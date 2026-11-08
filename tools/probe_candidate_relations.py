"""Frozen representation + candidate embedding -> true/IGNORE relation probe.

Candidate identities are provided by source annotations: this is a diagnostic,
not a deployable reader. Positive/negative marginals are symmetric and balanced.
"""
import argparse,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from models.resonant import ResonantLM
from models.checkpoints import load_model_and_memory
from tools.probe_gist_relations import documents,encode


class RelationProbe(nn.Module):
    def __init__(self,width,embedding,kind):
        super().__init__();self.kind=kind
        self.head=nn.Bilinear(width,embedding,1) if kind=='bilinear' else nn.Sequential(nn.Linear(width+embedding,128),nn.GELU(),nn.Linear(128,1))
    def forward(self,x,candidate):
        return (self.head(x,candidate) if self.kind=='bilinear' else self.head(torch.cat((x,candidate),-1))).squeeze(-1)


@torch.no_grad()
def report(probe,x,positive,negative):
    a=probe(x,positive);b=probe(x,negative)
    return {'pair_choice':float(((a>b).float()+.5*(a==b)).mean()),
            'binary_accuracy':float(((a>0).float().sum()+(b<0).float().sum())/(2*len(x))),
            'loss':float((F.softplus(-a).mean()+F.softplus(b).mean())/2)}


def fit(x,pos,neg,val,kind,seed,steps):
    torch.manual_seed(seed);probe=RelationProbe(x.shape[-1],pos.shape[-1],kind)
    optimizer=torch.optim.AdamW(probe.parameters(),lr=.003);rng=torch.Generator().manual_seed(8300000+seed)
    best=None;lowest=float('inf')
    for step in range(steps):
        idx=torch.randint(len(x),(128,),generator=rng)
        total=(F.softplus(-probe(x[idx],pos[idx])).mean()+F.softplus(probe(x[idx],neg[idx])).mean())/2
        optimizer.zero_grad();total.backward();optimizer.step()
        if (step+1)%100==0 or step+1==steps:
            score=report(probe,*val)['loss']
            if score<lowest:lowest=score;best={n:p.detach().clone() for n,p in probe.state_dict().items()}
    probe.load_state_dict(best);probe.eval();return probe


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--replay',required=True);p.add_argument('--output',required=True)
    p.add_argument('--train-roles',choices=['fixed','mixed'],default='mixed');p.add_argument('--seeds',type=int,default=3);p.add_argument('--steps',type=int,default=600)
    p.add_argument('--features',nargs='+',default=['gist_kv_2','raw_bag','oracle_value'])
    p.add_argument('--historical-only',action='store_true')
    p.add_argument('--encoders',nargs='+',choices=['parent','replay','random'],default=['parent','replay','random'])
    p.add_argument('--all-values',action='store_true')
    args=p.parse_args()
    if min(args.steps,args.seeds)<1:p.error('positive steps/seeds required')
    torch.set_num_threads(1);root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    ids={'train':documents(8400000,2000,5,args.train_roles),'validation':documents(8500000,256,5,args.train_roles),
         'mixed':documents(8600000,512,5,'mixed'),'fixed':documents(8700000,512,5,'fixed')}
    ids['swapped']=ids['mixed'].clone();ids['swapped'][:,:40]=ids['swapped'][:,:40].reshape(512,5,2,4).flip(2).flatten(1)
    parent,memory=load_model_and_memory(args.parent);replay,policy=load_model_and_memory(args.replay)
    torch.manual_seed(2026);random=ResonantLM(parent.cfg).eval();results=[]
    for encoder,model,mem in [('parent',parent,memory),('replay',replay,policy),('random',random,memory)]:
        if encoder not in args.encoders:continue
        model.requires_grad_(False);snapshot={n:p.detach().clone() for n,p in model.named_parameters()}
        features=args.features
        data={name:encode(model,mem,tokens,5,features,args.historical_only) for name,tokens in ids.items()}
        with torch.no_grad():
            candidates={name:(model.embedding(y[:,1]+257).clone(),model.embedding(y[:,2]+257).clone()) for name,(_,y) in data.items()}
        for feature in features:
            center=data['train'][0][feature].mean(0);scale=data['train'][0][feature].std(0).clamp_min(1e-5)
            x={name:(f[feature]-center)/scale for name,(f,_) in data.items()}
            # Candidate normalization depends on training candidates only.
            joined=torch.cat(candidates['train']);cm=joined.mean(0);cs=joined.std(0).clamp_min(1e-5)
            c={name:((a-cm)/cs,(b-cm)/cs) for name,(a,b) in candidates.items()}
            for kind in ('bilinear','mlp'):
                for seed in range(args.seeds):
                    probe=fit(x['train'],*c['train'],(x['validation'],*c['validation']),kind,seed,args.steps)
                    row={'encoder':encoder,'feature':feature,'probe':kind,'seed':seed,
                        'training':report(probe,x['train'],*c['train']),
                        'validation':report(probe,x['validation'],*c['validation']),
                        'tests':{name:report(probe,x[name],*c[name]) for name in ('mixed','fixed','swapped')}}
                    if args.all_values:
                        with torch.no_grad():
                            candidates_all=(model.embedding(torch.arange(257,265))-cm)/cs
                            for name in ('mixed','fixed','swapped'):
                                logits=torch.stack([probe(x[name],candidate[None].expand(len(x[name]),-1)) for candidate in candidates_all],-1)
                                truth=data[name][1][:,1]
                                row['tests'][name]['all_value_accuracy']=float((logits.argmax(-1)==truth).float().mean())
                        torch.save({'probe':probe.state_dict(),'feature_mean':center,'feature_scale':scale,
                                    'candidate_mean':cm,'candidate_scale':cs,'feature':feature,'kind':kind,
                                    'encoder':encoder,'arguments':vars(args)},root/f'{encoder}-{feature}-{kind}-{seed}.pt')
                    results.append(row)
            print(encoder,feature,'completed',flush=True)
        if any(not torch.equal(snapshot[n],p) for n,p in model.named_parameters()):raise RuntimeError('encoder changed')
        (root/'results.json').write_text(json.dumps({'arguments':vars(args),'encoder_unchanged':True,'results':results},indent=2))

if __name__=='__main__':main()
