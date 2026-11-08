"""Eight-way value reconstruction from a known historical page; frozen encoder."""
import argparse,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from models.checkpoints import load_model_and_memory
from tools.probe_gist_relations import documents,encode


@torch.no_grad()
def score(head,x,y):
    logits=head(x);return {'accuracy':float((logits.argmax(-1)==y).float().mean()),'loss':float(F.cross_entropy(logits,y))}


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=600);p.add_argument('--seeds',type=int,default=3)
    args=p.parse_args()
    if min(args.steps,args.seeds)<1:p.error('positive steps/seeds required')
    torch.set_num_threads(1);root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    model,memory=load_model_and_memory(args.checkpoint);model.requires_grad_(False)
    snapshot={n:p.detach().clone() for n,p in model.named_parameters()}
    tokens={'train':documents(8400000,2000,5,'mixed'),'val':documents(8500000,256,5,'mixed'),
            'mixed':documents(8600000,512,5,'mixed'),'fixed':documents(8700000,512,5,'fixed')}
    tokens['swapped']=tokens['mixed'].clone();tokens['swapped'][:,:40]=tokens['swapped'][:,:40].reshape(512,5,2,4).flip(2).flatten(1)
    features=['gist_kv_2','body_kv_2','body_int4_kv_2','raw_bag','oracle_value']
    data={name:encode(model,memory,ids,5,features,True) for name,ids in tokens.items()};results=[]
    for feature in features:
        center=data['train'][0][feature].mean(0);scale=data['train'][0][feature].std(0).clamp_min(1e-5)
        x={name:(f[feature]-center)/scale for name,(f,_) in data.items()}
        y={name:labels[:,1] for name,(_,labels) in data.items()}
        for kind in ('linear','mlp'):
            for seed in range(args.seeds):
                torch.manual_seed(seed)
                head=nn.Linear(x['train'].shape[-1],8) if kind=='linear' else nn.Sequential(nn.Linear(x['train'].shape[-1],128),nn.GELU(),nn.Linear(128,8))
                optimizer=torch.optim.AdamW(head.parameters(),lr=.01 if kind=='linear' else .003)
                rng=torch.Generator().manual_seed(9300000+seed);lowest=float('inf');best=None
                for step in range(args.steps):
                    idx=torch.randint(len(x['train']),(256,),generator=rng)
                    loss=F.cross_entropy(head(x['train'][idx]),y['train'][idx]);optimizer.zero_grad();loss.backward();optimizer.step()
                    if (step+1)%100==0 or step+1==args.steps:
                        val=score(head,x['val'],y['val'])['loss']
                        if val<lowest:lowest=val;best={n:p.detach().clone() for n,p in head.state_dict().items()}
                head.load_state_dict(best);head.eval()
                row={'feature':feature,'kind':kind,'seed':seed,'train':score(head,x['train'],y['train']),
                     'val':score(head,x['val'],y['val']),'tests':{name:score(head,x[name],y[name]) for name in ('mixed','fixed','swapped')}}
                with torch.no_grad():
                    a=head(x['mixed']).argmax(-1);b=head(x['swapped']).argmax(-1)
                    row['paired_correct']=float(((a==y['mixed'])&(b==y['swapped'])).float().mean())
                results.append(row)
                torch.save({'head':head.state_dict(),'feature_mean':center,'feature_scale':scale,
                            'arguments':vars(args),'feature':feature,'kind':kind},root/f'{feature}-{kind}-{seed}.pt')
        print(feature,'completed',flush=True)
    if any(not torch.equal(snapshot[n],p) for n,p in model.named_parameters()):raise RuntimeError('encoder changed')
    (root/'results.json').write_text(json.dumps({'arguments':vars(args),'encoder_unchanged':True,'results':results},indent=2))

if __name__=='__main__':main()
