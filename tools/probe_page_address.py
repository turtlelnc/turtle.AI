"""Frozen address probes on identical learned Top-K candidates; screening only."""
import argparse,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from models.checkpoints import load_model_and_memory
from tools.probe_gist_relations import documents
from tools.source_labels import derive_batch_targets
from tools.train_page_value_bridge import fingerprint

@torch.no_grad()
def dataset(model,memory,seed,count):
    ids=documents(seed,count,5,'mixed'); _,pages,_=derive_batch_targets(ids)
    data={k:[] for k in ('gist_k','gist_kv','body_int4','gist_kv_context')};labels=[];native=0
    captured={}
    hook=model.blocks[-1].attention.qkv.register_forward_hook(lambda m,i,o:captured.update(qkv=o))
    try:
        for start in range(0,count,32):
            source=ids[start:start+32]
            out=model(source,memory=memory,disk_backed=True,return_page_features=True)
            selected=out['page_features']['retrieved'][:,-1];safe=selected.clamp_min(0)
            rows=torch.arange(len(source))[:,None]
            _,k,v=captured.pop('qkv').chunk(3,-1)
            gist=torch.arange(5)*9+8
            gk=k[:,gist];gv=v[:,gist]
            from models.resonant import fake_int4
            indices=(torch.arange(5)[:,None]*9+torch.arange(8)).flatten()
            bk=k[:,indices];bv=v[:,indices]
            protected=torch.arange(40)<memory.sink_tokens
            bk=torch.where(protected[None,:,None],bk,fake_int4(bk))
            bv=torch.where(protected[None,:,None],bv,fake_int4(bv))
            body=torch.cat((bk,bv),-1).reshape(len(source),5,-1)
            raw=model.embedding(source[:,-1]);context=out['page_features']['query'][:,-1]
            for name,page,query in (('gist_k',gk,raw),('gist_kv',torch.cat((gk,gv),-1),raw),
                                    ('body_int4',body,raw),('gist_kv_context',torch.cat((gk,gv),-1),context)):
                x=torch.cat((query[:,None].expand(-1,selected.shape[1],-1),page[rows,safe]),-1)
                data[name].append(x.detach())
            labels.append(selected==pages[start:start+len(source),-1,None])
            truth=derive_batch_targets(source)[0][:,-1]
            native+=int((out['logits'][:,-1].argmax(-1)==truth).sum())
    finally:hook.remove()
    return {k:torch.cat(v) for k,v in data.items()},torch.cat(labels),native/count

@torch.no_grad()
def score(head,x,labels):
    covered=labels.any(-1);truth=labels.long().argmax(-1);pred=head(x).squeeze(-1).argmax(-1)
    correct=(pred==truth)&covered
    return {'coverage':float(covered.float().mean()),'conditional_accuracy':float(correct.sum()/covered.sum()),
            'all_document_accuracy':float(correct.float().mean()),
            'covered_ce':float(F.cross_entropy(head(x[covered]).squeeze(-1),truth[covered]))}

def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--seeds',type=int,default=3);p.add_argument('--features',default='gist_k,gist_kv,body_int4,gist_kv_context');p.add_argument('--hidden',type=int,default=128);args=p.parse_args()
    torch.set_num_threads(1);root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    model,memory=load_model_and_memory(args.checkpoint);model.requires_grad_(False);before=fingerprint(model)
    data={name:dataset(model,memory,seed,n) for name,seed,n in [('train',56100000,4096),('val',56200000,512),('test',56300000,1024)]}
    results=[]
    for feature in args.features.split(','):
        center=data['train'][0][feature].flatten(0,1).mean(0)
        scale=data['train'][0][feature].flatten(0,1).std(0).clamp_min(1e-5)
        xs={split:(d[0][feature]-center)/scale for split,d in data.items()}
        covered=data['train'][1].any(-1);trainx=xs['train'][covered];truth=data['train'][1][covered].long().argmax(-1)
        for seed in range(args.seeds):
            torch.manual_seed(seed);head=nn.Sequential(nn.Linear(trainx.shape[-1],args.hidden),nn.GELU(),nn.Linear(args.hidden,1))
            opt=torch.optim.AdamW(head.parameters(),lr=.003);rng=torch.Generator().manual_seed(56400000+seed)
            lowest=float('inf');best=None
            for step in range(args.steps):
                idx=torch.randint(len(trainx),(128,),generator=rng)
                loss=F.cross_entropy(head(trainx[idx]).squeeze(-1),truth[idx]);opt.zero_grad();loss.backward();opt.step()
                if (step+1)%100==0:
                    val=score(head,xs['val'],data['val'][1])['covered_ce']
                    if val<lowest:lowest=val;best={k:v.clone() for k,v in head.state_dict().items()}
            head.load_state_dict(best)
            row={'feature':feature,'seed':seed,'parameters':sum(p.numel() for p in head.parameters()),
                 'scores':{split:score(head,xs[split],data[split][1]) for split in data}}
            # Break query-page correspondence on test, keeping candidate pages fixed.
            corrupted=xs['test'].clone();corrupted[:,:,:model.cfg.width]=corrupted[:,:,:model.cfg.width].roll(1,0)
            row['query_shuffled']=score(head,corrupted,data['test'][1]);results.append(row)
            print(feature,seed,row['scores']['test'],flush=True)
        frozen=fingerprint(model)==before
        if not frozen:raise RuntimeError('parent changed')
        (root/'results.json').write_text(json.dumps({'arguments':vars(args),'parent_state_sha256':before,
          'parent_unchanged':frozen,'native_test':data['test'][2],'first_selected_baseline':float(data['test'][1][:,0].float().mean()),'results':results,
          'scope':'Frozen diagnostic heads, supervised source-page labels, same learned candidates; different input parameter counts. Negative probe results do not establish information loss.'},indent=2))
if __name__=='__main__':main()
