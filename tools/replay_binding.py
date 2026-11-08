"""Paired adaptation from one checkpoint: replay vs new-only vs half-new control.

Seeds vary adaptation data, not pretrained initialization. All branches start
with the identical model and optimizer. Final weights, not best validation, are tested.
"""
import argparse
from pathlib import Path
import json
import torch
from models.checkpoints import load_model_and_memory
from tools.associative_recall import batch
from tools.source_labels import derive_batch_targets
from tools.train_associative_memory import evaluate


def train_parts(new, old, branch):
    if branch not in ('new_only','half_new','replay'):raise ValueError('unknown adaptation control')
    if branch=='new_only':return [(new,1.)]
    first=tuple(t[:8] for t in new)
    return [(first,1.),(old,1. if branch=='replay' else 0.)]


def sample(rng,n,blocks):
    ids,_,_=batch(rng,n,blocks,working_blocks=2,key_count=128,value_count=8)
    labels,pages,_=derive_batch_targets(ids)
    return ids,labels,pages


@torch.no_grad()
def diagnostics(model,memory,seed,blocks):
    rng=torch.Generator().manual_seed(21300000+seed*1000+blocks)
    correct={'normal':0,'known_page_only':0}
    for _ in range(16):
        ids,labels,pages=sample(rng,8,blocks)
        for name,oracle in [('normal',False),('known_page_only',True)]:
            logits=model(ids,memory=memory,disk_backed=True,routing_targets=pages if oracle else None,
                         oracle_page_only=oracle)['logits'][:,-1]
            correct[name]+=int((logits.argmax(-1)==labels[:,-1]).sum())
    return {name:value/128 for name,value in correct.items()}


def run(args,seed,branch):
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model,memory=load_model_and_memory(args.checkpoint)
    if saved.get('continuation_current_blocks',saved['history'][-1]['blocks'])!=4:raise ValueError('requires two-answer-candidate checkpoint')
    if saved.get('mode')!='retrieval' or len(saved.get('continuation_validation',[]))<2 or min(saved['continuation_validation'][-2:])<.95:
        raise ValueError('requires a validation-qualified retrieval source')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003);optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['torch_rng_state'])
    new_rng=torch.Generator().manual_seed(51000+seed)
    old_rng=torch.Generator().manual_seed(61000+seed)
    history=[]
    before={str(b):evaluate(model,memory,'retrieval',seed,b,128,8,19300000+b*100000,128,8,True,2)
            for b in (4,5)}
    for step in range(1,args.steps+1):
        model.train();new=sample(new_rng,16,5);old=sample(old_rng,8,4)
        optimizer.zero_grad()
        for (ids,labels,pages),weight in train_parts(new,old,branch):
            for start in range(0,len(ids),4):
                stop=min(start+4,len(ids))
                loss=model(ids[start:stop],labels[start:stop],memory=memory,disk_backed=True,
                           routing_targets=pages[start:stop],routing_weight=.5,
                           reconstruction_weight=.05,precision_weight=0)['losses']['total']
                if not torch.isfinite(loss):raise RuntimeError('nonfinite adaptation loss')
                (loss*weight*(stop-start)/16).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%200==0 or step==args.steps:
            validation={str(b):evaluate(model,memory,'retrieval',seed,b,64,8,
                        17300000+step*101+b*10000,128,8,True,2)['accuracy'] for b in (4,5)}
            history.append({'step':step,'accuracy':validation})
            print(seed,branch,step,validation,flush=True)
    model.eval()
    tests={str(b):evaluate(model,memory,'retrieval',seed,b,128,8,19300000+b*100000,128,8,True,2)
           for b in (4,5)}
    directory=Path(args.output)/f'{branch}-{seed}';directory.mkdir(parents=True,exist_ok=True)
    old_validation=saved.pop('continuation_validation',[])
    saved.update(source_continuation_validation=old_validation,source_checkpoint=args.checkpoint,
                 adaptation_arguments=vars(args),adaptation_stage_blocks=[4,5],continuation_current_blocks=5,
                 model=model.state_dict(),optimizer=optimizer.state_dict(),adaptation_branch=branch,
                 adaptation_seed=seed,adaptation_steps=args.steps,adaptation_validation=history,adaptation_evaluations=tests,
                 torch_rng_state=torch.get_rng_state(),data_new_rng=new_rng.get_state(),
                 data_old_rng=old_rng.get_state())
    torch.save(saved,directory/'checkpoint.pt')
    return {'branch':branch,'adaptation_seed':seed,'steps':args.steps,'before':before,'tests':tests,
            'competition_diagnostic':{str(b):diagnostics(model,memory,seed,b) for b in (4,5)},
            'validation':history}


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=800);p.add_argument('--seeds',type=int,default=3)
    args=p.parse_args()
    if min(args.steps,args.seeds)<1:p.error('steps and seeds must be positive')
    torch.set_num_threads(1);root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in range(args.seeds):
        for branch in ('new_only','half_new','replay'):
            results.append(run(args,seed,branch))
            temporary=root/'results.tmp'
            temporary.write_text(json.dumps({'arguments':vars(args),'scope':'one initialization, paired adaptation-data seeds',
                                             'results':results},indent=2))
            temporary.replace(root/'results.json')

if __name__=='__main__':main()
