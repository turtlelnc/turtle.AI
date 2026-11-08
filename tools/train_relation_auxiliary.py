"""Matched relation-supervision screen; auxiliary head never supplies LM answers."""
import argparse,json
from pathlib import Path
import torch
from models.resonant import Config,ResonantLM
from tools.train_associative_memory import policy,evaluate,stream_check
from tools.replay_binding import sample
from tools.source_labels import derive_relation_targets


def run(args,seed,mode,use_aux):
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    cfg=dict(saved['config']);cfg['relation_reader']=True
    torch.manual_seed(110000+seed);model=ResonantLM(Config(**cfg))
    missing=model.load_state_dict(saved['model'],strict=False)
    if any(not n.startswith('relation_reader.') for n in missing.missing_keys) or missing.unexpected_keys:raise ValueError('unexpected migration')
    original=[p for n,p in model.named_parameters() if not n.startswith('relation_reader.')]
    added=list(model.relation_reader.parameters())
    optimizer=torch.optim.AdamW(original,lr=.003);optimizer.load_state_dict(saved['optimizer'])
    optimizer.add_param_group({'params':added,'lr':.003})
    memory=policy(mode,cfg['width'],10,gist_slots=2);supervision=policy('retrieval',cfg['width'],10,gist_slots=2)
    new_rng=torch.Generator().manual_seed(111000+seed);old_rng=torch.Generator().manual_seed(121000+seed)
    augmentation=torch.Generator().manual_seed(131000+seed)
    history=[]
    for step in range(1,args.steps+1):
        model.train();new=sample(new_rng,8,5);old=sample(old_rng,8,4);optimizer.zero_grad()
        for ids,labels,pages in (new,old):
            if args.shuffle_roles:
                count=(ids.shape[1]-2)//8;ids=ids.clone()
                records=ids[:,:count*8].reshape(len(ids),count,2,4).clone()
                swaps=torch.rand(len(ids),count,generator=augmentation)<.5
                records[swaps]=records[swaps].flip(1);ids[:,:count*8]=records.flatten(1)
            relation=derive_relation_targets(ids)
            for start in range(0,8,4):
                result=model(ids[start:start+4],labels[start:start+4],memory=memory,disk_backed=True,
                    routing_targets=pages[start:start+4],routing_weight=.5,reconstruction_weight=.05,precision_weight=0,
                    supervision_policy=supervision if mode=='full_supervised' else None,
                    relation_targets=relation[start:start+4],relation_weight=.1 if use_aux else 0.)
                loss=result['losses']['total']
                if not torch.isfinite(loss):raise RuntimeError('nonfinite relation training')
                (loss/4).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%200==0 or step==args.steps:
            validation={str(b):evaluate(model,memory,mode,seed,b,64,8,args.validation_base+step*101,128,8,True,2)['accuracy'] for b in (4,5)}
            history.append({'step':step,'accuracy':validation});print(seed,mode,use_aux,step,validation,flush=True)
    tests={str(b):evaluate(model,memory,mode,seed,b,128,8,args.test_base+b*10000,128,8,True,2) for b in (4,5)}
    name=f'{mode}-aux{int(use_aux)}-{seed}';directory=Path(args.output)/name;directory.mkdir(parents=True,exist_ok=True)
    saved.pop('continuation_validation',None)
    from dataclasses import asdict
    saved.update(config=cfg,model=model.state_dict(),optimizer=optimizer.state_dict(),
        policy={**asdict(memory),'working_dtype':str(memory.working_dtype)},
        auxiliary_relation_protocol=vars(args),auxiliary_relation_weight=.1 if use_aux else 0.,
        adaptation_seed=seed,adaptation_mode=mode,continuation_current_blocks=5,
        data_new_rng=new_rng.get_state(),data_old_rng=old_rng.get_state())
    torch.save(saved,directory/'checkpoint.pt')
    checks=[stream_check(model,memory,b,seed,directory/f'pages-{b}',128,8,True,2) for b in (4,5)]
    return {'mode':mode,'auxiliary':use_aux,'adaptation_seed':seed,'tests':tests,'validation':history,'stream_checks':checks}


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=600);p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--shuffle-roles',action='store_true')
    p.add_argument('--test-base',type=int,default=43300000)
    p.add_argument('--validation-base',type=int,default=41300000)
    args=p.parse_args()
    if min(args.steps,args.seeds)<1:p.error('positive steps/seeds required')
    torch.set_num_threads(1);root=Path(args.output);root.mkdir(parents=True,exist_ok=True);results=[]
    for seed in range(args.seeds):
        for mode in ('full_supervised','retrieval'):
            for aux in (False,True):
                results.append(run(args,seed,mode,aux))
                temporary=root/'results.tmp';temporary.write_text(json.dumps({'arguments':vars(args),'results':results},indent=2));temporary.replace(root/'results.json')

if __name__=='__main__':main()
