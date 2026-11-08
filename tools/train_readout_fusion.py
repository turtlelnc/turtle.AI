"""Paired joint/fixed/learned residual readout, with identical replay adaptation."""
import argparse,json
from dataclasses import asdict
from pathlib import Path
import torch
from models.resonant import Config,ResonantLM
from tools.replay_binding import sample,diagnostics
from tools.train_associative_memory import evaluate
from models.checkpoints import load_model_and_memory


def run(args,seed,variant):
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if saved.get('mode')!='retrieval' or len(saved.get('continuation_validation',[]))<2 or min(saved['continuation_validation'][-2:])<.95:
        raise ValueError('requires a qualified ungated retrieval parent')
    cfg=dict(saved['config']);cfg['readout_fusion']=variant!='joint'
    model=ResonantLM(Config(**cfg));missing=model.load_state_dict(saved['model'],strict=False)
    expected={f'blocks.{i}.attention.read_gate.{field}' for i in range(cfg['layers']) for field in ('weight','bias')} if cfg['readout_fusion'] else set()
    if set(missing.missing_keys)!=expected or missing.unexpected_keys:raise ValueError('unexpected migration keys')
    _,memory=load_model_and_memory(args.checkpoint)
    old=[p for n,p in model.named_parameters() if '.read_gate.' not in n]
    gates=[p for n,p in model.named_parameters() if '.read_gate.' in n]
    optimizer=torch.optim.AdamW(old,lr=.003);optimizer.load_state_dict(saved['optimizer'])
    if variant=='learned':optimizer.add_param_group({'params':gates,'lr':.003})
    elif variant=='fixed':
        for parameter in gates:parameter.requires_grad_(False)
    torch.set_rng_state(saved['torch_rng_state'])
    new_rng=torch.Generator().manual_seed(71000+seed);old_rng=torch.Generator().manual_seed(81000+seed)
    validation=[]
    for step in range(1,args.steps+1):
        model.train();new=sample(new_rng,16,5);old_data=sample(old_rng,8,4)
        optimizer.zero_grad()
        for ids,labels,pages in [tuple(x[:8] for x in new),old_data]:
            for start in range(0,8,4):
                loss=model(ids[start:start+4],labels[start:start+4],memory=memory,disk_backed=True,
                           routing_targets=pages[start:start+4],routing_weight=.5,
                           reconstruction_weight=.05,precision_weight=0)['losses']['total']
                if not torch.isfinite(loss):raise RuntimeError('nonfinite fusion loss')
                (loss/4).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%200==0 or step==args.steps:
            scores={str(b):evaluate(model,memory,'retrieval',seed,b,64,8,27300000+step*101,128,8,True,2)['accuracy'] for b in (4,5)}
            validation.append({'step':step,'accuracy':scores});print(seed,variant,step,scores,flush=True)
    tests={str(b):evaluate(model,memory,'retrieval',seed,b,128,8,29300000+b*10000,128,8,True,2) for b in (4,5)}
    destination=Path(args.output)/f'{variant}-{seed}';destination.mkdir(parents=True,exist_ok=True)
    saved.pop('continuation_validation',None)
    saved.update(config=cfg,model=model.state_dict(),optimizer=optimizer.state_dict(),
                 source_checkpoint=args.checkpoint,fusion_variant=variant,adaptation_seed=seed,
                 adaptation_steps=args.steps,continuation_current_blocks=5,fusion_arguments=vars(args),
                 adaptation_validation=validation,adaptation_evaluations=tests,
                 torch_rng_state=torch.get_rng_state(),
                 data_new_rng=new_rng.get_state(),data_old_rng=old_rng.get_state())
    torch.save(saved,destination/'checkpoint.pt')
    return {'variant':variant,'adaptation_seed':seed,'tests':tests,'validation':validation,
            'competition_diagnostic':{str(b):diagnostics(model,memory,seed+100,b) for b in (4,5)}}


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--output',required=True)
    p.add_argument('--seeds',type=int,default=3);p.add_argument('--steps',type=int,default=800)
    args=p.parse_args()
    if min(args.seeds,args.steps)<1:p.error('positive seeds/steps required')
    torch.set_num_threads(1);root=Path(args.output);root.mkdir(parents=True,exist_ok=True);results=[]
    for seed in range(args.seeds):
        for variant in ('joint','fixed','learned'):
            results.append(run(args,seed,variant))
            temporary=root/'results.tmp';temporary.write_text(json.dumps({'arguments':vars(args),'results':results},indent=2));temporary.replace(root/'results.json')

if __name__=='__main__':main()
