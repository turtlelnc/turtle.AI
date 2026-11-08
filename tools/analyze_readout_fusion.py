"""All-seed fusion summary, zero-shot migration and trained streaming parity."""
import argparse,json
from pathlib import Path
from statistics import mean
import torch
from models.resonant import Config,ResonantLM
from models.checkpoints import load_model_and_memory
from tools.train_associative_memory import evaluate,stream_check
from tools.replay_binding import sample


@torch.no_grad()
def gate_statistics(model,memory,seed,blocks):
    if not model.cfg.readout_fusion:return None
    samples=[[] for _ in model.blocks];hooks=[]
    for layer_id,layer in enumerate(model.blocks):
        def collect(module,inputs,output,index=layer_id):
            samples[index].extend((.5*output[:,-1].sigmoid()).flatten().tolist())
        hooks.append(layer.attention.read_gate.register_forward_hook(collect))
    try:
        rng=torch.Generator().manual_seed(31300000+seed*1000+blocks)
        for _ in range(16):
            ids,_,_=sample(rng,8,blocks);model(ids,memory=memory,disk_backed=True)
    finally:
        for hook in hooks:hook.remove()
    return [{'mean':mean(x),'min':min(x),'max':max(x)} for x in samples]


def main():
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();root=Path(args.root)
    data=json.loads((root/'results.json').read_text())
    if len(data['results'])!=data['arguments']['seeds']*3:raise ValueError('wait for all runs')
    torch.set_num_threads(1)
    parent=torch.load(data['arguments']['checkpoint'],map_location='cpu',weights_only=True)
    for row in data['results']:
        seed=row['adaptation_seed'];variant=row['variant']
        directory=root/f'{variant}-{seed}'
        model,memory=load_model_and_memory(directory/'checkpoint.pt')
        row['gate_statistics']={str(b):gate_statistics(model,memory,seed,b) for b in (4,5)}
        row['stream_checks']=[stream_check(model,memory,b,seed,directory/f'parity-{b}',128,8,True,2) for b in (4,5)]
        cfg=dict(parent['config']);cfg['readout_fusion']=variant!='joint'
        initial=ResonantLM(Config(**cfg));initial.load_state_dict(parent['model'],strict=False)
        row['before']={str(b):evaluate(initial,memory,'retrieval',seed,b,128,8,29300000+b*10000,128,8,True,2) for b in (4,5)}
    summary={}
    for variant in ('joint','fixed','learned'):
        rows=[r for r in data['results'] if r['variant']==variant]
        summary[variant]={}
        for b in ('4','5'):
            values=[r['tests'][b]['accuracy'] for r in rows]
            summary[variant][b]={'mean':mean(values),'min':min(values),'max':max(values),
                'before_mean':mean(r['before'][b]['accuracy'] for r in rows),
                'joint_answer_mean':mean(r['tests'][b]['counterfactual_joint_accuracy'] for r in rows)}
    for seed in range(data['arguments']['seeds']):
        rows={r['variant']:r for r in data['results'] if r['adaptation_seed']==seed}
        for b in ('4','5'):
            if rows['fixed']['before'][b]['accuracy']!=rows['learned']['before'][b]['accuracy']:
                raise ValueError('fixed and learned migration functions differ')
    checks=[c for r in data['results'] for c in r['stream_checks']]
    result={'summary':summary,'stream_checks':len(checks),'stream_top1_all_match':all(c['top1_matches_dense'] for c in checks),
            'max_stream_logit_difference':max(c['max_logit_difference'] for c in checks),'runs':data['results']}
    (root/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))

if __name__=='__main__':main()
