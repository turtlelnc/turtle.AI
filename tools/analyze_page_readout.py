"""All-seed page-readout summary, migration and trained streaming parity."""
import argparse,json
from pathlib import Path
from statistics import mean
import torch
from models.resonant import Config,ResonantLM
from models.checkpoints import load_model_and_memory
from tools.train_associative_memory import evaluate,stream_check



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
        row['stream_checks']=[stream_check(model,memory,b,seed,directory/f'parity-{b}',128,8,True,2) for b in (4,5)]
        cfg=dict(parent['config']);cfg['readout_fusion']=False
        cfg['page_score_weight']=0. if variant=='joint' else 1.
        cfg['page_score_detach']=variant=='detached'
        initial=ResonantLM(Config(**cfg));initial.load_state_dict(parent['model'])
        row['before']={str(b):evaluate(initial,memory,'retrieval',seed,b,128,8,39300000+b*10000,128,8,True,2) for b in (4,5)}
    summary={}
    for variant in ('joint','detached','differentiable'):
        rows=[r for r in data['results'] if r['variant']==variant]
        summary[variant]={}
        for b in ('4','5'):
            values=[r['tests'][b]['accuracy'] for r in rows]
            summary[variant][b]={'mean':mean(values),'min':min(values),'max':max(values),
                'before_mean':mean(r['before'][b]['accuracy'] for r in rows),
                'joint_answer_mean':mean(r['tests'][b]['counterfactual_joint_accuracy'] for r in rows),
                'page_recall_mean':mean(r['tests'][b]['route_hit_rate'] for r in rows)}
    for seed in range(data['arguments']['seeds']):
        rows={r['variant']:r for r in data['results'] if r['adaptation_seed']==seed}
        for b in ('4','5'):
            if rows['detached']['before'][b]['accuracy']!=rows['differentiable']['before'][b]['accuracy']:
                raise ValueError('detached and differentiable forward functions differ')
    checks=[c for r in data['results'] for c in r['stream_checks']]
    result={'summary':summary,'stream_checks':len(checks),'stream_top1_all_match':all(c['top1_matches_dense'] for c in checks),
            'max_stream_logit_difference':max(c['max_logit_difference'] for c in checks),'runs':data['results']}
    (root/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))

if __name__=='__main__':main()
