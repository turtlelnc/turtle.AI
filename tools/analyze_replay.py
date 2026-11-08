"""Summarize complete paired runs and recompute same-input page-only diagnostics."""
import argparse
import json
from pathlib import Path
from statistics import mean
import torch
from models.checkpoints import load_model_and_memory
from tools.replay_binding import diagnostics


def main():
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();root=Path(args.root)
    data=json.loads((root/'results.json').read_text())
    if len(data['results'])!=data['arguments']['seeds']*3:raise ValueError('wait for all paired branches')
    torch.set_num_threads(1)
    for seed in range(data['arguments']['seeds']):
        rows=[r for r in data['results'] if r['adaptation_seed']==seed]
        for blocks in ('4','5'):
            if len({r['before'][blocks]['accuracy'] for r in rows})!=1:
                raise ValueError('branches do not share the same before-test accuracy')
    for row in data['results']:
        model,memory=load_model_and_memory(root/f"{row['branch']}-{row['adaptation_seed']}"/'checkpoint.pt')
        row['competition_diagnostic']={str(b):diagnostics(model,memory,row['adaptation_seed'],b) for b in (4,5)}
    summary={}
    for branch in ('new_only','half_new','replay'):
        rows=[r for r in data['results'] if r['branch']==branch];metrics={}
        for blocks in ('4','5'):
            values=[r['tests'][blocks]['accuracy'] for r in rows]
            metrics[blocks]={'accuracy_mean':mean(values),'min':min(values),'max':max(values),
                'before_mean':mean(r['before'][blocks]['accuracy'] for r in rows),
                'joint_mean':mean(r['tests'][blocks]['counterfactual_joint_accuracy'] for r in rows),
                'paired_page_only_mean':mean(r['competition_diagnostic'][blocks]['known_page_only'] for r in rows),
                'paired_normal_mean':mean(r['competition_diagnostic'][blocks]['normal'] for r in rows)}
        summary[branch]=metrics
    result={'scope':data['scope'],'summary':summary,'runs':data['results']}
    (root/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
