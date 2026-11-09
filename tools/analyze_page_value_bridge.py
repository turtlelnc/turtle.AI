"""Query-switch and selector diagnostics on fresh five-page mixed-order data.

Source provenance is used for scoring only; bridge predictions receive no labels.
"""
import argparse
import json
import math
from pathlib import Path
import torch
from models.checkpoints import load_model_and_memory
from models.page_value_bridge import PageValueBridge
from tools.source_labels import extract_facts
from tools.replay_binding import sample
from tools.train_page_value_bridge import features, shuffled, fingerprint


@torch.no_grad()
def analyze(root, examples=512):
    data = json.loads((root/'results.json').read_text())
    model, memory = load_model_and_memory(data['arguments']['checkpoint'])
    model.requires_grad_(False)
    if fingerprint(model) != data['parent_sha256']: raise ValueError('wrong parent')
    bridges = []
    for row in data['runs']:
        saved = torch.load(root/f"bridge-{row['fit_seed']}.pt", weights_only=True)
        if saved['parent_sha256'] != data['parent_sha256']: raise ValueError('wrong adapter parent')
        bridge = PageValueBridge(model.cfg.width)
        bridge.load_state_dict(saved['bridge']); bridges.append(bridge)
    totals = [dict(native=0, bridge=0, native_joint=0, bridge_joint=0,
                   selector_hit=0, route_hit=0, true_page_head=0,
                   count=0, paired_count=0) for _ in bridges]
    rng = torch.Generator().manual_seed(55100005)
    augment = torch.Generator().manual_seed(55200005)
    for start in range(0,examples,16):
        ids, labels, pages = sample(rng,min(16,examples-start),5)
        ids = shuffled(ids,augment,5,'mixed')
        first = features(model,memory,ids)
        other = ids.clone(); valid=[]; other_answers=[]
        for i, tokens in enumerate(ids.tolist()):
            candidates = [f for f in extract_facts(tokens[:-2]).values()
                          if f.page < 3 and f.value != int(labels[i,-1])]
            valid.append(bool(candidates))
            fact = candidates[0] if candidates else None
            if fact is not None: other[i,-1] = fact.key
            other_answers.append(fact.value if fact else int(labels[i,-1]))
        second = features(model,memory,other)
        truth = labels[:,-1]; other_truth = torch.tensor(other_answers)
        paired = torch.tensor(valid)
        native = first['native'].argmax(-1) == truth
        native_other = second['native'].argmax(-1) == other_truth
        rows = torch.arange(len(ids))[:,None]
        for bridge, total in zip(bridges,totals):
            pred = bridge(**first).argmax(-1) == truth
            other_pred = bridge(**second).argmax(-1) == other_truth
            projected = first['query'] + first['query'] @ bridge.score_delta
            selected = first['selected']
            keys = first['keys'][rows,selected.clamp_min(0)]
            scores = (projected[:,None]*keys).sum(-1)/math.sqrt(model.cfg.width)
            scores = scores.masked_fill(selected<0,-torch.inf)
            picked = selected.gather(-1,scores.argmax(-1,keepdim=True)).squeeze(-1)
            total['selector_hit'] += int((picked==pages[:,-1]).sum())
            total['route_hit'] += int((selected==pages[:,-1,None]).any(-1).sum())
            head = first['values'][torch.arange(len(ids)),pages[:,-1]].argmax(-1)
            total['true_page_head'] += int((head==truth).sum())
            total['native'] += int(native.sum()); total['bridge'] += int(pred.sum())
            total['native_joint'] += int((native & native_other & paired).sum())
            total['bridge_joint'] += int((pred & other_pred & paired).sum())
            total['count'] += len(ids); total['paired_count'] += int(paired.sum())
    normalized=[]
    for total in totals:
        normalized.append({**total, 'accuracy':{k:v/total['paired_count' if k.endswith('_joint') else 'count']
                           for k,v in total.items() if k not in ('count','paired_count')}})
    result={'test_seed':55100005,'augmentation_seed':55200005,'blocks':5,
            'scope':'Fresh-data diagnostics only; source page labels used for scoring, never prediction.',
            'runs':normalized}
    (root/'query-switch-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root');args=parser.parse_args()
    torch.set_num_threads(1);analyze(Path(args.root))
