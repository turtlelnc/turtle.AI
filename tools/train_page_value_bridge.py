"""Frozen reader-to-answer bridge screen; no oracle routes or page labels in fitting."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
import torch
from torch.nn import functional as F
from models.checkpoints import load_model_and_memory
from models.page_value_bridge import PageValueBridge, bridge_inputs
from tools.replay_binding import sample


def shuffled(ids, rng, blocks, view):
    if view == 'fixed': return ids
    ids = ids.clone()
    records = ids[:, :blocks*8].reshape(len(ids), blocks, 2, 4).clone()
    swaps = torch.ones(len(ids), blocks, dtype=torch.bool) if view == 'swapped' else torch.rand(len(ids), blocks, generator=rng) < .5
    records[swaps] = records[swaps].flip(1)
    ids[:, :blocks*8] = records.flatten(1)
    return ids


def fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(bytes(tensor.contiguous().view(torch.uint8).flatten().tolist()))
    return digest.hexdigest()


@torch.no_grad()
def features(model, memory, ids):
    return bridge_inputs(model(ids, memory=memory, disk_backed=True, return_page_features=True))


@torch.no_grad()
def evaluate(model, memory, bridge, blocks, view, base, examples, mix_initial=-4.):
    # Identical source documents and augmentation across bridge fit seeds.
    rng = torch.Generator().manual_seed(base + blocks)
    augment = torch.Generator().manual_seed(base + 10000 + blocks)
    scores = dict(native=0, bridge=0, untrained_bridge=0, uniform_selector=0,
                  misaligned_values=0, native_and_bridge_joint=0, route_hit=0,
                  true_page_head=0)
    initial = PageValueBridge(model.cfg.width, mix_initial)
    ce = dict(native=0., bridge=0.)
    for start in range(0, examples, 16):
        ids, labels, pages = sample(rng, min(16, examples-start), blocks)
        ids = shuffled(ids, augment, blocks, view)
        x = features(model, memory, ids); truth = labels[:, -1]
        native = x['native'].argmax(-1)
        logp = bridge(**x); pred = logp.argmax(-1)
        scores['native'] += int((native == truth).sum())
        scores['bridge'] += int((pred == truth).sum())
        scores['untrained_bridge'] += int((initial(**x).argmax(-1) == truth).sum())
        uniform = dict(x); uniform['keys'] = torch.zeros_like(x['keys'])
        scores['uniform_selector'] += int((bridge(**uniform).argmax(-1) == truth).sum())
        # Rotate values across documents while preserving native logits, selected
        # IDs and relevance keys. This breaks content-to-document correspondence.
        broken = dict(x); broken['values'] = x['values'].roll(1, 0)
        scores['misaligned_values'] += int((bridge(**broken).argmax(-1) == truth).sum())
        scores['native_and_bridge_joint'] += int(((native == truth) & (pred == truth)).sum())
        scores['route_hit'] += int((x['selected'] == pages[:, -1, None]).any(-1).sum())
        row = torch.arange(len(ids))
        scores['true_page_head'] += int((x['values'][row, pages[:, -1]].argmax(-1) == truth).sum())
        ce['native'] += float(F.cross_entropy(x['native'], truth, reduction='sum'))
        ce['bridge'] += float(F.nll_loss(logp, truth, reduction='sum'))
    return {**{k: v/examples for k,v in scores.items()},
            'native_ce': ce['native']/examples, 'bridge_ce': ce['bridge']/examples,
            'examples': examples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint'); p.add_argument('--output', required=True)
    p.add_argument('--seeds', type=int, default=3)
    p.add_argument('--steps', type=int, default=600)
    p.add_argument('--examples', type=int, default=512)
    p.add_argument('--mix-initial', type=float, default=-4.)
    args = p.parse_args()
    if min(args.seeds,args.steps,args.examples) < 1 or args.examples % 16:
        p.error('positive seeds/steps and examples a multiple of 16 required')
    torch.set_num_threads(1)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    model, memory = load_model_and_memory(args.checkpoint)
    if not model.cfg.relation_reader or memory.top_k < 1:
        raise ValueError('a trained relation reader and bounded retrieval are required')
    model.requires_grad_(False)
    before = fingerprint(model)
    results = []
    for seed in range(args.seeds):
        torch.manual_seed(50100000 + seed)
        bridge = PageValueBridge(model.cfg.width, args.mix_initial)
        optimizer = torch.optim.AdamW(bridge.parameters(), lr=.003, weight_decay=0.)
        rng = torch.Generator().manual_seed(51100000 + seed)
        augment = torch.Generator().manual_seed(52100000 + seed)
        history = []
        for step in range(1, args.steps+1):
            optimizer.zero_grad()
            for blocks in (4,5):
                ids, labels, _ = sample(rng, 8, blocks)
                ids = shuffled(ids, augment, blocks, 'mixed')
                logp = bridge(**features(model, memory, ids))
                loss = F.nll_loss(logp, labels[:, -1]) / 2
                if not torch.isfinite(loss): raise RuntimeError('nonfinite bridge loss')
                loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.)
            optimizer.step()
            if step % 200 == 0 or step == args.steps:
                history.append({'step':step, 'mix':float(bridge.mix_logit.sigmoid().detach()),
                                'last_half_ce':float(loss.detach()*2)})
                print(seed, history[-1], flush=True)
        tests = {str(b):{view:evaluate(model,memory,bridge,b,view,53100000,args.examples,args.mix_initial)
                       for view in ('fixed','mixed','swapped')} for b in (4,5)}
        torch.save({'bridge':bridge.state_dict(),'width':model.cfg.width,
                    'parent_sha256':before,'arguments':vars(args),'fit_seed':seed}, root/f'bridge-{seed}.pt')
        results.append({'fit_seed':seed,'history':history,'tests':tests})
        summary = {str(b):{view:{k:mean(r['tests'][str(b)][view][k] for r in results)
                                  for k in tests[str(b)][view] if k!='examples'}
                          for view in ('fixed','mixed','swapped')} for b in (4,5)}
        unchanged = fingerprint(model) == before
        if not unchanged: raise RuntimeError('frozen parent mutated')
        data = {'arguments':vars(args),'parent_sha256':before,'frozen_parent_unchanged':unchanged,
                'bridge_parameters':sum(p.numel() for p in bridge.parameters()),
                'summary':summary,'runs':results,
                'scope':'Screening only: three fit-data seeds share one frozen backbone and test set; dense last-token adapter, no streaming support or latency claims.'}
        temp = root/'results.tmp'; temp.write_text(json.dumps(data,indent=2)); temp.replace(root/'results.json')
        print(json.dumps(summary),flush=True)


if __name__ == '__main__': main()
