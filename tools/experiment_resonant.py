"""Synthetic screening experiment, not evidence of real-world long-context gains."""
import argparse
import json
import time
from dataclasses import asdict
import torch
from models.resonant import Config, ResonantLM


def batch(generator, size):
    # Recall one random key/value association behind three filler blocks.
    keys = torch.randint(1, 5, (size,), generator=generator)
    values = torch.randint(5, 9, (size,), generator=generator)
    ids = torch.randint(9, 16, (size, 20), generator=generator)
    ids[:, 0] = 0
    ids[:, 1] = keys
    ids[:, 2] = values
    ids[:, -1] = keys
    labels = torch.full_like(ids, -100)
    labels[:, -1] = values
    return ids, labels


def run(steps, seed, mode):
    torch.manual_seed(seed)
    cfg = Config(vocab_size=16, width=32, layers=2, block_size=4,
                 window=32 if mode == 'full' else 4, sinks=1,
                 top_k=1 if mode == 'retrieval' else 0, max_tokens=32)
    model = ResonantLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    train_rng = torch.Generator().manual_seed(seed + 1000)
    started = time.perf_counter()
    initial = None
    for _ in range(steps):
        ids, labels = batch(train_rng, 32)
        # Pause training is not part of this synthetic task.
        result = model(ids, labels)
        loss = result['losses']['total']
        if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
        if initial is None: initial = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    elapsed = time.perf_counter() - started
    model.eval()
    ids, labels = batch(torch.Generator().manual_seed(99000 + seed), 256)
    with torch.no_grad():
        result = model(ids, labels)
        logits = result['logits'][:, -1]
        accuracy = float((logits.argmax(-1) == labels[:, -1]).float().mean())
        # Matched weights and input: force retained cold K/V for comparison.
        for block in model.blocks:
            block.attention.router.weight.zero_()
            block.attention.router.bias.fill_(100)
        reference = model(ids)['logits'][:, -1]
        disagreement = float((logits.argmax(-1) != reference.argmax(-1)).float().mean())
        gaps = logits.topk(2, dim=-1).values
    return {'mode': mode, 'seed': seed, 'steps': steps, 'config': asdict(cfg),
            'initial_train_loss': initial, 'heldout_total_loss': float(result['losses']['total']),
            'heldout_recall_accuracy': accuracy, 'training_seconds': elapsed,
            'mean_top2_margin': float((gaps[:, 0] - gaps[:, 1]).mean()),
            'fp32_cold_ablation_top1_disagreement': disagreement}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args()
    if min(args.steps, args.seeds) < 1: parser.error('steps and seeds must be positive')
    torch.set_num_threads(1)
    results = [run(args.steps, seed, mode) for seed in range(args.seeds)
               for mode in ('full', 'gist_only', 'retrieval')]
    print(json.dumps({'task': 'single association recall, 20 input tokens',
                      'chance_accuracy': 0.25, 'results': results}, indent=2))

if __name__ == '__main__': main()
