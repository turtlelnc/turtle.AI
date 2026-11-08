"""Aligned-policy multi-association training and held-out screening on CPU."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import torch
from models.resonant import Config, ResonantLM
from models.streaming_memory import MemoryConfig, ResonantStream
from models.memory_policy import cold_page_bytes
from tools.associative_recall import batch, BLOCK_SIZE, VOCAB_SIZE, VALUES
from tools.source_labels import derive_batch_targets


def policy(mode, width, max_blocks, dtype=torch.float32, gist_slots=2):
    if mode in ('full', 'full_supervised'):
        return MemoryConfig(block_size=BLOCK_SIZE, working_blocks=max_blocks + 1,
                            sink_tokens=1, gist_slots=max_blocks + 1,
                            cold_bytes=0, top_k=0, working_dtype=dtype)
    return MemoryConfig(block_size=BLOCK_SIZE, working_blocks=1, sink_tokens=1,
                        gist_slots=0 if mode == 'local' else gist_slots,
                        cold_bytes=4 * cold_page_bytes(width, BLOCK_SIZE),
                        top_k=2 if mode in ('retrieval', 'retrieval_unsupervised') else 0,
                        working_dtype=dtype)


@torch.no_grad()
def evaluate(model, memory, mode, seed, blocks, examples, batch_size, evaluation_seed=900000, key_count=256, value_count=32, contradictory=True, query_gap=1):
    model.eval()
    generator = torch.Generator().manual_seed(evaluation_seed + 1000 * seed + blocks)
    correct, route_hits, disagreements, count, ce_sum, margins = 0, 0, 0, 0, 0., 0.
    oracle_correct, counterfactual_count, counterfactual_joint = 0, 0, 0
    started = time.perf_counter()
    for start in range(0, examples, batch_size):
        ids, labels, targets = batch(generator, min(batch_size, examples - start), blocks, key_count=key_count,
                                     value_count=value_count, contradictory=contradictory, working_blocks=query_gap)
        result = model(ids, labels, memory=memory, disk_backed=True)
        logits = result['logits'][:, -1]
        predictions = logits.argmax(-1)
        correct += int((predictions == labels[:, -1]).sum())
        ce_sum += float(result['losses']['ce']) * len(ids)
        routes = model.blocks[-1].attention.last_retrieved[:, -1]
        if memory.top_k:
            route_hits += int((routes == targets[:, -1, None]).any(-1).sum())
        if memory.top_k:
            oracle = model(ids, memory=memory, disk_backed=True, routing_targets=targets,
                           oracle_routes=True)['logits'][:, -1]
            oracle_correct += int((oracle.argmax(-1) == labels[:, -1]).sum())
        # Keep the whole document fixed; ask a different key with a different answer.
        records = ids[:, :-2].reshape(len(ids), blocks, BLOCK_SIZE)
        cold_values = records[:, :blocks - query_gap, 2]
        alternatives = cold_values != labels[:, -1, None]
        paired_valid = alternatives.any(-1)
        if paired_valid.any():
            rows = torch.arange(len(ids))[paired_valid]
            other_blocks = alternatives.long().argmax(-1)[paired_valid]
            other_ids = ids[paired_valid].clone()
            other_ids[:, -1] = records[rows, other_blocks, 1]
            other_labels = records[rows, other_blocks, 2]
            other_logits = model(other_ids, memory=memory, disk_backed=True)['logits'][:, -1]
            counterfactual_joint += int(((predictions[paired_valid] == labels[paired_valid, -1]) &
                                         (other_logits.argmax(-1) == other_labels)).sum())
            counterfactual_count += len(rows)
        reference = model(ids, memory=memory, disk_backed=True, quantize_cold=False)['logits'][:, -1]
        disagreements += int((predictions != reference.argmax(-1)).sum())
        top = logits.topk(2, -1).values
        margins += float((top[:, 0] - top[:, 1]).sum())
        count += len(ids)
    return {'blocks': blocks, 'input_tokens': blocks * BLOCK_SIZE + 2, 'examples': count,
            'accuracy': correct / count, 'ce': ce_sum / count,
            'route_hit_rate': route_hits / count if memory.top_k else None,
            'oracle_page_accuracy': oracle_correct / count if memory.top_k else None,
            'counterfactual_examples': counterfactual_count,
            'counterfactual_joint_accuracy': counterfactual_joint / counterfactual_count if counterfactual_count else None,
            'random_topk_page_chance': min(memory.top_k, blocks - 1) / (blocks - 1) if memory.top_k else None,
            'position_prior_topk_chance': min(memory.top_k, blocks - query_gap) / (blocks - query_gap) if memory.top_k else None,
            'fp32_cold_top1_disagreement': disagreements / count,
            'mean_top2_margin': margins / count, 'evaluation_seconds': time.perf_counter() - started}


@torch.no_grad()
def stream_check(model, memory, blocks, seed, disk_root, key_count=256, value_count=32, contradictory=True, query_gap=1):
    generator = torch.Generator().manual_seed(800000 + seed + blocks)
    ids, labels, targets = batch(generator, 1, blocks, key_count=key_count, value_count=value_count,
                                 contradictory=contradictory, working_blocks=query_gap)
    model.eval()
    expected = model(ids, memory=memory, disk_backed=True)['logits'][0, -1]
    hits = []
    def trace(layer, query, positions, selected):
        if layer == len(model.blocks) - 1: hits[:] = selected
    stream = ResonantStream(model, memory, page_directory=disk_root if memory.top_k else None, trace=trace)
    actual = stream.prefill(ids[0].tolist())
    # Measure rather than hide any numerical selection/output divergence.
    agrees = torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
    return {'input_tokens': ids.shape[1], 'logits_close': agrees,
            'top1_matches_dense': int(actual.argmax()) == int(expected.argmax()), 'max_logit_difference': float((actual - expected).abs().max()),
            'correct': int(actual.argmax()) == int(labels[0, -1]),
            'route_hit': int(targets[0, -1]) in hits if memory.top_k else None,
            'stats': stream.stats()}


def run(args, seed, mode):
    torch.manual_seed(seed)
    cfg = Config(vocab_size=VOCAB_SIZE, width=args.width, layers=args.layers, block_size=BLOCK_SIZE,
                 max_tokens=max(args.eval_blocks + args.train_blocks) * BLOCK_SIZE + 2,
                 position_mode='sinusoidal', gist_local_only=args.local_gists)
    model = ResonantLM(cfg)
    memory = policy(mode, args.width, max(args.eval_blocks + args.train_blocks), gist_slots=args.gist_slots)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator().manual_seed(10000 + seed)
    started = time.perf_counter()
    training = []
    for step in range(args.steps):
        model.train()
        # Deterministic rotation avoids different random length schedules across modes.
        blocks = args.train_blocks[step % len(args.train_blocks)]
        ids, labels, targets = batch(generator, args.batch_size, blocks, key_count=args.keys,
                                     value_count=args.values, contradictory=not args.no_contradictions, working_blocks=args.query_gap)
        source_labels, source_pages, _ = derive_batch_targets(ids, BLOCK_SIZE)
        if not torch.equal(source_labels, labels) or not torch.equal(source_pages, targets):
            raise RuntimeError('source-derived labels disagree with generator validation oracle')
        labels, targets = source_labels, source_pages
        supervised_route = mode in ('retrieval', 'full_supervised')
        result = model(ids, labels, memory=memory, disk_backed=True,
                       routing_targets=targets if supervised_route else None,
                       routing_weight=args.routing_weight if supervised_route else 0,
                       reconstruction_weight=0.05, precision_weight=0,
                       supervision_policy=policy('retrieval', args.width, max(args.eval_blocks), gist_slots=args.gist_slots)
                           if mode == 'full_supervised' else None)
        loss = result['losses']['total']
        if not torch.isfinite(loss): raise RuntimeError('nonfinite training loss')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            row = {'step': step + 1, 'total': float(loss.detach()),
                   'ce': float(result['losses']['ce'].detach()),
                   'routing': float(result['losses']['routing'].detach()),
                   'batch_accuracy': float((result['logits'][:, -1].argmax(-1) == labels[:, -1]).float().mean())}
            training.append(row)
            print(f'{mode} seed={seed} step={step+1} ce={row["ce"]:.3f} acc={row["batch_accuracy"]:.3f}',
                  file=sys.stderr, flush=True)
    elapsed = time.perf_counter() - started
    directory = Path(args.output) / f'{mode}-seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    torch.save({'model': model.state_dict(), 'config': asdict(cfg),
                'policy': {**asdict(memory), 'working_dtype': str(memory.working_dtype)},
                'steps': args.steps, 'seed': seed, 'mode': mode, 'arguments': vars(args),
                'optimizer': optimizer.state_dict(), 'data_rng_state': generator.get_state(),
                'torch_rng_state': torch.get_rng_state()}, directory / 'checkpoint.pt')
    evaluation = [evaluate(model, memory, mode, seed, blocks, args.examples,
                           min(args.batch_size, max(1, 4096 // (blocks * BLOCK_SIZE + 2))), args.evaluation_seed, args.keys, args.values, not args.no_contradictions, args.query_gap)
                  for blocks in args.eval_blocks]
    checks = [stream_check(model, memory, blocks, seed, directory / f'pages-{blocks}', args.keys, args.values, not args.no_contradictions, args.query_gap)
              for blocks in args.stream_blocks]
    return {'mode': mode, 'seed': seed, 'config': asdict(cfg), 'training_seconds': elapsed,
            'routing_supervision_weight': args.routing_weight if mode in ('retrieval','full_supervised') else 0,
            'training': training, 'evaluation': evaluation, 'stream_checks': checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=600)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--width', type=int, default=48)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--gist-slots', type=int, default=2)
    parser.add_argument('--query-gap', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.003)
    parser.add_argument('--routing-weight', type=float, default=0.5)
    parser.add_argument('--keys', type=int, default=256)
    parser.add_argument('--values', type=int, default=32)
    parser.add_argument('--no-contradictions', action='store_true')
    parser.add_argument('--local-gists', action='store_true')
    parser.add_argument('--train-blocks', type=int, nargs='+', default=[4,8,12])
    parser.add_argument('--eval-blocks', type=int, nargs='+', default=[4,8,12,32,128])
    parser.add_argument('--stream-blocks', type=int, nargs='*', default=[12,128])
    parser.add_argument('--examples', type=int, default=128)
    parser.add_argument('--evaluation-seed', type=int, default=1900000)
    parser.add_argument('--modes', nargs='+', choices=['full','full_supervised','local','gist','retrieval','retrieval_unsupervised'],
                        default=['full','local','gist','retrieval'])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.steps, args.seeds, args.width, args.layers, args.batch_size, args.examples) < 1:
        parser.error('positive steps/seeds/width/batch/examples required')
    if args.gist_slots < 0 or args.query_gap < 1: parser.error('invalid gist slots/query gap')
    if any(b <= args.query_gap for b in args.train_blocks + args.eval_blocks + args.stream_blocks):
        parser.error('all block counts must exceed query gap')
    if not 1 <= args.keys <= 256 or not 2 <= args.values <= 32:
        parser.error('keys must be 1..256; values 2..32')
    if any(not 2 <= b <= args.keys for b in args.train_blocks + args.eval_blocks + args.stream_blocks):
        parser.error('block counts must be between 2 and 256')
    if any(b > max(args.eval_blocks + args.train_blocks) for b in args.stream_blocks):
        parser.error('stream checks must fit configured positional capacity')
    torch.set_num_threads(1)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in range(args.seeds):
        for mode in args.modes:
            results.append(run(args, seed, mode))
            (output / 'results.json').write_text(json.dumps({'arguments': vars(args),
                'value_only_chance': 1 / args.values, 'results': results}, indent=2))
    print(json.dumps({'output': str(output), 'completed_runs': len(results)}))

if __name__ == '__main__': main()
