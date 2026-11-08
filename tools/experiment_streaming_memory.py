"""Controlled KV paging screen: known embedding keys, NOT learned semantic retrieval."""
import argparse
import json
import math
from statistics import mean, median
import time
import tempfile
import torch
from models.streaming_memory import MemoryConfig, MemoryPool, ResonantStream
from models.resonant import Config, ResonantLM


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def screen(length, seed, cold_bytes, disk_directory=None):
    width, block_size = 64, 32
    cfg = MemoryConfig(block_size=block_size, working_blocks=2, sink_tokens=8,
                       gist_slots=8, cold_bytes=cold_bytes, top_k=2)
    if disk_directory is None:
        pool = MemoryPool(width, cfg)
    else:
        from models.disk_pages import DiskPageStore
        pool = MemoryPool(width, cfg, page_store=DiskPageStore(disk_directory, width))
    rng = torch.Generator().manual_seed(seed)
    blocks = length // block_size
    keys = torch.nn.functional.normalize(torch.randn(blocks, width, generator=rng), dim=-1)
    labels = torch.randint(width, (blocks,), generator=rng)
    values = torch.nn.functional.one_hot(labels, width).float()
    started = time.perf_counter()
    for bid in range(blocks):
        for offset in range(block_size):
            pool.append(bid * (block_size + 1) + offset, bid, keys[bid], values[bid])
        pool.finish(bid * (block_size + 1) + block_size, bid, keys[bid], values[bid])
        stats = pool.stats()
        assert stats['persistent_payload_bytes'] <= stats['payload_bound_bytes']
        assert stats['cold_payload_bytes'] <= cold_bytes
    insertion_seconds = time.perf_counter() - started
    # Test every historical block outside the hot window, including evicted pages.
    targets = list(range(blocks - cfg.working_blocks))
    retained = {page.block_id for page in pool.cold}
    route_hits, retained_hits, packed_hits, restored_hits = 0, 0, 0, 0
    accuracy = {name: [] for name in ('full', 'local', 'gist', 'retrieval_int4', 'retrieval_original_fp32')}
    latencies = {name: [] for name in accuracy}
    # Full baseline is a separate unbounded tensor store in the test harness.
    full_keys = keys.repeat_interleave(block_size, dim=0)
    full_values = values.repeat_interleave(block_size, dim=0)
    local_items = pool.anchors + [token for _, tokens, _ in pool.hot for token in tokens]
    gist_items = pool._working()
    def tensors(items):
        return torch.stack([t.key.float() for t in items]), torch.stack([t.value.float() for t in items])
    local_keys, local_values = tensors(local_items)
    gist_keys, gist_values = tensors(gist_items)
    def attention(query, k, v):
        return (k @ query / math.sqrt(width)).softmax(-1) @ v
    def original(bid, positions):
        return keys[bid].repeat(len(positions), 1), values[bid].repeat(len(positions), 1)
    for bid in targets:
        query = keys[bid] * 128
        hit = bid in [page.block_id for page in pool.retrieve(query)]
        route_hits += hit
        retained_hits += hit and bid in retained
        options = {
            'full': lambda: attention(query, full_keys, full_values),
            'local': lambda: attention(query, local_keys, local_values),
            'gist': lambda: attention(query, gist_keys, gist_values),
            'retrieval_int4': lambda: pool.attend(query)[0],
            # Oracle originals in the harness: NOT text JIT-prefill.
            'retrieval_original_fp32': lambda: pool.attend(query, original)[0],
        }
        for name, operation in options.items():
            started = time.perf_counter()
            output = operation()
            latencies[name].append((time.perf_counter() - started) * 1000)
            accuracy[name].append(int(output.argmax() == labels[bid]))
        packed_hits += accuracy['retrieval_int4'][-1] if bid in retained else 0
        restored_hits += accuracy['retrieval_original_fp32'][-1] if bid in retained else 0
    assert pool.peak_attention_tokens <= pool.stats()['attention_token_bound']
    retained_targets = len(set(targets) & retained)
    result = {'length': length, 'seed': seed, 'cold_budget_bytes': cold_bytes,
              'historical_queries': len(targets), 'retained_historical_pages': retained_targets,
              'page_retention_coverage': retained_targets / len(targets),
              'page_hit_rate_all': route_hits / len(targets),
              'page_hit_rate_retained': retained_hits / retained_targets if retained_targets else None,
              'value_accuracy_retained_int4': packed_hits / retained_targets if retained_targets else None,
              'value_accuracy_retained_original_fp32': restored_hits / retained_targets if retained_targets else None,
              'value_accuracy_all': {name: mean(values) for name, values in accuracy.items()},
              'query_latency_ms': {name: {'median': median(values), 'p95': percentile(values, .95)}
                                   for name, values in latencies.items()},
              'insertion_seconds': insertion_seconds, 'stats': pool.stats(),
              'full_baseline_kv_payload_bytes': full_keys.numel() * 4 + full_values.numel() * 4}
    return result


def neural_smoke(length, disk_directory=None):
    torch.manual_seed(27)
    model = ResonantLM(Config(vocab_size=16, width=16, layers=1, block_size=32,
                             max_tokens=length)).eval()
    cfg = MemoryConfig(block_size=32, working_blocks=2, sink_tokens=8,
                       gist_slots=8, cold_bytes=16384, top_k=1,
                       working_dtype=torch.float16)
    stream = ResonantStream(model, cfg, page_directory=disk_directory)
    started = time.perf_counter()
    for i in range(length):
        logits = stream.push(i % 16)
        if not torch.isfinite(logits).all(): raise RuntimeError('nonfinite streaming logits')
    stats = stream.stats()
    assert stats['persistent_payload_bytes'] <= stats['payload_bound_bytes']
    return {'length': length, 'seconds': time.perf_counter() - started,
            'stats': stats, 'note': 'random weights; integration/budget test only, no quality metric'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lengths', type=int, nargs='+', default=[1024,4096,16384])
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--cold-bytes', type=int, default=65536)
    parser.add_argument('--neural-smoke', action='store_true')
    parser.add_argument('--disk-backed', action='store_true')
    args = parser.parse_args()
    if args.seeds < 1 or args.cold_bytes < 0 or any(n < 96 or n % 32 for n in args.lengths):
        parser.error('positive seeds, nonnegative cold budget and lengths >=96 divisible by 32 required')
    torch.set_num_threads(1)
    results = []
    for length in args.lengths:
        for seed in range(args.seeds):
            if args.disk_backed:
                with tempfile.TemporaryDirectory(prefix='kv-pages-') as directory:
                    result = screen(length, seed, args.cold_bytes, directory)
            else:
                result = screen(length, seed, args.cold_bytes)
            results.append(result)
            print(f'completed KV screen length={length} seed={seed}', file=__import__('sys').stderr, flush=True)
    smoke = []
    if args.neural_smoke:
        for length in args.lengths:
            if args.disk_backed:
                with tempfile.TemporaryDirectory(prefix='kv-smoke-') as directory:
                    smoke.append(neural_smoke(length, directory))
            else:
                smoke.append(neural_smoke(length))
            print(f'completed neural smoke length={length}', file=__import__('sys').stderr, flush=True)
    print(json.dumps({'scope': 'known-key CPU paging screen; no learned retrieval or inversion claim',
                      'disk_backed': args.disk_backed, 'results': results, 'neural_smoke': smoke}, indent=2))

if __name__ == '__main__': main()
