"""Aggregate all seeds; add a query-blind source-value majority diagnostic."""
import argparse
import json
from statistics import mean
from pathlib import Path
import torch
from tools.associative_recall import batch, BLOCK_SIZE


def majority_baseline(arguments, seed, blocks):
    rng = torch.Generator().manual_seed(arguments['evaluation_seed'] + 1000 * seed + blocks)
    examples = arguments['examples']
    batch_size = min(arguments['batch_size'], max(1, 4096 // (blocks * BLOCK_SIZE + 2)))
    correct = 0
    for start in range(0, examples, batch_size):
        ids, labels, _ = batch(rng, min(batch_size, examples - start), blocks,
                              working_blocks=arguments['query_gap'], key_count=arguments['keys'],
                              value_count=arguments['values'], contradictory=not arguments['no_contradictions'])
        records = ids[:, :-2].reshape(len(ids), blocks, BLOCK_SIZE)
        # Ignore Query identity, but exploit which values occur most in candidate facts.
        guess = records[:, :blocks - arguments['query_gap'], 2].mode(-1).values
        correct += int((guess == labels[:, -1]).sum())
    return correct / examples


def summarize(document):
    args, runs = document['arguments'], document['results']
    result = {'protocol': args, 'value_only_chance': document['value_only_chance'], 'lengths': []}
    for blocks in args['eval_blocks']:
        row = {'blocks': blocks, 'tokens': blocks * BLOCK_SIZE + 2, 'modes': {}}
        seeds = sorted({run['seed'] for run in runs})
        row['query_blind_majority_accuracy'] = mean(majority_baseline(args, seed, blocks) for seed in seeds)
        for mode in args['modes']:
            samples = [e for run in runs if run['mode'] == mode for e in run['evaluation'] if e['blocks'] == blocks]
            if not samples: continue
            metrics = {}
            for field in ('accuracy','route_hit_rate','oracle_page_accuracy','counterfactual_joint_accuracy',
                          'fp32_cold_top1_disagreement','position_prior_topk_chance'):
                values = [sample[field] for sample in samples if sample[field] is not None]
                metrics[field] = {'mean': mean(values), 'min': min(values), 'max': max(values)} if values else None
            row['modes'][mode] = {'runs': len(samples), 'metrics': metrics}
        result['lengths'].append(row)
    checks = [check for run in runs for check in run['stream_checks']]
    result['streaming'] = {'checks': len(checks), 'all_top1_match': all(c['top1_matches_dense'] for c in checks),
                           'max_logit_difference': max((c['max_logit_difference'] for c in checks), default=None)}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='+')
    args = parser.parse_args()
    torch.set_num_threads(1)
    documents = [json.loads(Path(path).read_text()) for path in args.results]
    combined = documents[0]
    modes = list(combined['arguments']['modes'])
    seen = {(run['mode'], run['seed']) for run in combined['results']}
    for document in documents[1:]:
        for key, value in combined['arguments'].items():
            if key not in ('output', 'modes') and document['arguments'][key] != value:
                raise ValueError(f'incompatible experiment protocols: {key}')
        for run in document['results']:
            identity = (run['mode'], run['seed'])
            if identity in seen: raise ValueError(f'duplicate run: {identity}')
            seen.add(identity)
            combined['results'].append(run)
        modes.extend(mode for mode in document['arguments']['modes'] if mode not in modes)
    combined['arguments']['modes'] = modes
    print(json.dumps(summarize(combined), indent=2))

if __name__ == '__main__': main()
