"""Untrained distance extension of a validation-qualified sinusoidal checkpoint."""
import argparse
from dataclasses import asdict
import json
import torch
from models.resonant import Config, ResonantLM
from tools.train_associative_memory import policy, evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('checkpoint');p.add_argument('--blocks',type=int,nargs='+',default=[64,128])
    p.add_argument('--examples',type=int,default=32)
    args=p.parse_args();torch.set_num_threads(1)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if not all(stage['promoted'] for stage in saved['history']):
        raise ValueError('extension screen requires all curriculum gates passed')
    cfg=dict(saved['config'])
    if cfg['position_mode']!='sinusoidal':raise ValueError('cannot extend an untrained learned table')
    source=saved['arguments']
    if args.examples<1 or any(b<3 or b>source['keys'] for b in args.blocks):raise ValueError('invalid extension')
    cfg['max_tokens']=max(cfg['max_tokens'],max(args.blocks)*8+2)
    model=ResonantLM(Config(**cfg))
    weights=dict(saved['model']);old_positions=weights.pop('position.weight')
    torch.testing.assert_close(model.position.weight[:len(old_positions)],old_positions)
    missing=model.load_state_dict(weights,strict=False)
    if missing.missing_keys!=['position.weight'] or missing.unexpected_keys:raise ValueError('weight mismatch')
    memory=policy(saved['mode'],cfg['width'],max(args.blocks),gist_slots=2)
    results=[evaluate(model,memory,saved['mode'],saved['seed'],b,args.examples,1,
                      9500000,source['keys'],8,True,2) for b in args.blocks]
    print(json.dumps({'checkpoint':args.checkpoint,'seed':saved['seed'],'mode':saved['mode'],
        'scope':'validation-qualified single checkpoint; sinusoidal capacity extension without training',
        'training_max_tokens':max(source['stages'])*8+2,'results':results},indent=2))

if __name__=='__main__':main()
