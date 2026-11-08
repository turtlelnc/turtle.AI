"""Offline annotated-page-only mask ablation; never production retrieval."""
import argparse
import json
from pathlib import Path
import torch
from models.checkpoints import load_model_and_memory
from tools.associative_recall import batch
from tools.source_labels import derive_batch_targets


def main():
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args()
    torch.set_num_threads(1)
    results=[]
    for checkpoint in sorted(Path(args.root).glob('*/checkpoint.pt')):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        model,memory=load_model_and_memory(checkpoint)
        rng=torch.Generator().manual_seed(11300000+saved['seed'])
        scores={'normal':0,'known_page_only':0}
        for _ in range(8):
            ids,_,_=batch(rng,8,4,working_blocks=2,key_count=128,value_count=8)
            labels,pages,_=derive_batch_targets(ids)
            with torch.no_grad():
                for name,oracle in [('normal',False),('known_page_only',True)]:
                    out=model(ids,memory=memory,disk_backed=True,
                              routing_targets=pages if oracle else None,oracle_page_only=oracle)['logits'][:,-1]
                    scores[name]+=int((out.argmax(-1)==labels[:,-1]).sum())
        results.append({'checkpoint':str(checkpoint),'mode':saved['mode'],'seed':saved['seed'],
                        'examples':64,**{k:v/64 for k,v in scores.items()}})
    print(json.dumps({'scope':'ground-truth-page diagnostic only, not an achievable retrieval score',
                      'results':results},indent=2))

if __name__=='__main__':main()
