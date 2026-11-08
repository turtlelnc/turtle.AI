"""All-condition LM summary and same-input auxiliary-reader diagnostics."""
import argparse,json
from pathlib import Path
from statistics import mean
import torch
from models.checkpoints import load_model_and_memory
from tools.replay_binding import sample
from tools.source_labels import derive_relation_targets


@torch.no_grad()
def reader_metrics(model,memory,seed,blocks,swap=False,test_base=43300000):
    rng=torch.Generator().manual_seed(test_base+blocks*10000+seed*1000+blocks)
    augment=torch.Generator().manual_seed(test_base+900000+seed*1000+blocks)
    answer=all_correct=historic_correct=queried_correct=0;count=0
    for _ in range(16):
        ids,labels,pages=sample(rng,8,blocks)
        if swap:
            ids=ids.clone();records=ids[:,:blocks*8].reshape(8,blocks,2,4).clone()
            if swap=='mixed':
                choices=torch.rand(8,blocks,generator=augment)<.5
                records[choices]=records[choices].flip(1)
            else:records=records.flip(2)
            ids[:,:blocks*8]=records.flatten(1)
        truth=derive_relation_targets(ids)
        out=model(ids,memory=memory,disk_backed=True,relation_targets=truth)
        pred=out['relation_logits'].argmax(-1)
        all_correct+=int((pred==truth).sum());historic_correct+=int((pred[:,:blocks-2]==truth[:,:blocks-2]).sum())
        row=torch.arange(8);source=pages[:,-1]
        queried_correct+=int((pred[row,source]==labels[:,-1]).sum())
        answer+=int((out['logits'][:,-1].argmax(-1)==labels[:,-1]).sum());count+=8
    return {'native_answer':answer/count,'head_all_pages':all_correct/(count*blocks),
            'head_historical_pages':historic_correct/(count*(blocks-2)),
            'head_true_query_page':queried_correct/count}


def main():
    p=argparse.ArgumentParser();p.add_argument('root');args=p.parse_args();root=Path(args.root)
    data=json.loads((root/'results.json').read_text())
    if len(data['results'])!=data['arguments']['seeds']*4:raise ValueError('wait for all conditions')
    torch.set_num_threads(1)
    for row in data['results']:
        directory=root/f"{row['mode']}-aux{int(row['auxiliary'])}-{row['adaptation_seed']}"
        model,memory=load_model_and_memory(directory/'checkpoint.pt')
        base=data['arguments'].get('test_base',43300000)
        row['reader_diagnostic']={str(b):reader_metrics(model,memory,row['adaptation_seed'],b,test_base=base) for b in (4,5)}
        row['swapped_diagnostic']={str(b):reader_metrics(model,memory,row['adaptation_seed'],b,True,base) for b in (4,5)}
        row['mixed_diagnostic']={str(b):reader_metrics(model,memory,row['adaptation_seed'],b,'mixed',base) for b in (4,5)}
        for b in ('4','5'):
            if row['reader_diagnostic'][b]['native_answer']!=row['tests'][b]['accuracy']:raise RuntimeError('diagnostic altered answers')
    summary={}
    for mode in ('full_supervised','retrieval'):
        for auxiliary in (False,True):
            rows=[r for r in data['results'] if r['mode']==mode and r['auxiliary']==auxiliary]
            summary[f'{mode}-aux{int(auxiliary)}']={b:{
                'native_mean':mean(r['tests'][b]['accuracy'] for r in rows),
                'native_min':min(r['tests'][b]['accuracy'] for r in rows),
                'native_max':max(r['tests'][b]['accuracy'] for r in rows),
                'joint_answer_mean':mean(r['tests'][b]['counterfactual_joint_accuracy'] for r in rows),
                'true_page_head_mean':mean(r['reader_diagnostic'][b]['head_true_query_page'] for r in rows),
                'mixed_native_mean':mean(r['mixed_diagnostic'][b]['native_answer'] for r in rows),
                'mixed_true_page_head_mean':mean(r['mixed_diagnostic'][b]['head_true_query_page'] for r in rows),
                'swapped_native_mean':mean(r['swapped_diagnostic'][b]['native_answer'] for r in rows),
                'swapped_true_page_head_mean':mean(r['swapped_diagnostic'][b]['head_true_query_page'] for r in rows)
                } for b in ('4','5')}
    checks=[c for r in data['results'] for c in r['stream_checks']]
    result={'summary':summary,'stream_top1_all_match':all(c['top1_matches_dense'] for c in checks),
            'stream_checks':len(checks),'runs':data['results']}
    (root/'analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2))

if __name__=='__main__':main()
