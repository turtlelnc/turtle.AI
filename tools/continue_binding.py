"""Validation-selected continuation; separate from the fixed-budget seed screen."""
import argparse
import json
from pathlib import Path
import torch
from models.checkpoints import load_model_and_memory
from tools.associative_recall import batch
from tools.source_labels import derive_batch_targets
from tools.train_associative_memory import policy,evaluate
from tools.curriculum_memory import promotion


def main():
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--output',required=True);p.add_argument('--advance',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    results=[];output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    for path in sorted(list(Path(args.root).glob('*/checkpoint.pt'))+list(Path(args.root).glob('*.pt'))):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        history=saved['history'][-1];v=history['validation_accuracy']
        # Selection uses validation only, never the final test results.
        if args.advance:
            if not promotion(saved.get('continuation_validation',[]),.95,2):continue
        elif (history['promoted'] or promotion(saved.get('continuation_validation',[]),.95,2) or
              len(v)<2 or min(v[-2:])<.75 or v[-1]<v[-2]):continue
        model,memory=load_model_and_memory(path)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.003);optimizer.load_state_dict(saved['optimizer'])
        rng=torch.Generator();rng.set_state(saved['data_rng_state']);torch.set_rng_state(saved['torch_rng_state'])
        source=saved['arguments']
        if source.get('effective_batch_size')!=16 or source.get('microbatch')!=4 or source['keys']!=128:
            raise ValueError('continuation is defined for the fixed-128-key, 16/4 batch protocol only')
        blocks=saved.get('continuation_current_blocks',history['blocks'])+int(args.advance);seed=saved['seed'];mode=saved['mode']
        if blocks*8+2>model.cfg.max_tokens:raise ValueError('advanced stage exceeds checkpoint capacity')
        supervision=policy('retrieval',48,max(source['stages']),gist_slots=2)
        validations=[]
        for step in range(1,1201):
            model.train();ids,_,_=batch(rng,16,blocks,working_blocks=2,key_count=128,value_count=8)
            labels,pages,_=derive_batch_targets(ids);optimizer.zero_grad()
            for start in range(0,16,4):
                loss=model(ids[start:start+4],labels[start:start+4],memory=memory,disk_backed=True,
                           routing_targets=pages[start:start+4],routing_weight=.5,reconstruction_weight=.05,
                           precision_weight=0,supervision_policy=supervision if mode=='full_supervised' else None)['losses']['total']
                if not torch.isfinite(loss):raise RuntimeError('nonfinite continuation loss')
                (loss/4).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            if step%200==0:
                val=evaluate(model,memory,mode,seed,blocks,64,8,(3720000 if args.advance else 2720000)+step*101,128,8,True,2)
                validations.append(val['accuracy'])
                print(mode,seed,step,val['accuracy'],flush=True)
                if promotion(validations,.95,2):break
        test=evaluate(model,memory,mode,seed,blocks,128,8,(14300000 if args.advance else 12300000),128,8,True,2)
        saved.update(model=model.state_dict(),optimizer=optimizer.state_dict(),data_rng_state=rng.get_state(),
                     torch_rng_state=torch.get_rng_state(),continuation_validation=validations,
                     continuation_steps=step,continuation_current_blocks=blocks)
        torch.save(saved,output/f'{mode}-{seed}.pt')
        results.append({'source':str(path),'mode':mode,'seed':seed,'additional_steps':step,'blocks':blocks,'advanced':args.advance,
                        'validation':validations,'passed':promotion(validations,.95,2),'test':test})
    (output/'results.json').write_text(json.dumps({'selection':'failed gates with last two validation scores >=.75 and nondecreasing',
                                                   'results':results},indent=2))

if __name__=='__main__':main()
