"""Fixed-vocabulary, validation-gated matched-supervision curriculum."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import torch
from models.resonant import Config, ResonantLM
from tools.associative_recall import batch, VOCAB_SIZE
from tools.source_labels import derive_batch_targets
from tools.train_associative_memory import policy, evaluate, stream_check


def promotion(history, threshold, patience):
    return len(history) >= patience and all(x >= threshold for x in history[-patience:])


def run(args, seed, mode):
    torch.manual_seed(seed)
    cfg = Config(vocab_size=VOCAB_SIZE,width=48,layers=3,block_size=8,
                 max_tokens=max(args.stages)*8+2,position_mode='sinusoidal',gist_local_only=True)
    model = ResonantLM(cfg)
    memory = policy(mode,48,max(args.stages),gist_slots=2)
    supervision = policy('retrieval',48,max(args.stages),gist_slots=2)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.003)
    rng = torch.Generator().manual_seed(30000+seed)
    history, total_steps = [], 0
    directory = Path(args.output)/f'{mode}-{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    for stage_index, blocks in enumerate(args.stages):
        validations, promoted = [], False
        for step in range(1,args.stage_steps+1):
            model.train()
            # Deterministic rehearsal; same vocabulary, decoys and labels in every stage.
            length = args.stages[stage_index-1] if stage_index and step % 4 == 0 else blocks
            size = args.effective_batch_size or min(16,max(1,1024//(length*8+2)))
            ids,_,_ = batch(rng,size,length,working_blocks=2,key_count=args.keys,value_count=8)
            labels,pages,_ = derive_batch_targets(ids)
            optimizer.zero_grad()
            micro = args.microbatch if args.effective_batch_size else size
            for start in range(0,size,micro):
                end = min(size,start+micro)
                result = model(ids[start:end],labels[start:end],memory=memory,disk_backed=True,
                               routing_targets=pages[start:end],routing_weight=.5,
                               reconstruction_weight=.05,precision_weight=0,
                               supervision_policy=supervision if mode=='full_supervised' else None)
                loss = result['losses']['total']
                if not torch.isfinite(loss): raise RuntimeError('nonfinite curriculum loss')
                # Each microbatch is weighted by its actual example count, including remainders.
                (loss*((end-start)/size)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); total_steps += 1
            if step % args.check_every == 0 or step == args.stage_steps:
                validation = evaluate(model,memory,mode,seed,blocks,args.validation_examples,
                                      min(8,max(1,1024//(blocks*8+2))),
                                      args.validation_seed+stage_index*10000+step*101,args.keys,8,True,2)
                validations.append(validation['accuracy'])
                print(f'{mode} seed={seed} blocks={blocks} step={step} validation={validation["accuracy"]:.3f}',
                      file=sys.stderr,flush=True)
                promoted = promotion(validations,args.threshold,2)
                if promoted: break
        history.append({'blocks':blocks,'steps':step,'validation_accuracy':validations,
                        'promoted':promoted})
        torch.save({'model':model.state_dict(),'config':asdict(cfg),
                    'policy':{**asdict(memory),'working_dtype':str(memory.working_dtype)},
                    'optimizer':optimizer.state_dict(),'data_rng_state':rng.get_state(),
                    'torch_rng_state':torch.get_rng_state(),'history':history,
                    'arguments':vars(args),'mode':mode,'seed':seed},directory/'checkpoint.pt')
        # A failed competence gate halts length escalation, not the test report.
        if not promoted: break
    tests = [evaluate(model,memory,mode,seed,b,args.test_examples,min(8,max(1,1024//(b*8+2))),
                      args.test_seed,args.keys,8,True,2) for b in args.stages]
    check = stream_check(model,memory,args.stages[0],seed,directory/'pages',args.keys,8,True,2)
    return {'mode':mode,'seed':seed,'history':history,'total_steps':total_steps,
            'seconds':time.perf_counter()-started,'tests':tests,'stream_check':check}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--stages',type=int,nargs='+',default=[6,14,32])
    p.add_argument('--keys',type=int,default=128)
    p.add_argument('--stage-steps',type=int,default=1800)
    p.add_argument('--check-every',type=int,default=300)
    p.add_argument('--validation-examples',type=int,default=64)
    p.add_argument('--test-examples',type=int,default=128)
    p.add_argument('--effective-batch-size',type=int,default=None)
    p.add_argument('--microbatch',type=int,default=4)
    p.add_argument('--validation-seed',type=int,default=720000)
    p.add_argument('--test-seed',type=int,default=9300000)
    p.add_argument('--threshold',type=float,default=.95)
    args=p.parse_args()
    if (min(args.seeds,args.stage_steps,args.check_every,args.validation_examples,args.test_examples)<1 or
        not 0<args.threshold<=1 or not 3<=min(args.stages)<=max(args.stages)<=args.keys<=256 or
        args.stages!=sorted(set(args.stages))): p.error('invalid curriculum configuration')
    if args.microbatch<1 or (args.effective_batch_size is not None and args.effective_batch_size<1):
        p.error('batch sizes must be positive')
    torch.set_num_threads(1)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in range(args.seeds):
        for mode in ('full_supervised','retrieval'):
            results.append(run(args,seed,mode))
            temporary=root/'results.tmp'
            temporary.write_text(json.dumps({'arguments':vars(args),'results':results},indent=2))
            temporary.replace(root/'results.json')
    print(json.dumps({'runs':len(results),'output':str(root)}))

if __name__=='__main__':main()
