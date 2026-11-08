"""Frozen-encoder probes: decodability is not the model's own reading ability."""
import argparse,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from models.resonant import Config,ResonantLM
from models.checkpoints import load_model_and_memory
from tools.associative_recall import batch
from tools.source_labels import extract_facts


def documents(seed,count,blocks,roles):
    rng=torch.Generator().manual_seed(seed)
    ids,_,_=batch(rng,count,blocks,working_blocks=2,key_count=128,value_count=8)
    if roles!='fixed':
        records=ids[:,:blocks*8].reshape(count,blocks,2,4).clone()
        swaps=torch.rand(count,blocks,generator=rng)<.5
        records[swaps]=records[swaps].flip(1)
        ids[:,:blocks*8]=records.flatten(1)
    return ids


def fact_labels(ids,blocks):
    labels=[];true_offsets=[]
    for tokens in ids[:,:blocks*8].tolist():
        facts=extract_facts(tokens)
        by_page={fact.page:fact for fact in facts.values()}
        row=[];offsets=[]
        for page in range(blocks):
            fact=by_page[page];offset=fact.token_start-page*8+2
            other=page*8+(6 if offset==2 else 2)
            row.append([fact.key-1,fact.value-257,tokens[other]-257]);offsets.append(offset)
        labels.append(row);true_offsets.append(offsets)
    return torch.tensor(labels),torch.tensor(true_offsets)


@torch.no_grad()
def encode(model,memory,ids,blocks,feature_names=None,historical_only=False):
    names=feature_names or ['gist','contextual_mean','raw_bag','oracle_value']
    captured={};projected={};hooks=[]
    hooks.append(model.norm.register_forward_hook(lambda module,inputs,out:captured.update(hidden=out)))
    for layer_id,layer in enumerate(model.blocks):
        def capture(module,inputs,out,index=layer_id):projected[index]=out
        hooks.append(layer.attention.qkv.register_forward_hook(capture))
    features={name:[] for name in names};targets=[]
    try:
        for start in range(0,len(ids),32):
            source=ids[start:start+32];labels,offsets=fact_labels(source,blocks)
            model(source,memory=memory,disk_backed=True)
            h=captured.pop('hidden');rows=torch.arange(len(source))[:,None]
            pages=torch.arange(blocks)[None,:];base=pages*9
            tokens=source[:,:blocks*8].reshape(len(source),blocks,8)
            for name in names:
                if name=='gist':value=h[rows,base+8]
                elif name=='contextual_mean':value=h[rows[:,:,None],base[:,:,None]+torch.arange(8)].mean(2)
                elif name=='raw_bag':value=model.embedding(tokens.sort(-1).values).mean(2)
                elif name=='oracle_value':value=model.embedding(tokens.gather(-1,offsets[...,None]).squeeze(-1))
                elif name.startswith(('body_kv_','body_int4_kv_')):
                    index=int(name.rsplit('_',1)[1]);_,key,val=projected[index].chunk(3,-1)
                    key=key.to(memory.working_dtype).float()[rows[:,:,None],base[:,:,None]+torch.arange(8)]
                    val=val.to(memory.working_dtype).float()[rows[:,:,None],base[:,:,None]+torch.arange(8)]
                    if name.startswith('body_int4_'):
                        from models.resonant import PackedINT4
                        protected=(pages[:,:,None]*8+torch.arange(8))<memory.sink_tokens
                        key=torch.where(protected[...,None],key,PackedINT4.encode(key).decode().to(memory.working_dtype).float())
                        val=torch.where(protected[...,None],val,PackedINT4.encode(val).decode().to(memory.working_dtype).float())
                    value=torch.cat((key,val),-1).flatten(-2)
                elif name.startswith(('gist_k_','gist_v_','gist_kv_')):
                    index=int(name.rsplit('_',1)[1]);_,key,val=projected[index].chunk(3,-1)
                    key=key.to(memory.working_dtype).float()[rows,base+8]
                    val=val.to(memory.working_dtype).float()[rows,base+8]
                    value=torch.cat((key,val),-1) if name.startswith('gist_kv_') else key if name.startswith('gist_k_') else val
                else:raise ValueError(f'unknown probe feature: {name}')
                features[name].append(value[:,:blocks-2 if historical_only else blocks].flatten(0,1).cpu().clone())
            projected.clear();targets.append(labels[:,:blocks-2 if historical_only else blocks].flatten(0,1))
    finally:
        for hook in hooks:hook.remove()
    return {k:torch.cat(v) for k,v in features.items()},torch.cat(targets)


def make_probe(width,kind):
    return nn.Linear(width,144) if kind=='linear' else nn.Sequential(nn.Linear(width,128),nn.GELU(),nn.Linear(128,144))


def loss(logits,labels):
    return F.cross_entropy(logits[:,:128],labels[:,0])+F.cross_entropy(logits[:,128:136],labels[:,1])+F.cross_entropy(logits[:,136:],labels[:,2])


@torch.no_grad()
def metrics(probe,x,y):
    logits=probe(x);key=logits[:,:128].argmax(-1);value=logits[:,128:136].argmax(-1);decoy=logits[:,136:].argmax(-1)
    scores=logits[:,128:136];a=scores.gather(-1,y[:,1,None]).squeeze(-1);b=scores.gather(-1,y[:,2,None]).squeeze(-1)
    return {'key_accuracy':float((key==y[:,0]).float().mean()),'value_accuracy':float((value==y[:,1]).float().mean()),
            'decoy_accuracy':float((decoy==y[:,2]).float().mean()),
            'role_choice':float(((a>b).float()+.5*(a==b)).mean()),
            'both_values':float(((value==y[:,1])&(decoy==y[:,2])).float().mean()),
            'key_and_value':float(((key==y[:,0])&(value==y[:,1])).float().mean()),'loss':float(loss(logits,y))}


def fit(train_x,train_y,val_x,val_y,kind,seed,steps,shuffle=False):
    torch.manual_seed(seed);probe=make_probe(train_x.shape[-1],kind)
    rng=torch.Generator().manual_seed(7300000+seed)
    if shuffle:train_y=train_y[torch.randperm(len(train_y),generator=rng)]
    optimizer=torch.optim.AdamW(probe.parameters(),lr=.01 if kind=='linear' else .003)
    best=None;best_loss=float('inf')
    for step in range(steps):
        index=torch.randint(len(train_x),(256,),generator=rng)
        total=loss(probe(train_x[index]),train_y[index]);optimizer.zero_grad();total.backward();optimizer.step()
        if (step+1)%100==0 or step+1==steps:
            current=metrics(probe,val_x,val_y)['loss']
            if current<best_loss:best_loss=current;best={n:t.detach().clone() for n,t in probe.state_dict().items()}
    probe.load_state_dict(best);probe.eval();return probe


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True);p.add_argument('--replay',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=600);p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--features',nargs='+',default=['gist','contextual_mean','raw_bag','oracle_value'])
    p.add_argument('--train-roles',choices=['mixed','fixed'],default='mixed')
    args=p.parse_args()
    if min(args.steps,args.seeds)<1:p.error('positive steps/seeds required')
    torch.set_num_threads(1);root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    train_ids=documents(6400000,2000,5,args.train_roles);val_ids=documents(6500000,256,5,args.train_roles)
    test_ids=documents(6600000,512,5,'mixed')
    swapped=test_ids.clone();swapped[:,:40]=swapped[:,:40].reshape(512,5,2,4).flip(2).flatten(1)
    test_sets={'mixed':(test_ids,5),'paired_swapped':(swapped,5),
               'fixed_order':(documents(6700000,512,5,'fixed'),5),
               'longer':(documents(6800000,256,10,'mixed'),10)}
    parent,memory=load_model_and_memory(args.parent)
    torch.manual_seed(2026);random=ResonantLM(parent.cfg).eval()
    replay,replay_memory=load_model_and_memory(args.replay)
    results=[]
    for encoder,model,policy in [('parent',parent,memory),('replay',replay,replay_memory),('random',random,memory)]:
        model.requires_grad_(False);snapshot={n:p.detach().clone() for n,p in model.named_parameters()}
        train,train_y=encode(model,policy,train_ids,5,args.features);val,val_y=encode(model,policy,val_ids,5,args.features)
        tests={name:encode(model,policy,ids,blocks,args.features) for name,(ids,blocks) in test_sets.items()}
        if not torch.equal(tests['mixed'][1],tests['paired_swapped'][1]):raise RuntimeError('role swap changed semantic labels')
        for feature in train:
            center=train[feature].mean(0);scale=train[feature].std(0).clamp_min(1e-5)
            x=(train[feature]-center)/scale;vx=(val[feature]-center)/scale
            for kind in ('linear','mlp'):
                for seed in range(args.seeds):
                    probe=fit(x,train_y,vx,val_y,kind,seed,args.steps)
                    row={'encoder':encoder,'feature':feature,'probe':kind,'seed':seed,'validation':metrics(probe,vx,val_y),
                         'tests':{name:metrics(probe,(data[feature]-center)/scale,y) for name,(data,y) in tests.items()}}
                    original=probe((tests['mixed'][0][feature]-center)/scale)[:,128:136].argmax(-1)
                    alternate=probe((tests['paired_swapped'][0][feature]-center)/scale)[:,128:136].argmax(-1)
                    row['paired_prediction_agreement']=float((original==alternate).float().mean())
                    truth=tests['mixed'][1][:,1]
                    row['paired_value_correct']=float(((original==truth)&(alternate==truth)).float().mean())
                    results.append(row)
            print(encoder,feature,'completed',flush=True)
        # Shuffle the entire label tuple; select a representative encoded feature.
        control='gist' if 'gist' in train else args.features[0]
        center=train[control].mean(0);scale=train[control].std(0).clamp_min(1e-5)
        probe=fit((train[control]-center)/scale,train_y,(val[control]-center)/scale,val_y,'mlp',0,args.steps,True)
        results.append({'encoder':encoder,'feature':control+'_label_shuffle','probe':'mlp','seed':0,
                        'tests':{name:metrics(probe,(data[control]-center)/scale,y) for name,(data,y) in tests.items()}})
        if any(not torch.equal(snapshot[n],parameter) for n,parameter in model.named_parameters()):raise RuntimeError('encoder was modified')
        (root/'results.json').write_text(json.dumps({'arguments':vars(args),'encoder_unchanged':True,'results':results},indent=2))

if __name__=='__main__':main()
