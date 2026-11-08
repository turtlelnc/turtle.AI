"""Isolate same-key PAIR/IGNORE readout without any cross-page selection."""
import json
import argparse
import torch
from models.resonant import Config, ResonantLM
from tools.associative_recall import batch, QUERY
from tools.train_associative_memory import policy


def examples(rng,size,shuffle=False):
    records,_,_=batch(rng,size,2,key_count=128,value_count=8)
    source=records[:,:8].clone()
    answers=source[:,2].clone()
    if shuffle:
        swap=torch.rand(size,generator=rng)<.5
        source[swap]=source[swap].reshape(-1,2,4).flip(1).flatten(1)
    query=torch.stack((torch.full((size,),QUERY),source[:,1]),-1)
    ids=torch.cat((source,query),-1)
    labels=torch.full_like(ids,-100);labels[:,-1]=answers
    return ids,labels


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shuffle-roles',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(0)
    model=ResonantLM(Config(vocab_size=324,width=48,layers=3,block_size=8,
                           max_tokens=24,position_mode='sinusoidal',gist_local_only=True))
    memory=policy('full_supervised',48,2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.003)
    rng=torch.Generator().manual_seed(41000)
    for _ in range(800):
        ids,labels=examples(rng,16,args.shuffle_roles)
        loss=model(ids,labels,memory=memory,disk_backed=True,
                   reconstruction_weight=.05,precision_weight=0)['losses']['total']
        optimizer.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    model.eval();ids,labels=examples(torch.Generator().manual_seed(9400000),256,args.shuffle_roles)
    with torch.no_grad():
        output=model(ids,memory=memory,disk_backed=True)['logits'][:,-1]
    print(json.dumps({'scope':'single-page role/readout only; one seed; 800 steps',
                      'shuffled_roles':args.shuffle_roles,'heldout_examples':256,'accuracy':float((output.argmax(-1)==labels[:,-1]).float().mean()),
                      'value_only_chance':.125}))

if __name__=='__main__':main()
