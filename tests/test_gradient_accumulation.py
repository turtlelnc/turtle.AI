import unittest
import torch
from models.resonant import Config, ResonantLM
from tools.train_associative_memory import policy
from tools.associative_recall import batch

class AccumulationTests(unittest.TestCase):
    def test_remainder_matches_full_batch_gradient(self):
        torch.set_num_threads(1);torch.manual_seed(66)
        cfg=Config(vocab_size=324,width=8,layers=3,block_size=8,max_tokens=40,gist_local_only=True)
        model=ResonantLM(cfg)
        memory=policy('retrieval',8,4)
        ids,labels,pages=batch(torch.Generator().manual_seed(33),5,4,working_blocks=2)
        def loss(a,b):
            return model(ids[a:b],labels[a:b],memory=memory,disk_backed=True,
                         routing_targets=pages[a:b],routing_weight=.5)['losses']['total']
        loss(0,5).backward()
        reference={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad()
        for a,b in [(0,2),(2,4),(4,5)]: (loss(a,b)*(b-a)/5).backward()
        for n,p in model.named_parameters():
            if n in reference:torch.testing.assert_close(p.grad,reference[n],atol=2e-6,rtol=2e-5)

if __name__=='__main__':unittest.main()
