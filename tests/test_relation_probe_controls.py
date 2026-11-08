import unittest
import torch
from models.resonant import Config,ResonantLM
from models.streaming_memory import MemoryConfig
from tools.associative_recall import PAIR,IGNORE
from tools.probe_gist_relations import documents,encode
from tools.probe_candidate_relations import RelationProbe,report

class RelationControlTests(unittest.TestCase):
    def test_marker_flip_changes_truth_but_bag_cannot_discriminate(self):
        torch.set_num_threads(1);torch.manual_seed(71)
        model=ResonantLM(Config(vocab_size=324,width=8,layers=3,block_size=8,max_tokens=50)).eval()
        ids=documents(72,2,5,'mixed');flipped=ids.clone()
        records=flipped[:,:40].reshape(2,5,2,4).clone()
        records[:,:,:,0]=torch.where(records[:,:,:,0]==PAIR,IGNORE,PAIR)
        flipped[:,:40]=records.flatten(1)
        a,y=encode(model,MemoryConfig(block_size=8),ids,5,['raw_bag'])
        b,z=encode(model,MemoryConfig(block_size=8),flipped,5,['raw_bag'])
        self.assertTrue(torch.equal(a['raw_bag'],b['raw_bag']))
        self.assertTrue(torch.equal(y[:,1],z[:,2]));self.assertTrue(torch.equal(y[:,2],z[:,1]))
        probe=RelationProbe(8,8,'mlp')
        with torch.no_grad():positive=model.embedding(y[:,1]+257);negative=model.embedding(y[:,2]+257)
        first=report(probe,a['raw_bag'],positive,negative)['pair_choice']
        second=report(probe,b['raw_bag'],negative,positive)['pair_choice']
        self.assertAlmostEqual((first+second)/2,.5,places=6)

if __name__=='__main__':unittest.main()
