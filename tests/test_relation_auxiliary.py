import unittest
import torch
from models.resonant import Config,ResonantLM
from tools.associative_recall import batch
from tools.train_associative_memory import policy
from tools.source_labels import derive_relation_targets

class RelationAuxTests(unittest.TestCase):
    def test_relation_targets_follow_markers_after_record_swap(self):
        ids,_,_=batch(torch.Generator().manual_seed(74),2,4,working_blocks=2)
        targets=derive_relation_targets(ids)
        swapped=ids.clone()
        swapped[:,:32]=ids[:,:32].reshape(2,4,2,4).flip(2).flatten(1)
        self.assertTrue(torch.equal(targets,derive_relation_targets(swapped)))

    def test_aux_changes_gradients_not_answer_forward(self):
        torch.set_num_threads(1);torch.manual_seed(73)
        m=ResonantLM(Config(vocab_size=324,width=8,layers=3,block_size=8,max_tokens=50,relation_reader=True))
        ids,labels,_=batch(torch.Generator().manual_seed(74),2,4,working_blocks=2)
        targets=derive_relation_targets(ids)
        self.assertTrue(torch.equal(targets,ids[:,:32].reshape(2,4,8)[:,:,2]))
        memory=policy('retrieval',8,5)
        original=m(ids,memory=memory,disk_backed=True)['logits']
        out=m(ids,labels,memory=memory,disk_backed=True,relation_targets=targets,relation_weight=.5)
        torch.testing.assert_close(original,out['logits'],atol=0,rtol=0)
        out['losses']['relation'].backward()
        self.assertGreater(m.relation_reader[0].weight.grad.abs().sum().item(),0)
        self.assertGreater(m.blocks[-1].attention.qkv.weight.grad.abs().sum().item(),0)

if __name__=='__main__':unittest.main()
