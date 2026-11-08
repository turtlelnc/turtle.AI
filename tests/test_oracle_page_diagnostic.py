import unittest
import torch
from models.resonant import Config, ResonantLM
from tools.train_associative_memory import policy
from tools.associative_recall import batch

class OraclePageTests(unittest.TestCase):
    def test_only_annotated_past_page_and_self_are_visible(self):
        torch.set_num_threads(1)
        model=ResonantLM(Config(vocab_size=324,width=8,layers=3,block_size=8,max_tokens=40))
        ids,_,pages=batch(torch.Generator().manual_seed(66),2,4,working_blocks=2)
        model(ids,memory=policy('retrieval',8,4),disk_backed=True,
              routing_targets=pages,oracle_page_only=True)
        for layer in model.blocks:
            mask=layer.attention.last_mask
            for row in range(2):
                visible=mask[row,-1].nonzero().flatten().tolist()
                page=int(pages[row,-1]);expected=list(range(page*9,page*9+9))+[mask.shape[-1]-1]
                self.assertEqual(visible,expected)
            self.assertFalse(mask.triu(1).any())

    def test_inclusion_oracle_preserves_already_retrieved_set(self):
        torch.set_num_threads(1);torch.manual_seed(67)
        model=ResonantLM(Config(vocab_size=324,width=8,layers=1,block_size=8,max_tokens=40))
        memory=policy('retrieval',8,4)
        ids,_,_=batch(torch.Generator().manual_seed(68),2,4,working_blocks=2)
        original=model(ids,memory=memory,disk_backed=True)['logits']
        mask=model.blocks[0].attention.last_mask.clone()
        target=torch.full_like(ids,-100)
        target[:,-1]=model.blocks[0].attention.last_retrieved[:,-1,1]
        oracle=model(ids,memory=memory,disk_backed=True,routing_targets=target,oracle_routes=True)['logits']
        torch.testing.assert_close(original,oracle,atol=0,rtol=0)
        self.assertTrue(torch.equal(mask,model.blocks[0].attention.last_mask))

if __name__=='__main__':unittest.main()
