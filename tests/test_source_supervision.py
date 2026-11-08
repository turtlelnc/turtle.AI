import unittest
import torch
from tools.associative_recall import batch, PAIR, IGNORE, SEPARATOR
from tools.source_labels import derive_batch_targets, extract_facts, generate_examples
from models.resonant import Config, ResonantLM
from models.streaming_memory import MemoryConfig

class SourceSupervisionTests(unittest.TestCase):
    def test_derived_provenance(self):
        ids, expected, pages = batch(torch.Generator().manual_seed(24), 8, 64)
        labels, derived, spans = derive_batch_targets(ids)
        self.assertTrue(torch.equal(labels, expected))
        self.assertTrue(torch.equal(pages, derived))
        for row, (start, stop) in enumerate(spans):
            self.assertEqual(int(ids[row,start]), PAIR)
            self.assertEqual(stop-start, 4)
            self.assertEqual(int(ids[row,start+2]), int(labels[row,-1]))
        generated = list(generate_examples(ids[0,:-2].tolist(), 'document-1', max_queries=3))
        self.assertEqual(len(generated),3)
        for item in generated:
            self.assertEqual(item['source_id'],'document-1')
            self.assertLess(item['source_page'],62)
            self.assertEqual(item['input_ids'][item['source_span'][0]+2],item['answer'])
        with self.assertRaises(ValueError):
            extract_facts([PAIR,1,257,SEPARATOR]*2)
        with self.assertRaises(ValueError):
            extract_facts([PAIR,1,257,0])

    def test_auxiliary_supervision_does_not_change_full_visibility(self):
        torch.manual_seed(19)
        model = ResonantLM(Config(vocab_size=324,width=16,layers=3,block_size=8,
                                 max_tokens=64,gist_local_only=True))
        full = MemoryConfig(block_size=8,working_blocks=16,gist_slots=16,top_k=0)
        auxiliary = MemoryConfig(block_size=8,working_blocks=1,gist_slots=2,sink_tokens=1,top_k=2)
        ids, labels, pages = batch(torch.Generator().manual_seed(25), 2, 6, working_blocks=2)
        plain = model(ids, memory=full, disk_backed=True)['logits']
        out = model(ids,labels,memory=full,disk_backed=True,routing_targets=pages,
                    supervision_policy=auxiliary,routing_weight=.5)
        torch.testing.assert_close(out['logits'],plain,atol=0,rtol=0)
        self.assertEqual(out['losses']['routing_examples'],4)
        out['losses']['routing'].backward()
        self.assertGreater(model.blocks[1].attention.qkv.weight.grad.abs().sum().item(),0)

if __name__ == '__main__': unittest.main()
