import unittest
import torch
from models.resonant import Config, PackedINT4, ResonantLM

torch.set_num_threads(1)

class NativeMemoryTests(unittest.TestCase):
    def model(self, top_k=1):
        torch.manual_seed(7)
        return ResonantLM(Config(vocab_size=16, width=16, layers=2,
                                 block_size=3, window=2, sinks=1, top_k=top_k, max_tokens=32))

    def test_packed(self):
        x = torch.tensor([[0., -2., 1., 2., -1.]])
        packed = PackedINT4.encode(x)
        self.assertEqual(packed.storage_bytes(), 7)
        self.assertTrue(((packed.decode() - x).abs() <= packed.scale / 2 + 1e-6).all())

    def test_causal_prefix_and_gradients(self):
        m = self.model()
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        out = m(ids, (ids + 1) % 16)
        self.assertTrue(torch.isfinite(out['losses']['total']))
        out['losses']['total'].backward()
        for parameter in [m.gist_embedding, m.blocks[0].attention.router.weight,
                          m.blocks[0].attention.qkv.weight, m.reconstruct.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        m.eval()
        full = m(ids)['logits']
        for length in range(1, ids.shape[1]):
            prefix = m(ids[:, :length])['logits']
            torch.testing.assert_close(full[:, :length], prefix, atol=2e-6, rtol=2e-6)
        mask = m.blocks[0].attention.last_mask
        self.assertFalse(mask[0].triu(1).any())

    def test_route_and_ponder_bound(self):
        m = self.model(0).eval()
        m(torch.tensor([[1,2,3,4,5,6,7,8,9]]))
        mask = m.blocks[0].attention.last_mask[0]
        # Last ordinary query sees sink 0, old gists 3/7, but not old body token 1.
        self.assertTrue(mask[10, 0] and mask[10, 3] and mask[10, 7])
        self.assertFalse(mask[10, 1])
        result = m.generate(torch.tensor([[1,2]]), max_new_tokens=2,
                            epsilon=1e6, max_pauses=3)
        self.assertEqual(result['pauses'], 3)
        self.assertEqual(len(result['tokens']), 2)
        self.assertTrue(all(e['low_confidence'] for e in result['events'] if e['type'] == 'emit'))

if __name__ == '__main__': unittest.main()
