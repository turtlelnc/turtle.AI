import unittest
import torch
from models.page_value_bridge import PageValueBridge, bridge_inputs
from models.resonant import Config, ResonantLM
from tools.replay_binding import sample
from tools.train_associative_memory import policy


class PageValueBridgeTests(unittest.TestCase):
    def test_only_selected_pages_and_empty_rows(self):
        torch.manual_seed(45)
        bridge = PageValueBridge(4)
        native, query = torch.randn(2, 11), torch.randn(2, 4)
        keys, values = torch.randn(2, 3, 4), torch.randn(2, 3, 11)
        selected = torch.tensor([[0, -1], [-1, -1]])
        original = bridge(native, query, keys, selected, values)
        values[:, 1:] += 100 * torch.randn_like(values[:, 1:])
        actual = bridge(native, query, keys, selected, values)
        torch.testing.assert_close(actual, original, atol=0, rtol=0)
        torch.testing.assert_close(actual[1], native[1].log_softmax(-1))
        torch.testing.assert_close(actual.exp().sum(-1), torch.ones(2))

    def test_features_need_no_labels_and_preserve_causal_prefix(self):
        torch.set_num_threads(1); torch.manual_seed(46)
        model = ResonantLM(Config(vocab_size=324, width=8, layers=3,
                                 block_size=8, max_tokens=50, relation_reader=True)).eval()
        memory = policy('retrieval', 8, 5)
        ids, _, _ = sample(torch.Generator().manual_seed(47), 2, 4)
        plain = model(ids, memory=memory, disk_backed=True)
        out = model(ids, memory=memory, disk_backed=True, return_page_features=True)
        torch.testing.assert_close(plain['logits'], out['logits'], atol=0, rtol=0)
        prefix = model(ids[:, :18], memory=memory, disk_backed=True, return_page_features=True)
        torch.testing.assert_close(prefix['relation_logits'], out['relation_logits'][:, :2], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(prefix['page_features']['query'], out['page_features']['query'][:, :18], atol=1e-6, rtol=1e-5)
        bridge = PageValueBridge(8)
        logp = bridge(**bridge_inputs(out))
        (-logp[:, 257].mean()).backward()
        self.assertGreater(bridge.score_delta.grad.abs().sum().item(), 0)
        self.assertGreater(bridge.mix_logit.grad.abs().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        partial = model(ids[:, :2], memory=memory, disk_backed=True, return_page_features=True)
        self.assertEqual(partial['relation_logits'].shape, (2, 0, 324))


if __name__ == '__main__': unittest.main()
