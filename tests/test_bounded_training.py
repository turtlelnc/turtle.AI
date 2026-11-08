import tempfile
import unittest
import torch
from models.resonant import Config, ResonantLM
from models.streaming_memory import MemoryConfig, ResonantStream
from models.memory_policy import cold_history, cold_page_bytes


torch.set_num_threads(1)

class BoundedTrainingTests(unittest.TestCase):
    def test_exact_masks_logits_and_rounding(self):
        for disk in (False, True):
            for dtype in (torch.float32, torch.float16):
                with self.subTest(disk=disk, dtype=dtype), tempfile.TemporaryDirectory() as directory:
                    torch.manual_seed(28)
                    model = ResonantLM(Config(vocab_size=16, width=8, layers=2, block_size=3,
                                             max_tokens=32, position_mode='sinusoidal')).eval()
                    cfg = MemoryConfig(block_size=3, working_blocks=1, sink_tokens=2,
                                       gist_slots=1, cold_bytes=120, top_k=2, working_dtype=dtype)
                    ids = torch.tensor([[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,1,2,3,4,5]])
                    with torch.no_grad(): dense = model(ids, memory=cfg, disk_backed=disk)['logits'][0]
                    masks = [layer.attention.last_mask[0].clone() for layer in model.blocks]
                    def trace(layer, query, positions, hits):
                        expected = masks[layer][query].nonzero(as_tuple=True)[0].tolist()
                        self.assertEqual(sorted(positions), expected, (disk, dtype, layer, query))
                    stream = ResonantStream(model, cfg, page_directory=directory if disk else None, trace=trace)
                    self.assertTrue(all(layer.attention.last_mask is None for layer in model.blocks))
                    actual = torch.stack([stream.push(int(token)) for token in ids[0]])
                    # FP16 boundaries can amplify tiny GEMM differences by one half ULP.
                    torch.testing.assert_close(dense, actual, atol=4e-4 if dtype == torch.float16 else 3e-6,
                                               rtol=4e-4 if dtype == torch.float16 else 3e-6)

    def test_fifo_metadata_and_bytes(self):
        cfg = MemoryConfig(block_size=3, working_blocks=1, sink_tokens=2,
                           gist_slots=1, cold_bytes=120, top_k=1)
        self.assertEqual(cold_page_bytes(8, 1), 80)
        self.assertEqual(cold_page_bytes(8, 3), 112)
        fifo = cold_history(7, 8, cfg)
        self.assertEqual(fifo[2].nonzero().flatten().tolist(), [0])
        self.assertEqual(fifo[3].nonzero().flatten().tolist(), [1])
        disk = cold_history(7, 8, cfg, True)
        self.assertEqual(disk[6].nonzero().flatten().tolist(), [0,1,2,3,4])

    def test_routing_loss_and_causality(self):
        torch.manual_seed(17)
        model = ResonantLM(Config(vocab_size=16, width=16, layers=2, block_size=3,
                                 max_tokens=32, position_mode='sinusoidal'))
        memory = MemoryConfig(block_size=3, working_blocks=1, sink_tokens=1,
                              gist_slots=1, cold_bytes=512, top_k=1)
        ids = torch.tensor([[1,2,3,4,5,6,7,8,9,10,11,12,13]])
        labels, routes = torch.full_like(ids, -100), torch.full_like(ids, -100)
        labels[:, -1], routes[:, -1] = 5, 0
        result = model(ids, labels, memory=memory, routing_targets=routes,
                       disk_backed=True, routing_weight=1)
        self.assertEqual(result['losses']['routing_examples'], 1)
        unlabelled = model(ids, memory=memory, disk_backed=True)['logits']
        torch.testing.assert_close(result['logits'], unlabelled, atol=0, rtol=0)
        oracle = model(ids, memory=memory, disk_backed=True, routing_targets=routes,
                       oracle_routes=True)
        self.assertTrue(torch.isfinite(oracle['logits']).all())
        selected = model.blocks[-1].attention.last_retrieved[:, -1]
        self.assertTrue((selected == 0).any())
        result['losses']['total'].backward()
        self.assertGreater(model.blocks[1].attention.qkv.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(model.gist_embedding.grad).all())
        model.eval()
        with torch.no_grad():
            full = model(ids, memory=memory, disk_backed=True)['logits']
            for length in (2,3,5,9,12):
                prefix = model(ids[:, :length], memory=memory, disk_backed=True)['logits']
                torch.testing.assert_close(full[:, :length], prefix, atol=3e-6, rtol=3e-6)

    def test_local_gist_policy_and_auxiliary_gradient(self):
        torch.manual_seed(43)
        model = ResonantLM(Config(vocab_size=16, width=8, layers=2, block_size=3,
                                 max_tokens=32, gist_local_only=True)).eval()
        cfg = MemoryConfig(block_size=3, working_blocks=1, sink_tokens=1,
                           gist_slots=1, cold_bytes=120, top_k=1)
        ids = torch.tensor([[1,2,3,4,5,6,7,8,9,10,11,12,13]])
        routes = torch.full_like(ids, -100)
        routes[:, -1] = 0
        result = model(ids, memory=cfg, disk_backed=True, routing_targets=routes)
        result['losses']['routing'].backward()
        self.assertGreater(model.blocks[0].attention.qkv.weight.grad.abs().sum().item(), 0)
        masks = [layer.attention.last_mask[0].clone() for layer in model.blocks]
        # Gist of block 1: only anchor 0, its own body 4..6, and itself 7.
        self.assertEqual(masks[0][7].nonzero().flatten().tolist(), [0,4,5,6,7])
        with torch.no_grad():
            expected = model(ids, memory=cfg, disk_backed=True)['logits'][0]
        def trace(layer, query, positions, hits):
            self.assertEqual(sorted(positions), masks[layer][query].nonzero().flatten().tolist())
        with tempfile.TemporaryDirectory() as directory:
            stream = ResonantStream(model, cfg, page_directory=directory, trace=trace)
            actual = torch.stack([stream.push(int(token)) for token in ids[0]])
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-6)

if __name__ == '__main__': unittest.main()
