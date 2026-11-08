import tempfile
import unittest
import torch
from models.resonant import Config, ResonantLM
from models.streaming_memory import MemoryConfig, MemoryPool, ResonantStream
from models.source_replay import TokenArchive, LocalPrefillReplay

torch.set_num_threads(1)

class StreamingMemoryTests(unittest.TestCase):
    def test_eviction_and_budget(self):
        cfg = MemoryConfig(block_size=4, working_blocks=1, sink_tokens=2,
                           gist_slots=2, cold_bytes=240, top_k=1)
        pool = MemoryPool(8, cfg)
        for block in range(100):
            for offset in range(4):
                key = torch.arange(8).float() + block + offset
                pool.append(block * 5 + offset, block, key, key)
            pool.finish(block * 5 + 4, block, torch.ones(8), torch.ones(8))
            stats = pool.stats()
            self.assertLessEqual(stats['cold_payload_bytes'], cfg.cold_bytes)
            self.assertLessEqual(stats['persistent_payload_bytes'], stats['payload_bound_bytes'])
        self.assertGreater(pool.evictions, 0)
        self.assertLessEqual(pool.peak_attention_tokens, pool.stats()['attention_token_bound'])
        self.assertEqual([t.position for t in pool.anchors], [0, 1])
        _, details = pool.attend(torch.ones(8))
        self.assertEqual(len(details['positions']), len(set(details['positions'])))
        self.assertTrue(all(p <= pool.last_position for p in details['positions']))
        pool.reset()
        self.assertEqual(pool.stats()['persistent_payload_bytes'], 0)
        self.assertEqual(pool.retrieve(torch.ones(8)), [])

    def test_selected_kv_and_original_callback(self):
        cfg = MemoryConfig(block_size=2, working_blocks=1, sink_tokens=0,
                           gist_slots=0, cold_bytes=1024, top_k=1)
        pool = MemoryPool(3, cfg)
        originals = {}
        for block in range(3):
            key = torch.tensor([1., 0.13, -0.29]) if block == 0 else torch.tensor([-1., 0., 0.])
            value = torch.tensor([float(block), 0.1234, -0.9876])
            originals[block] = (key.repeat(2,1), value.repeat(2,1))
            for offset in range(2): pool.append(block*3+offset, block, key, value)
            pool.finish(block*3+2, block, key, value)
        query = torch.tensor([10., 0., 0.])
        self.assertEqual(pool.retrieve(query)[0].block_id, 0)
        packed_k, packed_v, positions, _ = pool.materialize(query)
        replay_k, replay_v, positions2, _ = pool.materialize(query, lambda bid, pos: originals[bid])
        self.assertEqual(positions, positions2)
        self.assertGreater((packed_k - replay_k).abs().max().item(), 0)
        self.assertGreater((packed_v - replay_v).abs().max().item(), 0)
        self.assertTrue(torch.isfinite(pool.attend(query)[0]).all())

    def test_stream_matches_untruncated_dense(self):
        torch.manual_seed(11)
        model = ResonantLM(Config(vocab_size=16, width=16, layers=2, block_size=3,
                                 window=32, sinks=1, top_k=0, max_tokens=32)).eval()
        ids = torch.tensor([[1,2,3,4,5,6,7,8,9]])
        cfg = MemoryConfig(block_size=3, working_blocks=10, sink_tokens=1,
                           gist_slots=20, top_k=0)
        stream = ResonantStream(model, cfg)
        actual = torch.stack([stream.push(int(token)) for token in ids[0]])
        with torch.no_grad(): expected = model(ids)['logits'][0]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertTrue(all(not t.key.requires_grad for p in stream.pools for t in p._working()))
        stream.reset()
        torch.testing.assert_close(stream.push(1), actual[0], atol=2e-6, rtol=2e-6)

    def test_source_prefill_and_session_isolation(self):
        torch.manual_seed(3)
        model = ResonantLM(Config(vocab_size=16, width=8, layers=2, block_size=2,
                                 max_tokens=32)).eval()
        with tempfile.TemporaryDirectory() as directory:
            archive = TokenArchive(directory, 2)
            replay = LocalPrefillReplay(model, archive)
            cfg = MemoryConfig(block_size=2, working_blocks=1, sink_tokens=1,
                               gist_slots=1, cold_bytes=1024, top_k=1)
            stream = ResonantStream(model, cfg, replay=replay, source_archive=archive)
            outputs = [stream.push(i % 16) for i in range(10)]
            self.assertTrue(torch.isfinite(torch.stack(outputs)).all())
            self.assertGreater(replay.prefill_count, 0)
            self.assertLessEqual(replay.cache_bytes(), 2 * 2 * 2 * 8 * 4)
            self.assertEqual(archive.get(0), [0, 1])
            stream.reset()
            self.assertEqual(replay.cache_bytes(), 0)
            with self.assertRaises(FileNotFoundError): archive.get(0)

    def test_invalid_admission_is_atomic(self):
        pool = MemoryPool(2, MemoryConfig(block_size=1))
        with self.assertRaises(ValueError): pool.append(0, 0, torch.ones(3), torch.ones(3))
        self.assertEqual(pool.seen_tokens, 0)
        pool.append(0, 0, torch.ones(2), torch.ones(2))
        with self.assertRaises(ValueError): pool.append(1, 0, torch.ones(2), torch.ones(2))
        self.assertEqual(pool.seen_tokens, 1)
        with self.assertRaises(ValueError): pool.finish(0, 0, torch.ones(2), torch.ones(2))

    def test_disk_recall_after_resident_eviction(self):
        from models.disk_pages import DiskPageStore
        with tempfile.TemporaryDirectory() as directory:
            backing = DiskPageStore(directory, 3, index_batch=2)
            cfg = MemoryConfig(block_size=2, working_blocks=1, sink_tokens=0,
                               gist_slots=0, cold_bytes=0, top_k=2)
            pool = MemoryPool(3, cfg, page_store=backing)
            for block in range(6):
                key = torch.tensor([1., 0., 0.]) if block in (0, 1) else torch.tensor([-1., 0., 0.])
                value = torch.tensor([float(block), .2, .3])
                for offset in range(2): pool.append(block*3+offset, block, key, value)
                pool.finish(block*3+2, block, key, value)
            self.assertEqual(len(pool.cold), 0)
            pages = pool.retrieve(torch.tensor([10., 0., 0.]))
            self.assertEqual([page.block_id for page in pages], [0, 1])
            self.assertEqual(pages[0].keys.shape, (2,3))
            self.assertGreater(backing.stats()['disk_bytes'], 0)
            self.assertEqual(backing.stats()['pages'], 5)
            output, details = pool.attend(torch.tensor([10., 0., 0.]))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(details['hits'], [0,1])
            pool.reset()
            self.assertEqual(pool.retrieve(torch.ones(3)), [])
            self.assertEqual(backing.stats()['pages'], 0)

    def test_streaming_pause_and_capacity(self):
        model = ResonantLM(Config(vocab_size=16, width=8, layers=1, block_size=2,
                                 max_tokens=12)).eval()
        stream = ResonantStream(model, MemoryConfig(block_size=2, working_dtype=torch.float16))
        stream.prefill([1,2])
        generated = stream.generate(max_new_tokens=2, epsilon=1e6, max_pauses=3)
        self.assertEqual(generated['pauses'], 3)
        self.assertEqual(len(generated['tokens']), 2)
        self.assertTrue(all(t.key.dtype == torch.float16 for t in stream.pools[0]._working()))
        while stream.can_push(): stream.push(1)
        before = stream.stats()
        with self.assertRaises(ValueError): stream.push(1)
        self.assertEqual(stream.stats(), before)
        stopped = stream.generate(2)
        self.assertEqual(stopped['stop_reason'], 'positional_capacity')
        self.assertEqual(stopped['tokens'], [])
        pool = MemoryPool(2, MemoryConfig(block_size=1, working_dtype=torch.float16))
        with self.assertRaises(ValueError): pool.append(0, 0, torch.ones(2)*1e10, torch.ones(2))
        self.assertEqual(pool.seen_tokens, 0)

    def test_partial_failure_requires_reset(self):
        model = ResonantLM(Config(vocab_size=16, width=8, layers=2, block_size=2,
                                 max_tokens=8)).eval()
        def fail_on_second_layer(layer, *_):
            if layer == 1: raise IOError('external source failure')
        stream = ResonantStream(model, MemoryConfig(block_size=2), capture=fail_on_second_layer)
        with self.assertRaises(IOError): stream.push(1)
        self.assertTrue(stream.failed)
        with self.assertRaises(RuntimeError): stream.push(1)
        stream.capture = None
        stream.reset()
        self.assertTrue(torch.isfinite(stream.push(1)).all())

if __name__ == '__main__': unittest.main()
