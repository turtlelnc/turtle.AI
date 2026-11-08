"""Disk-backed original-token archive and bounded, isolated JIT block prefill.

Local prefill reconstructs K/V from tokens, NOT the original long-context K/V.
Absolute positions and preceding context differ. This is an explicit ablation.
"""
import json
from pathlib import Path
import uuid
import torch
from .streaming_memory import MemoryConfig, ResonantStream


class TokenArchive:
    def __init__(self, root, block_size):
        if block_size <= 0: raise ValueError('invalid archive block size')
        self.root, self.block_size = Path(root), block_size
        self.new_session()

    def new_session(self):
        # Every reset gets a new namespace; previous sessions cannot be retrieved.
        self.directory = self.root / uuid.uuid4().hex
        self.directory.mkdir(parents=True, exist_ok=False)

    def _path(self, block_id):
        if not isinstance(block_id, int) or block_id < 0: raise ValueError('invalid block ID')
        return self.directory / f'{block_id}.json'

    def put(self, block_id, token_ids):
        if len(token_ids) != self.block_size or any(type(i) is not int or i < 0 for i in token_ids):
            raise ValueError('archive requires a complete token block')
        path = self._path(block_id)
        with path.open('x') as handle: json.dump(token_ids, handle)

    def get(self, block_id):
        path = self._path(block_id)
        # Bound reads as well as decoded token counts.
        if path.stat().st_size > self.block_size * 32 + 2: raise ValueError('oversized source block')
        ids = json.loads(path.read_text())
        if (not isinstance(ids, list) or len(ids) != self.block_size or
                any(type(i) is not int or i < 0 for i in ids)):
            raise ValueError('invalid source block')
        return ids


class LocalPrefillReplay:
    def __init__(self, model, archive):
        if model.cfg.block_size != archive.block_size: raise ValueError('block size mismatch')
        self.model, self.archive = model, archive
        self.reset()

    def reset(self):
        # One block only, across all layers; host-resident, constant-size cache.
        self.cached_id = None
        self.cache = {}
        self.prefill_count = 0

    def __call__(self, layer_id, block_id, positions):
        if self.cached_id != block_id:
            collected = {}
            def capture(layer, bid, position, key, value):
                collected.setdefault(layer, []).append((key.detach().float().cpu().clone(),
                                                       value.detach().float().cpu().clone()))
            cfg = MemoryConfig(block_size=self.archive.block_size, working_blocks=1,
                               sink_tokens=0, gist_slots=0, cold_bytes=0, top_k=0)
            local = ResonantStream(self.model, cfg, capture=capture)
            for token in self.archive.get(block_id): local.push(token)
            self.cache = {layer: (torch.stack([k for k, _ in pairs]),
                                  torch.stack([v for _, v in pairs]))
                          for layer, pairs in collected.items()}
            self.cached_id = block_id
            self.prefill_count += 1
        start = block_id * (self.archive.block_size + 1)
        offsets = [position - start for position in positions]
        if any(i < 0 or i >= self.archive.block_size for i in offsets):
            raise ValueError('source logical positions do not match block')
        keys, values = self.cache[layer_id]
        return keys[offsets], values[offsets]

    def cache_bytes(self):
        return sum(t.numel() * t.element_size() for pair in self.cache.values() for t in pair)
