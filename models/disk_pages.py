"""Append-only disk pages with a fixed-batch exact MIPS index scan.

No all-history in-RAM catalog. Query cost grows with history length; only selected
pages are loaded. Each pool/reset gets an isolated namespace.
"""
import heapq
from pathlib import Path
import struct
import uuid
import torch


class DiskPageStore:
    def __init__(self, root, width, index_batch=64):
        if min(width, index_batch) <= 0: raise ValueError('invalid disk index dimensions')
        self.root, self.width, self.index_batch = Path(root), width, index_batch
        self.reset()

    def reset(self):
        self.directory = self.root / uuid.uuid4().hex
        self.directory.mkdir(parents=True, exist_ok=False)
        self.index_path = self.directory / 'index.bin'
        self.index_path.touch()
        self.count, self.bytes, self.last_id = 0, 0, -1
        self.page_loads, self.searches, self.keys_scanned = 0, 0, 0

    def _path(self, block_id):
        if type(block_id) is not int or block_id < 0: raise ValueError('invalid page ID')
        return self.directory / f'{block_id}.pt'

    def put(self, page):
        if page.block_id <= self.last_id: raise ValueError('disk page IDs must increase')
        if page.gist.key.shape != (self.width,): raise ValueError('index width mismatch')
        tensors = {'block_id': page.block_id, 'positions': page.positions,
                   'k_data': page.keys.data, 'k_scale': page.keys.scale, 'k_shape': page.keys.shape,
                   'v_data': page.values.data, 'v_scale': page.values.scale, 'v_shape': page.values.shape,
                   'gist_position': page.gist.position, 'gist_key': page.gist.key,
                   'gist_value': page.gist.value}
        path = self._path(page.block_id)
        torch.save(tensors, path)
        record = struct.pack('<q', page.block_id) + struct.pack('<' + 'f' * self.width,
                                                                *page.gist.key.tolist())
        with self.index_path.open('ab') as handle: handle.write(record)
        self.count += 1
        self.bytes += path.stat().st_size + len(record)
        self.last_id = page.block_id

    def get(self, block_id):
        from .resonant import PackedINT4
        from .streaming_memory import ColdPage, TokenKV
        data = torch.load(self._path(block_id), map_location='cpu', weights_only=True)
        if data['block_id'] != block_id: raise ValueError('disk page ID mismatch')
        self.page_loads += 1
        return ColdPage(block_id, tuple(data['positions']),
                        PackedINT4(data['k_data'], data['k_scale'], tuple(data['k_shape'])),
                        PackedINT4(data['v_data'], data['v_scale'], tuple(data['v_shape'])),
                        TokenKV(data['gist_position'], data['gist_key'], data['gist_value']))

    def search(self, query, top_k, exclude=()):
        if query.shape != (self.width,) or not torch.isfinite(query).all() or top_k < 0:
            raise ValueError('invalid disk query')
        if not top_k: return []
        self.searches += 1
        query = query.detach().float().cpu()
        excluded = set(exclude)  # Resident IDs only, bounded by the resident cold pool.
        record_bytes = 8 + self.width * 4
        heap = []
        with self.index_path.open('rb') as handle:
            while True:
                data = handle.read(record_bytes * self.index_batch)
                if not data: break
                if len(data) % record_bytes: raise ValueError('truncated disk index')
                ids, vectors = [], []
                for start in range(0, len(data), record_bytes):
                    block_id = struct.unpack_from('<q', data, start)[0]
                    if block_id in excluded: continue
                    ids.append(block_id)
                    vectors.append(struct.unpack_from('<' + 'f' * self.width, data, start + 8))
                if not vectors: continue
                scores = torch.tensor(vectors, dtype=torch.float32) @ query
                if not torch.isfinite(scores).all(): raise ValueError('disk retrieval score overflow')
                self.keys_scanned += len(ids)
                for score, block_id in zip(scores.tolist(), ids):
                    candidate = (score, -block_id)
                    if len(heap) < top_k: heapq.heappush(heap, candidate)
                    elif candidate > heap[0]: heapq.heapreplace(heap, candidate)
        return [(score, -negative_id) for score, negative_id in sorted(heap, reverse=True)]

    def stats(self):
        return {'pages': self.count, 'disk_bytes': self.bytes,
                'page_loads': self.page_loads, 'searches': self.searches,
                'keys_scanned': self.keys_scanned, 'index_batch': self.index_batch}
