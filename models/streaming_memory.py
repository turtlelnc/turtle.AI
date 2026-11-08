"""Bounded inference-only KV paging for the Resonant reference model.

Tensor-payload bounds exclude model weights, Python metadata, allocator overhead,
external source storage and transient attention buffers. No fused GPU kernels.
"""
from collections import deque
from dataclasses import dataclass
from typing import Callable
import math
import torch
from .resonant import PackedINT4


def owned(tensor, device, dtype):
    # Never retain a view into a full sequence or autograd graph.
    return tensor.detach().to(device=device, dtype=dtype).clone()


def nbytes(tensor):
    return tensor.numel() * tensor.element_size()


@dataclass(frozen=True)
class MemoryConfig:
    block_size: int = 32
    working_blocks: int = 2
    sink_tokens: int = 8
    gist_slots: int = 8
    cold_bytes: int = 65536
    top_k: int = 2
    working_dtype: torch.dtype = torch.float32

    def __post_init__(self):
        if min(self.block_size, self.working_blocks) <= 0:
            raise ValueError('block_size and working_blocks must be positive')
        if min(self.sink_tokens, self.gist_slots, self.cold_bytes, self.top_k) < 0:
            raise ValueError('memory budgets cannot be negative')
        if self.working_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError('working memory must use a floating point dtype')


@dataclass
class TokenKV:
    position: int
    key: torch.Tensor
    value: torch.Tensor


@dataclass
class ColdPage:
    block_id: int
    positions: tuple
    keys: PackedINT4
    values: PackedINT4
    gist: TokenKV

    def bytes(self):
        return (self.keys.storage_bytes() + self.values.storage_bytes() +
                nbytes(self.gist.key) + nbytes(self.gist.value))


class MemoryPool:
    def __init__(self, width: int, cfg: MemoryConfig, device='cpu', page_store=None):
        if width <= 0: raise ValueError('width must be positive')
        self.width, self.cfg, self.device = width, cfg, torch.device(device)
        if page_store is not None and page_store.width != width: raise ValueError('disk index width mismatch')
        self.page_store = page_store
        self.reset()

    def reset(self):
        if self.page_store is not None: self.page_store.reset()
        self.anchors = []
        self.hot = deque()
        self.pending = []
        self.gists = deque()
        self.cold = deque()
        self.current_block = None
        self.last_position = -1
        self.cold_payload = 0
        self.evictions = 0
        self.peak_payload = 0
        self.peak_attention_tokens = 0
        self.completed_blocks = 0
        self.seen_tokens = 0

    def _validate_pair(self, key, value):
        if key.shape != (self.width,) or value.shape != (self.width,):
            raise ValueError('expected single-token vectors matching memory width')
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise ValueError('nonfinite KV')

    def _token(self, position, key, value):
        self._validate_pair(key, value)
        stored_key = owned(key, self.device, self.cfg.working_dtype)
        stored_value = owned(value, self.device, self.cfg.working_dtype)
        if not torch.isfinite(stored_key).all() or not torch.isfinite(stored_value).all():
            raise ValueError('KV overflows working precision')
        return TokenKV(position, stored_key, stored_value)

    def append(self, position, block_id, key, value):
        self._validate_pair(key, value)
        if position <= self.last_position or block_id != self.completed_blocks:
            raise ValueError('nonmonotonic position or unexpected block ID')
        if self.current_block is not None and self.current_block != block_id:
            raise ValueError('complete the previous block before starting a new one')
        if self.seen_tokens % self.cfg.block_size == 0 and self.current_block is not None:
            raise ValueError('block is full; summary required')
        item = self._token(position, key, value)
        self.current_block = block_id
        if self.seen_tokens < self.cfg.sink_tokens:
            self.anchors.append(item)
        else:
            self.pending.append(item)
        self.seen_tokens += 1
        self.last_position = position
        self._update_peak()

    def finish(self, position, block_id, key, value):
        self._validate_pair(key, value)
        if (self.current_block != block_id or block_id != self.completed_blocks or
                self.seen_tokens % self.cfg.block_size or position <= self.last_position):
            raise ValueError('summary requires a complete current block and increasing position')
        gist = self._token(position, key, value)
        self.gists.append(gist)
        while len(self.gists) > self.cfg.gist_slots: self.gists.popleft()
        self.hot.append((block_id, self.pending, gist))
        self.pending = []
        self.current_block = None
        self.completed_blocks += 1
        self.last_position = position
        while len(self.hot) > self.cfg.working_blocks:
            old_id, tokens, old_gist = self.hot.popleft()
            if not tokens: continue
            # Cold pages live on host memory. Only selected pages are upcast.
            page = ColdPage(old_id, tuple(t.position for t in tokens),
                            PackedINT4.encode(torch.stack([t.key.float().cpu() for t in tokens])),
                            PackedINT4.encode(torch.stack([t.value.float().cpu() for t in tokens])),
                            TokenKV(old_gist.position, owned(old_gist.key, 'cpu', torch.float32),
                                    owned(old_gist.value, 'cpu', torch.float32)))
            if self.page_store is not None: self.page_store.put(page)
            if page.bytes() > self.cfg.cold_bytes:
                self.evictions += 1
                continue
            while self.cold_payload + page.bytes() > self.cfg.cold_bytes:
                self.cold_payload -= self.cold.popleft().bytes()
                self.evictions += 1
            self.cold.append(page)
            self.cold_payload += page.bytes()
        self._update_peak()

    def _working(self):
        return self.anchors + [t for _, tokens, _ in self.hot for t in tokens] + self.pending + list(self.gists)

    def stats(self):
        # hot also owns gist objects even after their global slots are evicted.
        tensors = {}
        tokens = self._working() + [gist for _, _, gist in self.hot]
        for token in tokens:
            for tensor in (token.key, token.value):
                tensors[id(tensor)] = tensor
        warm = sum(nbytes(t) for t in tensors.values())
        element_bytes = torch.empty((), dtype=self.cfg.working_dtype).element_size()
        # Pending block + completed hot blocks, anchors and two bounded gist lists.
        warm_bound = (self.cfg.sink_tokens + self.cfg.block_size * (self.cfg.working_blocks + 1) +
                      self.cfg.gist_slots + self.cfg.working_blocks) * self.width * 2 * element_bytes
        return {'working_payload_bytes': warm, 'cold_payload_bytes': self.cold_payload,
                'persistent_payload_bytes': warm + self.cold_payload,
                'payload_bound_bytes': warm_bound + self.cfg.cold_bytes,
                'cold_pages': len(self.cold), 'evictions': self.evictions,
                'peak_payload_bytes': self.peak_payload,
                'peak_attention_tokens': self.peak_attention_tokens,
                'seen_tokens': self.seen_tokens,
                'attention_token_bound': (self.cfg.sink_tokens + self.cfg.block_size * (self.cfg.working_blocks + 1) +
                                          self.cfg.gist_slots + self.cfg.top_k * (self.cfg.block_size + 1) + 1),
                'disk': None if self.page_store is None else self.page_store.stats()}

    def _update_peak(self):
        self.peak_payload = max(self.peak_payload, self.stats()['persistent_payload_bytes'])

    def retrieve(self, query, top_k=None):
        if query.shape != (self.width,) or not torch.isfinite(query).all():
            raise ValueError('invalid memory query')
        k = self.cfg.top_k if top_k is None else top_k
        if type(k) is not int or k < 0 or k > self.cfg.top_k:
            raise ValueError('top_k must remain within the configured retrieval budget')
        if not k: return []
        ranked = []
        resident = {page.block_id: page for page in self.cold}
        if self.cold:
            index = torch.stack([p.gist.key for p in self.cold])
            scores = index @ query.detach().float().cpu()
            if not torch.isfinite(scores).all(): raise ValueError('retrieval score overflow')
            ranked.extend((float(score), page.block_id) for score, page in zip(scores, self.cold))
        if self.page_store is not None:
            ranked.extend(self.page_store.search(query, k, resident))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [resident[bid] if bid in resident else self.page_store.get(bid)
                for _, bid in ranked[:k]]

    def materialize(self, query, replay: Callable | None = None):
        items = {t.position: t for t in self._working()}
        hits = self.retrieve(query)
        for page in hits:
            if replay is None:
                keys, values = page.keys.decode(), page.values.decode()
            else:
                # Callback may JIT-prefill source or load original FP32 projections.
                # Different prefill context/positions generally change those projections.
                keys, values = replay(page.block_id, page.positions)
                if keys.shape != (len(page.positions), self.width) or values.shape != keys.shape:
                    raise ValueError('replay shape mismatch')
                if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
                    raise ValueError('nonfinite replay KV')
            keys = keys.to(device=self.device, dtype=self.cfg.working_dtype)
            values = values.to(device=self.device, dtype=self.cfg.working_dtype)
            if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
                raise ValueError('recalled KV overflows working precision')
            for i, position in enumerate(page.positions):
                if position not in items:
                    items[position] = TokenKV(position, keys[i], values[i])
            # Keep the selected page's summary visible even if its global slot expired.
            if page.gist.position not in items: items[page.gist.position] = page.gist
        ordered = sorted(items.values(), key=lambda t: t.position)
        if not ordered: raise ValueError('attention requires at least one token')
        self.peak_attention_tokens = max(self.peak_attention_tokens, len(ordered))
        keys = torch.stack([t.key.to(self.device).float() for t in ordered])
        values = torch.stack([t.value.to(self.device).float() for t in ordered])
        return keys, values, tuple(t.position for t in ordered), [p.block_id for p in hits]

    def attend(self, query, replay=None, read_gate=None, page_score_weight=0.0):
        keys, values, positions, hits = self.materialize(query, replay)
        scores = keys @ query.float() / math.sqrt(self.width)
        if not torch.isfinite(scores).all(): raise ValueError('attention score overflow')
        if page_score_weight:
            from .page_readout import stream_page_bias,preserve_recall_mass
            warm={t.position for t in self._working()}
            remote=torch.tensor([p not in warm for p in positions],device=scores.device)
            candidate=scores+page_score_weight*stream_page_bias(query,keys,positions,hits,warm,self.cfg.block_size)
            scores=preserve_recall_mass(scores,candidate,remote)
        if read_gate is None:
            weights = scores.softmax(-1)
        else:
            from .readout_fusion import fused_weights
            warm={t.position for t in self._working()}
            remote=torch.tensor([position not in warm for position in positions],device=scores.device)
            weights=fused_weights(scores,torch.ones_like(remote),remote,read_gate)
        return weights @ values, {'positions': positions, 'hits': hits, 'weights': weights}


class ResonantStream:
    """Single-session, batch-one inference. Original model weights are shared.

    The mask uses a bounded number of completed working blocks and Gists.
    Teacher-forced forward(memory=cfg, disk_backed=...) simulates the same policy;
    the legacy forward without a memory policy retains unlimited historical Gists.
    """
    def __init__(self, model, cfg: MemoryConfig, replay=None, source_archive=None, capture=None, page_directory=None, trace=None):
        if model.training: raise ValueError('call model.eval() before creating a stream')
        if model.cfg.block_size != cfg.block_size: raise ValueError('block size mismatch')
        if source_archive is not None and source_archive.block_size != cfg.block_size:
            raise ValueError('source archive block size mismatch')
        self.model, self.cfg, self.replay = model, cfg, replay
        self.source_archive, self.capture, self.trace = source_archive, capture, trace
        self.device = next(model.parameters()).device
        if page_directory is None:
            self.pools = [MemoryPool(model.cfg.width, cfg, self.device) for _ in model.blocks]
        else:
            from pathlib import Path
            from .disk_pages import DiskPageStore
            self.pools = [MemoryPool(model.cfg.width, cfg, self.device,
                                    DiskPageStore(Path(page_directory) / str(i), model.cfg.width))
                          for i in range(len(model.blocks))]
        self.reset()

    def reset(self):
        # A model reused after teacher forcing must not retain its dense debug masks.
        for layer in self.model.blocks:
            layer.attention.last_mask = None
            layer.attention.last_retrieved = None
        for pool in self.pools: pool.reset()
        self.logical_position = 0
        self.source_position = 0
        self.last_hidden = None
        self.last_logits = None
        self.failed = False
        self.pending_source = []
        if self.source_archive is not None: self.source_archive.new_session()
        if hasattr(self.replay, "reset"): self.replay.reset()

    @torch.no_grad()
    def _step(self, embedding, gist=False):
        if self.model.training: raise ValueError('stream cannot run while model is training')
        if self.logical_position >= self.model.position.num_embeddings:
            raise ValueError('logical position exceeds trained position table')
        block_id = (self.source_position - 1) // self.cfg.block_size if gist else self.source_position // self.cfg.block_size
        x = embedding.reshape(1, 1, -1) + self.model.position.weight[self.logical_position][None, None]
        for layer_id, (layer, pool) in enumerate(zip(self.model.blocks, self.pools)):
            normalized = layer.norm1(x)
            q, k, v = layer.attention.qkv(normalized).reshape(-1).chunk(3)
            if gist:
                # Summary sees its block, then becomes visible to future queries.
                # Self-attention must include the summary's own K/V too.
                if self.model.cfg.gist_local_only:
                    summary_items = sorted(pool.anchors + pool.pending, key=lambda t: t.position)
                    keys = torch.stack([t.key.to(self.device).float() for t in summary_items])
                    values = torch.stack([t.value.to(self.device).float() for t in summary_items])
                    positions, hits = tuple(t.position for t in summary_items), []
                else:
                    keys, values, positions, hits = pool.materialize(q, None if self.replay is None else
                                              lambda bid, pos: self.replay(layer_id, bid, pos))
                keys = torch.cat((keys, k[None].float()))
                values = torch.cat((values, v[None].float()))
                pool.peak_attention_tokens = max(pool.peak_attention_tokens, len(keys))
                scores = keys @ q.float() / math.sqrt(self.model.cfg.width)
                if not torch.isfinite(scores).all(): raise ValueError('attention score overflow')
                attention = scores.softmax(-1) @ values
                if self.trace is not None:
                    self.trace(layer_id, self.logical_position, positions + (self.logical_position,), hits)
                pool.finish(self.logical_position, block_id, k, v)
            else:
                if self.capture is not None:
                    self.capture(layer_id, block_id, self.logical_position, k, v)
                pool.append(self.logical_position, block_id, k, v)
                attention, details = pool.attend(q, None if self.replay is None else
                                          lambda bid, pos: self.replay(layer_id, bid, pos),
                    read_gate=(.5*layer.attention.read_gate(normalized).sigmoid()).reshape(())
                              if self.model.cfg.readout_fusion else None,
                    page_score_weight=self.model.cfg.page_score_weight)
                if self.trace is not None:
                    self.trace(layer_id, self.logical_position, details['positions'], details['hits'])
            x = x + layer.attention.output(attention.to(x.dtype))[None, None]
            x = x + layer.ff(layer.norm2(x))
        self.logical_position += 1
        if not gist: self.source_position += 1
        return self.model.norm(x)

    @torch.no_grad()
    def push(self, token_id):
        if self.failed: raise RuntimeError('stream failed; reset before reuse')
        if not isinstance(token_id, int) or not 0 <= token_id <= self.model.pause_id:
            raise ValueError('token ID out of range')
        # Validate room for the whole operation before mutating any layer cache.
        will_finish = (self.source_position + 1) % self.cfg.block_size == 0
        needed = 2 if will_finish else 1
        if self.source_position >= self.model.cfg.max_tokens or self.logical_position + needed > self.model.position.num_embeddings:
            raise ValueError('stream exceeds configured positional capacity; reset or use a trained extension')
        try:
            hidden = self._step(self.model.embedding.weight[token_id])
            self.last_hidden = hidden
            logits = self.model.head(hidden).reshape(-1).float()
            if self.source_archive is not None:
                self.pending_source.append(token_id)
            self.last_logits = logits
            if will_finish:
                if self.source_archive is not None:
                    self.source_archive.put((self.source_position - 1) // self.cfg.block_size, self.pending_source)
                    self.pending_source = []
                self._step(self.model.gist_embedding, gist=True)
            return logits
        except Exception:
            # A late layer or external replay may fail after earlier caches changed.
            # Never continue silently with a partially updated session.
            self.failed = True
            raise

    def can_push(self):
        needed = 2 if (self.source_position + 1) % self.cfg.block_size == 0 else 1
        return (self.source_position < self.model.cfg.max_tokens and
                self.logical_position + needed <= self.model.position.num_embeddings)

    @torch.no_grad()
    def prefill(self, token_ids):
        # Return just the latest logits: no unbounded output list in the session.
        found = False
        for token_id in token_ids:
            self.push(token_id)
            found = True
        if not found: raise ValueError('prefill requires at least one token')
        return self.last_logits

    @torch.no_grad()
    def generate(self, max_new_tokens=32, epsilon=0.2, max_pauses=3):
        if self.failed: raise RuntimeError('stream failed; reset before reuse')
        if self.last_logits is None: raise ValueError('prefill before generation')
        if (not isinstance(max_new_tokens, int) or not isinstance(max_pauses, int) or
                min(max_new_tokens, max_pauses) < 0 or epsilon < 0 or not math.isfinite(epsilon)):
            raise ValueError('invalid generation limits')
        tokens, events, pauses = [], [], 0
        reason = 'max_new_tokens'
        for _ in range(max_new_tokens):
            while self.can_push():
                top = self.last_logits.topk(min(2, self.last_logits.numel()))
                gap = float(top.values[0] - top.values[-1])
                if gap >= epsilon or pauses >= max_pauses: break
                self.push(self.model.pause_id)
                pauses += 1
                events.append({'type': 'pause', 'margin': gap})
            if not self.can_push():
                reason = 'positional_capacity'
                break
            top = self.last_logits.topk(min(2, self.last_logits.numel()))
            gap = float(top.values[0] - top.values[-1])
            token = int(top.indices[0])
            self.push(token)
            tokens.append(token)
            events.append({'type': 'emit', 'margin': gap, 'low_confidence': gap < epsilon})
        return {'tokens': tokens, 'pauses': pauses, 'events': events, 'stop_reason': reason}

    def stats(self):
        per_layer = [p.stats() for p in self.pools]
        return {'seen_tokens': self.source_position, 'logical_positions': self.logical_position,
                'persistent_payload_bytes': sum(s['persistent_payload_bytes'] for s in per_layer),
                'payload_bound_bytes': sum(s['payload_bound_bytes'] for s in per_layer),
                'layers': per_layer}
