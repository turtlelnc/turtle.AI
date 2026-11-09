"""Trainable CPU/GPU reference topology; dense teacher-forcing, not a fused KV engine."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 256
    width: int = 64
    layers: int = 2
    block_size: int = 32
    window: int = 32
    sinks: int = 8
    top_k: int = 2
    max_tokens: int = 2048
    retain_threshold: float = 0.5
    position_mode: str = "learned"
    gist_local_only: bool = False
    readout_fusion: bool = False
    page_score_weight: float = 0.0
    page_score_detach: bool = False
    relation_reader: bool = False

    def __post_init__(self):
        if min(self.vocab_size, self.width, self.layers, self.block_size,
               self.window, self.max_tokens) <= 0 or min(self.sinks, self.top_k) < 0:
            raise ValueError('invalid model dimensions or routing configuration')
        if not math.isfinite(self.page_score_weight) or self.page_score_weight<0:
            raise ValueError('page score weight must be finite and nonnegative')
        if self.position_mode not in ('learned', 'sinusoidal'):
            raise ValueError('unsupported positional encoding mode')
        if not 0 < self.retain_threshold < 1:
            raise ValueError('retain_threshold must be between zero and one')


def fake_int4(x):
    """Per-token symmetric INT4 simulation; straight-through derivative."""
    scale = x.detach().abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7
    quantized = (x / scale).round().clamp(-7, 7) * scale
    return x + (quantized - x).detach()


@dataclass
class PackedINT4:
    data: torch.Tensor
    scale: torch.Tensor
    shape: tuple

    @classmethod
    def encode(cls, value):
        if value.ndim < 1 or value.shape[-1] < 1 or not torch.isfinite(value).all():
            raise ValueError('expected finite nonempty vectors')
        value = value.detach().float()
        scale = value.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7
        q = ((value / scale).round().clamp(-7, 7) + 7).to(torch.uint8)
        if q.shape[-1] % 2: q = F.pad(q, (0, 1))
        return cls(q[..., 0::2] | (q[..., 1::2] << 4), scale, tuple(value.shape))

    def decode(self):
        q = torch.stack((self.data & 15, self.data >> 4), dim=-1).flatten(-2)
        return (q[..., :self.shape[-1]].float() - 7) * self.scale

    def storage_bytes(self):
        return self.data.numel() + self.scale.numel() * self.scale.element_size()


class MemoryAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.width, cfg.width * 3)
        self.router = nn.Linear(cfg.width, 1)
        self.output = nn.Linear(cfg.width, cfg.width)
        if cfg.readout_fusion:
            self.read_gate = nn.Linear(cfg.width,1)
            nn.init.zeros_(self.read_gate.weight)
            nn.init.constant_(self.read_gate.bias,-2.197224577)
        self.last_mask = None
        self.last_retrieved = None

    def routing_mask(self, q, k, positions, blocks, gist):
        n = positions.numel()
        row = torch.arange(n, device=q.device)
        causal = row[:, None] >= row[None, :]
        # Source positions, not original document absolute offsets, define locality.
        local = causal & ((blocks[:, None] == blocks[None, :]) |
                          ((positions[:, None] - positions[None, :]) < self.cfg.window) |
                          (positions[None, :] < self.cfg.sinks) | gist[None, :])
        if self.cfg.gist_local_only:
            summary_local = causal & ((blocks[:, None] == blocks[None, :]) |
                                      ((positions[None, :] < self.cfg.sinks) & ~gist[None, :]))
            local = torch.where(gist[:, None], summary_local, local)
        mask = local.unsqueeze(0).expand(q.shape[0], -1, -1).clone()
        if self.cfg.top_k:
            gist_indices = gist.nonzero(as_tuple=True)[0]
            if len(gist_indices):
                # Hard block selection has no derivative; selected Q/K/V still train.
                route_scores = torch.matmul(q.detach(), k[:, gist_indices].detach().transpose(-1, -2))
                eligible = row[:, None] > gist_indices[None, :]
                eligible &= blocks[:, None] > blocks[gist_indices][None, :]
                if self.cfg.gist_local_only: eligible &= ~gist[:, None]
                route_scores = route_scores.masked_fill(~eligible, -torch.inf)
                indices = route_scores.topk(min(self.cfg.top_k, len(gist_indices)), dim=-1).indices
                selected_valid = eligible.unsqueeze(0).expand(q.shape[0], -1, -1).gather(-1, indices)
                chosen_blocks = blocks[gist_indices][indices]
                selected = ((chosen_blocks[..., None] == blocks[None, None, None, :]) &
                            selected_valid[..., None]).any(dim=2)
                mask |= selected & causal
        self.last_mask = mask.detach()
        return mask, local

    def forward(self, x, positions, blocks, gist, memory=None, disk_backed=False,
                routing_targets=None, quantize_cold=True, oracle_routes=False, supervision_policy=None, oracle_page_only=False, collect_page_kv=False):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        if memory is not None:
            from .memory_policy import bounded_routes, working_round
            warm_k = working_round(k, memory.working_dtype)
            warm_v = working_round(v, memory.working_dtype)
            cold_k = working_round(fake_int4(warm_k), memory.working_dtype) if quantize_cold else warm_k
            cold_v = working_round(fake_int4(warm_v), memory.working_dtype) if quantize_cold else warm_v
            # Gist K/V remain at working precision, even when recalled from disk.
            cold_k = torch.where(gist[None, :, None], warm_k, cold_k)
            cold_v = torch.where(gist[None, :, None], warm_v, cold_v)
            mask, local, chosen, route_loss, count = bounded_routes(
                q, warm_k, positions, blocks, gist, memory, disk_backed, routing_targets, oracle_routes,
                self.cfg.gist_local_only, supervision_policy, oracle_page_only)
            scale = 1 / math.sqrt(self.cfg.width)
            scores = torch.where(local[None], q.float() @ warm_k.transpose(-1, -2) * scale,
                                 q.float() @ cold_k.transpose(-1, -2) * scale)
            # Stream Gist queries include their own unrounded projection before storage.
            diagonal = (q.float() * k.float()).sum(-1) * scale
            scores = scores + torch.diag_embed((diagonal - scores.diagonal(dim1=-2, dim2=-1)) * gist[None])
            if self.cfg.page_score_weight:
                from .page_readout import dense_page_bias,preserve_recall_mass
                candidate=scores+self.cfg.page_score_weight*dense_page_bias(
                    q,warm_k,chosen,blocks,local,gist,self.cfg.block_size,self.cfg.page_score_detach)
                scores=preserve_recall_mass(scores,candidate,mask & ~local[None])
            if self.cfg.readout_fusion:
                from .readout_fusion import fused_weights
                gate=.5*self.read_gate(x).sigmoid()*~gist[None,:,None]
                weights=fused_weights(scores,mask,~local[None],gate)
            else:
                weights = scores.masked_fill(~mask, -torch.inf).softmax(-1)
            out = (weights * local[None]) @ warm_v + (weights * ~local[None]) @ cold_v
            out = out + weights.diagonal(dim1=-2, dim2=-1)[..., None] * gist[None, :, None] * (v - warm_v)
            self.last_mask, self.last_retrieved = mask.detach(), chosen.detach()
            return self.output(out.to(x.dtype)), {'precision': q.sum() * 0,
                                                  'routing': route_loss, 'routing_count': count,
                                                  'page_query': q if collect_page_kv else None,
                                                  'page_keys': warm_k[:, gist] if collect_page_kv else None,
                                                  'page_kv': torch.cat((torch.where(((positions<memory.sink_tokens)|gist)[None,:,None],warm_k,cold_k),
                                                                        torch.where(((positions<memory.sink_tokens)|gist)[None,:,None],warm_v,cold_v)),-1) if collect_page_kv else None}
        if self.cfg.readout_fusion or self.cfg.page_score_weight:
            raise ValueError('selective readout requires an explicit memory policy')
        self.last_retrieved = None
        p = self.router(x).sigmoid()
        hard = (p > self.cfg.retain_threshold).to(p.dtype)
        retain = hard + p - p.detach() if self.training else hard
        # Anchors are protected. Local working memory is read from original K/V below.
        protected = ((positions < self.cfg.sinks) | gist)[None, :, None]
        retain = torch.where(protected, torch.ones_like(retain), retain)
        cold_k = retain * k + (1 - retain) * fake_int4(k)
        cold_v = retain * v + (1 - retain) * fake_int4(v)
        mask, local = self.routing_mask(q, k, positions, blocks, gist)
        scale = 1 / math.sqrt(self.cfg.width)
        warm_scores = q.float() @ k.float().transpose(-1, -2) * scale
        cold_scores = q.float() @ cold_k.float().transpose(-1, -2) * scale
        scores = torch.where(local[None], warm_scores, cold_scores)
        weights = scores.masked_fill(~mask, -torch.inf).softmax(-1)
        # Both local and recalled slices participate in one softmax denominator.
        warm_weights = weights * local[None]
        cold_weights = weights * ~local[None]
        out = warm_weights @ v.float() + cold_weights @ cold_v.float()
        unprotected = ~protected.expand_as(p)
        budget_loss = p[unprotected].mean() if unprotected.any() else p.sum() * 0
        return self.output(out.to(x.dtype)), {'precision': budget_loss,
                                              'routing': q.sum() * 0, 'routing_count': 0}


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.width)
        self.norm2 = nn.LayerNorm(cfg.width)
        self.attention = MemoryAttention(cfg)
        self.ff = nn.Sequential(nn.Linear(cfg.width, cfg.width * 4), nn.GELU(),
                                nn.Linear(cfg.width * 4, cfg.width))

    def forward(self, x, positions, blocks, gist, **policy):
        attention, cost = self.attention(self.norm1(x), positions, blocks, gist, **policy)
        x = x + attention
        return x + self.ff(self.norm2(x)), cost


class ResonantLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.pause_id = cfg.vocab_size
        self.embedding = nn.Embedding(cfg.vocab_size + 1, cfg.width)
        self.gist_embedding = nn.Parameter(torch.randn(cfg.width) * 0.02)
        # Logical, contiguous positions after gist insertion; no KV position reuse.
        self.position = nn.Embedding(cfg.max_tokens * 2 + 1, cfg.width)
        if cfg.position_mode == 'sinusoidal':
            positions = torch.arange(self.position.num_embeddings).float()[:, None]
            frequencies = torch.exp(torch.arange(0, cfg.width, 2).float() * (-math.log(10000.) / cfg.width))
            with torch.no_grad():
                self.position.weight[:, 0::2] = torch.sin(positions * frequencies)
                self.position.weight[:, 1::2] = torch.cos(positions * frequencies[:cfg.width // 2])
            self.position.weight.requires_grad_(False)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.width)
        self.head = nn.Linear(cfg.width, cfg.vocab_size)
        self.reconstruct = nn.Linear(cfg.width, cfg.width)
        if cfg.relation_reader:
            self.relation_reader = nn.Sequential(nn.Linear(cfg.block_size*cfg.width*2,128),nn.GELU(),nn.Linear(128,cfg.vocab_size))

    def expand(self, ids):
        if ids.ndim != 2 or ids.shape[1] < 1 or ids.shape[1] > self.cfg.max_tokens:
            raise ValueError('expected nonempty [batch, time] within max_tokens')
        embeddings = self.embedding(ids)
        pieces, ordinary, gist_indices, owners, source_positions = [], [], [], [], []
        offset = 0
        for start in range(0, ids.shape[1], self.cfg.block_size):
            stop = min(start + self.cfg.block_size, ids.shape[1])
            length = stop - start
            pieces.append(embeddings[:, start:stop])
            ordinary.extend(range(offset, offset + length))
            owners.extend([start // self.cfg.block_size] * length)
            source_positions.extend(range(start, stop))
            offset += length
            # Never summarize a partial block; its future is not available yet.
            if length == self.cfg.block_size:
                pieces.append(self.gist_embedding[None, None].expand(ids.shape[0], 1, -1))
                gist_indices.append(offset)
                owners.append(start // self.cfg.block_size)
                source_positions.append(stop - 1)
                offset += 1
        x = torch.cat(pieces, dim=1)
        device = ids.device
        gist = torch.zeros(offset, dtype=torch.bool, device=device)
        gist[gist_indices] = True
        return (x, embeddings, torch.tensor(ordinary, device=device), gist,
                torch.tensor(owners, device=device), torch.tensor(source_positions, device=device))

    def forward(self, ids, targets=None, margin=0.2, margin_weight=0.1,
                reconstruction_weight=0.1, precision_weight=0.01, memory=None,
                disk_backed=False, routing_targets=None, routing_weight=0.0,
                quantize_cold=True, oracle_routes=False, supervision_policy=None, oracle_page_only=False,
                relation_targets=None, relation_weight=0.0, return_page_features=False):
        x, original, ordinary, gist, owners, positions = self.expand(ids)
        x = x + self.position(torch.arange(x.shape[1], device=ids.device))[None]
        if memory is not None and memory.block_size != self.cfg.block_size:
            raise ValueError('training and model block sizes differ')
        if supervision_policy is not None and (memory is None or supervision_policy.block_size != self.cfg.block_size):
            raise ValueError('auxiliary supervision requires a matching bounded block size')
        expanded_targets = None
        if routing_targets is not None:
            if memory is None: raise ValueError('routing supervision requires bounded memory')
            if routing_targets.shape != ids.shape: raise ValueError('routing target shape mismatch')
            expanded_targets = torch.full(x.shape[:2], -100, dtype=torch.long, device=ids.device)
            expanded_targets[:, ordinary] = routing_targets
        need_pages = relation_targets is not None or return_page_features
        if need_pages and (memory is None or not self.cfg.relation_reader):
            raise ValueError('relation supervision requires a configured reader and memory policy')
        costs, routing_losses, routing_counts = [], [], []
        last_page_kv = None
        for layer_id, block in enumerate(self.blocks):
            # First-layer Gist keys have not yet seen block contents; supervise later layers.
            x, cost = block(x, positions, owners, gist, memory=memory, disk_backed=disk_backed,
                            routing_targets=expanded_targets if (layer_id > 0 or oracle_routes or oracle_page_only) else None,
                            quantize_cold=quantize_cold, oracle_routes=oracle_routes,
                            supervision_policy=supervision_policy, oracle_page_only=oracle_page_only,
                            collect_page_kv=need_pages and layer_id==len(self.blocks)-1)
            costs.append(cost['precision'])
            if cost.get('page_kv') is not None:last_page_kv=cost['page_kv']
            if cost['routing_count']:
                routing_losses.append(cost['routing'])
                routing_counts.append(cost['routing_count'])
        x = self.norm(x)
        logits = self.head(x[:, ordinary]).float()
        reconstruction = x.sum() * 0
        gist_hidden = x[:, gist]
        if gist_hidden.shape[1]:
            n = gist_hidden.shape[1] * self.cfg.block_size
            target = original[:, :n].detach().reshape(ids.shape[0], -1, self.cfg.block_size,
                                                       self.cfg.width).mean(2)
            reconstruction = F.mse_loss(self.reconstruct(gist_hidden), target)
        relation_loss=x.sum()*0
        relation_logits=None
        if need_pages:
            pages=ids.shape[1]//self.cfg.block_size
            indices=(torch.arange(pages,device=ids.device)[:,None]*(self.cfg.block_size+1)+torch.arange(self.cfg.block_size,device=ids.device)).flatten()
            page_input=last_page_kv[:,indices].reshape(ids.shape[0],pages,self.cfg.block_size*self.cfg.width*2)
            relation_logits=self.relation_reader(page_input).float()
        if relation_targets is not None:
            if relation_targets.shape!=(ids.shape[0],pages):raise ValueError('relation target shape mismatch')
            valid=relation_targets!=-100
            if valid.any():relation_loss=F.cross_entropy(relation_logits[valid],relation_targets[valid])
        route_loss = torch.stack(routing_losses).mean() if routing_losses else x.sum() * 0
        losses = {'reconstruction': reconstruction, 'precision': torch.stack(costs).mean(),
                  'routing': route_loss, 'routing_examples': sum(routing_counts),'relation':relation_loss}
        if targets is not None:
            if targets.shape != ids.shape: raise ValueError('target shape mismatch')
            valid = targets != -100
            if not valid.any(): raise ValueError('no supervised tokens')
            selected, labels = logits[valid], targets[valid]
            ce = F.cross_entropy(selected, labels)
            correct = selected.gather(-1, labels[:, None]).squeeze(-1)
            competitors = selected.scatter(-1, labels[:, None], -torch.inf).max(-1).values
            margin_loss = F.relu(margin - correct + competitors).mean()
            losses.update(ce=ce, margin=margin_loss)
            losses['total'] = (ce + margin_weight * margin_loss +
                               reconstruction_weight * reconstruction + precision_weight * losses['precision'] +
                               routing_weight * route_loss + relation_weight * relation_loss)
        page_features = None
        if return_page_features:
            page_features = {'query': cost['page_query'][:, ordinary],
                             'keys': cost['page_keys'],
                             'retrieved': self.blocks[-1].attention.last_retrieved[:, ordinary]}
        return {'logits': logits, 'losses': losses, 'relation_logits':relation_logits,
                'page_features': page_features}

    @torch.no_grad()
    def generate(self, ids, max_new_tokens=32, epsilon=0.2, max_pauses=3):
        if self.training: raise ValueError('call eval() before generation')
        if min(max_new_tokens, max_pauses) < 0 or epsilon < 0:
            raise ValueError('invalid generation limits')
        if ids.shape[0] != 1: raise ValueError('reference generation supports batch size one')
        visible, pauses, events = [], 0, []
        for _ in range(max_new_tokens):
            while True:
                logits = self(ids)['logits'][:, -1]
                top = logits.topk(min(2, logits.shape[-1]), dim=-1)
                gap = float(top.values[0, 0] - top.values[0, -1])
                room = ids.shape[1] < self.cfg.max_tokens
                if gap >= epsilon or pauses >= max_pauses or not room:
                    break
                ids = torch.cat((ids, ids.new_tensor([[self.pause_id]])), dim=1)
                pauses += 1
                events.append({'type': 'pause', 'margin': gap})
            token = int(top.indices[0, 0])
            visible.append(token)
            events.append({'type': 'emit', 'margin': gap,
                           'low_confidence': gap < epsilon})
            if ids.shape[1] >= self.cfg.max_tokens: break
            ids = torch.cat((ids, ids.new_tensor([[token]])), dim=1)
        return {'tokens': visible, 'pauses': pauses, 'events': events}
