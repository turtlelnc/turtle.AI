"""Teacher-forcing simulation of the streaming pool's visibility and FIFO rules.

Dense tensors are intentional for training. Disk backing changes available pages,
not the resident byte budget. No neural representations are stored here.
"""
from collections import deque
import torch
from torch.nn import functional as F


def cold_page_bytes(width, token_count):
    if token_count <= 0: return 0
    return 2 * token_count * ((width + 1) // 2 + 4) + 8 * width


def cold_history(block_count, width, cfg, disk_backed=False):
    table = torch.zeros(block_count, block_count, dtype=torch.bool)
    resident, used = deque(), 0
    all_pages = []
    for query_block in range(block_count):
        demoted = query_block - cfg.working_blocks - 1
        if demoted >= 0:
            protected = min(cfg.block_size, max(0, cfg.sink_tokens - demoted * cfg.block_size))
            size = cold_page_bytes(width, cfg.block_size - protected)
            if size:
                all_pages.append(demoted)
                if size <= cfg.cold_bytes:
                    while used + size > cfg.cold_bytes:
                        _, previous_size = resident.popleft()
                        used -= previous_size
                    resident.append((demoted, size)); used += size
        visible = all_pages if disk_backed else [bid for bid, _ in resident]
        table[query_block, visible] = True
    return table


def working_round(x, dtype):
    rounded = x.to(dtype).float()
    if not torch.isfinite(rounded).all(): raise ValueError('working precision overflow')
    return x + (rounded - x).detach()


def bounded_routes(q, warm_keys, positions, owners, gist, cfg, disk_backed=False,
                   routing_targets=None, oracle_routes=False, gist_local_only=False, supervision_policy=None, oracle_page_only=False):
    batch, length, width = q.shape
    rows = torch.arange(length, device=q.device)
    causal = rows[:, None] >= rows[None, :]
    query_blocks, key_blocks = owners[:, None], owners[None, :]
    body = ~gist[None, :]
    local_body = body & (key_blocks >= query_blocks - cfg.working_blocks) & (key_blocks <= query_blocks)
    anchors = body & (positions[None, :] < cfg.sink_tokens)
    recent_gists = gist[None, :] & (key_blocks < query_blocks) & (key_blocks >= query_blocks - cfg.gist_slots)
    self_gist = torch.eye(length, dtype=torch.bool, device=q.device) & gist[:, None]
    local = causal & (local_body | anchors | recent_gists | self_gist)
    if gist_local_only:
        summary_local = causal & (anchors | (body & (key_blocks == query_blocks)) | self_gist)
        local = torch.where(gist[:, None], summary_local, local)
    mask = local[None].expand(batch, -1, -1).clone()
    loss, count = q.sum() * 0, 0
    gist_indices = gist.nonzero(as_tuple=True)[0]
    chosen = torch.empty(batch, length, 0, dtype=torch.long, device=q.device)
    eligible = torch.empty(length, 0, dtype=torch.bool, device=q.device)
    if len(gist_indices):
        history = cold_history(int(owners.max()) + 1, width, cfg, disk_backed).to(q.device)
        eligible = history[owners][:, owners[gist_indices]]
        if gist_local_only: eligible = eligible & ~gist[:, None]
        # Scale affects training but not ranking. The index stores warm-round Gist K.
        scores = q.float() @ warm_keys[:, gist_indices].float().transpose(-1, -2) / width**0.5
        masked_scores = scores.masked_fill(~eligible[None], -torch.inf)
        if cfg.top_k:
            k = min(cfg.top_k, len(gist_indices))
            # Same deterministic score-descending, oldest-ID tie break as the pool.
            indices = masked_scores.detach().argsort(dim=-1, descending=True, stable=True)[..., :k]
            valid = eligible[None].expand(batch, -1, -1).gather(-1, indices)
            chosen = owners[gist_indices][indices].masked_fill(~valid, -1)
            if oracle_routes:
                if routing_targets is None: raise ValueError('oracle routing requires annotated target pages')
                safe = routing_targets.clamp(0, len(gist_indices) - 1)
                target_valid = ((routing_targets >= 0) & (routing_targets < len(gist_indices)) &
                                eligible[None].expand(batch, -1, -1).gather(-1, safe[..., None]).squeeze(-1))
                # Preserve the retrieved set if the target is already present.
                # Otherwise duplicates would remove a competing page and confound the oracle.
                insert = target_valid & ~(chosen == routing_targets[..., None]).any(-1)
                chosen[..., 0] = torch.where(insert, routing_targets, chosen[..., 0])
                valid[..., 0] |= insert
            selected = ((chosen[..., None] == owners[None, None, None, :]) &
                        valid[..., None]).any(2)
            mask |= selected & causal
        if routing_targets is not None:
            if routing_targets.shape != (batch, length): raise ValueError('expanded routing target shape mismatch')
            target = routing_targets.clamp(0, len(gist_indices) - 1)
            valid = (routing_targets >= 0) & (routing_targets < len(gist_indices))
            loss_eligible = eligible
            if supervision_policy is not None:
                supervised_history = cold_history(int(owners.max()) + 1, width, supervision_policy, disk_backed).to(q.device)
                loss_eligible = supervised_history[owners][:, owners[gist_indices]]
                if gist_local_only: loss_eligible &= ~gist[:, None]
            loss_scores = scores.masked_fill(~loss_eligible[None], -torch.inf)
            valid &= loss_eligible[None].expand(batch, -1, -1).gather(-1, target[..., None]).squeeze(-1)
            count = int(valid.sum())
            if count: loss = F.cross_entropy(loss_scores[valid], target[valid])
    if oracle_page_only:
        if routing_targets is None: raise ValueError('page-only diagnostic requires source annotations')
        valid_target = (routing_targets >= 0) & (routing_targets < owners[None, :])
        target_page = owners[None, None, :] == routing_targets[..., None]
        self_key = torch.eye(length,dtype=torch.bool,device=q.device)[None]
        # Diagnostic grants the known page even if the learned retriever missed it.
        only = (target_page | self_key) & causal[None]
        mask = torch.where(valid_target[...,None],only,mask)
    return mask, local, chosen, loss, count
