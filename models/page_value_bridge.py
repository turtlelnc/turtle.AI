"""Last-token research adapter over actually retrieved pages of a frozen model.

This is a dense diagnostic, not a streaming KV implementation. Its API consumes
no target pages or answer labels. Page values come from the learned reader.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def bridge_inputs(result):
    features = result['page_features']
    if features is None:
        raise ValueError('request return_page_features=True')
    # Last ordinary query only: completed pages can never include future tokens.
    return {'native': result['logits'][:, -1].detach(),
            'query': features['query'][:, -1].detach(),
            'keys': features['keys'].detach(),
            'selected': features['retrieved'][:, -1].detach(),
            'values': result['relation_logits'].detach()}


class PageValueBridge(nn.Module):
    def __init__(self, width, mix_initial=-4.):
        super().__init__()
        self.score_delta = nn.Parameter(torch.zeros(width, width))
        self.mix_logit = nn.Parameter(torch.tensor(float(mix_initial)))

    def forward(self, native, query, keys, selected, values):
        native_logp = F.log_softmax(native, -1)
        if selected.shape[-1] == 0 or keys.shape[1] == 0:
            return native_logp
        valid = (selected >= 0) & (selected < keys.shape[1])
        safe = selected.clamp(0, keys.shape[1] - 1)
        rows = torch.arange(len(query), device=query.device)[:, None]
        routed_keys = keys[rows, safe]
        projected = query + query @ self.score_delta
        scores = (projected[:, None] * routed_keys).sum(-1) / math.sqrt(query.shape[-1])
        # Invalid placeholders must carry exactly zero probability, even when
        # there are no historical pages for a row.
        has_pages = valid.any(-1)
        scores = scores.masked_fill(~valid, -torch.inf)
        scores = torch.where(has_pages[:, None], scores, torch.zeros_like(scores))
        weights = scores.softmax(-1) * valid
        page_probs = values[rows, safe].softmax(-1)
        memory_probs = (weights[..., None] * page_probs).sum(1)
        mix = self.mix_logit.sigmoid() * has_pages
        probs = (1 - mix[:, None]) * native_logp.exp() + mix[:, None] * memory_probs
        return probs.clamp_min(torch.finfo(probs.dtype).tiny).log()
