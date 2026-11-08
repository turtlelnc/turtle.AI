"""Relative selected-page scores bias recalled tokens; working/skeleton keys unchanged."""
import torch


def dense_page_bias(query,keys,chosen,owners,local,gist,block_size,detach=False):
    b,n,d=query.shape
    if chosen.shape[-1]==0:return query.new_zeros(b,n,n)
    valid=chosen>=0
    indices=((chosen.clamp_min(0)+1)*(block_size+1)-1).reshape(b,-1)
    selected=keys.gather(1,indices[...,None].expand(-1,-1,d)).reshape(b,n,-1,d)
    scores=(query[:,:,None,:].float()*selected.float()).sum(-1)/d**.5
    if detach:scores=scores.detach()
    centered=(scores-(scores*valid).sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1))*valid
    match=(chosen[...,None]==owners[None,None,None,:]) & valid[...,None]
    bias=(centered[...,None]*match).sum(2)
    return bias*(~local)[None]*~gist[None,:,None]


def stream_page_bias(query,keys,positions,hits,warm,block_size):
    if not hits:return query.new_zeros(len(positions))
    index={position:i for i,position in enumerate(positions)}
    summaries=torch.stack([keys[index[(page+1)*(block_size+1)-1]] for page in hits])
    scores=summaries.float() @ query.float()/query.numel()**.5
    centered=scores-scores.mean()
    by_page={page:centered[i] for i,page in enumerate(hits)}
    return torch.stack([by_page.get(position//(block_size+1),query.new_zeros(()))
                        if position not in warm else query.new_zeros(()) for position in positions])


def preserve_recall_mass(original,candidate,remote):
    """Keep remote exp-sum equal, so local attention probabilities are unchanged."""
    valid=remote.any(-1,keepdim=True)
    old=torch.where(valid,original.masked_fill(~remote,-torch.inf),torch.zeros_like(original))
    new=torch.where(valid,candidate.masked_fill(~remote,-torch.inf),torch.zeros_like(candidate))
    correction=old.logsumexp(-1,keepdim=True)-new.logsumexp(-1,keepdim=True)
    return torch.where(remote,candidate+correction,original)
