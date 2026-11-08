"""Bounded residual readout: at least half of the original joint context survives."""
import torch


def safe_softmax(scores,mask):
    valid=mask.any(-1,keepdim=True)
    logits=scores.masked_fill(~mask,-torch.inf)
    logits=torch.where(valid,logits,torch.zeros_like(logits))
    return logits.softmax(-1)*mask


def fused_weights(scores,visible,remote,gate):
    joint=safe_softmax(scores,visible)
    recall=safe_softmax(scores,visible & remote)
    fraction=gate*(visible & remote).any(-1,keepdim=True)
    return (1-fraction)*joint+fraction*recall
