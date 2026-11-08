"""Deterministically derive QA and page labels from explicit source records.

Only the synthetic PAIR/IGNORE grammar is supported. This is not automatic
natural-language understanding. Ambiguous or malformed sources fail explicitly.
"""
from dataclasses import dataclass
import torch
from .associative_recall import PAIR, IGNORE, SEPARATOR, QUERY, KEYS, VALUES

@dataclass(frozen=True)
class SourceFact:
    key: int
    value: int
    token_start: int
    token_end: int
    page: int


def extract_facts(tokens, block_size=8):
    if block_size < 4: raise ValueError('page must hold a complete record')
    facts = {}
    for start in range(0, len(tokens), 4):
        record = tokens[start:start+4]
        if len(record) != 4 or record[0] not in (PAIR, IGNORE) or record[3] != SEPARATOR:
            raise ValueError('malformed source record')
        if record[0] == IGNORE: continue
        _, key, value, _ = record
        if not 1 <= key <= KEYS or not KEYS < value <= KEYS + VALUES:
            raise ValueError('invalid fact key/value')
        if start // block_size != (start + 3) // block_size:
            raise ValueError('fact crosses page boundary')
        if key in facts: raise ValueError('duplicate fact key requires explicit revision semantics')
        facts[key] = SourceFact(key, value, start, start + 4, start // block_size)
    return facts


def derive_batch_targets(ids, block_size=8):
    """Query IDs already exist; labels are re-derived solely from source spans."""
    labels, pages = torch.full_like(ids, -100), torch.full_like(ids, -100)
    spans = []
    for row in range(len(ids)):
        tokens = ids[row].tolist()
        if tokens[-2] != QUERY: raise ValueError('missing final query')
        facts = extract_facts(tokens[:-2], block_size)
        if tokens[-1] not in facts: raise ValueError('query has no source fact')
        fact = facts[tokens[-1]]
        labels[row, -1], pages[row, -1] = fact.value, fact.page
        spans.append((fact.token_start, fact.token_end))
    return labels, pages, spans


def generate_examples(tokens, source_id, block_size=8, query_gap=2, max_queries=8):
    """Build bounded-count QA examples from source, without generator-side answers."""
    if block_size < 4 or not source_id or len(tokens) % block_size or query_gap < 0 or max_queries < 1:
        raise ValueError('invalid source identity, alignment or query limits')
    facts = extract_facts(tokens, block_size)
    completed = len(tokens) // block_size
    selected = [fact for fact in facts.values() if fact.page < completed - query_gap]
    for fact in selected[:max_queries]:
        yield {'source_id': source_id, 'input_ids': list(tokens) + [QUERY, fact.key],
               'answer': fact.value, 'source_page': fact.page,
               'source_span': [fact.token_start, fact.token_end]}


def derive_relation_targets(ids, block_size=8):
    pages=ids.shape[1]//block_size
    targets=torch.full((len(ids),pages),-100,dtype=torch.long,device=ids.device)
    for row,tokens in enumerate(ids[:,:pages*block_size].tolist()):
        for fact in extract_facts(tokens,block_size).values():
            if targets[row,fact.page]!=-100:raise ValueError('multiple facts require a structured relation target')
            targets[row,fact.page]=fact.value
    return targets
