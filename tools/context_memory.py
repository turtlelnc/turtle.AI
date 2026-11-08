"""Text JIT-RAG reference scheduler. No model calls or KV-cache reuse."""
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Callable


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    start_line: int
    end_line: int
    text: str
    skeleton: str


def chunk(source, start, end, text, skeleton):
    identity = hashlib.sha256(f'{source}:{start}:{end}:{text}'.encode()).hexdigest()[:20]
    return Chunk(identity, source, start, end, text, skeleton)


def cpp_chunks(source: str, text: str) -> list[Chunk]:
    """Tree-sitter boundaries; oversized functions remain intact for admission checks."""
    from tree_sitter import Language, Parser
    import tree_sitter_cpp
    raw = text.encode()
    root = Parser(Language(tree_sitter_cpp.language())).parse(raw).root_node
    if root.has_error:
        raise ValueError(f'{source}: C++ parse errors; refusing unreliable skeleton')
    result = []
    def visit(node):
        if node.type == 'function_definition':
            body = node.child_by_field_name('body')
            signature = raw[node.start_byte:body.start_byte].decode().strip()
            # Include enclosing type/namespace names to disambiguate methods.
            parents, parent = [], node.parent
            while parent is not None:
                if parent.type in ('class_specifier', 'struct_specifier', 'namespace_definition'):
                    name = parent.child_by_field_name('name')
                    if name: parents.append(raw[name.start_byte:name.end_byte].decode())
                parent = parent.parent
            prefix = '::'.join(reversed(parents))
            result.append(chunk(source, node.start_point.row + 1, node.end_point.row + 1,
                                raw[node.start_byte:node.end_byte].decode(),
                                f'{prefix}: {signature}' if prefix else signature))
            return
        for child in node.named_children:
            visit(child)
    visit(root)
    # Preserve imports, type/field declarations and global declarations as index entries.
    def declarations(node):
        if node.type == 'function_definition': return
        if node.type in ('preproc_include', 'field_declaration', 'declaration', 'alias_declaration'):
            value = raw[node.start_byte:node.end_byte].decode()
            result.append(chunk(source, node.start_point.row + 1, node.end_point.row + 1,
                                value, value))
            return
        for child in node.named_children: declarations(child)
    declarations(root)
    return sorted(result, key=lambda item: (item.start_line, item.id))


def document_chunks(source: str, text: str) -> list[Chunk]:
    """Markdown sections / paragraphs; claims require explicit caller annotations."""
    lines = text.splitlines(keepends=True)
    result, heading, start, buffer = [], '', 1, []
    def flush():
        if ''.join(buffer).strip():
            result.append(chunk(source, start, start + len(buffer) - 1,
                                ''.join(buffer), heading or '(untitled paragraph)'))
    for number, line in enumerate(lines, 1):
        if re.match(r'^#{1,6}\s', line):
            flush(); buffer = []; heading = line.strip(); start = number
        elif not line.strip():
            flush(); buffer = []; start = number + 1
            continue
        if not buffer: start = number
        buffer.append(line)
    flush()
    return result


def terms(text):
    # Characters support Chinese lexical matching; no semantic recall guarantee.
    return re.findall(r'[a-z0-9_]+|[\u4e00-\u9fff]', text.lower())


class Scheduler:
    def __init__(self, chunks: list[Chunk], count_tokens: Callable[[str], int]):
        if len({c.id for c in chunks}) != len(chunks):
            raise ValueError('duplicate chunk IDs')
        self.chunks = {c.id: c for c in chunks}
        self.count = count_tokens  # Must be the target model tokenizer for a token guarantee.

    def route(self, query: str, top_k=2):
        if top_k < 0: raise ValueError('negative top_k')
        query_terms = set(terms(query))
        scored = []
        for c in self.chunks.values():
            words = terms(c.skeleton + '\n' + c.text)
            score = sum(math.log1p(words.count(t)) for t in query_terms)
            if score > 0: scored.append((score, c.id))
        return [identity for _, identity in sorted(scored, key=lambda x: (-x[0], x[1]))[:top_k]]

    def assemble(self, system, skeleton, query, recent, ids, budget, reserve=0):
        if budget <= 0 or reserve < 0 or reserve >= budget:
            raise ValueError('invalid context budget')
        if len(set(ids)) != len(ids): raise ValueError('duplicate requested IDs')
        # Immutable base and constraints are never silently truncated.
        base = f'SYSTEM\n{system}\nSKELETON\n{skeleton}\n'
        tail = f'RECENT\n{recent}\nQUERY\n{query}\n'
        limit = budget - reserve
        if self.count(base + tail) > limit:
            raise ValueError('base skeleton/query/recent exceeds context budget')
        evidence, included, omitted = '', [], []
        for identity in ids:
            c = self.chunks[identity]  # Unknown IDs fail rather than fabricate evidence.
            section = (f'EVIDENCE {c.id} {c.source}:{c.start_line}-{c.end_line}\n'
                       f'{c.text}\nEND EVIDENCE\n')
            if self.count(base + evidence + section + tail) <= limit:
                evidence += section; included.append(identity)
            else: omitted.append(identity)
        prompt = base + evidence + tail
        return {'prompt': prompt, 'tokens': self.count(prompt), 'included': included,
                'omitted': omitted, 'reserved_generation_tokens': reserve}


class State:
    """Explicit, provenance-backed updates. Caller decides what is confirmed."""
    def __init__(self): self.facts = {}

    def write(self, key, value, source_id, expected=None):
        if not key or not source_id: raise ValueError('state updates require key and source')
        old = self.facts.get(key)
        if old is not None and old != (value, source_id) and expected != old:
            raise ValueError('conflicting state update requires explicit prior value')
        self.facts[key] = (value, source_id)


def fold_event(step, tool, status, summary, source_id):
    if not source_id: raise ValueError('event requires original observation reference')
    clean = lambda value: ' '.join(str(value).split())
    return f'[{step}] {clean(tool)} -> {clean(status)}: {clean(summary)} [ref={clean(source_id)}]'
