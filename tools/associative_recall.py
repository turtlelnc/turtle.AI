"""Random per-document associations with same-key contradictory IGNORE records.

Queries require the PAIR value. Location supervision is synthetic annotation,
never an input token. Held-out documents redraw all keys and values.
"""
import torch

KEYS = 256
VALUES = 32
PAIR, QUERY, IGNORE, SEPARATOR = 320, 321, 322, 323
VOCAB_SIZE = 324
BLOCK_SIZE = 8


def batch(generator, size, blocks, working_blocks=1, key_count=KEYS, value_count=VALUES,
          contradictory=True):
    if (not 1 <= key_count <= KEYS or not 2 <= value_count <= VALUES or size < 1 or
            not 2 <= blocks <= key_count or not 0 <= working_blocks < blocks):
        raise ValueError('invalid association task dimensions')
    # Top-k random continuous priorities implement sampling without replacement.
    keys = torch.rand(size, key_count, generator=generator).argsort(-1)[:, :blocks] + 1
    values = torch.randint(value_count, (size, blocks), generator=generator)
    decoys = (values + torch.randint(1, value_count, (size, blocks), generator=generator)) % value_count
    data = torch.empty(size, blocks, BLOCK_SIZE, dtype=torch.long)
    data[:, :, 0], data[:, :, 1] = PAIR, keys
    data[:, :, 2], data[:, :, 3] = values + KEYS + 1, SEPARATOR
    data[:, :, 4], data[:, :, 5] = IGNORE, keys
    data[:, :, 6], data[:, :, 7] = decoys + KEYS + 1, SEPARATOR
    if not contradictory:
        # Same record markers, but no contradictory key/value evidence.
        data[:, :, 5:7] = torch.randint(289, 320, (size, blocks, 2), generator=generator)
    # Query any association that has left the completed working-block ring.
    target_blocks = torch.randint(blocks - working_blocks, (size,), generator=generator)
    rows = torch.arange(size)
    query = torch.stack((torch.full((size,), QUERY), keys[rows, target_blocks]), -1)
    ids = torch.cat((data.flatten(1), query), -1)
    labels, routes = torch.full_like(ids, -100), torch.full_like(ids, -100)
    labels[:, -1] = values[rows, target_blocks] + KEYS + 1
    routes[:, -1] = target_blocks
    return ids, labels, routes
