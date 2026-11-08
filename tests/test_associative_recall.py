import unittest
import torch
from tools.associative_recall import batch, KEYS, VALUES, PAIR, QUERY, IGNORE, SEPARATOR


class AssociationTaskTests(unittest.TestCase):
    def test_unambiguous_keys_decoys_and_targets(self):
        ids, labels, routes = batch(torch.Generator().manual_seed(10), 16, 12)
        self.assertEqual(ids.shape, (16, 98))
        for row in range(len(ids)):
            records = ids[row, :-2].reshape(12, 8)
            self.assertEqual(len(set(records[:, 1].tolist())), 12)
            self.assertTrue((records[:, 0] == PAIR).all())
            self.assertTrue((records[:, 4] == IGNORE).all())
            self.assertTrue((records[:, 1] == records[:, 5]).all())
            self.assertTrue((records[:, 2] != records[:, 6]).all())
            target = int(routes[row, -1])
            self.assertLess(target, 11)
            self.assertEqual(int(ids[row, -2]), QUERY)
            self.assertEqual(int(ids[row, -1]), int(records[target, 1]))
            self.assertEqual(int(labels[row, -1]), int(records[target, 2]))
        self.assertTrue((labels[:, :-1] == -100).all())
        self.assertTrue((routes[:, :-1] == -100).all())

if __name__ == '__main__': unittest.main()
