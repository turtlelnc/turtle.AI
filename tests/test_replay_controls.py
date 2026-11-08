import unittest
import torch
from tools.replay_binding import train_parts

class ReplayControlTests(unittest.TestCase):
    def test_exposure_and_loss_weights(self):
        new=(torch.arange(16),)*3;old=(torch.arange(8)+100,)*3
        full=train_parts(new,old,'new_only')
        half=train_parts(new,old,'half_new')
        replay=train_parts(new,old,'replay')
        self.assertEqual(sum(len(x[0])*w for x,w in full),16)
        self.assertEqual(sum(len(x[0])*w for x,w in half),8)
        self.assertEqual(sum(len(x[0])*w for x,w in replay),16)
        self.assertTrue(torch.equal(half[0][0][0],replay[0][0][0]))
        self.assertEqual(half[1][1],0)
        self.assertEqual(replay[1][1],1)

if __name__=='__main__':unittest.main()
