import unittest
from tools.curriculum_memory import promotion

class CurriculumTests(unittest.TestCase):
    def test_requires_two_consecutive_passes(self):
        self.assertFalse(promotion([1.],.95,2))
        self.assertFalse(promotion([1.,.8,1.],.95,2))
        self.assertTrue(promotion([.8,.96,.98],.95,2))

if __name__=='__main__':unittest.main()
