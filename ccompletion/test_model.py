from .model import batch_attention_mask

import unittest


class TestModel(unittest.TestCase):
    def test_attention_mask1(self):
        batch = [
            [1],
            [1, 2],
            [1, 2, 3],
        ]
        mask = batch_attention_mask(batch)
        self.assertEqual(mask, [
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
        ])

    def test_attention_mask(self):
        batch = [
            [1, 2, 3],
            [1, 2],
            [1],
        ]
        mask = batch_attention_mask(batch)
        self.assertEqual(mask, [
            [1, 1, 1],
            [1, 1, 0],
            [1, 0, 0],
        ])


if __name__ == '__main__':
    unittest.main()
