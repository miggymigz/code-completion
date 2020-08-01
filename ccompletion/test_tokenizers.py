from .tokenizers import PygmentsTokenizer

import unittest


class TestPygmentsTokenizer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = PygmentsTokenizer()

    def test_split_last_word(self):
        # last token: word
        tokens = self.tokenizer.split(
            'import',
            normalize=False,
            byte_encoded=False,
        )
        self.assertEqual(tokens, ['import'])

    def test_split_last_newline(self):
        # last token: newline
        tokens = self.tokenizer.split(
            'import\n',
            normalize=False,
            byte_encoded=False,
        )
        self.assertEqual(tokens, ['import', '\n'])

    def test_split_last_space(self):
        # last token: space
        tokens = self.tokenizer.split(
            'import ',
            normalize=False,
            byte_encoded=False,
        )
        self.assertEqual(tokens, ['import', ' '])

    def test_split(self):
        tokens = self.tokenizer.split(
            'import tensorflow',
            normalize=True,
            byte_encoded=False,
        )

        self.assertEqual(tokens, ['import', ' ', 'tensorflow', '\n'])

    def test_split_unsplit(self):
        src = 'import tensorflow\n'
        tokens = self.tokenizer.split(src)
        output = self.tokenizer.unsplit(tokens)

        self.assertEqual(src, output)


if __name__ == '__main__':
    unittest.main()
