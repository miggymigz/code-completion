from ccompletion.tokenizer import PythonTokenizer

import fire


def train_tokenizer(
    dataset_dir: str = 'repositories',
    output_path: str = 'vocab.bpe'
):
    tokenizer = PythonTokenizer()


if __name__ == '__main__':
    fire.Fire(train_tokenizer)
