from ccompletion.tokenizer import PythonTokenizer

import fire
import os


def train_tokenizer(
    dataset_dir: str = 'repositories',
    output_path: str = 'vocab.bpe',
    n_vocab: int = 24_000,
    n_threads: int = -1,
):
    # check dataset_dir existence
    if not os.path.isdir(dataset_dir):
        print(f'[ERROR] "{dataset_dir}" is not a directory')
        return

    # check output_path existence
    if os.path.isfile(os.path.join('models', output_path)):
        print(f'[ERROR] "{output_path}" already exists')
        return

    # create models directory if it doesn't exist
    if not os.path.isdir('models'):
        os.mkdir('models')

    tokenizer = PythonTokenizer()
    tokenizer.train(
        dataset_dir=dataset_dir,
        output_path=output_path,
        n_vocab=n_vocab,
        n_threads=n_threads,
    )


if __name__ == '__main__':
    fire.Fire(train_tokenizer)
