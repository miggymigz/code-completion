from ccompletion.tokenizers import PygmentsTokenizer, PythonTokenizer

import fire
import os


def train_tokenizer(
    tokenizer: str = 'pygments',
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

    # rename output path to include tokenizer
    concatenated_output_path = f'_{tokenizer}_aux'
    output_path = os.path.join('models', f'{tokenizer}-{output_path}')

    tokenizer = get_tokenizer(tokenizer)
    tokenizer.train(
        dataset_dir=dataset_dir,
        concatenated_output_path=concatenated_output_path,
        output_path=output_path,
        n_vocab=n_vocab,
        n_threads=n_threads,
    )


def get_tokenizer(tokenizer: str):
    if tokenizer == 'pygments':
        return PygmentsTokenizer()

    if tokenizer == 'python':
        return PythonTokenizer()

    raise AssertionError(f'Unknown tokenizer: {tokenizer}')


if __name__ == '__main__':
    fire.Fire(train_tokenizer)
