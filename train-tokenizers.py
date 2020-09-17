from pathlib import Path
from ccompletion.tokenisers import PythonTokenizer, PygmentsTokenizer

import fire


def train_tokenizers(repos: str = 'repositories', n_vocab: int = 25_000):
    repos_path = Path(repos)
    assert repos_path.exists() and repos_path.is_dir()

    models_path = Path('models')
    models_path.mkdir(parents=False, exist_ok=True)

    # train `tokenize` tokenizer
    tokenize_path = models_path / 'tokenize'
    tokenize_path.mkdir(parents=False, exist_ok=True)
    tokenizer = PythonTokenizer()
    tokenizer.train(
        repos_path=repos_path,
        output_path=tokenize_path,
        n_vocab=n_vocab,
    )

    # train `Pygments` tokenizer
    pygments_path = models_path / 'pygments'
    pygments_path.mkdir(parents=False, exist_ok=True)
    tokenizer = PygmentsTokenizer()
    tokenizer.train(
        repos_path=repos_path,
        output_path=pygments_path,
        n_vocab=n_vocab,
    )


if __name__ == '__main__':
    fire.Fire(train_tokenizers)
