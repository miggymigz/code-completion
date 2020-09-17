from abc import ABC, abstractmethod
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Generator, List, Union

from pygments.lexers import PythonLexer
from tokenizers import SentencePieceBPETokenizer
from .utils import read_source_file

import autopep8
import chardet
import os
import tokenize as ptokenize


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    # of unicode characters in your vocab if you want to avoid UNKs.
    This means you need a large
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1)) + \
        list(range(ord("¡"), ord("¬")+1)) + \
        list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def count_files(directory: str) -> int:
    """
    Count total files contained in the directory
    """
    total = 0
    for _, _, files in os.walk(directory):
        total += len(files)

    return total


class BaseTokenizer(ABC):
    def __init__(
        self,
        vocab_path: Optional[Union[str, Path]] = None,
        errors: str = 'replace'
    ):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.errors = errors

        # initialize subword units tokenizer
        # if vocab_path is given and is valid
        if isinstance(vocab_path, str):
            vocab_path = Path(vocab_path)

        if vocab_path and vocab_path.is_dir():
            self.bpe = SentencePieceBPETokenizer(
                vocab_file=str(vocab_path / 'vocab.json'),
                merges_file=str(vocab_path / 'merges.txt'),
            )

    @abstractmethod
    def split(self, src: str, normalize: bool = True, byte_encode: bool = True) -> List[str]:
        """
        Splits source code (string) into tokens
        """
        pass

    @abstractmethod
    def unsplit(self, tokens: List[str], byte_encoded: bool = True) -> str:
        """
        Merges back the tokens split by `self.split` into a single string
        """
        pass

    def byte_encode(self, token):
        return ''.join(self.byte_encoder[c] for c in token.encode('utf-8'))

    def byte_decode(self, token):
        return bytearray([self.byte_decoder[c] for c in token]).decode('utf-8')

    def encode(self, src: str, return_ids: bool = True) -> List[int]:
        """
        Splits the given source code into tokens and then into subwords.
        After that, the subwords are then converted into ids if `return_ids` is true.
        """
        tokens = list(self.split(src))
        output = self.bpe.encode(' '.join(tokens))
        return output.ids if return_ids else output.tokens

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes the given subword ids (created by `self.encode`) into string.
        """
        tokens = self.bpe.decode(token_ids).split()
        return self.unsplit(tokens)

    def train(
        self,
        repos_path: Path,
        output_path: Path,
        n_vocab: int
    ):
        """
        Trains bpe to all of the python source files located in `repos_path`
        """
        if hasattr(self, 'bpe'):
            raise AssertionError(f'Cannot train on already trained tokenizer!')

        # concatenate dataset first before training BPE
        concatenated_output_path = output_path / '_aux_concat'
        if concatenated_output_path.exists():
            print('INFO - Will skip concatenating dataset files')
        else:
            self.__concatenate_dataset(
                repos_path=repos_path,
                output_path=concatenated_output_path,
            )

        # train BPE using _training_aux
        bpe = SentencePieceBPETokenizer()
        bpe.train([str(concatenated_output_path)], vocab_size=n_vocab)
        bpe.save_model(str(output_path))

    def __concatenate_dataset(self, repos_path: Path, output_path: Path):
        # make sure repos directory exists
        assert repos_path.exists() and repos_path.is_dir()

        # count number of python source files for progress tracking
        file_count = count_files(repos_path)

        with tqdm(total=file_count) as t, open(output_path, 'w', encoding='utf-8') as ofd:
            for root, _, files in os.walk(repos_path):
                for pf in files:
                    # all files should be python source files
                    assert pf.endswith('.py')

                    pf_path = os.path.join(root, pf)
                    t.set_description(pf_path)

                    # read python source file
                    # ignore if failed to decode
                    try:
                        src, enc = read_source_file(pf_path)
                    except UnicodeDecodeError:
                        t.write(f'WARN - Unknown encoding: {pf_path}')
                        t.update()
                        continue

                    try:
                        tokens = self.split(src, encoding=enc)
                    except (ptokenize.TokenError, IndentationError, SyntaxError):
                        # ignore python files that could not be tokenized
                        # as they may be used by test files e.g. (google/pytype/tokenerror1.py)
                        # this way, our dataset will only contain grammatical python source files
                        t.write(f'WARN - Malformed source file: {pf_path}')
                    else:
                        concatenated = ' '.join(tokens)
                        print(concatenated, file=ofd)
                    finally:
                        t.update()


class PythonTokenizer(BaseTokenizer):
    def split(
        self,
        src: str,
        normalize: bool = True,
        byte_encode: bool = True,
        encoding: str = 'utf-8'
    ) -> List[str]:
        tokens = []
        src_as_bytes = BytesIO(src.encode(encoding))

        for token_info in ptokenize.tokenize(src_as_bytes.readline):
            token_type = token_info.exact_type
            token_value = token_info.string
            token = f'{token_type}|{token_value}'

            if byte_encode:
                token = self.byte_encode(token)

            tokens.append(token)

        return tokens

    def unsplit(
        self,
        tokens: List[str],
        byte_encoded: bool = True
    ) -> str:
        result = []
        for token in tokens:
            if byte_encoded:
                token = self.byte_decode(token)

            token_type, token_value = token.split('|', 1)
            result.append((int(token_type), token_value))

        untokenized = ptokenize.untokenize(result)
        untokenized_decoded = untokenized.decode('utf-8', errors=self.errors)
        return untokenized_decoded


class PygmentsTokenizer(BaseTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lexer = PythonLexer()

    def split(
        self,
        src: str,
        normalize: bool = True,
        byte_encode: bool = True,
        encoding: str = 'utf-8'
    ) -> List[str]:
        tokens = [token for _, token in self.lexer.get_tokens(src)]

        if not normalize:
            last_src_char = src[-1]
            last_token = tokens[-1]

            # Pygments adds an extra linebreak token at the end
            if not last_src_char.isspace():
                del tokens[-1]
            elif last_src_char != last_token:
                tokens[-1] = last_src_char

        if byte_encode:
            tokens = [self.byte_encode(token) for token in tokens]

        return tokens

    def unsplit(
        self,
        tokens: List[str],
        byte_encoded: bool = True
    ) -> str:
        tokens = [self.byte_decode(token) for token in tokens]
        return ''.join(tokens)
