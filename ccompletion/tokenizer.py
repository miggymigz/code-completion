from functools import lru_cache
from io import BytesIO
from tqdm import tqdm
from typing import Optional, Generator

import codecs
import json
import os
import token
import tokenize as ptokenize
import youtokentome as yttm


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


class PythonTokenizer:
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        errors: str = 'replace'
    ):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        self.bpe = yttm.BPE(model=vocab_file) if vocab_file else None
        self.errors = errors

    def split(self, src: str) -> Generator[str, None, None]:
        """
        Splits source code (string) into python tokens
        """
        src_as_bytes = BytesIO(src.encode('utf-8'))
        for token_info in ptokenize.tokenize(src_as_bytes.readline):
            token_type = token_info.exact_type
            token_value = token_info.string
            token = f'{token_type}||{token_value}'
            yield ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))

    def unsplit(self, tokens: [str]) -> str:
        """
        Merges back the tokens split by `self.split`.
        """
        result = []
        for token in tokens:
            token_as_bytes = bytearray([self.byte_decoder[c] for c in token])
            token = token_as_bytes.decode('utf-8', errors=self.errors)
            token_type, token_value = token.split('||', 1)
            result.append((int(token_type), token_value))

        untokenized = ptokenize.untokenize(result)
        untokenized_decoded = untokenized.decode('utf-8', errors=self.errors)
        return untokenized_decoded

    def encode(self, src: str, return_ids: bool = True) -> [int]:
        """
        Splits the given source code into tokens and then into subwords.
        After that, the subwords are then converted into ids if `return_ids` is true.
        """
        tokens = list(self.split(src))
        output_type = yttm.OutputType.ID if return_ids else yttm.OutputType.SUBWORD
        bpe_result = self.bpe.encode(tokens, output_type=output_type)

        return [t for tt in bpe_result for t in tt]

    def decode(self, token_ids: [int]) -> str:
        """
        Decodes the given subword ids (created by `self.encode`) into string.
        """
        tokens = self.bpe.decode(token_ids)[0].split()
        return self.unsplit(tokens)

    def train(self, dataset_dir: str, output_path: str, n_vocab: int, n_threads: int = -1) -> None:
        """
        Trains bpe to all of the python source files located in `dataset_dir`
        """
        if self.bpe is not None:
            raise AssertionError(f'Cannot train on already trained tokenizer!')

        # concatenate dataset first before training BPE
        concatenated_output_path = '_training_aux'
        if os.path.isfile(concatenated_output_path):
            print('INFO - Will skip concatenating dataset files')
        else:
            self.__concatenate_dataset(
                dataset_dir=dataset_dir,
                output_path=concatenated_output_path,
            )

        # train BPE using _training_aux
        self.bpe = yttm.BPE.train(
            data=concatenated_output_path,
            model=output_path,
            vocab_size=n_vocab,
            n_threads=n_threads,
        )

    def __concatenate_dataset(self, dataset_dir: str, output_path: str) -> None:
        file_count = count_files(dataset_dir)
        with tqdm(total=file_count) as t, codecs.open(output_path, 'w') as ofd:
            for root, _, files in os.walk(dataset_dir):
                for pf in files:
                    # skip files that are not python src codes
                    if not pf.endswith('.py'):
                        t.update()
                        continue

                    pf_path = os.path.join(root, pf)
                    t.set_description(pf_path)

                    # open each src file and collate all unique tokens
                    with codecs.open(pf_path, 'r', 'utf8', errors=self.errors) as fd:
                        src_code = fd.read().strip()

                        try:
                            tokens = tuple(self.split(src_code))
                        except (ptokenize.TokenError, IndentationError, SyntaxError):
                            # ignore python files that could not be tokenized
                            # as they may be used by test files e.g. (google/pytype/tokenerror1.py)
                            # this way, our dataset will only contain grammatical python source files
                            t.write(f'WARN - Malformed source file: {pf_path}')
                        except (LookupError, UnicodeDecodeError):
                            # some python files in the repositories use encodings other than
                            t.write(f'WARN - Unsupported encoding: {pf_path}')
                        else:
                            concatenated = ' '.join(tokens)
                            print(concatenated, file=ofd)
                        finally:
                            t.update()
