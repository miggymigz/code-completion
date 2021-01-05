from dataclasses import dataclass
from lib2to3.pgen2.parse import ParseError
from pathlib import Path
from transformers import T5TokenizerFast

import ast
import autopep8
import datasets
import itertools


FEATURES = datasets.Features(
    {
        "src": datasets.Value("string"),
        "target": datasets.Value("string"),
    }
)


@dataclass
class PythonRepositoriesConfig(datasets.BuilderConfig):
    model: str = None
    clean: bool = False
    encoding: str = "utf-8"
    maxfilesize: int = 1 << 20  # 1MB


class PythonRepositoriesDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = PythonRepositoriesConfig

    def _info(self):
        return datasets.DatasetInfo(features=FEATURES)

    def _split_generators(self, dl_manager):
        if not self.config.data_dir:
            raise ValueError(
                "Specify the location of the repositories directory or zip file.")

        # prepare T5 tokenizer if model is T5
        if self.config.model == 't5':
            self.tokenizer = T5TokenizerFast.from_pretrained('t5-base')

        return [datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={
                "path": self.config.data_dir,
                "model": self.config.model,
                "clean": self.config.clean,
                "maxfilesize": self.config.maxfilesize,
            },
        )]

    def _generate_examples(self, path: str, model: str, clean: bool, maxfilesize: int):
        files = Path(path).rglob('*.py')

        # TODO: remove slice
        for f in itertools.islice(files, 100):
            # skip directories whose names end in .py
            if not f.is_file():
                continue

            # read file contents into memory
            with f.open('r', encoding=self.config.encoding) as fd:
                try:
                    src = fd.read().strip()
                    raise_error_if_empty(src)
                    ast.parse(src)

                    # skip files greater than `maxfilesize`
                    if f.stat().st_size > maxfilesize:
                        continue

                    # optional source code formatting
                    if clean:
                        src = autopep8.fix_code(src)

                    # yield samples
                    if model == 't5':
                        yield from self.__as_t5_example(str(f), src)
                    else:
                        yield str(f), {
                            'src': src,
                            'target': None,
                        }
                except (UnicodeDecodeError, ValueError, SyntaxError, ParseError):
                    pass

    def __as_t5_example(self, path: str, src: str):
        lines = src.splitlines()

        # atleast two lines is required to make a T5 sample
        if len(lines) < 2:
            return

        for i in range(1, len(lines)):
            inputs = self.tokenizer(lines[:i]).input_ids
            inputs = [_id for ids in inputs for _id in ids]

            # ignore remaining samples if model size can't accomodate
            if len(inputs) > self.tokenizer.model_max_length:
                break

            yield f'{path}-{i}', {
                'src': '\n'.join(lines[:i]),
                'target': lines[i],
            }


def raise_error_if_empty(src: str):
    if src == '':
        raise ValueError('empty src')
