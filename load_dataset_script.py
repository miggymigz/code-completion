from dataclasses import dataclass
from pathlib import Path

import datasets

FEATURES = datasets.Features(
    {
        "src": datasets.Value("string"),
    }
)


@dataclass
class PythonRepositoriesConfig(datasets.BuilderConfig):
    encoding: str = "utf-8"
    chunksize: int = 10 << 20  # 10MB


class PythonRepositoriesDataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = PythonRepositoriesConfig

    def _info(self):
        return datasets.DatasetInfo(features=FEATURES)

    def _split_generators(self, dl_manager):
        if not self.config.data_dir:
            raise ValueError(
                "Specify the location of the repositories directory or zip file.")

        repositories_path = self.config.data_dir
        return [datasets.SplitGenerator(
            name=datasets.Split.TRAIN,
            gen_kwargs={"path": repositories_path},
        )]

    def _generate_examples(self, path):
        for f in Path(path).rglob('*.py'):
            if f.is_file():
                with f.open('r', encoding=self.config.encoding) as fd:
                    yield str(f), {'src': fd.read().strip()}
