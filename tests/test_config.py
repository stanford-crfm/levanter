import dataclasses

import fsspec

from haliax.partitioning import ResourceAxis

import levanter.config
from levanter.data.text import LMDatasetConfig, LMMixtureDatasetConfig
from levanter.trainer import TrainerConfig


def test_main_wrapper_loads_from_fsspec():
    with fsspec.open("memory://test.yaml", "w") as f:
        f.write(
            """
        project: test
        """
        )

    args = ["--config_path", "memory://test.yaml", "--x", "2"]

    @dataclasses.dataclass
    class Config:
        project: str
        x: int = 1

    @levanter.config.main(args=args)
    def main(config: Config):
        assert config.project == "test"
        assert config.x == 2

    main()


def test_new_style_axis_mapping():
    config = TrainerConfig(
        tensor_parallel_axes=["a1", "a2"],
    )

    assert config.tensor_parallel_axes == ["a1", "a2"]
    assert config.compute_axis_mapping == {
        "batch": ResourceAxis.DATA,
        "a1": ResourceAxis.MODEL,
        "a2": ResourceAxis.MODEL,
    }
    assert config.parameter_axis_mapping == {
        "embed": ResourceAxis.DATA,
        "a1": ResourceAxis.MODEL,
        "a2": ResourceAxis.MODEL,
        "batch": ResourceAxis.DATA,
    }


def test_lm_dataset_config():
    @dataclasses.dataclass
    class Config:
        data: LMDatasetConfig = dataclasses.field(default_factory=LMDatasetConfig)

    yaml_config = """
    data:
        id: dlwh/wikitext_103_detokenized
        cache_dir: "gs://levanter-data/tokenized/wikitext"
    """
    args = ["--config_path", _write_yaml_to_memory(yaml_config)]

    @levanter.config.main(args=args)
    def main(config: Config):
        assert config.data.id == "dlwh/wikitext_103_detokenized"
        assert config.data.cache_dir == "gs://levanter-data/tokenized/wikitext"

    main()


def test_lm_mixture_dataset_config():
    @dataclasses.dataclass
    class Config:
        data_mixture: LMMixtureDatasetConfig = dataclasses.field(default_factory=LMMixtureDatasetConfig)

    yaml_config = """
    data_mixture:
        configs:
            pile:
                train_urls:
                    - gs://levanter-data/pile/train/{00..29}.jsonl.zst
                validation_urls:
                    - gs://levanter-data/pile/val.jsonl.zst
                cache_dir: "gs://levanter-data/tokenized/pile-old/"
                tokenizer: "EleutherAI/gpt-neox-20b"
            redpajama:
                id: togethercomputer/RedPajama-Data-1T
                cache_dir: gs://levanter-data/tokenized/redpajama/
                tokenizer: EleutherAI/gpt-neox-20b
                splits:
                    - train
                rows_per_chunk: 4096
        weights:
            pile: 0.6
            redpajama: 0.4
    """
    args = ["--config_path", _write_yaml_to_memory(yaml_config)]

    @levanter.config.main(args=args)
    def main(config: Config):
        assert config.data_mixture is not None

    main()


def _write_yaml_to_memory(yaml: str, path: str = "memory://test.yaml"):
    with fsspec.open(path, "w") as f:
        f.write(yaml)
    return path
