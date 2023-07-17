import dataclasses
from typing import Union

import fsspec

from haliax.partitioning import ResourceAxis

import levanter.config
from levanter.data.text import LMDatasetConfig, LMMixtureDatasetConfig
from levanter.trainer import TrainerConfig


def test_main_wrapper_loads_from_fsspec():
    yaml_config = """
    project: test
    """
    args = ["--config_path", _write_yaml_to_memory(yaml_config), "--x", "2"]

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
        data: Union[LMDatasetConfig, LMMixtureDatasetConfig] = dataclasses.field(default_factory=LMDatasetConfig)

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
        data: Union[LMDatasetConfig, LMMixtureDatasetConfig] = dataclasses.field(default_factory=LMDatasetConfig)

    yaml_config = """
    data:
        datasets:
            - cache_dir: "gs://levanter-data/tokenized/openwebtext/"
              tokenizer: "EleutherAI/gpt-neox-20b"
              weight: 0.5
            - cache_dir: "gs://levanter-data/tokenized/pile-old/"
              tokenizer: "EleutherAI/gpt-neox-20b"
              weight: 0.5
    """
    args = ["--config_path", _write_yaml_to_memory(yaml_config)]

    @levanter.config.main(args=args)
    def main(config: Config):
        assert len(config.data.datasets) == 2
        assert config.data.datasets[0].cache_dir == "gs://levanter-data/tokenized/openwebtext/"
        assert config.data.datasets[0].tokenizer == "EleutherAI/gpt-neox-20b"
        assert config.data.datasets[0].weight == 0.5
        assert config.data.datasets[1].cache_dir == "gs://levanter-data/tokenized/pile-old/"
        assert config.data.datasets[1].tokenizer == "EleutherAI/gpt-neox-20b"
        assert config.data.datasets[1].weight == 0.5

    main()


def _write_yaml_to_memory(yaml: str, path: str = "memory://test.yaml"):
    with fsspec.open(path, "w") as f:
        f.write(yaml)
    return path
