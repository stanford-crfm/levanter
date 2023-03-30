import dataclasses
import pathlib

import fsspec
import pytest
from git import InvalidGitRepositoryError, NoSuchPathError, Repo

import levanter.config
from levanter.config import WandbConfig


def test_infer_experiment_git_root():
    # make sure this test is running in a git repo
    try:
        repo = Repo(pathlib.Path(__file__), search_parent_directories=True)
    except (InvalidGitRepositoryError, NoSuchPathError):
        pytest.skip("test not running in a git repo")

    root = WandbConfig._infer_experiment_git_root()

    # ensure that 1) this is a git root and 2) this source file is underneath
    assert root is not None
    assert pathlib.Path(root).exists()
    repo = Repo(root)
    assert repo.working_dir == root
    print(root, __file__)
    assert pathlib.Path(__file__).is_relative_to(root), f"{__file__} is not relative to {root}"


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

    @levanter.config.main(args)
    def main(config: Config):
        assert config.project == "test"
        assert config.x == 2

    main()
