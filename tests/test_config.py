import pathlib

from git import Repo

from levanter.config import WandbConfig


def test_infer_experiment_git_root():
    # TODO: bit of a hack since we rely on the fact that this test runs in a git repo
    root = WandbConfig._infer_experiment_git_root()

    # ensure that 1) this is a git root and 2) this source file is underneath
    assert root is not None
    assert pathlib.Path(root).exists()
    repo = Repo(root)
    assert repo.working_dir == root
    print(root, __file__)
    assert pathlib.Path(__file__).is_relative_to(root), f"{__file__} is not relative to {root}"
