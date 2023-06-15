import dataclasses

import fsspec
import pyrallis
import pytest

import levanter.config
from haliax.partitioning import ResourceAxis
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


def test_class_registry():
    @levanter.config.config_registry
    @dataclasses.dataclass
    class Person:
        name: str

    @dataclasses.dataclass
    class Adult(Person):
        age: int

    @dataclasses.dataclass
    class Child(Person):
        favorite_toy: str

    Person.register_subclass("adult", Adult)
    Person.register_subclass("child", Child)

    assert pyrallis.decode(Person, {"adult": {"name": "bob", "age": 10}}) == Adult("bob", 10)
    assert pyrallis.decode(Person, {"child": {"name": "bob", "favorite_toy": "truck"}}) == Child("bob", "truck")

    with pytest.raises(Exception):  # pyrallis raises an Exception, not a ValueError
        pyrallis.decode(Person, {"adult": {"name": "bob", "age": 10, "favorite_toy": "truck"}})

    with pytest.raises(ValueError):
        pyrallis.decode(Person, {"baby": {"name": "bob"}})
