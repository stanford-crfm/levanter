# NOTE: Do not explicitly import wandb/other trackers here, as this will cause the tests to trivially pass.
import dataclasses
from typing import Tuple

import pytest
import yaml

from levanter.tracker import TrackerConfig


def test_tracker_plugin_stuff_works():
    assert TrackerConfig.get_choice_class("wandb") is not None
    with pytest.raises(KeyError):
        TrackerConfig.get_choice_class("foo")


def test_tracker_plugin_multi_parsing_work():
    config = """
    tracker:
        type: noop
    """
    parsed = yaml.safe_load(config)

    @dataclasses.dataclass
    class ConfigHolder:
        tracker: TrackerConfig | Tuple[TrackerConfig, ...]

    import draccus

    from levanter.tracker.tracker import NoopTrackerConfig

    assert isinstance(draccus.decode(ConfigHolder, parsed).tracker, NoopTrackerConfig)

    config = """
    tracker:
        - type: noop
        - type: wandb
    """
    parsed = yaml.safe_load(config)
    decoded = draccus.decode(ConfigHolder, parsed).tracker
    assert decoded == (NoopTrackerConfig(), TrackerConfig.get_choice_class("wandb")())
