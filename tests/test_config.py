import dataclasses
import jax # Added import
import fsspec
import copy # For deepcopying jax_config

from haliax.partitioning import ResourceAxis

import levanter.config
from levanter.data.text import HfSingleDatasetLMConfig, LMMixtureDatasetConfig
from levanter.trainer import TrainerConfig, DEFAULT_JAX_CONFIG # Added import
from levanter.tracker.tracker import NoopConfig # Added import


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
        "batch": (ResourceAxis.REPLICA, ResourceAxis.DATA),
        "a1": ResourceAxis.MODEL,
        "a2": ResourceAxis.MODEL,
    }
    assert config.parameter_axis_mapping == {
        "embed": ResourceAxis.DATA,
        "a1": ResourceAxis.MODEL,
        "a2": ResourceAxis.MODEL,
        "batch": (ResourceAxis.REPLICA, ResourceAxis.DATA),
    }


def test_lm_dataset_config():
    @dataclasses.dataclass
    class Config:
        data: HfSingleDatasetLMConfig = dataclasses.field(default_factory=HfSingleDatasetLMConfig)

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
        data: LMMixtureDatasetConfig = dataclasses.field(default_factory=LMMixtureDatasetConfig)

    yaml_config = """
    data:
        configs:
            owt:
                train_urls:
                    - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_train.{1..128}-of-128.jsonl.gz"
                validation_urls:
                    - "gs://pubmed-mosaic/openwebtext-sharded/openwebtext_val.{1..8}-of-8.jsonl.gz"
            wikitext:
                id: dlwh/wikitext_103_detokenized
        train_weights:
            owt: 0.6
            wikitext: 0.4
        tokenizer: gpt2
        cache_dir: "gs://levanter-data/tokenized/mixture"
    """
    args = ["--config_path", _write_yaml_to_memory(yaml_config)]

    @levanter.config.main(args=args)
    def main(config: Config):
        assert config.data is not None
        # TODO: assert more things

    main()


def test_jax_compilation_cache_config():
    # Store original JAX config values
    original_cache_dir = getattr(jax.config, "jax_compilation_cache_dir", None)
    original_min_compile_time = getattr(jax.config, "jax_persistent_cache_min_compile_time_secs", None)
    original_min_entry_size = getattr(jax.config, "jax_persistent_cache_min_entry_size_bytes", None)
    original_enable_xla_caches = getattr(jax.config, "jax_persistent_cache_enable_xla_caches", None)

    # JAX internal defaults (as of jax 0.4.23, these might change)
    # These are used to check against if the original values were None (meaning JAX was using its internal defaults)
    jax_default_min_compile_time = 1.0
    jax_default_min_entry_size = 0


    try:
        # Test Case 1: Local path for jax_compilation_cache_dir
        # Ensure other cache settings are NOT affected by TrainerConfig's direct fields (only by jax_config)
        trainer_config_local = TrainerConfig(
            jax_compilation_cache_dir="/tmp/test_jax_cache",
            require_accelerator=False,
            tracker=(NoopConfig(),),
        )
        trainer_config_local.initialize()

        assert jax.config.jax_compilation_cache_dir == "/tmp/test_jax_cache"
        # These should remain their original values or JAX's internal defaults if original was None
        assert jax.config.jax_persistent_cache_min_compile_time_secs == (original_min_compile_time if original_min_compile_time is not None else jax_default_min_compile_time)
        assert jax.config.jax_persistent_cache_min_entry_size_bytes == (original_min_entry_size if original_min_entry_size is not None else jax_default_min_entry_size)
        assert jax.config.jax_persistent_cache_enable_xla_caches == original_enable_xla_caches


        # Reset for next test case to ensure isolation
        jax.config.update("jax_compilation_cache_dir", original_cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", original_min_compile_time if original_min_compile_time is not None else jax_default_min_compile_time)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", original_min_entry_size if original_min_entry_size is not None else jax_default_min_entry_size)
        if original_enable_xla_caches is not None: # Only update if it was something, otherwise it might be unset
            jax.config.update("jax_persistent_cache_enable_xla_caches", original_enable_xla_caches)
        else: # If original was None, ensure it's None for the next test too, unless default jax_config sets it
            if "jax_persistent_cache_enable_xla_caches" not in DEFAULT_JAX_CONFIG:
                 jax.config.update("jax_persistent_cache_enable_xla_caches", None)


        # Test Case 2: GCS path for jax_compilation_cache_dir
        trainer_config_gcs = TrainerConfig(
            jax_compilation_cache_dir="gs://my-bucket/test_jax_cache",
            require_accelerator=False,
            tracker=(NoopConfig(),),
        )
        trainer_config_gcs.initialize()
        assert jax.config.jax_compilation_cache_dir == "gs://my-bucket/test_jax_cache"
        # These should also remain their original values or JAX's internal defaults
        assert jax.config.jax_persistent_cache_min_compile_time_secs == (original_min_compile_time if original_min_compile_time is not None else jax_default_min_compile_time)
        assert jax.config.jax_persistent_cache_min_entry_size_bytes == (original_min_entry_size if original_min_entry_size is not None else jax_default_min_entry_size)
        assert jax.config.jax_persistent_cache_enable_xla_caches == original_enable_xla_caches

    finally:
        # Restore original JAX config values
        jax.config.update("jax_compilation_cache_dir", original_cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", original_min_compile_time if original_min_compile_time is not None else jax_default_min_compile_time)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", original_min_entry_size if original_min_entry_size is not None else jax_default_min_entry_size)
        # Only update if it was something, otherwise it might be unset vs set to None by jax.config.update
        if original_enable_xla_caches is not None:
            jax.config.update("jax_persistent_cache_enable_xla_caches", original_enable_xla_caches)
        else: # if original was None, ensure it's set back to None if it's not in default jax_config
            current_default_jax_config = copy.deepcopy(DEFAULT_JAX_CONFIG)
            if "jax_persistent_cache_enable_xla_caches" not in current_default_jax_config:
                jax.config.update("jax_persistent_cache_enable_xla_caches", None)


def test_advanced_jax_cache_settings_via_jax_config():
    original_cache_dir = getattr(jax.config, "jax_compilation_cache_dir", None)
    original_min_compile_time = getattr(jax.config, "jax_persistent_cache_min_compile_time_secs", None)
    original_min_entry_size = getattr(jax.config, "jax_persistent_cache_min_entry_size_bytes", None)
    original_enable_xla_caches = getattr(jax.config, "jax_persistent_cache_enable_xla_caches", None)

    # JAX internal defaults
    jax_default_min_compile_time = 1.0
    jax_default_min_entry_size = 0

    try:
        custom_jax_config = copy.deepcopy(DEFAULT_JAX_CONFIG)
        custom_jax_config["jax_persistent_cache_min_compile_time_secs"] = 10.0
        custom_jax_config["jax_persistent_cache_min_entry_size_bytes"] = 2048
        custom_jax_config["jax_persistent_cache_enable_xla_caches"] = "xla_gpu_kernel_cache_file"
        # Optionally, set a custom cache dir via jax_config as well, or test it separately
        custom_jax_config["jax_compilation_cache_dir"] = "/tmp/advanced_test_cache"

        trainer_config = TrainerConfig(
            jax_config=custom_jax_config,
            require_accelerator=False,
            tracker=(NoopConfig(),),
        )
        trainer_config.initialize()

        assert jax.config.jax_compilation_cache_dir == "/tmp/advanced_test_cache"
        assert jax.config.jax_persistent_cache_min_compile_time_secs == 10.0
        assert jax.config.jax_persistent_cache_min_entry_size_bytes == 2048
        assert jax.config.jax_persistent_cache_enable_xla_caches == "xla_gpu_kernel_cache_file"

    finally:
        # Restore original JAX config values
        jax.config.update("jax_compilation_cache_dir", original_cache_dir)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", original_min_compile_time if original_min_compile_time is not None else jax_default_min_compile_time)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", original_min_entry_size if original_min_entry_size is not None else jax_default_min_entry_size)
        if original_enable_xla_caches is not None:
            jax.config.update("jax_persistent_cache_enable_xla_caches", original_enable_xla_caches)
        else:
            current_default_jax_config = copy.deepcopy(DEFAULT_JAX_CONFIG)
            if "jax_persistent_cache_enable_xla_caches" not in current_default_jax_config:
                 jax.config.update("jax_persistent_cache_enable_xla_caches", None)


def _write_yaml_to_memory(yaml: str, path: str = "memory://test.yaml"):
    with fsspec.open(path, "w") as f:
        f.write(yaml)
    return path
