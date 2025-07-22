from ray.runtime_env import RuntimeEnv

from levanter.utils.ray_utils import RayResources


class TestRayResources:
    def test_merge_env_vars_basic(self):
        """Test basic environment variable merging."""
        resources = RayResources(num_cpus=2, num_gpus=1)

        # Add some environment variables
        result = resources.merge_env_vars({"FOO": "bar", "BAZ": "qux"})

        assert result.num_cpus == 2
        assert result.num_gpus == 1
        assert result.runtime_env.get("env_vars") == {"FOO": "bar", "BAZ": "qux"}

    def test_merge_env_vars_additive(self):
        """Test that environment variable merging is additive."""
        # Start with existing env vars
        base_env = RuntimeEnv(env_vars={"EXISTING": "value", "FOO": "original"})
        resources = RayResources(num_cpus=1, runtime_env=base_env)

        # Merge new env vars
        result = resources.merge_env_vars({"FOO": "new_value", "NEW": "added"})

        expected_env_vars = {
            "EXISTING": "value",  # preserved
            "FOO": "new_value",   # overridden
            "NEW": "added"        # added
        }
        assert result.runtime_env.get("env_vars") == expected_env_vars

    def test_merge_env_vars_none(self):
        """Test that None env_vars returns original unchanged."""
        resources = RayResources(num_cpus=1)
        result = resources.merge_env_vars(None)

        assert result is resources  # should be the same instance

    def test_merge_env_vars_empty(self):
        """Test that empty env_vars returns original unchanged."""
        resources = RayResources(num_cpus=1)
        result = resources.merge_env_vars({})

        assert result is resources  # should be the same instance

    def test_merge_runtime_env_basic(self):
        """Test basic runtime environment merging."""
        resources = RayResources(num_cpus=2)

        # Merge a runtime environment with env_vars and pip
        new_runtime_env = RuntimeEnv(
            env_vars={"FOO": "bar"},
            pip={"packages": ["requests"]}
        )

        result = resources.merge_runtime_env(new_runtime_env)

        assert result.num_cpus == 2
        assert result.runtime_env.get("env_vars") == {"FOO": "bar"}
        assert result.runtime_env.get("pip") == {"packages": ["requests"], "pip_check": False}

    def test_merge_runtime_env_additive(self):
        """Test that runtime environment merging is additive."""
        # Start with existing runtime env
        base_env = RuntimeEnv(
            env_vars={"EXISTING": "value"},
            pip={"packages": ["numpy"]}
        )
        resources = RayResources(num_cpus=1, runtime_env=base_env)

        # Merge new runtime env
        new_env = RuntimeEnv(
            env_vars={"NEW": "value"},
            conda={"dependencies": ["pandas"]}
        )

        result = resources.merge_runtime_env(new_env)

        # Should preserve existing and add new
        assert result.runtime_env.get("env_vars") == {
            "EXISTING": "value",
            "NEW": "value"
        }
        assert result.runtime_env.get("pip") == {"packages": ["numpy"], "pip_check": False}
        assert result.runtime_env.get("conda") == {"dependencies": ["pandas"]}

    def test_merge_runtime_env_dict(self):
        """Test merging with a dict instead of RuntimeEnv."""
        resources = RayResources(num_cpus=1)

        runtime_env_dict = {
            "env_vars": {"FOO": "bar"},
            "working_dir": "/tmp/test"
        }

        result = resources.merge_runtime_env(runtime_env_dict)

        assert result.runtime_env.get("env_vars") == {"FOO": "bar"}
        assert result.runtime_env.get("working_dir") == "/tmp/test"

    def test_merge_runtime_env_none(self):
        """Test that None runtime_env returns original unchanged."""
        resources = RayResources(num_cpus=1)
        result = resources.merge_runtime_env(None)

        assert result is resources  # should be the same instance

    def test_merge_runtime_env_empty(self):
        """Test that empty runtime_env returns original unchanged."""
        resources = RayResources(num_cpus=1)
        result = resources.merge_runtime_env({})

        assert result is resources  # should be the same instance

    def test_merge_env_vars_delegates_to_merge_runtime_env(self):
        """Test that merge_env_vars correctly delegates to merge_runtime_env."""
        resources = RayResources(num_cpus=1)

        # Test that both methods produce the same result
        env_vars = {"FOO": "bar", "BAZ": "qux"}

        result1 = resources.merge_env_vars(env_vars)
        result2 = resources.merge_runtime_env({"env_vars": env_vars})

        assert result1.runtime_env.get("env_vars") == result2.runtime_env.get("env_vars")

    def test_immutability(self):
        """Test that original RayResources instances are not modified."""
        original = RayResources(num_cpus=1, runtime_env=RuntimeEnv(env_vars={"ORIGINAL": "value"}))

        # Perform merges
        result1 = original.merge_env_vars({"NEW": "value"})
        result2 = original.merge_runtime_env(RuntimeEnv(env_vars={"ANOTHER": "value"}))

        # Original should be unchanged
        assert original.runtime_env.get("env_vars") == {"ORIGINAL": "value"}

        # Results should be different instances
        assert result1 is not original
        assert result2 is not original
        assert result1 is not result2

    def test_complex_merge_scenario(self):
        """Test a complex scenario with multiple merges."""
        # Start with base resources
        base_env = RuntimeEnv(
            env_vars={"BASE": "value"},
            pip={"packages": ["numpy"]}
        )
        resources = RayResources(
            num_cpus=4,
            num_gpus=2,
            runtime_env=base_env
        )

        # First merge: add remote function runtime env
        remote_fn_env = RuntimeEnv(
            env_vars={"REMOTE": "fn_value"},
            conda={"dependencies": ["pandas"]}
        )
        resources = resources.merge_runtime_env(remote_fn_env)

        # Second merge: add MXLA environment variables
        mxla_env = {"MEGASCALE_COORDINATOR_ADDRESS": "localhost:8081"}
        resources = resources.merge_env_vars(mxla_env)

        # Check final state
        expected_env_vars = {
            "BASE": "value",
            "REMOTE": "fn_value",
            "MEGASCALE_COORDINATOR_ADDRESS": "localhost:8081"
        }

        assert resources.num_cpus == 4
        assert resources.num_gpus == 2
        assert resources.runtime_env.get("env_vars") == expected_env_vars
        assert resources.runtime_env.get("pip") == {"packages": ["numpy"], "pip_check": False}
        assert resources.runtime_env.get("conda") == {"dependencies": ["pandas"]}
