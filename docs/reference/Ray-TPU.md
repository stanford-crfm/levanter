# Ray TPU Remote Functions

Levanter provides a convenience decorator [levanter.infra.ray_tpu.tpu_remote][] for running Python functions on Cloud TPU slices through a Ray cluster. The decorator acquires the required TPU resources, launches the function on each host of the slice, and retries automatically if the job is preempted or encounters transient failures.

## Basic Usage

You can apply `tpu_remote` as a decorator:

```python
from levanter.infra.ray_tpu import tpu_remote

@tpu_remote(tpu_type="v4-8")
def add_one(x):
    import jax.numpy as jnp
    return jnp.add(x, 1)

result = add_one(1)
```

`tpu_remote` returns a Ray remote function. Calling the decorated function submits a job to the Ray cluster and blocks until the TPU work is complete.

The decorator may also be used as a function:

```python
from levanter.infra.ray_tpu import tpu_remote

def add(x, y):
    import jax.numpy as jnp
    return jnp.add(x, y)

remote_add = tpu_remote(add, tpu_type="v4-8")
result = remote_add(1, 2)
```

## Multiple Slices

If `num_slices` is greater than 1, the function is launched on each slice. The return value is a flat list containing the results from every host of all slices:

```python
@tpu_remote(tpu_type="v4-8", num_slices=2)
def host_id():
    import jax
    return jax.process_index()

ids = host_id()
# ids == [0, 1, ..., 15]
```

## Retry Behaviour

By default, `tpu_remote` will retry preempted jobs and transient failures. This can be controlled via the `max_retries_preemption` and `max_retries_failure` arguments.

!!! note
    `run_on_pod_ray` is deprecated. Use [tpu_remote][levanter.infra.ray_tpu.tpu_remote] for new code.

## API Reference

::: levanter.infra.ray_tpu.tpu_remote
