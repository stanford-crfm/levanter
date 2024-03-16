# Introduction

This is the very beginnings of a performance guide for Levanter. It's currently mostly a collection of notes and ideas,
but it will eventually be a comprehensive guide to optimizing Levanter (and potentially other JAX programs).



See also the [JAX Profiling Guide](https://jax.readthedocs.io/en/latest/profiling.html)

# Profiling

## Enabling the Profiler

Levanter uses JAX's built-in profiler. You can enable it by adding the `--trainer.profiler true` flag
to the command line. This will generate a trace file in the `./logs` directory, under `./logs/<run_id>/profiler/plugins/profile/<datetime>`.
(Yeah, it's a mess, but it's what JAX wants to do.)
It will also upload the information to the relevant tracker (such as Weights & Biases or TensorBoard).

Here are the full list of profiling related options:

| Argument                           | Description | Default |
|------------------------------------|-------------|---------|
| `--trainer.profiler`               | Enable the profiler | `false` |
| `--trainer.profiler_start_step`    | The step to start profiling | `5`     |
| `--trainer.profiler_num_steps`     | The number of steps to profile | `100`   |
| `--trainer.profiler_perfetto_link` | Whether to generate a Perfetto URL | `false` |

As usual, these can be specified in the yaml configuration file as well.

In a multi-process setup, each node will save a profile, but only the first node will upload it to the tracker.
All of them will be available in the `./logs` directory (on each node).


## Examining a Profile

See the [JAX Profiling Guide](https://jax.readthedocs.io/en/latest/profiling.html) for more information on how to examine a profile.

But as a rough cut, you can use [Perfetto](https://ui.perfetto.dev/) in Chrome (Firefox doesn't work super well with it)
to examine the `perfetto_trace.json.gz` file.
