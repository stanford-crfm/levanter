import jax


def log(metrics, *, step):
    """
    Log metrics to the global tracker.

    Args:
        metrics: Metrics to log. We use LoggableValues just to give you a sense of what you can log. Backends may
            support additional types.
        step: Step to log at
        commit: Whether to commit the metrics. If None, uses the default for the tracker.
    """
    print(metrics, step)


def _do_jit_log(metrics, *, step=None):
    try:
        log(metrics, step=step)
    except Exception as e:
        raise e


def jit_log(metrics, *, step=None):
    """uses jax effect callback to log to wandb from the host"""
    jax.debug.callback(_do_jit_log, metrics, step=step)
