from typing import Optional

from optax import MultiStepsState

from psithuros.jax_utils import jnp_to_python


def log_optimizer_hyperparams(opt_state, prefix: Optional[str] = None, *, step=None):
    import wandb
    if isinstance(opt_state, MultiStepsState):
        opt_state = opt_state.inner_opt_state

    def wrap_key(key):
        if prefix:
            return f"{prefix}/{key}"
        return key

    if hasattr(opt_state, "hyperparams"):
        params = {wrap_key(k): jnp_to_python(v) for k, v in opt_state.hyperparams.items()}
        print(params)
        wandb.log(params, step=step)
