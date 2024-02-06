# Trainer Abstraction Cleanup

## Current Status (2024-01-23)

### Trainer's current jobs

Trainer currently has these jobs:

* Handle registering and running callbacks
* Handle checkpointing (delegated to `Checkpointer`)
* Handling initialization, including loading from a checkpoint (partially delegated to `Checkpointer`)
* train_step/train_steps/training loop
* holding execution environment details (e.g. mixed precision policy, device mesh, etc)
* handles making data loaders (with the right sharding etc)
* sets up microbatching/grad accum (mostly factored out into nice pieces
* actually taking the step

It would be nice if this were orthogonalized as much as possible.

Hooks are already mostly delegated out to TrainerHooks so that's not too bad, and checkpoints are well encapsulated in the Checkpointer class,
other than the initialization/resume logic.

Execution Environment is just work to abstract, and dovetails well with other work (i.e. just-in-time mixed precision).

A lot of changes live in the doremi branch, because it needs an augmented trainer state to do its work



### Other things that bother me

* the cached_property loss_fn is smelly and actually behaves badly because jit(grad(jit(f))) doesn't work well
* I don't love the story for extending TrainerState

### TrainerState extension

We want TrainerState to be extensible, which means that:

* it needs to inheritable
* methods like train_step need to be able to be overridden
* core "train_step" logic needs to be reusable (e.g. the logic for accumulating gradients, etc) in a way that
  returns the right type (e.g. TrainerState or subclass)

In Haliax we do initialization with a static/classmethod on modules, rather than the ctor. It's useful to have
a "plain old constructor" for various modules

## Initialization/Resume


### Requirements

There are 3 core cases to consider:

1. No checkpoint, initialize from scratch
2. Checkpoint exists, load checkpoint and initialize "unserialized"/missing state
3. Partial checkpoint exists (e.g. only model), load checkpoint and initialize "unserialized"/missing state

Typically, (3) is a full checkpoint, but we only want to load the model. This is useful for things like
fine-tuning a model, where we want to load the model but not the optimizer state.

On top of that, we currently differentiate between passing in a model_init function and a model. This
complicates things a bit, but model_init is preferred because:

1. it's more memory/time efficient when initializing from checkpoint
2. it makes it easier to get sharding and mixed precision right immediately.

For (1), I think the time isn't a big deal, but we need a way of dealing
with the memory. One could maybe delete the passed in model (preserving only the shape)
once we determine the checkpoint exists?

For (2), we also want to get the mixed precision and sharding set up correctly immediately. Passing in a model_init
allows us to wrap it in the right jit and partition magic to get that right.
We can and should expose (2) as a function...


Another complexity is `is_trainable`, which is a FilterSpec that allows you to specify which parts of the model
are trainable. This is useful for things like fine-tuning, where you want to freeze some layers. We use is_trainable in
4 ways:

* only the is_trainable parts of a model get an optimizer_state associated with them
* we only serialize/deserialize the is_trainable parts of a model
* we only compute gradients for the is_trainable parts of a model
* We store the non-trainable parts of the model in compute precision, and the trainable parts in the param precision

### Current Initialization w/o checkpoint

This is conceptually what happens when there is no checkpointing:

```python
@hax.named_jit(out_axis_resources=parameter_mapping)
def initialize(optimizer, model_init, is_trainable, mp):
    model = model_init()
    trainable_model = eqx.filter(model, is_trainable)
    optimizer_state = optimizer.init(trainable_model)

    model = _cast_model_by_trainability(model, is_trainable, mp)

    state = TrainerState(
        _step=0,
        model=model,
        optimizer_state=optimizer_state,
        is_trainable=is_trainable,
    )

    state = hax.shard(state, parameter_mapping)

    return state


def _cast_model_by_trainability(model, is_trainable, mp):
    trainable_model, non_trainable_model = eqx.partition(model, is_trainable)
    non_trainable_model = mp.cast_to_compute(non_trainable_model)
    trainable_model = mp.cast_to_param(trainable_model)
    model = eqx.combine(trainable_model, non_trainable_model)
    return model
```



### Current logic for initialization w/ checkpoint

The logic for initial_state is pretty complex. There are 3 cases to consider:

1. No checkpoint, initialize from scratch
2. Checkpoint exists, load checkpoint and initialize "unserialized"/missing state
3. Partial checkpoint exists (e.g. only model), load checkpoint and initialize "unserialized"/missing state

At the moment the flow is:

```python

state_shape = eval_shape(_initialize_from_scratch(model_init_fn))
if checkpoint_exists:
    partial_state = load_checkpoint(state_shape, path)
elif partial_checkpoint_exists:
    partial_checkpoint = load_checkpoint(state_shape.model, path, subpath="model")
    partial_state = dataclasses.replace(partial_state, model=partial_checkpoint)

state = jit(lambda s: combine(s, _initialize_from_scratch(model_init_fn)), partial_state)
```

I'd like to hoist this out so it's not dependent on the Trainer class, and so that it's easier to test.

One of the things I was trying to accomplish was to define a checkpointed_or_initialize function that was just

```python
state_shape = eval_shape(f)
if checkpoint_exists:
    partial_state = load_checkpoint(state_shape, path)
else:
    partial_state = eqx.filter(state_shape, lamba v: False)

state = jit(lambda s: combine(s, f()), partial_state)

```

But this doesn't actually compose well: you can't really do IO inside of eval_shape, so you can't really combine two
of those... or can you
