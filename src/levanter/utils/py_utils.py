from dataclasses import dataclass


def non_caching_cycle(iterable):
    """Like itertools.cycle, but doesn't cache the iterable."""
    while True:
        yield from iterable


# https://stackoverflow.com/a/58336722/1736826 CC-BY-SA 4.0
def dataclass_with_default_init(_cls=None, *args, **kwargs):
    def wrap(cls):
        # Save the current __init__ and remove it so dataclass will
        # create the default __init__.
        user_init = getattr(cls, "__init__")
        delattr(cls, "__init__")

        # let dataclass process our class.
        result = dataclass(cls, *args, **kwargs)

        # Restore the user's __init__ save the default init to __default_init__.
        setattr(result, "__default_init__", result.__init__)
        setattr(result, "__init__", user_init)

        # Just in case that dataclass will return a new instance,
        # (currently, does not happen), restore cls's __init__.
        if result is not cls:
            setattr(cls, "__init__", user_init)

        return result

    # Support both dataclass_with_default_init() and dataclass_with_default_init
    if _cls is None:
        return wrap
    else:
        return wrap(_cls)
