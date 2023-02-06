def non_caching_cycle(iterable):
    """Like itertools.cycle, but doesn't cache the iterable."""
    while True:
        yield from iterable
