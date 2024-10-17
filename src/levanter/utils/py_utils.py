import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, TypeVar


def logical_cpu_core_count():
    """Returns the number of logical CPU cores available to the process."""
    num_cpus = os.getenv("SLURM_CPUS_ON_NODE", None)
    if num_cpus is not None:
        return int(num_cpus)

    try:
        return os.cpu_count()
    except NotImplementedError:
        return 1


def logical_cpu_memory_size():
    """Returns the total amount of memory in GB available to the process or logical memory for SLURM."""
    mem = os.getenv("SLURM_MEM_PER_NODE", None)
    tasks = os.getenv("SLURM_NTASKS_PER_NODE", None)
    if mem is not None and tasks is not None:
        return float(mem) / int(tasks) / 1024.0  # MEM_PER_NODE is in MB

    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return total / (1024.0**3)
    except ValueError:
        import psutil

        return psutil.virtual_memory().total / (1024.0**3)


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


# slightly modified from https://github.com/tensorflow/tensorflow/blob/14ea9d18c36946b09a1b0f4c0eb689f70b65512c/tensorflow/python/util/decorator_utils.py
# to make TF happy
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


class classproperty(object):  # pylint: disable=invalid-name
    """Class property decorator.

    Example usage:

    class MyClass(object):

      @classproperty
      def value(cls):
        return '123'

    > print MyClass.value
    123
    """

    def __init__(self, func):
        self._func = func

    def __get__(self, owner_self, owner_cls):
        return self._func(owner_cls)


class _CachedClassProperty(object):
    """Cached class property decorator.

    Transforms a class method into a property whose value is computed once
    and then cached as a normal attribute for the life of the class.  Example
    usage:

    >>> class MyClass(object):
    ...   @cached_classproperty
    ...   def value(cls):
    ...     print("Computing value")
    ...     return '<property of %s>' % cls.__name__
    >>> class MySubclass(MyClass):
    ...   pass
    >>> MyClass.value
    Computing value
    '<property of MyClass>'
    >>> MyClass.value  # uses cached value
    '<property of MyClass>'
    >>> MySubclass.value
    Computing value
    '<property of MySubclass>'

    This decorator is similar to `functools.cached_property`, but it adds a
    property to the class, not to individual instances.
    """

    def __init__(self, func):
        self._func = func
        self._cache = {}

    def __get__(self, obj, objtype):
        if objtype not in self._cache:
            self._cache[objtype] = self._func(objtype)
        return self._cache[objtype]

    def __set__(self, obj, value):
        raise AttributeError("property %s is read-only" % self._func.__name__)

    def __delete__(self, obj):
        raise AttributeError("property %s is read-only" % self._func.__name__)


# modification based on https://github.com/python/mypy/issues/2563
PropReturn = TypeVar("PropReturn")


def cached_classproperty(func: Callable[..., PropReturn]) -> PropReturn:
    return _CachedClassProperty(func)  # type: ignore


cached_classproperty.__doc__ = _CachedClassProperty.__doc__


def actual_sizeof(obj):
    """similar to sys.getsizeof, but recurses into dicts and lists and other objects"""
    seen = set()
    size = 0
    objects = [obj]
    while objects:
        need_to_see = []
        for obj in objects:
            if id(obj) in seen:
                continue
            seen.add(id(obj))
            size += sys.getsizeof(obj)
            if isinstance(obj, dict):
                need_to_see.extend(obj.values())
            elif hasattr(obj, "__dict__"):
                need_to_see.extend(obj.__dict__.values())
            elif isinstance(obj, (list, tuple, set, frozenset)):
                need_to_see.extend(obj)
        objects = need_to_see
    return size


class Stopwatch:
    """Resumable stop watch for tracking time per call"""

    def __init__(self):
        self._start_time = time.time()
        self._elapsed = 0.0
        self._n = 0

    def start(self):
        self._start_time = time.time()
        self._n += 1

    def stop(self):
        self._elapsed += time.time() - self._start_time

    def reset(self):
        self._elapsed = 0.0

    def elapsed(self):
        return self._elapsed

    def average(self):
        if self._n == 0:
            return 0.0
        return self._elapsed / self._n

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
