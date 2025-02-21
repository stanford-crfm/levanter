import os

import braceexpand
import fsspec
from fsspec.asyn import AsyncFileSystem


def exists(url, **kwargs) -> bool:
    """Check if a file exists on a remote filesystem."""
    fs, path = fsspec.core.url_to_fs(url, **kwargs)
    return fs.exists(path)


def mkdirs(path):
    """Create a directory and any necessary parent directories."""
    fs, path = fsspec.core.url_to_fs(path)
    fs.makedirs(path, exist_ok=True)


def expand_glob(url):
    expanded_urls = braceexpand.braceexpand(url)
    for expanded_url in expanded_urls:
        if "*" in expanded_url:
            fs = fsspec.core.url_to_fs(expanded_url)[0]
            globbed = fs.glob(expanded_url)
            # have to append the fs prefix back on
            protocol, _ = fsspec.core.split_protocol(expanded_url)
            if protocol is None:
                yield from globbed
            else:
                yield from [f"{protocol}://{path}" for path in globbed]
        else:
            yield expanded_url


def remove(url, *, recursive=False, **kwargs):
    """Remove a file from a remote filesystem."""
    # TODO: better to use a STS deletion policy or job for this one.
    fs, path = fsspec.core.url_to_fs(url, **kwargs)

    fs.rm(path, recursive=recursive)


async def async_remove(url, *, recursive=False, **kwargs):
    """Remove a file from a remote filesystem."""
    fs, path = fsspec.core.url_to_fs(url, **kwargs)

    if isinstance(fs, AsyncFileSystem):
        return await fs._rm(path, recursive=recursive)
    else:
        fs.rm(path, recursive=recursive)


def join_path(lhs, rhs):
    """
    Join parts of a path together. Similar to plain old os.path.join except when there is a protocol in the rhs, it
    is treated as an absolute path. However, the lhs protocol and rhs protocol must match if the rhs has one.

    """

    lhs_protocol, lhs_rest = fsspec.core.split_protocol(lhs)
    rhs_protocol, rhs_rest = fsspec.core.split_protocol(rhs)

    if rhs_protocol is not None and lhs_protocol is not None and lhs_protocol != rhs_protocol:
        raise ValueError(f"Cannot join paths with different protocols: {lhs} and {rhs}")

    if rhs_protocol is not None:
        return rhs
    else:
        return os.path.join(lhs, rhs)
