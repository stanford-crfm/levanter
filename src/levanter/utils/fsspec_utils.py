import fsspec


def exists(url, **kwargs) -> bool:
    """Check if a file exists on a remote filesystem."""
    fs, path = fsspec.core.url_to_fs(url, **kwargs)
    return fs.exists(path)
