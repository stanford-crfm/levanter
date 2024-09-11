import fsspec


def exists(url, **kwargs) -> bool:
    """Check if a file exists on a remote filesystem."""
    fs, path = fsspec.core.url_to_fs(url, **kwargs)
    return fs.exists(path)


def mkdirs(path):
    """Create a directory and any necessary parent directories."""
    fs, path = fsspec.core.url_to_fs(path)
    fs.makedirs(path, exist_ok=True)
