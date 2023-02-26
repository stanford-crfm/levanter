import tempfile
from urllib.parse import urlparse

import fsspec
from transformers import AutoTokenizer


def load_tokenizer(model_name_or_path, local_cache_dir=None):
    """Like AutoTokenizer.from_pretrained, but works with gs:// paths or anything on fsspec"""
    is_url_like = urlparse(model_name_or_path).scheme != ""
    if is_url_like:
        # tokenizers are directories, so we have to copy them locally
        if local_cache_dir is None:
            local_cache_dir = tempfile.mkdtemp()

        fs, path = fsspec.core.url_to_fs(model_name_or_path)
        fs.get(path, local_cache_dir, recursive=True)
        return AutoTokenizer.from_pretrained(local_cache_dir)
    else:
        return AutoTokenizer.from_pretrained(model_name_or_path)
