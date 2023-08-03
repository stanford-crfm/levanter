import contextlib
import logging
import os
import shutil
import tempfile
import time
import urllib.parse
from typing import Optional

import fsspec
import jax
import requests  # type: ignore
from fsspec import AbstractFileSystem
from jax.experimental.multihost_utils import sync_global_devices


logger = logging.getLogger(__name__)

SENTINEL_FILE = "/tmp/tpu_shutdown_sentinel"


def _checked_request(url):
    try:
        response = requests.get(url, headers={"Metadata-Flavor": "Google"})
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException:
        logger.exception(f"Could not get {url} from metadata server. Is this a TPU VM?", exc_info=True)
        raise


def shutdown_tpu_vm(sleep_seconds=60 * 5):
    """You should probably call this from atexit or something like that."""
    try:
        zone = _checked_request("http://metadata.google.internal/computeMetadata/v1/instance/zone")
        zone = zone.split("/")[-1]
        name = _checked_request("http://metadata.google.internal/computeMetadata/v1/attributes/instance-id")
    except requests.exceptions.RequestException:
        logger.warning("Could not get zone or instance-id from metadata server. Is this a TPU VM? Not shutting down.")
        return

    # the gcloud command we would run is something like:
    # gcloud compute tpus tpu-vm delete tpu-vm-1 --zone us-central1-a --quiet
    logger.critical(f"Shutting down TPU VM {name} in zone {zone} in {sleep_seconds} seconds")
    logger.critical(f"Create a file {SENTINEL_FILE} to cancel the shutdown")
    logger.critical(f"$ touch {SENTINEL_FILE}")

    time.sleep(sleep_seconds)
    if os.path.exists(SENTINEL_FILE):
        logger.critical(f"Found sentinel file {SENTINEL_FILE}, not shutting down TPU VM")
        return

    logger.critical(f"Shutting down TPU VM {name} in zone {zone}")
    if jax.process_index() != 0:
        logger.info(f"Letting process 0 handle the shutdown. We are process {jax.process_index()}")
        return

    os.system(f"gcloud compute tpus tpu-vm delete {name} --zone {zone} --quiet")


_sync_count = 0


@contextlib.contextmanager
def temp_dir_before_upload(path, process_should_upload: Optional[bool] = None):
    """
    Creates a temp_dir, yields it, then uploads it to the given path or url on exit using fsspec.
    If it's a local path instead of a url, it just yields the path.

    :param process_should_upload: If None, only upload if jax.process_index() == 0. Otherwise, upload if this is True.
    """

    if process_should_upload is None:
        process_should_upload = jax.process_index() == 0

    def _is_url_like(path):
        return urllib.parse.urlparse(path).scheme != ""

    try:
        tmpdir: Optional[str] = None
        if _is_url_like(path):
            tmpdir = tempfile.mkdtemp()
            local_path = tmpdir
        else:
            local_path = path

        yield local_path

        if tmpdir is not None:  # we're uploading to GCS or similar
            if process_should_upload:
                logger.info(f"Copying from temp dir to {path}")
                fs: AbstractFileSystem = fsspec.core.get_fs_token_paths(path, mode="wb")[0]
                fs.put(os.path.join(local_path, "*"), path, recursive=True)
                logger.info(f"Finished copying to {path}")
            else:
                logger.info(f"Waiting for process 0 to finish saving checkpoint to {path}")

            global _sync_count
            sync_global_devices(f"upload? {path}{_sync_count}")
            _sync_count += 1
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)
