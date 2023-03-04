import logging
import os
import time

import jax
import requests  # type: ignore


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
