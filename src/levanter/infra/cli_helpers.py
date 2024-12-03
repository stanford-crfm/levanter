import argparse
import base64
import os
import shlex
import subprocess
from typing import Optional

import yaml
from google.cloud import storage


# Oddly enough, there's no API to simply fetch the current gcloud configuration...
def gcloud_config():
    client = storage.Client()
    out: dict[str, str | None] = {
        "project": client.project,
    }
    try:
        out["zone"] = get_default_zone()
    except subprocess.CalledProcessError:
        out["zone"] = None

    return out


def get_default_zone() -> Optional[str]:
    try:
        result = subprocess.run(["gcloud", "config", "get-value", "compute/zone"], stdout=subprocess.PIPE, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def add_arg(parser: argparse.ArgumentParser, config: dict, flags: list[str], required=False, default=None, **kw):
    """Add an argument to the parser, using `config` or the environment to resolve default values."""
    key = flags[0].lstrip("-").replace("-", "_")
    if key in config:
        default = config[key]

    if key.upper() in os.environ:
        default = os.environ[key.upper()]

    if default is not None:
        kw["default"] = default
    elif required:
        kw["required"] = True

    parser.add_argument(*flags, **kw)


def load_config():
    if os.path.exists(".config"):
        return yaml.load(open(".config", "r"), Loader=yaml.SafeLoader)
    else:
        return {}


def get_git_commit():
    """Get the current git commit hash."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def make_docker_run_command(image_id, command, *, foreground, env, name="levanter"):
    docker_command = [
        "docker",
        "run",
        "-t" if foreground else "-d",
        f"--name={shlex.quote(name)}",
        "--privileged",
        "--shm-size=32gb",
        "--net=host",
        "--init",
        "--mount",
        "type=volume,source=levanter,target=/home/levanter",
        "-v",
        "/tmp:/tmp",
    ]

    # optionally add multislice env vars (if set by ray runtime env vars)
    for v in ["MEGASCALE_COORDINATOR_ADDRESS", "MEGASCALE_NUM_SLICES", "MEGASCALE_PORT", "MEGASCALE_SLICE_ID"]:
        v = shlex.quote(str(v))
        docker_command.extend(["-e", v])

    for k, v in env.items():
        v = shlex.quote(str(v))
        k = shlex.quote(str(k))
        docker_command.extend(["-e", f"{k}={v}"])

    docker_command.extend([image_id, *command])
    return docker_command


def default_run_id():
    """Generate a run ID for wandb and continuation.

    Wandb expects a base36 encoded ID of exactly 8 lowercase characters
    or it won't generate a display name."""
    rng_bytes = os.urandom(16)
    run_id = base64.b32encode(rng_bytes)[:8].lower()
    run_id = run_id.decode("utf-8")
    assert len(run_id) == 8
    for char in run_id:
        assert char in "abcdefghijklmnopqrstuvwxyz0123456789"
    return run_id


def add_capacity_type_args(parser, config):
    """
    Add capacity type arguments to the parser. This emulates the behavior of Google's `gcloud` CLI.
    The capacity type will be stored in the `capacity_type` attribute of the parsed arguments.

    Args:
        parser: The argparse parser to add arguments to.
        config: The configuration dictionary to use for defaults.


    """
    add_arg(
        parser,
        config,
        ["--capacity_type"],
        default=None,
        choices=["preemptible", "spot", "reserved", "on-demand", "best-effort"],
    )
    add_arg(
        parser,
        config,
        ["--preemptible"],
        required=False,
        action="store_const",
        const="preemptible",
        dest="capacity_type",
    )
    add_arg(parser, config, ["--spot"], required=False, action="store_const", const="spot", dest="capacity_type")
    add_arg(
        parser, config, ["--reserved"], required=False, action="store_const", const="reserved", dest="capacity_type"
    )
    add_arg(
        parser, config, ["--on-demand"], required=False, action="store_const", const="on-demand", dest="capacity_type"
    )
