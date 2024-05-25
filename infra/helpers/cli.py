import argparse
import os
import subprocess
import typing

from google.cloud import storage
import yaml


def run_command(*args, **kwargs):
    print("Running:", " ".join(list(args)))
    return subprocess.check_call(args, **kwargs)


def add_ssh_key(ssh_key_filename):
    # format 3072 SHA256:... key-name (RSA)
    key_hash = subprocess.check_output(["ssh-keygen", "-lf", ssh_key_filename]).decode("utf-8").split()[1]
    existing_keys = subprocess.check_output(["ssh-add", "-l"]).decode("utf-8").split("\n")
    for key in existing_keys:
        if key_hash in key:
            print('Found existing key in ssh-agent, skipping "ssh-add"')
            return

    subprocess.check_call(["ssh-add", ssh_key_filename])


def tpu_ssh(tpu_name, zone, *args, ignore_failure=False):
    add_ssh_key(os.path.expanduser("~/.ssh/google_compute_engine"))
    try:
        return run_command(
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "ssh",
            tpu_name,
            "--worker=all",
            f"--zone={zone}",
            "--command=%s" % " ".join(args),
        )
    except subprocess.CalledProcessError as e:
        if ignore_failure:
            print("Ignoring failure:", e)
        else:
            raise


# Oddly enough, there's no API to simply fetch the current gcloud configuration...
def gcloud_config():
    client = storage.Client()
    return {
        "project": client.project,
    }


def add_arg(
    parser: argparse.ArgumentParser, config: typing.Dict, flags: typing.List[str], required=False, default=None, **kw
):
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
