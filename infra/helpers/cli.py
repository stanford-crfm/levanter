import argparse
import concurrent.futures
import os
import subprocess
import typing

import yaml
from google.cloud import storage


def run_command(*args, **kwargs):
    print("Running:", " ".join(list(args)))
    return subprocess.check_call(args, **kwargs)


def add_ssh_key(ssh_key_filename):
    # format 3072 SHA256:... key-name (RSA)
    key_hash = subprocess.check_output(["ssh-keygen", "-lf", ssh_key_filename]).decode("utf-8").split()[1]
    existing_keys = subprocess.check_output(["ssh-add", "-l"]).decode("utf-8").split("\n")
    for key in existing_keys:
        if key_hash in key:
            return

    subprocess.check_call(["ssh-add", ssh_key_filename])


def tpu_ssh(tpu_name, zone, node_count, *args, ignore_failure=False):
    add_ssh_key(os.path.expanduser("~/.ssh/google_compute_engine"))
    try:
        if node_count > 1:
            return _tpu_ssh_multislice(tpu_name, zone, node_count, *args, ignore_failure=ignore_failure)

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


def _tpu_ssh_multislice(tpu_name, zone, node_count, *args, ignore_failure=False):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                run_command,
                "gcloud",
                "alpha",
                "compute",
                "tpus",
                "tpu-vm",
                "ssh",
                f"{tpu_name}-{i}",
                "--worker=all",
                f"--zone={zone}",
                "--command=%s" % " ".join(args),
            )
            for i in range(node_count)
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except subprocess.CalledProcessError as e:
                if ignore_failure:
                    print("Ignoring failure:", e)
                else:
                    raise


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


def get_default_zone() -> str:
    result = subprocess.run(["gcloud", "config", "get-value", "compute/zone"], stdout=subprocess.PIPE, text=True)
    return result.stdout.strip()


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


def get_git_commit():
    """Get the current git commit hash."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def make_docker_run_command(image_id, command, *, foreground, env):
    docker_command = [
        "docker",
        "run",
        "-t" if foreground else "-d",
        "--name=levanter",
        "--privileged",
        "--shm-size=32gb",
        "--net=host",
        "--init",
        "--mount",
        "type=volume,source=levanter,target=/home/levanter",
        "-v",
        "/tmp:/tmp",
    ]

    for k, v in env.items():
        docker_command.extend(["-e", k + f"='{str(v)}'"])

    docker_command.extend([image_id, " ".join(command)])
    return docker_command
