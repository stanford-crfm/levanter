#!/usr/bin/python

import argparse
import base64
import getpass
import os
import subprocess
import time

from infra import push_docker
from infra.helpers import cli


def setup_vm_docker(tpu_name, zone, docker_base_image):
    """Change docker permissions on `tpu_name`, remove any old runs, and setup the cache volume."""
    cli.tpu_ssh(
        tpu_name,
        zone,
        "sudo",
        "usermod",
        "-aG",
        "docker",
        getpass.getuser(),
        "&&",
        "sudo",
        "docker",
        "volume",
        "create",
        "--driver=local",
        "levanter",
        "&&",
        "sudo",
        "docker",
        "rm",
        "-f",
        "levanter",
    )


def list_tpus(zone):
    tpus = subprocess.check_output(
        [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "list",
            "--zone=" + zone,
        ]
    )
    rows = tpus.decode("utf-8").split("\n")
    header = rows[0].split()
    tpus = []
    for row in rows[1:]:
        if row:
            tpus.append(dict(zip(header, row.split())))
    return tpus


def start_tpu_vm(tpu_name, *, tpu_type, capacity_type, version, zone, autodelete):
    tpu_exists = any([tpu["NAME"] == tpu_name for tpu in list_tpus(zone)])
    if tpu_exists:
        if not autodelete:
            print("TPU already exists and autodelete is false, leaving it as is.")
            return

        print("TPU already exists, deleting...")
        cli.run_command(
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "delete",
            "--quiet",
            f"--zone={zone}",
            tpu_name,
        )

    print(f"Creating new TPU {tpu_name} in {zone} of type {tpu_type}...")
    command = [
        "gcloud",
        "alpha",
        "compute",
        "tpus",
        "tpu-vm",
        "create",
        tpu_name,
        f"--accelerator-type={tpu_type}",
        f"--version={version}",
        "--zone=" + zone,
        "--quiet",
    ]
    if capacity_type == "preemptible":
        command.append("--preemptible")
    elif capacity_type == "reserved":
        command.append("--reserved")
    elif capacity_type == "spot":
        command.append("--spot")
    elif capacity_type == "on-demand" or capacity_type is None:
        pass
    else:
        raise ValueError(f"Unknown capacity type: {capacity_type}")
    cli.run_command(*command)


def _default_run_id():
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = cli.load_config()

    cli.add_arg(
        parser, config, ["--autodelete"], default=False, action="store_true", help="Delete TPU if it already exists."
    )
    cli.add_arg(parser, config, ["--docker_base_image"], default="ghcr.io/rjpower/levanter:latest")
    cli.add_arg(parser, config, ["--docker_repository"], default="levanter")
    cli.add_arg(parser, config, ["--foreground"], default=False, action="store_true")
    cli.add_arg(parser, config, ["--image_name"], default=f"levanter-{getpass.getuser()}")
    cli.add_arg(
        parser, config, ["--capacity_type"], default=None, choices=["preemptible", "spot", "reserved", "on-demand"]
    )
    cli.add_arg(
        parser,
        config,
        ["--preemptible"],
        required=False,
        action="store_const",
        const="preemptible",
        dest="capacity_type",
    )
    cli.add_arg(parser, config, ["--spot"], required=False, action="store_const", const="spot", dest="capacity_type")
    cli.add_arg(
        parser, config, ["--reserved"], required=False, action="store_const", const="reserved", dest="capacity_type"
    )
    cli.add_arg(parser, config, ["--project"], default=cli.gcloud_config()["project"])
    cli.add_arg(parser, config, ["--tpu_name"], required=True)
    cli.add_arg(parser, config, ["--tpu_type"], required=True)
    cli.add_arg(parser, config, ["--version"], default="tpu-ubuntu2204-base")
    cli.add_arg(parser, config, ["--zone"], required=True)
    cli.add_arg(parser, config, ["--retries"], default=0, type=int)
    cli.add_arg(parser, config, ["--run_id"], default=_default_run_id(), type=str)
    cli.add_arg(parser, config, ["--docker_registry"], default="gcp", choices=["gcp", "ghcr"])
    cli.add_arg(parser, config, ["--github_user"], type=str)
    cli.add_arg(parser, config, ["--github_token"], type=str)

    parser.add_argument(
        "-e", "--env", 
        action="append", 
        nargs=2, 
        metavar=("KEY", "VALUE"), 
        default=list(config.get("env", {}).items())
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    autodelete = args.autodelete
    command = args.command
    docker_base_image = args.docker_base_image
    docker_repository = args.docker_repository
    foreground = args.foreground
    image_id = args.image_name
    capacity_type = args.capacity_type
    project = args.project
    if args.retries < 0:
        retries = 10000000
    else:
        retries = args.retries
    tpu_name = args.tpu_name
    tpu_type = args.tpu_type
    version = args.version
    zone = args.zone
    run_id = args.run_id
    registry = args.docker_registry
    github_user = args.github_user
    github_token = args.github_token

    region = "-".join(zone.split("-")[:-1])
    env = {k: v for k, v in args.env}

    if "WANDB_PROJECT" not in env:
        env["WANDB_PROJECT"] = "levanter"

    if command[0] == "--":
        command = command[1:]

    # make an image tag based on the unix timestamp to ensure we always pull the latest image
    tag = int(time.time())

    full_image_id = push_docker.push_to_gcp(
        project_id=project,
        region=region,
        repository=docker_repository,
        image_name=image_id,
        tag=tag,
        docker_file="docker/tpu/Dockerfile.incremental",
    )

    for i in range(retries + 1):
        try:
            start_tpu_vm(
                tpu_name=tpu_name,
                tpu_type=tpu_type,
                capacity_type=capacity_type,
                version=version,
                zone=zone,
                autodelete=autodelete,
            )

            # We don't technically need to setup on every run, but if we are working on a
            # stale VM or a VM from e.g. spin-up-vm.sh, this ensures things always work.
            setup_vm_docker(
                tpu_name=tpu_name,
                zone=zone,
                docker_base_image=docker_base_image,
            )

            # make an image tag based on the unix timestamp to ensure we always pull the latest image
            tag = int(time.time())

            if registry == "ghcr":
                full_image_id = push_docker.push_to_github(
                    local_image=image_id,
                    tag=tag,
                    github_user=github_user,
                    github_token=github_token,
                    docker_file="docker/tpu/Dockerfile.incremental",
                )
            elif registry == "gcp":
                full_image_id = push_docker.push_to_gcp(
                    project_id=project,
                    region=region,
                    repository=docker_repository,
                    image_name=image_id,
                    tag=tag,
                    docker_file="docker/tpu/Dockerfile.incremental",
                )
            else:
                raise ValueError(f"Unknown docker registry: {args.docker_registry}")

            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

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
                "-e",
                f"WANDB_DOCKER={image_id}",
                "-e",
                f"GIT_COMMIT={git_commit}",
                "-e",
                f"RUN_ID={run_id}",
            ]

            for k, v in env.items():
                docker_command.extend(["-e", k + f"='{str(v)}'"])

            docker_command.extend([full_image_id, " ".join(command)])

            print(f"Running on tpu_name... {tpu_name}")
            cli.tpu_ssh(tpu_name, zone, *docker_command)
        except subprocess.CalledProcessError as e:  # noqa: F841
            print("Error running command.")
            if i < retries - 1:
                print("Retrying... %d/%d" % (i + 1, retries))
