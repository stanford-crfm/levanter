#!/usr/bin/python

import argparse
import getpass
import subprocess
import time

from infra import push_docker

from infra.helpers import cli


def setup_vm_docker(tpu_name, zone, docker_base_image):
    cli.tpu_ssh(
        tpu_name,
        zone,
        "sudo",
        "usermod",
        "-aG",
        "docker",
        getpass.getuser(),
    )

    cli.tpu_ssh(
        tpu_name,
        zone,
        "docker",
        "pull",
        docker_base_image,
    )

    cli.tpu_ssh(tpu_name, zone, "docker", "volume", "create", "--driver=local", "levanter")


def list_tpus(zone):
    tpus = subprocess.check_output(
        [
            "gcloud",
            "alpha",
            "compute",
            "tpus",
            "tpu-vm",
            "list",
        ]
    )
    rows = tpus.decode("utf-8").split("\n")
    header = rows[0].split()
    tpus = []
    for row in rows[1:]:
        if row:
            tpus.append(dict(zip(header, row.split())))
    return tpus


def start_tpu_vm(
    tpu_name, *, tpu_type, preemptible, version, zone, autodelete, project, docker_repository, docker_base_image
):
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
            "delete",
            "--quiet",
            f"--zone={zone}",
            tpu_name,
        )

    print(f"Creating new TPU {tpu_name} in {zone} of type {tpu_type}...")
    cli.run_command(
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
        "--preemptible" if preemptible else "",
        "--quiet",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = cli.load_config()

    cli.add_arg(parser, config, ["--autodelete"], default=False, action="store_true")
    cli.add_arg(parser, config, ["--docker_base_image"], default="ghcr.io/rjpower/levanter:latest")
    cli.add_arg(parser, config, ["--docker_repository"], default="levanter")
    cli.add_arg(parser, config, ["--foreground"], default=False, action="store_true")
    cli.add_arg(parser, config, ["--image_name"], default=f"levanter-{getpass.getuser()}")
    cli.add_arg(parser, config, ["--preemptible"], default=False, action="store_true")
    cli.add_arg(parser, config, ["--project"], default=cli.gcloud_config()["project"])
    cli.add_arg(parser, config, ["--tpu"], required=True)
    cli.add_arg(parser, config, ["--tpu_type"])
    cli.add_arg(parser, config, ["--version"], default="tpu-ubuntu2204-base")
    cli.add_arg(parser, config, ["--zone"], required=True)
    cli.add_arg(parser, config, ["--retries"], default=0, type=int)

    parser.add_argument(
        "-e", "--env", action="append", nargs=2, metavar=("KEY", "VALUE"), default=config.get("env", {}).items()
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    autodelete = args.autodelete
    command = args.command
    docker_base_image = args.docker_base_image
    docker_repository = args.docker_repository
    foreground = args.foreground
    image_id = args.image_name
    preemptible = args.preemptible
    project = args.project
    if args.retries < 0:
        retries = 10000000
    else:
        retries = args.retries
    tpu_name = args.tpu
    tpu_type = args.tpu_type
    version = args.version
    zone = args.zone

    region = "-".join(zone.split("-")[:-1])
    env = {k: v for k, v in args.env}

    if "WANDB_PROJECT" not in env:
        env["WANDB_PROJECT"] = "levanter"

    if command[0] == "--":
        command = command[1:]

    for i in range(retries + 1):
        try:
            start_tpu_vm(
                tpu_name=tpu_name,
                tpu_type=tpu_type,
                preemptible=preemptible,
                version=version,
                zone=zone,
                autodelete=autodelete,
                project=project,
                docker_repository=docker_repository,
                docker_base_image=docker_base_image,
            )

            setup_vm_docker(
                tpu_name=tpu_name,
                zone=zone,
                docker_base_image=docker_base_image,
            )

            # make an image tag based on the unix timestamp to ensure we always pull the latest image
            tag = run_id = int(time.time())

            full_image_id = push_docker.push_to_gcp(
                project_id=project,
                region=region,
                repository=docker_repository,
                image_name=image_id,
                tag=tag,
                docker_file="docker/tpu/Dockerfile.incremental",
            )

            git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()

            cli.tpu_ssh(tpu_name, zone, "docker", "stop", "levanter", "-t", "1", ignore_failure=True)
            cli.tpu_ssh(tpu_name, zone, "docker", "rm", "-f", "levanter", ignore_failure=True)

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
        except subprocess.CalledProcessError as e:
            print("Error running command.")
            if i < retries - 1:
                print("Retrying... %d/%d" % (i + 1, retries))
