#!/usr/bin/python

"""
Build and deploy the Levanter base image to Artifact Registry or Docker Hub.

It is not necessary to run this yourself unless you are deploying a new base image: the launch
script will automatically build and deploy an image based on your current code.
"""

import argparse
import json
import os
import pty
import sys
import subprocess

from infra.helpers import cli


GCP_CLEANUP_POLICY = [
    {
        "name": "delete-stale",
        "action": {"type": "Delete"},
        "condition": {
            "olderThan": "86400s",
            "tagState": "ANY",
        },
    },
    {
        "name": "keep-latest",
        "action": {"type": "Keep"},
        "mostRecentVersions": {
            "keepCount": 5,
        },
    },
]


def _run(argv, **kw):
    buffer = []

    def _read(fd):
        data = os.read(fd, 1024)
        os.write(sys.stdout.fileno(), data)
        buffer.append(data)

    exit_code = pty.spawn(argv)
    if exit_code != 0:
        raise subprocess.CalledProcessError(exit_code, argv, output=b"".join(buffer))


def configure_gcp_docker(project_id, region, repository):
    """Setup Artifact registry repository and configure permissions to enable TPU access."""
    # check if the repository already exists
    try:
        _run(
            ["gcloud", "artifacts", "repositories", "describe", f"--location={region}", repository],
            stderr=subprocess.STDOUT,
        )
        print(f"Found existing artifact registry repository `{repository}`, skipping setup.")
        return
    except subprocess.CalledProcessError as e:
        if b"NOT_FOUND" not in e.output:
            raise

    # Activate artifact registry and setup the repository.
    _run(["gcloud", "services", "enable", "artifactregistry.googleapis.com"])

    try:
        _run(
            [
                "gcloud",
                "artifacts",
                "repositories",
                "create",
                repository,
                f"--location={region}",
                "--repository-format=docker",
            ],
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as e:
        # Ignore error if repository already exists.
        if b"ALREADY_EXISTS" not in e.output:
            print("Error creating repository: ", e.output)
            raise

    with open("/tmp/cleanup-policy.json", "w") as f:
        json.dump(GCP_CLEANUP_POLICY, f, indent=2)

    _run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "set-cleanup-policies",
            f"--location={region}",
            "--policy=/tmp/cleanup-policy.json",
            repository,
        ]
    )

    # Grant public read access ('allUsers') for TPU VMs
    _run(
        [
            "gcloud",
            "artifacts",
            "repositories",
            "add-iam-policy-binding",
            "--member=allUsers",
            "--role=roles/artifactregistry.reader",
            f"--location={region}",
            repository,
        ]
    )

    _run(
        [
            "gcloud",
            "--project",
            project_id,
            "artifacts",
            "repositories",
            "add-iam-policy-binding",
            repository,
            "--location",
            region,
            "--member",
            "allUsers",
            "--role",
            "roles/artifactregistry.reader",
        ]
    )

    _run(["gcloud", "auth", "configure-docker", "--quiet", f"{region}-docker.pkg.dev"])


def build_docker(docker_file, image_name, tag) -> str:
    """Builds a Docker image, enables artifact access, and pushes to Artifact Registry."""

    _run(
        [
            "docker",
            "buildx",
            "build",
            "--platform=linux/amd64",
            "-t",
            f"{image_name}:{tag}",
            "-f",
            docker_file,
            ".",
        ]
    )

    return f"{image_name}:{tag}"


# Disabled until we can figure out how Docker hub organizations work
def push_to_github(local_image, tag, github_user=None, github_token=None, docker_file=None):
    """Pushes a local Docker image to Docker Hub."""

    # Authenticate the docker service with Github if a token exists
    if github_token:
        login_process = subprocess.Popen(
            ["docker", "login", "ghcr.io", "-u", github_user, "--password-stdin"], stdin=subprocess.PIPE
        )
        print(login_process.communicate(input=github_token.encode(), timeout=10))

    remote_name = f"ghcr.io/{github_user}/{local_image}:{tag}"
    local_name = build_docker(docker_file=docker_file, image_name=local_image, tag=tag)

    _run(["docker", "tag", local_name, remote_name])
    _run(["docker", "push", remote_name])
    return remote_name


def push_to_gcp(project_id, region, repository, image_name, tag, docker_file) -> str:
    """Pushes a local Docker image to Artifact Registry."""
    configure_gcp_docker(project_id, region, repository)
    local_image = build_docker(docker_file=docker_file, image_name=image_name, tag=tag)

    artifact_repo = f"{region}-docker.pkg.dev/{project_id}/{repository}"

    full_image_name = f"{artifact_repo}/{image_name}:{tag}"
    _run(["docker", "tag", local_image, full_image_name])
    _run(["docker", "push", full_image_name])

    return f"{region}-docker.pkg.dev/{project_id}/{repository}/{image_name}:{tag}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and push Docker image to Artifact Registry.")
    config = cli.load_config()
    cli.add_arg(parser, config, ["--project"], help="GCP project ID")
    cli.add_arg(parser, config, ["--region"], help="Artifact Registry region (e.g., us-west4)")
    cli.add_arg(parser, config, ["--repository"], default="levanter", help="Artifact Registry repository name")
    cli.add_arg(parser, config, ["--image"], default="levanter", help="Docker image name.")
    cli.add_arg(parser, config, ["--tag"], default="latest", help="Docker image tag.")
    cli.add_arg(parser, config, ["--github_user"], default=None, help="Github user name.")
    cli.add_arg(parser, config, ["--github_token"], default=None, help="Github token.")
    cli.add_arg(parser, config, ["--docker_file"], default="docker/tpu/Dockerfile.base", help="Dockerfile to use.")

    # push to either github or GCP
    cli.add_arg(parser, config, ["--docker_target"], choices=["github", "gcp"], required=True)

    args = parser.parse_args()

    if args.docker_target == "github":
        assert args.github_user, "Must specify --github_user when pushing to Github"
        assert args.github_token, "Must specify --github_token when pushing to Github"
        push_to_github(args.image, args.tag, args.github_user, args.github_token, docker_file=args.docker_file)
    else:
        assert args.region, "Must specify --region when pushing to GCP"
        assert args.project, "Must specify --project when pushing to GCP"
        assert args.repository, "Must specify --repository when pushing to GCP"

        push_to_gcp(
            args.project,
            args.region,
            args.repository,
            args.image,
            args.tag,
            docker_file=args.docker_file,
        )
