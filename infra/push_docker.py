#!/usr/bin/python

"""
Build and deploy the Levanter base image to Artifact Registry or Docker Hub.

It is not necessary to run this yourself unless you are deploying a new base image: the launch
script will automatically build and deploy an image based on your current code.
"""

import argparse
import json
import pty
import subprocess
import sys

from .helpers import cli


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


def _run(argv):
    if sys.stdout.isatty():
        exit_code = pty.spawn(argv)
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, argv)
    else:
        subprocess.check_output(argv, stderr=subprocess.STDOUT)


def configure_gcp_docker(project_id, region, repository):
    """Setup Artifact registry repository and configure permissions to enable TPU access."""
    # check if the repository already exists
    try:
        _run(
            ["gcloud", "artifacts", "repositories", "describe", f"--location={region}", repository],
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


def push_to_github(local_id, github_user, github_token=None):
    """Pushes a local Docker image to Docker Hub."""

    # Authenticate the docker service with Github if a token exists
    if github_token:
        login_process = subprocess.Popen(
            ["docker", "login", "ghcr.io", "-u", github_user, "--password-stdin"], stdin=subprocess.PIPE
        )
        print(login_process.communicate(input=github_token.encode(), timeout=10))

    remote_name = f"ghcr.io/{github_user}/{local_id}"

    _run(["docker", "tag", local_id, remote_name])
    _run(["docker", "push", remote_name])
    return remote_name


def push_to_gcp(local_id, project_id, region, repository) -> str:
    """Pushes a local Docker image to Artifact Registry."""
    configure_gcp_docker(project_id, region, repository)

    artifact_repo = f"{region}-docker.pkg.dev/{project_id}/{repository}"

    full_image_name = f"{artifact_repo}/{local_id}"
    _run(["docker", "tag", local_id, full_image_name])
    _run(["docker", "push", full_image_name])

    return f"{artifact_repo}/{local_id}"


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
    cli.add_arg(parser, config, ["--docker_target"], choices=["github", "gcp", "ghcr"], required=True)

    args = parser.parse_args()

    local_id = build_docker(docker_file=args.docker_file, image_name=args.image, tag=args.tag)

    if args.docker_target in ["github", "ghcr"]:
        assert args.github_user, "Must specify --github_user when pushing to Github"
        assert args.github_token, "Must specify --github_token when pushing to Github"
        push_to_github(local_id=local_id, github_user=args.github_user, github_token=args.github_token)
    else:
        assert args.region, "Must specify --region when pushing to GCP"
        assert args.project, "Must specify --project when pushing to GCP"
        assert args.repository, "Must specify --repository when pushing to GCP"

        push_to_gcp(local_id, args.project, args.region, args.repository)
