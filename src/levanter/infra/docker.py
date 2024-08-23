import json
import os
import pty
import shutil
import subprocess
import sys
from pathlib import Path


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


def _rm(path):
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.is_file():
        os.remove(path)
    elif path.exists():
        raise RuntimeError(f"Remove failed. Path ({path}) is neither a directory nor a file.")


def _cp(src, dst):
    # delete dst if exists
    _rm(dst)

    if src.is_dir():
        shutil.copytree(src, dst)
    elif src.is_file():
        shutil.copy(src, dst)
    else:
        raise RuntimeError(f"Copy failed. Source path ({src}) is neither a directory nor a file. Check if it exists.")


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


def build_docker(docker_file, image_name, tag, mount_src) -> str:
    """Builds a Docker image, enables artifact access, and pushes to Artifact Registry."""
    # Copy external files temporarily to .mnt
    mount_dst = Path(".mnt")
    _cp(mount_src, mount_dst)

    # Get mounting path in docker image.
    levanter_path = Path("/opt/levanter")
    extra_context = levanter_path / mount_src
    _run(
        [
            "docker",
            "buildx",
            "build",
            "--build-arg",
            f"EXTRA_CTX={extra_context.resolve()}",
            "--platform=linux/amd64",
            "-t",
            f"{image_name}:{tag}",
            "-f",
            docker_file,
            ".",
        ]
    )
    # clean up after building
    _rm(mount_dst)

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
