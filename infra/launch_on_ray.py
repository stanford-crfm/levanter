#!/usr/bin/python
# Similar to launch.py, but this instead launches on a Ray cluster configured with auto-scaling TPUs

import argparse
import getpass
import os
import tempfile
import time
from pathlib import Path

import draccus
from ray.dashboard.modules.job.common import JobStatus
from ray.dashboard.modules.job.sdk import JobSubmissionClient

import levanter.infra.cli_helpers as cli
import levanter.infra.docker as docker


def main():
    parser = argparse.ArgumentParser()
    config = cli.load_config()

    cli.add_arg(parser, config, ["--docker_base_image"], default="ghcr.io/stanford-crfm/levanter-base:latest")
    cli.add_arg(parser, config, ["--docker_repository"], default="levanter")
    cli.add_arg(parser, config, ["--address"], default="http://127.0.0.1:8265")
    cli.add_arg(parser, config, ["--image_name"], default=f"levanter-{getpass.getuser()}")
    cli.add_capacity_type_args(parser, config)
    cli.add_arg(parser, config, ["--project"], default=cli.gcloud_config()["project"])
    cli.add_arg(parser, config, ["--tpu_type"], required=True)
    # TODO: bring node_count to Ray
    # cli.add_arg(parser, config, ["--node_count"], default=1, type=int)
    cli.add_arg(parser, config, ["--foreground"], default=False, action="store_true")
    cli.add_arg(parser, config, ["--retries"], default=10, type=int)
    cli.add_arg(parser, config, ["--run_id"], default=cli.default_run_id(), type=str)
    cli.add_arg(parser, config, ["--docker_registry"], default="gcp", choices=["gcp", "ghcr"])
    cli.add_arg(parser, config, ["--github_user"], type=str)
    cli.add_arg(parser, config, ["--github_token"], type=str)
    cli.add_arg(parser, config, ["--extra_context"], type=Path, required=False, default=None)
    cli.add_arg(parser, config, ["--zone"], default=None, type=str, required=False)

    parser.add_argument(
        "-e", "--env", action="append", nargs=2, metavar=("KEY", "VALUE"), default=list(config.get("env", {}).items())
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)

    args = parser.parse_args()

    command = args.command
    docker_repository = args.docker_repository
    image_id = args.image_name
    project = args.project
    if args.retries < 0:
        retries = 10000000
    else:
        retries = args.retries

    tpu_type = args.tpu_type

    zone = args.zone
    run_id = args.run_id
    registry = args.docker_registry
    github_user = args.github_user
    github_token = args.github_token
    extra_context = args.extra_context

    if zone is None:
        zone = cli.gcloud_config()["zone"]

    if zone is None:
        raise ValueError("Zone must be specified or set in gcloud config.")

    region = "-".join(zone.split("-")[:-1])

    if command[0] == "--":
        command = command[1:]

    # make an image tag based on the unix timestamp to ensure we always pull the latest image
    tag = int(time.time())

    with docker.copy_extra_ctx(extra_context) as extra_context:
        build_args = {"EXTRA_CTX": extra_context} if extra_context else {}
        base_image, base_tag = docker.split_image_and_tag(args.docker_base_image)
        build_args["IMAGE"] = base_image
        build_args["TAG"] = base_tag

        local_id = docker.build_docker(
            docker_file="docker/tpu/Dockerfile.incremental", image_name=image_id, tag=tag, build_args=build_args
        )

    if registry == "ghcr":
        full_image_id = docker.push_to_github(
            local_id=local_id,
            github_user=github_user,
            github_token=github_token,
        )
    elif registry == "gcp":
        full_image_id = docker.push_to_gcp(
            local_id=local_id,
            project_id=project,
            region=region,
            repository=docker_repository,
        )
    else:
        raise ValueError(f"Unknown docker registry: {registry}")

    env = {k: v for k, v in args.env}

    if "WANDB_PROJECT" not in env:
        env["WANDB_PROJECT"] = "levanter"

    env["GIT_COMMIT"] = cli.get_git_commit()
    env["RUN_ID"] = run_id
    env["WANDB_DOCKER"] = full_image_id

    # run_docker_on_pod(
    #     full_image_id,
    #     command=command,
    #     tpu_type=tpu_type,
    #     env=env,
    #     retries=retries,
    # )

    # Submit the job to the Ray cluster. We have to use the JobSubmissionClient to do this and stringify the arguments
    # we want:
    from levanter.infra.ray_tpu import RunOnPodConfig

    config = RunOnPodConfig(
        image_id=full_image_id,
        command=command,
        tpu_type=tpu_type,
        env=env,
        name="levanter",
        retries=retries,
    )

    with tempfile.NamedTemporaryFile(suffix=".yaml", prefix=f"launch-{run_id}-", dir=".") as f:
        yaml = draccus.dump(config)
        f.write(yaml.encode("utf-8"))
        f.flush()

        f_name = os.path.relpath(f.name)
        print(f"Submitting job with config path {f_name}")

        client = JobSubmissionClient(args.address)

        job_id = _make_unique_job_id(client, run_id)

        job_id = client.submit_job(
            entrypoint=f"python src/levanter/infra/ray_tpu.py --config_path {f_name}",
            runtime_env={"working_dir": "./"},
            job_id=job_id,
        )

        print(
            f"""
-------------------------------------------------------
Job '{job_id}' submitted successfully
-------------------------------------------------------

Next steps
  Query the logs of the job:
    ray job logs {job_id}
  Query the status of the job:
    ray job status {job_id}
  Request the job to be stopped:
    ray job stop {job_id}
"""
        )

    if args.foreground:

        async def tail_job(job_id):
            async for line in client.tail_job_logs(job_id):  # type: ignore
                print(line, end="")

                status = client.get_job_status(job_id)
                if status in {JobStatus.FAILED, JobStatus.SUCCEEDED, JobStatus.STOPPED}:
                    break

        print("Tailing job logs")
        wait_until_status(
            client, job_id, {JobStatus.RUNNING, JobStatus.FAILED, JobStatus.SUCCEEDED, JobStatus.STOPPED}
        )
        # tail_job(job_id)
        import asyncio

        asyncio.run(tail_job(job_id))


def wait_until_status(client, job_id, status_to_wait_for, timeout_seconds=5):
    start = time.time()
    while time.time() - start <= timeout_seconds:
        status = client.get_job_status(job_id)
        print(f"status: {status}")
        if status in status_to_wait_for:
            break
        time.sleep(1)


# try to make the job id be the same as the run id, but if it already exists, just make it unique
def _make_unique_job_id(client, run_id):
    job_id = run_id
    try:
        while client.get_job_status(job_id) is not None:
            job_id = f"{run_id}-{time.time_ns()}"
    except Exception as e:  # noqa
        if "does not exist" in str(e):
            pass
        else:
            raise
    return job_id


if __name__ == "__main__":
    main()
