# Ray TPU Job Manager

This is a quick design document to explain how our Ray TPU Job Manager works.

## Introduction

Please see the [Ray documentation](https://docs.ray.io/en/latest/index.html) for more information.
Ray is a resource-aware job scheduler, so you can specify the resources that a job requires:

```python
@ray.remote(num_cpus=4)
def my_cpu_job():
    ...
```

For GPUs, Ray lets you specify the number of GPUs you need:

```python
@ray.remote(num_gpus=1)
def my_gpu_job():
    ...
```

In Ray, TPUs are roughly represented the same way, but there are a number of problems with that approach.
In particular:

* Ray's granularity allows it to schedule a task on a single machine, not across multiple machines. In particular,
Ray can't directly schedule a task on a TPU slice that spans multiple machines.
* Google requires that only one process on a machine can access the TPU at a time.

This document explains how we work around those problems.

### Scheduling Slices of TPUs

TPU slices must be used in a SPMD manner (this is probably not quite true, but it's the easiest way to use them).
This means that you need to schedule the same task on multiple slices of the TPU at once. Ray can't do this directly.

Allen Wang (@allenwang28) at Google wrote [this gist](https://gist.github.com/allenwang28/e3400b9e9212b50aa1cda55ebeccea60#file-ray_tpu_task-py)
that is most of the way to a solution. The key idea is to schedule a task on the special `"TPU-XXXX-head"` resource
(where XXXX is the TPU type like `"v4-256"`). This gives you a "lock" on the TPU slice. Once you have the lock, you can
query the VM to get a unique resource that is shared only for this particular slice. You can then use this resource to
schedule K tasks on the K workers that are on the same slice. These tasks do the actual work.

Managing preemption then is just a question of rescheduling the job when it gets preempted: getting a new lock and
restarting the job.

### Dealing with `libtpu`

`libtpu` is a library that interfaces with the TPU. `libtpu` has a hard requirement that only one process on a machine
can access the TPU at a time. It manages this with a lockfile called `/tmp/libtpu_lockfile`. Ordinarily, this is fine,
as the lockfile is removed when the process exits. However, Ray maintains a pool of workers that don't ordinarily exit
between tasks. This means that the lockfile is not removed, and the next job that tries to access the TPU will fail.

As best I can tell, it's actually fine to remove this lockfile so long as you're not trying to access the TPU from
multiple processes on the same machine. Because we're using Ray to lock the resources, we're guaranteed that only one
process will be accessing the TPU at a time, so we just remove the lockfile when the task finishes.

## Ray+TPU+Docker

So above we have the core idea of how to use Ray with TPUs. However, there are a few additional complications when
we want our jobs to be running separate docker containers. Some of this is just dealing with Ray+Docker, but some of it
is Ray+TPU+Docker specific.

### Ray+Docker

We use a Docker container to run the core Ray scheduling process on each machine. We also want to use a different
Docker container to run actual jobs. In theory, Ray can run tasks inside task-specific docker images, but I've heard it
doesn't work well. We also want to avoid a full Docker-in-Docker setup (which I've also heard is tricky), so we
instead want the scheduler to launch sibling containers. To do that, we bind-mount the docker socket into the
scheduler container.

### Ray+TPU+Docker

Above we discussed dealing with the TPU lockfile. The only real remaining issues are:

* you have to use `--privileged` to use TPUs.
* There's a bug in Ray's Docker support that [causes the `TPU-<TPU_TYPE>-head` resource to be assigned to all workers](https://github.com/ray-project/ray/pull/47777), not just
the leader. We have a patch.
