from fabric import *

@task
def hello(c):
    c.run('echo "Hello World!"')


@task
def setup(c):
    "run first time setup: upload the setup file and run it"
    c: Connection
    c.put('setup-tpu-vm.sh', 'setup-tpu-vm.sh')
    c.run('bash setup-tpu-vm.sh')
    enable_nfs(c)

@task
def enable_nfs(c):
    c: Connection
    nfs_server = c.config['nfs_server']
    MOUNT_POINT = "/files"
    c.sudo(f"mkdir -p {MOUNT_POINT}")
    current_line = c.run(f'grep {nfs_server} /etc/fstab || true', warn=True, hide=True)
    desired_line = f'{nfs_server}:/propulsion {MOUNT_POINT} nfs defaults 0 0'
    if current_line.stdout != desired_line:
        print(f"Current: {current_line}\nDesired: {desired_line}")
        # first remove the bad line
        c.sudo(f'grep -v {nfs_server} /etc/fstab > /tmp/fstab.new')
        # then add the good line
        c.sudo(f'echo "{desired_line}" >> /tmp/fstab.new')
        # then move the new fstab back into place
        c.sudo('mv /tmp/fstab.new /etc/fstab')

    c.sudo('mount -a')