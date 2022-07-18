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