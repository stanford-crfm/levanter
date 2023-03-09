set -x
# broadly based on https://github.com/ayaka14732/tpu-starter

# tcmalloc interferes with intellij remote ide
sudo patch -f -b /etc/environment << EOF
2c2
< LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
---
> #LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
EOF

# don't complain if already applied
retCode=$?
[[ $retCode -le 1 ]] || exit $retCode

# install python 3.10, latest git, and nfs
sudo systemctl stop unattended-upgrades  # this frequently holds the apt lock
# sometimes apt-get update fails, so retry a few times
for i in {1..5}; do
  sudo apt-get install -y software-properties-common \
  && sudo add-apt-repository -y ppa:deadsnakes/ppa \
  && sudo add-apt-repository -y ppa:git-core/ppa \
  && sudo apt-get update \
  && sudo apt-get install -y python3.10-full python3.10-dev nfs-common git \
  && break
done
sudo systemctl start unattended-upgrades

VENV=~/venv310
# if the venv doesn't exist, make it
if [ ! -d "$VENV" ]; then
    echo "Creating virtualenv at $VENV"
    python3.10 -m venv $VENV
fi
source $VENV/bin/activate

pip install -U pip
pip install -U wheel

# jax and jaxlib
# libtpu sometimes has issues installing for clinical (probably firewall?)
# TODO: get these deps from somewhere reasonable
for i in {1..5}; do
    pip install -U "jax[tpu]==0.4.5" libtpu-nightly==0.1.dev20230216 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    if [ $? -eq 0 ]; then
        break
    fi
    if 5 == $i; then
        echo "Error installing jax, giving up"
        exit 1
    fi
    echo "Error installing jax, retrying in 5 seconds"
    sleep 5
done

# clone levanter
git clone https://github.com/stanford-crfm/levanter.git

echo $VENV > levanter/infra/venv_path.txt

cd levanter

pip install -e .
