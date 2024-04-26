# broadly based on https://github.com/ayaka14732/tpu-starter

# parse some arguments
# usage: ./setup-tpu-vm.sh -b|--branch <git commit or branch for levanter> -r <git repo for levanter>

if [ "$DEBUG" == "1" ]; then
  set -x
fi

REPO="https://github.com/stanford-crfm/levanter.git"
BRANCH=main

if [ "$GIT_BRANCH" != "" ]; then
  BRANCH="$GIT_BRANCH"
fi

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -b|--branch)
      BRANCH="$2"
      shift
      shift
      ;;
    -r|--repo)
      REPO="$2"
      shift
      shift
      ;;
    *)
      >&2 echo "Unknown option $1"
      exit 1
      ;;
  esac
done

# we frequently deal with commands failing, and we like to loop until they succeed. this function does that for us
function retry {
  for i in {1..5}; do
    $@
    if [ $? -eq 0 ]; then
      break
    fi
    if [ $i -eq 5 ]; then
      >&2 echo "Error running $*, giving up"
      exit 1
    fi
    >&2 echo "Error running $*, retrying in 5 seconds"
    sleep 5
  done
}

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


# set these env variables b/c it makes tensorstore behave better
if ! grep -q TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS /etc/environment; then
  # need sudo
  echo "TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS=60" | sudo tee -a /etc/environment > /dev/null
fi

if ! grep -q TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES /etc/environment; then
  echo "TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES=1024" | sudo tee -a /etc/environment > /dev/null
fi

# install python 3.10, latest git
sudo systemctl stop unattended-upgrades  # this frequently holds the apt lock
sudo systemctl disable unattended-upgrades
sudo apt remove -y unattended-upgrades
# if it's still running somehow, kill it
if [ $(ps aux | grep unattended-upgrade | wc -l) -gt 1 ]; then
  sudo kill -9 $(ps aux | grep unattended-upgrade | awk '{print $2}')
fi

# sometimes apt-get update fails, so retry a few times
retry sudo apt-get install -y software-properties-common
retry sudo add-apt-repository -y ppa:deadsnakes/ppa
retry sudo add-apt-repository -y ppa:git-core/ppa
retry sudo apt-get -qq update
retry sudo apt-get -qq install -y python3.10-full python3.10-dev git

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
#retry pip install -U "jax[tpu]==0.4.5" libtpu-nightly==0.1.dev20230216 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
retry pip install -U "jax[tpu]==0.4.26" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# clone levanter
git clone $REPO levanter
echo $VENV > levanter/infra/venv_path.txt

cd levanter

# checkout the branch we want

echo "Checking out branch $BRANCH"

git checkout $BRANCH

# install levanter

pip install -e .
