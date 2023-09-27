# broadly based on https://github.com/ayaka14732/tpu-starter

# parse some arguments
# usage: ./setup-tpu-vm.sh -b|--branch <git commit or branch for levanter> -r <git repo for levanter>

if [ "$DEBUG" == "1" ]; then
  set -x
fi

function get_metadata {
  # -f is fail on error, so we don't get 404 output
  curl -f -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" -H "Metadata-Flavor: Google"
}


# try to query project metadata to find username, branch, and repo

# try to get username from vm metadata
USERNAME=$(get_metadata levanter_user)
if [ $? -ne 0 ]; then
  >&2 echo "Error getting username from metadata, falling back to current user"
  USERNAME=$(whoami)
fi

echo "USERNAME" $USERNAME


# try to get branch from vm metadata
if [ "$BRANCH" == "" ]; then
  BRANCH=$(get_metadata levanter_branch)
fi

if [ "$BRANCH" != "" ]; then
  BRANCH="$GIT_BRANCH"
fi

if [ "$BRANCH" == "" ]; then
  BRANCH="main"
fi

# try to get repo from vm metadata
if [ "$REPO" == "" ]; then
  REPO=$(get_metadata levanter_repo)
fi

if [ "$REPO" == "" ]; then
  REPO="https://github.com/stanford-crfm/levanter.git"
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
    -u|--user)
      USERNAME="$2"
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

# jax and jaxlib
retry pip install -U "jax[tpu]==0.4.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# setup levanter as user in their home directory
HDIR=$(getent passwd $USERNAME | cut -d: -f6)

# clone levanter
# TODO: add secrets so we can ssh clone
echo "Cloning $REPO into $HDIR/levanter"
sudo -u $USERNAME git clone $REPO $HDIR/levanter

# checkout the branch we want

echo "Checking out branch $BRANCH"

sudo -u $USERNAME git -C $HDIR/levanter checkout $BRANCH

# install levanter

sudo -u $USERNAME  pip install -e $HDIR/levanter/
