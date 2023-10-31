# helper script used by spin-up-vm.sh and babysit-tpu-vm.sh


# Args we want
# VM name
# rest are flags:
# zone (default us-east1-d)
# type of box (default: v3-32
# vm image (default: tpu-vm-base)
# preemptible (default: false)
# autodelete (default: true)
# setup script (default: infra/helpers/setup-tpu-vm.sh)


# set defaults
ZONE="us-east1-d"
TYPE="v3-32"
VM_IMAGE="tpu-ubuntu2204-base"
PREEMPTIBLE=false
AUTODELETE=true
SETUP_SCRIPT="$SCRIPT_DIR/helpers/setup-tpu-vm.sh"

GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
GIT_REPO=$(git config --get remote.origin.url)

# if GIT_REPO looks like an ssh url, convert it to https
if [[ "$GIT_REPO" == git@* ]]; then
  GIT_REPO=$(echo "$GIT_REPO" | sed 's/:/\//g' | sed 's/git@/https:\/\//g')
fi

# parse args
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    -z|--zone)
      ZONE="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--type)
      TYPE="$2"
      shift # past argument
      shift # past value
      ;;
    -i|--image)
      VM_IMAGE="$2"
      shift # past argument
      shift # past value
      ;;
    -p|--preemptible)
      PREEMPTIBLE="true"
      shift # past argument
      ;;
    -a|--autodelete)
      AUTODELETE="false"
      shift # past argument
      ;;
    -s|--setup)
      SETUP_SCRIPT="$2"
      shift # past argument
      shift # past value
      ;;
    -b|--branch)
      GIT_BRANCH="$2"
      shift # past argument
      shift # past value
      ;;
    -r|--repo)
      GIT_REPO="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option, assume it's the vm name
      # error out if we already set a name
      if [ -n "$VM_NAME" ]; then
        echo "Error: VM name already set to $VM_NAME"
        exit 1
      fi
      VM_NAME="$1"
      shift # past argument
      ;;
  esac
done

# check if the branch we chose has been pushed to the remote
# if not, warn

# get the remote branch name
REMOTE_BRANCH=$(git ls-remote --heads origin "$GIT_BRANCH" | awk '{print $2}' | sed 's/refs\/heads\///g')
# if it's empty, warn
if [ -z "$REMOTE_BRANCH" ]; then
  >&2 echo "Warning: branch $GIT_BRANCH not found on remote $GIT_REPO"
else

  # make sure it's pushed
  LOCAL_COMMIT=$(git rev-parse --short "$GIT_BRANCH")
  REMOTE_COMMIT=$(git rev-parse --short "origin/$REMOTE_BRANCH")

  if [ "$LOCAL_COMMIT" != "$REMOTE_COMMIT" ]; then
   >&2 echo "Warning: branch $GIT_BRANCH not pushed to remote $GIT_REPO. Local commit: $LOCAL_COMMIT, remote commit: $REMOTE_COMMIT"
  fi
fi
