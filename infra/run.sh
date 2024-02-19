umask 000
LEV_ROOT=$(dirname "$(readlink -f $0)")/..

# Create a directory for Hugging Face cache in /dev/shm if it doesn't exist
HF_CACHE_DIR="/dev/shm/huggingface_cache"
mkdir -p "$HF_CACHE_DIR"

# Set the Hugging Face cache environment variable
export HF_HOME="$HF_CACHE_DIR"

# figure out venv, first check if we wrote a path in infra/venv_path
if [ ! -d "$VENV" ] && [ -f "$LEV_ROOT/infra/venv_path.txt" ]; then
  VENV=$(cat "$LEV_ROOT"/infra/venv_path.txt)
fi

# if we still don't have a venv, we'll look in our default
if [ ! -d "$VENV" ]; then
  VENV=/files/venv32
fi

source $VENV/bin/activate

PYTHONPATH=${LEV_ROOT}:${LEV_ROOT}/src:${LEV_ROOT}/examples:$PYTHONPATH "$@"
