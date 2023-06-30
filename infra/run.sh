umask 000
LEV_ROOT=$(dirname "$(readlink -f $0)")/..

# figure out venv, first check if we wrote a path in infra/venv_path
if [ ! -d "$VENV" ] && [ -f "$LEV_ROOT/infra/venv_path.txt" ]; then
  VENV=$(cat "$LEV_ROOT"/infra/venv_path.txt)
fi

# if we still don't have a venv, we'll look in our default
if [ ! -d "$VENV" ]; then
  VENV=/files/venv32
fi

source $VENV/bin/activate

pip install -U pyrallis
pip install -U torch torchvision torchaudio

PYTHONPATH=${LEV_ROOT}:${LEV_ROOT}/src:${LEV_ROOT}/examples:$PYTHONPATH "$@"
