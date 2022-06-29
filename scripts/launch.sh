PYTHON=$1
PDIR=$(dirname "$PYTHON")

cd "$(dirname $0)/.." || exit

source "$PDIR"/activate
export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH


nohup "$@" &
