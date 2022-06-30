umask 000
cd "$(dirname $0)/.." || exit

PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH nohup "$@"
