umask 000
PSI_ROOT=$(dirname "$(readlink -f $0)")/..

source /files/venv32/bin/activate

PYTHONPATH=${PSI_ROOT}:${PSI_ROOT}/src:${PSI_ROOT}/examples:$PYTHONPATH "$@"
