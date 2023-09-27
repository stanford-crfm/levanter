umask 000
LEV_ROOT=$(dirname "$(readlink -f $0)")/..

PYTHONPATH=${LEV_ROOT}:${LEV_ROOT}/src:${LEV_ROOT}/examples:$PYTHONPATH "$@"
