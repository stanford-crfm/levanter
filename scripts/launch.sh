#!/bin/bash
# This script is used for launching on TPU pods (or other direct run environments) via remote ssh with a virtual env
set -e
umask 000
LEV_ROOT=$(dirname "$(readlink -f $0)")/..

source /files/venv32/bin/activate

PYTHONPATH=${LEV_ROOT}:${LEV_ROOT}/src:${LEV_ROOT}/examples:$PYTHONPATH nohup "$@" > >(tee -a stdout-`hostname`.log) 2> >(tee -a stderr-`hostname`.log >&2)
