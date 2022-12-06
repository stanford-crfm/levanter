#!/bin/bash
set -e
umask 000
PSI_ROOT=$(dirname "$(readlink -f $0)")/..

# conda activate levanter

env

PYTHONPATH=${PSI_ROOT}:${PSI_ROOT}/src:${PSI_ROOT}/examples:$PYTHONPATH "$@" > >(tee -a stdout-`hostname`.log) 2> >(tee -a stderr-`hostname`.log >&2)
