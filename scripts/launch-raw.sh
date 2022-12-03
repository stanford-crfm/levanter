#!/bin/bash
set -e
umask 000
PSI_ROOT=$(dirname "$(readlink -f $0)")/..

# conda activate levanter

PYTHONPATH=${PSI_ROOT}:${PSI_ROOT}/src:${PSI_ROOT}/examples:$PYTHONPATH "$@"
