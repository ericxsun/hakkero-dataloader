#!/usr/bin/env bash
#

SELF_DIR=$(cd $(dirname ${BASH_SOURCE});pwd)

PROJECT_ROOT=${SELF_DIR}/..

# ----------------------------------------------------------------------------------------------------------------------
check_fmt() {
  cd ${PROJECT_ROOT}

  ${ROOT_PYTHON_BIN:+${ROOT_PYTHON_BIN}/}isort hakkero setup.py || exit 1
  ${ROOT_PYTHON_BIN:+${ROOT_PYTHON_BIN}/}autopep8 hakkero setup.py || exit 1
  ${ROOT_PYTHON_BIN:+${ROOT_PYTHON_BIN}/}black hakkero setup.py || exit 1

  cd ${SELF_DIR}
}

check_dev() {
  ${ROOT_PYTHON_BIN:+${ROOT_PYTHON_BIN}/}pip install -r ${SELF_DIR}/dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
}


# ----------------------------------------------------------------------------------------------------------------------
if [[ "$1" == "fmt" ]]; then
  check_fmt
elif [[ "$1" == "dev" ]]; then
  check_dev
else
  echo "not support cmd: $1"
fi
