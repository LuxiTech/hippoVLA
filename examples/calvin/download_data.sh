#!/bin/bash

# Usage:
#   bash download_data.sh D
#   bash download_data.sh D /mnt/data/jiangnan/calvin

set -euo pipefail

split="${1:-}"
target_dir="${2:-$(pwd)}"

mkdir -p "${target_dir}"
cd "${target_dir}"

if [ "${split}" = "D" ]; then
    echo "Downloading task_D_D to ${target_dir} ..."
    wget -c http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
    unzip -o task_D_D.zip
    echo "saved folder: ${target_dir}/task_D_D"
elif [ "${split}" = "ABC" ]; then
    echo "Downloading task_ABC_D to ${target_dir} ..."
    wget -c http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip
    unzip -o task_ABC_D.zip
    echo "saved folder: ${target_dir}/task_ABC_D"
elif [ "${split}" = "ABCD" ]; then
    echo "Downloading task_ABCD_D to ${target_dir} ..."
    wget -c http://calvin.cs.uni-freiburg.de/dataset/task_ABCD_D.zip
    unzip -o task_ABCD_D.zip
    echo "saved folder: ${target_dir}/task_ABCD_D"
elif [ "${split}" = "debug" ]; then
    echo "Downloading debug dataset to ${target_dir} ..."
    wget -c http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip
    unzip -o calvin_debug_dataset.zip
    echo "saved folder: ${target_dir}/calvin_debug_dataset"
else
    echo "Failed: Usage download_data.sh D|ABC|ABCD|debug [target_dir]"
    exit 1
fi