#!/bin/bash

configs=(
    'fix_clstm16'
)

for config in ${configs[@]}; do
    python3 -u mutual_info.py --config ${config} > mutual_info_${config}.log
done

