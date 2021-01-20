#!/bin/bash

configs=(
    'fix_sclstm128'
    'fix_sclstm64'
    'fix_sclstm32'
    'fix_sclstm16'
)

for config in ${configs[@]}; do
    python3 -u mutual_info.py --config ${config} > mutual_info_${config}.log
done

