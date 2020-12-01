#!/bin/bash

configs=(
    '0.1'
    '0.2'
    '0.3'
    '0.4'
    '0.5'
    '0.6'
    '0.7'
    '0.8'
    '0.9'
)

for config in ${configs[@]}; do
    python3 -u selected_dropout2.py --config fix_clstm16 --drate ${config} > select_fix_clsym16_${config}.log
done

