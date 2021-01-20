#!/bin/bash

configs=(
    'fix_sclstm16'
    'fix_sclstm32'
    'fix_sclstm64'
    'fix_sclstm128'
)

for config in ${configs[@]}; do
    python3 -u trans_entropy.py --config ${config} > transfer_entropy_${config}.log
done

