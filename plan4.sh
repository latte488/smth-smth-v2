#!/bin/bash

configs=(
    'fix_clstm64'
)

for config in ${configs[@]}; do
    python3 -u fix_boxcount.py --config ${config} > fix_boxcount_${config}.log
done

