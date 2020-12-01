#!/bin/bash

configs=(
    'crnn'
    'crnn512'
    'crnn256'
    'crnn128'
    'clstm'
    'clstm512'
    'clstm256'
    'clstm128'
)

for config in ${configs[@]}; do
    python3 -u boxcount.py --config ${config} > boxcount_${config}.log
done

