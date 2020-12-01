#!/bin/bash

configs=(
    'config_crnn256.json'
    'config_crnn128.json'
    'config_clstm256.json'
    'config_clstm128.json'
)

for config in ${configs[@]}; do
    python3 -u train.py -c configs/${config} --use_cuda --gpu 0 > ${config}.log
done

