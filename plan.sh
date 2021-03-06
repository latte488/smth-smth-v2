#!/bin/bash

configs=(
    'config_crnn512.json'
    'config_clstm512.json'
    #'config_cesn.json'
)

for config in ${configs[@]}; do
    python3 -u train.py -c configs/${config} --use_cuda --gpu 0 > ${config}.log
done

