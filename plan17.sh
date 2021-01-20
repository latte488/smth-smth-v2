#!/bin/bash

configs=(
    'config_fix_scrnn2048.json'
    'config_fix_scrnn8192.json'
)

date

for config in ${configs[@]}; do
    python3 -u fix_train.py -c configs/${config} --use_cuda --gpu 0 > ${config}.log
done

date

