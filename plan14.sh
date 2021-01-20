#!/bin/bash

names=(
    'fix_clstm16'
    'fix_clstm32'
    'fix_clstm64'
    'fix_clstm128'
    'fix_clstm256'
    'fix_clstm512'
    'fix_clstm1024'
    'fix_crnn1024'
    'fix_crnn2048'
    'fix_crnn4096'
    'fix_crnn8192'
)

modes=(
    'low'
    'middle'
    'high'
)

for name in ${names[@]}; do
    for mode in ${modes[@]}; do
        for ((drate = 1; drate < 10; drate++)) {
            python3 -u selected_dropout4.py --config ${name} --drate 0.9${drate} --mode ${mode} > transfer_entropy_sd_log/${name}_${mode}_${drate}.log
        }
    done
done

