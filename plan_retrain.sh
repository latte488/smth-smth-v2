#!/bin/bash

names=(
    'fix_sclstm128'
    'fix_sclstm64'
    'fix_sclstm32'
    'fix_sclstm16'
)

modes=(
    'low'
    'high'
)

for name in ${names[@]}; do
    for mode in ${modes[@]}; do
        for ((drate = 1; drate < 10; drate++)) {
            python3 -u en_train.py --config ${name} --drate 0.${drate} --mode ${mode} > re_train/en_${name}_${mode}_${drate}.log
            python3 -u mi_train.py --config ${name} --drate 0.${drate} --mode ${mode} > re_train/mi_${name}_${mode}_${drate}.log
            python3 -u te_train.py --config ${name} --drate 0.${drate} --mode ${mode} > re_train/te_${name}_${mode}_${drate}.log
        }
    done
done

