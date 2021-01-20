#!/bin/bash

names=(
    'fix_sclstm16'
    'fix_sclstm32'
    'fix_sclstm64'
    'fix_sclstm128'
)

modes=(
    'low'
    'high'
)

for name in ${names[@]}; do
    for mode in ${modes[@]}; do
        for ((drate = 1; drate < 10; drate++)) {
            python3 -u selected_dropout2.py --config ${name} --drate 0.${drate} --mode ${mode} > sd_log/${name}_${mode}_${drate}.log
            python3 -u selected_dropout3.py --config ${name} --drate 0.${drate} --mode ${mode} > entropy_sd_log/${name}_${mode}_${drate}.log
            python3 -u selected_dropout4.py --config ${name} --drate 0.${drate} --mode ${mode} > transfer_entropy_sd_log/${name}_${mode}_${drate}.log
        }
    done
done
