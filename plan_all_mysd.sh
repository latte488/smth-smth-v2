#!/bin/bash

names=(
    'fix_sclstm128'
)

for name in ${names[@]}; do
    for ((number = 0; number < 128; number++)) {
        python3 -u all_mysd.py --config ${name} --drate 100 --mode number${number} > all_mysd_log/${name}_${number}.log
    }
done

