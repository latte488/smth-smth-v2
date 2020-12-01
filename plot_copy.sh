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
    mkdir -p plots/${config}
    cp trained_models/${config}_scratch/plots/* plots/${config}
done

