import numpy as np
import matplotlib.pyplot as plt
import toml

names = [
    'fix_clstm16',
    'fix_clstm32',
    'fix_clstm64',
    'fix_clstm128',
    'fix_clstm256',
    'fix_clstm512',
    'fix_clstm1024',
    'fix_crnn1024',
    'fix_crnn2048',
    'fix_crnn4096',
    'fix_crnn8192',
]

modes = [
    'low',
    'middle',
    'high',
    'random',
]

drates = [(i + 1) / 10 for i in range(9)]
lefts = np.arange(len(drates))
width = 0.15

with open('selected_dropout_acc.toml', 'r') as f:
    acc_dict = toml.load(f)


for name in names:
    plt.figure(figsize=(6.4, 3.8))
    for mode_i, mode in enumerate(modes):
        if mode == 'random':
            acc_means = []
            acc_stds = []
            for drate in drates:
                random_accs = [acc_dict[f'{name}_{mode}{random_i}_{drate}'] for random_i in range(10)]
                random_accs = np.array(random_accs)
                acc_means.append(random_accs.mean())
                acc_stds.append(random_accs.std())
            plt.bar(lefts + width * mode_i, acc_means, yerr=acc_stds, label=f'{name[4:]} {mode}', width=width, align="center")
        else:
            accs = []
            for drate in drates:
                accs.append(acc_dict[f'{name}_{mode}_{drate}'])
            plt.bar(lefts + width * mode_i, accs, label=f'{name[4:]} {mode}', width=width, align="center")
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    plt.tick_params(bottom=False, direction='in')
    plt.xticks(lefts + width, [str(drate) for drate in drates])
    plt.xlabel('Dropout rate')
    plt.ylabel('Accuracy(%)')
    plt.ylim(0, 100)
    plt.savefig(f'sd_random_png/{name}.png')
