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
    'high',
]

drates = [(i + 1) / 10 for i in range(9)]
lefts = np.arange(len(drates))
width = 0.15

with open('selected_dropout_acc.toml', 'r') as f:
    mi_acc_dict = toml.load(f)

with open('entropy_selected_dropout_acc.toml', 'r') as f:
    entropy_acc_dict = toml.load(f)


for name in names:
    plt.figure(figsize=(6.4, 3.8))
    for mode_i, mode in enumerate(modes):
        if mode == 'random':
            mi_acc_means = []
            mi_acc_stds = []
            entropy_acc_means = []
            entropy_acc_stds = []
            for drate in drates:
                mi_random_accs = [mi_acc_dict[f'{name}_{mode}{random_i}_{drate}'] for random_i in range(10)]
                mi_random_accs = np.array(mi_random_accs)
                mi_acc_means.append(mi_random_accs.mean())
                mi_acc_stds.append(mi_random_accs.std())
                entropy_random_accs = [entropy_acc_dict[f'{name}_{mode}{random_i}_{drate}'] for random_i in range(10)]
                entropy_random_accs = np.array(entropy_random_accs)
                entropy_acc_means.append(entropy_random_accs.mean())
                entropy_acc_stds.append(entropy_random_accs.std())
            plt.bar(lefts + width * mode_i, mi_acc_means, yerr=mi_acc_stds, label=f'{name[4:]} {mode}', width=width, align="center")
            plt.bar(lefts + width * mode_i, entropy_acc_means, yerr=entropy_acc_stds, label=f'{name[4:]} {mode}', width=width, align="center")
        else:
            mi_accs = []
            entropy_accs = []
            for drate in drates:
                mi_accs.append(mi_acc_dict[f'{name}_{mode}_{drate}'])
                entropy_accs.append(entropy_acc_dict[f'{name}_{mode}_{drate}'])
            plt.bar(lefts + width * mode_i, mi_accs, label=f'{name[4:]} mutual info [{mode}]', width=width, align="center")
            plt.bar(lefts + width * (mode_i + 1), entropy_accs, label=f'{name[4:]} entropy [{mode}]', width=width, align="center")
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    plt.tick_params(bottom=False, direction='in')
    plt.xticks(lefts + width / 2, [str(drate) for drate in drates])
    plt.xlabel('Dropout rate')
    plt.ylabel('Accuracy')
    plt.ylim(0, 100)
    plt.savefig(f'both_sd_png/{name[4:]}.png')
