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
    'high',
]

drates = [float(f'0.9{i + 1}') for i in range(9)]
lefts = np.arange(len(drates))
width = 0.15

with open('selected_dropout_acc.toml', 'r') as f:
    mi_acc_dict = toml.load(f)

with open('transfer_entropy_selected_dropout_acc.toml', 'r') as f:
    te_acc_dict = toml.load(f)


for name in names:
    
    for mode_i, mode in enumerate(modes):
        
        fig = plt.figure(figsize=(6.4, 3.8))
        
        mi_accs = []
        te_accs = []
        
        for drate in drates:
            
            key = f'{name}_{mode}_{drate}'
            mi_accs.append(mi_acc_dict[key])
            te_accs.append(te_acc_dict[key])

        plt.plot(lefts, mi_accs, label=f'{name[4:]} {mode} mutual information')
        plt.plot(lefts, te_accs, label=f'{name[4:]} {mode} transfer entropy')

        plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
        plt.tick_params(bottom=False, direction='in')
        plt.xticks(lefts, [str(drate) for drate in drates])
        plt.xlabel('Dropout rate')
        plt.ylabel('Accuracy(%)')
        plt.ylim(0, 100)
        plt.savefig(f'all2_sd_png/{name}_{mode}.png')

        plt.close(fig)
        
