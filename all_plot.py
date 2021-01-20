import numpy as np
import matplotlib.pyplot as plt
import toml

names = [
    'fix_sclstm16',
    'fix_sclstm32',
    'fix_sclstm64',
    'fix_sclstm128',
]

modes = [
    'low',
    'high',
]

drates = [(i + 1) / 10 for i in range(9)]
lefts = np.arange(len(drates))
width = 0.15

with open('entropy_selected_dropout_acc.toml', 'r') as f:
    e_acc_dict = toml.load(f)

with open('selected_dropout_acc.toml', 'r') as f:
    mi_acc_dict = toml.load(f)

with open('transfer_entropy_selected_dropout_acc.toml', 'r') as f:
    te_acc_dict = toml.load(f)


for name in names:
    
    for mode_i, mode in enumerate(modes):
        
        fig = plt.figure(figsize=(6.4, 3.8))
        
        if mode == 'random':
            
            acc_means = []
            acc_stds = []
            
            for drate in drates:
                
                random_accs = [acc_dict[f'{name}_{mode}{random_i}_{drate}'] for random_i in range(10)]
                random_accs = np.array(random_accs)
                acc_means.append(random_accs.mean())
                acc_stds.append(random_accs.std())
            
            plt.plot(lefts, acc_means, yerr=acc_stds, label=f'{name[4:]} {mode}')
        
        else:
            
            e_accs = []
            mi_accs = []
            te_accs = []
            
            for drate in drates:
                
                key = f'{name}_{mode}_{drate}'
                e_accs.append(e_acc_dict[key])
                mi_accs.append(mi_acc_dict[key])
                te_accs.append(te_acc_dict[key])

            plt.plot(lefts, e_accs, label=f'Entropy')
            plt.plot(lefts, mi_accs, label=f'Mutual Information')
            plt.plot(lefts, te_accs, label=f'Transfer Entropy')

        plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
        plt.tick_params(bottom=False, direction='in')
        plt.xticks(lefts, [str(drate) for drate in drates])
        plt.xlabel('Pruning Rate')
        plt.ylabel('Accuracy(%)')
        plt.ylim(0, 100)
        plt.savefig(f'all_png/{name}_{mode}.png')

        plt.close(fig)
        
