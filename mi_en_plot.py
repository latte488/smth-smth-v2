import toml
import matplotlib.pyplot as plt
import numpy as np


def main():

    model_name = 'fix_clstm256'
    width = 0.15
    size = 10
    
    with open(f'mutual_info_{model_name}.toml', 'r') as f:
        mi_dict = toml.load(f)
    with open(f'entropy_{model_name}.toml', 'r') as f:
        entropy_dict = toml.load(f)

    mi_score = [float(value) for value in mi_dict.values()][:size]
    entropy_score = [float(value) for value in entropy_dict.values()][:size]

    lefts = np.array(range(len(mi_score)))

    plt.bar(lefts, mi_score, label=f'mutual info', width=width, align="center")
    plt.bar(lefts + width, entropy_score, label=f'entropy', width=width, align="center")
    plt.legend(bbox_to_anchor=(0, 0), loc='lower left')
    plt.tick_params(bottom=False, direction='in')
    plt.xticks(lefts + width / 2, lefts + 1)
    plt.xlabel('Neuron')
    plt.ylabel('Information')
    plt.savefig(f'both_{model_name}.png')

main()
