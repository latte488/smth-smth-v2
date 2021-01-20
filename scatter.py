import matplotlib.pyplot as plt
import numpy as np
import toml

def plot(filename, score1, label1, score2, label2, title):
    fig = plt.figure()
    plt.scatter(score1, score2)
    plt.xlabel(label1)
    plt.ylabel(label2)
    plt.title(title)
    plt.savefig(filename)
    plt.close(fig)

names = [
    'fix_sclstm16',
    'fix_sclstm32',
    'fix_sclstm64',
    'fix_sclstm128',
]


for name in names:
    with open(f'entropy_{name}.toml', 'r') as f:
        e_dict = toml.load(f)

    with open(f'mutual_info_{name}.toml', 'r') as f:
        mi_dict = toml.load(f)

    with open(f'transfer_entropy_{name}.toml', 'r') as f:
        te_dict = toml.load(f)


    e_score = [float(value) for value in e_dict.values()]
    mi_score = [float(value) for value in mi_dict.values()]
    te_score = [float(value) for value in te_dict.values()]

    corrcoef = np.corrcoef(np.array([e_score, mi_score, te_score]))

    e_mi_cc = corrcoef[0][1]
    e_te_cc = corrcoef[0][2]
    mi_te_cc = corrcoef[1][2]

    e_str = 'entropy'
    mi_str = 'mutual infomation'
    te_str = 'transfer entropy'
    plot(f'scatter/{name}_e_mi', e_score, e_str, mi_score, mi_str, f'Correlation coefficient: {e_mi_cc:.3}')
    plot(f'scatter/{name}_e_te', e_score, e_str, te_score, te_str, f'Correlation coefficient: {e_te_cc:.3}')
    plot(f'scatter/{name}_mi_te', mi_score, mi_str, te_score, te_str, f'Correlation coefficient: {mi_te_cc:.3}')
          
