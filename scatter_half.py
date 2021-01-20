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

def plot_e_mi_te(e_score, mi_score, te_score, label):

    corrcoef = np.corrcoef(np.array([e_score, mi_score, te_score]))

    e_mi_cc = corrcoef[0][1]
    e_te_cc = corrcoef[0][2]
    mi_te_cc = corrcoef[1][2]

    e_str = 'entropy'
    mi_str = 'mutual infomation'
    te_str = 'transfer entropy'
    plot(f'scatter_half/{name}_e_mi_{label}', e_score, e_str, mi_score, mi_str, f'Correlation coefficient: {e_mi_cc:.3}')
    plot(f'scatter_half/{name}_e_te_{label}', e_score, e_str, te_score, te_str, f'Correlation coefficient: {e_te_cc:.3}')
    plot(f'scatter_half/{name}_mi_te_{label}', mi_score, mi_str, te_score, te_str, f'Correlation coefficient: {mi_te_cc:.3}')
          
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

    e_score.sort()
    mi_score.sort()
    te_score.sort()

    q = len(e_score) // 4
    s = 0
    e = q
    plot_e_mi_te(e_score[s:e], mi_score[s:e], te_score[s:e], 'lowlow')
    s = e
    e = s + q
    plot_e_mi_te(e_score[s:e], mi_score[s:e], te_score[s:e], 'lowhigh')
    s = e
    e = s + q
    plot_e_mi_te(e_score[s:e], mi_score[s:e], te_score[s:e], 'highlow')
    s = e
    e = s + q
    plot_e_mi_te(e_score[s:e], mi_score[s:e], te_score[s:e], 'highhigh')

