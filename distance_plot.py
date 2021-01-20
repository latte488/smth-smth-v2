import toml
import matplotlib.pyplot as plt
import numpy as np


def mi_entropy_distance(model_name):

    with open(f'mutual_info_{model_name}.toml', 'r') as f:
        mi_dict = toml.load(f)
    with open(f'entropy_{model_name}.toml', 'r') as f:
        entropy_dict = toml.load(f)

    mi_score = [float(value) for value in mi_dict.values()]
    entropy_score = [float(value) for value in entropy_dict.values()]

    mi_score = np.array(mi_score)
    entropy_score = np.array(entropy_score)
    distance = entropy_score - mi_score
    
    print(distance.mean())

model_names = [
    'fix_clstm128',
    'fix_clstm256',
    'fix_clstm512',
    'fix_clstm1024',
]

for model_name in model_names:
    mi_entropy_distance(model_name)
