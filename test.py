import toml
import random

def main():
    filename = 'test'
    n_hidden = 64

    mutual_infos = []
    for i in range(n_hidden):
        mutual_infos.append(random.random())

    mutual_info_dict = {}
    for i in range(n_hidden):
        mutual_info_dict[f'{i:04}'] = mutual_infos[i]
    with open(f'mutual_info_{filename}.toml', 'w') as f:
        toml_str = toml.dump(mutual_info_dict, f)
    print(toml_str)


main()
