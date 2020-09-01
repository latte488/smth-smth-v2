import os
import numpy as np
from sklearn.decomposition import PCA

import matplotlib
matplotlib.use('Agg')
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_box_count(vector, tau=3):
    data_length = vector.shape[0]
    data_dims = vector.shape[1]

    result = []
    for _ in range(data_dims):
        result.append([])

    for t in range(0, (data_length - tau * 2)):
        d1 = vector[t, :]
        d2 = vector[t + tau, :]
        d3 = vector[t + tau * 2, :]

        for i in range(data_dims):
            result[i].append([d1[i], d2[i], d3[i]])

    result_array = np.array(result)
    return result_array


def compute_pca_box_count(vector, tau=3):
    data_length = vector.shape[0]
    data_dims = vector.shape[1]

    pca = PCA(n_components=data_dims)
    pca.fit(vector)
    transformed_x = pca.transform(vector)

    result = []
    for _ in range(data_dims):
        result.append([])

    for t in range(0, (data_length - tau * 2)):
        d1 = transformed_x[t, :]
        d2 = transformed_x[t + tau, :]
        d3 = transformed_x[t + tau * 2, :]

        for i in range(data_dims):
            result[i].append([d1[i], d2[i], d3[i]])

    result_array = np.array(result)
    return result_array


def plot_box_count(vector, label, mutual_info, result_dir, file_header, tau):
    label_type = ['action']
    cmap_type = ['gist_rainbow']
    max_label_values = [173]
    vis_labels = [label]

    # make directory to store images
    for ltype in label_type:
        os.makedirs(os.path.join(result_dir, "bcount_" + ltype), exist_ok=True)

    # compute box count
    box_count_result = compute_box_count(vector, tau=tau)
    seq_length = box_count_result.shape[1]

    for i in range(len(label_type)):
        color_map = plt.get_cmap(cmap_type[i])
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max_label_values[i])
        cm = color_map(norm(vis_labels[i]))

        for i_dim, bcount in enumerate(box_count_result):
            d1 = np.asarray(bcount[:, 0])
            d2 = np.asarray(bcount[:, 1])
            d3 = np.asarray(bcount[:, 2])

            fig = plt.figure(figsize=(6, 6))

            # hidden value
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            for j in range(seq_length - 1):
                ax.plot(d1[j:j+2], d2[j:j+2], d3[j:j+2],'-o', color=cm[j])

            ax.set_xlabel('t')
            ax.set_ylabel('t + tau')
            ax.set_zlabel('t + 2tau')
            ax.set_title(f'tau={tau}')

            # legend
            if max_label_values[i] == 8:
                l0 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(0)), label='0, 0')
                l1 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(1)), label='1, 0')
                l2 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(2)), label='2, 0')
                l3 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(3)), label='0, 1')
                l4 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(4)), label='1, 1')
                l5 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(5)), label='2, 1')
                l6 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(6)), label='0, 2')
                l7 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(7)), label='1, 2')
                l8 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(8)), label='2, 2')
                ax.legend(handles=[l0, l1, l2, l3, l4, l5, l6, l7, l8])
            else:
                l0 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(0)), label='0')
                l1 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(1)), label='1')
                l2 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(2)), label='2')
                ax.legend(handles=[l0, l1, l2])

            fig.tight_layout()
            _output_filename = os.path.join(result_dir,
                                            "bcount_" + label_type[i],
                                            file_header + "_%03d_tau_%02d.png" % (i_dim, tau))
            fig.savefig(_output_filename)
            plt.close()


def plot_pca_box_count(vector, label, mutual_info, result_dir, file_header, tau):
    label_type = ['action']
    cmap_type = ['gist_rainbow']
    max_label_values = [173]
    vis_labels = [label]

    # make directory to store images
    for ltype in label_type:
        os.makedirs(os.path.join(result_dir, "pca_bcount_" + ltype), exist_ok=True)

    # compute box count
    box_count_result = compute_pca_box_count(vector, tau=tau)
    seq_length = box_count_result.shape[1]

    for i in range(len(label_type)):
        color_map = plt.get_cmap(cmap_type[i])
        norm = matplotlib.colors.Normalize(vmin=0, vmax=max_label_values[i])
        cm = color_map(norm(vis_labels[i]))

        for i_dim, bcount in enumerate(box_count_result):
            d1 = np.asarray(bcount[:, 0])
            d2 = np.asarray(bcount[:, 1])
            d3 = np.asarray(bcount[:, 2])

            fig = plt.figure(figsize=(6, 6))

            # hidden value
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            for j in range(seq_length - 1):
                ax.plot(d1[j:j+2], d2[j:j+2], d3[j:j+2],'-o', color=cm[j])

            ax.set_xlabel('t')
            ax.set_ylabel('t + tau')
            ax.set_zlabel('t + 2tau')
            ax.set_title('tau=%d, MI(sp)=%f, MI(tp)=%f' % (tau, mutual_info[i_dim, 0], mutual_info[i_dim, 1]))

            # legend
            if max_label_values[i] == 8:
                l0 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(0)), label='0, 0')
                l1 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(1)), label='1, 0')
                l2 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(2)), label='2, 0')
                l3 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(3)), label='0, 1')
                l4 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(4)), label='1, 1')
                l5 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(5)), label='2, 1')
                l6 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(6)), label='0, 2')
                l7 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(7)), label='1, 2')
                l8 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(8)), label='2, 2')
                ax.legend(handles=[l0, l1, l2, l3, l4, l5, l6, l7, l8])
            else:
                l0 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(0)), label='0')
                l1 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(1)), label='1')
                l2 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(2)), label='2')
                ax.legend(handles=[l0, l1, l2])

            fig.tight_layout()
            _output_filename = os.path.join(result_dir,
                                            "pca_bcount_" + label_type[i],
                                            file_header + "_%03d_tau_%02d.png" % (i_dim, tau))
            fig.savefig(_output_filename)
            plt.close()
        
            if i_dim == 2:
                break
