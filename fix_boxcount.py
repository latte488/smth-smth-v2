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


def plot_box_count(vector, label, result_dir, file_header, tau):
    label_type = ['action']
    cmap_type = ['gist_rainbow']
    max_label_values = [173]
    vis_labels = [label]

    # make directory to store images
    os.makedirs(os.path.join(result_dir, "bcount_" + file_header), exist_ok=True)

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
                ax.plot(d1[j:j+2], d2[j:j+2], d3[j:j+2],'-o', color='b')

            ax.set_xlabel('t')
            ax.set_ylabel('t + tau')
            ax.set_zlabel('t + 2tau')
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-1.0, 1.0)
            ax.set_zlim(-1.0, 1.0)
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
                l0 = mlines.Line2D([0, 1], [0, 1], color='b', label='attract')
                #l1 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(1)), label='1')
                #l2 = mlines.Line2D([0, 1], [0, 1], color=color_map(norm(2)), label='2')
                ax.legend(handles=[l0])

            fig.tight_layout()
            _output_filename = os.path.join(result_dir,
                                            "bcount_" + file_header,
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


if __name__ == '__main__':

    import sys
    import time
    import signal
    import importlib

    import torch
    import torch.nn as nn

    from utils import *
    from callbacks import (PlotLearning, AverageMeter)
    from models.multi_column import MultiColumn
    import torchvision
    from transforms_video import *
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    config_name = args.config


    config = load_json_config(f'configs/config_{config_name}.json')

    # set column model
    file_name = config['conv_model']
    cnn_def = importlib.import_module("{}".format(file_name))

    # setup device - CPU or GPU
    device, device_ids = 'cuda', [0]
    print(" > Using device: {}".format(device))
    print(" > Active GPU ids: {}".format(device_ids))

    best_loss = float('Inf')

    if config["input_mode"] == "av":
        from data_loader_av import VideoFolder
    elif config["input_mode"] == "skvideo":
        from data_loader_skvideo import VideoFolder
    elif config["input_mode"] == "uiuc":
        from data_loader_uiuc import VideoFolder
    else:
        raise ValueError("Please provide a valid input mode")


    # set run output folder
    model_name = config["model_name"]
    output_dir = config["output_dir"]
    save_dir = os.path.join(output_dir, model_name)

    # assign Ctrl+C signal handler
    signal.signal(signal.SIGINT, ExperimentalRunCleaner(save_dir))

    # create model
    print(" > Creating model ... !")
    model = cnn_def.Model(config['num_classes']).to(device)

    # optionally resume from a checkpoint
    checkpoint_path = os.path.join(config['output_dir'],
                                   config['model_name'],
                                   'model_best.pth.tar')

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        print(" > Loaded checkpoint '{}' (epoch {})"
              .format(checkpoint_path, checkpoint['epoch']))
    else:
        print(" !#! No checkpoint found at '{}'".format(
            checkpoint_path))

    # define augmentation pipeline
    upscale_size_eval = int(config['input_spatial_size'] * config["upscale_factor_eval"])

    # Center crop videos during evaluation
    transform_eval_pre = ComposeMix([
            [Scale(upscale_size_eval), "img"],
            [torchvision.transforms.ToPILImage(), "img"],
            [torchvision.transforms.CenterCrop(config['input_spatial_size']), "img"],
             ])

    # Transforms common to train and eval sets and applied after "pre" transforms
    transform_post = ComposeMix([
            [torchvision.transforms.ToTensor(), "img"],
            [torchvision.transforms.Normalize(
                       mean=[0.485, 0.456, 0.406],  # default values for imagenet
                       std=[0.229, 0.224, 0.225]), "img"]
             ])

    val_data = VideoFolder(root=config['data_folder'],
                           json_file_input=config['json_data_val'],
                           json_file_labels=config['json_file_labels'],
                           clip_size=config['clip_size'],
                           nclips=config['nclips_val'],
                           step_size=config['step_size_val'],
                           is_val=True,
                           transform_pre=transform_eval_pre,
                           transform_post=transform_post,
                           get_item_id=True,
                           )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=False)

    model.eval()

    with torch.no_grad():
        for i, (input, target, item_id) in enumerate(val_loader):

            if config['nclips_val'] > 1:
                input_var = list(input.split(config['clip_size'], 2))
                for idx, inp in enumerate(input_var):
                    input_var[idx] = inp.to(device)
            else:
                input_var = [input.to(device)]

            # compute output and loss
            output = model(input_var)

            state = model.h[0]

            vector = np.array([s.cpu().numpy().flatten() for s in state])
            label = target.data.numpy()[0]
            print('start box_count')
            plot_box_count(vector, label, 'dir_box_count3', config_name, 3)

            break
