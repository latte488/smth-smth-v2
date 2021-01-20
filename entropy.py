import os
import numpy as np
from sklearn.metrics import mutual_info_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import toml
import time
from multiprocessing import Pool

def entropy(p_list):
    plogp_list = []
    for p in p_list:
        if p != 0:
            plogp_list.append(p * np.log2(p))
    return -sum(plogp_list)
    
def plot_entropy(vector, filename):
    output_count, neuron_size = vector.shape

    x = []
    entropys = []
    for i in range(neuron_size):
        x.append(i)
        fd, _ = np.histogram(vector[:, i], bins=8, range=(-1, 1))
        entropys.append(entropy(fd / output_count))

    entropy_dict = {}
    for i in range(neuron_size):
        entropy_dict[f'{i:04}'] = entropys[i]
    with open(f'entropy_{filename}.toml', 'w') as f:
        toml_str = toml.dump(entropy_dict, f)
    print(toml_str)

    # plot
    if filename is not None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(x, entropys, 0.35)
        ax.set_xlabel("Neuron")
        ax.set_ylabel("Entropy")
        #ax.set_xlim(0, 1)
        ax.set_ylim(0,  -(1/8) * np.log(1/8) * 8)
        fig.tight_layout()
        fig.savefig(f'entropy_{filename}')
        plt.close()



if __name__ == '__main__':

    import sys
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

        vector = []
        for i, (input, target, item_id) in enumerate(val_loader):

            if config['nclips_val'] > 1:
                input_var = list(input.split(config['clip_size'], 2))
                for idx, inp in enumerate(input_var):
                    input_var[idx] = inp.to(device)
            else:
                input_var = [input.to(device)]

            target1 = target // 8
            target2 = target % 8
            target1 = target1.to(device)
            target2 = target2.to(device)
            
            # compute output and loss
            output = model(input_var)

            state = model.h[0]

            for s in state:
                vector.append(s.cpu().numpy().flatten())
        vector = np.array(vector)
        print('start entropy')
        plot_entropy(vector, config_name)

