import os
import numpy as np
from sklearn.metrics import mutual_info_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import toml
import time
from multiprocessing import Pool

tau = 1

# xs.shape => (n, t)
# ys.shape => (n, t)
def transfer_entropy(xs, ys):

    count_next_x_x_y = {}
    count_x_y = {}
    count_next_x_x = {}
    count_x = {}
    n = 0

    for xseq, yseq in zip(xs, ys):
        
        for (next_x , (x,y)) in zip(xseq[tau:] , zip(xseq[:-tau] , yseq[:-tau])):
            
            count_next_x_x_y[(next_x, x, y)] = count_next_x_x_y.get((next_x, x, y), 0) + 1
            count_x_y[(x, y)] = count_x_y.get((x, y), 0) + 1
            count_next_x_x[(next_x, x)] = count_next_x_x.get((next_x, x), 0) + 1
            count_x[x] = count_x.get(x, 0) + 1
            n += 1

    transfer_entropys = []

    for next_x, x, y in count_next_x_x_y:
        
        p_next_x_x_y = count_next_x_x_y[(next_x, x, y)] / n
        p_x_y = count_x_y[(x, y)] / n
        p_next_x_x = count_next_x_x[(next_x, x)] / n
        p_x = count_x[x] / n
        
        p_next_x_l_x_y = p_next_x_x_y / p_x_y
        p_next_x_l_x = p_next_x_x / p_x

        te = p_next_x_x_y * np.log(p_next_x_l_x_y / p_next_x_l_x)
    
        transfer_entropys.append(te)


        print(f'p_next_x_x_y    : {p_next_x_x_y}')
        print(f'p_x_y           : {p_x_y}')
        print(f'p_next_x_x      : {p_next_x_x}')
        print(f'p_x             : {p_x}')
        print(f'p_next_x_l_x_y  : {p_next_x_l_x_y}')
        print(f'p_next_x_l_x    : {p_next_x_l_x}')
        print(f'transfer entropy: {te}')

    return sum(transfer_entropys)

def mutual_information(xs, ys):
    # todo: might be modified because of the difference of calculation proces

    digit_xs = []
    for x in xs:
        _, hist_edge = np.histogram(x, bins=8, range=(-1, 1))
        digit_xs.append(np.digitize(x, hist_edge))
    return transfer_entropy(digit_xs, ys)

def mutual_information_from_tuple(hidden_and_label):
    hidden, label = hidden_and_label
    return mutual_information(hidden, label)
    
def plot(xss, ys, filename):
    neuron_size = xss.shape[2]

    mutual_infos = []

    x = []
    iterable = []
    for i in range(neuron_size):
        x.append(i)
        #iterable.append((input_hidden[:, i], input_label))
        mutual_infos.append(mutual_information(xss[:, :, i], ys))

    mutual_info_dict = {}
    for i in range(neuron_size):
        mutual_info_dict[f'{i:04}'] = mutual_infos[i]
    tau_str = f'_{tau}' if tau != 1 else ''
    with open(f'transfer_entropy_{filename}{tau_str}.toml', 'w') as f:
        toml_str = toml.dump(mutual_info_dict, f)
    print(toml_str)
    
    # plot
    if filename is not None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.bar(x, mutual_infos, 0.35)
        ax.set_xlabel("neuron")
        ax.set_ylabel("mutual")
        #ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(f'transfer_entropy_{filename}')
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

        xss = []
        ys = []
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

            xseqs = []
            yseq = []
            for s in state:
                xseqs.append(s.cpu().numpy().flatten())
            for i in range(50):
                yseq.append(target1[0].cpu().numpy())
            for i in range(50):
                yseq.append(target2[0].cpu().numpy())
            
            xss.append(xseqs)
            ys.append(yseq)

        xss = np.array(xss)
        ys = np.array(ys)

        print('start transfer_entropy')
        plot(xss, ys, config_name)

