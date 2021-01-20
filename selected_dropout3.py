import os
import numpy as np
from sklearn.metrics import mutual_info_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import toml

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
    parser.add_argument('--drate', type=float)
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()

    config_name = args.config
    drate = args.drate

    config = load_json_config(f'configs/config_{config_name}.json')

    # set column model
    file_name = config['conv_model']
    cnn_def = importlib.import_module(f'{file_name}_select')

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

    with open(f'entropy_{model_name[:-8]}.toml', 'r') as f:
        entropy_dict = toml.load(f)

    score = [float(value) for value in entropy_dict.values()]
    score = np.array(score)
    d_rate = drate
    mode = args.mode

    # create model
    print(" > Creating model ... !")
    model = cnn_def.Model(config['num_classes'], score, d_rate, mode).to(device)

    # optionally resume from a checkpoint
    checkpoint_path = os.path.join(config['output_dir'],
                                   config['model_name'],
                                   'model_best.pth.tar')

    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'], strict=False)
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

    criterion = nn.CrossEntropyLoss().to(device)
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    logits_matrix = []
    features_matrix = []
    targets_list = []
    item_id_list = []

    end = time.time()

    with torch.no_grad():

        vector = []
        label = []
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

            loss_list = []
            for j in range(50):
                loss_list.append(criterion(output[j], target1))
            for j in range(50):
                loss_list.append(criterion(output[50 + j], target2))
            loss = torch.stack(loss_list).sum()

            # measure accuracy and record loss
            prec1_1, prec5_1 = accuracy(output[49].detach().cpu(), target1.detach().cpu(), topk=(1, 5))
            prec1_2, prec5_2 = accuracy(output[-1].detach().cpu(), target2.detach().cpu(), topk=(1, 5))
            prec1, prec5 = ((prec1_1 + prec1_2) / 2.0, (prec5_1 + prec5_2) / 2.0)
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config["print_freq"] == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        acc_filename = 'entropy_selected_dropout_acc.toml'
        with open(acc_filename, 'r') as f:
            accs = toml.load(f)

        accs[f'{config_name}_{args.mode}_{str(d_rate)}'] = top1.avg

        with open(acc_filename,'w') as f:
            toml.dump(accs, f)
