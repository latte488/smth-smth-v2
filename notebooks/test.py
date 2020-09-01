#!/usr/bin/env python
# coding: utf-8

# # Loads pre-trained model and get prediction on validation samples

# ### 1. Info
# Please provide path to the relevant config file

# In[1]:


config_file_path = "../configs/pretrained/config_model1.json"


# ### 2. Importing required modules

# In[2]:

import moviepy.editor as mpy
import os
import cv2
import sys
import importlib
import torch
import torchvision
import numpy as np

sys.path.insert(0, "../")

# imports for displaying a video an IPython cell
import io
import base64
from IPython.display import HTML


# In[3]:


from data_parser import WebmDataset
from data_loader_av import VideoFolder

from models.multi_column import MultiColumn
from transforms_video import *

from utils import load_json_config, remove_module_from_checkpoint_state_dict
from pprint import pprint


# ### 3. Loading configuration file, model definition and its path

# In[4]:


# Load config file
config = load_json_config(config_file_path)


# In[5]:


# set column model
column_cnn_def = importlib.import_module("{}".format(config['conv_model']))
model_name = config["model_name"]

print("=> Name of the model -- {}".format(model_name))

# checkpoint path to a trained model
checkpoint_path = os.path.join("../", config["output_dir"], config["model_name"], "model_best.pth.tar")
print("=> Checkpoint path --> {}".format(checkpoint_path))


# ### 3. Load model

# _Note: without cuda() for ease_

# In[6]:


model = MultiColumn(config['num_classes'], column_cnn_def.Model, int(config["column_units"]))
model.eval();


# In[7]:


print("=> loading checkpoint")
checkpoint = torch.load(checkpoint_path)
checkpoint['state_dict'] = remove_module_from_checkpoint_state_dict(
                                              checkpoint['state_dict'])
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {})"
      .format(checkpoint_path, checkpoint['epoch']))


# ### 4. Load data

# In[8]:


# Center crop videos during evaluation
transform_eval_pre = ComposeMix([
        [Scale(config['input_spatial_size']), "img"],
        [torchvision.transforms.ToPILImage(), "img"],
        [torchvision.transforms.CenterCrop(config['input_spatial_size']), "img"]
         ])

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
dict_two_way = val_data.classes_dict

raw_transform_post = ComposeMix([
        [torchvision.transforms.ToTensor(), "img"],
])

raw_data = VideoFolder(root=config['data_folder'],
                       json_file_input=config['json_data_val'],
                       json_file_labels=config['json_file_labels'],
                       clip_size=config['clip_size'],
                       nclips=config['nclips_val'],
                       step_size=config['step_size_val'],
                       is_val=True,
                       transform_pre=transform_eval_pre,
                       transform_post=raw_transform_post,
                       get_item_id=True,
                       )

for val_data_i, (input_data, target, item_id) in enumerate(val_data):
    input_data = input_data.unsqueeze(0)

    if config['nclips_val'] > 1:
        input_var = list(input_data.split(config['clip_size'], 2))
        for idx, inp in enumerate(input_var):
            input_var[idx] = torch.autograd.Variable(inp)
    else:
        input_var = [torch.autograd.Variable(input_data)]

    output = model(input_var).squeeze(0)
    output = torch.nn.functional.softmax(output, dim=0)

    pred_prob, pred_top5 = output.data.topk(5)
    pred_prob = pred_prob.numpy()
    pred_top5 = pred_top5.numpy()

    preds = []
    for i, pred in enumerate(pred_top5):
        preds.append(pred)

    if target in preds:
        print(f'{val_data_i} : pass')
        continue
    else:
        print(f'{val_data_i} : miss')

    input_data, target, item_id = raw_data[val_data_i]
    c, f, h, w = 0, 1, 2, 3
    input_data = input_data.permute(f, h, w, c)
    input_data = input_data.numpy()
    images = []
    for image in input_data:
        image = image * np.iinfo(np.uint8).max
        image = np.array(np.round(image), dtype=np.uint8)
        images.append(image)
    mpy.ImageSequenceClip(images, fps=30).write_gif(f"miss_videos/{item_id}_{dict_two_way[target].replace(' ', '_')}.gif");

