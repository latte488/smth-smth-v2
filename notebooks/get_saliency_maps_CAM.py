#!/usr/bin/env python
# coding: utf-8

# # To visualize Class Activation Mapping (CAM)
# This notebook visualizes the **_block5_** activations of model3D_1 trained on smth-smth data v2

# ### 1. Info
# Please provide path to the relevant config file 

# In[1]:


config_file_path = "../configs/pretrained/config_model1_224.json"


# ### 1. Importing required modules

# In[2]:


import os
import cv2
import sys
import importlib
import torch
import torchvision
import numpy as np

sys.path.insert(0, "../")


# In[3]:


# Insert path to the repo https://github.com/jacobgil/pytorch-grad-cam
# I assume it is downloaded in the parent directory of this repo
path_to_grad_cam_repo = "../../pytorch-grad-cam/"
if not os.path.exists(path_to_grad_cam_repo):
    raise ValueError("Path to Grad-CAM repo not found. Please correct the path.")
sys.path.insert(0, path_to_grad_cam_repo)


# In[4]:


from data_parser import WebmDataset
from data_loader_av import VideoFolder

from models.multi_column import MultiColumn
from transforms_video import *
from grad_cam_videos import GradCamVideo

from utils import load_json_config, remove_module_from_checkpoint_state_dict
from pprint import pprint


# ### 2. Loading configuration file, model definition and its path

# In[5]:


# Load config file
config = load_json_config(config_file_path)


# In[6]:


# set column model
column_cnn_def = importlib.import_module("{}".format(config['conv_model']))
model_name = config["model_name"]

print("=> Name of the model -- {}".format(model_name))

# checkpoint path to a trained model
checkpoint_path = os.path.join("../", config["output_dir"], config["model_name"], "model_best.pth.tar")
print("=> Checkpoint path --> {}".format(checkpoint_path))


# ### 3. Load model

# In[7]:


model = MultiColumn(config['num_classes'], column_cnn_def.Model, int(config["column_units"]))
model.eval();


# In[8]:


print("=> loading checkpoint")
checkpoint = torch.load(checkpoint_path)
checkpoint['state_dict'] = remove_module_from_checkpoint_state_dict(
                                              checkpoint['state_dict'])
model.load_state_dict(checkpoint['state_dict'])
print("=> loaded checkpoint '{}' (epoch {})"
      .format(checkpoint_path, checkpoint['epoch']))


# ### 4. Load data

# In[9]:


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


# #### 4.1. Manually selecting a sample to load from the loader

# #### 5.1. Select random sample (or specify the index)

# In[10]:


selected_indx = np.random.randint(len(val_data))

# OR, if you know the video id
# video_id = '96257'
# selected_indx = [x for x in range(len(val_data)) if val_data.csv_data[x].id == video_id][0]


# In[11]:


input_data, target, item_id = val_data[selected_indx]
input_data = input_data.unsqueeze(0)
print("Id of the video sample = {}".format(item_id))
print("True label --> {} ({})".format(target, dict_two_way[target]))


# In[12]:


if config['nclips_val'] > 1:
    input_var = list(input_data.split(config['clip_size'], 2))
    for idx, inp in enumerate(input_var):
        input_var[idx] = torch.autograd.Variable(inp)
else:
    input_var = [torch.autograd.Variable(input_data)]


# ### 5. CAM Stuff
# - You can choose the class of which you want to get CAM by changing "`target_index`"
# - By default, it selects the most probable class !

# In[13]:


target_index = None

grad_cam = GradCamVideo(model=model,
                   target_layer_names=["block5"],
                   class_dict=dict_two_way,
                   use_cuda=False,
                   input_spatial_size=config["input_spatial_size"])
input_to_model = input_var[0]
mask, output = grad_cam(input_to_model, target_index)


# In[14]:


output = model(input_var).squeeze(0)
output = torch.nn.functional.softmax(output, dim=0)

# compute top5 predictions
pred_prob, pred_top5 = output.data.topk(5)
pred_prob = pred_prob.numpy()
pred_top5 = pred_top5.numpy()


# ### 6. Writing CAM images to disk

# #### 6.1. CAM visualisation

# #### 6.1. Original input data visualisation

# In[15]:


unnormalize_op = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
input_data_unnormalised = unnormalize_op(input_to_model.data.cpu().squeeze(0))
input_data_unnormalised = input_data_unnormalised.permute(1, 2, 3, 0).numpy()  # (16x224x224x3)
input_data_unnormalised = np.flip(input_data_unnormalised, 3)

output_images_folder_cam_combined = os.path.join("cam_saved_images", str(item_id), "combined")

output_images_folder_original = os.path.join("cam_saved_images", str(item_id), "original")
output_images_folder_cam = os.path.join("cam_saved_images", str(item_id), "cam")

os.makedirs(output_images_folder_cam_combined, exist_ok=True)
os.makedirs(output_images_folder_cam, exist_ok=True)
os.makedirs(output_images_folder_original, exist_ok=True)

clip_size = mask.shape[0]

RESIZE_SIZE = 224
RESIZE_FLAG = 0
SAVE_INDIVIDUALS = 1

for i in range(clip_size):
    input_data_img = input_data_unnormalised[i, :, :, :]
    heatmap = cv2.applyColorMap(np.uint8(255 * mask[i]), cv2.COLORMAP_JET)
    if RESIZE_FLAG:
        input_data_img = cv2.resize(input_data_img, (RESIZE_SIZE, RESIZE_SIZE))
        heatmap = cv2.resize(heatmap, (RESIZE_SIZE, RESIZE_SIZE))
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(input_data_img)
    cam = cam / np.max(cam)
    combined_img = np.concatenate((np.uint8(255 * input_data_img), np.uint8(255 * cam)), axis=1)
    
    cv2.imwrite(os.path.join(output_images_folder_cam_combined, "img%02d.jpg" % (i + 1)), combined_img)
    if SAVE_INDIVIDUALS:
        cv2.imwrite(os.path.join(output_images_folder_cam, "img%02d.jpg" % (i + 1)), np.uint8(255 * cam))
        cv2.imwrite(os.path.join(output_images_folder_original, "img%02d.jpg" % (i + 1)), np.uint8(255 * input_data_img))


# In[16]:


# Write text file with sample info, predictions and true labels
with open(os.path.join(output_images_folder_cam_combined, "info.txt"), "w") as fp:
    fp.write("Evaluation file used = {}\n".format(config['json_data_val']))
    fp.write("Sample index = {}\n".format(selected_indx))
    fp.write("True label --> {} ({})\n".format(target, dict_two_way[target]))
    fp.write("\n##Top-5 predicted labels##\n")
    for i, elem in enumerate(pred_top5):
        fp.write("{}: {} --> {:.2f}\n".format(i + 1, dict_two_way[elem], pred_prob[i] * 100))
    fp.write("\nPredicted index chosen = {} ({})\n".format(pred_top5[0], dict_two_way[pred_top5[0]]))


# In[17]:


path_to_combined_gif = os.path.join(output_images_folder_cam_combined, "mygif.gif")
os.system("convert -delay 10 -loop 0 {}.jpg {}".format(
                                    os.path.join(output_images_folder_cam_combined, "*"),
                                    path_to_combined_gif))


# In[18]:


# To avoid caching media(images, gifs etc.) in IPynb
import random
__counter__ = random.randint(0,2e9)


# In[19]:


from IPython.display import HTML
HTML('<img src="{}?{}">'.format(path_to_combined_gif, __counter__))


# In[20]:


print("Id of the video sample = {}".format(item_id))
print("True label --> {} ({})".format(target, dict_two_way[target]))
print("\nTop-5 Predictions:")
for i, pred in enumerate(pred_top5):
    print("Top {} :== {}. Prob := {:.2f}%".format(i + 1, dict_two_way[pred], pred_prob[i] * 100))

