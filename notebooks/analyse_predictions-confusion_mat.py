#!/usr/bin/env python
# coding: utf-8

# ### Load config file

# In[1]:


config_file_path = "../configs/pretrained/config_model1.json"


# In[2]:


import os
import sys
import pickle
import numpy as np
from pprint import pprint

sys.path.insert(0, "../")

from utils import load_json_config

# imports for displaying a video an IPython cell
import io
import base64
from IPython.display import HTML


# In[3]:


# Load config file
config = load_json_config(config_file_path)


# ### Get predictions

# In[4]:


prediction_file_pickle_path = os.path.join('../', config['output_dir'], config['model_name'], 'test_results.pkl')


# In[5]:


with open(prediction_file_pickle_path, 'rb') as fp:
    logits_matrix, features_matrix, targets_list, item_id_list, class_to_idx = pickle.load(fp)


# In[6]:


logits_matrix.shape


# In[7]:


targets_list.shape


# ### Get confusion matrix

# In[8]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[9]:


preds = logits_matrix.argmax(axis=1)
preds.shape


# In[10]:


target_names = [class_to_idx[i] for i in range(int(len(class_to_idx) / 2))]


# In[11]:


target_names


# In[12]:


report = classification_report(targets_list, preds, target_names=target_names)


# In[13]:


confusion_mat = confusion_matrix(targets_list, preds)


# In[14]:


confusion_mat


# In[15]:


confusion_mat.shape


# In[16]:


confusions = {}
for i in range(174):
    support = np.sum(confusion_mat[i])
    for j in range(174):
        if i != j:
            confusions['{}:{}'.format(i, j)] = (confusion_mat[i, j] / support) * 100


# In[17]:


# sort confusions
import operator
confusions_sorted = sorted(confusions.items(), key=operator.itemgetter(1), reverse=True)


# ### Top-K confusions

# In[18]:


K = 20


# In[19]:


for i in range(K):
    elem = confusions_sorted[i]
    y_true = int(elem[0].split(":")[0])
    y_pred = int(elem[0].split(":")[1])
    print("{} --> {}:\t{:.2f}%".format(class_to_idx[y_true], class_to_idx[y_pred], elem[1]))


# ### Top-K pretending confusions

# In[20]:


k = 10
i = 0
while k > 0 and i < len(confusions_sorted):
    elem = confusions_sorted[i]
    y_true = class_to_idx[int(elem[0].split(":")[0])]
    y_pred = class_to_idx[int(elem[0].split(":")[1])]
    if ("Pretending" in y_true or "pretending" in y_true or
        "Pretending" in y_pred or "pretending" in y_pred):
        print("{} --> {}:\t{:.2f}%".format(y_true, y_pred, elem[1]))
        k -= 1
    i += 1
    

