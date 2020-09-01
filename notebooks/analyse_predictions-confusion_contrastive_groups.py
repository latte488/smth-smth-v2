#!/usr/bin/env python
# coding: utf-8

# ### Annotation file paths

# In[1]:


config_file_path = "../configs/pretrained/config_model1.json"
contrastive_grp_file = "../assets/contrastive_groups_list.txt"


# In[2]:


import os
import sys
import pickle
import numpy as np
from pprint import pprint

sys.path.insert(0, "../")

from utils import load_json_config


# In[3]:


# Load config file
config = load_json_config(config_file_path)


# ### Get predictions

# In[4]:


prediction_file_pickle_path = os.path.join('../', config['output_dir'], config['model_name'], 'test_results.pkl')


# In[5]:


## some weird hacky stuff
with open(prediction_file_pickle_path, 'rb') as fp:
    try:
        logits_matrix, features_mat, targets_list, item_id_list, class_to_idx = pickle.load(fp)
    except ValueError as e:
        if str(e) != "not enough values to unpack (expected 5, got 4)":
            raise
        else:
            with open(prediction_file_pickle_path, 'rb') as fp:
                logits_matrix, targets_list, item_id_list, class_to_idx = pickle.load(fp)


# In[6]:


logits_matrix.shape


# In[7]:


targets_list.shape


# ### Fetch mapping

# In[8]:


label_to_action_grp_dict = {}
action_grp_to_label_dict = {}
action_grp_to_target_dict = {}
merge_grp_dict = {}

grp_id = 0
flag = 0

with open(contrastive_grp_file, "r", encoding='utf-8') as fp:
    for row in fp:
        if row.startswith('# '):
#             import pdb; pdb.set_trace()
            if row[2].isdigit():
                mapping = []
                for c in row[2:].strip():
                    if c.isdigit():
                        mapping.append(int(c))
                merge_grp_dict[grp_id] = mapping
            continue
        elif not row.strip():
            if flag == 0:
                flag = 1
                continue
            else:
                flag = 0
                grp_id += 1
        elif row.startswith('##'):
            break
        else:
            label = row.strip().strip(",").strip("\"").strip("'")
            label_to_action_grp_dict[class_to_idx[label]] = grp_id
            if grp_id not in action_grp_to_label_dict:
                action_grp_to_label_dict[grp_id] = [label]
                action_grp_to_target_dict[grp_id] = [class_to_idx[label]]                
            else:                
                action_grp_to_label_dict[grp_id].append(label)
                action_grp_to_target_dict[grp_id].append(class_to_idx[label])


# In[9]:


merge_grp_dict


# In[10]:


for key, value in action_grp_to_target_dict.items():
    print(key, value)
    break


# In[11]:


def get_argmax_over_predefined_targets(targets, logits):
    return logits[targets].argmax()


# ### Generate results for action groups separately (confusion mat, avg precision etc.)

# In[12]:


# action_grp_preds_and_true = {}
# for logits, target in zip(logits_matrix, targets_list):
    
#     ## if label is not present in action groups generated
#     if target not in label_to_action_grp_dict:
#         continue
#     action_grp_belongingness = label_to_action_grp_dict[target]
#     targets_action_grp = action_grp_to_target_dict[action_grp_belongingness]
    
#     pred_ag_label = get_argmax_over_predefined_targets(targets_action_grp, logits)
#     true_ag_label = targets_action_grp.index(target)
    
#     if action_grp_belongingness not in action_grp_preds_and_true:
#         action_grp_preds_and_true[action_grp_belongingness] = {"y_pred": [pred_ag_label],
#                                                                "y_true": [true_ag_label]}
#     else:
#         action_grp_preds_and_true[action_grp_belongingness]["y_pred"].append(pred_ag_label)
#         action_grp_preds_and_true[action_grp_belongingness]["y_true"].append(true_ag_label)    


# In[13]:


action_grp_preds_and_true = []
for i in range(len(action_grp_to_target_dict)):
    true_ag_label = []
    pred_ag_label = []
    
    targets_action_grp = action_grp_to_target_dict[i]
    for logits, target in zip(logits_matrix, targets_list):
        if target not in targets_action_grp:
            continue
        else:
            true_ag_label.append(targets_action_grp.index(target))
            pred_ag_label.append(get_argmax_over_predefined_targets(targets_action_grp, logits))
    
    action_grp_preds_and_true.append({'y_true': true_ag_label, 'y_pred': pred_ag_label})


# In[14]:


def merge_classes(data, target_names, mapping_list):
    mapping = {key: val for key, val in enumerate(mapping_list)}
    
    y_pred_new = []
    for val in data['y_pred']:
        y_pred_new.append(mapping[val])
    
    y_true_new = []
    for val in data['y_true']:
        y_true_new.append(mapping[val])
    
    data['y_pred'] = y_pred_new
    data['y_true'] = y_true_new
    
    target_list_done = []
    target_names_new = []
    for i, elem in enumerate(mapping_list):
        if elem not in target_list_done:
            target_names_new.append(target_names[i])
            target_list_done.append(elem)

    return data, target_names_new


# In[15]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from pprint import pprint


# In[16]:


action_grp_ap_average = 0
action_grp_ap_most_probable_average = 0

for action_grp, data in enumerate(action_grp_preds_and_true):
    
    target_names = []
    for ind in action_grp_to_target_dict[action_grp]:
        target_names.append(class_to_idx[ind])
    
    ## pre-process potential class clubbing
    if action_grp in merge_grp_dict:
        data, target_names = merge_classes(data, target_names, merge_grp_dict[action_grp])
    

    confusion_mat = confusion_matrix(data['y_true'], data['y_pred'])
    report = classification_report(data['y_true'], data['y_pred'], target_names=target_names)

    
    
    """
    Calculate metrics for each label, and find their average, weighted by support 
    (the number of true instances for each label). This alters ‘macro’ to account 
    for label imbalance; it can result in an F-score that is not between precision and recall.
    """
    
    precision_recall_fscore_support_val = precision_recall_fscore_support(
                                                    data['y_true'],
                                                    data['y_pred'],
                                                    average='weighted'
                                                    )
    accuracy_group = accuracy_score(data['y_true'], data['y_pred'])
    
    print("#" * 80)
    print("{}: ACTION GROUP".format(action_grp + 1))
    print("\nConfusion Matrix:")
    print(confusion_mat)
    
    for i, name in enumerate(target_names):
        print("{} --> {}".format(i, name))
    
#     print(report)
    ap_most_probable = np.max(np.sum(confusion_mat, axis=1)) / np.sum(confusion_mat)
    print("Average precision = {:.2f}% ({:.2f}%)\n".format(
#                         precision_recall_fscore_support_val[0] * 100,
                        accuracy_group * 100,
                        ap_most_probable * 100))
#     action_grp_ap_average += precision_recall_fscore_support_val[0]
    action_grp_ap_average += accuracy_group
    action_grp_ap_most_probable_average += ap_most_probable
    
action_grp_ap_average /= len(action_grp_preds_and_true)
action_grp_ap_most_probable_average /= len(action_grp_preds_and_true)


# In[17]:


print("Average precision among all action groups = {:.2f}%".format(action_grp_ap_average * 100))
print("Average precision (most probable class) among all action groups= {:.2f}%".format(action_grp_ap_most_probable_average * 100))

