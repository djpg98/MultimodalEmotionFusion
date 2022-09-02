import os
import pickle
import sys
import re
from os.path import join, exists

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Datasets.IEMOCAP import DatasetIEMOCAP
from Utils.dataloaders import my_collate
from Utils.datasets import FusionTransformer
from Utils.script_utils import iterate_models_get_metric

METHOD_LIST = [
    'mlp_simple',
    'attention_mlp',
    'deep_fusion',
    'weighted_combination',
    'cross_modality',
    'tensorfusion',
    'embracenet',
    'embracenet_plus',
    'self_attention'
]

"""
INSTRUCTIONS:
    This script allows to obtain aditional metrics for all the saved models for a specific method. A function
    for evaluating said metric for the training and test datasets must be implemented in Utils/script_utils.py
    and made available in the iterate_models_get_metric function in said file (Look at F1 for example)
    Arguments:
        - method: Name of the method for which new metrics we need to be calculated
        - metric: Name of the metric to be calculated. This metric must be a key in the dictionary available_metrics
        in iterate_models_get_metric. Otherwise this will not work
        - save_report: (Pass a -s flag if you wish to activate this option) When calculating f1-score, additionally
        save a file with the sklearn classification_report
        - Omit modality flags: -nface (No face modality), -naudio (No audio modality), -ntext (No text modality) (Currently, 
        only one modality can be omitted)
"""

classes = {'exc':0, 'neu':1, 'sad':2, 'hap':0, 'ang':3, 'number': 4}

metric_kwargs = {}

method = sys.argv[1]
metric = sys.argv[2]

omit_modality = None

if metric == "basics":

    metric_kwargs['unweighted_loss_function'] = nn.CrossEntropyLoss()
    metric_kwargs['weighted_loss_function'] = nn.CrossEntropyLoss(weight=torch.tensor([0.8982412060301508,0.8100453172205438,1.2783075089392133,1.1495176848874598]))

if len(sys.argv) > 3:
    if "-s" in sys.argv[3:] and metric in ['F1', 'basics']:
        metric_kwargs['save_report'] = True

    #Check omit modality flags
    flag_pattern = re.compile(r"-n\w+")
    flag = flag_pattern.search(" ".join(list(map(lambda x: x.split()[0], sys.argv[3:]))))

    if flag is not None and flag.group() in ["-nface", "-naudio", "-ntext"]:
        omit_modality = flag.group()[2:]
    else: 
        omit_modality = None
        
saved_models_path = join('Saved Models', method)

if method not in METHOD_LIST:
    formated_method_list = ", ".join(METHOD_LIST)
    print(f"Error: Selected fusion method does not exist. Try with one of the following: {formated_method_list}")
    sys.exit(-1)

face_data = join('Data', 'facepreds_allsess_v4_55A.p')
audio_data = join('Data', 'audiopreds_allsess_4e_75A.p')
text_data = join('Data', 'text_preds_4e_6-A.p')

with open(face_data, 'rb') as dic:
    face_data = pickle.load(dic)
with open(audio_data, 'rb') as dic:
    audi_data = pickle.load(dic)
with open(text_data, 'rb') as dic:
    text_data = pickle.load(dic)

if metric == "inference_time_cpu":
    BatchSize = 1
else:
    BatchSize = 32

train_dataset = DatasetIEMOCAP(classes, face_data, audi_data,
                               text_data, 'average',
                               transform=FusionTransformer(''), omit_modality=omit_modality)
test_dataset = DatasetIEMOCAP(classes, face_data, audi_data,
                              text_data, 'average', mode = 'test',
                              transform=FusionTransformer(''), omit_modality=omit_modality)

train_dataloader = DataLoader(train_dataset,
                              batch_size=BatchSize, collate_fn=my_collate)
test_dataloader = DataLoader(test_dataset,
                             batch_size=BatchSize, collate_fn=my_collate)

encoded_path = os.fsencode(saved_models_path)

for model_dir in os.listdir(encoded_path):

    decoded_model = os.fsdecode(model_dir)

    model_path = join(saved_models_path, decoded_model)

    weighted_path = join(model_path, 'weighted')

    if exists(weighted_path):

        encoded_iter_dir = os.fsencode(weighted_path)

        iterate_models_get_metric(
            metric=metric, 
            encoded_iter_dir=encoded_iter_dir, 
            path_to_dir=weighted_path, 
            method=method, 
            configuration=decoded_model, 
            train_dataloader=train_dataloader, 
            test_dataloader=test_dataloader, 
            omit_modality=omit_modality,
            kwargs=metric_kwargs
        )

    unweighted_path = join(model_path, 'unweighted')

    if exists(unweighted_path):

        encoded_iter_dir = os.fsencode(unweighted_path)

        iterate_models_get_metric(
            metric=metric, 
            encoded_iter_dir=encoded_iter_dir, 
            path_to_dir=unweighted_path, 
            method=method, 
            configuration=decoded_model, 
            train_dataloader=train_dataloader, 
            test_dataloader=test_dataloader, 
            omit_modality=omit_modality,
            kwargs=metric_kwargs
        )
