import os
import pickle
import sys
from os.path import join, exists

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
    'embracenet'
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
"""

classes = {'exc':0, 'neu':1, 'sad':2, 'hap':0, 'ang':3, 'number': 4}

method = sys.argv[1]
metric = sys.argv[2]

metric_kwargs = {}

if len(sys.argv) > 3:
    if "-s" in sys.argv[3:]:
        metric_kwargs['save_report'] = True
        
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

BatchSize = 32

train_dataset = DatasetIEMOCAP(classes, face_data, audi_data,
                               text_data, 'average',
                               transform=FusionTransformer(''))
test_dataset = DatasetIEMOCAP(classes, face_data, audi_data,
                              text_data, 'average', mode = 'test',
                              transform=FusionTransformer(''))

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

        iterate_models_get_metric(metric, encoded_iter_dir, weighted_path, method, train_dataloader, test_dataloader, metric_kwargs)

    unweighted_path = join(model_path, 'unweighted')

    if exists(unweighted_path):

        encoded_iter_dir = os.fsencode(unweighted_path)

        iterate_models_get_metric(metric, encoded_iter_dir, unweighted_path, method, train_dataloader, test_dataloader, metric_kwargs)