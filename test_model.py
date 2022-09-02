import pickle
import sys
import os
import re
from os.path import join

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from Architectures.architectures import MLP_ARCHITECTURES, ATTENTION_MLP_ARCHITECTURES, TENSORFUSION_ARCHITECTURES
from Datasets.IEMOCAP import DatasetIEMOCAP
from Models.Attention import AttentionMLP
from Models.DeepFusion import DeepFusion, WeightedCombinationClassifier, CrossModalityClassifier
from Models.Embracenet import EmbracenetPlus, Wrapper
from Models.MLP import MLP
from Models.SelfAttention import SelfAttentionClassifier
from Models.TensorFusion import TensorFusion
from Parameters.parameters import DEEP_FUSION_PARAMETERS
from Utils.dataloaders import my_collate
from Utils.datasets import FusionTransformer
from Utils.training_functions import train_deep_fusion, train_embracenet, train_mlp

""" Script command line arguments (In order):
    - method: Name of the selected fusion method
    - model_name: Name of the model. Must be the name of an architecture in the architecture file
      This name is also going to be used in the results files (Plots, csvs and pth)
    - learning_rate: Learning rate used for training. If 0 it uses the optimizers' default
    The following flags are valid
    - Weight: If passed (Pass a '-w' flag after the three mandatory arguments), use weight for samples during training. 
    - Omit modality flags: -nface (No face modality), -naudio (No audio modality), -ntext (No text modality) (Currently, only one modality can be omitted)
"""

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

ARCHITECTURES = {
    'mlp_simple': MLP_ARCHITECTURES,
    'attention_mlp': ATTENTION_MLP_ARCHITECTURES,
    'tensorfusion': TENSORFUSION_ARCHITECTURES
}
classes = {'exc':0, 'neu':1, 'sad':2, 'hap':0, 'ang':3, 'number': 4}

face_data = join('Data', 'facepreds_allsess_v4_55A.p')
audio_data = join('Data', 'audiopreds_allsess_4e_75A.p')
text_data = join('Data', 'text_preds_4e_6-A.p')

with open(face_data, 'rb') as dic:
    face_data = pickle.load(dic)
with open(audio_data, 'rb') as dic:
    audi_data = pickle.load(dic)
with open(text_data, 'rb') as dic:
    text_data = pickle.load(dic)

device = torch.device('cpu')
method = sys.argv[1]
model_name = sys.argv[2]
learning_rate = float(sys.argv[3])

weight = False
omit_modality = None

if len(sys.argv) > 4:
    if "-w" in sys.argv[4:]:
        weight = True

    #Check omit modality flags
    flag_pattern = re.compile(r"-n\w+")
    flag = flag_pattern.search(" ".join(list(map(lambda x: x.split()[0], sys.argv[4:]))))

    if flag is not None and flag.group() in ["-nface", "-naudio", "-ntext"]:
        omit_modality = flag.group()[2:]

if method not in METHOD_LIST:
    formated_method_list = ", ".join(METHOD_LIST)
    print(f"Error: Selected fusion method does not exist. Try with one of the following: {formated_method_list}")
    sys.exit(-1)

if method in ARCHITECTURES.keys():

    try:
        ARCHITECTURES[method][model_name]
    except KeyError:
        available_names = ", ".join(ARCHITECTURES[method].keys())
        print(f"Error: Specified model architechture does not exist. Try with one of the following: {available_names}")
        sys.exit(-1)

BatchSize = 32

if weight:
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor([0.8982412060301508,0.8100453172205438,1.2783075089392133,1.1495176848874598]))
else:
    loss_function = nn.CrossEntropyLoss()
    
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

if method == 'mlp_simple':

    net_structure = MLP_ARCHITECTURES[model_name]

    model = MLP(
        device=device,
        name = model_name,
        net_structure=net_structure
    )

elif method == 'attention_mlp':

    attention_net_structure = ATTENTION_MLP_ARCHITECTURES[model_name]['attention_fusion']
    multimodal_net_structure = ATTENTION_MLP_ARCHITECTURES[model_name]['multimodal_fusion']


    model_list = [
        MLP(
            device=device,
            name = f'{model_name}_{i}',
            net_structure=attention_net_structure
        )

        for i in range(3)
    ]

    model = AttentionMLP(
        number_of_modes=3, 
        device=device,
        name = model_name,
        net_structure=multimodal_net_structure,
        method_list=model_list
    )

elif method == 'deep_fusion':

    loss_parameters = DEEP_FUSION_PARAMETERS[model_name]

    model = DeepFusion(
        device=device,
        name=model_name,
        modes=3,
        modality_size=4,
        cross_modality_activation=nn.ReLU()
    )

elif method == "weighted_combination":

    model = WeightedCombinationClassifier(
        device=device,
        name=model_name,
        modes=3,
        modality_size=4
    )

elif method == "cross_modality":

    model = CrossModalityClassifier(
        device=device, 
        name=model_name, 
        modes=3,
        modality_size=4,
        activation_function=nn.ReLU()
    )

elif method == "tensorfusion":

    net_structure = TENSORFUSION_ARCHITECTURES[model_name]

    model = TensorFusion( 
        device=device,
        name = model_name,
        number_of_modes=3,
        net_structure=net_structure,
    )

elif method == "embracenet":

    model = Wrapper(
        name=model_name,
        device=device,
        n_classes=4, 
        size_list=[4,4,4], 
        embracesize=16
    )

elif method == "embracenet_plus":

    model = EmbracenetPlus(
        name=model_name,
        device=device,
        additional_layer_size=32,
        n_classes=4,
        size_list=[4, 4, 4],
        embracesize=16
    )

elif method == "self_attention":

    model = SelfAttentionClassifier(
        device=device,
        name=model_name,
        modes=3,
        modality_size=4
    )

else:

    raise Exception("No method initializated")

optimizer = Adam(model.parameters())#SGD(model.parameters(), lr=learning_rate)

if method == "deep_fusion":
    train_deep_fusion(model, learning_rate, train_dataloader, 200, loss_function, loss_parameters,optimizer, 'deep_fusion', test_dataloader)
elif method == "embracenet" or method == "embracenet_plus":
    train_embracenet(model, learning_rate, train_dataloader, 200, loss_function, optimizer, method, test_dataloader)
else:
    train_mlp(model, learning_rate, train_dataloader, 200, loss_function, optimizer, method, test_dataloader)

if learning_rate != 0:
    base_name = f'model_{model_name}_lr_{str(learning_rate).replace(".", "")}'
else:
    base_name = f'model_{model_name}_adam'

results_path = join('Results', method)
torch.save(model.state_dict(), join('Saved Models', f'{base_name}.pth'))
os.system(f'Rscript plots.R {results_path} {base_name}')
os.system(f'rm Rplots.pdf')