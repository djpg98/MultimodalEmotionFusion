import pickle
import sys
import os
from os.path import join

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from Architectures.architectures import ATTENTION_MLP_ARCHITECTURES
from Datasets.IEMOCAP import DatasetIEMOCAP
from Models.Attention import AttentionMLP
from Models.MLP import MLP
from Utils.dataloaders import my_collate
from Utils.datasets import FusionTransformer
from Utils.training_functions import train_mlp

"""Script command line arguments (In order):
    - model_name: Name of the model. Must be the name of an architecture in the architecture file
      This name is also going to be used in the results files (Plots, csvs and pth)
    - learning_rate: Learning rate used for training. If 0 it uses the optimizers' default
    - Weight: If passed (As a '-w' flag), use weight for samples during training. 
"""

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

# IMPORTANT DO NOT FORGET: FOR NOW, FOR TESTING PURPOSES, ALL MLPS USED ARE
# GOING TO BE USING THE SAME ARCHITECTURE.
device = torch.device('cpu')
model_name = sys.argv[1]
learning_rate = float(sys.argv[2])

if (len(sys.argv) == 4 and sys.argv[3] == '-w'):
    weight = True
else:
    weight = False

try:
    ATTENTION_MLP_ARCHITECTURES[model_name]
except KeyError:
    available_names = ", ".join(ATTENTION_MLP_ARCHITECTURES.keys())
    print(f"Error: Specified model architecuture does not exist. Try with one of the following: {available_names}")
    sys.exit(-1)

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

BatchSize = 32

if weight:
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor([0.8982412060301508,0.8100453172205438,1.2783075089392133,1.1495176848874598]))
else:
    loss_function = nn.CrossEntropyLoss()

optimizer = Adam(model.parameters())#SGD(model.parameters(), lr=learning_rate)

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

train_mlp(model, learning_rate, train_dataloader, 60, loss_function, optimizer, 'attention_mlp', test_dataloader)
if learning_rate != 0:
    base_name = f'model_{model_name}_lr_{str(learning_rate).replace(".", "")}'
else:
    base_name = f'model_{model_name}_adam'

results_path = join('Results', 'attention_mlp')
torch.save(model.state_dict(), join('Saved Models', f'{base_name}.pth'))
os.system(f'Rscript plots.R {results_path} {base_name}')
os.system(f'rm Rplots.pdf')

