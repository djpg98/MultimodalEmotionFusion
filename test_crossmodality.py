import pickle
import sys
import os
from os.path import join

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader

from Datasets.IEMOCAP import DatasetIEMOCAP
from Models.DeepFusion import CrossModalityClassifier
from Utils.dataloaders import my_collate
from Utils.datasets import FusionTransformer
from Utils.training_functions import train_mlp

"""Script command line arguments (In order):
    - model_name: Name of the model. This name is going to be used in the results files (Plots, csvs and pth)
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

device = torch.device('cpu')
model_name = sys.argv[1]
learning_rate = float(sys.argv[2])

if (len(sys.argv) == 4 and sys.argv[3] == '-w'):
    weight = True
else:
    weight = False
    

model = CrossModalityClassifier(
    device=device, 
    name=model_name, 
    modes=3,
    modality_size=4,
    activation_function=nn.ReLU()
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

train_mlp(model, learning_rate, train_dataloader, 200, loss_function, optimizer, 'cross_modality', test_dataloader)
if learning_rate != 0:
    base_name = f'model_{model_name}_lr_{str(learning_rate).replace(".", "")}'
else:
    base_name = f'model_{model_name}_adam'

results_path = join('Results', 'cross_modality')
torch.save(model.state_dict(), join('Saved Models', f'{base_name}.pth'))
os.system(f'Rscript plots.R {results_path} {base_name}')
os.system(f'rm Rplots.pdf')