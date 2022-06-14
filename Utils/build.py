import torch
import torch.nn as nn


""" Given a layer's description (As specified in FILE NAME), the function builds said layer in
    pytorch
    Parameters:
        - layer_description: Description of the layer in the specified format
    Returns: A list that represents the described Layer
"""
def make_layer(layer_description):

    if layer_description['activation'] == "relu":

        return layer_description['repeat'] * [
            nn.Linear(
                in_features=layer_description['in_features'],
                out_features=layer_description['neurons']    
            ),
            nn.ReLU()
        ]

    elif layer_description['activation'] == 'softmax':

        return [
            nn.Softmax(dim=1)
        ]

    elif layer_description['activation'] == 'dropout':

        return [
            nn.Dropout(layer_description['rate'])
        ]
        
""" Given the description of a network architecture (As specified in FILE_NAME), the function builds
    said network in pytorch
    Parameters:
        - structure: Description of the network architecture in the specified format
    Returns: A nn.Sequential instance that represents the described network

"""
def build_model(architecture):

    layers = []

    for layer_description in architecture:

        layers += make_layer(layer_description)

    return nn.Sequential(*layers)

