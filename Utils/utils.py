import torch
import torch.nn as nn


""" Given a layer's description (As specified in FILE NAME), the function builds said layer in
    pytorch
    Parameters:
        - layer_description: Description of the layer in the specified format
    Returns: A list that represents the described Layer
"""
""" Dada una descripción de la capa (En el formato especificado en el archivo NOMBRE DEL ARCHIVO)
    construye la misma en pytorch
    Parámetros:
        - layer_description: Descripción de la capa en el formato especificado
    Retorna: Una lista que contiene los elementos que conforman la capa
"""
def make_layer(layer_description):

    if layer_description['activation'] == "relu":

        return [
            nn.Linear(
                in_features=layer_description['in_features'],
                out_features=layer_description['neurons']    
            ),
            nn.ReLU
        ]

    elif layer_description['activation'] == 'softmax':

        return [
            nn.Softmax()
        ]

""" Given the description of a network architecture (As specified in FILE_NAME), the function builds
    said network in pytorch
    Parameters:
        - structure: Description of the network architecture in the specified format
    Returns: A nn.Sequential instance that represents the described network

"""
""" Dada la especificación de una arquitectura (En el formato detallado en el archivo NOMBRE DEL
    ARCHIVO) construye la misma en pytorch
    Parámetros:
        - architecture: Descripción de la estructura de la red en el formato especificado
    Retorna: Un objeto de clase nn.Sequential que representa la red especificada  
"""
def build_model(architecture):

    layers = []

    for layer_description in architecture:

        layers += make_layer(layer_description)

    return nn.Sequential(layers)

