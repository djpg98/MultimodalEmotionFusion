import torch
import torch.nn as nn

from Utils.build import build_model

""" This class represents a multilayer perceptron. It is not a generic one
    the forward method is specifically designed to concatenate the results
    from different modalities before beginning forward propagation 
"""
class MLP(nn.module):

    """ Initialization method
        Parameters:
            - device: Device in which torch calculations will be performed
            - net_structure: Structure of the desired MLP, formatted according to
            the specification in (FILE NAME)
    """
    def __init__(self, device, net_structure):
        super(MLP, self).__init__()

        self.device = device
        self.layers = build_model(net_structure)

    """ Forward propagation method. This forward method is specifically designed to work 
        with the results of multiple modalities as input
        Parameters:
            - input_list: List containing the results from each modality
        Returns: The output of the MLP
    """
    def forward(self, input_list):

        concatenated_input = torch.cat(tuple([input_list[i] for i in range(len(input_list))]), dim=1)
        return self.layers(concatenated_input)

