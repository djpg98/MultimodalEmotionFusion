""" This file contains all the architectures tested for each fusion model. The explanation
    on how to specify a new architecture in this format is given in the file
    format_specification.txt in this same directory
"""

MLP = {
    'test' : [
        {
            'repeat': 1,
            'in_features': 12,
            'neurons': 3,
            'activation': 'relu'
        },

        {
            'repeat': 1,
            'in_features': 3,
            'neurons': 4,
            'activation': 'relu'
        }, 
        
        {
            'activation': 'softmax'
        }
    ]
}