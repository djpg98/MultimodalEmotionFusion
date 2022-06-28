""" This file contains all the architectures tested for each fusion model. The explanation
    on how to specify a new architecture in this format is given in the file
    format_specification.txt in this same directory
"""

MLP_ARCHITECTURES = {
    'l3-l4' : [
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
    ],

    'l5-l4' : [
        {
            'repeat': 1,
            'in_features': 12,
            'neurons': 5,
            'activation': 'relu'
        },

        {
            'repeat': 1,
            'in_features': 5,
            'neurons': 4,
            'activation': 'relu'
        }, 
        
        {
            'activation': 'softmax'
        }
    ],

    '2l3-l4' : [
        {
            'repeat': 1,
            'in_features': 12,
            'neurons': 3,
            'activation': 'relu'
        },

        {
            'repeat': 1,
            'in_features': 3,
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
    ],

    'l7-l4' : [
        {
            'repeat': 1,
            'in_features': 12,
            'neurons': 7,
            'activation': 'relu'
        },

        {
            'repeat': 1,
            'in_features': 7,
            'neurons': 4,
            'activation': 'relu'
        }, 
        
        {
            'activation': 'softmax'
        }        
    ],

    '2l7-l4' : [
        {
            'repeat': 1,
            'in_features': 12,
            'neurons': 7,
            'activation': 'relu'
        },

        {
            'repeat': 1,
            'in_features': 7,
            'neurons': 7,
            'activation': 'relu'
        },

        {
            'repeat': 1,
            'in_features': 7,
            'neurons': 4,
            'activation': 'relu'
        }, 
        
        {
            'activation': 'softmax'
        }    
    ],

    'l7-l3-l4' : [
        {
            'repeat': 1,
            'in_features': 12,
            'neurons': 7,
            'activation': 'relu'
        },

        {
            'repeat': 1,
            'in_features': 7,
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
    ],

    'l10-l4' : [
        {
            'repeat': 1,
            'in_features': 12,
            'neurons': 10,
            'activation': 'relu'
        },

        {
            'repeat': 1,
            'in_features': 10,
            'neurons': 4,
            'activation': 'relu'
        }, 

        {
            'activation': 'softmax'
        }
    ],

    'l500-l300-l4' : [
        {
            'repeat': 1,
            'in_features': 12,
            'neurons': 500,
            'activation': 'relu'
        },

        {
            'repeat': 1,
            'in_features': 500,
            'neurons': 300,
            'activation': 'relu'
        },

        {
            'repeat': 1,
            'in_features': 300,
            'neurons': 4,
            'activation': 'relu'
        }, 

        {
            'activation': 'softmax'
        }    
    ]


}

INTRA_FUSION_MLP = {
    'l5-l4' : [
        {
            'repeat': 1,
            'in_features': 8,
            'neurons': 5,
            'activation': 'relu'            
        },

        {
            'repeat': 1,
            'in_features': 5,
            'neurons': 4,
            'activation': 'relu'
        }, 
        
        {
            'activation': 'softmax'
        }          
    ],

    'l7-l4' : [
        {
            'repeat': 1,
            'in_features': 8,
            'neurons': 7,
            'activation': 'relu'
        },

        {
            'repeat': 1,
            'in_features': 7,
            'neurons': 4,
            'activation': 'relu'
        }, 
        
        {
            'activation': 'softmax'
        }     
    ],

    'l12-l4' : [
        {
            'repeat': 1,
            'in_features': 8,
            'neurons': 12,
            'activation': 'relu'
        },

        {
            'repeat': 1,
            'in_features': 12,
            'neurons': 4,
            'activation': 'relu'
        }, 
        
        {
            'activation': 'softmax'
        } 
    ]
}

ATTENTION_MLP_ARCHITECTURES = {
    'l7a-l10f' : {
        'attention_fusion': INTRA_FUSION_MLP['l7-l4'],
        'multimodal_fusion': MLP_ARCHITECTURES['l10-l4']
    },

    'l12a-l10f' : {
        'attention_fusion': INTRA_FUSION_MLP['l12-l4'],
        'multimodal_fusion': MLP_ARCHITECTURES['l10-l4']        
    },

    'l5a-l2l7f' : {
        'attention_fusion': INTRA_FUSION_MLP['l5-l4'],
        'multimodal_fusion': MLP_ARCHITECTURES['2l7-l4']          
    }
}

TENSORFUSION_ARCHITECTURES = {
    'test' : [
        {
            'repeat': 1,
            'in_features': 125,
            'neurons': 128,
            'activation': 'relu'
        },

        {
            'repeat': 1,
            'in_features': 128,
            'neurons': 128,
            'activation': 'relu'
        }, 

        {
            'repeat': 1,
            'in_features': 128,
            'neurons': 4,
            'activation': 'relu'
        }, 
        
        {
            'activation': 'softmax'
        } 
    ],

    'dropout_test' : [
        {
            'repeat': 1,
            'in_features': 125,
            'neurons': 128,
            'activation': 'relu'
        },

        {
            'activation': 'dropout',
            'rate': 0.25
        },

        {
            'repeat': 1,
            'in_features': 128,
            'neurons': 128,
            'activation': 'relu'
        }, 

        {
            'repeat': 1,
            'in_features': 128,
            'neurons': 4,
            'activation': 'relu'
        }, 
        
        {
            'activation': 'softmax'
        } 
    ],

    '83-dropout25' : [
        {
            'repeat': 1,
            'in_features': 125,
            'neurons': 83,
            'activation': 'relu'
        },

        {
            'activation': 'dropout',
            'rate': 0.25
        },

        {
            'repeat': 1,
            'in_features': 83,
            'neurons': 83,
            'activation': 'relu'
        }, 

        {
            'repeat': 1,
            'in_features': 83,
            'neurons': 4,
            'activation': 'relu'
        }, 
        
        {
            'activation': 'softmax'
        } 
    ],

        '83-dropout05' : [
        {
            'repeat': 1,
            'in_features': 125,
            'neurons': 83,
            'activation': 'relu'
        },

        {
            'activation': 'dropout',
            'rate': 0.5
        },

        {
            'repeat': 1,
            'in_features': 83,
            'neurons': 83,
            'activation': 'relu'
        }, 

        {
            'repeat': 1,
            'in_features': 83,
            'neurons': 4,
            'activation': 'relu'
        }, 
        
        {
            'activation': 'softmax'
        } 
    ],

    '83-no-dropout' : [
        {
            'repeat': 1,
            'in_features': 125,
            'neurons': 83,
            'activation': 'relu'
        },

        {
            'repeat': 1,
            'in_features': 83,
            'neurons': 83,
            'activation': 'relu'
        }, 

        {
            'repeat': 1,
            'in_features': 83,
            'neurons': 4,
            'activation': 'relu'
        }, 
        
        {
            'activation': 'softmax'
        } 
    ]
}