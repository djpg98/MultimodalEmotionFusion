import torch
import torch.nn as nn

from .MLP import MLP

""" This multiplies a batch of samples from one mode with the corresponding batch of samples from another
    Softmax is applied to each row of the resulting attention matrix as suggested by Choi, Song, Lee, 2018
    If one of the samples is a vector of zeros, then the attention matrix returned is the identity matrix
"""
class BimodalAttentionBlock(nn.Module):

    """ Class constructor
        Parameters:
            - device: Device in which torch calculations will be performed    
    """
    def __init__(self, device):

        super(BimodalAttentionBlock, self).__init__()

        self.device = device

    """ Forward propagation method. This forward method is specifically designed to work 
        with the results of two modalities as input
        Parameters:
            - input_list: List containing the results from the two modalities
        Returns: A list of attention matrixes for the batch
    """
    def forward(self, input1, input2):

        assert(len(input1) == len(input2))

        softmax_rows = nn.Softmax(dim=1)
        
        attention_list = []

        # For every sample in the batch, we multiply the two modalities
        # Reshape is necessary because samples come in unidimensional
        # array-like format, not exactly vectors that can be multiplied
        for sample in zip(input1, input2):
            #Check if one of the inputs is zero, if it is then append id matrix
            if not (torch.any(sample[0].bool()) or  torch.any(sample[1].bool())):

                attention_list.append(
                    softmax_rows(torch.mul(
                        torch.reshape(sample[0], (len(sample[0]), 1)),
                        torch.reshape(sample[1], (1, len(sample[1])))
                    ))                
                )

            else:

                attention_list.append(
                    torch.eye(len(sample[0]))
                )

        attention_matrix = torch.stack(attention_list, 0)
        return attention_matrix

""" For every pair of modes, it calculates the attention matrix and uses these matrixes to enrich each modes' 
    features (Similar to Choi, Song, Lee, 2018). If there are more than two modes, then for each sample in a
    given mode, we'll have more than one enriched version of that sample. These are fused with the method
    specified for the mode in method_list, so each sample in a given mode has only one enriched version
"""
class BimodalAttentionSet(nn.Module):

    """ Class constructor
        Parameters:
            - number_of_modes: Number of modalities to consider
            - device: Device in which torch calculations will be performed
            - name: Network name? I have no clue why I put this, but i must have had a good reason
            - method_list: List which contains the fusion methods that are going to be used to 
              combine the results of different enrriched samples of the same sample
    
    """
    def __init__(self, number_of_modes, device, name, method_list):

        super(BimodalAttentionSet, self).__init__()

        self.device = device
        self.name = name
        self.number_of_modes = number_of_modes

        self.combinations = [(i, j) for i in range(number_of_modes - 1) for j in range(i + 1, number_of_modes)]

        for i, j in self.combinations:
            setattr(self, f'attention_block_{i}_{j}', BimodalAttentionBlock(device))

        for i in range(self.number_of_modes):
            setattr(self, f'intra_fusion_{i}', method_list[i])

    """ Forward propagation method. This forward method is specifically designed to work 
        with the results of multiple modalities as input
        Parameters:
            - input_list: List containing the results from each modality
        Returns: An enrriched version of each sample in each mode
    """
    def forward(self, input_list):

        matrix_dict = {}

        for i, j in self.combinations:

            matrix_dict[(i, j)] = getattr(self, f'attention_block_{i}_{j}')(input_list[i], input_list[j])
            matrix_dict[(j, i)] = torch.transpose(matrix_dict[(i, j)], 1, 2) #Transposes the matrix for each sample without altering the samples order

        results = []

        for i in range(self.number_of_modes):

            modality_with_attention = []
            reshaped_input = torch.reshape(input_list[i], (input_list[i].shape[0], len(input_list[i][0]), 1))
            new_dimensions = (reshaped_input.shape[0], reshaped_input.shape[1])
            for j in range(self.number_of_modes):

                if i != j:

                    modality_with_attention.append(
                        torch.reshape(
                            torch.matmul(
                                matrix_dict[(j, i)], 
                                reshaped_input
                            ),
                            new_dimensions
                        )
                    )

            results.append(getattr(self, f'intra_fusion_{i}')(modality_with_attention)) #HERE IS THE PROBLEM

        return results

""" This method is a MLP Classifier that uses the attention mechanism implemented in BimodalAttentionSet in
    order to attemt an improvement in the classification results. It is called AttentionMLP because it uses
    a MLP to make the final classification of the data
"""
class AttentionMLP(nn.Module):

    """ Class constructor
        Parameters:
            - number_of_modes: Number of modalities to be considered
            - device: Device in which torch calculations will be performed
            - name: Network name
            - net_structure: Structure of the MLP that will be used to classify the features obtained
              by the BimodalAttentionSet
            - method_list: List which contains the fusion methods that are going to be used in the 
              BimodalAttentionSet to combine the results of different enrriched samples of the same sample
    """
    def __init__(self, number_of_modes, device, name, net_structure, method_list):

        super(AttentionMLP, self).__init__()

        self.device = device
        self.name = name
        
        self.bimodal_attention_set = BimodalAttentionSet(
            number_of_modes=number_of_modes,
            device=self.device,
            name=f'{self.name}_attention_set',
            method_list=method_list
        )

        self.classifier = MLP(
            device = self.device,
            name = f'{self.name}_MLP',
            net_structure=net_structure
        )

    """ Forward propagation method. This forward method is specifically designed to work 
        with the results of multiple modalities as input
        Parameters:
            - input_list: List containing the results from each modality
        Returns: The output of the MLP
    """
    def forward(self, input_list):

        input_with_attention = self.bimodal_attention_set(input_list)

        return self.classifier(input_with_attention)

                    



        

        


