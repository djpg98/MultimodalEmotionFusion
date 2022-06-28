import torch
import torch.nn as nn

""" Given an input from a single modality, it returns a quality weight
    for the modality. This weight is to be used by WeightedCombination
    when fusing modalities
"""
class WeightedMode(nn.Module):

    """ Class constructor:
        Parameters:
            - device: Device in which torch calculations will be performed
            - input_size: Number of features the inputs will have
    """
    def __init__(self, device, input_size):

        super(WeightedMode, self).__init__()

        self.device = device
        self.quality_factor = 1 / input_size
        self.linear = nn.Linear(in_features=input_size, out_features=1)
        self.activation = nn.Sigmoid()

    """ Forward propagation method
        Parameters:
            - input_list: Vector of features to be analyzed
        Returns: A quality weight for the given input
    """
    def forward(self, input_list):

        weight = self.linear(input_list)
        quality_weight = torch.mul(weight, self.quality_factor)
        rescaled_quality_weight = self.activation(quality_weight)

        return rescaled_quality_weight

""" Given a number of modalities, it assigns a quality weight to the outputs of
    each one (Using the WeightedMode class), normalizes them, sums the resulting 
    vectors into a single one, that is then used as the input for a 2-layer GRU
    It outputs a set of features
"""
class WeightedCombination(nn.Module):

    """ Class constructor
        Parameters:
            - device: Device in which torch calculations will be performed
            - name: Name of the network
            - modes: Number of modalities to be considered
            - modality_size: The number of features in each modality. All modalities must 
              have the same number of features
    """
    def __init__(self, device, name, modes, modality_size):

        super(WeightedCombination, self).__init__()

        self.device = device
        self.name = name
        self.modes = modes

        for i in range(self.modes):

            setattr(self, f'weighted_mode_{i}', WeightedMode(self.device, modality_size))

        self.softmax = nn.Softmax(dim=0)
        self.gru = nn.GRU(input_size=4, hidden_size=4, num_layers=2, batch_first=True)

    """ Forward propagation method
        Parameters:
            - input_list: List containing the results from each modality
        Returns: A set of features with the same size as the input features
        of each modality.
    """
    def forward(self, input_list):

        modality_results = []
        all_weights = [] 

        for i in range(self.modes):

            modality_weight = getattr(self, f'weighted_mode_{i}')(input_list[i])
            all_weights.append(modality_weight)

        normalized_weights = self.softmax(torch.stack(all_weights))

        for weight, input_vector in zip(normalized_weights, input_list):
            weighted_modality_output = torch.mul(weight, input_vector)
            modality_results.append(weighted_modality_output)
            
        combination = torch.sum(torch.stack(modality_results), dim=0) #dim is right
        reshaped_combination =  torch.stack(list(map(lambda y: torch.reshape(y, (1, len(input_list[2][0]))), combination)))
        result, hidden_state = self.gru(reshaped_combination)

        return result

""" A 1-layer network that obtains a set of features that represents the cross-modality 
    correlation between a specific modality (Referred in this class' description as the 
    main modality) and other considered modalities
"""
class CrossOneModality(nn.Module):

    """ Class constructor
        Parameters:
            - device: Device in which torch calculations will be performed
            - input_size: Number of features in the main modality (And all the other ones 
              considered, as all must have the same number of features)
            - output_size: Desired output size of the correlation vector obtained by this module
            - activation_function: Activation function to be used by this module
    """
    def __init__(self, device, input_size, output_size, number_of_modes, activation_function):

        super(CrossOneModality, self).__init__()

        self.device = device
        self.linear = nn.Linear(in_features=input_size*number_of_modes, out_features=output_size)
        self.activation = activation_function

    """ Forward propagation method
        Parameters:
            - mode: The index in the input_list that contains the main modality's features
            - input_list: List containing the results from each modality
        Returns: A set of features that have the dimension specified in the argument output_size
        of the constructor    
    """
    def forward(self, mode, input_list):

        correlations = []

        for i in range(len(input_list)):

            if i == mode:
                continue

            pair_correlation = torch.subtract(input_list[i], input_list[mode])
            correlations.append(pair_correlation)

        concatenated_correlations = torch.cat([correlations[i] for i in range(len(correlations))], dim=1)
        activation_input = self.linear(concatenated_correlations)
        result = self.activation(activation_input)

        return result

""" Given a number of modalities, it generates a correlation vector for each modality
    (Using the CrossOneModality class), then averages the resulting vectors
"""
class CrossModality(nn.Module):

    """ Class constructor
        Parameters:
            - device: Device in which torch calculations will be performed
            - name: Name of the network
            - modes: Number of modalities to be considered
            - modality_size: The number of features in each modality. All modalities must 
              have the same number of features
            - activation_function: Activation function to be used by the CrossOneModality Modules
            
        Warning: This implementation assumes that the output size of the CrossOneModality
        instances is modality_size, but nothing in the paper suggests that this is mandatory. So
        one could modify the constructor so the output size is not necessarily the same as the
        modality_size. Although be aware that if you do this you must also must change the in_features
        value in the initialization of the self.linear attribute in the constructor of the DeepFusion class
    """
    def __init__(self, device, name, modes, modality_size, activation_function):

        super(CrossModality, self).__init__()

        self.device = device
        self.name = name
        self.modes = modes

        for i in range(self.modes):

            setattr(self, f'cross_one_modality_{i}', CrossOneModality(
                device=self.device, 
                input_size=modality_size, 
                output_size=modality_size,
                number_of_modes=self.modes - 1,
                activation_function=activation_function
            ))

    """ Forward propagation method
        Parameters:
            - input_list: List containing the results from each modality
        Returns: An average of the correlation vectors obtained for each modality 
    """
    def forward(self, input_list):

        correlations = []

        for i in range(self.modes):

            output_vector = getattr(self, f'cross_one_modality_{i}')(i, input_list)
            correlations.append(output_vector)

        avg_correlation = torch.mean(
            torch.stack(
                tuple(correlations)
            ),
            dim=0
        )

        return avg_correlation

""" An adaptation of the DeepFusion Method proposed by Xue, Jiang, Miao, Yuan, Ma, Ma, 
    Wang, Yao, Xu, Zhang, Su (2019) for the architecture used in this project. Since the feature 
    vectors obtained from each modality are already the same size and are pretty short, some 
    steps that were taken to make all feature vectors the same size or reduce their 
    dimensionality are omitted. It returns the probability that the given sample belongs in
    a given class, for all considered classes

    Warning: Be aware the value of the in_features of self.linear is based on the assumption 
    that the output vector of the CrossModality module has the same size as the vectors obtained 
    from each individual modality. This is currently not a problem, as the CrossModality module
    was also designed with that assumption in mind. But, should that change in the future, do not 
    forget to make the appropiate changes here. I also wrote a warning in the description of said class.
    Do not ignore them
"""
class DeepFusion(nn.Module):

    """ Class constructor
        Parameters:
            - device: Device in which torch calculations will be performed
            - name: Name of the network
            - modes: Number of modalities to be considered
            - modality_size: The number of features in each modality. All modalities must 
              have the same number of features
            - cross_modality_activation: Activation function to be used by the CrossModality Modules
    """
    def __init__(self, device, name, modes, modality_size, cross_modality_activation):

        super(DeepFusion, self).__init__()

        self.device = device
        self.name = name

        self.weighted_combination_module = WeightedCombination(
            device=device, 
            name=f'{name}_weighted_c_module', 
            modes=modes,
            modality_size=modality_size
        )

        self.cross_modality_module = CrossModality(
            device=device, 
            name=f'{name}_cross_m_module', 
            modes=modes,
            modality_size=modality_size,
            activation_function=cross_modality_activation
        )

        self.linear = nn.Linear(in_features=2*modality_size, out_features=modality_size)

        self.softmax = nn.Softmax(dim=1)

    """ Forward propagation method. This forward method is specifically designed to work 
        with the results of multiple modalities as input
        Parameters:
            - input_list: List containing the results from each modality
        Returns: The probability that the given sample belongs in a given class, for all 
        considered classes
    """
    def forward(self, input_list):

        weighted_combination_result = self.weighted_combination_module(input_list)
        cross_modality_output = self.cross_modality_module(input_list)

        weighted_combination_result = torch.reshape(weighted_combination_result, cross_modality_output.shape)

        concatenated_output = torch.cat([weighted_combination_result, cross_modality_output], dim=1)
        combination_output = self.linear(concatenated_output)

        result = self.softmax(combination_output)

        return result, weighted_combination_result, combination_output

""" This is a version of the WeightedCombination Module that adds a softmax layer at the end for
    classification purposes
"""
class WeightedCombinationClassifier(nn.Module):

    """ Class constructor
        Parameters:
            - device: Device in which torch calculations will be performed
            - name: Name of the network
            - modes: Number of modalities to be considered
            - modality_size: The number of features in each modality. All modalities must 
              have the same number of features
    """
    def __init__(self, device, name, modes, modality_size):

        super(WeightedCombinationClassifier, self).__init__()

        self.device = device
        self.name = name

        self.weighted_combination_module = WeightedCombination(
            device=device, 
            name=f'{name}_weighted_c_module', 
            modes=modes,
            modality_size=modality_size
        )

        self.softmax = nn.Softmax(dim=1)

    """ Forward propagation method
        Parameters:
            - input_list: List containing the results from each modality
        Returns: A set of features with the same size as the input features
        of each modality.
    """
    def forward(self, input_list):

        weighted_combination_result = self.weighted_combination_module(input_list)

        weighted_combination_result = torch.reshape(weighted_combination_result, (weighted_combination_result.shape[0], 4))

        results = self.softmax(weighted_combination_result)

        return results

""" This is a version of the CrossModality Module that adds a softmax layer at the end for
    classification purposes
"""
class CrossModalityClassifier(nn.Module):

    """ Class constructor
        Parameters:
            - device: Device in which torch calculations will be performed
            - name: Name of the network
            - modes: Number of modalities to be considered
            - modality_size: The number of features in each modality. All modalities must 
              have the same number of features
            - activation_function: Activation function to be used by the CrossOneModality Modules
            
        Warning: This implementation assumes that the output size of the CrossOneModality
        instances is modality_size, but nothing in the paper suggests that this is mandatory. So
        one could modify the constructor so the output size is not necessarily the same as the
        modality_size. Although be aware that if you do this you must also must change the in_features
        value in the initialization of the self.linear attribute in the constructor of the DeepFusion class
    """
    def __init__(self, device, name, modes, modality_size, activation_function):

        super(CrossModalityClassifier, self).__init__()

        self.device = device
        self.name = name

        self.cross_modality_module = CrossModality(
            device=device, 
            name=f'{name}_cross_m_module', 
            modes=modes,
            modality_size=modality_size,
            activation_function=activation_function
        )

        self.softmax = nn.Softmax(dim=1)

    """ Forward propagation method
        Parameters:
            - input_list: List containing the results from each modality
        Returns: An average of the correlation vectors obtained for each modality 
    """
    def forward(self, input_list):
        cross_modality_output = self.cross_modality_module(input_list)
        result = self.softmax(cross_modality_output)

        return result

        
