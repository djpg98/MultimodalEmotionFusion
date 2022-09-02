import os
import sys
from os.path import join, exists

METHOD_SCRIPTS = [
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

"""
INSTRUCTIONS:
    This script requires four arguments and has an optional one:
    - method: Fusion method to be used
    - architechture: Fusion architechture to be used (Available architectures are listed at Architectures/architectures.py)
    Note: In the case of DeepFusion here goes the name of the parameter set for the loss function (Available parametere set
    are listed at Parameteres/parameters.py)
    - start: First iteration number
    - end: Last iteration number
    - weighted (Optional): If passed a Weighted loss function is used (Pass "-w" as the last argument). 
"""

method = sys.argv[1]
architechture = sys.argv[2]
iterations_start = int(sys.argv[3])
iterations_end = int(sys.argv[4])

if method not in METHOD_SCRIPTS:
    formated_method_list = ", ".join(METHOD_SCRIPTS)
    print(f"Error: Selected fusion method does not exist. Try with one of the following: {formated_method_list}")
    sys.exit(-1)

if (len(sys.argv) == 6 and sys.argv[5] == '-w'):
    directory = 'weighted'
    weight = True
else:
    directory = 'unweighted'
    weight = False

# These are the paths where results and saved models will be saved at first
original_plots_path = join('Results', method, 'Plots')
original_data_path = join('Results', method, 'Training Data')
original_saved_models_path = join('Saved Models')

# Check existence of new appropiate directories for results/saved models
plots_path = join(original_plots_path, architechture, directory)
data_path = join(original_data_path, architechture, directory)
saved_models_path = join(original_saved_models_path, method, architechture, directory)

if not exists(plots_path):
    os.makedirs(plots_path)

if not exists(data_path):
    os.makedirs(data_path)

if not exists(saved_models_path):
    os.makedirs(saved_models_path)

for i in range(iterations_start, iterations_end):

    if weight:
        status = os.system(f"python test_model.py {method} {architechture} 0 -w")
    else:
        status = os.system(f"python test_model.py {method} {architechture} 0")

    if status == -1:
        sys.exit(-1)
    else:
        #Renaming and moving
        base_name = f'model_{architechture}_adam'
        
        if weight:
            suffix = f'w_{i+1}'
        else:
            suffix = f'{i + 1}'

        os.rename(
            join(original_plots_path, f'{base_name}_acc.png'),
            join(plots_path, f'{base_name}_acc_{suffix}.png')
        )

        os.rename(
            join(original_plots_path, f'{base_name}_loss.png'),
            join(plots_path, f'{base_name}_loss_{suffix}.png')
        )

        os.rename(
            join(original_data_path, f'{base_name}_acc.csv'),
            join(data_path, f'{base_name}_acc_{suffix}.csv'),
        )

        os.rename(
            join(original_data_path, f'{base_name}_loss.csv'),
            join(data_path, f'{base_name}_loss_{suffix}.csv'),
        )

        os.rename(
            join(original_saved_models_path, f'{base_name}.pth'),
            join(saved_models_path, f'{base_name}_{suffix}.pth')
        )
