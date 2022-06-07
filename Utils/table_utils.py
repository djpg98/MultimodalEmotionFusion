import os
from os.path import join

import numpy as np
import pandas as pd

""" Iterates over all the results obtained for a specific model configuration. Appends 
    the train and validation (From the last epoch) accuracy and loss to some lists
    Parameters:
        - encoded_iter_dir: The directory where the results are stored. Must be previously encoded with os.fsencode()
        - path_to_dir: The path to the directory that is going to be iterated over
        - train_loss: List that stores the last epoch of training's loss for each of the training sessions in the directory
        - train_loss: List that stores the last epoch of training's accuracy for each of the training sessions in the directory
        - val_loss: List that stores the last validation's loss for each of the training sessions in the directory
        - val_acc: List that stores the last validation's accuracy for each of the training sessions in the directory
"""
def iterate_model_results(encoded_iter_dir, path_to_dir, train_loss, train_acc, val_loss, val_acc, train_f1_macro, train_f1_weighted, val_f1_macro, val_f1_weighted):

    macro_counter = 0
    weighted_counter = 0

    for file in os.listdir(encoded_iter_dir):

        file_name = os.fsdecode(file)

        if file_name[0] == "-":
            continue


        if "report" in file_name:
            continue
        
        file_path = join(path_to_dir, file_name)

        df = pd.read_csv(file_path)

        final_index = len(df) - 1

        train = df.iloc[final_index]['train']
        val = df.iloc[final_index]['val']


        if 'acc' in file_name:

            train_acc.append(train)
            val_acc.append(val)

        if 'loss' in file_name:

            train_loss.append(train)
            val_loss.append(val)

        if 'f1_macro' in file_name:

            train_f1_macro.append(train)
            val_f1_macro.append(val)

            macro_counter += 1

        if 'f1_weighted' in file_name:

            train_f1_weighted.append(train)
            val_f1_weighted.append(val)

            weighted_counter += 1

""" Calculates statistics (Avg, min, max, training repetitions) given the results of 
    a specific model's configuration (These should be a list, like the ones obtained from
    iterate_model_results) and adds them to a dictionary where the keys are the names 
    of the stats being calculated and the values are the list to which the results will
    be appended
    Parameters:
        - data_dict: Dictionary that stores the stats for each model configuration. It stores:
            * model: An identifier for this configuration
            * repetitions: Number of training repetitions
            * avg: An average for the values in the list provided
            * max: The maximum value in the list provided
            * min: The minimum value in the list provided
        - model_name: An identifier for the model's configuration
        - data_list: List with the values for which the statistics will be calculated
"""
def add_data(data_dict, model_name, data_list):

    if len(data_list) == 0:
        data_dict['model'].append(model_name)
        data_dict['repetitions'].append(0)
        data_dict['avg'].append(-1)
        data_dict['max'].append(-1)
        data_dict['min'].append(-1)
        data_dict['var'].append(-1)
    else:
        data_dict['model'].append(model_name)
        data_dict['repetitions'].append(len(data_list))
        data_dict['avg'].append(sum(data_list)/len(data_list))
        data_dict['max'].append(max(data_list))
        data_dict['min'].append(min(data_list))
        data_dict['var'].append(np.var(data_list))

""" Calculates the mean for each of the dataframe's columns
    Parameters:
        - df: Dataframe
"""
def add_columns_mean(df):

    return df.append({
        'model':'Total',
        'repetitions': df['repetitions'].mean(),
        'avg': df['avg'].mean(),
        'max': df['max'].mean(),
        'min': df['min'].mean(),
        'var': df['var'].mean()
    }, ignore_index=True)