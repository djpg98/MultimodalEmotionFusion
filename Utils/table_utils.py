import os
from os.path import join

import pandas as pd

def iterate_model_results(encoded_iter_dir, path_to_dir, train_loss, train_acc, val_loss, val_acc):

    for file in os.listdir(encoded_iter_dir):

        file_name = os.fsdecode(file)

        if file_name[0] == "-":
            continue
        
        file_path = join(path_to_dir, file_name)

        df = pd.read_csv(file_path)

        final_index = len(df) - 1

        train = df.iloc[final_index]['train']
        val = df.iloc[final_index]['val']


        if 'acc' in file_name:

            train_acc.append(train)
            val_acc.append(val)

        else:

            train_loss.append(train)
            val_loss.append(val)

def add_data(data_dict, model_name, data_list):

    if len(data_list) == 0:
        data_dict['model'].append(model_name)
        data_dict['repetitions'].append(0)
        data_dict['avg'].append(-1)
        data_dict['max'].append(-1)
        data_dict['min'].append(-1)
    else:
        data_dict['model'].append(model_name)
        data_dict['repetitions'].append(len(data_list))
        data_dict['avg'].append(sum(data_list)/len(data_list))
        data_dict['max'].append(max(data_list))
        data_dict['min'].append(min(data_list))

def add_columns_mean(df):

    return df.append({
        'model':'Total',
        'repetitions': df['repetitions'].mean(),
        'avg': df['avg'].mean(),
        'max': df['max'].mean(),
        'min': df['min'].mean()
    }, ignore_index=True)