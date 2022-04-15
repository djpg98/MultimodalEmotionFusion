import os
import sys
from os.path import join, exists

import pandas as pd

from Utils.table_utils import iterate_model_results, add_data, add_columns_mean

METHOD_LIST = [
    'mlp_simple',
    'attention_mlp',
    'deep_fusion',
    'weighted_combination',
    'cross_modality'
]

weighted_loss_train = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': []}
weighted_acc_train = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': []}
weighted_loss_val = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': []}
weighted_acc_val = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': []}

unweighted_loss_train = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': []}
unweighted_acc_train = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': []}
unweighted_loss_val = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': []}
unweighted_acc_val = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': []}

method = sys.argv[1]
data_path = join('Results', method, 'Training Data')

encoded_path = os.fsencode(data_path)

for model_dir in os.listdir(encoded_path):

    decoded_model = os.fsdecode(model_dir)

    model_path = join(data_path, decoded_model)

    weighted_path = join(model_path, 'weighted')

    if exists(weighted_path):

        encoded_iter_dir = os.fsencode(weighted_path)

        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []

        iterate_model_results(encoded_iter_dir, weighted_path, train_loss, train_acc, val_loss, val_acc)

        add_data(weighted_loss_train, decoded_model, train_loss)
        add_data(weighted_loss_val, decoded_model, val_loss)
        add_data(weighted_acc_train, decoded_model, train_acc)
        add_data(weighted_acc_val, decoded_model, val_acc)

    unweighted_path = join(model_path, 'unweighted')

    if exists(unweighted_path):

        encoded_iter_dir = os.fsencode(unweighted_path)

        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []

        iterate_model_results(encoded_iter_dir, unweighted_path, train_loss, train_acc, val_loss, val_acc)

        add_data(unweighted_loss_train, decoded_model, train_loss)
        add_data(unweighted_loss_val, decoded_model, val_loss)
        add_data(unweighted_acc_train, decoded_model, train_acc)
        add_data(unweighted_acc_val, decoded_model, val_acc)


df_weighted_loss_train = pd.DataFrame(data=weighted_loss_train)
df_weighted_acc_train = pd.DataFrame(data=weighted_acc_train)
df_weighted_loss_val = pd.DataFrame(data=weighted_loss_val)
df_weighted_acc_val = pd.DataFrame(data=weighted_acc_val)

df_unweighted_loss_train = pd.DataFrame(data=unweighted_loss_train)
df_unweighted_acc_train = pd.DataFrame(data=unweighted_acc_train)
df_unweighted_loss_val = pd.DataFrame(data=unweighted_loss_val)
df_unweighted_acc_val = pd.DataFrame(data=unweighted_acc_val)

df_weighted_loss_train = add_columns_mean(df_weighted_loss_train)
df_weighted_acc_train = add_columns_mean(df_weighted_acc_train)
df_weighted_loss_val = add_columns_mean(df_weighted_loss_val)
df_weighted_acc_val = add_columns_mean(df_weighted_acc_val)

df_unweighted_loss_train = add_columns_mean(df_unweighted_loss_train)
df_unweighted_acc_train = add_columns_mean(df_unweighted_acc_train)
df_unweighted_loss_val = add_columns_mean(df_unweighted_loss_val)
df_unweighted_acc_val = add_columns_mean(df_unweighted_acc_val)

with pd.ExcelWriter(f'{method}_tables.xlsx') as writer:

    df_weighted_loss_train.to_excel(writer, sheet_name='Weighted Loss Train')
    df_weighted_acc_train.to_excel(writer, sheet_name='Weighted Acc Train')
    df_weighted_loss_val.to_excel(writer, sheet_name='Weighted Loss Val')
    df_weighted_acc_val.to_excel(writer, sheet_name='Weighted Acc Val')

    df_unweighted_loss_train.to_excel(writer, sheet_name='Unweighted Loss Train')
    df_unweighted_acc_train.to_excel(writer, sheet_name='Unweighted Acc Train')
    df_unweighted_loss_val.to_excel(writer, sheet_name='Unweighted Loss Val')
    df_unweighted_acc_val.to_excel(writer, sheet_name='Unweighted Acc Val')
