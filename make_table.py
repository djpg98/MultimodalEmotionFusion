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
    'cross_modality',
    'tensorfusion',
    'embracenet'
]

weighted_loss_train = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': [], 'var': []}
weighted_acc_train = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': [], 'var': []}
weighted_loss_val = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': [], 'var': []}
weighted_acc_val = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': [], 'var': []}
weighted_f1_macro_train = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': [], 'var': []}
weighted_f1_weighted_train = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': [], 'var': []}
weighted_f1_macro_val = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': [], 'var': []}
weighted_f1_weighted_val = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': [], 'var': []}

weighted_metrics = [weighted_loss_train, weighted_acc_train, weighted_loss_val, weighted_acc_val, 
                    weighted_f1_macro_train, weighted_f1_weighted_train, weighted_f1_macro_val, weighted_f1_weighted_val]

unweighted_loss_train = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': [], 'var': []}
unweighted_acc_train = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': [], 'var': []}
unweighted_loss_val = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': [], 'var': []}
unweighted_acc_val = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': [], 'var': []}
unweighted_f1_macro_train = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': [], 'var': []}
unweighted_f1_weighted_train = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': [], 'var': []}
unweighted_f1_macro_val = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': [], 'var': []}
unweighted_f1_weighted_val = {'model': [], 'repetitions': [], 'avg': [], 'min': [], 'max': [], 'var': []}

unweighted_metrics = [unweighted_loss_train, unweighted_acc_train, unweighted_loss_val, unweighted_acc_val, 
                      unweighted_f1_macro_train, unweighted_f1_weighted_train, unweighted_f1_macro_val, unweighted_f1_weighted_val]

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
        train_f1_macro = []
        train_f1_weighted = []
        val_f1_macro = []
        val_f1_weighted = []

        weighted_lists = [train_loss, train_acc, val_loss, val_acc, train_f1_macro, train_f1_weighted, val_f1_macro,val_f1_weighted]

        iterate_model_results(encoded_iter_dir, weighted_path, train_loss, train_acc, val_loss, val_acc, train_f1_macro, train_f1_weighted, val_f1_macro, val_f1_weighted)

        for stats_row, data in zip(weighted_metrics, weighted_lists):
            add_data(stats_row, decoded_model, data)

    unweighted_path = join(model_path, 'unweighted')

    if exists(unweighted_path):

        encoded_iter_dir = os.fsencode(unweighted_path)

        train_loss = []
        train_acc = []
        val_loss = []
        val_acc = []
        train_f1_macro = []
        train_f1_weighted = []
        val_f1_macro = []
        val_f1_weighted = []

        unweighted_lists = [train_loss, train_acc, val_loss, val_acc, train_f1_macro, train_f1_weighted, val_f1_macro,val_f1_weighted]

        iterate_model_results(encoded_iter_dir, unweighted_path, train_loss, train_acc, val_loss, val_acc, train_f1_macro, train_f1_weighted, val_f1_macro, val_f1_weighted)

        for stats_row, data in zip(unweighted_metrics, unweighted_lists):
            add_data(stats_row, decoded_model, data)

df_names = ['loss_train', 'acc_train', 'loss_val', 'acc_val', 'f1_macro_train', 'f1_weighted_train', 'f1_macro_val', 'f1_weighted_val']

df_weighted = {}

for name, data in zip(df_names, weighted_metrics):

    df_weighted[name] = pd.DataFrame(data=data)

df_unweighted = {}

for name, data in zip(df_names, unweighted_metrics):

    df_unweighted[name] = pd.DataFrame(data=data)

for key, data in df_weighted.items():

    df_weighted[key] = add_columns_mean(df_weighted[key])

for key, data in df_unweighted.items():

    df_unweighted[key] = add_columns_mean(df_unweighted[key])

with pd.ExcelWriter(join('Tables', f'{method}_tables_train.xlsx')) as writer:

    df_weighted['loss_train'].to_excel(writer, sheet_name='Weighted Loss Train')
    df_weighted['acc_train'].to_excel(writer, sheet_name='Weighted Acc Train')
    df_weighted['f1_macro_train'].to_excel(writer, sheet_name='Weighted F1 Macro Train')
    df_weighted['f1_weighted_train'].to_excel(writer, sheet_name='Weighted F1 Weighted Train')

    df_unweighted['loss_train'].to_excel(writer, sheet_name='Unweighted Loss Train')
    df_unweighted['acc_train'].to_excel(writer, sheet_name='Unweighted Acc Train')
    df_unweighted['f1_macro_train'].to_excel(writer, sheet_name='Unweighted F1 Macro Train')
    df_unweighted['f1_weighted_train'].to_excel(writer, sheet_name='Unweighted F1 Weighted Train')

with pd.ExcelWriter(join('Tables', f'{method}_tables_val.xlsx')) as writer: 

    df_weighted['loss_val'].to_excel(writer, sheet_name='Weighted Loss Val')
    df_weighted['acc_val'].to_excel(writer, sheet_name='Weighted Acc Val')
    df_weighted['f1_macro_val'].to_excel(writer, sheet_name='Weighted F1 Macro Val')
    df_weighted['f1_weighted_val'].to_excel(writer, sheet_name='Weighted F1 Weighted Val')

    df_unweighted['loss_val'].to_excel(writer, sheet_name='Unweighted Loss Val')
    df_unweighted['acc_val'].to_excel(writer, sheet_name='Unweighted Acc Val')
    df_unweighted['f1_macro_val'].to_excel(writer, sheet_name='Uneighted F1 Macro Val')
    df_unweighted['f1_weighted_val'].to_excel(writer, sheet_name='Unweighted F1 Weighted Val')
