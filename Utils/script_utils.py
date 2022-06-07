import csv
import os
import re
import sys
from os.path import join, split

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm.notebook import tqdm

from Architectures.architectures import MLP_ARCHITECTURES, ATTENTION_MLP_ARCHITECTURES, TENSORFUSION_ARCHITECTURES
from Models.Attention import AttentionMLP
from Models.DeepFusion import DeepFusion, WeightedCombinationClassifier, CrossModalityClassifier
from Models.Embracenet import Wrapper
from Models.MLP import MLP
from Models.TensorFusion import TensorFusion
from Utils.results_saving import save_f1, save_results
    
def eval_f1(model, results_path, train_dataloader=None, test_dataloader=None, save_report=False):

    if train_dataloader is None and test_dataloader is None:
        raise Exception("At least one dataloader (Train or test) must be passed")

    model.eval()

    with torch.no_grad():

        train_expected_output_list = []
        train_actual_output_list = []

        if train_dataloader is not None:

            for batch in tqdm(train_dataloader):

                input_list = [batch['face'], batch['audio'], batch['text']]
                expected_value = torch.argmax(batch['label'], dim=-1)
                
                if isinstance(model, DeepFusion):
                    output_value = model(input_list)[0]
                elif isinstance(model, Wrapper):
                    batch.pop('label')
                    batch.pop('name')
                    output_value = model(**batch)
                else:
                    output_value = model(input_list)

                train_expected_output_list += expected_value.tolist()
                train_actual_output_list += torch.argmax(output_value, dim=1).tolist()

        test_expected_output_list = []
        test_actual_output_list = []

        if test_dataloader is not None:

            for batch in tqdm(test_dataloader):

                input_list = [batch['face'], batch['audio'], batch['text']]
                expected_value = torch.argmax(batch['label'], dim=-1)

                if isinstance(model, DeepFusion):
                    output_value = model(input_list)[0]
                elif isinstance(model, Wrapper):
                    batch.pop('label')
                    batch.pop('name')
                    output_value = model(**batch)
                else:
                    output_value = model(input_list)

                test_expected_output_list += expected_value.tolist()
                test_actual_output_list += torch.argmax(output_value, dim=1).tolist()

        save_f1(
            model_name=model.name,
            results_path=results_path,
            train_expected=train_expected_output_list,
            train_output=train_actual_output_list,
            test_expected=test_expected_output_list,
            test_output=test_actual_output_list,
            save_report=save_report
        )

def eval_basics(model, results_path, train_dataloader=None, test_dataloader=None, weighted_loss_function=None, unweighted_loss_function=None):

    if train_dataloader is None and test_dataloader is None:
        raise Exception("At least one dataloader (Train or test) must be passed")

    if "unweighted" in results_path:
        loss_function = unweighted_loss_function
    else:
        loss_function = weighted_loss_function

    model.eval()

    with torch.no_grad():

        train_loss = []
        train_acc = []
        train_expected_output_list = []
        train_actual_output_list = []

        if train_dataloader is not None:

            train_sum_loss = 0
            train_correct = 0
            train_total = 0

            for batch in tqdm(train_dataloader):

                input_list = [batch['face'], batch['audio'], batch['text']]
                expected_value = torch.argmax(batch['label'], dim=-1)
                
                if isinstance(model, DeepFusion):
                    output_value = model(input_list)[0]
                elif isinstance(model, Wrapper):
                    batch.pop('label')
                    batch.pop('name')
                    output_value = model(**batch)
                    loss = loss_function(output_value, expected_value)
                else:
                    output_value = model(input_list)
                    loss = loss_function(output_value, expected_value)

                train_sum_loss += loss.item()
                train_correct += (torch.argmax(output_value, dim=1) == expected_value).float().sum().item()
                train_total += batch['face'].shape[0]

                train_expected_output_list += expected_value.tolist()
                train_actual_output_list += torch.argmax(output_value, dim=1).tolist()

            epoch_loss = train_sum_loss / train_total
            epoch_acc = train_correct / train_total
            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)

        test_loss = []
        test_acc = []
        test_expected_output_list = []
        test_actual_output_list = []

        if test_dataloader is not None:

            test_sum_loss = 0
            test_correct = 0
            test_total = 0

            for batch in tqdm(test_dataloader):

                input_list = [batch['face'], batch['audio'], batch['text']]
                expected_value = torch.argmax(batch['label'], dim=-1)

                if isinstance(model, DeepFusion):
                    output_value = model(input_list)[0]
                elif isinstance(model, Wrapper):
                    batch.pop('label')
                    batch.pop('name')
                    output_value = model(**batch)
                    loss = loss_function(output_value, expected_value)
                else:
                    output_value = model(input_list)
                    loss = loss_function(output_value, expected_value)

                test_sum_loss += loss.item()
                test_correct += (torch.argmax(output_value, dim=1) == expected_value).float().sum().item()
                test_total += batch['face'].shape[0]
                
                test_expected_output_list += expected_value.tolist()
                test_actual_output_list += torch.argmax(output_value, dim=1).tolist()

            epoch_loss = test_sum_loss / test_total
            epoch_acc = test_correct / test_total
            test_loss.append(epoch_loss)
            test_acc.append(epoch_acc)


def iterate_models_get_metric(metric, encoded_iter_dir, path_to_dir, method, configuration, train_dataloader=None, test_dataloader=None, **kwargs):

    available_metrics = {
        'F1': eval_f1
    }

    try:
        metric_function = available_metrics[metric]
    except KeyError:
        formated_method_list = ", ".join(available_metrics.keys())
        print(f"Error: A function for evaluating the selected metric does not exist. Only the following are available: {formated_method_list}")
        sys.exit(-1)

    device = torch.device('cpu')

    for file in os.listdir(encoded_iter_dir):

        file_name = os.fsdecode(file)

        if file_name[0] == "-":
            continue

        if ".pth" not in file_name:
            continue

        file_path = join(path_to_dir, file_name)

        model_name = file_name.replace(".pth", "")

        #print("HERE")
        #print(file_path)
        #print(file_name)
        #print(model_name)

        if method == "mlp_simple":

            model = MLP(
                device=device, 
                name=model_name, 
                net_structure=MLP_ARCHITECTURES[configuration]
            )

        if method == "attention_mlp":

            attention_net_structure = ATTENTION_MLP_ARCHITECTURES[configuration]['attention_fusion']
            multimodal_net_structure = ATTENTION_MLP_ARCHITECTURES[configuration]['multimodal_fusion']

            model_list = [
                MLP(
                    device=device,
                    name = f'{model_name}_{i}',
                    net_structure=attention_net_structure
                )

                for i in range(3)
            ]

            model = AttentionMLP(
                number_of_modes=3, 
                device=device,
                name = model_name,
                net_structure=multimodal_net_structure,
                method_list=model_list
            )

        if method == 'deep_fusion':

            model = DeepFusion(
                device=device,
                name=model_name,
                modes=3,
                modality_size=4,
                cross_modality_activation=nn.ReLU()
            )

        if method == 'weighted_combination':

            model = WeightedCombinationClassifier(
                device=device,
                name=model_name,
                modes=3,
                modality_size=4
            )

        if method == 'cross_modality':

            model = CrossModalityClassifier(
                device=device, 
                name=model_name, 
                modes=3,
                modality_size=4,
                activation_function=nn.ReLU()
            )

        if method == 'tensorfusion':

            model = TensorFusion( 
                device=device,
                name = model_name,
                number_of_modes=3,
                net_structure=TENSORFUSION_ARCHITECTURES[configuration],
            )

        if method == 'embracenet':

            model = Wrapper(
                name=model_name,
                device=device,
                n_classes=4, 
                size_list=[4,4,4], 
                embracesize=16
            )

        print(file_path)

        model.load_state_dict(torch.load(file_path))

        loss_type = "unweighted" if "unweighted" in path_to_dir else "weighted"

        results_path = join('Results', method, 'Training Data', configuration, loss_type)

        metric_function(model, results_path, train_dataloader, test_dataloader, **kwargs["kwargs"])