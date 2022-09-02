import os
import sys
from os.path import join, exists

import torch
import torch.nn as nn

from Architectures.architectures import MLP_ARCHITECTURES, ATTENTION_MLP_ARCHITECTURES, TENSORFUSION_ARCHITECTURES
from Models.Attention import AttentionMLP
from Models.DeepFusion import DeepFusion, WeightedCombinationClassifier, CrossModalityClassifier
from Models.Embracenet import EmbracenetPlus, Wrapper
from Models.MLP import MLP
from Models.SelfAttention import SelfAttentionClassifier
from Models.TensorFusion import TensorFusion
from Parameters.parameters import DEEP_FUSION_PARAMETERS
from Utils.results_saving import save_f1, save_results_basic, save_confusion_matrix, save_time_report
from Utils.time_recording import record_inference_time_cpu_only
from Utils.training_functions import exec_model

def eval_inference_time_cpu(model, results_path, train_dataloader=None, test_dataloader=None, run_batch_cleaner=False):

    with torch.no_grad():

        avg_time, min_time, max_time = record_inference_time_cpu_only(
            model=model, 
            train_dataloader=train_dataloader, 
            test_dataloader=test_dataloader,
            run_batch_cleaner=run_batch_cleaner
        )

        save_time_report(
            model_name=model.name, 
            results_path=results_path,
            avg_time=avg_time,
            min_time=min_time,
            max_time=max_time
        )
    
def eval_f1(model, results_path, train_dataloader=None, test_dataloader=None, run_batch_cleaner=False, save_report=False):

    if train_dataloader is None and test_dataloader is None:
        raise Exception("At least one dataloader (Train or test) must be passed")

    with torch.no_grad():

        train_expected_output_list = []
        train_actual_output_list = []

        if train_dataloader is not None:

            exec_model(
                model=model,
                dataloader=train_dataloader,
                expected_output_list=train_expected_output_list,
                actual_output_list=train_actual_output_list,
                run_batch_cleaner=run_batch_cleaner
            )

        test_expected_output_list = []
        test_actual_output_list = []

        if test_dataloader is not None:

            exec_model(
                model=model,
                dataloader=test_dataloader,
                expected_output_list=test_expected_output_list,
                actual_output_list=test_actual_output_list,
                run_batch_cleaner=run_batch_cleaner
            )


        save_f1(
            model_name=model.name,
            results_path=results_path,
            train_expected=train_expected_output_list,
            train_output=train_actual_output_list,
            test_expected=test_expected_output_list,
            test_output=test_actual_output_list,
            save_report=save_report
        )

def eval_confusion_matrix(model, results_path, train_dataloader=None, test_dataloader=None, run_batch_cleaner=False, save_report=False):

        train_expected_output_list = []
        train_actual_output_list = []

        if train_dataloader is not None:

            exec_model(
                model=model,
                dataloader=train_dataloader,
                expected_output_list=train_expected_output_list,
                actual_output_list=train_actual_output_list,
                run_batch_cleaner=run_batch_cleaner
            )

        test_expected_output_list = []
        test_actual_output_list = []

        if test_dataloader is not None:

            exec_model(
                model=model,
                dataloader=test_dataloader,
                expected_output_list=test_expected_output_list,
                actual_output_list=test_actual_output_list,
                run_batch_cleaner=run_batch_cleaner
            )

        save_confusion_matrix(
            model_name=model.name,
            results_path=results_path,
            train_expected=train_expected_output_list,
            train_output=train_actual_output_list,
            test_expected=test_expected_output_list,
            test_output=test_actual_output_list,
        )


def eval_basics(model, results_path, train_dataloader=None, test_dataloader=None, run_batch_cleaner=False, weighted_loss_function=None, unweighted_loss_function=None, loss_parameters=None, save_report=False):

    if train_dataloader is None and test_dataloader is None:
        raise Exception("At least one dataloader (Train or test) must be passed")

    if "unweighted" in results_path:
        loss_function = unweighted_loss_function
    else:
        loss_function = weighted_loss_function

    with torch.no_grad():

        train_loss = []
        train_acc = []
        train_expected_output_list = []
        train_actual_output_list = []

        if train_dataloader is not None:

            exec_model(
                model=model,
                dataloader=train_dataloader,
                loss_function=loss_function,
                loss_list=train_loss,
                acc_list=train_acc,
                expected_output_list=train_expected_output_list,
                actual_output_list=train_actual_output_list,
                loss_parameters=loss_parameters,
                run_batch_cleaner=run_batch_cleaner
            )

        test_loss = []
        test_acc = []
        test_expected_output_list = []
        test_actual_output_list = []

        if test_dataloader is not None:

            exec_model(
                model=model,
                dataloader=test_dataloader,
                loss_function=loss_function,
                loss_list=test_loss,
                acc_list=test_acc,
                expected_output_list=test_expected_output_list,
                actual_output_list=test_actual_output_list,
                loss_parameters=loss_parameters,
                run_batch_cleaner=run_batch_cleaner
            )

        save_results_basic(
            model_name=model.name,
            results_path=results_path,
            train_loss=train_loss[0],
            train_acc=train_acc[0],
            test_loss=test_loss[0],
            test_acc=test_acc[0]
        )

        save_f1(
            model_name=model.name,
            results_path=results_path,
            train_expected=train_expected_output_list,
            train_output=train_actual_output_list,
            test_expected=test_expected_output_list,
            test_output=test_actual_output_list,
            save_report=save_report            
        )


def iterate_models_get_metric(metric, encoded_iter_dir, path_to_dir, method, configuration, train_dataloader=None, test_dataloader=None, omit_modality=None, **kwargs):

    available_metrics = {
        'F1': eval_f1,
        'basics': eval_basics,
        'confusion_matrix': eval_confusion_matrix,
        'inference_time_cpu': eval_inference_time_cpu,
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

            if metric =='basics':
                kwargs['kwargs']['loss_parameters'] = DEEP_FUSION_PARAMETERS[configuration]

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

        if method == 'embracenet_plus':

            model = EmbracenetPlus(
                name=model_name,
                device=device,
                additional_layer_size=32,
                n_classes=4,
                size_list=[4, 4, 4],
                embracesize=16
            )

        if method == "self_attention":

            model = SelfAttentionClassifier(
                device=device,
                name=model_name,
                modes=3,
                modality_size=4
            )

        print(file_path)

        model.load_state_dict(torch.load(file_path))

        loss_type = "unweighted" if "unweighted" in path_to_dir else "weighted"

        if omit_modality is not None:

            base_results_dir = f'Results (No {omit_modality})'
            run_batch_cleaner = True

        else:

            base_results_dir = 'Results'
            run_batch_cleaner = False

        print(base_results_dir)

        if metric ==  "confusion_matrix":
            results_path = join(base_results_dir, method, 'Confusion Matrix', configuration, loss_type)
        elif metric == "inference_time_cpu":
            results_path = join(base_results_dir, method, 'Time CPU', configuration, loss_type)
        else:
            results_path = join(base_results_dir, method, 'Training Data', configuration, loss_type)

        if not exists(results_path):
            os.makedirs(results_path)

        metric_function(model, results_path, train_dataloader, test_dataloader, run_batch_cleaner, **kwargs["kwargs"])