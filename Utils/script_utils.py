import csv
import os
import re
import sys
from os.path import join

import torch
from sklearn.metrics import classification_report
from tqdm.notebook import tqdm

from Architectures.architectures import MLP_ARCHITECTURES, ATTENTION_MLP_ARCHITECTURES, TENSORFUSION_ARCHITECTURES
from Models.Attention import AttentionMLP
from Models.DeepFusion import DeepFusion, WeightedCombinationClassifier, CrossModalityClassifier
from Models.Embracenet import EmbraceNet
from Models.MLP import MLP
from Models.TensorFusion import TensorFusion

#TO DO: Implement iterate_model_get_metric for other models

identifier_separator = re.compile(r"adam_w|adam|lr_\d+_w|lr_\d+")

def save_f1(model_name, results_path, train_expected, train_output, test_expected, test_output, save_report=False):

    labels = ["Happiness", "Neutral", "Sadness", "Anger"]
    header = []
    results_macro = []
    results_weighted = []

    if len(train_expected) != 0:
        train_summary = classification_report(train_expected, train_output, labels=labels, output_dict=True)
        header.append("train")
        results_macro.append(train_summary['macro avg']['f1-score'])
        results_weighted.append(train_summary['weighted avg']['f1-score'])

    if len(test_expected) != 0:
        test_summary = classification_report(test_expected, test_output, labels=labels, output_dict=True)
        header.append("val")
        results_macro.append(test_summary['macro avg']['f1-score'])
        results_weighted.append(test_summary['weighted avg']['f1-score'])

    name_sections = model_name.rpartition(identifier_separator.search(model_name).group())

    with open(join(results_path, name_sections[0] + name_sections[1] + "_f1_macro" + name_sections[2] + ".csv"), "w") as f1_macro_file:

        writer = csv.writer(f1_macro_file, delimiter=",")
        writer.writerow(header)
        writer.writerow(results_macro)

    with open(join(results_path, name_sections[0] + name_sections[1] + "_f1_weighted" + name_sections[2] + ".csv"), "w") as f1_weighted_file:

        writer = csv.writer(f1_weighted_file, delimiter=",")
        writer.writerow(header)
        writer.writerow(results_weighted)

    if save_report:

        with open(join(results_path, results_path, name_sections[0] + name_sections[1] + "_report" + name_sections[2] + ".csv"), "w") as report_file:
            
            if len(train_expected) != 0:
                train_report = classification_report(train_expected, train_output, labels=labels)
                report_file.write("TRAINING REPORT\n")
                report_file.write(train_report)
                report_file.write("\n\n")

            if len(test_expected) != 0:
                test_report = classification_report(test_expected, test_output, labels=labels)
                report_file.write("TEST REPORT\n")
                report_file.write(test_report)
                report_file.write("\n\n")


    
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

                output_value = model(input_list)

                train_expected_output_list += expected_value.tolist()
                train_actual_output_list += torch.argmax(output_value, dim=1).tolist()

        test_expected_output_list = []
        test_actual_output_list = []

        if test_dataloader is not None:

            for batch in tqdm(test_dataloader):

                input_list = [batch['face'], batch['audio'], batch['text']]
                expected_value = torch.argmax(batch['label'], dim=-1)

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

def iterate_models_get_metric(metric, encoded_iter_dir, path_to_dir, method, train_dataloader=None, test_dataloader=None, **kwargs):

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

        file_path = join(path_to_dir, file_name)

        if method == "mlp_simple":

            model = MLP(
                device=device, 
                method=file_name.replace(".pth", ""), 
                net_structure=MLP_ARCHITECTURES[method]
            )

            model.load_state_dict(torch.load(file_path))

            loss_type = "weighted" if "weighted" in path_to_dir else "unweighted"

            results_path = join('Results', 'mlp_simple', method, loss_type)

            metric_function(model, results_path, train_dataloader, test_dataloader, **kwargs)