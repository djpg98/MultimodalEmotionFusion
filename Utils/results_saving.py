import csv
import re
from os.path import join

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

IDENTIFIER_SEPARATOR = re.compile(r"adam_w|adam|lr_\d+_w|lr_\d+")

""" Saves the results (loss and acc) for each epoch (Training and validation) in csv files. 
    The files are saved in the Results folder under the name results_{model_name}_{metric}.csv
    Parameters:
        - model_name: Name of the model that produced the results
        - train_loss: List containing the loss for each epoch of training
        - train_acc: List containing the acc for each epoch of training
        - val_loss: List containing the loss for each validation performed during training. May be empty
        - val_acc: List containing the acc for each validation performed during training. May be empty
"""
def save_results(model_name, prefix, train_loss, train_acc, val_loss, val_acc):

    epochs = len(train_loss)

    if len(val_loss) != 0:

        header = ['epoch', 'train', 'val']
        results_loss = zip([i for i in range(1, epochs + 1)], train_loss, val_loss)
        results_acc = zip([i for i in range(1, epochs + 1)], train_acc, val_acc)

    else:

        header = ['epoch', 'train']
        results_loss = zip([i for i in range(1, epochs + 1)], train_loss)
        results_acc = zip([i for i in range(1, epochs + 1)], train_acc)


    with open(join('Results', prefix, 'Training Data', f'model_{model_name}_loss.csv'), 'w') as loss_file:

        writer = csv.writer(loss_file, delimiter=",")
        writer.writerow(header)
        writer.writerows(results_loss)

    with open(join('Results', prefix, 'Training Data', f'model_{model_name}_acc.csv'), 'w') as acc_file:

        writer = csv.writer(acc_file, delimiter=",")
        writer.writerow(header)
        writer.writerows(results_acc)

def save_results_basic(model_name, results_path, train_loss=None, train_acc=None, test_loss=None, test_acc=None):

    header = []
    results_loss = []
    results_acc = []

    if train_loss is not None:

        header += ['train']
        results_loss.append(train_loss)
        results_acc.append(train_acc)

    if test_loss is not None:

        header += ['val']
        results_loss.append(test_loss)
        results_acc.append(test_acc)

    name_sections = model_name.rpartition(IDENTIFIER_SEPARATOR.search(model_name).group())

    if '_w' in name_sections[1]:
        identifier_sections = name_sections[1].rpartition('_w')
        prefix = name_sections[0] + identifier_sections[0]
        suffix = identifier_sections[1] + name_sections[2]
    else:
        prefix  = name_sections[0] + name_sections[1]
        suffix = name_sections[2]

    loss_file_name = prefix + '_loss' + suffix + '.csv'

    with open(join(results_path, loss_file_name), 'w') as loss_file:

        writer = csv.writer(loss_file, delimiter=',')
        writer.writerow(header)
        writer.writerow(results_loss)

    acc_file_name = prefix + '_acc' + suffix + '.csv'

    with open(join(results_path, acc_file_name), 'w') as acc_file:

        writer = csv.writer(acc_file, delimiter=',')
        writer.writerow(header)
        writer.writerow(results_acc)


def save_f1(model_name, results_path, train_expected, train_output, test_expected, test_output, save_report=False):

    labels = ["Happiness", "Neutral", "Sadness", "Anger"]
    header = []
    results_macro = []
    results_weighted = []

    if len(train_expected) != 0:
        train_summary = classification_report(train_expected, train_output, target_names=labels, output_dict=True, zero_division=0)
        header.append("train")
        results_macro.append(train_summary['macro avg']['f1-score'])
        results_weighted.append(train_summary['weighted avg']['f1-score'])

    if len(test_expected) != 0:
        test_summary = classification_report(test_expected, test_output, target_names=labels, output_dict=True, zero_division=0)
        header.append("val")
        results_macro.append(test_summary['macro avg']['f1-score'])
        results_weighted.append(test_summary['weighted avg']['f1-score'])

    name_sections = model_name.rpartition(IDENTIFIER_SEPARATOR.search(model_name).group())

    if '_w' in name_sections[1]:
        identifier_sections = name_sections[1].rpartition('_w')
        prefix = name_sections[0] + identifier_sections[0]
        suffix = identifier_sections[1] + name_sections[2]
    else:
        prefix  = name_sections[0] + name_sections[1]
        suffix = name_sections[2]

    f1_macro_file_name = prefix + "_f1_macro" + suffix + ".csv"

    with open(join(results_path, f1_macro_file_name), "w") as f1_macro_file:

        writer = csv.writer(f1_macro_file, delimiter=",")
        writer.writerow(header)
        writer.writerow(results_macro)

    f1_weighted_file_name = prefix + "_f1_weighted" + suffix + ".csv"

    with open(join(results_path, f1_weighted_file_name), "w") as f1_weighted_file:

        writer = csv.writer(f1_weighted_file, delimiter=",")
        writer.writerow(header)
        writer.writerow(results_weighted)

    if save_report:

        report_file_name = prefix + "_report" + suffix + ".txt"

        with open(join(results_path, report_file_name), "w") as report_file:
            
            if len(train_expected) != 0:
                train_report = classification_report(train_expected, train_output, target_names=labels, zero_division=0)
                report_file.write("TRAINING REPORT\n")
                report_file.write(train_report)
                report_file.write("\n\n")

            if len(test_expected) != 0:
                test_report = classification_report(test_expected, test_output, target_names=labels, zero_division=0)
                report_file.write("TEST REPORT\n")
                report_file.write(test_report)
                report_file.write("\n\n")

def save_confusion_matrix(model_name, results_path, train_expected, train_output, test_expected, test_output):

    labels = ["Happiness", "Neutral", "Sadness", "Anger"]

    name_sections = model_name.rpartition(IDENTIFIER_SEPARATOR.search(model_name).group())

    if '_w' in name_sections[1]:
        identifier_sections = name_sections[1].rpartition('_w')
        prefix = name_sections[0] + identifier_sections[0]
        suffix = identifier_sections[1] + name_sections[2]
    else:
        prefix  = name_sections[0] + name_sections[1]
        suffix = name_sections[2]

    train_file_name = prefix + "_train_cm" + suffix + ".png"

    if len(train_expected) != 0:

        cm = confusion_matrix(train_expected, train_output)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)#.from_predictions(train_expected, train_output, display_labels=labels)
        figure = cm_display.plot()
        plt.savefig(join(results_path, train_file_name))
        plt.close()

    test_file_name = prefix + "_test_cm" + suffix + ".png"

    if len(test_expected) != 0:

        cm = confusion_matrix(test_expected, test_output)
        cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)#.from_predictions(train_expected, train_output, display_labels=labels)
        figure = cm_display.plot()
        plt.savefig(join(results_path, test_file_name))
        plt.close()

def save_time_report(model_name, results_path, avg_time, min_time, max_time):

    header = ['avg', 'min', 'max']

    name_sections = model_name.rpartition(IDENTIFIER_SEPARATOR.search(model_name).group())

    if '_w' in name_sections[1]:
        identifier_sections = name_sections[1].rpartition('_w')
        prefix = name_sections[0] + identifier_sections[0]
        suffix = identifier_sections[1] + name_sections[2]
    else:
        prefix  = name_sections[0] + name_sections[1]
        suffix = name_sections[2]

    file_name = prefix + "_time_report" + suffix + ".csv"

    with open(join(results_path, file_name), "w") as report_file:

        writer = csv.writer(report_file, delimiter=",")
        writer.writerow(header)
        writer.writerow([avg_time, min_time, max_time])


