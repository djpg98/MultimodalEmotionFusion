import csv
from os.path import join

import torch
from tqdm.notebook import tqdm


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

""" Trains a MLP.
    Parameters:
        - model: Model to be trained (Must be a pytorch model)
        - train_dataloader: Dataloader for the training data
        - loss_function: Function to be used as the loss function during training
        - optimizer: Method to be used as optimizer
        - validation_dataloader: Dataloader for the validation data
        - save: If True it calls save_results function defined above

"""
def train_mlp(model, learning_rate, train_dataloader, epochs, loss_function, optimizer, validation_dataloader=None, save=True):

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for current_epoch in range(epochs):

        model.train()

        train_sum_loss = 0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_dataloader):

            input_list = [batch['face'], batch['audio'], batch['text']]
            expected_value = torch.argmax(batch['label'], dim=-1)

            optimizer.zero_grad()
            output_value = model(input_list)
            loss = loss_function(output_value, expected_value)
            loss.backward()
            optimizer.step()

            train_sum_loss += loss.item()
            train_correct += (torch.argmax(output_value, dim=1) == expected_value).float().sum().item()
            train_total += input_list[0].shape[0]

        epoch_loss = train_sum_loss / train_total
        epoch_acc = train_correct / train_total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        if validation_dataloader is not None:

            model.eval()
            with torch.no_grad():

                val_sum_loss = 0
                val_correct = 0
                val_total = 0

                for batch in tqdm(validation_dataloader):

                    input_list = [batch['face'], batch['audio'], batch['text']]
                    expected_value = torch.argmax(batch['label'], dim=-1)

                    output_value = model(input_list)
                    loss = loss_function(output_value, expected_value)

                    val_sum_loss += loss.item()
                    val_correct += (torch.argmax(output_value, dim=1) == expected_value).float().sum().item()
                    val_total += input_list[0].shape[0]

            epoch_loss = val_sum_loss / val_total
            epoch_acc = val_correct / val_total
            val_loss.append(epoch_loss)
            val_acc.append(epoch_acc)

            print(f'epoch: {current_epoch}, train loss: {train_loss[-1]}, train acc: {train_acc[-1]}, val loss: {val_loss[-1]}, val acc: {val_acc[-1]}')

        else:

            print(f'epoch: {current_epoch}, train loss: {train_loss[-1]}, train acc: {train_acc[-1]}')

        if save:

            save_results(f'{model.name}_lr_{learning_rate}', 'mlp_simple', train_loss, train_acc, val_loss, val_acc)