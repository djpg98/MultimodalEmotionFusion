import torch
from tqdm.notebook import tqdm

from Models.DeepFusion import DeepFusion
from Models.MLP import MLP
from Models.Embracenet import Wrapper 
from Utils.results_saving import save_results

""" Trains a MLP.
    Parameters:
        - model: Model to be trained (Must be a pytorch model)
        - learning_rate: The learning rate used by the algorithm. In this function, it is only used to indicate
        the learning rate that was being used when certain results were obtained. Nothing else. If it's 0, then
        the default learning rate for Adam was being used 
        - train_dataloader: Dataloader for the training data
        - epochs: Number of epochs used in training
        - loss_function: Function to be used as the loss function during training
        - optimizer: Method to be used as optimizer
        - fusion_type: Fusion method being used
        - validation_dataloader: Dataloader for the validation data
        - save: If True it calls save_results function defined above

"""
def train_mlp(model, learning_rate, train_dataloader, epochs, loss_function, optimizer, fusion_type, validation_dataloader=None, save=True):

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
            #print(expected_value)
            #print(output_value)
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
            if learning_rate != 0:
                save_results(f'{model.name}_lr_{str(learning_rate).replace(".", "")}', fusion_type, train_loss, train_acc, val_loss, val_acc)
            else:
                save_results(f'{model.name}_adam', fusion_type, train_loss, train_acc, val_loss, val_acc)

""" Trains DeepFusion.
    Parameters:
        - model: Model to be trained (Must be a pytorch model)
        - learning_rate: The learning rate used by the algorithm. In this function, it is only used to indicate
        the learning rate that was being used when certain results were obtained. Nothing else. If it's 0, then
        the default learning rate for Adam was being used
        - train_dataloader: Dataloader for the training data
        - epochs: Number of epochs used in training
        - loss_function: Function to be used as the loss function during training
        - loss_parameters: Parameters of the loss function being used
        - optimizer: Method to be used as optimizer
        - fusion_type: Fusion method being used
        - validation_dataloader: Dataloader for the validation data
        - save: If True it calls save_results function defined above

"""
def train_deep_fusion(model, learning_rate, train_dataloader, epochs, loss_function, loss_parameters, optimizer, fusion_type, validation_dataloader=None, save=True):

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
            output_value, output_weighted_module, output_crossmodality = model(input_list)
            loss = loss_function(output_value, expected_value) 
            + loss_parameters["weighted_module"] * loss_function(output_weighted_module, expected_value)
            + loss_parameters["crossmodality"] * loss_function(output_crossmodality, expected_value)
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

                    output_value, output_weighted_module, output_crossmodality = model(input_list)
                    loss = loss_function(output_value, expected_value) 
                    + loss_parameters["weighted_module"] * loss_function(output_weighted_module, expected_value)
                    + loss_parameters["crossmodality"] * loss_function(output_crossmodality, expected_value)

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
            if learning_rate != 0:
                save_results(f'{model.name}_lr_{str(learning_rate).replace(".", "")}', fusion_type, train_loss, train_acc, val_loss, val_acc)
            else:
                save_results(f'{model.name}_adam', fusion_type, train_loss, train_acc, val_loss, val_acc)

def train_embracenet(model, learning_rate, train_dataloader, epochs, loss_function, optimizer, fusion_type, validation_dataloader=None, save=True):

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

            expected_value = torch.argmax(batch['label'], dim=-1)#.flatten()
            batch.pop('label')
            batch.pop('name')
            
            optimizer.zero_grad()
            output_value = model(**batch)
            loss = loss_function(output_value, expected_value)
            loss.backward()
            optimizer.step()

            train_sum_loss += loss.item()
            train_correct += (torch.argmax(output_value, dim=1) == expected_value).float().sum().item()
            train_total += batch['face'].shape[0]

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

                    expected_value = torch.argmax(batch['label'], dim=-1)#.flatten()
                    batch.pop('label')
                    batch.pop('name')

                    output_value = model(**batch)
                    loss = loss_function(output_value, expected_value)

                    val_sum_loss += loss.item()
                    val_correct += (torch.argmax(output_value, dim=1) == expected_value).float().sum().item()
                    val_total += batch['face'].shape[0]

            epoch_loss = val_sum_loss / val_total
            epoch_acc = val_correct / val_total
            val_loss.append(epoch_loss)
            val_acc.append(epoch_acc)

            print(f'epoch: {current_epoch}, train loss: {train_loss[-1]}, train acc: {train_acc[-1]}, val loss: {val_loss[-1]}, val acc: {val_acc[-1]}')

        else:

            print(f'epoch: {current_epoch}, train loss: {train_loss[-1]}, train acc: {train_acc[-1]}')

        if save:
            if learning_rate != 0:
                save_results(f'{model.name}_lr_{str(learning_rate).replace(".", "")}', fusion_type, train_loss, train_acc, val_loss, val_acc)
            else:
                save_results(f'{model.name}_adam', fusion_type, train_loss, train_acc, val_loss, val_acc)