import torch
from tqdm.notebook import tqdm

from Models.DeepFusion import DeepFusion
from Models.MLP import MLP
from Models.Embracenet import EmbracenetPlus, Wrapper 
from Utils.results_saving import save_results

""" This are based on JuanPablo Heredia (juan1t0 github) code for Embracenet training
    but are customized and adapted by me"""

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
            loss = loss_function(output_value, expected_value) + loss_parameters["weighted_module"] * loss_function(output_weighted_module, expected_value) + loss_parameters["crossmodality"] * loss_function(output_crossmodality, expected_value)
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
                    loss = loss_function(output_value, expected_value) + loss_parameters["weighted_module"] * loss_function(output_weighted_module, expected_value) + loss_parameters["crossmodality"] * loss_function(output_crossmodality, expected_value)

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

##### REDOING STUFF ######

def clean_batch(batch):

    index = 0

    for availabilities in batch['availabilities']:

        if sum(availabilities) < 1.0:

            for key in batch.keys():
                if type(batch[key]) == list:
                    batch[key].pop(index)
                else:
                    batch[key] = torch.cat((batch[key][:index], batch[key][index+1:]))
        else:

            index += 1

""" This function evaluates every sample in a dataset (Associated with a dataloader) with the specified model. Whether it trains said model
    or limits itself to evaluate (Test) said model, depends on whether an optimizer is passed. If an optimizer is passed, then it will 
    automatically enter training mode, and if it isn't, it will go into test mode. This function works for the following classifer classes: 
    MLP, AttentionMLP, DeepFusion, WeightedCombinationClassifier, CrossModalityClassifier, TensorFusion, Embracenet. If your model
    recieves the input in the same format as the MLP one, and doesn't use a loss function in the style the DeepFusion, you probably 
    can use this function for said model, but please check anyways, you can always customize it to fit yor model, or write another

    Parameters:
        - model: Model used to evaluate the samples
        - dataloader: Dataloader for the dataset to be evluated
        - loss_function: Loss function used to calculate the loss in each batch (Optional)
        - loss_list: List used to store the value of the evaluation's average loss (Optional)
        - acc_list: List used to store the overall accuracy of the evaluation (Optional)
        - expected_output_list: List that contains the expected output for each datapoint in the dataset
        - actual_output_list: List that contains the output obtained by the model for each datapoint in the dataset
        - loss_parameters: Parameters used for the loss function of DeepFusion (Required only if calculating the loss of a DeepFusion model)
        - optimizer: The optimizer using during the training process. As mentioned above, whether this argument is passed or not
        determines whether the model is evaluates the dataset in train or test mode
"""
def exec_model(model, dataloader, loss_function=None, loss_list=None, acc_list=None, expected_output_list=None, actual_output_list=None, loss_parameters=None, optimizer=None, run_batch_cleaner=False):

    if (loss_function is None and loss_list is not None):
        raise Exception("To save the model's loss value you must provide both a loss function and a list to store the value")

    if loss_function is None and optimizer:
        raise Exception("Can't train model without a loss function")

    if isinstance(model, DeepFusion) and loss_function is not None and loss_parameters is None:
        raise Exception("Calculating the loss of the DeepFusion model requires that loss_parameters is not None")

    if optimizer is None:
        model.eval()
        is_train = False
    else:
        is_train = True
        model.train()

    with torch.set_grad_enabled(is_train):

        sum_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(dataloader):

            if run_batch_cleaner:
                clean_batch(batch)

            expected_value = torch.argmax(batch['label'], dim=-1)#.flatten()
            input_list = [batch['face'], batch['audio'], batch['text']]

            if optimizer is not None:
                optimizer.zero_grad()

            if isinstance(model, DeepFusion):
                output_value, output_weighted_module, output_crossmodality = model(input_list)
                if loss_function is not None:
                    loss = loss_function(output_value, expected_value) + loss_parameters["weighted_module"] * loss_function(output_weighted_module, expected_value) + loss_parameters["crossmodality"] * loss_function(output_crossmodality, expected_value)
                    sum_loss += loss.item()
            else:
                if isinstance(model, Wrapper) or isinstance(model, EmbracenetPlus):
                    batch.pop('label')
                    batch.pop('name')
                    output_value = model(**batch)
                else:
                    output_value = model(input_list)

                if loss_function is not None:
                    loss = loss_function(output_value, expected_value)
                    sum_loss += loss.item()

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            correct += (torch.argmax(output_value, dim=1) == expected_value).float().sum().item()
            total += batch['face'].shape[0]

            if expected_output_list is not None:
                expected_output_list += expected_value.tolist()
            
            if actual_output_list is not None:
                actual_output_list += torch.argmax(output_value, dim=1).tolist()

    if loss_function is not None:
        loss_value = sum_loss / total
        
    acc = correct / total

    if loss_list is not None:
        loss_list.append(loss_value)

    if acc_list is not None:
        acc_list.append(acc)

""" This is a generic training function that uses the exec_model() function as a base for its iterations. So it works with all the 
    models said function works with
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
        - loss_parameters: Parameters of the loss function being used
"""
def train_model(model, learning_rate, train_dataloader, epochs, loss_function, optimizer, fusion_type, validation_dataloader=None, save=True, loss_parameters=None):

    # These lists store the accuracy and loss for each epoch of the training
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for current_epoch in range(epochs):
        
        # When an optimizer is passed to exec_model, it executes model.train()
        # and when it is not, it executes model.eval()
        # Which means this is a training run
        exec_model(
            model=model,
            dataloader=train_dataloader,
            loss_function=loss_function,
            loss_list=train_loss,
            acc_list=train_acc,
            loss_parameters=loss_parameters,
            optimizer=optimizer
        )

        if validation_dataloader is not None:
            # And this one here is a test/validation run
            exec_model(
                model=model,
                dataloader=validation_dataloader,
                loss_function=loss_function,
                loss_list=val_loss,
                acc_list=val_acc,
                loss_parameters=loss_parameters
            )

            print(f'epoch: {current_epoch}, train loss: {train_loss[-1]}, train acc: {train_acc[-1]}, val loss: {val_loss[-1]}, val acc: {val_acc[-1]}')

        else:

            print(f'epoch: {current_epoch}, train loss: {train_loss[-1]}, train acc: {train_acc[-1]}')      

        if save:
            if learning_rate != 0:
                save_results(f'{model.name}_lr_{str(learning_rate).replace(".", "")}', fusion_type, train_loss, train_acc, val_loss, val_acc)
            else:
                save_results(f'{model.name}_adam', fusion_type, train_loss, train_acc, val_loss, val_acc)     