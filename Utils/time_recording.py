import time

import torch
import torch.nn as nn
from tqdm.notebook import tqdm

from Models.DeepFusion import DeepFusion
from Models.MLP import MLP
from Models.Embracenet import EmbracenetPlus, Wrapper 
from Utils.training_functions import clean_batch

#Batch size must be 1
def record_inference_time_cpu_only(model, train_dataloader, test_dataloader, run_batch_cleaner=False):

    model.eval()
    times_list = []

    with torch.no_grad():

        for batch in tqdm(train_dataloader):

            if run_batch_cleaner:
                clean_batch(batch)

            input_list = [batch['face'], batch['audio'], batch['text']]

            if isinstance(model, DeepFusion):
                start = time.time()
                model(input_list)
                inference_time = time.time() - start
            else:
                if isinstance(model, Wrapper) or isinstance(model, EmbracenetPlus):
                    batch.pop('label')
                    batch.pop('name')
                    start = time.time()
                    model(**batch)
                    inference_time = time.time() - start
                else:
                    start = time.time()
                    model(input_list)
                    inference_time = time.time() - start

            times_list.append(inference_time)

        for batch in tqdm(test_dataloader):

            if run_batch_cleaner:
                clean_batch(batch)

            input_list = [batch['face'], batch['audio'], batch['text']]

            if isinstance(model, DeepFusion):
                start = time.time()
                model(input_list)
                inference_time = time.time() - start
            else:
                if isinstance(model, Wrapper) or isinstance(model, EmbracenetPlus):
                    batch.pop('label')
                    batch.pop('name')
                    start = time.time()
                    model(**batch)
                    inference_time = time.time() - start
                else:
                    start = time.time()
                    model(input_list)
                    inference_time = time.time() - start

            times_list.append(inference_time)

    avg_time = sum(times_list)/len(times_list)
    min_time = min(times_list)
    max_time = max(times_list)

    return (avg_time, min_time, max_time)