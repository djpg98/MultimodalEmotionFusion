import os
import sys

import Architectures.architectures as architectures

METHOD_LIST = [
    'mlp_simple',
    'attention_mlp'
]

method = sys.argv[1]
iterations_start = int(sys.argv[2])
iterations_end = int(sys.argv[3])

if method not in METHOD_LIST:
    formated_method_list = ", ".join(METHOD_LIST)
    print(f"Error: Selected fusion method does not exist. Try with one of the following: {formated_method_list}")
    sys.exit(-1)

if method == 'mlp_simple':
    architecture_list = architectures.MLP_ARCHITECTURES

if method == "attention_mlp":
    architecture_list = architectures.ATTENTION_MLP_ARCHITECTURES

for key in architecture_list.keys():

    status = os.system(f"python run_n_iterations.py {method} {key} {iterations_start} {iterations_end}")

    if status == -1:
        sys.exit(-1)

    status = os.system(f"python run_n_iterations.py {method} {key} {iterations_start} {iterations_end} -w")

    if status == -1:
        sys.exit(-1)

os.system(f'python make_tables {method}')