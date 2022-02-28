import os
import sys

import Architectures.architechtures as architectures

#For now, the only method available is mlp_simple
method = sys.argv[1]

if method == 'mlp_simple':
    architecture_list = architectures.MLP_ARCHITECTURES

for key in architecture_list.keys():

    os.system(f"python3 test_script.py {key} 0")