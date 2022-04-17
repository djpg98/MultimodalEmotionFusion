import torch
import torch.nn as nn

from Models.MLP import MLP
from Utils.fusion_utils import generate_outer_product_equation

class TensorFusion(nn.Module):

    def __init__(self, device, name, number_of_modes, net_structure):

        super(TensorFusion, self).__init__()
        self.device = device
        self.name = name
        self.number_of_modes = number_of_modes

        self.classifier = MLP(
            device = self.device,
            name = f'{self.name}_MLP',
            net_structure=net_structure
        )

    def forward(self, input_list):

        batch_size = len(input_list[0])
        
        input_list_new_col = [torch.cat((input_list[i], torch.ones(batch_size, 1)), dim=1) for i in range(len(input_list))]
        new_input_list = []

        for i in range(batch_size):

            current_tensor = input_list_new_col[0][i]

            for j in range(1, self.number_of_modes):

                current_tensor = torch.einsum(
                    generate_outer_product_equation(len(current_tensor.shape), len(input_list_new_col[j][i].shape)),
                    current_tensor,
                    input_list_new_col[j][i]
                )

            new_input_list.append(torch.flatten(current_tensor))

        result = self.classifier(new_input_list, concat=False)

        return result