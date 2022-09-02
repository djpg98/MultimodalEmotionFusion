import torch
import torch.nn as nn

class WeightedSum(nn.Module):

    def __init__(self, device, name, number_of_modes):
        super(WeightedSum, self).__init__()
        self.device = device
        self.name = name
        self.number_of_modes = number_of_modes

        self.linear = nn.Linear(in_features=number_of_modes, out_features=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_list):

        stacked_input = torch.stack([input_list[i] for i in range(len(input_list))], dim=-1)
        weighted_sum = self.linear(stacked_input)
        result = self.softmax(weighted_sum)

        result_reshaped = torch.reshape(result, (result.shape[0], result.shape[1]))

        return result_reshaped
