import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    def __init__(self, device, input_size):
        super(SelfAttention, self).__init__()

        self.device = device
        self.linear = nn.Linear(in_features=input_size, out_features=1)
        self.tahn = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_list):

        linear_result = self.linear(input_list)
        tahn_result = self.tahn(linear_result)
        attention_weight = self.sigmoid(tahn_result)
        result =  attention_weight * input_list

        return result

class SelfAttentionClassifier(nn.Module):

    def __init__(self, device, name, modes, modality_size) -> None:
        super(SelfAttentionClassifier, self).__init__()

        self.device = device
        self.name = name
        self.modes = modes

        for i in range(self.modes):

            setattr(self, f'self_attention_mode_{i}', SelfAttention(self.device, modality_size))

        self.linear = nn.Linear(in_features=modes*modality_size, out_features=modality_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_list):

        modality_results = []

        for i in range(self.modes):

            modality_output = getattr(self, f'self_attention_mode_{i}')(input_list[i])
            modality_results.append(modality_output)

        concatenated_input = torch.cat([input_list[i] for i in range(len(input_list))], dim=1)

        linear_result = self.linear(concatenated_input)
        result = self.softmax(linear_result)

        return result

