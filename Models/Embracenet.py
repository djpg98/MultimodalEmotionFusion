"Extracted from https://github.com/juan1t0/multimodalDLforER JuanPablo Heredia (juan1t0 github)"
import torch
import torch.nn as nn

from Models.WeightedSum import WeightedSum

"Based on the embracenet proposed by Jun-Ho Choi, Jong-Seok Lee (2019)"
class EmbraceNet(nn.Module):
	def __init__(self, device, input_size_list, embracement_size=256, bypass_docking=False, additional_layer_size=0):
		super(EmbraceNet, self).__init__()

		self.device = device
		self.input_size_list = input_size_list
		self.embracement_size = embracement_size
		self.bypass_docking = bypass_docking
		if (not bypass_docking):
			for i, input_size in enumerate(input_size_list):
				if additional_layer_size > 0:
					layers = [
						nn.Linear(input_size, additional_layer_size),
						nn.Dropout(),
						nn.Linear(additional_layer_size, embracement_size),
					]
					setattr(self, 'docking_%d' % (i), nn.Sequential(*layers))
				else:
					setattr(self, 'docking_%d' % (i), nn.Linear(input_size, embracement_size))

	def forward(self, input_list, availabilities=None, selection_probabilities=None):
		# check input data
		assert len(input_list) == len(self.input_size_list)
		num_modalities = len(input_list)
		batch_size = input_list[0].shape[0]

		# docking layer
		docking_output_list = []
		if (self.bypass_docking):
			docking_output_list = input_list
		else:
			for i, input_data in enumerate(input_list):
				x = getattr(self, 'docking_%d' % (i))(input_data)
				x = nn.functional.relu(x)
				docking_output_list.append(x)

		# check availabilities
		if (availabilities is None):
			availabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
		else:
			availabilities = availabilities.float()

		# adjust selection probabilities
		if (selection_probabilities is None):
			selection_probabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)

		selection_probabilities = torch.mul(selection_probabilities, availabilities)
		probability_sum = torch.sum(selection_probabilities, dim=-1, keepdim=True)
		selection_probabilities = torch.div(selection_probabilities, probability_sum)

		# stack docking outputs
		docking_output_stack = torch.stack(docking_output_list, dim=-1)  # [batch_size, embracement_size, num_modalities]

		# embrace
		modality_indices = torch.multinomial(selection_probabilities, num_samples=self.embracement_size, replacement=True)  # [batch_size, embracement_size]
		modality_toggles = nn.functional.one_hot(modality_indices, num_classes=num_modalities).float()  # [batch_size, embracement_size, num_modalities]

		embracement_output_stack = torch.mul(docking_output_stack, modality_toggles)
		embracement_output = torch.sum(embracement_output_stack, dim=-1)  # [batch_size, embracement_size]

		return embracement_output

class Wrapper(nn.Module):
	def __init__(self, name, device, n_classes=6, size_list=[6,6,6],
				embracesize=100, bypass_docking=False):
		super(Wrapper, self).__init__()
		self.name = name
		self.NClasses = n_classes
		self.Embrace = EmbraceNet(device=device,
								input_size_list=size_list,
								embracement_size=embracesize,
								bypass_docking=bypass_docking)
		self.classifier = False
		if embracesize != n_classes:
			self.classifier = True
			# setattr(self, 'docking_%d' % (i), nn.Linear(input_size, embracement_size))
			self.clf = nn.Sequential(nn.Linear(embracesize, n_classes),
									nn.Softmax(dim=-1))

	def forward(self, face, audio, text, availabilities):
		out = self.Embrace([face, audio, text], availabilities=availabilities)
		if self.classifier:
			out = self.clf(out)
		return out

class EmbracenetPlus(nn.Module):

	def __init__(self, name, device, additional_layer_size, n_classes=6, size_list=[6,6,6],
				embracesize=100, bypass_docking=False) -> None:
		super().__init__()

		super(EmbracenetPlus, self).__init__()
		self.name = name
		self.embracenet_modalities = EmbraceNet(
			device=device,
			input_size_list=size_list, 
			embracement_size=embracesize,
			additional_layer_size=additional_layer_size
		)
		self.weighted_sum = WeightedSum(
			device=device,
			name="weighted_sum",
			number_of_modes=len(size_list)
		)
		self.embracenet_methods = Wrapper(
			name=f"{name}_classifier",
			device=device,
			n_classes=n_classes, 
			size_list=[embracesize,size_list[0],sum(size_list)], 
			embracesize=embracesize
		)

		self.methods_availabilites = torch.tensor([1., 1., 1.])

	def forward(self, face, audio, text, availabilities):

		input_list = [face, audio, text]

		embracenet_results = self.embracenet_modalities(input_list, availabilities=availabilities)
		weighted_sum_results = self.weighted_sum(input_list)
		concat_modalities = torch.cat(input_list, dim=1)

		results = self.embracenet_methods(embracenet_results, weighted_sum_results, concat_modalities, availabilities=self.methods_availabilites)

		return results