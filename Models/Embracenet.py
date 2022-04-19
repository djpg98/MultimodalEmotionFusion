"Extracted from (CITE JUANPABLOS' GITHUB REPO)"
import torch
import torch.nn as nn

class EmbraceNet(nn.Module):
	def __init__(self, device, input_size_list, embracement_size=256, bypass_docking=False):
		super(EmbraceNet, self).__init__()

		self.device = device
		self.input_size_list = input_size_list
		self.embracement_size = embracement_size
		self.bypass_docking = bypass_docking
		if (not bypass_docking):
			for i, input_size in enumerate(input_size_list):
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

	def forward(self, face, audio, text, avs):
		out = self.Embrace([face, audio, text], availabilities=avs)
		if self.classifier:
			out = self.clf(out)
		return out