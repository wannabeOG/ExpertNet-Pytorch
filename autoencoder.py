import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class autoencoder_expert(nn.Module):
	def __init__(self, in_features, code_features):
		super(autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(in_features, code_features),
			nn.Relu)

		self.decoder = nn.Linear(code_features, in_features)

	def forward():
		






def fill_in_args():
"""This function is used to fill in the arguments needed to execute the file"""

