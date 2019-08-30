import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Autoencoder(nn.Module):
	def __init__(self, in_dim, code_dim = 100):
		super(autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(in_features, code_features),
			nn.Relu())
		self.decoder = nn.Linear(code_features, in_features)

	def forward(self, x):
		encoded_x = self.encoder(x)
		reconstructed_x = self.decode(x)
		return reconstructed_x





