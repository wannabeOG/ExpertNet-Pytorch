import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Autoencoder(nn.Module):
	def __init__(self, input_dims, code_dims = 100):
		super(autoencoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(input_dims, code_dims),
			nn.Relu())
		self.decoder = nn.Sequential(
			nn.Linear(code_dims, input_dims),
			nn.LogSigmoid())
	

	def forward(self, x):
		encoded_x = self.encoder(x)
		reconstructed_x = self.decode(x)
		return reconstructed_x



def encoder_criterion(preds, labels):
	loss = nn.MSELoss()
	return loss(outputs, preds)




