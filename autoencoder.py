import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Autoencoder(nn.Module):
    def __init__(self, input_dims, code_dims = 100):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, code_dims),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(code_dims, input_dims),
            nn.LogSigmoid())


    def forward(self, x):
        encoded_x = self.encoder(x)
        reconstructed_x = self.decoder(encoded_x)
        return reconstructed_x


def encoder_criterion(preds, labels):
	loss = nn.MSELoss()
	return loss(outputs, preds)


class Alexnet_FE(nn.module):
	def __init__(self, alexnet_model):
		super(Alexnet_FE, self).__init__()
		self.fe_model = nn.Sequential(*list(alexnet_model.children())[0][:-2])

	def forward(self, x):
		return self.fe_model(x)