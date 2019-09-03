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




def module_extractor(model):
	layer_dict = Dict()
	for name, module in model._modules.item():
		if (name == "features"):
			for namex, modulex in module._modules.items():
				layer_dict[name] = modulex

	return layer_dict

class Alexnet_FE(nn.module):
	def __init__(self, layer_dict):
		for key, value in layer_dict:
			self.add_module(key, value)

	def forward(self, x):
		for name, module in self._modules.items():
			x = module(x)
		return x