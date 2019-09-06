import torch 
import torch.nn as nn
import torch.nn.functional as F



class Autoencoder(nn.Module):
    def __init__(self, input_dims, code_dims = 100):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dims, code_dims),
            nn.ReLU())
        self.decoder = nn.Sequential(
            nn.Linear(code_dims, input_dims),
            nn.Sigmoid())


    def forward(self, x):
        encoded_x = self.encoder(x)
        reconstructed_x = self.decoder(encoded_x)
        return reconstructed_x


class Alexnet_FE(nn.module):
	def __init__(self, alexnet_model):
		super(Alexnet_FE, self).__init__()
		self.fe_model = nn.Sequential(*list(alexnet_model.children())[0][:-2])

	def forward(self, x):
		return self.fe_model(x)


class GenModel(nn.Module):
    def __init__(self, classes):
        super(GenModel, self).__init__()
        self.model = models.alexnet(pretrained = True)
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, classes)

    def forward(self, x):
        return self.model(x)
