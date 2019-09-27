#!/usr/bin/env python
# coding: utf-8

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


class Autoencoder(nn.Module):

	def __init__(self, input_dims = 256*13*13, code_dims = 100):
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



class Alexnet_FE(nn.Module):
	def __init__(self, alexnet_model):
		super(Alexnet_FE, self).__init__()
		self.fe_model = nn.Sequential(*list(alexnet_model.children())[0][:-2])
		self.fe_model.train = False
	
	def forward(self, x):
		return self.fe_model(x)


class GeneralModelClass(nn.Module):
	def __init__(self, classes):
		super(GeneralModelClass, self).__init__()
		self.Tmodel = models.alexnet(pretrained = True)
		self.Tmodel.classifier[-1] = nn.Linear(self.Tmodel.classifier[-1].in_features, classes)

	def forward(self, x):
		return self.Tmodel(x)

