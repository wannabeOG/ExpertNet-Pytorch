#!/usr/bin/env python
# coding: utf-8


"""
Module containing the realization of all the models that are to be used un t 
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class Autoencoder(nn.Module):
	"""
	The class defines the autoencoder model which takes in the features from the last convolutional layer of the 
	Alexnet model. The default value for the input_dims is 256*13*13.
	"""
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
	"""
	Create a feature extractor model from an Alexnet architecture, that is used to train the autoencoder model
	and get the most related model whilst training a new task in a sequence
	"""
	def __init__(self, alexnet_model):
		super(Alexnet_FE, self).__init__()
		self.fe_model = nn.Sequential(*list(alexnet_model.children())[0][:-2])
		self.fe_model.train = False
	
	def forward(self, x):
		return self.fe_model(x)


class GeneralModelClass(nn.Module):
	"""
	Define the model replacing the last linear layer with a linear layer with the required amount of classes
	for the new task.
	"""
	def __init__(self, classes):
		super(GeneralModelClass, self).__init__()
		self.Tmodel = models.alexnet(pretrained = True)
		self.Tmodel.classifier[-1] = nn.Linear(self.Tmodel.classifier[-1].in_features, classes)

	def forward(self, x):
		return self.Tmodel(x)

