#!/usr/bin/env python
# coding: utf-8

"""
This module is used to generate the experts for the tasks in the sequence that have been generated 
by the data_prep files. This model will also train the autoencoders that will be used to identify 
the tasks during the testing phase of the model

"""

import torch
torch.backends.cudnn.benchmark=True

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import numpy as np

import copy

from autoencoder import *

import os
import sys
from random import shuffle

sys.path.append(os.path.join(os.getcwd(), 'utils'))

from encoder_train import *
from encoder_utils import *

from model_train import *
from model_utils import *

from initial_model_train import *

#define the parser
parser = argparse.ArgumentParser(description='Generate models file')
parser.add_argument('--init_lr', default=0.1, type=float, help='Init learning rate')
parser.add_argument('--num_epochs_encoder', default=15, type=int, help='Number of epochs you want the encoder model to train on')
parser.add_argument('--num_epochs_model', default=40, type=int, help='Number of epochs you want  model to train on')
parser.add_argument('--batch_size', default=16, type=int, help='Mini batch size')
parser.add_argument('--use_gpu', default='False', help='Set the GPU flag either True or False to use the GPU', type=str)

args = parser.parse_args()

#get the arguments
use_gpu = args.use_gpu
num_epochs_encoder = args.num_epochs_encoder
num_epochs_model = args.num_epochs_model
batch_size = args.batch_size
lr = args.init_lr

#number of tasks in the sequence
no_of_tasks = 9

#transforms for the tiny-imagenet dataset. Applicable for the tasks 1-4
data_transforms_tin = {
		'train': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		]),
		'test': transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
	}


#transforms for the mnist dataset. Applicable for the tasks 5-9
data_transforms_mnist = {
	'train': transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize([0.1307,], [0.3081,])
		]),
		'test': transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize([0.1307,], [0.3081,])
		])
}

#code to generate the initial directories for storing the models
model_path = os.path.join(os.getcwd(), "models")

if not (os.path.isdir(model_path)):
	os.mkdir(model_path)

if not (os.path.isdir(os.path.join(model_path, "autoencoders"))):
	os.mkdir(os.path.join(model_path, "autoencoders"))

if not (os.path.isdir(os.path.join(model_path, "trained_models"))):
	os.mkdir(os.path.join(model_path, "trained_models"))


#Initial model 
pretrained_alexnet = models.alexnet(pretrained = True)

#Derives a feature extractor model from the Alexnet model
feature_extractor = Alexnet_FE(pretrained_alexnet)

#shuffle the items in a list so that mnist tasks and tiny imagenet tasks are iterspersed, uncomment these lines
#if you are also using MNIST dataset

#task_number = shuffle([x for x in range(1, 9+1)])

#Replace the "range(1, no_of_tasks+1)" in the for loop with task_numbe list

#shuffle over the tasks
for task_number in range(1, no_of_tasks+1):
	
	print ("Task Number {}".format(task_number))
	data_path = os.getcwd() + "/Data"
	encoder_path = os.getcwd() + "/models/autoencoders"
	#model_path = os.getcwd() + "/models/trained_models"

	path_task = data_path + "/Task_" + str(task_number)
	
	if (task_number >=1 and task_number <=4 ):
		image_folder = datasets.ImageFolder(path_task + "/" + 'train', transform = data_transforms_tin['train'])
	
	else:
		image_folder = datasets.ImageFolder(path_task + "/" + 'train', transform = data_transforms_mnist['train'])	
	
	dset_size = len(image_folder)

	device = torch.device("cuda:0" if use_gpu else "cpu")

	dset_loaders = torch.utils.data.DataLoader(image_folder, batch_size = batch_size,
													shuffle=True, num_workers=4)

	mypath = encoder_path + "/autoencoder_" + str(task_number)

	if os.path.isdir(mypath):
		############ check for the latest checkpoint file in the autoencoder ################
		onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
		max_train = -1
		flag = False

		model = Autoencoder(256*13*13)
		
		store_path = mypath
		
		for file in onlyfiles:
			if(file.endswith('pth.tr')):
				flag = True
				test_epoch = int(file[0])
				if(test_epoch > max_train): 
					max_epoch = test_epoch
					checkpoint_file_encoder = file
		#######################################################################################
		
		if (flag == False): 
			checkpoint_file_encoder = ""

	else:
		checkpoint_file_encoder = ""

	#get an autoencoder model and the path where the autoencoder model would be stored
	model, store_path = add_autoencoder(256*13*13, 100, task_number)

	#Define an optimizer for this model 
	optimizer_encoder = optim.Adam(model.parameters(), lr = 0.003, weight_decay= 0.0001)

	print ("Reached here for {}".format(task_number))
	print ()
	#Training the autoencoder
	autoencoder_train(model, feature_extractor, store_path, optimizer_encoder, encoder_criterion, dset_loaders, dset_size, num_epochs_encoder, checkpoint_file_encoder, use_gpu)

	#Train the model
	if(task_number == 1):
		train_model_1(len(image_folder.classes), feature_extractor, encoder_criterion, dset_loaders, dset_size, num_epochs_model , True, task_number,  lr = lr)
	else:	
		train_model(len(image_folder.classes), feature_extractor, encoder_criterion, dset_loaders, dset_size, num_epochs_model , True, task_number,  lr = lr)