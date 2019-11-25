"""
Module to train the model for the first task. Seperated from the rest of the code for the purpose of clarity
The paper treats a pretrained Alexnet model as the initial expert so this file also helps to recreate that setting
without overtly making the generate_models.py file complicated to read at the expense of some redundancy in the code.
"""

#!/usr/bin/env python
# coding: utf-8

import torch 
import os
from torchvision import models
from autoencoder import GeneralModelClass

import copy

import sys
sys.path.append(os.path.join(os.getcwd(), 'utils'))
from model_utils import *

def train_model_1(num_classes, feature_extractor, encoder_criterion, dset_loaders, dset_size, num_epochs, use_gpu, task_number, lr = 0.1, alpha = 0.01):
	""" 
	Inputs: 
		1) num_classes = The number of classes in the new task  
		2) feature_extractor = A reference to the feature extractor model  
		3) encoder_criterion = The loss criterion for training the Autoencoder
		4) dset_loaders = Dataset loaders for the model
		5) dset_size = Size of the dataset loaders
		6) num_of_epochs = Number of epochs for which the model needs to be trained
		7) use_gpu = A flag which would be set if the user has a CUDA enabled device
		8) task_number = A number which represents the task for which the model is being trained
		9) lr = initial learning rate for the model
		10) alpha = Tradeoff factor for the loss   

	Function: Trains the model on the first task specifically
		
	"""
	since = time.time()
	best_perform = 10e6
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	model_init = GeneralModelClass(num_classes)
	model_init.to(device)

	for param in model_init.Tmodel.classifier.parameters():
		param.requires_grad = True

	for param in model_init.Tmodel.features.parameters():
		param.requires_grad = False

	for param in model_init.Tmodel.features[8].parameters():
		param.requires_grad = True

	for param in model_init.Tmodel.features[10].parameters():
		param.requires_grad = True

		
	#model_init.to(device)
	print ("Initializing an Adam optimizer")
	optimizer = optim.Adam(model_init.Tmodel.parameters(), lr = 0.003, weight_decay= 0.0001)

	print ("Creating the directory for the new model")
	os.mkdir(os.path.join(os.getcwd(), "models", "trained_models", "model_1"))

	mypath = os.path.join(os.getcwd(), "models", "trained_models", "model_1")
	
	# Store the number of classes in the file for future use
	with open(os.path.join(mypath, 'classes.txt'), 'w') as file1:
		input_to_txtfile = str(num_classes)
		file1.write(input_to_txtfile)
		file1.close()

	for epoch in range(num_epochs):
		since = time.time()
		best_perform = 10e6
		
		print ("Epoch {}/{}".format(epoch+1, num_epochs))
		print ("-"*20)
		#print ("The training phase is ongoing".format(phase))
		
		running_loss = 0
		
		#scales the optimizer every 10 epochs 
		optimizer = exp_lr_scheduler(optimizer, epoch, lr)
		model_init = model_init.train(True)
		
		for data in dset_loaders:
			input_data, labels = data

			del data

			if (use_gpu):
				input_data = Variable(input_data.to(device))
				labels = Variable(labels.to(device)) 
			
			else:
				input_data  = Variable(input_data)
				labels = Variable(labels)
			
			output = model_init(input_data)
			#ref_output = ref_model(input_data)

			del input_data

			optimizer.zero_grad()
			#model_init.zero_grad()

			loss = model_criterion(output, labels, flag = "CE")
			
			del labels
			#del output

			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			
		epoch_loss = running_loss/dset_size


		print('Epoch Loss:{}'.format(epoch_loss))

		if(epoch != 0 and epoch != num_epochs -1 and (epoch+1) % 10 == 0):
			epoch_file_name = os.path.join(mypath, str(epoch+1)+'.pth.tar')
			torch.save({
			'epoch': epoch,
			'epoch_loss': epoch_loss, 
			'model_state_dict': model_init.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),

			}, epoch_file_name)


	torch.save(model_init.state_dict(), mypath + "/best_performing_model.pth")		
		

	del model_init




