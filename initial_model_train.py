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

			# loss_1 only takes in the outputs from the nodes of the old classes 

			#loss1_output = output[:, :num_of_classes_old]
			#loss2_output = output[:, num_of_classes_old:]
	
			loss = model_criterion(output, labels, flag = "CE")
			
			del labels
			#del output

			#total_loss = alpha*loss_1 + loss_2

			#del loss_1
			#del loss_2

			loss.backward()
			optimizer.step()

			running_loss += loss.item()
			
		epoch_loss = running_loss/dset_size


		print('Epoch Loss:{}'.format(epoch_loss))

		if(epoch != 0 and epoch != num_of_epochs -1 and (epoch+1) % 10 == 0):
			epoch_file_name = os.path.join(mypath, str(epoch+1)+'.pth.tar')
			torch.save({
			'epoch': epoch,
			'epoch_loss': epoch_loss, 
			'model_state_dict': model_init.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),

			}, epoch_file_name)


	torch.save(model_init.state_dict(), mypath + "/best_performing_model.pth")		
		

	del model_init




