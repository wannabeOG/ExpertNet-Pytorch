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

def train_model(num_classes, feature_extractor, encoder_criterion, dset_loaders, dset_size, num_epochs, use_gpu, task_number, lr = 0.1, alpha = 0.01):
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

	Function: Trains the model on the given task
		1) If the task relatedness is greater than 0.85, the function uses the Learning without Forgetting method
		2) If the task relatedness is lesser than 0.85, the function uses the normal finetuning procedure as outlined
			in the "Learning without Forgetting" paper ("https://arxiv.org/abs/1606.09282")

		Whilst implementing finetuning procedure, PyTorch does not provide the option to only partially freeze the 
		weights of a layer. In order to implement this idea, I manually zero the gradients from the older classes in
		order to ensure that these weights do not have a learning signal from the loss function. 

	"""	
	
	device = torch.device("cuda:0" if use_gpu else "cpu")

	print ("Determining the most related model")
	model_number, best_relatedness = get_initial_model(feature_extractor, dset_loaders, dset_size, encoder_criterion, use_gpu)
	
	# Load the most related model in the memory and finetune the model
	new_path = os.getcwd() + "/models/trained_models"
	path = os.getcwd() + "/models/trained_models/model_"
	path_to_dir = path + str(model_number) 
	file_name = path_to_dir + "/classes.txt" 
	file_object = open(file_name, 'r')
	
	num_of_classes_old = file_object.read()
	file_object.close()
	num_of_classes_old = int(num_of_classes_old)

	#Create a variable to store the new number of classes that this model is exposed to
	new_classes = num_of_classes_old + num_classes
	
	#Check the number of models that already exist
	num_ae = len(next(os.walk(new_path))[1])
	#num_ae = 0

	#If task_number is less than num_ae it suggests that the directory had already been created
	if (task_number <= num_ae):
		#Keeping it consistent with the usage of num_ae throughout this file
		num_ae = task_number-1

	
	print ("Checking if a prior training file exists")
	
	#mypath is the path where the model is going to be stored
	mypath = path + str(num_ae+1)

	#The conditional if the directory already exists
	if os.path.isdir(mypath):
		#mypath = path + str(num_ae+1)

		######################### check for the latest checkpoint file #######################
		onlyfiles = [f for f in os.listdir(mypath) if os.isfile(os.join(mypath, f))]
		max_train = -1
		flag = False

		#Check the latest epoch file that was created
		for file in onlyfiles:
			if(file.endswith('pth.tr')):
				flag = True
				test_epoch = file[0]
				if(test_epoch > max_train): 
					max_epoch = test_epoch
					checkpoint_file = file
		#######################################################################################
		
		if (flag == False): 
			checkpoint_file = ""

		
		#Steps to create a ref_model in order to prevent storing this model as well
		model_init = GeneralModelClass(num_of_classes_old)
		model_init.load_state_dict(torch.load(path_to_dir+"/best_performing_model.pth"))
		
		#Create (Recreate) the ref_model that has to be used
		ref_model = copy.deepcopy(model_init)
		ref_model.train(False)
		ref_model.to(device)
		
		######################## Code for loading the checkpoint file #########################
		
		if (os.path.isfile(mypath + "/" + checkpoint_file)):
			
			print ("Loading checkpoint '{}' ".format(checkpoint_file))
			checkpoint = torch.load(checkpoint_file)
			start_epoch = checkpoint['epoch']
			
			print ("Loading the model")
			model_init = GeneralModelClass(num_of_classes_old + num_classes)
			model_init = model_init.load_state_dict(checkpoint['state_dict'])
			
			print ("Loading the optimizer")
			optimizer = optimizer.load_state_dict(checkpoint['optimizer'])
			
			print ("Done")

		else:
			start_epoch = 0

		##########################################################################################



	#Will have to create a new directory since it does not exist at the moment
	else:
		print ("Creating the directory for the new model")
		os.mkdir(mypath)


	# Store the number of classes in the file for future use
		with open(os.path.join(mypath, 'classes.txt'), 'w') as file1:
			input_to_txtfile = str(new_classes)
			file1.write(input_to_txtfile)
			file1.close()

	# Load the most related model into memory
	
		print ("Loading the most related model")
		model_init = GeneralModelClass(num_of_classes_old)
		model_init.load_state_dict(torch.load(path_to_dir+"/best_performing_model.pth"))
		print ("Model loaded")

		#Create (Recreate) the ref_model that has to be used
		ref_model = copy.deepcopy(model_init)
		ref_model.train(False)
		ref_model.to(device)

		#print (ref_model)

		for param in model_init.Tmodel.classifier.parameters():
			param.requires_grad = True

		for param in model_init.Tmodel.features.parameters():
			param.requires_grad = False

		for param in model_init.Tmodel.features[8].parameters():
			param.requires_grad = True

		for param in model_init.Tmodel.features[10].parameters():
			param.requires_grad = True

		
		#model_init.to(device)
		print ()
		print ("Initializing an Adam optimizer")
		optimizer = optim.Adam(model_init.Tmodel.parameters(), lr = 0.003, weight_decay= 0.0001)


		# Reference model to compute the soft scores for the LwF(Learning without Forgetting) method
		
		
		#Actually makes the changes to the model_init, so slightly redundant
		print ("Initializing the model to be trained")
		model_init = initialize_new_model(model_init, num_classes, num_of_classes_old)
		#print (model_init)
		#model_init.to(device)
		start_epoch = 0

	#The training process format or LwF (Learning without Forgetting)
	# Add the start epoch code 
	
	if (best_relatedness > 0.85):

		model_init.to(device)
		ref_model.to(device)

		print ("Using the LwF approach")
		for epoch in range(start_epoch, num_epochs):			
			since = time.time()
			best_perform = 10e6
			
			print ("Epoch {}/{}".format(epoch+1, num_epochs))
			print ("-"*20)
			print ("The training phase is ongoing")
			
			running_loss = 0
			
			#scales the optimizer every 10 epochs 
			optimizer = exp_lr_scheduler(optimizer, epoch, lr)
			#model_init = model_init.train(True)
			
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
				ref_output = ref_model(input_data)
				del input_data

				optimizer.zero_grad()

				# loss_1 only takes in the outputs from the nodes of the old classes 

				loss1_output = output[:, :num_of_classes_old]
				loss2_output = output[:, num_of_classes_old:]

				print ()

				del output

				loss_1 = model_criterion(loss1_output, ref_output, flag = "Distill")
				del ref_output
				
				# loss_2 takes in the outputs from the nodes that were initialized for the new task
				
				loss_2 = model_criterion(loss2_output, labels, flag = "CE")
				del labels
				#del output

				total_loss = alpha*loss_1 + loss_2

				del loss_1
				del loss_2

				
				total_loss.backward()
				optimizer.step()

				running_loss += total_loss.item()
				
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
		del ref_model
	
	#Process for finetuning the model
	else:
		
		model_init.to(device)
		print ("Using the finetuning approach")
		
		for epoch in range(start_epoch, num_epochs):


			print ("Epoch {}/{}".format(epoch+1, num_epochs))
			print ("-"*20)

			optimizer = exp_lr_scheduler(optimizer, epoch, lr)
			model_init = model_init.train(True)
			
			running_loss = 0
			
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
				
				del input_data
				#del output
				
				optimizer.zero_grad()
				model_init.zero_grad()
				
				#Implemented as explained in the doc string
				loss = model_criterion(output[num_of_classes_old:], labels, flag = 'CE')

				del output
				del labels

				loss.backward()
				# Zero the gradients from the older classes
				model_init.Tmodel.classifier[-1].weight.grad[:num_of_classes_old,:] = 0  
				optimizer.step()

				running_loss += loss.item()
				
			epoch_loss = running_loss/dset_size[phase]

			print('Epoch Loss:{}'.format(epoch_loss))

			if(epoch != 0 and (epoch+1) % 5 == 0 and epoch != num_epochs -1):
				epoch_file_name = os.path.join(path_to_model, str(epoch+1)+'.pth.tar')
				torch.save({
				'epoch': epoch,
				'epoch_loss': epoch_loss, 
				'model_state_dict': model_init.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),

				}, epoch_file_name)


		torch.save(model_init.state_dict(), mypath + "/best_performing_model.pth")

		del model_init
		del ref_model