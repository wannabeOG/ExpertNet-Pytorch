import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from autoencoder import Autoencoder, Alexnet_FE

import os
import warnings
import time

warnings.filterwarnings("ignore")


def add_autoencoder(input_dims, code_dims):

	"""Inputs: 
		1) input_dims = input_dims of the features being fed into the autoencoder. Check the
		README.md for more details regarding the choice of input.
		2) code_dims = the dimenions of the "code" which is a lower dimensional representation of the 
		input data.

	Outputs:
		1) autoencoder = A reference to the autoencoder object that is created 
		2) store_path = Path to the directory where the trained model and the checkpoints will be stored
	"""	
		
	autoencoder = Autoencoder(input_dims, code_dims)
	og_path = os.getcwd()
	dir = og_path + "/models/autoencoders/"
	num_ae = len(next(os.walk(dir)[1]))
	store_path = dir + "/autoencoder_"+str(num_ae+1)
	os.mkdir(store_path)

	return autoencoder, store_path


def autoencoder_train(model, feature_extractor, path, optimizer, encoder_criterion, dset_loaders, dset_size, num_epochs, checkpoint_file, use_gpu):
	"""
	Inputs:
		1) model = A reference to the Autoencoder model that needs to be trained 
		2) feature_extractor = A reference to to the feature_extractor part of Alexnet; it returns the features
		   from the last convolutional layer of the Alexnet
		3) path = The path where the model will be stored
		4) optimizer = The optimizer to optimize the parameters of the Autoencoder
		5) encoder_criterion = The loss criterion for training the Autoencoder
		6) dset_loaders = Dataset loaders for the model
		7) dset_size = Size of the dataset loaders
		8) num_of_epochs = Number of epochs for which the model needs to be trained
		9) checkpoint_file = A checkpoint file which can be used to resume training; starting from the epoch at 
		   which the checkpoint file was created 
		10) use_gpu = A flag which would be set if the user has a CUDA enabled device 

	Outputs:
		1) model = A reference to the trained model


	Function:
		Returns a trained autoencoder model

	"""
	since = time.time()
	best_perform = 10e6
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	num_of_classes = 0

################################### Code for loading the checkpoint file ########################################
	
	if (os.path.isfile(path + "/" + checkpoint_file)):
		print ("Loading checkpoint '{}' ".format(checkpoint_file))
		checkpoint = torch.load(resume)
		start_epoch = checkpoint['epoch']
		print ("Loading the model")
		model = model.load_state_dict(checkpoint['state_dict'])
		print ("Loading the optimizer")
		optimizer = optimizer.load_state_dict(checkpoint['optimizer'])
		print ("Done")

	else:
		start_epoch = 0
#################################################################################################################
	

	for epoch in range(start_epoch, num_epochs):

		print ("Epoch {}/{}".format(epoch+1, num_epochs))
		print ("-"*10)

		# The model is evaluated at each epoch and the best performing model 
		# on the validation set is saved 

		for phase in ["train", "val"]:
			running_loss = 0
			
			if (phase == "train"):
				optimizer = exp_lr_scheduler(optimizer, epoch, lr)
				model = model.train(True)
			else:
				model = model.train(False)

			for data in dset_loaders[phase]:
				input_data, labels = data

				if (use_gpu):
					input_data = Variable(input_data.to(device)) 
				
				else:
					input_data  = Variable(input_data)

				# Input_to_ae is the features from the last convolutional layer
				# of an Alexnet trained on Imagenet 

				feature_extractor.to(device)
				input_to_ae = feature_extractor(input_data)
				input_to_ae = input_to_ae.view(input_to_ae.size(0), -1)

				optimizer.zero_grad()
				model.zero_grad()

				input_to_ae = input_to_ae.to(device)
				model.to(device)

				outputs = model(input_to_ae)
				loss = encoder_criterion(outputs, input_to_ae)

				if (phase == "train"):	
					loss.backward()
					optimizer.step()

				running_loss += loss.item()
			
			epoch_loss = running_loss/dset_size[phase]

			if(phase == "train"):
				print('Epoch Loss:{}, Epoch Accuracy:{}'.format(epoch_loss, epoch_accuracy))
				
				#Creates a checkpoint every 5 epochs
				if(epoch != 0 and (epoch+1) % 5 == 0):
					epoch_file_name = path +'/'+str(epoch+1)+'.pth.tar'
					torch.save({
					'epoch': epoch,
					'epoch_loss': epoch_loss, 
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),

					}, epoch_file_name)


			else:
				if (epoch_loss < best_perform):
					best_perform = epoch_loss
					torch.save(model.state_dict(), path + "/best_performing_model.pth")


	elapsed_time = time.time()-since
	print ("This procedure took {:.2f} minutes and {:.2f} seconds".format(elapsed_time//60, elapsed_time%60))
	print ("The best performing model has a {:.2f} loss on the validation set".format(best_perform))

	return model


	


	

