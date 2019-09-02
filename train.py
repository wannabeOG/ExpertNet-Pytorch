import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings

from torch.utils.data import Dataset, Dataloader
from torchvision import transforms, utils

from autoencoder import Autoencoder

import os


warnings.filterwarnings("ignore")

def add_autoencoder(input_dims, code_dims):
""" 
Inputs: 
	1) input_dims = input_dims of the data or the images being fed into the autoencoder. Check the
	README.md for more details regarding the choice of input.
	2) code_dims = the dimenions of the "code" which is a lower dimensional representation of the 
	input data.

Outputs:
	1) autoencoder = A reference to the autoencoder object that is created 
	2) store_path = Path to the directory where the trained model and the checkpoints will be stored
"""	
	autoencoder = Autoencoder(input_dims, code_dims)
	num_ae = len(next(os.walk(dir)[1]))
	path = os.getcwd()
	store_path = path + "/models/autoencoders/autoencoder_"+str(num_ae+1)
	os.mkdir(store_path)
	return autoencoder, store_path


def autoencoder_train(model, path, dset_loaders, num_epochs, optimizer, use_gpu):
	since = time.time()
####################### Needs code for loading data into this ##########################		
	





########################################################################################
	for epoch in range(start_epoch, num_epochs):
		running_loss = 0
		running_correct_predictions = 0

		print ("Epoch {}/{}".format(epoch, num_epochs-1))
		print ("-"*20)

		for data in dset_loaders[phase]:
			input_data, labels = data

			if (use_gpu):
				input_data, labels = Variable(input_data.cuda()), Variable(labels.cuda())
			else:
				input_data, labels = Variable(input_data), Variable(labels)

			optimizer.zero_grad()
			model.zero_grad()

			outputs = model(inputs)
			preds = torch.argmax(outputs, 1)
			loss = criterion(preds, labels)

			loss.backward()
			optimizer.step()

			running_loss += loss.data[0]
			running_correct_predictions += torch.sum(preds == labels.data)


		epoch_loss = running_loss/dset_size
		epoch_accuracy = running_accuracy/dset_size

		print('Epoch Loss:{}, Epoch Accuracy:{}'.format(epoch_loss, epoch_accuracy))

		epoch_file_name = export_dir+'/'+str(epoch)+'.pth.tar'
		torch.save({
			'epoch': epoch,
			'epoch_loss': epoch_loss, 
			'epoch_accuracy': epoch_accuracy, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
           
            }, epoch_file_name)

	elapsed_time = time.time()-since
	print ("This procedure took {:.2f} minutes and {:.2f} seconds".format(elapsed_time//60, elapsed_time%60))
	return model

	

def distillation_loss(preds, ref_scores, temperature):
	


