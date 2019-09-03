import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, Dataloader
from torchvision import transforms, utils, models

from autoencoder import Autoencoder, Alexnet_FE

import os
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")


def add_autoencoder(input_dims, code_dims):
""" 
Inputs: 
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


def autoencoder_train(model, feature_extractor, path, optimizer, encoder_criterion, dset_loaders, num_epochs, checkpoint_file, use_gpu):
	
	since = time.time()
	best_perform = 10e6
		
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

	for epoch in range(start_epoch, num_epochs):
		
		print ("Epoch {}/{}".format(epoch+1, num_epochs))
		print ("-"*10)

		for phase in ["train", "val"]:
			running_loss = 0
			running_correct_predictions = 0
			
			if (phase == "train"):
				model = model.train(True)
			else:
				model = model.train(False)

			for data in dset_loaders[phase]:
				input_data, labels = data
				
				if (use_gpu):
					input_data, labels = Variable(input_data.cuda()), Variable(labels.cuda())
				else:
					input_data, labels = Variable(input_data), Variable(labels)

				input_to_ae = feature_extractor(inputs_data)
				

				optimizer.zero_grad()
				model.zero_grad()

				outputs = model(input_to_ae)
				loss = encoder_criterion(outputs, inputs)

				if (phase == "train"):	
					loss.backward()
					optimizer.step()

				running_loss += loss.data[0]
				running_correct_predictions += torch.sum(preds == labels.data)


		epoch_loss = running_loss/dset_size
		epoch_accuracy = running_accuracy/dset_size

		if(phase == "train"):
			print('Epoch Loss:{}, Epoch Accuracy:{}'.format(epoch_loss, epoch_accuracy))
			
			if(epoch != 0 && (epoch+1) % 5 == 0):
				epoch_file_name = path +'/'+str(epoch+1)+'.pth.tar'
				torch.save({
					'epoch': epoch,
					'epoch_loss': epoch_loss, 
					'epoch_accuracy': epoch_accuracy, 
		            'model_state_dict': model.state_dict(),
		            'optimizer_state_dict': optimizer.state_dict(),
		           
		            }, epoch_file_name)

        else:
			
			if (epoch_loss < best_acc):
        		best_acc = epoch_loss
        		torch.save(model.state_dict(), path + "/best_performing_model.pth")				

	
	elapsed_time = time.time()-since
	print ("This procedure took {:.2f} minutes and {:.2f} seconds".format(elapsed_time//60, elapsed_time%60))
	print ("The best performing model has a {:.2f} loss on the validation set".format(best_acc))

	return model


def autoencoder_train(model, device, feature_extractor, path, optimizer, encoder_criterion, dset_loaders, dset_size, num_epochs, checkpoint_file, use_gpu):

	since = time.time()
	best_perform = 10e6

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

	for epoch in range(start_epoch, num_epochs):

		print ("Epoch {}/{}".format(epoch+1, num_epochs))
		print ("-"*10)

		for phase in ["train", "val"]:
			running_loss = 0
			running_correct_predictions = 0

			if (phase == "train"):
				model = model.train(True)
			else:
				model = model.train(False)

			for data in dset_loaders[phase]:
				input_data, _ = data

				if (use_gpu):
					input_data = Variable(input_data.to(device)) 
				
				else:
					input_data  = Variable(input_data)

				feature_extractor.to(device)

				input_to_ae = feature_extractor(input_data)

				input_to_ae = input_to_ae.view(-1, 256*13*13)

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


	


	


