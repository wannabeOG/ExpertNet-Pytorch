import torch 
import os
from torchvision import models

from model_utils import *
from autoencoder import GenModel


def train_model(num_classes, alpha = 0.01, optimizer, encoder_criterion, dset_loaders, dset_size, num_epochs, checkpoint_file, use_gpu):
""" 
Inputs: 
	1) num_classes = A reference to the Autoencoder model that needs to be trained 
	2) alpha = A constant which is used to determine the contributions of two distinct loss functions to the total
	   loss finally reported 
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


Function: Trains the model
	1) If the task relatedness is greater than 0.85, the function uses the Learning without Forgetting method
	2) If the task relatedness is lesser than 0.85, the function uses the normal finetuning procedure as outlined
	   in the "Learning without Forgetting" paper ("https://arxiv.org/abs/1606.09282")

   Whilst implementing finetuning procedure, PyTorch does not provide the option to only partially freeze the 
   weights of a layer. In order to implement this idea, I manually zero the gradients from the older classes in
   order to ensure that these weights do not have a learning signal from the loss function. 

"""	
	
	model_number, best_relatedness = get_initial_model(feature_extractor, dset_loaders, encoder_criterion, use_gpu)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
	
	# Load the most related model in the memory and finetune the model
	path = os.getcwd() + "/models/trained_models/model_"
	path_to_dir = path + str(model_number) 
	
	file_name = path_to_dir + "/classes.txt" 
	file_object = open(file_name, 'r')
	
	num_of_classes_old = file_object.read()
	file_object.close()
	
	# Create a directory for the new model
	num_ae = len(next(os.walk(path))[1])
	os.mkdir(path + str(num_ae+1))

	# Store the number of classes in the file for future use
	with open(os.path.join(path + str(num_ae+1), 'classes.txt'), 'w') as file1:
		input_to_txtfile = str(num_of_classes_old + num_classes)
		file1 = file1.write(input_to_txtfile)
		file1.close()

	# Load the most related model into memory
	model_init = GenModel(num_of_classes_old)
	model_init.load_state_dict(torch.load(path_to_dir+"/best_performing_model.pth", map_location = device))
	
	# Reference model to compute the soft scores for the LwF(Learning without Forgetting) method
	ref_model = copy.deepcopy(model_init)
	ref_model.train(False)

	#Actually makes the changes to the model_init, so slightly redundant
	model_init = initialize_new_model(model_init, num_classes, num_of_classes_old)
	model_init.to(device)

	#The training process for LwF (Learning without Forgetting)

	if (best_relatedness > 0.85):

		for epoch in range(start_epoch, num_epochs):

			print ("Epoch {}/{}".format(epoch+1, num_epochs))
			print ("-"*20)

			for phase in ["train", "val"]:
				running_loss = 0
				
				if (phase == "train"):
					model_init = model_init.train(True)
				else:
					model_init = model_init.train(False)

				for data in dset_loaders[phase]:
					input_data, labels = data

					if (use_gpu):
						input_data = Variable(input_data.to(device))
						labels = Variable(labels.to(device)) 
					
					else:
						input_data  = Variable(input_data)
						labels = Variable(labels)
					
					model_init.to(device)
					ref_model.to(device)
					
					output = model_init(input_data)
					ref_output = ref_model(input_data)

					optimizer.zero_grad()
					model_init.zero_grad()

					# loss_1 only takes in the outputs from the nodes of the old classes  
					loss_1 = model_criterion(output[:num_of_classes_old], ref_output, flag = "Distill")
					
					# loss_2 takes in the outputs from the nodes that were initialized for the new task
					loss_2 = model_criterion(output[num_of_classes_old:], labels, flag = "CE")

					total_loss = alpha*loss_1 + loss_2

					if (phase == "train"):	
						total_loss.backward()
						optimizer.step()

					running_loss += total_loss.item()
					
				epoch_loss = running_loss/dset_size[phase]

				if(phase == "train"):
					print('Epoch Loss:{}, Epoch Accuracy:{}'.format(epoch_loss, epoch_accuracy))

					if(epoch != 0 and (epoch+1) % 5 == 0):
						epoch_file_name = path +'/'+str(epoch+1)+'.pth.tar'
						torch.save({
						'epoch': epoch,
						'epoch_loss': epoch_loss, 
						'model_state_dict': model_init.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),

						}, epoch_file_name)


				else:
					if (epoch_loss < best_perform):
						best_perform = epoch_loss
						torch.save(model_init.state_dict(), path + "/best_performing_model.pth")
	

	
	#Process for finetuning the model
	else:
		
		for epoch in range(start_epoch, num_epochs):

			print ("Epoch {}/{}".format(epoch+1, num_epochs))
			print ("-"*10)

			for phase in ["train", "val"]:
				running_loss = 0
				running_correct_predictions = 0

				if (phase == "train"):
					model_init = model_init.train(True)
				else:
					model_init = model_init.train(False)

				for data in dset_loaders[phase]:
					input_data, labels = data

					if (use_gpu):
						input_data = Variable(input_data.to(device))
						labels = Variable(labels.to(device)) 
					
					else:
						input_data  = Variable(input_data)
						labels = Variable(labels)


					model_init.to(device)

					output = model_init(input_data)
					
					optimizer.zero_grad()
					model_init.zero_grad()
					
					#Since the
					loss = model_criterion(output[num_of_classes_old:], labels)

					if (phase == "train"):	
						loss.backward()
						# Zero the gradients from the older classes
						model_init.classifier[-1].weight.grad[:num_of_classes_old,:] = 0  
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
						'model_state_dict': model_init.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),

						}, epoch_file_name)


				else:
					if (epoch_loss < best_perform):
						best_perform = epoch_loss
						torch.save(model_init.state_dict(), path + "/best_performing_model.pth")


	return model_init



			


