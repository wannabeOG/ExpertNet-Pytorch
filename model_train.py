import torch 
import os
from torchvision import models

from model_utils import *
from autoencoder import GeneralModelClass

import copy

def train_model(num_classes, feature_extractor, encoder_criterion, dset_loaders, dset_size, num_epochs, checkpoint_file, use_gpu, alpha = 0.01):
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
	
	print ("Determining the most related model")
	model_number, best_relatedness = get_initial_model(feature_extractor, dset_loaders, dset_size, encoder_criterion, use_gpu)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
	

	# Load the most related model in the memory and finetune the model
	new_path = os.getcwd() + "/models/trained_models"
	path = os.getcwd() + "/models/trained_models/model_"
	path_to_dir = path + str(model_number) 
	
	file_name = path_to_dir + "/classes.txt" 
	file_object = open(file_name, 'r')
	
	num_of_classes_old = file_object.read()
	file_object.close()
	num_of_classes_old = int(num_of_classes_old)
	new_classes = num_of_classes_old + num_classes
	# Create a directory for the new model
	num_ae = len(next(os.walk(new_path))[1])

	print ("Creating the directory for the new model")
	
	os.mkdir(path + str(num_ae+1))

	path_to_model = path + str(num_ae+1)
	# Store the number of classes in the file for future use
	with open(os.path.join(path + str(num_ae+1), 'classes.txt'), 'w') as file1:
		input_to_txtfile = str(new_classes)
		file1.write(input_to_txtfile)
		file1.close()

	# Load the most related model into memory
	model_init = GeneralModelClass(num_of_classes_old)
	print ("The architecture of the model; most related to the given task")
	print (model_init)

	print ("Loading the model")
	model_init.load_state_dict(torch.load(path_to_dir+"/best_performing_model.pth"))
	print ("Model loaded")

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


	# Reference model to compute the soft scores for the LwF(Learning without Forgetting) method
	
	ref_model = copy.deepcopy(model_init)
	ref_model.train(False)
	ref_model.to(device)

	#Actually makes the changes to the model_init, so slightly redundant
	print ("Initializing the model to be trained")
	model_init = initialize_new_model(model_init, num_classes, num_of_classes_old)
	model_init.to(device)

	#The training process format or LwF (Learning without Forgetting)
	# Add the start epoch code 
	
	if (best_relatedness > 0.85):

		print ("Using the LwF approach")

		for epoch in range(num_epochs):
			
			since = time.time()
			best_perform = 10e6
			
			print ("Epoch {}/{}".format(epoch+1, num_epochs))
			print ("-"*20)

			for phase in ["train", "test"]:
				
				print ("The {}ing phase is ongoing".format(phase))
				
				running_loss = 0
				
				if (phase == "train"):
					model_init = model_init.train(True)
				else:
					model_init = model_init.train(False)
					model_init.eval()

				for data in dset_loaders[phase]:
					input_data, labels = data

					del data

					if (use_gpu):
						input_data = Variable(input_data.to(device))
						labels = Variable(labels.to(device)) 
					
					else:
						input_data  = Variable(input_data)
						labels = Variable(labels)
					
					#model_init.to(device)
					#ref_model.to(device)
					
					output = model_init(input_data)
					ref_output = ref_model(input_data)

					del input_data

					optimizer.zero_grad()
					model_init.zero_grad()

					# loss_1 only takes in the outputs from the nodes of the old classes 

					loss1_output = output[:, :num_of_classes_old]
					loss2_output = output[:, num_of_classes_old:]

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

					if (phase == "train"):	
						total_loss.backward()
						optimizer.step()

					running_loss += total_loss.item()
					
				epoch_loss = running_loss/dset_size[phase]

				if(phase == "train"):
					print('Epoch Loss:{}'.format(epoch_loss))

					if(epoch != 0 and (epoch+1) % 10 == 0):
						epoch_file_name = path_to_model +'/'+str(epoch+1)+'.pth.tar'
						torch.save({
						'epoch': epoch,
						'epoch_loss': epoch_loss, 
						'model_state_dict': model_init.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),

						}, epoch_file_name)


				else:
					if (epoch_loss < best_perform):
						best_perform = epoch_loss
						torch.save(model_init.state_dict(), path_to_model + "/best_performing_model.pth")
	
		del model_init
		del ref_model
	
	#Process for finetuning the model
	else:
		
		print ("Using the finetuning approach")
		
		for epoch in range(start_epoch, num_epochs):


			print ("Epoch {}/{}".format(epoch+1, num_epochs))
			print ("-"*10)

			for phase in ["train", "test"]:
				running_loss = 0
				running_correct_predictions = 0

				if (phase == "train"):
					model_init = model_init.train(True)
				else:
					model_init = model_init.train(False)

				for data in dset_loaders[phase]:
					input_data, labels = data
					del data

					if (use_gpu):
						input_data = Variable(input_data.to(device))
						labels = Variable(labels.to(device)) 
					
					else:
						input_data  = Variable(input_data)
						labels = Variable(labels)


					model_init.to(device)

					output = model_init(input_data)
					
					del input_data
					#del output
					
					optimizer.zero_grad()
					model_init.zero_grad()
					
					#Since the
					loss = model_criterion(output[num_of_classes_old:], labels)

					del output
					del labels

					if (phase == "train"):	
						loss.backward()
						# Zero the gradients from the older classes
						model_init.Tmodel.classifier[-1].weight.grad[:num_of_classes_old,:] = 0  
						optimizer.step()

					running_loss += loss.item()
					
				epoch_loss = running_loss/dset_size[phase]

				if(phase == "train"):
					print('Epoch Loss:{}'.format(epoch_loss))

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

		
		del ref_model				

	
	return model_init



			


