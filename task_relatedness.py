import torch 
import os
from torchvision import models

from autoencoder import Autoencoder

def task_metric(rerrror_comp, r_error_ref):
	return (1-(rerror_ref-r_error_comp)/r_error_comp)

#tested
def kaiming_initilaization(layer):
	nn.init.kaiming_normal_(layer.weight, nonlinearity='sigmoid')

def get_initial_model(feature_extractor, layer_dict, dset_loaders, encoder_criterion, use_gpu):
	
	path = os.getcwd()
	destination = path + "/models/autoencoders"
	num_ae = len(next(os.walk(destination))[1])
	best_relatedness = 0
	model_number = -999
	device = torch.device("cuda:0" if use_gpu else "cpu")

	model_path = destination + "/autoencoder_"+str(i+1) +"/best_performing_model.pth"
	model = Autoencoder(input_dims)
	model.load_state_dict(torch.load(model_path), map_location= device)
	
	for i in range(num_ae):
		
		model_path = destination + "/autoencoder_"+str(num_ae-i) +"/best_performing_model.pth"
		model = Autoencoder(input_dims)
		model.load_state_dict(torch.load(model_path), map_location= device)
		
		feature_extractor = Alexnet_FE(layer_dict)
		feature_extractor.to(device)
		
		model.train(False)

		for data in dset_loaders:
			
			input_data, _ = data
			input_data.to(device)
			input_to_ae = feature_extractor(input_data)
			input_to_ae = input_to_ae.view(input_to_ae.size(0), -1)

			output = model(input_to_ae)
			loss = encoder_criterion(outputs, input_data)
			if (i == 0): 
				rerror_comp = loss
			else:
				relatedness = task_metric(loss, rerrror_comp)
				
				if (relatedness > best_relatedness):
					best_relatedness = relatedness
					model_number = (num_ae - i)


	return model_number, best_relatedness			


#Tested out, make sure that model_init is a shallow copy of the model that is being passed
def initialize_new_model(model_init, num_classes, num_of_classes_old):
	weight_info = model_init.classifier[-1].weight.data
	model_init.classifier[-1] = nn.Linear(model_init.classifier[-1].in_features, num_of_classes_old+num_classes)
	kaiming_initilaization(model_init.classifier[-1])
	model_init.classifier[-1].weight[:num_of_classes_old, :] = weight_info
	return model_init 


def train_model(num_classes):
	
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


	model_init = GenModel(num_of_classes_old)
	model_init.load_state_dict(torch.load(path_to_dir+"/best_performing_model.pth", map_location = device))
	ref_model = copy.deepcopy(model_init)
	ref_model.train(False)

	#Actually makes the changes to the model_init, so slightly redundant
	model_init = initialize_new_model(model_init, num_classes, num_of_classes_old)
	model_init.to(device)

	#Process for LwF
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
					
					output = model(input_data)
					ref_output = ref_model(input_data)

					optimizer.zero_grad()
					model.zero_grad()
					
					loss_1 = model_criterion(output[:num_of_classes_old], ref_output, flag = "Distill")
					loss_2 = model_criterion(output[num_of_classes_old:], labels, flag = "CE")

					total_loss = loss_1 + alpha*loss_2

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
						'model_state_dict': model.state_dict(),
						'optimizer_state_dict': optimizer.state_dict(),

						}, epoch_file_name)


				else:
					if (epoch_loss < best_perform):
						best_perform = epoch_loss
						torch.save(model.state_dict(), path + "/best_performing_model.pth")
	

	
	#Process for finetuning the model
	else:
		
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
						input_data = Variable(input_data.to(device))
						labels = Variable(labels.to(device)) 
					
					else:
						input_data  = Variable(input_data)
						labels = Variable(labels)


					model.to(device)

					output = model(input_data)
					
					optimizer.zero_grad()
					model.zero_grad()
					
					loss = model_criterion(output[num_of_classes_old:], labels)

					if (phase == "train"):	
						loss.backward()
						model.classifier[-1].weight.grad[:num_of_classes_old,:] = 0  
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






			


