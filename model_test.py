import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

import torchvision
from torchvision import models, datasets

import os 
import shutil 
import pathlib 

from autoencoder import *
from encoder_train import *
from encoder_utils import *

#from model_utils import *

num_epochs = 5


#path = "/home/himal/expert-gate/models/trained_models/model_1"
path = "/home/wannabe/Desktop/Papers to Read/Incremental Learning/implementations/ExpertNet-Pytorch/models/trained_models/model_1"

model_init = GeneralModelClass(50)

print (model_init)

#model_init.load_state_dict(torch.load(path + "/best_performing_model.pth"))

#model = models.alexnet(pretrained=True)

################################################ Model parameters setting #################################################################

for param in model_init.Tmodel.classifier.parameters():
	param.requires_grad = True

for param in model_init.Tmodel.features.parameters():
    param.requires_grad = False
    
for param in model_init.Tmodel.features[8].parameters():
    param.requires_grad = True

for param in model_init.Tmodel.features[10].parameters():
    param.requires_grad = True


optimizer = optim.Adam(model_init.Tmodel.parameters(), lr = 0.003, weight_decay= 0.0001)

############################################################################################################################################


#path_task1 = "/home/himal/expert-gate/data/Task_1"
path_task1 = "/home/wannabe/Downloads/Test_datasets/tiny-imagenet/Task_1" 

data_transforms = {
	'train':transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	]),
	'test':transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])
}

image_folder = {x: datasets.ImageFolder(path_task1 + "/" + x, transform = data_transforms[x]) for x in ['train', 'test']}

#image_folder

dataset_sizes = {x: len(image_folder[x]) for x in ['train', 'test']}

#dataset_sizes

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")


dataloaders = {x: torch.utils.data.DataLoader(image_folder[x], batch_size=4,
												shuffle=True, num_workers=4) for x in ['train', 'test']}

#pretrained_alexnet = models.alexnet(pretrained = True)
#optimizer = optim.Adam(model.parameters(), lr = 0.003, weight_decay= 0.0001)

model_init.to(device)

for epoch in range(num_epochs):
	
	since = time.time()
	best_perform = 10e6
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	num_of_classes = 0

	print ("Epoch {}/{}".format(epoch+1, num_epochs))
	print ("------"*20)

	for phase in ["train", "test"]:
		running_loss = 0

		if (phase == "train"):
			model_init = model_init.train(True)
		else:
			model_init = model_init.train(False)
			model_init.eval()

		for data in dataloaders[phase]:
			input_data, labels = data

			del data

			#if (use_gpu):
			input_data = Variable(input_data.to(device))
			labels = Variable(labels.to(device)) 

			#else:
			#input_data  = Variable(input_data)
			#labels = Variable(labels)

			
			#ref_model.to(device)

			output = model_init(input_data)
			#ref_output = ref_model(input_data)
			
			del input_data
			
			optimizer.zero_grad()
			model_init.zero_grad()

			# loss_1 only takes in the outputs from the nodes of the old classes  
			#loss_1 = model_criterion(output[:num_of_classes_old], ref_output, flag = "Distill")

			# loss_2 takes in the outputs from the nodes that were initialized for the new task
			CEloss = nn.CrossEntropyLoss()
			loss_1 = CEloss(output, labels) 

			del output
			del labels

			total_loss = loss_1

			if (phase == "train"):	
				total_loss.backward()
				optimizer.step()

			running_loss += total_loss.item()

		
		epoch_loss = running_loss/dataset_sizes[phase]

			
		if(phase == "train"):
			print('Epoch Loss:{}'.format(epoch_loss))
			if(epoch != 0 and (epoch+1) % 10 == 0):
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
	

	#torch.save(model_init.state_dict(), path + "/best_performing_model.pth