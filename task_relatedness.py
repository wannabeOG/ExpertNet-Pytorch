import torch 
import os

from autoencoder import Autoencoder

def task_metric(rerrror_comp, r_error_ref):
	return (1-(rerror_ref-r_error_comp)/r_error_comp)

def kaiming_initilaization(layer):
	nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')

def get_initial_model(feature_extractor, layer_dict, dset_loaders, encoder_criterion, use_gpu):
	
	path = os.getcwd()
	destination = path + "/models/autoencoders"
	num_ae = len(next(os.walk(destination))[1])
	best_relatedness = 0
	model_number = -999
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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



def initialize_new_model(model_init, num_classes, num_of_classes_old):
	weight_info = model_init.classifier[-1].weight.data
	model_init.classifier[-1] = nn.Linear(model_init.classifier[-1].in_features, num_of_classes_old+num_classes)
	kaiming_initilaization(model_init.classifier[-1])
	model_init.classifier[-1].weight[:num_of_classes_old, :] = weight_info
	return model_init 


def train_model(num_classes):
	model_number, best_relatedness = get_initial_model(feature_extractor, dset_loaders, encoder_criterion, use_gpu)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
	
	if (best_relatedness > 0.85):


	else:
		path_to_dir = path + "/model_" + str(model_number) 
		
		file_name = path_to_dir + "/classes.txt" 
		file_object = open(file_name, 'r')
		num_of_classes_old = file_object.read()

		model_init = genmodel(num_of_classes_old)
		model_init.load_state_dict(torch.load(path_to_dir+"/best_performing_model.pth", map_location = device))
		
		ref_model = copy.deepcopy(model_init)
		ref_model.train(False)
		
		model_init = initialize_new_model(model_init, num_classes, num_of_classes_old)
		model_init.to(device)


		#Need to freeze weights see how to do that
		for data in dset_loaders:
			input_data, labels = data

			with torch.no_grad():
				ref_labels = ref_model(input_data)
			


