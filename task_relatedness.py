import torch 
import os

from autoencoder import Autoencoder

def task_metric(rerrror_comp, r_error_ref):
	return (1-(rerror_ref-r_error_comp)/r_error_comp)

def get_initial_model(dset_loaders, encoder_criterion, use_gpu):
	path = os.getcwd()
	destination = path + "/models/autoencoders"
	num_ae = len(next(os.walk(destination))[1])
	best_relatedness = 0
	model_number = -999

	for i in range(num_ae-1):
		model_path = destination + "/autoencoder_"+str(i+1) +"/best_performing_model.pth"
		model = Autoencoder(input_dims)
		model.load_state_dict(torch.load(model_path))
		model.train(False)

		for data in dset_loaders:
			
			input_data, _ = data
			output = model(input_data)
			loss = encoder_criterion(outputs, input_data)
			relatedness = task_metric(loss, rerrror_comp)
			
			if (relatedness > best_relatedness):
				best_relatedness = relatedness
				model_number = i+1


	return model_number, best_relatedness			


def train_model():
	model_number, best_relatedness = get_initial_model(dset_loaders, encoder_criterion, use_gpu)

	if (best_relatedness < 0.85):


	else:
		model = models.alexnet(pretrained = False)
		model.load_state_dict(torch.load(path_to_models + "/model_" + str(model_number) + "best_performing_model.pth"))

		for data in dset_loaders:
			input_data, labels = data
						


