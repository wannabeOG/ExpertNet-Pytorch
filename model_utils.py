def task_metric(r_errror_comp, r_error_ref):
	""" 
	Inputs: 
		1) r_error_comp = Reconstruction error for model 1
		2) r_error_ref = Reconstruction error for model 2  
		
	Outputs:
		1) task_metric = The task metric that was used in the paper
		   Max value = 1 indicates that the tasks are heavily related
		   Max value = 0 indicates that the tasks are not at all related 
		
	Function: This function returns the task metric

	"""
	return (1-(rerror_ref-r_error_comp)/r_error_comp)


def kaiming_initilaization(layer):
	nn.init.kaiming_normal_(layer.weight, nonlinearity='sigmoid')


def get_initial_model(feature_extractor, dset_loaders, encoder_criterion, use_gpu):
	""" 
	Inputs: 
		1) model_init = A reference to the model which needs to be initialized
		2) num_classes = The number of classes in the new task for which we need to train a expert  
		3) num_of_classes_old = The number of classes in the model that is used as a reference for
		   initializing the new model.  

	Outputs:
		1) autoencoder = A reference to the autoencoder object that is created 
		2) store_path = Path to the directory where the trained model and the checkpoints will be stored

	Function: This function takes in a reference model and initializes a new model with the reference model's
	weights (for the old task) and the weights for the new task are initialized using the kaiming initialization
	method

	"""	
	path = os.getcwd()
	destination = path + "/models/autoencoders"
	num_ae = len(next(os.walk(destination))[1])
	best_relatedness = 0
	model_number = -999
	device = torch.device("cuda:0" if use_gpu else "cpu")
	
	for i in range(num_ae):
		
		model_path = destination + "/autoencoder_"+str(num_ae-i) +"/best_performing_model.pth"
		model = Autoencoder()
		model.load_state_dict(torch.load(model_path), map_location= device)
		
	
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
	""" 
	Inputs: 
		1) model_init = A reference to the model which needs to be initialized
		2) num_classes = The number of classes in the new task for which we need to train a expert  
		3) num_of_classes_old = The number of classes in the model that is used as a reference for
		   initializing the new model.  

	Outputs:
		1) autoencoder = A reference to the autoencoder object that is created 
		2) store_path = Path to the directory where the trained model and the checkpoints will be stored

	Function: This function takes in a reference model and initializes a new model with the reference model's
	weights (for the old task) and the weights for the new task are initialized using the kaiming initialization
	method

	"""	
	weight_info = model_init.classifier[-1].weight.data
	model_init.classifier[-1] = nn.Linear(model_init.classifier[-1].in_features, num_of_classes_old+num_classes)
	kaiming_initilaization(model_init.classifier[-1])
	model_init.classifier[-1].weight[:num_of_classes_old, :] = weight_info
	return model_init 



def model_criterion(preds, labels, flag, T = 2):
	"""
		Temperature is used to produce softer values of probability and 
		this parameter is used only when the flag option is set with the "Distill"
		option
	"""

	if(flag == "CE"):
		loss = nn.CrossEntropyLoss()
		return loss(preds, labels)

	elif(flag == "Distill"):
		
		""" The labels are the teacher scores or the reference
			scores in this case
		"""	
		
		preds = F.softmax(preds)
		labels = F.softmax(labels)

		preds = preds.pow(1/T)
		labels = labels.pow(1/T)

		sum_preds = torch.sum(preds, dim = 1)
		sum_labels = torch.sum(preds, dim = 1)

		sum_preds_ref = torch.t(torch.t(sum_preds).repeat(preds.size(1), 1))
		sum_labels_ref = torch.t(torch.t(sum_labels).repeat(labels.size(1), 1))

		loss = torch.sum(-1*sum_preds_ref*sum_labels_ref, dim = 1)

		return loss
