def distillation_loss(preds, ref_scores, temperature):


def encoder_criterion(preds, labels):
	loss = nn.MSELoss()
	return loss(outputs, preds)

def model_criterion(preds, labels, flag, T = 2):
	
	if(flag == "CE"):
		loss = nn.CrossEntropyLoss()
		return loss(preds, labels)

	elif(flag == "Distill"):
		softmax = nn.Softmax(dim = 1)
		preds = softmax(preds)
		labels = softmax(labels)

		preds = preds.pow(1/T)
		pred_sum = preds.sum()
		preds = preds/pred_sum
		
		labels = labels.pow(1/T)
		label_sum = labels.sum()
		labels = labels/label_sum

		loss = nn.CrossEntropyLoss()
		return (preds, labels)