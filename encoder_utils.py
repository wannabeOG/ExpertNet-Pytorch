def encoder_criterion(preds, labels):
	loss = nn.MSELoss()
	return loss(outputs, preds)
