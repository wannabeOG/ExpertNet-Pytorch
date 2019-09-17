import torch.nn as nn
import torch

def encoder_criterion(outputs, inputs):
	loss = nn.MSELoss()
	return loss(outputs, inputs)

def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=10):
	"""
	Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
	
	"""
	lr = init_lr * (0.1**(epoch // lr_decay_epoch))
	print('lr is '+str(lr))

	if (epoch % lr_decay_epoch == 0):
		print('LR is set to {}'.format(lr))

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	return optimizer