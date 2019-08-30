import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def autoencoder_train():
	since = time.time()
####################### Needs code for loading data into this #########################







########################################################################################
	for epoch in range(start_epoch, num_epochs):
		print ("Epoch {}/{}".format(epoch, num_epochs-1))
		print ("-"*20)

		for data in dset_loaders[phase]:
			input_data, labels = data

			if (use_gpu):
				input_data, labels = Variable(input_data.cuda()), Variable(labels.cuda())
			else:
				input_data, labels = Variable(input_data), Variable(labels)

			optimizer.zero_grad()
			model.zero_grad()

			outputs = model(inputs)
			preds = torch.argmax(outputs, 1)
			loss = criterion(outputs, preds)

			loss.backward()
			optimizer.step()

			running_loss += loss.data[0]
			running_correct_predictions += torch.sum(preds == labels.data)


		epoch_loss = running_loss/dset_size
		epoch_accuracy = running_accuracy/dset_size

		print('Epoch Loss:{}, Epoch Accuracy:{}'.format(epoch_loss, epoch_accuracy))

		epoch_file_name = export_dir+'/'+str(epoch)+'.pth.tar'
		torch.save({
			'epoch': epoch,
			'epoch_loss': epoch_loss, 
			'epoch_accuracy': epoch_accuracy, 
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
           
            }, epoch_file_name)

	elapsed_time = time.time()-since
	print ("This procedure took {:.2f} minutes and {:.2f} seconds".format(elapsed_time//60, elapsed_time%60))
	return model

	
