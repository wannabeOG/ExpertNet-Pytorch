import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
from pathlib import Path
import shutil

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

def convert_tiny_imagenet(path):
	"""
	Called for train/val/test
	"""
	for directory in os.listdir(path):
		path = Path(path)
		if (path.name == 'train'):
			path_to_dir = os.path.join(path, directory)
			dest_path = path_to_dir
			#print (dest_path)
			for file in os.listdir(os.path.join(path_to_dir,  'images')):
				shutil.move(path_to_dir + "/" + "images" + "/" + file, dest_path + "/" + file)
			
			os.rmdir(path_to_dir + "/" + "images")

			for file in os.listdir(path_to_dir):
				if file.endswith('.txt'):
					os.remove(path_to_dir + "/" + file)

		else :
			shutil.rmtree(path)


def split_datasets_tiny_imagenet(base_path, dest_base_path):
	for directory in os.listdir(base_path):
		path_to_file = os.path.join(base_path, directory)    
		if os.path.isdir(path_to_file):
			#Train directory
			for idir in os.listdir(path_to_file):
			#Images directory = working_dir
				working_dir = os.path.join(path_to_file, idir)
				#dest_dir = os.path.join()
				os.mkdir(dest_base_path + "/" + idir)
				no_of_images, images_list = len(next(os.walk(working_dir))[2]), next(os.walk(working_dir))[2]
				#indices = list(range(no_of_images))
				no_of_val_images = math.floor(val_split*no_of_images)
				for i in range(no_of_val_images):
					shutil.move(os.path.join(working_dir,images_list[i]),dest_base_path + "/" + idir) 


data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


class GenDataset(Dataset):

	def __init__(self, path):
		super(GenDataset, self).__init__()
		self.dataset = torchvision.datasets.ImageFolder(path)


	def __len__(self):
		return len(self.dataset['train'])


	def __getitem__(self, index):
		image = self.dataset['train'][index]
		task = random.randint(0, len(self.task_defn) - 1)

		# now sample predictions based on task
		select_index = torch.LongTensor(self.task_defn[task])
		labels = image.gather(0, torch.LongTensor(select_index))
		task = torch.LongTensor([task])
		if self.opt.get('use_gpu'):
			image, task, labels = image.cuda(), task.cuda(), labels.cuda()
		return {'image': image, 'task': task, 'labels': labels}













