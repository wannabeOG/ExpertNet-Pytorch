import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import warnings
from pathlib import Path
import shutil

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


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
		self.transforms = data_transforms
		self.train = False
		
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










