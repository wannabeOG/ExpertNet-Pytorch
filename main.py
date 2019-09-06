import torch
torch.backends.cudnn.benchmark=True

import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as dsets
import torchvision.models as models
import torchvision.transforms as transforms

import argparse
import time
import numpy as np

import copy

from autoencoder import *
from dataloader import *

from encoder_train import *
from encoder_utils import *

from model_train import *
from model_utils import *


parser = argparse.ArgumentParser(description='Expert Gate')
parser.add_argument('--outfile', default='temp_0.1.csv', type=str, help='Output file name')
parser.add_argument('--data_file', default='', help='Path to the data folder to train the model on', type=str)
parser.add_argument('--init_lr', default=0.1, type=float, help='Init learning rate')
parser.add_argument('--num_epochs', default=40, type=int, help='Number of epochs')
parser.add_argument('--batch_size', default=16, type=int, help='Mini batch size')

args = parser.parse_args()



