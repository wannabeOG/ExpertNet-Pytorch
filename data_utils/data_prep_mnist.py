#!/usr/bin/env python
# coding: utf-8

"""
Python module to download the MNIST dataset and split the dataset into 5 tasks to bring the total to 
9 tasks in all for the sequence

"""
import warnings
import os
from pathlib import Path
import shutil
import gdown

import requests, zipfile, io
import time

since = time.time()

dir_list = [['0','1'],['2','3'],['4','5'],['6','7'],['8', '9']]


#dictionary mapping the digits to the task directory
#images belonging to '0' and '1' map to Task_5 directory 
#images belonging to '2' and '3' map to Task_6 directory
#and so on
task_dict = {'0': '5', '1': '5', 
			'2': '6', '3': '6', 
			'4': '7', '5': '7', 
			'6': '8', '7': '8', 
			'8': '9', '9': '9'}


#path to the data file, already been created
path_to_file =  "../Data"
path_to_zip_file = "./mnist_jpgfiles.zip"

#need to use it since G-Drive requires 2 GET Requests for larger files
r = gdown.download("https://drive.google.com/uc?id=1F6xICWB2ZqUouf274xJViMmMvLKC39fA",output="mnist_jpgfiles.zip",quiet=False)

#unzip the MNIST dataset the "Data" folder and delete the zippped directory
with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
	zip_ref.extractall(path_to_file)
os.remove(path_to_zip_file)


#train and test paths
train_path_oracle = os.path.join(path_to_file, "mnist_jpgfiles", "train")
test_path_oracle = os.path.join(path_to_file, "mnist_jpgfiles", "test")


#create directories for all the tasks
for digits in dir_list:
	digit_0 = digits[0]
	digit_1 = digits[1]

	dir_key = task_dict[digit_0]
	
	#create the directory
	task_dir = os.path.join(path_to_file, "Task_" + dir_key)
	os.mkdir(task_dir)

	#create the train and test sub directories within the Task folder
	train_path = os.path.join(task_dir, "train")
	test_path = os.path.join(task_dir, "test")

	os.mkdir(train_path)
	os.mkdir(test_path)

	#create the class directories within the test and train folders
	train_dest_0 = os.path.join(train_path, digit_0)
	train_dest_1 = os.path.join(train_path, digit_1)
	test_dest_0 = os.path.join(test_path, digit_0)
	test_dest_1 = os.path.join(test_path, digit_1)
	os.mkdir(train_dest_0)
	os.mkdir(train_dest_1)
	os.mkdir(test_dest_0)
	os.mkdir(test_dest_1)

#move the files to the appropriate folders
for file in os.listdir(train_path_oracle):
	
	digit = file[6]
	dir_key = task_dict[digit]

	shutil.move(os.path.join(train_path_oracle, file), os.path.join(path_to_file, "Task_" + dir_key, "train", digit))

#move the files to the appropriate folders
for file in os.listdir(test_path_oracle):
	digit = file[6]
	dir_key = task_dict[digit]

	shutil.move(os.path.join(test_path_oracle, file), os.path.join(path_to_file, "Task_" + dir_key, "test", digit))



# delete the unnecessary folders"
shutil.rmtree(os.path.join(path_to_file, "mnist_jpgfiles"))
shutil.rmtree(os.path.join(path_to_file, "__MACOSX"))

total_time = time.time() - since

print ("This process took {} minutes and {} seconds to execute".format(total_time//60, total_time%60))