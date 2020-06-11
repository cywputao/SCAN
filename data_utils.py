from __future__ import print_function
import sys
sys.dont_write_bytecode=True
import warnings
import torch
import glob
from torch.utils.data import random_split
from utils import *
import pandas as pd
from data import *
from termcolor import colored

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything()

def split_data(files, args):
	"""
	Split data into train, valid, test datasets (used only when dataset is grandcentral)
	Returns:
		traindataset, validdataset, testdataset
	"""
	data = dataset(files,args)
	data_size=len(data)
	valid_size = int(np.floor(0.1*data_size))
	test_size=valid_size
	train_size = data_size-valid_size-test_size
	traindataset, validdataset, testdataset = random_split(data, [train_size, valid_size, test_size])
	return traindataset, validdataset, testdataset

def load_data(data_dir,args):
	"""
	Loads training, validation, test datasets 
	Returns:
		traindataset, validdataset, testdataset
	"""
	if data_dir=='new_data/grandcentral/':
		csv_files = glob.glob(data_dir+'*.csv')
		traindataset, validdataset, testdataset = split_data(csv_files,args)
		torch.save(traindataset, 'data/grandcentral/train/traindataset.pt')
		torch.save(validdataset, 'data/grandcentral/val/validdataset.pt')
		torch.save(testdataset, 'data/grandcentral/test/testdataset.pt')
	elif data_dir=='data/grandcentral/':
		traindataset = torch.load(data_dir+'train/traindataset.pt')
		validdataset = torch.load(data_dir+'val/validdataset.pt')
		testdataset = torch.load(data_dir+'test/testdataset.pt')
	else:
		dataset_dir = 'datasets/'+str(args.data)+'/'
		if len(glob.glob(dataset_dir+'*'))==0 or args.split_data or not(args.drop_frames==0):
			print(colored("Initializing train, valid, test datasets for given args","blue"))
			print(colored(f"Processing training data from {data_dir}train", "blue"))
			train_files = glob.glob(data_dir+"train/*.txt")
			print(colored(f"Processing validation data from {data_dir}val", "blue"))
			valid_files = glob.glob(data_dir+'val/*.txt')
			print(colored(f"Processing test data from {data_dir}test", "blue"))
			test_files = glob.glob(data_dir+'test/*txt')
			traindataset = dataset(train_files,args)
			validdataset = dataset(valid_files,args)
			if (args.data=="stanford"):
				testdataset = traindataset
			else:
				testdataset = dataset(test_files,args)
		else:
			print(colored("loading datasets from " + str(dataset_dir),"blue"))
			traindataset = torch.load(dataset_dir+'train.pt')
			validdataset = torch.load(dataset_dir+'valid.pt')
			testdataset = torch.load(dataset_dir+'test.pt')
	return traindataset, validdataset, testdataset





