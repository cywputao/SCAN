from __future__ import print_function
import sys
import os
sys.dont_write_bytecode=True
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import *
from data_utils import *

seed_everything()

device = torch.device("cuda" if torch.cuda.is_available else "cpu")

def get_dirs(args):
	"""
	Initializes directories to save/load models and dataset
	Returns:
		net_dir : directory to save/load model
		data_dir: directory to save/load dataset
	""" 
	net_dir = str(args.model)+'/'+str(args.data)+'/'
	if not os.path.exists(args.model):
		os.mkdir(args.model)
	if not os.path.exists(str(args.model)+'/'+str(args.data)+'/'):
		os.mkdir(str(args.model)+'/'+str(args.data)+'/')
	data_dir = 'data/'+args.data+'/' 
	if args.data=='grandcentral' and (len(glob.glob(data_dir+'train/*'))==0 or args.split_data):
		data_dir = 'raw_data/grandcentral/'
	if not os.path.isdir(data_dir):
		os.makedirs(data_dir)
	return net_dir, data_dir

def get_batch(batch):
	"""
	Batch processing: all tensors to cuda if cuda is available and converts to batch first for batch size=1
	Returns:
		batch : Next batch 
	"""
	batch = [tensor.to(device) for tensor in batch]
	if not len(batch[0].size())==4: # not batch-first
		batch = [tensor.unsqueeze(0) for tensor in batch]
	return batch

def predict(batch,net):
	"""
	Predicts intent for one batch
	Args:
		Batch: Input Batch 
		Net: Model for prediction
	Returns:
		pred (Variable): Model prediction for given batch (batch_size, num_pedestrians, prediction_length, output_size)
		target (Variable): ground truth for given batch (batch_size, num_pedestrians, prediction_length, output_size)
		sequence (Variable): input sequence (batch_size, num_pedestrians, sequence_length, feature_size)
		temporal_attention_dict (Dict): Alignment Vectors for given batch
	""" 
	sequence,target,dist_matrix,bearing_matrix,heading_matrix,ip_mask,op_mask,pedestrians = get_batch(batch) 
	assert(not(pedestrians==0).any()), "No pedestrians in frame"
	sequence, dist_matrix, bearing_matrix, heading_matrix = Variable(sequence, requires_grad=True), Variable(dist_matrix, requires_grad=True), Variable(bearing_matrix, requires_grad=True), Variable(heading_matrix, requires_grad=True)
	pred, temporal_attention_dict = net(sequence, dist_matrix,bearing_matrix,heading_matrix,ip_mask,op_mask)
	assert(not torch.isnan(pred).any())
	target_mask = op_mask.unsqueeze(-1).expand(target.size())
	target = torch.where(~target_mask, torch.zeros_like(target), target)
	pred = torch.where(~target_mask, torch.zeros_like(pred), pred)
	return pred,target,sequence[...,:2],temporal_attention_dict, pedestrians

def get_optimizer(optimizer_type,net,learning_rate,learning_rate_domain=None):
	"""
	Optimizer to optimize model parameters during training
	Args:
		optimizer_type (str): Optimizer to use
		net : model to train
		learning_rate (float): learning rate to use for training
		learning_rate_domain (float): if not None, separate learning rate to use for learning pedestrian domain
	Returns:
		optimizer object
	"""
	optimizer = getattr(torch.optim,optimizer_type)(net.parameters(), lr=learning_rate)
	return optimizer
