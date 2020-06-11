from __future__ import print_function
import os
import sys
sys.dont_write_bytecode=True
import torch
import torch.nn as nn
import math
import numpy as np
from data import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything()

def get_valid_peds(targets):
	"""
	Computes valid denominator value for averaging error
	Args:
		targets (Variable): prediction value from model (batch_size, num_pedestrians, prediction_length, output_size)
	Returns:
		valid_denom (float): number of unmasked valid pedestrians in frame
	"""
	valid_denom = torch.nonzero(targets).size(0)
	return valid_denom/2

def displacement_error(pred,targets,eps=1e-24):
	"""
	Computes displacement error 
	Args:
		pred (Variable): prediction value from model (batch_size, num_pedestrians, prediction_length, output_size)
		targets (Variable): ground truth values (batch_size, num_pedestrians, prediction_length, output_size)
		eps (float): small value to avoid NaN gradients during backpropagation through torch.sqrt
	Returns:
		**multiplication by 10 to convert 1/10th of kilometer to meters***
		dist (Variable): displacement error in meters (batch_size, num_pedestrians, prediction_length)
	"""	
	
	dist = torch.sqrt(((pred-targets)**2).sum(dim=-1)+eps)
	dist = 15*dist
	return dist

def ADE(pred,targets, num_pedestrians):
	"""
	Computes average displacement error (ADE): mean squared displacement error 
	Args:
		pred (Variable): prediction value from model (batch_size, num_pedestrians, prediction_length, output_size)
		targets (Variable): ground truth values (batch_size, num_pedestrians, prediction_length, output_size)
	Returns:
		error (Variable): average displacement error (ADE)
	"""
	num_pedestrians = num_pedestrians.sum()
	dist = displacement_error(pred,targets)
	dist = dist.pow(2)
	prediction_length = pred.size(2)
	error = dist.sum()/(num_pedestrians*pred.size(2))
	return error

def FDE(pred,targets, num_pedestrians):
	"""
	Computes final displacement error (FDE)
	Args:
		pred (Variable): prediction value from model (batch_size, num_pedestrians, prediction_length, output_size)
		targets (Variable): ground truth values (batch_size, num_pedestrians, prediction_length, output_size)
	Returns:
		error (Variable): final displacement error (FDE)
	"""
	num_pedestrians = num_pedestrians.sum()
	dist = displacement_error(pred[...,-1,:],targets[...,-1,:])
	error = dist.sum()/(num_pedestrians)
	return error

def mean_displacement_error(pred,targets, num_pedestrians):
	"""
	Computes mean displacement error 
	Args:
		pred (Variable): prediction value from model (batch_size, num_pedestrians, prediction_length, output_size)
		targets (Variable): ground truth values (batch_size, num_pedestrians, prediction_length, output_size)
	Returns:
		error (Variable): mean displacement error 
	"""
	dist = displacement_error(pred,targets)
	prediction_length = pred.size(2)
	num_pedestrians = num_pedestrians.sum()
	error = dist.sum()/(num_pedestrians*pred.size(2))
	return error

