from __future__ import print_function
import sys
import os
sys.dont_write_bytecode=True

import torch
import numpy as np
import argparse
import random
import glob
import time
from tqdm import tqdm
from torch.utils.data import DataLoader,random_split

from data import *
import models
from train_utils import *
from data_utils import *
from metrics import *
from itertools import combinations 

np.set_printoptions(precision=2)
torch.set_printoptions(precision=2)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything(seed=12)

parser=argparse.ArgumentParser(description="pedestrian intent modeling")

# Mode
parser.add_argument('--train',action="store_true",help="training mode")
parser.add_argument('--test',action="store_true",help="test mode")

# Dataset Parameters
parser.add_argument('--split_data',action="store_true",help="split data into train, valid, test")
parser.add_argument('--validation_split',type=float,default=0.1,help="validation split")
parser.add_argument('--test_split',type=float,default=0.1,help="test split")
parser.add_argument('--data',type=str,choices=['grandcentral','zara1','zara2','eth','hotel','biwi','univ','stanford'],help="dataset to use")
parser.add_argument('--drop_frames', type=int, default=0, help="number of frames to drop per pedestrian in scene to account for track fragmentation")
parser.add_argument('--delim', type=str, choices=['tab', 'space'], help="delimiter for chosen data")

# Model Parameters
parser.add_argument('--model',type=str,default='model',choices=['spatial_temporal_model','spatial_model','vanilla_lstm','temporal_model'],help="choose model for spatial+temporal     attention, spatial_att_model for only spatial attention")
parser.add_argument('--feature_size',type=int,default=2,help="feature size")
parser.add_argument('--output_size',type=int,default=2,help="output size")
parser.add_argument('--embedding_dim',type=int,default=16, help="embedding dimension")
parser.add_argument('--enc_dim',type=int, default=32,help="encoder dimension")
parser.add_argument('--dec_dim', type=int, default=32,help="decoder dimension")
parser.add_argument('--att_dim', type=int, default=128, help="attn. embedding dimension")
parser.add_argument('--mlp_dim', type=int,default=None,help="mlp dim decoder")
parser.add_argument('--sequence_length',type=int,default=8,help="sequence length")
parser.add_argument('--prediction_length',type=int,default=12,help="prediction length")
parser.add_argument('--min_dist',type=float,default=0.5,help="minimum domain distance")
parser.add_argument('--param_domain',type=float,default=2,help="parameter for domain initialization")
parser.add_argument('--delta_rb',type=float,default=30,help="discretization of bearing angle")
parser.add_argument('--delta_heading',type=float,default=30,help="discretization of heading angle")
parser.add_argument('--dropout',default=0,type=float,help="dropout")

# Training Parameters
parser.add_argument('--use_saved',action="store_true",help="train a trained model further")
parser.add_argument('--epochs',type=int,default=100,help="training epochs")
parser.add_argument('--batch_size',type=int,default=64,help="batch size")
parser.add_argument('--learning_rate',type=float,default=0.001,help="learning rate")
parser.add_argument('--learning_rate_domain',type=float,help="learning rate")
parser.add_argument('--optimizer',type=str,default='Adam',help="optimizer")
parser.add_argument('--best_loss',type=float,default=1000,help="best loss value")
parser.add_argument('--scheduler',action="store_true",default=False,help="use learning rate scheduler")
parser.add_argument('--diff_lr', action="store_true", default=False, help="use different learning rate for learning domain")
parser.add_argument('--init_type', choices=['constant', 'custom'], default="constant", help="domain init type")

# plotting parameters
parser.add_argument('--plot_collisions', action="store_true", help="plot % cllisions")

args=parser.parse_args()

if torch.cuda.is_available():
#	gpu_id = 2
	gpu_id = get_free_gpu()
	torch.cuda.set_device(gpu_id)
	print("Using GPU: {}".format(gpu_id))

def evaluate_model(testdataset,net,netfile,test_batch_size,plot_attn=True,plot_traj=False):
	"""
	Evaluates trained model
	Args:
		testdataset (Dataset): Testing dataset 
		net (model): model architecture to evaluate
		netfile : trained parameters
		test_batch_size (int): batch_size for testing
		plot_attn, plot_traj (bool): True if plot attention scores and/or predicted trajectories else False
	"""
	net.eval()
	ade = float(0)
	mean_error = float(0)
	fde = float(0)
	testloader = DataLoader(testdataset,batch_size=test_batch_size,collate_fn=collate_function(),shuffle=True)
	numTest=len(testloader)
	with torch.no_grad():
		for b, batch in enumerate(testloader):
			pred, target, sequence, context_vector, pedestrians = predict(batch,net)
			ade_batch = ADE(pred,target,pedestrians)
			mean_batch = mean_displacement_error(pred,target,pedestrians)
			fde_batch = FDE(pred,target,pedestrians)
			ade+=ade_batch.item()
			mean_error+=mean_batch.item()
			fde+=fde_batch.item()
	ade/=(b+1)
	mean_error/=(b+1)
	fde/=(b+1)
	print(f'TEST Mean ADE: {mean_error:.2f}\tFDE: {fde:.2f}')
	return mean_error

def evaluate_collisions(testdataset,net,netfile,test_batch_size,thresholds):
	"""
	Evaluates trained model
	Args:
		testdataset (Dataset): Testing dataset 
		net (model): model architecture to evaluate
		netfile : trained parameters
		test_batch_size (int): batch_size for testing
		plot_attn, plot_traj (bool): True if plot attention scores and/or predicted trajectories else False
	"""
	net.eval()
	ade = float(0)
	mean_error = float(0)
	fde = float(0)
	testloader = DataLoader(testdataset,batch_size=test_batch_size,collate_fn=collate_function(),shuffle=True)
	numTest=len(testloader)
	coll_array=[]
	for threshold in thresholds:
		print("Evaluating collisions for threshold: {}".format(threshold))
		with torch.no_grad():
			num_coll = float(0)
			num_total = float(0)
			for b, batch in enumerate(testloader):
				pred, target, sequence, context_vector, pedestrians = predict(batch,net)
				pred = pred.squeeze(0).permute(1,0,2)
				dist_matrix = get_distance_matrix(pred,neighbors_dim=1) 
				count = torch.where(dist_matrix<threshold, torch.ones_like(dist_matrix), torch.zeros_like(dist_matrix))
				count = count.sum()-pedestrians*pred.size(0)
				count = count/2 # each collision is counted twice
				count = count.item()
				if (count>0):
					num_coll+=1
				num_total += 1 
			print(f"Distance Threshold: {threshold}; Num Collisions: {num_coll}; Num Total Situations: {num_total}")
			num_coll_percent = (num_coll/num_total)*100
			print(f"Distance Threshold: {threshold}; Num Collisions: {num_coll}; Num Total Situations: {num_total}; % collisions: {num_coll_percent}%") 
		coll_array+=[num_coll_percent]
	return coll_array

def train(traindataset, validdataset, testdataset, net, netfile, args):
	"""
	Training and validation of model
	"""
	best_loss=args.best_loss
	best_test = float(1000)
	trainloader = DataLoader(traindataset,batch_size=args.batch_size,collate_fn=collate_function(),shuffle=True,num_workers=4)
	validloader=DataLoader(validdataset,batch_size=validdataset.len,collate_fn=collate_function(),shuffle=False)
	plotter = Plotter(args)
	if not args.use_saved and hasattr(net,'spatialAttention'):
		if args.init_type=="constant":
			nn.init.constant_(net.spatialAttention.domain.data,args.param_domain)
		elif args.init_type=="custom":
			custom_init(net.spatialAttention.domain.data,args.param_domain)
	if args.use_saved:
		print(f"Loading Trained Parameters from {netfile}")
		net.load_state_dict(torch.load(netfile))
	if args.diff_lr and hasattr(net, 'spatialAttention'):
		net.spatialAttention.domain.requires_grad_(False)
		params = [p for p in net.parameters() if p.requires_grad]
		net.spatialAttention.domain.requires_grad_(True)
		optimizer = getattr(torch.optim, args.optimizer)([{'params': params},{'params': net.spatialAttention.domain, 'lr': 1e-02}], lr=args.learning_rate, amsgrad=True)
	else:
		optimizer = getattr(torch.optim, args.optimizer)(net.parameters(), lr=args.learning_rate)
	if args.scheduler:
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, 0.8)
	print("Training samples: %d \nValidation Samples: %d \nTesting Samples: %d" %(len(traindataset),len(validdataset),len(testdataset)))
	numBatches=len(trainloader)
	numValBatches=len(validloader)
	print("training...")
	for epoch in range(args.epochs):
		net.train()
		print(f"EPOCH: {epoch+1} ----------->")
		epoch_loss=0
		for b, batch in enumerate(trainloader):
			optimizer.zero_grad()
			pred, target,_,_, pedestrians = predict(batch,net)
			loss = mean_displacement_error(pred,target,pedestrians)
			epoch_loss+=loss.item()
			sys.stdout.write("[Epoch: {}/{}\t{}/{}]: {:.3f}\r".format(epoch+1, args.epochs, b+1, len(trainloader), loss.item()))
			loss.backward()
			optimizer.step()
		epoch_loss/=(b+1)
		print(f"TRAIN Mean ADE: {epoch_loss:.3f}")
		if ((epoch+1)%5)==0 and hasattr(net,'spatialAttention'):
			print("Learned Pedestrian Domain:")
			print((net.spatialAttention.domain.data))
		net.eval()
		valid_loss=0
		with torch.no_grad():
			for b, batch in enumerate(validloader):
				pred, target,_,_,pedestrians = predict(batch,net)
				loss = mean_displacement_error(pred,target,pedestrians)
				assert(not torch.isnan(loss).any() and not torch.isinf(loss).any())
				valid_loss+=loss.item()
				del loss
		valid_loss/=(b+1)
		if args.scheduler:
			scheduler.step()
		print(f"VALID Mean ADE: {valid_loss:.2f}\tBest Validation ADE: {best_loss:.2f}")
		if not (args.data=="stanford"):
			with torch.no_grad():
			#test_loss = 0
				test_loss = evaluate_model(testdataset,net,netfile,testdataset.len)
		else:
			test_loss = 0
		if(valid_loss<best_loss):
			best_loss=valid_loss
			print(colored("Saving Model...","red"))
			torch.save(net.state_dict(),netfile)
			best_test = test_loss 
		plotter.update(epoch_loss,valid_loss, test_loss, best_test)

def main(args):
	print(f"Using {device}")
	print("Initializing model..")
	if args.train or args.test:
		net = getattr(models,args.model)(args.sequence_length,args.prediction_length,args.feature_size,args.embedding_dim,args.enc_dim, args.dec_dim, args.att_dim, args.delta_rb,args.delta_heading,args.param_domain,args.dropout, args.mlp_dim).float().to(device)
		net_dir, data_dir = get_dirs(args)
		traindataset, validdataset,testdataset = load_data(data_dir,args)
		print(len(traindataset), len(validdataset), len(testdataset))
		print("\rDataset Loaded.")
		netfile = "models/{}/{}.pt".format(args.model, args.data)
		print(f"Saving trained parameters at {netfile}")
		if args.train:
			train(traindataset, validdataset, testdataset, net, netfile, args)
		if args.test:
			net.load_state_dict(torch.load(netfile, map_location=device))
			print("testing...")
			with torch.no_grad():
				evaluate_model(testdataset,net,netfile,len(testdataset))	
	elif args.plot_collisions:
		thresholds = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2]
		fig = plt.figure()
		ax = fig.add_subplot(111)
		labels = ['vanillaSCAN', 'SCAN'] 
		colors = ['blue', 'orange', 'red', 'green'] 
		for m, model_ in enumerate(['spatial_model', 'spatial_temporal_model']):
			net = getattr(models, model_)(args.sequence_length,args.prediction_length,args.feature_size,args.embedding_dim,args.enc_dim, args.dec_dim, args.att_dim, args.delta_rb,args.delta_heading,args.param_domain,args.dropout,args.mlp_dim).float().to(device)
			net_dir, data_dir = get_dirs(args)
			traindataset, validdataset,testdataset = load_data(data_dir,args)
			netfile = "models/{}/{}.pt".format(model_, args.data)
			net.load_state_dict(torch.load(netfile, map_location=device))		
			coll_array = evaluate_collisions(testdataset,net,netfile,1, thresholds)
			plt.plot(thresholds, coll_array, marker="o", markersize=7, color = colors[m], linewidth=2, label= "$\it{}$".format(labels[m]))
		plt.xticks(thresholds)
		plt.legend(loc="lower right")
		ax.set_xticklabels(thresholds, rotation=90)
		plt.title(args.data.upper())
		plt.savefig("collisions_{}.pdf".format(args.data),bbox_inches='tight')

if __name__ == '__main__':
	main(args)

