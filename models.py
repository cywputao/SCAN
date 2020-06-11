from __future__ import print_function
import sys
sys.dont_write_bytecode=True

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np
import time
import math

from utils import *
from attention import *

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything()

class modularLSTM(nn.Module):
	def __init__(self,feature_size,lstm_dim):
		super(modularLSTM,self).__init__()
		self.lstm=nn.LSTMCell(feature_size,lstm_dim)
	def forward(self,x,h_t,c_t):
		h_t, c_t = self.lstm(x, (h_t,c_t))
		return h_t,c_t

class vanilla_lstm(nn.Module):
	def __init__(self,sequence_length,prediction_length,feature_size,embedding_dim, enc_dim, dec_dim, att_dim, delta_bearing,delta_heading,domain_parameter, dropout=0, mlp_dim=None):
		super(vanilla_lstm,self).__init__()
		self.embedding_enc_dim = embedding_dim
		self.embedding_dec_dim = embedding_dim
		if not (dropout==0):
			self.encoder_embedding = nn.Sequential(nn.Linear(feature_size, self.embedding_enc_dim), nn.Dropout(p=dropout))
			self.decoder_embedding = nn.Sequential(nn.Linear(feature_size, self.embedding_dec_dim), nn.Dropout(p=dropout))
		else:
			self.encoder_embedding = nn.Linear(feature_size, self.embedding_enc_dim)
			self.decoder_embedding = nn.Linear(feature_size, self.embedding_dec_dim) 
		self.encoder = modularLSTM(self.embedding_enc_dim, enc_dim)
		self.decoder = modularLSTM(self.embedding_dec_dim,dec_dim)
		self.sequence_length=sequence_length
		self.prediction_length=prediction_length
		self.enc_dim=enc_dim
		self.dec_dim = dec_dim
		#self.act = nn.Tanh()
		self.feature_size=feature_size
		self.delta_bearing=delta_bearing
		self.delta_heading=delta_heading
		if mlp_dim:
			self.out = nn.Sequential(nn.Linear(self.dec_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, self.feature_size),nn.Tanh())
		else:
			self.out = nn.Sequential(nn.Linear(self.dec_dim, self.feature_size), nn.Tanh())
	def init_states(self, batch_size, num_pedestrians, lstm_dim):
		h_t = Variable(torch.zeros(batch_size*num_pedestrians, lstm_dim), requires_grad=True).to(device)
		c_t = Variable(torch.zeros(batch_size*num_pedestrians, lstm_dim), requires_grad=True).to(device)
		return h_t, c_t
	def encode(self, hidden_state, cell_state, sequence, distance, bearing, heading, input_mask, num_pedestrians): 
		sequence_emb = self.encoder_embedding(sequence)
		if hasattr(self, "act"):
			sequence_emb = self.act(sequence_emb)
		hidden_state, cell_state = self.encoder(sequence_emb.view(-1, self.embedding_enc_dim), hidden_state, cell_state)
		return hidden_state, cell_state
	def decode(self,hidden_state, cell_state,previous_sequence,encoded_input,input_mask,alignmentVector,prediction_timestep,batch_size,num_pedestrians,op_mask):
		prediction = torch.FloatTensor(batch_size, num_pedestrians, self.feature_size).to(device)
		prediction_mask = op_mask.unsqueeze(-1).expand(prediction.size())
		sequence_emb = self.decoder_embedding(previous_sequence)
		if hasattr(self, "act"):
			sequence_emb = self.act(sequence_emb)
		hidden_state, cell_state = self.decoder(sequence_emb.view(-1, self.embedding_dec_dim),hidden_state, cell_state)
		prediction = self.out(hidden_state.view(-1, num_pedestrians, self.dec_dim))
		return hidden_state, cell_state, prediction
	def forward(self,sequence,dist_matrix,bearing_matrix,heading_matrix,seq_mask,op_mask):
		alignmentVector = {}
		batch_size, num_pedestrians,_,_ = list(sequence.size())
		hidden_state, cell_state = self.init_states(batch_size, num_pedestrians, self.enc_dim)
		encoded_input = []
		for i in range(self.sequence_length):
			hidden_state, cell_state = self.encode(hidden_state, cell_state, sequence[:,:,i,...], dist_matrix[:,:,i,...], bearing_matrix[:,:,i,...], heading_matrix[:,:,i,...], seq_mask[:,:,i,...],num_pedestrians)
		previous_sequence = sequence[:,:,-1,...]
		output = torch.FloatTensor(batch_size, num_pedestrians, self.prediction_length, self.feature_size).to(device)
		_, cell_state = self.init_states(batch_size, num_pedestrians, self.dec_dim)
		for i in range(self.prediction_length):
			hidden_state, cell_state, previous_sequence = self.decode(hidden_state,cell_state,previous_sequence,encoded_input,seq_mask,alignmentVector,i,batch_size,num_pedestrians,seq_mask[:,:,-1])
			output[:,:,i,...] = previous_sequence
		return output, alignmentVector


class spatial_model(nn.Module):
	def __init__(self,sequence_length,prediction_length,feature_size,embedding_dim, enc_dim, dec_dim, att_dim, delta_bearing,delta_heading,domain_parameter, dropout=0, mlp_dim=None):
		super(spatial_model,self).__init__()
		self.embedding_enc_dim = embedding_dim
		self.embedding_dec_dim = embedding_dim
		if not (dropout==0):
			self.encoder_embedding = nn.Sequential(nn.Linear(feature_size, self.embedding_enc_dim), nn.Dropout(p=dropout))
			self.decoder_embedding = nn.Sequential(nn.Linear(feature_size, self.embedding_dec_dim), nn.Dropout(p=dropout))
		else:
			self.encoder_embedding = nn.Linear(feature_size, self.embedding_enc_dim)
			self.decoder_embedding = nn.Linear(feature_size, self.embedding_dec_dim) 
		self.encoder = modularLSTM(self.embedding_enc_dim, enc_dim)
		self.decoder = modularLSTM(self.embedding_dec_dim,dec_dim)
		self.sequence_length=sequence_length
		self.prediction_length=prediction_length
		self.enc_dim=enc_dim
		self.dec_dim = dec_dim
		self.feature_size=feature_size
		self.spatialAttention = spatialAttention(domain_parameter,att_dim,delta_bearing,delta_heading, dropout)
		if not (enc_dim==att_dim):
			if not (dropout==0):
				self.spatial_enc_in = nn.Sequential(nn.Linear(enc_dim, att_dim), nn.Dropout(p=dropout))
				self.enc_spatial_out = nn.Sequential(nn.Linear(att_dim, enc_dim), nn.Dropout(p=dropout))
			else:
				self.spatial_enc_in = nn.Linear(enc_dim, att_dim)
				self.enc_spatial_out = nn.Linear(att_dim, enc_dim)
		if not (dec_dim==att_dim):
			if not (dropout==0):
				self.spatial_dec_in = nn.Sequential(nn.Linear(dec_dim, att_dim), nn.Dropout(p=dropout))
				self.spatial_dec_out = nn.Sequential(nn.Linear(att_dim, dec_dim), nn.Dropout(p=dropout))
			else:
				self.spatial_dec_in = nn.Linear(dec_dim, att_dim)
				self.spatial_dec_out= nn.Linear(att_dim, dec_dim)
		self.delta_bearing=delta_bearing
		self.delta_heading=delta_heading
		if mlp_dim:
			self.out = nn.Sequential(nn.Linear(self.dec_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, self.feature_size),nn.Tanh())
		else:
			self.out = nn.Sequential(nn.Linear(self.dec_dim, self.feature_size), nn.Tanh())
	def init_states(self, batch_size, num_pedestrians, lstm_dim):
		h_t = Variable(torch.zeros(batch_size*num_pedestrians, lstm_dim), requires_grad=True).to(device)
		c_t = Variable(torch.zeros(batch_size*num_pedestrians, lstm_dim), requires_grad=True).to(device)
		return h_t, c_t
	def encode(self, hidden_state, cell_state, sequence, distance, bearing, heading, input_mask, num_pedestrians): 
		hidden_state = hidden_state.view(-1, num_pedestrians, self.enc_dim) 
		if hasattr(self, "spatial_enc_in"):
			hidden_state = self.spatial_enc_in(hidden_state) 
			if hasattr(self, "act"):
				hidden_state = self.act(hidden_state)
		weighted_hidden = self.spatialAttention(hidden_state, distance, bearing, heading, input_mask)
		if hasattr(self, "enc_spatial_out"):
			weighted_hidden = self.enc_spatial_out(weighted_hidden)
		sequence_emb = self.encoder_embedding(sequence)
		if hasattr(self, "act"):
			sequence_emb = self.act(sequence_emb)
		hidden_state, cell_state = self.encoder(sequence_emb.view(-1, self.embedding_enc_dim), weighted_hidden.view(-1, self.enc_dim), cell_state)
		return weighted_hidden, hidden_state, cell_state
	def decode(self,hidden_state, cell_state,previous_sequence,weighted_hidden,encoded_input,input_mask,alignmentVector,prediction_timestep,batch_size,num_pedestrians,op_mask):
		if not (prediction_timestep==0) and hasattr(self, "spatial_dec_out"):		
			weighted_hidden = self.spatial_dec_out(weighted_hidden)
		prediction = torch.FloatTensor(batch_size, num_pedestrians, self.feature_size).to(device)
		prediction_mask = op_mask.unsqueeze(-1).expand(prediction.size())
		sequence_emb = self.decoder_embedding(previous_sequence)
		if hasattr(self, "act"):
			sequence_emb = self.act(sequence_emb)
		hidden_state, cell_state = self.decoder(sequence_emb.view(-1, self.embedding_dec_dim), weighted_hidden.view(-1, self.dec_dim), cell_state)
		hidden_state = hidden_state.view(-1, num_pedestrians, self.dec_dim)
		prediction = self.out(hidden_state)
		return hidden_state, cell_state, prediction
	def forward(self,sequence,dist_matrix,bearing_matrix,heading_matrix,seq_mask,op_mask):
		alignmentVector = {}
		batch_size, num_pedestrians,_,_ = list(sequence.size())
		hidden_state, cell_state = self.init_states(batch_size, num_pedestrians, self.enc_dim)
		encoded_input = []
		for i in range(self.sequence_length):
			encoded_hidden, hidden_state, cell_state = self.encode(hidden_state, cell_state, sequence[:,:,i,...], dist_matrix[:,:,i,...], bearing_matrix[:,:,i,...], heading_matrix[:,:,i,...], seq_mask[:,:,i,...],num_pedestrians)
			encoded_input+=[encoded_hidden.unsqueeze(2)]
		encoded_input = torch.cat(encoded_input,dim=2)
		previous_sequence = sequence[:,:,-1,...]
		output = torch.FloatTensor(batch_size, num_pedestrians, self.prediction_length, self.feature_size).to(device)
		hidden_state = hidden_state.view(-1, num_pedestrians, self.enc_dim)
		if hasattr(self, 'spatial_enc_in'):
			hidden_state = self.spatial_enc_in(hidden_state)
			if hasattr(self, "act"):
				hidden_state=self.act(hidden_state)
		weighted_hidden = self.spatialAttention(hidden_state, dist_matrix[:,:,-1,...], bearing_matrix[:,:,-1,...], heading_matrix[:,:,-1,...], seq_mask[:,:,-1,...])
		if hasattr(self, 'spatial_dec_out'):
			weighted_hidden = self.spatial_dec_out(weighted_hidden)
		_, cell_state = self.init_states(batch_size, num_pedestrians, self.dec_dim)
		for i in range(self.prediction_length):
			hidden_state, cell_state, previous_sequence = self.decode(hidden_state,cell_state,previous_sequence,weighted_hidden,encoded_input,seq_mask,alignmentVector,i,batch_size,num_pedestrians,seq_mask[:,:,-1])
			output[:,:,i,...] = previous_sequence
			if not (i==0):
				distance, bearing, heading = get_features(previous_sequence,1,output[:,:,i-1,...])
			else:
				distance, bearing, heading = get_features(previous_sequence,1,sequence[:,:,-1,...])
			hidden_state = hidden_state.view(-1, num_pedestrians, self.dec_dim)
			if hasattr(self, 'spatial_dec_in'):
				hidden_state = self.spatial_dec_in(hidden_state)
				if hasattr(self, "act"):
					hidden_state=self.act(hidden_state)
			weighted_hidden = self.spatialAttention(hidden_state, distance,bearing,heading, seq_mask[:,:,-1])
		return output, alignmentVector



class spatial_temporal_model(nn.Module):
	def __init__(self,sequence_length,prediction_length,feature_size,embedding_dim, enc_dim, dec_dim, att_dim, delta_bearing,delta_heading,domain_parameter, dropout=0, mlp_dim=None):
		super(spatial_temporal_model,self).__init__()
		self.embedding_enc_dim = embedding_dim
		self.embedding_dec_dim = embedding_dim
		if not (dropout==0):
			self.encoder_embedding = nn.Sequential(nn.Linear(feature_size, self.embedding_enc_dim), nn.Dropout(p=dropout))
			self.decoder_embedding = nn.Sequential(nn.Linear(feature_size, self.embedding_dec_dim), nn.Dropout(p=dropout))
		else:
			self.encoder_embedding = nn.Linear(feature_size, self.embedding_enc_dim)
			self.decoder_embedding = nn.Linear(feature_size, self.embedding_dec_dim)
		self.encoder = modularLSTM(self.embedding_enc_dim, enc_dim)
		self.decoder = modularLSTM(self.embedding_dec_dim,dec_dim)
		self.sequence_length=sequence_length
		self.prediction_length=prediction_length
		self.enc_dim=enc_dim
		self.dec_dim = dec_dim
		self.feature_size=feature_size
		self.temporalAttention = temporalAttention(enc_dim, dec_dim, att_dim, sequence_length, dropout)
		self.spatialAttention = spatialAttention(domain_parameter,att_dim,delta_bearing,delta_heading, dropout)
		if not (enc_dim==att_dim):
			if not (dropout==0):
				self.spatial_enc_in = nn.Sequential(nn.Linear(enc_dim, att_dim), nn.Dropout(p=dropout))
				self.enc_spatial_out = nn.Sequential(nn.Linear(att_dim, enc_dim), nn.Dropout(p=dropout))
			else:
				self.spatial_enc_in = nn.Linear(enc_dim, att_dim)
				self.enc_spatial_out = nn.Linear(att_dim, enc_dim)
		if not (dec_dim==att_dim):
			if not (dropout==0):
				self.spatial_dec_in = nn.Sequential(nn.Linear(dec_dim, att_dim), nn.Dropout(p=dropout))
				self.spatial_dec_out = nn.Sequential(nn.Linear(att_dim, dec_dim), nn.Dropout(p=dropout))
			else:
				self.spatial_dec_in = nn.Linear(dec_dim, att_dim)
				self.spatial_dec_out= nn.Linear(att_dim, dec_dim)
		self.delta_bearing=delta_bearing
		self.delta_heading=delta_heading
		if mlp_dim:
			self.out = nn.Sequential(nn.Linear(self.dec_dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, self.feature_size),nn.Tanh())
		else:
			self.out = nn.Sequential(nn.Linear(self.dec_dim, self.feature_size), nn.Tanh())
	def init_states(self, batch_size, num_pedestrians, lstm_dim):
		h_t = Variable(torch.rand(batch_size*num_pedestrians, lstm_dim), requires_grad=True).to(device)
		c_t = Variable(torch.rand(batch_size*num_pedestrians, lstm_dim), requires_grad=True).to(device)
		return h_t, c_t
	def encode(self, hidden_state, cell_state, sequence, distance, bearing, heading, input_mask, num_pedestrians): 
		hidden_state = hidden_state.view(-1, num_pedestrians, self.enc_dim) 
		if hasattr(self, "spatial_enc_in"):
			hidden_state = self.spatial_enc_in(hidden_state) 
		weighted_hidden = self.spatialAttention(hidden_state, distance, bearing, heading, input_mask)
		if hasattr(self, "enc_spatial_out"):
			weighted_hidden = self.enc_spatial_out(weighted_hidden)
		sequence_emb = self.encoder_embedding(sequence)
		if hasattr(self, "act"):
			sequence_emb = self.act(sequence_emb)
		hidden_state, cell_state = self.encoder(sequence_emb.view(-1, self.embedding_enc_dim), weighted_hidden.view(-1, self.enc_dim), cell_state)
		return weighted_hidden, hidden_state, cell_state
	def decode(self,hidden_state, cell_state,previous_sequence,weighted_hidden,encoded_input,input_mask,alignmentVector,prediction_timestep,batch_size,num_pedestrians,op_mask):
		if not (prediction_timestep==0) and hasattr(self, "spatial_dec_out"):		
			weighted_hidden = self.spatial_dec_out(weighted_hidden)
		prediction = torch.FloatTensor(batch_size, num_pedestrians, self.feature_size).to(device)
		prediction_mask = op_mask.unsqueeze(-1).expand(prediction.size())
		attended_weighted_ped, alignment_vector = self.temporalAttention(encoded_input, weighted_hidden, input_mask)
		sequence_emb = self.decoder_embedding(previous_sequence)
		if hasattr(self, "act"):
			sequence_emb = self.act(sequence_emb)
		hidden_state, cell_state = self.decoder(sequence_emb.view(-1, self.embedding_dec_dim), attended_weighted_ped.view(-1, self.dec_dim), cell_state)
		hidden_state = hidden_state.view(-1, num_pedestrians, self.dec_dim)
		prediction = self.out(hidden_state)
		alignmentVector[prediction_timestep] = alignment_vector
		return hidden_state, cell_state, prediction, alignmentVector 
	def forward(self,sequence,dist_matrix,bearing_matrix,heading_matrix,seq_mask,op_mask):
		alignmentVector = {}
		batch_size, num_pedestrians,_,_ = list(sequence.size())
		hidden_state, cell_state = self.init_states(batch_size, num_pedestrians, self.enc_dim)
		encoded_input = []
		for i in range(self.sequence_length):
			encoded_hidden, hidden_state, cell_state = self.encode(hidden_state, cell_state, sequence[:,:,i,...], dist_matrix[:,:,i,...], bearing_matrix[:,:,i,...], heading_matrix[:,:,i,...], seq_mask[:,:,i,...],num_pedestrians)
			encoded_input+=[encoded_hidden.unsqueeze(2)]
		encoded_input = torch.cat(encoded_input,dim=2)
		previous_sequence = sequence[:,:,-1,...]
		output = torch.FloatTensor(batch_size, num_pedestrians, self.prediction_length, self.feature_size).to(device)
		hidden_state = hidden_state.view(-1, num_pedestrians, self.enc_dim)
		if hasattr(self, 'spatial_enc_in'):
			hidden_state = self.spatial_enc_in(hidden_state)
		weighted_hidden = self.spatialAttention(hidden_state, dist_matrix[:,:,-1,...], bearing_matrix[:,:,-1,...], heading_matrix[:,:,-1,...], seq_mask[:,:,-1,...])
		if hasattr(self, 'spatial_dec_out'):
			weighted_hidden = self.spatial_dec_out(weighted_hidden)
		_, cell_state = self.init_states(batch_size, num_pedestrians, self.dec_dim)
		for i in range(self.prediction_length):
			hidden_state, cell_state, previous_sequence, alignmentVector = self.decode(hidden_state,cell_state,previous_sequence,weighted_hidden,encoded_input,seq_mask,alignmentVector,i,batch_size,num_pedestrians,seq_mask[:,:,-1])
			output[:,:,i,...] = previous_sequence
			if not (i==0):
				distance, bearing, heading = get_features(previous_sequence,1,output[:,:,i-1,...])
			else:
				distance, bearing, heading = get_features(previous_sequence,1,sequence[:,:,-1,...])
			hidden_state = hidden_state.view(-1, num_pedestrians, self.dec_dim)
			if hasattr(self, 'spatial_dec_in'):
				hidden_state = self.spatial_dec_in(hidden_state)
			weighted_hidden = self.spatialAttention(hidden_state, distance,bearing,heading, seq_mask[:,:,-1])
		return output, alignmentVector


