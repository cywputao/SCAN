from __future__ import print_function
import sys
sys.dont_write_bytecode=True

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed_everything()

def masked_softmax(vec,dim=1, epsilon=1e-5, T=100):
    mask = ~(vec==0)
    mask = mask.bool()
    exps = torch.exp(vec)
    masked_exps = exps * mask.float() 
    masked_exps = masked_exps + epsilon 
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return (masked_exps/masked_sums)

class spatialAttention(nn.Module):
	def __init__(self,domain_parameter,att_dim, delta_bearing,delta_heading, dropout):
		super(spatialAttention,self).__init__()
		self.delta_bearing=delta_bearing
		self.delta_heading=delta_heading
		self.domain = Variable(torch.FloatTensor(int(360/delta_heading),int(360/delta_bearing)).to(device),requires_grad=True)
		self.domain = nn.Parameter(self.domain)
		self.relu=nn.ReLU()
		self.tanh=nn.Tanh()
		self.softmax = nn.Softmax(dim=2)
		if not (dropout==0):
			self.linear = nn.Sequential(nn.Linear(2*att_dim, att_dim), nn.Dropout(p=dropout))
		else:
			self.linear = nn.Linear(2*att_dim, att_dim)
	def forward(self, hidden_state, distance_matrix, bearing_matrix, heading_matrix,sequence_mask):
		num_pedestrians = hidden_state.size(0)
		weights = self.compute_weights(distance_matrix, bearing_matrix, heading_matrix, sequence_mask)
		weighted_hidden = torch.bmm(weights, hidden_state)
		weighted_hidden = torch.cat((weighted_hidden, hidden_state), dim=2)
		weighted_hidden = self.linear(weighted_hidden) 
		weighted_hidden = self.tanh(weighted_hidden)
		return weighted_hidden
	def compute_weights(self, distance_matrix, bearing_matrix, heading_matrix, sequence_mask):
		idx1, idx2 = torch.floor(heading_matrix/self.delta_heading), torch.floor(bearing_matrix/self.delta_bearing)
		idx1, idx2 = idx1.clamp(0, int(360/self.delta_heading)-1), idx2.clamp(0, int(360/self.delta_bearing)-1)
		weights_mask = sequence_mask.unsqueeze(-1).expand(distance_matrix.size())
		weights_mask = weights_mask.mul(weights_mask.transpose(1,2))
		mask_val=0
		weights = self.relu(self.domain[idx1.clone().long(), idx2.clone().long()]-distance_matrix)
		ped_ix = range(weights.size(1))
		ped_mask = torch.ones_like(weights)
		ped_mask[:, ped_ix, ped_ix] = 0
		weights.data.masked_fill_(mask=~ped_mask.bool(), value=mask_val)
		weights.data.masked_fill_(mask=~weights_mask.bool(), value=mask_val)
		if hasattr(self, "softmax"): weights = masked_softmax(weights, dim=2)
		return weights

class temporalAttention(nn.Module):
	def __init__(self,enc_dim, dec_dim, att_dim, sequence_length, dropout=0):
		super(temporalAttention,self).__init__()
		self.sequence_length=sequence_length
		self.linear = nn.Linear(2*dec_dim, dec_dim)
		self.softmax=nn.Softmax(dim=2)
		if not (enc_dim==dec_dim):
			if not (dropout==0):
				self.enc_emb = nn.Sequential(nn.Linear(enc_dim, dec_dim), nn.Dropout(p=dropout))
			else:
				self.enc_emb = nn.Linear(enc_dim, dec_dim)
	def compute_score(self,hidden_encoder,hidden_decoder,sequence_mask):
		score = torch.matmul(hidden_encoder, hidden_decoder.unsqueeze(-1)).squeeze(-1)
		score.data.masked_fill_(mask=~sequence_mask, value=float(-1e24))
		score = self.softmax(score)
		return score
	def forward(self,hidden_encoder,hidden_decoder,sequence_mask):
		if hasattr(self, "enc_emb"):
			enc_emb = self.enc_emb(hidden_encoder)
		else:
			enc_emb = hidden_encoder
		dec_emb = hidden_decoder
		score = self.compute_score(enc_emb, dec_emb, sequence_mask) 
		context_vector = torch.matmul(score.unsqueeze(2), enc_emb)
		attn_features = [dec_emb, context_vector.squeeze(2)]
		attn_ip = torch.cat(attn_features, dim=2)
		output = self.linear(attn_ip)
		output = F.tanh(output)
		return output, score

