# Learning a distance metric for CFair.

import torch
import torch.nn
import torch.optim
import torch.nn.utils as torch_utils
import numpy as np
import pandas as pd

SIZE = 16  # convariance matrix size.
FEATURES = ['rrt', 'gcs', 'sofa_24hours',
			'heart_rate', 'sbp_art', 'dbp_art', 'mbp_cuff', 'resp_rate',
			'temperature', 'spo2', 'glucose', 'gender', 'admission_age',
			'charlson_comorbidity_index', 'apsiii', 'bmi']



class DisLearner(torch.nn.Module):
	
	def __init__(self, size, axis=1):
		"""
		Train a covariance matrix H in distance measurement.
		Given two vector x and y, their distance is represented as x^T * H * y.
		:param size: dimension of features.
		:param axis: axis for normalization in calculating weights.
		"""

		super(DisLearner, self).__init__()
		self.var = torch.nn.Parameter(2 * torch.rand([size, size], requires_grad=True) - 1)
		# self.H = torch.triu(self.var, diagonal=0) + \
		#          torch.triu(self.var, diagonal=1).transpose(0, 1)
		self.axis = axis
	
	def get_H(self):
		H = torch.triu(self.var, diagonal=0) + \
			torch.triu(self.var, diagonal=1).transpose(0, 1)
		return H
	
	def weight_map(self, D, sigma=1.0):
		# transform the pairwise distance tensor to the weights, following 2 steps.
		# 1. map each element d in the tensor to d' = exp(-d/sigma).
		# 2. perform normalization along the specific axis, default: the second axis.
		# output = torch.exp(-D/sigma)  # step 1.
		output = torch.abs(D)
		norm = output.sum(dim=self.axis) + 1e-5  # avoid zero division error.
		output = output / norm.unsqueeze(dim=self.axis)  # step 2.
		return output
	
	def forward(self, Dx_outer_prod, Dx_abs):
		n_0, n_1 = Dx_abs.shape
		H = self.get_H()
		H_expand = H.unsqueeze(0).unsqueeze(0).repeat(n_0, n_1, 1, 1)
		D = torch.sum(Dx_outer_prod * H_expand, dim=(-2, -1))  # pair-wise distances
		A = self.weight_map(D, sigma=1.0)  # weight matrix, shape of n_0, n_1
		out = (A * Dx_abs).sum()  # objective function.
		return out


def load_data(file_0, file_1):

	## Load patient feature matrix.
	df_0 = pd.read_csv(file_0, index_col=0)
	df_1 = pd.read_csv(file_1, index_col=0)
	
	# covert it to tensor and move it to cuda
	X_0 = torch.tensor(df_0[features].values).to(device)  # shape of n_0 * size
	X_1 = torch.tensor(df_1[features].values).to(device)  # shape of n_1 * size
	n_0 = X_0.shape[0]
	n_1 = X_1.shape[0]
	_X_0 = X_0.unsqueeze(1).repeat_interleave(n_1, dim=1)
	_X_1 = X_1.unsqueeze(0).repeat_interleave(n_0, dim=0)
	Dx = _X_0 - _X_1  # shape of n_0 * n_1 * size
	Dx_abs = Dx.abs().sum(dim=-1)  # shape of n_0, n_1
	
	# calculate outer product along the last dimension,
	# using pytorch's broadcasting capabilities.
	Dx_outer_prod = torch.matmul(Dx.unsqueeze(-1), Dx.unsqueeze(-2))  # (n_0, n_1, size, size)
	return Dx_outer_prod, Dx_abs


if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	max_iteration = 20
	# size = 3
	# features = ['rrt', 'heart_rate', 'bmi']
	size = SIZE
	features = FEATURES
	file_0 = "PSM_step_black.csv"
	file_1 = "PSM_step_white.csv"
	
	# load and transform patient feature data.
	Dx_outer_prod, Dx_abs = load_data(file_0, file_1)
	
	# load distance learner model
	DisL = DisLearner(size=size)
	
	# move tensors to gpu device
	DisL = DisL.to(device)
	Dx_abs = Dx_abs.to(device)
	Dx_outer_prod = Dx_outer_prod.to(device)
	
	# train
	optimizer = torch.optim.Adam(DisL.parameters(), lr=0.1)
	for i in range(max_iteration):
		print("Iteration: " + str(i))
		loss = DisL(Dx_outer_prod, Dx_abs)
		
		# compute gradient
		optimizer.zero_grad()
		loss.backward()
		
		# clip gradients
		max_norm = 1.0
		torch_utils.clip_grad_norm_(DisL.parameters(), max_norm)
		
		# update parameters
		optimizer.step()
		print(loss.item())
		if i % 10 == 1:
			torch.save(DisL.get_H().data, 'cov_matrix.pt')