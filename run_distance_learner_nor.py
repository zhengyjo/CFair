# Learning a distance metric for CFair.

import torch
import torch.nn
import torch.optim
import torch.nn.utils as torch_utils
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

SIZE = 16  # convariance matrix size.
FEATURES = ['rrt_nor', 'gcs_nor', 'sofa_24hours_nor',
       'heart_rate_nor', 'sbp_art_nor', 'dbp_art_nor', 'mbp_cuff_nor',
       'resp_rate_nor', 'temperature_nor', 'spo2_nor', 'glucose_nor',
       'gender_nor', 'admission_age_nor', 'charlson_comorbidity_index_nor',
       'apsiii_nor', 'bmi_nor']



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
	
	def forward(self, Dx_outer_prod, Dx_abs, mask=None):
		n_0, n_1 = Dx_abs.shape
		H = self.get_H()
		H_expand = H.unsqueeze(0).unsqueeze(0).repeat(n_0, n_1, 1, 1)
		D = torch.sum(Dx_outer_prod * H_expand, dim=(-2, -1))  # pair-wise distances
		A = self.weight_map(D, sigma=1.0)  # weight matrix, shape of n_0, n_1
		out = A * Dx_abs
		if mask is not None:
			out = out * mask
		out = out.sum()  # objective function.
		return out


def load_data(file_0, file_1, file_map=None):

	## Load patient feature matrix.
	df_0 = pd.read_csv(file_0, index_col=0)
	df_1 = pd.read_csv(file_1, index_col=0)
	mask = torch.zeros([len(df_0), len(df_1)])

	# covert it to tensor and move it to cuda
	X_0 = torch.tensor(df_0[features].values) # shape of n_0 * size
	X_1 = torch.tensor(df_1[features].values)  # shape of n_1 * size
	n_0 = X_0.shape[0]
	n_1 = X_1.shape[0]
	_X_0 = X_0.unsqueeze(1).repeat_interleave(n_1, dim=1)
	_X_1 = X_1.unsqueeze(0).repeat_interleave(n_0, dim=0)
	Dx = _X_0 - _X_1  # shape of n_0 * n_1 * size

	# load and process mask matrix if map_filt exist.
	if file_map is not None:
		print("Add mask matrix.")
		with open(file_map, 'rb') as f:
			dic_match = pickle.load(f)

		G_0 = set()
		G_1 = set()
		for i in dic_match.keys():
			m = list(dic_match[i])
			if len(m) >= 1:
				G_0.add(i)
				G_1.update(m)
				mask[i, m] = 1

		G_0 = list(G_0)
		G_1 = list(G_1)
		mask = mask[G_0, :][:, G_1]
		Dx = Dx[G_0, :, :][:, G_1, :]
		Dx = Dx * mask[:, :, np.newaxis]

	Dx_abs = Dx.abs().sum(dim=-1)  # shape of n_0, n_1
	
	# calculate outer product along the last dimension,
	# using pytorch's broadcasting capabilities.
	Dx_outer_prod = torch.matmul(Dx.unsqueeze(-1), Dx.unsqueeze(-2))  # (n_0, n_1, size, size)
	if file_map is not None:
		return Dx_outer_prod, Dx_abs, mask
	else:
		return Dx_outer_prod, Dx_abs


if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	max_iteration = 1000
	# size = 3
	# features = ['rrt', 'heart_rate', 'bmi']
	size = SIZE
	features = FEATURES
	file_0 = "black_5_9.csv"
	file_1 = "white_5_9.csv"
	file_map = "./90_psm_matching_candidates_5_12"
	# load and transform patient feature data.
	Dx_outer_prod, Dx_abs, mask = load_data(file_0, file_1, file_map)

	# load distance learner model
	DisL = DisLearner(size=size)
	
	# move tensors to gpu device
	DisL = DisL.to(device)
	Dx_abs = Dx_abs.to(device)
	Dx_outer_prod = Dx_outer_prod.to(device)
	mask = mask.to(device)
	
	# train
	optimizer = torch.optim.Adam(DisL.parameters(), lr=0.1)
	loss_values = []
	for i in range(max_iteration):
		#print("Iteration: " + str(i))
		loss = DisL(Dx_outer_prod, Dx_abs, mask)
		
		# compute gradient
		optimizer.zero_grad()
		loss.backward()
		loss_values.append(loss.item())
		
		# clip gradients
		max_norm = 1.0
		torch_utils.clip_grad_norm_(DisL.parameters(), max_norm)
		
		# update parameters
		optimizer.step()
		#print(loss.item())
		if i % 100 == 1:
			print(i)
			print(loss.item())
			torch.save(DisL.get_H().data, 'cov_matrix.pt')

	# plot the loss values
	plt.figure(figsize=(10, 5), dpi=100)
	plt.plot(loss_values)
	plt.xlabel('Iteration')
	plt.ylabel('Loss')
	plt.title('Loss across Iterations')
	plt.savefig('loss_plot_nor.png')

	# plot heatmap of covaraince matrix.
	cov = DisL.get_H().data.cpu()
	plt.figure(figsize=(10, 10), dpi=100)
	feat_names = ['rrt', 'gcs', 'sofa',
				  'heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate',
				  'temperature', 'spo2', 'glucose', 'gender', 'age',
				  'cci', 'apsiii', 'bmi']
	plt.imshow(cov, cmap="magma_r", interpolation='nearest')
	plt.colorbar()
	plt.yticks(range(len(feat_names)), feat_names, rotation=0)
	plt.xticks(range(len(feat_names)), feat_names, rotation=90)
	plt.title("Heatmap of learned covariance matrix.")
	plt.savefig('heatmap_cov_nor.png')