import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
init = nn.init.xavier_uniform_
import gc
from info_nce import InfoNCE


class DataHandler:
	def __init__(self):
		if args.data == 'yelp':
			predir = '../Data/yelp/'
		if args.data == 'yelp-14k':
			predir = '../Data/yelp-14k/'
		if args.data == 'yelp3':
			predir = '../Data/yelp3/'
		if args.data == 'yelp-small':
			predir = '../Data/yelp-small/'
		if args.data == 'sparse_yelp-small':
			predir = '../Data/sparse_yelp-small/'
		if args.data == 'sparse_gowalla-5k':
			predir = '../Data/sparse_gowalla-5k/'
		if args.data == 'sparse_gowalla-5k-005':
			predir = '../Data/sparse_gowalla-5k-noise005/'
		if args.data == 'sparse_gowalla-5k-01':
			predir = '../Data/sparse_gowalla-5k-noise01/'
		if args.data == 'sparse_gowalla-5k-015':
			predir = '../Data/sparse_gowalla-5k-noise015/'
		if args.data == 'sparse_gowalla-5k-02':
			predir = '../Data/sparse_gowalla-5k-noise02/'
		if args.data == 'yelp-small2-n005':
			predir = '../Data/yelp-small2-n005/'
		if args.data == 'yelp-small2-n01':
			predir = '../Data/yelp-small2-n01/'
		if args.data == 'yelp-small2-n015':
			predir = '../Data/yelp-small2-n015/'
		if args.data == 'yelp-small2-n02':
			predir = '../Data/yelp-small2-n02/'
		if args.data == 'yelp-small2':
			predir = '../Data/yelp-small2/'
		elif args.data == 'amazon':
			predir = '../Data/amazon/'
		elif args.data == 'amazon2':
			predir = '../Data/amazon2/'
		elif args.data == 'amazon-20k':
			predir = '../Data/amazon-20k/'
		if args.data == 'sparse_amazon-small':
			predir = '../Data/sparse_amazon-small/'
		elif args.data == 'amazon-small':
			predir = '../Data/amazon-small/'
		elif args.data == 'ml10m2':
			predir = '../Data/ml10m2/'
		elif args.data == 'ml10m':
			predir = '../Data/ml10m/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'

	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat):
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		# mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)
		# make cuda tensor
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).cuda()
    
	def Threshold(self, x, threshold, value):
		return t.where(x > threshold, x, t.tensor(value, device=x.device))
    
	def convert_to_torch_sparse(self, coo_matrix):
		indices = t.tensor(np.vstack((coo_matrix.row, coo_matrix.col)), dtype=t.long)
		values = t.tensor(coo_matrix.data, dtype=t.float32)
		shape = t.Size(coo_matrix.shape)
		return t.sparse_coo_tensor(indices, values, shape)

	def makeHyperAdj(self, mat):
		k_intra = args.hyperk
		# ハイパーグラフを構築
		mat = mat.to(device="cuda:0")
		hypergraph, hypergraph_temp, hypergraph_WL = self.HyperGraph(mat, k_intra)
		# hypergraph, _, _ = self.HyperGraph(mat, k_intra)
		return hypergraph, hypergraph_temp, hypergraph_WL
		# return hypergraph, _, _
  
	def makeRingHyperAdj(self, mat):
		k_intra = args.hyperk
		# ハイパーグラフを構築
		mat = mat.to(device="cuda:0")

		hypergraph, hypergraph_temp, hypergraph_WL = self.HyperGraph(mat, k_intra)
		_, _, hypergraph_WL_2 = self.HyperGraph(mat, k_intra-1)
		_, _, hypergraph_WL_1 = self.HyperGraph(mat, k_intra-2)
		zeros_column = t.zeros((hypergraph_WL.shape[0], int(hypergraph_WL.shape[1]/k_intra))).to(device="cuda:0")
  
		slices = int(hypergraph_WL.shape[1] / k_intra)

		sliced_hypergraph_WL_1 = hypergraph_WL[:, :slices]
		sliced_hypergraph_WL_2 = hypergraph_WL[:, -slices:]
		hypergraph_WL_2_0 = t.concat([hypergraph_WL_2, zeros_column], axis=1)
		hypergraph_WL_1_0 = t.concat([hypergraph_WL_1, zeros_column], axis=1)
		hypergraph_WL_1_0 = t.concat([hypergraph_WL_1_0, zeros_column], axis=1)
  
		hypergraph_WL_1_ = t.concat([hypergraph_WL_1, sliced_hypergraph_WL_1], axis=1)
		hypergraph_WL_1_ = t.concat([hypergraph_WL_1_, zeros_column], axis=1)
		hypergraph_WL_2_ = t.concat([hypergraph_WL_2, sliced_hypergraph_WL_2], axis=1)
		


		hypergraph_WL_3_2 = hypergraph_WL - hypergraph_WL_2_
		hypergraph_WL_2_1 = hypergraph_WL_2_0 - hypergraph_WL_1_
		hypergraph_WL_0 = hypergraph_WL_3_2 + hypergraph_WL_2_1 + hypergraph_WL_1_0
		hypergraph_WL_0 = hypergraph_WL_1_0
		# hypergraph_WL_0 = t.rand(hypergraph_WL.shape[0], hypergraph_WL.shape[1], device="cuda:0")
		# hypergraph, _, _ = self.HyperGraph(mat, k_intra)
		# print(hypergraph_WL.type())
		# print(hypergraph_WL.size())
		hypergraph_, hypergraph_temp = self.Laplcian_matrix(hypergraph_WL_0)
		hypergraph, hypergraph_temp = self.Laplcian_matrix(hypergraph_WL)
		print(t.equal(hypergraph_WL, hypergraph_WL_0))
		print(t.equal(hypergraph, hypergraph_))
		return hypergraph_, hypergraph_temp, hypergraph_WL
		# return hypergraph, _, _

	def makeEncodeuHyperAdj(self, mat):
		k_intra = 2
		# ハイパーグラフを構築
		mat = mat.to(device="cuda:0")
		hypergraph, hypergraph_temp, hypergraph_WL = self.EncodeuHyperGraph(mat, k_intra)
		return hypergraph, hypergraph_temp, hypergraph_WL
    
    
	def makeEncodeiHyperAdj(self, mat):
		k_intra = 2
		# ハイパーグラフを構築
		mat = mat.to(device="cuda:0")
		hypergraph, hypergraph_temp, hypergraph_WL = self.EncodeiHyperGraph(mat, k_intra)
		return hypergraph, hypergraph_temp, hypergraph_WL

	def Laplcian_matrix(self, H):
		D_uv = t.diag(H.sum(axis=1))
		D_u = t.diag(t.mm(H, t.diag(H.sum(axis=0))).sum(axis=0))

		D_uv_div = t.nan_to_num(t.pow(D_uv.diag(), -1/2), posinf=0, neginf=0).diag()
		D_u_div = t.nan_to_num(t.pow(D_u.diag(), -1/2), posinf=0, neginf=0).diag()

		temp = t.mm(D_uv_div, H)
		temp = t.mm(temp, D_u_div)
		temp = t.mm(temp, H.T)
		H_Lap = t.mm(temp, D_uv_div)

		edgetemp = t.mm(H.T, D_uv_div)
		#H_Lap = t.mm(temp, D_u_div)
		'''
		# version 1.0
		H_Lap = torch.mm(H, H.T)
		H_L = torch.diag(H_Lap.sum(axis=0)) - H_Lap
		D_div = torch.nan_to_num(torch.pow(H_Lap.diag(), -1/2), posinf=0, neginf=0).diag()
		temp = torch.mm(D_div, H_L)
		H_Lap = torch.mm(H_L, D_div)
		'''
		return H_Lap, edgetemp

	# def Laplcian_matrix_edge(self, H):
	# 	D_uv = t.diag(H.sum(axis=1))
	# 	D_u = t.diag(t.mm(H, t.diag(H.sum(axis=0))).sum(axis=0))

	# 	D_uv_div = t.nan_to_num(t.pow(D_uv.diag(), -1/2), posinf=0, neginf=0).diag()
	# 	D_u_div = t.nan_to_num(t.pow(D_u.diag(), -1/2), posinf=0, neginf=0).diag()

	# 	temp = t.mm(D_uv_div, H)
	# 	temp = t.mm(temp, D_u_div)
	# 	# temp = t.mm(temp, H.T)
	# 	# H_Lap = t.mm(temp, D_uv_div)
	# 	#H_Lap = t.mm(temp, D_u_div)
	# 	'''
	# 	# version 1.0
	# 	H_Lap = torch.mm(H, H.T)
	# 	H_L = torch.diag(H_Lap.sum(axis=0)) - H_Lap
	# 	D_div = torch.nan_to_num(torch.pow(H_Lap.diag(), -1/2), posinf=0, neginf=0).diag()
	# 	temp = torch.mm(D_div, H_L)
	# 	H_Lap = torch.mm(H_L, D_div)
	# 	'''
	# 	return temp
        
	def HyperGraph(self, R, k_intra):
		def Redundant_remove(x):
			temp = self.Threshold(x, t.median(x.diag()), 0)
			F_k = self.Threshold(temp, 0.0000001, 1)
			return F_k

		H_lst = [R.to_dense()]
		D_u = t.zeros(R.shape[1], R.shape[1]).cuda()
		for k in range(k_intra-1):
			dense_R = R.to_dense()

			H_k = t.mm(dense_R, self.Threshold(t.matrix_power(t.mm(dense_R.T, dense_R).to_dense(), k), 1, 1))
			D_uv_k = t.diag(H_k.sum(axis=0))
			D_u_k = t.diag(t.mm(H_k, D_uv_k).sum(axis=0))
			D_u += D_u_k

			if k > 0:
				F_uk = Redundant_remove(D_u)
				H_k = t.mm(H_k, F_uk)

			H_lst.append(H_k)
		H = t.cat(H_lst, dim=1)
		# 疎行列に変換
		# H_sparse = self.convert_to_torch_sparse(sp.coo_matrix(H))
		# H = self.min_max_normalize(H)
		Lap_H, H_temp = self.Laplcian_matrix(H)
		return Lap_H, H_temp, H



	def EncodeuHyperGraph(self, R, k_intra):
		def Redundant_remove(x):
			temp = self.Threshold(x, t.median(x.diag()), 0)
			F_k = self.Threshold(temp, 0.0000001, 1)
			return F_k

		class HyperGraphConvolution(nn.Module):
			def __init__(self, input_features, output_features):
				super(HyperGraphConvolution, self).__init__()
				self.weight = nn.Parameter(t.FloatTensor(input_features, output_features))
				nn.init.xavier_uniform_(self.weight)
				self.uEmbeds = t.randn(args.user, args.latdim)
				self.relu = t.nn.ReLU() 
				self.hgnnLayer = HGNNLayer()

			def forward(self, x, H):
				uuHyper = H
				uuHyper = uuHyper.to(device="cuda:0")
				hyperLats = []
				for i in range(args.gnn_layer):
					hyperULat = self.hgnnLayer(F.dropout(uuHyper, p=1-0.5), x[:args.user])
					hyperLats.append(t.concat([hyperULat], dim=0))
				embeds = sum(hyperLats)
				return embeds

			def Laplcian_matrix(self, H):
				D_uv = t.diag(H.sum(axis=1))
				D_u = t.diag(t.mm(H, t.diag(H.sum(axis=0))).sum(axis=0))

				D_uv_div = t.nan_to_num(t.pow(D_uv.diag(), -1/2), posinf=0, neginf=0).diag()
				# print(D_uv_div)
				D_u_div = t.nan_to_num(t.pow(D_u.diag(), -1/2), posinf=0, neginf=0).diag()
				# print(D_u_div)

				temp = t.mm(D_uv_div, H)
				temp = t.mm(temp, D_u_div)
				temp = t.mm(temp, H.T)
				H_Lap = t.mm(temp, D_uv_div)

				edgetemp = t.mm(H.T, D_uv_div)
				# H_Lap = t.mm(temp, D_u_div)
				return H_Lap, edgetemp

		class HGNNLayer(nn.Module):
			def __init__(self):
				super(HGNNLayer, self).__init__()
				self.act = nn.LeakyReLU(negative_slope=0.5)
			
			def forward(self, embeds, adj):
				# lat = self.act(adj.T @ embeds)
				# ret = self.act(adj @ lat)
				# print(adj.size(), embeds.size())
				lat = (adj.T @ embeds)
				ret = (adj @ lat)
				return ret

		class GraphAutoencoder(nn.Module):
			def __init__(self, input_features, hidden_dim, output_dim):
				super(GraphAutoencoder, self).__init__()
				
				self.uHyper = nn.Parameter(init(t.empty(args.user, output_dim))).to(device="cuda:0")
				self.encoder = HyperGraphConvolution(input_features, hidden_dim)
			

			def forward(self, x, adjacency_matrix):
				# print(x.size(), adjacency_matrix.size())
				x_out = t.relu(self.encoder(adjacency_matrix, x))
				# print(x.device, self.uHyper.device)
				x_hats = t.relu(self.encoder(self.uHyper, x))
				# print(x_out.size(), x_hats.size())
				return x_out, x_hats

		H_lst = [R.to_dense()]
		D_u = t.zeros(R.shape[1], R.shape[1]).cuda()
		for k in range(k_intra-1):
			dense_R = R.to_dense()

			H_k = t.mm(dense_R, self.Threshold(t.matrix_power(t.mm(dense_R.T, dense_R).to_dense(), k), 1, 1))
			D_uv_k = t.diag(H_k.sum(axis=0))
			D_u_k = t.diag(t.mm(H_k, D_uv_k).sum(axis=0))
			D_u += D_u_k

			if k > 0:
				F_uk = Redundant_remove(D_u)
				H_k = t.mm(H_k, F_uk)

			H_lst.append(H_k)
		H = t.cat(H_lst, dim=1)
  
		# Encode
		# グラフオートエンコーダーのモデル、損失関数、オプティマイザの初期化
		input_features = args.latdim  # ノードの特徴量の次元数
		output_dim = args.hyperNum
		hidden_dim = 64  # 隠れ層の次元数
		learning_rate = 0.0000001
		Encodemodel = GraphAutoencoder(input_features, hidden_dim, output_dim)
		# model1 = GraphAutoencoder(input_features, hidden_dim, output_dim)
		criterion = nn.MSELoss()
		# criterion = InfoNCE()
		optimizer = optim.Adam(Encodemodel.parameters(), lr=learning_rate)

		# ダミーデータの準備（ノードの特徴量と隣接行列）
		nodefeature = t.randn(args.user, args.latdim)
		nodefeature = nodefeature.to(device="cuda:0")
		adjacency_matrix = H
		adjacency_matrix = adjacency_matrix.to(device="cuda:0")
		


		# トレーニングの実行
		num_epochs = args.encode_epoch

		for epoch in range(num_epochs):
			# 勾配を初期化
			optimizer.zero_grad()

			# フォワードパス
			outputs, output_hat = Encodemodel(nodefeature, adjacency_matrix)
			
			# 損失の計算
			loss = criterion(outputs, output_hat)
			# loss = criterion(outputs, outputs, output_hat)
			
			# バックワードパス
			loss.backward()
			
			# パラメータの更新
			optimizer.step()

			# 途中結果の表示
			print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

		print('Training complete!')
		Encode_H = Encodemodel.uHyper
		# print(Encode_H.size())
		# print(H.size())
		del adjacency_matrix, H
		gc.collect()
		Encode_H = t.rand(5432, 8192, device="cuda:0")

  
		# 疎行列に変換
		# H_sparse = self.convert_to_torch_sparse(sp.coo_matrix(H))
		# H = self.min_max_normalize(H)
		Lap_H, H_temp = self.Laplcian_matrix(Encode_H)
		return Lap_H, H_temp, Encode_H

	def EncodeiHyperGraph(self, R, k_intra):
		def Redundant_remove(x):
			temp = self.Threshold(x, t.median(x.diag()), 0)
			F_k = self.Threshold(temp, 0.0000001, 1)
			return F_k

		class HyperGraphConvolution(nn.Module):
			def __init__(self, input_features, output_features):
				super(HyperGraphConvolution, self).__init__()
				self.weight = nn.Parameter(t.FloatTensor(input_features, output_features))
				nn.init.xavier_uniform_(self.weight)
				self.uEmbeds = t.randn(args.item, args.latdim)
				self.relu = t.nn.ReLU() 
				self.hgnnLayer = HGNNLayer()

			def forward(self, x, H):
				uuHyper = H
				uuHyper = uuHyper.to(device="cuda:0")
				hyperLats = []
				for i in range(args.gnn_layer):
					hyperULat = self.hgnnLayer(F.dropout(uuHyper, p=1-0.5), x[:args.item])
					hyperLats.append(t.concat([hyperULat], dim=0))
				embeds = sum(hyperLats)
				return embeds

			def Laplcian_matrix(self, H):
				D_uv = t.diag(H.sum(axis=1))
				D_u = t.diag(t.mm(H, t.diag(H.sum(axis=0))).sum(axis=0))

				D_uv_div = t.nan_to_num(t.pow(D_uv.diag(), -1/2), posinf=0, neginf=0).diag()
				# print(D_uv_div)
				D_u_div = t.nan_to_num(t.pow(D_u.diag(), -1/2), posinf=0, neginf=0).diag()
				# print(D_u_div)

				temp = t.mm(D_uv_div, H)
				temp = t.mm(temp, D_u_div)
				temp = t.mm(temp, H.T)
				H_Lap = t.mm(temp, D_uv_div)

				edgetemp = t.mm(H.T, D_uv_div)
				# H_Lap = t.mm(temp, D_u_div)
				return H_Lap, edgetemp

		class HGNNLayer(nn.Module):
			def __init__(self):
				super(HGNNLayer, self).__init__()
				self.act = nn.LeakyReLU(negative_slope=0.5)
			
			def forward(self, embeds, adj):
				# lat = self.act(adj.T @ embeds)
				# ret = self.act(adj @ lat)
				# print(adj.size(), embeds.size())
				lat = (adj.T @ embeds)
				ret = (adj @ lat)
				return ret

		class GraphAutoencoder(nn.Module):
			def __init__(self, input_features, hidden_dim, output_dim):
				super(GraphAutoencoder, self).__init__()
				
				self.uHyper = nn.Parameter(init(t.empty(args.item, output_dim))).to(device="cuda:0")
				# self.uHyper = torch.randn(1600, 1024)
				self.encoder = HyperGraphConvolution(input_features, hidden_dim)
			

			def forward(self, x, adjacency_matrix):
				x_out = t.relu(self.encoder(adjacency_matrix, x))
				x_hats = t.relu(self.encoder(self.uHyper, x))
				return x_out, x_hats

		H_lst = [R.to_dense()]
		D_u = t.zeros(R.shape[1], R.shape[1]).cuda()
		for k in range(k_intra-1):
			dense_R = R.to_dense()

			H_k = t.mm(dense_R, self.Threshold(t.matrix_power(t.mm(dense_R.T, dense_R).to_dense(), k), 1, 1))
			D_uv_k = t.diag(H_k.sum(axis=0))
			D_u_k = t.diag(t.mm(H_k, D_uv_k).sum(axis=0))
			D_u += D_u_k

			if k > 0:
				F_uk = Redundant_remove(D_u)
				H_k = t.mm(H_k, F_uk)

			H_lst.append(H_k)
		H = t.cat(H_lst, dim=1)
  
		# Encode
		# グラフオートエンコーダーのモデル、損失関数、オプティマイザの初期化
		input_features = args.latdim   # ノードの特徴量の次元数
		output_dim = args.hyperNum
		hidden_dim = 64  # 隠れ層の次元数
		learning_rate = 0.0000001
		Encodemodel = GraphAutoencoder(input_features, hidden_dim, output_dim)
		# model1 = GraphAutoencoder(input_features, hidden_dim, output_dim)
		criterion = nn.MSELoss()
		# criterion = InfoNCE()
		optimizer = optim.Adam(Encodemodel.parameters(), lr=learning_rate)

		# ダミーデータの準備（ノードの特徴量と隣接行列）
		nodefeature = t.randn(args.item, args.latdim)
		nodefeature = nodefeature.to(device="cuda:0")
		# adjacency_matrix = H
		adjacency_matrix = H.to(device="cuda:0")


		# トレーニングの実行
		num_epochs = args.encode_epoch

		for epoch in range(num_epochs):
			# 勾配を初期化
			optimizer.zero_grad()

			# フォワードパス
			outputs, output_hat = Encodemodel(nodefeature, adjacency_matrix)
			# print(outputs.size(), output_hat.size())
			
			# 損失の計算
			loss = criterion(outputs, output_hat)
			# loss = criterion(outputs, outputs, output_hat)
			
			# バックワードパス
			loss.backward()
			
			# パラメータの更新
			optimizer.step()

			# 途中結果の表示
			print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

		print('Training complete!')
		Encode_H = Encodemodel.uHyper
		print(Encode_H.size())
		del adjacency_matrix
		gc.collect()
		Encode_H = t.rand(7236, 8192, device="cuda:0")

  
		# 疎行列に変換
		# H_sparse = self.convert_to_torch_sparse(sp.coo_matrix(H))
		# H = self.min_max_normalize(H)
		Lap_H, H_temp = self.Laplcian_matrix(Encode_H)
		return Lap_H, H_temp, Encode_H

	def LearnedHyperGraph(self, H, S):
		def nabla_residual_hypergraph(S, padd):
			values, indices = t.sort(S.view(-1))
			bottom_indices = indices[:int(padd * S.numel())]
			bottom_indices = (bottom_indices // S.size(1), bottom_indices % S.size(1))
			nabla_H = t.zeros_like(S)
			nabla_H[bottom_indices] = 1
			return nabla_H

		def delta_residual_hypergraph(S, padd):
			values, indices = t.sort(S.view(-1))
			top_indices = indices[-int(padd * S.numel()):]
			top_indices = (top_indices // S.size(1), top_indices % S.size(1))
			delta_H = t.zeros_like(S)
			delta_H[top_indices] = 1
			return delta_H

		def create_H_hat(original_H, delta_H, nabla_H):
			H_hat = t.clip(original_H + delta_H - nabla_H, 0, None)  # 0以下の値を0に、1以上の値を1にクリップ
			return H_hat
		padd = 0.005  # paddの設定（例として0.2）

		# ∆Hと∇Hの計算
		delta_H = delta_residual_hypergraph(S, padd)
		nabla_H = nabla_residual_hypergraph(S, padd)
		# print(S.size())
		# print(H.size(), delta_H.size(), nabla_H.size())
		H_hat = create_H_hat(H, delta_H, nabla_H)
		Learined_H, _ = self.Laplcian_matrix(H_hat)

		return Learined_H

	def min_max_normalize(self,tensor):
		min_val = tensor.min()
		max_val = tensor.max()
		normalized_tensor = 0.1 * (tensor - min_val) / (max_val - min_val) + 0.00000000001
		return normalized_tensor
    
	def LoadData(self):
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		args.user, args.item = trnMat.shape
		self.torchBiAdj = self.makeTorchAdj(trnMat)
		trnMat_torch = self.convert_to_torch_sparse(trnMat)
		self.uuHyperAdj, self.uuHypertemp, self.uuHyper = self.makeHyperAdj(trnMat_torch)
		# self.uuHyper, _, _ = self.makeRingHyperAdj(trnMat_torch)\
		self.uuHyperAdj = self.min_max_normalize(self.uuHyperAdj)
		# self.iiHyper, _, _ = self.makeRingHyperAdj(t.t(trnMat_torch))
		self.iiHyperAdj, self.iiHypertemp, self.iiHyper = self.makeHyperAdj(t.t(trnMat_torch))
		self.iiHyperAdj = self.min_max_normalize(self.iiHyperAdj)
		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)


	# def LoadData(self):
	# 	trnMat = self.loadOneFile(self.trnfile)
	# 	tstMat = self.loadOneFile(self.tstfile)
	# 	args.user, args.item = trnMat.shape
	# 	self.torchBiAdj = self.makeTorchAdj(trnMat)
	# 	trnMat_torch = self.convert_to_torch_sparse(trnMat)
	# 	self.uuHyperAdj, self.uuHypertemp, self.uuHyper = self.makeEncodeuHyperAdj(trnMat_torch)
	# 	# self.uuHyperAdj = self.min_max_normalize(self.uuHyperAdj)
	# 	self.iiHyperAdj, self.iiHypertemp, self.iiHyper = self.makeEncodeiHyperAdj(t.t(trnMat_torch))
	# 	# self.iiHyperAdj = self.min_max_normalize(self.iiHyperAdj)
	# 	trnData = TrnData(trnMat)
	# 	self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
	# 	tstData = TstData(tstMat, trnMat)
	# 	self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)
  
class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
