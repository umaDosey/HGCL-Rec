import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
from Utils.TimeLogger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader

class DataHandler:
	def __init__(self):
		if args.data == 'yelp':
			predir = '../Data/yelp/'
		if args.data == 'yelp2':
			predir = '../Data/yelp2/'
		elif args.data == 'amazon':
			predir = '../Data/amazon/'
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
		k_intra = 1
		# ハイパーグラフを構築
		mat = mat.to(device="cuda:0")
		hypergraph = self.HyperGraph(mat, k_intra)
		return hypergraph
    
	def Laplcian_matrix(self,H):
		D_uv = t.diag(H.sum(axis=1))
		D_u = t.diag(t.mm(H, t.diag(H.sum(axis=0))).sum(axis=0))

		D_uv_div = t.nan_to_num(t.pow(D_uv.diag(), -1/2), posinf=0, neginf=0).diag()
		D_u_div = t.nan_to_num(t.pow(D_u.diag(), -1/2), posinf=0, neginf=0).diag()

		temp = t.mm(D_uv_div, H)
		temp = t.mm(temp, D_u_div)
		temp = t.mm(temp, H.T)
		H_Lap = t.mm(temp, D_uv_div)
		#H_Lap = t.mm(temp, D_u_div)
		'''
		# version 1.0
		H_Lap = torch.mm(H, H.T)
		H_L = torch.diag(H_Lap.sum(axis=0)) - H_Lap
		D_div = torch.nan_to_num(torch.pow(H_Lap.diag(), -1/2), posinf=0, neginf=0).diag()
		temp = torch.mm(D_div, H_L)
		H_Lap = torch.mm(H_L, D_div)
		'''
		return H_Lap
        
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
		H = self.Laplcian_matrix(H)
		return H

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
		self.uuHyperAdj = self.makeHyperAdj(trnMat_torch)
		# self.uuHyperAdj = self.min_max_normalize(self.uuHyperAdj)
		self.iiHyperAdj = self.makeHyperAdj(t.t(trnMat_torch))
		# self.iiHyperAdj = self.min_max_normalize(self.iiHyperAdj)
		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

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
