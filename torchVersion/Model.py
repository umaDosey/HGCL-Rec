from statistics import mean
import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import pairPredict, contrastLoss
from DataHandler import DataHandler
from sklearn.metrics.pairwise import cosine_similarity
from torchmetrics.functional import pairwise_cosine_similarity
import gc

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		self.gcnLayer = GCNLayer()
		self.hgnnLayer = HGNNLayer()
		self.uHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
		self.iHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
		# self.uHyper = nn.Parameter(t.zeros(args.latdim, args.hyperNum))
		# self.iHyper = nn.Parameter(t.zeros(args.latdim, args.hyperNum))

		self.edgeDropper = SpAdjDropEdge()

	def forward(self, adj, keepRate):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
		lats = [embeds]
		gnnLats = []
		hyperLats = []
		# print(self.uHyper)
		# print(self.uHyper.type())
		# print(self.iHyper)
		# print(self.uHyper)
		uuHyper = self.uEmbeds @ self.uHyper
		iiHyper = self.iEmbeds @ self.iHyper
		# print(uuHyper)
		# print(iiHyper)
		# print(uuHyper.size())
		# print(iiHyper.size())
		for i in range(args.gnn_layer):
			temEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), lats[-1])
			hyperULat = self.hgnnLayer(F.dropout(uuHyper, p=1-keepRate), lats[-1][:args.user])
			hyperILat = self.hgnnLayer(F.dropout(iiHyper, p=1-keepRate), lats[-1][args.user:])
			gnnLats.append(temEmbeds)
			hyperLats.append(t.concat([hyperULat, hyperILat], dim=0))
			# lats.append(temEmbeds + hyperLats[-1])
			lats.append(hyperLats[-1])
			# lats.append(temEmbeds)
		embeds = sum(lats)
		# print(embeds.size())
		# print(gnnLats.size())
		# print(hyperLats.size())
		return embeds, gnnLats, hyperLats

	def calcLosses(self, ancs, poss, negs, adj, keepRate):
		embeds, gcnEmbedsLst, hyperEmbedsLst = self.forward(adj, keepRate)
		uEmbeds, iEmbeds = embeds[:args.user], embeds[args.user:]
		
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().mean()
		# bprLoss = t.maximum(t.zeros_like(scoreDiff), 1 - scoreDiff).mean() * 40

		sslLoss = 0
		for i in range(args.gnn_layer):
			embeds1 = gcnEmbedsLst[i].detach()
			embeds2 = hyperEmbedsLst[i]
			sslLoss += contrastLoss(embeds1[:args.user], embeds2[:args.user], t.unique(ancs), args.temp) + contrastLoss(embeds1[args.user:], embeds2[args.user:], t.unique(poss), args.temp)
		return bprLoss, sslLoss
	
	def predict(self, adj):
		embeds, _, _ = self.forward(adj, 1.0)
		return embeds[:args.user], embeds[args.user:]

class Model2(nn.Module):
	def __init__(self):
		super(Model2, self).__init__()

		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		self.gcnLayer = GCNLayer()
		self.hgnnLayer = HGNNLayer()
		self.uHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
		self.iHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
		# self.uHyper = nn.Parameter(t.zeros(args.latdim, args.hyperNum))
		# self.iHyper = nn.Parameter(t.zeros(args.latdim, args.hyperNum))

		self.edgeDropper = SpAdjDropEdge()

	def forward(self, adj, keepRate):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
		lats = [embeds]
		gnnLats = []
		hyperLats = []
		# print(self.uHyper)
		# print(self.uHyper.type())
		# print(self.iHyper)
		# print(self.uHyper)
		uuHyper = self.uEmbeds @ self.uHyper
		iiHyper = self.iEmbeds @ self.iHyper
		# print(uuHyper)
		# print(iiHyper)
		# print(uuHyper.size())
		# print(iiHyper.size())
		for i in range(args.gnn_layer):
			temEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), lats[-1])
			hyperULat = self.relu(t.mm(uuHyper, self.uEmbeds)) + self.uEmbeds
			hyperILat = self.relu(t.mm(iiHyper, self.iEmbeds)) + self.iEmbeds
			hyperULat_temp = self.relu(t.mm(uuHypertemp, self.uEmbeds))
			hyperILat_temp = self.relu(t.mm(iiHypertemp, self.iEmbeds))
			gnnLats.append(temEmbeds)
			hyperLats.append(t.concat([hyperULat, hyperILat], dim=0))
			hyperLats_temp.append(t.concat([hyperULat_temp, hyperILat_temp], dim=0))
			# lats.append(temEmbeds + hyperLats[-1])
			lats.append(temEmbeds)
		embeds = sum(lats)
		hyembeds = sum(hyperLats)
		hytempembeds = sum(hyperLats_temp)
		# print(hytempembeds.size())
  
		# print(embeds.size())
		# print(gnnLats.shape)
		# print(hyperLats.shape)
		HyperuEmbeds, HyperiEmbeds = hyembeds[:args.user], hyembeds[args.user:]
		# HyperuEmbedstemp, HyperiEmbedstemp = hytempembeds[:args.item*2], hytempembeds[args.item*2:]
		HyperuEmbedstemp, HyperiEmbedstemp = hytempembeds[:args.hyperNum], hytempembeds[args.hyperNum:]
		# print(HyperuEmbedstemp.size())
		# print(HyperiEmbedstemp.size())
		# print(HyperuEmbeds)
		# HyperuEmbeds = t.stack(HyperuEmbeds, dim=0)
		# print(HyperuEmbeds.size())
		# HyperuEmbedstemp = t.stack(HyperuEmbedstemp, dim=0)
		# print(HyperuEmbedstemp.size())
		# HyperuEmbeds = t.tensor(HyperuEmbeds)
		# HyperuEmbedstemp = t.tensor(HyperuEmbedstemp)

		uuHyperSimi = pairwise_cosine_similarity(HyperuEmbeds, HyperuEmbedstemp)
		iiHyperSimi = pairwise_cosine_similarity(HyperiEmbeds, HyperiEmbedstemp)
		# print(uuHyperSimi)
		uuLearnedHyper = self.handler.LearnedHyperGraph(uuHyper_wL, uuHyperSimi)
		iiLearnedHyper = self.handler.LearnedHyperGraph(iiHyper_wL, iiHyperSimi)

		LearnedhyperLats = []
		for i in range(args.gnn_layer):
			LearnedhyperULat = self.relu(t.mm(uuLearnedHyper, self.uEmbeds)) + self.uEmbeds
			LearnedhyperILat = self.relu(t.mm(iiLearnedHyper, self.iEmbeds)) + self.iEmbeds
			LearnedhyperLats.append(t.concat([LearnedhyperULat, LearnedhyperILat], dim=0))
  
		# print(embeds)
		return embeds, gnnLats, hyperLats, LearnedhyperLats

	def calcLosses(self, ancs, poss, negs, adj, keepRate):
		embeds, gcnEmbedsLst, hyperEmbedsLst , hyperEmbedsTempLst= self.forward(adj, keepRate)
		uEmbeds, iEmbeds = embeds[:args.user], embeds[args.user:]



		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().mean()
		# bprLoss = t.maximum(t.zeros_like(scoreDiff), 1 - scoreDiff).mean() * 40

		sslLoss = 0
		SLsslLoss = 0
		for i in range(args.gnn_layer):
			embeds1 = gcnEmbedsLst[i].detach()
			embeds3 = hyperEmbedsLst[i]
			embeds2 = hyperEmbedsTempLst[i]
			sslLoss += contrastLoss(embeds1[:args.user], embeds2[:args.user], t.unique(ancs), args.temp) + contrastLoss(embeds1[args.user:], embeds2[args.user:], t.unique(poss), args.temp)
			SLsslLoss += contrastLoss(embeds2[:args.user], embeds3[:args.user], t.unique(ancs), args.temp) + contrastLoss(embeds2[args.user:], embeds3[args.user:], t.unique(poss), args.temp)
		return bprLoss, sslLoss, SLsslLoss
	
	def predict(self, adj):
		embeds, _, _ = self.forward(adj, 1.0)
		return embeds[:args.user], embeds[args.user:]



class PropModel(nn.Module):
	def __init__(self):
		super(PropModel, self).__init__()
		self.soft = t.nn.Softmax(dim=1)
		self.f = nn.Sigmoid()
		self.relu = t.nn.ReLU()    
		self.handler = DataHandler()
		self.handler.LoadData()
		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		# print(self.uEmbeds)
		# print(self.iEmbeds)
		self.gcnLayer = GCNLayer()
		self.hgnnLayer = HGNNLayer()
		# self.uHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
		# self.iHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
		# self.uuHyper = t.rand(5432, 5432, device="cuda:0")
		# self.iiHyper = t.rand(7236, 7236, device="cuda:0")

		self.edgeDropper = SpAdjDropEdge()

	def forward(self, adj, keepRate):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
		# print(embeds)
		lats = [embeds]
		gnnLats = []
		hyperLats = []
		# print(self.uHyper)
		# print(self.uHyper.type())
		# print(self.iHyper)
		# print(self.uHyper)
		# print(self.handler.torchBiAdj)bash
		uuHyper = self.handler.uuHyperAdj.to(device="cuda:0").detach()
		iiHyper = self.handler.iiHyperAdj.to(device="cuda:0").detach()
		# uuHyper = t.rand(5432, 5432, device="cuda:0")
		# iiHyper = t.rand(7236, 7236, device="cuda:0")
		# uuHyper = t.rand(14651, 14651, device="cuda:0")
		# iiHyper = t.rand(12367, 12367, device="cuda:0")
		# uuHyper = self.uuHyper
		# iiHyper = self.iiHyper

		# テンソルを nn.Parameter に変換し、requires_grad=True を適用
		# uuHyper = nn.Parameter(uuHyper.float(), requires_grad=True)
		# iiHyper = nn.Parameter(iiHyper.float(), requires_grad=True)
        
		# print(uuHyper)
		# print(iiHyper)
		# print(uuHyper.size())
		# print(iiHyper.size())
		for i in range(args.gnn_layer):
			temEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), lats[-1])
			hyperULat = self.relu(t.mm(uuHyper, self.uEmbeds)) + self.uEmbeds
			hyperILat = self.relu(t.mm(iiHyper, self.iEmbeds)) + self.iEmbeds
			gnnLats.append(temEmbeds)
			hyperLats.append(t.concat([hyperULat, hyperILat], dim=0))
			# lats.append(temEmbeds + hyperLats[-1])
			# print([len(v) for v in hyperLats])
			lats.append(temEmbeds)

		embeds = sum(lats)
  
		# print(embeds.size())
		# print(gnnLats.shape)
		# print(hyperLats.shape)
  
		# print(embeds)
		return embeds, gnnLats, hyperLats

	def calcLosses(self, ancs, poss, negs, adj, keepRate):
		embeds, gcnEmbedsLst, hyperEmbedsLst = self.forward(adj, keepRate)
		uEmbeds, iEmbeds = embeds[:args.user], embeds[args.user:]
		
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().mean()
		# bprLoss = t.maximum(t.zeros_like(scoreDiff), 1 - scoreDiff).mean() * 40

		sslLoss = 0
		for i in range(args.gnn_layer):
			embeds1 = gcnEmbedsLst[i].detach()
			embeds2 = hyperEmbedsLst[i]
			sslLoss += contrastLoss(embeds1[:args.user], embeds2[:args.user], t.unique(ancs), args.temp) + contrastLoss(embeds1[args.user:], embeds2[args.user:], t.unique(poss), args.temp)
		return bprLoss, sslLoss
	
	def predict(self, adj):
		embeds, _, _ = self.forward(adj, 1.0)
		return embeds[:args.user], embeds[args.user:]


class PropModel2(nn.Module):
	def __init__(self):
		super(PropModel2, self).__init__()
		self.soft = t.nn.Softmax(dim=1)
		self.f = nn.Sigmoid()
		self.relu = t.nn.ReLU()    
		self.handler = DataHandler()
		self.handler.LoadData()
		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		# print(self.uEmbeds)
		# print(self.iEmbeds)
		self.gcnLayer = GCNLayer()
		self.hgnnLayer = HGNNLayer()
		# self.uHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
		# self.iHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))

		self.edgeDropper = SpAdjDropEdge()
		

	def forward(self, adj, keepRate):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
		# print(embeds)
		lats = [embeds]
		gnnLats = []
		hyperLats = []
		hyperLats_temp = []
		# print(self.uHyper)
		# print(self.uHyper.type())
		# print(self.iHyper)
		# print(self.uHyper)
		# print(self.handler.torchBiAdj)
		uuHyper = self.handler.uuHyperAdj.to(device="cuda:0").detach()
		iiHyper = self.handler.iiHyperAdj.to(device="cuda:0").detach()
		uuHyper_wL = self.handler.uuHyper.to(device="cuda:0").detach()
		iiHyper_wL = self.handler.iiHyper.to(device="cuda:0").detach()
		uuHypertemp = self.handler.uuHypertemp.to(device="cuda:0").detach()
		iiHypertemp = self.handler.iiHypertemp.to(device="cuda:0").detach()
		# print(uuHyper.size(), iiHyper.size(), uuHyper_wL.size(), iiHyper_wL.size(), uuHypertemp.size(), iiHypertemp.size())
		# uuHyper = t.rand(5432, 5432, device="cuda:0")
		# iiHyper = t.rand(7236, 7236, device="cuda:0")
		# uuHyper_wL = t.rand(5432, 8192, device="cuda:0")
		# iiHyper_wL = t.rand(7236, 8192, device="cuda:0")
		# uuHypertemp = t.rand(8192, 5432, device="cuda:0")
		# iiHypertemp = t.rand(8192, 7236, device="cuda:0")
		# print(uuHyper.size())
		# print(uuHypertemp.type())
		# print(uuHypertemp.T.size())
		# テンソルを nn.Parameter に変換し、requires_grad=True を適用
		# uuHyper = nn.Parameter(uuHyper.float(), requires_grad=True)
		# iiHyper = nn.Parameter(iiHyper.float(), requires_grad=True)
        
		# print(uuHyper)
		# print(iiHyper)
		# print(uuHyper.size())
		# print(iiHyper.size())
		# print(self.uEmbeds.device)
		# print(uuHyper.device)
		for i in range(args.gnn_layer):
			temEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), lats[-1])
			hyperULat = self.relu(t.mm(uuHyper, self.uEmbeds)) + self.uEmbeds
			hyperILat = self.relu(t.mm(iiHyper, self.iEmbeds)) + self.iEmbeds
			hyperULat_temp = self.relu(t.mm(uuHypertemp, self.uEmbeds))
			hyperILat_temp = self.relu(t.mm(iiHypertemp, self.iEmbeds))
			gnnLats.append(temEmbeds)
			hyperLats.append(t.concat([hyperULat, hyperILat], dim=0))
			hyperLats_temp.append(t.concat([hyperULat_temp, hyperILat_temp], dim=0))
			# lats.append(temEmbeds + hyperLats[-1])
			lats.append(temEmbeds)
		embeds = sum(lats)
		hyembeds = sum(hyperLats)
		hytempembeds = sum(hyperLats_temp)
		# print(hytempembeds.size())
  
		# print(embeds.size())
		# print(gnnLats.shape)
		# print(hyperLats.shape)
		HyperuEmbeds, HyperiEmbeds = hyembeds[:args.user], hyembeds[args.user:]
		HyperuEmbedstemp, HyperiEmbedstemp = hytempembeds[:args.item*args.hyperk], hytempembeds[args.item*args.hyperk:]
		# HyperuEmbedstemp, HyperiEmbedstemp = hytempembeds[:args.hyperNum], hytempembeds[args.hyperNum:]
		# print(HyperuEmbedstemp.size())
		# print(HyperiEmbedstemp.size())
		# print(HyperuEmbeds)
		# HyperuEmbeds = t.stack(HyperuEmbeds, dim=0)
		# print(HyperuEmbeds.size())
		# HyperuEmbedstemp = t.stack(HyperuEmbedstemp, dim=0)
		# print(HyperuEmbedstemp.size())
		# HyperuEmbeds = t.tensor(HyperuEmbeds)
		# HyperuEmbedstemp = t.tensor(HyperuEmbedstemp)

		uuHyperSimi = pairwise_cosine_similarity(HyperuEmbeds, HyperuEmbedstemp)
		iiHyperSimi = pairwise_cosine_similarity(HyperiEmbeds, HyperiEmbedstemp)
		# print(uuHyperSimi.size())
		uuLearnedHyper = self.handler.LearnedHyperGraph(uuHyper_wL, uuHyperSimi)
		iiLearnedHyper = self.handler.LearnedHyperGraph(iiHyper_wL, iiHyperSimi)

		del uuHyperSimi, iiHyperSimi
		gc.collect()

		LearnedhyperLats = []
		for i in range(args.gnn_layer):
			LearnedhyperULat = self.relu(t.mm(uuLearnedHyper, self.uEmbeds)) + self.uEmbeds
			LearnedhyperILat = self.relu(t.mm(iiLearnedHyper, self.iEmbeds)) + self.iEmbeds
			LearnedhyperLats.append(t.concat([LearnedhyperULat, LearnedhyperILat], dim=0))
  
		# print(embeds)
		return embeds, gnnLats, hyperLats, LearnedhyperLats

	def calcLosses(self, ancs, poss, negs, adj, keepRate):
		embeds, gcnEmbedsLst, hyperEmbedsLst , hyperEmbedsTempLst= self.forward(adj, keepRate)
		uEmbeds, iEmbeds = embeds[:args.user], embeds[args.user:]



		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().mean()
		# bprLoss = t.maximum(t.zeros_like(scoreDiff), 1 - scoreDiff).mean() * 40

		sslLoss = 0
		SLsslLoss = 0
		for i in range(args.gnn_layer):
			embeds1 = gcnEmbedsLst[i].detach()
			embeds3 = hyperEmbedsLst[i]
			embeds2 = hyperEmbedsTempLst[i]
			sslLoss += contrastLoss(embeds1[:args.user], embeds2[:args.user], t.unique(ancs), args.temp) + contrastLoss(embeds1[args.user:], embeds2[args.user:], t.unique(poss), args.temp)
			SLsslLoss += contrastLoss(embeds2[:args.user], embeds3[:args.user], t.unique(ancs), args.temp) + contrastLoss(embeds2[args.user:], embeds3[args.user:], t.unique(poss), args.temp)
		return bprLoss, sslLoss, SLsslLoss
	
	def predict(self, adj):
		embeds, _, _, _ = self.forward(adj, 1.0)
		return embeds[:args.user], embeds[args.user:]

class PropModel3(nn.Module):
	def __init__(self):
		super(PropModel3, self).__init__()
		self.soft = t.nn.Softmax(dim=1)
		self.f = nn.Sigmoid()
		self.relu = t.nn.ReLU()    
		self.handler = DataHandler()
		self.handler.LoadData()
		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		# print(self.uEmbeds)
		# print(self.iEmbeds)
		self.gcnLayer = GCNLayer()
		self.hgnnLayer = HGNNLayer()
		# self.uHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
		# self.iHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))

		self.edgeDropper = SpAdjDropEdge()
		

	def forward(self, adj, keepRate):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
		# print(embeds)
		lats = [embeds]
		gnnLats = []
		hyperLats = []
		hyperLats_temp = []
		# print(self.uHyper)
		# print(self.uHyper.type())
		# print(self.iHyper)
		# print(self.uHyper)
		# print(self.handler.torchBiAdj)
		uuHyper = self.handler.uuHyperAdj.to(device="cuda:0").detach()
		iiHyper = self.handler.iiHyperAdj.to(device="cuda:0").detach()
		# uuHyper_wL = self.handler.uuHyper.to(device="cuda:0").detach()
		# iiHyper_wL = self.handler.iiHyper.to(device="cuda:0").detach()
		# uuHypertemp = self.handler.uuHypertemp.to(device="cuda:0").detach()
		# iiHypertemp = self.handler.iiHypertemp.to(device="cuda:0").detach()
		# print(uuHyper.size())
		# print(uuHypertemp.type())
		# print(uuHypertemp.T.size())
		# テンソルを nn.Parameter に変換し、requires_grad=True を適用
		# uuHyper = nn.Parameter(uuHyper.float(), requires_grad=True)
		# iiHyper = nn.Parameter(iiHyper.float(), requires_grad=True)
        
		# print(uuHyper)
		# print(iiHyper)
		# print(uuHyper.size())
		# print(iiHyper.size())
		# print(self.uEmbeds.device)
		# print(uuHyper.device)
		for i in range(args.gnn_layer):
			temEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), lats[-1])
			hyperULat = self.relu(t.mm(uuHyper, self.uEmbeds)) + self.uEmbeds
			hyperILat = self.relu(t.mm(iiHyper, self.iEmbeds)) + self.iEmbeds
			# hyperULat_temp = self.relu(t.mm(uuHypertemp, self.uEmbeds))
			# hyperILat_temp = self.relu(t.mm(iiHypertemp, self.iEmbeds))
			gnnLats.append(temEmbeds)
			hyperLats.append(t.concat([hyperULat, hyperILat], dim=0))
			# hyperLats_temp.append(t.concat([hyperULat_temp, hyperILat_temp], dim=0))
			# lats.append(temEmbeds + hyperLats[-1])
			lats.append(temEmbeds)
		embeds = sum(lats)
		hyembeds = sum(hyperLats)
		# hytempembeds = sum(hyperLats_temp)
		# print(hytempembeds.size())
  
		# print(embeds.size())
		# print(gnnLats.shape)
		# print(hyperLats.shape)
		# HyperuEmbeds, HyperiEmbeds = hyembeds[:args.user], hyembeds[args.user:]
		# HyperuEmbedstemp, HyperiEmbedstemp = hytempembeds[:args.item*2], hytempembeds[args.item*2:]
		# HyperuEmbedstemp, HyperiEmbedstemp = hytempembeds[:args.hyperNum], hytempembeds[args.hyperNum:]
		# print(HyperuEmbedstemp.size())
		# print(HyperiEmbedstemp.size())
		# print(HyperuEmbeds)
		# HyperuEmbeds = t.stack(HyperuEmbeds, dim=0)
		# print(HyperuEmbeds.size())
		# HyperuEmbedstemp = t.stack(HyperuEmbedstemp, dim=0)
		# print(HyperuEmbedstemp.size())
		# HyperuEmbeds = t.tensor(HyperuEmbeds)
		# HyperuEmbedstemp = t.tensor(HyperuEmbedstemp)

		# uuHyperSimi = pairwise_cosine_similarity(HyperuEmbeds, HyperuEmbedstemp)
		# iiHyperSimi = pairwise_cosine_similarity(HyperiEmbeds, HyperiEmbedstemp)
		# # print(uuHyperSimi)
		# uuLearnedHyper = self.handler.LearnedHyperGraph(uuHyper_wL, uuHyperSimi)
		# iiLearnedHyper = self.handler.LearnedHyperGraph(iiHyper_wL, iiHyperSimi)

		# LearnedhyperLats = []
		# for i in range(args.gnn_layer):
		# 	LearnedhyperULat = self.relu(t.mm(uuLearnedHyper, self.uEmbeds)) + self.uEmbeds
		# 	LearnedhyperILat = self.relu(t.mm(iiLearnedHyper, self.iEmbeds)) + self.iEmbeds
			# LearnedhyperLats.append(t.concat([LearnedhyperULat, LearnedhyperILat], dim=0))
  
		# print(embeds)
		return embeds, gnnLats, hyperLats

	def calcLosses(self, ancs, poss, negs, adj, keepRate):
		embeds, gcnEmbedsLst, hyperEmbedsLst = self.forward(adj, keepRate)
		uEmbeds, iEmbeds = embeds[:args.user], embeds[args.user:]
		
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().mean()
		# bprLoss = t.maximum(t.zeros_like(scoreDiff), 1 - scoreDiff).mean() * 40

		sslLoss = 0
		for i in range(args.gnn_layer):
			embeds1 = gcnEmbedsLst[i].detach()
			embeds2 = hyperEmbedsLst[i]
			sslLoss += contrastLoss(embeds1[:args.user], embeds2[:args.user], t.unique(ancs), args.temp) + contrastLoss(embeds1[args.user:], embeds2[args.user:], t.unique(poss), args.temp)
		return bprLoss, sslLoss
	
	def predict(self, adj):
		embeds, _, _, _ = self.forward(adj, 1.0)
		return embeds[:args.user], embeds[args.user:]

class DHCF(nn.Module):
	def __init__(self):
		super(DHCF, self).__init__()
		self.soft = t.nn.Softmax(dim=1)
		self.f = nn.Sigmoid()
		self.relu = t.nn.ReLU()
		self.MLP = t.nn.Linear(args.latdim, args.latdim)
		self.handler = DataHandler()
		self.handler.LoadData()
		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		# print(self.uEmbeds)
		# print(self.iEmbeds)
		self.gcnLayer = GCNLayer()
		self.hgnnLayer = HGNNLayer()
		# self.uHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
		# self.iHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))

		self.edgeDropper = SpAdjDropEdge()

	def forward(self, adj, keepRate):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
		# print(embeds)
		# lats = [embeds]
		gnnLats = []
		hyperLats = []
		# print(self.uHyper)
		# print(self.uHyper.type())
		# print(self.iHyper)
		# print(self.uHyper)
		# print(self.handler.torchBiAdj)
		uuHyper = self.handler.uuHyperAdj
		iiHyper = self.handler.iiHyperAdj
		# テンソルを nn.Parameter に変換し、requires_grad=True を適用
		# uuHyper = nn.Parameter(uuHyper.float(), requires_grad=True)
		# iiHyper = nn.Parameter(iiHyper.float(), requires_grad=True)
        
		# print(uuHyper)
		# print(iiHyper)
		# print(uuHyper.size())
		# print(iiHyper.size())
		for i in range(args.gnn_layer):
			# temEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), lats[-1])
			hyperULat = self.relu(t.mm(uuHyper, self.uEmbeds)) + self.uEmbeds
			hyperILat = self.relu(t.mm(iiHyper, self.iEmbeds)) + self.iEmbeds
			# gnnLats.append(temEmbeds)
			hyperLats.append(t.concat([hyperULat, hyperILat], dim=0))
			# lats.append(temEmbeds + hyperLats[-1])
			# lats.append(temEmbeds)
		embeds = sum(hyperLats)
		embeds = self.relu(self.MLP(embeds))

  
		# print(embeds)
		return embeds, gnnLats, hyperLats

	def calcLosses(self, ancs, poss, negs, adj, keepRate):
		embeds, _, _ = self.forward(adj, keepRate)
		uEmbeds, iEmbeds = embeds[:args.user], embeds[args.user:]
		
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().mean()
		# bprLoss = t.maximum(t.zeros_like(scoreDiff), 1 - scoreDiff).mean() * 40
		sslLoss = 0
		"""
		for i in range(args.gnn_layer):
			embeds1 = gcnEmbedsLst[i].detach()
			embeds2 = hyperEmbedsLst[i]
			sslLoss += contrastLoss(embeds1[:args.user], embeds2[:args.user], t.unique(ancs), args.temp) + contrastLoss(embeds1[args.user:], embeds2[args.user:], t.unique(poss), args.temp)
		"""
		return bprLoss, sslLoss
	
	def predict(self, adj):
		embeds, _, _ = self.forward(adj, 1.0)
		return embeds[:args.user], embeds[args.user:]

class MF(nn.Module):
	def __init__(self):
		super(MF, self).__init__()
		self.soft = t.nn.Softmax(dim=1)
		self.f = nn.Sigmoid()
		self.relu = t.nn.ReLU()
		self.MLP = t.nn.Linear(args.latdim, args.latdim)
		# self.handler = DataHandler()
		# self.handler.LoadData()
		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		# print(self.uEmbeds)
		# print(self.iEmbeds)
		self.gcnLayer = GCNLayer()
		self.hgnnLayer = HGNNLayer()
		# self.uHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
		# self.iHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))

		self.edgeDropper = SpAdjDropEdge()

	def forward(self, adj, keepRate):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], dim=0)
		# print(embeds)
		lats = [embeds]
		gnnLats = []
		hyperLats = []
		# print(self.uHyper)
		# print(self.uHyper.type())
		# print(self.iHyper)
		# print(self.uHyper)
		# print(self.handler.torchBiAdj)
		# uuHyper = self.handler.uuHyperAdj
		# iiHyper = self.handler.iiHyperAdj
		# テンソルを nn.Parameter に変換し、requires_grad=True を適用
		# uuHyper = nn.Parameter(uuHyper.float(), requires_grad=True)
		# iiHyper = nn.Parameter(iiHyper.float(), requires_grad=True)
        
		# print(uuHyper)
		# print(iiHyper)
		# print(uuHyper.size())
		# print(iiHyper.size())
		# for i in range(args.gnn_layer):
		# 	temEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), lats[-1])
		# 	hyperULat = self.relu(t.mm(uuHyper, self.uEmbeds)) + self.uEmbeds
		# 	hyperILat = self.relu(t.mm(iiHyper, self.iEmbeds)) + self.iEmbeds
		# 	gnnLats.append(temEmbeds)
		# 	hyperLats.append(t.concat([hyperULat, hyperILat], dim=0))
		# 	# lats.append(temEmbeds + hyperLats[-1])
		# 	lats.append(temEmbeds)
		# embeds = sum(hyperLats)
		# embeds = self.relu(self.MLP(embeds))

  
		# print(embeds)
		return embeds, gnnLats, hyperLats

	def calcLosses(self, ancs, poss, negs, adj, keepRate):
		embeds, gcnEmbedsLst, hyperEmbedsLst = self.forward(adj, keepRate)
		uEmbeds, iEmbeds = embeds[:args.user], embeds[args.user:]
		
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
		bprLoss = - (scoreDiff).sigmoid().log().mean()
		# bprLoss = t.maximum(t.zeros_like(scoreDiff), 1 - scoreDiff).mean() * 40
		sslLoss = 0
		"""
		for i in range(args.gnn_layer):
			embeds1 = gcnEmbedsLst[i].detach()
			embeds2 = hyperEmbedsLst[i]
			sslLoss += contrastLoss(embeds1[:args.user], embeds2[:args.user], t.unique(ancs), args.temp) + contrastLoss(embeds1[args.user:], embeds2[args.user:], t.unique(poss), args.temp)
		"""
		return bprLoss, sslLoss
	
	def predict(self, adj):
		embeds, _, _ = self.forward(adj, 1.0)
		return embeds[:args.user], embeds[args.user:]

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()
		self.act = nn.LeakyReLU(negative_slope=args.leaky)

	def forward(self, adj, embeds):
		return (t.spmm(adj, embeds))

class HGNNLayer(nn.Module):
	def __init__(self):
		super(HGNNLayer, self).__init__()
		self.act = nn.LeakyReLU(negative_slope=args.leaky)
	
	def forward(self, adj, embeds):
		# lat = self.act(adj.T @ embeds)
		# ret = self.act(adj @ lat)
		lat = (adj.T @ embeds)
		ret = (adj @ lat)
		return ret

class SpAdjDropEdge(nn.Module):
	def __init__(self):
		super(SpAdjDropEdge, self).__init__()

	def forward(self, adj, keepRate):
		if keepRate == 1.0:
			return adj
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
		newVals = vals[mask] / keepRate
		newIdxs = idxs[:, mask]
		return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)
