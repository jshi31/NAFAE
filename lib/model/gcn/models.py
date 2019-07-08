import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gcn.layers import GraphConvolution
from model.gcn.layers import GraphConvolutionCustom

"""
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
"""

class GCN(nn.Module):
    def __init__(self, d_in, d_out, dropout=0., bias=True):
        super(GCN, self).__init__()
        self.gc = GraphConvolution(d_in, d_out, bias)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x

class ResGCN(nn.Module):
    def __init__(self, d_in, d_out, dropout=0., bias=True):
        super(ResGCN, self).__init__()
        self.gc = GraphConvolution(d_in, d_out, bias)
        self.dropout = dropout

    def forward(self, x, y, adj):
        x = F.relu(self.gc(x, adj) + y)
        x = F.dropout(x, self.dropout, training=self.training)
        return x

class GCNCustom(nn.Module):
    def __init__(self, d_in, d_out, dropout=0., bias=True):
        super(GCNCustom, self).__init__()
        self.gc = GraphConvolutionCustom(d_in, d_out, bias)
        self.dropout = dropout

    def forward(self, x, y, adj, adj_sumrow):
        x = F.relu(self.gc(x, y, adj, adj_sumrow))
        x = F.dropout(x, self.dropout, training=self.training)
        return x

