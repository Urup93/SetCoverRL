from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch import nn
import torch
from datetime import datetime

class Q_function(Module):
    def __init__(self, n_feat, n_hid):
        super(Q_function, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_feat, n_hid),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(n_hid, n_hid),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(n_hid, 1)
        )

    def forward(self, features):
        return self.classifier(features)


class FeatureProcessing(Module):
    def __init__(self):
        super(FeatureProcessing, self).__init__()

    def forward(self, adj, cur_sub_idx, uni_feat, sub_feat, original_sub_feat):
        current_subset = sub_feat[cur_sub_idx, :]
        neighbor_uni_feat = torch.sum(uni_feat[adj[:, cur_sub_idx] > 0, :], dim=0)
        uni_feat_sum = torch.sum(uni_feat, dim=0)
        sub_feat_sum = torch.sum(sub_feat, dim=0)
        features = torch.cat((original_sub_feat, current_subset, neighbor_uni_feat, uni_feat_sum, sub_feat_sum))
        return features


class BipartiteGraphConvNet(Module):
    def __init__(self, n_uni_feat, n_sub_feat):
        super(BipartiteGraphConvNet, self).__init__()
        self.s1 = GraphConvLayer(n_uni_feat, n_sub_feat)
        self.s2 = GraphConvLayer(n_uni_feat, n_sub_feat)
        self.s3 = GraphConvLayer(n_uni_feat, n_sub_feat)
        self.u1 = GraphConvLayer(n_sub_feat, n_uni_feat)
        self.u2 = GraphConvLayer(n_sub_feat, n_uni_feat)
        self.u3 = GraphConvLayer(n_sub_feat, n_uni_feat)

    def forward(self, adj, uni_feat, sub_feat):
        time = datetime.now()
        adjT = self.normalize(adj.t())
        adj = self.normalize(adj)
        print('BGCN: Finished normalizing in: ', datetime.now()-time, ' seconds')
        time = datetime.now()
        sub_feat_ = self.s1(adjT, uni_feat)

        uni_feat_ = self.u1(adj, sub_feat)

        sub_feat = self.s1(adjT, uni_feat_)
        uni_feat = self.u1(adj, sub_feat_)

        sub_feat_ = self.s1(adjT, uni_feat)
        uni_feat_ = self.u1(adj, sub_feat)
        print('BGCN: Finished running convolutions in: ', datetime.now()-time, ' seconds')
        return sub_feat_, uni_feat_

    def normalize(self, adj):
        d_mat = torch.diag(1/torch.sum(adj, dim=1))
        return torch.mm(d_mat, adj)


class GraphConvLayer(torch.nn.Module):
    def __init__(self, in_feat, out_feat):
        super(GraphConvLayer, self).__init__()
        w = torch.empty(in_feat, out_feat)
        self.weight = Parameter(torch.nn.init.xavier_normal_(w))

    def forward(self, adj, in_feat):
        features = torch.mm(adj, in_feat)
        features = torch.mm(features, self.weight)
        return features


class SubsetRanking(Module):
    def __init__(self, n_uni_feat=32, n_sub_feat=32, n_hid=64):
        self.n_uni_feat = n_uni_feat
        self.n_sub_feat = n_sub_feat
        super(SubsetRanking, self).__init__()
        self.BGCN = BipartiteGraphConvNet(n_uni_feat, n_sub_feat)
        self.FP = FeatureProcessing()
        self.Q_func = Q_function(2*n_uni_feat+3*n_sub_feat, n_hid)

    def forward(self, state):
        original_uni_feat, original_sub_feat, adj = state
        sub_feat, uni_feat = self.BGCN(adj, original_uni_feat, original_sub_feat)
        n_sub = sub_feat.size()[0]
        feat_mat = torch.empty(n_sub, 2*self.n_uni_feat+3*self.n_sub_feat)
        time = datetime.now()
        for cur_sub in range(n_sub):
            feat = self.FP(adj, cur_sub, uni_feat, sub_feat, original_sub_feat[cur_sub, :])
            feat_mat[cur_sub, :] = feat
        print('Finished feature processing in: ', datetime.now()-time, ' seconds')
        time = datetime.now()
        q_val = self.Q_func(feat_mat)
        print('Q-function: Finished evaluting in: ', datetime.now()-time, ' seconds')
        return q_val

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


