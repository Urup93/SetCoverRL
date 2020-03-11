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
    def __init__(self, input_uni_feat, input_sub_feat, output_uni_feat, output_sub_feat):
        super(BipartiteGraphConvNet, self).__init__()
        self.s1 = GraphConvLayer(input_uni_feat, output_sub_feat)
        self.s2 = GraphConvLayer(output_uni_feat, output_sub_feat)
        self.u1 = GraphConvLayer(input_sub_feat, output_uni_feat)
        self.u2 = GraphConvLayer(output_sub_feat, output_uni_feat)

    def forward(self, adj, uni_feat, sub_feat):

        sub_feat_ = self.s1(adj.t(), uni_feat)
        uni_feat_ = self.u1(adj, sub_feat)

        sub_feat_ = torch.tanh(sub_feat_ - torch.mean(sub_feat_, dim=0))
        uni_feat_ = torch.tanh(uni_feat_ - torch.mean(uni_feat_, dim=0))

        sub_feat = self.s2(adj.t(), uni_feat_)
        uni_feat = self.u2(adj, sub_feat_)

        sub_feat = torch.tanh(sub_feat - torch.mean(sub_feat, dim=0))
        uni_feat = torch.tanh(uni_feat - torch.mean(uni_feat, dim=0))

        return sub_feat, uni_feat

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
    def __init__(self, input_uni_feat, input_sub_feat, output_uni_feat, output_sub_feat, n_hid=64):
        super(SubsetRanking, self).__init__()
        self.BGCN = BipartiteGraphConvNet(input_uni_feat, input_sub_feat, output_uni_feat, output_sub_feat)
        self.FP = FeatureProcessing()
        self.Q_func = Q_function(2*output_sub_feat, n_hid)

    def forward(self, state):
        original_uni_feat, original_sub_feat, adj = state
        sub_feat, uni_feat = self.BGCN(adj, original_uni_feat, original_sub_feat)
        #print('sub feat: ', sub_feat[1:5, :])
        n_sub = sub_feat.size()[0]
        sum_sub_feat = torch.sum(sub_feat, dim=0).repeat(n_sub, 1)
        feat_mat = torch.cat((sub_feat, sum_sub_feat), dim=1)
        #print('feat mat: ', feat_mat[1:5, 1:5])
        q_val = self.Q_func(feat_mat)
        return q_val

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))


