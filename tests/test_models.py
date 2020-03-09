from unittest import TestCase
import torch
from models import *


class Test(TestCase):
    def test_q_function(self):
        ...

    def test_feature_processing(self):
        ...

    def test_big_juice(self):
        ...

    def test_graph_conv_layer(self):
        ...

    def test_bigger_juice(self):
        n_sub_feat = 3
        n_uni_feat = 2
        Big = SubsetRanking(n_uni_feat, n_sub_feat, 4)
        inst = torch.zeros(3, 2)
        inst[0, :] = torch.Tensor([0, 1])
        inst[1, :] = torch.Tensor([1, 1])
        inst[2, :] = torch.Tensor([1, 1])
        uni_feat = torch.arange(6).reshape(3, 2).float()
        sub_feat = torch.arange(6).reshape(2, 3).float()
        feat_and_state = uni_feat, sub_feat, inst
        print(Big(feat_and_state))

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
print(dir_path)

