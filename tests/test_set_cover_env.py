from unittest import TestCase
import torch
from set_cover_env import *
import numpy as np


class TestSetCoverEnv(TestCase):
    def setUp(self):
        self.env = SetCoverEnv(5, 3, 1)
        inst = torch.zeros(3, 5)
        inst[0, :] = torch.Tensor([0, 1, 0, 0, 1])
        inst[1, :] = torch.Tensor([1, 0, 1, 1, 1])
        inst[2, :] = torch.Tensor([1, 1, 0, 1, 1])
        self.env.instance = inst.int()

    def test_step(self):
        state, reward, done, _ = self.env.step(1)
        self.assertEqual(-1, reward)
        self.assertEqual(False, done)
        state, reward, done, _ = self.env.step(1)
        self.assertEqual(-1, reward)
        self.assertEqual(True, done)
        np.testing.assert_allclose(np.array([0, 1, 1, 0, 0]), self.env.solution)

    def test_generate_features(self):
        sub_feat, uni_feat = generate_features(self.env.instance)
        np.testing.assert_almost_equal(np.array([2/3, 2/3, 1/3, 2/3, 3/3]), np.array(sub_feat[:, 0].tolist()))
        np.testing.assert_almost_equal(np.array([4/5, 3/5, 4/5, 4/5, 2/3]), np.array(sub_feat[:, 1].tolist()))
        np.testing.assert_almost_equal(np.array([4/5, 2/5, 4/5, 4/5, 2/5]), np.array(sub_feat[:, 2].tolist()))
        np.testing.assert_almost_equal(np.array([2/5, 4/5, 4/5]), np.array(uni_feat[:, 0].tolist()))


