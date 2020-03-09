from unittest import TestCase
import torch
from set_cover_env import SetCoverEnv
from Agent import Q_Agent

class TestQ_Agent(TestCase):
    def setUp(self):
        #Env
        self.env = SetCoverEnv(5, 3, 1)
        inst = torch.zeros(3, 5)
        inst[0, :] = torch.Tensor([0, 1, 0, 0, 1])
        inst[1, :] = torch.Tensor([1, 0, 1, 1, 1])
        inst[2, :] = torch.Tensor([1, 1, 0, 1, 1])
        self.env.instance = inst

        #Define test model
        class StubModel(torch.nn.Module):
            def __init__(self):
                super(StubModel, self).__init__()
                self.weight1 = torch.nn.Parameter(torch.eye(5))
                self.weight2 = torch.nn.Parameter(torch.eye(4))
                self.weight3 = torch.nn.Parameter(torch.eye(3))
                self.i = 0

            def forward(self, state):
                self.i += 1
                if self.i == 1:
                    return torch.mv(self.weight1, torch.Tensor([0, 1/4, 1/2, 1, 0]))
                if self.i == 2:
                    return torch.mv(self.weight2, torch.Tensor([0, 1/4, 1/2, 0]))
                if self.i == 3:
                    return torch.mv(self.weight3, torch.Tensor([0, 1/4, 0]))


        #Agent
        self.agent = Q_Agent(self.env, StubModel())

    def test_get_action(self):
        self.agent.epsilon = 0
        action, value = self.agent.get_action(self.env.get_state())
        self.assertEqual(3, action)
        self.assertEqual(1, value)

    def test_q_step(self):
        self.agent.epsilon = 0
        action, value = self.agent.get_action(self.env.get_state())
        next_state, reward, is_solution, _ = self.env.step(action)
        target, error = self.agent.q_step(value, reward, next_state)
        self.assertEqual(-0.5, target.item())
        self.assertEqual((1.5**2), error.item())

    def test_train(self):
        self.agent.epsilon = 0
