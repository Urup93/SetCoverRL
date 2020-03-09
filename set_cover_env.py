import gym
import numpy as np
import torch
from datetime import datetime

def _generate_instance(u, s, p):
    A = np.random.choice([0, 1], replace=True, size=(u, s), p=[1 - p, p])
    vertix_nb = np.sum(A, axis=1)
    for i in range(u):
        if vertix_nb[i] == 0:
            A[i, np.random.randint(0, s)] = 1
    subset_degree = np.sum(A, axis=0)
    for j in range(s):
        if subset_degree[j]==0:
            A[np.random.randint(0, u), j] = 1
    A = torch.from_numpy(A).float()
    return A

def generate_features(adj):
    print('Generating features')
    time = datetime.now()
    n_uni, n_sub = adj.size()
    feature_subset_degree = (torch.sum(adj, dim=0)/n_uni).reshape(n_sub, 1)
    feature_mean_covered_degree = torch.empty(n_sub, 1)
    feature_min_covered_degree = torch.empty(n_sub, 1)
    for i in range(n_sub):
        covered_degrees = torch.sum(adj[adj[:, i] > 0, :], dim=1)/n_sub
        feature_mean_covered_degree[i] = torch.mean(covered_degrees)
        feature_min_covered_degree[i] = torch.min(covered_degrees)
    uni_feats = (torch.sum(adj, dim=1)/n_sub).reshape(n_uni, 1)
    sub_feats = torch.cat((feature_subset_degree, feature_mean_covered_degree, feature_min_covered_degree), dim=1)
    print('Finished generating features in', datetime.now()-time)
    return sub_feats, uni_feats

class SetCoverEnv(gym.Env):
    def __init__(self, s, u, p):
        self.s = s
        self.u = u
        self.p = p
        self.instance = _generate_instance(u, s, p)
        self.sub_feat, self.uni_feat = generate_features(self.instance)
        self.solution = np.zeros(s, dtype=int)

    def step(self, action: int):
        sol_action = action + sum(self.solution[0:action])
        while True:
            if self.solution[sol_action] == 0:
                print('Action found: ', sol_action)
                self.solution[sol_action] = 1
                break
            sol_action = sol_action + 1
        reward = -1
        self.red_state(action)
        done = self.is_solution()
        return self.get_state(), reward, done, {}

    def red_state(self, action):
        not_covered = self.instance[:, action] < 1
        self.uni_feat = self.uni_feat[not_covered, :]
        self.sub_feat = torch.cat((self.sub_feat[0:action, :], self.sub_feat[action+1:, :]), dim=0)
        self.instance = self.instance[not_covered, :]
        self.instance = torch.cat((self.instance[:, 0:action], self.instance[:, action + 1:]), dim=1)
        self.sub_feat = self.sub_feat[torch.sum(self.instance, dim=0) > 0, :]
        self.instance = self.instance[:, torch.sum(self.instance, dim=0) > 0]

    def get_solution(self):
        return self.solution

    def get_state(self):
        return self.uni_feat, self.sub_feat, self.instance

    def get_action_space(self):
        return self.instance.size()[1]

    def is_solution(self):
        return self.instance.size()[0] == 0

    def reset(self, u=None, s=None, p=None):
        if u is None or s is None or p is None:
            self.__init__(self.u, self.s, self.p)
        else:
            self.__init__(u, s, p)

    def set_instance(self, adj):
        self.instance = adj
        self.sub_feat, self.uni_feat = generate_features(adj)
        self.solution = np.zeros(adj.size()[1], dtype=int)

    def render(self, mode='human'):
        ...

    def close(self):
        ...

