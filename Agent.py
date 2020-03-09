import random
import torch
from MemoryBuffer import *

class Q_Agent:
    def __init__(self, env, model):
        self.env = env
        self.epsilon = 0.3
        self.model = model
        self.optim = torch.optim.Adam(model.parameters())
        self.loss = torch.nn.MSELoss()

    def get_action(self, state):
        num_subsets = self.env.get_action_space()
        q_values = self.model(state)
        if random.random() < self.epsilon:
            action = random.randint(0, num_subsets-1)
        else:
            action = torch.argmax(q_values).item()
        q_value = q_values[action]
        return action, q_value

    def q_step(self, q_value, reward, next_state):
        if self.env.is_solution():
            target = torch.Tensor([reward])
        else:
            target = torch.Tensor([reward]) + torch.max(self.model(next_state))
        error = self.loss(q_value.flatten(), target.flatten())
        self.optim.zero_grad()
        error.backward()
        self.optim.step()
        return target, error

    def train(self):
        is_solution = False
        state = self.env.get_state()
        err = []
        while not is_solution:
            action, q_value = self.get_action(state)
            next_state, reward, is_solution, _ = self.env.step(action)
            target, error = self.q_step(q_value, reward, next_state)
            err.append(error.item())
            state = next_state
        return sum(err)/len(err)


class DDQN_Agent:
    def __init__(self, env, model, n_step, memory_capacity):
        self.env = env
        self.epsilon = 0.3
        self.model = model
        self.optim = torch.optim.Adam(model.parameters())
        self.loss = torch.nn.MSELoss()
        self.n_step = n_step
        self.memory_buffer = ReplayBuffer(memory_capacity)

    def get_action(self, state):
        num_subsets = self.env.get_action_space()
        q_values = self.model(state)
        if random.random() < self.epsilon:
            action = random.randint(0, num_subsets-1)
        else:
            action = torch.argmax(q_values).item()
        q_value = q_values[action]
        return action, q_value

    def n_q_step(self, q_value, reward_sum, next_state, is_solution):
        if is_solution:
            target = torch.Tensor([reward_sum])
        else:
            target = torch.Tensor([reward_sum]) + torch.max(self.model(next_state))
        error = self.loss(q_value.flatten(), target.flatten())
        self.optim.zero_grad()
        error.backward(retain_graph=True)
        self.optim.step()
        return target, error

    def train(self):
        is_solution = False
        state = self.env.get_state()
        err = []
        t=0
        reward_memory = deque(maxlen=self.n_step)
        q_value_memory = deque(maxlen=self.n_step)
        while not is_solution:
            action, q_value = self.get_action(state)
            t = t+1
            next_state, reward, is_solution, _ = self.env.step(action)
            q_value_memory.append(q_value)
            reward_memory.append(reward)
            if t >= self.n_step:
                self.memory_buffer.add(q_value[0], sum(reward_memory), next_state, is_solution)
                q_value, reward_sum, n_state, was_solution = self.memory_buffer.sample()
                target, error = self.n_q_step(q_value, reward_sum, n_state, was_solution)
                err.append(error.item())
            state = next_state
            if t < self.n_step and is_solution:
                self.memory_buffer.add(q_value[0], sum(reward_memory), next_state, is_solution)
                q_value, reward_sum, n_state, was_solution = self.memory_buffer.sample()
                target, error = self.n_q_step(q_value, reward_sum, n_state, was_solution)
                err.append(error.item())
        return sum(err)/len(err)

    def eval(self):
        state = self.env.get_state()
        is_solution = False
        while not is_solution:
            print('Computing Q-values and deciding on an action')
            q_values = self.model(state)
            action = torch.argmax(q_values).item()
            state, reward, is_solution, _ = self.env.step(action)
            print('Uncovered elements remaining: ', self.env.instance.size()[0])
        return self.env.get_solution

