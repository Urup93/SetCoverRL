from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen= buffer_size)

    def add(self, q_value, reward_sum, next_state, is_solution):
        experience = (q_value, reward_sum, next_state, is_solution)
        self.buffer.append(experience)

    def size(self):
        return self.count

    def sample(self):
        q_value, reward_sum, next_state, is_solution = random.sample(self.buffer, 1)[0]
        return q_value, reward_sum, next_state, is_solution

    def clear(self):
        self.buffer.clear()