from unittest import TestCase
from MemoryBuffer import *

class TestReplayBuffer(TestCase):
    def test_sample(self):
        buffer = ReplayBuffer(2)
        buffer.add(1, 2, 3, 4)
        buffer.add(5, 6, 7, 8)
        buffer.add(9, 10, 11, 12)
        print(buffer.sample())

        a, b, c, d = buffer.sample()
        print(a, b, c, d)


