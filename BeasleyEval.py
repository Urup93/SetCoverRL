import os
from Agent import *
from models import *
from set_cover_env import *
import math
import RandomTing.solver as solver

def read_as_adj(path):
    with open(path, 'r') as file:
        if 'rail' in path:
            # Read amount of rows and cols
            line = file.readline()
            size = line.split()
            size = [int(size[0]), int(size[1])]

            # Init adj mat and cost vec
            adj = np.zeros(size, dtype=int)
            cost = np.zeros(size[1])
            col = 0

            # Read subsets line by line and update adj mat accordingly
            while True:
                line = file.readline()
                if not line:
                    break
                line = line.split()
                cost[col] = int(line[0])
                for row in line[2:len(line)]:
                    adj[int(row)-1, col] = 1
                col += 1
        elif 'scp' in path:
            # Read amount of cols and rows
            line = file.readline()
            size = line.split()
            size = [int(size[1]), int(size[0])]

            # Init adj mat and cost vec
            adj = np.zeros(size)
            cost = np.zeros(size[1])

            #Read costs and store in costs
            lines_with_cost = range(math.ceil(len(cost)/15))
            for i in lines_with_cost:
                line = file.readline()
                line = line.split()
                if i == int(lines_with_cost[-1]):
                    cost[i*15:] = np.array(list(map(int, line)))
                else:
                    cost[i*15:i*15+15] = np.array(list(map(int, line)))
            #Read subset length and subset
            col = 0
            while True:
                line = file.readline()
                if not line:
                    break
                subset_length = int(line.split()[0])
                lines_defining_subset = range(math.ceil(subset_length/15))
                for i in lines_defining_subset:
                    line = file.readline()
                    line = line.split()
                    for row in line:
                        adj[int(row)-1, col] = 1
                col += 1
        else:
            print('Error in format')
        return adj, cost


s, u, p = 1, 1, 1
env = SetCoverEnv(s, u, p)
model = SubsetRanking(n_uni_feat=1, n_sub_feat=3, n_hid=64)
model.eval()
dir_path = os.path.dirname(os.path.realpath(__file__))
if os.path.isfile(dir_path):
    model.load_model(dir_path)
agent = DDQN_Agent(env, model, 5, 40)

path = 'rail582.txt'
print('Loading data from ', path)
adj, cost = read_as_adj(path)
print(adj)
cost = np.ones_like(cost)
adj_torch = torch.from_numpy(adj).float()
print('Finished loading')
env.set_instance(adj_torch)

print('evaling model')
agent.eval()

solution = env.get_solution()
print(sum(solution))

print('computing greedy solution')
greedy = solver.greedySolver()
greedy_sol, c = greedy.solve(adj, cost)

print(len(greedy_sol))