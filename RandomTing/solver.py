from SetCoverPy import setcover
import numpy as np
import docplex.mp.model as cpx
import pandas as pd
import math
from datetime import datetime
from scipy import sparse


def generate_adj_mat(u, s, p):
    adj = np.zeros((u, s))
    for i in range(u):
        for j in range(s):
            if np.random.rand() < p:
                adj[i,j] = 1
        if np.sum(adj[i,:]) == 0:
            idx = np.random.choice(range(s))
            adj[i, idx] = 1
    return adj


class greedySolver:
    def solve(self, adj, cost, grasp = False):
        u, s = np.shape(adj)
        solution = []
        c = 0
        elements_covered = set()
        while len(elements_covered) != u:
            scores = cost / np.sum(adj, axis=0)
            if grasp:
                best_s = np.random.choice(range(s), p=1-(scores / np.sum(scores)))
            else:
                best_s = np.argmin(scores)
            for row in range(u):
                if adj[row, best_s] == 1:
                    elements_covered.add(row)
                    adj[row, :] = 0
            solution.append(best_s)
            c += cost[best_s]
        return solution, c


class cplexSolver:
    def solve(self, adj):
        u, s = np.shape(adj)
        U = range(1, u + 1)
        S = range(1, s + 1)
        opt_model = cpx.Model(name="SCP Model")
        x_vars = {i: opt_model.binary_var(name="x_{0}".format(i)) for i in S}
        constraints = {j: opt_model.add_constraint(ct=opt_model.sum(adj[j - 1, i - 1] * x_vars[i] for i in S) >= 1, ctname="constraint_{0}".format(j)) for j in U}
        objective = opt_model.sum(x_vars[i] for i in S)
        opt_model.minimize(objective)
        opt_model.solve()
        opt_df = pd.DataFrame.from_dict(x_vars, orient="index", columns=["variable_object"])
        opt_df.reset_index(inplace=True)
        opt_df["solution_value"] = opt_df["variable_object"].apply(lambda item: item.solution_value)
        return opt_df


def read_as_adj(path):
    with open(path, 'r') as file:
        if 'rail' in path:
            # Read amount of rows and cols
            line = file.readline()
            size = line.split()
            size = [int(size[0]), int(size[1])]

            # Init adj mat and cost vec
            adj = np.zeros(size, dtype=bool)
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


if __name__ == '__main__':
    path = 'rail582.txt'
    print('Loading data from ', path)
    adj, cost = read_as_adj(path)

    #print('read adj mat: ', '\n', adj, '\n')
    #print('read cost: ', '\n', cost)

    # cplex = cplexSolver()
    # time = datetime.now()
    # cplex_sol = cplex.solve(adj)
    # cplex_time = datetime.now()-time

    print('Running set cover py')
    scppy = setcover.SetCover(adj, cost)
    time = datetime.now()
    scppy_solution, time_used = scppy.SolveSCP()
    scppy_time = datetime.now()-time
    print('SCP py time: ', scppy_time)
    print('SCP solution: ', scppy_solution)


    print('Running greedy solver')
    gs = greedySolver()
    time = datetime.now()
    greedy_solution, greedy_cost = gs.solve(adj, cost, False)
    greedy_time = datetime.now()-time
    print('Greedy time: ', greedy_time)
    print('Greedy cost: ', greedy_cost)


    # result = pd.DataFrame({'Model': ['Greedy', 'CPlex', 'SCPPy'],
    #                        'Time': [greedy_time, cplex_time, scppy_time],
    #                        'Solution Quality': [len(greedy_solution), sum(cplex_sol['solution_value']), scppy_solution]})
    # result.set_index(['Model'])
    # print(result)
    #
    # print('CPLEX time: ', cplex_time)
    # print('CPLEX solution: ', sum(cplex_sol['solution_value']))













