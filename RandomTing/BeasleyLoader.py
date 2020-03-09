#Features from SVD
import math
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime


def adj_to_features(adj):
    pca = PCA(n_components=128)
    pca.fit(adj.T)
    return pca.transform(adj.T)

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


path = 'rail582.txt'
print('Loading data from ', path)
adj, cost = read_as_adj(path)
print('Finished loading')
time = datetime.now()
print('Running PCA')
data = StandardScaler().fit_transform(adj)
print(adj_to_features(data))
print('Finished PCA in: ', datetime.now()-time, ' seconds')
