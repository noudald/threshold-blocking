import numpy as np

from tqdm import tqdm

ns = 1000
nf = 2
k = 3

rng = np.random.RandomState(37)
data = rng.uniform(0, 1, size=(ns, nf))

cost = np.zeros((ns, ns))
for i in range(ns):
    for j in range(i + 1, ns):
        cost_ = np.sqrt(np.sum((data[i,:] - data[j,:])**2))
        cost[i, j] = cost_
        cost[j, i] = cost_

vertices = np.array(range(ns))
edges = np.zeros((ns, ns))

for v in tqdm(vertices):
    edges[v, v] = 1
    for idx in np.argpartition(cost[v,:], k)[:k]:
        edges[v, idx] = 1
        edges[idx, v] = 1

print(vertices)
print(edges)


# Step 2.a: Create second power graph
edges_2 = edges.copy()

for v0 in tqdm(vertices):
    for v1 in [v for v in vertices if v > v0]:
        if edges[v0, v1] == 1:
            edges_2[v0, :] = np.maximum(edges_2[v0, :], edges[:, v1].T)
            edges_2[:, v0] = edges_2[v0, :]

print(edges_2)
