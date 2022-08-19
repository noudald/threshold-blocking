import numpy as np

from scipy.spatial import distance_matrix
from tqdm import tqdm

ns = 20
nf = 2
k = 2

rng = np.random.RandomState(37)
data = rng.uniform(0, 1, size=(ns, nf))

print('Calculate cost matrix')
cost = distance_matrix(data, data)


# Step 1. Construct (k - 1)-nearest neighbor subgraph.
vertices = np.array(range(ns))
edges = np.zeros((ns, ns), dtype=int)

for v in tqdm(vertices, bar_format='Construct (k - 1)-nearest neighbor subgraph {l_bar}{bar:10}{r_bar}{bar:-10b}'):
    edges[v, v] = 1
    for idx in np.argpartition(cost[v,:], k)[:k]:
        edges[v, idx] = 1
        edges[idx, v] = 1


# Step 2.a: Create second power graph
edges_2 = edges.copy()

for v0 in tqdm(vertices, bar_format='Calculate second power graph                {l_bar}{bar:10}{r_bar}{bar:-10b}'):
    for v1 in [v for v in vertices if v > v0]:
        if edges[v0, v1] == 1:
            edges_2[v0, :] = np.maximum(edges_2[v0, :], edges[:, v1].T)
            edges_2[:, v0] = edges_2[v0, :]

# Step 2.b: Create maximal independent set of vertices in second power graph,
#   called block seeds.
seeds = []
for v in tqdm(vertices, bar_format='Collect seeds                               {l_bar}{bar:10}{r_bar}{bar:-10b}'):
    if not np.any(edges_2[v, np.array(seeds, dtype=int)]):
        seeds.append(v)

# Step 3: For each seed, create a block comprising its closed neighborhood.
blocks = []
unassigned = list(vertices)
for s in tqdm(seeds, bar_format='Create threshold blocks                     {l_bar}{bar:10}{r_bar}{bar:-10b}'):
    block = []
    for v in vertices:
        if v in unassigned:
            if edges[s, v] == 1:
                block.append(v)
                unassigned.remove(v)

    blocks.append(block)


# Step 4: Add unassign vertices to block with smallest distance.
print('Add all remaining vertices to threshold blocks')

def connected(v, block):
    for b in block:
        if edges[v, b] == 1:
            return True
    return False

def distance(v, block):
    return np.mean(cost[v, block])

for v in unassigned:
    i = np.argmin([distance(v, block) for block in blocks])
    blocks[i].append(v)
unassigned = []

def within_distance(block):
    s = 0
    for b0 in block:
        for b1 in block:
            s += cost[b0, b1]
    return s

for block in blocks:
    print(
        block,
        'size',
        len(block),
        round(within_distance(block), 2),
        round(within_distance(rng.choice(vertices, len(block), replace=False)), 2)
    )
