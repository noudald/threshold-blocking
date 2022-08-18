import numpy as np

ns = 1000
nf = 4
k = 10

rng = np.random.RandomState(37)
data = rng.uniform(0, 1, size=(ns, nf))

cost = np.zeros((ns, ns))
for i in range(ns):
    for j in range(i + 1, ns):
        cost_ = np.sqrt(np.sum((data[i,:] - data[j,:])**2))
        cost[i, j] = cost_
        cost[j, i] = cost_

# Step 1: Construct (k - 1)-nearest neighbor subgraph of G
vertices = np.array(range(ns))
edges = []

for v in vertices:
    for idx in np.argpartition(cost[v,:], k)[:k]:
        if idx != v and (idx, v) not in edges:
            edges.append((v, idx))

print('V', vertices)
print('E', edges)

# Step 2.a: Create second power graph
edges_2 = []
for e0 in edges:
    edges_2.append(e0)
    for e1 in edges:
        if e0[1] == e1[0] and (e0[0], e1[1]) not in edges_2:
            edges_2.append((e0[0], e1[1]))
        if e0[1] == e1[1] and e0[0] != e1[0]:
            f_0 = e0[0]
            f_1 = e1[0]
            if f_0 < f_1:
                f = (f_0, f_1)
            else:
                f = (f_1, f_0)
            if f not in edges_2:
                edges_2.append(f)


print('E^2', edges_2)

# Step 2.b: Create maximal independent set of vertices in second power graph, called block seeds.
def is_independent(v, seeds, edges):
    if v in seeds:
        return False

    for (e0, e1) in edges:
        if e0 == v and e1 in seeds:
            return False
        if e1 == v and e0 in seeds:
            return False

    return True

seeds = []
for v in vertices:
    if is_independent(v, seeds, edges_2):
        seeds.append(v)

print('Seeds', seeds)

# Step 3: For each seed, create a block comprising its closed neighborhood.
unassigned = list(vertices.copy())
blocks = []
for s in seeds:
    block = [s]
    unassigned.remove(s)
    for e in edges:
        if e[0] == s and e[1] in unassigned:
            block.append(e[1])
            unassigned.remove(e[1])
        elif e[1] == s and e[0] in unassigned:
            block.append(e[0])
            unassigned.remove(e[0])
    blocks.append(block)


# Step 4: Add unassigned vertices to blocks.
print('Unassigned', unassigned)

for block in blocks:
    for vb in block:
        for vu in unassigned:
            if (vu, vb) in edges or (vb, vu) in edges:
                block.append(vu)
                unassigned.remove(vu)

print('Blocks', blocks)
