import numpy as np
from scipy.spatial import distance_matrix

from thresholdblocking import threshold_blocking


ns = 100
nf = 2
k = 10

rng = np.random.RandomState(37)
data = rng.uniform(0, 1, size=(ns, nf))

print('Calculate cost matrix')
cost = distance_matrix(data, data)

blocks = threshold_blocking(cost, k)

def within_distance(block):
    return np.mean(cost[block, :][:, block])

def block_distance(block1, block2):
    return np.mean(cost[block1, :][:, block2])

for block in blocks:
    print(
        block,
        'size',
        len(block),
        round(within_distance(block), 4),
    )

inner_distance = [within_distance(block) for block in blocks]

outer_distance = []
for block1 in blocks:
    for block2 in blocks:
        if block1 == block2:
            continue
        outer_distance.append(block_distance(block1, block2))

print(f'Average within block distance: {np.mean(inner_distance):.2f}')
print(f'Average between block distance: {np.mean(outer_distance):.2f}')