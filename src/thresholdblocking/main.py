import numpy as np

from tqdm import tqdm


def threshold_blocking(cost, k):
    if isinstance(cost, list):
        cost = np.array(cost)

    pbar = tqdm(
        total=5*cost.shape[0],
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
    )

    # Step 1. Construct (k - 1)-nearest neighbor subgraph.
    vertices = np.array(range(cost.shape[0]))
    edges = np.zeros(cost.shape, dtype=int)

    pbar.set_description('Construct nearest neighbor subgraph')
    for v in vertices:
        edges[v, v] = 1
        for idx in np.argpartition(cost[v,:], k)[:k]:
            edges[v, idx] = 1
            edges[idx, v] = 1

        pbar.update(1)


    # Step 2.a: Create second power graph
    edges_2 = edges.copy()
    pbar.set_description('Calculate second power graph       ')
    for v0 in vertices:
        for v1 in [v for v in vertices if v > v0]:
            if edges[v0, v1] == 1:
                edges_2[v0, :] = np.maximum(edges_2[v0, :], edges[:, v1].T)
                edges_2[:, v0] = edges_2[v0, :]

        pbar.update(1)
        pbar.refresh()

    # Step 2.b: Create maximal independent set of vertices in second power graph,
    #   called block seeds.
    seeds = []
    pbar.set_description('Collect seeds                      ')
    for v in vertices:
        if not np.any(edges_2[v, np.array(seeds, dtype=int)]):
            seeds.append(v)

        pbar.update(1)
        pbar.refresh()

    # Step 3: For each seed, create a block comprising its closed neighborhood.
    blocks = []
    unassigned = list(vertices)
    pbar_step = len(vertices) // len(seeds)
    pbar.set_description('Create threshold blocks            ')
    for s in seeds:
        block = []
        for v in vertices:
            if v in unassigned:
                if edges[s, v] == 1:
                    block.append(v)
                    unassigned.remove(v)

        blocks.append(block)

        pbar.update(pbar_step)
        pbar.refresh()


    # Step 4: Add unassign vertices to block with smallest distance.
    def connected(v, block):
        for b in block:
            if edges[v, b] == 1:
                return True
        return False

    def distance(v, block):
        return np.mean(cost[v, block])

    pbar_step = len(vertices) // len(unassigned) if len(unassigned) > 0 else 0
    for v in unassigned:
        i = np.argmin([distance(v, block) for block in blocks])
        blocks[i].append(v)

        pbar.update(pbar_step)
        pbar.refresh()

    unassigned = []

    # Step 5: Split blocks that are too big.
    new_blocks = []
    for block in blocks:
        if len(block) >= 2*k:
            split = [list(b) for b in np.array_split(block, len(block) // k)]
            new_blocks.extend(split)
        else:
            new_blocks.append(block)

    pbar.update(5*cost.shape[0] - pbar.n)

    return new_blocks
