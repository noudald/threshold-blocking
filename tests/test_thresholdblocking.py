import numpy as np

from thresholdblocking import threshold_blocking


def test_simple_case_1():
    blocks = threshold_blocking([
        [0.0, 1.0],
        [1.0, 0.0]
    ], 1)

    assert blocks[0] == [0]
    assert blocks[1] == [1]

def test_simple_case_2():
    blocks = threshold_blocking([
        [0.0, 0.1, 1.0, 1.0],
        [0.1, 0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 0.1],
        [1.0, 1.0, 0.1, 0.0]
    ], 2)

    assert blocks[0] == [0, 1]
    assert blocks[1] == [2, 3]

def test_complex_case():
    rng = np.random.RandomState(37)
    x = rng.uniform(0, 1, size=(100, 100))
    cost = x @ x.T

    blocks = threshold_blocking(cost, 10)
    assert len(blocks) > 8
    for block in blocks:
        assert len(block) >= 10

    assert sorted([i for block in blocks for i in block]) == list(range(100))
