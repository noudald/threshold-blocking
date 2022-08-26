from thresholdblocking import threshold_blocking


def test_threshold_blocking():
    blocks = threshold_blocking([[1, 1], [1, 1]], 1)

    assert len(blocks) > 0
    for block in blocks:
        assert len(blocks) > 1
