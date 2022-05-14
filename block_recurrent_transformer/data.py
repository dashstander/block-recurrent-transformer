from typing import List, Tuple


def get_seq_indices(seq_len: int, window_len: int) -> List[Tuple[int, int]]:
    prev = 0
    inds = []
    for curr in range(window_len, seq_len + 1, window_len):
        inds.append((prev, curr + 1))
    return inds


def long_sequence_splitter(batch, window_len: int):
    """
    """
    seq_len = batch.shape[1]
    for i, j in get_seq_indices(seq_len, window_len):
        yield batch[:, i:j]
