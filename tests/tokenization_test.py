#%%
import sys
src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory

import numpy as np

from src.tokenizer import GridTokenizer, GridSerializer

def test_grid_serialize_tokenize(h=4, w=5):
    tokenizer = GridTokenizer()
    x = np.random.randint(0, 10, (h, w))
    x_str, indices = GridSerializer.serialize_array(x)
    x_tokenized = tokenizer.encode(x_str)
    x_decoded = tokenizer.decode(x_tokenized)
    x_reconst = GridSerializer.deserialize_array(x_decoded)

    grid_indices = []
    for r, c in indices:
        if c == 0 or c == w + 1:
            continue
        else:
            grid_indices.append((r, c - 1))

    assert len(grid_indices) == h * w
    for r, c in grid_indices:
        assert x[r, c] == x_reconst[r, c]

    assert np.all(x == x_reconst)

test_grid_serialize_tokenize(1, 1)
test_grid_serialize_tokenize(4, 5)
test_grid_serialize_tokenize(5, 5)
test_grid_serialize_tokenize(1, 100)
test_grid_serialize_tokenize(100, 1)

#%%