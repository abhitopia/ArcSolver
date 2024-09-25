#%%
import sys
src_path = '/Users/abhishekaggarwal/synced_repos/ArcSolver'
sys.path.append(src_path)  # replace "/path/to/src" with the actual path to the src directory

import torch
from src.mask_utils import create_enc_dec_mask

# Updated Test Case
def test_create_attention_mask():
    # B = 1
    # I_max = 5
    # O_max = 5
    x_valid_mask = torch.tensor([[1, 1, 1, 1, 0],
                                 [1, 1, 0, 0, 0],
                                 [1, 1, 1, 1, 1]], dtype=torch.bool)

    y_valid_mask = torch.tensor([[1, 1, 0, 0],
                                 [1, 1, 1, 1],
                                 [1, 0, 0, 0]], dtype=torch.bool)

    expected_mask_0 = torch.tensor([
        # x tokens (positions 0 to 4)
        [1, 1, 1, 1, 0, 0, 0, 0, 0],  # x[0] attends to x[0-2]
        [1, 1, 1, 1, 0, 0, 0, 0, 0],  # x[1] attends to x[0-2]
        [1, 1, 1, 1, 0, 0, 0, 0, 0],  # x[2] attends to x[0-2]
        [1, 1, 1, 1, 0, 0, 0, 0, 0],  # Padding token
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Padding token
        # y tokens (positions 5 to 9)
        [1, 1, 1, 1, 0, 1, 0, 0, 0],  # y[0] attends to x[0-2], y[0]
        [1, 1, 1, 1, 0, 1, 1, 0, 0],  # y[1] attends to x[0-2], y[0-1]
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Padding token
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Padding token
    ], dtype=torch.bool).unsqueeze(0)  # Add batch dimension


    expected_mask_1 = torch.tensor([
        # x tokens (positions 0 to 4)
        [1, 1, 0, 0, 0, 0, 0, 0, 0],  # x[0] attends to x[0-2]
        [1, 1, 0, 0, 0, 0, 0, 0, 0],  # x[1] attends to x[0-2]
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # x[2] attends to x[0-2]
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Padding token
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Padding token
        # y tokens (positions 5 to 9)
        [1, 1, 0, 0, 0, 1, 0, 0, 0],  # y[0] attends to x[0-2], y[0]
        [1, 1, 0, 0, 0, 1, 1, 0, 0],  # y[1] attends to x[0-2], y[0-1]
        [1, 1, 0, 0, 0, 1, 1, 1, 0],  # Padding token
        [1, 1, 0, 0, 0, 1, 1, 1, 1],  # Padding token
    ], dtype=torch.bool).unsqueeze(0)  # Add batch dimension


    expected_mask_2 = torch.tensor([
        # x tokens (positions 0 to 4)
        [1, 1, 1, 1, 1, 0, 0, 0, 0],  # x[0] attends to x[0-2]
        [1, 1, 1, 1, 1, 0, 0, 0, 0],  # x[1] attends to x[0-2]
        [1, 1, 1, 1, 1, 0, 0, 0, 0],  # x[2] attends to x[0-2]
        [1, 1, 1, 1, 1, 0, 0, 0, 0],  # Padding token
        [1, 1, 1, 1, 1, 0, 0, 0, 0],  # Padding token
        # y tokens (positions 5 to 9)
        [1, 1, 1, 1, 1, 1, 0, 0, 0],  # y[0] attends to x[0-2], y[0]
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # y[1] attends to x[0-2], y[0-1]
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Padding token
        [0, 0, 0, 0, 0, 0, 0, 0, 0],  # Padding token
    ], dtype=torch.bool).unsqueeze(0)  # Add batch dimension


    attention_mask = create_enc_dec_mask(x_valid_mask, y_valid_mask)
    # print(attention_mask[0].unsqueeze(0).shape, expected_mask_0.shape)
    assert torch.equal(attention_mask[0].unsqueeze(0), expected_mask_0), "The attention mask for batch 0 does not match the expected mask."
    assert torch.equal(attention_mask[1].unsqueeze(0), expected_mask_1), "The attention mask for batch 1 does not match the expected mask."
    assert torch.equal(attention_mask[2].unsqueeze(0), expected_mask_2), "The attention mask for batch 2 does not match the expected mask."

    print("Test passed: The attention mask matches the expected output.")

# Run the test
test_create_attention_mask()

# %%
