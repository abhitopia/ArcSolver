
import torch


def create_enc_dec_mask(enc_valid_mask, dec_valid_mask):
    """
    Creates an attention mask of shape [B, L, L] where invalid tokens attend only to themselves.

    Args:
        enc_valid_mask (torch.Tensor): Boolean tensor of shape [B, I_max], True for valid x tokens.
        dec_valid_mask (torch.Tensor): Boolean tensor of shape [B, O_max], True for valid y tokens.

    Returns:
        torch.Tensor: Attention mask of shape [B, L, L].
    """
    B, I_max = enc_valid_mask.shape
    By, O_max = dec_valid_mask.shape
    assert B == By, "Batch sizes of x and y must match."

    L = I_max + O_max

    # Initialize the attention mask with zeros (False)
    M = torch.zeros((B, L, L), dtype=torch.bool, device=enc_valid_mask.device)

    # x-x attention: valid x tokens attend to valid x tokens
    x_mask = enc_valid_mask.unsqueeze(2) & enc_valid_mask.unsqueeze(1)  # [B, I_max, I_max]
    M[:, :I_max, :I_max] = x_mask

    # y-x attention: valid y tokens attend to valid x tokens
    yx_mask = dec_valid_mask.unsqueeze(2) & enc_valid_mask.unsqueeze(1)  # [B, O_max, I_max]
    M[:, I_max:, :I_max] = yx_mask

    # y-y attention with causal mask
    causal_mask_y = torch.tril(torch.ones((1, O_max, O_max), dtype=torch.bool, device=enc_valid_mask.device))
    yy_mask = (dec_valid_mask.unsqueeze(2) & dec_valid_mask.unsqueeze(1)) & causal_mask_y  # [B, O_max, O_max]
    M[:, I_max:, I_max:] = yy_mask

    # Adjust M so that invalid tokens attend only to themselves
    invalid_enc_positions = ~enc_valid_mask  # [B, I_max]
    invalid_dec_positions = ~dec_valid_mask  # [B, O_max]
    invalid_positions = torch.cat([invalid_enc_positions, invalid_dec_positions], dim=1)  # [B, L]

    # Expand invalid_positions to match M's dimensions
    invalid_positions_row = invalid_positions.unsqueeze(2)  # [B, L, 1]
    invalid_positions_col = invalid_positions.unsqueeze(1)  # [B, 1, L]

    # Create a mask for invalid positions
    invalid_positions_matrix = invalid_positions_row | invalid_positions_col  # [B, L, L]

    # Set M to False where either the row or column is an invalid position
    M[invalid_positions_matrix] = False

    # Set diagonal elements for invalid positions to True (allow self-attention)
    diag_indices = torch.arange(L, device=enc_valid_mask.device)
    M[:, diag_indices, diag_indices] |= invalid_positions

    return M