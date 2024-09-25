
import torch


def create_enc_dec_mask(enc_valid_mask, dec_valid_mask):
    """
    Creates an attention mask of shape [B, I+O, I+O] based on the provided valid masks for x and y.

    Args:
        enc_valid_mask (torch.Tensor): A boolean tensor of shape [B, I_max], where True indicates valid tokens in x.
        dec_valid_mask (torch.Tensor): A boolean tensor of shape [B, O_max], where True indicates valid tokens in y.

    Returns:
        torch.Tensor: The attention mask of shape [B, I+O, I+O].
    """
    B, I_max = enc_valid_mask.shape
    By, O_max = dec_valid_mask.shape
    assert B == By, "Batch sizes of x and y must match."

    L = I_max + O_max

    # Initialize the attention mask with zeros
    M = torch.zeros((B, L, L), dtype=torch.bool, device=enc_valid_mask.device)

    # x-x attention: valid x tokens attend to valid x tokens
    x_mask = enc_valid_mask.unsqueeze(2) & enc_valid_mask.unsqueeze(1)  # Shape: [B, I_max, I_max]
    M[:, :I_max, :I_max] = x_mask

    # y-x attention: valid y tokens attend to valid x tokens
    yx_mask = dec_valid_mask.unsqueeze(2) & enc_valid_mask.unsqueeze(1)  # Shape: [B, O_max, I_max]
    M[:, I_max:, :I_max] = yx_mask

    # y-y attention with causal mask
    causal_mask_y = torch.tril(torch.ones((1, O_max, O_max), dtype=torch.bool, device=enc_valid_mask.device))
    yy_mask = (dec_valid_mask.unsqueeze(2) & dec_valid_mask.unsqueeze(1)) & causal_mask_y  # Shape: [B, O_max, O_max]
    M[:, I_max:, I_max:] = yy_mask

    # x cannot attend to y tokens (already initialized to False)
    # Ensure padding tokens do not partake in any computation (already handled by valid masks)

    return M