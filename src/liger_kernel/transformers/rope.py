from liger_kernel.ops.rope import LigerRopeFunction

def apply_rotary_positional_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    cos_embeddings: torch.Tensor,
    sin_embeddings: torch.Tensor,
    position_ids: torch.Tensor = None,
    unsqueeze_dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Positional Embedding (RoPE) operation to query and key states.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, num_query_heads, sequence_length, head_dim).
        key (torch.Tensor): Key tensor of shape (batch_size, num_kv_heads, sequence_length, head_dim).
        cos_embeddings (torch.Tensor): Cosine embeddings tensor of shape (1, sequence_length, head_dim).
        sin_embeddings (torch.Tensor): Sine embeddings tensor of shape (1, sequence_length, head_dim).
        position_ids (torch.Tensor, optional): Position ids tensor. Defaults to None.
        unsqueeze_dim (int, optional): Dimension to unsqueeze. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Query and key tensors after applying the RoPE operation.
    """

    return LigerRopeFunction.apply(
        query, key, cos_embeddings, sin_embeddings, position_ids, unsqueeze_dim
    )
