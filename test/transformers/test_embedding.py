import pytest
import torch

from torch.nn import Embedding

from liger_kernel.transformers.experimental.embedding import LigerEmbedding
from liger_kernel.utils import infer_device

device = infer_device()

SLEEP_SECONDS = 0.1


@pytest.mark.skip(reason="LigerEmbedding is under experimentation")
@pytest.mark.parametrize(
    "num_embeddings, embedding_dim, padding_idx",
    [
        (100, 64, None),
        (100, 64, None),
        (1000, 128, None),
        (100, 60, None),
        (100, 60, None),
        (1000, 120, None),
        (1000, 500, None),
        (30522, 768, None),
        (100, 64, 0),
        (1000, 128, 50),
        (30522, 768, 1),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol, device",
    [
        (torch.float32, 1e-6, 1e-5, device),
    ],
)
def test_embedding_correctness(num_embeddings, embedding_dim, padding_idx, dtype, atol, rtol, device):
    print(f"\nTesting embedding with size: ({num_embeddings}, {embedding_dim}), padding_idx: {padding_idx}")
    torch.manual_seed(42)

    torch_embedding = Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx).to(dtype).to(device)
    liger_embedding = LigerEmbedding(num_embeddings, embedding_dim, padding_idx=padding_idx).to(dtype).to(device)
    liger_embedding.weight.data.copy_(torch_embedding.weight.data)

    if padding_idx is not None:
        input_ids = torch.randint(0, num_embeddings, (32 * 10,), device=device)
        input_ids[torch.randint(0, 32 * 10, (32 * 10 // 10,))] = padding_idx
    else:
        input_ids = torch.randint(0, num_embeddings, (32 * 10,), device=device)

    torch_output = torch_embedding(input_ids).view(32, 10, -1)
    liger_output = liger_embedding(input_ids).view(32, 10, -1)

    assert torch.allclose(torch_output, liger_output, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(torch_output)

    torch_output.backward(grad_output)
    liger_output.backward(grad_output)

    assert torch.allclose(torch_embedding.weight.grad, liger_embedding.weight.grad, atol=atol, rtol=rtol)


INT32_MAX = 2**31 - 1


def _free_device_memory_bytes():
    if not torch.cuda.is_available():
        return 0
    free, _ = torch.cuda.mem_get_info()
    return free


@pytest.mark.skipif(
    _free_device_memory_bytes() < 24 * 1024**3,
    reason="Needs about 20 GiB of free GPU memory for a >2^31-element output and its gradient",
)
def test_embedding_output_offsets_beyond_int32():
    """tokens * embedding_dim > 2^31: row offsets into output / grad_output must not wrap."""
    from liger_kernel.ops.experimental.embedding import LigerEmbeddingFunction

    num_embeddings, embedding_dim = 1024, 128
    num_tokens = INT32_MAX // embedding_dim + 1024
    assert num_tokens * embedding_dim > INT32_MAX

    torch.manual_seed(0)
    weight = torch.randn(num_embeddings, embedding_dim, device=device, dtype=torch.float32, requires_grad=True)
    indices = torch.randint(0, num_embeddings, (num_tokens,), device=device)

    output = LigerEmbeddingFunction.apply(weight, indices)
    # Rows past the int32 boundary must hold the gathered embeddings.
    tail = indices[-2048:]
    torch.testing.assert_close(output[-2048:], weight.detach()[tail], rtol=0, atol=0)

    # With an all-ones gradient each weight row's gradient is its exact hit count.
    output.backward(torch.ones_like(output))
    del output
    counts = torch.bincount(indices, minlength=num_embeddings).to(torch.float32)
    expected = counts[:, None].expand(-1, embedding_dim)
    torch.testing.assert_close(weight.grad, expected, rtol=0, atol=0)


@pytest.mark.skipif(
    _free_device_memory_bytes() < 24 * 1024**3,
    reason="Needs about 17 GiB of free GPU memory for a >2^31-element table and its gradient",
)
def test_embedding_table_offsets_beyond_int32_with_int32_indices():
    """num_embeddings * embedding_dim > 2^31 with int32 indices: table offsets must not wrap."""
    from liger_kernel.ops.experimental.embedding import LigerEmbeddingFunction

    embedding_dim = 128
    num_embeddings = INT32_MAX // embedding_dim + 1024
    assert num_embeddings * embedding_dim > INT32_MAX

    weight = torch.zeros(num_embeddings, embedding_dim, device=device, dtype=torch.float32)
    last_rows = torch.arange(num_embeddings - 512, num_embeddings, device=device)
    weight[last_rows] = last_rows.to(torch.float32)[:, None]
    weight.requires_grad_(True)
    indices = last_rows.to(torch.int32)

    output = LigerEmbeddingFunction.apply(weight, indices)
    torch.testing.assert_close(output, weight.detach()[last_rows], rtol=0, atol=0)

    output.backward(torch.ones_like(output))
    assert torch.all(weight.grad[last_rows] == 1)
    assert int(weight.grad.sum().item()) == last_rows.numel() * embedding_dim
