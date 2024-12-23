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
