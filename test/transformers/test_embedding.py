import pytest
import torch
from torch.nn import Embedding

from liger_kernel.transformers.embedding import LigerEmbedding

SLEEP_SECONDS = 0.1

@pytest.mark.parametrize(
    "num_embeddings, embedding_dim",
    [
        (100, 64),
        (100, 64),
        (1000, 128),
        (1000, 128),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-6, 1e-5),
    ],
)
def test_embedding_correctness(num_embeddings, embedding_dim, dtype, atol, rtol):
    torch.manual_seed(42)

    torch_embedding = Embedding(num_embeddings, embedding_dim).to(dtype).to("cuda")
    liger_embedding = LigerEmbedding(num_embeddings, embedding_dim).to(dtype).to("cuda")
    liger_embedding.embeddings.data.copy_(torch_embedding.weight.data)

    input_ids = torch.randint(0, num_embeddings, (32 * 10,), device="cuda")
    torch_output = torch_embedding(input_ids).view(32, 10, -1)
    liger_output = liger_embedding(input_ids).view(32, 10, -1)

    assert torch.allclose(torch_output, liger_output, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(torch_output)
    torch_output.backward(grad_output)
    liger_output.backward(grad_output)

    assert torch.allclose(torch_embedding.weight.grad, liger_embedding.embeddings.grad, atol=atol, rtol=rtol)
