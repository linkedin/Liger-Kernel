import pytest
import torch
from torch.nn import Embedding
from liger_kernel.transformers.embedding import LigerEmbedding
import time

SLEEP_SECONDS = 0.1

@pytest.mark.parametrize(
    "num_embeddings, embedding_dim",
    [
        (100, 64),
        (100, 64),
        (1000, 128),
        (100, 60),
        (100, 60),
        (1000, 120),
        (1000, 500),
        (30522, 768)
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-6, 1e-5),
    ],
)
def test_embedding_correctness(num_embeddings, embedding_dim, dtype, atol, rtol):
    print(f"\nTesting with embedding with the size: ({num_embeddings}, {embedding_dim})")
    torch.manual_seed(42)

    torch_embedding = Embedding(num_embeddings, embedding_dim).to(dtype).to("cuda")
    liger_embedding = LigerEmbedding(num_embeddings, embedding_dim).to(dtype).to("cuda")
    liger_embedding.weight.data.copy_(torch_embedding.weight.data)

    input_ids = torch.randint(0, num_embeddings, (32 * 10,), device="cuda")
    
    start_time = time.time()
    torch_output = torch_embedding(input_ids).view(32, 10, -1)
    torch_forward_time = time.time() - start_time
    print(f"nn.Embedding forward time: {torch_forward_time:.6f} seconds")

    start_time = time.time()
    liger_output = liger_embedding(input_ids).view(32, 10, -1)
    liger_forward_time = time.time() - start_time
    print(f"LigerEmbedding forward time: {liger_forward_time:.6f} seconds")
    print(f"Forward pass speedup: {torch_forward_time / liger_forward_time:.2f}x")

    assert torch.allclose(torch_output, liger_output, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(torch_output)
    
    start_time = time.time()
    torch_output.backward(grad_output)
    torch_backward_time = time.time() - start_time
    print(f"nn.Embedding backward time: {torch_backward_time:.6f} seconds")

    start_time = time.time()
    liger_output.backward(grad_output)
    liger_backward_time = time.time() - start_time
    print(f"LigerEmbedding backward time: {liger_backward_time:.6f} seconds")
    print(f"Backward pass speedup: {torch_backward_time / liger_backward_time:.2f}x")

    assert torch.allclose(torch_embedding.weight.grad, liger_embedding.weight.grad, atol=atol, rtol=rtol)
