import pytest
import torch

from liger_kernel.ops.experimental.mm_int8int2 import matmul
from liger_kernel.ops.experimental.mm_int8int2 import pack_weights
from liger_kernel.ops.experimental.mm_int8int2 import unpack_weights
from liger_kernel.utils import infer_device

device = infer_device()


# input_features = size*4 when the weight matrix is unpacked
@pytest.mark.skip(reason="mm_int8int2 is under experimentation")
@pytest.mark.parametrize(
    "size",
    [
        2048,
        1024,
        512,
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [1, 2, 3, 8],
)
@pytest.mark.parametrize(
    "seq_len",
    [1, 7, 16, 2048],
)
@pytest.mark.parametrize(
    "out_features",
    [
        1024,
        2048,
        4096,
        10000,
    ],
)
@pytest.mark.parametrize(
    "atol, rtol, device",
    [
        (1e-2, 1e-2, device),
    ],
)
def test_kernel_correctness(batch_size, seq_len, out_features, size, atol, rtol, device):
    print(f"\nTesting kernel with size: {size}, atol: {atol}, rtol: {rtol}")

    # Generate the random tensors
    ht = torch.randint(-127, 127, (batch_size, seq_len, size * 4), device=device, dtype=torch.int8)
    u = torch.randint(0, 255, (out_features, size), device=device, dtype=torch.uint8)

    # Calculate dimensions
    B, M, N = ht.size()

    # Compute triton output
    triton_output = matmul(ht.view(B * M, N), u.T.contiguous()).view(B, M, -1)

    # Unpack weights and compute torch output
    unpacked = unpack_weights(u.T, bits=2).T
    torch_output = torch.matmul(ht.to(torch.float32), unpacked.T.contiguous().to(torch.float32))

    # Print the results (optional, can be commented out)
    print("triton_output =", triton_output)
    print("torch_output =", torch_output)

    # Check if outputs are close within the given tolerances
    assert torch.allclose(triton_output, torch_output.to(torch.int32), atol=atol, rtol=rtol), "Results differ"


@pytest.mark.skip(reason="mm_int8int2 is under experimentation")
@pytest.mark.parametrize(
    "size",
    [
        2048,
        1024,
        512,
    ],
)
@pytest.mark.parametrize(
    "out_features",
    [
        1024,
        2048,
        4096,
        10000,
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        device,
    ],
)
def test_unpack_pack_correctness(out_features, size, device):
    u = torch.randint(0, 255, (out_features, size), device=device, dtype=torch.uint8)

    assert (pack_weights(unpack_weights(u.T), 2) == u.T).all(), "Packed weights do not match original weights."
