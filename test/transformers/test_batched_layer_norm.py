import numpy
import pytest
import torch
import keras
from liger_kernel.transformers.functional import liger_layer_norm,liger_batch_norm

import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


@pytest.mark.parametrize(
    "hidden_size",
    [
        32,
        32,
        32,
        32,
    ],
)
@pytest.mark.parametrize(
    "batch_size, seq_len",
    [
        (2, 32),
        (8, 32),
        (2, 32),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
    ],
)

def test_liger_bacthed_layer_norm(batch_size, seq_len, hidden_size, dtype, atol, rtol):
    torch.manual_seed(0)

    x = torch.randn(
        batch_size, seq_len, hidden_size, dtype=dtype, device="cuda", requires_grad=True
    )
    keras_ln = keras.layers.BatchNormalization(epsilon=1e-6)

    keras_ln.training = False

    axis = -1

    eps = 1e-6


    liger_output = liger_batch_norm(x, axis,  eps)

    x = x.detach().cpu().numpy()
    keras_output = keras_ln(x)

    print(torch.Tensor(numpy.array(keras_output)).to('cuda'))
    print(liger_output)
    assert torch.allclose(liger_output,torch.Tensor(numpy.array(keras_output)).to('cuda'), atol=atol, rtol=rtol)


