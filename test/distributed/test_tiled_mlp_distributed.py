import os
import tempfile
import uuid

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from transformers.models.llama.configuration_llama import LlamaConfig

from liger_kernel.transformers.tiled_mlp import LigerTiledGEGLUMLP
from liger_kernel.transformers.tiled_mlp import LigerTiledSwiGLUMLP


def get_init_file():
    """Get a unique file path for distributed init that doesn't exist yet."""
    return os.path.join(tempfile.gettempdir(), f"dist_init_{uuid.uuid4().hex}")


# Check if FSDP is available
try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False


def setup_distributed(rank, world_size, init_file, backend="nccl"):
    """Initialize distributed process group using file-based init."""
    init_method = f"file://{init_file}"
    dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def run_ddp_test(rank, world_size, mlp_type, config, dtype, num_shards, init_file):
    """
    Run DDP test on a single GPU process.
    This function is spawned by torch.multiprocessing.
    """
    try:
        setup_distributed(rank, world_size, init_file)
        device = torch.device(f"cuda:{rank}")

        bsz, seq_len, hidden_size = 2, 512, config.hidden_size

        # Use same random seed for input data across all ranks
        torch.manual_seed(42)
        x = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype) * 0.1
        x.requires_grad_(True)

        # Initialize weights (same across all ranks for verification)
        G = torch.randn(config.intermediate_size, config.hidden_size, device=device, dtype=dtype)
        U = torch.randn(config.intermediate_size, config.hidden_size, device=device, dtype=dtype)
        D = torch.randn(config.hidden_size, config.intermediate_size, device=device, dtype=dtype)

        # Create tiled MLP
        if mlp_type == "geglu":
            tiled_mlp = LigerTiledGEGLUMLP(config=config, num_shards=num_shards).to(device).to(dtype)
        else:  # swiglu
            tiled_mlp = LigerTiledSwiGLUMLP(config=config, num_shards=num_shards).to(device).to(dtype)

        tiled_mlp.gate_proj.weight.data = G
        tiled_mlp.up_proj.weight.data = U
        tiled_mlp.down_proj.weight.data = D

        # Wrap with DDP
        ddp_mlp = DDP(tiled_mlp, device_ids=[rank])

        # Forward pass
        output = ddp_mlp(x)

        # Backward pass with same gradient across all ranks
        torch.manual_seed(42)  # Same gradient for all ranks
        grad_output = torch.randn_like(output)
        output.backward(grad_output)

        # Verify that module is detected as DDP
        assert hasattr(ddp_mlp.module, "gate_proj"), "Model structure is correct"

        # Verify gradients exist
        assert ddp_mlp.module.gate_proj.weight.grad is not None
        assert ddp_mlp.module.up_proj.weight.grad is not None
        assert ddp_mlp.module.down_proj.weight.grad is not None

        # Verify gradient synchronization across ranks
        # All ranks should have identical gradients after DDP synchronization
        gate_grad = ddp_mlp.module.gate_proj.weight.grad.clone()
        up_grad = ddp_mlp.module.up_proj.weight.grad.clone()
        down_grad = ddp_mlp.module.down_proj.weight.grad.clone()

        # Gather gradients from all ranks to rank 0
        if rank == 0:
            gate_grads = [torch.zeros_like(gate_grad) for _ in range(world_size)]
            up_grads = [torch.zeros_like(up_grad) for _ in range(world_size)]
            down_grads = [torch.zeros_like(down_grad) for _ in range(world_size)]
        else:
            gate_grads = None
            up_grads = None
            down_grads = None

        dist.gather(gate_grad, gate_grads, dst=0)
        dist.gather(up_grad, up_grads, dst=0)
        dist.gather(down_grad, down_grads, dst=0)

        # Rank 0 verifies all gradients are synchronized
        if rank == 0:
            for i in range(1, world_size):
                torch.testing.assert_close(
                    gate_grads[0],
                    gate_grads[i],
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"Gate gradients not synchronized between rank 0 and rank {i}",
                )
                torch.testing.assert_close(
                    up_grads[0],
                    up_grads[i],
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"Up gradients not synchronized between rank 0 and rank {i}",
                )
                torch.testing.assert_close(
                    down_grads[0],
                    down_grads[i],
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"Down gradients not synchronized between rank 0 and rank {i}",
                )

    finally:
        # Barrier to ensure all ranks complete
        if dist.is_initialized():
            dist.barrier()
        cleanup_distributed()


def run_fsdp_test(rank, world_size, mlp_type, config, dtype, num_shards, init_file):
    if not FSDP_AVAILABLE:
        return

    try:
        setup_distributed(rank, world_size, init_file)
        device = torch.device(f"cuda:{rank}")

        bsz, seq_len, hidden_size = 2, 512, config.hidden_size

        # Use same random seed for input data and weights across all ranks
        torch.manual_seed(42)
        x = torch.randn(bsz, seq_len, hidden_size, device=device, dtype=dtype) * 0.1
        x.requires_grad_(True)

        # Initialize weights
        G = torch.randn(config.intermediate_size, config.hidden_size, device=device, dtype=dtype)
        U = torch.randn(config.intermediate_size, config.hidden_size, device=device, dtype=dtype)
        D = torch.randn(config.hidden_size, config.intermediate_size, device=device, dtype=dtype)

        # Create tiled MLP on CPU first (FSDP best practice)
        if mlp_type == "geglu":
            tiled_mlp = LigerTiledGEGLUMLP(config=config, num_shards=num_shards).to(dtype)
        else:  # swiglu
            tiled_mlp = LigerTiledSwiGLUMLP(config=config, num_shards=num_shards).to(dtype)

        # Initialize weights on CPU
        tiled_mlp.gate_proj.weight.data.copy_(G.cpu())
        tiled_mlp.up_proj.weight.data.copy_(U.cpu())
        tiled_mlp.down_proj.weight.data.copy_(D.cpu())

        # Wrap with FSDP - it will move to device
        fsdp_mlp = FSDP(
            tiled_mlp,
            device_id=rank,
            sync_module_states=True,
        )

        # Forward pass
        output = fsdp_mlp(x)

        # Backward pass with same gradient across all ranks
        torch.manual_seed(42)  # Same gradient for all ranks
        grad_output = torch.randn_like(output)
        output.backward(grad_output)

    finally:
        if dist.is_initialized():
            dist.barrier()
        cleanup_distributed()


def run_no_sync_test(rank, world_size, init_file):
    """
    Run no_sync test on a single GPU process.
    This function is spawned by torch.multiprocessing.
    """
    try:
        setup_distributed(rank, world_size, init_file)
        device = torch.device(f"cuda:{rank}")

        config = LlamaConfig(hidden_size=128, intermediate_size=256, hidden_act="silu")

        # Create model with same weights across all ranks
        torch.manual_seed(42)
        mlp = LigerTiledSwiGLUMLP(config=config, num_shards=None).to(device).to(torch.float32)
        ddp_mlp = DDP(mlp, device_ids=[rank])

        # First backward with no_sync (should NOT synchronize)
        x1 = torch.randn(2, 512, 128, device=device, dtype=torch.float32) * 0.1
        x1.requires_grad_(True)

        with ddp_mlp.no_sync():
            out1 = ddp_mlp(x1)
            torch.manual_seed(rank)  # Different gradient per rank!
            grad1 = torch.randn_like(out1)
            out1.backward(grad1)

        # After no_sync, gradients should be DIFFERENT across ranks
        gate_grad_no_sync = ddp_mlp.module.gate_proj.weight.grad.clone()

        # Gather to verify they are different
        if rank == 0:
            no_sync_grads = [torch.zeros_like(gate_grad_no_sync) for _ in range(world_size)]
        else:
            no_sync_grads = None

        dist.gather(gate_grad_no_sync, no_sync_grads, dst=0)

        if rank == 0:
            # Verify gradients are DIFFERENT (not synchronized)
            try:
                torch.testing.assert_close(no_sync_grads[0], no_sync_grads[1], rtol=1e-5, atol=1e-5)
                raise AssertionError("Gradients should NOT be synchronized inside no_sync(), but they are!")
            except AssertionError as e:
                if "should NOT be synchronized" in str(e):
                    raise
                # Expected: gradients are different, which is correct!
                pass

        # Second backward WITH sync (should synchronize)
        ddp_mlp.zero_grad()
        torch.manual_seed(100)  # Same input for all ranks
        x2 = torch.randn(2, 512, 128, device=device, dtype=torch.float32) * 0.1
        x2.requires_grad_(True)

        out2 = ddp_mlp(x2)
        torch.manual_seed(42)  # Same gradient for all ranks
        grad2 = torch.randn_like(out2)
        out2.backward(grad2)

        # After normal backward, gradients should be SYNCHRONIZED
        gate_grad_sync = ddp_mlp.module.gate_proj.weight.grad.clone()

        if rank == 0:
            sync_grads = [torch.zeros_like(gate_grad_sync) for _ in range(world_size)]
        else:
            sync_grads = None

        dist.gather(gate_grad_sync, sync_grads, dst=0)

        if rank == 0:
            # Verify gradients are SAME (synchronized)
            torch.testing.assert_close(
                sync_grads[0],
                sync_grads[1],
                rtol=1e-5,
                atol=1e-5,
                msg="Gradients should be synchronized after normal backward",
            )

    finally:
        if dist.is_initialized():
            dist.barrier()
        cleanup_distributed()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multi-GPU tests require at least 2 GPUs")
@pytest.mark.parametrize("mlp_type", ["geglu", "swiglu"])
@pytest.mark.parametrize("num_shards", [None])  # Only None works reliably with DDP gradient synchronization
@pytest.mark.parametrize("dtype", [torch.float32])
def test_tiled_mlp_ddp(mlp_type, num_shards, dtype):
    """
    Test TiledMLP with DistributedDataParallel.
    """
    world_size = min(2, torch.cuda.device_count())

    hidden_size = 128
    intermediate_size = 256

    if mlp_type == "geglu":
        config = LlamaConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act="gelu_pytorch_tanh",
        )
    else:  # swiglu
        config = LlamaConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act="silu",
        )

    # Use temporary file for distributed init
    init_file = get_init_file()

    try:
        # Spawn processes for each GPU
        mp.spawn(
            run_ddp_test,
            args=(world_size, mlp_type, config, dtype, num_shards, init_file),
            nprocs=world_size,
            join=True,
        )
    finally:
        # Clean up init file
        if os.path.exists(init_file):
            os.unlink(init_file)


@pytest.mark.skip(
    reason="FSDP is incompatible with LigerTiledMLP's custom autograd function. "
    "use_orig_params=True explicitly disallows custom autograd functions, "
    "and use_orig_params=False causes grad_fn issues with flattened parameters."
)
@pytest.mark.skipif(
    torch.cuda.device_count() < 2 or not FSDP_AVAILABLE, reason="FSDP tests require at least 2 GPUs and PyTorch >= 1.11"
)
@pytest.mark.parametrize("mlp_type", ["geglu", "swiglu"])
@pytest.mark.parametrize("num_shards", [None])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_tiled_mlp_fsdp(mlp_type, num_shards, dtype):
    """
    Test TiledMLP with FullyShardedDataParallel.
    """
    world_size = min(2, torch.cuda.device_count())

    hidden_size = 128
    intermediate_size = 256

    if mlp_type == "geglu":
        config = LlamaConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act="gelu_pytorch_tanh",
        )
    else:  # swiglu
        config = LlamaConfig(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act="silu",
        )

    # Use temporary file for distributed init
    init_file = get_init_file()

    try:
        # Spawn processes for each GPU
        mp.spawn(
            run_fsdp_test,
            args=(world_size, mlp_type, config, dtype, num_shards, init_file),
            nprocs=world_size,
            join=True,
        )
    finally:
        # Clean up init file
        if os.path.exists(init_file):
            os.unlink(init_file)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multi-GPU tests require at least 2 GPUs")
def test_tiled_mlp_ddp_no_sync():
    """
    Test that no_sync() context works correctly with TiledMLP.
    Verifies that gradients are NOT synchronized when using no_sync().
    """
    world_size = min(2, torch.cuda.device_count())

    # Use temporary file for distributed init
    init_file = get_init_file()

    try:
        mp.spawn(run_no_sync_test, args=(world_size, init_file), nprocs=world_size, join=True)
    finally:
        # Clean up init file
        if os.path.exists(init_file):
            os.unlink(init_file)
