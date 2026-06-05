"""Mode 1 — monkey-patch Megatron-Core to use Liger RMSNorm.

Adapted from Megatron's ``examples/run_simple_mcore_train_loop.py``. The
relevant additions (vs. that file) are:

  1. ``apply_liger_kernel_to_megatron(rms_norm=True)`` called once at the
     top of ``model_provider()``. This patches both
     ``LocalSpecProvider.layer_norm`` (per-layer norm slots) and
     ``transformer_block.LayerNormImpl`` (block-level ``final_layernorm``).

  2. ``normalization="RMSNorm"`` added to ``TransformerConfig`` so the
     model actually has RMSNorm slots to patch (Megatron defaults to
     ``LayerNorm``).

  3. ``_print_norm_classes`` after model construction — prints the
     resolved class for every norm slot so you can verify Liger took
     over.

Run with:
    torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 \\
        examples/megatron/run_mode1_monkey_patch.py
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterator, Tuple

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from megatron.core import dist_checkpointing, parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset
from megatron.core.datasets.utils import compile_helpers
from megatron.core.distributed import (
    DistributedDataParallel,
    DistributedDataParallelConfig,
)
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.transformer.transformer_config import TransformerConfig

# --- Liger integration: Mode 1 ---------------------------------------------
from liger_kernel.megatron import apply_liger_kernel_to_megatron

# ---------------------------------------------------------------------------

_SEQUENCE_LENGTH = 64
_NUM_ITERS = 5


def initialize_distributed(tp: int = 2, pp: int = 1) -> None:
    parallel_state.destroy_model_parallel()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )
    parallel_state.initialize_model_parallel(tp, pp)


def model_provider() -> GPTModel:
    # ↓↓ Mode 1 — patch once, everything below picks up Liger ↓↓
    apply_liger_kernel_to_megatron(rms_norm=True)
    # ↑↑ ------------------------------------------------------ ↑↑

    cfg = TransformerConfig(
        num_layers=2,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        normalization="RMSNorm",
    )
    return GPTModel(
        config=cfg,
        transformer_layer_spec=get_gpt_layer_local_spec(normalization="RMSNorm"),
        vocab_size=100,
        max_sequence_length=_SEQUENCE_LENGTH,
    )


def get_train_data_iterator() -> Iterator:
    if torch.distributed.get_rank() == 0:
        compile_helpers()
    torch.distributed.barrier()
    cfg = GPTDatasetConfig(
        random_seed=0,
        sequence_length=_SEQUENCE_LENGTH,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        tokenizer=MegatronTokenizer.from_pretrained(
            metadata_path={"library": "null-text"},
            vocab_size=_SEQUENCE_LENGTH,
        ),
        mid_level_dataset_surplus=0.005,
    )
    datasets = BlendedMegatronDatasetBuilder(
        MockGPTDataset, [1000, None, None], lambda: True, cfg
    ).build()
    return iter(DataLoader(datasets[0], batch_size=8, shuffle=True))


def forward_step_func(data_iterator, model):
    def loss_func(loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        return loss, {"lm loss": loss}

    data = next(data_iterator)
    tokens = data["tokens"].cuda()
    attn = data["attention_mask"].cuda()
    pos = data["position_ids"].cuda()
    labels = data["labels"].cuda()
    loss_mask = data["loss_mask"].cuda()
    out = model(tokens, pos, attn, labels=labels)
    return out, partial(loss_func, loss_mask)


def _print_norm_classes(model: torch.nn.Module) -> None:
    print("\n=== Resolved norm classes ===")
    target = (
        "LayerNorm",
        "RMSNorm",
        "FusedLayerNorm",
        "WrappedTorchNorm",
        "LigerMegatronRMSNorm",
    )
    for name, mod in model.named_modules():
        if type(mod).__name__ in target and "layernorm" in name.lower():
            cls = type(mod)
            print(f"  {name:50s}  {cls.__module__}.{cls.__name__}")
    print()


def main() -> None:
    initialize_distributed(tp=2, pp=1)
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)

    gpt_model = model_provider().cuda()

    if torch.distributed.get_rank() == 0:
        print("\n=== Full model tree (mode 1: monkey-patch) ===")
        print(gpt_model)
        _print_norm_classes(gpt_model)

    ddp_cfg = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
    )
    model = DistributedDataParallel(
        config=gpt_model.config, ddp_config=ddp_cfg, module=gpt_model
    )

    optim = Adam(model.parameters())
    data_iter = get_train_data_iterator()
    fwd_bwd = get_forward_backward_func()

    for it in range(_NUM_ITERS):
        optim.zero_grad()
        out = fwd_bwd(
            forward_step_func=forward_step_func,
            data_iterator=data_iter,
            model=model,
            num_microbatches=1,
            seq_length=_SEQUENCE_LENGTH,
            micro_batch_size=8,
            decoder_seq_length=_SEQUENCE_LENGTH,
            forward_only=False,
        )
        finalize_model_grads([model])
        optim.step()
        loss_val = float(out[0]["lm loss"].detach())
        if torch.distributed.get_rank() == 0:
            print(f"[mode1] iter {it}  loss={loss_val:.6f}")

    ckpt_path = Path(os.getcwd()) / "ckpt_mode1"
    ckpt_path.mkdir(exist_ok=True)
    underlying = model.module if hasattr(model, "module") else model
    dist_checkpointing.save(
        sharded_state_dict=underlying.sharded_state_dict(prefix=""),
        checkpoint_dir=str(ckpt_path),
    )
    sd = underlying.sharded_state_dict(prefix="")
    underlying.load_state_dict(
        dist_checkpointing.load(sharded_state_dict=sd, checkpoint_dir=str(ckpt_path))
    )
    if torch.distributed.get_rank() == 0:
        print("Successfully loaded the model")


if __name__ == "__main__":
    main()
