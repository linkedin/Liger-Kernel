"""Mode 2 — hand-assembled spec + GPTModel subclass using Liger directly.

Adapted from Megatron's ``examples/run_simple_mcore_train_loop.py``. The
relevant additions (vs. that file) are:

  1. Direct imports of ``LigerMegatronRMSNorm`` and ``LigerMegatronCrossEntropy``
     (no monkey-patch).

  2. ``model_provider()`` assembles a ``TransformerBlockSubmodules`` by hand,
     placing ``LigerMegatronRMSNorm`` into every norm slot:
       - per-layer ``input_layernorm`` and ``pre_mlp_layernorm``
       - the block-level ``layer_norm`` field that backs
         ``decoder.final_layernorm``
     This is the slot-level integration path — verbose but maximally explicit.
     It is the only way to control ``final_layernorm`` from user code without
     monkey-patching.

  3. ``_LigerCEGPTModel(GPTModel)`` overrides
     ``LanguageModule.compute_language_model_loss`` to route the loss through
     a ``LigerMegatronCrossEntropy`` instance. Cross-entropy has no spec slot
     in Megatron, so subclassing is the symmetric "hand-built" path.

  4. ``_print_norm_classes`` + ``_print_ce_class`` after model construction
     — print the resolved class for every norm slot AND the resolved CE
     class on the model so you can verify Liger took over.

Run with:
    torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 \\
        examples/megatron/run_mode2_hand_spec.py
"""

from __future__ import annotations

import os

from functools import partial
from pathlib import Path
from typing import Iterator

import torch

from megatron.core import dist_checkpointing
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset
from megatron.core.datasets.utils import compile_helpers
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.tensor_parallel.layers import RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.tokenizers import MegatronTokenizer
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from torch.optim import Adam
from torch.utils.data import DataLoader

# --- Liger integration: Mode 2 ---------------------------------------------
from liger_kernel.megatron import LigerMegatronCrossEntropy
from liger_kernel.megatron import LigerMegatronRMSNorm

# ---------------------------------------------------------------------------

_SEQUENCE_LENGTH = 64
_NUM_ITERS = 5
_NUM_LAYERS = 2
_LABEL_SMOOTHING = 0.1


class _LigerCEGPTModel(GPTModel):
    """``GPTModel`` subclass that routes its loss through ``LigerMegatronCrossEntropy``.

    Megatron's CE is not a spec slot — ``LanguageModule.compute_language_model_loss``
    calls ``fused_vocab_parallel_cross_entropy`` directly. The symmetric "hand-built"
    integration is therefore to subclass ``GPTModel`` and override that method.
    """

    def __init__(self, *args, liger_ce_label_smoothing: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.liger_ce = LigerMegatronCrossEntropy(
            ignore_index=-100,
            label_smoothing=liger_ce_label_smoothing,
            reduction="none",
        )

    def compute_language_model_loss(self, labels, logits):
        # LanguageModule contract: input labels are [b, s], output loss is [b, s].
        # LigerMegatronCrossEntropy matches the fused signature, which expects [s, b].
        labels_sb = labels.transpose(0, 1).contiguous()  # [s, b]
        loss_sb = self.liger_ce(logits, labels_sb, self.pg_collection.tp)  # [s, b]
        return loss_sb.transpose(0, 1).contiguous()  # [b, s]


def initialize_distributed(tp: int = 2, pp: int = 1) -> None:
    parallel_state.destroy_model_parallel()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    parallel_state.initialize_model_parallel(tp, pp)


def model_provider() -> GPTModel:
    cfg = TransformerConfig(
        num_layers=_NUM_LAYERS,
        hidden_size=12,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        normalization="RMSNorm",
    )

    # ↓↓ Mode 2 — explicit slot-level placement of LigerMegatronRMSNorm ↓↓
    per_layer = ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=LigerMegatronRMSNorm,
            self_attention=ModuleSpec(
                module=SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=LigerMegatronRMSNorm,
            mlp=partial(
                MLP.as_mlp_submodule,
                submodules=MLPSubmodules(
                    linear_fc1=ColumnParallelLinear,
                    linear_fc2=RowParallelLinear,
                    activation_func=None,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            sharded_state_dict_keys_map={
                "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
                "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
            },
        ),
    )
    block_spec = TransformerBlockSubmodules(
        layer_specs=[per_layer] * _NUM_LAYERS,
        layer_norm=LigerMegatronRMSNorm,  # ← block-level final_layernorm
    )
    # ↑↑ ----------------------------------------------------------------- ↑↑

    return _LigerCEGPTModel(
        config=cfg,
        transformer_layer_spec=block_spec,
        vocab_size=128,
        max_sequence_length=_SEQUENCE_LENGTH,
        liger_ce_label_smoothing=_LABEL_SMOOTHING,
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
    datasets = BlendedMegatronDatasetBuilder(MockGPTDataset, [1000, None, None], lambda: True, cfg).build()
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


def _print_ce_class(model: torch.nn.Module) -> None:
    """Show that ``compute_language_model_loss`` will route through Liger."""
    ce = getattr(model, "liger_ce", None)
    print("=== Resolved CE class ===")
    if ce is None:
        print("  model.liger_ce  → (not set; subclass missing)")
    else:
        print(f"  model.liger_ce            →  {type(ce).__module__}.{type(ce).__name__}")
        print(f"  ce.label_smoothing        →  {ce.label_smoothing}")
        print(f"  ce.ignore_index           →  {ce.ignore_index}")
    print()


def main() -> None:
    # TP=1, DP=2 — CE patch (TP=1 only). Norms are correct under any TP value, so
    # demonstrating both Liger features in one script means running data-parallel.
    initialize_distributed(tp=1, pp=1)
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)

    gpt_model = model_provider().cuda()

    if torch.distributed.get_rank() == 0:
        print("\n=== Full model tree (mode 2: hand-built spec) ===")
        print(gpt_model)
        _print_norm_classes(gpt_model)
        _print_ce_class(gpt_model)

    ddp_cfg = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,
        use_distributed_optimizer=False,
    )
    model = DistributedDataParallel(config=gpt_model.config, ddp_config=ddp_cfg, module=gpt_model)

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
            print(f"[mode2] iter {it}  loss={loss_val:.6f}")

    ckpt_path = Path(os.getcwd()) / "ckpt_mode2"
    ckpt_path.mkdir(exist_ok=True)
    underlying = model.module if hasattr(model, "module") else model
    dist_checkpointing.save(
        sharded_state_dict=underlying.sharded_state_dict(prefix=""),
        checkpoint_dir=str(ckpt_path),
    )
    sd = underlying.sharded_state_dict(prefix="")
    underlying.load_state_dict(dist_checkpointing.load(sharded_state_dict=sd, checkpoint_dir=str(ckpt_path)))
    if torch.distributed.get_rank() == 0:
        print("Successfully loaded the model")


if __name__ == "__main__":
    main()
