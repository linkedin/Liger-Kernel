from types import SimpleNamespace

import pytest
import torch

from liger_kernel.utils import infer_device

# LigerORPOTrainer subclasses trl.trainer.ORPOTrainer, so trl must be importable.
pytest.importorskip("trl")

from transformers import LlamaConfig  # noqa: E402
from transformers import LlamaForCausalLM  # noqa: E402

from liger_kernel.transformers.trainer import LigerORPOTrainer  # noqa: E402

device = infer_device()


def test_concatenated_forward_without_fsdp():
    """Regression test for #1229: concatenated_forward must not assume the model is FSDP-wrapped.

    Previously the lm_head forward always went through _FSDPForwardRedirection, which asserts the
    module is FSDP and therefore crashed on a single, non-FSDP device. Here we run it on a plain
    LlamaForCausalLM and only expect a finite loss.
    """
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    model = LlamaForCausalLM(config).to(device)

    batch_size, seq_len = 2, 8
    input_ids = torch.randint(0, config.vocab_size, (2 * batch_size, seq_len), device=device)
    concatenated_batch = {
        "concatenated_input_ids": input_ids,
        "concatenated_attention_mask": torch.ones_like(input_ids),
        "concatenated_labels": input_ids.clone(),
    }

    # Build the trainer without running ORPOTrainer.__init__ and stub concatenated_inputs, so the
    # test exercises only the forward path that used to be FSDP-only, free of dataset plumbing.
    trainer = LigerORPOTrainer.__new__(LigerORPOTrainer)
    trainer.is_encoder_decoder = False
    trainer.aux_loss_enabled = False
    trainer.label_pad_token_id = -100
    trainer.padding_value = 0
    trainer.beta = 0.1
    trainer.accelerator = SimpleNamespace(device=device)
    trainer.concatenated_inputs = lambda *args, **kwargs: concatenated_batch

    loss, aux_outputs = trainer.concatenated_forward(model, batch={})

    assert loss.isfinite().all()
    assert len(aux_outputs) > 0
