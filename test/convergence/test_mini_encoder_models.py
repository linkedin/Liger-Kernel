from test.utils import (
    DEFAULT_DATASET_PATH,
    MiniModelConfig,
    assert_verbose_allclose,
    set_seed,
    simple_collate_fn
)

import pytest
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers.models.bert import BertConfig, BertForSequenceClassification

from liger_kernel.transformers import apply_liger_kernel_to_bert

MINI_BERT_CONFIG = MiniModelConfig(
    liger_kernel_patch_func=apply_liger_kernel_to_bert,
    model_class=BertForSequenceClassification,
    mini_model_config=BertConfig(
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=2,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        num_labels=2,
    ),
)

def create_model(with_liger=False):
    if with_liger:
        apply_liger_kernel_to_bert(embedding=True, cross_entropy=True)
    return MINI_BERT_CONFIG.model_class(MINI_BERT_CONFIG.mini_model_config).to("cuda")

def run_mini_bert(num_steps=10, with_liger=False):
    set_seed(42)
    model = create_model(with_liger)
    train_dataset = load_from_disk(DEFAULT_DATASET_PATH)
    loader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=simple_collate_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    losses = []
    for _ in range(num_steps):
        batch = next(iter(loader))
        batch = {k: v.to("cuda") for k, v in batch.items()}
        batch['labels'] = (batch['labels'][:, 0] % 2).long()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    return {"loss": losses, "model": model}

@pytest.mark.parametrize(
    "num_steps, loss_atol, loss_rtol, param_atol, param_rtol",
    [(10, 1e-6, 1e-5, 5e-3, 1e-5)],
)
def test_mini_bert(num_steps, loss_atol, loss_rtol, param_atol, param_rtol):
    expected_output = run_mini_bert(num_steps)
    actual_output = run_mini_bert(num_steps, with_liger=True)

    assert_verbose_allclose(
        expected_output["loss"],
        actual_output["loss"],
        atol=loss_atol,
        rtol=loss_rtol,
        msg="Loss mismatch",
    )

    for (name, p1), (_, p2) in zip(
        expected_output["model"].named_parameters(),
        actual_output["model"].named_parameters(),
    ):
        assert_verbose_allclose(
            p1, p2, atol=param_atol, rtol=param_rtol, msg=f"Parameter mismatch: {name}"
        )
