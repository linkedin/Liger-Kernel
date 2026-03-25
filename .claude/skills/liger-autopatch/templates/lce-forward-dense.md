# Template: lce_forward for Dense Models

Use this as the basis for generating `src/liger_kernel/transformers/model/{model_type}.py` for a standard dense (non-MoE, non-VL) model.

## Reference

The canonical implementation is `src/liger_kernel/transformers/model/llama.py`. Read it before generating.

## Structure

The file needs these imports and two functions: `lce_forward` and (for most models) a reuse of `lce_maybe_trainable_lm_head` from `llama.py`.

### Imports

```python
from typing import TYPE_CHECKING
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch

from liger_kernel.transformers.model.llama import lce_maybe_trainable_lm_head
from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import LigerCausalLMOutputWithPast

if TYPE_CHECKING:
    from transformers.cache_utils import Cache
```

### lce_forward function

The function signature must exactly match the HF model's `ForCausalLM.forward`, including all model-specific parameters. The body follows this pattern:

1. **Set defaults from config** — output_attentions, output_hidden_states, return_dict
2. **Call base model** — `outputs = self.model(...)`
3. **Extract hidden states** — `hidden_states = outputs[0]` (or `outputs.last_hidden_state`)
4. **Slice for logits_to_keep** — `kept_hidden_states = hidden_states[:, slice_indices, :]`
5. **Handle shift_labels** — `shift_labels = kwargs.pop("shift_labels", None)`
6. **skip_logits logic** — default to True during training with labels
7. **Compute loss** — use `lce_maybe_trainable_lm_head` when skip_logits, else `self.lm_head`
8. **Return** — `LigerCausalLMOutputWithPast` or tuple

### Variations from Llama

Adapt these based on the model profile:

- **Logit softcapping**: Add `final_logit_softcapping=self.config.final_logit_softcapping` to the `LigerForCausalLMLoss` call
- **Hidden state access**: Use `outputs.last_hidden_state` if the model uses that pattern
- **Extra forward params**: Add any model-specific params to the `self.model(...)` call
- **pretraining_tp**: Remove the check if the model doesn't have this config field
- **No lce_maybe_trainable_lm_head reuse**: If the model needs a custom loss path (e.g., different softcapping logic), write a local `_liger_for_causal_lm_loss` function instead

### When NOT to reuse llama's lce_maybe_trainable_lm_head

If the model passes extra kwargs to `LigerForCausalLMLoss` (like `final_logit_softcapping`), you need a local helper:

```python
def _liger_for_causal_lm_loss(lm_head, hidden_states, hidden_size, labels, shift_labels, softcap, **loss_kwargs):
    return LigerForCausalLMLoss(
        hidden_states=hidden_states,
        lm_head_weight=lm_head.weight,
        labels=labels,
        hidden_size=hidden_size,
        shift_labels=shift_labels,
        final_logit_softcapping=softcap,
        **loss_kwargs,
    )
```

And inline the PEFT/FSDP handling from llama.py's `lce_maybe_trainable_lm_head`.
