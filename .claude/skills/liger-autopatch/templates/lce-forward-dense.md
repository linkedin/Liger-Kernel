# Template: lce_forward for Dense Models

Reference implementation: `src/liger_kernel/transformers/model/llama.py`. Read it first.

## Structure

The file needs `lce_forward` and reuses `lce_maybe_trainable_lm_head` from `llama.py`.

### Imports

```python
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import torch
from liger_kernel.transformers.model.llama import lce_maybe_trainable_lm_head
from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss, unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import LigerCausalLMOutputWithPast
if TYPE_CHECKING:
    from transformers.cache_utils import Cache
```

### lce_forward body pattern

1. Set defaults from config (output_attentions, output_hidden_states, return_dict)
2. Call `self.model(...)` with all forwarded params
3. Extract hidden states: `outputs[0]` or `outputs.last_hidden_state`
4. Slice for `logits_to_keep`
5. Extract `shift_labels = kwargs.pop("shift_labels", None)`
6. `skip_logits` logic — default True during training with labels
7. Compute loss via `lce_maybe_trainable_lm_head` (skip_logits) or `self.lm_head` (else)
8. Return `LigerCausalLMOutputWithPast`

### Variations

- **Logit softcapping**: Pass `final_logit_softcapping=self.config.final_logit_softcapping` to `LigerForCausalLMLoss`. Requires a local helper instead of reusing `lce_maybe_trainable_lm_head`.
- **Hidden state access**: Use `outputs.last_hidden_state` if HF model uses that
- **pretraining_tp**: Remove check if model doesn't have this config field

### When NOT to reuse llama's lce_maybe_trainable_lm_head

If the model passes extra kwargs to `LigerForCausalLMLoss` (like softcapping), write a local `_liger_for_causal_lm_loss` and inline the PEFT/FSDP handling from llama.py.
