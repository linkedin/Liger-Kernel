# Pixtral vision encoder does not require a custom forward function.
# The Liger kernel optimizations for Pixtral (RMSNorm, SwiGLU, RoPE) are applied
# via class/function-level monkey patching in monkey_patch.py, which is sufficient
# since the vision encoder has no cross-entropy loss to fuse.
