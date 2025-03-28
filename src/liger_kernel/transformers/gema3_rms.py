from .rms_norm import LigerRMSNorm


class LigerRMSNormForGemma3(LigerRMSNorm):
    """Gemma3RMSNorm has a dim argument not hidden_size used in q_norm and k_norm."""

    def __init__(self, dim, eps=0.000001, offset=1.0, casting_mode="gemma", init_fn="zeros", in_place=False):
        super().__init__(dim, eps, offset, casting_mode, init_fn, in_place)
