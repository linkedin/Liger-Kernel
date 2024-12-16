try:
    from liger_kernel.transformers.trainer.orpo_trainer import (  # noqa: F401
        LigerORPOTrainer,
    )
except ImportError:
    raise ImportError("Please `pip install trl` to use LigerORPOTrainer")
