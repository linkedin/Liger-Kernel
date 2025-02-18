try:
    from liger_kernel.transformers.trainer.orpo_trainer import LigerORPOTrainer  # noqa: F401
except ImportError:
    raise ImportError("Please `pip install trl` to use LigerORPOTrainer")
