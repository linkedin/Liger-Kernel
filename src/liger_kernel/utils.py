import torch


def infer_device():
    """
    Get current device name based on available devices
    """
    if torch.cuda.is_available():  # Works for both Nvidia and AMD
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"


def transformers_version_dispatch(
    required_version: str,
    before_fn,
    after_fn,
    before_args: tuple = (),
    after_args: tuple = (),
    before_kwargs: dict = None,
    after_kwargs: dict = None,
):
    """
    Dispatches to different functions based on package version comparison.

    Args:
        required_version: Version to compare against (e.g. "4.48.0")
        before_fn: Function to call if package_version < required_version
        after_fn: Function to call if package_version >= required_version
        before_args: Positional arguments for before_fn
        after_args: Positional arguments for after_fn
        before_kwargs: Keyword arguments for before_fn
        after_kwargs: Keyword arguments for after_fn

    Returns:
        Result from either before_fn or after_fn

    Example:
        >>> rotary_emb = transformers_version_dispatch(
        ...     "4.48.0",
        ...     LlamaRotaryEmbedding,
        ...     LlamaRotaryEmbedding,
        ...     before_args=(head_dim,),
        ...     after_args=(LlamaConfig(head_dim=head_dim),),
        ...     before_kwargs={'device': device},
        ...     after_kwargs={'device': device}
        ... )
    """
    from packaging import version
    from transformers import __version__ as transformers_version

    before_kwargs = before_kwargs or {}
    after_kwargs = after_kwargs or {}

    if version.parse(transformers_version) < version.parse(required_version):
        return before_fn(*before_args, **before_kwargs)
    else:
        return after_fn(*after_args, **after_kwargs)
