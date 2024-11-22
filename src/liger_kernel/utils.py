import torch


def infer_device():
    """
    Get current device name based on available devices
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"
