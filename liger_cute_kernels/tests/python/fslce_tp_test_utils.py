import time

import torch.distributed as dist


def strided_tp_group(world_size: int, stride: int):
    if stride == 0:
        return dist.group.WORLD
    if stride < 1 or world_size % stride != 0:
        raise ValueError(f"tp-group-stride must divide world size {world_size}, got {stride}")

    rank = dist.get_rank()
    selected = None
    for offset in range(stride):
        ranks = list(range(offset, world_size, stride))
        group = dist.new_group(ranks)
        if rank in ranks:
            selected = group
    assert selected is not None
    return selected


def stagger_tp_group(global_rank: int, stride: int, delay_ms: int) -> None:
    if stride and delay_ms:
        time.sleep((global_rank % stride) * delay_ms / 1000.0)
