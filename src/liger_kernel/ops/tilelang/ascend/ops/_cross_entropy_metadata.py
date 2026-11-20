"""Fused target validation, int32 conversion and valid-row statistics for Ascend CE."""

import tilelang
import tilelang.ascend.language as T
import torch


@tilelang.jit
def target_metadata_kernel(V, words, cores=64, tile=1024):

    @T.prim_func
    def main(
        TargetWords_handle: T.handle,
        Labels_handle: T.handle,
        Meta_handle: T.handle,
        ignore_index: T.int32,
        N: T.int64,
    ):
        TargetWords = T.match_buffer(TargetWords_handle, (N * words,), "int32")
        Labels = T.match_buffer(Labels_handle, (N,), "int32")
        Meta = T.match_buffer(Meta_handle, (cores, 8), "int32")
        with T.Kernel(cores) as core:
            raw = T.alloc_shared((tile * words,), "int32")
            labels = T.alloc_shared((tile,), "int32")
            totals = T.alloc_shared((8,), "int32")
            with T.SimtVF(threads=128):
                for i in T.Parallel(8):
                    totals[i] = 0
            for step in T.serial(T.ceildiv(N, cores * tile)):
                begin = (step * cores + core) * tile
                if begin < N:
                    valid = T.cast(T.min(tile, N - begin), "int32")
                    T.copy(TargetWords[begin * words : (begin + valid) * words], raw[: valid * words])
                    with T.SimtVF(threads=128):
                        live = T.alloc_fragment((tile,), "int32")
                        bad = T.alloc_fragment((tile,), "int32")
                        count = T.alloc_fragment((1,), "int32")
                        invalid = T.alloc_fragment((1,), "int32")
                        for i in T.Parallel(tile):
                            if i < valid:
                                low = raw[i * words]
                                high = raw[i * words + 1] if words == 2 else T.if_then_else(low < 0, -1, 0)
                                ignored = (low == ignore_index) & (high == T.if_then_else(ignore_index < 0, -1, 0))
                                in_range = (high == 0) & (low >= 0) & (low < V)
                                live[i] = T.if_then_else(ignored, 0, 1)
                                bad[i] = T.if_then_else(ignored | in_range, 0, 1)
                                labels[i] = low
                            else:
                                live[i] = 0
                                bad[i] = 0
                        T.reduce_sum(live, count, dim=0)
                        T.reduce_sum(bad, invalid, dim=0)
                        for i in T.Parallel(1):
                            totals[0] = totals[0] + count[0]
                            totals[1] = totals[1] + invalid[0]
                    T.copy(labels[:valid], Labels[begin : begin + valid])
            T.copy(totals, Meta[core, :])

    return main


def prepare_targets(target, classes, ignore_index):
    """Validate full-width integer labels before exposing their int32 kernel view.

    Ascend vector arithmetic uses int32 labels. Inspect both words of an int64
    target so values such as 2**32 cannot wrap to a valid class. Counts remain
    integer through the cross-core reduction; only one small D2H copy is used.
    """
    target = target.contiguous()
    labels = torch.empty(target.numel(), device=target.device, dtype=torch.int32)
    cores = torch.npu.get_device_properties(target.device).vector_core_num
    partial = torch.empty((cores, 8), device=target.device, dtype=torch.int32)
    with torch.npu.device(target.device):
        target_metadata_kernel(classes, target.element_size() // 4, cores)(
            target.view(torch.int32), labels, partial, ignore_index, target.numel()
        )
    totals = partial.sum(dim=0).cpu().tolist()
    if totals[1]:
        raise ValueError("target is out of bounds")
    return labels, int(totals[0])
