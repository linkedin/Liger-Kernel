import os
import time

from dataclasses import dataclass

import torch
import transformers

from accelerate.utils.constants import FSDP_SHARDING_STRATEGY
from transformers import TrainerControl
from transformers import TrainerState
from transformers import TrainingArguments

from liger_kernel.utils import infer_device

# https://simple.wikipedia.org/wiki/Byte
# For memory, we use binary system
M_BIN_UNIT = 2**20
# For metrics (tflops), we use decimal system
T_DEC_UNIT = 10**12


def round_to_n_decimal(x, n):
    return round(x, n)


@dataclass
class Precision:
    """
    Precision is a dataclass to store the number of decimal points for each metric.
    """

    n_decimal_time: int
    n_decimal_memory: int
    n_decimal_TPS: int
    n_decimal_MFU: int


@dataclass
class State:
    """
    State is a dataclass to store the internal state of the efficiency callback.
    """

    n_warmup_steps: int = 0
    total_peak_memory_allocated: float = float("-inf")
    total_peak_memory_reserved: float = float("-inf")

    step_start_time: float = 0.0
    elapsed_time: float = 0.0

    elapsed_step: int = 0

    step_start_tokens_seen: int = 0
    elapsed_tokens_seen: int = 0

    step_start_flos: float = 0.0
    elapsed_flos: float = 0.0

    global_start_step: int = 0


@dataclass
class Time:
    """
    Time is a dataclass to store the time-related metrics.
    """

    step: int = 0
    step_time_sec: float = 0.0
    avg_step_time_sec: float = 0.0
    time_to_completion_sec: float = 0.0
    estimated_total_time_sec: float = 0.0


@dataclass
class Memory:
    """
    Memory is a dataclass to store the memory-related metrics.
    """

    step_peak_memory_allocated_MB: float = 0.0
    total_peak_memory_allocated_MB: float = 0.0


@dataclass
class TPS:
    """
    TPS is a dataclass to store the tokens per second metrics.
    """

    step_tokens_per_second: float = 0.0
    avg_tokens_per_second: float = 0.0


@dataclass
class MFU:
    """
    MFU is a dataclass to store the MFU metrics.
    """

    step_MFU: float = 0.0
    avg_MFU: float = 0.0


class EfficiencyCallback(transformers.TrainerCallback):
    """
    EfficiencyCallback is a callback to track the efficiency of the training process.
    The tracked stats include: step time, memory, throughput, and MFU.

    It requires including `--include_num_input_tokens_seen` and `logging_steps=1` in the training arguments.

    Args:
        n_warmup_steps: number of warmup steps
            The stats in the first n_warmup_steps will not be added into the aggregated stats
            This is because the first few steps might take longer due to jit compliation and other initialization overheads
        n_decimal_time: number of decimal points for time
        n_decimal_memory: number of decimal points for memory
        n_decimal_TPS: number of decimal points for TPS
        n_decimal_MFU: number of decimal points for MFU in percentage
    """

    def __init__(
        self,
        n_warmup_steps=2,
        n_decimal_time=2,
        n_decimal_memory=2,
        n_decimal_TPS=2,
        n_decimal_MFU=4,
    ):
        self.state = State(
            n_warmup_steps,
        )

        self.precision = Precision(
            n_decimal_time,
            n_decimal_memory,
            n_decimal_TPS,
            n_decimal_MFU,
        )

        self.time = Time()
        self.memory = Memory()
        self.tps = TPS()
        self.mfu = MFU()
        self.device = infer_device()

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of the initialization of the [`Trainer`].
        """
        if not args.include_num_input_tokens_seen:
            raise Exception(
                'Please pass training argument "--include_num_input_tokens_seen" to track tokens per second'
            )
        if args.logging_steps != 1:
            raise Exception("Please set logging_steps=1 to track the efficiency metrics accurately")

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # if loaded from checkpoints, global_start_step is not 1 but state.global_step
        self.state.global_start_step = state.global_step

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float],
        **kwargs,
    ):
        if state.global_step < (self.state.global_start_step + self.state.n_warmup_steps):
            return
        else:
            # spread self.time, self.memory, self.tps, self.mfu to logs
            # logs.update(self.time.__dict__)
            logs.update(self.memory.__dict__)
            logs.update(self.tps.__dict__)
            # logs.update(self.mfu.__dict__)

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the beginning of a training step. If using gradient accumulation, one training step might take
        several inputs.
        """
        # memory
        getattr(torch, self.device).reset_peak_memory_stats()

        # time
        self.state.step_start_time = time.perf_counter()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step < (self.state.global_start_step + self.state.n_warmup_steps):
            # The end the current step_start_tokens_seen and step_start_flos are the start of next iteration

            # tokens
            self.state.step_start_tokens_seen = state.num_input_tokens_seen
            # flos
            self.state.step_start_flos = state.total_flos
            return

        # time
        current_time = time.perf_counter()
        step_time = current_time - self.state.step_start_time
        self.state.elapsed_time += step_time

        # step
        global_step = state.global_step
        self.state.elapsed_step += 1
        avg_step_time = self.state.elapsed_time / self.state.elapsed_step

        self.time.step = global_step
        self.time.step_time_sec = round_to_n_decimal(step_time, self.precision.n_decimal_time)
        self.time.avg_step_time_sec = round_to_n_decimal(avg_step_time, self.precision.n_decimal_time)
        self.time.time_to_completion_sec = round_to_n_decimal(
            avg_step_time * (state.max_steps - global_step),
            self.precision.n_decimal_time,
        )
        self.time.estimated_total_time_sec = round_to_n_decimal(
            avg_step_time * state.max_steps, self.precision.n_decimal_time
        )

        # memory
        step_peak_memory_allocated = getattr(torch, self.device).memory.max_memory_allocated()
        step_peak_memory_reserved = getattr(torch, self.device).memory.max_memory_reserved()

        self.memory.step_peak_memory_allocated_MB = round_to_n_decimal(
            step_peak_memory_allocated / M_BIN_UNIT, self.precision.n_decimal_memory
        )
        self.state.total_peak_memory_allocated = max(self.state.total_peak_memory_allocated, step_peak_memory_allocated)
        self.memory.total_peak_memory_allocated_MB = round_to_n_decimal(
            self.state.total_peak_memory_allocated / M_BIN_UNIT,
            self.precision.n_decimal_memory,
        )

        self.memory.step_peak_memory_reserved_MB = round_to_n_decimal(
            step_peak_memory_reserved / M_BIN_UNIT, self.precision.n_decimal_memory
        )

        self.state.total_peak_memory_reserved = max(self.state.total_peak_memory_reserved, step_peak_memory_reserved)

        self.memory.total_peak_memory_reserved_MB = round_to_n_decimal(
            self.state.total_peak_memory_reserved / M_BIN_UNIT,
            self.precision.n_decimal_memory,
        )

        # tokens
        step_tokens_seen = state.num_input_tokens_seen - self.state.step_start_tokens_seen

        self.state.elapsed_tokens_seen += step_tokens_seen

        self.tps.step_tokens_per_second = round_to_n_decimal(
            step_tokens_seen / step_time,
            self.precision.n_decimal_TPS,
        )

        self.tps.avg_tokens_per_second = round_to_n_decimal(
            self.state.elapsed_tokens_seen / self.state.elapsed_time,
            self.precision.n_decimal_TPS,
        )

        # flos
        step_flos = state.total_flos - self.state.step_start_flos
        self.state.elapsed_flos += step_flos

        # MFU
        # 1. Definition
        #
        # MFU is defined as (achieved TPS) / (theoretical maximum TPS) = (achieved floating point operations per sec) / (theoretical maximum floating point operations per sec)
        # Crucially, the "theoretical maximum" throughput only accounts for the required operations to compute the forward+backward passes, and not rematerialization. MFU therefore allows fair comparisons
        # between training runs on different systems, as the numerator is simply the observed tokens-per-second, and the denominator is only dependent on the model architecture and published maximum FLOPs for a given system.
        # Ref: https://arxiv.org/pdf/2204.02311
        # The benefit of MFU is that it
        #
        # 2. Implementation in huggingface
        #
        # current_flos = 6 * estimate_tokens(input_dict) * num_parameters()
        # total_flos = sum(current_flos) # across all GPUs
        # Ref: https://github.com/huggingface/transformers/blob/616bb11d487aabc231bb230b245c42214ea4b254/src/transformers/modeling_utils.py#L1196
        #
        # 3. Derive MFU on rank 0
        #
        # rank_0_flos = tatal_flos / n_gpus = measured_flos / effecitve_n_gpus
        # rank_0_MFU = rank_0_flos / step_time
        #
        # For FSDP, num_parameters() is (1 / n_gpus) of the total parameters. So, the effective_n_gpus = 1
        # For HSDP, num_parameters() is (1 / local_world_size) of the total parameters. So, the effective_n_gpus = n_nodes
        # For no sharding and zero-2, num_parameters() is the total parameters. So, the effective_n_gpus = n_gpus

        num_gpus = EfficiencyCallback._get_effective_num_gpus()
        step_achieved_tflops = step_flos / step_time / num_gpus / T_DEC_UNIT

        avg_achieved_tflops = self.state.elapsed_flos / self.state.elapsed_time / num_gpus / T_DEC_UNIT

        precision_bits = 16 if args.bf16 or args.fp16 else 32
        gpu_peak_tflops = EfficiencyCallback._get_gpu_peak_tflops(precision_bits)

        self.mfu.step_MFU = round_to_n_decimal(step_achieved_tflops / gpu_peak_tflops, self.precision.n_decimal_MFU)

        self.mfu.avg_MFU = round_to_n_decimal(avg_achieved_tflops / gpu_peak_tflops, self.precision.n_decimal_MFU)

        # The end the current step_start_tokens_seen and step_start_flos are the start of next iteration

        # tokens
        self.state.step_start_tokens_seen = state.num_input_tokens_seen
        # flos
        self.state.step_start_flos = state.total_flos

    @staticmethod
    def _get_effective_num_gpus():
        # Calculate the number of effective GPUs for the total FLOPs in order to calculate the single GPU FLOP
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if transformers.utils.strtobool(os.environ.get("ACCELERATE_USE_FSDP", "false")):
            sharding_strategy = os.environ.get("FSDP_SHARDING_STRATEGY", FSDP_SHARDING_STRATEGY[0]).upper()

            # Either specified as string or enum number
            if sharding_strategy in {
                "FULL_SHARD",
                str(FSDP_SHARDING_STRATEGY.index("FULL_SHARD") + 1),
            }:
                return 1

            elif sharding_strategy in {
                "HYBRID_SHARD",
                str(FSDP_SHARDING_STRATEGY.index("HYBRID_SHARD") + 1),
            }:
                return world_size // int(os.environ.get("LOCAL_WORLD_SIZE", 1))
            else:
                return world_size

        assert world_size != 0, (
            "WORLD_SIZE should be set to a positive integer. For single GPU training, please explicitly set WORLD_SIZE=1."
        )

        # TODO: add deepspeed support
        return world_size

    @staticmethod
    def _get_gpu_peak_tflops(precision_bits: int = 16):
        if precision_bits not in {16, 32}:
            raise Exception(f"Precision bits {precision_bits} is not supported")

        device_name = getattr(torch, infer_device()).get_device_name()

        if "A100" in device_name:
            # data from https://www.nvidia.com/en-us/data-center/a100/
            return 312 if precision_bits == 16 else 156
        elif "H100" in device_name:
            # data from https://www.nvidia.com/en-us/data-center/h100/
            # NOTE: Specifications are one-half lower without sparsity.
            if "NVL" in device_name:
                return 1979 if precision_bits == 16 else 989
            elif "PCIe" in device_name:
                return 756 if precision_bits == 16 else 378
            else:  # for SXM and other variants
                return 989 if precision_bits == 16 else 494
        elif "V100" in device_name:
            if "NVL" in device_name:
                return 125
            else:
                return 112
        return None
