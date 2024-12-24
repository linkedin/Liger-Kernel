import time

from dataclasses import dataclass

import torch
import transformers

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
    step_peak_memory_reserved_MB: float = 0.0
    total_peak_memory_allocated_MB: float = 0.0
    total_peak_memory_reserved_MB: float = 0.0


@dataclass
class TPS:
    """
    TPS is a dataclass to store the tokens per second metrics.
    """

    step_tokens_per_second: float = 0.0
    avg_tokens_per_second: float = 0.0


class EfficiencyCallback(transformers.TrainerCallback):
    """
    EfficiencyCallback is a callback to track the efficiency of the training process.
    The tracked stats include: step time, memory, and throughput.

    It requires including `--include_num_input_tokens_seen` and `logging_steps=1` in the training arguments.

    Args:
        n_warmup_steps: number of warmup steps
            The stats in the first n_warmup_steps will not be added into the aggregated stats
            This is because the first few steps might take longer due to jit compliation and other initialization overheads
        n_decimal_time: number of decimal points for time
        n_decimal_memory: number of decimal points for memory
        n_decimal_TPS: number of decimal points for TPS
    """

    def __init__(self, n_warmup_steps=2, n_decimal_time=2, n_decimal_memory=2, n_decimal_TPS=2):
        self.state = State(
            n_warmup_steps,
        )

        self.precision = Precision(n_decimal_time, n_decimal_memory, n_decimal_TPS)

        self.time = Time()
        self.memory = Memory()
        self.tps = TPS()
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
            # spread self.time, self.memory, self.tps to logs
            logs.update(self.time.__dict__)
            logs.update(self.memory.__dict__)
            logs.update(self.tps.__dict__)

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
            # The end the current step_start_tokens_seen is the start of next iteration

            # tokens
            self.state.step_start_tokens_seen = state.num_input_tokens_seen
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

        # The end the current step_start_tokens_seen is the start of next iteration

        # tokens
        self.state.step_start_tokens_seen = state.num_input_tokens_seen
