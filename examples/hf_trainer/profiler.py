import os
import socket
import time

import torch
import transformers

# Define ANSI color codes as constants
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def trace_handler(
    dir_name: str,
    enable_trace: bool = True,
    enable_memory_timeline: bool = True,
):

    # make dir_name a absolute path
    dir_name = os.path.abspath(dir_name)

    def handler_fn(prof) -> None:

        os.makedirs(dir_name, exist_ok=True)

        file_prefix = f"{socket.gethostname()}_{os.getpid()}_{time.time_ns()}"

        if enable_trace is True:
            # Export chrome trace
            trace_name = f"{file_prefix}.pt.trace.json"
            trace_path = os.path.join(dir_name, trace_name)
            prof.export_chrome_trace(trace_path)
            print(f"{GREEN}Profiler data saved to {trace_path}{RESET}")
            print(
                f"{YELLOW}Please `pip install torch-tb-profiler` and run `tensorboard --logdir {dir_name}` to view the trace{RESET}"
            )

        if enable_memory_timeline is True:
            # Export memory timeline
            memory_timeline_name = f"{file_prefix}.html"
            memory_timeline_path = os.path.join(dir_name, memory_timeline_name)
            prof.export_memory_timeline(memory_timeline_path)
            print(f"{GREEN}Memory timeline data saved to {memory_timeline_path}{RESET}")
            print(
                f"{YELLOW}Please download {memory_timeline_path} and open with browser to view the memory timeline.{RESET}"
            )

    return handler_fn


class ProfilerCallback(transformers.TrainerCallback):
    """
    ProfilerCallback uses PyTorch Profiler to profile the training process. It skips the first `warmup` steps and profiles the next `active` steps.
    It is recommended to only enable ProfilerCallback on rank 0 process for distributed training.
    For example,

    .. code-block:: python

        if not torch.distributed.distributed_c10d.is_initialized() or torch.distributed.get_rank() == 0:
            callbacks.append(ProfilerCallback())


    If enable_trace is True, it will export the trace data to `output_dir`.
    If enable_memory_timeline is True, it will export the memory timeline data to `output_dir`.
    At least one of enable_trace and enable_memory_timeline should be True.

    Args:
        warmup: Number of steps to warmup the profiler
        active: Number of steps to profile
        enable_trace: Enable trace output
        enable_memory_timeline: Enable memory timeline output
        output_dir: Output directory for profiler data
    """

    def __init__(
        self,
        warmup: int = 2,
        active: int = 2,
        enable_trace: bool = True,
        enable_memory_timeline: bool = True,
        output_dir: str = "./profiler-output",
    ):

        assert (
            enable_trace or enable_memory_timeline
        ), "At least one of enable_trace and enable_memory_timeline should be True."

        self.prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=0, warmup=warmup, active=active, repeat=1, skip_first=0
            ),
            on_trace_ready=trace_handler(
                output_dir, enable_trace, enable_memory_timeline
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self.prof.start()

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()
