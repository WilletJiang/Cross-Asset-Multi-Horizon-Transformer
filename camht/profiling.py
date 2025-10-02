"""Performance profiling and monitoring utilities.

Throughput tracking, CUDA events, memory profiling for extreme performance optimization.
性能分析和监控工具：吞吐量跟踪、CUDA事件、内存分析
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class PerfMetrics:
    """Performance metrics for a training/inference run."""

    # Timing
    total_time: float = 0.0
    data_time: float = 0.0
    forward_time: float = 0.0
    backward_time: float = 0.0
    optimizer_time: float = 0.0

    # Throughput
    samples_processed: int = 0
    batches_processed: int = 0

    # Memory (in MB)
    peak_memory_allocated: float = 0.0
    peak_memory_reserved: float = 0.0

    # Compute
    total_flops: float = 0.0

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_time = 0.0
        self.data_time = 0.0
        self.forward_time = 0.0
        self.backward_time = 0.0
        self.optimizer_time = 0.0
        self.samples_processed = 0
        self.batches_processed = 0
        self.peak_memory_allocated = 0.0
        self.peak_memory_reserved = 0.0
        self.total_flops = 0.0

    def throughput(self) -> float:
        """Samples per second."""
        if self.total_time <= 0:
            return 0.0
        return self.samples_processed / self.total_time

    def compute_utilization(self) -> float:
        """Fraction of time spent on actual compute (forward + backward)."""
        if self.total_time <= 0:
            return 0.0
        return (self.forward_time + self.backward_time) / self.total_time

    def data_loading_overhead(self) -> float:
        """Fraction of time spent on data loading."""
        if self.total_time <= 0:
            return 0.0
        return self.data_time / self.total_time

    def print_summary(self) -> None:
        """Print performance summary."""
        table = Table(title="Performance Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total Time", f"{self.total_time:.2f}s")
        table.add_row("Data Loading Time", f"{self.data_time:.2f}s ({self.data_loading_overhead()*100:.1f}%)")
        table.add_row("Forward Time", f"{self.forward_time:.2f}s")
        table.add_row("Backward Time", f"{self.backward_time:.2f}s")
        table.add_row("Optimizer Time", f"{self.optimizer_time:.2f}s")
        table.add_row("Compute Utilization", f"{self.compute_utilization()*100:.1f}%")
        table.add_row("Samples Processed", f"{self.samples_processed}")
        table.add_row("Batches Processed", f"{self.batches_processed}")
        table.add_row("Throughput", f"{self.throughput():.2f} samples/s")
        table.add_row("Peak Memory Allocated", f"{self.peak_memory_allocated:.2f} MB")
        table.add_row("Peak Memory Reserved", f"{self.peak_memory_reserved:.2f} MB")

        console.print(table)


class CUDATimer:
    """CUDA event-based timer for accurate GPU timing.

    More accurate than CPU timing because it measures actual GPU execution time.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        if self.enabled:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        self.elapsed_time = 0.0

    def __enter__(self):
        if self.enabled:
            self.start_event.record()
        return self

    def __exit__(self, *args):
        if self.enabled:
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed_time = self.start_event.elapsed_time(self.end_event) / 1000.0  # ms to s

    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed_time


class CPUTimer:
    """Simple CPU timer."""

    def __init__(self):
        self.elapsed_time = 0.0
        self.start_time = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_time = time.perf_counter() - self.start_time

    def elapsed(self) -> float:
        """Get elapsed time in seconds."""
        return self.elapsed_time


@contextmanager
def profile_section(name: str, metrics: Optional[PerfMetrics] = None, use_cuda: bool = True):
    """Context manager for profiling a code section.

    Args:
        name: Section name (e.g., "forward", "backward")
        metrics: PerfMetrics object to update
        use_cuda: Use CUDA events for timing (more accurate for GPU operations)

    Example:
        with profile_section("forward", metrics, use_cuda=True):
            output = model(input)
    """
    timer = CUDATimer() if use_cuda else CPUTimer()

    with timer:
        yield

    if metrics is not None:
        if name == "data":
            metrics.data_time += timer.elapsed()
        elif name == "forward":
            metrics.forward_time += timer.elapsed()
        elif name == "backward":
            metrics.backward_time += timer.elapsed()
        elif name == "optimizer":
            metrics.optimizer_time += timer.elapsed()


def update_memory_stats(metrics: PerfMetrics) -> None:
    """Update memory statistics in metrics."""
    if torch.cuda.is_available():
        allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.max_memory_reserved() / 1024**2  # MB
        metrics.peak_memory_allocated = max(metrics.peak_memory_allocated, allocated)
        metrics.peak_memory_reserved = max(metrics.peak_memory_reserved, reserved)


def reset_memory_stats() -> None:
    """Reset CUDA memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def print_memory_summary() -> None:
    """Print current memory usage."""
    if not torch.cuda.is_available():
        return

    allocated = torch.cuda.memory_allocated() / 1024**2
    reserved = torch.cuda.memory_reserved() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    max_reserved = torch.cuda.max_memory_reserved() / 1024**2

    console.print("[cyan]GPU Memory Usage:")
    console.print(f"  Current Allocated: {allocated:.2f} MB")
    console.print(f"  Current Reserved:  {reserved:.2f} MB")
    console.print(f"  Peak Allocated:    {max_allocated:.2f} MB")
    console.print(f"  Peak Reserved:     {max_reserved:.2f} MB")


@contextmanager
def torch_profiler_context(
    output_dir: str,
    activities: Optional[list] = None,
    record_shapes: bool = True,
    profile_memory: bool = True,
    with_stack: bool = False,
):
    """Context manager for PyTorch profiler.

    Args:
        output_dir: Directory to save profiler traces
        activities: List of activities to profile (default: [ProfilerActivity.CPU, ProfilerActivity.CUDA])
        record_shapes: Record tensor shapes
        profile_memory: Profile memory usage
        with_stack: Record stack traces (slow but useful for debugging)

    Example:
        with torch_profiler_context("./profiler_logs"):
            for batch in loader:
                output = model(batch)
                loss.backward()
    """
    from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler

    if activities is None:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=tensorboard_trace_handler(output_dir),
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:
        yield prof


def estimate_model_flops(model: torch.nn.Module, input_shape: tuple, device: str = "cuda") -> float:
    """Estimate model FLOPs using forward pass.

    Note: This is an approximation and may not be exact for all operations.

    Args:
        model: Model to profile
        input_shape: Input tensor shape (batch_size, ...)
        device: Device to run on

    Returns:
        Estimated FLOPs in GFLOPs
    """
    try:
        from torch.utils.flop_counter import FlopCounterMode

        model = model.to(device)
        dummy_input = torch.randn(*input_shape, device=device)

        with FlopCounterMode(model, display=False) as flop_counter:
            _ = model(dummy_input)
            total_flops = flop_counter.get_total_flops()

        return total_flops / 1e9  # Convert to GFLOPs
    except Exception as e:
        console.print(f"[yellow]Failed to estimate FLOPs: {e}")
        return 0.0


def print_gpu_utilization() -> None:
    """Print GPU utilization using nvidia-smi (if available)."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )

        lines = result.stdout.strip().split("\n")
        console.print("[cyan]GPU Utilization:")
        for i, line in enumerate(lines):
            gpu_util, mem_used, mem_total = line.split(", ")
            console.print(f"  GPU {i}: {gpu_util}% utilization, {mem_used}/{mem_total} MB memory")
    except Exception:
        pass  # nvidia-smi not available or failed

