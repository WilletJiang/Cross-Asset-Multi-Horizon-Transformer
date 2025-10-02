from __future__ import annotations

import functools
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from rich.console import Console

try:
    from collections.abc import Mapping, Sequence
except ImportError:  # Python <3.9 fallback
    from typing import Mapping, Sequence  # type: ignore

from omegaconf import OmegaConf


console = Console()


def _parse_harmonic_range(input_value: Any) -> Tuple[float, float]:
    if isinstance(input_value, Sequence) and not isinstance(input_value, (str, bytes)):
        seq = list(input_value)
        if len(seq) != 2:
            raise ValueError("Harmonic freq_range must be (min_period, max_period)")
        return float(seq[0]), float(seq[1])
    if isinstance(input_value, Mapping) and {"min", "max"}.issubset(input_value):
        return float(input_value["min"]), float(input_value["max"])
    if isinstance(input_value, str):
        cleaned = input_value.replace("period", "")
        for token in ("=", "min", "max"):
            cleaned = cleaned.replace(token, "")
        delim = ":" if ":" in cleaned else ","
        parts = [p.strip() for p in cleaned.split(delim) if p.strip()]
        if len(parts) == 2:
            return float(parts[0]), float(parts[1])
    raise TypeError("Unsupported harmonic freq_range format")


def _bool_from_cfg(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no"}
    return bool(value)


def resolve_time2vec_kwargs(model_cfg: Any) -> Dict[str, Any]:
    if not isinstance(model_cfg, Mapping):
        try:
            model_cfg = OmegaConf.to_container(model_cfg, resolve=True)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            model_cfg = dict(model_cfg)

    activation = model_cfg.get("time2vec_activation", "sin")
    include_linear = _bool_from_cfg(model_cfg.get("time2vec_include_linear", True))
    freq_init = str(model_cfg.get("time2vec_freq_init", "random") or "random")
    kwargs: Dict[str, Any] = {
        "periodic_activation": activation,
        "include_linear": include_linear,
        "freq_init": freq_init,
    }

    freq_range_cfg = model_cfg.get("time2vec_freq_range")
    if freq_init == "harmonic":
        if freq_range_cfg is None:
            console.print("[yellow][ConfigWarning][Time2Vec] freq_init=harmonic but no freq_range provided; fallback to random")
            kwargs["freq_init"] = "random"
        else:
            try:
                kwargs["freq_range"] = _parse_harmonic_range(freq_range_cfg)
            except Exception as err:  # noqa: BLE001
                console.print(f"[yellow][ConfigWarning][Time2Vec] freq_range parsing failed ({err}); fallback to random")
                kwargs["freq_init"] = "random"

    return kwargs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class SDPPolicy:
    force_math: bool = False
    enable_flash: bool = True
    enable_mem_efficient: bool = True


def configure_sdp(policy: Optional[SDPPolicy] = None) -> None:
    """Configure scaled_dot_product_attention kernel preferences.

    Keep shapes static and use torch.compile for best performance.
    """
    if policy is None:
        policy = SDPPolicy()
    torch.backends.cuda.enable_flash_sdp(policy.enable_flash)
    torch.backends.cuda.enable_mem_efficient_sdp(policy.enable_mem_efficient)
    torch.backends.cuda.enable_math_sdp(policy.force_math)


def maybe_compile(
    model: torch.nn.Module,
    enabled: bool,
    *,
    mode: str = "max-autotune",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> torch.nn.Module:
    """Compile model with extreme performance settings.

    Args:
        model: Model to compile
        enabled: Whether to compile
        mode: Compilation mode (default, reduce-overhead, max-autotune, max-autotune-no-cudagraphs)
        fullgraph: Force full graph compilation (breaks on graph breaks)
        dynamic: Enable dynamic shapes (slower but more flexible)
    """
    if not enabled:
        return model

    try:
        # Inductor config for extreme performance
        import torch._dynamo
        torch._dynamo.config.suppress_errors = not fullgraph

        compile_kwargs = {
            "mode": mode,
            "fullgraph": fullgraph,
            "dynamic": dynamic,
        }

        model = torch.compile(model, **compile_kwargs)
        console.print(f"[green]torch.compile enabled: mode={mode}, fullgraph={fullgraph}, dynamic={dynamic}")
    except Exception as err:  # noqa: BLE001
        console.print(f"[yellow]torch.compile failed: {err}. Continuing without compile.")

    return model


def env_flag(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "y"}


# ============================================================================
# Advanced Compilation Utilities | 高级编译工具
# ============================================================================


def configure_inductor_aggressive() -> None:
    """Configure Inductor for maximum performance.

    Sets aggressive optimization flags for torch.compile.
    """
    try:
        import torch._inductor.config as inductor_config

        # Maximum autotuning
        inductor_config.max_autotune = True
        inductor_config.max_autotune_gemm = True
        inductor_config.coordinate_descent_tuning = True

        # Freezing for better optimization
        inductor_config.freezing = True

        # Memory optimizations
        inductor_config.memory_planning = True

        # CUDA graph when possible
        inductor_config.triton.cudagraphs = True

        # Kernel fusion
        inductor_config.aggressive_fusion = True

        # Cache compiled kernels
        inductor_config.fx_graph_cache = True

        console.print("[green]Inductor configured for extreme performance")
    except Exception as e:
        console.print(f"[yellow]Failed to configure Inductor: {e}")


def configure_dynamo_aggressive() -> None:
    """Configure TorchDynamo for maximum performance.

    极致性能的TorchDynamo配置。
    """
    try:
        import torch._dynamo.config as dynamo_config

        # Disable dynamic shapes for better optimization
        dynamo_config.dynamic_shapes = False
        dynamo_config.assume_static_by_default = True

        # Guard optimization
        dynamo_config.guard_nn_modules = True

        # Cache
        dynamo_config.cache_size_limit = 256

        # Suppress non-critical errors
        dynamo_config.suppress_errors = True

        console.print("[green]TorchDynamo configured for extreme performance")
    except Exception as e:
        console.print(f"[yellow]Failed to configure TorchDynamo: {e}")


def setup_extreme_performance_mode() -> None:
    """Setup all extreme performance configurations.

    一键配置极致性能模式。
    """
    configure_inductor_aggressive()
    configure_dynamo_aggressive()

    # CUDA optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    console.print("[green]✓ Extreme performance mode activated!")


def explain_graph_breaks(model: torch.nn.Module, *args, **kwargs) -> None:
    """Analyze and explain graph breaks in a model.

    Useful for debugging torch.compile issues.
    分析并解释模型中的图断点。

    Args:
        model: Model to analyze
        *args: Example inputs
        **kwargs: Example keyword inputs

    Example:
        >>> explain_graph_breaks(model, dummy_input)
    """
    try:
        import torch._dynamo as dynamo

        explanation = dynamo.explain(model)(*args, **kwargs)

        console.print("[cyan]Graph Break Analysis:")
        console.print(f"Graph Count: {explanation.graph_count}")
        console.print(f"Graph Break Count: {explanation.graph_break_count}")

        if explanation.graph_break_count > 0:
            console.print("[yellow]Graph breaks detected:")
            for i, reason in enumerate(explanation.break_reasons, 1):
                console.print(f"  {i}. {reason}")
        else:
            console.print("[green]No graph breaks! Model is fully fusible.")

        console.print(f"\nCompile Time: {explanation.compile_time:.2f}s")
    except Exception as e:
        console.print(f"[yellow]Failed to explain graph breaks: {e}")


def compare_compiled_vs_eager(
    model: torch.nn.Module,
    *args,
    mode: str = "max-autotune",
    runs: int = 100,
    **kwargs,
) -> dict[str, Any]:
    """Compare compiled vs eager mode performance.

    比较编译模式与eager模式的性能。

    Args:
        model: Model to compare
        *args: Example inputs
        mode: Compilation mode
        runs: Number of profiling runs
        **kwargs: Example keyword inputs

    Returns:
        Dictionary with comparison results
    """
    import time
    import copy

    # Eager mode
    model_eager = copy.deepcopy(model)
    model_eager.eval()

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = model_eager(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Profile eager
    start = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            _ = model_eager(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    eager_time = time.perf_counter() - start

    # Compiled mode
    model_compiled = torch.compile(model, mode=mode)
    model_compiled.eval()

    # Warmup (includes compilation)
    compile_start = time.perf_counter()
    for _ in range(3):
        with torch.no_grad():
            _ = model_compiled(*args, **kwargs)
    compile_time = time.perf_counter() - compile_start

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Profile compiled
    start = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            _ = model_compiled(*args, **kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    compiled_time = time.perf_counter() - start

    # Results
    speedup = eager_time / compiled_time

    results = {
        "eager_time": eager_time,
        "compiled_time": compiled_time,
        "compile_overhead": compile_time,
        "speedup": speedup,
        "eager_throughput": runs / eager_time,
        "compiled_throughput": runs / compiled_time,
    }

    console.print("\n[cyan]Compilation Comparison:")
    console.print(f"  Eager Time: {eager_time:.3f}s")
    console.print(f"  Compiled Time: {compiled_time:.3f}s")
    console.print(f"  Compilation Overhead: {compile_time:.3f}s")
    console.print(f"  Speedup: {speedup:.2f}x")
    console.print(f"  Break-even: {compile_time/max(eager_time-compiled_time, 0.001):.1f} runs")

    return results


def _maybe_log_config_warning(name: str, reason: str) -> None:
    from camht.distributed import is_main_process

    if is_main_process():
        console.print(f"[yellow][ConfigWarning][{name}] {reason}")

