from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Generator, Iterable, List, Sequence, Tuple

import numpy as np
import polars as pl

try:
    from skfolio.model_selection import CombinatorialPurgedCV  # type: ignore
except Exception:  # noqa: BLE001
    CombinatorialPurgedCV = None  # type: ignore


@dataclass
class CPCVSpec:
    n_splits: int
    embargo_days: int
    test_fold_size: int = 2


def cpcv_splits(dates: Sequence[int], spec: CPCVSpec) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    n = len(dates)
    if spec.n_splits <= 1:
        raise ValueError("CPCV requires at least 2 splits")
    if spec.test_fold_size < 1 or spec.test_fold_size >= spec.n_splits:
        raise ValueError("test_fold_size must be between 1 and n_splits-1")

    indices = np.arange(n)

    if CombinatorialPurgedCV is not None:
        cpcv = CombinatorialPurgedCV(n_splits=spec.n_splits, embargo=spec.embargo_days, test_fold_size=spec.test_fold_size)
        for tr, te in cpcv.split(indices):
            yield np.asarray(tr, dtype=np.int64), np.asarray(te, dtype=np.int64)
        return

    # High-fidelity CPCV fallback
    fold_sizes = np.full(spec.n_splits, n // spec.n_splits, dtype=np.int64)
    fold_sizes[: n % spec.n_splits] += 1
    fold_boundaries = np.cumsum(fold_sizes)
    fold_indices: List[np.ndarray] = []
    start = 0
    for end in fold_boundaries:
        fold_indices.append(indices[start:end])
        start = end

    embargo = int(spec.embargo_days)
    for test_combo in combinations(range(spec.n_splits), spec.test_fold_size):
        test_idx = np.concatenate([fold_indices[i] for i in test_combo])
        test_idx.sort()

        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        if embargo > 0:
            offsets = np.arange(-embargo, embargo + 1)
            impacted = (test_idx[:, None] + offsets[None, :]).reshape(-1)
            impacted = impacted[(impacted >= 0) & (impacted < n)]
            train_mask[impacted] = False

        train_idx = indices[train_mask]
        if train_idx.size == 0:
            continue
        yield train_idx, test_idx

