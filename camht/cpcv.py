from __future__ import annotations

from dataclasses import dataclass
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


def cpcv_splits(dates: Sequence[int], spec: CPCVSpec) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    if CombinatorialPurgedCV is None:
        # Fallback: simple time series split with embargo
        n = len(dates)
        fold = n // spec.n_splits
        for i in range(spec.n_splits):
            test_idx = np.arange(i * fold, min(n, (i + 1) * fold))
            train_idx = np.concatenate([np.arange(0, max(0, i * fold - spec.embargo_days)), np.arange(min(n, (i + 1) * fold + spec.embargo_days), n)])
            yield train_idx, test_idx
        return

    # Proper CPCV
    # skfolio expects timestamps; but we have ordinal date_id; use direct indices with embargo
    n = len(dates)
    indices = np.arange(n)
    cpcv = CombinatorialPurgedCV(n_splits=spec.n_splits, embargo=spec.embargo_days)
    for tr, te in cpcv.split(indices):
        yield np.array(tr), np.array(te)

