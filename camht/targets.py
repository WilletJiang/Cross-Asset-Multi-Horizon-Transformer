from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import polars as pl


@dataclass
class TargetDef:
    name: str
    lag: int
    expr: str  # e.g., "A - B" or "A"


def parse_target_pairs(csv_path: str) -> Dict[str, TargetDef]:
    df = pl.read_csv(csv_path)
    out: Dict[str, TargetDef] = {}
    for row in df.iter_rows(named=True):
        name = row["target"]
        lag = int(row["lag"])
        expr = str(row["pair"])
        out[name] = TargetDef(name=name, lag=lag, expr=expr)
    return out


def compute_pair_series(frame: pl.DataFrame, expr: str) -> pl.Series:
    """Compute a pair expression over a frame and return a Series.

    Supported: "colA - colB" or single column name.
    """
    if "-" in expr:
        a, b = [s.strip() for s in expr.split("-")]
        return frame[a] - frame[b]
    return frame[expr.strip()]


def compute_target_matrix(frame: pl.DataFrame, date_col: str, pairs: Dict[str, TargetDef]) -> Tuple[pl.DataFrame, List[str]]:
    """Return DataFrame with date_col + all target columns computed from pairs.

    Targets are raw pair series (not returns). Use downstream to compute returns.
    """
    out = pl.DataFrame({date_col: frame[date_col]})
    order: List[str] = []
    for tname, tdef in pairs.items():
        s = compute_pair_series(frame, tdef.expr)
        out = out.with_columns(pl.Series(name=tname, values=s))
        order.append(tname)
    return out, order

