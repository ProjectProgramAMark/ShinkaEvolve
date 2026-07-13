"""Trusted r4 adaptive-eligibility and dual-finalist selection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class AdaptiveEligibility:
    eligible: bool
    nonclone_actions_at_five_percent: tuple[int, ...]
    pre_probability: tuple[float, ...]
    post_probability: tuple[float, ...]
    total_variation: float
    total_variation_interval: tuple[float, float]


def _probability(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (6,) or not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError(f"{label} must be one finite nonnegative six-vector")
    total = float(np.sum(array))
    if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        raise ValueError(f"{label} must sum to one")
    return array / total


def _aggregate(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    pre_weights = np.asarray([row.get("pre_count", 1.0) for row in rows], dtype=float)
    post_weights = np.asarray([row.get("post_count", 1.0) for row in rows], dtype=float)
    if (
        not np.all(np.isfinite(pre_weights))
        or not np.all(np.isfinite(post_weights))
        or np.any(pre_weights <= 0)
        or np.any(post_weights <= 0)
    ):
        raise ValueError("adaptive probability counts must be finite and positive")
    pre = np.average(
        [_probability(row["pre_probability"], "pre") for row in rows],
        axis=0,
        weights=pre_weights,
    )
    post = np.average(
        [_probability(row["post_probability"], "post") for row in rows],
        axis=0,
        weights=post_weights,
    )
    return pre, post


def classify_adaptive(
    observations: Iterable[dict[str, Any]],
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 20260713,
) -> AdaptiveEligibility:
    """Classify behavior without adding it to the optimization score.

    Each observation is one coherent founder/repeat summary with a founder ID
    and expected pre/post action-probability vectors. Realized action counts are
    intentionally not accepted.
    """
    rows = list(observations)
    if not rows:
        raise ValueError("adaptive classification requires observations")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    founders: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        founder = row.get("founder_id")
        if not isinstance(founder, str) or not founder:
            raise ValueError("adaptive observation lacks founder_id")
        founders.setdefault(founder, []).append(row)
    founder_ids = sorted(founders)
    if len(founder_ids) < 2:
        raise ValueError("adaptive classification requires multiple founders")

    pre, post = _aggregate(rows)
    pre_total = sum(float(row.get("pre_count", 1.0)) for row in rows)
    post_total = sum(float(row.get("post_count", 1.0)) for row in rows)
    mean_probability = (pre_total * pre + post_total * post) / (pre_total + post_total)
    exercised = tuple(index for index in range(1, 6) if mean_probability[index] >= 0.05)
    total_variation = float(0.5 * np.sum(np.abs(post - pre)))

    generator = np.random.default_rng(bootstrap_seed)
    draws = np.empty(bootstrap_replicates, dtype=float)
    for index in range(bootstrap_replicates):
        selected = generator.choice(founder_ids, size=len(founder_ids), replace=True)
        sample: list[dict[str, Any]] = []
        for founder in selected:
            founder_rows = founders[str(founder)]
            repeat_indices = generator.integers(0, len(founder_rows), len(founder_rows))
            sample.extend(founder_rows[item] for item in repeat_indices)
        sampled_pre, sampled_post = _aggregate(sample)
        draws[index] = 0.5 * np.sum(np.abs(sampled_post - sampled_pre))
    interval = tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))
    eligible = len(exercised) >= 2 and total_variation >= 0.20 and interval[0] > 0.0
    return AdaptiveEligibility(
        eligible=eligible,
        nonclone_actions_at_five_percent=exercised,
        pre_probability=tuple(float(value) for value in pre),
        post_probability=tuple(float(value) for value in post),
        total_variation=total_variation,
        total_variation_interval=interval,
    )


def select_finalists(
    ranked_records: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Always select the unrestricted champion and, when present, an adaptive one."""
    valid = [
        record for record in ranked_records if record.get("development_integrity_valid")
    ]
    if not valid:
        raise ValueError("no valid development finalist")
    valid.sort(
        key=lambda item: (-float(item["development_score"]), int(item["generation"]))
    )
    result = {"unrestricted": valid[0]}
    adaptive = [
        record
        for record in valid
        if record.get("adaptive_eligibility", {}).get("eligible") is True
    ]
    if adaptive:
        result["adaptive"] = adaptive[0]
    return result
