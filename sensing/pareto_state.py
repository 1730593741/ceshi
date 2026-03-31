"""Population-to-state sensing for the closed-loop controller.

The optimizer operates on full populations, but the controller should react to a
compact and stable summary. This module defines that summary as ``ParetoState``
and computes a set of metrics that characterize improvement, feasibility,
diversity, and front geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from optimizers.nsga2.population import Individual
from optimizers.nsga2.selection import dominates
from sensing.feasibility_metrics import feasible_ratio, mean_constraint_violation
from sensing.hypervolume import HypervolumeCalculator, SimplifiedHypervolumeCalculator


@dataclass(slots=True)
class ParetoState:
    """Structured snapshot of the optimizer's search state.

    Besides the basic MVP metrics such as hypervolume and feasible ratio, the
    project also tracks a few richer geometric indicators that are useful for
    later experiments and ablations.
    """

    generation: int
    hv: float
    delta_hv: float
    feasible_ratio: float
    rank1_ratio: float
    mean_cv: float
    diversity_score: float
    crowding_entropy: float
    d_dec: float
    d_front: float
    stagnation_len: int
    rank1_objectives: list[tuple[float, ...]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state into a dictionary suitable for JSON logging."""
        return asdict(self)


class ParetoStateSensor:
    """Build ``ParetoState`` objects from population snapshots."""

    def __init__(self, hv_calculator: HypervolumeCalculator | None = None) -> None:
        self._hv_calculator = hv_calculator or SimplifiedHypervolumeCalculator()

    def sense(
        self,
        *,
        generation: int,
        population: list[Individual],
        previous_state: ParetoState | None = None,
        reference_point: tuple[float, ...] | None = None,
        stagnation_tolerance: float = 1e-12,
    ) -> ParetoState:
        """Generate a structured state from the current population.

        ``previous_state`` is optional but recommended, because it enables the
        sensor to compute both ``delta_hv`` and the stagnation counter.
        """
        if generation < 0:
            raise ValueError("generation must be >= 0")

        violations = [ind.constraint_violation for ind in population]
        objectives = [ind.objectives for ind in population if ind.objectives]

        current_feasible_ratio = feasible_ratio(violations)
        current_mean_cv = mean_constraint_violation(violations)
        # Single non-dominated scan: reuse rank1 list for ratio + downstream metrics.
        rank1_individuals = _rank1_individuals(population)
        current_rank1_ratio = len(rank1_individuals) / len(population) if population else 0.0
        current_diversity = _compute_diversity_score(objectives)
        current_crowding_entropy = _compute_crowding_entropy(rank1_individuals)
        current_d_dec = _compute_decision_diversity(population)
        current_d_front = _compute_front_separation(population, rank1_individuals)

        hv_reference = reference_point or _default_reference_point(objectives)
        current_hv = self._hv_calculator.compute(objectives, hv_reference) if objectives else 0.0

        prev_hv = previous_state.hv if previous_state else 0.0
        delta_hv = current_hv - prev_hv

        if previous_state is None:
            stagnation_len = 0
        elif delta_hv <= stagnation_tolerance:
            stagnation_len = previous_state.stagnation_len + 1
        else:
            stagnation_len = 0

        return ParetoState(
            generation=generation,
            hv=current_hv,
            delta_hv=delta_hv,
            feasible_ratio=current_feasible_ratio,
            rank1_ratio=current_rank1_ratio,
            mean_cv=current_mean_cv,
            diversity_score=current_diversity,
            crowding_entropy=current_crowding_entropy,
            d_dec=current_d_dec,
            d_front=current_d_front,
            stagnation_len=stagnation_len,
            rank1_objectives=[tuple(ind.objectives) for ind in rank1_individuals if ind.objectives],
        )



def _compute_rank1_ratio(population: list[Individual]) -> float:
    """Compute the fraction of individuals that lie on the first front."""
    if not population:
        return 0.0

    rank1_size = sum(1 for candidate in population if _is_nondominated(candidate, population))
    return rank1_size / len(population)



def _rank1_individuals(population: list[Individual]) -> list[Individual]:
    """Return all nondominated individuals without mutating stored ranks."""
    return [candidate for candidate in population if _is_nondominated(candidate, population)]



def _is_nondominated(candidate: Individual, population: list[Individual]) -> bool:
    """Return ``True`` when no other individual dominates ``candidate``."""
    for other in population:
        if other is candidate:
            continue
        if dominates(other, candidate):
            return False
    return True



def _compute_diversity_score(objectives: list[tuple[float, ...]]) -> float:
    """Estimate objective-space diversity via mean distance to the centroid.

    This metric is intentionally simple and cheap to compute every generation,
    making it suitable for the control loop's online sensing step.
    """
    if len(objectives) <= 1:
        return 0.0

    matrix = np.asarray(objectives, dtype=float)
    centroid = matrix.mean(axis=0)
    distances = np.linalg.norm(matrix - centroid, axis=1)
    return float(np.mean(distances))



def _compute_crowding_entropy(rank1_individuals: list[Individual], eps: float = 1e-12) -> float:
    """Measure how evenly spread rank-1 points are in objective space.

    Steps:
    1. For each rank-1 point, compute the nearest-neighbor distance.
    2. Normalize those distances into a probability mass function.
    3. Return normalized Shannon entropy in ``[0, 1]``.

    Fallback behavior:
    - fewer than three rank-1 points -> ``0.0``;
    - near-zero aggregate spacing -> ``0.0``.
    """
    if len(rank1_individuals) < 3:
        return 0.0

    front = np.asarray([ind.objectives for ind in rank1_individuals], dtype=float)
    if front.ndim != 2 or front.size == 0:
        return 0.0

    distance_matrix = np.linalg.norm(front[:, None, :] - front[None, :, :], axis=2)
    np.fill_diagonal(distance_matrix, np.inf)
    nearest = distance_matrix.min(axis=1)
    nearest = np.nan_to_num(nearest, nan=0.0, posinf=0.0, neginf=0.0)

    total = float(np.sum(nearest))
    if total <= eps:
        return 0.0

    probs = np.clip(nearest / total, eps, 1.0)
    entropy = float(-np.sum(probs * np.log(probs)))
    normalizer = float(np.log(len(probs)))
    if normalizer <= eps:
        return 0.0

    return float(np.clip(entropy / normalizer, 0.0, 1.0))



def _compute_decision_diversity(population: list[Individual]) -> float:
    """Measure decision-space diversity with normalized Hamming distance.

    Assignment-style genomes are discrete vectors, so mismatch ratio is more
    interpretable here than Euclidean distance.
    """
    if len(population) <= 1:
        return 0.0

    distances: list[float] = []
    for i, lhs in enumerate(population):
        for rhs in population[i + 1 :]:
            distances.append(_normalized_hamming(lhs.genome, rhs.genome))

    if not distances:
        return 0.0
    return float(np.mean(np.asarray(distances, dtype=float)))



def _normalized_hamming(lhs: list[int], rhs: list[int]) -> float:
    """Return normalized Hamming distance with a length-mismatch penalty."""
    max_len = max(len(lhs), len(rhs))
    if max_len == 0:
        return 0.0

    overlap = min(len(lhs), len(rhs))
    mismatches = sum(1 for idx in range(overlap) if lhs[idx] != rhs[idx])
    mismatches += abs(len(lhs) - len(rhs))
    return mismatches / max_len



def _compute_front_separation(
    population: list[Individual],
    rank1_individuals: list[Individual],
    eps: float = 1e-12,
) -> float:
    """Quantify how distinctly separated rank-1 points are from dominated ones.

    This is a multi-objective analogue of a front-separation ratio:

    - ``inter``: mean nearest distance from dominated points to the rank-1 set;
    - ``intra``: mean pairwise distance within the rank-1 set;
    - ``d_front = inter / (inter + intra + eps)`` in ``[0, 1]``.

    Interpretation:
    - high values mean the front is compact and clearly separated;
    - low values mean dominated and rank-1 points are intermixed.
    """
    if not rank1_individuals:
        return 0.0

    # Derive dominated set from pre-computed rank1 to avoid a third O(n²) scan.
    rank1_ids = {id(ind) for ind in rank1_individuals}
    dominated = [ind for ind in population if id(ind) not in rank1_ids]
    if not dominated:
        return 1.0

    front = np.asarray([ind.objectives for ind in rank1_individuals], dtype=float)
    dominated_objs = np.asarray([ind.objectives for ind in dominated], dtype=float)

    pairwise_fd = np.linalg.norm(dominated_objs[:, None, :] - front[None, :, :], axis=2)
    inter = float(np.mean(np.min(pairwise_fd, axis=1)))

    if len(rank1_individuals) <= 1:
        intra = 0.0
    else:
        pairwise_ff = np.linalg.norm(front[:, None, :] - front[None, :, :], axis=2)
        upper = pairwise_ff[np.triu_indices(len(rank1_individuals), k=1)]
        intra = float(np.mean(upper)) if upper.size > 0 else 0.0

    score = inter / (inter + intra + eps)
    return float(np.clip(np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))



def _default_reference_point(objectives: list[tuple[float, ...]]) -> tuple[float, ...]:
    """Construct a conservative default hypervolume reference point."""
    if not objectives:
        return (1.0, 1.0)

    matrix = np.asarray(objectives, dtype=float)
    maxima = matrix.max(axis=0)
    padding = np.maximum(np.abs(maxima) * 0.1, 1e-6)
    return tuple((maxima + padding).tolist())
