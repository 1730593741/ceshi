"""选择组件 用于 constrained NSGA-II."""

from __future__ import annotations

import math
import random

from optimizers.nsga2.population import Individual



def dominates(lhs: Individual, rhs: Individual) -> bool:
    """返回 约束支配关系 用于 minimization 目标."""
    if lhs.feasible and not rhs.feasible:
        return True
    if not lhs.feasible and rhs.feasible:
        return False

    if not lhs.feasible and not rhs.feasible:
        return lhs.constraint_violation < rhs.constraint_violation

    lhs_no_worse = all(lhs_obj <= rhs_obj for lhs_obj, rhs_obj in zip(lhs.objectives, rhs.objectives))
    lhs_strictly_better = any(lhs_obj < rhs_obj for lhs_obj, rhs_obj in zip(lhs.objectives, rhs.objectives))
    return lhs_no_worse and lhs_strictly_better



def non_dominated_sort(individuals: list[Individual]) -> list[list[Individual]]:
    """将个体分组为 Pareto 前沿 (rank 0, 1, ...).

    Uses integer indices instead of ``id()`` to avoid key collisions when
    the same Individual object is referenced multiple times in the list.
    """
    if not individuals:
        return []

    n = len(individuals)
    domination_count = [0] * n
    dominates_list: list[list[int]] = [[] for _ in range(n)]
    first_front: list[int] = []

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(individuals[i], individuals[j]):
                dominates_list[i].append(j)
                domination_count[j] += 1
            elif dominates(individuals[j], individuals[i]):
                dominates_list[j].append(i)
                domination_count[i] += 1

    for i in range(n):
        if domination_count[i] == 0:
            individuals[i].rank = 0
            first_front.append(i)

    front_indices: list[list[int]] = [first_front]
    level = 0
    while level < len(front_indices) and front_indices[level]:
        next_front: list[int] = []
        for p_idx in front_indices[level]:
            for q_idx in dominates_list[p_idx]:
                domination_count[q_idx] -= 1
                if domination_count[q_idx] == 0:
                    individuals[q_idx].rank = level + 1
                    next_front.append(q_idx)
        if next_front:
            front_indices.append(next_front)
        level += 1

    return [[individuals[i] for i in front] for front in front_indices]



def assign_crowding_distance(front: list[Individual]) -> None:
    """原地为一个 前沿 分配拥挤距离."""
    if not front:
        return

    for ind in front:
        ind.crowding_distance = 0.0

    if len(front) <= 2:
        for ind in front:
            ind.crowding_distance = math.inf
        return

    n_obj = len(front[0].objectives)
    for obj_idx in range(n_obj):
        sorted_front = sorted(front, key=lambda ind: ind.objectives[obj_idx])
        sorted_front[0].crowding_distance = math.inf
        sorted_front[-1].crowding_distance = math.inf

        obj_min = sorted_front[0].objectives[obj_idx]
        obj_max = sorted_front[-1].objectives[obj_idx]
        if obj_max == obj_min:
            continue

        span = obj_max - obj_min
        for idx in range(1, len(sorted_front) - 1):
            if math.isinf(sorted_front[idx].crowding_distance):
                continue
            prev_val = sorted_front[idx - 1].objectives[obj_idx]
            next_val = sorted_front[idx + 1].objectives[obj_idx]
            sorted_front[idx].crowding_distance += (next_val - prev_val) / span



def tournament_pick(a: Individual, b: Individual, rng: random.Random) -> Individual:
    """按 rank、拥挤距离以及随机打破平局来选择一个个体."""
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b
    if a.crowding_distance > b.crowding_distance:
        return a
    if b.crowding_distance > a.crowding_distance:
        return b
    return a if rng.random() < 0.5 else b



def parent_selection(individuals: list[Individual], n_parents: int, rng: random.Random) -> list[Individual]:
    """通过有放回二元锦标赛选择父代."""
    if n_parents <= 0:
        return []
    if len(individuals) < 2:
        raise ValueError("Need at least two individuals for tournament selection")

    selected: list[Individual] = []
    for _ in range(n_parents):
        cand_a = rng.choice(individuals)
        cand_b = rng.choice(individuals)
        selected.append(tournament_pick(cand_a, cand_b, rng))
    return selected



def environmental_selection(individuals: list[Individual], population_size: int) -> list[Individual]:
    """先按 前沿 再按拥挤距离选择下一代."""
    fronts = non_dominated_sort(individuals)
    next_population: list[Individual] = []

    for front in fronts:
        assign_crowding_distance(front)
        if len(next_population) + len(front) <= population_size:
            next_population.extend(front)
            continue

        sorted_front = sorted(front, key=lambda ind: ind.crowding_distance, reverse=True)
        remaining = population_size - len(next_population)
        next_population.extend(sorted_front[:remaining])
        break

    return next_population
