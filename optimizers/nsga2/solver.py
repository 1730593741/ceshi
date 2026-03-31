"""DWTA-focused NSGA-II solver used by paper experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Sequence

from controller.operator_space import OperatorCapabilities, OperatorParams
from optimizers.nsga2.operators import (
    matrix_block_crossover,
    mutate_dwta_allocation,
)
from optimizers.nsga2.population import Individual
from optimizers.nsga2.selection import (
    assign_crowding_distance,
    environmental_selection,
    non_dominated_sort,
    parent_selection,
)
from src.dwta.constraints import constraint_breakdown as dwta_constraint_breakdown
from src.dwta.encoding import random_allocation
from src.dwta.live_cache import DWTALiveCache
from src.dwta.model import DWTABenchmarkData
from src.dwta.objectives import compute_objectives as compute_dwta_objectives
from src.dwta.repair import repair_allocation


@dataclass(slots=True)
class NSGA2Config:
    """Configuration for the minimal NSGA-II loop."""

    population_size: int = 40
    generations: int = 30
    crossover_prob: float = 0.9
    mutation_prob: float = 0.1
    repair_prob: float = 1.0
    eta_c: float | None = None
    eta_m: float | None = None
    local_search_prob: float | None = None
    seed: int | None = None


class NSGA2Solver:
    """NSGA-II solver for DWTA problems only."""

    def __init__(
        self,
        *,
        n_tasks: int,
        n_resources: int,
        cost_matrix: Sequence[Sequence[float]],
        task_loads: Sequence[float],
        capacities: Sequence[float],
        task_time_windows: Sequence[Sequence[float]] | None = None,
        resource_time_windows: Sequence[Sequence[float]] | None = None,
        compatibility_matrix: Sequence[Sequence[int]] | None = None,
        resource_stage_levels: Sequence[int] | None = None,
        stage_transitions: Sequence[Sequence[int]] | None = None,
        config: NSGA2Config,
        dwta_data: DWTABenchmarkData | None = None,
        dwta_live_cache: DWTALiveCache | None = None,
    ) -> None:
        if config.population_size <= 1:
            raise ValueError("population_size must be > 1")
        if config.generations < 0:
            raise ValueError("generations must be >= 0")
        if not 0.0 <= config.crossover_prob <= 1.0:
            raise ValueError("crossover_prob must be in [0, 1]")
        if not 0.0 <= config.mutation_prob <= 1.0:
            raise ValueError("mutation_prob must be in [0, 1]")
        if not 0.0 <= config.repair_prob <= 1.0:
            raise ValueError("repair_prob must be in [0, 1]")

        self.config = config
        self.rng = random.Random(config.seed)
        self.dwta_data = dwta_data
        self.dwta_live_cache = dwta_live_cache
        self._applied_dwta_event_indices: set[int] = set()

        # DWTA uses a benchmark object while keeping a uniform solver interface.
        if self.dwta_data is None:
            raise ValueError("NSGA2Solver now requires dwta_data (non-DWTA mode removed)")

        if self.dwta_live_cache is not None:
            live = self.dwta_live_cache.get_snapshot()
            dwta_n_weapons = live.n_weapons
            dwta_n_targets = live.n_targets
            ammo_capacities = live.ammo_capacities.astype(float).tolist()
        else:
            dwta_n_weapons = self.dwta_data.n_weapons
            dwta_n_targets = self.dwta_data.n_targets
            ammo_capacities = [float(value) for value in self.dwta_data.ammo_capacities]
        self.problem_mode = "dwta"
        self.n_tasks = dwta_n_weapons * dwta_n_targets
        self.n_resources = 1
        self.cost_matrix = [[0.0] for _ in range(self.n_tasks)]
        self.task_loads = [0.0] * self.n_tasks
        self.capacities = ammo_capacities
        self.task_time_windows = None
        self.resource_time_windows = None
        self.compatibility_matrix = None
        self.resource_stage_levels = None
        self.stage_transitions = None

    def get_operator_capabilities(self) -> OperatorCapabilities:
        """Describe which operator dimensions are meaningful for this problem.

        The controller uses this metadata to avoid proposing adjustments that
        would have no effect in the active solver/problem combination.
        """
        return OperatorCapabilities(
            supports_eta_c=True,
            supports_eta_m=True,
            supports_repair_prob=True,
            supports_local_search_prob=True,
        )

    def get_operator_params(self) -> OperatorParams:
        """Return a unified snapshot of the current operator parameters."""
        return OperatorParams(
            mutation_prob=self.config.mutation_prob,
            crossover_prob=self.config.crossover_prob,
            eta_c=self.config.eta_c,
            eta_m=self.config.eta_m,
            repair_prob=self.config.repair_prob,
            local_search_prob=self.config.local_search_prob,
        )

    def set_operator_params(self, params: OperatorParams) -> None:
        """Update runtime parameters using capability-aware application rules."""
        if not 0.0 <= params.crossover_prob <= 1.0:
            raise ValueError("crossover_prob must be in [0, 1]")
        if not 0.0 <= params.mutation_prob <= 1.0:
            raise ValueError("mutation_prob must be in [0, 1]")
        self.config.mutation_prob = params.mutation_prob
        self.config.crossover_prob = params.crossover_prob

        capabilities = self.get_operator_capabilities()
        if capabilities.supports_repair_prob and params.repair_prob is not None:
            if not 0.0 <= params.repair_prob <= 1.0:
                raise ValueError("repair_prob must be in [0, 1]")
            self.config.repair_prob = params.repair_prob
        if capabilities.supports_eta_c:
            self.config.eta_c = params.eta_c
        if capabilities.supports_eta_m:
            self.config.eta_m = params.eta_m
        if capabilities.supports_local_search_prob and params.local_search_prob is not None:
            if not 0.0 <= params.local_search_prob <= 1.0:
                raise ValueError("local_search_prob must be in [0, 1]")
            self.config.local_search_prob = params.local_search_prob

    def set_operator_probs(self, *, mutation_prob: float, crossover_prob: float) -> None:
        """Backward-compatible wrapper retained for early controller code paths."""
        self.set_operator_params(
            OperatorParams(
                mutation_prob=mutation_prob,
                crossover_prob=crossover_prob,
                repair_prob=self.config.repair_prob,
            )
        )

    def _evaluate(self, genome: list[int]) -> Individual:
        """Evaluate one genome and package the result as an ``Individual``."""
        objectives = compute_dwta_objectives(
            genome,
            n_weapons=self.dwta_data.n_weapons,
            n_targets=self.dwta_data.n_targets,
            required_damage=self.dwta_data.required_damage,
            lethality_matrix=self.dwta_data.lethality_matrix,
            live_cache=self.dwta_live_cache,
        )
        breakdown = dwta_constraint_breakdown(
            genome,
            ammo_capacities=self.dwta_data.ammo_capacities,
            compatibility_matrix=self.dwta_data.compatibility_matrix,
            n_targets=self.dwta_data.n_targets,
            live_cache=self.dwta_live_cache,
        )
        cv = breakdown.total
        return Individual(
            genome=genome,
            objectives=objectives,
            constraint_violation=cv,
            feasible=cv <= 0.0,
            constraint_components={
                "capacity": breakdown.capacity,
                "compatibility": breakdown.compatibility,
                "time_window": 0.0,
                "stage_transition": 0.0,
            },
        )

    def initialize_population(self) -> list[Individual]:
        """Create the initial repaired population used by the closed loop."""
        compatibility_matrix = (
            self.dwta_live_cache.get_snapshot().compatibility_mask.astype(int).tolist()
            if self.dwta_live_cache is not None
            else self.dwta_data.compatibility_matrix
        )
        return [
            self._evaluate(
                repair_allocation(
                    random_allocation(
                        self.dwta_data.ammo_capacities,
                        compatibility_matrix,
                        self.rng,
                    ),
                    ammo_capacities=self.dwta_data.ammo_capacities,
                    compatibility_matrix=self.dwta_data.compatibility_matrix,
                    n_targets=self.dwta_data.n_targets,
                    rng=self.rng,
                    live_cache=self.dwta_live_cache,
                )
            )
            for _ in range(self.config.population_size)
        ]

    def _annotate_population(self, individuals: list[Individual]) -> None:
        """Refresh NSGA-II rank/crowding annotations in-place for a population."""
        fronts = non_dominated_sort(individuals)
        for front in fronts:
            assign_crowding_distance(front)

    def reevaluate_population(self, population: list[Individual]) -> list[Individual]:
        """Re-evaluate existing genomes under the current environment state."""
        refreshed = [self._evaluate(individual.genome) for individual in population]
        self._annotate_population(refreshed)
        return refreshed

    def apply_runtime_events(self, *, generation: int) -> list[dict[str, Any]]:
        """Apply DWTA scripted events scheduled at a given generation.

        Returns a list of structured event payloads for logging. Static problems
        and non-scripted DWTA modes return an empty list.
        """
        if self.dwta_live_cache is None:
            return []
        environment = self.dwta_live_cache.environment
        script = environment.script
        if script is None:
            return []

        applied_events: list[dict[str, Any]] = []
        for idx, event in enumerate(script.waves):
            if idx in self._applied_dwta_event_indices:
                continue
            if event.trigger_generation != generation:
                continue
            applied_events.append(environment.apply_wave_event(event))
            self._applied_dwta_event_indices.add(idx)

        if applied_events:
            self.dwta_live_cache.invalidate()
            snapshot = self.dwta_live_cache.refresh(force=True)
            self.dwta_data = snapshot.as_benchmark_data()

        return applied_events

    def _make_offspring(self, population: list[Individual]) -> list[Individual]:
        """Generate one offspring batch using selection, variation, and repair.

        For DWTA problems, the controller-adjustable operator knobs map to a
        matrix-aware variation process:

        - ``mutation_prob`` controls per-cell mutation trigger rate;
        - ``eta_m`` shapes mutation locality;
        - ``local_search_prob`` enables an additional row-level improvement move;
        - ``eta_c`` biases matrix block crossover behavior;
        - ``repair_prob`` keeps its standard post-variation feasibility role.
        """
        self._annotate_population(population)
        parents = parent_selection(population, self.config.population_size, self.rng)

        dwta_mutation_step = max(1.0, self.config.mutation_prob * 10.0)

        # Cache the compatibility matrix once per generation — snapshot is
        # stable within a single _make_offspring call.
        compat_matrix = (
            self.dwta_live_cache.get_snapshot().compatibility_mask.astype(int).tolist()
            if self.dwta_live_cache is not None
            else self.dwta_data.compatibility_matrix
        )

        offspring: list[Individual] = []
        for idx in range(0, len(parents), 2):
            parent_a = parents[idx]
            parent_b = parents[(idx + 1) % len(parents)]

            child_a, child_b = matrix_block_crossover(
                parent_a.genome,
                parent_b.genome,
                n_weapons=self.dwta_data.n_weapons,
                n_targets=self.dwta_data.n_targets,
                crossover_prob=self.config.crossover_prob,
                rng=self.rng,
                eta_c=self.config.eta_c,
            )
            child_a = mutate_dwta_allocation(
                child_a,
                n_weapons=self.dwta_data.n_weapons,
                n_targets=self.dwta_data.n_targets,
                compatibility_matrix=compat_matrix,
                mutation_prob=self.config.mutation_prob,
                rng=self.rng,
                mutation_step=dwta_mutation_step,
                eta_m=self.config.eta_m,
                local_search_prob=self.config.local_search_prob,
            )
            child_b = mutate_dwta_allocation(
                child_b,
                n_weapons=self.dwta_data.n_weapons,
                n_targets=self.dwta_data.n_targets,
                compatibility_matrix=compat_matrix,
                mutation_prob=self.config.mutation_prob,
                rng=self.rng,
                mutation_step=dwta_mutation_step,
                eta_m=self.config.eta_m,
                local_search_prob=self.config.local_search_prob,
            )

            repaired_a = child_a
            repaired_b = child_b
            if self.rng.random() < self.config.repair_prob:
                repaired_a = repair_allocation(
                    child_a,
                    ammo_capacities=self.dwta_data.ammo_capacities,
                    compatibility_matrix=self.dwta_data.compatibility_matrix,
                    n_targets=self.dwta_data.n_targets,
                    rng=self.rng,
                    live_cache=self.dwta_live_cache,
                )
            if self.rng.random() < self.config.repair_prob:
                repaired_b = repair_allocation(
                    child_b,
                    ammo_capacities=self.dwta_data.ammo_capacities,
                    compatibility_matrix=self.dwta_data.compatibility_matrix,
                    n_targets=self.dwta_data.n_targets,
                    rng=self.rng,
                    live_cache=self.dwta_live_cache,
                )

            offspring.append(self._evaluate(repaired_a))
            if len(offspring) < self.config.population_size:
                offspring.append(self._evaluate(repaired_b))

        return offspring[: self.config.population_size]

    def evolve_one_generation(self, population: list[Individual]) -> list[Individual]:
        """Advance the population by one NSGA-II generation."""
        offspring = self._make_offspring(population)
        merged = population + offspring
        return environmental_selection(merged, self.config.population_size)

    def run(self) -> list[Individual]:
        """Run the configured number of generations and return the final population."""
        population = self.initialize_population()

        for _ in range(self.config.generations):
            population = self.evolve_one_generation(population)

        self._annotate_population(population)
        return population

    def solve(self) -> list[Individual]:
        """Alias for ``run`` kept for callers that prefer solver-style naming."""
        return self.run()
