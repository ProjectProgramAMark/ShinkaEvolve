import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome_summary,
    parent_stats,
    population_stats,
    operator_stats,
    rng,
):
    """Stress-budget bandit scheduler for six heredity actions."""
    success = operator_stats[0]
    usage = operator_stats[1]
    evidence = operator_stats[2]
    node_fraction = parent_genome_summary[0]
    connection_fraction = parent_genome_summary[1]
    energy_fraction = parent_stats[0]
    intake_ema = parent_stats[1]
    age_fraction = parent_stats[2]
    alive_fraction = population_stats[0]
    mean_energy_fraction = population_stats[1]
    population_change_ema = population_stats[2]
    birth_rate_ema = population_stats[3]
    death_rate_ema = population_stats[4]
    mean_intake_ema = population_stats[5]
    decline = jnp.clip(-population_change_ema, 0.0, 1.0)
    growth = jnp.clip(population_change_ema, 0.0, 1.0)
    death = jnp.clip(death_rate_ema, 0.0, 1.0)
    sparse = jnp.clip(1.0 - alive_fraction, 0.0, 1.0)
    energy_short = jnp.clip(0.55 - mean_energy_fraction, 0.0, 1.0)
    intake_short = jnp.clip(0.54 - mean_intake_ema, 0.0, 1.0)
    birth_short = jnp.clip(0.28 - birth_rate_ema, 0.0, 1.0)
    parent_power = jnp.clip(
        0.52 * energy_fraction + 0.34 * intake_ema + 0.14 * (1.0 - age_fraction),
        0.0,
        1.0,
    )
    compact = jnp.clip(1.0 - 0.50 * node_fraction - 0.50 * connection_fraction, 0.0, 1.0)
    leverage = jnp.clip(
        parent_power
        * (0.45 + 0.55 * energy_fraction)
        * (0.45 + 0.55 * intake_ema)
        * (0.58 + 0.42 * compact),
        0.0,
        1.0,
    )
    fast_stress = jnp.clip(0.82 * decline + 0.72 * death + 0.34 * sparse, 0.0, 1.0)
    resource_stress = jnp.clip(0.44 * intake_short + 0.34 * energy_short + 0.20 * birth_short, 0.0, 1.0)
    stress = jnp.clip(
        fast_stress * (0.44 + 0.56 * resource_stress) + 0.16 * sparse * resource_stress,
        0.0,
        1.0,
    )
    stable = jnp.clip(
        alive_fraction
        * (1.0 - decline)
        * (1.0 - death)
        * (0.62 + 0.38 * mean_energy_fraction)
        * (0.62 + 0.38 * mean_intake_ema),
        0.0,
        1.0,
    )
    no_spend = jnp.clip(stable * (1.0 - 0.88 * stress) * (0.82 + 0.18 * growth), 0.0, 1.0)
    fragile = jnp.clip(stable * (1.0 - parent_power) * (0.48 + 0.52 * (1.0 - compact)), 0.0, 1.0)
    prior = jnp.array([0.38, 0.46, 0.33, 0.12, 0.10, 0.18], dtype=jnp.float32)
    estimate = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    trust = jnp.clip(0.10 + 0.90 * evidence, 0.0, 1.0)
    novelty = jnp.clip((1.0 - usage) * (1.0 - 0.62 * evidence), 0.0, 1.0)
    clone_failure = jnp.clip(evidence[0] * (0.39 - success[0]) + usage[0] * evidence[0] * 0.10, 0.0, 1.0)
    conservative_proven = jnp.clip((estimate[1] - estimate[0] + 0.03) * trust[1] + 0.24 * clone_failure, 0.0, 1.0)
    standard_proven = jnp.clip((estimate[2] - estimate[0] + 0.04) * trust[2] + 0.30 * clone_failure, 0.0, 1.0)
    mixed_proven = jnp.clip((estimate[5] - estimate[0] + 0.02) * trust[5] + 0.12 * clone_failure, 0.0, 1.0)
    conservative_fatigue = jnp.clip(usage[1] * evidence[1] * (0.45 - success[1]), 0.0, 1.0)
    standard_fatigue = jnp.clip(usage[2] * evidence[2] * (0.43 - success[2]), 0.0, 1.0)
    rescue_budget = jnp.clip(
        stress
        * leverage
        * (0.18 + 0.82 * clone_failure + 0.26 * novelty[2])
        * (1.0 - 0.78 * no_spend)
        * (1.0 - 0.55 * fragile),
        0.0,
        1.0,
    )
    renewal_budget = jnp.clip(
        stable
        * leverage
        * conservative_proven
        * (0.16 + 0.84 * clone_failure)
        * (1.0 - 0.86 * stress)
        * (1.0 - 0.70 * conservative_fatigue)
        * (1.0 - 0.58 * fragile),
        0.0,
        1.0,
    )
    probe_budget = jnp.clip(
        stress
        * leverage
        * novelty
        * (0.35 + 0.65 * clone_failure)
        * (1.0 - 0.92 * no_spend),
        0.0,
        1.0,
    )
    mutation_tax = jnp.clip(no_spend * (0.88 + 0.12 * growth) + 0.66 * fragile, 0.0, 1.0)
    hard_stress = jnp.clip(
        0.52 * decline + 0.50 * death + 0.32 * birth_short + 0.18 * sparse,
        0.0,
        1.0,
    )
    exploit_window = jnp.clip(
        leverage
        * (0.42 + 0.58 * parent_power)
        * (0.30 + 0.70 * birth_rate_ema)
        * (0.58 + 0.42 * compact)
        * (1.0 - 0.72 * fragile),
        0.0,
        1.0,
    )
    evidence_release = jnp.clip(
        0.56 * clone_failure
        + 0.20 * conservative_proven
        + 0.24 * standard_proven
        + 0.08 * novelty[1]
        + 0.10 * novelty[2],
        0.0,
        1.0,
    )
    stress_release = jnp.clip(
        stress
        * hard_stress
        * exploit_window
        * evidence_release
        * (1.0 - 0.98 * no_spend)
        * (1.0 - 0.70 * stable),
        0.0,
        1.0,
    )
    clone_logit = (
        4.72
        + 1.62 * no_spend
        + 0.88 * fragile
        + 0.30 * growth
        - 1.85 * rescue_budget * hard_stress
        - 0.42 * renewal_budget
        - 4.10 * stress_release
        - 1.05 * clone_failure * hard_stress
    )
    conservative_logit = (
        -2.24
        + 1.85 * renewal_budget
        + 3.10 * stress_release * (0.46 + 0.54 * conservative_proven)
        + 0.28 * hard_stress * conservative_proven
        + 0.22 * fragile * conservative_proven
        - 2.28 * mutation_tax
        - 0.82 * conservative_fatigue
    )
    standard_logit = (
        -4.30
        + 3.85 * rescue_budget * hard_stress * (0.40 + 0.60 * standard_proven)
        + 5.60 * stress_release * (0.44 + 0.56 * standard_proven)
        + 0.82 * hard_stress * leverage * novelty[2] * clone_failure
        - 2.72 * no_spend
        - 1.42 * fragile
        - 0.80 * standard_fatigue
    )
    exploratory_logit = (
        -6.55
        + 0.95 * probe_budget[3] * compact
        + 0.42 * rescue_budget * novelty[3]
        - 1.20 * no_spend
        - 0.70 * fragile
    )
    structural_logit = (
        -6.75
        + 0.70 * probe_budget[4] * compact
        + 0.24 * rescue_budget * novelty[4]
        - 1.20 * no_spend
        - 0.75 * fragile
    )
    mixed_logit = (
        -6.35
        + 1.25 * rescue_budget * mixed_proven * compact
        + 0.62 * probe_budget[5] * compact
        - 1.35 * no_spend
        - 0.78 * fragile
    )
    logits = jnp.array(
        [
            clone_logit,
            conservative_logit,
            standard_logit,
            exploratory_logit,
            structural_logit,
            mixed_logit,
        ],
        dtype=jnp.float32,
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END