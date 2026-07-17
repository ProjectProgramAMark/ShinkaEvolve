import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome_summary,
    parent_stats,
    population_stats,
    operator_stats,
    rng,
):
    """Trusted evidence mutation-budget heredity scheduler."""
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
    scarcity = jnp.clip(1.0 - alive_fraction, 0.0, 1.0)
    energy_deficit = jnp.clip((0.55 - mean_energy_fraction) * 1.45, 0.0, 1.0)
    intake_deficit = jnp.clip((0.50 - mean_intake_ema) * 1.55, 0.0, 1.0)
    birth_deficit = jnp.clip((0.24 - birth_rate_ema) * 1.80, 0.0, 1.0)
    parent_quality = jnp.clip(0.62 * energy_fraction + 0.38 * intake_ema, 0.0, 1.0)
    parent_lag = jnp.clip(mean_energy_fraction - energy_fraction + mean_intake_ema - intake_ema, 0.0, 1.0)
    maturity = jnp.clip(age_fraction, 0.0, 1.0)
    genome_load = jnp.clip(0.54 * node_fraction + 0.46 * connection_fraction, 0.0, 1.0)
    compact = jnp.clip(1.0 - genome_load, 0.0, 1.0)
    stress = jnp.clip(
        0.34 * decline
        + 0.26 * death
        + 0.18 * energy_deficit
        + 0.14 * intake_deficit
        + 0.08 * scarcity,
        0.0,
        1.0,
    )
    recovery_room = jnp.clip(
        0.40 * birth_deficit
        + 0.24 * compact
        + 0.20 * parent_quality
        + 0.16 * (1.0 - growth),
        0.0,
        1.0,
    )
    fragility = jnp.clip(
        0.38 * death
        + 0.24 * scarcity
        + 0.22 * energy_deficit
        + 0.16 * parent_lag,
        0.0,
        1.0,
    )
    mutation_budget = jnp.clip(
        0.10
        + 0.30 * stress * recovery_room
        + 0.16 * birth_deficit * parent_quality
        + 0.08 * compact * (1.0 - death)
        - 0.18 * fragility * (1.0 - parent_quality),
        0.02,
        0.46,
    )
    prior = jnp.array([0.22, 0.51, 0.48, 0.39, 0.34, 0.40], dtype=jnp.float32)
    trust = jnp.clip(evidence * (0.45 + 0.55 * (1.0 - usage)), 0.0, 1.0)
    posterior = jnp.clip(prior + trust * (success - prior), 0.0, 1.0)
    clone_anchor = posterior[0]
    relative_value = jnp.clip(posterior - clone_anchor + 0.50, 0.0, 1.0)
    underused = jnp.clip(1.0 - usage, 0.0, 1.0)
    uncertainty = jnp.clip((1.0 - evidence) * underused, 0.0, 1.0)
    role = jnp.array([0.00, 0.34, 0.46, 0.38, 0.30, 0.42], dtype=jnp.float32)
    stress_role = jnp.array([0.00, 0.08, 0.25, 0.34, 0.16, 0.28], dtype=jnp.float32)
    compact_role = jnp.array([0.00, 0.10, 0.05, 0.10, 0.26, 0.16], dtype=jnp.float32)
    fragile_penalty = jnp.array([0.00, -0.03, -0.11, -0.26, -0.30, -0.22], dtype=jnp.float32)
    role_score = jnp.clip(
        role
        + stress_role * stress
        + compact_role * compact
        + fragile_penalty * fragility
        + 0.10 * relative_value
        + 0.035 * uncertainty,
        0.0,
        1.0,
    )
    mutation_shape = mutation_budget * role_score
    clone_guard = jnp.clip(
        1.0
        - mutation_budget
        + 0.28 * fragility
        + 0.18 * (1.0 - parent_quality)
        + 0.10 * growth,
        0.34,
        1.0,
    )
    base = jnp.array([1.18, -1.58, -1.82, -2.72, -3.02, -2.64], dtype=jnp.float32)
    value_gain = jnp.array([0.35, 1.20, 1.26, 1.06, 0.96, 1.08], dtype=jnp.float32)
    budget_gain = jnp.array([-2.10, 3.05, 3.35, 3.65, 3.15, 3.35], dtype=jnp.float32)
    ecology_gain = jnp.array(
        [
            1.65 * clone_guard,
            0.58 * parent_quality + 0.20 * maturity,
            0.50 * stress + 0.24 * parent_quality,
            0.76 * stress + 0.20 * compact,
            0.46 * compact + 0.20 * stress,
            0.54 * stress + 0.26 * compact,
        ],
        dtype=jnp.float32,
    )
    logits = (
        base
        + value_gain * (posterior - 0.42)
        + budget_gain * mutation_shape
        + ecology_gain
        - 0.36 * usage
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END