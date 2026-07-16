import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome_summary,
    parent_stats,
    population_stats,
    operator_stats,
    rng,
):
    """Vector-regime sparse heredity scheduler."""
    success = operator_stats[0]
    usage = operator_stats[1]
    evidence = operator_stats[2]
    node_fraction = parent_genome_summary[0]
    connection_fraction = parent_genome_summary[1]
    energy_fraction = parent_stats[0]
    intake_ema = parent_stats[1]
    alive_fraction = population_stats[0]
    mean_energy_fraction = population_stats[1]
    population_change_ema = population_stats[2]
    birth_rate_ema = population_stats[3]
    death_rate_ema = population_stats[4]
    mean_intake_ema = population_stats[5]
    decline = jnp.clip(-population_change_ema, 0.0, 1.0)
    growth = jnp.clip(population_change_ema, 0.0, 1.0)
    death = jnp.clip(death_rate_ema, 0.0, 1.0)
    birth_gap = jnp.clip(0.28 - birth_rate_ema, 0.0, 1.0)
    energy_gap = jnp.clip(0.56 - mean_energy_fraction, 0.0, 1.0)
    intake_gap = jnp.clip(0.54 - mean_intake_ema, 0.0, 1.0)
    parent_quality = jnp.clip(0.58 * energy_fraction + 0.42 * intake_ema, 0.0, 1.0)
    compact = jnp.clip(1.0 - 0.55 * node_fraction - 0.45 * connection_fraction, 0.0, 1.0)
    calm = jnp.clip(
        alive_fraction
        * (1.0 - death)
        * (1.0 - decline)
        * (0.58 + 0.42 * mean_energy_fraction),
        0.0,
        1.0,
    )
    stress_raw = jnp.clip(
        0.58 * decline
        + 0.52 * death
        + 0.28 * energy_gap
        + 0.32 * intake_gap
        + 0.16 * birth_gap
        + 0.12 * (1.0 - alive_fraction),
        0.0,
        1.0,
    )
    parent_edge = jnp.clip(
        0.5
        + 0.55 * (energy_fraction - mean_energy_fraction)
        + 0.45 * (intake_ema - mean_intake_ema),
        0.0,
        1.0,
    )
    reproductive_gap = jnp.clip(death + birth_gap - birth_rate_ema, 0.0, 1.0)
    ecology_need = jnp.clip(
        0.50 * stress_raw + 0.30 * reproductive_gap + 0.20 * intake_gap,
        0.0,
        1.0,
    )
    pressure = jnp.clip(
        ecology_need * (0.36 + 0.64 * parent_edge) * (0.44 + 0.56 * compact),
        0.0,
        1.0,
    )
    rescue = jnp.clip(
        intake_gap
        * (0.28 * decline + 0.22 * death + 0.16 * energy_gap + 0.28 * reproductive_gap)
        * (0.32 + 0.68 * parent_edge),
        0.0,
        1.0,
    )
    prior = jnp.array([0.24, 0.58, 0.52, 0.34, 0.32, 0.36], dtype=jnp.float32)
    learned = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    scarce = jnp.clip(1.0 - usage, 0.0, 1.0)
    evidence_gap = jnp.clip(1.0 - evidence, 0.0, 1.0)
    trusted = jnp.clip(evidence * (success - 0.34) * 3.00, 0.0, 1.0)
    clone_bad = jnp.clip(evidence[0] * (0.30 - success[0]) * 2.20, 0.0, 1.0)
    mutation_trust = jnp.clip(
        0.42 * trusted[1]
        + 0.42 * trusted[2]
        + 0.04 * trusted[3]
        + 0.03 * trusted[4]
        + 0.09 * trusted[5],
        0.0,
        1.0,
    )
    ecology_gate = jnp.clip(
        (0.30 + 0.70 * pressure)
        * (0.42 + 0.58 * parent_edge)
        * (0.50 + 0.50 * mutation_trust),
        0.0,
        1.0,
    )
    probe = evidence_gap * scarce * ecology_gate
    op_value = (
        1.48 * learned
        + 0.18 * scarce * ecology_gate
        + 0.10 * probe
        + 0.95 * trusted
        - 0.28 * usage
        - jnp.array([0.44, 0.00, 0.00, 0.00, 0.00, 0.00], dtype=jnp.float32) * clone_bad
    )
    base = jnp.array([4.18, -0.78, -1.46, -5.38, -6.20, -5.74], dtype=jnp.float32)
    calm_gain = jnp.array([2.36, 0.44, 0.06, -0.48, -0.44, -0.36], dtype=jnp.float32)
    pressure_gain = jnp.array([-1.44, 0.44, 1.06, 0.78, 0.34, 0.62], dtype=jnp.float32)
    rescue_gain = jnp.array([-0.34, 0.70, 1.08, 0.58, 0.20, 0.48], dtype=jnp.float32)
    quality_gain = jnp.array([0.00, 0.80, 0.54, 0.10, 0.00, 0.10], dtype=jnp.float32)
    shape_gain = jnp.array([0.00, 0.00, 0.00, 0.06, 0.44, 0.18], dtype=jnp.float32)
    growth_gain = jnp.array([0.16, 0.02, -0.12, -0.24, -0.18, 0.18], dtype=jnp.float32)
    value_gain = jnp.array([0.32, 1.04, 1.18, 0.62, 0.50, 0.58], dtype=jnp.float32)
    logits = (
        base
        + calm_gain * calm
        + pressure_gain * pressure
        + rescue_gain * rescue
        + quality_gain * parent_quality
        + shape_gain * compact
        + growth_gain * growth
        + value_gain * op_value
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END