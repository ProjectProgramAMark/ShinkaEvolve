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
    pressure = jnp.clip(
        0.78 * decline
        + 0.68 * death
        + 0.38 * energy_gap
        + 0.44 * intake_gap
        + 0.18 * birth_gap
        + 0.16 * (1.0 - alive_fraction),
        0.0,
        1.0,
    )
    rescue = jnp.clip(
        intake_gap * (0.52 * decline + 0.34 * death + 0.24 * energy_gap),
        0.0,
        1.0,
    )
    prior = jnp.array([0.22, 0.56, 0.50, 0.40, 0.37, 0.39], dtype=jnp.float32)
    learned = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    scarce = jnp.clip(1.0 - usage, 0.0, 1.0)
    evidence_gap = jnp.clip(1.0 - evidence, 0.0, 1.0)
    probe = evidence_gap * scarce
    op_value = 1.85 * learned + 0.38 * scarce + 0.14 * probe - 0.22 * usage
    base = jnp.array([4.34, -1.12, -2.04, -4.88, -5.58, -5.26], dtype=jnp.float32)
    calm_gain = jnp.array([2.12, 0.62, 0.22, -0.24, -0.20, -0.18], dtype=jnp.float32)
    pressure_gain = jnp.array([-2.34, 0.10, 1.18, 1.75, 1.42, 1.55], dtype=jnp.float32)
    rescue_gain = jnp.array([-0.62, 0.52, 1.06, 1.42, 0.82, 1.04], dtype=jnp.float32)
    quality_gain = jnp.array([0.00, 0.88, 0.44, 0.10, 0.00, 0.06], dtype=jnp.float32)
    shape_gain = jnp.array([0.00, 0.00, 0.00, 0.12, 0.68, 0.20], dtype=jnp.float32)
    growth_gain = jnp.array([0.08, 0.04, -0.10, -0.16, -0.10, 0.28], dtype=jnp.float32)
    value_gain = jnp.array([0.46, 0.78, 0.92, 0.70, 0.68, 0.68], dtype=jnp.float32)
    recovery_probe = jnp.clip(
        (0.42 + 0.58 * parent_quality)
        * (0.35 + 0.65 * compact)
        * (0.28 + 0.72 * birth_gap)
        * (1.0 - 0.62 * death)
        * (0.52 + 0.48 * jnp.clip(pressure + rescue, 0.0, 1.0)),
        0.0,
        1.0,
    )
    trust_center = jnp.clip(evidence * (success - prior), -0.35, 0.35)
    probe_gain = jnp.array([-0.34, 0.18, 0.42, 1.20, 0.86, 1.04], dtype=jnp.float32)
    trust_gain = jnp.array([0.10, 0.56, 0.74, 0.58, 0.50, 0.54], dtype=jnp.float32)
    logits = (
        base
        + calm_gain * calm
        + pressure_gain * pressure
        + rescue_gain * rescue
        + quality_gain * parent_quality
        + shape_gain * compact
        + growth_gain * growth
        + value_gain * op_value
        + probe_gain * recovery_probe
        + trust_gain * trust_center
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END