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
    viability = jnp.clip(
        alive_fraction
        * (0.45 + 0.55 * mean_energy_fraction)
        * (1.0 - 0.55 * death),
        0.0,
        1.0,
    )
    normalized_need = jnp.clip(
        0.46 * stress_raw
        + 0.24 * intake_gap * (1.0 - mean_intake_ema)
        + 0.18 * energy_gap * (1.0 - mean_energy_fraction)
        + 0.12 * birth_gap * (1.0 - birth_rate_ema),
        0.0,
        1.0,
    )
    pressure = jnp.clip(
        normalized_need
        * (0.35 + 0.65 * parent_edge)
        * (0.38 + 0.62 * viability),
        0.0,
        1.0,
    )
    rescue = jnp.clip(
        intake_gap
        * (0.28 * decline + 0.22 * death + 0.18 * energy_gap + 0.18 * birth_gap)
        * (0.42 + 0.58 * parent_edge)
        * (0.50 + 0.50 * viability),
        0.0,
        1.0,
    )
    prior = jnp.array([0.24, 0.58, 0.52, 0.36, 0.34, 0.38], dtype=jnp.float32)
    learned = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    scarce = jnp.clip(1.0 - usage, 0.0, 1.0)
    evidence_gap = jnp.clip(1.0 - evidence, 0.0, 1.0)
    trusted = jnp.clip(evidence * (success - 0.34) * 3.20, 0.0, 1.0)
    clone_trust = jnp.clip(evidence[0] * (success[0] - 0.22) * 2.20, 0.0, 1.0)
    relative_trust = jnp.clip(trusted - 0.28 * clone_trust, 0.0, 1.0)
    mutation_trust = jnp.clip(
        0.46 * relative_trust[1]
        + 0.40 * relative_trust[2]
        + 0.04 * relative_trust[3]
        + 0.03 * relative_trust[4]
        + 0.07 * relative_trust[5],
        0.0,
        1.0,
    )
    ecology_gate = jnp.clip(
        (0.20 + 0.80 * pressure)
        * (0.48 + 0.52 * parent_edge)
        * (0.56 + 0.44 * mutation_trust),
        0.0,
        1.0,
    )
    clone_bad = jnp.clip(evidence[0] * (0.30 - success[0]) * 2.40, 0.0, 1.0)
    parametric_release = jnp.clip(
        clone_bad
        * (0.24 + 0.76 * ecology_gate)
        * (0.64 + 0.36 * mutation_trust),
        0.0,
        1.0,
    )
    probe = evidence_gap * scarce * ecology_gate
    op_value = (
        1.46 * learned
        + 0.16 * scarce * ecology_gate
        + 0.08 * probe
        + 0.96 * relative_trust
        - 0.30 * usage
    )
    base = jnp.array([4.20, -0.88, -1.62, -5.42, -6.16, -5.78], dtype=jnp.float32)
    calm_gain = jnp.array([2.38, 0.46, 0.08, -0.44, -0.38, -0.32], dtype=jnp.float32)
    pressure_gain = jnp.array([-1.22, 0.40, 0.98, 0.70, 0.30, 0.54], dtype=jnp.float32)
    rescue_gain = jnp.array([-0.34, 0.66, 1.02, 0.54, 0.18, 0.42], dtype=jnp.float32)
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
        + parametric_release
        * jnp.array([-0.54, 0.82, 0.92, 0.08, 0.04, 0.06], dtype=jnp.float32)
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END