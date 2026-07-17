import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome_summary,
    parent_stats,
    population_stats,
    operator_stats,
    rng,
):
    """Homeostatic evidence-gated heredity scheduler."""
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
    energy_gap = jnp.clip(0.52 - mean_energy_fraction, 0.0, 1.0)
    intake_gap = jnp.clip(0.50 - mean_intake_ema, 0.0, 1.0)
    birth_gap = jnp.clip(0.22 - birth_rate_ema, 0.0, 1.0)
    crowd_gap = jnp.clip(0.82 - alive_fraction, 0.0, 1.0)
    parent_energy_edge = jnp.clip(energy_fraction - mean_energy_fraction + 0.18, 0.0, 1.0)
    parent_intake_edge = jnp.clip(intake_ema - mean_intake_ema + 0.18, 0.0, 1.0)
    parent_vigor = jnp.clip(
        0.48 * energy_fraction
        + 0.30 * intake_ema
        + 0.22 * parent_energy_edge * parent_intake_edge,
        0.0,
        1.0,
    )
    compact = jnp.clip(1.0 - 0.52 * node_fraction - 0.48 * connection_fraction, 0.0, 1.0)
    maturity = jnp.clip(1.0 - 0.55 * age_fraction, 0.0, 1.0)
    calm = jnp.clip(
        alive_fraction
        * (1.0 - death)
        * (1.0 - decline)
        * (0.52 + 0.48 * mean_energy_fraction)
        * (0.55 + 0.45 * mean_intake_ema),
        0.0,
        1.0,
    )
    raw_need = jnp.clip(
        0.46 * decline
        + 0.34 * energy_gap
        + 0.34 * intake_gap
        + 0.24 * birth_gap
        + 0.14 * crowd_gap,
        0.0,
        1.0,
    )
    collapse = jnp.clip(0.74 * death + 0.38 * decline + 0.18 * crowd_gap, 0.0, 1.0)
    moderate_need = jnp.clip(raw_need * (1.0 - 0.68 * collapse) * (1.0 - 0.48 * growth), 0.0, 1.0)
    opportunity = jnp.clip(
        moderate_need
        * (0.34 + 0.66 * parent_vigor)
        * (0.42 + 0.58 * compact)
        * (0.55 + 0.45 * maturity)
        * (1.0 - 0.52 * calm),
        0.0,
        1.0,
    )
    prior = jnp.array([0.20, 0.55, 0.50, 0.38, 0.34, 0.37], dtype=jnp.float32)
    learned = jnp.clip(prior + evidence * (success - prior), 0.0, 1.0)
    trust = jnp.clip(evidence * (success - prior), -0.32, 0.32)
    clone_edge = trust[0]
    relative_trust = jnp.clip(trust - clone_edge, -0.22, 0.34)
    underused = jnp.clip(1.0 - usage, 0.0, 1.0)
    credible_probe = jnp.clip((1.0 - evidence) * underused * opportunity, 0.0, 1.0)
    conservative_gate = jnp.clip(
        opportunity
        * (0.72 + 0.28 * parent_vigor)
        * (1.0 - 0.35 * death),
        0.0,
        1.0,
    )
    standard_gate = jnp.clip(
        opportunity
        * jnp.clip(0.25 + 0.75 * raw_need, 0.0, 1.0)
        * (1.0 - 0.48 * death),
        0.0,
        1.0,
    )
    exploratory_gate = jnp.clip(
        opportunity
        * compact
        * birth_gap
        * (1.0 - 0.62 * death)
        * (0.20 + 0.80 * parent_intake_edge),
        0.0,
        1.0,
    )
    structural_gate = jnp.clip(
        opportunity
        * compact
        * compact
        * jnp.clip(0.18 + 0.82 * birth_gap, 0.0, 1.0)
        * (1.0 - 0.70 * death),
        0.0,
        1.0,
    )
    mixed_gate = jnp.clip(
        opportunity
        * compact
        * jnp.clip(0.22 + 0.78 * parent_vigor, 0.0, 1.0)
        * jnp.clip(0.20 + 0.80 * birth_gap, 0.0, 1.0)
        * (1.0 - 0.64 * death),
        0.0,
        1.0,
    )
    gates = jnp.array(
        [
            calm,
            conservative_gate,
            standard_gate,
            exploratory_gate,
            structural_gate,
            mixed_gate,
        ],
        dtype=jnp.float32,
    )
    base = jnp.array([4.82, -1.20, -1.88, -5.18, -5.76, -5.42], dtype=jnp.float32)
    need_gain = jnp.array([-2.10, 0.88, 1.28, 1.10, 0.78, 0.96], dtype=jnp.float32)
    calm_gain = jnp.array([1.62, 0.18, -0.10, -0.36, -0.34, -0.32], dtype=jnp.float32)
    value_gain = jnp.array([0.34, 0.98, 1.08, 0.66, 0.58, 0.62], dtype=jnp.float32)
    trust_gain = jnp.array([0.04, 0.86, 0.96, 0.52, 0.42, 0.48], dtype=jnp.float32)
    probe_gain = jnp.array([-0.10, 0.12, 0.22, 0.52, 0.36, 0.44], dtype=jnp.float32)
    gate_gain = jnp.array([0.30, 1.34, 1.18, 0.88, 0.64, 0.78], dtype=jnp.float32)
    usage_cost = jnp.array([0.20, 0.34, 0.38, 0.32, 0.28, 0.30], dtype=jnp.float32)
    logits = (
        base
        + need_gain * moderate_need
        + calm_gain * calm
        + value_gain * learned
        + trust_gain * relative_trust
        + probe_gain * credible_probe
        + gate_gain * gates
        - usage_cost * usage
    )
    clone_guard = jnp.clip(collapse + calm + growth, 0.0, 1.0)
    logits = logits + jnp.array([1.10, -0.26, -0.34, -0.58, -0.62, -0.58], dtype=jnp.float32) * clone_guard
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END