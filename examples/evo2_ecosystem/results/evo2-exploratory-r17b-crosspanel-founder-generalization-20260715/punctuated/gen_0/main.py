import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome_summary,
    parent_stats,
    population_stats,
    operator_stats,
    rng,
):
    """Regime-mixed evidence scheduler for six heredity actions."""
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
    low_alive = jnp.clip(1.0 - alive_fraction, 0.0, 1.0)
    energy_gap = jnp.clip(0.58 - mean_energy_fraction, 0.0, 1.0)
    intake_gap = jnp.clip(0.56 - mean_intake_ema, 0.0, 1.0)
    birth_gap = jnp.clip(0.31 - birth_rate_ema, 0.0, 1.0)
    parent_vigor = jnp.clip(0.58 * energy_fraction + 0.34 * intake_ema + 0.08 * (1.0 - age_fraction), 0.0, 1.0)
    compact = jnp.clip(1.0 - 0.46 * node_fraction - 0.54 * connection_fraction, 0.0, 1.0)
    simple_parent = jnp.clip(0.55 + 0.45 * compact, 0.0, 1.0)
    stress_core = jnp.clip(
        0.66 * decline
        + 0.58 * death
        + 0.32 * low_alive
        + 0.34 * intake_gap
        + 0.24 * energy_gap
        + 0.12 * birth_gap,
        0.0,
        1.0,
    )
    injury_rescue = jnp.clip(
        (0.58 * decline + 0.48 * death + 0.22 * low_alive)
        * (0.38 + 0.42 * intake_gap + 0.20 * energy_gap),
        0.0,
        1.0,
    )
    stable_world = jnp.clip(
        alive_fraction
        * (1.0 - decline)
        * (1.0 - death)
        * (0.52 + 0.48 * mean_energy_fraction)
        * (1.0 - 0.72 * injury_rescue),
        0.0,
        1.0,
    )
    renewal_world = jnp.clip(stable_world * parent_vigor * simple_parent * (1.0 - 0.45 * growth), 0.0, 1.0)
    probe_world = jnp.clip(injury_rescue * (1.0 - 0.82 * stable_world) * (0.42 + 0.58 * intake_gap), 0.0, 1.0)
    prior = jnp.array([0.15, 0.61, 0.51, 0.35, 0.31, 0.41], dtype=jnp.float32)
    learned = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    unused = jnp.clip((1.0 - usage) * (1.0 - 0.52 * evidence), 0.0, 1.0)
    utility = jnp.clip(1.50 * learned + 0.26 * unused - 0.18 * usage, 0.0, 2.0)
    clone_bad = jnp.clip(evidence[0] * (0.43 - success[0]), 0.0, 1.0)
    conservative_edge = jnp.clip(learned[1] - learned[0] + 0.38 * clone_bad, 0.0, 1.0)
    standard_edge = jnp.clip(learned[2] - learned[0] + 0.28 * clone_bad, 0.0, 1.0)
    mixed_edge = jnp.clip(learned[5] - learned[0] + 0.18 * clone_bad, 0.0, 1.0)
    stable_lock = jnp.clip(stable_world * (0.60 + 0.30 * growth) * (1.0 - 0.82 * injury_rescue), 0.0, 1.0)
    renewal_param = jnp.clip(renewal_world * conservative_edge * (1.0 - 0.70 * injury_rescue), 0.0, 1.0)
    rescue_param = jnp.clip((0.70 * stress_core + 1.02 * injury_rescue + 0.16 * low_alive) * standard_edge * (1.0 - 0.40 * stable_lock), 0.0, 1.0)
    conservative_gate = jnp.clip(0.82 * renewal_param + 0.16 * injury_rescue * conservative_edge + 0.12 * stable_world * clone_bad, 0.0, 1.0)
    standard_gate = jnp.clip(
        (0.42 * probe_world * (0.45 + 0.45 * utility[2]) + 0.52 * rescue_param)
        * (1.0 - 0.46 * stable_lock),
        0.0,
        1.0,
    )
    mixed_gate = jnp.clip(
        (0.22 * probe_world * (0.34 + 0.40 * utility[5]) + 0.18 * rescue_param * mixed_edge)
        * compact
        * (1.0 - 0.54 * stable_lock),
        0.0,
        1.0,
    )
    exploratory_gate = jnp.clip(
        probe_world * unused[3] * (0.14 + 0.24 * compact) * injury_rescue * (1.0 - 0.58 * stable_lock),
        0.0,
        1.0,
    )
    structural_gate = jnp.clip(
        probe_world * unused[4] * compact * 0.15 * injury_rescue * (1.0 - 0.58 * stable_lock),
        0.0,
        1.0,
    )
    anchor = jnp.array([4.50, -1.02, -2.70, -6.05, -6.34, -5.94], dtype=jnp.float32)
    stable_policy = jnp.array([2.42, 0.58, -1.48, -1.96, -2.08, -1.78], dtype=jnp.float32)
    renewal_policy = jnp.array([-0.58, 1.86, 0.14, -0.36, -0.44, -0.24], dtype=jnp.float32)
    stress_policy = jnp.array([-2.72, -0.02, 0.96, 0.45, 0.28, 0.58], dtype=jnp.float32)
    rescue_policy = jnp.array([-1.22, 0.28, 1.82, 0.46, 0.22, 0.82], dtype=jnp.float32)
    utility_policy = jnp.array([0.24, 0.78, 1.00, 0.36, 0.28, 0.46], dtype=jnp.float32)
    pulse = jnp.array(
        [
            -0.92 * standard_gate - 0.52 * mixed_gate - 1.54 * renewal_param - 0.90 * clone_bad * injury_rescue - 0.26 * stable_world * clone_bad,
            2.24 * conservative_gate + 0.16 * injury_rescue + 0.16 * stable_lock * conservative_edge,
            2.46 * standard_gate + 0.70 * rescue_param - 0.42 * stable_lock,
            0.70 * exploratory_gate,
            0.48 * structural_gate,
            1.18 * mixed_gate,
        ],
        dtype=jnp.float32,
    )
    growth_trim = jnp.array(
        [0.12 * growth, 0.04 * growth, -0.24 * growth, -0.20 * growth, -0.18 * growth, 0.06 * growth],
        dtype=jnp.float32,
    )
    logits = (
        anchor
        + stable_policy * stable_world
        + renewal_policy * renewal_world
        + stress_policy * stress_core
        + rescue_policy * injury_rescue
        + utility_policy * utility
        + pulse
        + growth_trim
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END