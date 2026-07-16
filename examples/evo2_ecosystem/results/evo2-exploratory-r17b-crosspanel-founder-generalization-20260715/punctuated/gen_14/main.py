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
    prior = jnp.array([0.18, 0.62, 0.48, 0.28, 0.24, 0.34], dtype=jnp.float32)
    learned = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    credible = jnp.clip(0.22 + 0.78 * evidence, 0.0, 1.0)
    unused = jnp.clip((1.0 - usage) * (1.0 - 0.58 * evidence), 0.0, 1.0)
    utility = jnp.clip((1.42 * learned + 0.22 * unused - 0.24 * usage) * credible, 0.0, 2.0)
    clone_bad = jnp.clip(evidence[0] * (0.46 - success[0]) + 0.20 * usage[0] * evidence[0] * (learned[1] - learned[0]), 0.0, 1.0)
    conservative_edge = jnp.clip((learned[1] - learned[0]) * (0.55 + 0.45 * evidence[1]) + 0.46 * clone_bad, 0.0, 1.0)
    standard_edge = jnp.clip((learned[2] - learned[0]) * (0.38 + 0.62 * evidence[2]) + 0.30 * clone_bad + 0.16 * unused[2] * injury_rescue, 0.0, 1.0)
    mixed_edge = jnp.clip((learned[5] - learned[0]) * (0.36 + 0.64 * evidence[5]) + 0.16 * clone_bad, 0.0, 1.0)
    conservative_fatigue = jnp.clip(evidence[1] * usage[1] * (0.42 - success[1]), 0.0, 1.0)
    stable_mutation_debt = jnp.clip(stable_world * (1.0 - injury_rescue) * (0.34 + 0.66 * conservative_fatigue), 0.0, 1.0)
    standard_rescue_credit = jnp.clip(injury_rescue * (0.18 + 0.82 * stress_core) * (0.46 * unused[2] + 0.54 * standard_edge), 0.0, 1.0)
    stable_sham_guard = jnp.clip(stable_world * (1.0 - injury_rescue) * (0.62 + 0.38 * mean_energy_fraction) * (1.0 - 0.42 * clone_bad), 0.0, 1.0)
    injury_tail_window = jnp.clip(injury_rescue * stress_core * (0.46 + 0.54 * clone_bad) * (1.0 - 0.44 * stable_sham_guard), 0.0, 1.0)
    birth_pressure = jnp.clip(birth_gap * stable_world * parent_vigor * simple_parent * (1.0 - 0.76 * injury_rescue), 0.0, 1.0)
    conservative_reseed = jnp.clip(birth_pressure * conservative_edge * (0.18 + 0.82 * clone_bad) * (0.36 + 0.64 * evidence[1]) * (1.0 - 0.72 * conservative_fatigue), 0.0, 1.0)
    standard_tail_probe = jnp.clip(
        injury_rescue
        * stress_core
        * (0.42 + 0.58 * clone_bad)
        * (0.34 + 0.66 * unused[2])
        * (0.58 + 0.42 * standard_edge)
        * (1.0 - 0.62 * stable_sham_guard * (1.0 - 0.72 * injury_tail_window)),
        0.0,
        1.0,
    )
    stable_lock = jnp.clip(stable_world * (0.72 + 0.22 * growth) * (1.0 - 0.90 * injury_rescue) * (1.0 - 0.34 * clone_bad), 0.0, 1.0)
    fragile_lock = jnp.clip(stable_world * (1.0 - parent_vigor) * (0.48 + 0.52 * (1.0 - compact)), 0.0, 1.0)
    renewal_param = jnp.clip(renewal_world * conservative_edge * (1.0 - 0.72 * injury_rescue) * (1.0 - 0.35 * fragile_lock) * (1.0 - 0.72 * conservative_fatigue), 0.0, 1.0)
    rescue_param = jnp.clip((0.74 * stress_core + 1.12 * injury_rescue + 0.18 * low_alive) * standard_edge * (1.0 - 0.52 * stable_lock) * (1.0 - 0.42 * fragile_lock), 0.0, 1.0)
    conservative_gate = jnp.clip((1.00 * renewal_param + 0.20 * injury_rescue * conservative_edge + 0.22 * stable_world * clone_bad * conservative_edge + 0.10 * fragile_lock * conservative_edge + 0.36 * conservative_reseed) * (1.0 - 0.58 * stable_mutation_debt), 0.0, 1.0)
    standard_gate = jnp.clip(
        (0.30 * probe_world * (0.34 + 0.52 * utility[2]) + 0.66 * rescue_param + 0.24 * standard_rescue_credit + 0.40 * standard_tail_probe + 0.18 * injury_tail_window * (0.42 * unused[2] + 0.58 * standard_edge))
        * (1.0 - 0.54 * stable_lock)
        * (1.0 - 0.64 * fragile_lock)
        * (1.0 - 0.70 * stable_sham_guard * (1.0 - 0.78 * injury_tail_window)),
        0.0,
        1.0,
    )
    mixed_gate = jnp.clip(
        (0.14 * probe_world * (0.30 + 0.38 * utility[5]) + 0.12 * rescue_param * mixed_edge)
        * compact
        * (1.0 - 0.64 * stable_lock)
        * (1.0 - 0.58 * fragile_lock)
        * (1.0 - 0.68 * stable_sham_guard),
        0.0,
        1.0,
    )
    exploratory_gate = jnp.clip(
        probe_world * unused[3] * (0.10 + 0.20 * compact) * injury_rescue * (1.0 - 0.68 * stable_lock) * (1.0 - 0.70 * fragile_lock),
        0.0,
        1.0,
    )
    structural_gate = jnp.clip(
        probe_world * unused[4] * compact * 0.12 * injury_rescue * (1.0 - 0.68 * stable_lock) * (1.0 - 0.70 * fragile_lock),
        0.0,
        1.0,
    )
    anchor = jnp.array([3.32, -0.44, -2.96, -6.10, -6.42, -5.88], dtype=jnp.float32)
    stable_policy = jnp.array([1.06, 0.76, -1.72, -2.12, -2.26, -1.92], dtype=jnp.float32)
    renewal_policy = jnp.array([-0.86, 2.18, 0.08, -0.48, -0.52, -0.32], dtype=jnp.float32)
    stress_policy = jnp.array([-2.88, -0.12, 1.08, 0.34, 0.22, 0.52], dtype=jnp.float32)
    rescue_policy = jnp.array([-1.46, 0.22, 1.96, 0.34, 0.16, 0.74], dtype=jnp.float32)
    utility_policy = jnp.array([0.08, 1.16, 1.04, 0.32, 0.24, 0.42], dtype=jnp.float32)
    pulse = jnp.array(
        [
            -1.00 * standard_gate - 0.52 * mixed_gate - 1.20 * renewal_param - 1.14 * clone_bad * injury_rescue - 0.30 * stable_world * clone_bad + 0.38 * fragile_lock + 0.86 * stable_mutation_debt + 0.42 * stable_sham_guard - 0.24 * conservative_reseed - 0.42 * standard_tail_probe - 0.22 * injury_tail_window,
            2.46 * conservative_gate + 0.08 * injury_rescue + 0.20 * stable_lock * conservative_edge + 0.16 * fragile_lock - 1.14 * stable_mutation_debt - 0.52 * conservative_fatigue + 0.64 * conservative_reseed - 0.18 * stable_sham_guard,
            2.98 * standard_gate + 0.92 * rescue_param + 0.38 * standard_rescue_credit + 0.88 * standard_tail_probe + 0.34 * injury_tail_window - 0.64 * stable_lock - 0.46 * fragile_lock - 0.30 * stable_sham_guard,
            0.56 * exploratory_gate,
            0.36 * structural_gate,
            1.08 * mixed_gate,
        ],
        dtype=jnp.float32,
    )
    growth_trim = jnp.array(
        [0.16 * growth, 0.00 * growth, -0.34 * growth, -0.24 * growth, -0.20 * growth, 0.04 * growth],
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