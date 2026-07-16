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
    reproduction_ready = jnp.clip(
        parent_vigor
        * (0.46 + 0.54 * energy_fraction)
        * (0.40 + 0.60 * intake_ema)
        * (0.55 + 0.45 * birth_rate_ema),
        0.0,
        1.0,
    )
    stress_opportunity = jnp.clip(injury_rescue * (0.40 + 0.60 * stress_core) * (0.35 + 0.65 * reproduction_ready), 0.0, 1.0)
    prior = jnp.array([0.42, 0.44, 0.38, 0.14, 0.12, 0.20], dtype=jnp.float32)
    learned = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    credible = jnp.clip(0.12 + 0.88 * evidence, 0.0, 1.0)
    evidence_gate = jnp.clip(0.18 + 0.82 * evidence, 0.0, 1.0)
    unused = jnp.clip((1.0 - usage) * (1.0 - 0.58 * evidence), 0.0, 1.0)
    utility = jnp.clip((1.30 * learned + 0.16 * unused - 0.30 * usage) * credible, 0.0, 2.0)
    evidence_margin = jnp.clip((success - success[0] + 0.06) * evidence, 0.0, 1.0)
    clone_bad = jnp.clip(evidence[0] * (0.40 - success[0]) + 0.16 * usage[0] * evidence[0] * (learned[1] - learned[0]), 0.0, 1.0)
    conservative_edge = jnp.clip((learned[1] - learned[0]) * (0.22 + 0.78 * evidence[1]) + 0.42 * clone_bad + 0.34 * evidence_margin[1], 0.0, 1.0)
    standard_edge = jnp.clip((learned[2] - learned[0]) * (0.18 + 0.82 * evidence[2]) + 0.22 * clone_bad + 0.58 * evidence_margin[2] + 0.18 * unused[2] * stress_opportunity, 0.0, 1.0)
    mixed_edge = jnp.clip((learned[5] - learned[0]) * (0.16 + 0.84 * evidence[5]) + 0.12 * clone_bad + 0.24 * evidence_margin[5], 0.0, 1.0)
    conservative_fatigue = jnp.clip(evidence[1] * usage[1] * (0.42 - success[1]), 0.0, 1.0)
    stable_mutation_debt = jnp.clip(stable_world * (1.0 - injury_rescue) * (0.34 + 0.66 * conservative_fatigue), 0.0, 1.0)
    stress_probe_credit = jnp.clip(stress_opportunity * unused[2] * (1.0 - 0.72 * stable_world) * (0.46 + 0.54 * clone_bad), 0.0, 1.0)
    standard_rescue_credit = jnp.clip(stress_opportunity * (0.24 * unused[2] + 0.76 * standard_edge) * (0.42 + 0.58 * evidence_gate[2]), 0.0, 1.0)
    stable_lock = jnp.clip(stable_world * (0.80 + 0.18 * growth) * (1.0 - 0.94 * injury_rescue) * (1.0 - 0.26 * clone_bad), 0.0, 1.0)
    fragile_lock = jnp.clip(stable_world * (1.0 - parent_vigor) * (0.48 + 0.52 * (1.0 - compact)), 0.0, 1.0)
    stable_escape = jnp.clip(stable_world * reproduction_ready * clone_bad * conservative_edge * evidence_gate[1], 0.0, 1.0)
    renewal_param = jnp.clip(renewal_world * conservative_edge * reproduction_ready * (0.16 + 0.84 * clone_bad) * (1.0 - 0.78 * injury_rescue) * (1.0 - 0.42 * fragile_lock) * (1.0 - 0.78 * conservative_fatigue), 0.0, 1.0)
    rescue_param = jnp.clip((0.50 * stress_core + 1.10 * injury_rescue + 0.12 * low_alive) * standard_edge * reproduction_ready * (1.0 - 0.58 * stable_lock) * (1.0 - 0.46 * fragile_lock), 0.0, 1.0)
    conservative_gate = jnp.clip((0.42 * renewal_param + 0.08 * injury_rescue * conservative_edge + 0.30 * stable_escape + 0.06 * fragile_lock * conservative_edge) * (1.0 - 0.66 * stable_mutation_debt), 0.0, 1.0)
    standard_gate = jnp.clip(
        (0.22 * probe_world * (0.30 + 0.48 * utility[2]) + 0.72 * rescue_param + 0.30 * standard_rescue_credit + 0.34 * stress_probe_credit)
        * (0.35 + 0.65 * reproduction_ready)
        * (1.0 - 0.64 * stable_lock)
        * (1.0 - 0.68 * fragile_lock),
        0.0,
        1.0,
    )
    mixed_gate = jnp.clip(
        (0.10 * probe_world * (0.22 + 0.32 * utility[5]) + 0.10 * rescue_param * mixed_edge)
        * compact
        * reproduction_ready
        * evidence_gate[5]
        * (1.0 - 0.70 * stable_lock)
        * (1.0 - 0.62 * fragile_lock),
        0.0,
        1.0,
    )
    exploratory_gate = jnp.clip(
        probe_world * unused[3] * (0.06 + 0.16 * compact) * injury_rescue * reproduction_ready * evidence_gate[3] * (1.0 - 0.72 * stable_lock) * (1.0 - 0.72 * fragile_lock),
        0.0,
        1.0,
    )
    structural_gate = jnp.clip(
        probe_world * unused[4] * compact * 0.08 * injury_rescue * reproduction_ready * evidence_gate[4] * (1.0 - 0.72 * stable_lock) * (1.0 - 0.72 * fragile_lock),
        0.0,
        1.0,
    )
    anchor = jnp.array([4.10, -1.18, -3.35, -6.40, -6.65, -6.25], dtype=jnp.float32)
    stable_policy = jnp.array([1.22, 0.16, -2.10, -2.35, -2.45, -2.20], dtype=jnp.float32)
    renewal_policy = jnp.array([-0.18, 0.92, -0.10, -0.54, -0.58, -0.42], dtype=jnp.float32)
    stress_policy = jnp.array([-3.16, -0.22, 1.20, 0.30, 0.16, 0.46], dtype=jnp.float32)
    rescue_policy = jnp.array([-1.78, 0.04, 2.34, 0.24, 0.08, 0.52], dtype=jnp.float32)
    utility_policy = jnp.array([0.08, 0.72, 1.12, 0.24, 0.18, 0.32], dtype=jnp.float32)
    pulse = jnp.array(
        [
            -1.18 * standard_gate - 0.46 * mixed_gate - 0.42 * renewal_param - 1.18 * clone_bad * injury_rescue - 0.20 * stable_world * clone_bad - 0.66 * stress_probe_credit + 0.40 * fragile_lock + 0.92 * stable_mutation_debt,
            1.64 * conservative_gate + 0.04 * injury_rescue + 0.10 * stable_escape + 0.16 * fragile_lock - 1.22 * stable_mutation_debt - 0.52 * conservative_fatigue,
            3.18 * standard_gate + 1.04 * rescue_param + 0.44 * standard_rescue_credit + 0.92 * stress_probe_credit - 0.68 * stable_lock - 0.52 * fragile_lock,
            0.38 * exploratory_gate,
            0.24 * structural_gate,
            0.74 * mixed_gate,
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