import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome_summary,
    parent_stats,
    population_stats,
    operator_stats,
    rng,
):
    """Risk-budgeted lower-confidence heredity scheduler for six actions."""
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
    low_alive = jnp.clip(1.0 - alive_fraction, 0.0, 1.0)
    death = jnp.clip(death_rate_ema, 0.0, 1.0)
    energy_gap = jnp.clip(0.56 - mean_energy_fraction, 0.0, 1.0)
    intake_gap = jnp.clip(0.54 - mean_intake_ema, 0.0, 1.0)
    birth_gap = jnp.clip(0.28 - birth_rate_ema, 0.0, 1.0)
    parent_vigor = jnp.clip(
        0.52 * energy_fraction + 0.36 * intake_ema + 0.12 * (1.0 - age_fraction),
        0.0,
        1.0,
    )
    compact = jnp.clip(1.0 - 0.50 * node_fraction - 0.50 * connection_fraction, 0.0, 1.0)
    stress = jnp.clip(
        0.52 * decline
        + 0.42 * death
        + 0.26 * low_alive
        + 0.24 * intake_gap
        + 0.18 * energy_gap
        + 0.10 * birth_gap,
        0.0,
        1.0,
    )
    injury_signal = jnp.clip(
        (0.62 * decline + 0.48 * death + 0.22 * low_alive)
        * (0.30 + 0.46 * intake_gap + 0.24 * energy_gap),
        0.0,
        1.0,
    )
    stable_signal = jnp.clip(
        alive_fraction
        * (1.0 - decline)
        * (1.0 - death)
        * (0.42 + 0.58 * mean_energy_fraction)
        * (0.48 + 0.52 * mean_intake_ema),
        0.0,
        1.0,
    )
    fragile_signal = jnp.clip(
        (0.46 * low_alive + 0.34 * energy_gap + 0.30 * intake_gap)
        * (1.0 - parent_vigor)
        + 0.28 * death,
        0.0,
        1.0,
    )
    prior = jnp.array([0.19, 0.55, 0.48, 0.31, 0.28, 0.36], dtype=jnp.float32)
    risk = jnp.array([0.00, 0.16, 0.39, 0.72, 0.82, 0.55], dtype=jnp.float32)
    posterior = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    uncertainty = jnp.clip((1.0 - evidence) * (1.0 - 0.35 * usage), 0.0, 1.0)
    lower_value = jnp.clip(posterior - 0.18 * uncertainty - 0.12 * usage * risk, 0.0, 1.0)
    clone_floor = jnp.clip(0.20 + 0.55 * evidence[0] * success[0], 0.0, 1.0)
    conservative_edge = jnp.clip(lower_value[1] - clone_floor + 0.22 * evidence[0] * (0.35 - success[0]), 0.0, 1.0)
    standard_edge = jnp.clip(lower_value[2] - clone_floor + 0.16 * evidence[0] * (0.35 - success[0]), 0.0, 1.0)
    mixed_edge = jnp.clip(lower_value[5] - clone_floor + 0.10 * evidence[0] * (0.35 - success[0]), 0.0, 1.0)
    rescue_window = jnp.clip(
        injury_signal
        * (0.36 + 0.64 * parent_vigor)
        * (0.42 + 0.58 * stress)
        * (1.0 - 0.62 * fragile_signal)
        * (1.0 - 0.74 * stable_signal * (1.0 - injury_signal)),
        0.0,
        1.0,
    )
    stable_caution = jnp.clip(stable_signal * (1.0 - injury_signal) * (0.58 + 0.42 * growth), 0.0, 1.0)
    mutation_budget = jnp.clip(
        0.035
        + 0.24 * injury_signal
        + 0.30 * rescue_window
        + 0.12 * stress * parent_vigor
        + 0.08 * conservative_edge
        - 0.40 * stable_caution
        - 0.38 * fragile_signal,
        0.0,
        0.66,
    )
    standard_budget = jnp.clip(
        mutation_budget
        * (0.16 + 0.52 * injury_signal + 0.58 * rescue_window)
        * (0.28 + 0.72 * standard_edge)
        * (1.0 - 0.84 * stable_caution)
        * (0.46 + 0.54 * parent_vigor),
        0.0,
        1.0,
    )
    conservative_budget = jnp.clip(
        mutation_budget
        * (0.48 + 0.36 * stable_signal + 0.28 * rescue_window)
        * (0.42 + 0.58 * conservative_edge)
        * (1.0 - 0.36 * injury_signal)
        * (1.0 - 0.34 * stable_caution),
        0.0,
        1.0,
    )
    mixed_budget = jnp.clip(
        mutation_budget
        * injury_signal
        * compact
        * (0.24 + 0.76 * mixed_edge)
        * (1.0 - 0.55 * fragile_signal),
        0.0,
        1.0,
    )
    rare_probe = jnp.clip(
        mutation_budget
        * injury_signal
        * compact
        * (1.0 - stable_signal)
        * (1.0 - fragile_signal)
        * 0.10,
        0.0,
        1.0,
    )
    clone_guard = jnp.clip(
        1.12
        + 2.35 * stable_caution
        + 1.70 * fragile_signal
        + 0.36 * growth
        - 1.38 * mutation_budget
        - 0.82 * rescue_window * standard_edge
        - 0.54 * evidence[0] * (0.35 - success[0]),
        -2.0,
        4.0,
    )
    base = jnp.array([2.95, -0.82, -2.25, -6.30, -6.45, -6.10], dtype=jnp.float32)
    value_push = jnp.array(
        [
            -0.30 * conservative_edge - 0.46 * standard_edge - 0.18 * mixed_edge,
            2.55 * conservative_edge,
            2.25 * standard_edge,
            0.58 * lower_value[3] * uncertainty[3],
            0.42 * lower_value[4] * uncertainty[4],
            1.15 * mixed_edge,
        ],
        dtype=jnp.float32,
    )
    budget_push = jnp.array(
        [
            clone_guard,
            3.15 * conservative_budget + 0.25 * stable_signal * conservative_edge,
            3.65 * standard_budget + 0.72 * rescue_window * standard_edge,
            1.10 * rare_probe,
            0.82 * rare_probe * compact,
            2.05 * mixed_budget,
        ],
        dtype=jnp.float32,
    )
    risk_brake = jnp.array(
        [
            0.0,
            -0.10 * fragile_signal,
            -1.28 * stable_caution - 0.50 * fragile_signal,
            -1.25 * stable_signal - 0.92 * fragile_signal,
            -1.35 * stable_signal - 1.05 * fragile_signal,
            -0.76 * stable_signal * (1.0 - injury_signal) - 0.64 * fragile_signal,
        ],
        dtype=jnp.float32,
    )
    logits = base + value_push + budget_push + risk_brake
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END