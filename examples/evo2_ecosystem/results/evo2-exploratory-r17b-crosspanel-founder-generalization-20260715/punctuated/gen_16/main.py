import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome_summary,
    parent_stats,
    population_stats,
    operator_stats,
    rng,
):
    """Regime-mixture heredity scheduler for six bounded action logits."""
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
    energy_gap = jnp.clip(0.57 - mean_energy_fraction, 0.0, 1.0)
    intake_gap = jnp.clip(0.55 - mean_intake_ema, 0.0, 1.0)
    birth_gap = jnp.clip(0.27 - birth_rate_ema, 0.0, 1.0)
    parent_vigor = jnp.clip(
        0.50 * energy_fraction + 0.38 * intake_ema + 0.12 * (1.0 - age_fraction),
        0.0,
        1.0,
    )
    compact = jnp.clip(1.0 - 0.48 * node_fraction - 0.52 * connection_fraction, 0.0, 1.0)
    stress = jnp.clip(
        0.46 * decline
        + 0.36 * death
        + 0.20 * low_alive
        + 0.22 * intake_gap
        + 0.17 * energy_gap
        + 0.10 * birth_gap,
        0.0,
        1.0,
    )
    stable = jnp.clip(
        alive_fraction
        * (1.0 - decline)
        * (1.0 - death)
        * (0.50 + 0.50 * mean_energy_fraction)
        * (0.50 + 0.50 * mean_intake_ema)
        * (0.75 + 0.25 * growth),
        0.0,
        1.0,
    )
    repair = jnp.clip(
        stress
        * (0.40 + 0.60 * parent_vigor)
        * (0.50 + 0.50 * decline + 0.30 * death)
        * (1.0 - 0.62 * stable),
        0.0,
        1.0,
    )
    fragile = jnp.clip(
        (0.42 * low_alive + 0.34 * energy_gap + 0.34 * intake_gap + 0.22 * death)
        * (1.0 - 0.62 * parent_vigor)
        + 0.20 * birth_gap,
        0.0,
        1.0,
    )
    prior = jnp.array([0.24, 0.54, 0.50, 0.30, 0.27, 0.38], dtype=jnp.float32)
    risk = jnp.array([0.00, 0.14, 0.36, 0.74, 0.86, 0.58], dtype=jnp.float32)
    posterior = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    uncertainty = jnp.clip((1.0 - evidence) * (1.0 - 0.30 * usage), 0.0, 1.0)
    trust = jnp.clip(posterior - 0.16 * uncertainty - 0.10 * usage * risk, 0.0, 1.0)
    clone_weak = jnp.clip(evidence[0] * (0.40 - success[0]) * (0.45 + 0.55 * usage[0]), 0.0, 1.0)
    cons_good = jnp.clip(trust[1] - trust[0] + 0.16 * clone_weak, 0.0, 1.0)
    std_good = jnp.clip(trust[2] - trust[0] + 0.22 * clone_weak, 0.0, 1.0)
    mix_good = jnp.clip(trust[5] - trust[0] + 0.10 * clone_weak, 0.0, 1.0)
    std_bad = jnp.clip(evidence[2] * (0.47 - success[2]) + 0.14 * usage[2], 0.0, 1.0)
    high_conf_repair = jnp.clip(
        repair
        * (0.50 + 0.50 * clone_weak)
        * (0.34 + 0.66 * std_good)
        * (1.0 - 0.70 * std_bad)
        * (1.0 - 0.68 * fragile),
        0.0,
        1.0,
    )
    quiet_guard = jnp.clip(stable * (1.0 - repair) + 0.80 * fragile, 0.0, 1.0)
    conservative_lane = jnp.clip(
        (0.18 + 0.52 * stable + 0.36 * repair)
        * (0.30 + 0.70 * cons_good)
        * (1.0 - 0.48 * fragile),
        0.0,
        1.0,
    )
    standard_lane = jnp.clip(
        high_conf_repair
        * (0.45 + 0.55 * parent_vigor)
        * (1.0 - 0.86 * stable * (1.0 - repair)),
        0.0,
        1.0,
    )
    mixed_lane = jnp.clip(
        repair
        * compact
        * (0.22 + 0.78 * mix_good)
        * (1.0 - 0.62 * fragile)
        * (1.0 - 0.58 * stable),
        0.0,
        1.0,
    )
    probe_lane = jnp.clip(
        0.055
        * repair
        * compact
        * (1.0 - quiet_guard)
        * (0.35 + 0.65 * uncertainty[3]),
        0.0,
        1.0,
    )
    base = jnp.array([3.15, -0.95, -2.65, -6.55, -6.70, -6.35], dtype=jnp.float32)
    regime_push = jnp.array(
        [
            1.65 * quiet_guard + 0.42 * growth - 1.18 * high_conf_repair - 0.28 * clone_weak,
            3.05 * conservative_lane + 0.28 * stable * cons_good,
            4.10 * standard_lane + 0.62 * high_conf_repair,
            0.90 * probe_lane * trust[3],
            0.68 * probe_lane * compact * trust[4],
            2.25 * mixed_lane,
        ],
        dtype=jnp.float32,
    )
    evidence_push = jnp.array(
        [
            -0.20 * cons_good - 0.42 * std_good - 0.16 * mix_good,
            1.90 * cons_good,
            2.20 * std_good,
            0.42 * uncertainty[3] * trust[3],
            0.30 * uncertainty[4] * trust[4],
            1.05 * mix_good,
        ],
        dtype=jnp.float32,
    )
    brake = jnp.array(
        [
            0.0,
            -0.12 * fragile,
            -1.55 * stable * (1.0 - repair) - 0.64 * fragile - 0.70 * std_bad,
            -1.35 * stable - 1.00 * fragile,
            -1.48 * stable - 1.12 * fragile,
            -0.86 * stable * (1.0 - repair) - 0.72 * fragile,
        ],
        dtype=jnp.float32,
    )
    logits = base + regime_push + evidence_push + brake
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END