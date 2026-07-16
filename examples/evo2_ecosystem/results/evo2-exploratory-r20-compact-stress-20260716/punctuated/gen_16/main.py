import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome_summary,
    parent_stats,
    population_stats,
    operator_stats,
    rng,
):
    """Tiered ecological rescue scheduler for six heredity operators."""
    success = operator_stats[0]
    usage = operator_stats[1]
    evidence = operator_stats[2]
    node_frac = jnp.clip(parent_genome_summary[0], 0.0, 1.0)
    conn_frac = jnp.clip(parent_genome_summary[1], 0.0, 1.0)
    parent_energy = jnp.clip(parent_stats[0], 0.0, 1.0)
    parent_intake = jnp.clip(parent_stats[1], 0.0, 1.0)
    parent_age = jnp.clip(parent_stats[2], 0.0, 1.0)
    alive = jnp.clip(population_stats[0], 0.0, 1.0)
    mean_energy = jnp.clip(population_stats[1], 0.0, 1.0)
    pop_change = jnp.clip(population_stats[2], -1.0, 1.0)
    births = jnp.clip(population_stats[3], 0.0, 1.0)
    deaths = jnp.clip(population_stats[4], 0.0, 1.0)
    mean_intake = jnp.clip(population_stats[5], 0.0, 1.0)
    compact = jnp.clip(1.0 - 0.58 * node_frac - 0.42 * conn_frac, 0.0, 1.0)
    complexity = jnp.clip(0.58 * node_frac + 0.42 * conn_frac, 0.0, 1.0)
    parent_ready = jnp.clip(
        0.50 * parent_energy
        + 0.30 * parent_intake
        + 0.20 * parent_age,
        0.0,
        1.0,
    )
    colony_ready = jnp.clip(
        0.55 * mean_energy
        + 0.35 * mean_intake
        + 0.10 * alive,
        0.0,
        1.0,
    )
    ready = jnp.clip(parent_ready * (0.35 + 0.65 * colony_ready), 0.0, 1.0)
    contraction = jnp.clip(-pop_change, 0.0, 1.0)
    loss_signal = jnp.clip(
        0.48 * contraction
        + 0.34 * deaths
        + 0.18 * jnp.clip(0.82 - alive, 0.0, 1.0),
        0.0,
        1.0,
    )
    starvation = jnp.clip(
        0.55 * jnp.clip(0.48 - mean_energy, 0.0, 1.0)
        + 0.45 * jnp.clip(0.42 - mean_intake, 0.0, 1.0),
        0.0,
        1.0,
    )
    reproduction_gap = jnp.clip(
        (0.16 - births) * (0.45 + 0.55 * alive),
        0.0,
        1.0,
    )
    decline = jnp.clip(
        1.15 * loss_signal
        + 0.75 * starvation
        + 0.95 * reproduction_gap,
        0.0,
        1.0,
    )
    yellow = jnp.clip((decline - 0.10) * 2.8, 0.0, 1.0)
    red = jnp.clip((decline - 0.30) * 2.6, 0.0, 1.0)
    black = jnp.clip((decline - 0.55) * 2.2, 0.0, 1.0)
    clone_quality = jnp.clip(success[0] * evidence[0], 0.0, 1.0)
    clone_overstay = jnp.clip(
        usage[0] * evidence[0] * jnp.clip(0.46 - success[0], 0.0, 1.0),
        0.0,
        1.0,
    )
    novelty = jnp.clip(1.0 - evidence, 0.0, 1.0)
    proven = jnp.clip(evidence * success, 0.0, 1.0)
    anti_crowd = jnp.clip(1.0 - usage, 0.0, 1.0)
    relative = jnp.clip(success - success[0] + 0.30, 0.0, 1.0)
    credit = jnp.clip(
        0.42 * proven
        + 0.28 * novelty
        + 0.22 * relative
        + 0.18 * anti_crowd
        - 0.20 * clone_quality,
        0.0,
        1.0,
    )
    standard_rescue = jnp.clip(
        ready
        * (0.55 * yellow + 0.45 * red)
        * (0.50 + 0.50 * compact)
        * (0.45 + 0.55 * credit[2])
        * (0.70 + 0.30 * clone_overstay),
        0.0,
        1.0,
    )
    conservative_rescue = jnp.clip(
        ready
        * yellow
        * (0.35 + 0.65 * starvation)
        * (0.30 + 0.70 * credit[1])
        * (0.35 + 0.65 * anti_crowd[1]),
        0.0,
        1.0,
    )
    exploratory_rescue = jnp.clip(
        ready
        * red
        * compact
        * (0.25 + 0.75 * credit[3])
        * (0.50 + 0.50 * novelty[3]),
        0.0,
        1.0,
    )
    structural_rescue = jnp.clip(
        ready
        * black
        * compact
        * (0.20 + 0.80 * credit[4])
        * (0.45 + 0.55 * novelty[4]),
        0.0,
        1.0,
    )
    mixed_rescue = jnp.clip(
        ready
        * red
        * jnp.clip(0.70 * compact + 0.30 * complexity, 0.0, 1.0)
        * (0.25 + 0.75 * credit[5])
        * (0.45 + 0.55 * novelty[5]),
        0.0,
        1.0,
    )
    total_mutation = jnp.clip(
        0.40 * conservative_rescue
        + 0.95 * standard_rescue
        + 0.30 * exploratory_rescue
        + 0.25 * structural_rescue
        + 0.32 * mixed_rescue,
        0.0,
        1.0,
    )
    stable_clone = jnp.clip(
        (1.0 - yellow)
        * (0.55 + 0.45 * colony_ready)
        + 0.25 * clone_quality
        - 0.30 * clone_overstay,
        0.0,
        1.0,
    )
    logits = jnp.array(
        [
            4.35 + 2.10 * stable_clone - 6.60 * total_mutation - 1.80 * red,
            -4.70 + 5.10 * conservative_rescue + 1.15 * credit[1],
            -4.35 + 8.90 * standard_rescue + 1.55 * credit[2] + 2.20 * red,
            -6.10 + 5.20 * exploratory_rescue + 0.95 * credit[3],
            -6.45 + 5.00 * structural_rescue + 0.80 * credit[4],
            -6.00 + 5.60 * mixed_rescue + 0.95 * credit[5],
        ],
        dtype=jnp.float32,
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END