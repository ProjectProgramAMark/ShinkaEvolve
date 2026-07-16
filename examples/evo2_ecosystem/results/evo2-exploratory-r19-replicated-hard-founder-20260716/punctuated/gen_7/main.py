import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome_summary,
    parent_stats,
    population_stats,
    operator_stats,
    rng,
):
    """Evidence-weighted stress bandit for six heredity actions."""
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
    energy_short = jnp.clip(0.54 - mean_energy_fraction, 0.0, 1.0)
    intake_short = jnp.clip(0.54 - mean_intake_ema, 0.0, 1.0)
    birth_short = jnp.clip(0.28 - birth_rate_ema, 0.0, 1.0)
    parent_vigor = jnp.clip(
        0.52 * energy_fraction + 0.38 * intake_ema + 0.10 * (1.0 - age_fraction),
        0.0,
        1.0,
    )
    compact = jnp.clip(1.0 - 0.50 * node_fraction - 0.50 * connection_fraction, 0.0, 1.0)
    stress = jnp.clip(
        0.54 * decline
        + 0.46 * death
        + 0.24 * low_alive
        + 0.30 * intake_short
        + 0.18 * energy_short
        + 0.12 * birth_short,
        0.0,
        1.0,
    )
    stable = jnp.clip(
        alive_fraction
        * (1.0 - decline)
        * (1.0 - death)
        * (0.42 + 0.58 * mean_energy_fraction)
        * (0.45 + 0.55 * mean_intake_ema),
        0.0,
        1.0,
    )
    rescue_context = jnp.clip(
        stress
        * (0.35 + 0.65 * parent_vigor)
        * (0.40 + 0.60 * compact)
        * (1.0 - 0.72 * stable),
        0.0,
        1.0,
    )
    quiet_context = jnp.clip(stable * (1.0 - stress) * (0.70 + 0.30 * growth), 0.0, 1.0)
    prior = jnp.array([0.32, 0.47, 0.38, 0.18, 0.15, 0.24], dtype=jnp.float32)
    learned = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    confidence = jnp.clip(0.10 + 0.90 * evidence, 0.0, 1.0)
    novelty = jnp.clip((1.0 - usage) * (1.0 - 0.45 * evidence), 0.0, 1.0)
    clone_score = learned[0]
    advantage = jnp.clip((learned - clone_score + 0.025) * confidence, -1.0, 1.0)
    conservative_ok = jnp.clip(advantage[1] + 0.16 * novelty[1] + 0.10 * (success[1] - success[0]), 0.0, 1.0)
    standard_ok = jnp.clip(advantage[2] + 0.10 * novelty[2] + 0.06 * stress, 0.0, 1.0)
    exploratory_ok = jnp.clip(advantage[3] + 0.06 * novelty[3], 0.0, 1.0)
    structural_ok = jnp.clip(advantage[4] + 0.05 * novelty[4], 0.0, 1.0)
    mixed_ok = jnp.clip(advantage[5] + 0.08 * novelty[5], 0.0, 1.0)
    clone_overused = jnp.clip(usage[0] * evidence[0] * (0.42 - success[0]), 0.0, 1.0)
    conservative_bad = jnp.clip(usage[1] * evidence[1] * (0.40 - success[1]), 0.0, 1.0)
    mutation_budget = jnp.clip(
        0.018
        + 0.105 * rescue_context
        + 0.055 * stress * clone_overused
        - 0.040 * quiet_context
        - 0.035 * conservative_bad * quiet_context,
        0.0,
        0.20,
    )
    conservative_share = jnp.clip(
        0.68
        + 0.22 * conservative_ok
        - 0.18 * stress
        - 0.20 * conservative_bad,
        0.05,
        0.92,
    )
    standard_share = jnp.clip(
        0.16
        + 0.62 * stress * standard_ok
        + 0.18 * clone_overused
        - 0.12 * quiet_context,
        0.02,
        0.78,
    )
    mixed_share = jnp.clip(0.06 + 0.28 * stress * mixed_ok * compact, 0.01, 0.34)
    exploratory_share = jnp.clip(0.018 * stress * exploratory_ok * compact, 0.0, 0.08)
    structural_share = jnp.clip(0.012 * stress * structural_ok * compact, 0.0, 0.06)
    shares = jnp.array(
        [
            0.0,
            conservative_share,
            standard_share,
            exploratory_share,
            structural_share,
            mixed_share,
        ],
        dtype=jnp.float32,
    )
    share_sum = jnp.clip(jnp.sum(shares[1:]), 0.001, 10.0)
    mutation_probs = mutation_budget * shares / share_sum
    clone_prob = jnp.clip(1.0 - jnp.sum(mutation_probs[1:]), 0.80, 0.9995)
    probs = jnp.array(
        [
            clone_prob,
            mutation_probs[1],
            mutation_probs[2],
            mutation_probs[3],
            mutation_probs[4],
            mutation_probs[5],
        ],
        dtype=jnp.float32,
    )
    probs = jnp.clip(probs, 0.00002, 0.99996)
    logits = jnp.log(probs) - jnp.log(probs[0])
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END