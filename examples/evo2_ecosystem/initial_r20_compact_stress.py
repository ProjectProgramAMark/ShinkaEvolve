import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome_summary,
    parent_stats,
    population_stats,
    operator_stats,
    rng,
):
    """Compact stress-gated probe for the six trusted heredity operators."""
    success = operator_stats[0]
    evidence = operator_stats[2]
    parent_ready = jnp.clip(
        0.55 * parent_stats[0] + 0.45 * parent_stats[1], 0.0, 1.0
    )
    decline = jnp.clip(-population_stats[2], 0.0, 1.0)
    death = jnp.clip(population_stats[4], 0.0, 1.0)
    low_alive = jnp.clip(1.0 - population_stats[0], 0.0, 1.0)
    low_energy = jnp.clip(0.55 - population_stats[1], 0.0, 1.0)
    low_intake = jnp.clip(0.55 - population_stats[5], 0.0, 1.0)
    stress = jnp.clip(
        1.8 * decline
        + 1.5 * death
        + 0.6 * low_alive
        + 0.5 * low_energy
        + 0.7 * low_intake,
        0.0,
        1.0,
    )
    standard_edge = jnp.clip(
        (success[2] - success[0] + 0.05) * evidence[2], 0.0, 1.0
    )
    rescue = jnp.clip(
        stress * parent_ready * (0.25 + 0.45 * (1.0 - evidence[2]) + 0.30 * standard_edge),
        0.0,
        1.0,
    )
    base = jnp.array([6.0, -3.0, -6.0, -7.0, -7.0, -7.0], dtype=jnp.float32)
    rescue_direction = jnp.array(
        [-8.0, 0.5, 10.0, 0.0, 0.0, 0.0], dtype=jnp.float32
    )
    return jnp.clip(base + rescue * rescue_direction, -7.0, 7.0)
# EVOLVE-BLOCK-END
