import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(parent_genome, parent_stats, population_stats, rng):
    """Score clone, parametric, structural, and mixed mutation."""
    node_fraction = jnp.clip(parent_genome[0], 0.0, 1.0)
    connection_fraction = jnp.clip(parent_genome[1], 0.0, 1.0)
    energy_fraction = jnp.clip(parent_stats[0], 0.0, 1.0)
    intake_ema = jnp.clip(parent_stats[1], 0.0, 1.0)
    alive_fraction = jnp.clip(population_stats[0], 0.0, 1.0)
    population_change_ema = jnp.clip(population_stats[1], -1.0, 1.0)
    action_diversity = jnp.clip(population_stats[2], 0.0, 1.0)
    lineage_entropy = jnp.clip(population_stats[3], 0.0, 1.0)
    productivity = 0.68 * energy_fraction + 0.32 * intake_ema
    compactness = 1.0 - (0.60 * node_fraction + 0.40 * connection_fraction)
    stagnation = jnp.clip(-population_change_ema, 0.0, 1.0)
    diversity_gap = 1.0 - (0.60 * action_diversity + 0.40 * lineage_entropy)
    crowding = alive_fraction * (1.0 - stagnation)
    sparse_parent = compactness * (1.0 - connection_fraction)
    rescue_pressure = stagnation * diversity_gap
    rescue_gate = rescue_pressure * (1.0 - crowding)
    builder_fit = productivity * sparse_parent
    clone = 0.05 + 0.90 * productivity + 0.50 * crowding - 0.85 * diversity_gap - 0.35 * stagnation
    parametric = 1.10 + 1.38 * productivity + 0.50 * crowding + 0.28 * connection_fraction - 0.28 * rescue_gate
    structural = 0.00 + 0.58 * productivity + 1.20 * sparse_parent + 1.45 * rescue_gate + 0.35 * builder_fit - 0.42 * crowding
    mixed = 0.08 + 0.78 * productivity + 0.70 * sparse_parent + 1.05 * rescue_gate + 0.25 * builder_fit - 0.45 * crowding
    return jnp.clip(
        jnp.array([clone, parametric, structural, mixed], dtype=jnp.float32),
        -4.0,
        4.0,
    )


# EVOLVE-BLOCK-END