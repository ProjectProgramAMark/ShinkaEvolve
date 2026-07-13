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
    genome_load = 0.5 * (node_fraction + connection_fraction)
    compact_genome = 1.0 - genome_load
    parent_health = 0.65 * energy_fraction + 0.35 * intake_ema
    low_diversity = 1.0 - 0.5 * (action_diversity + lineage_entropy)
    recovery_gap = jnp.clip(-population_change_ema, 0.0, 1.0)
    expansion_room = 1.0 - alive_fraction
    rescue_pressure = jnp.clip(
        0.50 * recovery_gap + 0.35 * expansion_room + 0.25 * low_diversity,
        0.0,
        1.0,
    )
    structural_window = parent_health * compact_genome * rescue_pressure
    exploit_window = parent_health * (1.0 - 0.45 * rescue_pressure)
    clone = 0.08 + 0.45 * (1.0 - parent_health) + 0.12 * alive_fraction
    parametric = 0.55 + 1.05 * exploit_window + 0.25 * genome_load + 0.18 * action_diversity
    structural = 0.30 + 1.55 * structural_window + 0.20 * recovery_gap - 0.18 * genome_load
    mixed = 0.34 + 1.15 * parent_health * rescue_pressure + 0.18 * low_diversity - 0.10 * genome_load
    return jnp.array(
        [clone, parametric, structural, mixed],
        dtype=jnp.float32,
    )


# EVOLVE-BLOCK-END