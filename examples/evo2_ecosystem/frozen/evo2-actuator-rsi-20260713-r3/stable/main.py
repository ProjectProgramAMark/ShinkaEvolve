import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(parent_genome, parent_stats, population_stats, rng):
    """Score clone, parametric, structural, and mixed mutation."""
    node_frac = jnp.clip(parent_genome[0], 0.0, 1.0)
    conn_frac = jnp.clip(parent_genome[1], 0.0, 1.0)
    energy = jnp.clip(parent_stats[0], 0.0, 1.0)
    intake = jnp.clip(parent_stats[1], 0.0, 1.0)
    alive = jnp.clip(population_stats[0], 0.0, 1.0)
    pop_delta = jnp.clip(population_stats[1], -1.0, 1.0)
    diversity = jnp.clip(population_stats[2], 0.0, 1.0)
    lineage = jnp.clip(population_stats[3], 0.0, 1.0)
    occupancy = 0.55 * node_frac + 0.45 * conn_frac
    vigor = 0.68 * energy + 0.32 * intake
    open_capacity = 1.0 - occupancy
    stable_field = 0.58 * alive + 0.24 * diversity + 0.18 * lineage
    expansion_gate = vigor * open_capacity * stable_field
    crowding = occupancy * alive * (1.0 - diversity)
    clone = -0.10 - 0.10 * vigor + 0.05 * (1.0 - alive)
    parametric = 0.72 + 0.18 * vigor * occupancy + 0.08 * crowding - 0.04 * pop_delta
    structural = 0.86 + 0.30 * expansion_gate - 0.05 * crowding
    mixed = 1.0 + 0.035 * vigor + 0.025 * alive - 0.045 * expansion_gate
    return jnp.clip(
        jnp.array([clone, parametric, structural, mixed], dtype=jnp.float32),
        -4.0,
        4.0,
    )


# EVOLVE-BLOCK-END