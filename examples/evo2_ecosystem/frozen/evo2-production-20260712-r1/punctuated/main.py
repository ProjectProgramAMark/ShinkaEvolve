import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(parent_genome, parent_stats, population_stats, rng):
    """Score clone, parametric, structural, and mixed mutation."""
    node_frac = jnp.clip(parent_genome[0], 0.0, 1.0)
    conn_frac = jnp.clip(parent_genome[1], 0.0, 1.0)
    energy = jnp.clip(parent_stats[0], 0.0, 1.0)
    intake = jnp.tanh(parent_stats[1])
    alive = jnp.clip(population_stats[0], 0.0, 1.0)
    pop_delta = jnp.tanh(population_stats[1])
    diversity = jnp.clip(population_stats[2], 0.0, 1.0)
    entropy = jnp.clip(population_stats[3], 0.0, 1.0)
    complexity = 0.55 * node_frac + 0.45 * conn_frac
    productivity = jnp.clip(0.68 * energy + 0.32 * (0.5 + 0.5 * intake), 0.0, 1.0)
    fragility = jnp.clip(0.55 * (1.0 - alive) + 0.45 * (0.5 - 0.5 * pop_delta), 0.0, 1.0)
    convergence = jnp.clip(0.55 * (1.0 - diversity) + 0.45 * (1.0 - entropy), 0.0, 1.0)
    headroom = 1.0 - complexity
    stability = jnp.clip(0.62 * alive + 0.38 * (0.5 + 0.5 * pop_delta), 0.0, 1.0)
    innovation = jnp.clip(convergence * (0.35 + 0.65 * headroom), 0.0, 1.0)
    recovery = jnp.clip(fragility * (1.0 - convergence), 0.0, 1.0)
    clone = (
        0.70 * recovery
        + 0.34 * productivity
        - 0.42 * innovation
        - 0.18 * headroom
    )
    parametric = (
        0.88 * productivity
        + 0.42 * fragility
        + 0.20 * complexity
        - 0.18 * innovation
    )
    structural = (
        0.96 * innovation * stability
        + 0.34 * productivity
        + 0.22 * headroom
        - 0.62 * fragility
        - 0.10 * complexity
    )
    mixed = (
        0.72 * innovation
        + 0.58 * productivity
        + 0.24 * headroom
        + 0.14 * stability
        - 0.38 * fragility
    )
    scores = jnp.array([clone, parametric, structural, mixed], dtype=jnp.float32)
    return jnp.clip(scores, -4.0, 4.0)
# EVOLVE-BLOCK-END