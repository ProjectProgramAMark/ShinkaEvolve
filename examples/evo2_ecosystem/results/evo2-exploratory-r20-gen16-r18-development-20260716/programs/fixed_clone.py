import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome_summary,
    parent_stats,
    population_stats,
    operator_stats,
    rng,
): 
    """Always select the frozen clone operator."""
    return jnp.array([8.0, -8.0, -8.0, -8.0, -8.0, -8.0], dtype=jnp.float32)


# EVOLVE-BLOCK-END
