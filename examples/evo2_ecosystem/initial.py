import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(parent_genome, parent_stats, population_stats, rng):
    """Score clone, parametric, structural, and mixed mutation."""
    return jnp.array(
        [0.0, 0.0, 0.0, 1.0],
        dtype=jnp.float32,
    )


# EVOLVE-BLOCK-END
