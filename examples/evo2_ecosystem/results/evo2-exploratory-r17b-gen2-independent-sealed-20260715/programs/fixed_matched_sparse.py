import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome_summary,
    parent_stats,
    population_stats,
    operator_stats,
    rng,
):
    """Use the finalist's development-set operator mixture without conditioning."""
    return jnp.array(
        [8.0, 6.6828525, 3.0278746, -8.0, -8.0, -8.0],
        dtype=jnp.float32,
    )
# EVOLVE-BLOCK-END
