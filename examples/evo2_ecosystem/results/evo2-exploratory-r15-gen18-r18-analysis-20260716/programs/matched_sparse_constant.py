import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome_summary,
    parent_stats,
    population_stats,
    operator_stats,
    rng,
):
    """Constant mixture matched to R15-gen18 development operator usage."""
    del parent_genome_summary, parent_stats, population_stats, operator_stats, rng
    probabilities = jnp.array(
        [
            0.9815712900096993,
            0.011639185257032008,
            0.004849660523763337,
            0.0019398642095053346,
            1.0e-8,
            1.0e-8,
        ],
        dtype=jnp.float32,
    )
    return jnp.log(probabilities)
# EVOLVE-BLOCK-END
