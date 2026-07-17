# System Instructions


Discover a better six-action heredity scheduler for an embodied CPPN ecosystem.

make_offspring receives bounded parent, population, and per-operator success,
usage, and evidence summaries. Return six finite logits for clone,
conservative parametric, standard parametric, exploratory parametric,
structural, and mixed mutation. The paired ancestor for this run is
Exact frozen clone. R15 generation 18 discovered an ecology-conditioned sparse heredity scheduler that beat clone on all three R18 development repeats but over-triggered mutation on independent R18 sealed founders and failed all three sealed repeats. R21 changes training diversity only: all sixteen exposed non-sealed R18 founders are now training. Discover normalized ecology and trusted-operator-evidence signals that reduce founder-specific over-triggering and transfer across founders. Keep a genuine non-clone ecology-conditioned mechanism; do not collapse to a constant mutation mixture. Do not use or infer founder identity, seed, scenario identity, event time, future state, or hidden partitions.. Maximize the paired candidate-minus-ancestor ecological
score across matched uninjured and persistent head-actuator-injury worlds; the first two of six hinge gains fall from 1.0 to 0.1 at the event. Use
operator evidence to adapt choices rather than merely returning the ancestor.

The array ABI is exact:
- parent_genome_summary = [node_fraction, connection_fraction]
- parent_stats = [energy_fraction, intake_ema, age_fraction]
- population_stats = [alive_fraction, mean_energy_fraction,
  population_change_ema, birth_rate_ema, death_rate_ema, mean_intake_ema]
- operator_stats rows are [success_ema, usage_ema, evidence_ema] and columns
  use the six-operator order above
- rng is opaque and must not be read

Index operator_stats exactly as:
  success = operator_stats[0]
  usage = operator_stats[1]
  evidence = operator_stats[2]
Each resulting vector has six entries. Never use operator_stats[:, 0],
operator_stats[:, 1], or operator_stats[:, 2]; those transpose the ABI.

Use only the approved jax.numpy expression grammar already demonstrated by the
parent program. Only pure bounded jax.numpy expressions are valid.

You MUST respond using an edit name, description, and the exact SEARCH/REPLACE diff format shown below to indicate changes:

<NAME>
A shortened name summarizing the edit you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
A description and argumentation process of the edit you are proposing.
</DESCRIPTION>

<DIFF>
<<<<<<< SEARCH
# Original code to find and replace (must match exactly including indentation)
=======
# New replacement code
>>>>>>> REPLACE

</DIFF>


Example of a valid diff format:
<DIFF>
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

</DIFF>

* You may only modify text that lies below a line containing "EVOLVE-BLOCK-START" and above the next "EVOLVE-BLOCK-END". Everything outside those markers is read-only.
* Do not repeat the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the SEARCH/REPLACE blocks.  
* Every block’s SEARCH section must be copied **verbatim** from the current file, including indentation.
* You can propose multiple independent edits. SEARCH/REPLACE blocks follow one after another. DO NOT ADD ANY OTHER TEXT BETWEEN THESE BLOCKS.
* Make sure the file still runs after your changes.

# Previous Messages

[]

# User Request

Here are the performance metrics of a set of previously implemented programs:

# Prior programs

```python
import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome_summary,
    parent_stats,
    population_stats,
    operator_stats,
    rng,
):
    """Trusted evidence mutation-budget heredity scheduler."""
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
    scarcity = jnp.clip(1.0 - alive_fraction, 0.0, 1.0)
    energy_deficit = jnp.clip((0.55 - mean_energy_fraction) * 1.45, 0.0, 1.0)
    intake_deficit = jnp.clip((0.50 - mean_intake_ema) * 1.55, 0.0, 1.0)
    birth_deficit = jnp.clip((0.24 - birth_rate_ema) * 1.80, 0.0, 1.0)
    parent_quality = jnp.clip(0.62 * energy_fraction + 0.38 * intake_ema, 0.0, 1.0)
    parent_lag = jnp.clip(mean_energy_fraction - energy_fraction + mean_intake_ema - intake_ema, 0.0, 1.0)
    maturity = jnp.clip(age_fraction, 0.0, 1.0)
    genome_load = jnp.clip(0.54 * node_fraction + 0.46 * connection_fraction, 0.0, 1.0)
    compact = jnp.clip(1.0 - genome_load, 0.0, 1.0)
    stress = jnp.clip(
        0.34 * decline
        + 0.26 * death
        + 0.18 * energy_deficit
        + 0.14 * intake_deficit
        + 0.08 * scarcity,
        0.0,
        1.0,
    )
    recovery_room = jnp.clip(
        0.40 * birth_deficit
        + 0.24 * compact
        + 0.20 * parent_quality
        + 0.16 * (1.0 - growth),
        0.0,
        1.0,
    )
    fragility = jnp.clip(
        0.38 * death
        + 0.24 * scarcity
        + 0.22 * energy_deficit
        + 0.16 * parent_lag,
        0.0,
        1.0,
    )
    mutation_budget = jnp.clip(
        0.10
        + 0.30 * stress * recovery_room
        + 0.16 * birth_deficit * parent_quality
        + 0.08 * compact * (1.0 - death)
        - 0.18 * fragility * (1.0 - parent_quality),
        0.02,
        0.46,
    )
    prior = jnp.array([0.22, 0.51, 0.48, 0.39, 0.34, 0.40], dtype=jnp.float32)
    trust = jnp.clip(evidence * (0.45 + 0.55 * (1.0 - usage)), 0.0, 1.0)
    posterior = jnp.clip(prior + trust * (success - prior), 0.0, 1.0)
    clone_anchor = posterior[0]
    relative_value = jnp.clip(posterior - clone_anchor + 0.50, 0.0, 1.0)
    underused = jnp.clip(1.0 - usage, 0.0, 1.0)
    uncertainty = jnp.clip((1.0 - evidence) * underused, 0.0, 1.0)
    role = jnp.array([0.00, 0.34, 0.46, 0.38, 0.30, 0.42], dtype=jnp.float32)
    stress_role = jnp.array([0.00, 0.08, 0.25, 0.34, 0.16, 0.28], dtype=jnp.float32)
    compact_role = jnp.array([0.00, 0.10, 0.05, 0.10, 0.26, 0.16], dtype=jnp.float32)
    fragile_penalty = jnp.array([0.00, -0.03, -0.11, -0.26, -0.30, -0.22], dtype=jnp.float32)
    role_score = jnp.clip(
        role
        + stress_role * stress
        + compact_role * compact
        + fragile_penalty * fragility
        + 0.10 * relative_value
        + 0.035 * uncertainty,
        0.0,
        1.0,
    )
    mutation_shape = mutation_budget * role_score
    clone_guard = jnp.clip(
        1.0
        - mutation_budget
        + 0.28 * fragility
        + 0.18 * (1.0 - parent_quality)
        + 0.10 * growth,
        0.34,
        1.0,
    )
    base = jnp.array([1.18, -1.58, -1.82, -2.72, -3.02, -2.64], dtype=jnp.float32)
    value_gain = jnp.array([0.35, 1.20, 1.26, 1.06, 0.96, 1.08], dtype=jnp.float32)
    budget_gain = jnp.array([-2.10, 3.05, 3.35, 3.65, 3.15, 3.35], dtype=jnp.float32)
    ecology_gain = jnp.array(
        [
            1.65 * clone_guard,
            0.58 * parent_quality + 0.20 * maturity,
            0.50 * stress + 0.24 * parent_quality,
            0.76 * stress + 0.20 * compact,
            0.46 * compact + 0.20 * stress,
            0.54 * stress + 0.26 * compact,
        ],
        dtype=jnp.float32,
    )
    logits = (
        base
        + value_gain * (posterior - 0.42)
        + budget_gain * mutation_shape
        + ecology_gain
        - 0.36 * usage
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: -0.02
score: -0.02; sham_auc_delta: 0.01; survival_rate: 0.84; mean_births: 43.12; mean_deaths: 28.16; mean_generation_gain: 4.41; operator_fraction: {'clone': 0.8442028985507246, 'parametric_conservative': 0.07318840579710145, 'parametric_standard': 0.04492753623188406, 'parametric_exploratory': 0.013043478260869565, 'structural': 0.007246376811594203, 'mixed': 0.017391304347826087}; pre_selection_probability: [0.885813666655954, 0.05049985752696485, 0.03178232258795637, 0.011123452220210987, 0.008152666097853036, 0.012628030691262894]; post_selection_probability: [0.8678773336332014, 0.059093286585146676, 0.036033632396760704, 0.012812769061448358, 0.00960575297202082, 0.01457723478530009]; operator_success_ema: [0.1870273679960519, 0.4350810730829835, 0.4681265139952302, 0.49344374518841505, 0.4937783945351839, 0.49204026255756617]; operator_usage_ema: [0.65460882242769, 0.05472841550363228, 0.027531878564332146, 0.010650380165316164, 0.007022048412181903, 0.018951713194837794]; operator_evidence_ema: [0.7719567231833935, 0.16400716360658407, 0.10670939972624183, 0.03062500711530447, 0.012500002980232239, 0.021562505047768354]; shock_auc_delta: -0.02; ancestor_survival_rate: 1.00

Text feedback:
paired ancestor delta=-0.0208; sham delta=0.0102; shock delta=-0.0199; survival=0.84; mean births=43.1; survival is scored, not a validity gate.


# Current program

Here is the current program we are trying to improve (you will need to propose a modification to it below):

```python
import jax.numpy as jnp


# EVOLVE-BLOCK-START
def make_offspring(
    parent_genome_summary,
    parent_stats,
    population_stats,
    operator_stats,
    rng,
):
    """Vector-regime sparse heredity scheduler."""
    success = operator_stats[0]
    usage = operator_stats[1]
    evidence = operator_stats[2]
    node_fraction = parent_genome_summary[0]
    connection_fraction = parent_genome_summary[1]
    energy_fraction = parent_stats[0]
    intake_ema = parent_stats[1]
    alive_fraction = population_stats[0]
    mean_energy_fraction = population_stats[1]
    population_change_ema = population_stats[2]
    birth_rate_ema = population_stats[3]
    death_rate_ema = population_stats[4]
    mean_intake_ema = population_stats[5]
    decline = jnp.clip(-population_change_ema, 0.0, 1.0)
    growth = jnp.clip(population_change_ema, 0.0, 1.0)
    death = jnp.clip(death_rate_ema, 0.0, 1.0)
    birth_gap = jnp.clip(0.28 - birth_rate_ema, 0.0, 1.0)
    energy_gap = jnp.clip(0.56 - mean_energy_fraction, 0.0, 1.0)
    intake_gap = jnp.clip(0.54 - mean_intake_ema, 0.0, 1.0)
    parent_quality = jnp.clip(0.58 * energy_fraction + 0.42 * intake_ema, 0.0, 1.0)
    compact = jnp.clip(1.0 - 0.55 * node_fraction - 0.45 * connection_fraction, 0.0, 1.0)
    calm = jnp.clip(
        alive_fraction
        * (1.0 - death)
        * (1.0 - decline)
        * (0.58 + 0.42 * mean_energy_fraction),
        0.0,
        1.0,
    )
    pressure = jnp.clip(
        0.78 * decline
        + 0.68 * death
        + 0.38 * energy_gap
        + 0.44 * intake_gap
        + 0.18 * birth_gap
        + 0.16 * (1.0 - alive_fraction),
        0.0,
        1.0,
    )
    rescue = jnp.clip(
        intake_gap * (0.52 * decline + 0.34 * death + 0.24 * energy_gap),
        0.0,
        1.0,
    )
    prior = jnp.array([0.22, 0.56, 0.50, 0.40, 0.37, 0.39], dtype=jnp.float32)
    learned = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    scarce = jnp.clip(1.0 - usage, 0.0, 1.0)
    evidence_gap = jnp.clip(1.0 - evidence, 0.0, 1.0)
    probe = evidence_gap * scarce
    op_value = 1.85 * learned + 0.38 * scarce + 0.14 * probe - 0.22 * usage
    base = jnp.array([4.34, -1.12, -2.04, -4.88, -5.58, -5.26], dtype=jnp.float32)
    calm_gain = jnp.array([2.12, 0.62, 0.22, -0.24, -0.20, -0.18], dtype=jnp.float32)
    pressure_gain = jnp.array([-2.34, 0.10, 1.18, 1.75, 1.42, 1.55], dtype=jnp.float32)
    rescue_gain = jnp.array([-0.62, 0.52, 1.06, 1.42, 0.82, 1.04], dtype=jnp.float32)
    quality_gain = jnp.array([0.00, 0.88, 0.44, 0.10, 0.00, 0.06], dtype=jnp.float32)
    shape_gain = jnp.array([0.00, 0.00, 0.00, 0.12, 0.68, 0.20], dtype=jnp.float32)
    growth_gain = jnp.array([0.08, 0.04, -0.10, -0.16, -0.10, 0.28], dtype=jnp.float32)
    value_gain = jnp.array([0.46, 0.78, 0.92, 0.70, 0.68, 0.68], dtype=jnp.float32)
    logits = (
        base
        + calm_gain * calm
        + pressure_gain * pressure
        + rescue_gain * rescue
        + quality_gain * parent_quality
        + shape_gain * compact
        + growth_gain * growth
        + value_gain * op_value
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Here are the performance metrics of the program:

Combined score to maximize: -0.01
score: -0.01; sham_auc_delta: 0.03; survival_rate: 0.97; mean_births: 47.53; mean_deaths: 29.12; mean_generation_gain: 4.75; operator_fraction: {'clone': 0.9644970414201184, 'parametric_conservative': 0.02564102564102564, 'parametric_standard': 0.009861932938856016, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9579785623048481, 0.029402373260573336, 0.011773677967619477, 0.00038254866682875316, 0.00021864835728417363, 0.0002441739927969154]; post_selection_probability: [0.9707659325300956, 0.021195806793390275, 0.007543118055463299, 0.00022304685250117696, 0.00012828753909549179, 0.00014383115280601098]; operator_success_ema: [0.15698173036798835, 0.4704886404797435, 0.48741505295038223, 0.5, 0.5, 0.5]; operator_usage_ema: [0.8236149279400706, 0.017205326883413363, 0.004452917535672896, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.8399839205667377, 0.07846876792609692, 0.03718750877305865, 0.0, 0.0, 0.0]; shock_auc_delta: -0.01; ancestor_survival_rate: 0.94

Here is additional text feedback about the current program:

paired ancestor delta=-0.0077; sham delta=0.0271; shock delta=-0.0085; survival=0.97; mean births=47.5; survival is scored, not a validity gate.


# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
