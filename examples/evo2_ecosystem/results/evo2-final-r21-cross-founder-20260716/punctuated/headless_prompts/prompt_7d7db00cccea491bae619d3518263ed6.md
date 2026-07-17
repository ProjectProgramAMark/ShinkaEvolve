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

Performance metrics:
Combined score to maximize: -0.01
score: -0.01; sham_auc_delta: 0.03; survival_rate: 0.97; mean_births: 47.53; mean_deaths: 29.12; mean_generation_gain: 4.75; operator_fraction: {'clone': 0.9644970414201184, 'parametric_conservative': 0.02564102564102564, 'parametric_standard': 0.009861932938856016, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9579785623048481, 0.029402373260573336, 0.011773677967619477, 0.00038254866682875316, 0.00021864835728417363, 0.0002441739927969154]; post_selection_probability: [0.9707659325300956, 0.021195806793390275, 0.007543118055463299, 0.00022304685250117696, 0.00012828753909549179, 0.00014383115280601098]; operator_success_ema: [0.15698173036798835, 0.4704886404797435, 0.48741505295038223, 0.5, 0.5, 0.5]; operator_usage_ema: [0.8236149279400706, 0.017205326883413363, 0.004452917535672896, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.8399839205667377, 0.07846876792609692, 0.03718750877305865, 0.0, 0.0, 0.0]; shock_auc_delta: -0.01; ancestor_survival_rate: 0.94

Text feedback:
paired ancestor delta=-0.0077; sham delta=0.0271; shock delta=-0.0085; survival=0.97; mean births=47.5; survival is scored, not a validity gate.

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
    stress_raw = jnp.clip(
        0.58 * decline
        + 0.52 * death
        + 0.28 * energy_gap
        + 0.32 * intake_gap
        + 0.16 * birth_gap
        + 0.12 * (1.0 - alive_fraction),
        0.0,
        1.0,
    )
    parent_edge = jnp.clip(
        0.5
        + 0.55 * (energy_fraction - mean_energy_fraction)
        + 0.45 * (intake_ema - mean_intake_ema),
        0.0,
        1.0,
    )
    viability = jnp.clip(
        alive_fraction
        * (0.45 + 0.55 * mean_energy_fraction)
        * (1.0 - 0.55 * death),
        0.0,
        1.0,
    )
    normalized_need = jnp.clip(
        0.46 * stress_raw
        + 0.24 * intake_gap * (1.0 - mean_intake_ema)
        + 0.18 * energy_gap * (1.0 - mean_energy_fraction)
        + 0.12 * birth_gap * (1.0 - birth_rate_ema),
        0.0,
        1.0,
    )
    pressure = jnp.clip(
        normalized_need
        * (0.35 + 0.65 * parent_edge)
        * (0.38 + 0.62 * viability),
        0.0,
        1.0,
    )
    rescue = jnp.clip(
        intake_gap
        * (0.28 * decline + 0.22 * death + 0.18 * energy_gap + 0.18 * birth_gap)
        * (0.42 + 0.58 * parent_edge)
        * (0.50 + 0.50 * viability),
        0.0,
        1.0,
    )
    prior = jnp.array([0.24, 0.58, 0.52, 0.36, 0.34, 0.38], dtype=jnp.float32)
    learned = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    scarce = jnp.clip(1.0 - usage, 0.0, 1.0)
    evidence_gap = jnp.clip(1.0 - evidence, 0.0, 1.0)
    trusted = jnp.clip(evidence * (success - 0.34) * 3.20, 0.0, 1.0)
    clone_trust = jnp.clip(evidence[0] * (success[0] - 0.22) * 2.20, 0.0, 1.0)
    relative_trust = jnp.clip(trusted - 0.28 * clone_trust, 0.0, 1.0)
    mutation_trust = jnp.clip(
        0.46 * relative_trust[1]
        + 0.40 * relative_trust[2]
        + 0.04 * relative_trust[3]
        + 0.03 * relative_trust[4]
        + 0.07 * relative_trust[5],
        0.0,
        1.0,
    )
    ecology_gate = jnp.clip(
        (0.20 + 0.80 * pressure)
        * (0.48 + 0.52 * parent_edge)
        * (0.56 + 0.44 * mutation_trust),
        0.0,
        1.0,
    )
    probe = evidence_gap * scarce * ecology_gate
    op_value = (
        1.46 * learned
        + 0.16 * scarce * ecology_gate
        + 0.08 * probe
        + 0.96 * relative_trust
        - 0.30 * usage
    )
    base = jnp.array([4.20, -0.88, -1.62, -5.42, -6.16, -5.78], dtype=jnp.float32)
    calm_gain = jnp.array([2.38, 0.46, 0.08, -0.44, -0.38, -0.32], dtype=jnp.float32)
    pressure_gain = jnp.array([-1.22, 0.40, 0.98, 0.70, 0.30, 0.54], dtype=jnp.float32)
    rescue_gain = jnp.array([-0.34, 0.66, 1.02, 0.54, 0.18, 0.42], dtype=jnp.float32)
    quality_gain = jnp.array([0.00, 0.80, 0.54, 0.10, 0.00, 0.10], dtype=jnp.float32)
    shape_gain = jnp.array([0.00, 0.00, 0.00, 0.06, 0.44, 0.18], dtype=jnp.float32)
    growth_gain = jnp.array([0.16, 0.02, -0.12, -0.24, -0.18, 0.18], dtype=jnp.float32)
    value_gain = jnp.array([0.32, 1.04, 1.18, 0.62, 0.50, 0.58], dtype=jnp.float32)
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

Performance metrics:
Combined score to maximize: 0.00
score: 0.00; sham_auc_delta: 0.01; survival_rate: 0.97; mean_births: 45.72; mean_deaths: 28.69; mean_generation_gain: 4.56; operator_fraction: {'clone': 0.9719753930280246, 'parametric_conservative': 0.019822282980177717, 'parametric_standard': 0.008202323991797676, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9737904441040174, 0.018950822013141834, 0.007142033493888061, 5.141861076915917e-05, 2.5986505139709773e-05, 3.930525424111371e-05]; post_selection_probability: [0.9833589602403324, 0.01226665853562621, 0.004311462956132095, 2.753877442080916e-05, 1.3923197542844233e-05, 2.1453123145737507e-05]; operator_success_ema: [0.16254484746605158, 0.47853795997798443, 0.4894091384485364, 0.5, 0.5, 0.5]; operator_usage_ema: [0.8009050181135535, 0.012211624496558215, 0.003103788651060313, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.822869130410254, 0.05381251219660044, 0.030937507282942533, 0.0, 0.0, 0.0]; shock_auc_delta: 0.01; ancestor_survival_rate: 0.94

Text feedback:
paired ancestor delta=0.0003; sham delta=0.0073; shock delta=0.0138; survival=0.97; mean births=45.7; survival is scored, not a validity gate.


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
    stress_raw = jnp.clip(
        0.58 * decline
        + 0.52 * death
        + 0.28 * energy_gap
        + 0.32 * intake_gap
        + 0.16 * birth_gap
        + 0.12 * (1.0 - alive_fraction),
        0.0,
        1.0,
    )
    parent_edge = jnp.clip(
        0.5
        + 0.55 * (energy_fraction - mean_energy_fraction)
        + 0.45 * (intake_ema - mean_intake_ema),
        0.0,
        1.0,
    )
    viability = jnp.clip(
        alive_fraction
        * (0.45 + 0.55 * mean_energy_fraction)
        * (1.0 - 0.55 * death),
        0.0,
        1.0,
    )
    normalized_need = jnp.clip(
        0.46 * stress_raw
        + 0.24 * intake_gap * (1.0 - mean_intake_ema)
        + 0.18 * energy_gap * (1.0 - mean_energy_fraction)
        + 0.12 * birth_gap * (1.0 - birth_rate_ema),
        0.0,
        1.0,
    )
    pressure = jnp.clip(
        normalized_need
        * (0.35 + 0.65 * parent_edge)
        * (0.38 + 0.62 * viability),
        0.0,
        1.0,
    )
    rescue = jnp.clip(
        intake_gap
        * (0.28 * decline + 0.22 * death + 0.18 * energy_gap + 0.18 * birth_gap)
        * (0.42 + 0.58 * parent_edge)
        * (0.50 + 0.50 * viability),
        0.0,
        1.0,
    )
    prior = jnp.array([0.24, 0.58, 0.52, 0.36, 0.34, 0.38], dtype=jnp.float32)
    learned = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    scarce = jnp.clip(1.0 - usage, 0.0, 1.0)
    evidence_gap = jnp.clip(1.0 - evidence, 0.0, 1.0)
    trusted = jnp.clip(evidence * (success - 0.34) * 3.20, 0.0, 1.0)
    clone_trust = jnp.clip(evidence[0] * (success[0] - 0.22) * 2.20, 0.0, 1.0)
    relative_trust = jnp.clip(trusted - 0.28 * clone_trust, 0.0, 1.0)
    mutation_trust = jnp.clip(
        0.46 * relative_trust[1]
        + 0.40 * relative_trust[2]
        + 0.04 * relative_trust[3]
        + 0.03 * relative_trust[4]
        + 0.07 * relative_trust[5],
        0.0,
        1.0,
    )
    ecology_gate = jnp.clip(
        (0.20 + 0.80 * pressure)
        * (0.48 + 0.52 * parent_edge)
        * (0.56 + 0.44 * mutation_trust),
        0.0,
        1.0,
    )
    probe = evidence_gap * scarce * ecology_gate
    op_value = (
        1.46 * learned
        + 0.16 * scarce * ecology_gate
        + 0.08 * probe
        + 0.96 * relative_trust
        - 0.30 * usage
    )
    base = jnp.array([4.20, -0.88, -1.62, -5.42, -6.16, -5.78], dtype=jnp.float32)
    calm_gain = jnp.array([2.38, 0.46, 0.08, -0.44, -0.38, -0.32], dtype=jnp.float32)
    pressure_gain = jnp.array([-1.22, 0.40, 0.98, 0.70, 0.30, 0.54], dtype=jnp.float32)
    rescue_gain = jnp.array([-0.34, 0.66, 1.02, 0.54, 0.18, 0.42], dtype=jnp.float32)
    quality_gain = jnp.array([0.00, 0.80, 0.54, 0.10, 0.00, 0.10], dtype=jnp.float32)
    shape_gain = jnp.array([0.00, 0.00, 0.00, 0.06, 0.44, 0.18], dtype=jnp.float32)
    growth_gain = jnp.array([0.16, 0.02, -0.12, -0.24, -0.18, 0.18], dtype=jnp.float32)
    value_gain = jnp.array([0.32, 1.04, 1.18, 0.62, 0.50, 0.58], dtype=jnp.float32)
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

Combined score to maximize: 0.01
score: 0.01; sham_auc_delta: 0.01; survival_rate: 0.97; mean_births: 46.25; mean_deaths: 28.84; mean_generation_gain: 4.56; operator_fraction: {'clone': 0.972972972972973, 'parametric_conservative': 0.018243243243243244, 'parametric_standard': 0.008783783783783784, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9737923176942673, 0.018949534711057106, 0.00714145689807107, 5.141575447287806e-05, 2.598539224013983e-05, 3.9303215427645604e-05]; post_selection_probability: [0.9833074015293395, 0.01230681553910794, 0.004322715050153185, 2.761247213294899e-05, 1.395742283000327e-05, 2.1506185303412473e-05]; operator_success_ema: [0.16043149179313332, 0.4778451547026634, 0.48784663807600737, 0.5, 0.5, 0.5]; operator_usage_ema: [0.8104869686067104, 0.01115340930846287, 0.0032068678265204653, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.8272673869505525, 0.0569375129416585, 0.03406250802800059, 0.0, 0.0, 0.0]; shock_auc_delta: 0.02; ancestor_survival_rate: 0.97

Here is additional text feedback about the current program:

paired ancestor delta=0.0054; sham delta=0.0078; shock delta=0.0167; survival=0.97; mean births=46.2; survival is scored, not a validity gate.


# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
