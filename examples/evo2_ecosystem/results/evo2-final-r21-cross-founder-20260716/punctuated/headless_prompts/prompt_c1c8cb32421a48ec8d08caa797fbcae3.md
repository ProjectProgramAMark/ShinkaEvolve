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
    clone_bad = jnp.clip(evidence[0] * (0.30 - success[0]) * 2.40, 0.0, 1.0)
    parametric_release = jnp.clip(
        clone_bad
        * (0.24 + 0.76 * ecology_gate)
        * (0.64 + 0.36 * mutation_trust),
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
        + parametric_release
        * jnp.array([-0.54, 0.82, 0.92, 0.08, 0.04, 0.06], dtype=jnp.float32)
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: -0.00
score: -0.00; sham_auc_delta: 0.01; survival_rate: 1.00; mean_births: 46.59; mean_deaths: 28.78; mean_generation_gain: 4.62; operator_fraction: {'clone': 0.9691482226693494, 'parametric_conservative': 0.021462105969148222, 'parametric_standard': 0.009389671361502348, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9737904441040174, 0.018950822013141834, 0.007142033493888061, 5.141861076915917e-05, 2.5986505139709773e-05, 3.930525424111371e-05]; post_selection_probability: [0.9824611546964984, 0.012917115477469598, 0.004556177709649203, 2.8704585375912163e-05, 1.4501435002551337e-05, 2.2338194797448668e-05]; operator_success_ema: [0.16783555608708411, 0.47526917792856693, 0.4870967511087656, 0.5, 0.5, 0.5]; operator_usage_ema: [0.8191232718527317, 0.014299036360171158, 0.0035526936990208924, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.8319622669368982, 0.06318751443177462, 0.03687500860542059, 0.0, 0.0, 0.0]; shock_auc_delta: -0.00; ancestor_survival_rate: 0.91

Text feedback:
paired ancestor delta=-0.0012; sham delta=0.0092; shock delta=-0.0022; survival=1.00; mean births=46.6; survival is scored, not a validity gate.

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
Combined score to maximize: 0.01
score: 0.01; sham_auc_delta: 0.01; survival_rate: 0.97; mean_births: 46.25; mean_deaths: 28.84; mean_generation_gain: 4.56; operator_fraction: {'clone': 0.972972972972973, 'parametric_conservative': 0.018243243243243244, 'parametric_standard': 0.008783783783783784, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9737923176942673, 0.018949534711057106, 0.00714145689807107, 5.141575447287806e-05, 2.598539224013983e-05, 3.9303215427645604e-05]; post_selection_probability: [0.9833074015293395, 0.01230681553910794, 0.004322715050153185, 2.761247213294899e-05, 1.395742283000327e-05, 2.1506185303412473e-05]; operator_success_ema: [0.16043149179313332, 0.4778451547026634, 0.48784663807600737, 0.5, 0.5, 0.5]; operator_usage_ema: [0.8104869686067104, 0.01115340930846287, 0.0032068678265204653, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.8272673869505525, 0.0569375129416585, 0.03406250802800059, 0.0, 0.0, 0.0]; shock_auc_delta: 0.02; ancestor_survival_rate: 0.97

Text feedback:
paired ancestor delta=0.0054; sham delta=0.0078; shock delta=0.0167; survival=0.97; mean births=46.2; survival is scored, not a validity gate.


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

Combined score to maximize: -0.00
score: -0.00; sham_auc_delta: 0.02; survival_rate: 1.00; mean_births: 47.06; mean_deaths: 28.69; mean_generation_gain: 4.62; operator_fraction: {'clone': 0.9734395750332006, 'parametric_conservative': 0.017928286852589643, 'parametric_standard': 0.008632138114209827, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9737904441040174, 0.018950822013141834, 0.007142033493888061, 5.141861076915917e-05, 2.5986505139709773e-05, 3.930525424111371e-05]; post_selection_probability: [0.9828917611390352, 0.0125992150104139, 0.004443944169906899, 2.849750806035445e-05, 1.4401591704427119e-05, 2.2178486801749386e-05]; operator_success_ema: [0.16809866460971534, 0.4778451547026634, 0.49027813598513603, 0.5, 0.5, 0.5]; operator_usage_ema: [0.8315547909587622, 0.009498004823399242, 0.003933471336495131, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.8366774972528219, 0.0569375129416585, 0.030937507282942533, 0.0, 0.0, 0.0]; shock_auc_delta: 0.00; ancestor_survival_rate: 0.97

Here is additional text feedback about the current program:

paired ancestor delta=-0.0018; sham delta=0.0209; shock delta=0.0023; survival=1.00; mean births=47.1; survival is scored, not a validity gate.


# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
