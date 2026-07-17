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
    recovery_probe = jnp.clip(
        (0.42 + 0.58 * parent_quality)
        * (0.35 + 0.65 * compact)
        * (0.28 + 0.72 * birth_gap)
        * (1.0 - 0.62 * death)
        * (0.52 + 0.48 * jnp.clip(pressure + rescue, 0.0, 1.0)),
        0.0,
        1.0,
    )
    trust_center = jnp.clip(evidence * (success - prior), -0.35, 0.35)
    probe_gain = jnp.array([-0.34, 0.18, 0.42, 1.20, 0.86, 1.04], dtype=jnp.float32)
    trust_gain = jnp.array([0.10, 0.56, 0.74, 0.58, 0.50, 0.54], dtype=jnp.float32)
    logits = (
        base
        + calm_gain * calm
        + pressure_gain * pressure
        + rescue_gain * rescue
        + quality_gain * parent_quality
        + shape_gain * compact
        + growth_gain * growth
        + value_gain * op_value
        + probe_gain * recovery_probe
        + trust_gain * trust_center
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: -0.00
score: -0.00; sham_auc_delta: 0.02; survival_rate: 0.97; mean_births: 47.16; mean_deaths: 28.62; mean_generation_gain: 5.12; operator_fraction: {'clone': 0.9622266401590457, 'parametric_conservative': 0.027170311464546057, 'parametric_standard': 0.010603048376408217, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9516882627411226, 0.03323761096834082, 0.013909903560987615, 0.0005443272437381072, 0.00028589066757771095, 0.0003339967837730274]; post_selection_probability: [0.9657249235121979, 0.02430427002423644, 0.009255376139935904, 0.0003333664589960038, 0.00017580177888291615, 0.00020625011992531184]; operator_success_ema: [0.18122283497359604, 0.4691088665276766, 0.4851512061432004, 0.5, 0.5, 0.5]; operator_usage_ema: [0.8107076790183783, 0.019431392014666926, 0.006749733434844529, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.8309835367836058, 0.07046564109623432, 0.04031250951811671, 0.0, 0.0, 0.0]; shock_auc_delta: -0.00; ancestor_survival_rate: 0.97

Text feedback:
paired ancestor delta=-0.0047; sham delta=0.0195; shock delta=-0.0041; survival=0.97; mean births=47.2; survival is scored, not a validity gate.

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
    rescue_window = jnp.clip(
        (0.40 + 0.60 * parent_quality)
        * (0.38 + 0.62 * compact)
        * (0.34 + 0.66 * birth_gap)
        * jnp.clip(pressure + rescue - 0.46 * calm, 0.0, 1.0)
        * (1.0 - 0.58 * death),
        0.0,
        1.0,
    )
    trusted_edge = jnp.clip(evidence * (success - prior), -0.28, 0.28)
    cautious_probe = jnp.clip(probe * rescue_window * (1.0 - 0.44 * calm), 0.0, 1.0)
    probe_gain = jnp.array([-0.24, 0.14, 0.28, 0.86, 0.58, 0.72], dtype=jnp.float32)
    trust_gain = jnp.array([0.08, 0.48, 0.62, 0.44, 0.38, 0.42], dtype=jnp.float32)
    logits = (
        base
        + calm_gain * calm
        + pressure_gain * pressure
        + rescue_gain * rescue
        + quality_gain * parent_quality
        + shape_gain * compact
        + growth_gain * growth
        + value_gain * op_value
        + probe_gain * cautious_probe
        + trust_gain * trusted_edge
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: -0.00
score: -0.00; sham_auc_delta: 0.02; survival_rate: 1.00; mean_births: 47.44; mean_deaths: 28.97; mean_generation_gain: 4.69; operator_fraction: {'clone': 0.9617918313570487, 'parametric_conservative': 0.027009222661396576, 'parametric_standard': 0.011198945981554678, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9568675484573632, 0.03005639404842728, 0.012170641405279176, 0.00041319327755716807, 0.00023126123440370225, 0.00026096360862108047]; post_selection_probability: [0.9701477333556774, 0.021569698817970218, 0.007758202369129935, 0.0002378533218893432, 0.00013450723080065592, 0.00015200051937483308]; operator_success_ema: [0.15232757409103215, 0.4706109846010804, 0.48632646165788174, 0.5, 0.5, 0.5]; operator_usage_ema: [0.8235504915937781, 0.018061269955069292, 0.005328565835952759, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.8405768573284149, 0.07846876792609692, 0.03718750877305865, 0.0, 0.0, 0.0]; shock_auc_delta: 0.01; ancestor_survival_rate: 0.94

Text feedback:
paired ancestor delta=-0.0019; sham delta=0.0171; shock delta=0.0120; survival=1.00; mean births=47.4; survival is scored, not a validity gate.


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
    rescue_window = jnp.clip(
        (0.40 + 0.60 * parent_quality)
        * (0.38 + 0.62 * compact)
        * (0.34 + 0.66 * birth_gap)
        * jnp.clip(pressure + rescue - 0.46 * calm, 0.0, 1.0)
        * (1.0 - 0.58 * death),
        0.0,
        1.0,
    )
    trusted_edge = jnp.clip(evidence * (success - prior), -0.28, 0.28)
    clone_trust = jnp.clip(evidence[0] * (success[0] - prior[0]), -0.28, 0.28)
    nonclone_edge = jnp.clip(trusted_edge - clone_trust, -0.18, 0.32)
    transfer_gate = jnp.clip(
        (0.30 + 0.70 * rescue_window)
        * (0.22 + 0.78 * jnp.clip(pressure + rescue - 0.55 * calm, 0.0, 1.0))
        * (0.55 + 0.45 * parent_quality)
        * (1.0 - 0.50 * growth),
        0.0,
        1.0,
    )
    cautious_probe = jnp.clip(
        probe * rescue_window * (1.0 - 0.44 * calm) * (0.18 + 0.82 * transfer_gate),
        0.0,
        1.0,
    )
    probe_gain = jnp.array([-0.18, 0.10, 0.22, 0.58, 0.38, 0.48], dtype=jnp.float32)
    trust_gain = jnp.array([0.06, 0.44, 0.58, 0.34, 0.30, 0.34], dtype=jnp.float32)
    relative_gain = jnp.array([-0.46, 0.42, 0.56, 0.20, 0.16, 0.18], dtype=jnp.float32)
    logits = (
        base
        + calm_gain * calm
        + pressure_gain * pressure
        + rescue_gain * rescue
        + quality_gain * parent_quality
        + shape_gain * compact
        + growth_gain * growth
        + value_gain * op_value
        + probe_gain * cautious_probe
        + trust_gain * trusted_edge
        + relative_gain * nonclone_edge * transfer_gate
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Here are the performance metrics of the program:

Combined score to maximize: -0.00
score: -0.00; sham_auc_delta: 0.00; survival_rate: 1.00; mean_births: 47.12; mean_deaths: 28.53; mean_generation_gain: 4.66; operator_fraction: {'clone': 0.9641909814323607, 'parametric_conservative': 0.02586206896551724, 'parametric_standard': 0.009946949602122016, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9578063916741756, 0.029493388474771853, 0.01184412196539996, 0.0003880724737731119, 0.00022087457565296638, 0.0002471751624334342]; post_selection_probability: [0.9700324170291423, 0.021684852475300432, 0.007768025406403467, 0.00023229292964970226, 0.0001330057047198352, 0.00014938965641704273]; operator_success_ema: [0.1667772502405569, 0.4673636555671692, 0.48775905184447765, 0.5, 0.5, 0.5]; operator_usage_ema: [0.8251970903947949, 0.01753069805999985, 0.004333576187491417, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.8339836094528437, 0.08068751823157072, 0.03718750877305865, 0.0, 0.0, 0.0]; shock_auc_delta: 0.00; ancestor_survival_rate: 0.97

Here is additional text feedback about the current program:

paired ancestor delta=-0.0040; sham delta=0.0023; shock delta=0.0002; survival=1.00; mean births=47.1; survival is scored, not a validity gate.


# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
