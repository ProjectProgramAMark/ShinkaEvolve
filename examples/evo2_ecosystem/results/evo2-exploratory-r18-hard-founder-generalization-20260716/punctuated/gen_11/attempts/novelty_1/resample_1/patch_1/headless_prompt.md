# System Instructions


Discover a better six-action heredity scheduler for an embodied CPPN ecosystem.

make_offspring receives bounded parent, population, and per-operator success,
usage, and evidence summaries. Return six finite logits for clone,
conservative parametric, standard parametric, exploratory parametric,
structural, and mixed mutation. The paired ancestor for this run is
exact frozen clone. On this exact hard founder panel, continuous standard mutation versus clone scores -0.02765 robust: sham -0.01643 and injury +0.01089, so mutation helps after injury but damages stable worlds and creates severe founder tails. The R17 generation-2 parent was the only source with three all-positive fresh development repeats, but it failed the exposed sealed panel at -0.03315 robust while using about 76% clone, 23% conservative mutation, and 0.7% standard mutation. Discover an ecology-conditioned scheduler that mutates only when ecological stress and accumulated operator success/evidence justify it and when reproduction can exploit the variation. Retain injury benefit without paying stable-world cost or fragile-founder tails. Do not use founder, seed, scenario, event, or future information. Avoid broad fixed mutation, unconditional clone, and unnecessary complexity; prefer a small readable mechanism.. Maximize the paired candidate-minus-ancestor ecological
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
    """Regime-mixed evidence scheduler for six heredity actions."""
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
    low_alive = jnp.clip(1.0 - alive_fraction, 0.0, 1.0)
    energy_gap = jnp.clip(0.58 - mean_energy_fraction, 0.0, 1.0)
    intake_gap = jnp.clip(0.56 - mean_intake_ema, 0.0, 1.0)
    birth_gap = jnp.clip(0.31 - birth_rate_ema, 0.0, 1.0)
    parent_vigor = jnp.clip(0.58 * energy_fraction + 0.34 * intake_ema + 0.08 * (1.0 - age_fraction), 0.0, 1.0)
    compact = jnp.clip(1.0 - 0.46 * node_fraction - 0.54 * connection_fraction, 0.0, 1.0)
    simple_parent = jnp.clip(0.55 + 0.45 * compact, 0.0, 1.0)
    stress_core = jnp.clip(
        0.66 * decline
        + 0.58 * death
        + 0.32 * low_alive
        + 0.34 * intake_gap
        + 0.24 * energy_gap
        + 0.12 * birth_gap,
        0.0,
        1.0,
    )
    injury_rescue = jnp.clip(
        (0.58 * decline + 0.48 * death + 0.22 * low_alive)
        * (0.38 + 0.42 * intake_gap + 0.20 * energy_gap),
        0.0,
        1.0,
    )
    stable_world = jnp.clip(
        alive_fraction
        * (1.0 - decline)
        * (1.0 - death)
        * (0.52 + 0.48 * mean_energy_fraction)
        * (1.0 - 0.72 * injury_rescue),
        0.0,
        1.0,
    )
    renewal_world = jnp.clip(stable_world * parent_vigor * simple_parent * (1.0 - 0.45 * growth), 0.0, 1.0)
    probe_world = jnp.clip(injury_rescue * (1.0 - 0.82 * stable_world) * (0.42 + 0.58 * intake_gap), 0.0, 1.0)
    reproduction_ready = jnp.clip(
        parent_vigor
        * (0.46 + 0.54 * energy_fraction)
        * (0.40 + 0.60 * intake_ema)
        * (0.55 + 0.45 * birth_rate_ema),
        0.0,
        1.0,
    )
    stress_opportunity = jnp.clip(injury_rescue * (0.40 + 0.60 * stress_core) * (0.35 + 0.65 * reproduction_ready), 0.0, 1.0)
    prior = jnp.array([0.42, 0.44, 0.38, 0.14, 0.12, 0.20], dtype=jnp.float32)
    learned = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    credible = jnp.clip(0.12 + 0.88 * evidence, 0.0, 1.0)
    evidence_gate = jnp.clip(0.18 + 0.82 * evidence, 0.0, 1.0)
    unused = jnp.clip((1.0 - usage) * (1.0 - 0.58 * evidence), 0.0, 1.0)
    utility = jnp.clip((1.30 * learned + 0.16 * unused - 0.30 * usage) * credible, 0.0, 2.0)
    evidence_margin = jnp.clip((success - success[0] + 0.06) * evidence, 0.0, 1.0)
    clone_bad = jnp.clip(evidence[0] * (0.40 - success[0]) + 0.16 * usage[0] * evidence[0] * (learned[1] - learned[0]), 0.0, 1.0)
    conservative_edge = jnp.clip((learned[1] - learned[0]) * (0.22 + 0.78 * evidence[1]) + 0.42 * clone_bad + 0.34 * evidence_margin[1], 0.0, 1.0)
    standard_edge = jnp.clip((learned[2] - learned[0]) * (0.18 + 0.82 * evidence[2]) + 0.22 * clone_bad + 0.58 * evidence_margin[2] + 0.18 * unused[2] * stress_opportunity, 0.0, 1.0)
    mixed_edge = jnp.clip((learned[5] - learned[0]) * (0.16 + 0.84 * evidence[5]) + 0.12 * clone_bad + 0.24 * evidence_margin[5], 0.0, 1.0)
    conservative_fatigue = jnp.clip(evidence[1] * usage[1] * (0.42 - success[1]), 0.0, 1.0)
    stable_mutation_debt = jnp.clip(stable_world * (1.0 - injury_rescue) * (0.34 + 0.66 * conservative_fatigue), 0.0, 1.0)
    stress_probe_credit = jnp.clip(stress_opportunity * unused[2] * (1.0 - 0.72 * stable_world) * (0.46 + 0.54 * clone_bad), 0.0, 1.0)
    standard_rescue_credit = jnp.clip(stress_opportunity * (0.24 * unused[2] + 0.76 * standard_edge) * (0.42 + 0.58 * evidence_gate[2]), 0.0, 1.0)
    stable_lock = jnp.clip(stable_world * (0.80 + 0.18 * growth) * (1.0 - 0.94 * injury_rescue) * (1.0 - 0.26 * clone_bad), 0.0, 1.0)
    fragile_lock = jnp.clip(stable_world * (1.0 - parent_vigor) * (0.48 + 0.52 * (1.0 - compact)), 0.0, 1.0)
    stable_escape = jnp.clip(stable_world * reproduction_ready * clone_bad * conservative_edge * evidence_gate[1], 0.0, 1.0)
    renewal_param = jnp.clip(renewal_world * conservative_edge * reproduction_ready * (0.16 + 0.84 * clone_bad) * (1.0 - 0.78 * injury_rescue) * (1.0 - 0.42 * fragile_lock) * (1.0 - 0.78 * conservative_fatigue), 0.0, 1.0)
    rescue_param = jnp.clip((0.50 * stress_core + 1.10 * injury_rescue + 0.12 * low_alive) * standard_edge * reproduction_ready * (1.0 - 0.58 * stable_lock) * (1.0 - 0.46 * fragile_lock), 0.0, 1.0)
    conservative_gate = jnp.clip((0.42 * renewal_param + 0.08 * injury_rescue * conservative_edge + 0.30 * stable_escape + 0.06 * fragile_lock * conservative_edge) * (1.0 - 0.66 * stable_mutation_debt), 0.0, 1.0)
    standard_gate = jnp.clip(
        (0.22 * probe_world * (0.30 + 0.48 * utility[2]) + 0.72 * rescue_param + 0.30 * standard_rescue_credit + 0.34 * stress_probe_credit)
        * (0.35 + 0.65 * reproduction_ready)
        * (1.0 - 0.64 * stable_lock)
        * (1.0 - 0.68 * fragile_lock),
        0.0,
        1.0,
    )
    mixed_gate = jnp.clip(
        (0.10 * probe_world * (0.22 + 0.32 * utility[5]) + 0.10 * rescue_param * mixed_edge)
        * compact
        * reproduction_ready
        * evidence_gate[5]
        * (1.0 - 0.70 * stable_lock)
        * (1.0 - 0.62 * fragile_lock),
        0.0,
        1.0,
    )
    exploratory_gate = jnp.clip(
        probe_world * unused[3] * (0.06 + 0.16 * compact) * injury_rescue * reproduction_ready * evidence_gate[3] * (1.0 - 0.72 * stable_lock) * (1.0 - 0.72 * fragile_lock),
        0.0,
        1.0,
    )
    structural_gate = jnp.clip(
        probe_world * unused[4] * compact * 0.08 * injury_rescue * reproduction_ready * evidence_gate[4] * (1.0 - 0.72 * stable_lock) * (1.0 - 0.72 * fragile_lock),
        0.0,
        1.0,
    )
    anchor = jnp.array([4.10, -1.18, -3.35, -6.40, -6.65, -6.25], dtype=jnp.float32)
    stable_policy = jnp.array([1.22, 0.16, -2.10, -2.35, -2.45, -2.20], dtype=jnp.float32)
    renewal_policy = jnp.array([-0.18, 0.92, -0.10, -0.54, -0.58, -0.42], dtype=jnp.float32)
    stress_policy = jnp.array([-3.16, -0.22, 1.20, 0.30, 0.16, 0.46], dtype=jnp.float32)
    rescue_policy = jnp.array([-1.78, 0.04, 2.34, 0.24, 0.08, 0.52], dtype=jnp.float32)
    utility_policy = jnp.array([0.08, 0.72, 1.12, 0.24, 0.18, 0.32], dtype=jnp.float32)
    pulse = jnp.array(
        [
            -1.18 * standard_gate - 0.46 * mixed_gate - 0.42 * renewal_param - 1.18 * clone_bad * injury_rescue - 0.20 * stable_world * clone_bad - 0.66 * stress_probe_credit + 0.40 * fragile_lock + 0.92 * stable_mutation_debt,
            1.64 * conservative_gate + 0.04 * injury_rescue + 0.10 * stable_escape + 0.16 * fragile_lock - 1.22 * stable_mutation_debt - 0.52 * conservative_fatigue,
            3.18 * standard_gate + 1.04 * rescue_param + 0.44 * standard_rescue_credit + 0.92 * stress_probe_credit - 0.68 * stable_lock - 0.52 * fragile_lock,
            0.38 * exploratory_gate,
            0.24 * structural_gate,
            0.74 * mixed_gate,
        ],
        dtype=jnp.float32,
    )
    growth_trim = jnp.array(
        [0.16 * growth, 0.00 * growth, -0.34 * growth, -0.24 * growth, -0.20 * growth, 0.04 * growth],
        dtype=jnp.float32,
    )
    logits = (
        anchor
        + stable_policy * stable_world
        + renewal_policy * renewal_world
        + stress_policy * stress_core
        + rescue_policy * injury_rescue
        + utility_policy * utility
        + pulse
        + growth_trim
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: -0.00
score: -0.00; sham_auc_delta: -0.00; survival_rate: 0.75; mean_births: 24.12; mean_deaths: 16.06; mean_generation_gain: 3.56; operator_fraction: {'clone': 0.9844559585492227, 'parametric_conservative': 0.015544041450777202, 'parametric_standard': 0.0, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.979256386756897, 0.01811006300151348, 0.002446857336908579, 5.825178726809099e-05, 4.983479768270627e-05, 7.860259298468008e-05]; post_selection_probability: [0.9842137813213325, 0.014079274399028648, 0.001573492707740072, 4.271517891971771e-05, 3.750807955579408e-05, 5.325857903569288e-05]; operator_success_ema: [0.30259986966848373, 0.48781249672174454, 0.5, 0.5, 0.5, 0.5]; operator_usage_ema: [0.5880571380257607, 0.011678626236971468, 0.0, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.5996920578181744, 0.02437500562518835, 0.0, 0.0, 0.0, 0.0]; shock_auc_delta: -0.01; ancestor_survival_rate: 0.75

Text feedback:
paired ancestor delta=-0.0043; sham delta=-0.0037; shock delta=-0.0074; survival=0.75; mean births=24.1; survival is scored, not a validity gate.

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
    """Regime-mixed evidence scheduler for six heredity actions."""
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
    low_alive = jnp.clip(1.0 - alive_fraction, 0.0, 1.0)
    energy_gap = jnp.clip(0.58 - mean_energy_fraction, 0.0, 1.0)
    intake_gap = jnp.clip(0.56 - mean_intake_ema, 0.0, 1.0)
    birth_gap = jnp.clip(0.31 - birth_rate_ema, 0.0, 1.0)
    parent_vigor = jnp.clip(0.58 * energy_fraction + 0.34 * intake_ema + 0.08 * (1.0 - age_fraction), 0.0, 1.0)
    compact = jnp.clip(1.0 - 0.46 * node_fraction - 0.54 * connection_fraction, 0.0, 1.0)
    simple_parent = jnp.clip(0.55 + 0.45 * compact, 0.0, 1.0)
    stress_core = jnp.clip(
        0.66 * decline
        + 0.58 * death
        + 0.32 * low_alive
        + 0.34 * intake_gap
        + 0.24 * energy_gap
        + 0.12 * birth_gap,
        0.0,
        1.0,
    )
    injury_rescue = jnp.clip(
        (0.58 * decline + 0.48 * death + 0.22 * low_alive)
        * (0.38 + 0.42 * intake_gap + 0.20 * energy_gap),
        0.0,
        1.0,
    )
    stable_world = jnp.clip(
        alive_fraction
        * (1.0 - decline)
        * (1.0 - death)
        * (0.52 + 0.48 * mean_energy_fraction)
        * (1.0 - 0.72 * injury_rescue),
        0.0,
        1.0,
    )
    renewal_world = jnp.clip(stable_world * parent_vigor * simple_parent * (1.0 - 0.45 * growth), 0.0, 1.0)
    probe_world = jnp.clip(injury_rescue * (1.0 - 0.82 * stable_world) * (0.42 + 0.58 * intake_gap), 0.0, 1.0)
    reproduction_ready = jnp.clip(
        parent_vigor
        * (0.46 + 0.54 * energy_fraction)
        * (0.40 + 0.60 * intake_ema)
        * (0.55 + 0.45 * birth_rate_ema),
        0.0,
        1.0,
    )
    stress_opportunity = jnp.clip(injury_rescue * (0.40 + 0.60 * stress_core) * (0.35 + 0.65 * reproduction_ready), 0.0, 1.0)
    prior = jnp.array([0.42, 0.46, 0.30, 0.14, 0.12, 0.20], dtype=jnp.float32)
    learned = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    credible = jnp.clip(0.12 + 0.88 * evidence, 0.0, 1.0)
    evidence_gate = jnp.clip(0.18 + 0.82 * evidence, 0.0, 1.0)
    unused = jnp.clip((1.0 - usage) * (1.0 - 0.58 * evidence), 0.0, 1.0)
    utility = jnp.clip((1.30 * learned + 0.16 * unused - 0.30 * usage) * credible, 0.0, 2.0)
    evidence_margin = jnp.clip((success - success[0] + 0.06) * evidence, 0.0, 1.0)
    clone_bad = jnp.clip(evidence[0] * (0.40 - success[0]) + 0.16 * usage[0] * evidence[0] * (learned[1] - learned[0]), 0.0, 1.0)
    conservative_edge = jnp.clip((learned[1] - learned[0]) * (0.22 + 0.78 * evidence[1]) + 0.42 * clone_bad + 0.34 * evidence_margin[1], 0.0, 1.0)
    standard_edge = jnp.clip((learned[2] - learned[0]) * (0.18 + 0.82 * evidence[2]) + 0.20 * clone_bad + 0.58 * evidence_margin[2] + 0.08 * unused[2] * stress_opportunity * clone_bad, 0.0, 1.0)
    mixed_edge = jnp.clip((learned[5] - learned[0]) * (0.16 + 0.84 * evidence[5]) + 0.12 * clone_bad + 0.24 * evidence_margin[5], 0.0, 1.0)
    conservative_fatigue = jnp.clip(evidence[1] * usage[1] * (0.42 - success[1]), 0.0, 1.0)
    stable_mutation_debt = jnp.clip(stable_world * (1.0 - injury_rescue) * (0.34 + 0.66 * conservative_fatigue), 0.0, 1.0)
    stress_probe_credit = jnp.clip(stress_opportunity * unused[2] * reproduction_ready * (0.08 + 0.92 * clone_bad) * (1.0 - 0.88 * stable_world) * (0.28 + 0.72 * stress_core), 0.0, 1.0)
    standard_rescue_credit = jnp.clip(stress_opportunity * (0.24 * unused[2] + 0.76 * standard_edge) * (0.42 + 0.58 * evidence_gate[2]), 0.0, 1.0)
    stable_lock = jnp.clip(stable_world * (0.80 + 0.18 * growth) * (1.0 - 0.94 * injury_rescue) * (1.0 - 0.26 * clone_bad), 0.0, 1.0)
    fragile_lock = jnp.clip(stable_world * (1.0 - parent_vigor) * (0.48 + 0.52 * (1.0 - compact)), 0.0, 1.0)
    stable_escape = jnp.clip(stable_world * reproduction_ready * clone_bad * conservative_edge * evidence_gate[1], 0.0, 1.0)
    renewal_param = jnp.clip(renewal_world * conservative_edge * reproduction_ready * (0.16 + 0.84 * clone_bad) * (1.0 - 0.78 * injury_rescue) * (1.0 - 0.42 * fragile_lock) * (1.0 - 0.78 * conservative_fatigue), 0.0, 1.0)
    rescue_param = jnp.clip((0.50 * stress_core + 1.10 * injury_rescue + 0.12 * low_alive) * standard_edge * reproduction_ready * (1.0 - 0.58 * stable_lock) * (1.0 - 0.46 * fragile_lock), 0.0, 1.0)
    conservative_gate = jnp.clip((0.42 * renewal_param + 0.08 * injury_rescue * conservative_edge + 0.30 * stable_escape + 0.06 * fragile_lock * conservative_edge) * (1.0 - 0.66 * stable_mutation_debt), 0.0, 1.0)
    standard_gate = jnp.clip(
        (0.22 * probe_world * (0.30 + 0.48 * utility[2]) + 0.72 * rescue_param + 0.30 * standard_rescue_credit + 0.34 * stress_probe_credit)
        * (0.35 + 0.65 * reproduction_ready)
        * (1.0 - 0.64 * stable_lock)
        * (1.0 - 0.68 * fragile_lock),
        0.0,
        1.0,
    )
    mixed_gate = jnp.clip(
        (0.10 * probe_world * (0.22 + 0.32 * utility[5]) + 0.10 * rescue_param * mixed_edge)
        * compact
        * reproduction_ready
        * evidence_gate[5]
        * (1.0 - 0.70 * stable_lock)
        * (1.0 - 0.62 * fragile_lock),
        0.0,
        1.0,
    )
    exploratory_gate = jnp.clip(
        probe_world * unused[3] * (0.06 + 0.16 * compact) * injury_rescue * reproduction_ready * evidence_gate[3] * (1.0 - 0.72 * stable_lock) * (1.0 - 0.72 * fragile_lock),
        0.0,
        1.0,
    )
    structural_gate = jnp.clip(
        probe_world * unused[4] * compact * 0.08 * injury_rescue * reproduction_ready * evidence_gate[4] * (1.0 - 0.72 * stable_lock) * (1.0 - 0.72 * fragile_lock),
        0.0,
        1.0,
    )
    anchor = jnp.array([4.10, -1.18, -3.35, -6.40, -6.65, -6.25], dtype=jnp.float32)
    stable_policy = jnp.array([1.22, 0.16, -2.10, -2.35, -2.45, -2.20], dtype=jnp.float32)
    renewal_policy = jnp.array([-0.18, 0.92, -0.10, -0.54, -0.58, -0.42], dtype=jnp.float32)
    stress_policy = jnp.array([-3.16, -0.22, 1.20, 0.30, 0.16, 0.46], dtype=jnp.float32)
    rescue_policy = jnp.array([-1.78, 0.04, 2.34, 0.24, 0.08, 0.52], dtype=jnp.float32)
    utility_policy = jnp.array([0.08, 0.72, 1.12, 0.24, 0.18, 0.32], dtype=jnp.float32)
    pulse = jnp.array(
        [
            -1.18 * standard_gate - 0.46 * mixed_gate - 0.42 * renewal_param - 1.18 * clone_bad * injury_rescue - 0.20 * stable_world * clone_bad - 0.42 * stress_probe_credit + 0.40 * fragile_lock + 0.92 * stable_mutation_debt,
            1.64 * conservative_gate + 0.04 * injury_rescue + 0.10 * stable_escape + 0.16 * fragile_lock - 1.22 * stable_mutation_debt - 0.52 * conservative_fatigue,
            3.18 * standard_gate + 1.04 * rescue_param + 0.44 * standard_rescue_credit + 1.36 * stress_probe_credit - 0.68 * stable_lock - 0.52 * fragile_lock,
            0.38 * exploratory_gate,
            0.24 * structural_gate,
            0.74 * mixed_gate,
        ],
        dtype=jnp.float32,
    )
    growth_trim = jnp.array(
        [0.16 * growth, 0.00 * growth, -0.34 * growth, -0.24 * growth, -0.20 * growth, 0.04 * growth],
        dtype=jnp.float32,
    )
    logits = (
        anchor
        + stable_policy * stable_world
        + renewal_policy * renewal_world
        + stress_policy * stress_core
        + rescue_policy * injury_rescue
        + utility_policy * utility
        + pulse
        + growth_trim
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: -0.00
score: -0.00; sham_auc_delta: -0.00; survival_rate: 0.75; mean_births: 24.12; mean_deaths: 16.06; mean_generation_gain: 3.69; operator_fraction: {'clone': 0.9844559585492227, 'parametric_conservative': 0.015544041450777202, 'parametric_standard': 0.0, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9795059943199158, 0.01796391710639, 0.0023453486990183594, 5.764587374869734e-05, 4.931799514451995e-05, 7.778443832648919e-05]; post_selection_probability: [0.9844538003561043, 0.0139248921269817, 0.0014899565992111871, 4.206352634485354e-05, 3.692996020141609e-05, 5.2363152527894794e-05]; operator_success_ema: [0.30669865291565657, 0.48781249672174454, 0.5, 0.5, 0.5, 0.5]; operator_usage_ema: [0.5880571380257607, 0.011678626236971468, 0.0, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.6007343828678131, 0.02437500562518835, 0.0, 0.0, 0.0, 0.0]; shock_auc_delta: -0.00; ancestor_survival_rate: 0.75

Text feedback:
paired ancestor delta=-0.0005; sham delta=-0.0027; shock delta=-0.0025; survival=0.75; mean births=24.1; survival is scored, not a validity gate.


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
    """Regime-mixed evidence scheduler for six heredity actions."""
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
    low_alive = jnp.clip(1.0 - alive_fraction, 0.0, 1.0)
    energy_gap = jnp.clip(0.58 - mean_energy_fraction, 0.0, 1.0)
    intake_gap = jnp.clip(0.56 - mean_intake_ema, 0.0, 1.0)
    birth_gap = jnp.clip(0.31 - birth_rate_ema, 0.0, 1.0)
    parent_vigor = jnp.clip(0.58 * energy_fraction + 0.34 * intake_ema + 0.08 * (1.0 - age_fraction), 0.0, 1.0)
    compact = jnp.clip(1.0 - 0.46 * node_fraction - 0.54 * connection_fraction, 0.0, 1.0)
    simple_parent = jnp.clip(0.55 + 0.45 * compact, 0.0, 1.0)
    stress_core = jnp.clip(
        0.66 * decline
        + 0.58 * death
        + 0.32 * low_alive
        + 0.34 * intake_gap
        + 0.24 * energy_gap
        + 0.12 * birth_gap,
        0.0,
        1.0,
    )
    injury_rescue = jnp.clip(
        (0.58 * decline + 0.48 * death + 0.22 * low_alive)
        * (0.38 + 0.42 * intake_gap + 0.20 * energy_gap),
        0.0,
        1.0,
    )
    stable_world = jnp.clip(
        alive_fraction
        * (1.0 - decline)
        * (1.0 - death)
        * (0.52 + 0.48 * mean_energy_fraction)
        * (1.0 - 0.72 * injury_rescue),
        0.0,
        1.0,
    )
    renewal_world = jnp.clip(stable_world * parent_vigor * simple_parent * (1.0 - 0.45 * growth), 0.0, 1.0)
    probe_world = jnp.clip(injury_rescue * (1.0 - 0.82 * stable_world) * (0.42 + 0.58 * intake_gap), 0.0, 1.0)
    reproduction_ready = jnp.clip(
        parent_vigor
        * (0.46 + 0.54 * energy_fraction)
        * (0.40 + 0.60 * intake_ema)
        * (0.55 + 0.45 * birth_rate_ema),
        0.0,
        1.0,
    )
    stress_opportunity = jnp.clip(injury_rescue * (0.40 + 0.60 * stress_core) * (0.35 + 0.65 * reproduction_ready), 0.0, 1.0)
    prior = jnp.array([0.42, 0.44, 0.30, 0.14, 0.12, 0.20], dtype=jnp.float32)
    learned = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    credible = jnp.clip(0.12 + 0.88 * evidence, 0.0, 1.0)
    evidence_gate = jnp.clip(0.18 + 0.82 * evidence, 0.0, 1.0)
    unused = jnp.clip((1.0 - usage) * (1.0 - 0.58 * evidence), 0.0, 1.0)
    utility = jnp.clip((1.30 * learned + 0.16 * unused - 0.30 * usage) * credible, 0.0, 2.0)
    evidence_margin = jnp.clip((success - success[0] + 0.06) * evidence, 0.0, 1.0)
    clone_bad = jnp.clip(evidence[0] * (0.40 - success[0]) + 0.16 * usage[0] * evidence[0] * (learned[1] - learned[0]), 0.0, 1.0)
    conservative_edge = jnp.clip((learned[1] - learned[0]) * (0.22 + 0.78 * evidence[1]) + 0.42 * clone_bad + 0.34 * evidence_margin[1], 0.0, 1.0)
    standard_edge = jnp.clip((learned[2] - learned[0]) * (0.18 + 0.82 * evidence[2]) + 0.22 * clone_bad + 0.58 * evidence_margin[2] + 0.10 * unused[2] * injury_rescue, 0.0, 1.0)
    mixed_edge = jnp.clip((learned[5] - learned[0]) * (0.16 + 0.84 * evidence[5]) + 0.12 * clone_bad + 0.24 * evidence_margin[5], 0.0, 1.0)
    conservative_fatigue = jnp.clip(evidence[1] * usage[1] * (0.42 - success[1]), 0.0, 1.0)
    stable_mutation_debt = jnp.clip(stable_world * (1.0 - injury_rescue) * (0.34 + 0.66 * conservative_fatigue), 0.0, 1.0)
    stress_probe_credit = jnp.clip(stress_opportunity * unused[2] * reproduction_ready * (0.10 + 0.90 * clone_bad) * (1.0 - 0.90 * stable_world) * (0.24 + 0.76 * stress_core), 0.0, 1.0)
    standard_rescue_credit = jnp.clip(stress_opportunity * (0.22 * unused[2] + 0.78 * standard_edge) * (0.38 + 0.62 * evidence_gate[2]), 0.0, 1.0)
    stable_lock = jnp.clip(stable_world * (0.80 + 0.18 * growth) * (1.0 - 0.94 * injury_rescue) * (1.0 - 0.26 * clone_bad), 0.0, 1.0)
    fragile_lock = jnp.clip(stable_world * (1.0 - parent_vigor) * (0.48 + 0.52 * (1.0 - compact)), 0.0, 1.0)
    stable_escape = jnp.clip(stable_world * reproduction_ready * clone_bad * conservative_edge * evidence_gate[1], 0.0, 1.0)
    renewal_param = jnp.clip(renewal_world * conservative_edge * reproduction_ready * (0.16 + 0.84 * clone_bad) * (1.0 - 0.78 * injury_rescue) * (1.0 - 0.42 * fragile_lock) * (1.0 - 0.78 * conservative_fatigue), 0.0, 1.0)
    rescue_param = jnp.clip((0.50 * stress_core + 1.10 * injury_rescue + 0.12 * low_alive) * standard_edge * reproduction_ready * (1.0 - 0.58 * stable_lock) * (1.0 - 0.46 * fragile_lock), 0.0, 1.0)
    conservative_gate = jnp.clip((0.42 * renewal_param + 0.08 * injury_rescue * conservative_edge + 0.30 * stable_escape + 0.06 * fragile_lock * conservative_edge) * (1.0 - 0.66 * stable_mutation_debt), 0.0, 1.0)
    standard_gate = jnp.clip(
        (0.22 * probe_world * (0.30 + 0.48 * utility[2]) + 0.72 * rescue_param + 0.30 * standard_rescue_credit)
        * (0.35 + 0.65 * reproduction_ready)
        * (1.0 - 0.64 * stable_lock)
        * (1.0 - 0.68 * fragile_lock),
        0.0,
        1.0,
    )
    mixed_gate = jnp.clip(
        (0.10 * probe_world * (0.22 + 0.32 * utility[5]) + 0.10 * rescue_param * mixed_edge)
        * compact
        * reproduction_ready
        * evidence_gate[5]
        * (1.0 - 0.70 * stable_lock)
        * (1.0 - 0.62 * fragile_lock),
        0.0,
        1.0,
    )
    exploratory_gate = jnp.clip(
        probe_world * unused[3] * (0.06 + 0.16 * compact) * injury_rescue * reproduction_ready * evidence_gate[3] * (1.0 - 0.72 * stable_lock) * (1.0 - 0.72 * fragile_lock),
        0.0,
        1.0,
    )
    structural_gate = jnp.clip(
        probe_world * unused[4] * compact * 0.08 * injury_rescue * reproduction_ready * evidence_gate[4] * (1.0 - 0.72 * stable_lock) * (1.0 - 0.72 * fragile_lock),
        0.0,
        1.0,
    )
    anchor = jnp.array([4.10, -1.18, -3.35, -6.40, -6.65, -6.25], dtype=jnp.float32)
    stable_policy = jnp.array([1.22, 0.16, -2.10, -2.35, -2.45, -2.20], dtype=jnp.float32)
    renewal_policy = jnp.array([-0.18, 0.92, -0.10, -0.54, -0.58, -0.42], dtype=jnp.float32)
    stress_policy = jnp.array([-3.16, -0.22, 1.20, 0.30, 0.16, 0.46], dtype=jnp.float32)
    rescue_policy = jnp.array([-1.78, 0.04, 2.34, 0.24, 0.08, 0.52], dtype=jnp.float32)
    utility_policy = jnp.array([0.08, 0.72, 1.12, 0.24, 0.18, 0.32], dtype=jnp.float32)
    pulse = jnp.array(
        [
            -1.18 * standard_gate - 0.46 * mixed_gate - 0.42 * renewal_param - 1.18 * clone_bad * injury_rescue - 0.20 * stable_world * clone_bad - 0.46 * stress_probe_credit + 0.40 * fragile_lock + 0.92 * stable_mutation_debt,
            1.54 * conservative_gate + 0.04 * injury_rescue + 0.10 * stable_escape + 0.16 * fragile_lock - 1.22 * stable_mutation_debt - 0.52 * conservative_fatigue,
            3.18 * standard_gate + 1.04 * rescue_param + 0.44 * standard_rescue_credit + 1.18 * stress_probe_credit - 0.68 * stable_lock - 0.52 * fragile_lock,
            0.38 * exploratory_gate,
            0.24 * structural_gate,
            0.74 * mixed_gate,
        ],
        dtype=jnp.float32,
    )
    growth_trim = jnp.array(
        [0.16 * growth, 0.00 * growth, -0.34 * growth, -0.24 * growth, -0.20 * growth, 0.04 * growth],
        dtype=jnp.float32,
    )
    logits = (
        anchor
        + stable_policy * stable_world
        + renewal_policy * renewal_world
        + stress_policy * stress_core
        + rescue_policy * injury_rescue
        + utility_policy * utility
        + pulse
        + growth_trim
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Here are the performance metrics of the program:

Combined score to maximize: -0.00
score: -0.00; sham_auc_delta: -0.01; survival_rate: 0.75; mean_births: 24.12; mean_deaths: 16.06; mean_generation_gain: 3.56; operator_fraction: {'clone': 0.9844559585492227, 'parametric_conservative': 0.015544041450777202, 'parametric_standard': 0.0, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9795515847206115, 0.017920244485139847, 0.0023434262070804834, 5.764036439359188e-05, 4.931318428134546e-05, 7.777699909638613e-05]; post_selection_probability: [0.9844587468320415, 0.01391526730176771, 0.0014942050613518361, 4.218189654975071e-05, 3.705062753007431e-05, 5.251564347677881e-05]; operator_success_ema: [0.3040203219279647, 0.48781249672174454, 0.5, 0.5, 0.5, 0.5]; operator_usage_ema: [0.5880571380257607, 0.011678626236971468, 0.0, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.598533920943737, 0.02437500562518835, 0.0, 0.0, 0.0, 0.0]; shock_auc_delta: -0.00; ancestor_survival_rate: 0.75

Here is additional text feedback about the current program:

paired ancestor delta=-0.0003; sham delta=-0.0057; shock delta=-0.0004; survival=0.75; mean births=24.1; survival is scored, not a validity gate.


# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
