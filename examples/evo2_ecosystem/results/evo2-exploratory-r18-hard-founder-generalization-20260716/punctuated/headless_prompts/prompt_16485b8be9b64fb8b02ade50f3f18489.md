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
    prior = jnp.array([0.18, 0.62, 0.48, 0.28, 0.24, 0.34], dtype=jnp.float32)
    learned = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    credible = jnp.clip(0.22 + 0.78 * evidence, 0.0, 1.0)
    unused = jnp.clip((1.0 - usage) * (1.0 - 0.58 * evidence), 0.0, 1.0)
    utility = jnp.clip((1.42 * learned + 0.22 * unused - 0.24 * usage) * credible, 0.0, 2.0)
    clone_bad = jnp.clip(evidence[0] * (0.46 - success[0]) + 0.20 * usage[0] * evidence[0] * (learned[1] - learned[0]), 0.0, 1.0)
    conservative_edge = jnp.clip((learned[1] - learned[0]) * (0.55 + 0.45 * evidence[1]) + 0.46 * clone_bad, 0.0, 1.0)
    standard_edge = jnp.clip((learned[2] - learned[0]) * (0.38 + 0.62 * evidence[2]) + 0.30 * clone_bad + 0.16 * unused[2] * injury_rescue, 0.0, 1.0)
    mixed_edge = jnp.clip((learned[5] - learned[0]) * (0.36 + 0.64 * evidence[5]) + 0.16 * clone_bad, 0.0, 1.0)
    conservative_fatigue = jnp.clip(evidence[1] * usage[1] * (0.42 - success[1]), 0.0, 1.0)
    stable_mutation_debt = jnp.clip(stable_world * (1.0 - injury_rescue) * (0.34 + 0.66 * conservative_fatigue), 0.0, 1.0)
    standard_rescue_credit = jnp.clip(injury_rescue * (0.18 + 0.82 * stress_core) * (0.46 * unused[2] + 0.54 * standard_edge), 0.0, 1.0)
    stable_lock = jnp.clip(stable_world * (0.72 + 0.22 * growth) * (1.0 - 0.90 * injury_rescue) * (1.0 - 0.34 * clone_bad), 0.0, 1.0)
    fragile_lock = jnp.clip(stable_world * (1.0 - parent_vigor) * (0.48 + 0.52 * (1.0 - compact)), 0.0, 1.0)
    renewal_param = jnp.clip(renewal_world * conservative_edge * (1.0 - 0.72 * injury_rescue) * (1.0 - 0.35 * fragile_lock) * (1.0 - 0.72 * conservative_fatigue), 0.0, 1.0)
    rescue_param = jnp.clip((0.74 * stress_core + 1.12 * injury_rescue + 0.18 * low_alive) * standard_edge * (1.0 - 0.52 * stable_lock) * (1.0 - 0.42 * fragile_lock), 0.0, 1.0)
    conservative_gate = jnp.clip((1.00 * renewal_param + 0.20 * injury_rescue * conservative_edge + 0.22 * stable_world * clone_bad * conservative_edge + 0.10 * fragile_lock * conservative_edge) * (1.0 - 0.58 * stable_mutation_debt), 0.0, 1.0)
    standard_gate = jnp.clip(
        (0.36 * probe_world * (0.36 + 0.52 * utility[2]) + 0.62 * rescue_param + 0.22 * standard_rescue_credit)
        * (1.0 - 0.58 * stable_lock)
        * (1.0 - 0.64 * fragile_lock),
        0.0,
        1.0,
    )
    mixed_gate = jnp.clip(
        (0.18 * probe_world * (0.30 + 0.38 * utility[5]) + 0.16 * rescue_param * mixed_edge)
        * compact
        * (1.0 - 0.64 * stable_lock)
        * (1.0 - 0.58 * fragile_lock),
        0.0,
        1.0,
    )
    exploratory_gate = jnp.clip(
        probe_world * unused[3] * (0.10 + 0.20 * compact) * injury_rescue * (1.0 - 0.68 * stable_lock) * (1.0 - 0.70 * fragile_lock),
        0.0,
        1.0,
    )
    structural_gate = jnp.clip(
        probe_world * unused[4] * compact * 0.12 * injury_rescue * (1.0 - 0.68 * stable_lock) * (1.0 - 0.70 * fragile_lock),
        0.0,
        1.0,
    )
    anchor = jnp.array([3.32, -0.44, -2.96, -6.10, -6.42, -5.88], dtype=jnp.float32)
    stable_policy = jnp.array([1.06, 0.76, -1.72, -2.12, -2.26, -1.92], dtype=jnp.float32)
    renewal_policy = jnp.array([-0.86, 2.18, 0.08, -0.48, -0.52, -0.32], dtype=jnp.float32)
    stress_policy = jnp.array([-2.88, -0.12, 1.08, 0.34, 0.22, 0.52], dtype=jnp.float32)
    rescue_policy = jnp.array([-1.46, 0.22, 1.96, 0.34, 0.16, 0.74], dtype=jnp.float32)
    utility_policy = jnp.array([0.08, 1.16, 1.04, 0.32, 0.24, 0.42], dtype=jnp.float32)
    pulse = jnp.array(
        [
            -1.02 * standard_gate - 0.58 * mixed_gate - 1.26 * renewal_param - 1.12 * clone_bad * injury_rescue - 0.34 * stable_world * clone_bad + 0.34 * fragile_lock + 0.72 * stable_mutation_debt,
            2.56 * conservative_gate + 0.10 * injury_rescue + 0.22 * stable_lock * conservative_edge + 0.18 * fragile_lock - 1.08 * stable_mutation_debt - 0.46 * conservative_fatigue,
            2.74 * standard_gate + 0.82 * rescue_param + 0.34 * standard_rescue_credit - 0.54 * stable_lock - 0.44 * fragile_lock,
            0.56 * exploratory_gate,
            0.36 * structural_gate,
            1.08 * mixed_gate,
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
Combined score to maximize: -0.02
score: -0.02; sham_auc_delta: -0.02; survival_rate: 0.75; mean_births: 21.00; mean_deaths: 15.25; mean_generation_gain: 3.38; operator_fraction: {'clone': 0.8005952380952381, 'parametric_conservative': 0.18154761904761904, 'parametric_standard': 0.017857142857142856, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.8287639943036166, 0.16011554138226944, 0.010540202006020329, 0.00018229150208009577, 0.00011812515136129646, 0.0002798566592074084]; post_selection_probability: [0.80209046336886, 0.18996776813921862, 0.007503904121583456, 0.00013620654678798267, 9.668938116802776e-05, 0.00020497660772565574]; operator_success_ema: [0.32296791207045317, 0.44702629931271076, 0.49080172739923, 0.5, 0.5, 0.5]; operator_usage_ema: [0.4518947936594486, 0.09486122918315232, 0.011235607322305441, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.5084658619016409, 0.16277872491627932, 0.018750004470348358, 0.0, 0.0, 0.0]; shock_auc_delta: -0.03; ancestor_survival_rate: 0.75

Text feedback:
paired ancestor delta=-0.0223; sham delta=-0.0173; shock delta=-0.0255; survival=0.75; mean births=21.0; survival is scored, not a validity gate.

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
    """Stress-budget bandit scheduler for six heredity actions."""
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
    sparse = jnp.clip(1.0 - alive_fraction, 0.0, 1.0)
    energy_short = jnp.clip(0.55 - mean_energy_fraction, 0.0, 1.0)
    intake_short = jnp.clip(0.54 - mean_intake_ema, 0.0, 1.0)
    birth_short = jnp.clip(0.28 - birth_rate_ema, 0.0, 1.0)
    parent_power = jnp.clip(
        0.52 * energy_fraction + 0.34 * intake_ema + 0.14 * (1.0 - age_fraction),
        0.0,
        1.0,
    )
    compact = jnp.clip(1.0 - 0.50 * node_fraction - 0.50 * connection_fraction, 0.0, 1.0)
    leverage = jnp.clip(
        parent_power
        * (0.45 + 0.55 * energy_fraction)
        * (0.45 + 0.55 * intake_ema)
        * (0.58 + 0.42 * compact),
        0.0,
        1.0,
    )
    fast_stress = jnp.clip(0.82 * decline + 0.72 * death + 0.34 * sparse, 0.0, 1.0)
    resource_stress = jnp.clip(0.44 * intake_short + 0.34 * energy_short + 0.20 * birth_short, 0.0, 1.0)
    stress = jnp.clip(
        fast_stress * (0.44 + 0.56 * resource_stress) + 0.16 * sparse * resource_stress,
        0.0,
        1.0,
    )
    stable = jnp.clip(
        alive_fraction
        * (1.0 - decline)
        * (1.0 - death)
        * (0.62 + 0.38 * mean_energy_fraction)
        * (0.62 + 0.38 * mean_intake_ema),
        0.0,
        1.0,
    )
    no_spend = jnp.clip(stable * (1.0 - 0.88 * stress) * (0.82 + 0.18 * growth), 0.0, 1.0)
    fragile = jnp.clip(stable * (1.0 - parent_power) * (0.48 + 0.52 * (1.0 - compact)), 0.0, 1.0)
    prior = jnp.array([0.38, 0.46, 0.33, 0.12, 0.10, 0.18], dtype=jnp.float32)
    estimate = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    trust = jnp.clip(0.10 + 0.90 * evidence, 0.0, 1.0)
    novelty = jnp.clip((1.0 - usage) * (1.0 - 0.62 * evidence), 0.0, 1.0)
    clone_failure = jnp.clip(evidence[0] * (0.39 - success[0]) + usage[0] * evidence[0] * 0.10, 0.0, 1.0)
    conservative_proven = jnp.clip((estimate[1] - estimate[0] + 0.03) * trust[1] + 0.24 * clone_failure, 0.0, 1.0)
    standard_proven = jnp.clip((estimate[2] - estimate[0] + 0.04) * trust[2] + 0.30 * clone_failure, 0.0, 1.0)
    mixed_proven = jnp.clip((estimate[5] - estimate[0] + 0.02) * trust[5] + 0.12 * clone_failure, 0.0, 1.0)
    conservative_fatigue = jnp.clip(usage[1] * evidence[1] * (0.45 - success[1]), 0.0, 1.0)
    standard_fatigue = jnp.clip(usage[2] * evidence[2] * (0.43 - success[2]), 0.0, 1.0)
    rescue_budget = jnp.clip(
        stress
        * leverage
        * (0.18 + 0.82 * clone_failure + 0.26 * novelty[2])
        * (1.0 - 0.78 * no_spend)
        * (1.0 - 0.55 * fragile),
        0.0,
        1.0,
    )
    renewal_budget = jnp.clip(
        stable
        * leverage
        * conservative_proven
        * (0.16 + 0.84 * clone_failure)
        * (1.0 - 0.86 * stress)
        * (1.0 - 0.70 * conservative_fatigue)
        * (1.0 - 0.58 * fragile),
        0.0,
        1.0,
    )
    probe_budget = jnp.clip(
        stress
        * leverage
        * novelty
        * (0.35 + 0.65 * clone_failure)
        * (1.0 - 0.92 * no_spend),
        0.0,
        1.0,
    )
    mutation_tax = jnp.clip(no_spend * (0.80 + 0.20 * growth) + 0.55 * fragile, 0.0, 1.0)
    relief_window = jnp.clip(
        stress
        * leverage
        * (0.20 + 0.80 * clone_failure)
        * (0.40 + 0.60 * novelty[2])
        * (1.0 - 0.92 * no_spend)
        * (1.0 - 0.62 * fragile),
        0.0,
        1.0,
    )
    conservative_bridge = jnp.clip(
        relief_window
        * (0.36 + 0.64 * novelty[1])
        * (0.56 + 0.44 * compact)
        * (1.0 - 0.60 * conservative_fatigue),
        0.0,
        1.0,
    )
    standard_bridge = jnp.clip(
        relief_window
        * (0.46 + 0.54 * standard_proven)
        * (0.72 + 0.28 * compact)
        * (1.0 - 0.70 * standard_fatigue),
        0.0,
        1.0,
    )
    clone_logit = (
        4.55
        + 1.35 * no_spend
        + 0.72 * fragile
        + 0.24 * growth
        - 2.55 * rescue_budget
        - 0.58 * renewal_budget
        - 1.55 * relief_window
        - 0.82 * clone_failure * stress
    )
    conservative_logit = (
        -2.05
        + 2.25 * renewal_budget
        + 1.85 * conservative_bridge
        + 0.38 * stress * conservative_proven
        + 0.30 * fragile * conservative_proven
        - 1.70 * mutation_tax
        - 0.72 * conservative_fatigue
    )
    standard_logit = (
        -4.10
        + 4.85 * rescue_budget * (0.42 + 0.58 * standard_proven)
        + 3.25 * standard_bridge
        + 0.92 * stress * leverage * novelty[2] * clone_failure
        - 1.95 * no_spend
        - 1.10 * fragile
        - 0.68 * standard_fatigue
    )
    exploratory_logit = (
        -6.55
        + 0.95 * probe_budget[3] * compact
        + 0.42 * rescue_budget * novelty[3]
        - 1.20 * no_spend
        - 0.70 * fragile
    )
    structural_logit = (
        -6.75
        + 0.70 * probe_budget[4] * compact
        + 0.24 * rescue_budget * novelty[4]
        - 1.20 * no_spend
        - 0.75 * fragile
    )
    mixed_logit = (
        -6.35
        + 1.25 * rescue_budget * mixed_proven * compact
        + 0.62 * probe_budget[5] * compact
        - 1.35 * no_spend
        - 0.78 * fragile
    )
    logits = jnp.array(
        [
            clone_logit,
            conservative_logit,
            standard_logit,
            exploratory_logit,
            structural_logit,
            mixed_logit,
        ],
        dtype=jnp.float32,
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: -0.00
score: -0.00; sham_auc_delta: -0.02; survival_rate: 0.75; mean_births: 24.50; mean_deaths: 16.06; mean_generation_gain: 3.56; operator_fraction: {'clone': 1.0, 'parametric_conservative': 0.0, 'parametric_standard': 0.0, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9987265586853027, 0.0010958320461213588, 0.00014015886408742518, 1.2551590043585747e-05, 1.015338821162004e-05, 1.4735287149960641e-05]; post_selection_probability: [0.9989213748284947, 0.0009298340144447372, 0.00011671821364509322, 1.0731585770537356e-05, 8.909420757207268e-06, 1.2433372187295082e-05]; operator_success_ema: [0.2979920394718647, 0.5, 0.5, 0.5, 0.5, 0.5]; operator_usage_ema: [0.6048246137797832, 0.0, 0.0, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.6118159275501966, 0.0, 0.0, 0.0, 0.0, 0.0]; shock_auc_delta: 0.00; ancestor_survival_rate: 0.75

Text feedback:
paired ancestor delta=-0.0012; sham delta=-0.0164; shock delta=0.0034; survival=0.75; mean births=24.5; survival is scored, not a validity gate.


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
    """Stress-budget bandit scheduler for six heredity actions."""
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
    sparse = jnp.clip(1.0 - alive_fraction, 0.0, 1.0)
    energy_short = jnp.clip(0.55 - mean_energy_fraction, 0.0, 1.0)
    intake_short = jnp.clip(0.54 - mean_intake_ema, 0.0, 1.0)
    birth_short = jnp.clip(0.28 - birth_rate_ema, 0.0, 1.0)
    parent_power = jnp.clip(
        0.52 * energy_fraction + 0.34 * intake_ema + 0.14 * (1.0 - age_fraction),
        0.0,
        1.0,
    )
    compact = jnp.clip(1.0 - 0.50 * node_fraction - 0.50 * connection_fraction, 0.0, 1.0)
    leverage = jnp.clip(
        parent_power
        * (0.45 + 0.55 * energy_fraction)
        * (0.45 + 0.55 * intake_ema)
        * (0.58 + 0.42 * compact),
        0.0,
        1.0,
    )
    fast_stress = jnp.clip(0.82 * decline + 0.72 * death + 0.34 * sparse, 0.0, 1.0)
    resource_stress = jnp.clip(0.44 * intake_short + 0.34 * energy_short + 0.20 * birth_short, 0.0, 1.0)
    stress = jnp.clip(
        fast_stress * (0.44 + 0.56 * resource_stress) + 0.16 * sparse * resource_stress,
        0.0,
        1.0,
    )
    stable = jnp.clip(
        alive_fraction
        * (1.0 - decline)
        * (1.0 - death)
        * (0.62 + 0.38 * mean_energy_fraction)
        * (0.62 + 0.38 * mean_intake_ema),
        0.0,
        1.0,
    )
    no_spend = jnp.clip(stable * (1.0 - 0.88 * stress) * (0.82 + 0.18 * growth), 0.0, 1.0)
    fragile = jnp.clip(stable * (1.0 - parent_power) * (0.48 + 0.52 * (1.0 - compact)), 0.0, 1.0)
    prior = jnp.array([0.38, 0.46, 0.33, 0.12, 0.10, 0.18], dtype=jnp.float32)
    estimate = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    trust = jnp.clip(0.10 + 0.90 * evidence, 0.0, 1.0)
    novelty = jnp.clip((1.0 - usage) * (1.0 - 0.62 * evidence), 0.0, 1.0)
    clone_failure = jnp.clip(evidence[0] * (0.39 - success[0]) + usage[0] * evidence[0] * 0.10, 0.0, 1.0)
    conservative_proven = jnp.clip((estimate[1] - estimate[0] + 0.03) * trust[1] + 0.24 * clone_failure, 0.0, 1.0)
    standard_proven = jnp.clip((estimate[2] - estimate[0] + 0.04) * trust[2] + 0.30 * clone_failure, 0.0, 1.0)
    mixed_proven = jnp.clip((estimate[5] - estimate[0] + 0.02) * trust[5] + 0.12 * clone_failure, 0.0, 1.0)
    conservative_fatigue = jnp.clip(usage[1] * evidence[1] * (0.45 - success[1]), 0.0, 1.0)
    standard_fatigue = jnp.clip(usage[2] * evidence[2] * (0.43 - success[2]), 0.0, 1.0)
    rescue_budget = jnp.clip(
        stress
        * leverage
        * (0.18 + 0.82 * clone_failure + 0.26 * novelty[2])
        * (1.0 - 0.78 * no_spend)
        * (1.0 - 0.55 * fragile),
        0.0,
        1.0,
    )
    renewal_budget = jnp.clip(
        stable
        * leverage
        * conservative_proven
        * (0.16 + 0.84 * clone_failure)
        * (1.0 - 0.86 * stress)
        * (1.0 - 0.70 * conservative_fatigue)
        * (1.0 - 0.58 * fragile),
        0.0,
        1.0,
    )
    probe_budget = jnp.clip(
        stress
        * leverage
        * novelty
        * (0.35 + 0.65 * clone_failure)
        * (1.0 - 0.92 * no_spend),
        0.0,
        1.0,
    )
    mutation_tax = jnp.clip(no_spend * (0.80 + 0.20 * growth) + 0.55 * fragile, 0.0, 1.0)
    clone_logit = (
        4.55
        + 1.35 * no_spend
        + 0.72 * fragile
        + 0.24 * growth
        - 2.55 * rescue_budget
        - 0.58 * renewal_budget
        - 0.82 * clone_failure * stress
    )
    conservative_logit = (
        -2.05
        + 2.25 * renewal_budget
        + 0.38 * stress * conservative_proven
        + 0.30 * fragile * conservative_proven
        - 1.95 * mutation_tax
        - 0.72 * conservative_fatigue
    )
    standard_logit = (
        -4.10
        + 4.85 * rescue_budget * (0.42 + 0.58 * standard_proven)
        + 0.92 * stress * leverage * novelty[2] * clone_failure
        - 2.20 * no_spend
        - 1.10 * fragile
        - 0.68 * standard_fatigue
    )
    exploratory_logit = (
        -6.55
        + 0.95 * probe_budget[3] * compact
        + 0.42 * rescue_budget * novelty[3]
        - 1.20 * no_spend
        - 0.70 * fragile
    )
    structural_logit = (
        -6.75
        + 0.70 * probe_budget[4] * compact
        + 0.24 * rescue_budget * novelty[4]
        - 1.20 * no_spend
        - 0.75 * fragile
    )
    mixed_logit = (
        -6.35
        + 1.25 * rescue_budget * mixed_proven * compact
        + 0.62 * probe_budget[5] * compact
        - 1.35 * no_spend
        - 0.78 * fragile
    )
    logits = jnp.array(
        [
            clone_logit,
            conservative_logit,
            standard_logit,
            exploratory_logit,
            structural_logit,
            mixed_logit,
        ],
        dtype=jnp.float32,
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Here are the performance metrics of the program:

Combined score to maximize: -0.00
score: -0.00; sham_auc_delta: -0.01; survival_rate: 0.75; mean_births: 24.50; mean_deaths: 16.06; mean_generation_gain: 3.62; operator_fraction: {'clone': 1.0, 'parametric_conservative': 0.0, 'parametric_standard': 0.0, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9988153100013732, 0.0010178464325144886, 0.00013027802051510662, 1.2261593365110457e-05, 9.919057411025279e-06, 1.4395602120202966e-05]; post_selection_probability: [0.9990173146041513, 0.0008456326237531129, 0.00010590930745905613, 1.0422427241356188e-05, 8.663518656248925e-06, 1.2075658084184567e-05]; operator_success_ema: [0.297983730211854, 0.5, 0.5, 0.5, 0.5, 0.5]; operator_usage_ema: [0.6048246137797832, 0.0, 0.0, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.6107849273830652, 0.0, 0.0, 0.0, 0.0, 0.0]; shock_auc_delta: -0.00; ancestor_survival_rate: 0.75

Here is additional text feedback about the current program:

paired ancestor delta=-0.0019; sham delta=-0.0068; shock delta=-0.0006; survival=0.75; mean births=24.5; survival is scored, not a validity gate.


# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
