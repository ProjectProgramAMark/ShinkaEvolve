# System Instructions


Discover a better six-action heredity scheduler for an embodied CPPN ecosystem.

make_offspring receives bounded parent, population, and per-operator success,
usage, and evidence summaries. Return six finite logits for clone,
conservative parametric, standard parametric, exploratory parametric,
structural, and mixed mutation. The paired ancestor for this run is
Exact frozen clone. R19 used three repeats and found no positive candidate; its large parent repeatedly yielded nearly constant 98-99% clone plus conservative mutation. R20 resets only the editable program to a compact stress-gated clone-to-standard rescue pulse while keeping the complete hard panel, three-repeat score, physics, operators, and hidden partitions unchanged. Determine whether observable decline, death, population loss, energy, intake, parent readiness, and operator evidence can identify moments when mutation is worth its cost. Improve the small mechanism structurally and keep realized mutation ecology-conditioned. Do not use founder, seed, scenario, event, future, or hidden information. Avoid decorative complexity and unconditional mutation.. Maximize the paired candidate-minus-ancestor ecological
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
    """Homeostatic evidence-bandit scheduler for six heredity operators."""
    success = operator_stats[0]
    usage = operator_stats[1]
    evidence = operator_stats[2]
    parent_energy = jnp.clip(parent_stats[0], 0.0, 1.0)
    parent_intake = jnp.clip(parent_stats[1], 0.0, 1.0)
    parent_age = jnp.clip(parent_stats[2], 0.0, 1.0)
    node_frac = jnp.clip(parent_genome_summary[0], 0.0, 1.0)
    conn_frac = jnp.clip(parent_genome_summary[1], 0.0, 1.0)
    alive = jnp.clip(population_stats[0], 0.0, 1.0)
    mean_energy = jnp.clip(population_stats[1], 0.0, 1.0)
    pop_loss = jnp.clip(-population_stats[2], 0.0, 1.0)
    births = jnp.clip(population_stats[3], 0.0, 1.0)
    deaths = jnp.clip(population_stats[4], 0.0, 1.0)
    mean_intake = jnp.clip(population_stats[5], 0.0, 1.0)
    ready = jnp.clip(
        0.45 * parent_energy + 0.35 * parent_intake + 0.20 * parent_age,
        0.0,
        1.0,
    )
    scarcity = jnp.clip(
        0.55 * jnp.clip(0.55 - mean_energy, 0.0, 1.0)
        + 0.45 * jnp.clip(0.55 - mean_intake, 0.0, 1.0),
        0.0,
        1.0,
    )
    demographic = jnp.clip(
        0.45 * pop_loss
        + 0.35 * deaths
        + 0.20 * jnp.clip(0.90 - alive, 0.0, 1.0),
        0.0,
        1.0,
    )
    stagnation = jnp.clip(
        jnp.clip(0.12 - births, 0.0, 1.0) * (0.35 + 0.65 * alive),
        0.0,
        1.0,
    )
    stress = jnp.clip(
        1.35 * demographic + 0.75 * scarcity + 0.45 * stagnation,
        0.0,
        1.0,
    )
    compact = jnp.clip(1.0 - 0.55 * node_frac - 0.45 * conn_frac, 0.0, 1.0)
    large = jnp.clip(0.55 * node_frac + 0.45 * conn_frac, 0.0, 1.0)
    rescue_need = jnp.clip(
        0.50 * demographic
        + 0.30 * scarcity
        + 0.25 * stagnation
        + 0.20 * pop_loss
        + 0.15 * deaths
        + 0.15 * jnp.clip(0.45 - births, 0.0, 1.0),
        0.0,
        1.0,
    )
    standard_probe = jnp.clip(
        (0.55 + 0.45 * jnp.clip(1.0 - evidence[2], 0.0, 1.0))
        * (0.45 + 0.55 * jnp.clip(success[2] - success[0] + 0.35, 0.0, 1.0))
        * (0.45 + 0.55 * jnp.clip(1.0 - usage[2], 0.0, 1.0)),
        0.0,
        1.0,
    )
    rescue_pulse = jnp.clip(
        ready
        * rescue_need
        * standard_probe
        * (0.45 + 0.55 * jnp.clip(usage[0] - 0.25, 0.0, 1.0)),
        0.0,
        1.0,
    )
    mutation_budget = jnp.clip(
        ready
        * stress
        * (0.30 + 0.40 * compact + 0.20 * jnp.clip(success[0], 0.0, 1.0))
        * (0.35 + 0.65 * jnp.clip(1.0 - usage[0], 0.0, 1.0))
        + 0.75 * rescue_pulse,
        0.0,
        1.0,
    )
    trial = jnp.clip(1.0 - evidence, 0.0, 1.0)
    reliable = jnp.clip(evidence * success, 0.0, 1.0)
    clone_edge = jnp.clip(success[0] * evidence[0], 0.0, 1.0)
    op_advantage = jnp.clip(
        0.60 * reliable
        + 0.30 * trial
        + 0.20 * jnp.clip(success - success[0] + 0.20, 0.0, 1.0)
        - 0.35 * usage
        - 0.15 * clone_edge,
        0.0,
        1.0,
    )
    conservative_gate = jnp.clip(
        mutation_budget * (0.65 + 0.35 * scarcity) * (0.55 + 0.45 * op_advantage[1]),
        0.0,
        1.0,
    )
    standard_gate = jnp.clip(
        mutation_budget * (0.75 + 0.25 * demographic) * (0.50 + 0.50 * op_advantage[2]),
        0.0,
        1.0,
    )
    exploratory_gate = jnp.clip(
        mutation_budget * demographic * compact * (0.30 + 0.70 * op_advantage[3]),
        0.0,
        1.0,
    )
    structural_gate = jnp.clip(
        mutation_budget * demographic * compact * (0.20 + 0.80 * op_advantage[4]),
        0.0,
        1.0,
    )
    mixed_gate = jnp.clip(
        mutation_budget
        * stress
        * jnp.clip(0.75 * compact + 0.25 * large, 0.0, 1.0)
        * (0.25 + 0.75 * op_advantage[5]),
        0.0,
        1.0,
    )
    clone_hold = jnp.clip(
        1.0
        - 0.65 * mutation_budget
        + 0.25 * jnp.clip(mean_energy - 0.45, 0.0, 1.0)
        + 0.20 * jnp.clip(mean_intake - 0.45, 0.0, 1.0)
        + 0.20 * clone_edge,
        0.0,
        1.0,
    )
    logits = jnp.array(
        [
            5.4 + 1.0 * clone_hold - 6.8 * mutation_budget - 7.2 * rescue_pulse,
            -4.5 + 4.4 * conservative_gate + 1.0 * op_advantage[1] + 1.0 * rescue_pulse,
            -4.9 + 7.4 * standard_gate + 1.5 * op_advantage[2] + 10.8 * rescue_pulse,
            -6.4 + 5.2 * exploratory_gate + 1.0 * op_advantage[3],
            -6.8 + 4.6 * structural_gate + 0.8 * op_advantage[4],
            -6.5 + 5.4 * mixed_gate + 0.9 * op_advantage[5],
        ],
        dtype=jnp.float32,
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: -0.00
score: -0.00; sham_auc_delta: -0.00; survival_rate: 0.75; mean_births: 20.19; mean_deaths: 15.06; mean_generation_gain: 3.50; operator_fraction: {'clone': 1.0, 'parametric_conservative': 0.0, 'parametric_standard': 0.0, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9992827913340401, 0.00024271502608077272, 0.0004152392006397028, 2.3171696418156737e-05, 1.4355859521481975e-05, 2.1724010580595967e-05]; post_selection_probability: [0.999267760032601, 0.0002227383896355617, 0.000454117057185981, 2.1699037310017987e-05, 1.3489957895229819e-05, 2.0189766407689275e-05]; operator_success_ema: [0.30635405611246824, 0.5, 0.5, 0.5, 0.5, 0.5]; operator_usage_ema: [0.5201807003468275, 0.0, 0.0, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.552668122574687, 0.0, 0.0, 0.0, 0.0, 0.0]; shock_auc_delta: 0.00; ancestor_survival_rate: 0.75

Text feedback:
paired ancestor delta=-0.0001; sham delta=-0.0009; shock delta=0.0000; survival=0.75; mean births=20.2; survival is scored, not a validity gate.

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
    """Tiered ecological rescue scheduler for six heredity operators."""
    success = operator_stats[0]
    usage = operator_stats[1]
    evidence = operator_stats[2]
    node_frac = jnp.clip(parent_genome_summary[0], 0.0, 1.0)
    conn_frac = jnp.clip(parent_genome_summary[1], 0.0, 1.0)
    parent_energy = jnp.clip(parent_stats[0], 0.0, 1.0)
    parent_intake = jnp.clip(parent_stats[1], 0.0, 1.0)
    parent_age = jnp.clip(parent_stats[2], 0.0, 1.0)
    alive = jnp.clip(population_stats[0], 0.0, 1.0)
    mean_energy = jnp.clip(population_stats[1], 0.0, 1.0)
    pop_change = jnp.clip(population_stats[2], -1.0, 1.0)
    births = jnp.clip(population_stats[3], 0.0, 1.0)
    deaths = jnp.clip(population_stats[4], 0.0, 1.0)
    mean_intake = jnp.clip(population_stats[5], 0.0, 1.0)
    compact = jnp.clip(1.0 - 0.58 * node_frac - 0.42 * conn_frac, 0.0, 1.0)
    complexity = jnp.clip(0.58 * node_frac + 0.42 * conn_frac, 0.0, 1.0)
    parent_ready = jnp.clip(
        0.50 * parent_energy
        + 0.30 * parent_intake
        + 0.20 * parent_age,
        0.0,
        1.0,
    )
    colony_ready = jnp.clip(
        0.55 * mean_energy
        + 0.35 * mean_intake
        + 0.10 * alive,
        0.0,
        1.0,
    )
    ready = jnp.clip(parent_ready * (0.35 + 0.65 * colony_ready), 0.0, 1.0)
    contraction = jnp.clip(-pop_change, 0.0, 1.0)
    loss_signal = jnp.clip(
        0.48 * contraction
        + 0.34 * deaths
        + 0.18 * jnp.clip(0.82 - alive, 0.0, 1.0),
        0.0,
        1.0,
    )
    starvation = jnp.clip(
        0.55 * jnp.clip(0.48 - mean_energy, 0.0, 1.0)
        + 0.45 * jnp.clip(0.42 - mean_intake, 0.0, 1.0),
        0.0,
        1.0,
    )
    reproduction_gap = jnp.clip(
        (0.16 - births) * (0.45 + 0.55 * alive),
        0.0,
        1.0,
    )
    decline = jnp.clip(
        1.15 * loss_signal
        + 0.75 * starvation
        + 0.95 * reproduction_gap,
        0.0,
        1.0,
    )
    yellow = jnp.clip((decline - 0.10) * 2.8, 0.0, 1.0)
    red = jnp.clip((decline - 0.30) * 2.6, 0.0, 1.0)
    black = jnp.clip((decline - 0.55) * 2.2, 0.0, 1.0)
    clone_quality = jnp.clip(success[0] * evidence[0], 0.0, 1.0)
    clone_overstay = jnp.clip(
        usage[0] * evidence[0] * jnp.clip(0.46 - success[0], 0.0, 1.0),
        0.0,
        1.0,
    )
    novelty = jnp.clip(1.0 - evidence, 0.0, 1.0)
    proven = jnp.clip(evidence * success, 0.0, 1.0)
    anti_crowd = jnp.clip(1.0 - usage, 0.0, 1.0)
    relative = jnp.clip(success - success[0] + 0.30, 0.0, 1.0)
    credit = jnp.clip(
        0.42 * proven
        + 0.28 * novelty
        + 0.22 * relative
        + 0.18 * anti_crowd
        - 0.20 * clone_quality,
        0.0,
        1.0,
    )
    standard_rescue = jnp.clip(
        ready
        * (0.55 * yellow + 0.45 * red)
        * (0.50 + 0.50 * compact)
        * (0.45 + 0.55 * credit[2])
        * (0.70 + 0.30 * clone_overstay),
        0.0,
        1.0,
    )
    conservative_rescue = jnp.clip(
        ready
        * yellow
        * (0.35 + 0.65 * starvation)
        * (0.30 + 0.70 * credit[1])
        * (0.35 + 0.65 * anti_crowd[1]),
        0.0,
        1.0,
    )
    exploratory_rescue = jnp.clip(
        ready
        * red
        * compact
        * (0.25 + 0.75 * credit[3])
        * (0.50 + 0.50 * novelty[3]),
        0.0,
        1.0,
    )
    structural_rescue = jnp.clip(
        ready
        * black
        * compact
        * (0.20 + 0.80 * credit[4])
        * (0.45 + 0.55 * novelty[4]),
        0.0,
        1.0,
    )
    mixed_rescue = jnp.clip(
        ready
        * red
        * jnp.clip(0.70 * compact + 0.30 * complexity, 0.0, 1.0)
        * (0.25 + 0.75 * credit[5])
        * (0.45 + 0.55 * novelty[5]),
        0.0,
        1.0,
    )
    total_mutation = jnp.clip(
        0.40 * conservative_rescue
        + 0.95 * standard_rescue
        + 0.30 * exploratory_rescue
        + 0.25 * structural_rescue
        + 0.32 * mixed_rescue,
        0.0,
        1.0,
    )
    stable_clone = jnp.clip(
        (1.0 - yellow)
        * (0.55 + 0.45 * colony_ready)
        + 0.25 * clone_quality
        - 0.30 * clone_overstay,
        0.0,
        1.0,
    )
    logits = jnp.array(
        [
            4.35 + 2.10 * stable_clone - 6.60 * total_mutation - 1.80 * red,
            -4.70 + 5.10 * conservative_rescue + 1.15 * credit[1],
            -4.35 + 8.90 * standard_rescue + 1.55 * credit[2] + 2.20 * red,
            -6.10 + 5.20 * exploratory_rescue + 0.95 * credit[3],
            -6.45 + 5.00 * structural_rescue + 0.80 * credit[4],
            -6.00 + 5.60 * mixed_rescue + 0.95 * credit[5],
        ],
        dtype=jnp.float32,
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: 0.00
score: 0.00; sham_auc_delta: 0.01; survival_rate: 0.75; mean_births: 20.19; mean_deaths: 15.06; mean_generation_gain: 3.50; operator_fraction: {'clone': 0.9969040247678018, 'parametric_conservative': 0.0, 'parametric_standard': 0.0030959752321981426, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9983372057185453, 0.00042111532735254834, 0.001064231598694973, 6.554750657147345e-05, 3.996275999644935e-05, 7.192990383105901e-05]; post_selection_probability: [0.9989567623831409, 0.00028082422785634967, 0.0006435425548450087, 4.346403447390698e-05, 2.7511343917351017e-05, 4.7881323779743025e-05]; operator_success_ema: [0.30664277262985706, 0.5, 0.4988175667822361, 0.5, 0.5, 0.5]; operator_usage_ema: [0.5173603855073452, 0.0, 0.002820312976837158, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.5509085692465305, 0.0, 0.006250001490116119, 0.0, 0.0, 0.0]; shock_auc_delta: 0.00; ancestor_survival_rate: 0.75

Text feedback:
paired ancestor delta=0.0001; sham delta=0.0070; shock delta=0.0001; survival=0.75; mean births=20.2; survival is scored, not a validity gate.


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
    """Homeostatic evidence-bandit scheduler for six heredity operators."""
    success = operator_stats[0]
    usage = operator_stats[1]
    evidence = operator_stats[2]
    parent_energy = jnp.clip(parent_stats[0], 0.0, 1.0)
    parent_intake = jnp.clip(parent_stats[1], 0.0, 1.0)
    parent_age = jnp.clip(parent_stats[2], 0.0, 1.0)
    node_frac = jnp.clip(parent_genome_summary[0], 0.0, 1.0)
    conn_frac = jnp.clip(parent_genome_summary[1], 0.0, 1.0)
    alive = jnp.clip(population_stats[0], 0.0, 1.0)
    mean_energy = jnp.clip(population_stats[1], 0.0, 1.0)
    pop_loss = jnp.clip(-population_stats[2], 0.0, 1.0)
    births = jnp.clip(population_stats[3], 0.0, 1.0)
    deaths = jnp.clip(population_stats[4], 0.0, 1.0)
    mean_intake = jnp.clip(population_stats[5], 0.0, 1.0)
    ready = jnp.clip(
        0.45 * parent_energy + 0.35 * parent_intake + 0.20 * parent_age,
        0.0,
        1.0,
    )
    scarcity = jnp.clip(
        0.55 * jnp.clip(0.55 - mean_energy, 0.0, 1.0)
        + 0.45 * jnp.clip(0.55 - mean_intake, 0.0, 1.0),
        0.0,
        1.0,
    )
    demographic = jnp.clip(
        0.45 * pop_loss
        + 0.35 * deaths
        + 0.20 * jnp.clip(0.90 - alive, 0.0, 1.0),
        0.0,
        1.0,
    )
    stagnation = jnp.clip(
        jnp.clip(0.12 - births, 0.0, 1.0) * (0.35 + 0.65 * alive),
        0.0,
        1.0,
    )
    stress = jnp.clip(
        1.35 * demographic + 0.75 * scarcity + 0.45 * stagnation,
        0.0,
        1.0,
    )
    compact = jnp.clip(1.0 - 0.55 * node_frac - 0.45 * conn_frac, 0.0, 1.0)
    large = jnp.clip(0.55 * node_frac + 0.45 * conn_frac, 0.0, 1.0)
    mutation_budget = jnp.clip(
        ready
        * stress
        * (0.30 + 0.40 * compact + 0.20 * jnp.clip(success[0], 0.0, 1.0))
        * (0.35 + 0.65 * jnp.clip(1.0 - usage[0], 0.0, 1.0)),
        0.0,
        1.0,
    )
    trial = jnp.clip(1.0 - evidence, 0.0, 1.0)
    reliable = jnp.clip(evidence * success, 0.0, 1.0)
    clone_edge = jnp.clip(success[0] * evidence[0], 0.0, 1.0)
    op_advantage = jnp.clip(
        0.60 * reliable
        + 0.30 * trial
        + 0.20 * jnp.clip(success - success[0] + 0.20, 0.0, 1.0)
        - 0.35 * usage
        - 0.15 * clone_edge,
        0.0,
        1.0,
    )
    conservative_gate = jnp.clip(
        mutation_budget * (0.65 + 0.35 * scarcity) * (0.55 + 0.45 * op_advantage[1]),
        0.0,
        1.0,
    )
    standard_gate = jnp.clip(
        mutation_budget * (0.75 + 0.25 * demographic) * (0.50 + 0.50 * op_advantage[2]),
        0.0,
        1.0,
    )
    exploratory_gate = jnp.clip(
        mutation_budget * demographic * compact * (0.30 + 0.70 * op_advantage[3]),
        0.0,
        1.0,
    )
    structural_gate = jnp.clip(
        mutation_budget * demographic * compact * (0.20 + 0.80 * op_advantage[4]),
        0.0,
        1.0,
    )
    mixed_gate = jnp.clip(
        mutation_budget
        * stress
        * jnp.clip(0.75 * compact + 0.25 * large, 0.0, 1.0)
        * (0.25 + 0.75 * op_advantage[5]),
        0.0,
        1.0,
    )
    clone_hold = jnp.clip(
        1.0
        - 0.65 * mutation_budget
        + 0.25 * jnp.clip(mean_energy - 0.45, 0.0, 1.0)
        + 0.20 * jnp.clip(mean_intake - 0.45, 0.0, 1.0)
        + 0.20 * clone_edge,
        0.0,
        1.0,
    )
    logits = jnp.array(
        [
            5.6 + 1.4 * clone_hold - 6.2 * mutation_budget,
            -4.2 + 5.0 * conservative_gate + 1.1 * op_advantage[1],
            -4.8 + 7.0 * standard_gate + 1.4 * op_advantage[2],
            -6.4 + 5.2 * exploratory_gate + 1.0 * op_advantage[3],
            -6.8 + 4.6 * structural_gate + 0.8 * op_advantage[4],
            -6.5 + 5.4 * mixed_gate + 0.9 * op_advantage[5],
        ],
        dtype=jnp.float32,
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END

```

Here are the performance metrics of the program:

Combined score to maximize: -0.00
score: -0.00; sham_auc_delta: 0.01; survival_rate: 0.75; mean_births: 24.50; mean_deaths: 16.06; mean_generation_gain: 3.69; operator_fraction: {'clone': 1.0, 'parametric_conservative': 0.0, 'parametric_standard': 0.0, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.999832878112793, 8.731464069569483e-05, 6.338931649224833e-05, 6.435727809730451e-06, 3.998347383458168e-06, 5.954096141067567e-06]; post_selection_probability: [0.9998870303756312, 5.9485965375536614e-05, 4.141361011656502e-05, 4.74728116445774e-06, 2.95646178496929e-06, 4.340951359712594e-06]; operator_success_ema: [0.2983854589983821, 0.5, 0.5, 0.5, 0.5, 0.5]; operator_usage_ema: [0.6048246137797832, 0.0, 0.0, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.6104020196944475, 0.0, 0.0, 0.0, 0.0, 0.0]; shock_auc_delta: 0.00; ancestor_survival_rate: 0.75

Here is additional text feedback about the current program:

paired ancestor delta=-0.0001; sham delta=0.0102; shock delta=0.0024; survival=0.75; mean births=24.5; survival is scored, not a validity gate.


# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
