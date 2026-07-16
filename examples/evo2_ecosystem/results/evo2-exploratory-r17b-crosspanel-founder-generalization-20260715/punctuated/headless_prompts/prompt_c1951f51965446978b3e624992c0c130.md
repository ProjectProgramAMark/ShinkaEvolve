# System Instructions


Discover a better six-action heredity scheduler for an embodied CPPN ecosystem.

make_offspring receives bounded parent, population, and per-operator success,
usage, and evidence summaries. Return six finite logits for clone,
conservative parametric, standard parametric, exploratory parametric,
structural, and mixed mutation. The paired ancestor for this run is
exact frozen clone. The Shinka-generated R16b generation-11 scheduler beat clone on untouched development founders in all three repeats but missed the sealed robust aggregate because one injury founder/world produced a large negative tail. Continuous standard mutation helps injury and harms sham worlds. Improve worst-founder consistency across this eight-founder cross-panel: use ecological stress and accumulated operator success, usage, and evidence to preserve injury benefit while avoiding costly or unlucky mutation in stable and fragile founder regimes. Do not merely increase average mutation or clone unconditionally.. Maximize the paired candidate-minus-ancestor ecological
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
    """Risk-budgeted lower-confidence heredity scheduler for six actions."""
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
    low_alive = jnp.clip(1.0 - alive_fraction, 0.0, 1.0)
    death = jnp.clip(death_rate_ema, 0.0, 1.0)
    energy_gap = jnp.clip(0.56 - mean_energy_fraction, 0.0, 1.0)
    intake_gap = jnp.clip(0.54 - mean_intake_ema, 0.0, 1.0)
    birth_gap = jnp.clip(0.28 - birth_rate_ema, 0.0, 1.0)
    parent_vigor = jnp.clip(
        0.52 * energy_fraction + 0.36 * intake_ema + 0.12 * (1.0 - age_fraction),
        0.0,
        1.0,
    )
    compact = jnp.clip(1.0 - 0.50 * node_fraction - 0.50 * connection_fraction, 0.0, 1.0)
    stress = jnp.clip(
        0.52 * decline
        + 0.42 * death
        + 0.26 * low_alive
        + 0.24 * intake_gap
        + 0.18 * energy_gap
        + 0.10 * birth_gap,
        0.0,
        1.0,
    )
    injury_signal = jnp.clip(
        (0.62 * decline + 0.48 * death + 0.22 * low_alive)
        * (0.30 + 0.46 * intake_gap + 0.24 * energy_gap),
        0.0,
        1.0,
    )
    stable_signal = jnp.clip(
        alive_fraction
        * (1.0 - decline)
        * (1.0 - death)
        * (0.42 + 0.58 * mean_energy_fraction)
        * (0.48 + 0.52 * mean_intake_ema),
        0.0,
        1.0,
    )
    fragile_signal = jnp.clip(
        (0.46 * low_alive + 0.34 * energy_gap + 0.30 * intake_gap)
        * (1.0 - parent_vigor)
        + 0.28 * death,
        0.0,
        1.0,
    )
    prior = jnp.array([0.19, 0.55, 0.48, 0.31, 0.28, 0.36], dtype=jnp.float32)
    risk = jnp.array([0.00, 0.16, 0.39, 0.72, 0.82, 0.55], dtype=jnp.float32)
    posterior = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    uncertainty = jnp.clip((1.0 - evidence) * (1.0 - 0.35 * usage), 0.0, 1.0)
    lower_value = jnp.clip(posterior - 0.18 * uncertainty - 0.12 * usage * risk, 0.0, 1.0)
    clone_floor = jnp.clip(0.20 + 0.55 * evidence[0] * success[0], 0.0, 1.0)
    clone_underperform = jnp.clip(evidence[0] * (0.36 - success[0]), 0.0, 1.0)
    conservative_edge = jnp.clip(lower_value[1] - clone_floor + 0.24 * clone_underperform, 0.0, 1.0)
    standard_edge = jnp.clip(lower_value[2] - clone_floor + 0.20 * clone_underperform, 0.0, 1.0)
    mixed_edge = jnp.clip(lower_value[5] - clone_floor + 0.10 * clone_underperform, 0.0, 1.0)
    rescue_confidence = jnp.clip(
        clone_underperform
        * stress
        * parent_vigor
        * (0.42 + 0.58 * injury_signal)
        * (1.0 - 0.70 * stable_signal * (1.0 - injury_signal))
        * (1.0 - 0.58 * fragile_signal),
        0.0,
        1.0,
    )
    mutation_budget = jnp.clip(
        0.05
        + 0.32 * injury_signal
        + 0.18 * stress * parent_vigor
        + 0.16 * rescue_confidence
        + 0.10 * conservative_edge
        - 0.31 * stable_signal * (1.0 - injury_signal)
        - 0.34 * fragile_signal,
        0.0,
        0.62,
    )
    standard_budget = jnp.clip(
        mutation_budget
        * (0.20 + 0.68 * injury_signal + 0.42 * rescue_confidence)
        * (0.34 + 0.66 * standard_edge)
        * (1.0 - 0.70 * stable_signal * (1.0 - injury_signal))
        * (0.48 + 0.52 * parent_vigor),
        0.0,
        1.0,
    )
    conservative_budget = jnp.clip(
        mutation_budget
        * (0.50 + 0.44 * stable_signal + 0.18 * rescue_confidence)
        * (0.42 + 0.58 * conservative_edge)
        * (1.0 - 0.42 * injury_signal),
        0.0,
        1.0,
    )
    mixed_budget = jnp.clip(
        mutation_budget
        * injury_signal
        * compact
        * (0.24 + 0.76 * mixed_edge)
        * (1.0 - 0.55 * fragile_signal),
        0.0,
        1.0,
    )
    rare_probe = jnp.clip(
        mutation_budget
        * injury_signal
        * compact
        * (1.0 - stable_signal)
        * (1.0 - fragile_signal)
        * 0.10,
        0.0,
        1.0,
    )
    clone_guard = jnp.clip(
        1.0
        + 2.15 * stable_signal * (1.0 - injury_signal)
        + 1.55 * fragile_signal
        + 0.32 * growth
        - 1.55 * mutation_budget
        - 0.48 * rescue_confidence
        - 0.58 * clone_underperform,
        -2.0,
        4.0,
    )
    base = jnp.array([2.95, -0.82, -2.25, -6.30, -6.45, -6.10], dtype=jnp.float32)
    value_push = jnp.array(
        [
            -0.30 * conservative_edge - 0.46 * standard_edge - 0.18 * mixed_edge,
            2.55 * conservative_edge,
            2.25 * standard_edge,
            0.58 * lower_value[3] * uncertainty[3],
            0.42 * lower_value[4] * uncertainty[4],
            1.15 * mixed_edge,
        ],
        dtype=jnp.float32,
    )
    budget_push = jnp.array(
        [
            clone_guard,
            3.15 * conservative_budget + 0.25 * stable_signal * conservative_edge,
            3.45 * standard_budget + 0.52 * injury_signal * standard_edge + 0.58 * rescue_confidence * standard_edge,
            1.10 * rare_probe,
            0.82 * rare_probe * compact,
            2.05 * mixed_budget,
        ],
        dtype=jnp.float32,
    )
    risk_brake = jnp.array(
        [
            0.0,
            -0.10 * fragile_signal,
            -1.05 * stable_signal * (1.0 - injury_signal) - 0.46 * fragile_signal,
            -1.25 * stable_signal - 0.92 * fragile_signal,
            -1.35 * stable_signal - 1.05 * fragile_signal,
            -0.76 * stable_signal * (1.0 - injury_signal) - 0.64 * fragile_signal,
        ],
        dtype=jnp.float32,
    )
    logits = base + value_push + budget_push + risk_brake
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: -0.01
score: -0.01; sham_auc_delta: -0.01; survival_rate: 1.00; mean_births: 52.94; mean_deaths: 32.19; mean_generation_gain: 5.75; operator_fraction: {'clone': 0.9905548996458088, 'parametric_conservative': 0.008264462809917356, 'parametric_standard': 0.0011806375442739079, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9880288870711076, 0.01029246467116632, 0.0016083021856550324, 2.2918598532621553e-05, 1.8622431525727734e-05, 2.8796947031208364e-05]; post_selection_probability: [0.9925747452879982, 0.006470387702365573, 0.0009102783400377781, 1.4176994533673358e-05, 1.1668911056338455e-05, 1.8730151455299994e-05]; operator_success_ema: [0.1288941774982959, 0.49114320427179337, 0.5, 0.5, 0.5, 0.5]; operator_usage_ema: [0.8986866660416126, 0.006495013658422977, 0.00124129478354007, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.9122335612773895, 0.030000006780028343, 0.0, 0.0, 0.0, 0.0]; shock_auc_delta: -0.01; ancestor_survival_rate: 1.00

Text feedback:
paired ancestor delta=-0.0140; sham delta=-0.0084; shock delta=-0.0149; survival=1.00; mean births=52.9; survival is scored, not a validity gate.

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
    """Risk-budgeted lower-confidence heredity scheduler for six actions."""
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
    low_alive = jnp.clip(1.0 - alive_fraction, 0.0, 1.0)
    death = jnp.clip(death_rate_ema, 0.0, 1.0)
    energy_gap = jnp.clip(0.56 - mean_energy_fraction, 0.0, 1.0)
    intake_gap = jnp.clip(0.54 - mean_intake_ema, 0.0, 1.0)
    birth_gap = jnp.clip(0.28 - birth_rate_ema, 0.0, 1.0)
    parent_vigor = jnp.clip(
        0.52 * energy_fraction + 0.36 * intake_ema + 0.12 * (1.0 - age_fraction),
        0.0,
        1.0,
    )
    compact = jnp.clip(1.0 - 0.50 * node_fraction - 0.50 * connection_fraction, 0.0, 1.0)
    stress = jnp.clip(
        0.52 * decline
        + 0.42 * death
        + 0.26 * low_alive
        + 0.24 * intake_gap
        + 0.18 * energy_gap
        + 0.10 * birth_gap,
        0.0,
        1.0,
    )
    injury_signal = jnp.clip(
        (0.62 * decline + 0.48 * death + 0.22 * low_alive)
        * (0.30 + 0.46 * intake_gap + 0.24 * energy_gap),
        0.0,
        1.0,
    )
    stable_signal = jnp.clip(
        alive_fraction
        * (1.0 - decline)
        * (1.0 - death)
        * (0.42 + 0.58 * mean_energy_fraction)
        * (0.48 + 0.52 * mean_intake_ema),
        0.0,
        1.0,
    )
    fragile_signal = jnp.clip(
        (0.46 * low_alive + 0.34 * energy_gap + 0.30 * intake_gap)
        * (1.0 - parent_vigor)
        + 0.28 * death,
        0.0,
        1.0,
    )
    prior = jnp.array([0.19, 0.55, 0.48, 0.31, 0.28, 0.36], dtype=jnp.float32)
    risk = jnp.array([0.00, 0.16, 0.39, 0.72, 0.82, 0.55], dtype=jnp.float32)
    posterior = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    uncertainty = jnp.clip((1.0 - evidence) * (1.0 - 0.35 * usage), 0.0, 1.0)
    lower_value = jnp.clip(posterior - 0.18 * uncertainty - 0.12 * usage * risk, 0.0, 1.0)
    clone_floor = jnp.clip(0.20 + 0.55 * evidence[0] * success[0], 0.0, 1.0)
    conservative_edge = jnp.clip(lower_value[1] - clone_floor + 0.22 * evidence[0] * (0.35 - success[0]), 0.0, 1.0)
    standard_edge = jnp.clip(lower_value[2] - clone_floor + 0.16 * evidence[0] * (0.35 - success[0]), 0.0, 1.0)
    mixed_edge = jnp.clip(lower_value[5] - clone_floor + 0.10 * evidence[0] * (0.35 - success[0]), 0.0, 1.0)
    mutation_budget = jnp.clip(
        0.05
        + 0.34 * injury_signal
        + 0.18 * stress * parent_vigor
        + 0.10 * conservative_edge
        - 0.30 * stable_signal * (1.0 - injury_signal)
        - 0.34 * fragile_signal,
        0.0,
        0.62,
    )
    standard_budget = jnp.clip(
        mutation_budget
        * (0.22 + 0.78 * injury_signal)
        * (0.36 + 0.64 * standard_edge)
        * (1.0 - 0.70 * stable_signal * (1.0 - injury_signal))
        * (0.48 + 0.52 * parent_vigor),
        0.0,
        1.0,
    )
    conservative_budget = jnp.clip(
        mutation_budget
        * (0.52 + 0.48 * stable_signal)
        * (0.42 + 0.58 * conservative_edge)
        * (1.0 - 0.42 * injury_signal),
        0.0,
        1.0,
    )
    mixed_budget = jnp.clip(
        mutation_budget
        * injury_signal
        * compact
        * (0.24 + 0.76 * mixed_edge)
        * (1.0 - 0.55 * fragile_signal),
        0.0,
        1.0,
    )
    rare_probe = jnp.clip(
        mutation_budget
        * injury_signal
        * compact
        * (1.0 - stable_signal)
        * (1.0 - fragile_signal)
        * 0.10,
        0.0,
        1.0,
    )
    clone_guard = jnp.clip(
        1.0
        + 2.15 * stable_signal * (1.0 - injury_signal)
        + 1.55 * fragile_signal
        + 0.32 * growth
        - 1.65 * mutation_budget
        - 0.62 * evidence[0] * (0.35 - success[0]),
        -2.0,
        4.0,
    )
    base = jnp.array([2.95, -0.82, -2.25, -6.30, -6.45, -6.10], dtype=jnp.float32)
    value_push = jnp.array(
        [
            -0.30 * conservative_edge - 0.46 * standard_edge - 0.18 * mixed_edge,
            2.55 * conservative_edge,
            2.25 * standard_edge,
            0.58 * lower_value[3] * uncertainty[3],
            0.42 * lower_value[4] * uncertainty[4],
            1.15 * mixed_edge,
        ],
        dtype=jnp.float32,
    )
    budget_push = jnp.array(
        [
            clone_guard,
            3.15 * conservative_budget + 0.25 * stable_signal * conservative_edge,
            3.35 * standard_budget + 0.45 * injury_signal * standard_edge,
            1.10 * rare_probe,
            0.82 * rare_probe * compact,
            2.05 * mixed_budget,
        ],
        dtype=jnp.float32,
    )
    risk_brake = jnp.array(
        [
            0.0,
            -0.10 * fragile_signal,
            -1.05 * stable_signal * (1.0 - injury_signal) - 0.46 * fragile_signal,
            -1.25 * stable_signal - 0.92 * fragile_signal,
            -1.35 * stable_signal - 1.05 * fragile_signal,
            -0.76 * stable_signal * (1.0 - injury_signal) - 0.64 * fragile_signal,
        ],
        dtype=jnp.float32,
    )
    logits = base + value_push + budget_push + risk_brake
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: -0.00
score: -0.00; sham_auc_delta: 0.00; survival_rate: 1.00; mean_births: 56.12; mean_deaths: 33.44; mean_generation_gain: 5.44; operator_fraction: {'clone': 0.9866369710467706, 'parametric_conservative': 0.011135857461024499, 'parametric_standard': 0.0022271714922048997, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9874971321650914, 0.0107393957142319, 0.0016907152320657457, 2.3741919929826897e-05, 1.929686529495354e-05, 2.9720442086857346e-05]; post_selection_probability: [0.992812761845249, 0.0062678220404169495, 0.0008759436819279729, 1.3801685556226304e-05, 1.135103978612276e-05, 1.8299615706653778e-05]; operator_success_ema: [0.09541275771334767, 0.4793633781373501, 0.49699357710778713, 0.5, 0.5, 0.5]; operator_usage_ema: [0.9184901230037212, 0.009292905517213512, 0.002179100818466395, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.938767172396183, 0.04937501158565283, 0.006250001490116119, 0.0, 0.0, 0.0]; shock_auc_delta: 0.01; ancestor_survival_rate: 1.00

Text feedback:
paired ancestor delta=-0.0018; sham delta=0.0032; shock delta=0.0069; survival=1.00; mean births=56.1; survival is scored, not a validity gate.


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
    """Risk-budgeted lower-confidence heredity scheduler for six actions."""
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
    low_alive = jnp.clip(1.0 - alive_fraction, 0.0, 1.0)
    death = jnp.clip(death_rate_ema, 0.0, 1.0)
    energy_gap = jnp.clip(0.56 - mean_energy_fraction, 0.0, 1.0)
    intake_gap = jnp.clip(0.54 - mean_intake_ema, 0.0, 1.0)
    birth_gap = jnp.clip(0.28 - birth_rate_ema, 0.0, 1.0)
    parent_vigor = jnp.clip(
        0.52 * energy_fraction + 0.36 * intake_ema + 0.12 * (1.0 - age_fraction),
        0.0,
        1.0,
    )
    compact = jnp.clip(1.0 - 0.50 * node_fraction - 0.50 * connection_fraction, 0.0, 1.0)
    stress = jnp.clip(
        0.52 * decline
        + 0.42 * death
        + 0.26 * low_alive
        + 0.24 * intake_gap
        + 0.18 * energy_gap
        + 0.10 * birth_gap,
        0.0,
        1.0,
    )
    injury_signal = jnp.clip(
        (0.62 * decline + 0.48 * death + 0.22 * low_alive)
        * (0.30 + 0.46 * intake_gap + 0.24 * energy_gap),
        0.0,
        1.0,
    )
    stable_signal = jnp.clip(
        alive_fraction
        * (1.0 - decline)
        * (1.0 - death)
        * (0.42 + 0.58 * mean_energy_fraction)
        * (0.48 + 0.52 * mean_intake_ema),
        0.0,
        1.0,
    )
    fragile_signal = jnp.clip(
        (0.46 * low_alive + 0.34 * energy_gap + 0.30 * intake_gap)
        * (1.0 - parent_vigor)
        + 0.28 * death,
        0.0,
        1.0,
    )
    prior = jnp.array([0.19, 0.55, 0.48, 0.31, 0.28, 0.36], dtype=jnp.float32)
    risk = jnp.array([0.00, 0.16, 0.39, 0.72, 0.82, 0.55], dtype=jnp.float32)
    posterior = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    uncertainty = jnp.clip((1.0 - evidence) * (1.0 - 0.35 * usage), 0.0, 1.0)
    lower_value = jnp.clip(posterior - 0.18 * uncertainty - 0.12 * usage * risk, 0.0, 1.0)
    clone_floor = jnp.clip(0.20 + 0.55 * evidence[0] * success[0], 0.0, 1.0)
    conservative_edge = jnp.clip(lower_value[1] - clone_floor + 0.22 * evidence[0] * (0.35 - success[0]), 0.0, 1.0)
    standard_edge = jnp.clip(lower_value[2] - clone_floor + 0.16 * evidence[0] * (0.35 - success[0]), 0.0, 1.0)
    mixed_edge = jnp.clip(lower_value[5] - clone_floor + 0.10 * evidence[0] * (0.35 - success[0]), 0.0, 1.0)
    rescue_window = jnp.clip(
        injury_signal
        * (0.36 + 0.64 * parent_vigor)
        * (0.42 + 0.58 * stress)
        * (1.0 - 0.62 * fragile_signal)
        * (1.0 - 0.74 * stable_signal * (1.0 - injury_signal)),
        0.0,
        1.0,
    )
    stable_caution = jnp.clip(stable_signal * (1.0 - injury_signal) * (0.58 + 0.42 * growth), 0.0, 1.0)
    mutation_budget = jnp.clip(
        0.035
        + 0.24 * injury_signal
        + 0.30 * rescue_window
        + 0.12 * stress * parent_vigor
        + 0.08 * conservative_edge
        - 0.40 * stable_caution
        - 0.38 * fragile_signal,
        0.0,
        0.66,
    )
    standard_budget = jnp.clip(
        mutation_budget
        * (0.16 + 0.52 * injury_signal + 0.58 * rescue_window)
        * (0.28 + 0.72 * standard_edge)
        * (1.0 - 0.84 * stable_caution)
        * (0.46 + 0.54 * parent_vigor),
        0.0,
        1.0,
    )
    conservative_budget = jnp.clip(
        mutation_budget
        * (0.48 + 0.36 * stable_signal + 0.28 * rescue_window)
        * (0.42 + 0.58 * conservative_edge)
        * (1.0 - 0.36 * injury_signal)
        * (1.0 - 0.34 * stable_caution),
        0.0,
        1.0,
    )
    mixed_budget = jnp.clip(
        mutation_budget
        * injury_signal
        * compact
        * (0.24 + 0.76 * mixed_edge)
        * (1.0 - 0.55 * fragile_signal),
        0.0,
        1.0,
    )
    rare_probe = jnp.clip(
        mutation_budget
        * injury_signal
        * compact
        * (1.0 - stable_signal)
        * (1.0 - fragile_signal)
        * 0.10,
        0.0,
        1.0,
    )
    clone_guard = jnp.clip(
        1.12
        + 2.35 * stable_caution
        + 1.70 * fragile_signal
        + 0.36 * growth
        - 1.38 * mutation_budget
        - 0.82 * rescue_window * standard_edge
        - 0.54 * evidence[0] * (0.35 - success[0]),
        -2.0,
        4.0,
    )
    base = jnp.array([2.95, -0.82, -2.25, -6.30, -6.45, -6.10], dtype=jnp.float32)
    value_push = jnp.array(
        [
            -0.30 * conservative_edge - 0.46 * standard_edge - 0.18 * mixed_edge,
            2.55 * conservative_edge,
            2.25 * standard_edge,
            0.58 * lower_value[3] * uncertainty[3],
            0.42 * lower_value[4] * uncertainty[4],
            1.15 * mixed_edge,
        ],
        dtype=jnp.float32,
    )
    budget_push = jnp.array(
        [
            clone_guard,
            3.15 * conservative_budget + 0.25 * stable_signal * conservative_edge,
            3.65 * standard_budget + 0.72 * rescue_window * standard_edge,
            1.10 * rare_probe,
            0.82 * rare_probe * compact,
            2.05 * mixed_budget,
        ],
        dtype=jnp.float32,
    )
    risk_brake = jnp.array(
        [
            0.0,
            -0.10 * fragile_signal,
            -1.28 * stable_caution - 0.50 * fragile_signal,
            -1.25 * stable_signal - 0.92 * fragile_signal,
            -1.35 * stable_signal - 1.05 * fragile_signal,
            -0.76 * stable_signal * (1.0 - injury_signal) - 0.64 * fragile_signal,
        ],
        dtype=jnp.float32,
    )
    logits = base + value_push + budget_push + risk_brake
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Here are the performance metrics of the program:

Combined score to maximize: -0.01
score: -0.01; sham_auc_delta: 0.00; survival_rate: 1.00; mean_births: 54.56; mean_deaths: 33.38; mean_generation_gain: 5.75; operator_fraction: {'clone': 0.9885452462772051, 'parametric_conservative': 0.010309278350515464, 'parametric_standard': 0.001145475372279496, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9888983855122014, 0.009472147953745565, 0.0015630416057415698, 2.164286170706251e-05, 1.7586907999525703e-05, 2.71927969822029e-05]; post_selection_probability: [0.9922097069876534, 0.0067335306234068745, 0.0010108417997991823, 1.4490669224431917e-05, 1.2023889911122072e-05, 1.938466973093838e-05]; operator_success_ema: [0.12557107373140752, 0.4844752438366413, 0.5, 0.5, 0.5, 0.5]; operator_usage_ema: [0.9025752358138561, 0.008515870664268732, 0.00124129478354007, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.91295076161623, 0.03625000827014446, 0.0, 0.0, 0.0, 0.0]; shock_auc_delta: -0.00; ancestor_survival_rate: 1.00

Here is additional text feedback about the current program:

paired ancestor delta=-0.0114; sham delta=0.0034; shock delta=-0.0018; survival=1.00; mean births=54.6; survival is scored, not a validity gate.


# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
