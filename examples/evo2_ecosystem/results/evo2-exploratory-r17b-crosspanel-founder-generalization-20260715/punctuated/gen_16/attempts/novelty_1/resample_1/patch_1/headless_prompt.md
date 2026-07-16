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

Design a completely different algorithm approach to solve the same problem.
Ignore the current implementation and think of alternative algorithmic strategies that could achieve better performance.
You MUST respond using a short summary name, description and the full code:

<NAME>
A shortened name summarizing the code you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
Explain the completely different algorithmic approach you are taking and why it should perform better than the current implementation.
</DESCRIPTION>

<CODE>
```{language}
# The completely new algorithm implementation here.
```
</CODE>

* Keep the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the code.
* Your algorithm should solve the same problem but use a fundamentally different approach.
* Ensure the same inputs and outputs are maintained.
* Think outside the box - consider different data structures, algorithms, or paradigms.
* Use the <NAME>, <DESCRIPTION>, and <CODE> delimiters to structure your response. It will be parsed afterwards.

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
    standard_bad = jnp.clip(evidence[2] * (0.45 - success[2]) + 0.12 * usage[2], 0.0, 1.0)
    standard_trial = jnp.clip(
        injury_signal
        * stress
        * parent_vigor
        * (0.62 + 0.38 * uncertainty[2])
        * (1.0 - stable_signal)
        * (1.0 - 0.72 * fragile_signal)
        * (1.0 - standard_bad),
        0.0,
        1.0,
    )
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
        + 0.10 * standard_trial
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
        * (1.0 - 0.58 * standard_bad)
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
            3.65 * standard_budget + 0.72 * rescue_window * standard_edge + 0.72 * standard_trial,
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

Performance metrics:
Combined score to maximize: 0.00
score: 0.00; sham_auc_delta: 0.01; survival_rate: 1.00; mean_births: 56.50; mean_deaths: 33.81; mean_generation_gain: 5.56; operator_fraction: {'clone': 0.9922566371681416, 'parametric_conservative': 0.0055309734513274336, 'parametric_standard': 0.0022123893805309734, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9885558605194091, 0.009744741874081747, 0.0016316784811871393, 2.2096397671183306e-05, 1.7960114928428083e-05, 2.765459378549297e-05]; post_selection_probability: [0.9921749387111963, 0.006756841863325129, 0.0010218895016072308, 1.4641859397384781e-05, 1.210690608872247e-05, 1.9570843401030974e-05]; operator_success_ema: [0.08566290664020926, 0.4908840097486973, 0.4970061741769314, 0.5, 0.5, 0.5]; operator_usage_ema: [0.925118263810873, 0.003578165029466618, 0.00227265345165506, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.9417471028864384, 0.025000005960464478, 0.006250001490116119, 0.0, 0.0, 0.0]; shock_auc_delta: 0.01; ancestor_survival_rate: 1.00

Text feedback:
paired ancestor delta=0.0001; sham delta=0.0121; shock delta=0.0095; survival=1.00; mean births=56.5; survival is scored, not a validity gate.


# Current program

Here is the current program we are trying to improve (you will need to propose a new program with the same inputs and outputs as the original program, but with improved internal implementation):

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
    standard_bad = jnp.clip(evidence[2] * (0.45 - success[2]) + 0.12 * usage[2], 0.0, 1.0)
    clone_pressure = jnp.clip(evidence[0] * (0.42 - success[0]) * (0.35 + 0.65 * usage[0]), 0.0, 1.0)
    operator_confidence = jnp.clip(0.55 * lower_value[2] + 0.25 * lower_value[1] + 0.20 * lower_value[5], 0.0, 1.0)
    standard_trial = jnp.clip(
        injury_signal
        * stress
        * parent_vigor
        * (0.62 + 0.38 * uncertainty[2])
        * (0.82 + 0.18 * clone_pressure)
        * (1.0 - stable_signal)
        * (1.0 - 0.74 * fragile_signal)
        * (1.0 - standard_bad),
        0.0,
        1.0,
    )
    rescue_window = jnp.clip(
        injury_signal
        * (0.36 + 0.64 * parent_vigor)
        * (0.42 + 0.58 * stress)
        * (1.0 - 0.62 * fragile_signal)
        * (1.0 - 0.74 * stable_signal * (1.0 - injury_signal)),
        0.0,
        1.0,
    )
    rescue_release = jnp.clip(
        injury_signal
        * stress
        * (0.42 + 0.58 * parent_vigor)
        * (0.35 + 0.65 * clone_pressure)
        * (0.55 + 0.45 * operator_confidence)
        * (1.0 - 0.80 * stable_signal)
        * (1.0 - 0.68 * fragile_signal)
        * (1.0 - standard_bad),
        0.0,
        1.0,
    )
    stable_caution = jnp.clip(stable_signal * (1.0 - injury_signal) * (0.60 + 0.40 * growth), 0.0, 1.0)
    mutation_budget = jnp.clip(
        0.035
        + 0.22 * injury_signal
        + 0.26 * rescue_window
        + 0.20 * rescue_release
        + 0.08 * standard_trial
        + 0.12 * stress * parent_vigor
        + 0.08 * conservative_edge
        - 0.44 * stable_caution
        - 0.42 * fragile_signal,
        0.0,
        0.66,
    )
    standard_budget = jnp.clip(
        mutation_budget
        * (0.14 + 0.50 * injury_signal + 0.54 * rescue_window + 0.46 * rescue_release)
        * (0.28 + 0.72 * standard_edge)
        * (1.0 - 0.88 * stable_caution)
        * (1.0 - 0.60 * standard_bad)
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
        + 2.45 * stable_caution
        + 1.82 * fragile_signal
        + 0.36 * growth
        - 1.34 * mutation_budget
        - 0.82 * rescue_window * standard_edge
        - 0.90 * rescue_release * standard_edge
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
            3.70 * standard_budget + 0.72 * rescue_window * standard_edge + 0.88 * rescue_release * standard_edge + 0.64 * standard_trial,
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
            -1.42 * stable_caution - 0.54 * fragile_signal,
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
score: -0.01; sham_auc_delta: 0.00; survival_rate: 1.00; mean_births: 56.31; mean_deaths: 33.62; mean_generation_gain: 5.38; operator_fraction: {'clone': 0.9922308546059934, 'parametric_conservative': 0.005549389567147614, 'parametric_standard': 0.0022197558268590455, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9890696321214948, 0.009327497173632895, 0.0015375898751829353, 2.1309416167371508e-05, 1.7321347591600247e-05, 2.666297015301617e-05]; post_selection_probability: [0.9924922070897988, 0.006509930404507691, 0.0009530705427958994, 1.4179115544370671e-05, 1.1686769363980706e-05, 1.892232325089589e-05]; operator_success_ema: [0.09014228824526072, 0.4908840097486973, 0.49699357710778713, 0.5, 0.5, 0.5]; operator_usage_ema: [0.9247418753802776, 0.003578165029466618, 0.00227265345165506, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.9420637711882591, 0.025000005960464478, 0.006250001490116119, 0.0, 0.0, 0.0]; shock_auc_delta: -0.00; ancestor_survival_rate: 1.00

Here is additional text feedback about the current program:

paired ancestor delta=-0.0064; sham delta=0.0022; shock delta=-0.0030; survival=1.00; mean births=56.3; survival is scored, not a validity gate.


# Task

Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation.
