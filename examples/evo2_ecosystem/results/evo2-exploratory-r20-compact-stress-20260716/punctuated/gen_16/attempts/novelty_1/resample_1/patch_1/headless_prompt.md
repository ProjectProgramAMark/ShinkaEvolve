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
    clone_fatigue = jnp.clip(
        usage[0] * evidence[0] * jnp.clip(0.55 - success[0], 0.0, 1.0),
        0.0,
        1.0,
    )
    decline_pressure = jnp.clip(
        0.40 * pop_loss
        + 0.30 * deaths
        + 0.20 * jnp.clip(0.65 - mean_intake, 0.0, 1.0)
        + 0.10 * jnp.clip(0.85 - alive, 0.0, 1.0),
        0.0,
        1.0,
    )
    standard_probe = jnp.clip(
        (1.0 - evidence[2])
        * (0.50 + 0.50 * jnp.clip(success[2] - success[0] + 0.25, 0.0, 1.0))
        * (0.60 + 0.40 * jnp.clip(1.0 - usage[2], 0.0, 1.0)),
        0.0,
        1.0,
    )
    rescue_pulse = jnp.clip(
        ready
        * standard_probe
        * (0.45 * stress + 0.35 * decline_pressure + 0.20 * clone_fatigue)
        * (0.45 + 0.55 * compact),
        0.0,
        1.0,
    )
    mutation_budget = jnp.clip(
        ready
        * stress
        * (0.22 + 0.34 * compact + 0.28 * clone_fatigue + 0.16 * standard_probe)
        * (0.40 + 0.60 * jnp.clip(mean_energy + mean_intake, 0.0, 1.0))
        + 0.85 * rescue_pulse,
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
            4.9 + 1.1 * clone_hold - 7.4 * mutation_budget - 6.5 * rescue_pulse - 1.4 * clone_fatigue,
            -4.1 + 5.2 * conservative_gate + 1.0 * op_advantage[1] + 1.2 * rescue_pulse,
            -4.2 + 7.8 * standard_gate + 1.5 * op_advantage[2] + 10.5 * rescue_pulse,
            -6.5 + 4.8 * exploratory_gate + 0.9 * op_advantage[3] + 0.5 * rescue_pulse * demographic,
            -6.9 + 4.2 * structural_gate + 0.7 * op_advantage[4] + 0.4 * rescue_pulse * demographic,
            -6.2 + 5.6 * mixed_gate + 0.9 * op_advantage[5] + 0.8 * rescue_pulse * stress,
        ],
        dtype=jnp.float32,
    )
    return jnp.clip(logits, -7.0, 7.0)
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: -0.00
score: -0.00; sham_auc_delta: -0.01; survival_rate: 0.75; mean_births: 20.06; mean_deaths: 14.94; mean_generation_gain: 3.50; operator_fraction: {'clone': 0.9906542056074766, 'parametric_conservative': 0.006230529595015576, 'parametric_standard': 0.003115264797507788, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9967465365634245, 0.0007606523816802484, 0.0023741989451296188, 3.8280108984669345e-05, 2.3719788527266836e-05, 5.659026839352651e-05]; post_selection_probability: [0.9950602145975891, 0.0010278613447740623, 0.0037609509692141194, 4.849514996952163e-05, 3.0092977627037737e-05, 7.235306097453749e-05]; operator_success_ema: [0.31038595363497734, 0.49687499925494194, 0.4988175667822361, 0.5, 0.5, 0.5]; operator_usage_ema: [0.5154354907572269, 0.0014794492453802377, 0.002820312976837158, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.5485913120210171, 0.006250001490116119, 0.006250001490116119, 0.0, 0.0, 0.0]; shock_auc_delta: 0.00; ancestor_survival_rate: 0.75

Text feedback:
paired ancestor delta=-0.0009; sham delta=-0.0074; shock delta=0.0001; survival=0.75; mean births=20.1; survival is scored, not a validity gate.

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

Performance metrics:
Combined score to maximize: -0.00
score: -0.00; sham_auc_delta: 0.01; survival_rate: 0.75; mean_births: 24.50; mean_deaths: 16.06; mean_generation_gain: 3.69; operator_fraction: {'clone': 1.0, 'parametric_conservative': 0.0, 'parametric_standard': 0.0, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.999832878112793, 8.731464069569483e-05, 6.338931649224833e-05, 6.435727809730451e-06, 3.998347383458168e-06, 5.954096141067567e-06]; post_selection_probability: [0.9998870303756312, 5.9485965375536614e-05, 4.141361011656502e-05, 4.74728116445774e-06, 2.95646178496929e-06, 4.340951359712594e-06]; operator_success_ema: [0.2983854589983821, 0.5, 0.5, 0.5, 0.5, 0.5]; operator_usage_ema: [0.6048246137797832, 0.0, 0.0, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.6104020196944475, 0.0, 0.0, 0.0, 0.0, 0.0]; shock_auc_delta: 0.00; ancestor_survival_rate: 0.75

Text feedback:
paired ancestor delta=-0.0001; sham delta=0.0102; shock delta=0.0024; survival=0.75; mean births=24.5; survival is scored, not a validity gate.


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

Here are the performance metrics of the program:

Combined score to maximize: -0.00
score: -0.00; sham_auc_delta: -0.00; survival_rate: 0.75; mean_births: 20.19; mean_deaths: 15.06; mean_generation_gain: 3.50; operator_fraction: {'clone': 1.0, 'parametric_conservative': 0.0, 'parametric_standard': 0.0, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9992827913340401, 0.00024271502608077272, 0.0004152392006397028, 2.3171696418156737e-05, 1.4355859521481975e-05, 2.1724010580595967e-05]; post_selection_probability: [0.999267760032601, 0.0002227383896355617, 0.000454117057185981, 2.1699037310017987e-05, 1.3489957895229819e-05, 2.0189766407689275e-05]; operator_success_ema: [0.30635405611246824, 0.5, 0.5, 0.5, 0.5, 0.5]; operator_usage_ema: [0.5201807003468275, 0.0, 0.0, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.552668122574687, 0.0, 0.0, 0.0, 0.0, 0.0]; shock_auc_delta: 0.00; ancestor_survival_rate: 0.75

Here is additional text feedback about the current program:

paired ancestor delta=-0.0001; sham delta=-0.0009; shock delta=0.0000; survival=0.75; mean births=20.2; survival is scored, not a validity gate.


# Task

Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation.
