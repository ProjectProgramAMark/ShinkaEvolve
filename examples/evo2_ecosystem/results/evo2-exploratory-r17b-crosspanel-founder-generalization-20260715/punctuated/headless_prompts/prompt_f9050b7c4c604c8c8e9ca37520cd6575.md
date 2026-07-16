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
    prior = jnp.array([0.15, 0.61, 0.51, 0.35, 0.31, 0.41], dtype=jnp.float32)
    learned = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    unused = jnp.clip((1.0 - usage) * (1.0 - 0.52 * evidence), 0.0, 1.0)
    utility = jnp.clip(1.50 * learned + 0.26 * unused - 0.18 * usage, 0.0, 2.0)
    clone_bad = jnp.clip(evidence[0] * (0.43 - success[0]), 0.0, 1.0)
    conservative_edge = jnp.clip(learned[1] - learned[0] + 0.38 * clone_bad, 0.0, 1.0)
    standard_edge = jnp.clip(learned[2] - learned[0] + 0.28 * clone_bad, 0.0, 1.0)
    mixed_edge = jnp.clip(learned[5] - learned[0] + 0.18 * clone_bad, 0.0, 1.0)
    stable_lock = jnp.clip(stable_world * (0.60 + 0.30 * growth) * (1.0 - 0.82 * injury_rescue), 0.0, 1.0)
    renewal_param = jnp.clip(renewal_world * conservative_edge * (1.0 - 0.70 * injury_rescue), 0.0, 1.0)
    rescue_param = jnp.clip((0.70 * stress_core + 1.02 * injury_rescue + 0.16 * low_alive) * standard_edge * (1.0 - 0.40 * stable_lock), 0.0, 1.0)
    conservative_gate = jnp.clip(0.82 * renewal_param + 0.16 * injury_rescue * conservative_edge + 0.12 * stable_world * clone_bad, 0.0, 1.0)
    standard_gate = jnp.clip(
        (0.42 * probe_world * (0.45 + 0.45 * utility[2]) + 0.52 * rescue_param)
        * (1.0 - 0.46 * stable_lock),
        0.0,
        1.0,
    )
    mixed_gate = jnp.clip(
        (0.22 * probe_world * (0.34 + 0.40 * utility[5]) + 0.18 * rescue_param * mixed_edge)
        * compact
        * (1.0 - 0.54 * stable_lock),
        0.0,
        1.0,
    )
    exploratory_gate = jnp.clip(
        probe_world * unused[3] * (0.14 + 0.24 * compact) * injury_rescue * (1.0 - 0.58 * stable_lock),
        0.0,
        1.0,
    )
    structural_gate = jnp.clip(
        probe_world * unused[4] * compact * 0.15 * injury_rescue * (1.0 - 0.58 * stable_lock),
        0.0,
        1.0,
    )
    anchor = jnp.array([4.50, -1.02, -2.70, -6.05, -6.34, -5.94], dtype=jnp.float32)
    stable_policy = jnp.array([2.42, 0.58, -1.48, -1.96, -2.08, -1.78], dtype=jnp.float32)
    renewal_policy = jnp.array([-0.58, 1.86, 0.14, -0.36, -0.44, -0.24], dtype=jnp.float32)
    stress_policy = jnp.array([-2.72, -0.02, 0.96, 0.45, 0.28, 0.58], dtype=jnp.float32)
    rescue_policy = jnp.array([-1.22, 0.28, 1.82, 0.46, 0.22, 0.82], dtype=jnp.float32)
    utility_policy = jnp.array([0.24, 0.78, 1.00, 0.36, 0.28, 0.46], dtype=jnp.float32)
    pulse = jnp.array(
        [
            -0.92 * standard_gate - 0.52 * mixed_gate - 1.54 * renewal_param - 0.90 * clone_bad * injury_rescue - 0.26 * stable_world * clone_bad,
            2.24 * conservative_gate + 0.16 * injury_rescue + 0.16 * stable_lock * conservative_edge,
            2.46 * standard_gate + 0.70 * rescue_param - 0.42 * stable_lock,
            0.70 * exploratory_gate,
            0.48 * structural_gate,
            1.18 * mixed_gate,
        ],
        dtype=jnp.float32,
    )
    growth_trim = jnp.array(
        [0.12 * growth, 0.04 * growth, -0.24 * growth, -0.20 * growth, -0.18 * growth, 0.06 * growth],
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
Combined score to maximize: -0.05
score: -0.05; sham_auc_delta: -0.03; survival_rate: 1.00; mean_births: 53.81; mean_deaths: 32.25; mean_generation_gain: 5.50; operator_fraction: {'clone': 0.9419279907084785, 'parametric_conservative': 0.05574912891986063, 'parametric_standard': 0.0023228803716608595, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.9495696172322312, 0.043692087882185635, 0.006589327364751737, 4.82398242377102e-05, 2.9013802541407107e-05, 7.171333330517558e-05]; post_selection_probability: [0.9496735579483992, 0.04816015191845127, 0.0021112117160867142, 1.7549843980012448e-05, 1.2115515944372107e-05, 2.5375580034134062e-05]; operator_success_ema: [0.09563396940939128, 0.4592452831566334, 0.4937499985098839, 0.5, 0.5, 0.5]; operator_usage_ema: [0.861191313713789, 0.0588186095119454, 0.0003376032691448927, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.9220747575163841, 0.13628496881574392, 0.012500002980232239, 0.0, 0.0, 0.0]; shock_auc_delta: -0.03; ancestor_survival_rate: 1.00

Text feedback:
paired ancestor delta=-0.0542; sham delta=-0.0320; shock delta=-0.0328; survival=1.00; mean births=53.8; survival is scored, not a validity gate.


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
    prior = jnp.array([0.18, 0.62, 0.48, 0.28, 0.24, 0.34], dtype=jnp.float32)
    learned = jnp.clip(evidence * success + (1.0 - evidence) * prior, 0.0, 1.0)
    credible = jnp.clip(0.22 + 0.78 * evidence, 0.0, 1.0)
    unused = jnp.clip((1.0 - usage) * (1.0 - 0.58 * evidence), 0.0, 1.0)
    utility = jnp.clip((1.42 * learned + 0.22 * unused - 0.24 * usage) * credible, 0.0, 2.0)
    clone_bad = jnp.clip(evidence[0] * (0.46 - success[0]) + 0.20 * usage[0] * evidence[0] * (learned[1] - learned[0]), 0.0, 1.0)
    conservative_edge = jnp.clip((learned[1] - learned[0]) * (0.55 + 0.45 * evidence[1]) + 0.46 * clone_bad, 0.0, 1.0)
    standard_edge = jnp.clip((learned[2] - learned[0]) * (0.38 + 0.62 * evidence[2]) + 0.30 * clone_bad + 0.16 * unused[2] * injury_rescue, 0.0, 1.0)
    mixed_edge = jnp.clip((learned[5] - learned[0]) * (0.36 + 0.64 * evidence[5]) + 0.16 * clone_bad, 0.0, 1.0)
    stable_lock = jnp.clip(stable_world * (0.72 + 0.22 * growth) * (1.0 - 0.90 * injury_rescue) * (1.0 - 0.34 * clone_bad), 0.0, 1.0)
    fragile_lock = jnp.clip(stable_world * (1.0 - parent_vigor) * (0.48 + 0.52 * (1.0 - compact)), 0.0, 1.0)
    renewal_param = jnp.clip(renewal_world * conservative_edge * (1.0 - 0.72 * injury_rescue) * (1.0 - 0.35 * fragile_lock), 0.0, 1.0)
    rescue_param = jnp.clip((0.74 * stress_core + 1.12 * injury_rescue + 0.18 * low_alive) * standard_edge * (1.0 - 0.52 * stable_lock) * (1.0 - 0.42 * fragile_lock), 0.0, 1.0)
    conservative_gate = jnp.clip(1.00 * renewal_param + 0.20 * injury_rescue * conservative_edge + 0.22 * stable_world * clone_bad * conservative_edge + 0.10 * fragile_lock * conservative_edge, 0.0, 1.0)
    standard_gate = jnp.clip(
        (0.36 * probe_world * (0.36 + 0.52 * utility[2]) + 0.62 * rescue_param)
        * (1.0 - 0.62 * stable_lock)
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
            -1.02 * standard_gate - 0.58 * mixed_gate - 1.26 * renewal_param - 1.12 * clone_bad * injury_rescue - 0.34 * stable_world * clone_bad + 0.34 * fragile_lock,
            2.56 * conservative_gate + 0.10 * injury_rescue + 0.22 * stable_lock * conservative_edge + 0.18 * fragile_lock,
            2.74 * standard_gate + 0.82 * rescue_param - 0.58 * stable_lock - 0.44 * fragile_lock,
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

Here are the performance metrics of the program:

Combined score to maximize: -0.01
score: -0.01; sham_auc_delta: -0.01; survival_rate: 1.00; mean_births: 58.75; mean_deaths: 34.75; mean_generation_gain: 5.69; operator_fraction: {'clone': 0.6680851063829787, 'parametric_conservative': 0.3276595744680851, 'parametric_standard': 0.00425531914893617, 'parametric_exploratory': 0.0, 'structural': 0.0, 'mixed': 0.0}; pre_selection_probability: [0.7957204627990723, 0.19624717553456625, 0.007580253879229228, 0.00013933569348106782, 9.684570832177997e-05, 0.00021589803509414196]; post_selection_probability: [0.6439878946618188, 0.35372707994678354, 0.0021088508724034587, 5.456678085972237e-05, 4.912952338404293e-05, 7.248309034193996e-05]; operator_success_ema: [0.11330145667307079, 0.293432064820081, 0.48749999701976776, 0.5, 0.5, 0.5]; operator_usage_ema: [0.5637496430426836, 0.38157287053763866, 0.001259637385373935, 0.0, 0.0, 0.0]; operator_evidence_ema: [0.9078761860728264, 0.5414526667445898, 0.025000005960464478, 0.0, 0.0, 0.0]; shock_auc_delta: 0.03; ancestor_survival_rate: 1.00

Here is additional text feedback about the current program:

paired ancestor delta=-0.0053; sham delta=-0.0131; shock delta=0.0270; survival=1.00; mean births=58.8; survival is scored, not a validity gate.


# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
