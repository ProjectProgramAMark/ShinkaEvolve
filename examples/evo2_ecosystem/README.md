# Evo²-Ecosystem heredity task

ShinkaEvolve edits only `make_offspring` in `initial.py`. The function receives
bounded ecological summaries and returns four scores for trusted clone,
parametric, structural, and mixed TensorNEAT mutation kernels. It never receives
CPPN graph tensors, raw genomes, simulator state, scenario identity, files, or
non-training manifests.

## Canonical matched run

`run_spec.json` is canonical JSON and `run_spec.sha256` authenticates its exact
bytes. The one shared spec fixes both arms' run ID, 15-generation budget, outer
seed, model, three numerical evaluations per candidate, top-K of three,
training/development/sealed manifest hashes, trusted source hashes, archive
settings, bootstrap configuration, and the preregistered analysis-plan hash.
It also binds `microcosmos/uv.lock`, which pins TensorNEAT to commit
`7e872c7191699516a9db7e3c7e6e991fd622386b`. The launcher pins the proposal package
to `npx -y @roberttlange/headless@0.4.0` and records the exact command under
`results/<run_id>/<regime>/launch.json`.
The sealed comparison set is fixed to clone, fixed parametric, fixed mixed, and
stress-responsive mutation baselines.

With no additional argument, every launcher and freezer continues to select
this schema-v1 pilot specification. Prospective studies use schema-v2 specs,
selected either explicitly with `--run-spec /absolute/path/to/spec.json` or by
portable name with `--profile NAME` (resolved as `run_specs/NAME.json`). The two
forms are mutually exclusive. A schema-v2 spec additionally binds exact paths
and hashes for all four manifests, the founder-bank index, the preregistration,
the implementation plan, every trusted source, the candidate output width, and
the result/frozen artifact roots. No production schema-v2 profile is checked in
until those prospective artifacts have been generated and frozen.

The launcher materializes the selected canonical spec under its run root, then
passes only that materialized path and its SHA-256 to evaluator subprocesses.
Those arguments are trusted job configuration, not candidate inputs. Candidate
code still receives only the four bounded numeric summaries. Holdout permission
checks, development selection, frozen output, and lineage export all resolve
their paths from the same selected spec. For example, after a profile exists:

```bash
conda run -n sakana python examples/evo2_ecosystem/run_evo.py \
  --regime stable --profile actuator
conda run -n sakana python examples/evo2_ecosystem/freeze_finalist.py \
  --regime stable --profile actuator
conda run -n sakana python examples/evo2_ecosystem/program_lineage.py \
  --regime stable --profile actuator
```

Before either prospective search, make both holdout manifests and every
development/sealed founder genotype unreadable. The founder index remains
readable so its canonical hash and partition metadata can be authenticated;
training evaluation opens only the training artifacts named by its manifest.

```bash
chmod 000 \
  ../microcosmos/experiments/evo2_ecosystem/heredity_adaptation/manifests/development.json \
  ../microcosmos/experiments/evo2_ecosystem/heredity_adaptation/manifests/sealed.json \
  ../microcosmos/experiments/evo2_ecosystem/founders/heredity_adaptation/dev-*.npz \
  ../microcosmos/experiments/evo2_ecosystem/founders/heredity_adaptation/sealed-*.npz
```

Run the first fresh arm:

```bash
systemd-run --user --scope --quiet \
  -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=2G \
  conda run -n sakana python examples/evo2_ecosystem/run_evo.py --regime stable
```

After it completes and writes `stable.complete.json`, isolate its entire arm
before starting the second:

```bash
chmod 000 \
  examples/evo2_ecosystem/results/evo2-production-20260712-r1/stable

systemd-run --user --scope --quiet \
  -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=2G \
  conda run -n sakana python examples/evo2_ecosystem/run_evo.py --regime punctuated
```

The second launcher verifies the sibling completion marker and refuses to run
unless the completed sibling directory is unreadable. Only after the second arm
also completes may the first arm be restored for trusted selection:

```bash
chmod 700 \
  examples/evo2_ecosystem/results/evo2-production-20260712-r1/stable
```

Fresh launch fails if its arm directory is nonempty. To continue an interrupted
arm, add `--resume`; resume is accepted only when the stored run specification
matches exactly and is recorded separately.

## Development selection and freezing

Do not unlock anything until **both** arms have terminated and each has the full
matched budget. Then unlock the development manifest and development founders
only; the sealed manifest and sealed founders stay mode `000`:

```bash
chmod 644 \
  ../microcosmos/experiments/evo2_ecosystem/heredity_adaptation/manifests/development.json \
  ../microcosmos/experiments/evo2_ecosystem/founders/heredity_adaptation/dev-*.npz

test ! -r ../microcosmos/experiments/evo2_ecosystem/heredity_adaptation/manifests/sealed.json
test ! -r ../microcosmos/experiments/evo2_ecosystem/founders/heredity_adaptation/sealed-00.npz
```

Run trusted finalist selection one arm at a time:

```bash
systemd-run --user --scope --quiet \
  -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=2G \
  conda run -n sakana python examples/evo2_ecosystem/freeze_finalist.py --regime stable

systemd-run --user --scope --quiet \
  -p MemoryHigh=20G -p MemoryMax=24G -p MemorySwapMax=2G \
  conda run -n sakana python examples/evo2_ecosystem/freeze_finalist.py --regime punctuated
```

The selector derives all paths and the top-K from the run spec; it accepts no
result path, budget, model, manifest, or relabeling override. It verifies both
completed arms and every generation's candidate, evaluator, simulator,
training-manifest, and repeat hashes before loading development. Each frozen
directory contains the exact winning `main.py`, `freeze_record.json`, and a
hashed `archive_lineage.json`. The sealed manifest is unlocked only for the
later, one-time final comparison after both finalists and all baselines are
irrevocably frozen.

Before unlocking the sealed manifest, authenticate each champion's actual
`parent_id` chain and freeze the predeclared program-lineage representatives:

```bash
conda run -n sakana python \
  examples/evo2_ecosystem/program_lineage.py --regime stable
conda run -n sakana python \
  examples/evo2_ecosystem/program_lineage.py --regime punctuated
```

This opens the completed archive database read-only, verifies its hash and the
declared champion against the frozen source, and follows only actual parent
links (not archive inspirations). It writes canonical, write-once
`program_lineage_selection.json` and `.sha256` records under the arm's selection
directory. Source-identical archive copies are evaluated once, while the JSON
retains every program ID, parent ID, and generation occurrence that proves the
deduplication. The command refuses to run if the sealed manifest is readable.
