# Evaluation Artifacts (PRICAI 2026 baseline study)

Raw evaluation logs, the aggregated per-condition summary, and the trained path-planning
weights behind the baseline result table of the PRICAI 2026 paper (see the repository
root `README.md` for the citation).

## Contents

| Path | Contents |
| --- | --- |
| `summary.csv` | Per-condition aggregate over the seeds of each condition (long format) |
| `logs/<map>/<safe\|unsafe>/*.txt` | One raw stdout log per evaluation run (504 files) |
| `models/*.th` | Trained IQL/QMIX weights used by the runs (240 files) |

## Naming

Log files: `<map>_<agent_num>_<planner>_<assigner>_seed<k>.txt`, where `<planner>` is
`iql`, `qmix`, or `pbs` (= PP), `<assigner>` is `fifo` or `tp`, and `seed<k>` selects the
trained model. The `safe/` and `unsafe/` parent directory says whether the safety layer was
enabled (`safe_mode`), i.e. `safe/..._qmix_...` is SafeQMIX and `unsafe/..._qmix_...` is
plain QMIX. PP carries no weights and is unaffected by the safety layer, so its logs sit
under `unsafe/` with `seed0` only.

Weight files: `<map>_<agent_num>_<planner>_<safe|unsafe>_seed<k>.th`, five seeds per
combination (4 maps x 3 team sizes x {iql, qmix} x {safe, unsafe} x 5 seeds = 240). One
network is shared by all agents, so each file holds a single `RNNAgent` state dict.

The `safe`/`unsafe` tag records the training environment, not a runtime switch: `safe` weights
come from a `drp_safe-*` env key and `unsafe` weights from the corresponding `drp-*` key.
`MARLPolicy` reads one fixed location for both,

```
src/all_policy/models/safe/<map_name>_<agent_num>_<path_planner>.th
```

so to reproduce a condition, copy the matching weight file there under that name -- the
`_<safe|unsafe>_seed<k>` part of the name is dropped -- and set `safe_mode` to match the tag.

## Log format

Every completed run ends with a single machine-readable line:

```
[RESULT] condition=<map>/<n>agent/<safe|unsafe>_<planner>_<assigner> model_seed=<k> n_ep=10
         time_limit=1000 eval_seed=0 episode_seed_base=10000 step=<mean steps>
         goal_account=<mean node arrivals> 1agent_goal_account=<per-agent mean>
         task_completion=<TC> collision_rate=<fraction of episodes ending in a collision>
         time_sec=<ET>
```

`task_completion` and `time_sec` are the TC and ET columns of the paper's table.
`step` is the mean episode length: it stays at the 1000-step limit whenever no collision
occurs, and falls well below it for the plain learners, which end their episodes early.
`collision_rate` and `step` are recorded here but are not reported in the paper.

The lines above the `[RESULT]` line are Gym/NumPy deprecation warnings from the pinned
dependency versions and carry no result data.

## Aggregation

`summary.csv` has one row per (condition, metric) pair with columns
`condition,metric,n_seeds,mean,std,sem,per_seed`. `std` is the sample standard deviation
(denominator `n_seeds - 1`) and `sem` is `std / sqrt(n_seeds)`; `per_seed` lists the
individual run values, seed 0 first. For the learning methods `n_seeds = 5` and `std` is
therefore the model-to-model spread reported as `+- s.d.` in the paper. PP carries no learned
weights and is costly to run, so it was measured once per condition: `n_seeds = 1` and
`std`/`sem` are `nan`.

Every row of `summary.csv` is reproducible from the `logs/` files shipped here.

## Coverage against the published table

Of the 120 conditions in the paper's result table, 119 are backed by the logs here, and their
TC and ET values agree with `summary.csv`. PP is costly to run, so each of its conditions was
measured once, and on `map_8x5` a later run took the place of three of those logs. That leaves
two notes:

- `map_8x5/4agent/unsafe_pbs_fifo` (published TC 93, ET 2189.84 s): the log of the run behind
  this cell is no longer available, because a later attempt aborted on an interpreter-level
  fault in a dependency and replaced it. The condition therefore has no log here and no row in
  `summary.csv`.
- `map_8x5/3agent/unsafe_pbs_tp`: the log included here gives ET 1023.19 s against the
  published 1066.5 s, the condition having been timed again after the table was fixed. TC
  (91.7) is identical. PP's execution time is wall-clock time from an unoptimized
  single-threaded planner and is sensitive to machine load; the paper states that its ET values
  bound this implementation rather than the search paradigm.

For `map_8x5/3agent/unsafe_pbs_fifo`, also timed again, both logs are kept: `..._seed0.txt` is
the run behind the published ET (989.737 s) and `..._seed0_rerun.txt` is the later one
(962.568 s). TC (72.9) is identical in both.
