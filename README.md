# LDRP: A Configurable Benchmark for Lifelong Delivery Robot Routing Problems on Non-Grid Maps

LDRP is a configurable, Gym-based benchmark in which a team of delivery robots serves a continual stream of
pickup-and-delivery tasks on a non-grid route network while avoiding collisions. A policy combines
a **task-allocation** method with a **path-planning** method, and any pair can be swapped in.
The environment extends the DRP benchmark of Ding et al. with an integrated allocation interface,
a lifelong task stream, and a three-component task-management API.

## Publication

This repository accompanies the paper

> Donghui Lin, Masahiro Kaji, Shiyao Ding, and Fumito Uwano.
> **LDRP: A Configurable Benchmark for Lifelong Delivery Robot Routing Problems on Non-Grid Maps.**
> In *23rd Pacific Rim International Conference on Artificial Intelligence (PRICAI 2026)*,
> Guangzhou, China, November 2026, Lecture Notes in Artificial Intelligence, Springer. To appear.

The paper formulates LDRP, describes this benchmark environment, and reports an
allocation × planning baseline study (FIFO / TP × PP / IQL / QMIX / SafeIQL / SafeQMIX) on the four
evaluation maps below with three to five robots. If you use this code, maps, or configurations,
please cite:

```bibtex
@inproceedings{Lin2026LDRP,
  author    = {Lin, Donghui and Kaji, Masahiro and Ding, Shiyao and Uwano, Fumito},
  title     = {{LDRP}: A Configurable Benchmark for Lifelong Delivery Robot Routing Problems on Non-Grid Maps},
  booktitle = {23rd Pacific Rim International Conference on Artificial Intelligence (PRICAI 2026)},
  series    = {Lecture Notes in Artificial Intelligence},
  publisher = {Springer},
  year      = {2026},
  note      = {To appear}
}
```

## Repository Layout

| Path | Contents |
| --- | --- |
| `test.py` | Run one experiment condition (map × team size × planner × assigner) |
| `run.py` | Batch runner sweeping multiple conditions in parallel (logs to `logs/`) |
| `train.py` | Train MARL path planners (IQL/QMIX) via EPyMARL |
| `runner.py` | Episode loop shared by `test.py` (reports task completion and execution time) |
| `drpload_test.py` | Interactive GUI sanity check (see `src/main/README.md`) |
| `src/main/` | The LDRP Gym environment (`drp_env`), maps (CSV), environment configs |
| `src/all_policy/` | Path planners: `pbs.py` (prioritized planning, PP), `policy.py` (MARL model loader) |
| `src/task_assign/` | Task allocation: `task_policy/fifo.py` (FIFO), `task_policy/tp.py` (Token Passing's task-selection rule) |
| `src/config/default.yaml` | Experiment configuration used by `test.py` |
| `src/epymarl/` | EPyMARL framework used for training the learning baselines |

## Installation

Requires Python 3.9+ (tested on 3.10; the pinned matplotlib/networkx versions do not support
3.8). Clone or download this repository, then:

```
pip install -e ./src/main
pip install -r requirements.txt
```

To train the learning baselines you also need the EPyMARL dependencies:

```
pip install -r ./src/epymarl/requirements.txt
```

## Running Benchmark Experiments

Baseline methods map to code options as follows:

| Method | Code option | Notes |
| --- | --- | --- |
| PP (prioritized planner) | `path_planner: "pbs"` | Search-based; no trained model needed (see "About pbs") |
| IQL / QMIX | `path_planner: "iql"` / `"qmix"` | Loads a trained model (see below) |
| SafeIQL / SafeQMIX | same as above + `safe_mode: true` | Safety layer replaces colliding actions |
| FIFO (first-in, first-out) | `task_assigner: "fifo"` | Hands each idle agent the oldest unstarted task; legacy alias: `"random"` |
| TP (task-selection rule of Token Passing) | `task_assigner: "tp"` | Nearest-pickup claims in turn; no token-held path reservations |

Representative evaluation maps are `map_5x4` and `map_8x5` (simple maps with many short edges)
and `map_aoba00` and `map_aoba01` (complex maps with fewer but longer edges). More maps are
available under `src/main/drp_env/map/`; a map is defined by CSV node/edge files, so new maps
need no code changes.

To run a single condition, edit `src/config/default.yaml` (map, team size, planner, assigner,
`safe_mode`, episode count `test_num`) and run:

```
python3 test.py
```

or override map/team/planner/assigner from the command line:

```
python3 test.py map_8x5 4 pbs tp
```

The script prints average **task completion** (TC) and average **execution time** (ET) per
episode. For stable numbers, average many episodes per condition (e.g., `test_num: 1000`) and,
for the learning methods, multiple independently trained models.

To sweep many conditions as a batch experiment, edit the lists at the top of `run.py` and run
`python3 run.py`; per-condition logs are written under `logs/`.

### Trained Models

Model weights are not bundled. Train IQL/QMIX with EPyMARL — `train.py` shows the invocation
(edit the `env_args.key` to choose map/team size; `drp_safe-*` keys train with the safety
layer) — then place the resulting weights at:

```
src/all_policy/models/safe/<map_name>_<agent_num>_<path_planner>.th   (e.g. map_8x5_4_qmix.th)
```

which is where the `MARLPolicy` class in `src/all_policy/policy.py` loads them from.

## About the Policy Implementation

A policy returns a joint action combining path planning (`agents_action`) and task assignment
(`task_assign`), as in `src/policy.py`:
`joint_action = {"pass": agents_action, "task": task_assign}`.

`agents_action` is the same as in the conventional DRP. It is an array whose length equals the number of agents, and each element indicates the node that the corresponding agent moves to.

`task_assign` is an array whose length equals the number of agents, and each element indicates which task (by index) in the task list `env.current_tasklist` is assigned to the corresponding agent.

A new task can only be assigned to an agent that is not currently executing a task.
Use -1 to indicate that no assignment is made.

Example) Given the task list `[[1,2],[5,3],[8,9]]` and the assignment `[1,0,-1]`, task [5,3] is assigned to agent 0 and task [1,2] is assigned to agent 1, while agent 2 is not assigned any task.

`PolicyManager` (`src/all_policy/policy_manager.py`) handles path planning and `TaskManager`
(`src/task_assign/task_manager.py`) handles task assignment, so both are available for use.

- Task-related information
	- `env.current_tasklist`: the list of all currently unexecuted tasks
	(e.g., 3 tasks, `[[1,2],[5,3],[8,9]]`)
	- `env.assigned_list`: the list of which agent each unexecuted task is assigned to; -1 if unassigned
	(e.g., 3 tasks; task 0 is assigned to agent 1, task 1 is assigned to agent 0, and task 2 is unassigned. `[1,0,-1]`)
	- `env.assigned_tasks`: the task information assigned to each agent, including tasks currently being executed
	(e.g., 3 agents; agents 0 and 1 have tasks assigned, agent 2 has none. `[[1,2],[3,4],[]]`)

## About Task Generation and Processing

Tasks are added to `env.current_tasklist` at each step.
When and what kind of tasks are added is decided at the start of each episode, so runs are
reproducible. To run in an environment with tasks enabled, set the `task_flag` argument of
`gym.make` to `True`. To use your own task set instead of randomly generated tasks, pass your
prepared task list to the `task_list` argument of `gym.make`.

### Task Processing Flow
Task assignment &rarr; the agent heads to the pickup location &rarr; the agent picks up the task (execution starts)
 &rarr; the agent delivers the task to the delivery location &rarr; delivery complete &rarr; wait for the next assignment

## About Changes to drp_env.py

- Note that the change at line 200 made for pbs may affect reinforcement learning.
- For continuing (non-terminating) problems, the change ensures that an agent's `avail_actions` is not fixed to the goal node even when the agent has reached its destination.

## About pbs

The `pbs` planner option implements prioritized planning (referred to as PP, to distinguish
it from full Priority-Based Search, which systematically searches over priority orderings):
agents are planned one at a time by a space--time search at step resolution, keeping clearance
from the reserved trajectories of already-planned agents. Priorities favor committed, mid-edge
agents, and an agent for which no clear path is found is greedily promoted to the top and the
team replanned. The planner never declares failure: when this priority repair is exhausted,
agents hold or continue toward their current targets, so congestion shows up as low task
completion rather than as an error.
