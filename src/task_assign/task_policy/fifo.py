import copy

class FIFO:
    """First-in, first-out task allocation.

    Deterministic: each idle agent (scanned in agent-id order) receives the
    oldest unstarted task that is not yet assigned. Because task endpoints are
    generated uniformly at random by the environment, the assigned task is
    still spatially random with respect to the agent.
    """

    def __init__(self):
        pass

    def assign_task(self, env):
        current_tasklist = copy.deepcopy(env.current_tasklist)
        assigned_list = copy.deepcopy(env.assigned_list)
        assigned_tasks = copy.deepcopy(env.assigned_tasks)
        task_assign = []

        task_idx = 0
        for i in range(env.agent_num):
            if assigned_tasks[i] == [] and len(current_tasklist) - task_idx > 0:
                assigned = False
                while task_idx < len(current_tasklist):
                    if assigned_list[task_idx] == -1:
                        task_assign.append(task_idx)
                        task_idx += 1
                        assigned = True
                        break
                    else:
                        task_idx += 1
                if not assigned:
                    # Every remaining task is already assigned to another
                    # agent; without this entry the returned list would be
                    # shorter than agent_num and misalign downstream.
                    task_assign.append(-1)
            else:
                task_assign.append(-1)

        return task_assign
