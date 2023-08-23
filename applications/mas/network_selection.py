import copy

import numpy as np
from sympy import symbols, Eq, solve, Rational


def gen_task_combinations(affinity, tasks, rtn, index, path, path_dict, task_overlap):
    if index >= len(tasks):
        return

    for i in range(index, len(tasks)):
        cur_task = tasks[i]
        new_path = path
        new_dict = {k: v for k, v in path_dict.items()}

        # Building from a tree with two or more tasks...
        if new_path:
            new_dict[cur_task] = 0.
            for prev_task in path_dict:
                new_dict[prev_task] += affinity[prev_task][cur_task]
                new_dict[cur_task] += affinity[cur_task][prev_task]
            new_path = '{}|{}'.format(new_path, cur_task)
            rtn[new_path] = new_dict
        else:  # First element in a new-formed tree
            new_dict[cur_task] = 0.
            new_path = cur_task

        gen_task_combinations(affinity, tasks, rtn, i + 1, new_path, new_dict, task_overlap)

        if '|' not in new_path:
            if task_overlap:
                new_dict[cur_task] = -1e6
            else:
                new_dict[cur_task] = average_of_self_to_others_and_others_to_self(cur_task, affinity)

            rtn[new_path] = new_dict


def average_of_self_to_others(cur_task, affinity):
    scores = [score for task, score in affinity[cur_task].items() if task != cur_task]
    return sum(scores) / len(scores)


def average_of_others_to_self(cur_task, affinity):
    scores = [score for source_task, a in affinity.items() for target_task, score in a.items()
              if source_task != cur_task and target_task == cur_task]
    return sum(scores) / len(scores)


def average_of_self_to_others_and_others_to_self(cur_task, affinity):
    scores1 = [score for task, score in affinity[cur_task].items() if task != cur_task]
    scores2 = [score for source_task, a in affinity.items() for target_task, score in a.items()
               if source_task != cur_task and target_task == cur_task]
    return (sum(scores1) + sum(scores2)) / (len(scores1) + len(scores2))


def select_groups(affinity, rtn_tup, index, cur_group, best_group, best_val, splits, task_overlap=True):
    # Check if this group covers all tasks.
    num_tasks = len(affinity.keys())
    if task_overlap:
        task_set = set()
        for group in cur_group:
            for task in group.split('|'): task_set.add(task)
    else:
        task_set = list()
        for group in cur_group:
            for task in group.split('|'):
                if task in task_set:
                    return
                else:
                    task_set.append(task)
    if len(task_set) == num_tasks:
        best_tasks = {task: -1e6 for task in task_set}

        # Compute the per-task best scores for each task and average them together.
        for group in cur_group:
            for task in cur_group[group]:
                best_tasks[task] = max(best_tasks[task], cur_group[group][task])
        group_avg = np.mean(list(best_tasks.values()))

        # Compare with the best grouping seen thus far.
        if group_avg > best_val[0]:
            # print(cur_group)
            if task_overlap or no_task_overlap(cur_group, num_tasks):
                best_val[0] = group_avg
                best_group.clear()
                for entry in cur_group:
                    best_group[entry] = cur_group[entry]

    # Base case.
    if len(cur_group.keys()) == splits:
        return

    # Back to combinatorics
    for i in range(index, len(rtn_tup)):
        selected_group, selected_dict = rtn_tup[i]

        new_group = {k: v for k, v in cur_group.items()}
        new_group[selected_group] = selected_dict

        if len(new_group.keys()) <= splits:
            select_groups(affinity, rtn_tup, i + 1, new_group, best_group, best_val, splits, task_overlap)


def task_grouping(affinity, task_overlap=True, split=3):
    tasks = list(affinity.keys())
    rtn = {}

    gen_task_combinations(affinity, tasks=tasks, rtn=rtn, index=0, path='', path_dict={}, task_overlap=task_overlap)

    # Normalize by the number of times the accuracy of any given element has been summed.
    # i.e. (a,b,c) => [acc(a|b) + acc(a|c)]/2 + [acc(b|a) + acc(b|c)]/2 + [acc(c|a) + acc(c|b)]/2
    for group in rtn:
        if '|' in group:
            for task in rtn[group]:
                rtn[group][task] /= (len(group.split('|')) - 1)

    assert (len(rtn.keys()) == 2 ** len(affinity.keys()) - 1)
    rtn_tup = [(key, val) for key, val in rtn.items()]

    # if not task_overlap:
    #     rtn_tup = calculate_self_affinity(affinity, rtn_tup)
    selected_group = {}
    selected_val = [-100000000]
    select_groups(affinity, rtn_tup, index=0, cur_group={}, best_group=selected_group, best_val=selected_val,
                  splits=split, task_overlap=task_overlap)
    return list(selected_group.keys())


def rtn_tup_to_dict(rtn_tup):
    d = {}
    for tup in rtn_tup:
        d[tup[0]] = tup[1]
    return d


def rtn_dict_to_tup(rtn_dict):
    rtn_tup = []
    for key, value in rtn_dict.items():
        rtn_tup.append((key, value))
    return rtn_tup


def calculate_self_affinity(affinity, rtn_tup):
    rtn_dict = rtn_tup_to_dict(rtn_tup)

    task_names = list(affinity.keys())
    tasks = symbols(" ".join(task_names))
    for i, t in enumerate(task_names):
        rtn_dict[t] = tasks[i]

    equations = []
    for i, task in enumerate(task_names):
        task_combs = [comb for comb in rtn_dict.keys() if task in comb]
        count = len(task_combs) - 1
        eq = Rational(0)
        name1 = task + "|"
        name2 = "|" + task
        for comb in task_combs:
            if comb == task:
                eq -= count * rtn_dict[comb]
                continue
            sub_comb = comb.replace(name1, "") if name1 in comb else comb.replace(name2, "")
            sub = rtn_dict[sub_comb] if "|" not in sub_comb else sum(rtn_dict[sub_comb].values())
            eq += sum(rtn_dict[comb].values()) - sub
        equations.append(Eq(eq, 0))
    sol = solve(equations, tasks)
    for i, t in enumerate(task_names):
        rtn_dict[t] = {t: sol[tasks[i]]}

    rtn_tup = rtn_dict_to_tup(rtn_dict)
    return rtn_tup


def no_task_overlap(group, num_tasks):
    task_set = list()
    for combination in group.keys():
        for task in combination.split("|"):
            if task not in task_set:
                task_set.append(task)
            else:
                return False
    return len(task_set) == num_tasks


def average_task_affinity_among_clients(affinities):
    result = copy.deepcopy(affinities[0])
    for task, affinity in result.items():
        for target_task, score in affinity.items():
            total = score
            for a in affinities[1:]:
                total += a[task][target_task]
            result[task][target_task] = total / len(affinities)
    return result


def run(affinities):
    results = []
    averaged_affinity = average_task_affinity_among_clients(affinities)
    groups = task_grouping(averaged_affinity, task_overlap=True)
    results.append(groups)
    for i, a in enumerate(affinities):
        print("client", i)
        groups = task_grouping(a, task_overlap=True)
        results.append(groups)
    print(results)
    return results
