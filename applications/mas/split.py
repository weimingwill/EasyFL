import argparse
import collections
import copy
import functools
import json
import operator
import re
from pprint import pprint

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 30})
import numpy as np

import network_selection


MAPPING = {
    'ss_l': 's',
    'depth_l': 'd',
    'norm_l': 'n',
    'key_l': 'k',
    'edge2d_l': 't',
    'edge_l': 'e',
    'shade_l': 'r',
    'rgb_l': 'a',
    'pc_l': 'c',
}

COLOR_MAP = {
    'ss_l': 'tab:blue',
    'depth_l': 'tab:orange',
    'norm_l': 'tab:green',
    'key_l': 'tab:red',
    'edge2d_l': 'tab:purple',
    'edge_l': 'tab:brown',
    'shade_l': 'tab:pink',
    'rgb_l': 'tab:gray',
    'pc_l': 'tab:olive',
}


class Affinity:
    def __init__(self, args=None):
        self.affinities = {}
        self.args = args
        self.task_overlap = args.task_overlap
        self.split = args.split

    def add(self, round_id, client_id, affinity):
        if self.args.preprocess:
            affinity = self.preprocess_affinity(affinity)

        for scores in affinity.values():
            if isinstance(scores, list) and scores[0]['ss_l'] == 0.0:
                return
            else:
                break

        if round_id not in self.affinities:
            self.affinities[round_id] = {client_id: affinity}
        else:
            self.affinities[round_id][client_id] = affinity

    def get_round_affinities(self, round_id):
        return list(self.affinities[round_id].values())

    def average_affinities(self, affinities):
        result = copy.deepcopy(affinities[0])
        for task, affinity in result.items():
            for target_task, score in affinity.items():
                total = score
                for a in affinities[1:]:
                    total += a[task][target_task]
                result[task][target_task] = total / len(affinities)
        return result

    def average_affinity_of_clients(self, max_round=100):
        affinities = {}
        for round_id, affinity in self.affinities.items():
            if round_id >= max_round:
                continue
            result = self.average_affinities(list(affinity.values()))
            affinities[round_id] = result
        return affinities

    def average_affinity_of_rounds(self, max_round=100):
        affinities = self.average_affinity_of_clients(max_round)
        return self.average_affinities(list(affinities.values()))

    def preprocess_affinity(self, affinity):
        for task, scores in affinity.items():
            result = dict(functools.reduce(operator.add, map(collections.Counter, scores)))
            affinity[task] = result
        return affinity

    def network_selection(self, rounds, specific_round=False):
        results = {}
        # Network selection of specific round
        if specific_round:
            for round_id in rounds:
                round_affinities = self.get_round_affinities(round_id)
                # Network selection of average
                averaged_affinity = self.average_affinities(round_affinities)
                result = network_selection.task_grouping(averaged_affinity, task_overlap=self.task_overlap,
                                                         split=self.split)
                results[round_id] = {"average": result}
                # pprint(averaged_affinity)
                if not self.args.average_only:
                    for client, a in self.affinities[round_id].items():
                        result = network_selection.task_grouping(a, task_overlap=self.task_overlap, split=self.split)
                        results[round_id][client] = result
        # Average task affinity of all rounds
        for round_id in rounds:
            affinities = self.average_affinity_of_rounds(round_id)
            results[f"average_{round_id}"] = network_selection.task_grouping(affinities, task_overlap=self.task_overlap,
                                                                             split=self.split)
        # Convert string formats from loss to single letter
        return results


def extract_task_affinity(line):
    r = re.search(r'[\d\w\-\[\]\:\ ,]* Round (\d+) - Client (\w+) transference: (\{[\{\}\[\]\'\-\_\d\w\: .,]*\}\n)',
                  line)

    if not r:
        return
    return r.groups()


def run(args):
    A = Affinity(args)
    with open(args.filename, 'r') as f:
        for line in f:
            data = extract_task_affinity(line)
            if not data:
                continue
            round_id, client_id, affinity = data
            round_id = int(round_id)
            affinity = affinity.replace("'", "\"")
            affinity = json.loads(affinity)
            A.add(round_id, client_id, affinity)
        else:
            results = A.network_selection(args.rounds)
            results_str = json.dumps(results)
            for loss_name, char in MAPPING.items():
                results_str = results_str.replace(loss_name, char)
            results = json.loads(results_str)
            pprint(results)


def construct_analyze_parser(parser):
    parser.add_argument('-f', '--filename', type=str, metavar='PATH', default="./train.log")
    parser.add_argument('-s', '--split', type=int, default=3)
    parser.add_argument('-o', '--task_overlap', action='store_true')
    parser.add_argument('-p', '--preprocess', action='store_true')
    parser.add_argument('-a', '--average_only', action='store_true')
    parser.add_argument('-r', '--rounds', nargs="*", default=[10], type=int)
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split')
    parser = construct_analyze_parser(parser)
    args = parser.parse_args()
    print("args:", args)
    run(args)
