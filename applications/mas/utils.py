import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.std = 0
        self.sum = 0
        self.sumsq = 0
        self.count = 0
        self.lst = []

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        # self.sumsq += float(val)**2
        self.count += n
        self.avg = self.sum / self.count
        self.lst.append(self.val)
        self.std = np.std(self.lst)


class ProgressTable:
    def __init__(self, table_list):
        if len(table_list) == 0:
            print()
            return
        self.lens = defaultdict(int)
        self.table_list = table_list
        self.construct(table_list)

    def construct(self, table_list):
        self.lens = defaultdict(int)
        self.table_list = table_list
        for i in table_list:
            for ii, to_print in enumerate(i):
                for title, val in to_print.items():
                    self.lens[(title, ii)] = max(self.lens[(title, ii)], max(len(title), len(val)))

    def print_table_header(self):
        for ii, to_print in enumerate(self.table_list[0]):
            for title, val in to_print.items():
                print('{0:^{1}}'.format(title, self.lens[(title, ii)]), end=" ")

    def print_table_content(self):
        for i in self.table_list:
            print()
            for ii, to_print in enumerate(i):
                for title, val in to_print.items():
                    print('{0:^{1}}'.format(val, self.lens[(title, ii)]), end=" ", flush=True)

    def print_all_table(self):
        self.print_table_header()
        self.print_table_content()

    def print_table(self, header_condition, content_condition):
        if header_condition:
            self.print_table_header()
        if content_condition:
            self.print_table_content()

    def update_table_list(self, table_list):
        self.construct(table_list)


def print_table(table_list):
    if len(table_list) == 0:
        print()
        return

    lens = defaultdict(int)
    for i in table_list:
        for ii, to_print in enumerate(i):
            for title, val in to_print.items():
                lens[(title, ii)] = max(lens[(title, ii)], max(len(title), len(val)))

    # printed_table_list_header = []
    for ii, to_print in enumerate(table_list[0]):
        for title, val in to_print.items():
            print('{0:^{1}}'.format(title, lens[(title, ii)]), end=" ")
    for i in table_list:
        print()
        for ii, to_print in enumerate(i):
            for title, val in to_print.items():
                print('{0:^{1}}'.format(val, lens[(title, ii)]), end=" ", flush=True)
    print()
