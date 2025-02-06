import collections
import os
import io
import json
import time
import dataclasses
from fileinput import filename
from typing import Optional

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import blobfile as bf

import azstoragetorch.io


@dataclasses.dataclass
class FileOp:
    name: str
    args: list


@dataclasses.dataclass
class WriteOp(FileOp):
    name: str = dataclasses.field(default="write", init=False)


@dataclasses.dataclass
class FlushOp(FileOp):
    name: str = dataclasses.field(default="flush", init=False)


@dataclasses.dataclass
class CloseOp(FileOp):
    name: str = dataclasses.field(default="close", init=False)
    args: list = dataclasses.field(init=False)

    def __post_init__(self):
        self.args = []


class WriteIOSpy(io.IOBase):
    def __init__(self):
        self.recorded_operations = []

    def write(self, b):
        self.recorded_operations.append(FileOp("write", [len(b), type(b)]))
        return len(b)

    def flush(self):
        self.recorded_operations.append(FlushOp([]))

    def seekable(self):
        return False

    def close(self):
        self.recorded_operations.append(CloseOp())


def load_from_filename(filename):
    return torch.load(filename, weights_only=True)


def save_with_spy(model):
    spy = WriteIOSpy()
    torch.save(model, spy)
    return spy


def get_op_distribution(spy):
    counts = collections.defaultdict(int)
    for op in spy.recorded_operations:
        counts[op.name] += 1
    return counts


def get_op_counts(spy, op_type="read"):
    counts = collections.defaultdict(int)
    for op in spy.recorded_operations:
        if op.name == op_type:
            counts[op.args[0]] += 1
    read_amounts = sorted(counts, key=lambda x: x)
    return {k: counts[k] for k in read_amounts}

def get_write_position_points(writes):
    operation_index = 0
    write_positions = []
    for write in writes:
        write_positions.append(operation_index)
        operation_index += write
    return write_positions


def get_write_types(spy):
    write_types = collections.defaultdict(int)
    for op in spy.recorded_operations:
        if op.name == "write":
            write_types[op.args[1]] += 1
    return write_types


def plot_spy(spy):
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(3, 1)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, :])
    ax3 = fig.add_subplot(gs[2, :])


    formatter = mticker.ScalarFormatter(useOffset=False, useLocale=True)
    formatter.set_scientific(False)
    ax1.yaxis.set_major_formatter(formatter)
    ax2.yaxis.set_major_formatter(formatter)
    import locale
    locale.setlocale(locale.LC_ALL, '')

    writes = [
        op.args[0] for op in spy.recorded_operations if op.name == "write"
    ]
    write_positions = get_write_position_points(writes)
    ax1.plot(write_positions, '-', color='blue')

    ax2.bar(range(len(writes)), writes, color='orange')

    write_counts = get_op_counts(spy, "write")
    ax3.bar(range(len(write_counts.keys())), write_counts.values(), color='green')
    ax3.set_xticks(range(len(write_counts.keys())))
    ax3.set_xticklabels(write_counts.keys())

    ax1.set_xlabel('Operation Index')
    ax1.set_ylabel('Position / Size')
    ax1.set_title('Write on File')

    ax2.set_xlabel('Operation Index')
    ax2.set_ylabel('Write Size')
    ax2.set_title('Write size at index')
    ax2.set_yscale('log')

    ax3.set_xlabel('Write Size')
    ax3.set_ylabel('Number of occurrences')
    ax3.set_title('Write size distribution')

    plt.subplots_adjust(hspace=0.5)
    plt.show()


def main():
    # model = "bert_model.pth"
    model = "large_model.pth"
    print('Loading model...')
    model = load_from_filename(model)
    print('Saving model to spy...')
    spy = save_with_spy(model)
    # Uncomment to get the distribution of operations
    # print(get_op_distribution(spy))
    # Uncomment to generate plots
    plot_spy(spy)
    # Uncomment to get types passed to write. It should be solely memoryviews.
    # print(get_write_types(spy))


if __name__ == "__main__":
    main()
