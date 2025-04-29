import argparse
import collections
import glob
import os
import json
import statistics

import tabulate


def summarize(output_dir):
    if output_dir is None:
        output_dir = _get_latest_multirun_dir()
    tables = collections.defaultdict(lambda: collections.defaultdict(dict))
    results_dirs = _get_results_dirs(output_dir)
    for result_dir in results_dirs:
        metadata = _add_result_dir_to_tables(result_dir, tables)
    _display_tables(tables, metadata)


def _get_latest_multirun_dir():
    return max(glob.glob('multirun/*/*/'))

def _get_results_dirs(output_dir):
    return [
        os.path.join(output_dir, results_dir)
        for results_dir in sorted(glob.glob('**/results/', root_dir=output_dir), key=lambda x: int(x.split(os.sep)[-3]))
    ]

def _add_result_dir_to_tables(result_dir, tables):
    with open(os.path.join(result_dir, "metadata.json"), 'r') as f:
        metadata = json.load(f)
    mean, stddev = _calculate_mean_and_stddev(result_dir)
    tables[metadata['dataset-cls-method']][metadata['dataset-cls']][metadata['prefix']] = f"{mean:.3f} ± {stddev:.3f}"
    return {
        'num-runs': metadata['num-runs'],
        'batch-size': metadata['batch-size'],
        'num-workers': metadata['num-workers'],
    }


def _calculate_mean_and_stddev(result_dir):
    durations = []
    for result in glob.glob('*.txt', root_dir=result_dir):
        with open(os.path.join(result_dir, result), 'r') as f:
            duration = float(f.read())
            durations.append(duration)
    return statistics.mean(durations), statistics.stdev(durations)


def _display_tables(tables, metadata):
    num_runs = metadata['num-runs']
    batch_size = metadata['batch-size']
    num_workers = metadata['num-workers']
    count = 0
    for cls_method, table in tables.items():
        print(f'\nTable {count}: {cls_method} dataloader average in seconds over {num_runs} runs (batch_size={batch_size}, num_workers={num_workers})')
        _display_table(table)
        count += 1

def _display_table(table):
    tabulate_table = []
    headers = None
    for dataset_impl, prefix in table.items():
        if headers is None:
            headers = ['Implementation', *prefix.keys()]
        tabulate_table.append([dataset_impl, *prefix.values()])
    print(tabulate.tabulate(tabulate_table, headers=headers, tablefmt="fancy_grid"))


def main():
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument(
        'output_dir', nargs='?', help='The output directory to summarize'
    )
    args = parser.parse_args()
    summarize(args.output_dir)


if __name__ == "__main__":
    main()
