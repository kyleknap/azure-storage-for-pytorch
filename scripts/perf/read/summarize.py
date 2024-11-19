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
        num_runs = _add_result_dir_to_tables(result_dir, tables)
    print(tables)
    _display_tables(tables, num_runs)


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
    tables[metadata['read-method']][metadata['filelike-impl']][metadata['model-size']] = f"{mean:.3f} ± {stddev:.3f}"
    return metadata['num-runs']


def _calculate_mean_and_stddev(result_dir):
    durations = []
    for result in glob.glob('*.txt', root_dir=result_dir):
        with open(os.path.join(result_dir, result), 'r') as f:
            duration = float(f.read())
            durations.append(duration)
    return statistics.mean(durations), statistics.stdev(durations)


def _display_tables(tables, num_runs):
    count = 0
    for read_method, table in tables.items():
        if read_method == 'torch-load':
            read_method = 'torch.load()'
        if read_method == 'readall':
            read_method = 'read()'
        print(f'\nTable {count}: {read_method} average latency in seconds over {num_runs} runs')
        _display_table(table)
        count += 1

def _display_table(table):
    tabulate_table = []
    headers = None
    for filelike_impl, model_sizes in table.items():
        if headers is None:
            headers = ['Implementation', *model_sizes.keys()]
        tabulate_table.append([filelike_impl, *model_sizes.values()])
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
