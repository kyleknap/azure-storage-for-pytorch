import argparse
import collections
import glob
import os
import json
import re
import statistics

import tabulate

_DEFAULT_MAX_CONCURRENCY = 8
_DEFAULT_CONNECTION_DATA_BLOCK_SIZE = 64 * 1024
_DEFAULT_PARTITION_THRESHOLD = 4 * 1024 * 1024
_DEFAULT_PARTITION_SIZE = 8 * 1024 * 1024
TUNE_COL_PARAM_TO_REGEX = {
    'max-concurrency': re.compile(rf'^c:(\d+)_d:{_DEFAULT_CONNECTION_DATA_BLOCK_SIZE}_t:{_DEFAULT_PARTITION_THRESHOLD}_s:{_DEFAULT_PARTITION_SIZE}$'),
    'connection-data-block-size': re.compile(rf'^c:{_DEFAULT_MAX_CONCURRENCY}_d:(\d+)_t:{_DEFAULT_PARTITION_THRESHOLD}_s:{_DEFAULT_PARTITION_SIZE}$'),
    'partition-threshold': re.compile(rf'^c:{_DEFAULT_MAX_CONCURRENCY}_d:{_DEFAULT_CONNECTION_DATA_BLOCK_SIZE}_t:(\d+)_s:{_DEFAULT_PARTITION_SIZE}$'),
    'partition-size': re.compile(rf'^c:{_DEFAULT_MAX_CONCURRENCY}_d:{_DEFAULT_CONNECTION_DATA_BLOCK_SIZE}_t:{_DEFAULT_PARTITION_THRESHOLD}_s:(\d+)$'),
}


def summarize(output_dir, tune_param):
    if output_dir is None:
        output_dir = _get_latest_multirun_dir()
    tables = collections.defaultdict(lambda: collections.defaultdict(dict))
    results_dirs = _get_results_dirs(output_dir)
    for result_dir in results_dirs:
        num_runs = _add_result_dir_to_tables(result_dir, tables)
    _display_tables(tables, num_runs, tune_param)


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
    conf_id = _get_conf_id(metadata)
    tables[metadata['read-method']][conf_id][metadata['model-size']] = f"{mean:.3f} ± {stddev:.3f}"
    return metadata['num-runs']


def _calculate_mean_and_stddev(result_dir):
    durations = []
    for result in glob.glob('*.txt', root_dir=result_dir):
        with open(os.path.join(result_dir, result), 'r') as f:
            duration = float(f.read())
            durations.append(duration)
    return statistics.mean(durations), statistics.stdev(durations)

def _get_conf_id(metadata):
    return f"c:{metadata['max-concurrency']}_d:{metadata['connection-data-block-size']}_t:{metadata['partition-threshold']}_s:{metadata['partition-size']}"

def _display_tables(tables, num_runs, tune_param):
    count = 0
    for read_method, table in tables.items():
        if read_method == 'torch-load':
            read_method = 'torch.load()'
        if read_method == 'readall':
            read_method = 'read()'
        print(f'\nTable {count}: {read_method} average latency in seconds over {num_runs} runs')
        _display_table(table, tune_param)
        count += 1

def _display_table(table, tune_param):
    tabulate_table = []
    headers = None
    left_col_header = tune_param if tune_param is not None else 'Tune permutation'
    for conf_id, model_sizes in table.items():
        if headers is None:
            headers = [left_col_header, *model_sizes.keys()]
        col_display = _get_col_display(conf_id, tune_param)
        if col_display is None:
            continue
        tabulate_table.append([col_display, *model_sizes.values()])
    print(tabulate.tabulate(tabulate_table, headers=headers, tablefmt="fancy_grid"))

def _get_col_display(conf_id, tune_param):
    if tune_param is None:
        return conf_id
    match = TUNE_COL_PARAM_TO_REGEX[tune_param].match(conf_id)
    if match:
        return str(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument(
        'output_dir', nargs='?', help='The output directory to summarize'
    )
    parser.add_argument(
        '--tune-param',
        choices=['max-concurrency', 'connection-data-block-size', 'partition-threshold', 'partition-size'],
        help='The parameter to compare across runs. All other parameters will be fixed at default values'
    )
    args = parser.parse_args()
    summarize(args.output_dir, args.tune_param)


if __name__ == "__main__":
    main()
