# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import random
import string


KB = 1024
MB = 1024**2
GB = 1024**3

NUM_RUNS_ENV_VAR = "AZSTORAGETORCH_NUM_RUNS"


def get_num_runs():
    return os.environ.get(NUM_RUNS_ENV_VAR, 5)


def sample_data(data_length=20):
    return os.urandom(data_length)


def random_resource_name(name_length=8):
    return "".join(
        random.choices(string.ascii_lowercase + string.digits, k=name_length)
    )


def get_human_readable_size(size):
    if size < KB:
        return f"{size} B"
    elif size < MB:
        return f"{size / KB:.2f} KB"
    elif size < GB:
        return f"{size / MB:.2f} MB"
    else:
        return f"{size / GB:.2f} GB"
