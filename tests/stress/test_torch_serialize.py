import os
import pathlib

import pytest
import torch

from azstoragetorch.io import BlobIO
from utils import get_num_runs

STRESS_DIRNAME = os.path.abspath(os.path.dirname(__file__))


def get_local_weight_files():
    local_weights_dir = pathlib.Path(STRESS_DIRNAME, "local-weights")
    return list(local_weights_dir.iterdir())


def load_weights_from_file(weight_file):
    return torch.load(weight_file, weights_only=True)


def assert_state_dict(expected_state_dict, actual_state_dict):
    assert expected_state_dict.keys() == actual_state_dict.keys()
    for key in expected_state_dict.keys():
        assert torch.equal(expected_state_dict[key], actual_state_dict[key])


@pytest.mark.parametrize("weight_file", get_local_weight_files())
@pytest.mark.parametrize("run_number", range(get_num_runs()))
def test_roundtrip_torch_load_save(weight_file, run_number, container_client):
    weights = load_weights_from_file(weight_file)
    blob_url = f"{container_client.url}/weights/{run_number}/{weight_file.name}"

    with BlobIO(blob_url, "wb") as f:
        torch.save(weights, f)

    with BlobIO(blob_url, "rb") as f:
        loaded_weights = torch.load(f, weights_only=True)

    assert_state_dict(weights, loaded_weights)
