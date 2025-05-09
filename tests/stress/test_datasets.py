from concurrent.futures import ThreadPoolExecutor
import collections
import hashlib
import random
import os

import pytest
import torch.utils.data

from azstoragetorch.datasets import BlobDataset, IterableBlobDataset
from utils import get_num_runs

from utils import KB, MB, sample_data, get_human_readable_size

DEFAULT_MIN_DATA_SIZE = 1
DEFAULT_MAX_DATA_SIZE = 64 * KB
DEFAULT_MAX_LARGE_DATA_SIZE = 32 * MB
DEFAULT_MIN_SAMPLE_SIZE = 1000
DEFAULT_MAX_SAMPLE_SIZE = 50_000

MIN_RANDOM_DATA_SIZE_ENV_VAR = "AZSTORAGETORCH_MIN_RANDOM_DATA_SIZE"
MAX_RANDOM_DATA_SIZE_ENV_VAR = "AZSTORAGETORCH_MAX_RANDOM_DATA_SIZE"
MAX_RANDOM_LARGE_DATA_SIZE_ENV_VAR = "AZSTORAGETORCH_MIN_RANDOM_LARGE_DATA_SIZE"
MIN_RANDOM_SAMPLE_SIZE_ENV_VAR = "AZSTORAGETORCH_MIN_RANDOM_SAMPLE_SIZE"
MAX_RANDOM_SAMPLE_SIZE_ENV_VAR = "AZSTORAGETORCH_MAX_RANDOM_SAMPLE_SIZE"


def get_random_data_size():
    min_size = os.environ.get(MIN_RANDOM_DATA_SIZE_ENV_VAR, DEFAULT_MIN_DATA_SIZE)
    max_size = os.environ.get(MAX_RANDOM_DATA_SIZE_ENV_VAR, DEFAULT_MAX_DATA_SIZE)
    return random.randint(min_size, max_size)


def get_possibly_large_random_data_size():
    min_size = os.environ.get(MIN_RANDOM_DATA_SIZE_ENV_VAR, DEFAULT_MIN_DATA_SIZE)
    max_size = os.environ.get(
        MAX_RANDOM_LARGE_DATA_SIZE_ENV_VAR, DEFAULT_MAX_LARGE_DATA_SIZE
    )
    return random.randint(min_size, max_size)


def get_random_sample_size():
    min_size = os.environ.get(MIN_RANDOM_SAMPLE_SIZE_ENV_VAR, DEFAULT_MIN_SAMPLE_SIZE)
    max_size = os.environ.get(MAX_RANDOM_SAMPLE_SIZE_ENV_VAR, DEFAULT_MAX_SAMPLE_SIZE)
    return random.randint(min_size, max_size)


def generate_and_upload_data_samples(num_samples, container_client, blob_prefix=""):
    futures = []
    with ThreadPoolExecutor() as executor:
        for i in range(num_samples):
            if i % 100 == 0:
                size = get_possibly_large_random_data_size()
            else:
                size = get_random_data_size()
            futures.append(
                executor.submit(
                    generate_and_upload_data_sample,
                    i,
                    size,
                    container_client,
                    blob_prefix=blob_prefix,
                )
            )
    samples = [future.result() for future in futures]
    return sorted(samples, key=lambda x: x["url"])


def generate_and_upload_data_sample(index, size, container_client, blob_prefix=""):
    content = sample_data(size)
    name = f"{hashlib.md5(content).hexdigest()}-{index}"
    blob_name = f"{blob_prefix}{name}.txt"
    container_client.upload_blob(name=blob_name, data=content)
    return {
        "url": f"{container_client.url}/{blob_name}",
        "data": content,
    }


@pytest.fixture(scope="module")
def prefix():
    return "dataset/"


@pytest.fixture(scope="module")
def data_samples(container_client, prefix):
    return generate_and_upload_data_samples(
        get_random_sample_size(), container_client, blob_prefix=prefix
    )


def get_num_samples_and_total_size(data_samples):
    num_samples = len(data_samples)
    total_size = sum(len(sample["data"]) for sample in data_samples)
    return num_samples, total_size


def batched(data_samples, batch_size=1, num_worker_shards=1):
    batched_samples = []
    keys = data_samples[0].keys()
    worker_shards = collections.defaultdict(list)
    for i, data_sample in enumerate(data_samples):
        worker_id = i % num_worker_shards
        worker_shards[worker_id].append(data_sample)
        if len(worker_shards[worker_id]) == batch_size:
            samples = worker_shards.pop(worker_id)
            batch = {}
            for key in keys:
                batch[key] = [item[key] for item in samples]
            batched_samples.append(batch)
    for worker_id, samples in worker_shards.items():
        batch = {}
        for key in keys:
            batch[key] = [item[key] for item in samples]
        batched_samples.append(batch)
    return batched_samples


def get_cls_args_kwargs(cls_method, data_samples, container_url, prefix):
    if cls_method == "from_blob_urls":
        return ([sample["url"] for sample in data_samples],), {}
    elif cls_method == "from_container_url":
        return (container_url,), {"prefix": prefix}


@pytest.mark.parametrize("dataset_cls", [BlobDataset, IterableBlobDataset])
@pytest.mark.parametrize("cls_method", ["from_blob_urls", "from_container_url"])
def test_multiworker_and_multiepoch(
    dataset_cls, cls_method, container_client, data_samples, prefix
):
    num_samples, total_size = get_num_samples_and_total_size(data_samples)
    print(
        f"Num samples in dataset: {num_samples} totaling:  {get_human_readable_size(total_size)}"
    )
    args, kwargs = get_cls_args_kwargs(
        cls_method, data_samples, container_client.url, prefix
    )
    dataset = getattr(dataset_cls, cls_method)(*args, **kwargs)

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=8)
    num_epochs = get_num_runs()
    expected_worker_shards = 1
    if isinstance(dataset, IterableBlobDataset):
        expected_worker_shards = 8

    for i in range(num_epochs):
        print(f"Epoch {i + 1}/{num_epochs}")
        loaded_samples = list(loader)
        assert loaded_samples == batched(
            data_samples, batch_size=32, num_worker_shards=expected_worker_shards
        )
