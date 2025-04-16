# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
from collections.abc import Callable, Iterable
import dataclasses
import urllib.parse
from typing import Union, Any

import pytest
import torch.utils.data

from azstoragetorch.datasets import BlobDataset, IterableBlobDataset


def generate_and_upload_data_samples(num_samples, container_client, blob_prefix=""):
    samples = []
    for i in range(num_samples):
        blob_name = f"{blob_prefix}{i}.txt"
        content = f"Sample content for {blob_name}".encode("utf-8")
        container_client.upload_blob(name=blob_name, data=content)
        samples.append(
            {
                "url": f"{container_client.url}/{blob_name}",
                "data": content,
            }
        )
    return samples


def parse_blob_name_from_url(blob_url):
    parsed = urllib.parse.urlparse(blob_url)
    return parsed.path.split("/", 2)[-1]


def sort_samples(samples):
    return sorted(samples, key=lambda x: x["url"])


def blob_properties_only_transform(blob):
    return {
        "url": blob.url,
        "blob_name": blob.blob_name,
        "container_name": blob.container_name,
    }


def get_blob_properties_only_data_samples(data_samples, container_name):
    return [
        {
            "url": sample["url"],
            "blob_name": parse_blob_name_from_url(sample["url"]),
            "container_name": container_name,
        }
        for sample in data_samples
    ]


def get_blob_urls(data_samples):
    return [sample["url"] for sample in data_samples]


@pytest.fixture(scope="module")
def blob_prefix():
    return "prefix/"


@pytest.fixture(scope="module")
def prefixed_data_samples(blob_prefix, dataset_container):
    return generate_and_upload_data_samples(
        5, dataset_container, blob_prefix=blob_prefix
    )


@pytest.fixture(scope="module")
def data_samples(prefixed_data_samples, dataset_container):
    return (
        generate_and_upload_data_samples(5, dataset_container) + prefixed_data_samples
    )


@pytest.fixture(scope="module")
def other_data_samples(other_dataset_container):
    return generate_and_upload_data_samples(
        5, other_dataset_container, blob_prefix="other-"
    )


@pytest.fixture(scope="module")
def dataset_container(create_container):
    container_client = create_container()
    yield container_client
    container_client.delete_container()


@pytest.fixture(scope="module")
def other_dataset_container(create_container):
    container_client = create_container()
    yield container_client
    container_client.delete_container()


@dataclasses.dataclass
class DatasetCase:
    dataset_from_url_method: Callable
    url: Union[str, Iterable[str]]
    expected_data_samples: list[dict[str, Any]]
    dataset_from_url_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dataset(self) -> BlobDataset:
        return self.dataset_from_url_method(self.url, **self.dataset_from_url_kwargs)


@pytest.fixture(scope="module")
def create_dataset_case(data_samples):
    def _create_dataset_case(
        dataset_from_url_method,
        url,
        expected_data_samples=None,
        dataset_from_url_kwargs=None,
    ):
        if expected_data_samples is None:
            expected_data_samples = data_samples
        if dataset_from_url_kwargs is None:
            dataset_from_url_kwargs = {}
        return DatasetCase(
            dataset_from_url_method=dataset_from_url_method,
            url=url,
            expected_data_samples=expected_data_samples,
            dataset_from_url_kwargs=dataset_from_url_kwargs,
        )

    return _create_dataset_case


@pytest.fixture(scope="module")
def create_from_container_url_dataset_case(dataset_container, create_dataset_case):
    def _create_from_container_url_dataset_case(
        dataset_cls, container=None, **create_dataset_case_kwargs
    ):
        if container is None:
            container = dataset_container
        return create_dataset_case(
            dataset_cls.from_container_url,
            url=container.url,
            **create_dataset_case_kwargs,
        )

    return _create_from_container_url_dataset_case


@pytest.fixture(scope="module")
def create_from_blob_urls_dataset_case(create_dataset_case, data_samples):
    def _create_from_blob_urls_dataset_case(
        dataset_cls, url=None, **create_dataset_case_kwargs
    ):
        if url is None:
            url = get_blob_urls(data_samples)
        return create_dataset_case(
            dataset_cls.from_blob_urls, url=url, **create_dataset_case_kwargs
        )

    return _create_from_blob_urls_dataset_case


@pytest.fixture(scope="module")
def prefix_case_kwargs(blob_prefix, prefixed_data_samples):
    return {
        "expected_data_samples": prefixed_data_samples,
        "dataset_from_url_kwargs": {"prefix": blob_prefix},
    }


@pytest.fixture(scope="module")
def transform_case_kwargs(dataset_container, data_samples):
    return {
        "expected_data_samples": get_blob_properties_only_data_samples(
            data_samples,
            dataset_container.container_name,
        ),
        "dataset_from_url_kwargs": {"transform": blob_properties_only_transform},
    }


@pytest.fixture(scope="module")
def single_blob_url_case_kwargs(data_samples):
    return {
        "url": get_blob_urls(data_samples)[0],
        "expected_data_samples": data_samples[:1],
    }


@pytest.fixture(scope="module")
def different_containers_case_kwargs(data_samples, other_data_samples):
    urls = get_blob_urls(data_samples) + get_blob_urls(other_data_samples)
    return {
        "url": urls,
        "expected_data_samples": data_samples + other_data_samples,
    }


@pytest.fixture(scope="module")
def map_dataset_from_container_url_case(create_from_container_url_dataset_case):
    return create_from_container_url_dataset_case(BlobDataset)


@pytest.fixture(scope="module")
def map_dataset_from_container_url_with_prefix_case(
    create_from_container_url_dataset_case, prefix_case_kwargs
):
    return create_from_container_url_dataset_case(BlobDataset, **prefix_case_kwargs)


@pytest.fixture(scope="module")
def map_dataset_from_container_url_with_transform_case(
    create_from_container_url_dataset_case, transform_case_kwargs
):
    return create_from_container_url_dataset_case(BlobDataset, **transform_case_kwargs)


@pytest.fixture(scope="module")
def map_dataset_from_blob_urls_case(create_from_blob_urls_dataset_case):
    return create_from_blob_urls_dataset_case(BlobDataset)


@pytest.fixture(scope="module")
def map_dataset_from_blob_urls_with_single_blob_url_case(
    create_from_blob_urls_dataset_case, single_blob_url_case_kwargs
):
    return create_from_blob_urls_dataset_case(
        BlobDataset, **single_blob_url_case_kwargs
    )


@pytest.fixture(scope="module")
def map_dataset_from_blob_urls_with_transform_case(
    create_from_blob_urls_dataset_case, transform_case_kwargs
):
    return create_from_blob_urls_dataset_case(BlobDataset, **transform_case_kwargs)


@pytest.fixture(scope="module")
def map_dataset_from_blob_urls_with_different_containers_case(
    create_from_blob_urls_dataset_case, different_containers_case_kwargs
):
    return create_from_blob_urls_dataset_case(
        BlobDataset, **different_containers_case_kwargs
    )


@pytest.fixture(scope="module")
def iterable_dataset_from_container_url_case(create_from_container_url_dataset_case):
    return create_from_container_url_dataset_case(IterableBlobDataset)


@pytest.fixture(scope="module")
def iterable_dataset_from_container_url_with_prefix_case(
    create_from_container_url_dataset_case, prefix_case_kwargs
):
    return create_from_container_url_dataset_case(
        IterableBlobDataset, **prefix_case_kwargs
    )


@pytest.fixture(scope="module")
def iterable_dataset_from_container_url_with_transform_case(
    create_from_container_url_dataset_case, transform_case_kwargs
):
    return create_from_container_url_dataset_case(
        IterableBlobDataset, **transform_case_kwargs
    )


@pytest.fixture(scope="module")
def iterable_dataset_from_blob_urls_case(create_from_blob_urls_dataset_case):
    return create_from_blob_urls_dataset_case(IterableBlobDataset)


@pytest.fixture(scope="module")
def iterable_dataset_from_blob_urls_with_single_blob_url_case(
    create_from_blob_urls_dataset_case, single_blob_url_case_kwargs
):
    return create_from_blob_urls_dataset_case(
        IterableBlobDataset, **single_blob_url_case_kwargs
    )


@pytest.fixture(scope="module")
def iterable_dataset_from_blob_urls_with_transform_case(
    create_from_blob_urls_dataset_case, transform_case_kwargs
):
    return create_from_blob_urls_dataset_case(
        IterableBlobDataset, **transform_case_kwargs
    )


@pytest.fixture(scope="module")
def iterable_dataset_from_blob_urls_with_different_containers_case(
    create_from_blob_urls_dataset_case, different_containers_case_kwargs
):
    return create_from_blob_urls_dataset_case(
        IterableBlobDataset, **different_containers_case_kwargs
    )


@pytest.fixture
def dataset_case(request):
    return request.getfixturevalue(f"{request.param}_case")


class TestDatasets:
    MAP_CASES = [
        "map_dataset_from_container_url",
        "map_dataset_from_container_url_with_prefix",
        "map_dataset_from_container_url_with_transform",
        "map_dataset_from_blob_urls",
        "map_dataset_from_blob_urls_with_single_blob_url",
        "map_dataset_from_blob_urls_with_transform",
        "map_dataset_from_blob_urls_with_different_containers",
    ]
    ITERABLE_CASES = [
        "iterable_dataset_from_container_url",
        "iterable_dataset_from_container_url_with_prefix",
        "iterable_dataset_from_container_url_with_transform",
        "iterable_dataset_from_blob_urls",
        "iterable_dataset_from_blob_urls_with_single_blob_url",
        "iterable_dataset_from_blob_urls_with_transform",
        "iterable_dataset_from_blob_urls_with_different_containers",
    ]

    ALL_CASES = MAP_CASES + ITERABLE_CASES
    # TODO: Add case for large dataset that requires multiple pages.

    def batched(self, data_samples, batch_size=1):
        batched_samples = []
        keys = data_samples[0].keys()
        for i in range(0, len(data_samples), batch_size):
            batch = data_samples[i : i + batch_size]
            sample = {}
            for key in keys:
                sample[key] = [item[key] for item in batch]
            batched_samples.append(sample)
        return batched_samples

    @pytest.mark.parametrize("dataset_case", MAP_CASES, indirect=True)
    def test_map_dataset(self, dataset_case):
        dataset = dataset_case.to_dataset()
        assert isinstance(dataset, BlobDataset)
        assert len(dataset) == len(dataset_case.expected_data_samples)
        for i in range(len(dataset)):
            assert dataset[i] == dataset_case.expected_data_samples[i]

    @pytest.mark.parametrize("dataset_case", ITERABLE_CASES, indirect=True)
    def test_iterable_dataset(self, dataset_case):
        dataset = dataset_case.to_dataset()
        assert isinstance(dataset, IterableBlobDataset)
        assert list(dataset) == dataset_case.expected_data_samples

    @pytest.mark.parametrize("dataset_case", ALL_CASES, indirect=True)
    def test_default_loader(self, dataset_case):
        dataset = dataset_case.to_dataset()
        loader = torch.utils.data.DataLoader(dataset)
        loaded_samples = list(loader)
        assert loaded_samples == self.batched(
            dataset_case.expected_data_samples, batch_size=1
        )

    @pytest.mark.parametrize("dataset_case", ALL_CASES, indirect=True)
    def test_can_load_across_multiple_epochs(self, dataset_case):
        dataset = dataset_case.to_dataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=None)
        for _ in range(2):
            loaded_samples = list(loader)
            assert loaded_samples == dataset_case.expected_data_samples

    @pytest.mark.parametrize("dataset_case", ALL_CASES, indirect=True)
    def test_loader_with_workers(self, dataset_case):
        dataset = dataset_case.to_dataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=4)
        loaded_samples = list(loader)
        assert loaded_samples == dataset_case.expected_data_samples

    @pytest.mark.parametrize("dataset_case", ALL_CASES, indirect=True)
    def test_loader_with_batch_size(self, dataset_case):
        dataset = dataset_case.to_dataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        loaded_samples = list(loader)
        assert loaded_samples == self.batched(
            dataset_case.expected_data_samples, batch_size=4
        )

    @pytest.mark.parametrize("dataset_case", MAP_CASES, indirect=True)
    def test_loader_with_shuffle(self, dataset_case):
        dataset = dataset_case.to_dataset()
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=True)
        loaded_samples = list(loader)
        assert sort_samples(loaded_samples) == sort_samples(
            dataset_case.expected_data_samples
        )
