# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
from unittest import mock
import pytest

from azure.core.credentials import AzureSasCredential

from azstoragetorch.datasets import BlobDataset, IterableBlobDataset, Blob
from azstoragetorch._client import (
    AzStorageTorchBlobClient,
    AzStorageTorchBlobClientFactory,
)


@pytest.fixture
def create_mock_azstoragetorch_blob_client(blob_url, blob_content):
    def _create_mock_azstoragetorch_blob_client(url=None, data=None):
        if url is None:
            url = blob_url
        if data is None:
            data = blob_content
        client = mock.Mock(AzStorageTorchBlobClient)
        client.url = url
        client.get_blob_size.return_value = len(data)
        client.download.return_value = data
        return client

    return _create_mock_azstoragetorch_blob_client


@pytest.fixture
def mock_azstoragetorch_blob_client(create_mock_azstoragetorch_blob_client):
    return create_mock_azstoragetorch_blob_client()


@pytest.fixture
def blob(mock_azstoragetorch_blob_client):
    return Blob(mock_azstoragetorch_blob_client)


@pytest.fixture
def mock_azstoragetorch_blob_client_factory():
    return mock.Mock(AzStorageTorchBlobClientFactory)


@pytest.fixture(autouse=True)
def azstoragetorch_blob_factory_patch(mock_azstoragetorch_blob_client_factory):
    with mock.patch(
        "azstoragetorch._client.AzStorageTorchBlobClientFactory",
        mock_azstoragetorch_blob_client_factory,
    ):
        mock_azstoragetorch_blob_client_factory.return_value = (
            mock_azstoragetorch_blob_client_factory
        )
        yield mock_azstoragetorch_blob_client_factory


@pytest.fixture
def data_samples(container_url):
    return [
        {"url": f"{container_url}/blob{i}", "data": f"sample data {i}".encode("utf-8")}
        for i in range(10)
    ]


@pytest.fixture
def data_sample_blob_urls(data_samples):
    return [sample["url"] for sample in data_samples]


@pytest.fixture
def data_sample_blob_clients(data_samples, create_mock_azstoragetorch_blob_client):
    return [
        create_mock_azstoragetorch_blob_client(**data_sample)
        for data_sample in data_samples
    ]


class TestBlob:
    def test_url(self, blob, mock_azstoragetorch_blob_client, blob_url):
        mock_azstoragetorch_blob_client.url = blob_url
        assert blob.url == blob_url

    def test_blob_name(self, blob, mock_azstoragetorch_blob_client, blob_name):
        mock_azstoragetorch_blob_client.blob_name = blob_name
        assert blob.blob_name == blob_name

    def test_container_name(
        self, blob, mock_azstoragetorch_blob_client, container_name
    ):
        mock_azstoragetorch_blob_client.container_name = container_name
        assert blob.container_name == container_name

    def test_reader(self, blob, mock_azstoragetorch_blob_client):
        with mock.patch(
            "azstoragetorch.datasets.BlobIO", spec=True
        ) as mock_blob_io_cls:
            reader = blob.reader()
            assert reader is mock_blob_io_cls.return_value
            mock_blob_io_cls.assert_called_once_with(
                blob.url,
                "rb",
                _azstoragetorch_blob_client=mock_azstoragetorch_blob_client,
            )


class TestBlobDataset:
    def assert_expected_dataset(self, dataset, expected_data_samples):
        assert isinstance(dataset, BlobDataset)
        assert len(dataset) == len(expected_data_samples)
        for i in range(len(dataset)):
            assert dataset[i] == expected_data_samples[i]

    def assert_factory_calls_from_container_url(
        self,
        mock_azstoragetorch_blob_client_factory,
        expected_container_url,
        expected_prefix=None,
        expected_credential=None,
    ):
        mock_azstoragetorch_blob_client_factory.assert_called_once_with(
            credential=expected_credential
        )
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.assert_called_once_with(
            expected_container_url, prefix=expected_prefix
        )

    def assert_factory_calls_from_blob_urls(
        self,
        mock_azstoragetorch_blob_client_factory,
        expected_blob_urls,
        expected_credential=None,
    ):
        mock_azstoragetorch_blob_client_factory.assert_called_once_with(
            credential=expected_credential
        )
        assert (
            mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.call_args_list
            == [mock.call(url) for url in expected_blob_urls]
        )

    def test_from_container_url(
        self,
        container_url,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.return_value = data_sample_blob_clients
        dataset = BlobDataset.from_container_url(container_url)
        self.assert_expected_dataset(dataset, expected_data_samples=data_samples)
        self.assert_factory_calls_from_container_url(
            mock_azstoragetorch_blob_client_factory,
            expected_container_url=container_url,
        )

    def test_from_container_url_with_prefix(
        self,
        container_url,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.return_value = data_sample_blob_clients
        dataset = BlobDataset.from_container_url(container_url, prefix="prefix/")
        self.assert_expected_dataset(dataset, expected_data_samples=data_samples)
        self.assert_factory_calls_from_container_url(
            mock_azstoragetorch_blob_client_factory,
            expected_container_url=container_url,
            expected_prefix="prefix/",
        )

    def test_from_container_url_with_credential(
        self,
        container_url,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_clients,
    ):
        credential = AzureSasCredential("sas_token")
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.return_value = data_sample_blob_clients
        dataset = BlobDataset.from_container_url(container_url, credential=credential)
        self.assert_expected_dataset(dataset, expected_data_samples=data_samples)
        self.assert_factory_calls_from_container_url(
            mock_azstoragetorch_blob_client_factory,
            expected_container_url=container_url,
            expected_credential=credential,
        )

    def test_from_container_url_with_transform(
        self,
        container_url,
        mock_azstoragetorch_blob_client_factory,
        data_sample_blob_urls,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.return_value = data_sample_blob_clients
        dataset = BlobDataset.from_container_url(
            container_url, transform=lambda x: x.url
        )
        self.assert_expected_dataset(
            dataset, expected_data_samples=data_sample_blob_urls
        )
        self.assert_factory_calls_from_container_url(
            mock_azstoragetorch_blob_client_factory,
            expected_container_url=container_url,
        )

    def test_from_blob_urls(
        self,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_urls,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.side_effect = (
            data_sample_blob_clients
        )
        dataset = BlobDataset.from_blob_urls(data_sample_blob_urls)
        self.assert_expected_dataset(dataset, expected_data_samples=data_samples)
        self.assert_factory_calls_from_blob_urls(
            mock_azstoragetorch_blob_client_factory,
            expected_blob_urls=data_sample_blob_urls,
        )

    def test_from_blob_urls_with_single_blob_url(
        self,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_urls,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.return_value = data_sample_blob_clients[
            0
        ]
        dataset = BlobDataset.from_blob_urls(data_sample_blob_urls[0])
        self.assert_expected_dataset(dataset, expected_data_samples=[data_samples[0]])
        self.assert_factory_calls_from_blob_urls(
            mock_azstoragetorch_blob_client_factory,
            expected_blob_urls=[data_sample_blob_urls[0]],
        )

    def test_from_blob_urls_with_credential(
        self,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_urls,
        data_sample_blob_clients,
    ):
        credential = AzureSasCredential("sas_token")
        mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.side_effect = (
            data_sample_blob_clients
        )
        dataset = BlobDataset.from_blob_urls(
            data_sample_blob_urls, credential=credential
        )
        self.assert_expected_dataset(dataset, expected_data_samples=data_samples)
        self.assert_factory_calls_from_blob_urls(
            mock_azstoragetorch_blob_client_factory,
            expected_blob_urls=data_sample_blob_urls,
            expected_credential=credential,
        )

    def test_from_blob_urls_with_transform(
        self,
        mock_azstoragetorch_blob_client_factory,
        data_sample_blob_urls,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.side_effect = (
            data_sample_blob_clients
        )
        dataset = BlobDataset.from_blob_urls(
            data_sample_blob_urls, transform=lambda x: x.url
        )
        self.assert_expected_dataset(
            dataset, expected_data_samples=data_sample_blob_urls
        )
        self.assert_factory_calls_from_blob_urls(
            mock_azstoragetorch_blob_client_factory,
            expected_blob_urls=data_sample_blob_urls,
        )


class TestIterableBlobDataset:
    def assert_expected_dataset_instantiation(
        self, dataset, mock_azstoragetorch_blob_client_factory, expected_credential=None
    ):
        assert isinstance(dataset, IterableBlobDataset)
        # An iterable dataset can instaniate a blob client factory but should not immediately be
        # attempting to create blob clients. Those should be created in downstream calls to the
        # instantiated dataset
        mock_azstoragetorch_blob_client_factory.assert_called_once_with(
            credential=expected_credential
        )
        assert not mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.called
        assert (
            not mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.called
        )

    def assert_expected_dataset(self, dataset, expected_data_samples):
        assert isinstance(dataset, IterableBlobDataset)
        assert list(dataset) == expected_data_samples

    def assert_expected_dataset_from_blob_url(
        self,
        dataset,
        mock_azstoragetorch_blob_client_factory,
        expected_data_samples,
        expected_blob_urls,
    ):
        assert isinstance(dataset, IterableBlobDataset)
        for i, data_sample in enumerate(dataset):
            # For from_blob_url, the dataset should be creating blob clients per sample iteration instead
            # of all at once. Iterate through the dataset one at a time to confirm this behavior
            assert data_sample == expected_data_samples[i]
            assert (
                mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.call_args
                == mock.call(expected_blob_urls[i])
            )
            assert (
                mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.call_count
                == i + 1
            )
        assert (
            mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.call_count
            == len(expected_data_samples)
        )

    def test_from_container_url(
        self,
        container_url,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.return_value = data_sample_blob_clients
        dataset = IterableBlobDataset.from_container_url(container_url)
        self.assert_expected_dataset_instantiation(
            dataset, mock_azstoragetorch_blob_client_factory
        )
        self.assert_expected_dataset(dataset, expected_data_samples=data_samples)
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.assert_called_once_with(
            container_url, prefix=None
        )

    def test_from_container_url_with_prefix(
        self,
        container_url,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.return_value = data_sample_blob_clients
        dataset = IterableBlobDataset.from_container_url(
            container_url, prefix="prefix/"
        )
        self.assert_expected_dataset_instantiation(
            dataset, mock_azstoragetorch_blob_client_factory
        )
        self.assert_expected_dataset(dataset, expected_data_samples=data_samples)
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.assert_called_once_with(
            container_url, prefix="prefix/"
        )

    def test_from_container_url_with_credential(
        self,
        container_url,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_clients,
    ):
        credential = AzureSasCredential("sas_token")
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.return_value = data_sample_blob_clients
        dataset = IterableBlobDataset.from_container_url(
            container_url, credential=credential
        )
        self.assert_expected_dataset_instantiation(
            dataset,
            mock_azstoragetorch_blob_client_factory,
            expected_credential=credential,
        )
        self.assert_expected_dataset(dataset, expected_data_samples=data_samples)
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.assert_called_once_with(
            container_url, prefix=None
        )

    def test_from_container_url_with_transform(
        self,
        container_url,
        mock_azstoragetorch_blob_client_factory,
        data_sample_blob_urls,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.return_value = data_sample_blob_clients
        dataset = IterableBlobDataset.from_container_url(
            container_url, transform=lambda x: x.url
        )
        self.assert_expected_dataset_instantiation(
            dataset, mock_azstoragetorch_blob_client_factory
        )
        self.assert_expected_dataset(
            dataset, expected_data_samples=data_sample_blob_urls
        )
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.assert_called_once_with(
            container_url, prefix=None
        )

    def test_from_blob_urls(
        self,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_urls,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.side_effect = (
            data_sample_blob_clients
        )
        dataset = IterableBlobDataset.from_blob_urls(data_sample_blob_urls)
        self.assert_expected_dataset_instantiation(
            dataset, mock_azstoragetorch_blob_client_factory
        )
        self.assert_expected_dataset_from_blob_url(
            dataset,
            mock_azstoragetorch_blob_client_factory,
            expected_data_samples=data_samples,
            expected_blob_urls=data_sample_blob_urls,
        )

    def test_from_blob_urls_with_single_blob_url(
        self,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_urls,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.return_value = data_sample_blob_clients[
            0
        ]
        dataset = IterableBlobDataset.from_blob_urls(data_sample_blob_urls[0])
        self.assert_expected_dataset_instantiation(
            dataset, mock_azstoragetorch_blob_client_factory
        )
        self.assert_expected_dataset_from_blob_url(
            dataset,
            mock_azstoragetorch_blob_client_factory,
            expected_data_samples=[data_samples[0]],
            expected_blob_urls=[data_sample_blob_urls[0]],
        )

    def test_from_blob_urls_with_credential(
        self,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_urls,
        data_sample_blob_clients,
    ):
        credential = AzureSasCredential("sas_token")
        mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.side_effect = (
            data_sample_blob_clients
        )
        dataset = IterableBlobDataset.from_blob_urls(
            data_sample_blob_urls, credential=credential
        )
        self.assert_expected_dataset_instantiation(
            dataset,
            mock_azstoragetorch_blob_client_factory,
            expected_credential=credential,
        )
        self.assert_expected_dataset_from_blob_url(
            dataset,
            mock_azstoragetorch_blob_client_factory,
            expected_data_samples=data_samples,
            expected_blob_urls=data_sample_blob_urls,
        )

    def test_from_blob_urls_with_transform(
        self,
        mock_azstoragetorch_blob_client_factory,
        data_sample_blob_urls,
        data_sample_blob_clients,
    ):
        mock_azstoragetorch_blob_client_factory.get_blob_client_from_url.side_effect = (
            data_sample_blob_clients
        )
        dataset = IterableBlobDataset.from_blob_urls(
            data_sample_blob_urls, transform=lambda x: x.url
        )
        self.assert_expected_dataset_instantiation(
            dataset,
            mock_azstoragetorch_blob_client_factory,
        )
        self.assert_expected_dataset_from_blob_url(
            dataset,
            mock_azstoragetorch_blob_client_factory,
            expected_data_samples=data_sample_blob_urls,
            expected_blob_urls=data_sample_blob_urls,
        )

    @pytest.mark.parametrize(
        "worker_info,expected_data_indices",
        [
            # No workers and one worker should return all data samples
            (None, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            (mock.Mock(id=0, num_workers=1), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
            # Two workers should return every other data sample
            (mock.Mock(id=0, num_workers=2), [0, 2, 4, 6, 8]),
            (mock.Mock(id=1, num_workers=2), [1, 3, 5, 7, 9]),
            # Three workers should return every third data sample
            (mock.Mock(id=0, num_workers=3), [0, 3, 6, 9]),
            (mock.Mock(id=1, num_workers=3), [1, 4, 7]),
            (mock.Mock(id=2, num_workers=3), [2, 5, 8]),
            # A worker should only return a single data sample if the number of workers
            # is greater than or equal to the number of data samples
            (mock.Mock(id=0, num_workers=10), [0]),
            (mock.Mock(id=9, num_workers=10), [9]),
            (mock.Mock(id=7, num_workers=20), [7]),
            # And if a worker is not in the range of data samples, it should return no data samples
            (mock.Mock(id=10, num_workers=20), []),
        ],
    )
    def test_worker_sharding(
        self,
        container_url,
        mock_azstoragetorch_blob_client_factory,
        data_samples,
        data_sample_blob_clients,
        worker_info,
        expected_data_indices,
    ):
        mock_azstoragetorch_blob_client_factory.yield_blob_clients_from_container_url.return_value = data_sample_blob_clients
        dataset = IterableBlobDataset.from_container_url(container_url)
        self.assert_expected_dataset_instantiation(
            dataset, mock_azstoragetorch_blob_client_factory
        )
        with mock.patch(
            "torch.utils.data.get_worker_info", spec=True
        ) as mock_get_worker_info:
            mock_get_worker_info.return_value = worker_info
            expected_data_samples = [data_samples[i] for i in expected_data_indices]
            self.assert_expected_dataset(
                dataset, expected_data_samples=expected_data_samples
            )
