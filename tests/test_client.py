# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import concurrent.futures
from unittest import mock
import os
import pytest

from azure.core.credentials import AzureSasCredential
import azure.core.exceptions
from azure.core.pipeline.transport import HttpRequest, HttpResponse
from azure.storage.blob import BlobClient, BlobProperties, StorageErrorCode
from azure.storage.blob._generated._azure_blob_storage import AzureBlobStorage
from azure.storage.blob._generated.operations import BlobOperations
from azure.storage.blob._generated.models import ModifiedAccessConditions

from azstoragetorch._client import AzStorageTorchBlobClient

MB = 1024 * 1024
DEFAULT_PARTITION_DOWNLOAD_THRESHOLD = 16 * MB
DEFAULT_PARTITION_SIZE = 16 * MB
EXPECTED_RETRYABLE_READ_EXCEPTIONS = [
    azure.core.exceptions.IncompleteReadError,
    azure.core.exceptions.HttpResponseError,
    azure.core.exceptions.DecodeError,
]


@pytest.fixture(autouse=True)
def sleep_patch():
    with mock.patch("time.sleep") as patched_sleep:
        yield patched_sleep


@pytest.fixture
def blob_etag():
    return "blob-etag"


@pytest.fixture
def blob_properties(blob_length, blob_etag):
    return BlobProperties(**{"Content-Length": blob_length, "ETag": blob_etag})


@pytest.fixture
def mock_generated_sdk_storage_client():
    mock_generated_sdk_client = mock.Mock(AzureBlobStorage)
    mock_generated_sdk_client.blob = mock.Mock(BlobOperations)
    return mock_generated_sdk_client


@pytest.fixture
def mock_sdk_blob_client(mock_generated_sdk_storage_client):
    mock_sdk_client = mock.Mock(BlobClient)
    mock_sdk_client._client = mock_generated_sdk_storage_client
    return mock_sdk_client


@pytest.fixture
def single_threaded_executor():
    return concurrent.futures.ThreadPoolExecutor(max_workers=1)


@pytest.fixture
def azstoragetorch_blob_client(mock_sdk_blob_client, single_threaded_executor):
    return AzStorageTorchBlobClient(
        mock_sdk_blob_client, executor=single_threaded_executor
    )


@pytest.fixture
def http_response_error(blob_url):
    mock_http_response = mock.Mock(HttpResponse)
    mock_http_response.reason = "message"
    mock_http_response.status_code = 400
    mock_http_response.headers = {}
    mock_http_response.content_type = "application/xml"
    mock_http_response.text.return_value = ""
    return azure.core.exceptions.HttpResponseError(response=mock_http_response)


def random_bytes(length):
    return os.urandom(length)


def slice_bytes(content, range_value):
    start, end = range_value.split("-")
    return content[int(start) : int(end) + 1]


def to_bytes_iterator(content, chunk_size=64 * 1024, exception_to_raise=None):
    for i in range(0, len(content), chunk_size):
        yield content[i : i + chunk_size]
        if exception_to_raise is not None:
            raise exception_to_raise


class NonRetryableException(Exception):
    pass


class TestAzStorageTorchBlobClient:
    def assert_expected_download_calls(
        self, mock_generated_sdk_storage_client, expected_ranges, expected_etag
    ):
        expected_download_calls = [
            mock.call(
                range=f"bytes={expected_range}",
                modified_access_conditions=ModifiedAccessConditions(
                    if_match=expected_etag
                ),
            )
            for expected_range in expected_ranges
        ]
        assert (
            mock_generated_sdk_storage_client.blob.download.call_args_list
            == expected_download_calls
        )

    def test_from_blob_url(self, blob_url, mock_sdk_blob_client):
        credential = AzureSasCredential("sas")
        with mock.patch("azure.storage.blob.BlobClient") as patched_sdk_blob_client:
            client = AzStorageTorchBlobClient.from_blob_url(blob_url, credential)
            assert isinstance(client, AzStorageTorchBlobClient)
            patched_sdk_blob_client.from_blob_url.assert_called_once_with(
                blob_url,
                credential=credential,
                connection_data_block_size=256 * 1024,
            )

    def test_get_blob_size(
        self, azstoragetorch_blob_client, mock_sdk_blob_client, blob_properties
    ):
        mock_sdk_blob_client.get_blob_properties.return_value = blob_properties
        assert azstoragetorch_blob_client.get_blob_size() == blob_properties.size
        mock_sdk_blob_client.get_blob_properties.assert_called_once_with()

    def test_get_blob_size_caches_result(
        self, azstoragetorch_blob_client, mock_sdk_blob_client, blob_properties
    ):
        mock_sdk_blob_client.get_blob_properties.return_value = blob_properties
        assert azstoragetorch_blob_client.get_blob_size() == blob_properties.size
        assert azstoragetorch_blob_client.get_blob_size() == blob_properties.size
        mock_sdk_blob_client.get_blob_properties.assert_called_once_with()

    def test_close(self, mock_sdk_blob_client):
        mock_executor = mock.Mock(concurrent.futures.Executor)
        client = AzStorageTorchBlobClient(mock_sdk_blob_client, mock_executor)
        client.close()
        mock_executor.shutdown.assert_called_once_with()

    @pytest.mark.parametrize(
        "blob_size, download_offset, download_length, expected_ranges",
        [
            # Small single GET full, download
            (10, None, None, ["0-9"]),
            # Small download with offset
            (10, 5, None, ["5-9"]),
            # Small download with length
            (10, 0, 5, ["0-4"]),
            # Small download with offset and length
            (10, 3, 4, ["3-6"]),
            # Small download with length past blob size
            (10, 5, 10, ["5-9"]),
            # Small download of portion of large blob
            (32 * MB, 10, 10, ["10-19"]),
            # Download just below partitioned threshold
            (
                DEFAULT_PARTITION_DOWNLOAD_THRESHOLD - 1,
                None,
                None,
                [f"0-{DEFAULT_PARTITION_DOWNLOAD_THRESHOLD - 2}"],
            ),
            # Download at partitioned threshold
            (
                DEFAULT_PARTITION_DOWNLOAD_THRESHOLD,
                None,
                None,
                [f"0-{DEFAULT_PARTITION_DOWNLOAD_THRESHOLD - 1}"],
            ),
            # Download just above partitioned threshold
            (
                DEFAULT_PARTITION_DOWNLOAD_THRESHOLD + 1,
                None,
                None,
                [
                    f"0-{DEFAULT_PARTITION_DOWNLOAD_THRESHOLD - 1}",
                    f"{DEFAULT_PARTITION_DOWNLOAD_THRESHOLD}-{DEFAULT_PARTITION_DOWNLOAD_THRESHOLD}",
                ],
            ),
            # Large download with multiple partitions
            (
                4 * DEFAULT_PARTITION_SIZE,
                None,
                None,
                [
                    f"0-{DEFAULT_PARTITION_SIZE - 1}",
                    f"{DEFAULT_PARTITION_SIZE}-{2 * DEFAULT_PARTITION_SIZE - 1}",
                    f"{2 * DEFAULT_PARTITION_SIZE}-{3 * DEFAULT_PARTITION_SIZE - 1}",
                    f"{3 * DEFAULT_PARTITION_SIZE}-{4 * DEFAULT_PARTITION_SIZE - 1}",
                ],
            ),
            # Large download with offset
            (
                4 * DEFAULT_PARTITION_SIZE,
                10,
                None,
                [
                    f"10-{10 + (DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + DEFAULT_PARTITION_SIZE}-{10 + (2 * DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + (2 * DEFAULT_PARTITION_SIZE)}-{10 + (3 * DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + (3 * DEFAULT_PARTITION_SIZE)}-{4 * DEFAULT_PARTITION_SIZE - 1}",
                ],
            ),
            # Large download with length
            (
                4 * DEFAULT_PARTITION_SIZE,
                None,
                2 * DEFAULT_PARTITION_SIZE + 5,
                [
                    f"0-{DEFAULT_PARTITION_SIZE - 1}",
                    f"{DEFAULT_PARTITION_SIZE}-{2 * DEFAULT_PARTITION_SIZE - 1}",
                    f"{2 * DEFAULT_PARTITION_SIZE}-{2 * DEFAULT_PARTITION_SIZE + 4}",
                ],
            ),
            # Large download with offset and length
            (
                4 * DEFAULT_PARTITION_SIZE,
                10,
                2 * DEFAULT_PARTITION_SIZE + 5,
                [
                    f"10-{10 + (DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + DEFAULT_PARTITION_SIZE}-{10 + (2 * DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + (2 * DEFAULT_PARTITION_SIZE)}-{10 + (2 * DEFAULT_PARTITION_SIZE + 4)}",
                ],
            ),
            # Large download with length past blob size
            (
                4 * DEFAULT_PARTITION_SIZE,
                10,
                5 * DEFAULT_PARTITION_SIZE,
                [
                    f"10-{10 + (DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + DEFAULT_PARTITION_SIZE}-{10 + (2 * DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + (2 * DEFAULT_PARTITION_SIZE)}-{10 + (3 * DEFAULT_PARTITION_SIZE - 1)}",
                    f"{10 + (3 * DEFAULT_PARTITION_SIZE)}-{4 * DEFAULT_PARTITION_SIZE - 1}",
                ],
            ),
        ],
    )
    def test_download(
        self,
        blob_size,
        download_offset,
        download_length,
        expected_ranges,
        azstoragetorch_blob_client,
        mock_sdk_blob_client,
        mock_generated_sdk_storage_client,
        blob_properties,
    ):
        blob_properties.size = blob_size
        mock_sdk_blob_client.get_blob_properties.return_value = blob_properties

        content = random_bytes(blob_size)
        mock_generated_sdk_storage_client.blob.download.side_effect = [
            to_bytes_iterator(slice_bytes(content, expected_range))
            for expected_range in expected_ranges
        ]
        download_kwargs = {}
        expected_download_content = content
        if download_offset is not None:
            download_kwargs["offset"] = download_offset
            expected_download_content = expected_download_content[download_offset:]
        if download_length is not None:
            download_kwargs["length"] = download_length
            expected_download_content = expected_download_content[:download_length]
        assert (
            azstoragetorch_blob_client.download(**download_kwargs)
            == expected_download_content
        )
        self.assert_expected_download_calls(
            mock_generated_sdk_storage_client,
            expected_ranges=expected_ranges,
            expected_etag=blob_properties.etag,
        )

    @pytest.mark.parametrize(
        "response_error_code,expected_sdk_exception,expected_storage_error_code",
        [
            (
                "BlobNotFound",
                azure.core.exceptions.ResourceNotFoundError,
                StorageErrorCode.BLOB_NOT_FOUND,
            ),
            (
                "ConditionNotMet",
                azure.core.exceptions.ResourceModifiedError,
                StorageErrorCode.CONDITION_NOT_MET,
            ),
        ],
    )
    def test_maps_download_exceptions(
        self,
        azstoragetorch_blob_client,
        mock_sdk_blob_client,
        mock_generated_sdk_storage_client,
        blob_properties,
        http_response_error,
        response_error_code,
        expected_sdk_exception,
        expected_storage_error_code,
    ):
        http_response_error.response.text.return_value = (
            f'<?xml version="1.0" encoding="utf-8"?>'
            f" <Error><Code>{response_error_code}</Code>"
            f" <Message>message</Message>"
            f"</Error>"
        )

        mock_sdk_blob_client.get_blob_properties.return_value = blob_properties
        mock_generated_sdk_storage_client.blob.download.side_effect = (
            http_response_error
        )
        with pytest.raises(expected_sdk_exception) as exc_info:
            azstoragetorch_blob_client.download()
        assert exc_info.value.error_code == expected_storage_error_code

    @pytest.mark.parametrize(
        "retryable_exception_cls", EXPECTED_RETRYABLE_READ_EXCEPTIONS
    )
    def test_retries_reads(
        self,
        retryable_exception_cls,
        azstoragetorch_blob_client,
        mock_sdk_blob_client,
        mock_generated_sdk_storage_client,
        blob_properties,
        sleep_patch,
    ):
        content = random_bytes(10)
        blob_properties.size = len(content)
        mock_sdk_blob_client.get_blob_properties.return_value = blob_properties
        mock_generated_sdk_storage_client.blob.download.side_effect = [
            to_bytes_iterator(content, exception_to_raise=retryable_exception_cls()),
            to_bytes_iterator(content),
        ]
        assert azstoragetorch_blob_client.download() == content
        self.assert_expected_download_calls(
            mock_generated_sdk_storage_client, ["0-9", "0-9"], blob_properties.etag
        )
        assert sleep_patch.call_count == 1

    @pytest.mark.parametrize(
        "retryable_exception_cls", EXPECTED_RETRYABLE_READ_EXCEPTIONS
    )
    def test_raises_after_retries_exhausted(
        self,
        retryable_exception_cls,
        azstoragetorch_blob_client,
        mock_sdk_blob_client,
        mock_generated_sdk_storage_client,
        blob_properties,
        sleep_patch,
    ):
        content = random_bytes(10)
        blob_properties.size = len(content)
        mock_sdk_blob_client.get_blob_properties.return_value = blob_properties
        mock_generated_sdk_storage_client.blob.download.side_effect = [
            to_bytes_iterator(content, exception_to_raise=retryable_exception_cls()),
            to_bytes_iterator(content, exception_to_raise=retryable_exception_cls()),
            to_bytes_iterator(content, exception_to_raise=retryable_exception_cls()),
        ]
        with pytest.raises(retryable_exception_cls):
            azstoragetorch_blob_client.download()
        self.assert_expected_download_calls(
            mock_generated_sdk_storage_client,
            ["0-9", "0-9", "0-9"],
            blob_properties.etag,
        )
        assert sleep_patch.call_count == 2

    def test_does_not_retry_on_non_retryable_exceptions(
        self,
        azstoragetorch_blob_client,
        mock_sdk_blob_client,
        mock_generated_sdk_storage_client,
        blob_properties,
    ):
        content = random_bytes(10)
        blob_properties.size = len(content)
        mock_sdk_blob_client.get_blob_properties.return_value = blob_properties
        mock_generated_sdk_storage_client.blob.download.side_effect = [
            to_bytes_iterator(content, exception_to_raise=NonRetryableException()),
        ]
        with pytest.raises(NonRetryableException):
            azstoragetorch_blob_client.download()
        self.assert_expected_download_calls(
            mock_generated_sdk_storage_client,
            ["0-9"],
            blob_properties.etag,
        )
