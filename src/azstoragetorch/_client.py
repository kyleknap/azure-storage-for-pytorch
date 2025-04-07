# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------

import concurrent.futures
import functools
import io
import logging
import math
import os
import random
import threading
import time
import uuid
import urllib.parse
from typing import Optional, List, Tuple, Iterator, Union, Literal

import azure.core.exceptions
from azure.core.credentials import (
    AzureSasCredential,
    TokenCredential,
)
from azure.identity import DefaultAzureCredential
import azure.storage.blob
import azure.storage.blob._generated.models
from azure.storage.blob._shared.response_handlers import process_storage_error
from azure.core.pipeline import Pipeline
from azure.core.pipeline.transport import RequestsTransport


_LOGGER = logging.getLogger(__name__)

SDK_CREDENTIAL_TYPE = Optional[
    Union[
        AzureSasCredential,
        TokenCredential,
    ]
]
AZSTORAGETORCH_CREDENTIAL_TYPE = Union[SDK_CREDENTIAL_TYPE, Literal[False]]
SUPPORTED_WRITE_BYTES_LIKE_TYPE = Union[bytes, bytearray, memoryview]
STAGE_BLOCK_FUTURE_TYPE = concurrent.futures.Future[str]


class AzStorageTorchBlobClientFactory:
    _SOCKET_CONNECTION_TIMEOUT = 20
    _SOCKET_READ_TIMEOUT = 60
    _CONNECTION_DATA_BLOCK_SIZE = 256 * 1024

    def __init__(
        self,
        account_url: Optional[str] = None,
        credential: AZSTORAGETORCH_CREDENTIAL_TYPE = None,
    ):
        self._account_url = account_url
        self._sdk_credential = self._get_sdk_credential(credential)
        self._transport = self._get_transport()

    def get_blob_client_from_url(self, blob_url: str) -> "AzStorageTorchBlobClient":
        self._validate_url_matches_account(blob_url)
        blob_sdk_client = self._get_sdk_blob_client_from_url(blob_url)
        return AzStorageTorchBlobClient(blob_sdk_client)

    def get_blob_clients_from_container_url(
        self, container_url: str, name_starts_with: Optional[str] = None
    ) -> list["AzStorageTorchBlobClient"]:
        self._validate_url_matches_account(container_url)
        container_client = self._get_sdk_container_client_from_url(container_url)
        return [
            AzStorageTorchBlobClient(container_client.get_blob_client(blob_name))
            for blob_name in container_client.list_blob_names(
                name_starts_with=name_starts_with
            )
        ]

    @functools.cached_property
    def _shared_pipeline(self) -> Optional[Pipeline]:
        if self._account_url is not None:
            blob_service_client = azure.storage.blob.BlobServiceClient(
                account_url=self._account_url,
                **self._get_sdk_client_kwargs(self._account_url, reuse_pipeline=False),
            )
            return blob_service_client._pipeline
        return None

    def _validate_url_matches_account(self, resource_url: str) -> None:
        if not resource_url.startswith(self._account_url):
            raise ValueError(
                f"URL '{resource_url}' does not start with account URL '{self._account_url}'."
            )

    def _get_sdk_credential(
        self, credential: AZSTORAGETORCH_CREDENTIAL_TYPE
    ) -> SDK_CREDENTIAL_TYPE:
        if credential is False:
            return None
        if credential is None:
            return DefaultAzureCredential()
        if isinstance(credential, (AzureSasCredential, TokenCredential)):
            return credential
        raise TypeError(f"Unsupported credential: {type(credential)}")

    def _get_transport(self) -> RequestsTransport:
        return RequestsTransport(
            connection_timeout=self._SOCKET_CONNECTION_TIMEOUT,
            read_timeout=self._SOCKET_READ_TIMEOUT,
        )

    def _ensure_url_matches_account(self, resource_url: str) -> None:
        if not resource_url.startswith(self._account_url):
            raise ValueError(
                f"URL '{resource_url}' does not start with account URL '{self._account_url}'."
            )

    def _get_sdk_blob_client_from_url(self, blob_url: str):
        return azure.storage.blob.BlobClient.from_blob_url(
            blob_url,
            **self._get_sdk_client_kwargs(blob_url),
        )

    def _get_sdk_container_client_from_url(self, container_url: str):
        return azure.storage.blob.ContainerClient.from_container_url(
            container_url,
            **self._get_sdk_client_kwargs(container_url),
        )

    def _get_sdk_client_kwargs(
        self, resource_url: str, reuse_pipeline: bool = True
    ) -> dict:
        kwargs = {
            "connection_data_block_size": self._CONNECTION_DATA_BLOCK_SIZE,
            "transport": self._transport,
        }
        if not self._url_has_sas_token(resource_url):
            kwargs["credential"] = self._sdk_credential
            if reuse_pipeline and self._shared_pipeline is not None:
                kwargs["_pipeline"] = self._shared_pipeline
        return kwargs

    def _url_has_sas_token(self, resource_url: str) -> bool:
        parsed_url = urllib.parse.urlparse(resource_url)
        if parsed_url.query is None:
            return False
        parsed_qs = urllib.parse.parse_qs(parsed_url.query)
        # The signature is always required in a valid SAS token. So look for the "sig"
        # key to determine if the URL has a SAS token.
        return "sig" in parsed_qs


class AzStorageTorchBlobClient:
    _CONNECTION_DATA_BLOCK_SIZE = 256 * 1024
    _PARTITIONED_DOWNLOAD_THRESHOLD = 16 * 1024 * 1024
    _PARTITION_SIZE = 16 * 1024 * 1024
    _NUM_DOWNLOAD_ATTEMPTS = 3
    _STAGE_BLOCK_SIZE = 32 * 1024 * 1024
    _RETRYABLE_READ_EXCEPTIONS = (
        azure.core.exceptions.IncompleteReadError,
        azure.core.exceptions.HttpResponseError,
        azure.core.exceptions.DecodeError,
    )

    def __init__(
        self,
        sdk_blob_client: azure.storage.blob.BlobClient,
        executor: Optional[concurrent.futures.Executor] = None,
        max_in_flight_requests: Optional[int] = None,
    ):
        self._sdk_blob_client = sdk_blob_client
        self._generated_sdk_storage_client = self._sdk_blob_client._client
        if max_in_flight_requests is None:
            max_in_flight_requests = self._get_max_in_flight_requests()
        if executor is None:
            executor = concurrent.futures.ThreadPoolExecutor(max_in_flight_requests)
        self._executor = executor
        # The standard thread pool executor does not bound the number of tasks submitted to it.
        # This semaphore introduces bound so that the number of submitted, in-progress
        # futures are not greater than the available workers. This is important for cases where we
        # buffer data into memory for uploads as is prevents large amounts of memory from being
        # submitted to the executor when there are no workers available to upload it.
        self._max_in_flight_semaphore = threading.Semaphore(max_in_flight_requests)

    @classmethod
    def from_blob_url(
        cls, blob_url: str, sdk_credential: SDK_CREDENTIAL_TYPE
    ) -> "AzStorageTorchBlobClient":
        sdk_blob_client = azure.storage.blob.BlobClient.from_blob_url(
            blob_url,
            credential=sdk_credential,
            connection_data_block_size=cls._CONNECTION_DATA_BLOCK_SIZE,
        )
        return AzStorageTorchBlobClient(sdk_blob_client)

    def get_blob_size(self) -> int:
        return self._blob_properties.size

    def download(self, offset: int = 0, length: Optional[int] = None) -> bytes:
        length = self._update_download_length_from_blob_size(offset, length)
        if length < self._PARTITIONED_DOWNLOAD_THRESHOLD:
            return self._download_with_retries(offset, length)
        else:
            return self._partitioned_download(offset, length)

    def stage_blocks(
        self, data: SUPPORTED_WRITE_BYTES_LIKE_TYPE
    ) -> List[STAGE_BLOCK_FUTURE_TYPE]:
        if not data:
            raise ValueError("Data must not be empty.")
        stage_block_partitions = self._get_stage_block_partitions(data)
        futures = []
        for pos, length in stage_block_partitions:
            self._max_in_flight_semaphore.acquire()
            future = self._executor.submit(self._stage_block, data[pos : pos + length])
            future.add_done_callback(self._release_in_flight_semaphore)
            futures.append(future)
        return futures

    def commit_block_list(self, block_ids: List[str]) -> None:
        blob_blocks = [azure.storage.blob.BlobBlock(block_id) for block_id in block_ids]
        self._sdk_blob_client.commit_block_list(blob_blocks)

    def close(self) -> None:
        self._executor.shutdown()

    def _get_max_in_flight_requests(self) -> int:
        # Ideally we would just match this value to the max workers of the executor. However
        # the executor class does not publicly expose its max worker count. So, instead we copy
        # the max worker calculation from the executor class and inject it into both the executor
        # and semaphore
        #
        # In Python 3.13, os.process_cpu_count() was added and the ThreadPoolExecutor updated to
        # use os.process_cpu_count() instead of os.cpu_count() when calculating default max workers.
        # To match ThreadPoolExecutor defaults across Python versions, we use process_cpu_count
        # if available, otherwise fall back to os.cpu_count().
        cpu_count_fn = getattr(os, "process_cpu_count", os.cpu_count)
        return min(32, (cpu_count_fn() or 1) + 4)

    @functools.cached_property
    def _blob_properties(self) -> azure.storage.blob.BlobProperties:
        return self._sdk_blob_client.get_blob_properties()

    def _update_download_length_from_blob_size(
        self, offset: int, length: Optional[int] = None
    ) -> int:
        length_from_offset = self.get_blob_size() - offset
        if length is not None:
            return min(length, length_from_offset)
        return length_from_offset

    def _partitioned_download(self, offset: int, length: int) -> bytes:
        futures = []
        for read_partition in self._get_partitions(
            offset, length, self._PARTITION_SIZE
        ):
            futures.append(
                self._executor.submit(self._download_with_retries, *read_partition)
            )
        return b"".join(f.result() for f in futures)

    def _get_partitions(
        self, offset: int, length: int, partition_size: int
    ) -> List[Tuple[int, int]]:
        end = offset + length
        num_partitions = math.ceil(length / partition_size)
        partitions = []
        for i in range(num_partitions):
            start = offset + i * partition_size
            if start >= end:
                break
            size = min(partition_size, end - start)
            partitions.append((start, size))
        return partitions

    def _download_with_retries(self, pos: int, length: int) -> bytes:
        attempt = 0
        while self._attempts_remaining(attempt):
            stream = self._get_download_stream(pos, length)
            try:
                return self._read_stream(stream)
            except self._RETRYABLE_READ_EXCEPTIONS:
                backoff_time = self._get_backoff_time(attempt)
                attempt += 1
                if not self._attempts_remaining(attempt):
                    raise
                _LOGGER.debug(
                    "Sleeping %s seconds and retrying download from caught streaming exception (attempts remaining: %s).",
                    backoff_time,
                    self._attempts_remaining(attempt),
                    exc_info=True,
                )
                time.sleep(backoff_time)

    def _get_download_stream(self, pos: int, length: int) -> Iterator[bytes]:
        try:
            return self._generated_sdk_storage_client.blob.download(
                range=f"bytes={pos}-{pos + length - 1}",
                modified_access_conditions=azure.storage.blob._generated.models.ModifiedAccessConditions(
                    if_match=self._blob_properties.etag
                ),
            )
        except azure.core.exceptions.HttpResponseError as e:
            # TODO: This is so that we properly map exceptions from the generated client to the correct
            # exception class and error code. In the future, prior to a GA, we should consider pulling
            # in this function or a derivative of it if we plan to continue to raise Azure Python SDK
            # exceptions from this library (i.e. instead of raising our own exception classes).
            process_storage_error(e)

    def _attempts_remaining(self, attempt_number: int) -> int:
        return max(self._NUM_DOWNLOAD_ATTEMPTS - attempt_number, 0)

    def _get_backoff_time(self, attempt_number: int) -> float:
        # Backoff time uses exponential backoff with full jitter as a starting point to have at least
        # some delay before retrying. For exceptions that we get while streaming data, it will likely be
        # because of environment's network (e.g. high network load) so the approach will give some amount
        # of backoff and randomness before attempting to stream again. In the future, we should
        # consider other approaches such as adapting/throttling stream reading speeds to reduce occurrences
        # of connection errors due to an overwhelmed network.
        return min(random.uniform(0, 2**attempt_number), 20)

    def _read_stream(self, stream: Iterator[bytes]) -> bytes:
        content = io.BytesIO()
        for chunk in stream:
            content.write(chunk)
        return content.getvalue()

    def _get_stage_block_partitions(
        self, data: SUPPORTED_WRITE_BYTES_LIKE_TYPE
    ) -> List[Tuple[int, int]]:
        return self._get_partitions(0, len(data), self._STAGE_BLOCK_SIZE)

    def _stage_block(self, data: SUPPORTED_WRITE_BYTES_LIKE_TYPE) -> str:
        block_id = str(uuid.uuid4())
        self._sdk_blob_client.stage_block(block_id, data)
        return block_id

    def _release_in_flight_semaphore(self, _: STAGE_BLOCK_FUTURE_TYPE) -> None:
        self._max_in_flight_semaphore.release()
