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
from typing import Optional, List, Tuple, Iterator, Union, Type

import azure.core.exceptions
import azure.storage.blob
import azure.storage.blob._generated.models
from azure.storage.blob._shared.response_handlers import process_storage_error
from azure.storage.blob._shared.base_client import TransportWrapper
from azure.core.pipeline.transport import RequestsTransport


from azstoragetorch._utils import SDK_CREDENTIAL_TYPE


_LOGGER = logging.getLogger(__name__)

SUPPORTED_WRITE_BYTES_LIKE_TYPE = Union[bytes, bytearray, memoryview]
STAGE_BLOCK_FUTURE_TYPE = concurrent.futures.Future[str]


class AzStorageTorchBlobClientFactory:
    def __init__(self, sdk_credential: SDK_CREDENTIAL_TYPE):
        self._sdk_credential = sdk_credential
        # TODO: We probably want to share the executor and semaphore here.
        # TODO: Figure out how to 1) instantiate from containr client and 2) share client resources
        # given arbitrary blob urls
        self._client = None
        self._sdk_client = None
        self._sdk_container_client = None
        self._transport = None
        self._pipeline = None
        self._blob_properties = None
        self._blob_url = None

    # Probably want to accept blob properties here too so we can proxy data from the list
    def get_blob_client(self, blob_url: str, **kwargs) -> "AzStorageTorchBlobClient":
        start = time.time()
        blob_client = self._get_blob_client(blob_url, **kwargs)
        print(f"get_blob_client took {time.time() - start} seconds")
        return blob_client

    def _get_blob_client(self, blob_url: str, **kwargs) -> "AzStorageTorchBlobClient":
        if kwargs.get("cached", False):
            return self._get_blob_client_cached(blob_url)
        if kwargs.get("cached_sdk_client", False):
            return self._get_blob_client_cached_sdk_client(blob_url)
        if kwargs.get("cached_container_client", False):
            return self._get_blob_client_cached_container_client(blob_url)
        if kwargs.get("cached_transport", False):
            return self._get_blob_client_cached_transport(blob_url)
        if kwargs.get("cached_pipeline", False):
            return self._get_blob_client_cached_pipeline(blob_url)
        if kwargs.get("cached_blob_properties", False):
            return self._get_blob_client_cached_blob_properties(blob_url)
        return self._get_azstoragetorch_blob_client_from_url(blob_url)
    
    def _get_blob_client_cached(self, blob_url: str):
        if self._client is None:
            self._client = self._get_azstoragetorch_blob_client_from_url(blob_url)
            self._client.close = lambda: None  # To not close executor
        return self._client
    
    def _get_blob_client_cached_sdk_client(self, blob_url: str):
        if self._sdk_client is None:
            self._sdk_client = self._get_sdk_blob_client_from_url(blob_url)
        return AzStorageTorchBlobClient(self._sdk_client)
    
    def _get_blob_client_cached_container_client(self, blob_url: str):
        if self._sdk_container_client is None:
            self._sdk_container_client = self._get_sdk_container_client_from_url(blob_url)
        blob_name = blob_url[len(self._sdk_container_client.url) + 1 :]
        blob_client = self._sdk_container_client.get_blob_client(blob_name)
        return AzStorageTorchBlobClient(blob_client)

    def _get_blob_client_cached_transport(self, blob_url: str):
        if self._transport is None:
            # self._transport = TransportWrapper(RequestsTransport(connection_timeout=20, read_timeout=60))
            self._transport = RequestsTransport(connection_timeout=20, read_timeout=60)
        sdk_client = self._get_sdk_blob_client_from_url(blob_url, transport=self._transport)
        return AzStorageTorchBlobClient(sdk_client)
    
    def _get_blob_client_cached_pipeline(self, blob_url: str):
        if self._pipeline is None:
            sdk_blob_client = self._get_sdk_blob_client_from_url(blob_url)
            self._pipeline = sdk_blob_client._pipeline
        sdk_blob_client = self._get_sdk_blob_client_from_url(blob_url, pipeline=self._pipeline)
        return AzStorageTorchBlobClient(sdk_blob_client)
    
    def _get_blob_client_cached_blob_properties(self, blob_url: str):
        if self._blob_properties is None:
            client = self._get_azstoragetorch_blob_client_from_url(blob_url)
            client.get_blob_size()
            self._blob_properties = client._blob_properties
            self._blob_url = blob_url

        # if self._transport is None:
        #     self._transport = RequestsTransport(connection_timeout=20, read_timeout=60)
        # sdk_client = self._get_sdk_blob_client_from_url(self._blob_url, transport=self._transport)
        # client = AzStorageTorchBlobClient(sdk_client)
        client = self._get_azstoragetorch_blob_client_from_url(self._blob_url)
        client._blob_properties = self._blob_properties
        return client


    def _get_azstoragetorch_blob_client_from_url(self, blob_url: str):
        return AzStorageTorchBlobClient.from_blob_url(
            blob_url, sdk_credential=self._sdk_credential
        )
    
    def _get_sdk_blob_client_from_url(self, blob_url: str, transport=None, pipeline=None):
        return azure.storage.blob.BlobClient.from_blob_url(
            blob_url, credential=self._sdk_credential,
            connection_data_block_size=AzStorageTorchBlobClient._CONNECTION_DATA_BLOCK_SIZE,
            transport=transport,
            _pipeline=pipeline,
        )
    
    def _get_sdk_container_client_from_url(self, blob_url: str):
        parsed = urllib.parse.urlparse(blob_url)
        container_name = parsed.path.split("/")[1]
        container_url = f"{parsed.scheme}://{parsed.netloc}/{container_name}"
        return azure.storage.blob.ContainerClient.from_container_url(
            container_url, credential=self._sdk_credential,
            connection_data_block_size=AzStorageTorchBlobClient._CONNECTION_DATA_BLOCK_SIZE,
        )


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
