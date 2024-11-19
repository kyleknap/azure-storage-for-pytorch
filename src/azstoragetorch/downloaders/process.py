import concurrent.futures
import copy
import io
from typing import Optional

from azure.storage.blob import BlobClient

from azstoragetorch.downloaders import BaseBlobDownloader
from azstoragetorch.downloaders.utils import get_partitioned_reads


BLOB_CLIENT = None


def init_process(blob_url, credentials):
    global BLOB_CLIENT
    BLOB_CLIENT = BlobClient.from_blob_url(
        blob_url,
        credential=credentials,
        connection_data_block_size=64 * 1024,
    )


def download(pos, length, blob_client=None):
    if blob_client is None:
        global BLOB_CLIENT
        blob_client = BLOB_CLIENT
    stream = blob_client._client.blob.download(
        range=f"bytes={pos}-{pos + length - 1}",
    )
    content = io.BytesIO()
    for chunk in stream:
        content.write(chunk)
    ret = content.getvalue()
    return ret


class ProcessPoolDownloader(BaseBlobDownloader):
    _DEFAULT_MAX_CONCURRENCY = 8
    _CONNECTION_DATA_BLOCK_SIZE = 64 * 1024
    _THRESHOLD = 4 * 1024 * 1024
    _PARTITION_SIZE = 8 * 1024 * 1024
    _EXECUTOR_CLS = concurrent.futures.ProcessPoolExecutor

    def __init__(self, blob_url, credential):
        super().__init__(blob_url, credential)
        self._blob_client = BlobClient.from_blob_url(
            blob_url,
            credential=credential,
            connection_data_block_size=64 * 1024,
        )
        self._blob_size = None
        self._pool = self._EXECUTOR_CLS(
            max_workers=self._DEFAULT_MAX_CONCURRENCY,
            initializer=init_process,
            initargs=(blob_url, copy.deepcopy(credential)),
        )

    def get_blob_size(self) -> int:
        if self._blob_size is None:
            self._blob_size = self._blob_client.get_blob_properties().size
        return self._blob_size

    def download(self, offset: int = 0, length: Optional[int] = None):
        length = self._update_download_length_from_blob_size(offset, length)
        if length < self._THRESHOLD:
            return download(offset, length, self._blob_client)
        else:
            return self._partitioned_download(offset, length)

    def close(self):
        self._pool.shutdown()

    def _update_download_length_from_blob_size(self, offset: int, length: Optional[int] = None) -> int:
        length_from_offset = self.get_blob_size() - offset
        if length is not None:
            return min(length, length_from_offset)
        return length_from_offset

    def _partitioned_download(self, offset: int, length: int):
        futures = []
        for partition in get_partitioned_reads(offset, length, self._PARTITION_SIZE):
            futures.append(self._pool.submit(download, *partition))
        return b"".join(f.result() for f in futures)
