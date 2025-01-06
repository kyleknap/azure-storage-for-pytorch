import concurrent.futures
import io
from typing import Optional

from azure.storage.blob import BlobClient
import certifi
import urllib3

from azstoragetorch.downloaders import BaseBlobDownloader
from azstoragetorch.downloaders.utils import get_partitioned_reads


class SDKDownloader(BaseBlobDownloader):
    _DEFAULT_MAX_CONCURRENCY = 8
    _CONNECTION_DATA_BLOCK_SIZE = 64 * 1024

    def __init__(self, blob_url: str, sdk_credential):
        super().__init__(blob_url, sdk_credential)
        self._blob_client = BlobClient.from_blob_url(
            self._blob_url,
            credential=self._sdk_credential,
            connection_data_block_size=self._CONNECTION_DATA_BLOCK_SIZE,
            max_single_get_size=4 * 1024 * 1024,
            max_chunk_get_size=8 * 1024 * 1024,
        )
        self._blob_size = None

    def download(self, offset: int = 0, length: Optional[int] = None) -> bytes:
        sdk_downloader = self._blob_client.download_blob(
            max_concurrency=self._DEFAULT_MAX_CONCURRENCY, offset=offset, length=length
        )
        if self._blob_size is None:
            self._blob_size = sdk_downloader.size
        return sdk_downloader.read()

    def get_blob_size(self) -> int:
        if self._blob_size is None:
            self._blob_size = self._blob_client.get_blob_properties().size
        return self._blob_size

    def close(self):
        pass


class GeneratedSDKDownloader(SDKDownloader):
    _DEFAULT_MAX_CONCURRENCY = 8
    _THRESHOLD = 4 * 1024 * 1024
    _PARTITION_SIZE = 8 * 1024 * 1024

    def __init__(self, blob_url: str, sdk_credential):
        super().__init__(blob_url, sdk_credential)
        self._generated_blob_client = self._blob_client._client
        self._pool = concurrent.futures.ThreadPoolExecutor(max_workers=self._DEFAULT_MAX_CONCURRENCY)

    def download(self, offset: int = 0, length: Optional[int] = None) -> bytes:
        length = self._update_download_length_from_blob_size(offset, length)
        if length < self._THRESHOLD:
            return self._download(offset, length)
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
            futures.append(self._pool.submit(self._download, *partition))
        return b"".join(f.result() for f in futures)

    def _download(self, pos: int, length: int):
        stream = self._generated_blob_client.blob.download(
            range=f"bytes={pos}-{pos + length - 1}",
        )
        content = io.BytesIO()
        for chunk in stream:
            content.write(chunk)
        return content.getvalue()


class Urllib3Downloader(GeneratedSDKDownloader):
    def __init__(self, blob_url, credential):
        self._blob_url = blob_url
        self._credential = credential
        self._blob_size = None
        self._request_pool = urllib3.PoolManager(
            maxsize=10,
            ca_certs=certifi.where(),
        )
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._DEFAULT_MAX_CONCURRENCY
        )

    def get_blob_size(self):
        if self._blob_size is None:
            self._blob_size = self._get_blob_size()
        return self._blob_size

    def close(self):
        self._pool.shutdown()
        self._request_pool.clear()

    def _get_blob_size(self):
        resp = self._request_pool.request(
            "HEAD",
            f"{self._blob_url}",
            headers={"x-ms-version": "2025-01-05"},
        )
        self._raise_for_status(resp)
        return int(resp.headers["Content-Length"])

    def _download(self, pos, length):
        resp = self._request_pool.request(
            "GET",
            f"{self._blob_url}",
            headers={
                "x-ms-version": "2025-01-05",
                "Range": f"bytes={pos}-{pos + length - 1}",
            },
            preload_content=False,
        )
        self._raise_for_status(resp)
        content = io.BytesIO()
        for chunk in resp.stream(self._CONNECTION_DATA_BLOCK_SIZE):
            content.write(chunk)
        return content.getvalue()

    def _raise_for_status(self, response):
        if response.status >= 300:
            raise RuntimeError(
                f"Response failed: ({response.status}) {response.reason}"
            )
