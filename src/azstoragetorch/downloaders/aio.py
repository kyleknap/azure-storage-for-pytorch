import asyncio

from azure.storage.blob.aio import BlobClient

from azstoragetorch.downloaders import BaseBlobDownloader


class AsyncSDKDownloader(BaseBlobDownloader):
    _DEFAULT_MAX_CONCURRENCY = 8
    _CONNECTION_DATA_BLOCK_SIZE = 64 * 1024

    def __init__(self, blob_url, sdk_credential):
        super().__init__(blob_url, sdk_credential)
        self._blob_client = BlobClient.from_blob_url(
            blob_url,
            credential=sdk_credential,
            connection_data_block_size=self._CONNECTION_DATA_BLOCK_SIZE,
            max_single_get_size=4 * 1024 * 1024,
            max_chunk_get_size=8 * 1024 * 1024,
        )
        self._blob_size = None
        self._runner = asyncio.Runner()


    def download(self, offset: int = 0, length=None):
        sdk_downloader = self._runner.run(self._get_downloader(offset, length))
        if self._blob_size is None:
            self._blob_size = sdk_downloader.size
        return self._runner.run(self._read(sdk_downloader))

    def get_blob_size(self):
        if self._blob_size is None:
            self._blob_size = self._runner.run(self._get_blob_properties()).size
        return self._blob_size

    def close(self):
        self._runner.run(self._close_client())
        self._runner.close()

    async def _get_blob_properties(self):
        return await self._blob_client.get_blob_properties()

    async def _get_downloader(self, pos, size):
        return await self._blob_client.download_blob(
            max_concurrency=self._DEFAULT_MAX_CONCURRENCY,
            offset=pos,
            length=size,
        )

    async def _read(self, sdk_downloader):
        return await sdk_downloader.read()

    async def _close_client(self):
        await self._blob_client.close()