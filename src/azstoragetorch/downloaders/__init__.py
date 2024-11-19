from typing import Optional


class BaseBlobDownloader:
    def __init__(self, blob_url: str, sdk_credential):
        self._blob_url = blob_url
        self._sdk_credential = sdk_credential

    def get_blob_size(self) -> int:
        raise NotImplementedError("get_blob_size")

    def download(self, offset: int = 0, length: Optional[int] = None) -> bytes:
        raise NotImplementedError("download")

    def close(self) -> None:
        raise NotImplementedError("close")