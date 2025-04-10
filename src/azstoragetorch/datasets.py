# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------

import dataclasses
import urllib.parse
from typing import Optional

import torch.utils.data

from azstoragetorch.io import BlobIO
from azstoragetorch import _client


@dataclasses.dataclass
class Blob:
    _blob_client: _client.AzStorageTorchBlobClient

    def stream(self) -> BlobIO:
        return BlobIO("placeholder", "rb", azstoragetorch_blob_client=self._blob_client)


class BlobDataset(torch.utils.data.Dataset):
    def __init__(self, blobs: list[Blob], transform=None):
        self._blobs = blobs
        self._transform = transform

    @classmethod
    def from_blob_urls(
        cls,
        blob_urls: list[str],
        *,
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
        transform=None,
        **blob_factory_kwargs,
    ):
        blob_client_factory = cls._get_blob_client_factory(blob_urls[0], credential, **blob_factory_kwargs)
        blobs = [
            Blob(blob_client_factory.get_blob_client_from_url(blob_url))
            for blob_url in blob_urls
        ]
        return cls(blobs, transform=transform)

    @classmethod
    def from_container_url(
        cls,
        container_url: str,
        *,
        name_starts_with: Optional[str],
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
        transform=None,
        **blob_factory_kwargs,
    ):
        blob_client_factory = cls._get_blob_client_factory(container_url, credential, **blob_factory_kwargs)
        blobs = [
            Blob(blob_client)
            for blob_client in blob_client_factory.get_blob_clients_from_container_url(
                container_url, name_starts_with=name_starts_with
            )
        ]
        return cls(blobs, transform=transform)

    def __getitem__(self, index):
        blob = self._blobs[index]
        if self._transform:
            return self._transform(blob)
        return blob

    def __len__(self):
        return len(self._blobs)

    @staticmethod
    def _get_blob_client_factory(
        resource_url: str, credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE,
        list_type: str = "name",
        proxy_blob_properties: bool = False,
    ) -> _client.AzStorageTorchBlobClientFactory:
        parsed_url = urllib.parse.urlparse(resource_url)
        account_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        return _client.AzStorageTorchBlobClientFactory(
            account_url=account_url, credential=credential,
            list_type=list_type,
            proxy_blob_properties=proxy_blob_properties,
        )


class BlobIterableDataset(torch.utils.data.IterableDataset):
    pass
