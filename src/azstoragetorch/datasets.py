# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------

import dataclasses
from typing import Optional

import torch.utils.data
import azure.storage.blob

from azstoragetorch.io import BlobIO
from azstoragetorch._client import AzStorageTorchBlobClientFactory as _AzStorageTorchBlobClientFactory
from azstoragetorch import _utils


@dataclasses.dataclass
class Blob:
    url: str
    _blob_client_factory: _AzStorageTorchBlobClientFactory
    
    def stream(self) -> BlobIO:
        return BlobIO(self.url, "rb", azstoragetorch_blob_client_factory=self._blob_client_factory)


class BlobDataset(torch.utils.data.Dataset):
    # TODO: still need to add a transform in here
    def __init__(self, blobs: list[Blob], transform=None):
        self._blobs = blobs
        self._transform = transform

    @classmethod
    def from_container_url(cls, container_url: str, *, name_starts_with: Optional[str], credential: _utils.AZSTORAGETORCH_CREDENTIAL_TYPE = None, list_type: str = "full", transform=None):
        sdk_credential = _utils.to_sdk_credential(container_url, credential)
        container_client = azure.storage.blob.ContainerClient.from_container_url(container_url, credential=sdk_credential)
        blob_client_factory = _AzStorageTorchBlobClientFactory(sdk_credential)
        if list_type == "full":
            blobs = [
                Blob(f"{container_url}/{blob.name}", blob_client_factory) for blob in container_client.list_blobs(name_starts_with=name_starts_with)
            ]
        elif list_type == "name":
            blobs = [
                Blob(f"{container_url}/{blob_name}", blob_client_factory) for blob_name in container_client.list_blob_names(name_starts_with=name_starts_with)
            ]
        return cls(blobs, transform=transform)
    
    @classmethod
    def from_blob_urls(cls, blob_urls: list[str], *, credential: _utils.AZSTORAGETORCH_CREDENTIAL_TYPE = None, transform=None):
        # TODO: Need to figure out if we want to override the credential per url. probably should...
        # TODO: Need to figure out how we share transports across blobs especially if they are not same account or container
        sdk_credential = _utils.to_sdk_credential(blob_urls[0], credential)
        blob_client_factory = _AzStorageTorchBlobClientFactory(sdk_credential)
        blobs = [
            Blob(url, blob_client_factory) for url in blob_urls
        ]
        return cls(blobs, transform=transform)
    
    def __getitem__(self, index):
        blob = self._blobs[index]
        if self._transform:
            return self._transform(blob)
        return blob
    
    def __len__(self):
        return len(self._blobs)
    



class BlobIterableDataset(torch.utils.data.IterableDataset):
    pass