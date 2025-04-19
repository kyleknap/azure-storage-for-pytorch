# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------

from collections.abc import Callable, Iterable, Iterator
from typing import Optional, Union, TypeVar, TypedDict

import torch.utils.data

from azstoragetorch.io import BlobIO
from azstoragetorch import _client


TransformOutputType_co = TypeVar("TransformOutputType_co", covariant=True)
""":py:class:`~typing.TypeVar` representing the return type of a dataset's ``transform`` callable.

This can be of any type.
"""

_TRANSFORM_TYPE = Callable[["Blob"], TransformOutputType_co]


class BlobToDictTransformOutput(TypedDict):
    """A dictionary representation of :py:class:`Blob`.

    For example::

        {
            "url": "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>/<my-blob-name>",
            "data": b"<my-blob-content>"
        }

    This is return type of the default ``transform`` for :py:class:`BlobDataset`
    and :py:class:`IterableBlobDataset`.
    """
    url: str
    """The full endpoint URL of the blob."""
    data: bytes
    """The content of the blob as :py:class:`bytes`."""


def _blob_to_dict(blob: "Blob") -> BlobToDictTransformOutput:
    with blob.reader() as f:
        content = f.read()
    ret: BlobToDictTransformOutput = {
        "url": blob.url,
        "data": content,
    }
    return ret


class Blob:
    """Object representing a single blob in a dataset.

    Datasets instantiate :py:class:`Blob` objects and pass them directly to a dataset's
    ``transform`` callable. Within the ``transform`` callable, use properties and methods
    to access a blob's properties and content. For example::
    
        def to_bytes(blob: Blob) -> bytes:
            with blob.reader() as f:
                return f.read()

        dataset = BlobDataset.from_blob_urls(
            "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>/<my-blob-name>"
            transform=to_bytes
        )
        print(len(dataset[0]))  # prints the length of the blob content
        
    
    Instaniating directly using ``__init__()`` is not supported.
    """
    def __init__(self, blob_client: _client.AzStorageTorchBlobClient):
        self._blob_client = blob_client

    @property
    def url(self) -> str:
        """The full endpoint URL of the blob."""
        return self._blob_client.url

    @property
    def blob_name(self) -> str:
        """The name of the blob."""
        return self._blob_client.blob_name

    @property
    def container_name(self) -> str:
        """The name of the blob's container."""
        return self._blob_client.container_name

    def reader(self) -> BlobIO:
        """Open file-like object for reading the blob's content.

        :returns: A file-like object for reading the blob's content.
        """
        return BlobIO(
            self._blob_client.url, "rb", _azstoragetorch_blob_client=self._blob_client
        )


class BlobDataset(torch.utils.data.Dataset[TransformOutputType_co]):
    """Map-style dataset for blobs in Azure Blob Storage.

    Data samples returned from dataset map directly one-to-one to blobs in Azure Blob Storage.
    Use :py:meth:`from_blob_urls` or :py:meth:`from_container_url` to create an instance of
    this dataset. Instanitating directly using ``__init__()`` is not supported. TODO: add sample? Maybe with torch loader?
    """
    def __init__(
        self, blobs: Iterable[Blob], transform: _TRANSFORM_TYPE = _blob_to_dict
    ):
        self._blobs = list(blobs)
        self._transform = transform

    @classmethod
    def from_blob_urls(
        cls,
        blob_urls: Union[str, Iterable[str]],
        *,
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
        transform: _TRANSFORM_TYPE = _blob_to_dict,
    ) -> "BlobDataset":
        """Instantiate dataset from provided blob URLs.

        **Sample usage**::

            container_url = "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"
            dataset = BlobDataset.from_blob_urls([
                f"{container_url}/<my-blob-name-1>",
                f"{container_url}/<my-blob-name-2>",
                f"{container_url}/<my-blob-name-3>",
            ])

        :param blob_urls: The full endpoint URLs to the blobs to be used for dataset.
            Can be a single URL or an iterable of URLs. URLs respect SAS tokens,
            snapshots, and version IDs in their query strings.
        :param credential: The credential to use for authentication. If not specified,
            :py:class:`azure.identity.DefaultAzureCredential` will be used. When set to
            ``False``, anonymous requests will be made. If a URL contains a SAS token,
            this parameter is ignored for that URL.
        :param transform: A callable that accepts a :py:class:`Blob` object representing a blob
            in the dataset and returns a transformed output to be used as output from the dataset.
            The default transform returns a dictionary representing the blob
            (see :py:class:`BlobToDictTransformOutput`).

        :returns: Dataset formed from the provided blob URLs.
        """
        blobs = _BlobUrlsBlobIterable(blob_urls, credential=credential)
        return cls(blobs, transform=transform)

    @classmethod
    def from_container_url(
        cls,
        container_url: str,
        *,
        prefix: Optional[str] = None,
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
        transform: _TRANSFORM_TYPE = _blob_to_dict,
    ) -> "BlobDataset":
        """Instantiate dataset by listing blobs from provided container URL.

        **Sample usage**::

            dataset = BlobDataset.from_container_url(
                "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>",
            )

        :param container_url: The full endpoint URL to the container to be used for dataset.
            The URL respects SAS tokens in its query string.

        :param prefix: The prefix to filter blobs by. Only blobs whose names begin with
            ``prefix`` will be included in the dataset. If not specified, all blobs
            in the container will be included in the dataset.
        :param credential: The credential to use for authentication. If not specified,
            :py:class:`azure.identity.DefaultAzureCredential` will be used. When set to
            ``False``, anonymous requests will be made. If a URL contains a SAS token,
            this parameter is ignored for that URL.
        :param transform: A callable that accepts a :py:class:`Blob` object representing a blob
            in the dataset and returns a transformed output to be used as output from the dataset.
            The default transform returns a dictionary representing the blob
            (see :py:class:`BlobToDictTransformOutput`).

        :returns: Dataset formed from the blobs in the provided container URL.
        """
        blobs = _ContainerUrlBlobIterable(
            container_url, prefix=prefix, credential=credential
        )
        return cls(blobs, transform=transform)

    def __getitem__(self, index: int) -> TransformOutputType_co:
        """Retrieve the blob at the specified index in the dataset.

        :param index: The index of the blob to retrieve.
        :returns: The blob, with ``transform`` applied, at the specified index.
        """
        blob = self._blobs[index]
        return self._transform(blob)

    def __len__(self) -> int:
        """Return the number of blobs in the dataset.

        :returns: The number of blobs in the dataset.
        """
        return len(self._blobs)


class IterableBlobDataset(torch.utils.data.IterableDataset[TransformOutputType_co]):
    """Iterable-style dataset for blobs in Azure Blob Storage.

    Data samples returned from dataset map directly one-to-one to blobs in Azure Blob Storage.
    Use :py:meth:`from_blob_urls` or :py:meth:`from_container_url` to create an instance of
    this dataset. Instanitating directly using ``__init__()`` is not supported. TODO: add sample? Maybe with torch loader?
    """
    def __init__(
        self, blobs: Iterable[Blob], transform: _TRANSFORM_TYPE = _blob_to_dict
    ):
        self._blobs = blobs
        self._transform = transform

    @classmethod
    def from_blob_urls(
        cls,
        blob_urls: Union[str, Iterable[str]],
        *,
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
        transform: _TRANSFORM_TYPE = _blob_to_dict,
    ) -> "IterableBlobDataset":
        """Instantiate dataset from provided blob URLs.

        **Sample usage**::

            container_url = "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>"
            dataset = IterableBlobDataset.from_blob_urls([
                f"{container_url}/<my-blob-name-1>",
                f"{container_url}/<my-blob-name-2>",
                f"{container_url}/<my-blob-name-3>",
            ])

        :param blob_urls: The full endpoint URLs to the blobs to be used for dataset.
            Can be a single URL or an iterable of URLs. URLs respect SAS tokens,
            snapshots, and version IDs in their query strings.
        :param credential: The credential to use for authentication. If not specified,
            :py:class:`azure.identity.DefaultAzureCredential` will be used. When set to
            ``False``, anonymous requests will be made. If a URL contains a SAS token,
            this parameter is ignored for that URL.
        :param transform: A callable that accepts a :py:class:`Blob` object representing a blob
            in the dataset and returns a transformed output to be used as output from the dataset.
            The default transform returns a dictionary representing the blob
            (see :py:class:`BlobToDictTransformOutput`).

        :returns: Dataset formed from the provided blob URLs.
        """
        blobs = _BlobUrlsBlobIterable(blob_urls, credential=credential)
        return cls(blobs, transform=transform)

    @classmethod
    def from_container_url(
        cls,
        container_url: str,
        *,
        prefix: Optional[str] = None,
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
        transform: _TRANSFORM_TYPE = _blob_to_dict,
    ) -> "IterableBlobDataset":
        """Instantiate dataset by listing blobs from provided container URL.

        **Sample usage**::

            dataset = IterableBlobDataset.from_container_url(
                "https://<my-storage-account-name>.blob.core.windows.net/<my-container-name>",
            )

        :param container_url: The full endpoint URL to the container to be used for dataset.
            The URL respects SAS tokens in its query string.

        :param prefix: The prefix to filter blobs by. Only blobs whose names begin with
            ``prefix`` will be included in the dataset. If not specified, all blobs
            in the container will be included in the dataset.
        :param credential: The credential to use for authentication. If not specified,
            :py:class:`azure.identity.DefaultAzureCredential` will be used. When set to
            ``False``, anonymous requests will be made. If a URL contains a SAS token,
            this parameter is ignored for that URL.
        :param transform: A callable that accepts a :py:class:`Blob` object representing a blob
            in the dataset and returns a transformed output to be used as output from the dataset.
            The default transform returns a dictionary representing the blob
            (see :py:class:`BlobToDictTransformOutput`).

        :returns: Dataset formed from the blobs in the provided container URL.
        """
        blobs = _ContainerUrlBlobIterable(
            container_url, prefix=prefix, credential=credential
        )
        return cls(blobs, transform=transform)

    def __iter__(self) -> Iterable[TransformOutputType_co]:
        """Iterate over the blobs in the dataset.

        :returns: An iterator over the blobs, with ``transform`` applied, in the dataset.
            The ``transform`` is applied lazily to each blob as it is yielded.
        """
        worker_info = torch.utils.data.get_worker_info()
        for i, blob in enumerate(self._blobs):
            if self._should_yield_from_worker_shard(worker_info, i):
                yield self._transform(blob)

    def _should_yield_from_worker_shard(self, worker_info, blob_index: int) -> bool:
        if worker_info is None:
            return True
        return blob_index % worker_info.num_workers == worker_info.id


class _BaseBlobIterable(Iterable[Blob]):
    def __init__(self, credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None):
        self._credential = credential
        self._blob_client_factory = _client.AzStorageTorchBlobClientFactory(
            credential=self._credential
        )

    def __iter__(self) -> Iterator[Blob]:
        raise NotImplementedError("__iter__")


class _ContainerUrlBlobIterable(_BaseBlobIterable):
    def __init__(
        self,
        container_url: str,
        prefix: Optional[str] = None,
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
    ):
        super().__init__(credential)
        self._container_url = container_url
        self._prefix = prefix

    def __iter__(self) -> Iterator[Blob]:
        blob_clients = self._blob_client_factory.yield_blob_clients_from_container_url(
            self._container_url, prefix=self._prefix
        )
        for blob_client in blob_clients:
            yield Blob(blob_client)


class _BlobUrlsBlobIterable(_BaseBlobIterable):
    def __init__(
        self,
        blob_urls: Union[str, Iterable[str]],
        credential: _client.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
    ):
        super().__init__(credential)
        if isinstance(blob_urls, str):
            blob_urls = [blob_urls]
        self._blob_urls = blob_urls

    def __iter__(self) -> Iterator[Blob]:
        for blob_url in self._blob_urls:
            yield Blob(self._blob_client_factory.get_blob_client_from_url(blob_url))
