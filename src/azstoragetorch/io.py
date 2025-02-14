# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------

import io
import os
from typing import get_args, Optional, Union, Literal, Type
import urllib.parse

from azure.identity import DefaultAzureCredential
from azure.core.credentials import (
    AzureSasCredential,
    TokenCredential,
)

from azstoragetorch._client import SDK_CREDENTIAL_TYPE as _SDK_CREDENTIAL_TYPE
from azstoragetorch._client import AzStorageTorchBlobClient as _AzStorageTorchBlobClient


_SUPPORTED_MODES = Literal["rb", "wb"]
_SUPPORTED_WRITE_TYPES = Union[bytes, bytearray]
_AZSTORAGETORCH_CREDENTIAL_TYPE = Union[_SDK_CREDENTIAL_TYPE, Literal[False]]


class BlobIO(io.IOBase):
    _READLINE_PREFETCH_SIZE = 4 * 1024 * 1024
    _READLINE_TERMINATOR = b"\n"
    _WRITE_BUFFER_SIZE = 4 * 1024 * 1024

    def __init__(
        self,
        blob_url: str,
        mode: _SUPPORTED_MODES,
        *,
        credential: _AZSTORAGETORCH_CREDENTIAL_TYPE = None,
        **_internal_only_kwargs,
    ):
        self._blob_url = blob_url
        self._validate_mode(mode)
        self._mode = mode
        self._client = self._get_azstoragetorch_blob_client(
            blob_url,
            credential,
            _internal_only_kwargs.get(
                "azstoragetorch_blob_client_cls", _AzStorageTorchBlobClient
            ),
        )

        self._position = 0
        self._closed = False
        # TODO: Consider using a bytearray and/or memoryview for readline buffer. There may be performance
        #  gains in regards to reducing the number of copies performed when consuming from buffer.
        self._readline_buffer = b""
        self._write_buffer = b""
        self._stage_block_ids = []

    def close(self) -> None:
        if not self.closed and self.writable():
            self._commit_blob()
        self._close_client()
        self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    def fileno(self) -> int:
        raise OSError("BlobIO object has no fileno")

    def flush(self) -> None:
        self._validate_not_closed()
        self._flush()

    def read(self, size: Optional[int] = -1, /) -> bytes:
        if size is not None:
            self._validate_is_integer("size", size)
            self._validate_min("size", size, -1)
        self._validate_readable()
        self._validate_not_closed()
        self._invalidate_readline_buffer()
        return self._read(size)

    def readable(self) -> bool:
        if self._is_read_mode():
            self._validate_not_closed()
            return True
        return False

    def readline(self, size: Optional[int] = -1, /) -> bytes:
        if size is not None:
            self._validate_is_integer("size", size)
        self._validate_readable()
        self._validate_not_closed()
        return self._readline(size)

    def seek(self, offset: int, whence: int = os.SEEK_SET, /) -> int:
        self._validate_is_integer("offset", offset)
        self._validate_is_integer("whence", whence)
        self._validate_seekable()
        self._validate_not_closed()
        self._invalidate_readline_buffer()
        return self._seek(offset, whence)

    def seekable(self) -> bool:
        return self.readable()

    def tell(self) -> int:
        self._validate_not_closed()
        return self._position

    def write(self, b: _SUPPORTED_WRITE_TYPES, /) -> int:
        self._validate_supported_write_type(b)
        self._validate_writable()
        self._validate_not_closed()
        return self._write(b)

    def writable(self) -> bool:
        if self._is_write_mode():
            self._validate_not_closed()
            return True
        return False

    def _validate_mode(self, mode: str) -> None:
        if mode not in get_args(_SUPPORTED_MODES):
            raise ValueError(f"Unsupported mode: {mode}")

    def _validate_is_integer(self, param_name: str, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"{param_name} must be an integer, not: {type(value)}")

    def _validate_min(self, param_name: str, value: int, min_value: int) -> None:
        if value < min_value:
            raise ValueError(
                f"{param_name} must be greater than or equal to {min_value}"
            )

    def _validate_supported_write_type(self, b: _SUPPORTED_WRITE_TYPES) -> None:
        if not isinstance(b, get_args(_SUPPORTED_WRITE_TYPES)):
            raise TypeError(
                f"Unsupported type for write: {type(b)}. Supported types: {get_args(_SUPPORTED_WRITE_TYPES)}"
            )

    def _validate_readable(self) -> None:
        if not self._is_read_mode():
            raise io.UnsupportedOperation("read")

    def _validate_seekable(self) -> None:
        if not self._is_read_mode():
            raise io.UnsupportedOperation("seek")

    def _validate_writable(self) -> None:
        if not self._is_write_mode():
            raise io.UnsupportedOperation("write")

    def _is_read_mode(self) -> bool:
        return self._mode == "rb"

    def _is_write_mode(self) -> bool:
        return self._mode == "wb"

    def _validate_not_closed(self) -> None:
        if self.closed:
            raise ValueError("I/O operation on closed file")

    def _invalidate_readline_buffer(self) -> None:
        # NOTE: We invalidate the readline buffer for any out-of-band read() or seek() in order to simplify
        # caching logic for readline(). In the future, we can consider reusing the buffer for read() calls.
        self._readline_buffer = b""

    def _get_azstoragetorch_blob_client(
        self,
        blob_url: str,
        credential: _AZSTORAGETORCH_CREDENTIAL_TYPE,
        azstoragetorch_blob_client_cls: Type[_AzStorageTorchBlobClient],
    ) -> _AzStorageTorchBlobClient:
        return azstoragetorch_blob_client_cls.from_blob_url(
            blob_url,
            self._get_sdk_credential(blob_url, credential),
        )

    def _get_sdk_credential(
        self, blob_url: str, credential: _AZSTORAGETORCH_CREDENTIAL_TYPE
    ) -> _SDK_CREDENTIAL_TYPE:
        if credential is False or self._blob_url_has_sas_token(blob_url):
            return None
        if credential is None:
            return DefaultAzureCredential()
        if isinstance(credential, (AzureSasCredential, TokenCredential)):
            return credential
        raise TypeError(f"Unsupported credential: {type(credential)}")

    def _blob_url_has_sas_token(self, blob_url: str) -> bool:
        parsed_url = urllib.parse.urlparse(blob_url)
        if parsed_url.query is None:
            return False
        parsed_qs = urllib.parse.parse_qs(parsed_url.query)
        # The signature is always required in a valid SAS token. So look for the "sig"
        # key to determine if the URL has a SAS token.
        return "sig" in parsed_qs

    def _readline(self, size: Optional[int]) -> bytes:
        consumed = b""
        if size == 0 or self._is_at_end_of_blob():
            return consumed

        limit = self._get_limit(size)
        if self._readline_buffer:
            consumed = self._consume_from_readline_buffer(consumed, limit)
        while self._should_download_more_for_readline(consumed, limit):
            self._readline_buffer = self._client.download(
                offset=self._position, length=self._READLINE_PREFETCH_SIZE
            )
            consumed = self._consume_from_readline_buffer(consumed, limit)
        return consumed

    def _get_limit(self, size: Optional[int]) -> int:
        if size is None or size < 0:
            # If size is not provided, set the initial limit to the blob size as BlobIO
            # will never read more than the size of the blob in a single readline() call.
            return self._client.get_blob_size()
        return size

    def _consume_from_readline_buffer(self, consumed: bytes, limit: int) -> bytes:
        limit -= len(consumed)
        find_pos = self._readline_buffer.find(self._READLINE_TERMINATOR, 0, limit)
        end = find_pos + 1
        if find_pos == -1:
            buffer_length = len(self._readline_buffer)
            end = min(buffer_length, limit)
        consumed += self._readline_buffer[:end]
        self._readline_buffer = self._readline_buffer[end:]
        self._position += end
        return consumed

    def _should_download_more_for_readline(self, consumed: bytes, limit: int) -> bool:
        if consumed.endswith(self._READLINE_TERMINATOR):
            return False
        if self._is_at_end_of_blob():
            return False
        if len(consumed) == limit:
            return False
        return True

    def _read(self, size: Optional[int]) -> bytes:
        if size == 0 or self._is_at_end_of_blob():
            return b""
        download_length = size
        if size is not None and size < 0:
            download_length = None
        content = self._client.download(offset=self._position, length=download_length)
        self._position += len(content)
        return content

    def _seek(self, offset: int, whence: int) -> int:
        new_position = self._compute_new_position(offset, whence)
        if new_position < 0:
            raise ValueError("Cannot seek to negative position")
        self._position = new_position
        return self._position

    def _compute_new_position(self, offset: int, whence: int) -> int:
        if whence == os.SEEK_SET:
            return offset
        if whence == os.SEEK_CUR:
            return self._position + offset
        if whence == os.SEEK_END:
            return self._client.get_blob_size() + offset
        raise ValueError(f"Unsupported whence: {whence}")

    def _flush(self) -> None:
        if self._write_buffer:
            self._stage_block_ids.extend(self._client.stage_blocks(self._write_buffer))
            self._write_buffer = b""

    def _write(self, b: Union[bytes, bytearray]) -> int:
        self._write_buffer += b
        if len(self._write_buffer) >= self._WRITE_BUFFER_SIZE:
            self._flush()
        self._position += len(b)
        return len(b)

    def _commit_blob(self) -> None:
        self._flush()
        self._client.commit_block_list(self._stage_block_ids)

    def _close_client(self) -> None:
        if not self._closed:
            self._client.close()

    def _is_at_end_of_blob(self) -> bool:
        return self._position >= self._client.get_blob_size()
