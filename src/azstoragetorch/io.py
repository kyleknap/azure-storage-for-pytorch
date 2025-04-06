# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------

import concurrent.futures
import io
import os
from typing import get_args, Optional, Literal, List

import azure.storage.blob

from azstoragetorch import _client
from azstoragetorch import _utils
from azstoragetorch.exceptions import FatalBlobIOWriteError


_SUPPORTED_MODES = Literal["rb", "wb"]


class BlobIO(io.IOBase):
    _READLINE_PREFETCH_SIZE = 4 * 1024 * 1024
    _READLINE_TERMINATOR = b"\n"
    _WRITE_BUFFER_SIZE = 32 * 1024 * 1024

    def __init__(
        self,
        blob_url: str,
        mode: _SUPPORTED_MODES,
        *,
        credential: _utils.AZSTORAGETORCH_CREDENTIAL_TYPE = None,
        **_internal_only_kwargs,
    ):
        self._blob_url = blob_url
        self._validate_mode(mode)
        self._mode = mode
        self._client = self._get_azstoragetorch_blob_client(
            blob_url,
            credential,
            _internal_only_kwargs.get("azstoragetorch_blob_client_factory"),
            _internal_only_kwargs.get("blob_properties"),
        )

        self._position = 0
        self._closed = False
        # TODO: Consider using a bytearray and/or memoryview for readline buffer. There may be performance
        #  gains in regards to reducing the number of copies performed when consuming from buffer.
        self._readline_buffer = b""
        self._write_buffer = bytearray()
        self._all_stage_block_futures: List[_client.STAGE_BLOCK_FUTURE_TYPE] = []
        self._in_progress_stage_block_futures: List[_client.STAGE_BLOCK_FUTURE_TYPE] = []
        self._stage_block_exception: Optional[BaseException] = None

    def close(self) -> None:
        if self.closed:
            return
        try:
            # Any errors that occur while flushing or committing the block list are considered non-recoverable
            # when using the BlobIO interface. So if an error occurs, we still close the BlobIO to further indicate
            # that a new BlobIO instance will be needed and also avoid possibly calling the flush/commit logic again
            # during garbage collection.
            if self.writable():
                self._commit_blob()
        finally:
            # TODO: We need to figure out if we actually want to close the executor here cause if we share it as part of
            # a dataset we may not want to close it to avoid spinning up threads and closing them all the time.
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

    def write(self, b: _client.SUPPORTED_WRITE_BYTES_LIKE_TYPE, /) -> int:
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

    def _validate_supported_write_type(self, b: _client.SUPPORTED_WRITE_BYTES_LIKE_TYPE) -> None:
        if not isinstance(b, get_args(_client.SUPPORTED_WRITE_BYTES_LIKE_TYPE)):
            raise TypeError(
                f"Unsupported type for write: {type(b)}. Supported types: {get_args(_client.SUPPORTED_WRITE_BYTES_LIKE_TYPE)}"
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
        credential: _utils.AZSTORAGETORCH_CREDENTIAL_TYPE,
        azstoragetorch_blob_client_factory: Optional[_client.AzStorageTorchBlobClientFactory] = None,
        blob_properties: Optional[azure.storage.blob.BlobProperties] = None,
    ) -> _client.AzStorageTorchBlobClient:
        if azstoragetorch_blob_client_factory is None:
            print('better not be here')
            sdk_credential = _utils.to_sdk_credential(blob_url, credential)
            azstoragetorch_blob_client_factory = _client.AzStorageTorchBlobClientFactory(sdk_credential)
        return azstoragetorch_blob_client_factory.get_blob_client(
            blob_url, blob_properties=blob_properties
        )

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
        self._check_for_stage_block_exceptions(wait=False)
        self._flush_write_buffer()
        self._check_for_stage_block_exceptions(wait=True)

    def _flush_write_buffer(self) -> None:
        if self._write_buffer:
            futures = self._client.stage_blocks(memoryview(self._write_buffer))
            self._all_stage_block_futures.extend(futures)
            self._in_progress_stage_block_futures.extend(futures)
            self._write_buffer = bytearray()

    def _write(self, b: _client.SUPPORTED_WRITE_BYTES_LIKE_TYPE) -> int:
        self._check_for_stage_block_exceptions(wait=False)
        write_length = len(b)
        self._write_buffer.extend(b)
        if len(self._write_buffer) >= self._WRITE_BUFFER_SIZE:
            self._flush_write_buffer()
        self._position += write_length
        return write_length

    def _commit_blob(self) -> None:
        self._flush()
        block_ids = [f.result() for f in self._all_stage_block_futures]
        self._raise_if_duplicate_block_ids(block_ids)
        self._client.commit_block_list(block_ids)

    def _raise_if_duplicate_block_ids(self, block_ids: List[str]) -> None:
        # An additional safety measure to ensure we never reuse block IDs within a BlobIO instance. This
        # should not be an issue with UUID4 for block IDs, but that may not always be the case if
        # block ID generation changes in the future.
        if len(block_ids) != len(set(block_ids)):
            raise RuntimeError(
                "Unexpected duplicate block IDs detected. Not committing blob."
            )

    def _check_for_stage_block_exceptions(self, wait: bool = True) -> None:
        # Before doing any additional processing, raise if an exception has already
        # been processed especially if it is going to require us to wait for all
        # in-progress futures to complete.
        self._raise_if_fatal_write_error()
        self._process_stage_block_futures_for_errors(wait)

    def _process_stage_block_futures_for_errors(self, wait: bool) -> None:
        if wait:
            concurrent.futures.wait(
                self._in_progress_stage_block_futures,
                return_when=concurrent.futures.FIRST_EXCEPTION,
            )
        futures_still_in_progress = []
        for future in self._in_progress_stage_block_futures:
            if future.done():
                if (
                    self._stage_block_exception is None
                    and future.exception() is not None
                ):
                    self._stage_block_exception = future.exception()
            else:
                futures_still_in_progress.append(future)
        self._in_progress_stage_block_futures = futures_still_in_progress
        self._raise_if_fatal_write_error()

    def _raise_if_fatal_write_error(self) -> None:
        if self._stage_block_exception is not None:
            raise FatalBlobIOWriteError(self._stage_block_exception)

    def _close_client(self) -> None:
        self._client.close()

    def _is_at_end_of_blob(self) -> bool:
        return self._position >= self._client.get_blob_size()
