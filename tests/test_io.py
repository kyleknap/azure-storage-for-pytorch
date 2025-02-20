# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import random
import string
from unittest import mock
import pytest

from azure.core.credentials import AzureSasCredential, AzureNamedKeyCredential
from azure.identity import DefaultAzureCredential

from azstoragetorch.io import BlobIO
from azstoragetorch._client import AzStorageTorchBlobClient


EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE = 4 * 1024 * 1024


@pytest.fixture
def sas_token():
    return "sp=r&st=2024-10-28T20:22:30Z&se=2024-10-29T04:22:30Z&spr=https&sv=2022-11-02&sr=c&sig=signature"


@pytest.fixture
def mock_azstoragetorch_blob_client(blob_content, blob_length):
    mock_blob_client = mock.Mock(AzStorageTorchBlobClient)
    mock_blob_client.from_blob_url.return_value = mock_blob_client
    mock_blob_client.get_blob_size.return_value = blob_length
    mock_blob_client.download.return_value = blob_content
    return mock_blob_client


@pytest.fixture
def blob_io(blob_url, mock_azstoragetorch_blob_client):
    return BlobIO(
        blob_url,
        mode="rb",
        azstoragetorch_blob_client_cls=mock_azstoragetorch_blob_client,
    )


def random_ascii_letter_bytes(size):
    return "".join(random.choices(string.ascii_letters, k=size)).encode("utf-8")


class TestBlobIO:
    def test_credential_defaults_to_azure_default_credential(
        self, blob_url, mock_azstoragetorch_blob_client
    ):
        BlobIO(
            blob_url,
            mode="rb",
            azstoragetorch_blob_client_cls=mock_azstoragetorch_blob_client,
        )
        mock_azstoragetorch_blob_client.from_blob_url.assert_called_once_with(
            blob_url, mock.ANY
        )
        creds = mock_azstoragetorch_blob_client.from_blob_url.call_args[0][1]
        assert isinstance(creds, DefaultAzureCredential)

    @pytest.mark.parametrize(
        "credential",
        [
            DefaultAzureCredential(),
            AzureSasCredential("sas"),
        ],
    )
    def test_respects_user_provided_credential(
        self, blob_url, mock_azstoragetorch_blob_client, credential
    ):
        BlobIO(
            blob_url,
            mode="rb",
            credential=credential,
            azstoragetorch_blob_client_cls=mock_azstoragetorch_blob_client,
        )
        mock_azstoragetorch_blob_client.from_blob_url.assert_called_once_with(
            blob_url, credential
        )

    def test_anonymous_credential(self, blob_url, mock_azstoragetorch_blob_client):
        BlobIO(
            blob_url,
            mode="rb",
            credential=False,
            azstoragetorch_blob_client_cls=mock_azstoragetorch_blob_client,
        )
        mock_azstoragetorch_blob_client.from_blob_url.assert_called_once_with(
            blob_url, None
        )

    def test_detects_sas_token_in_blob_url(
        self, blob_url, mock_azstoragetorch_blob_client, sas_token
    ):
        BlobIO(
            blob_url + "?" + sas_token,
            mode="rb",
            azstoragetorch_blob_client_cls=mock_azstoragetorch_blob_client,
        )
        # The SDK prefers the explict credential over the one in the URL. So if a SAS token is
        # in the URL, we do not want it automatically injecting the DefaultAzureCredential.
        mock_azstoragetorch_blob_client.from_blob_url.assert_called_once_with(
            blob_url + "?" + sas_token, None
        )

    def test_credential_defaults_to_azure_default_credential_for_snapshot_url(
        self, blob_url, mock_azstoragetorch_blob_client
    ):
        snapshot_url = f"{blob_url}?snapshot=2024-10-28T20:34:36.1724588Z"
        BlobIO(
            snapshot_url,
            mode="rb",
            azstoragetorch_blob_client_cls=mock_azstoragetorch_blob_client,
        )
        mock_azstoragetorch_blob_client.from_blob_url.assert_called_once_with(
            snapshot_url, mock.ANY
        )
        creds = mock_azstoragetorch_blob_client.from_blob_url.call_args[0][1]
        assert isinstance(creds, DefaultAzureCredential)

    @pytest.mark.parametrize(
        "credential",
        [
            "key",
            {"account_name": "name", "account_key": "key"},
            AzureNamedKeyCredential("name", "key"),
        ],
    )
    def test_raises_for_unsupported_credential(self, blob_url, credential):
        with pytest.raises(TypeError, match="Unsupported credential"):
            BlobIO(blob_url, mode="rb", credential=credential)

    @pytest.mark.parametrize(
        "unsupported_mode",
        [
            "r",
            "r+",
            "r+b",
            "w",
            "wb",
            "w+",
            "w+b",
            "a",
            "ab",
            "a+",
            "a+b",
            "x",
            "xb",
            "x+",
            "x+b",
            "unknownmode",
        ],
    )
    def test_raises_for_unsupported_mode(self, blob_url, unsupported_mode):
        with pytest.raises(ValueError, match="Unsupported mode"):
            BlobIO(blob_url, mode=unsupported_mode)

    def test_close(self, blob_io, mock_azstoragetorch_blob_client):
        assert not blob_io.closed
        blob_io.close()
        assert blob_io.closed
        mock_azstoragetorch_blob_client.close.assert_called_once()

    def test_can_call_close_multiple_times(
        self, blob_io, mock_azstoragetorch_blob_client
    ):
        blob_io.close()
        blob_io.close()
        assert blob_io.closed
        mock_azstoragetorch_blob_client.close.assert_called_once()

    def test_context_manager_closes_blob_io(
        self, blob_io, mock_azstoragetorch_blob_client
    ):
        assert not blob_io.closed
        with blob_io:
            pass
        assert blob_io.closed
        mock_azstoragetorch_blob_client.close.assert_called_once()

    def test_del_closes_blob_io(self, blob_io, mock_azstoragetorch_blob_client):
        assert not blob_io.closed
        blob_io.__del__()
        assert blob_io.closed
        mock_azstoragetorch_blob_client.close.assert_called_once()

    @pytest.mark.parametrize(
        "method,args",
        [
            ("isatty", []),
            ("flush", []),
            ("read", []),
            ("readable", []),
            ("readline", []),
            ("seek", [1]),
            ("seekable", []),
            ("tell", []),
        ],
    )
    def test_raises_after_close(self, blob_io, method, args):
        blob_io.close()
        with pytest.raises(ValueError, match="I/O operation on closed file"):
            getattr(blob_io, method)(*args)

    def test_fileno_raises(self, blob_io):
        with pytest.raises(OSError, match="BlobIO object has no fileno"):
            blob_io.fileno()

    def test_isatty(self, blob_io):
        assert not blob_io.isatty()

    def test_flush_on_readable_is_noop(self, blob_url):
        blob_io = BlobIO(blob_url, mode="rb")
        try:
            blob_io.flush()
        except Exception as e:
            pytest.fail(
                f"Unexpected exception: {e}. flush() should be a no-op for readable BlobIO objects."
            )

    def test_readable(self, blob_url):
        blob_io = BlobIO(blob_url, mode="rb")
        assert blob_io.readable()

    def test_read(self, blob_io, blob_content, mock_azstoragetorch_blob_client):
        assert blob_io.read() == blob_content
        assert blob_io.tell() == len(blob_content)
        mock_azstoragetorch_blob_client.download.assert_called_once_with(
            offset=0, length=None
        )

    def test_read_with_size(
        self, blob_io, blob_content, mock_azstoragetorch_blob_client
    ):
        mock_azstoragetorch_blob_client.download.return_value = blob_content[:1]
        assert blob_io.read(1) == blob_content[:1]
        assert blob_io.tell() == 1
        mock_azstoragetorch_blob_client.download.assert_called_once_with(
            offset=0, length=1
        )

    def test_read_multiple_times(
        self, blob_io, blob_content, mock_azstoragetorch_blob_client
    ):
        mock_azstoragetorch_blob_client.download.side_effect = [
            blob_content[:1],
            blob_content[1:2],
            blob_content[2:],
        ]
        assert blob_io.read(1) == blob_content[:1]
        assert blob_io.read(1) == blob_content[1:2]
        assert blob_io.read() == blob_content[2:]
        assert mock_azstoragetorch_blob_client.download.call_args_list == [
            mock.call(offset=0, length=1),
            mock.call(offset=1, length=1),
            mock.call(offset=2, length=None),
        ]
        assert blob_io.tell() == len(blob_content)

    def test_read_after_seek(
        self, blob_io, blob_content, mock_azstoragetorch_blob_client
    ):
        offset = 2
        mock_azstoragetorch_blob_client.download.return_value = blob_content[offset:]
        assert blob_io.seek(offset) == offset
        assert blob_io.read() == blob_content[offset:]
        assert blob_io.tell() == len(blob_content)
        mock_azstoragetorch_blob_client.download.assert_called_once_with(
            offset=offset, length=None
        )

    def test_read_beyond_end(
        self, blob_io, blob_content, mock_azstoragetorch_blob_client
    ):
        assert blob_io.read() == blob_content
        assert blob_io.read() == b""
        assert blob_io.tell() == len(blob_content)
        mock_azstoragetorch_blob_client.download.assert_called_once_with(
            offset=0, length=None
        )

    def test_read_size_zero(self, blob_io, mock_azstoragetorch_blob_client):
        assert blob_io.read(0) == b""
        assert blob_io.tell() == 0
        mock_azstoragetorch_blob_client().download.assert_not_called()

    @pytest.mark.parametrize("size", [-1, None])
    def test_read_size_synonyms_for_read_all(
        self, blob_io, mock_azstoragetorch_blob_client, size
    ):
        assert blob_io.read(size) == b"blob content"
        mock_azstoragetorch_blob_client.download.assert_called_once_with(
            offset=0, length=None
        )

    @pytest.mark.parametrize("size", [0.5, "1"])
    @pytest.mark.parametrize("read_method", ["read", "readline"])
    def test_read_methods_raise_for_unsupported_size_types(
        self, blob_io, size, read_method
    ):
        with pytest.raises(TypeError, match="must be an integer"):
            getattr(blob_io, read_method)(size)

    def test_read_raises_for_less_than_negative_one_size(self, blob_io):
        with pytest.raises(ValueError, match="must be greater than or equal to -1"):
            blob_io.read(-2)

    @pytest.mark.parametrize(
        "lines",
        [
            [b"line1\n", b"line2\n"],
            # No newlines
            [b"line1-no-new-line"],
            # Content does not end with newline
            [b"line1\n", b"line2-no-new-line"],
            # Multiple newlines in succession
            [b"line1\n", b"\n", b"\n", b"line2\n"],
            # Lines with additional whitespace characters
            [b"line1 \t\r\f\v\n", b"line2\n"],
        ],
    )
    def test_readline(self, blob_io, mock_azstoragetorch_blob_client, lines):
        content = b"".join(lines)
        mock_azstoragetorch_blob_client.download.return_value = content
        mock_azstoragetorch_blob_client.get_blob_size.return_value = len(content)
        current_expected_position = 0
        for line in lines:
            assert blob_io.readline() == line
            current_expected_position += len(line)
            assert blob_io.tell() == current_expected_position
        assert blob_io.tell() == len(content)
        mock_azstoragetorch_blob_client.download.assert_called_once_with(
            offset=0, length=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE
        )

    @pytest.mark.parametrize(
        "size,content,expected_readline_return_val",
        [
            # Size less than first line
            (2, b"line1\nline2\n", b"li"),
            # Size larger than first line
            (8, b"line1\nline2\n", b"line1\n"),
            # Size larger than content
            (100, b"line1\nline2\n", b"line1\n"),
            # Size larger than expected prefetch size
            (
                EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE + 1,
                b"line1\nline2\n",
                b"line1\n",
            ),
            # Size of None is synonymous with size not being set
            (
                None,
                b"line1\nline2\n",
                b"line1\n",
            ),
            # Size of -1 is synonymous with size not being set
            (
                -1,
                b"line1\nline2\n",
                b"line1\n",
            ),
            # Size less than -1 is synonymous with size not being set. Note that is different behavior than read()
            # which throws validation errors for sizes < -1. This behavior was chosen to stay consistent with
            # file-like objects from open().
            (
                -2,
                b"line1\nline2\n",
                b"line1\n",
            ),
        ],
    )
    def test_readline_with_size(
        self,
        blob_io,
        mock_azstoragetorch_blob_client,
        size,
        content,
        expected_readline_return_val,
    ):
        mock_azstoragetorch_blob_client.download.return_value = content[
            :EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE
        ]
        mock_azstoragetorch_blob_client.get_blob_size.return_value = len(content)
        assert blob_io.readline(size) == expected_readline_return_val
        assert blob_io.tell() == len(expected_readline_return_val)
        mock_azstoragetorch_blob_client.download.assert_called_once_with(
            offset=0, length=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE
        )

    def test_readline_multiple_prefetches(
        self, blob_io, mock_azstoragetorch_blob_client
    ):
        first_download_prefetch = random_ascii_letter_bytes(
            EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE
        )
        second_download_prefetch = b"\n" + random_ascii_letter_bytes(
            EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE - 1
        )
        final_download_prefetch = random_ascii_letter_bytes(100)
        download_batches = [
            first_download_prefetch,
            second_download_prefetch,
            final_download_prefetch,
        ]
        blob_size = sum([len(batch) for batch in download_batches])
        mock_azstoragetorch_blob_client.download.side_effect = download_batches
        mock_azstoragetorch_blob_client.get_blob_size.return_value = blob_size

        assert blob_io.readline() == first_download_prefetch + b"\n"
        assert blob_io.tell() == len(first_download_prefetch) + 1
        # First readline() should have resulted in two prefetches because the first newline is in the second prefetch
        assert mock_azstoragetorch_blob_client.download.call_args_list == [
            mock.call(offset=0, length=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE),
            mock.call(
                offset=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE,
                length=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE,
            ),
        ]
        assert (
            blob_io.readline() == second_download_prefetch[1:] + final_download_prefetch
        )
        assert blob_io.tell() == blob_size
        # Second readline should result in triggering the final prefetch because there are no newlines for the rest
        # of the blob content.
        assert mock_azstoragetorch_blob_client.download.call_args_list == [
            mock.call(offset=0, length=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE),
            mock.call(
                offset=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE,
                length=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE,
            ),
            mock.call(
                offset=2 * EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE,
                length=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE,
            ),
        ]

    def test_readline_size_across_multiple_prefetches(
        self, blob_io, mock_azstoragetorch_blob_client
    ):
        newline_position = EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE + 6
        content = (
            random_ascii_letter_bytes(newline_position)
            + b"\n"
            + random_ascii_letter_bytes(100)
        )
        mock_azstoragetorch_blob_client.download.side_effect = [
            content[:EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE],
            content[EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE:],
        ]
        mock_azstoragetorch_blob_client.get_blob_size.return_value = len(content)
        readline_size = EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE + 2
        # First readline should have triggered two prefetches but stopped short of returning the newline in second
        # prefetch
        assert blob_io.readline(readline_size) == content[:readline_size]
        assert blob_io.tell() == readline_size
        assert mock_azstoragetorch_blob_client.download.call_args_list == [
            mock.call(offset=0, length=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE),
            mock.call(
                offset=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE,
                length=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE,
            ),
        ]
        # Second readline should now reach newline from second prefetch
        assert (
            blob_io.readline(readline_size)
            == content[readline_size : newline_position + 1]
        )
        assert blob_io.tell() == newline_position + 1
        # Third readline should return rest of content and no additional downloads should have been
        # made since the first readline
        assert blob_io.readline(readline_size) == content[newline_position + 1 :]
        assert blob_io.tell() == len(content)
        assert mock_azstoragetorch_blob_client.download.call_count == 2

    def test_readline_mixed_with_read(self, blob_io, mock_azstoragetorch_blob_client):
        content = b"line1\nline2\nline3\n"
        mock_azstoragetorch_blob_client.download.side_effect = [
            content,
            b"line2",
            b"\nline3\n",
        ]
        mock_azstoragetorch_blob_client.get_blob_size.return_value = len(content)

        assert blob_io.readline() == b"line1\n"
        assert blob_io.tell() == 6
        assert blob_io.read(5) == b"line2"
        assert blob_io.tell() == 11
        assert blob_io.readline() == b"\n"
        assert blob_io.tell() == 12
        assert blob_io.readline() == b"line3\n"
        assert blob_io.tell() == len(content)
        assert mock_azstoragetorch_blob_client.download.call_args_list == [
            # First readline() will result in full prefetch
            mock.call(offset=0, length=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE),
            # Out of band read(), downloads only requested range and invalidates prefetch cache
            mock.call(offset=6, length=5),
            # Second readline() will result in full prefetch
            mock.call(offset=11, length=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE),
        ]

    def test_readline_mixed_with_seek(self, blob_io, mock_azstoragetorch_blob_client):
        content = b"line1\nline2\nline3\n"
        mock_azstoragetorch_blob_client.download.side_effect = [
            content,
            b"\nline3\n",
        ]
        mock_azstoragetorch_blob_client.get_blob_size.return_value = len(content)

        assert blob_io.readline() == b"line1\n"
        assert blob_io.tell() == 6
        assert blob_io.seek(11) == 11
        assert blob_io.tell() == 11
        assert blob_io.readline() == b"\n"
        assert blob_io.tell() == 12
        assert blob_io.readline() == b"line3\n"
        assert blob_io.tell() == len(content)
        assert mock_azstoragetorch_blob_client.download.call_args_list == [
            # First readline() will result in full prefetch
            mock.call(offset=0, length=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE),
            # Second readline() will result in full prefetch from prior out-of-band seek()
            mock.call(offset=11, length=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE),
        ]

    def test_readline_beyond_end(self, blob_io, mock_azstoragetorch_blob_client):
        content = b"line1\nline2\n"
        mock_azstoragetorch_blob_client.download.return_value = content
        mock_azstoragetorch_blob_client.get_blob_size.return_value = len(content)
        blob_io.seek(0, os.SEEK_END)
        assert blob_io.readline() == b""
        assert blob_io.tell() == len(content)
        mock_azstoragetorch_blob_client.download.assert_not_called()

    def test_readline_size_zero(self, blob_io, mock_azstoragetorch_blob_client):
        assert blob_io.readline(0) == b""
        assert blob_io.tell() == 0
        mock_azstoragetorch_blob_client().download.assert_not_called()

    def test_readlines(self, blob_io, mock_azstoragetorch_blob_client):
        lines = [b"line1\n", b"line2\n", b"line3\n"]
        content = b"".join(lines)
        mock_azstoragetorch_blob_client.download.return_value = content
        mock_azstoragetorch_blob_client.get_blob_size.return_value = len(content)
        assert blob_io.readlines() == lines
        assert blob_io.tell() == len(content)
        mock_azstoragetorch_blob_client.download.assert_called_once_with(
            offset=0, length=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE
        )

    def test_next(self, blob_io, mock_azstoragetorch_blob_client):
        lines = [b"line1\n", b"line2\n", b"line3\n"]
        content = b"".join(lines)
        mock_azstoragetorch_blob_client.download.return_value = content
        mock_azstoragetorch_blob_client.get_blob_size.return_value = len(content)
        iterated_lines = [line for line in blob_io]
        assert iterated_lines == lines
        assert blob_io.tell() == len(content)
        mock_azstoragetorch_blob_client.download.assert_called_once_with(
            offset=0, length=EXPECTED_DEFAULT_READLINE_PREFETCH_SIZE
        )

    def test_seekable(self, blob_url):
        blob_io = BlobIO(blob_url, mode="rb")
        assert blob_io.seekable()

    def test_seek(self, blob_io):
        assert blob_io.seek(1) == 1
        assert blob_io.tell() == 1

    def test_seek_multiple_times(self, blob_io):
        assert blob_io.seek(1) == 1
        assert blob_io.tell() == 1
        assert blob_io.seek(2) == 2
        assert blob_io.tell() == 2
        assert blob_io.seek(0) == 0
        assert blob_io.tell() == 0

    def test_seek_beyond_end(
        self, blob_io, blob_length, mock_azstoragetorch_blob_client
    ):
        # Note: Sort of quirky behavior that you can seek past the end of a
        # file and return a position that is larger than the size of the file.
        # However, this was chosen to be consistent in behavior with file-like
        # objects from open()
        assert blob_io.seek(blob_length + 1) == blob_length + 1
        assert blob_io.tell() == blob_length + 1
        assert blob_io.read(1) == b""
        mock_azstoragetorch_blob_client.download.assert_not_called()

    def test_seek_cur(self, blob_io):
        assert blob_io.seek(1, os.SEEK_CUR) == 1
        assert blob_io.tell() == 1
        assert blob_io.seek(1, os.SEEK_CUR) == 2
        assert blob_io.tell() == 2

    def test_seek_end(self, blob_io, blob_length):
        assert blob_io.seek(0, os.SEEK_END) == blob_length
        assert blob_io.tell() == blob_length

    def test_seek_negative_offset(self, blob_io, blob_length):
        assert blob_io.seek(-1, os.SEEK_END) == blob_length - 1
        assert blob_io.tell() == blob_length - 1

    def test_seek_raises_when_results_in_negative_position(self, blob_io):
        with pytest.raises(ValueError, match="Cannot seek to negative position"):
            blob_io.seek(-1)

    def test_seek_raises_for_unsupported_whence(self, blob_io):
        with pytest.raises(ValueError, match="Unsupported whence"):
            blob_io.seek(0, 4)

    @pytest.mark.parametrize(
        "offset,whence", [(0.5, 0), ("1", 0), (None, 0), (0, 0.5), (0, "1"), (0, None)]
    )
    def test_seek_raises_for_unsupported_arg_types(self, blob_io, offset, whence):
        with pytest.raises(TypeError, match="must be an integer"):
            blob_io.seek(offset, whence)

    def test_tell_starts_at_zero(self, blob_io):
        assert blob_io.tell() == 0

    def test_writeable(self, blob_url):
        blob_io = BlobIO(blob_url, mode="rb")
        assert not blob_io.writable()
