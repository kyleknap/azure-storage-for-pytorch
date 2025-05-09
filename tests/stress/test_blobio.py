# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import dataclasses
import datetime
import hashlib
import io
import json
import os
import random
import typing

import pytest

from azstoragetorch.io import BlobIO
from utils import sample_data


STRESS_DIRNAME = os.path.abspath(os.path.dirname(__file__))


KB = 1024
MB = 1024 ** 2
GB = 1024 ** 3
DEFAULT_MIN_SIZE = 0
DEFAULT_MAX_SIZE = 700 * MB
MIN_RANDOM_SIZE_ENV_VAR = "AZSTORAGETORCH_MIN_RANDOM_SIZE"
MAX_RANDOM_SIZE_ENV_VAR = "AZSTORAGETORCH_MAX_RANDOM_SIZE"


@pytest.fixture()
def data_size():
    min_size = os.environ.get(MIN_RANDOM_SIZE_ENV_VAR, DEFAULT_MIN_SIZE)
    max_size = os.environ.get(MAX_RANDOM_SIZE_ENV_VAR, DEFAULT_MAX_SIZE)
    return random.randint(min_size, max_size)


@pytest.fixture
def data(data_size):
    return sample_data(data_size)


@pytest.fixture
def data_md5(data):
    return hashlib.md5(data).hexdigest()


@pytest.fixture
def container_url():
    # return container_client.url
    return "https://myaccount.blob.core.windows.net/mycontainer"


@pytest.fixture
def blob_url(container_url, data_md5):
    return f"{container_url}/{data_md5}.bin"


@pytest.fixture
def recorder():
    return OperationRecorder()


@pytest.fixture
def assert_matches_original_data(data, data_md5, blob_url, save_failure):
    def _assert_data(provided_data):
        try:
            assert len(provided_data) == len(data)
            assert hashlib.md5(provided_data).hexdigest() == hashlib.md5(data).hexdigest()
        except AssertionError as e:
            save_failure(e)
            raise e
    return _assert_data


@pytest.fixture
def failures_directory():
    return os.path.join(STRESS_DIRNAME, 'failures')


@pytest.fixture
def failure_session_directory(failures_directory):
    session_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    return os.path.join(failures_directory, session_name)


@pytest.fixture
def save_failure(failure_session_directory, data, blob_url, recorder):
    def _save_failure(exception):
        print(f'Saving data from failed run in: {failure_session_directory}')
        os.makedirs(failure_session_directory, exist_ok=True)
        with open(os.path.join(failure_session_directory, 'data.bin'), 'wb') as f:
            f.write(data)
        with open(os.path.join(failure_session_directory, 'failure.json'), 'w') as f:
            content = {
                "blob_url": blob_url,
                "exception": str(exception),
                "operations": recorder.to_dict()["operations"],
            }
            f.write(json.dumps(content, indent=2))
    return _save_failure


CURRENT_DATA = None


def blobio(url, mode):
    # return BlobIO(url, mode)
    if mode == "wb":
        write_io = io.BytesIO()
        write_io.close = lambda: save_to_current_data(write_io)
        return write_io
    elif mode == "rb":
        return io.BytesIO(CURRENT_DATA)


def save_to_current_data(bytes_io):
    global CURRENT_DATA
    CURRENT_DATA = bytes_io.getvalue()


def test_blobio_integrity(
    data,
    blob_url,
    recorder,
    assert_matches_original_data
):
    # Round trip data, reading and writing all
    # roundtrip_data = roundtrip(blob_url, data, recorder)
    # assert_matches_original_data(roundtrip_data)

    # Sequential read and write of random length
    # recorder.reset()
    # roundtrip_data = roundtrip(
    #     blob_url, data, recorder,
    #     read_fn=_random_seq_read,
    #     write_fn=_random_seq_write
    # )
    # assert_matches_original_data(roundtrip_data)


    # Random reads, seeks, and tells. Compare with BytesIO
    recorder.reset()
    planner = RandomReadOperationPlanner(num_operations=10_00, expected_data_length=len(data))
    planned_operations = planner.generate_plan()
    print(planned_operations)
    run_plan(data, planned_operations, recorder)


def roundtrip(blob_url, data, recorder, read_fn=None, write_fn=None):
    if read_fn is None:
        read_fn = _readall
    if write_fn is None:
        write_fn = _writeall
    with blobio(blob_url, "wb") as blob_io:
        write_fn(blob_io, data, recorder)
    with blobio(blob_url, "rb") as blob_io:
        return read_fn(blob_io, recorder, expected_data_length=len(data))


def run_plan(data, planned_operations, recorder):
    f = io.BytesIO(data)
    f2 = io.BytesIO(data)
    for operation in planned_operations:
        recorder.record(operation)
        ret = getattr(f, operation.name)(*operation.kwargs.values())
        ret2 = getattr(f2, operation.name)(*operation.kwargs.values())
        assert ret == ret2
        recorder.record_tell()
        assert f.tell() == f2.tell()
        # TODO: Seeking negative is capped at 0 for 1 and 2 whence? Need to check blobio

def _writeall(blob_io, data, recorder):
    recorder.record_write(data)
    blob_io.write(data)


def _random_seq_write(blob_io, data, recorder):
    amount_written = 0
    while amount_written < len(data):
        write_size = random.randint(0, len(data) - amount_written)
        data_to_write = data[amount_written:amount_written + write_size]
        recorder.record_write(data_to_write)
        blob_io.write(data_to_write)
        amount_written += write_size


def _readall(blob_io, recorder, **kwargs):
    recorder.record_read()
    return blob_io.read()


def _random_seq_read(blob_io, recorder, expected_data_length, **kwargs):
    data = b''
    read_size = None
    ret_data = b''
    while _still_has_data(read_size, ret_data):
        read_size = random.randint(0, expected_data_length)
        recorder.record_read(amt=read_size)
        ret_data = blob_io.read(read_size)
        data += ret_data
    return data


def _still_has_data(read_size, return_data):
    if not read_size:
        return True
    if return_data:
        return True
    return False


@dataclasses.dataclass
class Operation:
    name: str
    kwargs: dict[str, typing.Any] = dataclasses.field(default_factory=dict)


class OperationRecorder:
    def __init__(self):
        self.operations = []

    def reset(self):
        self.operations = []

    def to_dict(self):
        return {
            "operations": [
                dataclasses.asdict(operation)
                for operation in self.operations
            ]
        }

    def record(self, operation):
        self.operations.append(operation)

    def record_read(self, amt=None):
        self.record(
            Operation(
                name="read",
                kwargs={"amt": amt},
            )
        )

    def record_readline(self, amt=None):
        self.record(
            Operation(
                name="readline",
                kwargs={"amt": amt},
            )
        )

    def record_write(self, data):
        self.record(
            Operation(
                name="write",
                kwargs={"amt": len(data)},
            )
        )

    def record_seek(self, offset, whence):
        self.record(
            Operation(
                name="seek",
                kwargs={"offset": offset, "whence": whence},
            )
        )

    def record_tell(self):
        self.record(Operation(name="tell"))


class RandomReadOperationPlanner:
    _OPERATION_CHOICES = [
        "read",
        "readline",
        "seek",
    ]

    def __init__(self, num_operations, expected_data_length):
        self._num_operations = num_operations
        self._expected_data_length = expected_data_length

    def generate_plan(self):
        planned_operations = []
        for _ in range(self._num_operations):
            operation_name = random.choice(self._OPERATION_CHOICES)
            planned_operations.append(
                getattr(self, f"_get_random_{operation_name}")()
            )
        return planned_operations

    def _get_random_read(self):
        return Operation(
            name="read",
            kwargs={"amt": self._get_read_amt()},
        )

    def _get_random_readline(self):
        return Operation(
            name="readline",
            kwargs={"amt": self._get_read_amt()},
        )

    def _get_random_seek(self):
        random_whence = random.choice(
            [os.SEEK_SET, os.SEEK_CUR, os.SEEK_END]
        )
        if random_whence == os.SEEK_SET:
            random_offset = random.randint(0, self._expected_data_length)
        elif random_whence == os.SEEK_CUR:
            random_offset = random.randint(-self._expected_data_length, self._expected_data_length)
        else:
            random_offset = random.randint(-self._expected_data_length, 0)
        return Operation(
            name="seek",
            kwargs={
                "offset": random_offset,
                "whence": random_whence,
            },
        )

    def _get_read_amt(self):
        random_size = random.randint(0, self._expected_data_length)
        return random.choice([None, random_size])
