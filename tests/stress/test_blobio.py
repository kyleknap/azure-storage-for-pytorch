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
from utils import MB, sample_data, get_human_readable_size, get_num_runs


STRESS_DIRNAME = os.path.abspath(os.path.dirname(__file__))


DEFAULT_MIN_SIZE = 0
DEFAULT_MAX_SIZE = 700 * MB
MIN_RANDOM_SIZE_ENV_VAR = "AZSTORAGETORCH_MIN_RANDOM_SIZE"
MAX_RANDOM_SIZE_ENV_VAR = "AZSTORAGETORCH_MAX_RANDOM_SIZE"
NUM_RANDOM_OPERATIONS_ENV_VAR = "AZSTORAGETORCH_NUM_RANDOM_OPERATIONS"


@pytest.fixture()
def data_size():
    min_size = os.environ.get(MIN_RANDOM_SIZE_ENV_VAR, DEFAULT_MIN_SIZE)
    max_size = os.environ.get(MAX_RANDOM_SIZE_ENV_VAR, DEFAULT_MAX_SIZE)
    return random.randint(min_size, max_size)


@pytest.fixture()
def num_random_operations():
    return os.environ.get(NUM_RANDOM_OPERATIONS_ENV_VAR, 100)


@pytest.fixture
def data(data_size):
    return sample_data(data_size)


@pytest.fixture
def data_md5(data):
    return hashlib.md5(data).hexdigest()


@pytest.fixture
def container_url(container_client):
    return container_client.url


@pytest.fixture
def blob_url(container_url, data_md5):
    return f"{container_url}/{data_md5}.bin"


@pytest.fixture
def recorder():
    return OperationRecorder()


@pytest.fixture
def random_operation_planner(num_random_operations, data_size):
    return RandomReadOperationPlanner(
        num_operations=num_random_operations,
        expected_data_length=data_size,
    )


@pytest.fixture
def assert_matches_original_data(data, save_failure):
    def _assert_data(provided_data):
        try:
            assert len(provided_data) == len(data)
            assert (
                hashlib.md5(provided_data).hexdigest() == hashlib.md5(data).hexdigest()
            )
        except AssertionError as e:
            save_failure(e)
            raise e

    return _assert_data


@pytest.fixture
def assert_matches_value(save_failure):
    def _assert_value(actual_value, expected_value):
        try:
            assert actual_value == expected_value
        except AssertionError as e:
            save_failure(e)
            raise e

    return _assert_value


@pytest.fixture
def failures_directory():
    return os.path.join(STRESS_DIRNAME, "failures")


@pytest.fixture
def failure_session_directory(failures_directory):
    session_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return os.path.join(failures_directory, session_name)


@pytest.fixture
def save_failure(failure_session_directory, data, blob_url, recorder):
    def _save_failure(exception):
        print(f"Saving data from failed run in: {failure_session_directory}")
        os.makedirs(failure_session_directory, exist_ok=True)
        with open(os.path.join(failure_session_directory, "data.bin"), "wb") as f:
            f.write(data)
        with open(os.path.join(failure_session_directory, "failure.json"), "w") as f:
            content = {
                "blob_url": blob_url,
                "exception": str(exception),
                "operations": recorder.to_dict()["operations"],
            }
            f.write(json.dumps(content, indent=2))

    return _save_failure


def blobio(url, mode):
    return BlobIO(url, mode)


@pytest.mark.parametrize("run_number", range(get_num_runs()))
def test_blobio_integrity(
    data,
    blob_url,
    recorder,
    random_operation_planner,
    assert_matches_original_data,
    assert_matches_value,
    run_number,
):
    print(f"Integrity check using blob size: {get_human_readable_size(len(data))}")
    # Round trip data, reading and writing all
    roundtrip_data = roundtrip(blob_url, data, recorder)
    assert_matches_original_data(roundtrip_data)

    # Sequential read and write of random length
    recorder.reset()
    roundtrip_data = roundtrip(
        blob_url, data, recorder, read_fn=_random_seq_read, write_fn=_random_seq_write
    )
    assert_matches_original_data(roundtrip_data)

    # Random reads, seeks, and tells. Compare with BytesIO
    recorder.reset()
    planned_operations = random_operation_planner.generate_plan()
    run_plan(blob_url, data, planned_operations, recorder, assert_matches_value)


def roundtrip(blob_url, data, recorder, read_fn=None, write_fn=None):
    if read_fn is None:
        read_fn = _readall
    if write_fn is None:
        write_fn = _writeall
    with blobio(blob_url, "wb") as blob_io:
        write_fn(blob_io, data, recorder)
    with blobio(blob_url, "rb") as blob_io:
        return read_fn(blob_io, recorder, expected_data_length=len(data))


def run_plan(blob_url, data, planned_operations, recorder, assert_matches_value):
    expected = io.BytesIO(data)
    current_pos = 0
    with blobio(blob_url, "rb") as f:
        for operation in planned_operations:
            _adjust_planned_seek_if_needed(operation, current_pos)
            recorder.record(operation)
            ret = getattr(f, operation.name)(*operation.kwargs.values())
            expected_ret = getattr(expected, operation.name)(*operation.kwargs.values())
            assert_matches_value(ret, expected_ret)
            recorder.record_tell()
            assert_matches_value(f.tell(), expected.tell())
            current_pos = expected.tell()


def _adjust_planned_seek_if_needed(planned_operation, current_pos):
    if planned_operation.name == "seek":
        offset = planned_operation.kwargs["offset"]
        whence = planned_operation.kwargs["whence"]
        if whence == os.SEEK_CUR and current_pos + offset < 0:
            planned_operation.kwargs["offset"] = -current_pos


def _writeall(blob_io, data, recorder):
    recorder.record_write(data)
    blob_io.write(data)


def _random_seq_write(blob_io, data, recorder):
    amount_written = 0
    while amount_written < len(data):
        write_size = random.randint(0, len(data) - amount_written)
        data_to_write = data[amount_written : amount_written + write_size]
        recorder.record_write(data_to_write)
        blob_io.write(data_to_write)
        amount_written += write_size


def _readall(blob_io, recorder, **kwargs):
    recorder.record_read()
    return blob_io.read()


def _random_seq_read(blob_io, recorder, expected_data_length, **kwargs):
    data = b""
    read_size = None
    ret_data = b""
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
                dataclasses.asdict(operation) for operation in self.operations
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
            planned_operations.append(getattr(self, f"_get_random_{operation_name}")())
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
        random_whence = random.choice([os.SEEK_SET, os.SEEK_CUR, os.SEEK_END])
        if random_whence == os.SEEK_SET:
            random_offset = random.randint(0, self._expected_data_length)
        elif random_whence == os.SEEK_CUR:
            random_offset = random.randint(
                -self._expected_data_length, self._expected_data_length
            )
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
