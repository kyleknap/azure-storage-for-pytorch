# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------


class AZStorageTorchError(Exception):
    """Base class for exceptions raised by ``azstoragetorch``."""

    pass


class FatalBlobIOWriteError(AZStorageTorchError):
    """Raised when a fatal error occurs during :py:class:`~azstoragetorch.io.BlobIO` write operations.

    When this exception is raised, it indicates no more writing can be performed
    on the :py:class:`~azstoragetorch.io.BlobIO` object and no blocks staged as part of this
    :py:class:`~azstoragetorch.io.BlobIO` will be committed. It is recommended to create a
    new :py:class:`~azstoragetorch.io.BlobIO` object and retry all writes when attempting retries.
    """

    _MSG_FORMAT = (
        "Fatal error occurred while writing data. No data written using this BlobIO instance "
        "will be committed to blob. Encountered exception:\n{underlying_exception}"
    )

    def __init__(self, underlying_exception: BaseException):
        super().__init__(
            self._MSG_FORMAT.format(underlying_exception=underlying_exception)
        )


class ClientRequestIdMismatchError(AZStorageTorchError):
    """Raised when a client request ID in a response does not match the ID in it's originating request.

    If receiving this error as part of using both an azstoragetorch dataset and a
    PyTorch DataLoader, it may be because the dataset is being accessed in both the
    main process and a DataLoader's worker process. This can cause unintentional
    sharing of resources. To fix this error, consider not accessing the dataset's
    samples in the main process or not using workers with the DataLoader.
    """

    _MSG_FORMAT = (
        "Client request ID: {request_client_id} does not match echoed client request ID: "
        "{response_client_id}.  Service request ID: {service_request_id}. "
        "If receiving this error as part of using both an azstoragetorch dataset and a "
        "PyTorch DataLoader, it may be because the dataset is being accessed in both the "
        "main process and a DataLoader's worker process. This can cause unintentional "
        "sharing of resources. To fix this error, consider not accessing the dataset's "
        "samples in the main process or not using workers with the DataLoader."
    )

    def __init__(
        self, request_client_id: str, response_client_id: str, service_request_id: str
    ):
        super().__init__(
            self._MSG_FORMAT.format(
                request_client_id=request_client_id,
                response_client_id=response_client_id,
                service_request_id=service_request_id,
            )
        )
