# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import pytest

from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from utils import random_resource_name


@pytest.fixture(scope="package")
def account_url():
    account_name = os.environ.get("AZSTORAGETORCH_STORAGE_ACCOUNT_NAME")
    if account_name is None:
        raise ValueError(
            f'"AZSTORAGETORCH_STORAGE_ACCOUNT_NAME" environment variable must be set to run end to end tests.'
        )
    return f"https://{account_name}.blob.core.windows.net"


@pytest.fixture(scope="package")
def container_client(account_url):
    blob_service_client = BlobServiceClient(
        account_url, credential=DefaultAzureCredential()
    )
    container_name = random_resource_name()
    container = blob_service_client.create_container(name=container_name)
    yield container
    container.delete_container()
