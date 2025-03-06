import pytest


@pytest.fixture
def blob_url():
    return "https://myaccount.blob.core.windows.net/mycontainer/myblob"


@pytest.fixture
def blob_content():
    return b"blob content"


@pytest.fixture
def blob_length(blob_content):
    return len(blob_content)
