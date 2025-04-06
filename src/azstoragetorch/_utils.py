# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE in the project root for
# license information.
# --------------------------------------------------------------------------
from typing import Optional, Union, Literal
import urllib.parse

from azure.identity import DefaultAzureCredential
from azure.core.credentials import (
    AzureSasCredential,
    TokenCredential,
)


SDK_CREDENTIAL_TYPE = Optional[
    Union[
        AzureSasCredential,
        TokenCredential,
    ]
]
AZSTORAGETORCH_CREDENTIAL_TYPE = Union[SDK_CREDENTIAL_TYPE, Literal[False]]

# TODO: Probably need to make resource url optional for the blob_urls interface
def to_sdk_credential(
    resource_url: str, credential: AZSTORAGETORCH_CREDENTIAL_TYPE
) -> SDK_CREDENTIAL_TYPE:
    if credential is False or _url_has_sas_token(resource_url):
        return None
    if credential is None:
        return DefaultAzureCredential()
    if isinstance(credential, (AzureSasCredential, TokenCredential)):
        return credential
    # Hack to not worry how to handle query string in uri
    if isinstance(credential, str):
        return credential
    raise TypeError(f"Unsupported credential: {type(credential)}")


def _url_has_sas_token(resource_url: str) -> bool:
    parsed_url = urllib.parse.urlparse(resource_url)
    if parsed_url.query is None:
        return False
    parsed_qs = urllib.parse.parse_qs(parsed_url.query)
    # The signature is always required in a valid SAS token. So look for the "sig"
    # key to determine if the URL has a SAS token.
    return "sig" in parsed_qs
