import os

import azure.storage.blob
from azure.identity import DefaultAzureCredential


ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
LOCAL_WEIGHTS_DIR = os.path.join(ROOT_DIR, "local-models")
LOCAL_WEIGHTS_TO_DELETE = [
    "resnet18_weights_saved.pth",
    "resnet18_weights_saved_by_filelike.pth"
]

CONTAINER_URL = "https://azstoragetorchdev.blob.core.windows.net/demo"
BLOBS_TO_DELETE = [
    "models/resnet18_weights_saved.pth",
]


def cleanup():
    for filename in LOCAL_WEIGHTS_TO_DELETE:
        file_path = os.path.join(LOCAL_WEIGHTS_DIR, filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    for blob_name in BLOBS_TO_DELETE:
        blob_url = f"{CONTAINER_URL}/{blob_name}"
        blob_client = azure.storage.blob.BlobClient.from_blob_url(blob_url, credential=DefaultAzureCredential())
        if blob_client.exists():
            blob_client.delete_blob()
    

if __name__ == "__main__":
    cleanup()