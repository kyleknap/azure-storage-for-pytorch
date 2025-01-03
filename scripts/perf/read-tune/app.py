import logging
import json
import os
import time
from io import BytesIO

import hydra
import hydra.core.hydra_config
import torch
import adlfs
import blobfile

from azstoragetorch.io import BlobIO
import azstoragetorch.downloaders.thread
import azstoragetorch.downloaders.aio
import azstoragetorch.downloaders.process

LOGGER = logging.getLogger(__name__)

ROOTDIR = os.path.dirname(os.path.abspath(__file__))


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def read_tune(cfg):
    assert_env_vars()
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    write_result_metadata(results_dir, cfg)
    for i in range(cfg["num-runs"]):
        LOGGER.info("Run %s - Loading model for run", i)
        start = time.time()
        with get_readable_file(cfg) as f:
            if cfg["read-method"]["name"] == "torch-load":
                torch.load(f, weights_only=True)
            elif cfg["read-method"]["name"] == "readall":
                f.read()
        duration = time.time() - start
        del f
        LOGGER.info(f"Run %s - Seconds to load model: %s", i, duration)
        with open(os.path.join(results_dir, f"{i}.txt"), "w") as f:
            f.write(f"{duration}\n")
        print(duration)

def assert_env_vars():
    if "AZURE_STORAGE_SAS_TOKEN" not in os.environ:
        raise ValueError("AZURE_STORAGE_SAS_TOKEN environment variable must be set")


def write_result_metadata(results_dir, cfg):
    with open(os.path.join(results_dir, "metadata.json"), "w") as f:
        f.write(json.dumps(
            {
                'read-method': cfg["read-method"]["name"],
                'max-concurrency': cfg["max-concurrency"],
                'connection-data-block-size': cfg["connection-data-block-size"],
                'partition-threshold': cfg["partition-threshold"],
                'partition-size': cfg["partition-size"],
                'model-size': cfg["model"]["size"],
                'num-runs': cfg["num-runs"],
            },
            indent=2,
        ))


def get_readable_file(cfg):
    blob_url = get_blob_url(cfg["blob"], cfg["model"]["name"])
    sas_token = retrieve_sas_token(cfg)
    if sas_token is not None:
        blob_url += "?" + sas_token
    downloader_cls = azstoragetorch.downloaders.thread.GeneratedSDKDownloader
    downloader_cls._DEFAULT_MAX_CONCURRENCY = cfg["max-concurrency"]
    downloader_cls._CONNECTION_DATA_BLOCK_SIZE = cfg["connection-data-block-size"]
    downloader_cls._PARTITION_SIZE = cfg["partition-size"]
    downloader_cls._THRESHOLD = cfg["partition-threshold"]
    return BlobIO(blob_url, mode="rb", _downloader_cls=downloader_cls)



def get_blob_url(blob_cfg, name):
    return f'https://{blob_cfg["account"]}.blob.core.windows.net/{blob_cfg["container"]}/{name}'


def retrieve_sas_token(cfg):
    return os.environ.get(cfg["blob"]["sas-env-var"], None)


if __name__ == "__main__":
    read_tune()