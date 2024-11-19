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
LOCAL_MODELS_DIR = os.path.join(ROOTDIR, "local-models")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def torch_load(cfg):
    assert_env_vars()
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    write_result_metadata(results_dir, cfg)
    for i in range(cfg["num-runs"]):
        LOGGER.info("Run %s - Loading model for run", i)
        kwargs = get_readable_file_kwargs(cfg)
        start = time.time()
        with get_readable_file(cfg, **kwargs) as f:
            if cfg["read-method"]["name"] == "torch-load":
                torch.load(f, weights_only=True)
            elif cfg["read-method"]["name"] == "readall":
                f.read()
        duration = time.time() - start
        del f
        del kwargs
        LOGGER.info(f"Run %s - Seconds to load model: %s", i, duration)
        with open(os.path.join(results_dir, f"{i}.txt"), "w") as f:
            f.write(f"{duration}\n")
        print(duration)

def assert_env_vars():
    if "AZURE_STORAGE_SAS_TOKEN" not in os.environ:
        raise ValueError("AZURE_STORAGE_SAS_TOKEN environment variable must be set")
    if "AZURE_STORAGE_KEY" not in os.environ:
        raise ValueError("AZURE_STORAGE_KEY environment variable must be set")


def write_result_metadata(results_dir, cfg):
    with open(os.path.join(results_dir, "metadata.json"), "w") as f:
        f.write(json.dumps(
            {
                'read-method': cfg["read-method"]["name"],
                'filelike-impl': cfg["filelike-impl"]["name"],
                'model-size': cfg["model"]["size"],
                'num-runs': cfg["num-runs"],
            },
            indent=2,
        ))


def get_readable_file(cfg, **kwargs):
    filelike_impl_name = cfg["filelike-impl"]["name"]
    if filelike_impl_name.startswith("blobio"):
        blob_url = get_blob_url(cfg["blob"], cfg["model"]["name"])
        sas_token = retrieve_sas_token(cfg)
        if sas_token is not None:
            blob_url += "?" + sas_token
        downloader_cls = get_blobio_downloader_cls(cfg["filelike-impl"]["blobio-impl"])
        return BlobIO(blob_url, mode="rb", _downloader_cls=downloader_cls)
    elif filelike_impl_name == "open":
        return open_local_model(cfg)
    elif filelike_impl_name == "bytesio":
        return BytesIO(kwargs.pop("preloaded_model_bytes"))
    elif filelike_impl_name == "adlfs":
        fs = adlfs.AzureBlobFileSystem(
            account_name=cfg["blob"]["account"],
            sas_token=retrieve_sas_token(cfg),
        )
        return fs.open(f"abfs://{cfg['blob']['container']}/{cfg['model']['name']}", "rb")
    elif filelike_impl_name == "blobfile":
        blob_cfg = cfg["blob"]
        blobfile_uri = f"az://{blob_cfg['account']}/{blob_cfg['container']}/{cfg['model']['name']}"
        return blobfile.BlobFile(blobfile_uri, "rb")
    raise ValueError(f"Unknown filelike-impl: {cfg['filelike-impl']}")


def get_readable_file_kwargs(cfg):
    if cfg["filelike-impl"]["name"] == "bytesio":
        # Preload model bytes to not include read from disk in latency measurement
        return {"preloaded_model_bytes": preload_model_bytes(cfg)}
    return {}

def open_local_model(cfg):
    return open(os.path.join(LOCAL_MODELS_DIR, cfg["model"]["name"]), "rb")


def preload_model_bytes(cfg):
    with open_local_model(cfg) as f:
        return f.read()


def get_blob_url(blob_cfg, name):
    return f'https://{blob_cfg["account"]}.blob.core.windows.net/{blob_cfg["container"]}/{name}'


def retrieve_sas_token(cfg):
    return os.environ.get(cfg["blob"]["sas-env-var"], None)


def get_blobio_downloader_cls(blobio_impl_cfg):
    return getattr(
        getattr(azstoragetorch.downloaders, blobio_impl_cfg["module"]),
        blobio_impl_cfg["class"],
    )


if __name__ == "__main__":
    torch_load()