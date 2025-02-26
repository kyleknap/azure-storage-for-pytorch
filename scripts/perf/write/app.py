import logging
import json
import os
import tempfile
import time
from io import BytesIO, IOBase

import hydra
import hydra.core.hydra_config
import torch
import adlfs
import blobfile

from azstoragetorch.io import BlobIO

LOGGER = logging.getLogger(__name__)

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MODELS_DIR = os.path.join(ROOTDIR, "local-models")


class NoopIO(IOBase):
    def writable(self):
        return True

    def write(self, b):
        return len(b)

    def flush(self):
        pass

    def close(self):
        pass


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def write_perf(cfg):
    assert_env_vars()
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    write_result_metadata(results_dir, cfg)
    content = preload_model(cfg)
    for i in range(cfg["num-runs"]):
        LOGGER.info("Run %s - Saving model for run", i)
        start = time.time()
        with get_writeable_file(cfg) as f:
            if cfg["write-method"]["name"] == "torch-save":
                torch.save(content, f)
            elif cfg["write-method"]["name"] == "writeall":
                f.write(content)
        duration = time.time() - start
        del f
        LOGGER.info(f"Run %s - Seconds to write model: %s", i, duration)
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
                'write-method': cfg["write-method"]["name"],
                'filelike-impl': cfg["filelike-impl"]["name"],
                'model-size': cfg["model"]["size"],
                'num-runs': cfg["num-runs"],
            },
            indent=2,
        ))


def get_writeable_file(cfg, **kwargs):
    filelike_impl_name = cfg["filelike-impl"]["name"]
    if filelike_impl_name.startswith("blobio"):
        blob_url = get_blob_url(cfg["blob"], cfg["model"]["name"])
        sas_token = retrieve_sas_token(cfg)
        if sas_token is not None:
            blob_url += "?" + sas_token
        blob_io = BlobIO(blob_url, mode="wb")
        if filelike_impl_name == "blobio-no-wait":
            blob_io._WAIT_FOR_WRITES = False
        return blob_io
    elif filelike_impl_name == "open":
        return tempfile.TemporaryFile("wb")
    elif filelike_impl_name == "bytesio":
        return BytesIO()
    elif filelike_impl_name == "noop":
        return NoopIO()
    elif filelike_impl_name == "adlfs":
        fs = adlfs.AzureBlobFileSystem(
            account_name=cfg["blob"]["account"],
            sas_token=retrieve_sas_token(cfg),
        )
        return fs.open(f"abfs://{cfg['blob']['container']}/write-output/{cfg['model']['name']}", "wb")
    elif filelike_impl_name == "blobfile":
        blob_cfg = cfg["blob"]
        blobfile_uri = f"az://{blob_cfg['account']}/{blob_cfg['container']}/write-output/{cfg['model']['name']}"
        return blobfile.BlobFile(blobfile_uri, "wb")
    raise ValueError(f"Unknown filelike-impl: {cfg['filelike-impl']}")


def open_local_model(cfg):
    return open(os.path.join(LOCAL_MODELS_DIR, cfg["model"]["name"]), "rb")


def preload_model(cfg):
    if cfg["write-method"]["name"] == "torch-save":
        with open_local_model(cfg) as f:
            return torch.load(f, weights_only=True)
    if cfg["write-method"]["name"] == "writeall":
        return preload_model_bytes(cfg)


def preload_model_bytes(cfg):
    with open_local_model(cfg) as f:
        return f.read()


def get_blob_url(blob_cfg, name):
    return f'https://{blob_cfg["account"]}.blob.core.windows.net/{blob_cfg["container"]}/write-output/{name}'


def retrieve_sas_token(cfg):
    return os.environ.get(cfg["blob"]["sas-env-var"], None)


if __name__ == "__main__":
    write_perf()