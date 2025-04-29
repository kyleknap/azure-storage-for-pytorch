import logging
import functools
import json
import os
import time

import hydra
import hydra.core.hydra_config
import torch
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AzureSasCredential
import azure.storage.blob


import azstoragetorch.datasets

LOGGER = logging.getLogger(__name__)

ROOTDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def dataset_perf(cfg):
    assert_env_vars()
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    write_result_metadata(results_dir, cfg)

    create_dataset = get_dataset_factory(cfg)
    for i in range(cfg["num-runs"]):
        LOGGER.info("Run %s - Loading dataset", i)
        start = time.time()
        dataset = create_dataset()
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg["batch-size"],
            num_workers=cfg["num-workers"],
        )
        count = 0
        size = 0
        for batch in dataloader:
            count += len(batch["url"])
            size += sum([len(x) for x in batch["data"]])
        print(f"Count: {count}, Size: {size}")
        duration = time.time() - start
        del dataset
        del dataloader
        LOGGER.info(f"Run %s - Seconds to load: %s", i, duration)
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
                'dataset-cls': cfg["dataset-cls"],
                'dataset-cls-method': cfg["dataset-cls-method"],
                'prefix': cfg["blob"]["prefix"],
                'num-runs': cfg["num-runs"],
                'batch-size': cfg["batch-size"],
                'num-workers': cfg["num-workers"],
            },
            indent=2,
        ))


def get_dataset_factory(cfg, **kwargs):
    dataset_cls = getattr(azstoragetorch.datasets, cfg["dataset-cls"])
    method_name = cfg["dataset-cls-method"]
    dataset_cls_method = getattr(dataset_cls, method_name)
    sas = retrieve_sas_token(cfg)
    container_url = get_container_url(cfg["blob"])
    prefix = cfg["blob"]["prefix"]
    cls_method_kwargs = {}
    if method_name == "from_blob_urls":
        resource_url = get_blob_urls(container_url, prefix, sas)
    elif method_name == "from_container_url":
        resource_url = container_url
        if sas is not None:
            resource_url += f"?{sas}"
        cls_method_kwargs["prefix"] = prefix
    return functools.partial(dataset_cls_method, resource_url, **cls_method_kwargs)


def get_container_url(blob_cfg):
    return f'https://{blob_cfg["account"]}.blob.core.windows.net/{blob_cfg["container"]}'


def get_blob_urls(container_url, prefix, sas=None):
    if sas is None:
        credential = DefaultAzureCredential()
    else:
        credential = AzureSasCredential(sas)
    container_client = azure.storage.blob.ContainerClient.from_container_url(
        container_url, credential=credential
    )
    return [
        to_blob_url(container_url, blob_name, sas)
        for blob_name in container_client.list_blob_names(name_starts_with=prefix)
    ]

def to_blob_url(container_url, blob_name, sas=None):
    blob_url = f"{container_url}/{blob_name}"
    if sas is not None:
        blob_url += f"?{sas}"
    return blob_url


def retrieve_sas_token(cfg):
    return os.environ.get(cfg["blob"]["sas-env-var"], None)


if __name__ == "__main__":
    dataset_perf()