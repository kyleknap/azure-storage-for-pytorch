import logging
import functools
import json
import os
import time

import hydra
import hydra.core.hydra_config
import torch
from azure.identity import DefaultAzureCredential
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
        dataloader = torch.utils.data.DataLoader(dataset, num_workers=cfg["num-workers"])
        count = 0
        for sample in dataloader:
            count += 1
        print(f"Count: {count}")
        duration = time.time() - start
        del dataset
        del dataloader
        LOGGER.info(f"Run %s - Seconds to load: %s", i, duration)
        with open(os.path.join(results_dir, f"{i}.txt"), "w") as f:
            f.write(f"{duration}\n")
        print(duration)


def assert_env_vars():
    return
    if "AZURE_STORAGE_SAS_TOKEN" not in os.environ:
        raise ValueError("AZURE_STORAGE_SAS_TOKEN environment variable must be set")


def write_result_metadata(results_dir, cfg):
    with open(os.path.join(results_dir, "metadata.json"), "w") as f:
        f.write(json.dumps(
            {
                'dataset-cls': cfg["dataset-cls"],
                'dataset-cls-method': cfg["dataset-cls-method"],
                'num-runs': cfg["num-runs"],
                'num-workers': cfg["num-workers"],
                'list-type': cfg["list-type"],
                'transform': cfg["transform"],
            },
            indent=2,
        ))


def get_dataset_factory(cfg, **kwargs):
    dataset_cls = getattr(azstoragetorch.datasets, cfg["dataset-cls"])
    method_name = cfg["dataset-cls-method"]
    dataset_cls_method = getattr(dataset_cls, method_name)
    # credential = retrieve_sas_token(cfg)
    credential = None
    cls_method_kwargs = {
        "credential": credential,
        "transform": noop_transform,
    }
    container_url = get_container_url(cfg["blob"])
    name_starts_with = cfg["blob"]["name-starts-with"]
    if method_name == "from_blob_urls":
        if credential is None:
            credential = DefaultAzureCredential()
        container_client = azure.storage.blob.ContainerClient.from_container_url(
            container_url, credential=credential
        )
        resource_url = [
            f"{container_url}/{blob_name}"
            for blob_name in container_client.list_blob_names(name_starts_with=name_starts_with)
        ]
    elif method_name == "from_container_url":
        resource_url = container_url
        cls_method_kwargs["name_starts_with"] = name_starts_with
        cls_method_kwargs["list_type"] = cfg["list-type"]
        cls_method_kwargs["proxy_blob_properties"] = cfg["proxy-blob-properties"]
    if cfg["transform"] == "read":
        cls_method_kwargs["transform"] = read_transform
    cls_method_kwargs["share_pipeline"] = cfg["share-pipeline"]
    return functools.partial(dataset_cls_method, resource_url, **cls_method_kwargs)


def get_container_url(blob_cfg):
    return f'https://{blob_cfg["account"]}.blob.core.windows.net/{blob_cfg["container"]}'


def retrieve_sas_token(cfg):
    return os.environ.get(cfg["blob"]["sas-env-var"], None)


def noop_transform(blob):
    return torch.tensor([])


def read_transform(blob):
    with blob.stream() as f:
        return f.read()


if __name__ == "__main__":
    dataset_perf()