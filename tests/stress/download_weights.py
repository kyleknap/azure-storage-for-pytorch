import dataclasses
import os
import pathlib
import typing

from transformers import (
    BertModel,
    AutoModelForImageClassification,
    AutoModelForCausalLM,
)
import torch

STRESS_DIRNAME = os.path.abspath(os.path.dirname(__file__))


@dataclasses.dataclass
class ModelInfo:
    name: str
    hf_id: str
    cls_method: typing.Any


MODELS_TO_DOWNLOAD = [
    ModelInfo(
        name="resnet50",
        hf_id="microsoft/resnet-50",
        cls_method=AutoModelForImageClassification.from_pretrained,
    ),
    ModelInfo(
        name="bert-large-uncased",
        hf_id="bert-large-uncased",
        cls_method=BertModel.from_pretrained,
    ),
    ModelInfo(
        name="phi4",
        hf_id="microsoft/phi-4",
        cls_method=AutoModelForCausalLM.from_pretrained,
    ),
]


def main():
    local_weights_dir = pathlib.Path(STRESS_DIRNAME, "local-weights")
    local_weights_dir.mkdir(parents=True, exist_ok=True)
    for model_info in MODELS_TO_DOWNLOAD:
        load_and_save_model(local_weights_dir, model_info)


def load_and_save_model(local_weight_dir, model_info):
    print(f"Downloading {model_info.name} weights...")
    weights_file = local_weight_dir / f"{model_info.name}.pth"
    if weights_file.exists():
        print(f"File {weights_file} already exists. Skipping download.")
        return
    model = model_info.cls_method(model_info.hf_id)
    save_model_state_dict(model, weights_file)


def save_model_state_dict(model, file_path):
    torch.save(model.state_dict(), file_path)


if __name__ == "__main__":
    main()
