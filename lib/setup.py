import timm
import torch
from torch.utils.data import DataLoader
from torchvision import models as torchvision_models

from lib.dataset import get_imagenet, get_imagenette, wrap_imagenet_model


def setup_device(**kwargs):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return torch.device(device)


def setup_dataloader(**kwargs):
    dataset = kwargs.get("dataset", "imagenette")

    if dataset == "imagenet":
        dataset = get_imagenet()
    elif dataset == "imagenette":
        try:
            dataset = get_imagenette(download=True)
        except RuntimeError as e:
            # wierdly, Imagenette raises error if already downloaded (at least in some torchvision versions)
            print(e)
            dataset = get_imagenette(download=False)

    seed = kwargs.get("seed", 314)
    batch_size = kwargs.get("batch_size", 10)

    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )

    return loader


def setup_model(device, **kwargs):
    model_name = kwargs.get("model_name", "resnet50")
    model_source = kwargs.get("model_source", "torchvision")

    if model_source == "torchvision":
        backbone = torchvision_models.get_model(model_name, pretrained=True)
    elif model_source == "timm":
        backbone = timm.create_model(model_name, pretrained=True)

    model = wrap_imagenet_model(backbone)
    model = model.to(device)
    model.eval()

    return model


def setup_notebook(**kwargs):
    device = setup_device(**kwargs)

    return (device, setup_dataloader(**kwargs), setup_model(device=device, **kwargs))
