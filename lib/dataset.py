import torch
from captum.attr._utils.lrp_rules import EpsilonRule
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import transforms as T
from torchvision.datasets import ImageFolder, Imagenette

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a "text: imagenet 1000 class idx to human readable labels (Fox, E ..."
IMAGENETTE_TO_IMAGENET = [
    0,  # tench
    217,  # English springer
    482,  # cassette player
    491,  # chain saw
    497,  # church
    566,  # French horn
    569,  # garbage truck
    571,  # gas pump
    574,  # golf ball
    701,  # parachute
]
CLASS_NAMES = {
    0: ("tench", "Tinca tinca"),
    217: ("spaniel", "English springer", "English springer spaniel"),
    482: (
        "cas. player",
        "cassette player",
    ),
    491: ("chain saw", "chainsaw"),
    497: ("church", "church building"),
    566: ("horn", "French horn"),
    569: ("dustcart", "garbage truck"),
    571: ("gas pump", "gasoline pump", "petrol pump", "island dispenser"),
    574: ("golf ball",),
    701: ("parachute", "chute"),
}


class ImagenetToImagenetteHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "imagenet_indices", torch.tensor(IMAGENETTE_TO_IMAGENET, dtype=torch.long)
        )

    def forward(self, logits_imagenet: torch.Tensor) -> torch.Tensor:
        return logits_imagenet.index_select(dim=-1, index=self.imagenet_indices)


def imagenette_label_to_imagenet(label):
    return IMAGENETTE_TO_IMAGENET[label]


def my_normalize():
    return T.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )


def my_denormalize(x):
    return (x + 1) / 2  # Map from [-1,1] to [0,1]


class ImageNetNormalize(nn.Module):
    def __init__(self):
        super().__init__()
        self.imagenet_norm = T.Normalize(
            mean=MEAN,
            std=STD,
        )

    def forward(self, x):
        return self.imagenet_norm(x)


class FromMyNormalizeToImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        # self.imagenet_norm = ImageNetNormalize()
        self.mean = MEAN
        self.std = STD
        self.rule = EpsilonRule()

    def forward(self, x):
        x = (x + 1) / 2  # Map from [-1,1] to [0,1]
        return T.functional.normalize(x, self.mean, self.std, inplace=False)


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            my_normalize(),
        ]
    )


class ConditionalTransform(nn.Module):
    def __init__(self):
        super().__init__()
        self.resize_224 = T.Resize(224)
        self.resize_256 = T.Resize(256)
        self.crop_224 = T.CenterCrop(224)
        self.to_tensor = T.ToTensor()
        self.normalize = my_normalize()
        # self.normalize = ImageNetNormalize()

    def __call__(self, img):
        # img: PIL.Image
        w, h = img.size
        if w == h:
            img = self.resize_224(img)
        else:
            img = self.resize_256(img)
            img = self.crop_224(img)
        img = self.to_tensor(img)
        img = self.normalize(img)
        return img


def get_imagenette(download=False):
    return Imagenette(
        root="../data",
        split="val",  # or "train"
        size="160px",  # can also be "320" or "full"
        download=download,
        transform=ConditionalTransform(),
        target_transform=imagenette_label_to_imagenet,
    )


def get_imagenet():
    return ImageFolder(
        root="../data/imagenet/val",
        # transform=get_transform(),
        transform=ConditionalTransform(),
        # target_transform=None,
    )


def get_examples(loader, classes=None):
    class_indices = [0, 2, 4, 6, 8] if classes is None else classes
    target_classes = [imagenette_label_to_imagenet(l) for l in class_indices]
    # target_classes = class_indices

    selected = {}
    seen_classes = set()

    for batch in loader:
        images, labels = batch
        for img, label in zip(images, labels):
            label = label.item()
            if label in target_classes and label not in seen_classes:
                selected[label] = img.unsqueeze(0)
                seen_classes.add(label)
            if len(seen_classes) == len(target_classes):
                break
        if len(seen_classes) == len(target_classes):
            break

    # Concat images
    images = torch.cat([selected[c] for c in target_classes], dim=0)  # [5, C, H, W]
    labels = torch.tensor(target_classes)  # [5]

    return images, labels


def wrap_imagenet_model(model):
    # return nn.Sequential(ImageNetNormalize(), model, ImagenetToImagenetteHead())
    # return nn.Sequential(FromMyNormalizeToImageNet(), model, ImagenetToImagenetteHead())
    return nn.Sequential(FromMyNormalizeToImageNet(), model)
    # return model
