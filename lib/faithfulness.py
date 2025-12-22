import torch
import torch.nn.functional as F
from torch import einsum


def get_target_list(labels, n_classes):
    ret = []
    for i in range(n_classes):
        ret.append(torch.ones_like(labels) * i)

    return ret


def evaluate_directional_faithfulness(
    attribution_method, images, labels, n_classes=10, normalize=True
):
    attributions = []
    target_list = get_target_list(labels, n_classes=n_classes)
    for target in target_list:
        attributions.append(attribution_method.attribute(images, target=target))

    attributions = torch.stack(attributions)  # (n_classes, batch_size, C, H, W)

    if normalize:
        attributions = F.normalize(attributions, dim=(2, 3, 4))

    scores = einsum("bchw, nbchw -> bn", images, attributions)

    acc = (scores.max(dim=1)[1] == labels).float().mean()
    print(f"Directional Faithfulness Accuracy: {acc.item():.4f}")

    return acc, scores, attributions
