import torch
import torch.nn.functional as F
from torch import einsum


def get_target_list(preds, n_classes):
    ret = []
    for i in range(n_classes):
        ret.append(torch.ones_like(preds) * i)

    return ret


def evaluate_response_faithfulness(
    attribution_method, images, preds, n_classes=10, attribute_kwargs=None
):
    if attribute_kwargs is None:
        attribute_kwargs = {}
    attributions = []
    target_list = get_target_list(preds, n_classes=n_classes)
    for target in target_list:
        attributions.append(
            attribution_method.attribute(images, target=target, **attribute_kwargs)
        )
    attributions = torch.stack(attributions, dim=1)  # (batch_size, n_classes, C, H, W)

    scores = einsum("bchw, bnchw -> bn", images, attributions)

    acc = (scores.max(dim=1)[1] == preds).float().mean()
    print(f"Directional Faithfulness Accuracy: {acc.item():.4f}")
    print(f"Scores shape: {scores.shape}")

    return acc, scores, attributions
