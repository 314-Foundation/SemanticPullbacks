import torch
import torch.nn as nn
from torchattacks.attack import Attack


class PGA(Attack):
    r"""
    Projected Gradient Ascent.
    [https://arxiv.org/abs/1706.06083]
    """

    def __init__(
        self,
        model,
        alpha=20.0,
        steps=10,
        eps=100,
        relative_alpha=False,
        self_explain=False,
        use_cross_entropy_loss=False,
        pnorm=2,
        clip_min=-1.0,
        clip_max=1.0,
        clip_margin=0.0,
        eps_for_division=1e-20,
    ):
        super().__init__("PGA", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

        self.clip_margin = clip_margin
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eps_for_division = eps_for_division
        self.supported_mode = ["default", "targeted"]

        self.use_cross_entropy_loss = use_cross_entropy_loss
        self.ce_loss = nn.CrossEntropyLoss()

        self.pnorm = pnorm
        self.relative_alpha = relative_alpha
        self.self_explain = self_explain

    def compute_loss(self, outputs, target):
        if self.self_explain:
            return 0.5 * (outputs**2).sum()
        if self.use_cross_entropy_loss:
            return -self.ce_loss(outputs, target)
        else:
            return outputs.flatten(1)[torch.arange(len(target)), target].sum()

    def clip_images_(self, images):
        if self.clip_margin is not None:
            return torch.clamp_(
                images,
                min=self.clip_min - self.clip_margin,
                max=self.clip_max + self.clip_margin,
            ).detach()

    def forward(self, images, labels):
        r"""
        Overridden.
        """

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self.get_target_label(images, labels)

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.get_logits(adv_images)

            # Calculate loss
            if self.targeted:
                cost = self.compute_loss(outputs, target_labels)
            else:
                cost = -self.compute_loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]

            adv_images = adv_images.detach()

            adv_images_norms = (
                torch.norm(adv_images.flatten(1), p=self.pnorm, dim=1)
                .clamp_min(self.eps_for_division)
                .view(-1, 1, 1, 1)
            )

            grad_norms = (
                torch.norm(grad.flatten(1), p=self.pnorm, dim=1)
                .clamp_min(self.eps_for_division)
                .view(-1, 1, 1, 1)
            )

            if self.alpha is not None:
                grad = grad / grad_norms
                if self.relative_alpha:
                    grad = grad * adv_images_norms
                grad = grad * self.alpha

            adv_images = adv_images + grad

            if self.eps is not None:
                delta = adv_images - images
                delta_norms = torch.norm(
                    delta.flatten(1), p=self.pnorm, dim=1
                ).clamp_min(self.eps_for_division)
                factor = self.eps / delta_norms
                factor = torch.min(factor, torch.ones_like(delta_norms))
                delta = delta * factor.view(-1, 1, 1, 1)

                adv_images = images + delta

            self.clip_images_(adv_images)

        return adv_images, grad
