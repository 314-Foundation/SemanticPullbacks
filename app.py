import ast
import os

import gradio as gr
import numpy as np
import spaces
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import Imagenette
from torchvision.models import densenet121, resnet50, vgg11_bn

from lib.dataset import (
    ConditionalTransform,
    FromMyNormalizeToImageNet,
    imagenette_label_to_imagenet,
)
from lib.helpers import maxpool2d_param_extractor, replace_module_with_custom_
from lib.pga import PGA
from lib.surrogates import SurrogateSoftMaxPool2d, TwoWayReLU

if torch.cuda.is_available():
    device_ = "cuda"
elif torch.backends.mps.is_available():
    device_ = "mps"
else:
    device_ = "cpu"
DEVICE = torch.device(device_)


####
## Data
####


INIT_CLASS = 497

# Predefined class names (shortened for demo)
PREDEFINED_CLASSES = {
    0: "tench",
    217: "English springer",
    482: "cassette player",
    491: "chain saw",
    497: "church",
    566: "French horn",
    569: "garbage truck",
    571: "gas pump",
    574: "golf ball",
    701: "parachute",
    9: "ostrich",
}


def load_imagenet_labels(path="imagenet1000_clsidx_to_labels.txt"):
    with open(path, "r") as f:
        labels = ast.literal_eval(f.read())
    return labels


IMAGENET_LABELS = load_imagenet_labels()


def get_class_name(target_class):
    return IMAGENET_LABELS.get(int(target_class), "Unknown")


def get_dataset(download=False):
    return Imagenette(
        root="./data",
        split="val",  # or "train"
        size="160px",  # can also be "320" or "full"
        download=download,
        transform=None,
        target_transform=imagenette_label_to_imagenet,
    )


# Predefined images from Imagenette val
try:
    DATASET = get_dataset(download=True)
except RuntimeError as e:
    # wierdly, Imagenette raises error if already downloaded (at least in some torchvision versions)
    print(e)
    DATASET = get_dataset(download=False)


def sample_val_img():
    idx = np.random.randint(0, len(DATASET))
    img, _ = DATASET[idx]

    return img


# Load predefined images from examples folder
EXAMPLES_DIR = "examples"
predefined_files = sorted(
    [
        os.path.join(EXAMPLES_DIR, f)
        for f in os.listdir(EXAMPLES_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
)
# PREDEFINED_IMAGES = predefined_files
# [os.path.join(EXAMPLES_DIR, fname)] for fname in predefined_files
PREDEFINED_IMAGES = [
    np.array(Image.open(fname).convert("RGB")) for fname in predefined_files
]


def tensor_to_gradio_image(tensor):
    # tensor: [B, C, H, W] lub [C, H, W]
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    grid = vutils.make_grid(tensor, nrow=1, normalize=True, scale_each=True)
    # grid: [C, H, W]
    img = grid.permute(1, 2, 0).cpu().numpy()  # [H, W, C]
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return img


####
## Model
####


# Model mapping
MODEL_MAP = {
    "ResNet50": resnet50,
    "VGG11_BN": vgg11_bn,
    "DenseNet121": densenet121,
}
INIT_MODEL_NAME = "ResNet50"
INIT_MODEL_TEMP = 0.3

current_model = None
current_model_params = None


def get_model(model_name, temp=0.3):
    global current_model, current_model_params

    params = (model_name, temp)
    if current_model is not None and current_model_params == params:
        return current_model

    backbone = MODEL_MAP[model_name](pretrained=True)
    model = nn.Sequential(FromMyNormalizeToImageNet(), backbone)
    model.eval()

    replace_module_with_custom_(
        model, lambda: TwoWayReLU(temperature=temp), original_cls=nn.ReLU
    )
    replace_module_with_custom_(
        model,
        lambda **params: SurrogateSoftMaxPool2d(**params, temperature=temp),
        original_cls=nn.MaxPool2d,
        param_extractor=maxpool2d_param_extractor,
    )

    model = model.to(DEVICE)

    current_model = model
    current_model_params = params
    return model


@spaces.GPU
def run_pullback(
    input_image,
    model_name,
    target_class,
    steps,
    alpha,
    eps,
    temp,
):
    image_transform = ConditionalTransform()
    img_tensor = T.ToPILImage()(input_image)
    img_tensor = image_transform(img_tensor).unsqueeze(0).to(DEVICE)

    model = get_model(model_name, temp=temp)

    # Prepare target
    target = torch.tensor([target_class]).to(DEVICE)

    predicted_top5 = model(img_tensor).topk(5, dim=1)[1].flatten().tolist()  # [0])

    # Compute gradients/perturbation
    atk = PGA(
        model,
        alpha=alpha,
        steps=steps,
        eps=eps,
    )
    atk.set_mode_targeted_by_label()
    perturbed_img, grad = atk(img_tensor, target)

    # Visualize

    diff_img = perturbed_img - img_tensor

    img_tensor = tensor_to_gradio_image(img_tensor)
    diff_img = tensor_to_gradio_image(diff_img)
    perturbed_img = tensor_to_gradio_image(perturbed_img)

    return (perturbed_img, diff_img), (perturbed_img, img_tensor), predicted_top5


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # Excitation Pullbacks: Enhancing Human Perception with AI

        Instead of generating hallucinations, our method amplifies the real predictive features in the data, transforming AI into a transparent tool for high-stakes decision-making.
        The current implementation relies on a single architectural hyperparameter (temp) shared across all hidden neurons. Future work will enable neuron-specific adjustment, which is expected to significantly enhance explanation quality.
        """
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                Choose an input image - either sample from [Imagenette](https://github.com/fastai/imagenette) dataset, select a predefined example or upload your own. Square images are resized to 224x224 pixels, others are first resized to 256x256 and then center-cropped to 224x224 pixels.
                """
            )
            input_image = gr.Image(
                type="numpy", label="Input Image", value=PREDEFINED_IMAGES[4]
            )
            sample_from_val = gr.Button("Sample from Imagenette val")
            examples = gr.Examples(
                examples=predefined_files,
                # examples=PREDEFINED_IMAGES,
                inputs=[input_image],
                label=f"Example images from Imagenette val (corresponding to Example classes)",
            )

        with gr.Column():
            gr.Markdown(
                """
                Select a target class and amplify it's features -  compute Projected Gradient Ascent along the Excitaton Pullback. Very low temperature approximates (noisy) vanilla gradients, while very high temperature linearizes the model.
                """
            )
            with gr.Row():
                # with gr.Column():
                target_class = gr.Dropdown(
                    label="Target Class (ImageNet)",
                    info="idx - class name",
                    choices=[
                        (f"{idx} - {name}", int(idx))
                        for idx, name in IMAGENET_LABELS.items()
                    ],
                    value=INIT_CLASS,
                    allow_custom_value=False,
                    # render=False,
                )
                # with gr.Column():
                model_name = gr.Dropdown(
                    list(MODEL_MAP.keys()),
                    value=INIT_MODEL_NAME,
                    label="Model",
                    info="ImageNet-pretrained ReLU model",
                )
            with gr.Row():
                examples = gr.Examples(
                    # examples=[
                    #     [os.path.join(EXAMPLES_DIR, fname)] for fname in predefined_files
                    # ],
                    # examples=predefined_files,
                    examples=list(PREDEFINED_CLASSES.keys()),
                    example_labels=[
                        f"{cls} - {PREDEFINED_CLASSES[cls]}"
                        for cls in PREDEFINED_CLASSES.keys()
                    ],
                    inputs=[target_class],
                    label=f'Example classes (corresponding to Example images + "ostrich")',
                    examples_per_page=11,
                )
            with gr.Row():
                steps = gr.Number(
                    value=10,
                    label="Steps",
                    info="N steps for Projected Gradient Ascent",
                    maximum=1000,
                    minimum=1,
                    precision=0,
                )
                alpha = gr.Number(
                    value=20,
                    label="Alpha",
                    info="Step size (in L2 norm)",
                    minimum=1.0,
                    step=1.0,
                )
                eps = gr.Number(
                    value=100,
                    label="Eps",
                    info="Maximum perturbation (in L2 norm)",
                    minimum=10,
                    step=10,
                )
                temp = gr.Number(
                    value=INIT_MODEL_TEMP,
                    label="Temp",
                    info="Temperature for soft gating (sigmoid)",
                    minimum=0.01,
                    step=0.01,
                )
            run_button = gr.ClearButton(components=None, value="Amplify features!")
            with gr.Row():
                predicted_class_name_output = gr.Textbox(
                    label="Input image predicted top5 labels",
                    interactive=False,
                )
    with gr.Row():
        # with gr.Column():
        diff_img = gr.ImageSlider(
            # diff_img = gr.Image(
            label="Perturbed / Difference",
            # max_height=800,
            max_height=500,
            # show_fullscreen_button=False,
            interactive=False,
            slider_position=50,
            # show_fullscreen_button=False,
        )
        perturbed_img = gr.ImageSlider(
            # perturbed_img = gr.Image(
            label="Perturbed / Input",
            # max_height=800,
            max_height=500,
            # show_fullscreen_button=False,
            interactive=False,
            slider_position=50,
        )

    run_button.add(perturbed_img)
    run_button.add(diff_img)
    sample_from_val.click(fn=sample_val_img, outputs=input_image)

    run_button.click(
        fn=run_pullback,
        inputs=[
            input_image,
            model_name,
            target_class,
            steps,
            alpha,
            eps,
            temp,
        ],
        outputs=[diff_img, perturbed_img, predicted_class_name_output],
    )

if __name__ == "__main__":
    get_model(INIT_MODEL_NAME, INIT_MODEL_TEMP)  # preload default model
    demo.launch()
