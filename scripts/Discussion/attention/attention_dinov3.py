import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchvision.io import read_image
from pathlib import Path
import sys
import os

os.environ["XFORMERS_DISABLED"] = "1"

# -------------------------------------------------------
# find project root
# -------------------------------------------------------

current = Path(__file__).resolve()

for parent in current.parents:
    if (parent / "src").exists():
        sys.path.append(str(parent / "src"))
        break
else:
    raise RuntimeError("Could not find project root containing 'src' folder.")

from model_lib.neural_nets import fine_tuning

# -------------------------------------------------------
# settings
# -------------------------------------------------------

IMAGE_FOLDER = "z_field_example_choosen"
CHECKPOINT_PATH = "data/1_a_field_fine_tuned_mlp_dinov3/cnn_run_0/model_6.pth"
#CHECKPOINT_PATH = "data/2_fine-tuned-mlp/1_dinov3-1imgs_mlp_field_images/cnn_run_0/model_34.pth" #not

image_prefix = "mlp_dinov3_fine_tuned"


SAVE_FOLDER = "z_heatmaps"
#SAVE_FOLDER = "z_dinov3_patch_grad_not"

Path(SAVE_FOLDER).mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# load model
# -------------------------------------------------------

model = fine_tuning(
    backbone_name="vit_small_patch16_dinov3",
    pretrained=True,
    out_dim=128
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)
model.eval()

print("Model loaded successfully")

# -------------------------------------------------------
# transforms
# -------------------------------------------------------

geom_transform = torch.nn.Sequential(
    v2.Resize(256, antialias=True),
    v2.CenterCrop(224),
)

norm_transform = v2.Normalize(
    mean=[0.485,0.456,0.406],
    std=[0.229,0.224,0.225]
)

# -------------------------------------------------------
# hook storage
# -------------------------------------------------------
features = []
grads = []

def feature_hook(module, inp, out):

    out.retain_grad()

    features.append(out)

    def save_grad(grad):
        grads.append(grad)

    out.register_hook(save_grad)

layers = [-1, -2, -3]

for l in layers:
    model.backbone.blocks[l].register_forward_hook(feature_hook)
# -------------------------------------------------------
# image list
# -------------------------------------------------------

image_paths = sorted(
    list(Path(IMAGE_FOLDER).glob("*.jpg")) +
    list(Path(IMAGE_FOLDER).glob("*.png")) +
    list(Path(IMAGE_FOLDER).glob("*.jpeg"))
)

# -------------------------------------------------------
# loop
# -------------------------------------------------------

for IMAGE_PATH in image_paths:
    features = []
    grads = []

    print("Processing:", IMAGE_PATH)

    torch_img = read_image(str(IMAGE_PATH)).float() / 255.0
    img_geom = geom_transform(torch_img)
    img_model = norm_transform(img_geom)

    img_tensor = img_model.unsqueeze(0).unsqueeze(0).to(DEVICE)

    img_np = img_geom.permute(1,2,0).numpy()

    mask = torch.zeros((1,1), dtype=torch.bool).to(DEVICE)

    # forward
    output = model(img_tensor, mask).squeeze()

    model.zero_grad()
    output.backward()

    cams = []

    for f, g in zip(features, grads):

        # remove CLS + registers
        patch_tokens = f[:,5:,:]
        patch_grads  = g[:,5:,:]

        # Grad-CAM weights
        #weights = patch_grads.mean(dim=-1, keepdim=True)

        cam = torch.relu((patch_grads * patch_tokens).sum(dim=-1))

        cam = cam.reshape(14,14)

        cams.append(cam)

    # average across layers
    cam = torch.stack(cams).mean(0)

    importance = cam.detach().cpu().numpy()


    # normalize
    importance -= importance.min()
    importance /= importance.max() + 1e-8

    # upsample to image resolution
    importance = cv2.resize(importance, (224,224), interpolation=cv2.INTER_CUBIC)

    # smooth slightly
    importance = cv2.GaussianBlur(importance, (5,5), 0)


    print("Heatmap checksum:", float(importance.sum()))

    save_path = Path(SAVE_FOLDER) / f"{image_prefix}_{Path(IMAGE_PATH).stem}.png"

    plt.figure(figsize=(4,4))
    plt.imshow(img_np)
    plt.imshow(importance, cmap="inferno", alpha=0.6)
    plt.axis("off")
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()