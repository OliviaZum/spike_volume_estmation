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

from model_lib.neural_nets import fine_tuning

# -------------------------------------------------------
# SETTINGS
# -------------------------------------------------------

IMAGE_FOLDER = "z_field_example_choosen"
SAVE_FOLDER = "z_eigencams"

# ---------- CHOOSE MODEL ----------

BACKBONE = "resnet50"
CHECKPOINT_PATH = "data/1_resnet50_1imgs_mlp_field_images/cnn_run_0/model_19.pth"
#CHECKPOINT_PATH = "data/1_a_field_fine_tuned_mlp_resnet50/cnn_run_0/model_6.pth"

IMAGE_PREFIX = "eigencam_resnet50"

# BACKBONE = "resnet18"
# CHECKPOINT_PATH = "data/1_resnet18_1imgs_mlp_field_images/cnn_run_0/model_25.pth"
# IMAGE_PREFIX = "eigencam_resnet18"

Path(SAVE_FOLDER).mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------

model = fine_tuning(
    backbone_name=BACKBONE,
    pretrained=True,
    out_dim=128
)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("Model loaded")

# -------------------------------------------------------
# IMAGE TRANSFORMS
# -------------------------------------------------------

geom_transform = torch.nn.Sequential(
    v2.Resize(256, antialias=True),
    v2.CenterCrop(224),
)

norm_transform = v2.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# -------------------------------------------------------
# FEATURE HOOK
# -------------------------------------------------------

feature_maps = []

def forward_hook(module, input, output):
    feature_maps.append(output.detach())

target_layer = model.backbone.layer4
target_layer.register_forward_hook(forward_hook)

# -------------------------------------------------------
# PROCESS IMAGES
# -------------------------------------------------------

image_paths = sorted(Path(IMAGE_FOLDER).glob("*"))

for IMAGE_PATH in image_paths:

    print("Processing:", IMAGE_PATH.name)

    feature_maps.clear()

    # load image
    torch_img = read_image(str(IMAGE_PATH)).float() / 255.0
    img_geom = geom_transform(torch_img)
    img_model = norm_transform(img_geom)

    img_tensor = img_model.unsqueeze(0).unsqueeze(0).to(DEVICE)
    img_np = img_geom.permute(1,2,0).numpy()

    mask = torch.zeros((1,1), dtype=torch.bool).to(DEVICE)

    # forward pass
    with torch.no_grad():
        _ = model(img_tensor, mask)

    fmap = feature_maps[0]  # shape: (1, C, H, W)

    print("Feature map shape:", fmap.shape)
    print("Feature mean:", fmap.abs().mean().item())

    B, C, H, W = fmap.shape

    # -------------------------------------------------------
    # EigenCAM
    # -------------------------------------------------------

    cam = fmap[0].abs().max(dim=0)[0].cpu().numpy()

    # normalize
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # resize
    cam = cv2.resize(cam, (224,224), interpolation=cv2.INTER_LINEAR)

    # amplify contrast (important!)
    cam = np.power(cam, 3)

    # stretch dynamic range
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    # -------------------------------------------------------
    # PLOT
    # -------------------------------------------------------

    save_path = Path(SAVE_FOLDER) / f"{IMAGE_PREFIX}_{IMAGE_PATH.stem}.png"

    plt.figure(figsize=(4,4))
    plt.imshow(img_np)
    plt.imshow(cam, cmap="inferno", alpha=0.75)
    plt.axis("off")

    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

print("Done.")