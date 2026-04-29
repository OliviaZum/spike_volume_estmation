import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from torchvision.io import read_image
from pathlib import Path
import sys
import os
import torch.nn.functional as F

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
SAVE_FOLDER = "z_heatmaps_new"

#CHECKPOINT_PATH = "data/1_a_field_fine_tuned_mlp_dinov3/cnn_run_0/model_6.pth"
CHECKPOINT_PATH = "data/2_fine-tuned-mlp/1_dinov3-1imgs_mlp_field_images/cnn_run_0/model_34.pth" #not

image_prefix = "mlp_dinov3_not_fine_tuned_overlay"

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
# hook to capture tokens
# -------------------------------------------------------



tokens_list = []
grads_list = []

def token_hook(module, inp, out):
    tokens_list.append(out)
    out.register_hook(lambda g: grads_list.append(g))

# layers to visualize
layers = [ -1, -2, -3]

for l in layers:
    model.backbone.blocks[l].register_forward_hook(token_hook)

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

    print("Processing:", IMAGE_PATH)

    tokens_list = []
    grads_list = []

    torch_img = read_image(str(IMAGE_PATH)).float() / 255.0

    img_geom = geom_transform(torch_img)
    img_model = norm_transform(img_geom)

    img_tensor = img_model.unsqueeze(0).unsqueeze(0).to(DEVICE)

    img_np = img_geom.permute(1,2,0).numpy()

    mask = torch.zeros((1,1), dtype=torch.bool).to(DEVICE)

    output = model(img_tensor, mask)
    score = output.squeeze()

    model.zero_grad()
    score.backward()

    # -------------------------------------------------------
    # compute similarity for each layer
    # -------------------------------------------------------


    heatmaps = []

    for tokens, grads in zip(tokens_list, grads_list):

        patch_tokens = tokens[:,5:,:]      # remove CLS + registers
        patch_grads  = grads[:,5:,:]

        cam = (patch_tokens * patch_grads).sum(dim=-1)

        heatmaps.append(cam)

    similarity = torch.stack(heatmaps).mean(0)
    similarity = similarity.clamp(min=0)

    # -------------------------------------------------------
    # sharpen similarity
    # -------------------------------------------------------


    heatmap = similarity.reshape(14,14).detach().cpu().numpy()

    # normalize
    p_low = np.percentile(heatmap, 60)
    p_high = np.percentile(heatmap, 99)

    heatmap = np.clip((heatmap - p_low) / (p_high - p_low + 1e-8), 0, 1)

    # resize
    heatmap = cv2.resize(
        heatmap,
        (224,224),
        interpolation=cv2.INTER_CUBIC
    )

    heatmap = cv2.GaussianBlur(heatmap,(5,5),0)

    print("Heatmap checksum:", float(heatmap.sum()))

    save_path = Path(SAVE_FOLDER) / f"{image_prefix}_{Path(IMAGE_PATH).stem}.png"


    plt.figure(figsize=(4,4))
    plt.imshow(img_np)
    plt.imshow(
        heatmap,
        cmap="inferno",
        alpha=0.75
    )
    plt.axis("off")

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()