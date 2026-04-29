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
# project root
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
MODE = "gradcam++" #scorecam / gradcam++
IMAGE_FOLDER = "z_field_example_choosen"
SAVE_FOLDER = "z_heatmaps_new"

#CHECKPOINT_PATH = "data/1_resnet50_1imgs_mlp_field_images/cnn_run_0/model_19.pth" #not 
CHECKPOINT_PATH = "data/1_a_field_fine_tuned_mlp_resnet50/cnn_run_0/model_6.pth"

BACKBONE = "resnet50"
image_prefix = "mlp_resnet50_fine_tuned_overlay"


#CHECKPOINT_PATH = "data/1_resnet18_1imgs_mlp_field_images/cnn_run_0/model_25.pth" #not
#CHECKPOINT_PATH = "data/1_a_field_fine_tuned_mlp_resnet18/cnn_run_0/model_5.pth"


#BACKBONE = "resnet18"
#image_prefix = "mlp_resnet18_fine_tuned_overlay"

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

print("Model loaded.")

# -------------------------------------------------------
# TRANSFORMS
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

gradients = []
feature_maps = []

def forward_hook(module, input, output):
    feature_maps.append(output.detach())
def backward_hook(module, grad_input, grad_output):
    gradients.append(grad_output[0].detach())


target_layers = [
    model.backbone.layer2,
    model.backbone.layer3,
    model.backbone.layer4

]
for layer in target_layers:
    layer.register_forward_hook(forward_hook)
    layer.register_full_backward_hook(backward_hook)

# -------------------------------------------------------
# PROCESS IMAGES
# -------------------------------------------------------
image_paths = sorted(Path(IMAGE_FOLDER).glob("*"))

for IMAGE_PATH in image_paths:

    print("Processing:", IMAGE_PATH.name)

    feature_maps.clear()
    gradients.clear()

    # load image
    torch_img = read_image(str(IMAGE_PATH)).float() / 255.0
    img_geom = geom_transform(torch_img)
    img_model = norm_transform(img_geom)

    img_tensor = img_model.unsqueeze(0).unsqueeze(0).to(DEVICE)
    img_np = img_geom.permute(1,2,0).numpy()

    mask = torch.zeros((1,1), dtype=torch.bool).to(DEVICE)

    # forward once to get feature maps
    
    baseline_output = model(img_tensor, mask).sum()

    fmap2 = feature_maps[0]
    fmap3 = feature_maps[1]
    fmap4 = feature_maps[2]

         # (1, C, 7, 7)
    print("Layer3 feature map:", fmap3.shape)
    print("Layer4 feature map:", fmap4.shape)

    # -------------------------------------------------------
    # DEBUG: visualize individual feature channels
    # -------------------------------------------------------
    #debug_dir = Path(SAVE_FOLDER) / "debug_channels"
    #debug_dir.mkdir(exist_ok=True)

    #for i in range(10):

        #fmap_i = fmap[0, i].cpu().numpy()

        #fmap_i = (fmap_i - fmap_i.min()) / (fmap_i.max() - fmap_i.min() + 1e-8)
        #fmap_i = cv2.resize(fmap_i, (224,224))

        #plt.figure(figsize=(3,3))
        #plt.imshow(img_np)
        #plt.imshow(fmap_i, cmap="inferno", alpha=0.6)
        #plt.axis("off")
        #plt.savefig(debug_dir / f"{IMAGE_PATH.stem}_channel{i}.png", dpi=200, bbox_inches="tight")
        #plt.close()

    max_channels = 128

    # feature map used for Score-CAM
    fmap = fmap4
    B, C, H, W = fmap.shape
    scorecam = torch.zeros((H, W), device=DEVICE)

    # -------------------------------------------------------
    # SCORE-CAM
    # -------------------------------------------------------
    if MODE == "scorecam":

        for c in range(min(C, max_channels)):

            activation = fmap[0, c]

            act = activation.clone()
            act = (act - act.min()) / (act.max() - act.min() + 1e-8)

            act_up = torch.nn.functional.interpolate(
                act.unsqueeze(0).unsqueeze(0),
                size=(224,224),
                mode="bilinear",
                align_corners=False
            )[0,0]

            masked = img_tensor.clone()
            masked[:,:,0] *= act_up
            masked[:,:,1] *= act_up
            masked[:,:,2] *= act_up

            with torch.no_grad():
                score = model(masked, mask).mean()

            weight = torch.relu(score - baseline_output).item()
            print(f"channel {c} weight: {weight}")

            scorecam += weight * activation

        scorecam = torch.relu(scorecam)
        scorecam = scorecam.cpu().numpy()

        scorecam = cv2.resize(scorecam, (224,224), interpolation=cv2.INTER_NEAREST)
        scorecam = scorecam / (scorecam.max() + 1e-8)
    
    elif MODE == "gradcam++":

        model.zero_grad()
        baseline_output.backward()


        grad4 = gradients[0]
        grad3 = gradients[1]
        grad2 = gradients[2]

        # -----------------------------
        # GradCAM++ for layer3
        # -----------------------------

        # ----- layer2 -----
        weights2 = grad2.mean(dim=(2,3))
        cam2 = (weights2[0,:,None,None] * fmap2[0]).sum(0)
        cam2 = torch.relu(cam2).detach().cpu().numpy()

        # ----- layer3 -----
        weights3 = grad3.mean(dim=(2,3))
        cam3 = (weights3[0,:,None,None] * fmap3[0]).sum(0)
        cam3 = torch.relu(cam3).detach().cpu().numpy()

        # ----- layer4 -----
        weights4 = grad4.mean(dim=(2,3))
        cam4 = (weights4[0,:,None,None] * fmap4[0]).sum(0)
        cam4 = torch.relu(cam4).detach().cpu().numpy()


        # -----------------------------
        # Resize CAMs to image size
        # -----------------------------
        if BACKBONE == "Resnet18":
            cam3 = cv2.resize(cam3, (224,224), interpolation=cv2.INTER_CUBIC)
            cam4 = cv2.resize(cam4, (224,224), interpolation=cv2.INTER_CUBIC)
        else: 
            cam3 = cv2.resize(cam3, (224,224), interpolation=cv2.INTER_LINEAR)
            cam4 = cv2.resize(cam4, (224,224), interpolation=cv2.INTER_LINEAR)
            cam2 = cv2.resize(cam2, (224,224), interpolation=cv2.INTER_LINEAR)
            cam2 = cam2 / (cam2.max() + 1e-8)


        cam3 = cam3 / (cam3.max() + 1e-8)
        cam4 = cam4 / (cam4.max() + 1e-8)

        # -----------------------------
        # Fuse layer3 + layer4
        # -----------------------------
        if BACKBONE == "Resnet18":
            scorecam = 0.8 * cam3 + 0.2 * cam4
            #scorecam = cam3 * cam4
            scorecam = scorecam / (scorecam.max() + 1e-8)
        else: 
            scorecam = 0.8 * cam3 + 0.2 * cam4
            scorecam = scorecam / (scorecam.max() + 1e-8)       


    # -------------------------------------------------------
    # PLOT
    # -------------------------------------------------------
    save_path = Path(SAVE_FOLDER) / f"{image_prefix}_{Path(IMAGE_PATH).stem}.png"


    plt.figure(figsize=(4,4))
    plt.imshow(img_np)
    plt.imshow(scorecam, cmap="inferno", alpha=0.75)
    plt.axis("off")
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()

print("Done.")