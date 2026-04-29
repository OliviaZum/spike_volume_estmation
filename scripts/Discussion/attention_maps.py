import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from PIL import Image
import sys
from pathlib import Path
from torchvision.io import read_image
import os
os.environ["XFORMERS_DISABLED"] = "1"

# Go up until we find the project root (folder containing "src")
current = Path(__file__).resolve()

for parent in current.parents:
    if (parent / "src").exists():
        sys.path.append(str(parent / "src"))
        break
else:
    raise RuntimeError("Could not find project root containing 'src' folder.")

# ---- IMPORT YOUR MODEL CLASS ----
from model_lib.neural_nets import fine_tuning

##############################################################
#Load trained model
#CHECKPOINT_PATH = "data/1_a_field_fine_tuned_lstm_dinov3/cnn_run_0/model_3.pth"
IMAGE_FOLDER = "z_field_example_choosen"
mode = "rollout_grad" 

CHECKPOINT_PATH = "data/2_fine-tuned-mlp/1_dinov2-1imgs_mlp_field_images/cnn_run_0/model_19.pth" #not
#CHECKPOINT_PATH = "data/1_a_field_fine_tuned_mlp_dinov2/cnn_run_0/model_4.pth"

SAVE_FOLDER = "z_heatmaps_new"
backbone_model_name = "dinov2_vits14" #vit_small_patch16_dinov3 / dinov2_vits14
image_prefix = "mlp_dinov2_not_fine_tuned_overlay"
saved_qkv = []
# "rollout" / "gradient" / "rollout_grad"
#
############################################################

image_paths = sorted(
    list(Path(IMAGE_FOLDER).glob("*.jpg")) +
    list(Path(IMAGE_FOLDER).glob("*.png")) +
    list(Path(IMAGE_FOLDER).glob("*.jpeg"))
)
Path(SAVE_FOLDER).mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load model
model = fine_tuning(backbone_name=backbone_model_name, 
                    pretrained=True,
                    out_dim=128)


checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model_pretrained = fine_tuning(
    backbone_name=backbone_model_name,
    pretrained=True,
    out_dim=128
).to(DEVICE)

model.to(DEVICE)
model_pretrained.to(DEVICE)

print("\n===== TRUE BACKBONE DIFFERENCE =====")

diff_sum = 0
param_count = 0

for (name_pre, p_pre), (name_ft, p_ft) in zip(
    model_pretrained.backbone.named_parameters(),
    model.backbone.named_parameters()
):
    assert name_pre == name_ft
    diff = torch.mean(torch.abs(p_pre - p_ft)).item()
    diff_sum += diff
    param_count += 1

print("Mean backbone parameter change:", diff_sum / param_count)

print("\n===== LAST BLOCK DIFFERENCE =====")

diff_sum = 0
param_count = 0

for (name_pre, p_pre), (name_ft, p_ft) in zip(
    model_pretrained.backbone.blocks[-1].named_parameters(),
    model.backbone.blocks[-1].named_parameters()
):
    diff = torch.mean(torch.abs(p_pre - p_ft)).item()
    print(name_ft, diff)

    diff_sum += diff
    param_count += 1

print("Mean last block change:", diff_sum / param_count)




for name, p in model.named_parameters():
    if "backbone" not in name:
        print(name, p.abs().sum().item())

print(sum(
    p.abs().sum().item()
    for p in model.backbone.blocks[-1].parameters()
))

print("Checkpoint keys:", checkpoint.keys())
print(type(model.backbone))

model.to(DEVICE)
if mode == "rollout_grad":
    model.train()
else:
    model.eval()

print("Model loaded sucessfully.")

geom_transform = torch.nn.Sequential(
    v2.Resize(256, antialias=True),
    v2.CenterCrop(224),
)
# create normalized tensor for model
norm_transform = v2.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

if mode == "rollout":
    #Register attention hooks
    attentions = []

    def make_hook(attn_module):
        def hook(module, input, output):
            # output: (B, N, 3*dim)
            B, N, C = output.shape
            dim = C // 3

            qkv = output.reshape(B, N, 3, attn_module.num_heads, dim // attn_module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)

            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) / np.sqrt(q.shape[-1])
            
            attn = torch.softmax(attn, dim=-1)
            print("ATTN checksum:", attn[0,0].sum().item())

            attentions.append(attn.detach())

        return hook

    # register hooks
    for blk in model.backbone.blocks:
        blk.attn.qkv.register_forward_hook(make_hook(blk.attn))

elif mode == "rollout_grad":
    attentions = []
    grads = []

    if backbone_model_name == "dinov2_vits14":

        def save_attn(module, input, output):
            attn = input[0]           # attention before dropout
            attn.retain_grad()

            attentions.append(attn)

            def save_grad(grad):
                grads.append(grad)

            attn.register_hook(save_grad)
        for blk in model.backbone.blocks:
            blk.attn.attn_drop.register_forward_hook(save_attn)

    elif backbone_model_name == "vit_small_patch16_dinov3":

        print("dinov3")

        

        def qkv_hook(module, input, output):

            print("\n--- QKV HOOK ---")
            print("Layer:", len(saved_qkv))
            print("Output shape:", output.shape)

            B, N, C = output.shape
            print("Batch:", B)
            print("Tokens:", N)
            print("Channels:", C)
            print("Expected tokens for DINOv3 small: 201 (CLS + 4 register + 196 patches)", N)

            saved_qkv.append(output.detach())
        
        #def qkv_hook(module, input, output):
        #    print("QKV HOOK SHAPE:", output.shape)
        #    print("QKV SAMPLE:", output[0,0,:10].detach().cpu())
        #    saved_qkv.append(output)

        def proj_hook(module, input, output):
            grad_tensor = input[0]
            grad_tensor.retain_grad()

            def save_grad(grad):
                grads.append(grad)

            grad_tensor.register_hook(save_grad)

        for blk in model.backbone.blocks:
            blk.attn.qkv.register_forward_hook(qkv_hook)
            blk.attn.proj.register_forward_hook(proj_hook)

        

#load image: 
#img_pil = Image.open(IMAGE_PATH).convert("RGB")
#img_tensor = transform(img_pil).unsqueeze(0).unsqueeze(0).to(DEVICE)  
# shape: (B=1, T=1, C, H, W)
    # exact geometric transform (same as dataset, without random flips)



# ---- LOAD IMAGE EXACTLY LIKE DATASET ----

for IMAGE_PATH in image_paths:
    print("Processing:", IMAGE_PATH)

    torch_img = read_image(str(IMAGE_PATH)).float() / 255.0
    img_geom = geom_transform(torch_img)
    img_model = norm_transform(img_geom)

    # model expects (B, T, C, H, W)
    img_tensor = img_model.unsqueeze(0).unsqueeze(0).to(DEVICE)


    # visualization image (H, W, C)
    img_np = img_geom.permute(1, 2, 0).numpy()

    mask = torch.zeros((1,1), dtype=torch.bool).to(DEVICE)

    #forward: 
    # -------------------------

    if mode == "gradient":
        # -------- GRADIENT SALIENCY --------
        model.train()

        img_tensor.requires_grad_(True)

        output = model(img_tensor, mask)
        output = output.squeeze()

        model.zero_grad()
        output.backward()

        grad = img_tensor.grad.detach()

        saliency = grad.abs().mean(dim=2)   # average RGB
        saliency = saliency.squeeze().cpu().numpy()

        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

        mask_map = saliency



    if mode == "rollout":
        attentions.clear()
        with torch.no_grad():
            _ = model(img_tensor, mask)

        print(f"Collected {len(attentions)} attention layers.")

        #attention rollout: 
        def attention_rollout(attentions):
            print("Collected layers:", len(attentions))
            if len(attentions) > 0:
                print("Shape:", attentions[0].shape)
            result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)

            for attn in attentions:
                attn = attn.mean(dim=1)  # average heads
                attn = attn + torch.eye(attn.size(-1)).to(attn.device)
                attn = attn / attn.sum(dim=-1, keepdim=True)
                result = torch.matmul(attn, result)

            return result

        rollout = attention_rollout(attentions)
        # CLS token attention to patches

        if backbone_model_name == "dinov2_vits14":
            cls_attention = rollout[:, 0, 1:]
        
        elif backbone_model_name == "vit_small_patch16_dinov3":

            cls_to_register = rollout[:,0,1:5]
            register_to_patch = rollout[:,1:5,5:]

            cls_to_patches = (cls_to_register.unsqueeze(-1) * register_to_patch).sum(dim=1)

            mask_map = cls_to_patches.reshape(14,14).detach().cpu().numpy()


        

        #reshape to patch grid: 
        num_patches = cls_attention.shape[-1]
        grid_size = int(np.sqrt(num_patches))  # should be 16 for DINOv2 (224/14)

        mask_map = cls_attention.reshape(grid_size, grid_size).cpu().numpy()
        mask_map = cv2.resize(mask_map, (224, 224))
        mask_map = mask_map - mask_map.min()
        mask_map = mask_map / (mask_map.max() + 1e-8)
        

        print("Heatmap checksum:", float(mask_map.sum()))

    elif mode == "rollout_grad":

        attentions.clear()
        grads.clear()
        saved_qkv.clear()

        if backbone_model_name == "dinov2_vits14":
            model.train()   # IMPORTANT


        elif backbone_model_name == "vit_small_patch16_dinov3":
            torch.cuda.empty_cache()
            model.zero_grad(set_to_none=True)
            model.eval()
            

        output = model(img_tensor, mask)
        output = output.squeeze()

        model.zero_grad()
        output.backward()

        
            # reconstruct attention matrices for DINOv3
        if backbone_model_name == "vit_small_patch16_dinov3":

            for qkv in saved_qkv:

                B, N, C = qkv.shape
                num_heads = model.backbone.blocks[0].attn.num_heads
                
                embed_dim = model.backbone.embed_dim
                head_dim = (C // 3) // num_heads

                qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()

                q = qkv[0]
                k = qkv[1]
                v = qkv[2]

                print("\n--- QKV SPLIT ---")
                print("Q shape:", q.shape)
                print("K shape:", k.shape)
                print("V shape:", v.shape)

                attn = (q @ k.transpose(-2, -1)) / np.sqrt(head_dim)
                attn = torch.softmax(attn, dim=-1)
                print("ATTN mean:", attn.mean(), "CLS row mean:", attn[:, :, 0].mean())
                print("ATTN checksum:", attn[0,0].sum().item())

                print("\nRAW ATTN CLS->first 20 tokens:")
                print(attn[0,0,0,:20].detach().cpu())

                attentions.append(attn)

        print(f"Collected {len(attentions)} attention layers.")
        print(f"Collected {len(grads)} gradient layers.")

        if backbone_model_name == "vit_small_patch16_dinov3":
            min_len = min(len(attentions), len(grads))
            attentions = attentions[:min_len]
            grads = grads[:min_len]
        elif backbone_model_name == "dinov2_vits14":
            attentions = attentions[-4:]
            grads = grads[-4:]


        def attention_rollout_grad(attentions, grads):

            result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)

            for i in range(len(attentions)):

                attn = attentions[i]
                grad = grads[i]

                if i == 0:
                    print("ATTN SHAPE:", attn.shape)
                    print("GRAD SHAPE:", grad.shape)
                    print("ATTN CLS->tokens first 20:")
                    print(attn[0,0,0,5:25].detach().cpu())

                    print("GRAD token norm first 20:")
                    print(grad.norm(dim=-1)[0,:20].detach().cpu())

                if backbone_model_name == "dinov2_vits14": 
                    attn = attn * torch.relu(grad)
                elif backbone_model_name == "vit_small_patch16_dinov3": 
                    grad_attn = grad.norm(dim=-1)
                    grad_attn = grad_attn.unsqueeze(1).unsqueeze(-1)
                    attn = attn * grad_attn
                    print("attn mean:", attn.mean().item(), "attn std:", attn.std().item())
                    

                attn = attn.mean(dim=1)
                if backbone_model_name == "dinov2_vits14": 
                    attn = attn + torch.eye(attn.size(-1)).to(attn.device)
                    attn = attn / attn.sum(dim=-1, keepdim=True)

                elif backbone_model_name == "vit_small_patch16_dinov3": 
                    attn = attn + torch.eye(attn.size(-1), device=attn.device)
                    attn = attn / attn.sum(dim=-1, keepdim=True)

                result = torch.matmul(attn, result)

            return result


        rollout = attention_rollout_grad(attentions, grads)
        attn_last = attentions[-1].mean(dim=1) 
        
        if backbone_model_name == "vit_small_patch16_dinov3": 
            # remove CLS and register self-loop
            rollout[:, 0, 0] = 0
            rollout = rollout / rollout.sum(dim=-1, keepdim=True)
        print("ROLLOUT SHAPE:", rollout.shape)
        print("CLS attention first 20 tokens:")
        print(rollout[0,0,:20].detach().cpu())
        print("Max token index:", rollout[0,0].argmax().item())

        if backbone_model_name == "dinov2_vits14":
            cls_attention = rollout[:, 0, 1:] 
            num_patches = cls_attention.shape[-1]
            grid_size = int(np.sqrt(num_patches))
            mask_map = cls_attention.reshape(grid_size, grid_size).detach().cpu().numpy()
        elif backbone_model_name == "vit_small_patch16_dinov3":

            cls_to_register = rollout[:,0,1:5]
            register_to_patch = rollout[:,1:5,5:]
            cls_to_patches = (cls_to_register.unsqueeze(-1) * register_to_patch).sum(dim=1)
            mask_map = cls_to_patches.reshape(14,14).detach().cpu().numpy()

        print("MASK MAP SHAPE BEFORE RESIZE:", mask_map.shape)
        mask_map = cv2.resize(mask_map, (224, 224))
        mask_map = mask_map / (mask_map.max() + 1e-8)
        mask_map = cv2.resize(mask_map, (224, 224), interpolation=cv2.INTER_NEAREST)
        print("Heatmap checksum:", float(mask_map.sum()))
        
        print("Heatmap first 10 values:", mask_map.flatten()[:10])


    #plot: 
    #plot:
    #plt.figure(figsize=(6,6))
    #plt.imshow(img_np)   # already correctly aligned
    #plt.imshow(mask_map, cmap='inferno', alpha=0.5)
    #plt.axis("off")
    #plt.tight_layout()
    #plt.savefig(f"lstm_dinov2_attention_{Path(IMAGE_PATH).stem}.png")
    save_path = Path(SAVE_FOLDER) / f"{image_prefix}_{Path(IMAGE_PATH).stem}.png"

    #if backbone_model_name == "dinov2_vits14":

    plt.figure(figsize=(4,4))   # smaller canvas
    plt.imshow(img_np)
    heat = plt.imshow(
        mask_map,
        cmap='inferno',
        alpha=0.75,
        interpolation="nearest"   # IMPORTANT: no blur
    )
    plt.axis("off")
    plt.savefig(save_path,dpi=200, bbox_inches="tight",pad_inches=0)
    plt.close()

    

