import torch
from supnerf.trainer import SUPNeRFTrainer

config = {
    "data_root": "data/wheat_sample",
    "save_dir": "results/wheat_sample",
    "epochs": 50,
    "batch_size": 1,
    "img_size": 224,
    "use_pose_refinement": False,
    "use_scale_refinement": True,
    "lr": 5e-4,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

if __name__ == "__main__":
    trainer = SUPNeRFTrainer(config)
    trainer.train()