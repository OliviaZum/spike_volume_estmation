import torch
import json
import numpy as np
import os
import re

# Load the saved file
saved_data = torch.load("data/1_lstm_dinov2_multi_loss_random_train/mlp_run_0/model_12.pth")
state_dict = saved_data["model_state_dict"]

for key, value in state_dict.items():
    print(f"{key}: {value.shape}")

# Inspect keys in the saved data
print(saved_data.keys())

#dict_keys(['model_class', 'model_state_dict', 'model_params', 'optimizer_class', 
# 'optimizer_state_dict', 'optimizer_params', 'lr_scheduler_class', 'lr_scheduler_state_dict', 'lr_scheduler_params', 'version'])
print("epoch:")
print(saved_data['epoch'])  # Replace 'model_params' with the actual key name if different

print(saved_data['lr_scheduler_state_dict'])

print(saved_data['model_params'])




