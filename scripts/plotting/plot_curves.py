import json
import matplotlib.pyplot as plt

#with open("data/3_trans_dinov3_1imgs_loss_seq_to_one/training_losses.json") as f:
#with open("data/3_trans_dinov2_1imgs_loss_seq_to_one/training_losses.json") as f:



#new

#with open("data/3_lstm_dinov2_1imgs_loss/mlp_run_0/training_losses.json") as f:
#with open("data/3_lstm_dinov3_1imgs_loss/mlp_run_0/training_losses.json") as f:
#with open("data/4_trans_dinov2_6imgs_loss_seq_to_seq/training_losses.json") as f:
#with open("data/4_trans_dinov3_6imgs_loss_seq_to_seq/training_losses.json") as f:
#with open("data/3_mlp_dinov2_6imgs/mlp_run_0/training_losses.json") as f:
#with open("data/3_mlp_dinov3_6imgs//mlp_run_0/training_losses.json") as f:

#with open("data/3z_flex_lstm_dinov2_6imgs/cnn_run_0/training_losses.json") as f:
#with open("data/3z_flex_lstm_dinov3_6imgs/cnn_run_0/training_losses.json") as f:
#with open("data/3z_flex_mlp_dinov2_6imgs/cnn_run_0/training_losses.json") as f:
#with open("data/3z_flex_mlp_dinov3_6imgs/cnn_run_0/training_losses.json") as f:

#with open("data/3z_flex_lstm_resnet18_6imgs/cnn_run_0/training_losses.json") as f:
#with open("data/3z_flex_lstm_resnet50_6imgs/cnn_run_0/training_losses.json") as f:
#with open("data/3z_flex_mlp_resnet18_6imgs/cnn_run_0/training_losses.json") as f:
#with open("data/3z_flex_mlp_resnet50_6imgs/cnn_run_0/training_losses.json") as f:

with open("data/5_trans_dinov2_6imgs_loss_seq_to_seq/training_losses.json") as f:


#neu: 
#data/3_mlp_dinov2_6imgs
#data/3_mlp_dinov3_6imgs
#data/3z_flex_lstm_dinov2_6imgs
#data/3z_flex_lstm_dinov3_6imgs
#data/3z_flex_lstm_dinov2_6imgs
#data/3z_flex_mlp_dinov2_6imgs
#data/3z_flex_mlp_dinov3_6imgs


    data = json.load(f)

plt.figure(figsize=(6,4))

plt.plot(data["train_loss"][2:800], label="Train", linewidth=2)
plt.plot(data["val_loss"][2:800], label="Validation", color="darkred", linewidth=2)

plt.xlabel("Epoch", fontsize=20)
plt.ylabel("Loss", fontsize=20)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.ylim(0,800)

plt.legend(fontsize=15)

plt.tight_layout()

#plt.savefig("data/3_lstm_dinov2_1imgs_loss/mlp_run_0/training_curve.png", dpi=900)
#plt.savefig("data/3_lstm_dinov3_1imgs_loss/mlp_run_0/training_curve.png", dpi=900)
#plt.savefig("data/4_trans_dinov2_6imgs_loss_seq_to_seq/training_curve.png", dpi=900)
#plt.savefig("data/4_trans_dinov3_6imgs_loss_seq_to_seq/training_curve.png", dpi=900)
#plt.savefig("data/3_mlp_dinov2_6imgs/mlp_run_0/training_curve.png", dpi=900)
#plt.savefig("data/3_mlp_dinov3_6imgs/mlp_run_0/training_curve.png", dpi=900)

#plt.savefig("data/3z_flex_lstm_dinov2_6imgs/cnn_run_0/training_curve.png", dpi=900)
#plt.savefig("data/3z_flex_lstm_dinov3_6imgs/cnn_run_0/training_curve.png", dpi=900)
#plt.savefig("data/3z_flex_mlp_dinov2_6imgs/cnn_run_0/training_curve.png", dpi=900)
#plt.savefig("data/3z_flex_mlp_dinov3_6imgs/cnn_run_0/training_curve.png", dpi=900)

#plt.savefig("data/3z_flex_lstm_resnet18_6imgs/cnn_run_0/training_curve.png", dpi=900)
#plt.savefig("data/3z_flex_lstm_resnet50_6imgs/cnn_run_0/training_curve.png", dpi=900)
#plt.savefig("data/3z_flex_mlp_resnet18_6imgs/cnn_run_0/training_curve.png", dpi=900)
#plt.savefig("data/3z_flex_mlp_resnet50_6imgs/cnn_run_0/training_curve.png", dpi=900)

plt.savefig("data/5_trans_dinov2_6imgs_loss_seq_to_seq/training_curve.png", dpi=900)




#neu: 
#data/3_mlp_dinov2_6imgs
#data/3_mlp_dinov3_6imgs
#data/3z_flex_lstm_dinov2_6imgs
#data/3z_flex_lstm_dinov3_6imgs
#data/3z_flex_lstm_dinov2_6imgs
#data/3z_flex_mlp_dinov2_6imgs
#data/3z_flex_mlp_dinov3_6imgs




#plt.savefig("data/3_trans_dinov2_1imgs_loss_seq_to_one/training_curve.png", dpi=900)
#plt.savefig("data/3_trans_dinov3_1imgs_loss_seq_to_one/training_curve.png", dpi=900)
