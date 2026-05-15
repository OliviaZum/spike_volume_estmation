import click 
import numpy as np
import model_lib.neural_nets as neural_nets
import model_lib.training as training
import torch.nn as nn
import torch
from torchvision.transforms import v2
from icecream import ic
from model_lib.inferrer import SingleImageExperimentCreator, MultiImageExperimentCreator
from pathlib import Path
from accelerate.utils import set_seed
import torchvision.models as models
import timm


def volume_estimation(dataset_path: str | Path, output_path: str | Path, mapping_file: str | Path): 
    """
    Trains an MLP/LSTM/Transformer/CNN to predict volume of wheat spikes. 
     
    Args: 
        dataset_path: Path to image dataset
        output_path: Path to output folder 
        mapping_file: Path to mappin file
     
    """
    verbose = True

    # Fix seeds to make all the code reproducible
    set_seed(11397)
    random_split_generator = np.random.default_rng(414) 

    # ------------------------ Pretrained Model ---------------------------#
    #if s =>l s = 384, b: 768, l: 1024, g: 1536, inputsize = 1024, s => b: 7.., g => ......
  
    # ------------------- Fine-Tuning -------------------------
    # resnet18 /resnet50 / dinov2_vits14 / vit_small_patch16_dinov3 / fomo4wheat_base
    backbone_fine_tuning = "vit_small_patch16_dinov3"

    # ------------------- Vision Model without fine-tuning -------------------------
    pre_train = "dinov3" #resnet18 / dinov2 / dinov3

    # ----------------------------------------------------------
    model_type = "fine_tuning" # "LSTM" / "CNN" / "Transformer" / "MLP" / fine_tuning
    #Go to neural_nets.py and make sure that the correct model is called fine_tuning, and not fine_tuning_notused
    print(model_type)

    if model_type in ["LSTM", "Transformer", "MLP"]:
        if pre_train == "dinov2": 
            print("Using DinoV2")
            pretrained_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14") 
        elif pre_train == "dinov3": 
            print("Using DinoV3")
            pretrained_model = timm.create_model("vit_small_patch16_dinov3", pretrained=True)
            pretrained_model.reset_classifier(0)
        elif pre_train == "resnet18":
            print("Using Resnet18")
            pretrained_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            pretrained_model.fc = torch.nn.Identity() 

    use_checkpoint = False
    if use_checkpoint:
        print("Checkpoints used")
        #checkpoint_path = "data/1_resnet50_1imgs_mlp_field_images/cnn_run_0/model_19.pth"
        checkpoint_path = "data/1_resnet50_1imgs_mlp_field_images/cnn_run_0/model_19.pth"
    else: 
        checkpoint_path=None
    
    # ------------------------ Image Transformations ---------------------------#
    model_transform = [v2.Resize(size=256, antialias=True),
                       v2.CenterCrop(224),
                       v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
    
    random_rotations = False
    value_column = 'volume'
    #mean and std of training set, used to standardize the volumes for stable training
    Y_MEAN = 4769.53
    Y_STD = 1235.42

    #Fine-tune field images
    #Y_MEAN = 4714
    #Y_STD = 1111

    # ------------------------ Training Parameters ---------------------------#
   
    
   #MLP and LSTM setup
    mlp_num_hidden_layers = [3]
    mlp_hidden_size = 2

    if model_type in ["LSTM", "Transformer", "MLP"]:
        if pre_train == "resnet18":
            mlp_input_size = 512
            mlp_input_size = 512
            lstm_input_size = 512
            transformer_input_size = 512
        elif pre_train == "dinov2": 
            mlp_input_size = 384
            lstm_input_size = 384
            transformer_input_size = 384
        elif pre_train == "dinov3":
            print("size dinov3")
            mlp_input_size = 384
            lstm_input_size = 384
            transformer_input_size = 384
    else:
        print("Using: ")
        print(backbone_fine_tuning)

    lstm_hidden_size = 512
    lstm_layers = 3

    #general setup
    num_epochs = 5000
    num_worker = 16
    batch_size = 32
  
    #max_sequence_len: number of training images
    #eval_max_sequence_len: number of images in evaluation
    #if random_sequence_len = True, a random number of images between min and max is choosen
    #if eval_random_sequence_len = True, a random number of images between
    #min and data/1_mlp_basic_dinov2_1imgsmax is chosen for inference. 
    #if eval_random_sequence_len = False, the max number is chosen
    #if (eval_)random_choice = False, then the first images of each plant are chosen in training or evaluation
    #if (eval_)random_choice = True, the images of each plants are choosen randomly in training or evaluation
    min_sequence_len = 6
    max_sequence_len = 6
    random_sequence_len = False
    random_choice = True #true. Model should see all the images. 
    print("random choice")
    print(random_choice)

    eval_max_sequence_len = 6
    eval_min_sequence_len = 6
    print("eval sequence len")
    print(eval_min_sequence_len)
    eval_random_sequence_len = False #false, for comparison with baselines 
    eval_random_choice = False #false, for comparison with baselines

    drop_out_rates = [0.1] # transformer
    #drop_out_rates = [0.5] #lstm / MLP
    activations = [nn.CELU]
    normalization = nn.LayerNorm
    scaled_loss = True
    #only for lstm / transformer
    last_time_step = True
    print("last_time_step")
    print(last_time_step)

    def_loss_function = torch.nn.MSELoss()

    # scaled loss for normally distributed data
    class ScaledMSELoss(nn.Module):
        def __init__(self, reduction='mean'):
            super(ScaledMSELoss, self).__init__()
            self.reduction = reduction

        def forward(self, input, target, weight):
            loss = (input - target) ** 2 * weight
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        

    # ------------------------ Evaluation Parameters ---------------------------#
    #Prediction based in single images ("single") or multiple images ("multiple")
    num_of_img_inference = "multiple"

    if num_of_img_inference == "single": 
        experiment_creator = SingleImageExperimentCreator(ignore_artificial=True)
    else: 
        experiment_creator = MultiImageExperimentCreator(ignore_artificial=True, 
                                                             eval_min_seq_len=eval_min_sequence_len, 
                                                             eval_max_seq_len=eval_max_sequence_len,
                                                             eval_random_seq_len=eval_random_sequence_len, 
                                                             random_choice=eval_random_choice)
    

    # ---------------------------------- Different Models -----------------------------------#


    # ---------------------------------------------------#
    # -------------------- MLP setup --------------------#
    # ---------------------------------------------------#
    if model_type == "MLP":
        if scaled_loss: 
            loss_function = ScaledMSELoss()
        else: 
            loss_function=def_loss_function

        cnt = 0
        for activation in activations:
            for dropout_rate in drop_out_rates: 
                for layers in mlp_num_hidden_layers:

                    model_params_mlp_ah = {
                            'num_layer' : layers,
                            'activation' : activation,
                            #'input_size' : mlp_input_size * max_sequence_len,
                            'input_size' : mlp_input_size,
                            'hidden_div' : mlp_hidden_size,
                            'output_size' : 1,
                            'normalize' : normalization,
                            'dropout_rate' : dropout_rate,
                            }
                    mlp_model = neural_nets.MLP(**model_params_mlp_ah)
                        

                    # ---- Training ----- #
                    optimizer_params = {
                        'lr' : 0.0001,
                    }
                    optimizer = torch.optim.Adam(mlp_model.parameters(), **optimizer_params)

                    scheduler_params = {
                        'start_factor' : 1.0,
                        'end_factor' : 0.001,
                        'total_iters' : num_epochs,
                    }
                    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_params)
  
                    
                    training.train_run(
                        dataset_path=dataset_path,
                        pretrained_transform=model_transform, 
                        output_path=Path(output_path) / f"mlp_run_{cnt}",
                        mapping_file=mapping_file,
                        value_column=value_column,
                        random_split_generator=random_split_generator,
                        #augmentation_factor=augmentation_factor, why?
                        image_rotations=random_rotations,
                        Y_MEAN = Y_MEAN,
                        Y_STD = Y_STD,
                        pretrained_model=pretrained_model,
                        min_sequence_length=max_sequence_len, # same as max seqence deliberately, why?
                        max_sequence_length=max_sequence_len,
                        random_sequence_length=random_sequence_len,
                        random_choice = random_choice,
                        model = mlp_model,
                        model_params=model_params_mlp_ah,
                        optimizer=optimizer,
                        optimizer_params=optimizer_params,
                        scheduler=scheduler,
                        scheduler_params=scheduler_params,
                        loss_function=loss_function,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        num_worker=num_worker,
                        experiment_creator=experiment_creator,
                        verbose=verbose, 
                        scaled_loss = scaled_loss, 
                        last_time_step=last_time_step, 
                        checkpoint_path=checkpoint_path
                    )

                    cnt += 1
                    
    # ---------------------------------------------------#
    # -------------------- CNN setup --------------------#
    # ---------------------------------------------------#

    if model_type == "fine_tuning":
        if scaled_loss: 
            loss_function = ScaledMSELoss()
        else: 
            loss_function=def_loss_function

        cnt = 0
        model_params_fine_tuning = {
            "backbone_name": backbone_fine_tuning,
            "pretrained": True,
            "out_dim": 128,
        }

        fine_tuning_model = neural_nets.fine_tuning(**model_params_fine_tuning)
        fine_tuning_model.freeze_backbone_layers()


        optimizer_params = {
            'lr' : 0.0001,
        }
        optimizer = torch.optim.Adam(fine_tuning_model.parameters(), **optimizer_params)
        

        scheduler_params = {
            'start_factor' : 1,
            'end_factor' : 0.001,
            'total_iters' : num_epochs,
        }
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_params)
        
        training.train_run(
            dataset_path=dataset_path,
            pretrained_transform=model_transform, 
            output_path=Path(output_path) / f"ft_run_{cnt}",
            mapping_file=mapping_file,
            value_column=value_column,
            random_split_generator=random_split_generator,
            image_rotations=random_rotations,
            Y_MEAN = Y_MEAN,
            Y_STD = Y_STD,
            pretrained_model=None,
            min_sequence_length=min_sequence_len, 
            max_sequence_length=max_sequence_len,
            random_sequence_length=random_sequence_len,
            random_choice = random_choice,
            model = fine_tuning_model,
            model_params=model_params_fine_tuning,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            loss_function=loss_function,
            num_epochs=num_epochs,
            batch_size=batch_size,
            num_worker=num_worker,
            experiment_creator=experiment_creator,
            verbose=verbose, 
            scaled_loss = scaled_loss, 
            last_time_step=last_time_step, 
            checkpoint_path=checkpoint_path
        )
        cnt += 1


    # ---------------------------------------------------#
    # -------------------- LSTM setup -------------------#
    # ---------------------------------------------------#
    if model_type == "LSTM":
        if scaled_loss: 
            loss_function = ScaledMSELoss()
        else: 
            loss_function=def_loss_function

        cnt = 0
        for activation in activations:
            for dropout_rate in drop_out_rates: 

                    model_params_LSTM_fp = {
                            'num_layers' : lstm_layers,
                            'activation' : activation,
                            'input_size' : lstm_input_size,
                            'hidden_div' : lstm_hidden_size,
                            'output_size' : 1,
                            'normalize' : normalization,
                            'dropout_rate' : dropout_rate,
                            }
                    LSTM_model = neural_nets.LSTM(**model_params_LSTM_fp)    

                    optimizer_params = {
                        'lr' : 0.0001,
                    }
                    optimizer = torch.optim.Adam(LSTM_model.parameters(), **optimizer_params)
                    

                    scheduler_params = {
                        'start_factor' : 1,
                        'end_factor' : 0.001,
                        'total_iters' : num_epochs,
                    }
                    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_params)
                    
                    training.train_run(
                        dataset_path=dataset_path,
                        pretrained_transform=model_transform, 
                        output_path=Path(output_path) / f"mlp_run_{cnt}",
                        mapping_file=mapping_file,
                        value_column=value_column,
                        random_split_generator=random_split_generator,
                        image_rotations=random_rotations,
                        Y_MEAN = Y_MEAN,
                        Y_STD = Y_STD,
                        pretrained_model=pretrained_model,
                        min_sequence_length=min_sequence_len, 
                        max_sequence_length=max_sequence_len,
                        random_sequence_length=random_sequence_len,
                        random_choice = random_choice,
                        model = LSTM_model,
                        model_params=model_params_LSTM_fp,
                        optimizer=optimizer,
                        optimizer_params=optimizer_params,
                        scheduler=scheduler,
                        scheduler_params=scheduler_params,
                        loss_function=loss_function,
                        num_epochs=num_epochs,
                        batch_size=batch_size,
                        num_worker=num_worker,
                        experiment_creator=experiment_creator,
                        verbose=verbose, 
                        scaled_loss = scaled_loss, 
                        last_time_step=last_time_step, 
                        checkpoint_path=checkpoint_path
                    )
                    cnt += 1


    # ---------------------------------------------------#
    # -------------- TRANSFORMER setup ------------------#
    # ---------------------------------------------------#

    #delete RandomHorizontalFlip(p=0.5) and RandomVerticalFlip(p=0.5) in data_sets.py

    if model_type == "Transformer":
        if scaled_loss: 
            loss_function = ScaledMSELoss()
        else: 
            loss_function=def_loss_function
        
        cnt = 0
        for activation in activations:
            for dropout_rate in drop_out_rates: 
                attention_params = {
                            'dropout_rate' : dropout_rate,
                            'input_size' : transformer_input_size
                            }

                attention_model = neural_nets.AttentionBased(**attention_params)

                optimizer_params = {
                    'lr' : 0.0001, #10^-3 /10^-4 #standard: 0.001
                }
                optimizer = torch.optim.AdamW(attention_model.parameters(), **optimizer_params)

                scheduler_params = {
                    'start_factor' : 1,
                    'end_factor' : 0.001,
                    'total_iters' : num_epochs,
                }
                scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_params)

                training.train_run(
                    dataset_path=dataset_path,
                    pretrained_transform=model_transform,
                    output_path=Path(output_path),
                    mapping_file=mapping_file,
                    value_column=value_column,
                    random_split_generator=random_split_generator, 
                    image_rotations=random_rotations,
                    Y_MEAN = Y_MEAN,
                    Y_STD = Y_STD,
                    pretrained_model=pretrained_model,
                    min_sequence_length=min_sequence_len,
                    max_sequence_length=max_sequence_len,
                    random_sequence_length=random_sequence_len,
                    random_choice = random_choice,
                    model = attention_model,
                    model_params=attention_params,
                    optimizer=optimizer,
                    optimizer_params=optimizer_params,
                    scheduler=scheduler,
                    scheduler_params=scheduler_params,
                    loss_function=loss_function,
                    num_epochs=num_epochs,
                    batch_size=batch_size,
                    num_worker=num_worker,
                    experiment_creator=experiment_creator,
                    verbose=verbose, 
                    scaled_loss = scaled_loss, 
                    last_time_step=last_time_step, 
                    checkpoint_path=checkpoint_path
                )
       



@click.command()
@click.option("--dataset-path", type=str, required=True, help="Path to the dataset, required to contain the images and a mapping file.")
@click.option("--output-path", type=str, required=True, help="Path to output the computed results.")
@click.option("--mapping-file", type=str, required = True, help = "Path to mapping file" )
def main_training(dataset_path : str | Path,
                  output_path : str | Path,
                  mapping_file : str | Path):
     
    
    volume_estimation(dataset_path=dataset_path, 
                output_path=output_path, 
                mapping_file=mapping_file)


if __name__ == "__main__":
    main_training()
    
   
