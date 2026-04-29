import data_lib.data_sets as data_sets
import data_lib.split_dataset as split_dataset
import lib.stats_lib as stats_lib
import numpy as np
import pandas as pd
import time
import torch
import time 
from torch import nn
from accelerate import Accelerator
from data_lib.data_saver import Saver
from icecream import ic
from model_lib.inferrer import Inferrer, ExperimentCreator
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional
import model_lib.neural_nets as neural_nets
import json




def inverse_freq_weights(labels, num_bins=10, Y_MEAN=0, Y_STD=1):
    device = labels.device 
    true_values = labels * Y_STD + Y_MEAN  # De-normalize
    y_np = true_values.detach().cpu().numpy()
    hist, bin_edges = np.histogram(y_np, bins=num_bins)
    bin_counts = np.maximum(hist, 1)
    bin_indices = np.digitize(y_np, bin_edges[:-1], right=True) - 1
    weights = 1.0 / bin_counts[bin_indices]
    weights = weights / np.max(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_run(dataset_path : Path | str,
              pretrained_transform : list,
              output_path : Path | str,
              mapping_file : Path | str,
              value_column : str,
              random_split_generator : np.random.Generator,
              image_rotations : bool,
              Y_MEAN : float,
              Y_STD : float,
              pretrained_model : torch.nn.Module,
              max_sequence_length : int,
              min_sequence_length : int,
              random_sequence_length : bool,
              random_choice : bool,
              model : torch.nn.Module,
              model_params : Dict[str, any],
              optimizer : torch.optim.Optimizer,
              optimizer_params : Dict[str, any],
              scheduler : Optional[torch.optim.lr_scheduler.LRScheduler],
              scheduler_params : Dict[str, any],
              loss_function : torch.nn.Module,
              num_epochs : int,
              batch_size : int,
              num_worker : int,
              experiment_creator : ExperimentCreator,
              verbose : bool = False, 
              scaled_loss: bool = False, 
              last_time_step: bool = True, 
              checkpoint_path: Optional[str] = None):
    """Compute one training run including training, inference and computation of stats

    Args:
        dataset_path (Path | str): Directory of the dataset
        pretrained_transform (list): Model specific transform settings which should include crop, normalize and resize.
        output_path (Path | str): Directory to store all the output
        mapping_file (Path | str): Csv file containing Image -> Plant / value mapping 
        value_column (str): Column in mapping file to be regarded as value
        random_split_generator (np.random.Generator): Randomness used to create the train / val / test split
        image_rotations (bool): Rotate the images during training in every epoch randomly
        Y_MEAN (float): Mean volume of training dataset 
        Y_STD (float): Standard deviation of volume of training dataset 
        pretrained_model (torch.nn.Module): Pretrained model to extract features from
        max_sequence_length (int): Maximal seqence lenght for one plant
        min_sequence_length (int): Minimal seqence lenght for one plant 
        random_sequence_length (bool): Whether or not to use random seqence lenth
        random_choice (bool): Whether or not to randomly choose images in training 
        model (torch.nn.Module): Model to be trained
        model_params (Dict[str, any]): Model parameters with which the model was initialized
        optimizer (torch.optim.Optimizer): Optimizer which trained the model
        optimizer_params (Dict[str, any]): Initialization parameters for optimizer
        scheduler (torch.optim.lr_scheduler.LRScheduler | None): scheduler to adapt learning rate
        scheduler_params (Dict[str, any]): Parameters for scheduler
        loss_function (torch.nn.Module): Loss function used for training
        num_epochs (int): Number of epochs
        batch_size (int): Batch size
        num_worker (int): Number of workers
        experiment_creator (ExperimentCreator): Experiment creator for inference
        verbose (bool, optional): Defaults to False
        scaled_loss (bool) : Defaults to False, if loss is used that pays attention to volumes further away from mean
        last_time_step (bool): Whether to take only last time step of lstm to calculate the loss       
        checkpoint_path (str): Checkpoint model path in case you want to use one
    """

    # convert to Path variables in case of given strings
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    mapping_file = Path(mapping_file)

    if output_path.exists() and output_path.is_dir() and verbose:
        print(f"Output path {str(output_path)} already existed. Continue...")
    elif output_path.exists() and not output_path.is_dir():
        print(f"Output path {str(output_path)} is not a directory! Aborting...")
        return
    elif verbose:
        print(f"Creating output directory {str(output_path)}.")
    output_path.mkdir(parents=True, exist_ok=True)

    # Check mapping file
    if not mapping_file.exists():
        print(f'Mapping file at {str(mapping_file)} does not exits. Aborting...')
        return
    try:
        df_mapping = pd.read_csv(mapping_file)
    except Exception as e:
        print((f"Error while reading the mapping file. Aborting...\n\n"
               f"Got the following error {str(e)}"))
        return
    
    # compute output paths for mapping files and predictions
    save_mapping_plants = output_path / "mapping_all_plants.json"
    save_mapping_train = output_path / "mapping_train.json"
    save_mapping_val = output_path / "mapping_val.json"
    save_mapping_test = output_path / "mapping_test.json"
    save_predictions_path = output_path / "predictions.json"

    # Saver Class for saving the output data
    saver = Saver(output_path / 'saved_models') 

    # Create directory to store evaluation plots and statistics
    charts_output_path = output_path / "results"
    if charts_output_path.exists() and charts_output_path.is_dir():
        print((f"Output path {str(charts_output_path)} already existed. Continue...\n"
               f"Press ctrl-C to interrupt process if you wish to abort."))
    elif charts_output_path.exists() and not charts_output_path.is_dir():
        print(f"Output path {str(charts_output_path)} is not a directory! Aborting...")
        return
    elif verbose:
        print(f"Creating output directory {str(charts_output_path)}.")
    charts_output_path.mkdir(parents=True, exist_ok=True)

    
    # ------------------ Compute split ------------------#
    if verbose:
        print(f"Computing embeddings split.")
    split_dataset.compute_and_store_split(
        df_mapping=df_mapping,
        mapping_plants_path=save_mapping_plants,
        mapping_train_path=save_mapping_train,
        mapping_val_path=save_mapping_val,
        mapping_test_path=save_mapping_test,
        value_column=value_column,
        random_generator=random_split_generator,
       rel_test_set_size=0.2,
        rel_val_set_size=0.1,
        verbose=verbose,
    )

    # ------------------ Create Dataset ------------------#
    dataset_params = {
        'value_column' : value_column,
        'max_seq_len' : max_sequence_length,
        'min_seq_len' : min_sequence_length,
        'random_seq_len' : random_sequence_length,
        'random_choice' : random_choice,
        'rotate' : image_rotations,
        'Y_SCALE' : Y_STD,
        'Y_SHIFT' : Y_MEAN,
        'dtype' : torch.float32,
        'transform' : pretrained_transform
    }
    train_dataset = data_sets.MultiImageTrainDataset(image_dir=dataset_path,
                                                     plant_mapping_path=save_mapping_train,
                                                     **dataset_params)
    val_dataset = data_sets.MultiImageTrainDataset(image_dir=dataset_path,
                                                     plant_mapping_path=save_mapping_val,
                                                     **dataset_params)
    



    

    # --------------- training the model ----------------#
    model = train_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        pretrained_model=pretrained_model,
        model_params=model_params,
        optimizer=optimizer,
        optimizer_params=optimizer_params,
        loss_function=loss_function,
        scheduler=scheduler,
        scheduler_params=scheduler_params,
        model=model,
        num_epochs=num_epochs,
        output_path=output_path,
        Y_STD=Y_STD, 
        Y_MEAN=Y_MEAN,
        num_workers=num_worker,
        batch_size=batch_size,
        loss_scaling = Y_MEAN if value_column == 'volume' else 1,
        verbose=True, 
        scaled_loss=scaled_loss, 
        last_time_step=last_time_step,
        checkpoint_path=checkpoint_path
        )

    # Save model locally on disk
    saved_model_path = saver.save_log(model=model,
                            model_params=model_params,
                            optimizer=optimizer,
                            optimizer_params=optimizer_params,
                            lr_scheduler=scheduler,
                            lr_scheduler_params=scheduler_params)
    

    # --------- Model inferrence for evaluation ---------#
    # create / get experiment
    plant_mapping = pd.read_json(save_mapping_plants, orient='index', convert_axes=False)
    experiment = experiment_creator.create_experiment(plant_mapping=plant_mapping,)

    # Compute Predictions for all data including test data
    inferrer = Inferrer(saved_model_path=saved_model_path, 
                        output_name=value_column,
                        Y_STD=Y_STD, 
                        Y_MEAN=Y_MEAN,
                        pretrained_model=pretrained_model,
                        verbose=verbose)
    
    dataset = data_sets.ImageInferenceDataset(image_dir=dataset_path,
                                              transform=pretrained_transform,
                                              model_sequence_size=max_sequence_length,
                                              experiment=experiment)
    experiment_output = inferrer.infer_experiment(dataset=dataset)
    experiment_output.to_json(save_predictions_path, orient='index')
    ic(experiment_output)


    # --------- Compute experiment statistics -----------#
    # Make sure plant_ids are not converted to int in a weird manner
    df_pred = pd.read_json(save_predictions_path, orient='index' , dtype={'plant_id' : str})

    # The index of the mapping files are plant_ids so convert_axes = False otherwise it's converted to int
    df_train = pd.read_json(save_mapping_train, orient='index', convert_axes=False)
    df_test = pd.read_json(save_mapping_test, orient='index', convert_axes=False)
    df_val = pd.read_json(save_mapping_val, orient='index', convert_axes=False)
    
    fig_train, stat_train = stats_lib.compare(mapping_true=df_train,
                                            mapping_predicted=df_pred,
                                            output_name=value_column,
                                            name="train",
                                            output_folder=charts_output_path,
                                            verbose=verbose)
    fig_test, stat_test = stats_lib.compare(mapping_true=df_test,
                                                mapping_predicted=df_pred,
                                                output_name=value_column,
                                                name="test",
                                                output_folder=charts_output_path)
    fig_val, stat_val = stats_lib.compare(mapping_true=df_val,
                                        mapping_predicted=df_pred,
                                        output_name=value_column,
                                        name="val",
                                        output_folder=charts_output_path)
    
    # ------------ Finish a successful run --------------#



def train_model(train_dataset: Dataset,
                val_dataset : Dataset,
                model: torch.nn.Module,
                pretrained_model : torch.nn.Module,
                model_params: Dict[str, any],
                optimizer: torch.optim.Optimizer,
                optimizer_params: Dict[str, any], 
                loss_function: torch.nn.Module,
                scheduler: torch.optim.lr_scheduler,
                scheduler_params: Dict[str, any],
                num_epochs: int,
                output_path: Path,
                Y_STD: float, 
                Y_MEAN: float,
                num_workers: int = 1,
                batch_size: int = 256,
                loss_scaling : float = 1,
                scaled_loss: bool = False,
                last_time_step: bool = True,
                verbose : bool = False, 
                checkpoint_path: Optional[str] = None) -> Tuple[torch.nn.Module, List[float], List[float]]:
                
    """Train the given model and log the epoch losses.

    Args:
        train_dataset (Dataset): Train data and labels
        val_dataset (Dataset): Validation data and labels
        model (torch.nn.Module): Neural network to be trained
        pretrained_model (torch.nn.Module): Pretrained model to use
        model_params (Dict[str, any]): Model parameters with which the model was initialized
        optimizer (torch.optim.Opitmizer): optimizer used for weight updating
        optimizer_params (Dict[str, any]): Initialization parameters for optimizer
        loss_funciton (torch.nn.Module): loss function to be minimized
        scheduler (torch.optim.lr_scheduler): mechanisim to updated the learning rate over the epochs
        scheduler_params (Dict[str, any]): Parameters for scheduler
        num_epochs (int): Number of epochs to be trained
        output_path (Path | str): Directory to store all the output
        Y_MEAN (float): Mean volume of training dataset 
        Y_STD (float): Standard deviation of volume of training dataset 
        num_workers (int, optional): Number of threads to be used. Defaults to 8
        batch_size (int, optional): Batch size for training. Defaults to 256
        loss_scaling (float, optional): Scaling factor to bring the loss in an interpretable size. Defaults to 1
        scaled_loss (bool) : Defaults to False, if loss is used that pays attention to volumes further away from mean 
        last_time_step (bool): Whether to take only last time step of lstm to calculate the loss       
        verbose (bool, optional) : Defaults to False
        checkpoint_path (str, optional): path to checkpoint model in case you want to use one

    Returns:
        Tuple[torch.nn.Module, List[float], List[float]] : the trained model, training loss, and validation loss
    """

    #attention_model, attention_params = data_saver.Saver.load_model(trained_model_path)
    # Load checkpoint into the original model BEFORE wrapping
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint_model = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint_model:
            model.load_state_dict(checkpoint_model["model_state_dict"])
        else:
            model.load_state_dict(checkpoint_model)

    
    saver = Saver(output_path)
    output_path = Path(output_path)
    accelerator = Accelerator()

    # Define a data loader for the dataset
    def collate_fn(batch : List[Tuple[torch.Tensor, float, torch.Tensor]]):
        tensors, labels, masks = zip(*batch)
        tensors = torch.stack(tensors=tensors, dim=0)
        labels = torch.tensor(labels, dtype=tensors.dtype)
        masks = torch.stack(tensors=masks, dim=0)
        return tensors, labels, masks

    # get data loaders
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              persistent_workers=True,
                              pin_memory=True,
                              collate_fn=collate_fn)

    
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            persistent_workers=True,
                            pin_memory=True,
                            collate_fn=collate_fn)

   
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **scheduler_params)
    


    model, pretrained_model, train_loader, val_loader, scheduler = accelerator.prepare(
        model, pretrained_model, train_loader, val_loader, scheduler
    )
    
    best_val_loss = float(1300)
    train_losses = []
    val_losses = [] 


    # training loop (tqdm is just for nice output but deactivated)
    for epoch in tqdm(range(num_epochs), disable=True):
        start_epoch = time.time()
        
        # Training
        eval_time = 0.0
        epoch_train_loss = 0.0
        model.train()
        is_multi_view_cnn = isinstance(model, neural_nets.fine_tuning)
        for features, labels, mask in train_loader:

            if features.dim() == 5:
                if is_multi_view_cnn:
                    # For real CNN, keep raw images in shape (B, T, C, H, W)
                    pass
                else:
                    with torch.no_grad():
                        start_eval_time = time.time()
                        features = pretrained_model(features[~mask])
                        eval_time += time.time() - start_eval_time

                    features = nn.utils.rnn.pad_sequence(
                        features.split((~mask).sum(dim=1).tolist()),
                        batch_first=True,
                    )
                    mask = ~(features.any(dim=-1))
                
            if scaled_loss == True and last_time_step == True:
                outputs = model(features, mask).squeeze(1)
                weights = inverse_freq_weights(labels, num_bins=10, Y_MEAN=Y_MEAN, Y_STD=Y_STD)
                loss = loss_function(outputs, labels, weights)

            elif scaled_loss == True and last_time_step == False:
                outputs = model.forward_(features, mask).squeeze(2)  # shape: [batch, seq_len]
                n_sequence = features.shape[1]
                n_batch = features.shape[0]

                weights_1d = inverse_freq_weights(labels, num_bins=10, Y_MEAN=Y_MEAN, Y_STD=Y_STD)  
                weights_seq = weights_1d.unsqueeze(1).repeat(1, n_sequence) 

                loss = loss_function(outputs, labels.repeat(n_sequence).reshape(n_sequence, n_batch).T, weights_seq)
                
            elif scaled_loss == False and last_time_step == True:
                #training based on last model output 
                outputs = model.forward(features, mask).squeeze(1)
                loss = loss_function(outputs, labels)

            elif scaled_loss == False and last_time_step == False:
                #training based on all sequence model output
                outputs = model.forward_(features, mask).squeeze(2)
                n_sequence = features.shape[1]
                n_batch = features.shape[0]

                loss = loss_function(outputs, labels.repeat(n_sequence).reshape(n_sequence, n_batch).T)

            optimizer.zero_grad()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 0.1)

            optimizer.step()
            epoch_train_loss += loss.item()
        if scheduler is not None:
            scheduler.step()
            
        #if epoch % 5 == 0:
        epoch_val_loss = 0.0
        model.eval()
        with torch.no_grad():
            is_multi_view_cnn = isinstance(model, neural_nets.fine_tuning)
            for features, labels, mask in val_loader:
                features = features[:, :6]
                mask = mask[:, :6]

                if features.dim() == 5:
                    if is_multi_view_cnn:
                        # For real CNN, keep raw images in shape (B, T, C, H, W)
                        pass
                    else:
                    
                        features = pretrained_model(features[~mask])
                        features = nn.utils.rnn.pad_sequence(
                            features.split((~mask).sum(dim=1).tolist()),
                            batch_first=True,
                        )
                        
                        mask = ~(features.any(dim=-1))
                  
                if scaled_loss: 
                    outputs = model(features, mask).squeeze(1)
                    weights = inverse_freq_weights(labels, num_bins=10, Y_MEAN=Y_MEAN, Y_STD=Y_STD)
                    loss_val = loss_function(outputs, labels, weights)

                else:
                    outputs_val = model(features, mask).squeeze(1)
                    loss_val = loss_function(outputs_val, labels)
                
                epoch_val_loss += loss_val.item()

        
        time_epoch = time.time() - start_epoch
        train_loss = epoch_train_loss / len(train_loader)
        val_loss = epoch_val_loss / len(val_loader)

        train_losses.append(train_loss * loss_scaling)
        val_losses.append(val_loss * loss_scaling)


        loss_data = {
            "train_loss": train_losses,
            "val_loss": val_losses
        }

        loss_file = output_path / "training_losses.json"

        with open(loss_file, "w") as f:
            json.dump(loss_data, f)

        print(f"Saved loss history to {loss_file}")
        
        if val_loss < best_val_loss:
            
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss  # Update best loss
            
            saver.save(
                model=model,
                model_params=model_params,
                optimizer=optimizer,
                optimizer_params=optimizer_params,
                lr_scheduler=scheduler,
                lr_scheduler_params=scheduler_params,
                epoch=epoch
            )
        else:
            print("Val loss did not improve")
        
        learning_rate = optimizer.param_groups[0]['lr']
        if True:
            print(f'\t{"Time: ":7s}{time_epoch:>6.4f}',end='')
            print(f'\t{"Evaluation time: ":7s}{eval_time:>6.4f}')
            print(f'{"epoch":7s}{epoch:>5n}\t{"Train Loss:":11s}{train_loss * loss_scaling:>7.4f}', end="")
            print(f'\t{"Val Loss:":10}{val_loss * loss_scaling:>7.4f}', end="")
            print(f'\t{"Current lr:":>12s}{learning_rate:>10.0e}')



    return model

