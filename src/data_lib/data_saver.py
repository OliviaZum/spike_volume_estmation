import torch
import model_lib.neural_nets
from pathlib import Path
from typing import Dict, Optional, Tuple

class Saver:
    def __init__(self, save_dir : Path | str = 'saved_models') -> None:
        """Saver classe which saves models in the save_dir with increasing indices.

        The filed self.count gets increased by every call of the function save.

        Args:
            save_dir (Path | str, optional): directory to save model at. Defaults to 'saved_models'.
        """
        self.count = 0
        self.save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    def compute_path_from_index(self, index : int) -> Path:
        """Comptue output path

        Args:
            index (int): Model index

        Returns:
            Path : Path for model with index `index`.
        """
        compute_path = self.save_dir / f"model_{index}.pth"
        return compute_path

    def save(self,
            model : torch.nn.Module,
            model_params : Dict[str, any],
            optimizer : torch.optim.Optimizer,
            optimizer_params : Dict[str, any],
            lr_scheduler : Optional[torch.optim.lr_scheduler.LRScheduler],
            lr_scheduler_params : Dict[str, any],
            epoch : int,
            ) -> Path:
        """Saves model and optimizer state

        Args:
            model (torch.nn.Module): Model
            model_params (Dict[str, any]): Model parameter for initialization
            optimizer (torch.optim.Optimizer): Optimizer
            optimizer_params (Dict[str, any]): Optimizer parameter for initialization
            lr_scheduler (torch.optim.Optimizer): Scheduler
            lr_scheduler_params (Dict[str, any]): Scheduler parameter for initialization
            epoch: At which epoch the model was saved
        
        Returns:
            Path : path to the saved model

        """
        
        if lr_scheduler is None:
            scheduler_dict = None
        else:
            scheduler_dict = lr_scheduler.state_dict()

        saved_data = {
            'model_class' : type(model).__name__,
            'model_state_dict': model.state_dict(),
            'model_params' : model_params,
            'optimizer_class' : type(optimizer).__name__,
            'optimizer_state_dict' : optimizer.state_dict(),
            'optimizer_params' : optimizer_params,
            'lr_scheduler_class' : type(lr_scheduler).__name__,
            'lr_scheduler_state_dict' : scheduler_dict,
            'lr_scheduler_params' : lr_scheduler_params,
            'epoch' : epoch,
            'version' : 'final_0.1.0' # version used for the load model function
        }

        output_path = self.compute_path_from_index(self.count)
        torch.save(saved_data, output_path)
        
        # Save the same content with a .pt extension
        pt_output_path = output_path.with_suffix('.pt')  # Change the file extension to .pt
        torch.save(saved_data, pt_output_path)

        
        self.count += 1
        return output_path
    
    def save_log(self,
            model : torch.nn.Module,
            model_params : Dict[str, any],
            optimizer : torch.optim.Optimizer,
            optimizer_params : Dict[str, any],
            lr_scheduler : Optional[torch.optim.lr_scheduler.LRScheduler],
            lr_scheduler_params : Dict[str, any]
            ) -> Path:
        """Saves model and optimizer state

        Args:
            model (torch.nn.Module): Model
            model_params (Dict[str, any]): Model parameter for initialization
            optimizer (torch.optim.Optimizer): Optimizer
            optimizer_params (Dict[str, any]): Optimizer parameter for initialization
            lr_scheduler (torch.optim.Optimizer): Scheduler
            lr_scheduler_params (Dict[str, any]): Scheduler parameter for initialization
            epoch: At which epoch the model was saved
        
        Returns:
            Path : path to the saved model

        """
        
        if lr_scheduler is None:
            scheduler_dict = None
        else:
            scheduler_dict = lr_scheduler.state_dict()

        saved_data = {
            'model_class' : type(model).__name__,
            'model_state_dict': model.state_dict(),
            'model_params' : model_params,
            'optimizer_class' : type(optimizer).__name__,
            'optimizer_state_dict' : optimizer.state_dict(),
            'optimizer_params' : optimizer_params,
            'lr_scheduler_class' : type(lr_scheduler).__name__,
            'lr_scheduler_state_dict' : scheduler_dict,
            'lr_scheduler_params' : lr_scheduler_params,
            'version' : 'final_0.1.0' # version used for the load model function
        }

        output_path = self.compute_path_from_index(self.count)
        torch.save(saved_data, output_path)

        self.count += 1
        return output_path
    
    
    
    @staticmethod
    def load_model(model_path : Path | str) -> Tuple[torch.nn.Module, Dict[str, any]]:
        """Loads the model from a checkpoint

        Args:
            model_path (Path | str): Path to the saved model

        Returns:
            Tuple[torch.nn.Module: Model with pretrained weights
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            print(f'Path {str(model_path)} does not exist.')
        
        try:
            checkpoint = torch.load(model_path)
        except Exception as e:
            print(f'Failed to extract Model with error {e}.')
        
        if 'version' not in checkpoint.keys():
            # first models did not have wandb backups and were stored code dependent
            model = checkpoint["model_state_dict"]
            return model, None

        elif checkpoint['version'] == 'final_0.1.0':
            model_class = checkpoint['model_class']
            model_params = checkpoint['model_params']
            model_state = checkpoint['model_state_dict']

            # initialize model
            model : torch.nn.Module = getattr(model_lib.neural_nets, model_class)(**model_params)
            model.load_state_dict(model_state)
            return model, model_params
            
        

        


