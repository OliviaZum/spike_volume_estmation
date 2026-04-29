import numpy as np
import torch
import random

class SeedSetter:
    
    def __init__(self,  seed : int = 1071997):
        """Class to make code reproducable

        Args:
            seed (int, optional): Defaults to 1071997.
        """
        self.reset_seed(seed)
    
    def reset_seed(self, seed = 1071997):
        """Initialized seed for reproducability

        Args:
            seed (int, optional): Defaults to 1071997.
        """
        
        # set seeds for reproducability
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        
        self.initial_seed = seed

        # make sure code is reproducable
        # https://pytorch.org/docs/stable/notes/randomness.html        
        self.th_generator = torch.Generator()
        self.th_generator.manual_seed(seed)
        
        def seed_worker(worker_id):
            np.random.seed(self.initial_seed)
            random.seed(self.initial_seed)
        
        # make function available
        self.seed_worker = seed_worker