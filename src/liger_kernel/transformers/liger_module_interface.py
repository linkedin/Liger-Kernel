from abc import abstractmethod
import torch

class LigerModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def __reset_params(self, **kwargs):
        """
        A function to reassign class attributes based on kwargs of forward method
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    