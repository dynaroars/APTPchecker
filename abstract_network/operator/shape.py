import numpy as np
import torch

from .base import AbstractBase

class AbstractShape(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)

    @staticmethod
    def shape(x):
        return x.shape if isinstance(x, torch.Tensor) else torch.tensor(x).shape

    def forward(self, x):
        return AbstractShape.shape(x)
    
    
    def build_solver(self, *v, model, C=None):
        if not isinstance(v[0], torch.Tensor):
            # e.g., v[0] input shape (8, 7, 7) => output its shape (1, 8, 7, 7)
            assert len(v[0]) == 1, f'{len(v[0])=}'
            self.solver_vars = torch.tensor(np.array(v[0]).shape).long()
        else:
            self.solver_vars = torch.tensor(self.forward(v[0])).long()
        
        return self.solver_vars
