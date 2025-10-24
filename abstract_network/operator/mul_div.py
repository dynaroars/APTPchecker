import numpy as np
import torch

from .base import AbstractBase
    
class AbstractMul(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        
    def forward(self, x, y):
        return x * y
    
    def build_solver(self, *v, model, C=None):
        assert isinstance(v[1], torch.Tensor)
        if isinstance(v[0], torch.Tensor):
            self.solver_vars = self.forward(*v)
        else:
            assert len(v[0]) == 1, f'{len(v[0])=}'
            gvar_array = np.array(v[0]) * v[1].cpu().numpy()
            self.solver_vars = gvar_array
        return self.solver_vars
        