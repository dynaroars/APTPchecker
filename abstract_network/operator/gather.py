import torch

from .base import AbstractBase

class AbstractGather(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr.get('axis', 0)
        
    def forward(self, x, indices):
        if self.axis == -1:
            self.axis = len(x.shape) - 1
        
        if isinstance(indices, torch.Tensor):
            x = x.to(indices.device)
        
        if indices.ndim == 0:
            if indices == -1:
                indices = x.shape[self.axis] + indices
            return torch.index_select(x, dim=self.axis, index=indices).squeeze(self.axis)
        elif indices.ndim == 1:
            return torch.index_select(x, dim=self.axis, index=indices)
        
        raise ValueError(f'Unsupported shapes in Gather: data={x.shape}, indices={indices.shape}, axis={self.axis}')
    
    def build_solver(self, *v, model, C=None):
        self.solver_vars = self.forward(v[0], v[1])
        return self.solver_vars
