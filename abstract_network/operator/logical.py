import torch

from .base import AbstractBase
    
class AbstractEqual(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        
    def forward(self, x, y):
        return x == y
    
    def build_solver(self, *v, model, C=None):
        self.solver_vars = self.forward(*v)
        return self.solver_vars
    

class AbstractWhere(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        
        
    def forward(self, condition, x, y):
        return torch.where(condition.to(torch.bool), x, y)

    def build_solver(self, *v, model, C=None):
        self.solver_vars = self.forward(*v)
        return self.solver_vars