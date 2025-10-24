
from .base import AbstractBase

class AbstractSqueeze(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        if 'axes' in attr:
            assert attr['axes'] == 1
            self.axes = attr['axes'][0]
        else:
            self.axes = None
            
    def forward(self, *x):
        data = x[0]
        axes = self.axes if self.axes is not None else x[1].item()
        return data.squeeze(axes)
    
    def build_solver(self, *v, model, C=None):
        self.solver_vars = self.forward(*v)
        return self.solver_vars
    
class AbstractUnsqueeze(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        if 'axes' in attr:
            assert attr['axes'] == 1
            self.axes = attr['axes'][0]
        else:
            self.axes = None
            
    def forward(self, *x):
        data = x[0]
        axes = self.axes if self.axes is not None else x[1].item()
        return data.unsqueeze(axes)
    
    def build_solver(self, *v, model, C=None):
        self.solver_vars = self.forward(*v)
        return self.solver_vars