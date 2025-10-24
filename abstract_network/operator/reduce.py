import torch

from .base import AbstractBase

class AbstractReduce(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.axes = attr.get('axes', None)
        self.keepdim = bool(attr['keepdims']) if 'keepdims' in attr else True
        
        
    def _parse_input_and_axis(self, *x):
        if len(x) > 1:
            self.axes = tuple(item.item() for item in tuple(x[1]))
        self.axes = self.make_axis_non_negative(self.axes)
        return x[0]
    
class AbstractReduceMean(AbstractReduce):
    
    def forward(self, *x):
        x = self._parse_input_and_axis(*x)
        return torch.mean(x, dim=self.axes, keepdim=self.keepdim)