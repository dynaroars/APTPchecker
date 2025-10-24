
from .base import AbstractBase

class AbstractConstant(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.value = attr['value'].to(self.device)
        # print(f'\t- [+] Init constant: {self.value=}')
        
        # visualize only
        self.color = 'burlywood'
        
    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name}, value={self.value})'
    
    def forward(self):
        return self.value.to(self.device)
    
    def build_solver(self, *v, model, C=None):
        self.solver_vars = self.value
        return self.solver_vars
    
    
    
class AbstractConstantOfShape(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.value = attr['value'].to(self.device)
        
        # visualize only
        self.color = 'burlywood'
        
    def forward(self, x):
        return self.value.expand(*list(x))
    
    
    def build_solver(self, *v, model, C=None):
        self.solver_vars = self.forward(*v)
        return self.solver_vars