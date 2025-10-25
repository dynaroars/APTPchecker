from .base import AbstractBase

class AbstractInput(AbstractBase):
    
    def __init__(self, ori_name, value, perturbation=None, input_index=None, options=None, attr=None):
        super().__init__(options=options, attr=attr)
        self.ori_name = ori_name
        self.value = value
        self.perturbation = perturbation
        self.input_index = input_index
        self.color = 'yellow' # visualize only
    
    def forward(self):
        return self.value
