from .input import AbstractInput

class AbstractParameter(AbstractInput):

    def __init__(self, ori_name, value, perturbation=None, options=None, attr=None):
        super().__init__(ori_name=ori_name, value=None, perturbation=perturbation, attr=attr)
        self.register_parameter('param', value)

        # visualize only
        self.color = 'coral'
        
    def register_parameter(self, name, param):
        """Override register_parameter() hook to register only needed parameters."""
        if name == 'param':
            return super().register_parameter(name, param)
        else:
            object.__setattr__(self, name, param)

    def forward(self):
        return self.param.requires_grad_(self.training)
    