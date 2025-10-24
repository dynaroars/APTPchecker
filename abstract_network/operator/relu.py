import torch.nn.functional as F
import gurobipy as grb
import numpy as np

from .base import AbstractBase

class AbstractRelu(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.leaky_alpha = attr.get('alpha', 0)
        
    def forward(self, x):
        if self.leaky_alpha > 0:
            return F.leaky_relu(x, negative_slope=self.leaky_alpha)
        return F.relu(x)
        
    def build_solver(self, *v, model, C=None):
        if self.leaky_alpha > 0:
            raise NotImplementedError

        # pre-check
        assert len(v[0]) == 1, f'{len(v[0])=}'
        gvars_array = np.array(v[0])
        this_layer_shape = self.output_shape
        assert gvars_array.shape == this_layer_shape
        assert not any((_.lb == -float('inf') or _.ub == float('inf')) for _ in gvars_array.reshape(-1))

        # constant
        zero_var = model.getVarByName("zero")
        
        # current layer constraints
        new_layer_gurobi_vars = []
        for neuron_idx, pre_var in enumerate(gvars_array.reshape(-1)):
            if pre_var.lb >= 0: # active
                var = pre_var
            elif pre_var.ub <= 0: # inactive
                var = zero_var
            else: # unstable
                # post-relu var
                var = model.addVar(
                    lb=0, 
                    ub=pre_var.ub, 
                    obj=0, 
                    vtype=grb.GRB.CONTINUOUS, 
                    name=f'ReLU{self.name}_{neuron_idx}',
                )
                # binary indicator
                a = model.addVar(vtype=grb.GRB.BINARY, name=f'aReLU{self.name}_{neuron_idx}')
                # relu constraints
                model.addConstr(pre_var - pre_var.lb * (1 - a) >= var, name=f'ReLU{self.name}_{neuron_idx}_a_0')
                model.addConstr(var >= pre_var, name=f'ReLU{self.name}_{neuron_idx}_a_1')
                model.addConstr(pre_var.ub * a >= var, name=f'ReLU{self.name}_{neuron_idx}_a_2')

            new_layer_gurobi_vars.append(var)

        new_layer_gurobi_vars = np.array(new_layer_gurobi_vars).reshape(this_layer_shape)
        self.solver_vars = new_layer_gurobi_vars
        model.update()
        return self.solver_vars
    