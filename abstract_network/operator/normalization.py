import torch.nn.functional as F
import gurobipy as grb
import numpy as np
import torch

from .base import AbstractBase

class AbstractBatchNormalization(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        
        assert not attr['training_mode']
        self.epsilon = attr['epsilon']
        self.momentum = round(1 - attr['momentum'], 8)
        self.use_mean = self.options.get("mean", True)
        self.use_var = self.options.get("var", True)
        self.use_affine = self.options.get("affine", True)
        
        
    def forward(self, x, w, b, m, v):
        current_mean = m.data
        current_var = v.data
        training = False
        
        # check unused mean or var
        if not self.use_mean:
            current_mean = torch.zeros_like(current_mean)
        if not self.use_var:
            current_var = torch.ones_like(current_var)
            
        if not self.use_affine:
            w = torch.ones_like(w)
            b = torch.zeros_like(b)
            
        if not self.use_mean or not self.use_var:
            # If mean or variance is disabled
            w = w / torch.sqrt(current_var + self.epsilon)
            b = b - current_mean * w
            shape = (1, -1) + (1,) * (x.ndim - 2)
            result = w.view(*shape) * x + b.view(*shape)
        else:
            result = F.batch_norm(x, m, v, w, b, training, self.momentum, self.epsilon)
        return result
            
            
    def build_solver(self, *v, model, C=None):
        # pre-check
        assert len(v[0]) == 1, f'{len(v[0])=}'
        gvars_array = np.array(v[0])
        this_layer_shape = self.output_shape
        assert this_layer_shape == gvars_array.shape

        # check unused mean or var
        current_mean = v[3]
        current_var = v[4]
        if not self.use_mean:
            current_mean = torch.zeros_like(current_mean)
        if not self.use_var:
            current_var = torch.ones_like(current_var)
            
        # current layer weights and bias
        weight, bias = v[1], v[2]
        if not self.use_affine:
            weight = torch.ones_like(weight)
            bias = torch.zeros_like(bias)

        tmp_bias = bias - current_mean / torch.sqrt(current_var + self.epsilon) * weight
        tmp_weight = weight / torch.sqrt(current_var + self.epsilon)

        # current layer constraints
        new_layer_gurobi_vars = []
        neuron_idx = 0
        if len(this_layer_shape) == 3:
            for out_chan_idx in range(this_layer_shape[1]):
                out_chan_vars = []
                for out_row_idx in range(this_layer_shape[2]):
                    # print(this_layer_bias.shape, out_chan_idx, out_lbs.size(1))
                    lin_expr = tmp_bias[out_chan_idx].item() + tmp_weight[out_chan_idx].item() * gvars_array[0, out_chan_idx, out_row_idx]
                    
                    var = model.addVar(
                        lb=-float('inf'), 
                        ub=float('inf'), 
                        obj=0, 
                        vtype=grb.GRB.CONTINUOUS,
                        name=f'lay{self.name}_{neuron_idx}',
                    )
                    model.addConstr(lin_expr == var, name=f'lay{self.name}_{neuron_idx}_eq')
                    neuron_idx += 1

                    out_chan_vars.append(var)
                new_layer_gurobi_vars.append(out_chan_vars)
        elif len(this_layer_shape) == 4:
            for out_chan_idx in range(this_layer_shape[1]):
                out_chan_vars = []
                for out_row_idx in range(this_layer_shape[2]):
                    out_row_vars = []
                    for out_col_idx in range(this_layer_shape[3]):
                        # print(this_layer_bias.shape, out_chan_idx, out_lbs.size(1))
                        lin_expr = tmp_bias[out_chan_idx].item() + tmp_weight[out_chan_idx].item() * gvars_array[0, out_chan_idx, out_row_idx, out_col_idx]
                        
                        var = model.addVar(
                            lb=-float('inf'), 
                            ub=float('inf'), 
                            obj=0, 
                            vtype=grb.GRB.CONTINUOUS,
                            name=f'lay{self.name}_{neuron_idx}',
                        )
                        model.addConstr(lin_expr == var, name=f'lay{self.name}_{neuron_idx}_eq')
                        neuron_idx += 1

                        out_row_vars.append(var)
                    out_chan_vars.append(out_row_vars)
                new_layer_gurobi_vars.append(out_chan_vars)
        else:
            raise NotImplementedError(f'{this_layer_shape=}')

        self.solver_vars = np.array(new_layer_gurobi_vars).reshape(this_layer_shape)
        model.update()
        return self.solver_vars
        