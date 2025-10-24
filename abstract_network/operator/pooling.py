import torch.nn.functional as F
import gurobipy as grb
import numpy as np
import torch

from .base import AbstractBase

class AbstractAveragePool(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        
        self.kernel_size = attr['kernel_shape']
        assert len(self.kernel_size) == 2
        
        self.stride = attr['strides']
        assert len(self.stride) == 2
        
        # TODO: generalize
        assert ('pads' not in attr) or (attr['pads'][0] == attr['pads'][2])
        assert ('pads' not in attr) or (attr['pads'][1] == attr['pads'][3])
        if 'pads' not in attr:
            self.padding = [0, 0]
        else:
            self.padding = [attr['pads'][0], attr['pads'][1]]
            
        self.ceil_mode = bool(attr['ceil_mode'])
        assert not self.ceil_mode
        
        self.count_include_pad = bool(attr['count_include_pad'])
        assert self.count_include_pad
    
    def forward(self, x):
        return F.avg_pool2d(
            x, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding, 
            ceil_mode=self.ceil_mode, 
            count_include_pad=self.count_include_pad,
        )
    
    def build_solver(self, *v, model, C=None):
        # TODO: check correctness
        # pre-check
        gvars_array = np.array(v[0])
        prev_layer_shape = gvars_array.shape
        this_layer_shape = self.output_shape
        assert len(prev_layer_shape) == len(this_layer_shape) == 4
        assert this_layer_shape[2] == (prev_layer_shape[2] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        assert this_layer_shape[3] == (prev_layer_shape[3] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # current layer constraints
        coeff = 1.0 / (self.kernel_size[0] * self.kernel_size[1])
        neuron_idx = 0
        new_layer_gurobi_vars = []
        for out_chan_idx in range(this_layer_shape[1]):
            out_chan_vars = []
            for out_row_idx in range(this_layer_shape[2]):
                out_row_vars = []
                for out_col_idx in range(this_layer_shape[3]):
                    # init linear expression
                    lin_expr = 0.0
                    for ker_row_idx in range(self.kernel_size[0]):
                        in_row_idx = -self.padding[0] + self.stride[0] * out_row_idx + ker_row_idx
                        if (in_row_idx < 0) or (in_row_idx == prev_layer_shape[2]):
                            # This is padding -> value of 0
                            continue
                        for ker_col_idx in range(self.kernel_size[1]):
                            in_col_idx = -self.padding[1] + self.stride[1] * out_col_idx + ker_col_idx
                            if (in_col_idx < 0) or (in_col_idx == prev_layer_shape[3]):
                                # This is padding -> value of 0
                                continue
                            prev_var = gvars_array[0][out_chan_idx][in_row_idx][in_col_idx]
                            assert isinstance(prev_var, grb.Var)
                            lin_expr += coeff * prev_var
                    
                    # add the output var and constraint
                    var = model.addVar(
                        lb=-float('inf'), 
                        ub=float('inf'), 
                        obj=0, 
                        vtype=grb.GRB.CONTINUOUS, 
                        name=f'lay{self.name}_{neuron_idx}'
                    )
                    model.addConstr(lin_expr == var, name=f'lay{self.name}_{neuron_idx}_eq')
                    neuron_idx += 1

                    out_row_vars.append(var)
                out_chan_vars.append(out_row_vars)
            new_layer_gurobi_vars.append(out_chan_vars)

        assert np.array(new_layer_gurobi_vars).shape == this_layer_shape[1:]
        self.solver_vars = np.array([new_layer_gurobi_vars])
        model.update()
        return self.solver_vars
    
    

class AbstractMaxPool(AbstractBase):
    
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.kernel_size = attr['kernel_shape']
        assert len(self.kernel_size) == 2
        
        self.stride = attr['strides']
        assert len(self.stride) == 2
        
        # TODO: generalize
        assert ('pads' not in attr) or (attr['pads'][0] == attr['pads'][2])
        assert ('pads' not in attr) or (attr['pads'][1] == attr['pads'][3])
        self.padding = [attr['pads'][0], attr['pads'][1]]
        
        self.ceil_mode = bool(attr['ceil_mode'])
        assert not self.ceil_mode
        
        
    def forward(self, x):
        output, _ = F.max_pool2d(x, self.kernel_size, self.stride, self.padding, return_indices=True, ceil_mode=self.ceil_mode)
        return output


    def build_solver(self, *v, model, C=None):
        # TODO: check correctness
        # pre-check
        gvars_array = np.array(v[0])
        prev_layer_shape = gvars_array.shape
        this_layer_shape = self.output_shape
        assert len(prev_layer_shape) == len(this_layer_shape) == 4
        assert this_layer_shape[2] == (prev_layer_shape[2] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        assert this_layer_shape[3] == (prev_layer_shape[3] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # current layer constraints
        prev_upper = torch.tensor([_.ub for _ in gvars_array.reshape(-1)]).view(prev_layer_shape)
        prev_ubs = self.forward(prev_upper).detach().cpu().numpy()
        new_layer_gurobi_vars = []
        neuron_idx = 0
        for out_chan_idx in range(this_layer_shape[1]):
            out_chan_vars = []
            for out_row_idx in range(this_layer_shape[2]):
                out_row_vars = []
                for out_col_idx in range(this_layer_shape[3]):
                    a_sum = 0.0
                    var = model.addVar(
                        lb=-float('inf'), 
                        ub=float('inf'), 
                        obj=0, vtype=grb.GRB.CONTINUOUS, 
                        name=f'lay{self.name}_{neuron_idx}',
                    )
                    for ker_row_idx in range(self.kernel_size[0]):
                        in_row_idx = -self.padding[0] + self.stride[0] * out_row_idx + ker_row_idx
                        if (in_row_idx < 0) or (in_row_idx == prev_layer_shape[2]):
                            # This is padding -> value of 0
                            continue
                        for a_idx, ker_col_idx in enumerate(range(self.kernel_size[1])):
                            in_col_idx = -self.padding[1] + self.stride[1] * out_col_idx + ker_col_idx
                            if (in_col_idx < 0) or (in_col_idx == prev_layer_shape[3]):
                                # This is padding -> value of 0
                                continue
                            prev_var = gvars_array[0][out_chan_idx][in_row_idx][in_col_idx]
                            assert isinstance(prev_var, grb.Var)
                            a = model.addVar(vtype=grb.GRB.BINARY, name=f'aMaxPool{self.name}_{neuron_idx}_{a_idx}')
                            a_sum += a
                            model.addConstr(var >= prev_var)
                            model.addConstr(var <= prev_var + (1 - a) * prev_ubs[0, out_chan_idx, out_row_idx, out_col_idx])
                    model.addConstr(a_sum == 1, name=f'lay{self.name}_{neuron_idx}_eq')
                    neuron_idx += 1
                    
                    out_row_vars.append(var)
                out_chan_vars.append(out_row_vars)
            new_layer_gurobi_vars.append(out_chan_vars)
                            
        assert np.array(new_layer_gurobi_vars).shape == this_layer_shape[1:]
        self.solver_vars = np.array([new_layer_gurobi_vars])
        model.update()
        return self.solver_vars
    