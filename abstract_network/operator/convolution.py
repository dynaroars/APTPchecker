import torch.nn.functional as F
import gurobipy as grb
import numpy as np

from .base import AbstractBase
from ..helper.im2col import im2col


class AbstractConv(AbstractBase):

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)

        assert len(attr["kernel_shape"]) == 2
        assert attr["pads"][0] == attr["pads"][2]
        assert attr["pads"][1] == attr["pads"][3]
        self.padding = [attr["pads"][0], attr["pads"][1]]
        self.F_conv = F.conv2d

        self.stride = attr["strides"]
        self.dilation = attr["dilations"]
        self.groups = attr["group"]

    def forward(self, *x):
        # x: (input, weight, bias)
        bias = x[2] if len(x) == 3 else None
        output = self.F_conv(
            x[0],
            x[1],
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output

    def build_solver(self, *v, model, C=None):
        # pre-check
        gvars_array = np.array(v[0])
        prev_layer_shape = gvars_array.shape  # pre_layer_shape (1, 3, 32, 32)
        this_layer_shape = self.output_shape
        assert gvars_array.shape[1] % self.groups == 0, f"{gvars_array.shape[1]=}"
        assert this_layer_shape[1] % self.groups == 0, f"{this_layer_shape[1]=} {self.groups=}"
        assert len(prev_layer_shape) == len(this_layer_shape) == 4
        assert not any(
            (_.lb == -float("inf") or _.ub == float("inf")) for _ in gvars_array.reshape(-1)
        )
        # current layer weight
        this_layer_weight = v[1].detach().cpu().numpy()
        # current layer bias
        this_layer_bias = v[2].detach().cpu().numpy() if len(v) == 3 else None

        assert (
            this_layer_weight.shape[1] == prev_layer_shape[1] // self.groups
        ), f"{this_layer_weight.shape[1]=} {prev_layer_shape[1]=} {self.groups=}"
        group_input_channel = prev_layer_shape[1] // self.groups
        group_weight_channel = this_layer_weight.shape[0] // self.groups
        if self.padding != [0, 0]:
            gvars_array = np.pad(
                gvars_array,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding[0], self.padding[0]),
                    (self.padding[1], self.padding[1]),
                ),
                mode="constant",  # TODO: Generalize
                constant_values=0,  # TODO: Generalize
            )

        # current layer constraints
        new_layer_gurobi_vars = []
        count = 0
        for image_idx in range(this_layer_shape[0]):
            for group_idx in range(self.groups):
                group_gvars = gvars_array[
                    image_idx,
                    group_idx * group_input_channel : (group_idx + 1) * group_input_channel,
                ]
                group_weight = this_layer_weight[
                    group_idx * group_weight_channel : (group_idx + 1) * group_weight_channel
                ]
                group_bias = (
                    this_layer_bias[
                        group_idx * group_weight_channel : (group_idx + 1) * group_weight_channel
                    ]
                    if this_layer_bias is not None
                    else 0
                )

                unfold_data = im2col(
                    group_gvars,
                    kernel_h=group_weight.shape[2],
                    kernel_w=group_weight.shape[3],
                    pad_h=self.padding[0],
                    pad_w=self.padding[1],
                    stride_h=self.stride[0],
                    stride_w=self.stride[1],
                    dilation_h=self.dilation[0],
                    dilation_w=self.dilation[1],
                )
                for out_chan_idx in range(group_weight_channel):
                    # init linear expression
                    weight = group_weight[out_chan_idx].flatten()
                    bias = group_bias[out_chan_idx]
                    num_output = unfold_data.shape[-1]
                    # init linear constraint LHS implied by the conv operation
                    for output_idx in range(num_output):
                        gvars = unfold_data[:, output_idx]
                        lin_expr = grb.LinExpr(weight, gvars) + bias

                        # add the output var and constraint
                        var = model.addVar(
                            lb=-float("inf"),
                            ub=float("inf"),
                            obj=0,
                            vtype=grb.GRB.CONTINUOUS,
                            name=f"lay{self.name}_{count}",
                        )
                        model.addConstr(lin_expr == var, name=f"lay{self.name}_{count}_eq")
                        count += 1
                        new_layer_gurobi_vars.append(var)
        self.solver_vars = np.array(new_layer_gurobi_vars).reshape(this_layer_shape)
        model.update()
        return self.solver_vars
