import warnings
import torch

from helper.network.read_onnx import parse_onnx

warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
from abstract_network import AbstractNetwork

def test_solver():
    device = 'cpu'
    # onnx_path = 'data/fnn.onnx'
    # onnx_path = 'data/cnn.onnx'
    onnx_path = 'example/resnet.onnx'
    # load instance
    model, input_shape, output_shape = parse_onnx(onnx_path)
    model.eval()
    # objectives = parse_vnnlib(vnnlib_path, input_shape)
    
    abs_net = AbstractNetwork(model, input_shape, device)
    
    x = torch.randn(input_shape, device=device)
    y1 = model(x)
    y2 = abs_net(x)
    print(f'{y1=}')
    print(f'{y2=}')
    diff = torch.norm(y1 - y2).item()
    print(f'{diff=}')
    assert torch.allclose(y1, y2)
    abs_net.visualize('example/graph')
    # exit()
    
    x_L = torch.randn(input_shape, device=device)
    x_U = x_L + 1.0
    C = None


    abs_net.build_solver_module(
        x_L=x_L, 
        x_U=x_U, 
        C=C,
        timeout_per_neuron=2.5,
    )
    
if __name__ == "__main__":
    test_solver()