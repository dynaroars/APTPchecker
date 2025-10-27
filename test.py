import warnings
import torch

from helper.network.read_onnx import parse_onnx

warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
from abstract_network import AbstractNetwork

def test_solver():
    device = 'cpu'
    # onnx_path = 'data/cnn.onnx'
    # onnx_path = 'example/resnet.onnx'
    # onnx_path = 'example/fnn.onnx'
    onnx_path = 'example/sample.onnx'
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
    C = torch.tensor([1.0, 0.0], device=device).view(1, 1, -1)
    C = None

    abs_net.build_solver_module(
        x_L=x, 
        x_U=x, 
        C=C,
        timeout=2.5,
    )
    print(f'{y1=}')
    
    print(abs_net.split_nodes())
    print(abs_net.final_node().solver_vars)
    
if __name__ == "__main__":
    test_solver()