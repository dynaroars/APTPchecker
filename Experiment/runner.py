import torch
import tqdm
import os

from utils import get_total_instances, get_benchmark_list
from verifier import neuralsat, abcrown, marabou
from argument import parse_args

def main():
    
    torch.set_default_dtype(torch.float64)
    args = parse_args()
    
    if args.verifier == "neuralsat":
        verify_func = neuralsat.verify
    elif args.verifier == "abcrown":
        verify_func = abcrown.verify
    elif args.verifier == "marabou":
        verify_func = marabou.verify
    else:
        raise ValueError(f"Invalid verifier: {args.verifier=}")
    
    total_instances = get_total_instances(args)
    print(f'{total_instances=}')
    pbar = tqdm.tqdm(total=total_instances)
    
    print(f'[+] Running {args.verifier=} {args.split_type=}')
    
    stats = {
        'sat': 0,
        'unsat': 0,
        'timeout': 0,
        'error': 0,
    }
    
    for benchmark in get_benchmark_list(args):
        pbar.set_description(f'{benchmark=}')
        output_dir = os.path.join(args.result_dir, args.verifier, args.split_type, benchmark)
        os.makedirs(output_dir, exist_ok=True)
        # print(f'{output_dir=}')
        
        benchmark_dir = os.path.join(args.benchmark_dir, benchmark)
        instances_file = os.path.join(benchmark_dir, 'instances.csv')
        assert os.path.exists(instances_file), f"Instances file does not exist: {instances_file=}"
        
        with open(instances_file, 'r') as f:
            instances = f.readlines()
        
        for instance in instances:
            onnx, vnnlib, _ = instance.strip().split(',')
            onnx_path = os.path.abspath(os.path.join(benchmark_dir, onnx))
            assert os.path.exists(onnx_path), f"ONNX file does not exist: {onnx_path=}"
            vnnlib_path = os.path.abspath(os.path.join(benchmark_dir, vnnlib))
            assert os.path.exists(vnnlib_path), f"VNNLIB file does not exist: {vnnlib_path=}"
            output_path = os.path.abspath(os.path.join(output_dir, f'{os.path.splitext(os.path.basename(onnx))[0]}_{os.path.splitext(os.path.basename(vnnlib))[0]}'))
            # print(f'{onnx_path=}')
            # print(f'{vnnlib_path=}')
            # print(f'{output_path=}')
            status = verify_func(args, onnx_path, vnnlib_path, output_path, args.timeout)
            stats[status] += 1
            pbar.update(1)
            pbar.set_postfix(**stats)
            # exit()
        
        # print()
        
    

if __name__ == "__main__":
    main()