import argparse
import os

from utils import set_seed

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark_dir", type=str, default="benchmark/", help="Root directory for benchmark")
    p.add_argument("--result_dir", type=str, default="result/", help="Root directory for result")
    p.add_argument("--verifier", type=str, required=True, choices=["neuralsat", "abcrown", "marabou"])
    p.add_argument("--split_type", type=str, required=True, choices=["input", "hidden"])
    p.add_argument("--verifier_dir", type=str, required=True, help="Verifier directory")
    p.add_argument("--timeout", type=int, default=100, help="Timeout")
    p.add_argument("--device", type=str, default="cuda:0", help="Device to run the verifier")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    args.home_dir = os.getcwd()
    args.verifier_dir = os.path.abspath(os.path.join(args.home_dir, args.verifier_dir))
    set_seed(args.seed)
    
    return args
