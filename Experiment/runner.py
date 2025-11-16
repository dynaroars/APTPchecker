import argparse
import tqdm
import time
import os

##############################################################
TIMEOUT = 1000
ROOT_DIR = os.path.dirname(__file__)
RESULT_DIR = f'{ROOT_DIR}/result'
BENCHMARK_DIR = f'{ROOT_DIR}/benchmark'
##############################################################
print(f'{RESULT_DIR = }')
print(f'{BENCHMARK_DIR = }')
os.makedirs(RESULT_DIR, exist_ok=True)
##############################################################
cmd =        './vnncomp_scripts/run_instance.sh v1 {category} {onnx_path} {vnnlib_path} {res_file} {timeout} > {log_file}'
cmd_wo_log = './vnncomp_scripts/run_instance.sh v1 {category} {onnx_path} {vnnlib_path} {res_file} {timeout}'
##############################################################

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tool', type=str, required=True)
    args = parser.parse_args()   

    for benchmark_name in ['fnn_small', 'fnn_medium', 'cnn_small', 'cnn_medium']:
        benchmark_path = os.path.join(BENCHMARK_DIR, benchmark_name)
        OUTPUT_DIR = f'{RESULT_DIR}/{args.tool}/{benchmark_name}'
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        instance_csv  = f'{benchmark_path}/instances.csv'

        instances = []
        for line in open(instance_csv).read().strip().split('\n'):
            onnx_path, vnnlib_path, _ = line.split(',')
            onnx_path = os.path.join(benchmark_path, onnx_path)
            vnnlib_path = os.path.join(benchmark_path, vnnlib_path)
            assert os.path.exists(onnx_path)
            assert os.path.exists(vnnlib_path)
            instances.append((onnx_path, vnnlib_path))
            

        pbar = tqdm.tqdm(instances)
        pbar.set_description(f'Benchmark {benchmark_name} (timeout={TIMEOUT})')
        for idx, (onnx_path, vnnlib_path) in enumerate(pbar):
            onnx_name = os.path.splitext(os.path.basename(onnx_path))[0]
            vnnlib_name = os.path.splitext(os.path.basename(vnnlib_path))[0]
            output_name = f'{OUTPUT_DIR}/net_{onnx_name}_spec_{vnnlib_name}'
            
            config_dict = {
                'category': benchmark_name,
                'onnx_path': onnx_path,
                'vnnlib_path': vnnlib_path,
                'timeout': TIMEOUT,
                'res_file': f'{output_name}.res',
                'log_file': f'{output_name}.log',
                'error_file': f'{output_name}.error',
                'command_file': f'{output_name}.command',
            }
            
            print('\n-----------------------------------------------------')
            print(cmd_wo_log.format(**config_dict))
            print('-----------------------------------------------------\n')
            
            with open(config_dict['command_file'], 'w') as fp:
                fp.write(f'{cmd_wo_log.format(**config_dict)}\n')

            if os.path.exists(config_dict['res_file']):
                status = open(config_dict['res_file']).read().strip().lower()
                continue

            if os.path.exists(config_dict['error_file']):
                continue

            
            os.system(cmd.format(**config_dict))

            if not os.path.exists(config_dict['res_file']):
                # error
                with open(config_dict['error_file'], 'w') as fp:
                    fp.write(f'error,0.0\n')

        #     # TODO: remove 
        #     break
        # break