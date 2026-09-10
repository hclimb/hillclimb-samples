import argparse
from importlib import import_module
import json
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from utils.numerics import compare
from utils.workload import build_inputs, call_native, preserve_inputs, replace_values_in_place
from verifiers.reference import reference
from verifiers.sequence import benchmark_sequence


def timeout_handler(signum, frame):
    raise TimeoutError('Worker phase exceeded its time limit')


def time_call(operation, *arguments, calls=1):
    if not isinstance(calls, int) or calls <= 0:
        raise ValueError('Expected a positive call count')
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(calls):
        answer = operation(*arguments)
    torch.cuda.synchronize()
    return answer, (time.perf_counter() - started) / calls


def benchmark_case(implementation, case, seed, tolerances):
    inputs = build_inputs(case, seed, seed + 17)
    preserved = preserve_inputs(inputs)
    scheduler, _ = implementation.get_mla_metadata()
    answer, cold_seconds = time_call(call_native, implementation, inputs, scheduler)
    expected = reference(preserved)
    result = dict(cold_seconds=cold_seconds, numerical=compare(answer, expected, tolerances))
    del preserved, expected, answer

    # Reuse metadata and storage with new values on every shape, including scored batches.
    replacement = build_inputs(case, seed, seed + 29)
    replace_values_in_place(inputs, replacement)
    expected = reference(replacement)
    answer = call_native(implementation, inputs, scheduler)
    torch.cuda.synchronize()
    result['same_storage'] = compare(answer, expected, tolerances)
    del answer, expected
    if case['throughput_weight'] > 0:
        result.update(benchmark_sequence(implementation, inputs, replacement, scheduler, seed, tolerances))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--implementation', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case')
    arguments = parser.parse_args()
    sys.path.insert(0, str(arguments.implementation.resolve()))
    import flash_mla
    if 'B200' not in torch.cuda.get_device_name():
        raise RuntimeError('This workload requires one NVIDIA B200')
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'utils/manifest.json').read_text())
    tolerances = [manifest['output_tolerance'], manifest['lse_tolerance']]
    import_module('torch._inductor.config')
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(manifest['warmup_seconds'])
    with torch.inference_mode():
        for case in [manifest['cases'][14], manifest['cases'][18]]:
            inputs = build_inputs(case, arguments.seed + 1, arguments.seed + 2)
            scheduler, _ = flash_mla.get_mla_metadata()
            call_native(flash_mla, inputs, scheduler)
        torch.cuda.synchronize()
        del inputs, scheduler
        torch.cuda.empty_cache()
        signal.alarm(manifest['workload_seconds'])
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        report = dict(cases={}, torch=torch.__version__, gpu=torch.cuda.get_device_name(), seed=arguments.seed)
        for case_id, case in enumerate(manifest['cases']):
            if arguments.case and case['name'] != arguments.case:
                continue
            print('Running', case['name'], flush=True)
            seed = arguments.seed + case_id * 100003
            result = benchmark_case(flash_mla, case, seed, tolerances)
            report['cases'][case['name']] = result
            arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False))
            print(case['name'], result, flush=True)
        report['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
        report['workload_seconds'] = time.monotonic() - started
        signal.alarm(0)
        arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
