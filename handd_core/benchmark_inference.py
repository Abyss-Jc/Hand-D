"""Opt-in HD-12 CPU deployment spike. Does not change the Active Model.

Run with temporary ONNX dependencies:
  uv run --frozen --with onnx --with onnxruntime --with onnxscript \\
    python -m handd_core.benchmark_inference --candidate <verified-dir>
"""
from __future__ import annotations

import argparse
from importlib import metadata
from contextlib import redirect_stdout
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
import platform
import sys

import numpy as np
import torch

from handd_core.model_artifact import load_model_artifact


def _ms_stats(samples: list[float]) -> dict:
    return {'median':round(float(median(samples)),5),
            'p95':round(float(np.percentile(samples,95)),5)}


def _package_size(name: str) -> int | None:
    """Installed file sizes, NOT a measured compressed/packaged sidecar size."""
    try:
        package=metadata.distribution(name)
        paths=[package.locate_file(path) for path in package.files or ()]
        return sum(p.stat().st_size for p in paths if p.is_file())
    except (metadata.PackageNotFoundError, OSError):
        return None


def compare_backends(candidate_dir: str | Path, *, batch_sizes: tuple[int,...]=(1,8,32),
                     warmup: int=20, iterations: int=100, seed: int=42) -> dict:
    """Verify artifact, export its model, then measure strictly comparable CPU logits.

    A seeded synthetic 69-feature input is repeatable but NOT real hand data.
    Runtime/media processing, Tauri IPC and GPU providers are not measured.
    """
    if (not isinstance(batch_sizes,tuple) or not batch_sizes
            or any(type(i) is not int or i<1 or i>2048 for i in batch_sizes)
            or len(set(batch_sizes))!=len(batch_sizes)
            or type(warmup) is not int or warmup<0 or warmup>100000
            or type(iterations) is not int or not 1<=iterations<=100000
            or type(seed) is not int):
        raise ValueError('batch_sizes, warmup, iterations and seed must be valid')
    root=Path(candidate_dir)
    started=perf_counter()
    predictor,manifest=load_model_artifact(root)
    torch_load_ms=(perf_counter()-started)*1000
    try:
        import onnx
        import onnxruntime as ort
        import onnxscript
    except ImportError as error:
        raise RuntimeError('ONNX benchmark requires optional onnx and onnxruntime; '
                           'use uv run --frozen --with onnx --with onnxruntime --with onnxscript') from error

    np_rng=np.random.default_rng(seed)
    original_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with TemporaryDirectory(prefix='handd-hd12-') as folder:
            output=Path(folder)/'model.onnx'
            export_start=perf_counter()
            # Export a dynamic batch contract using the current torch.export
            # ONNX exporter. Batch=2 avoids specializing the dynamic axis to 1.
            # The modern exporter logs progress to stdout. Reserve stdout
            # for a machine-readable report when invoked as a CLI.
            with redirect_stdout(sys.stderr):
                torch.onnx.export(
                    predictor.model, (torch.zeros(2,69,dtype=torch.float32),), str(output),
                    input_names=['features'],output_names=['logits'],
                    dynamic_shapes=({0:torch.export.Dim('batch',min=1,max=2048)},),
                    opset_version=18,dynamo=True,
                )
            export_ms=(perf_counter()-export_start)*1000
            onnx.checker.check_model(str(output))
            opts=ort.SessionOptions()
            opts.intra_op_num_threads=1
            opts.inter_op_num_threads=1
            init_start=perf_counter()
            session=ort.InferenceSession(str(output),opts,
                                         providers=['CPUExecutionProvider'])
            ort_startup_ms=(perf_counter()-init_start)*1000
            providers=session.get_providers()
            if providers!=['CPUExecutionProvider']:
                raise RuntimeError(f'Expected explicit CPUExecutionProvider, got {providers}')
            rows=[]
            max_error=0.0
            mismatches=0
            for batch_size in batch_sizes:
                features=np_rng.normal(0,0.35,(batch_size,69)).astype(np.float32)
                reference=predictor(features)
                candidate=session.run(['logits'],{'features':features})[0]
                if reference.shape!=candidate.shape:
                    raise ValueError(f'ONNX shape mismatch {reference.shape} != {candidate.shape}')
                max_error=max(max_error,float(np.max(np.abs(reference-candidate))))
                mismatches+=int(np.count_nonzero(np.argmax(reference,axis=1)
                                                 !=np.argmax(candidate,axis=1)))
                def torch_run():
                    predictor(features)
                def ort_run():
                    session.run(['logits'],{'features':features})
                for _ in range(warmup):
                    torch_run();ort_run()
                latencies={}
                for label,fn in [('torch',torch_run),('onnxruntime',ort_run)]:
                    elapsed=[]
                    for _ in range(iterations):
                        tick=perf_counter()
                        fn()
                        elapsed.append((perf_counter()-tick)*1000)
                    latencies[label]=_ms_stats(elapsed)
                rows.append({'batch_size':batch_size,**latencies})
            graph_size=output.stat().st_size
            # torch.export stores tensors in model.onnx.data. Counting only
            # the .onnx graph dramatically undercounts deployable bytes.
            external_size=sum(p.stat().st_size for p in output.parent.glob(
                output.name+'.*') if p.is_file())
            output_size=graph_size+external_size
    finally:
        torch.set_num_threads(original_threads)
    result='equivalent' if max_error<=1e-4 and mismatches==0 else 'mismatch'
    return {
        'result':result,
        'exporter':'torch.export dynamo=True opset 18',
        'contract':{'input_features':manifest['input_features'],
                    'feature_transform':manifest['feature_transform'],
                    'architecture':manifest['architecture'],
                    'label_order':manifest['label_order'],
                    'weights_sha256':manifest['weights_sha256']},
        'input_source':f'seeded_synthetic_float32_69_seed_{seed}',
        'environment':{'os':platform.platform(),'architecture':platform.machine(),
                       'python':platform.python_version(),'torch':torch.__version__,
                       'onnx':onnx.__version__,'onnxruntime':ort.__version__,
                       'onnxscript':onnxscript.__version__,
                       'torch_cpu_threads':1},
        'onnxruntime':{'providers':providers},
        'numerical':{'max_abs_logit_error':max_error,'prediction_mismatches':mismatches,
                     'absolute_tolerance':1e-4},
        'latency_ms':rows,
        'startup_ms':{'torch_artifact_load':round(torch_load_ms,3),
                      'onnx_session':round(ort_startup_ms,3),
                      'export_one_time':round(export_ms,3)},
        'footprint_bytes':{'candidate_weights':(root/'weights.pth').stat().st_size,
                           'onnx_model':output_size,
                           'onnx_graph':graph_size,
                           'onnx_external_data':external_size,
                           'torch_distribution_installed':_package_size('torch'),
                           'onnxruntime_distribution_installed':_package_size('onnxruntime')},
        'limitations': ['CPU synthetic model-only latency, not end-to-end gesture latency',
                        'Installed package file sizes are not frozen sidecar sizes',
                        'No camera, GPU, macOS/Windows, or human-gesture accuracy assessed'],
    }


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description='HD-12 isolated PyTorch/ONNX CPU benchmark')
    parser.add_argument('--candidate',required=True,type=Path,
                        help='verified Candidate folder containing manifest.json')
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args(argv)
    result=compare_backends(args.candidate,warmup=args.warmup,
                            iterations=args.iterations,seed=args.seed)
    print(json.dumps(result,indent=2,sort_keys=True))
    return 0 if result['result']=='equivalent' else 2


if __name__=='__main__':
    raise SystemExit(main())
