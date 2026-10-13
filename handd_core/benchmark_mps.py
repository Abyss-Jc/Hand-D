"""Opt-in macOS Metal benchmark. No webcam, model mutation or workspace writes.

uv run --frozen python -m handd_core.benchmark_mps --candidate /path/to/candidate
uv run --frozen python -m handd_core.benchmark_mps  # diagnostic legacy model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
from statistics import median
from time import perf_counter

import numpy as np
import torch

from handd_core.legacy_model import load_legacy_checkpoint
from handd_core.model_artifact import load_model_artifact
from handd_core.mps_runtime import prepare_mps_predictor


def main(argv=None) -> int:
    cli=argparse.ArgumentParser(description='Hand-D experimental PyTorch CPU / MPS benchmark')
    cli.add_argument('--candidate',type=Path,help='verified Model Artifact directory')
    cli.add_argument('--legacy-checkpoint',type=Path,
                     default=Path('models/gesture_mlp.pth'),
                     help='unverified diagnostic checkpoint used if --candidate omitted')
    cli.add_argument('--iterations',type=int,default=100)
    cli.add_argument('--warmup',type=int,default=20)
    args=cli.parse_args(argv)
    if not 1<=args.iterations<=2000 or not 0<=args.warmup<=500:
        cli.error('iterations must be 1..2000 and warmup must be 0..500')
    status={
        'mps':{'built':bool(torch.backends.mps.is_built()),
               'available':bool(torch.backends.mps.is_available())},
        'platform':platform.platform(),
        'architecture':platform.machine(),
        'torch':torch.__version__,
        'scope':'model-only float32 synthetic features; not webcam, FPS or accuracy',
    }
    if not status['mps']['available']:
        print(json.dumps({'status':'unavailable',**status},indent=2))
        return 2
    try:
        reference,_=(load_model_artifact(args.candidate) if args.candidate
                     else load_legacy_checkpoint(args.legacy_checkpoint))
        device=prepare_mps_predictor(reference)
    except (RuntimeError,ValueError,OSError) as error:
        print(json.dumps({'status':'failed',**status,
                          'error':str(error)[:300]},indent=2))
        return 2
    rows=[]
    max_error=0.0
    mismatches=0
    rng=np.random.default_rng(2026)
    for batch_size in (1,8,32):
        points=rng.normal(0,.35,(batch_size,69)).astype(np.float32)
        truth=reference(points)
        accelerated=device(points)
        max_error=max(max_error,float(np.max(np.abs(truth-accelerated))))
        mismatches+=int(np.count_nonzero(np.argmax(truth,axis=1)
                                        !=np.argmax(accelerated,axis=1)))
        results={}
        for name,predictor in [('cpu',reference),('mps',device)]:
            for _ in range(args.warmup):predictor(points)
            durations=[]
            for _ in range(args.iterations):
                start=perf_counter()
                predictor(points) # output is already on CPU, so MPS is synchronized
                durations.append((perf_counter()-start)*1000)
            results[name]={
                'median_ms':round(float(median(durations)),5),
                'p95_ms':round(float(np.percentile(durations,95)),5),
            }
        rows.append({'batch_size':batch_size,**results})
    passed=max_error<=1e-4 and mismatches==0
    print(json.dumps({
        'status':'equivalent' if passed else 'mismatch',
        **status,'model_kind':'verified_candidate' if args.candidate else 'legacy_unverified',
        'labels':list(reference.label_order),
        'max_abs_logit_error':max_error,
        'prediction_mismatches':mismatches,
        'latency_ms':rows,
    },indent=2))
    return 0 if passed else 2


if __name__=='__main__':
    raise SystemExit(main())
