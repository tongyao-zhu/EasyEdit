"""Finite single-GPU diagnostic allocation with a persistent resource ledger."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--python', required=True)
    p.add_argument('--hostname', required=True)
    p.add_argument('--seconds', type=int, default=1800)
    a = p.parse_args()
    if not 1 <= a.seconds <= 3600 or socket.gethostname() != a.hostname:
        raise RuntimeError('Invalid resource policy or node')
    a.root.mkdir(exist_ok=False)
    ledger = a.root / 'resources.jsonl'
    def event(data):
        data['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with ledger.open('a') as f:
            f.write(json.dumps(data) + '\n'); f.flush(); os.fsync(f.fileno())
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used',
                                   '--format=csv,noheader,nounits'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    first = gpu.splitlines()[0].split(', ')
    if first[0] != '0' or 'A100' not in first[2] or int(first[3]) > 100 or first[1] in apps:
        event({'event': 'preflight_failed', 'gpu': gpu, 'apps': apps})
        raise RuntimeError('GPU 0 is not idle or not the authorized A100')
    base = Path(__file__).resolve().parents[1]
    model = json.loads((base.parent / 'model_location.json').read_text())['path']
    command = [a.python, '-m', 'ouro_integration.validate', '--model', model,
               '--request', str(base.parent / 'atomic_request.json'),
               '--model-manifest', str(base.parent / 'model_manifest.json'),
               '--output', str(a.root / 'validation')]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    started = time.monotonic(); status = 'failed_to_start'; child = None
    event({'event': 'open', 'GPU_count': 1, 'GPU_index': 0, 'uuid': first[1],
           'host': socket.gethostname(), 'max_seconds': a.seconds, 'command': command})
    try:
        with (a.root / 'worker.log').open('x') as log:
            child = subprocess.Popen(command, cwd=base, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            event({'event': 'started', 'pid': child.pid})
            try:
                code = child.wait(timeout=a.seconds)
                status = 'completed' if code == 0 else 'failed'
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL); child.wait()
                status, code = 'watchdog_stopped', child.returncode
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL); child.wait()
        elapsed = time.monotonic() - started
        event({'event': 'close', 'status': status, 'seconds': elapsed,
               'GPU_hours': elapsed / 3600, 'returncode': child.returncode if child else None})
        (a.root / 'DONE.json').write_text(json.dumps({'status': status, 'GPU_hours': elapsed / 3600}) + '\n')


if __name__ == '__main__':
    main()
