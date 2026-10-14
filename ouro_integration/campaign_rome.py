"""Finite, no-retry campaign: two fixed shards after separately completed acceptance."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from .benchmark import verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
    spec=verify(r)['spec']
    assert socket.gethostname()==spec['hostname']
    with (r/'LAUNCH_LOCK').open('x') as f:f.write(str(os.getpid()))
    ledger=r/'resources.jsonl';active={};opened=[];closed=[];global_start=time.monotonic()
    def event(row):
        row['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        with ledger.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    def idle(indices):
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.used','--format=csv,noheader,nounits'],text=True)
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        devices={int(x.split(', ')[0]):x.split(', ') for x in rows.splitlines()}
        for i in indices:
            dev=devices[i]
            assert dev[1]==spec['gpu_uuids'][i] and 'A100' in dev[2] and int(dev[3])<=100 and dev[1] not in apps
    def start(name,gpu,module,extra,limit):
        import sys
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 HF_MODULES_CACHE=str(r/'hf_modules'/name),PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
        command=[sys.executable,'-m',module,'--root',str(r)]+extra
        t=time.monotonic();event({'event':'open','job':name,'GPU_count':1,'GPU_index':gpu,'limit_seconds':limit,'command':command});opened.append(name)
        try:
            with (r/f'{name}.log').open('x') as log:
                child=subprocess.Popen(command,cwd=r/'repository',env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        except BaseException:
            elapsed=time.monotonic()-t;event({'event':'close','job':name,'status':'launch_failed','seconds':elapsed,'GPU_hours':elapsed/3600});closed.append(name);raise
        active[name]=(child,t,limit)
        event({'event':'started','job':name,'pid':child.pid})
    def stop(child):
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    def close(name,status):
        child,t,_=active.pop(name);elapsed=time.monotonic()-t
        event({'event':'close','job':name,'status':status,'returncode':child.returncode,'seconds':elapsed,'GPU_hours':elapsed/3600});closed.append(name)
    result='failed';error=None
    try:
        assert json.loads((r/'acceptance/entry_report.json').read_text())['status']=='passed_single_case_no_mom2_integration_only'
        assert json.loads((r/'acceptance/entry_DONE.json').read_text())['status']=='completed'
        deadline=datetime.datetime.fromisoformat(spec['global_authorized_deadline_utc'])
        if (deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds()<spec['resource_policy']['seconds_per_science_shard']+30:
            raise RuntimeError('Insufficient remaining node authorization for frozen batch')
        idle([0,1])
        for i in [0,1]:start(f'science{i}',i,'ouro_integration.benchmark_rome',['--shard',str(i)],spec['resource_policy']['seconds_per_science_shard'])
        failed=False
        while active:
            for name,(child,t,limit) in list(active.items()):
                if child.poll() is not None:
                    failed |= child.returncode!=0
                    close(name,'completed' if child.returncode==0 else 'failed')
                elif time.monotonic()-t>=limit or time.monotonic()-global_start>=spec['resource_policy']['global_seconds'] or datetime.datetime.now(datetime.timezone.utc)>=deadline:
                    stop(child);close(name,'watchdog_stopped');failed=True
            if failed:
                for name,(child,_,_) in list(active.items()):stop(child);close(name,'stopped_after_peer_failure')
                raise RuntimeError('Scientific worker failed or reached frozen deadline; no retry')
            if active:time.sleep(2)
        result='completed'
    except BaseException as exc:error=repr(exc)
    finally:
        for name,(child,_,_) in list(active.items()):stop(child);close(name,'manager_cleanup')
        assert sorted(opened)==sorted(closed)
        rows=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
        total=sum(x['GPU_hours'] for x in rows if x['event']=='close')
        (r/'DONE.json').write_text(json.dumps({'status':result,'error':error,'GPU_hours':total,'opened':opened,'closed':closed,'elapsed_seconds':time.monotonic()-global_start},indent=2)+'\n')
        # Postprocess only after every GPU allocation has closed; never GPU imports here.
        import sys
        with (r/'analysis.log').open('x') as log:
            code=subprocess.call([sys.executable,'-m','ouro_integration.analyze_rome','--root',str(r)],cwd=r/'repository',
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT)
        (r/'POSTPROCESS_STATUS.json').write_text(json.dumps({'returncode':code})+'\n')


if __name__=='__main__':main()
