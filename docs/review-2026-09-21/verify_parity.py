#!/usr/bin/env python3
"""Check exact ordinary proof bytes across feature configurations on Linux."""
import fcntl, hashlib, json, os, subprocess, tarfile, time
from pathlib import Path
root=Path('/root/atlas-consolidation')
with (root/'cargo-validation.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    source=root/('ordinary-parity-'+str(time.time_ns()));source.mkdir()
    with tarfile.open(root/'ordinary.tar.gz') as archive: archive.extractall(source)
    subprocess.run(['git','apply',str(root/'ordinary.patch')],cwd=source,check=True)
    manifest=json.loads((root/'ordinary.manifest.json').read_text())
    for filename,expected in manifest['files'].items():
        assert hashlib.sha256((source/filename).read_bytes()).hexdigest()==expected,filename
    logs=root/'logs'/source.name;logs.mkdir()
    env=os.environ.copy();env.update(PATH='/root/.cargo/bin:'+env['PATH'],CARGO_TARGET_DIR=str(root/'target'),RUSTFLAGS='-D warnings',RAYON_NUM_THREADS='8',CARGO_BUILD_JOBS='16',CARGO_PROFILE_DEV_DEBUG='0',CARGO_PROFILE_TEST_DEBUG='0')
    clean=['cargo','+1.95.0','clean']
    for p in ['common','atlas-onnx-tracer','joltworks','jolt-atlas-core','jolt-atlas']:clean+=['-p',p]
    with (logs/'clean.log').open('w') as out:subprocess.run(clean,cwd=source,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
    record=dict(source_head=manifest['head'],authenticated_files=len(manifest['files']),commands=[],proof_hashes={})
    variants=[('parallel',[]),('serial',['--no-default-features']),('optional',['--no-default-features','--features','affine-msm,check-fixed-tables'])]
    for name,features in variants:
        output=logs/name
        cmd=['cargo','+1.95.0','run','--profile','test','-p','jolt-atlas-core','--example','verifier_feature_parity']+features+['--',str(output)]
        started=time.time()
        with (logs/(name+'.log')).open('w') as out:
            p=subprocess.run(cmd,cwd=source,env=env,stdout=out,stderr=subprocess.STDOUT,timeout=1800)
        record['commands'].append(dict(command=cmd,exit_code=p.returncode,seconds=time.time()-started))
        if p.returncode==0:
            record['proof_hashes'][name]={f:hashlib.sha256((output/f).read_bytes()).hexdigest() for f in ['proof.bin','compact.bin','io.json']}
        print(name,p.returncode,flush=True)
        (logs/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
    record['identical']=len(record['proof_hashes'])==3 and len({json.dumps(x,sort_keys=True) for x in record['proof_hashes'].values()})==1
    (logs/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
    print('parity',record['identical'],str(logs),flush=True)
