#!/usr/bin/env python3
"""Verify tracked source files after validation, including Cargo.lock."""
import hashlib,json,sys
from pathlib import Path
root=Path('/root/atlas-consolidation')
manifest=json.loads((root/'audit-inputs.json').read_text())
result={}
for name,record in manifest.items():
    errors=[]
    for filename,digest in record['files'].items():
        p=root/name/filename
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:errors.append(filename)
    result[name]=dict(head=record['head'],authenticated_files=len(record['files']),changed_files=errors)
(root/'after-validation-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
assert not any(x['changed_files'] for x in result.values())
