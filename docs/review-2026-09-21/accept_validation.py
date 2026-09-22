#!/usr/bin/env python3
"""Gate publication on preserved, scoped validation and exact source identities."""
import datetime, json, subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
selection=json.loads((root/'selected-validation.json').read_text())
branches=['tensor-correctness','ordinary-verifier','native-foundations','native-operators','native-generation','native-prover','native-verifier']
names=['correctness','ordinary','foundation','operators','generation','prover','verifier']
expected={'correctness':[0,0,101,101,0,101], 'ordinary':[0,0,0,0,0,0,101,0,0,101], 'foundation':[0]*6, 'operators':[0]*6,'generation':[0]*4,'prover':[0]*6,'verifier':[0]*6}
heads={b:subprocess.check_output(['git','rev-parse','review/'+b],cwd=root/'repo',text=True).strip() for b in branches}
for name,branch in zip(names,branches):
 d=json.loads((root/'linux-logs'/selection[name]/'summary.json').read_text())
 assert [c['exit_code'] for c in d['commands']]==expected[name],name
 assert d['local_crates_cleaned'],name
 if name=='ordinary':
  assert d['source_head']=='d190e1419f5f1231818e69253cab5b7dda3d7165'
  delta=subprocess.check_output(['git','diff',d['source_head'],heads[branch]],cwd=root/'repo')
  assert delta==(root/'ordinary-final-backend-fixes.patch').read_bytes()
 else:assert d['source_head']==heads[branch],name
for name,count in [('ordinary-backends-v2',5),('ordinary-parity-1790007702185215279',3),('ordinary-fixed-tables-final',1)]:
 d=json.loads((root/'linux-logs'/name/'summary.json').read_text())
 assert d['source_head']==heads['ordinary-verifier']
 assert len(d['commands'])==count and all(c['exit_code']==0 for c in d['commands']),name
 if 'parity' in name:assert d['identical'] and len(d['proof_hashes'])==3
for name in ['zk-diagnostics-clean-comparison.json','ordinary-zk-diagnostics-comparison.json','ordinary-zk-test-diagnostics-comparison.json']:
 d=json.loads((root/name).read_text());assert not d['added'],name
core=root/'linux-logs/foundation-core-zk-final'
assert '127 passed; 26 failed; 29 ignored;' in (core/'00.log').read_text()
audit=json.loads((root/'after-validation-audit.json').read_text());assert len(audit)==12 and not any(d['changed_files'] for d in audit.values())
for name,data in json.loads((root/'audit-inputs.json').read_text()).items():
 assert audit[name]['head']==data['head'] and audit[name]['authenticated_files']==len(data['files'])
for i in range(1,8):
 s=(root/'bodies'/f'{i:02d}.md').read_text();assert 'Pending exact' not in s and 'being collected' not in s
result=dict(ready_for_publication=True,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),heads=heads,selection=selection,limitations=['Experimental ONNX ZK dispatcher has recorded compile/runtime failures; native API tests have their own passing scope.','Performance observations are preserved component comparisons, not a newly measured complete-model consolidation result.','Ordinary matrix and final backend cleanup heads are distinguished with exact diff and supplementary final-head checks.'])
(root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
print('Validated publication gate for seven review units and twelve source snapshots.')
