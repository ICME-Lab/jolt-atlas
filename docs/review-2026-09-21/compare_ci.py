from pathlib import Path
from collections import Counter
import datetime,json,re

def errors(path):
 s=Path(path).read_text()
 s=re.sub(r'\x1b\[[0-9;]*m','',s)
 s=re.sub(r'^\d{4}-\d\d-\d\dT\S+\s+','',s,flags=re.M)
 s=s.replace('##[error]','')
 # Lib and lib-test diagnostics can render the same types with different qualifiers.
 s=s.replace('onnx_proof::clamp_lookups::SaturatingAccClampOperands','SaturatingAccClampOperands').replace('std::default::Default','Default')
 return Counter((m.group(1),m.group(2)) for m in re.finditer(r'^(error(?:\[[^\]]+\])?: [^\n]+)\n\s*--> ([^\n]+)',s,re.M))
base=errors('linux-logs/baseline-ci-v2/00.log')
parent=errors('linux-logs/correctness-ci-v2/03.log')
results={}
for n in [369,370]:
 c=errors(f'ci-pr-{n}-zk.log')
 results[str(n)]=dict(ci_primary_diagnostics=sum(c.values()),unchanged_round_two_base_primary_diagnostics=sum(base.values()),added_over_base=list((c-base).items()),removed_from_base=list((base-c).items()),added_over_correctness_core_clippy=list((c-parent).items()),removed_from_correctness_core_clippy=list((parent-c).items()))
print(json.dumps(results,indent=2))
Path('ci-zk-diagnostics-comparison.json').write_text(json.dumps(dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),base='cfcffb05dcc76599f4e5b3d01c6e13f5fa653e13',scope='Compare every primary title and source location, normalizing only onnx_proof::clamp_lookups::SaturatingAccClampOperands to SaturatingAccClampOperands and std::default::Default to Default. Raw spelling differences are preserved separately. CI/local command differences remain in raw logs.',prs=results),indent=2)+'\n')
assert all(x['ci_primary_diagnostics']==41 and not x['added_over_base'] and not x['added_over_correctness_core_clippy'] for x in results.values())
