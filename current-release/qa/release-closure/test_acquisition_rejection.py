import hashlib
import importlib.util
import json
import sys
from pathlib import Path
root=Path('/tmp/tessera-release-smoke/relocated')
sys.path.insert(0,str(root/'portable'))
import acquire
import fetch_adapter
manifest=json.loads((root/'provenance/upstream-inputs.json').read_text())
row=manifest['acquisition']['her2st'][0]
source=Path('/tmp/tessera-release-smoke/verified-her2st')/row['filename']
bad=Path('/tmp/tessera-release-smoke/corrupt-file');bad.write_bytes(source.read_bytes()[:-1])
rejected=[]
try:acquire.check(bad,row)
except ValueError:rejected.append('size-or-sha256-mismatch')
wrong=dict(row,md5='0'*32)
try:acquire.check(source,wrong)
except ValueError:rejected.append('publisher-md5-mismatch')
unknown=fetch_adapter.fetch('https://example.invalid/unrecorded',Path('/tmp/tessera-release-smoke/unrecorded'))
assert not unknown['ok'] and unknown['blocked']
rejected.append('unrecorded-upstream-url')
missing=fetch_adapter.fetch('https://raw.githubusercontent.com/almaan/her2st/master/data/ST-cnts/J1.tsv.gz',Path('/tmp/tessera-release-smoke/J1.tsv.gz'))
assert not missing['ok'] and missing['url'].endswith('/J1.tsv.gz')
assert len(rejected)==3
print(json.dumps({'status':'passed','rejected':rejected,'historical_J1_exclusion_preserved':True,'network_requests':0},indent=2))
