#!/usr/bin/env python3
"""Small official DOI-metadata reads; no assay downloads or dependency changes."""
import json
import re
import urllib.parse
import urllib.request
from common import ROOT,dump

DOIS={'he2022cosmx':'10.1038/s41587-022-01483-z',
      'janesick2023breast':'10.1038/s41467-023-43458-x',
      'sun2026consensus':'10.1038/s41592-026-03194-8',
      'crowell2025tva':'10.1101/2025.06.23.660674',
      'yerly2024wound':'10.1101/2024.05.31.596823',
      'moses2022museum':'10.1038/s41592-022-01409-2',
      'rao2021architecture':'10.1038/s41586-021-03634-9',
      'hu2021spagcn':'10.1038/s41592-021-01255-8',
      'long2023graphst':'10.1038/s41467-023-36796-3',
      'ren2022spaceflow':'10.1038/s41467-022-31739-w',
      'shang2022spatialpca':'10.1038/s41467-022-34879-1',
      'yuan2024mender':'10.1038/s41467-023-44367-9',
      'yang2025staig':'10.1038/s41467-025-56276-0',
      'liang2026cellniche':'10.1038/s41467-026-71759-4',
      'zimmerman2021pseudoreplication':'10.1038/s41467-021-21038-1',
      'roberts2017crossvalidation':'10.1111/ecog.02881'}

def escape(s):
    return str(s).replace('&',r'\&').replace('_',r'\_').replace('\u2010','-').replace('\u2011','-')

def main():
    records={};entries=[]
    out=ROOT/'source/manuscript'
    all_dois=dict(DOIS)
    for key,body in re.findall(r'@\w+\{([^,]+),\s*(.*?)(?=\n@|\Z)',(out/'base_refs.bib').read_text(),re.S):
        match=re.search(r'doi=\{([^}]+)\}',body)
        if match: all_dois[key]=match.group(1)
    for key,doi in all_dois.items():
        req=urllib.request.Request('https://api.crossref.org/works/'+urllib.parse.quote(doi,safe=''),
            headers={'User-Agent':'Tessera-local-reference-audit/1.0'})
        with urllib.request.urlopen(req,timeout=30) as r: message=json.load(r)['message']
        records[key]=message
        title=message['title'][0];year=message.get('published',message.get('created'))['date-parts'][0][0]
        authors=message.get('author',[])[:3]
        names=['{}, {}'.format(a.get('family',''),a.get('given','')) for a in authors]
        if len(message.get('author',[]))>3:names.append('others')
        typ='misc' if message.get('type')=='posted-content' else 'article'
        fields={'title':'{'+escape(title)+'}','author':' and '.join(names),'year':year,'doi':doi}
        if typ=='article':
            fields['journal']=escape(message['container-title'][0])
            for k in ['volume','issue','page','article-number']:
                if k in message:fields[{'issue':'number','page':'pages','article-number':'pages'}.get(k,k)]=message[k]
        else: fields.update(howpublished='bioRxiv',
                            note='Preprint; cited for source study, not independent validation. DOI: '+doi)
        if key in DOIS:
            entries.append('@'+typ+'{'+key+',\n'+',\n'.join('  '+k+'={'+str(v)+'}' for k,v in fields.items())+'\n}')
        print(key,title,year,flush=True)
    dump(out/'ledgers/reference_metadata.json',records)
    dump(out/'ledgers/reference_audit.json',{
        'checked_date':'2026-10-11','metadata_authority':'Crossref publisher deposits',
        'doi_count':len(records),'new_related_work_references':11,
        'limits':'Metadata identity is not scientific validation. Source-study preprints remain preprints; no unsupported journal upgrade.',
        'identities':{k:{'doi':v['DOI'],'title':v['title'][0],
            'published':v.get('published',v.get('created'))['date-parts'][0],
            'type':v['type']} for k,v in records.items()}})
    (out/'new_refs.bib').write_text('\n\n'.join(entries)+'\n')

if __name__=='__main__':main()
