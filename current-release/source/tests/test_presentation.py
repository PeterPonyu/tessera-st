"""Scientific and document regressions for the presentation-only revision."""
from datetime import date
from pathlib import Path
import json
import re
import sys
import xml.etree.ElementTree as ET
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'source/scripts'))
from common import ARMS, OBJECTS, SEEDS, read, sha
from build_revision import figure_blocks, PANELS

MAN = ROOT / 'source/manuscript'
BASELINE = read(ROOT / 'source/tests/fixtures/presentation_baseline.json')['sha256']


@pytest.mark.parametrize('path', ['results/run-lock.json', 'results/transfer/run-lock.json'])
def test_frozen_estimator_and_protocol_identities(path):
    for filename, expected in read(ROOT / path)['code_sha256'].items():
        assert sha(ROOT / filename) == expected, filename


def test_every_primary_prediction_remains_bound_to_input():
    count = 0
    for obj in OBJECTS:
        meta = read(ROOT / 'data' / obj / 'input.json')
        assert sha(ROOT / 'data' / obj / 'input.npz') == meta['matrix_sha256']
        for arm in ARMS:
            for seed in SEEDS:
                stem = ROOT / 'results' / obj / f'{arm}-seed{seed}'
                rec = read(stem.with_suffix('.json'))
                assert rec['input_sha256'] == meta['matrix_sha256']
                assert sha(stem.with_suffix('.npz')) == rec['prediction_sha256']
                count += 1
    assert count == 351


def test_every_transfer_prediction_transform_and_weight_is_bound():
    count = 0
    for obj in OBJECTS[:5]:
        for arm in ['PCA50', 'SelfOnly', 'Tessera']:
            for seed in SEEDS:
                stem = ROOT / 'results/transfer' / f'{obj}-{arm}-seed{seed}'
                rec = read(stem.with_suffix('.json'))
                assert sha(stem.with_suffix('.npz')) == rec['prediction_sha256']
                assert sha(stem.parent / (obj + '-transform.npz')) == rec['transform_sha256']
                if arm != 'PCA50':
                    assert sha(stem.with_suffix('.pt')) == rec['weight_sha256']
                assert rec['patient_disjoint'] and not rec['target_transform_fitted']
                assert obj not in rec['source_patients'] and len(rec['source_patients']) == 4
                count += 1
    assert count == 45


@pytest.mark.parametrize('filename', ['all_primary_scores.csv', 'all_transfer_scores.csv', 'primary_summary.csv', 'seed_stability.csv'])
def test_numerical_ledgers_unchanged(filename):
    assert sha(MAN / 'ledgers' / filename) == BASELINE[filename]


@pytest.mark.parametrize('filename', ['manuscript_numbers.json', 'patient_contrasts.json'])
def test_reported_numbers_and_patient_intervals_unchanged(filename):
    assert sha(MAN / 'ledgers' / filename) == BASELINE[filename]


def test_citations_resolve_and_reference_timeline_is_not_future():
    text = (MAN / 'paper.tex').read_text()
    citations = set(k.strip() for group in re.findall(r'\\cite\w*\{([^}]+)\}', text) for k in group.split(','))
    bib = '\n'.join((MAN / f).read_text() for f in ['base_refs.bib', 'new_refs.bib'])
    keys = re.findall(r'@\w+\{([^,]+),', bib)
    assert len(keys) == len(set(keys))
    assert citations <= set(keys)
    audit = read(MAN / 'ledgers/reference_audit.json')
    assert audit['doi_count'] == 34 and audit['new_related_work_references'] == 11
    for key, item in audit['identities'].items():
        assert key in keys
        dt = item.get('published')
        if dt:
            assert date(*(dt + [1] * (3 - len(dt)))) <= date(2026, 10, 9), key
    assert audit['identities']['yuan2024mender']['published'][0] == 2024
    for key in ['yerly2024wound', 'crowell2025tva']:
        assert audit['identities'][key]['type'] == 'posted-content'


def test_introduction_and_discussion_have_comparable_substantial_lengths():
    text = (MAN / 'paper.tex').read_text()
    intro = text.split(r'\section{Introduction}')[1].split(r'\begin{figure}')[0]
    discussion = text.split(r'\section{Discussion}')[1].split(r'\section*')[0]
    a, b = len(intro.split()), len(discussion.split())
    assert a >= 500 and b >= 500  # concise after the 2026-10-10 major revision
    assert .75 <= b / a <= 1.35, (a, b)


def test_prose_word_counts_exclude_layout_comments():
    from render_document_qa import section_words
    text = (r'\section{Discussion}' + '\n'
            'Direct prose with measured evidence. % Hidden layout note\n'
            + r'\FloatBarrier' + '\n' + r'\section*{Availability}')
    assert section_words(text, 'Discussion', r'\section*') == 5


def test_every_figure_has_a_complete_caption_and_verified_pdf():
    entries = figure_blocks(MAN / 'paper.tex') + figure_blocks(MAN / 'supporting.tex')
    assert len(entries) == 21 and len({f for f, _ in entries}) == 21
    manifest = read(MAN / 'ledgers/figure_manifest.json')['figures']
    for filename, caption in entries:
        assert len(caption.split()) >= 25
        assert sha(MAN / 'figures' / filename) == manifest[filename]
    assert sum(PANELS[:12]) == 50 and sum(PANELS[12:]) == 25


def test_editable_schematics_have_declared_panel_structure():
    ns = {'s': 'http://www.w3.org/2000/svg'}
    for name, count in [('figure01-workflow', 6), ('figure13-architecture', 5), ('figure14-target-lineage', 5)]:
        svg = ET.parse(MAN / 'figures' / (name + '.svg')).getroot()
        labels = [e.text for e in svg.findall('.//s:text', ns)]
        assert {letter for letter in 'abcdef'[:count]} <= set(labels)
        assert len(svg.findall('.//s:text[@data-role="panel-label"]',ns))==count
        assert len(svg.findall('.//s:text[@data-role="panel-title"]',ns))==count
        for element in svg.findall('.//s:text',ns):
            role=element.get('data-role')
            assert element.get('font-weight')==('bold' if role=='panel-label' else 'normal')
            assert element.get('font-family')=='TeX Gyre Termes, Times New Roman, serif'
        caption=dict(figure_blocks(MAN/'paper.tex')+figure_blocks(MAN/'supporting.tex'))[name+'.pdf']
        assert 'Conceptual' in caption or 'conceptual' in caption or 'not claimed architectural' in caption


def test_methods_expose_implementation_critical_details():
    text = (MAN / 'paper.tex').read_text()
    assert len(re.findall(r'\\label\{eq:', text)) >= 26
    for phrase in ['multiset', 'without deduplication', 'stopgrad', 'v_{ji,k}',
                   'source-fitted', 'uncalibrated', 'untrained', 'query budget',
                   'other patients', 'sign-flip']:
        assert phrase in text, phrase


def test_panel_references_are_valid_and_confined_to_results_prose():
    text=(MAN/'paper.tex').read_text()
    results=text.split(r'\section{Results}')[1].split(r'\section{Discussion}')[0]
    rest=text.replace(results,'')
    # Command definitions do not count as panel references.
    assert not re.search(r'\\(?:sup)?panelref\{(?:fig:|S\d)',rest)
    labels={re.search(r'\\label\{([^}]+)\}',block).group(1):count
            for block,count in zip(re.findall(r'\\begin\{figure\}.*?\\end\{figure\}',text,re.S),PANELS[:12])}
    cited=set()
    for label,subset in re.findall(r'\\panelref\{([^}]+)\}\{([^}]+)\}',results):
        assert label in labels
        for letter in re.findall('[a-z]',subset):
            assert ord(letter)-ord('a')<labels[label],(label,subset)
        cited.add(label)
    assert len(cited)==10  # Every empirical main figure, not the two schematics.
    for number,subset in re.findall(r'\\suppanelref\{S(\d)\}\{([^}]+)\}',results):
        for letter in re.findall('[a-z]',subset):
            assert ord(letter)-ord('a')<PANELS[11+int(number)]
    assert not re.search(r'\\ref\{fig:[^}]+\}[A-Z]',text)


def test_rendered_statistical_figures_use_separate_regular_titles():
    audit=read(ROOT/'qa/statistical_figure_style_audit.json')
    assert len(audit)==18
    for name,record in audit.items():
        assert record['only_panel_labels_bold'] and record['separate_text_artists']
        assert record['labels']==[chr(97+i) for i in range(len(record['labels']))]
        assert len(record['labels'])==len(record['titles'])
        assert record['pdf_sha256']==sha(MAN/'figures'/(name+'.pdf'))
        assert record['method_color_contract']['Tessera']=='#bd5b30'
        assert record['method_color_contract']['NoContrast']=='#82618e'
        assert record['panel_label_size_points']>=13.5


def test_exported_figure_font_audit_is_bound_to_all_final_figures():
    audit=read(ROOT/'qa/figure_contract_audit.json')
    assert audit['figure_count']==21 and audit['panel_count']==75
    assert audit['all_only_panel_labels_bold']
    assert audit['all_text_black'] and audit['all_labels_larger_than_regular_text']
    for name,record in audit['figures'].items():
        assert record['pdf_sha256']==sha(MAN/'figures'/(name+'.pdf'))
        assert record['only_panel_labels_bold']
        assert record['all_text_black'] and record['labels_larger_than_regular_text']
        assert all(re.fullmatch(r'[a-z]',run) for run in record['bold_pdf_runs'])


def test_map_packing_preserves_geometry_and_aspect():
    import numpy as np
    from render_figures import plt,compact_map_grid
    fig,axes=plt.subplots(2,3,figsize=(7,5))
    coords=np.array([[0.,0.],[2.,1.],[3.,2.]])
    for ax in axes.ravel():
        ax.scatter(coords[:,0],coords[:,1]);ax.set_aspect('equal')
    before=[(ax.get_xlim(),ax.get_ylim()) for ax in axes.ravel()]
    compact_map_grid(fig,axes)
    assert [(ax.get_xlim(),ax.get_ylim()) for ax in axes.ravel()]==before
    for ax in axes.ravel():
        assert ax.get_aspect()==1.
        assert np.array_equal(ax.collections[0].get_offsets(),coords)
    plt.close(fig)


@pytest.mark.parametrize('path,expected',[
    ('M0 0 H30','M0 0 H18'),('M30 0 H0','M30 0 H12'),
    ('M0 0 V30','M0 0 V18'),('M0 30 V0','M0 30 V12'),
    ('M0 0 H20 V30','M0 0 H20 V18')])
def test_svg_arrow_shaft_stops_at_head_base_without_moving_tip(path,expected):
    from artwork import arrow_shaft
    assert arrow_shaft(path)==expected


def test_all_schematic_arrows_have_explicit_page_readable_head_geometry():
    ns={'s':'http://www.w3.org/2000/svg'}
    for name in ['figure01-workflow','figure13-architecture','figure14-target-lineage']:
        root=ET.parse(MAN/'figures'/(name+'.svg')).getroot()
        arrows=root.findall('.//s:g[@data-role="arrow"]',ns)
        assert arrows and not root.findall('.//s:marker',ns)
        scale=496.8/float(root.get('viewBox').split()[2])
        for arrow in arrows:
            path=arrow.find('s:path',ns);head=arrow.find('s:polygon',ns)
            assert float(arrow.get('data-head-length'))*scale==pytest.approx(4.396460177,abs=.001)
            assert float(arrow.get('data-head-width'))*scale>3
            assert head.get('fill')==path.get('stroke')
            assert float(path.get('stroke-width'))==2.8
            assert path.get('stroke-linecap')=='butt'
            tip=tuple(map(float,arrow.get('data-tip').split(',')))
            base=tuple(map(float,arrow.get('data-base').split(',')))
            corners=[tuple(map(float,p.split(','))) for p in head.get('points').split()]
            assert corners[1]==tip
            assert tuple((a+b)/2 for a,b in zip(corners[0],corners[2]))==base
            last=list(re.finditer(r'([MHV])([^MHV]+)',path.get('d')))[-1]
            index=0 if last[1]=='H' else 1
            assert float(last[2])==base[index]


def test_schematic_arrow_bodies_have_visible_terminal_legs():
    from artwork import workflow,architecture,lineage
    for drawing in [workflow(),architecture(),lineage()]:
        root=ET.fromstring(drawing)
        for path in root.findall('.//{http://www.w3.org/2000/svg}path[@data-role="arrow-body"]'):
            x=y=0
            for part in re.finditer(r'([MHV])([^MHV]+)',path.get('d')):
                before=(x,y);values=list(map(float,part[2].split()))
                if part[1]=='M':x,y=values
                elif part[1]=='H':x,=values
                else:y,=values
            index=0 if part[1]=='H' else 1
            assert abs((x,y)[index]-before[index])>=8


def test_explicit_seam_map_layout_preserves_points_and_sets_small_gaps():
    import numpy as np
    from render_figures import plt,compose_seam_and_map
    fig=plt.figure(figsize=(7.2,6.55));grid=fig.add_gridspec(3,1,height_ratios=[.88,1.5,.88])
    seam=fig.add_subplot(grid[1]);map_ax=fig.add_axes([.7,.4,.2,.2]);bar=fig.add_axes([.95,.4,.01,.2])
    coords=np.array([[0.,0.],[1.,2.],[2.,3.]])
    map_ax.scatter(coords[:,0],coords[:,1]);map_ax.set_aspect('equal');map_ax.invert_yaxis()
    before=(map_ax.get_xlim(),map_ax.get_ylim())
    compose_seam_and_map(fig,seam,map_ax,bar,grid[1].get_position(fig));map_ax.apply_aspect()
    assert (map_ax.get_xlim(),map_ax.get_ylim())==before
    assert np.array_equal(map_ax.collections[0].get_offsets(),coords)
    assert map_ax.get_aspect()==1.
    scale=fig.get_figwidth()*72
    assert (map_ax.get_position().x0-seam.get_position().x1)*scale==pytest.approx(14)
    assert (bar.get_position().x0-map_ax.get_position().x1)*scale==pytest.approx(5)
    plt.close(fig)


def test_symmetric_graph_counts_reciprocal_entries_with_multiplicity():
    import numpy as np
    from run_experiments import graph
    xy = np.array([[0., 0.], [1., 0.], [2., 0.]])
    edge, _ = graph(xy, np.array(['a', 'a', 'a']), 'Tessera', 1)
    unique, counts = np.unique(edge.T, axis=0, return_counts=True)
    assert edge.shape == (2, 12) and len(unique) == 6
    assert np.array_equal(counts, np.full(6, 2))
    assert np.array_equal(np.bincount(edge[1]), np.full(3, 4))


def test_homogeneity_denominators_and_seam_auroc_ties_match_methods():
    import numpy as np
    from run_experiments import scores
    truth=np.array([0,0,1,-1]);pred=np.array([0,1,1,1])
    edge=np.array([[0,0,1,1,2,2,3,3],[1,1,2,3,1,3,2,0]])
    result=scores(truth,pred,edge,gate=np.full(8,.5))
    assert result['reference_edge_homogeneity']==.5  # 2/4 valid entries
    assert result['predicted_edge_homogeneity']==.625  # 5/8 all entries
    assert result['gate_seam_auroc']==.5  # all scores tied


def test_figure_methods_map_covers_every_semantic_section_and_is_hash_bound():
    audit=read(ROOT/'qa/figure_methods_audit.json')
    assert audit['manuscript_sha256']==sha(MAN/'paper.tex')
    assert audit['supporting_sha256']==sha(MAN/'supporting.tex')
    assert audit['equation_count']==32
    assert len(audit['figures'])==21
    assert sum(len(f['panel_methods']) for f in audit['figures'])==75
    for figure in audit['figures']:
        assert figure['pdf_sha256']==sha(MAN/'figures'/figure['filename'])
        for panel in figure['panel_methods'].values():
            assert panel['definition'] and set(panel['equations'])<=set(audit['equation_labels'])
    for filename,expected in audit['computation_source_sha256'].items():
        assert sha(ROOT/filename)==expected


def test_statistical_markers_remain_readable_in_final_document_placements():
    audit=read(ROOT/'qa/document_qa_manifest.json')['documents']
    markers=[]
    for name,document in audit.items():
        assert document['sha256']==sha(ROOT/document['path'])
        for page in document.get('statistical_panel_markers_at_final_placement',{}).values():
            markers.extend(page)
    assert len(markers)==sum(PANELS)  # Journal derivatives retain all 75 panel labels as readable text.
    assert all(re.fullmatch('[a-z]',m['letter']) and m['size_points']>=12 for m in markers)


def test_documented_contrast_matches_per_edge_sampler_and_detachment():
    import torch
    import torch.nn.functional as F
    from tessera_st.losses import boundary_contrastive_loss
    z = torch.tensor([[1., 2.], [2., -1.], [-1., 3.]], requires_grad=True)
    edge = torch.tensor([[0, 1], [1, 2]])
    gate = torch.tensor([.2, .7], requires_grad=True)
    actual = boundary_contrastive_loss(z, edge, gate, .2, 10,
                generator=torch.Generator().manual_seed(11))
    neg = torch.randint(0, 3, (2, 10), generator=torch.Generator().manual_seed(11))
    zz = F.normalize(z, dim=-1)
    positive = (zz[edge[0]] * zz[edge[1]]).sum(-1) / .2
    negative = (zz[neg] * zz[edge[0]][:, None, :]).sum(-1) / .2
    per_edge = torch.logsumexp(torch.cat([positive[:, None], negative], dim=1), dim=1) - positive
    expected = (per_edge * gate.detach()).sum() / (gate.detach().sum() + 1e-8)
    assert torch.allclose(actual, expected, atol=1e-6)
    actual.backward()
    assert gate.grad is None and z.grad is not None and torch.isfinite(z.grad).all()
