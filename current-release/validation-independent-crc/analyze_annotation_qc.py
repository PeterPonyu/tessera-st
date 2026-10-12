#!/usr/bin/env python3
"""Disclosed QC-label sensitivity; no fitting or target-dependent scale search."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, roc_auc_score

import run_pilot as pilot
from common import read, dump, sha, encode

HERE = pilot.HERE
OUT = HERE / 'annotation-qc'


def main():
    protocol = HERE / 'ANNOTATION_QC_PROTOCOL.md'
    if sha(protocol) != read(protocol.with_suffix('.lock.json'))['sha256']:
        raise ValueError('QC sensitivity differs from its lock')
    primary = read(pilot.OUT / 'run-receipt.json')
    OUT.mkdir(exist_ok=True)
    arms, edges, nulls = [], [], []
    for section in pilot.SECTIONS:
        d = np.load(HERE / 'inputs' / section / 'input.npz', allow_pickle=False)
        raw = d['ref_name'].copy()
        raw[np.char.lower(np.char.strip(raw)) == 'exclude'] = ''
        y, classes = encode(raw)
        if (y >= 0).sum() < 100 or len(classes) < 2:
            raise ValueError('QC-filtered eligibility failed')
        boundary = np.load(pilot.OUT / section / 'boundary_scores.npz', allow_pickle=False)
        pair = boundary['unique_edge_index']
        valid = (y[pair[0]] >= 0) & (y[pair[1]] >= 0)
        seam = y[pair[0]][valid] != y[pair[1]][valid]
        if seam.sum() < 20 or (~seam).sum() < 20:
            raise ValueError('QC-filtered seam eligibility failed')
        patient = section.split('_')[1]
        for arm in pilot.ARMS:
            for seed in pilot.SEEDS:
                p = np.load(pilot.OUT / section / f'{arm}-seed{seed}.npz', allow_pickle=False)
                pred = KMeans(len(classes), n_init=10, random_state=seed).fit_predict(p['embedding'])
                ari = float(adjusted_rand_score(y[y >= 0], pred[y >= 0]))
                arms.append(dict(section=section, patient=patient, arm=arm, seed=seed, ari=ari,
                    n_annotated=int((y >= 0).sum()), scoring_K=len(classes)))
                dest = OUT / section
                dest.mkdir(exist_ok=True)
                np.savez_compressed(dest / f'{arm}-seed{seed}.npz', labels=pred)
        for name in ['Gate', 'RawFeature', 'UniformMeanFeature']:
            values = boundary[name]
            edges.append(dict(section=section, patient=patient, score=name,
                auroc=float(roc_auc_score(seam, values[valid])), n_annotated=int((y >= 0).sum()),
                n_scored_edges=int(valid.sum()), n_seam_edges=int(seam.sum()),
                n_within_edges=int((~seam).sum())))
        rng = np.random.RandomState(20261011)
        for iteration in range(1000):
            shuffled = rng.permutation(y)
            ok = (shuffled[pair[0]] >= 0) & (shuffled[pair[1]] >= 0)
            target = shuffled[pair[0]][ok] != shuffled[pair[1]][ok]
            if len(np.unique(target)) < 2:
                raise ValueError('degenerate QC label shuffle')
            for name in ['Gate', 'RawFeature', 'UniformMeanFeature']:
                nulls.append(dict(section=section, score=name, shuffle=iteration,
                    auroc=float(roc_auc_score(target, boundary[name][ok]))))
        np.savez_compressed(OUT / section / 'target.npz', ref=y, scored_mask=valid, seam=seam)
    arms, edges, nulls = pd.DataFrame(arms), pd.DataFrame(edges), pd.DataFrame(nulls)
    for name, frame in [('arms', arms), ('seams', edges), ('label_shuffle', nulls)]:
        frame.to_csv(OUT / (name + '.csv'), index=False)
    patients = arms.groupby(['patient', 'arm']).ari.mean().unstack().reindex([s.split('_')[1] for s in pilot.SECTIONS])
    patients['Tessera_minus_UniformSameGraph'] = patients.Tessera - patients.UniformSameGraph
    patients['NoGate_minus_Tessera'] = patients.NoGate - patients.Tessera
    patients.to_csv(OUT / 'patients.csv')
    seams = edges.pivot(index='patient', columns='score', values='auroc').reindex(patients.index)
    seams['Gate_minus_RawFeature'] = seams.Gate - seams.RawFeature
    seams['Gate_minus_UniformMeanFeature'] = seams.Gate - seams.UniformMeanFeature
    seams.to_csv(OUT / 'patient_seams.csv')
    statistics = {name: pilot.bootstrap(seams[name]) for name in seams}
    statistics.update({name: pilot.bootstrap(patients[name]) for name in
                       ['Tessera_minus_UniformSameGraph', 'NoGate_minus_Tessera']})
    criteria = dict(gate_discrimination=statistics['Gate']['lower'] > .5,
        gate_advantage_over_raw=statistics['Gate_minus_RawFeature']['lower'] > 0,
        learned_ari_gain_over_mean_margin_0_02=statistics['Tessera_minus_UniformSameGraph']['lower'] > .02)
    dump(OUT / 'run-receipt.json', dict(protocol_sha256=sha(protocol), script_sha256=sha(Path(__file__)),
        primary_receipt_sha256=sha(pilot.OUT / 'run-receipt.json'),
        primary_bindings=primary['bindings'], new_model_fits=0, patients=3,
        statistics=statistics, pilot_criteria=criteria,
        output_sha256={str(p.relative_to(HERE)): sha(p) for p in sorted(OUT.rglob('*'))
                       if p.is_file() and p.name != 'run-receipt.json'}))
    print('QC SENSITIVITY', {'statistics': statistics, 'pilot_criteria': criteria}, flush=True)


if __name__ == '__main__':
    main()
