#!/usr/bin/env python3
"""Registration inspection and descriptive pilot figures, kept outside the paper."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

import run_pilot as pilot
from common import read, dump, sha

HERE = pilot.HERE
OUT = HERE / 'figures'


def main():
    OUT.mkdir(exist_ok=True)
    registration = []
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    for section in pilot.SECTIONS:
        base = HERE / 'inputs' / section
        if not (base / 'input.json').exists():
            continue
        meta = read(base / 'input.json')
        d = np.load(base / 'input.npz', allow_pickle=False)
        image = np.asarray(Image.open(base / 'tissue_hires_image.png').convert('RGB'))
        xy = d['coords'] * meta['image_scalefactor']
        inside = ((xy >= 0).all(1) & (xy[:, 0] < image.shape[1]) & (xy[:, 1] < image.shape[0]))
        if not inside.all():
            raise ValueError('spot centers outside released H&E image')
        fig, axes = plt.subplots(1, 2, figsize=(11, 6.6))
        fig.subplots_adjust(left=.04, right=.98, top=.90, bottom=.25, wspace=.08)
        for ax in axes:
            ax.imshow(image)
            ax.set_xlim(0, image.shape[1]); ax.set_ylim(image.shape[0], 0)
            ax.set_aspect('equal'); ax.set_xlabel('Image pixels'); ax.set_ylabel('Image pixels')
        axes[0].set_title('Released H&E and retained tissue spots')
        axes[0].scatter(*xy.T, s=7, facecolors='none', edgecolors='#2166ac', linewidths=.4)
        axes[1].set_title('Pathologist categories; exclude is quality control')
        for i, name in enumerate(meta['classes']):
            mask = d['ref_name'] == name
            color = '#999999' if name.strip().lower() == 'exclude' else plt.get_cmap('tab20')(i)
            axes[1].scatter(*xy[mask].T, s=14, c=[color], edgecolors='white', linewidths=.15,
                            label=f'{name} (n={mask.sum()})', alpha=.85)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8, frameon=False)
        fig.suptitle(f'{meta["patient"]} — {section}; exact barcode join {meta["match_rate"]:.0%}', fontsize=12)
        fig.savefig(OUT / f'{section}-registration.png', dpi=160)
        plt.close(fig)
        registration.append(dict(section=section, image_shape=list(image.shape),
            centers_inside_image=int(inside.sum()), centers=len(xy),
            join_rate=meta['match_rate'], source_image_sha256=sha(base / 'tissue_hires_image.png')))
    if (HERE / 'annotation-qc/patients.csv').exists():
        ari = pd.read_csv(HERE / 'annotation-qc/patients.csv').set_index('patient')
        seams = pd.read_csv(HERE / 'annotation-qc/patient_seams.csv').set_index('patient')
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
        fig.subplots_adjust(left=.08, right=.98, bottom=.17, top=.82, wspace=.28)
        for j, (table, columns, title, ylabel) in enumerate([
            (ari, ['PCA50', 'UniformSameGraph', 'Tessera', 'NoGate'],
             'Compartment agreement (QC category omitted)', 'Adjusted Rand index'),
            (seams, ['Gate', 'RawFeature', 'UniformMeanFeature'],
             'Unique-edge seam discrimination', 'AUROC')]):
            ax = axes[j]
            for i, (patient, row) in enumerate(table.iterrows()):
                ax.plot(range(len(columns)), row[columns], '-o', label=patient,
                        color=plt.get_cmap('tab10')(i), linewidth=1, markersize=4)
            ax.set_xticks(range(len(columns)),
                          ['PCA50', 'Graph mean', 'Tessera', 'NoGate'] if j == 0 else ['Gate', 'Raw features', 'Mean features'])
            ax.margins(x=.15)
            ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis='y', alpha=.18)
            if j == 1:
                ax.axhline(.5, color='#888888', linestyle='--', linewidth=.7)
            ax.legend(frameon=False, fontsize=8)
        fig.suptitle('Untouched colorectal pilot: three patients, fixed methods, no scale search', fontsize=11)
        fig.savefig(OUT / 'pilot-summary.png', dpi=180)
        fig.savefig(OUT / 'pilot-summary.pdf')
        plt.close(fig)
    dump(OUT / 'figure-receipt.json', dict(script_sha256=sha(Path(__file__)),
        registration_geometry=registration,
        evidence_sha256={str(p.relative_to(HERE)): sha(p) for p in
                         sorted((HERE / 'annotation-qc').glob('*.csv'))},
        figure_sha256={p.name: sha(p) for p in sorted(OUT.glob('*'))
                       if p.suffix in ['.png', '.pdf']}))


if __name__ == '__main__':
    main()
