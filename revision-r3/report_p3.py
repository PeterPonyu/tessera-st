"""Print the P3 results from summary/summary.json as compact tables, for writing and checking the text.

Usage: python report_p3.py   (reads TESSERA_R3_OUT or results/revision-r3)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_r3 as c3  # noqa: E402

ENC = {'tessera': 'Tessera-ST', 'graphst': 'GraphST', 'stagate': 'STAGATE'}


def f(v, d=3):
    return 'NA' if v is None else f'{v:+.{d}f}'


def main():
    s = json.loads((c3.OUT / 'summary' / 'summary.json').read_text())
    if 'P3' not in s:
        print('no P3 in summary.json')
        return
    print('== P3 primary (mean over units, 95% t-interval, SESOI, Holm)')
    for cls, tests in s['P3'].items():
        for key, t in tests.items():
            e, cp = key.split('|')
            print(f'{cls:9s} {ENC[e]:10s} {cp:16s} est {f(t["estimate"])} CI [{f(t["ci"][0])}, {f(t["ci"][1])}] '
                  f'sesoi {t["sesoi"]:.3f} pos {t["units_positive"]}/{t["n_units"]} '
                  f'pH_sup {t["p_holm_encoder_superior"]:.3f} pH_inf {t["p_holm_counterpart_superior"]:.3f} '
                  f'-> {t["verdict"]}  cohorts {", ".join(f"{c} {f(v)}" for c, v in t["cohort_means"].items())}')
    d = s.get('P3_descriptive', {})
    print('\n== component removals (arm minus released; mean over units)')
    for cls, by in d.get('arms', {}).items():
        for e, arms in by.items():
            print(f'{cls:9s} {ENC[e]:10s} ' + '; '.join(f'{a} {f(t["estimate"])} ({t["units_positive"]}/{t["n_units"]} up)'
                                                     for a, t in arms.items()))
    print('\n== encoder minus propagation by steps')
    for cls, by in d.get('depth', {}).items():
        for e, steps in by.items():
            print(f'{cls:9s} {ENC[e]:10s} ' + '  '.join(f'L{L} {f(t["estimate"])}' for L, t in steps.items()))
    print('\n== Leiden replication (descriptive)')
    for cls, by in d.get('leiden', {}).items():
        for e, tests in by.items():
            print(f'{cls:9s} {ENC[e]:10s} ' + '; '.join(
                f'{k} {f(t["estimate"])} [{f(t["ci"][0])}, {f(t["ci"][1])}] {t["units_positive"]}/{t["n_units"]}'
                for k, t in tests.items()))
    print('\n== Tessera-ST minus self-inclusive propagation (secondary)')
    for cls, by in d.get('selfloop', {}).items():
        print(f'{cls:9s} ' + '; '.join(f'L{L} {f(t["estimate"])} [{f(t["ci"][0])}, {f(t["ci"][1])}] '
                                      f'{t["units_positive"]}/{t["n_units"]}' for L, t in by.items()))
    print('\n== calibration cohorts (encoder minus counterpart; mean over patients)')
    for g, by in d.get('calibration', {}).items():
        print(f'{g:9s} ' + ' | '.join(f'{ENC[e]} prop {f(t["propagation"]["estimate"])} '
                                     f'({t["propagation"]["units_positive"]}/{t["propagation"]["n_units"]}) '
                                     f'mean {f(t["same_graph_mean"]["estimate"])}' for e, t in by.items()))
    for g, by in d.get('calibration_selfloop', {}).items():
        print(f'{g:9s} Tessera-ST minus self-inclusive: ' + '  '.join(f'L{L} {f(t["estimate"])}' for L, t in by.items()))


if __name__ == '__main__':
    main()
