#!/usr/bin/env python3
"""Schema-only inspection of a downloaded h5ad, table, or archive of tables (round-3 cohorts).

The JSON receipt lists column names, dtypes and the numbers of unique and missing values; for
categorical, text or boolean columns with at most 60 distinct values it also lists those values and
their counts. Matrices and embeddings are described by type, shape and dtype only, numeric columns by
dtype and counts only, so no expression value, coordinate or score enters the receipt.
"""
from __future__ import annotations

import argparse
import fnmatch
import gzip
import io
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_cohorts as cc  # noqa: E402

MAX_CATEGORIES = 60
TABLES = ('.csv', '.tsv', '.txt', '.csv.gz', '.tsv.gz', '.txt.gz')


def column(series: pd.Series) -> dict:
    info = dict(dtype=str(series.dtype), n_unique=int(series.nunique(dropna=True)), n_missing=int(series.isna().sum()))
    listed = (isinstance(series.dtype, pd.CategoricalDtype) or pd.api.types.is_bool_dtype(series)
              or not pd.api.types.is_numeric_dtype(series))
    if listed and info['n_unique'] <= MAX_CATEGORIES:
        info['categories'] = {str(k): int(v) for k, v in series.value_counts(dropna=True).items()}
    return info


def table(frame: pd.DataFrame) -> dict:
    return dict(n_rows=len(frame), index=dict(name=frame.index.name, dtype=str(frame.index.dtype),
                                              n_unique=int(frame.index.nunique())),
                columns={str(name): column(frame.iloc[:, i]) for i, name in enumerate(frame.columns)})


def array(value) -> dict:
    """Type, shape, dtype and sparse format of a matrix or embedding; never its values."""
    if isinstance(value, pd.DataFrame):
        return dict(type='DataFrame', shape=list(value.shape), columns={str(c): str(t) for c, t in value.dtypes.items()})
    return dict(type=type(value).__name__, shape=[int(s) for s in getattr(value, 'shape', ())],
                dtype=str(getattr(value, 'dtype', '')), format=getattr(value, 'format', None))


def uns(mapping, depth: int = 0) -> dict:
    """Keys of ``.uns`` with their types; short strings (such as a unit statement) are kept verbatim."""
    out = {}
    for key, value in mapping.items():
        if isinstance(value, str):
            out[key] = value if len(value) <= 200 else dict(type='str', length=len(value))
        elif hasattr(value, 'items') and not isinstance(value, pd.DataFrame) and depth < 3:
            out[key] = uns(value, depth + 1)
        elif isinstance(value, (bool, int, float, np.generic)):
            out[key] = dict(type=type(value).__name__)
        else:
            out[key] = array(value)
    return out


def inspect_h5ad(path: Path) -> dict:
    import anndata as ad
    a = ad.read_h5ad(path, backed='r')        # X stays on disk and is never read
    try:
        return dict(kind='h5ad', n_obs=int(a.n_obs), n_vars=int(a.n_vars), X=array(a.X),
                    obs=table(a.obs), var=table(a.var),
                    layers={k: array(a.layers[k]) for k in a.layers}, obsm={k: array(a.obsm[k]) for k in a.obsm},
                    obsp=sorted(a.obsp.keys()), uns=uns(a.uns),
                    raw=None if a.raw is None else dict(X=array(a.raw.X), var=table(a.raw.var)))
    finally:
        a.file.close()


def read_table(name: str, data: bytes) -> pd.DataFrame:
    if name.endswith('.gz'):
        name, data = name[:-3], gzip.decompress(data)
    if name.endswith('.txt'):
        return pd.read_csv(io.BytesIO(data), sep=None, engine='python')
    return pd.read_csv(io.BytesIO(data), sep='\t' if name.endswith('.tsv') else ',', low_memory=False)


def inspect_archive(path: Path, patterns: list[str]) -> dict:
    """Member names and sizes; the schema of table members matching ``patterns``."""
    wanted = lambda name: (name.endswith(TABLES) and not Path(name).name.startswith('._')
                           and any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(Path(name).name, p) for p in patterns))
    members, tables = [], {}
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            for item in z.infolist():
                if not item.is_dir():
                    members.append(dict(name=item.filename, bytes=item.file_size))
                    if wanted(item.filename):
                        tables[item.filename] = table(read_table(item.filename, z.read(item)))
    else:
        with tarfile.open(path, 'r|*') as tar:
            for item in tar:
                if item.isfile():
                    members.append(dict(name=item.name, bytes=item.size))
                    if wanted(item.name):
                        tables[item.name] = table(read_table(item.name, tar.extractfile(item).read()))
    return dict(kind='archive', members=members, tables=tables)


def inspect(path, patterns=()) -> dict:
    path = Path(path)
    if path.suffix == '.h5ad':
        body = inspect_h5ad(path)
    elif path.name.endswith(TABLES):
        body = dict(kind='table', table=table(read_table(path.name, path.read_bytes())))
    else:
        body = inspect_archive(path, list(patterns))
    return dict(file=path.name, bytes=path.stat().st_size, sha256=cc.sha256(path), inspected_utc=cc.utc(),
                script_sha256=cc.sha256(Path(__file__)), **body)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('path', type=Path)
    parser.add_argument('--member', action='append', default=[],
                        help='archive members to read as tables (glob, repeatable); others are listed only')
    parser.add_argument('--out', type=Path, help='receipt path (default: <path>.inspect.json)')
    args = parser.parse_args(argv)
    out = args.out or args.path.with_name(args.path.name + '.inspect.json')
    cc.dump(out, inspect(args.path, args.member))
    print('INSPECTED', args.path.name, '->', out, flush=True)


if __name__ == '__main__':
    main()
