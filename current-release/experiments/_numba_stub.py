"""Hack: stub numba so numba-dependent SOTA (SEDR / GraphST / scanpy) import under numpy>=2.3.

dl env has numpy 2.5; real numba refuses to import (needs <=2.3). We never change the env — we just
inject a fake `numba` into sys.modules BEFORE those packages import it, so `@njit` etc become
identity decorators and the kernels run as plain Python (slower, but correct). Call install() first.
"""

import sys
import types


class _T:  # stand-in for numba type objects: indexable + callable, returns itself
    def __getitem__(self, i):
        return self

    def __call__(self, *a, **k):
        return self


def _deco(*a, **k):
    # njit(f), @njit, AND njit(f, cache=True, parallel=True)  -> return the function itself
    if a and callable(a[0]):
        return a[0]
    # @njit(...), njit(signature), njit(parallel=True)  -> identity decorator
    return lambda f: f


def install():
    if isinstance(sys.modules.get("numba"), types.ModuleType) and getattr(
            sys.modules["numba"], "_is_stub", False):
        return
    nb = types.ModuleType("numba")
    nb._is_stub = True
    for n in ["njit", "jit", "vectorize", "guvectorize", "generated_jit", "stencil", "cfunc",
              "jitclass", "overload"]:
        setattr(nb, n, _deco)
    nb.prange = range
    import numpy as _np
    nb.pndindex = lambda *a: _np.ndindex(*(a[0] if len(a) == 1 and isinstance(a[0], tuple) else a))
    nb.literally = lambda x: x
    nb.literal_unroll = lambda x: x
    nb.__version__ = "0.0.0-stub"
    nb.objmode = _T()
    nb.set_num_threads = lambda *a, **k: None
    nb.get_num_threads = lambda *a, **k: 1
    nb.config = types.SimpleNamespace(NUMBA_DEFAULT_NUM_THREADS=1, THREADING_LAYER="workqueue")

    tm = types.ModuleType("numba.types")
    for t in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64",
              "float32", "float64", "double", "boolean", "void", "intp", "uintp", "none",
              "byte", "char", "Array", "Tuple", "UniTuple", "List", "DictType", "ListType",
              "string", "unicode_type", "containers"]:
        setattr(tm, t, _T())
    nb.types = tm

    tym = types.ModuleType("numba.typed")
    tym.Dict = dict
    tym.List = list
    nb.typed = tym

    ce = types.ModuleType("numba.core.types")
    for t in dir(tm):
        if not t.startswith("_"):
            setattr(ce, t, getattr(tm, t))
    cm = types.ModuleType("numba.core")
    cm.types = ce

    ext = types.ModuleType("numba.extending")
    ext.overload = _deco
    ext.overload_method = _deco
    ext.register_jitable = _deco
    ext.get_cython_function_address = lambda *a, **k: 0
    ext.intrinsic = _deco
    nb.extending = ext

    npm = types.ModuleType("numba.np")
    npmu = types.ModuleType("numba.np.ufunc")
    npm.ufunc = npmu
    nb.np = npm

    # numba.experimental (jitclass / structref) — pulled in by pynndescent via scanpy's louvain/leiden,
    # which SpaGCN's resolution search triggers. Additive: does not affect the other stubbed importers.
    exp = types.ModuleType("numba.experimental")
    exp.jitclass = _deco
    sref = types.ModuleType("numba.experimental.structref")
    sref.register = _deco
    sref.StructRefProxy = object
    sref.define_boxing = lambda *a, **k: None
    sref.new = lambda *a, **k: None
    exp.structref = sref
    nb.experimental = exp

    for name, mod in [("numba", nb), ("numba.types", tm), ("numba.typed", tym),
                      ("numba.core", cm), ("numba.core.types", ce), ("numba.extending", ext),
                      ("numba.np", npm), ("numba.np.ufunc", npmu),
                      ("numba.experimental", exp), ("numba.experimental.structref", sref)]:
        sys.modules[name] = mod
