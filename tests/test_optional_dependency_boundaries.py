import subprocess
import sys


def test_core_import_and_numpy_regression_do_not_require_pandas():
    code = r"""
import builtins
_real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "pandas" or name.startswith("pandas."):
        raise ImportError("pandas unavailable")
    return _real_import(name, *args, **kwargs)
builtins.__import__ = guarded
import probstats
from probstats.inference import linear_regression
fit = linear_regression([[0.0], [1.0], [2.0]], [1.0, 3.0, 5.0])
assert fit.predict([[3.0]]).shape == (1,)
assert probstats.mean([1, 2, 3]) == 2.0
"""
    subprocess.run([sys.executable, "-c", code], check=True)
