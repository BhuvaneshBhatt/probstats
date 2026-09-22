"""Minimal installed-package behavior expected from every release artifact."""

import subprocess
import sys


def test_public_smoke_in_fresh_interpreter():
    code = """
import probstats
from probstats import Normal, mean, quantile

if not probstats.__version__:
    raise RuntimeError("missing package version")
if mean([1, 2, 3]) != 2.0:
    raise RuntimeError("sample mean smoke check failed")
if mean(Normal(0, 1)) != 0:
    raise RuntimeError("distribution mean smoke check failed")
if quantile(Normal(0, 1), 0.5) != 0:
    raise RuntimeError("distribution quantile smoke check failed")
"""
    subprocess.run([sys.executable, "-c", code], check=True)
