import ast
from pathlib import Path

import probstats

_EXPECTED_INTERNAL_GROUPS = {
    "inference.py": {
        "_inference_likelihood",
        "_inference_nonparametric",
        "_inference_profile",
        "_inference_regression",
        "_inference_resampling",
    },
    "testing.py": {
        "_testing_core",
        "_testing_dependence",
        "_testing_diagnostics",
        "_testing_parametric",
        "_testing_variance",
    },
    "data.py": {
        "_data_core",
        "_data_exponential",
        "_data_rolling",
        "_data_sequence",
        "_data_smoothing",
    },
}


def test_large_public_modules_are_import_only_facades():
    package_root = Path(probstats.__file__).resolve().parent
    for filename, expected_modules in _EXPECTED_INTERNAL_GROUPS.items():
        tree = ast.parse((package_root / filename).read_text())
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            for node in tree.body
        )
        imported = {
            node.module.rsplit(".", 1)[-1]
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert expected_modules <= imported
