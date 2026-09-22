"""Optional algebra/tensor backends expose named capability objects."""

import ast
from pathlib import Path

from probstats.algebraic import _dependencies

ROOT = Path(__file__).parents[1] / "src" / "probstats" / "algebraic"


def test_backend_loaders_return_named_capabilities():
    semialg = _dependencies.require_semialg()
    assert callable(semialg.analyze_equality_ideal)
    assert callable(semialg.ideal_degree)
    assert callable(semialg.singular_locus)

    tensor = _dependencies.require_tensoratlas()
    assert callable(tensor.TensorArray)
    assert callable(tensor.grouped_flatten)
    assert callable(tensor.segre_equations)
    assert callable(tensor.segre_parameterization)

    likelihood = _dependencies.require_likelihood_semialg()
    assert callable(likelihood.elimination_ideal_qq)
    assert callable(likelihood.saturate_ideal_qq)
    assert callable(likelihood.solve_zero_dimensional_system)

    latent = _dependencies.require_latent_backends()
    assert callable(latent.elimination_ideal_qq)
    assert callable(latent.generic_cp_identifiability)
    assert callable(latent.secant_expected_dimension)

    moments = _dependencies.require_moment_tensoratlas()
    assert callable(moments.cp_decompose)
    assert callable(moments.tensor_rank)
    assert callable(moments.waring_decompose)


def test_backend_results_not_positionally_unpacked():
    loader_names = {
        "require_semialg",
        "require_tensoratlas",
        "require_toric_semialg",
        "require_likelihood_semialg",
        "require_latent_backends",
        "require_moment_tensoratlas",
    }
    offenders = []
    for path in ROOT.glob("*.py"):
        if path.name == "_dependencies.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Name):
                continue
            if value.func.id not in loader_names:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            if any(isinstance(target, (ast.Tuple, ast.List)) for target in targets):
                offenders.append((path.name, node.lineno, value.func.id))
    assert offenders == []
