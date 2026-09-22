import importlib

import pytest

PUBLIC_MODULES = (
    "probstats.bayes",
    "probstats.distributions",
    "probstats.functionals",
    "probstats.information",
    "probstats.random_matrix",
    "probstats.smoothing",
    "probstats.spaces",
    "probstats.stats",
    "probstats.survival",
    "probstats.symbolic",
)


@pytest.mark.parametrize("module_name", PUBLIC_MODULES)
def test_public_module_exports_are_unique_existing_names(module_name):
    module = importlib.import_module(module_name)
    exports = module.__all__
    assert len(exports) == len(set(exports))
    assert all(
        isinstance(name, str) and name and not name.startswith("_") for name in exports
    )
    assert all(hasattr(module, name) for name in exports)
