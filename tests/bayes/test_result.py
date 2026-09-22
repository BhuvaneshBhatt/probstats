import pytest

from probstats.bayes import (
    InferenceKind,
    InferenceResult,
)
from probstats.bayes.core import InferenceStep


def test_result_records_and_explains_provenance():
    result = InferenceResult(
        posterior="posterior",
        kind=InferenceKind.EXACT,
        steps=(InferenceStep("conjugacy", "Matched Beta-Binomial rule", True),),
        metadata={"engine": "test"},
    )
    assert result.is_exact
    assert "conjugacy [exact]" in result.explain()
    assert result.metadata["engine"] == "test"


def test_result_metadata_is_immutable():
    result = InferenceResult(
        posterior=None, kind=InferenceKind.APPROXIMATE, metadata={"x": 1}
    )
    with pytest.raises(TypeError):
        result.metadata["x"] = 2
