import probstats
from probstats import algebra, bayes, inference
from probstats.algebra import ConjugacySignature, DistributionMetadata
from probstats.inference import OptimizationResult


def test_bayesian_facade_does_not_reexport_probability_distributions():
    for name in ("Normal", "Beta", "StudentT"):
        assert not hasattr(bayes, name)
    assert probstats.Normal is probstats.distributions.Normal
    assert probstats.Beta is probstats.distributions.Beta
    assert probstats.StudentT is probstats.distributions.StudentT


def test_structural_metadata_remains_in_algebra_namespace():
    assert ConjugacySignature is algebra.ConjugacySignature
    assert DistributionMetadata is algebra.DistributionMetadata
    assert not hasattr(bayes, "ConjugacySignature")
    assert not hasattr(bayes, "DistributionMetadata")


def test_colliding_result_types_remain_in_owning_namespaces():
    assert probstats.bayes is bayes
    assert OptimizationResult is inference.OptimizationResult
    assert not hasattr(bayes, "OptimizationResult")
