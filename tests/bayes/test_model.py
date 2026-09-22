import pytest
import sympy as sp

from probstats.bayes import (
    Factor,
    Model,
    Observation,
    Parameter,
    RandomVariable,
)
from probstats.bayes.core import (
    ModelValidationError,
    ObservationError,
)
from probstats.distributions import SymbolicDistribution


def make_normal_location_model():
    theta = Parameter("theta")
    y = RandomVariable("y")
    t = sp.Symbol("t", real=True)
    z = sp.Symbol("z", real=True)
    prior = SymbolicDistribution(t, -(t**2) / 2, normalizer_known=False)
    likelihood = SymbolicDistribution(z, -((z - theta.symbol) ** 2) / 2)
    return (
        theta,
        y,
        Model(
            variables=(theta, y),
            factors=(
                Factor.from_distribution(theta, prior, name="prior"),
                Factor.from_distribution(y, likelihood, name="likelihood"),
            ),
        ),
    )


def test_model_builds_joint_log_density_and_dependency_graph():
    theta, y, model = make_normal_location_model()
    expected = -(theta.symbol**2) / 2 - (y.symbol - theta.symbol) ** 2 / 2
    assert sp.simplify(model.joint_log_density() - expected) == 0
    assert model.parents_of(y) == (theta,)
    assert model.parents_of(theta) == ()


def test_observe_returns_new_immutable_model_and_substitutes_data():
    theta, y, model = make_normal_location_model()
    observed = model.observe(y=2)
    assert model.observations == {}
    assert observed.observations == {"y": 2}
    assert observed.latent_variables == (theta,)
    assert observed.observed_variables == (y,)
    expected = -(theta.symbol**2) / 2 - (2 - theta.symbol) ** 2 / 2
    assert sp.simplify(observed.joint_log_density() - expected) == 0


def test_model_build_accepts_observation_objects():
    theta, y, model = make_normal_location_model()
    rebuilt = Model.build(
        model.variables,
        model.factors,
        observations=(Observation(y, 1.5),),
    )
    assert rebuilt.observations == {"y": 1.5}
    assert rebuilt.latent_variables == (theta,)


def test_duplicate_variable_names_are_rejected():
    with pytest.raises(ModelValidationError, match="Duplicate variable"):
        Model((Parameter("x"), RandomVariable("x")))


def test_factor_target_must_belong_to_model():
    x = Parameter("x")
    z = Parameter("z")
    with pytest.raises(ModelValidationError, match="not a model variable"):
        Model((x,), (Factor.from_log_density(z, -(z.symbol**2)),))


def test_observing_unknown_variable_is_rejected():
    _, _, model = make_normal_location_model()
    with pytest.raises(ObservationError, match="Unknown observed variable"):
        model.observe(nope=1)


def test_observation_is_checked_against_support_when_decidable():
    p = Parameter("p", support=sp.Interval(0, 1))
    model = Model((p,))
    with pytest.raises(ObservationError, match="outside support"):
        model.observe(p=2)
