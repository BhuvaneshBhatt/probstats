import math

import numpy as np
import pytest

from probstats.survival import (
    SurvivalData,
    breslow_test,
    fleming_harrington_test,
    kaplan_meier,
    logrank_test,
    median_survival,
    nelson_aalen,
    risk_table,
    tarone_ware_test,
)


def test_survival_data_validation_and_convenience_constructor():
    data = SurvivalData.from_event_censor_times([1, 2], [3])
    assert data.n == 3
    assert data.events == 2
    assert data.censored == 1
    with pytest.raises(ValueError):
        SurvivalData([1, 2], [1])
    with pytest.raises(ValueError):
        SurvivalData([1], [1], entry=[2])


def test_risk_table_ties_events_before_censoring_and_delayed_entry():
    data = SurvivalData(
        [2, 2, 3, 4],
        [1, 0, 1, 1],
        entry=[0, 0, 2.5, 0],
    )
    table = risk_table(data)
    assert np.array_equal(table.time, [2, 3, 4])
    assert np.array_equal(table.at_risk, [3, 2, 1])
    assert np.array_equal(table.events, [1, 1, 1])
    assert np.array_equal(table.censored, [1, 0, 0])


def test_kaplan_meier_matches_hand_calculation_and_median():
    data = SurvivalData([1, 2, 2, 3], [1, 1, 0, 1])
    fit = kaplan_meier(data)
    # t=1: 3/4; t=2: (3/4)*(1-1/3)=1/2; t=3 -> 0.
    assert np.allclose(fit.event_times, [1, 2, 3])
    assert np.allclose(fit.survival, [0.75, 0.5, 0.0])
    assert fit.survival_at(1.5) == pytest.approx(0.75)
    assert fit.survival_at(2) == pytest.approx(0.5)
    assert fit.median_survival == 2
    assert median_survival(data) == 2
    assert np.all((fit.lower >= 0) & (fit.upper <= 1))


def test_greenwood_variance_matches_hand_calculation():
    data = SurvivalData([1, 2, 3, 4], [1, 1, 0, 1])
    fit = kaplan_meier(data)
    # At t=2: S=(3/4)*(2/3)=1/2, Greenwood sum=1/(4*3)+1/(3*2)=1/4.
    assert fit.survival[1] == pytest.approx(0.5)
    assert fit.greenwood_sum[1] == pytest.approx(0.25)
    assert fit.variance[1] == pytest.approx(0.0625)
    assert fit.standard_error[1] == pytest.approx(0.25)


def test_nelson_aalen_matches_hand_calculation():
    data = SurvivalData([1, 2, 2, 3], [1, 1, 0, 1])
    fit = nelson_aalen(data)
    assert np.allclose(fit.cumulative_hazard, [1 / 4, 1 / 4 + 1 / 3, 1 / 4 + 1 / 3 + 1])
    assert fit.cumulative_hazard_at(1.5) == pytest.approx(0.25)
    assert fit.survival_at(0) == pytest.approx(1.0)


def test_median_is_infinite_when_km_never_reaches_half():
    data = SurvivalData([1, 2, 3, 4], [1, 0, 0, 0])
    assert math.isinf(kaplan_meier(data).median_survival)


def test_logrank_identical_groups_has_zero_statistic():
    first = SurvivalData([1, 2, 3, 4], [1, 1, 0, 1])
    second = SurvivalData([1, 2, 3, 4], [1, 1, 0, 1])
    result = logrank_test(first, second)
    assert result.statistic == pytest.approx(0.0)
    assert result.pvalue == pytest.approx(1.0)
    assert result.degrees_of_freedom == 1


def test_logrank_detects_strong_separation_and_group_vector_api():
    time = np.r_[np.arange(1, 11), np.arange(11, 21)]
    group = np.array(["early"] * 10 + ["late"] * 10)
    result = logrank_test(time, groups=group)
    assert result.statistic > 10
    assert result.pvalue < 0.01


def test_weighted_logrank_family_and_fleming_harrington_special_case():
    first = SurvivalData([1, 2, 3, 8, 9], [1, 1, 0, 1, 0])
    second = SurvivalData([2, 4, 5, 7, 10], [1, 0, 1, 1, 0])
    ordinary = logrank_test(first, second)
    fh00 = fleming_harrington_test(first, second, p=0, q=0)
    assert fh00.statistic == pytest.approx(ordinary.statistic)
    assert fh00.pvalue == pytest.approx(ordinary.pvalue)
    assert breslow_test(first, second).statistic >= 0
    assert tarone_ware_test(first, second).statistic >= 0


def test_logrank_supports_more_than_two_groups_and_delayed_entry():
    samples = (
        SurvivalData([2, 4, 7], [1, 1, 0], entry=[0, 0, 1]),
        SurvivalData([3, 5, 8], [1, 0, 1], entry=[0, 2, 0]),
        SurvivalData([1, 6, 9], [1, 1, 0], entry=[0, 0, 4]),
    )
    result = logrank_test(*samples)
    assert result.degrees_of_freedom == 2
    assert 0 <= result.pvalue <= 1
