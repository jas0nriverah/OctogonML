"""Distribution algebra, point-in-time fitting boundaries and deployable V3 artifacts."""

import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp
from scipy.stats import nbinom

import cagemetrics.duration as duration
from cagemetrics.duration import (
    CountMixture,
    DurationMean,
    V3Engine,
    fit_duration,
    mixture_forecast,
    swap_perspectives,
)
from cagemetrics.features import FEATURE_COLUMNS


def feature_frame(n=1, date="2024-01-01"):
    data = {name: np.arange(n, dtype=float) + i + 1 for i, name in enumerate(FEATURE_COLUMNS)}
    data["scheduled_rounds"] = np.repeat(3.0, n)
    return pd.DataFrame(data).assign(event_date=date)


@pytest.mark.parametrize("weight", [0.0, 0.15, 0.8, 1.0])
def test_mixture_probability_functions_equal_weighted_component_distributions(weight):
    distribution = CountMixture(weight, 12.0, 83.0, (0.2, 0.5))
    x = np.arange(250)
    early = nbinom(5.0, 1 / (1 + 0.2 * 12))
    limit = nbinom(2.0, 1 / (1 + 0.5 * 83))
    expected = (1 - weight) * early.pmf(x) + weight * limit.pmf(x)
    np.testing.assert_allclose(distribution.pmf(x), expected, rtol=1e-12)
    np.testing.assert_allclose(
        distribution.cdf(x), (1 - weight) * early.cdf(x) + weight * limit.cdf(x), rtol=1e-12
    )
    np.testing.assert_allclose(
        distribution.sf(x), (1 - weight) * early.sf(x) + weight * limit.sf(x), rtol=1e-12
    )
    np.testing.assert_allclose(distribution.logpmf(x), np.log(expected), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(distribution.cdf(x) + distribution.sf(x), 1.0)


def test_logpmf_retains_extreme_tail_information_and_batch_broadcasting():
    distribution = CountMixture([0.0, 0.5, 1.0], [1.0, 10.0, 20.0], [30.0, 100.0, 200.0], (0.1, 0.3))
    actual = distribution.logpmf(10000)
    with np.errstate(divide="ignore"):
        expected = logsumexp(
            np.stack(
                [
                    np.log1p(-distribution.weight) + distribution.early.logpmf(10000),
                    np.log(distribution.weight) + distribution.limit.logpmf(10000),
                ]
            ),
            axis=0,
        )
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected)
    assert distribution.pmf(np.arange(10)[:, None]).shape == (10, 3)
    assert distribution.ppf(0.9).shape == (3,)


@pytest.mark.parametrize("line", [-1.0, 0.0, 0.5, 20.0, 20.5, 1000000.0])
def test_discrete_forecast_partition_push_and_rendered_mass(line):
    distribution = CountMixture(0.3, 10.0, 60.0, (0.2, 0.4))
    result = mixture_forecast(distribution, line)
    assert sum(result[key] for key in ["p_over", "p_under", "p_push"]) == pytest.approx(1.0)
    assert result["p_push"] == pytest.approx(float(distribution.pmf(line)) if line.is_integer() else 0.0)
    mass = sum(row["probability"] for row in result["distribution"]["pmf"])
    assert mass + result["distribution"]["tail_probability"] == pytest.approx(1.0)
    assert result["interval_low"] <= result["median"] <= result["interval_high"]


def test_variance_includes_between_regime_variability():
    distribution = CountMixture(0.25, 10.0, 110.0, (0.2, 0.1))
    within = 0.75 * (10 + 0.2 * 10**2) + 0.25 * (110 + 0.1 * 110**2)
    between = 0.25 * 0.75 * (110 - 10) ** 2
    assert distribution.mean() == pytest.approx(35.0)
    assert distribution.variance() == pytest.approx(within + between)
    assert mixture_forecast(distribution, 40.5)["distribution"]["variance"] == pytest.approx(within + between)


@pytest.mark.parametrize("q", [0.001, 0.1, 0.5, 0.9, 0.999])
def test_quantiles_are_exact_smallest_integer_crossings(q):
    distribution = CountMixture([0.0, 0.4, 1.0], [0.0, 10.0, 90.0], [10.0, 100.0, 180.0], (0.0, 0.3))
    values = distribution.ppf(q)
    assert np.issubdtype(values.dtype, np.integer)
    assert (distribution.cdf(values) >= q).all()
    assert (distribution.cdf(values - 1) < q).all()


def test_quantiles_are_not_truncated_by_visual_pmf_cap():
    distribution = CountMixture(0.8, 5000.0, 25000.0, (0.1, 0.2))
    result = mixture_forecast(distribution, 10000.5)
    assert result["median"] > 2000
    assert result["interval_high"] > 2000
    assert result["distribution"]["pmf"][-1]["x"] == 2000
    assert result["distribution"]["tail_probability"] > 0
    assert distribution.cdf(result["interval_high"] - 0.1) < 0.9
    assert distribution.cdf(result["interval_high"]) >= 0.9


def test_quantile_bracket_overflow_is_rejected_without_integer_cast_warning():
    import warnings

    distribution = CountMixture(0.5, 1e20, 2e20, (0.1, 0.2))
    with warnings.catch_warnings(record=True) as captured:
        with pytest.raises(ValueError, match="supported integer range"):
            distribution.ppf(0.9)
    assert not captured


@pytest.mark.parametrize(
    "weight,early,limit,alphas",
    [
        (-0.1, 10.0, 20.0, (0.1, 0.1)),
        (1.1, 10.0, 20.0, (0.1, 0.1)),
        (np.nan, 10.0, 20.0, (0.1, 0.1)),
        (0.5, -1.0, 20.0, (0.1, 0.1)),
        (0.5, np.inf, 20.0, (0.1, 0.1)),
        (0.5, 10.0, np.nan, (0.1, 0.1)),
        (0.5, 10.0, 20.0, (-0.1, 0.1)),
        (0.5, 10.0, 20.0, (0.1, np.inf)),
        (0.5, 10.0, 20.0, (0.1,)),
    ],
)
def test_invalid_mixture_parameters_rejected(weight, early, limit, alphas):
    with pytest.raises(ValueError):
        CountMixture(weight, early, limit, alphas)


def test_shape_and_forecast_boundary_validation():
    with pytest.raises(ValueError):
        CountMixture([0.2, 0.3], [1, 2, 3], 10.0, (0.1, 0.2))
    distribution = CountMixture([0.2, 0.3], 10.0, 50.0, (0.1, 0.2))
    with pytest.raises(ValueError, match="Exactly one"):
        mixture_forecast(distribution, 40.5)
    for q in [0.0, 1.0, -1.0, np.nan, np.inf]:
        with pytest.raises(ValueError):
            distribution.ppf(q)
    with pytest.raises(ValueError):
        mixture_forecast(CountMixture(0.5, 1.0, 10.0, (0.1, 0.2)), np.nan)


class DirectionalGate:
    def predict_proba(self, frame):
        values = np.clip(frame.own_prior_bouts.to_numpy() / 100.0, 0.01, 0.99)
        return np.column_stack([1 - values, values])


class ConstantMean:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.repeat(self.value, len(frame))


class AthleteSpecificMean:
    def __init__(self, scale):
        self.scale = scale

    def predict(self, frame):
        return frame.own_prior_bouts.to_numpy() * self.scale + 1


def test_swap_is_involution_and_gate_is_symmetric_with_separate_marginals():
    frame = feature_frame(4)
    swapped = swap_perspectives(frame)
    pd.testing.assert_frame_equal(swap_perspectives(swapped), frame[FEATURE_COLUMNS])
    assert swapped.scheduled_rounds.equals(frame.scheduled_rounds)
    model = DurationMean(DirectionalGate(), ConstantMean(10), ConstantMean(80))
    weight, early, limit = model.components(frame)
    np.testing.assert_allclose(weight, model.components(swapped)[0])
    np.testing.assert_allclose(model.predict(frame), (1 - weight) * early + weight * limit)
    directional = DurationMean(DirectionalGate(), AthleteSpecificMean(2), AthleteSpecificMean(10))
    np.testing.assert_allclose(directional.components(frame)[0], directional.components(swapped)[0])
    assert not np.allclose(directional.predict(frame), directional.predict(swapped))


def training_frames():
    train = feature_frame(120, "2022-01-01").assign(
        reaches_limit=np.tile([0, 1], 60),
        target=np.arange(120) + 10,
        duration_seconds=np.tile([300, 900], 60),
    )
    cal = feature_frame(40, "2023-01-01").assign(
        reaches_limit=np.tile([0, 1], 20),
        target=np.arange(40) + 200,
        duration_seconds=np.tile([300, 900], 20),
    )
    return train, cal


def test_fit_uses_closed_feature_set_training_only_and_regime_specific_calibration(monkeypatch):
    train, cal = training_frames()
    fitted = []
    calibrated = []
    gate_fits = []

    class RecordingMean(ConstantMean):
        def fit(self, x, y):
            fitted.append((x.copy(), y.copy()))
            return self

    class RecordingGate(DirectionalGate):
        def fit(self, x, y):
            gate_fits.append((x.copy(), y.copy()))
            return self

    monkeypatch.setattr(duration, "make_pipeline", lambda *args: RecordingGate())
    monkeypatch.setattr(duration, "_estimator", lambda kind: RecordingMean(30.0))

    def alpha(y, mu):
        calibrated.append((y.copy(), mu.copy()))
        return 0.1 + len(calibrated) / 10

    monkeypatch.setattr(duration, "_alpha", alpha)
    artifact = fit_duration(train, cal)
    assert artifact["component_alphas"] == pytest.approx([0.2, 0.3])
    assert list(gate_fits[0][0].columns) == FEATURE_COLUMNS
    np.testing.assert_array_equal(gate_fits[0][1], train.reaches_limit)
    for regime, (x, y), (actual, mu) in zip([0, 1], fitted, calibrated):
        assert list(x.columns) == FEATURE_COLUMNS
        assert not {"target", "duration_seconds", "reaches_limit", "event_date"} & set(x.columns)
        np.testing.assert_array_equal(y, train.loc[train.reaches_limit == regime, "target"])
        np.testing.assert_array_equal(actual, cal.loc[cal.reaches_limit == regime, "target"])
        np.testing.assert_array_equal(mu, np.repeat(30.0, 20))
    np.testing.assert_allclose(artifact["model"].predict(cal), 30.0)


@pytest.mark.parametrize("damage", ["train_size", "date_overlap", "regime_missing", "regime_too_small"])
def test_invalid_fit_boundaries_rejected(damage):
    train, cal = training_frames()
    if damage == "train_size":
        train = train.iloc[:50]
    elif damage == "date_overlap":
        cal["event_date"] = "2022-01-01"
    elif damage == "regime_missing":
        train["reaches_limit"] = 0
    else:
        cal.loc[cal.index >= 5, "reaches_limit"] = 1
    with pytest.raises(ValueError):
        fit_duration(train, cal)


@pytest.fixture
def fitted_artifact():
    train, cal = training_frames()
    # A real small fitted gate and count heads exercise the sklearn serialization path.
    train["target"] = np.tile([10.0, 70.0], 60)
    cal["target"] = np.tile([12.0, 65.0], 20)
    result = fit_duration(train, cal)
    return result | {"candidate": "duration_mixture", "version": "v3-test", "columns": list(FEATURE_COLUMNS)}


def write_artifact(folder, artifact, **metadata_overrides):
    path = folder / "model.joblib"
    joblib.dump(artifact, path)
    metadata = {
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_version": "v3-test",
        "calibration_cutoff_exclusive": "2023-02-01",
        "warnings": ["Unit-test fitted artifact"],
        **metadata_overrides,
    }
    (folder / "metadata.json").write_text(json.dumps(metadata))


def test_real_fitted_artifact_engine_probability_and_metadata(tmp_path, fitted_artifact):
    write_artifact(tmp_path, fitted_artifact)
    engine = V3Engine(tmp_path)
    result = engine.predict(feature_frame(1, "2024-01-01"), 40.5)
    assert result["model_version"] == "v3-test"
    assert result["features"].keys() == set(FEATURE_COLUMNS)
    assert result["warnings"] == ["Unit-test fitted artifact"]
    assert sum(result[k] for k in ["p_over", "p_under", "p_push"]) == pytest.approx(1.0)
    assert np.isfinite(result["mean"])
    for frame in [
        feature_frame(2),
        feature_frame().drop(columns="event_date"),
        feature_frame(1, "2022-01-01"),
    ]:
        with pytest.raises(ValueError):
            engine.predict(frame, 40.5)


@pytest.mark.parametrize("damage", ["checksum", "version", "candidate"])
def test_engine_rejects_unverified_artifacts(tmp_path, damage):
    artifact = {"candidate": "duration_mixture", "version": "v3-test", "columns": FEATURE_COLUMNS}
    if damage == "version":
        artifact["version"] = "different"
    elif damage == "candidate":
        artifact["candidate"] = "other"
    write_artifact(tmp_path, artifact)
    if damage == "checksum":
        (tmp_path / "model.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        V3Engine(tmp_path)


@pytest.mark.parametrize("invalid_date", ["not-a-date", "2024-99-99", "2024/01/01", ""])
def test_engine_rejects_malformed_inference_dates(tmp_path, fitted_artifact, invalid_date):
    write_artifact(tmp_path, fitted_artifact)
    engine = V3Engine(tmp_path)
    with pytest.raises(ValueError):
        engine.predict(feature_frame(1, invalid_date), 40.5)


@pytest.mark.parametrize("invalid_cutoff", ["not-a-date", "2023-13-01"])
def test_engine_rejects_malformed_metadata_calibration_cutoff(tmp_path, invalid_cutoff):
    artifact = {"candidate": "duration_mixture", "version": "v3-test", "columns": FEATURE_COLUMNS}
    write_artifact(tmp_path, artifact, calibration_cutoff_exclusive=invalid_cutoff)
    with pytest.raises(ValueError):
        V3Engine(tmp_path)


def test_engine_allows_cutoff_boundary_but_rejects_previous_day(tmp_path, fitted_artifact):
    write_artifact(tmp_path, fitted_artifact)
    engine = V3Engine(tmp_path)
    assert engine.predict(feature_frame(1, "2023-02-01"), 40.5)["model_version"] == "v3-test"
    with pytest.raises(ValueError):
        engine.predict(feature_frame(1, "2023-01-31"), 40.5)
