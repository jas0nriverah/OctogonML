"""Fixed study selection rejects temporal leakage, misalignment and unsafe promotion."""

from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cagemetrics import winner_v3 as v3
from cagemetrics import winner_v3_training as study


def development_rows(*, sparse=40, degrade_sparse=False):
    frames = []
    summaries = []
    strength = {
        "baseline_v2": 0.55,
        "reliability_hgb": 0.64,
        "weekly_rank_hgb": 0.70,
        "blend_50": 0.62,
        "weekly_blend_50": 0.67,
    }
    for candidate in v3.PROTOCOL["candidates"]:
        for year in [2021, 2022, 2023]:
            rows = []
            for i in range(100):
                target = i % 2
                quality = strength[candidate]
                if degrade_sparse and candidate == "weekly_rank_hgb":
                    quality = 0.54 if i < sparse else 0.9
                rows.append(
                    {
                        "bout_id": f"{year}-{i}",
                        "event_id": f"{year}-e{i // 5}",
                        "event_date": f"{year}-02-01",
                        "fighter_a_id": f"a{i}",
                        "fighter_b_id": f"b{i}",
                        "target": target,
                        "min_prior_bouts": 2 if i < sparse else 5,
                        "max_prior_bouts": 10,
                        "candidate": candidate,
                        "probability": quality if target else 1 - quality,
                        "stage": "development",
                    }
                )
            frame = pd.DataFrame(rows)
            frames.append(frame)
            summaries.append({"candidate": candidate, "year": year, **study._metrics(frame)})
    return summaries, pd.concat(frames, ignore_index=True)


def test_fixed_selection_and_gates_use_paired_development_rows_only():
    summaries, rows = development_rows()
    candidate, scores, gate = study.select_candidate(summaries, rows)
    assert candidate == "weekly_rank_hgb"
    assert set(scores) == set(v3.PROTOCOL["candidates"])
    assert gate["passed"] and all(gate["checks"].values())
    assert gate["subgroups"]["sparse_veteran"]["n_bouts"] == 120
    assert gate["paired_event_bootstrap"]["events"] == 60
    assert gate["paired_event_bootstrap"]["log_loss"]["ci_95"][1] < 0


@pytest.mark.parametrize(
    "defect",
    [
        "missing_summary",
        "duplicate_summary",
        "missing_candidate",
        "duplicate_prediction",
        "later_year",
        "target_drift",
        "fighter_drift",
        "event_drift",
        "nonfinite",
        "summary_drift",
    ],
)
def test_selection_rejects_incomplete_or_misaligned_comparisons(defect):
    summaries, rows = development_rows()
    summaries = deepcopy(summaries)
    index = rows.index[rows.candidate == "weekly_rank_hgb"][0]
    if defect == "missing_summary":
        summaries.pop()
    elif defect == "duplicate_summary":
        summaries[-1] = summaries[0]
    elif defect == "missing_candidate":
        rows = rows[rows.candidate != "blend_50"]
    elif defect == "duplicate_prediction":
        rows = pd.concat([rows, rows.iloc[[0]]])
    elif defect == "later_year":
        rows.loc[index, "event_date"] = "2024-02-01"
    elif defect == "target_drift":
        rows.loc[index, "target"] = 1 - rows.loc[index, "target"]
    elif defect == "fighter_drift":
        rows.loc[index, "fighter_a_id"] = "wrong-id"
    elif defect == "event_drift":
        rows.loc[index, "event_id"] = "wrong-event"
    elif defect == "nonfinite":
        rows.loc[index, "probability"] = np.nan
    elif defect == "summary_drift":
        summaries[0]["log_loss"] += 0.01
    with pytest.raises(ValueError):
        study.select_candidate(summaries, rows)


def test_sparse_veteran_regression_blocks_an_aggregate_improvement():
    summaries, rows = development_rows(degrade_sparse=True)
    candidate, _, gate = study.select_candidate(summaries, rows)
    assert candidate == "weekly_rank_hgb"
    assert gate["checks"]["log_loss_gain"]
    assert not gate["checks"]["sparse_veteran_nonregression"]
    assert not gate["passed"]


def test_small_sparse_group_is_warned_not_falsely_claimed_validated():
    summaries, rows = development_rows(sparse=9)
    _, _, gate = study.select_candidate(summaries, rows)
    assert gate["subgroups"]["sparse_veteran"]["n_bouts"] == 27
    assert gate["subgroups"]["sparse_veteran"]["passed"] is None
    assert gate["warnings"]


def test_prior_counts_exclude_same_day_and_future_and_use_both_athletes():
    frame = pd.DataFrame([{"fighter_a_id": "a", "fighter_b_id": "b", "event_date": "2022-01-01"}])
    history = SimpleNamespace(
        dates={"a": ["2020-01-01", "2022-01-01", "2030-01-01"], "b": ["2019-01-01", "2020-01-01"]}
    )
    result = study.attach_prior_counts(frame, history).iloc[0]
    assert result.min_prior_bouts == 1 and result.max_prior_bouts == 2


def test_existing_different_outputs_are_not_overwritten(tmp_path):
    target = tmp_path / "protocol.json"
    target.write_bytes(b"original")
    study._write_once(target, b"original")
    with pytest.raises(ValueError, match="Immutable"):
        study._write_once(target, b"replacement")
    assert target.read_bytes() == b"original"


def test_runner_declares_before_fit_locks_selection_before2024_and_reuses(monkeypatch, tmp_path):
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    reports = tmp_path / "reports"
    artifacts = tmp_path / "new"
    rows = []
    for year in [2021, 2022, 2023, 2024, 2026]:
        for i in range(10):
            row = {
                "bout_id": f"{year}-{i}",
                "event_id": f"e{year}-{i}",
                "event_date": f"{year}-08-01",
                "fighter_a_id": "a",
                "fighter_b_id": "b",
                "target": i % 2,
            }
            row.update({name: 0.0 for name in v3.WEEKLY_FEATURES})
            rows.append(row)
    frame = pd.DataFrame(rows)
    labels = frame[study.IDENTITY[:6]].assign(outcome="decisive", winner_id="a")
    label_bytes = labels.to_csv(index=False).encode()
    (baseline_dir / "fight_outcomes.csv").write_bytes(label_bytes)
    (baseline_dir / "model.joblib").write_bytes(b"frozen-baseline-model")
    (baseline_dir / "context.joblib").write_bytes(b"frozen-baseline-context")
    (baseline_dir / "metadata.json").write_text("{}")
    baseline = SimpleNamespace(
        labels=labels,
        context=None,
        sha256="b" * 64,
        outcome_labels_sha256=hashlib.sha256(label_bytes).hexdigest(),
        metadata={"model_version": "v2-fixture"},
    )
    baseline.prepare = lambda bundle: baseline
    monkeypatch.setattr(v3.v2, "WinnerV2Engine", lambda directory: baseline)
    monkeypatch.setattr(v3.v2, "_validate_labels", lambda labels: None)
    monkeypatch.setattr(v3.v2, "training_frame", lambda *args: frame.copy())
    monkeypatch.setattr(v3, "extend_frame", lambda bundle, frame: frame)
    monkeypatch.setattr(v3, "extend_weekly_frame", lambda bundle, frame, *args: frame)
    monkeypatch.setattr(
        v3,
        "ReliabilityHistory",
        lambda bundle: SimpleNamespace(dates={"a": ["2020-01-01"], "b": ["2020-01-01"] * 8}),
    )
    monkeypatch.setattr(
        study, "load_ranking_context", lambda directory: SimpleNamespace(provenance={"csv_sha256": "c" * 64})
    )
    calls = []

    def fit(frame, candidate, train_before, calibration_before, components=None):
        assert (reports / "winner_v3_protocol.json").exists()
        if train_before == "2023-07-01":
            assert (reports / "winner_v3_selection.json").exists()
        calls.append((candidate, train_before, calibration_before))
        if candidate in ("blend_50", "weekly_blend_50"):
            assert components["baseline_v2"]["train_before"] == train_before
            assert (
                components["weekly_rank_hgb" if candidate == "weekly_blend_50" else "reliability_hgb"][
                    "calibration_before"
                ]
                == calibration_before
            )
        return {
            "candidate": candidate,
            "train_before": train_before,
            "calibration_before": calibration_before,
            "train_rows": int((frame.event_date < train_before).sum()),
            "calibration_rows": int(
                ((frame.event_date >= train_before) & (frame.event_date < calibration_before)).sum()
            ),
            "feature_names": v3.feature_names(candidate),
            "temperature": 1.0,
        }

    def probability(artifact, frame):
        quality = {
            "baseline_v2": 0.55,
            "reliability_hgb": 0.65,
            "weekly_rank_hgb": 0.75,
            "blend_50": 0.60,
            "weekly_blend_50": 0.70,
        }[artifact["candidate"]]
        return np.where(frame.target == 1, quality, 1 - quality)

    monkeypatch.setattr(v3, "fit_candidate", fit)
    monkeypatch.setattr(v3, "probability", probability)
    monkeypatch.setattr(v3, "validate_artifact", lambda artifact: None)
    bundle = SimpleNamespace(provenance={"fixture": True}, bouts=pd.DataFrame({"event_date": ["2026-10-17"]}))
    result = study.run_winner_v3_study(bundle, tmp_path, artifacts, reports, baseline_dir)
    assert result["selected_candidate"] == "weekly_rank_hgb" and result["promotion_passed"]
    assert result["training_cutoff_exclusive"] == "2026-07-01"
    assert result["calibration_cutoff_exclusive"] == "2026-10-18"
    scored = pd.read_csv(reports / "winner_v3_predictions.csv")
    assert set(scored[scored.stage == "retrospective_2024"].candidate) == {"baseline_v2", "weekly_rank_hgb"}
    assert len([x for x in calls if x[1] == "2023-07-01"]) == 2
    assert (artifacts / "baseline/context.joblib").read_bytes() == b"frozen-baseline-context"
    assert (artifacts / "fight_outcomes.csv").read_bytes() == label_bytes
    assert result["artifact_sha256"] == hashlib.sha256((artifacts / "model.joblib").read_bytes()).hexdigest()
    assert (
        result["ranking_context_sha256"]
        == hashlib.sha256((artifacts / "ranking_context.joblib").read_bytes()).hexdigest()
    )
    count = len(calls)
    before = (artifacts / "metadata.json").read_bytes()
    engine = SimpleNamespace(metadata=json.loads(before))
    engine.prepare = lambda bundle: engine
    monkeypatch.setattr(v3, "WinnerV3Engine", lambda directory: engine)
    reused = study.run_winner_v3_study(bundle, tmp_path, artifacts, reports, baseline_dir)
    assert reused["artifact_reused"] and len(calls) == count
    assert (artifacts / "metadata.json").read_bytes() == before
