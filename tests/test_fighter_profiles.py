"""Every canonical identity has an observed profile, including absent histories."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from cagemetrics.data import DatasetBundle
from cagemetrics.fighters import FighterCatalog, public_photos
from cagemetrics.config import Settings
from cagemetrics.db import Repository, SourceSnapshot


@pytest.fixture
def bundle():
    bouts, stats = [], []
    for i, (seconds, landed, attempted, absorbed, faced) in enumerate(
        [
            (60, 10, 20, 5, 10),
            (900, 30, 100, 20, 50),
            (300, 20, 50, 8, 40),
            (600, 40, 80, 15, 60),
            (120, 10, 30, 6, 20),
            (300, 25, 50, 10, 40),
        ]
    ):
        bid = str(i)
        bouts.append(
            dict(
                bout_id=bid,
                event_id="e" + bid,
                event_date=f"2020-01-{i + 1:02}",
                event_name="Card " + bid,
                fighter_a_id="a",
                fighter_b_id="b",
                duration_seconds=seconds,
                scheduled_rounds=3,
                division="heavyweight" if i == 5 else "lightweight",
                method="Decision" if seconds == 900 else "KO/TKO",
                source="fixture",
            )
        )
        stats.extend(
            [
                dict(
                    bout_id=bid,
                    fighter_id="a",
                    opponent_id="b",
                    sig_landed=landed,
                    sig_attempted=attempted,
                    source="fixture",
                ),
                dict(
                    bout_id=bid,
                    fighter_id="b",
                    opponent_id="a",
                    sig_landed=absorbed,
                    sig_attempted=faced,
                    source="fixture",
                ),
            ]
        )
    return DatasetBundle(
        pd.DataFrame(
            [
                {"fighter_id": i, "name": n}
                for i, n in [("a", "Álpha"), ("b", "Beta"), ("c", "No History"), ("d", "Beta")]
            ]
        ),
        pd.DataFrame(bouts),
        pd.DataFrame(stats),
        {"fixture": {"commit": "pinned"}},
        {},
    )


def test_stable_complete_directory_pagination_filter_and_canonical_identity(bundle):
    catalog = FighterCatalog(bundle, "snapshot")
    page1 = catalog.directory("2020-02-01", page=1, per_page=2)
    page2 = catalog.directory("2020-02-01", page=2, per_page=2)
    assert [r["fighter_id"] for r in page1["fighters"] + page2["fighters"]] == ["a", "b", "d", "c"]
    assert page1["total_canonical"] == 4 and page1["pagination"]["total_pages"] == 2
    assert page1["pagination"]["has_next"] and not page2["pagination"]["has_next"]
    assert page1 == catalog.directory("2020-02-01", page=1, per_page=2)
    assert catalog.directory("2020-02-01", q="alpha")["fighters"][0]["fighter_id"] == "a"
    assert catalog.directory("2020-02-01", q="Beta")["total"] == 2
    assert catalog.directory("2020-02-01", division="heavyweight")["total"] == 2
    assert catalog.directory("2020-02-01", division="lightweight")["total"] == 0
    assert catalog.directory("2020-01-06", division="lightweight")["total"] == 2
    assert catalog.directory("2020-02-01", page=100)["fighters"] == []
    assert page1["source_snapshot_id"] == "snapshot"


def test_profile_exposure_ratios_samples_last5_and_no_invented_record(bundle):
    profile = FighterCatalog(bundle, "source").profile("a", "2020-02-01")
    career, last5 = profile["windows"]["career"], profile["windows"]["last5"]
    assert career["covered_bouts"] == 6 and last5["covered_bouts"] == 5
    assert career["sig_landed_per_min"] == pytest.approx(135 / 38)
    assert last5["sig_landed_per_min"] == pytest.approx(125 / 37)
    assert career["accuracy"] == pytest.approx(135 / 330)
    assert profile["history"][0]["bout_id"] == "5"
    assert profile["coverage"]["fighter_start"] == "2020-01-01"
    assert profile["coverage"]["fighter_end"] == "2020-01-06"
    assert profile["data_window"]["end"] == "2020-01-06"
    assert profile["profile_sources"]["verified_attributes"] == {}
    assert not {"wins", "losses", "height", "reach", "age", "record"} & set(profile["fighter"])


def test_absent_history_and_zero_denominators_are_null_not_model_priors(bundle):
    catalog = FighterCatalog(bundle, "source")
    empty = catalog.profile("c", "2020-02-01")
    assert empty["history"] == [] and empty["fighter"]["covered_bouts"] == 0
    assert empty["fighter"]["last_fight_date"] is None and empty["fighter"]["latest_division"] is None
    assert empty["windows"]["career"]["sig_landed_per_min"] is None
    assert empty["windows"]["career"]["observed_minutes"] is None
    zero = deepcopy(bundle)
    zero.stats[["sig_landed", "sig_attempted"]] = 0
    summary = FighterCatalog(zero, "source").profile("a", "2020-02-01")["windows"]["career"]
    assert summary["accuracy"] is None and summary["defense"] is None
    assert summary["sig_landed_per_min"] == 0
    with pytest.raises(KeyError):
        catalog.profile("not-canonical", "2020-02-01")


def test_same_day_future_mutations_cannot_change_profile_or_division(bundle):
    original = FighterCatalog(bundle, "source").profile("a", "2020-01-03")
    changed = deepcopy(bundle)
    future = changed.bouts.loc[changed.bouts.event_date >= "2020-01-03", "bout_id"]
    changed.stats.loc[changed.stats.bout_id.isin(future), ["sig_landed", "sig_attempted"]] = 9000
    changed.bouts.loc[changed.bouts.bout_id.isin(future), "division"] = "middleweight"
    assert FighterCatalog(changed, "source").profile("a", "2020-01-03") == original
    assert original["fighter"]["covered_bouts"] == 2
    assert original["fighter"]["latest_division"] == "Lightweight"


def test_photo_public_manifest_is_trimmed_sourced_and_invalidates_on_file_change(tmp_path, bundle):
    settings = SimpleNamespace(data_dir=tmp_path, project_dir=tmp_path)
    path = tmp_path / "fighter_photos.json"
    data = {
        "observed_at": "2026-10-22T00:00:00Z",
        "source": "ESPN profile API",
        "athletes": {"large": "duplicated"},
        "fighters": {
            "a": {
                "name": "Álpha",
                "photo_url": "https://example.com/a.png",
                "profile_url": "https://example.com/profile/a",
                "source_url": "https://example.com/source/a",
                "credit": "Source",
                "observed_at": "2026-10-22T00:00:00Z",
                "espn_id": "123",
                "sha256": "private-large-source-metadata",
            }
        },
    }
    path.write_text(json.dumps(data))
    public = public_photos(settings)
    assert set(public) == {"observed_at", "source", "fighters"}
    assert "sha256" not in public["fighters"]["a"]
    profile = FighterCatalog(bundle, "source").profile("a", "2020-02-01", public)
    assert profile["fighter"]["photo_url"] == "https://example.com/a.png"
    assert profile["profile_sources"]["photo"]["observed_at"] == "2026-10-22T00:00:00Z"
    data["fighters"]["a"]["photo_url"] = "https://example.com/updated-long-name.png"
    path.write_text(json.dumps(data))
    assert public_photos(settings)["fighters"]["a"]["photo_url"].endswith("updated-long-name.png")


@pytest.fixture
def http_client(tmp_path, bundle, monkeypatch):
    import cagemetrics.api as api

    repo = Repository(f"sqlite:///{tmp_path / 'profiles.db'}")
    repo.create_schema()
    repo.store_dataset(bundle)

    class FrozenForecaster:
        def __init__(self, settings, repository):
            self.bundle = repository.dataset()
            self.snapshot_id = repository.latest_snapshot_id()
            self.metadata = {"model_version": "profiles-test"}
            self.sha256 = "a" * 64
            self.predictions = 0

        def feature_frame(self, *args):
            return pd.DataFrame({"own_prior_bouts": [0.0]})

        def predict(self, *args):
            self.predictions += 1
            return {"mean": 40.0}

    monkeypatch.setattr(api, "LiveForecaster", FrozenForecaster)
    settings = Settings(
        tmp_path, tmp_path / "data", tmp_path / "artifacts", tmp_path / "reports", str(repo.engine.url)
    )
    app = api.create_app(settings)
    with TestClient(app) as client:
        yield client, repo, app
    repo.engine.dispose()


def test_http_profiles_directory_validation_deeplinks_and_old_list_contract(http_client):
    client, repo, app = http_client
    assert client.get("/health/ready").status_code == 200
    assert client.get("/api/v1/fighters").json()["total"] == 4
    directory = client.get("/api/v1/fighter-directory?per_page=2&page=1").json()
    assert directory["total_canonical"] == 4
    assert len(directory["fighters"]) == 2
    assert client.get("/api/v1/fighter-directory?per_page=61").status_code == 422
    assert client.get("/api/v1/fighter-directory?page=0").status_code == 422
    assert client.get("/api/v1/fighters/no-such-id").status_code == 404
    assert client.get("/api/v1/fighters/a?as_of=not-a-date").status_code == 422
    profile = client.get("/api/v1/fighters/a?as_of=2020-01-03").json()
    assert profile["fighter"]["covered_bouts"] == 2
    assert profile["data_window"]["end"] == "2020-01-02"
    assert client.get("/api/v1/fighters/c").json()["history"] == []
    assert "text/html" in client.get("/fighters").headers["content-type"]
    assert "text/html" in client.get("/fighters/a").headers["content-type"]
    assert client.get("/fighters/no-such-id").status_code == 404
    assert app.state.forecaster.predictions == 1  # Only startup warmup; profiles never forecast.


def test_http_profiles_refuse_silent_source_snapshot_changes(http_client):
    client, repo, app = http_client
    original = client.get("/api/v1/fighters/a").json()["source_snapshot_id"]
    with repo.sessions.begin() as session:
        session.add(
            SourceSnapshot(
                snapshot_id="changed", manifest={}, observed_at=datetime(2099, 1, 1, tzinfo=timezone.utc)
            )
        )
    assert client.get("/api/v1/fighter-directory").status_code == 503
    assert original == app.state.forecaster.snapshot_id
