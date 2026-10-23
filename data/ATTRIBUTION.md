# Data and model attribution

## Completed fights

Historical base: Noah Bair, **UFC Data v0.1.1 (2026)**, [Zenodo citation](https://doi.org/10.5281/zenodo.23458636), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The pinned source repository is [nlbair/ufc-data](https://github.com/nlbair/ufc-data), revision `a9c65430b3c5cb6e6351940f36e03b2adeb908d0`. The retained source documentation credits UFCStats, MMA Decisions, BestFightOdds, the martj42 official ranking archive, and UFCalendar. Original notice: [CC-BY-4.0-notice.txt](licenses/CC-BY-4.0-notice.txt).

October 17 increment: [Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats), revision `3e2f3db5751c14b5d4ea52e41921982ca23b9b23`, repository license GPL-3.0. Original license: [GPL-3.0.txt](licenses/GPL-3.0.txt), also preserved with the raw snapshot. No scraper implementation was copied.

Project transformations include exact canonical identity mapping, round aggregation, standard-format filtering, no-contest/overturned exclusions, outcome orientation, point-in-time features, and separate grappling/bio/ranking normalization. The completed snapshot retains 6,639 bouts, 13,278 athlete-bout observations, and 2,074 fighters through October 17, 2026. Original URLs, commits, bytes, hashes, and observation timestamps are recorded in [processed provenance](processed/provenance.json); exclusions appear in [quality](processed/quality.json).

Raw files are Git-ignored. They are retained in the prepared project archive and can be recovered explicitly from pinned sources. Versioned download caches use `raw/<owner>/<commit>/<file>`. Normalized records and derived reports do not establish that later-collected source values were available at a historical prediction time. Recent replays are reconstructed after events, not issued pre-event predictions. No commercial betting returns or market-line accuracy are claimed.

## Historical auxiliary context

Winner V2's auxiliary observations come from the same pinned wide UFC Data source and Greco1899 round/profile files. Exact bout/fighter IDs bind all 13,278 grappling rows. The [auxiliary provenance](processed/winner_context_provenance.json) and [quality audit](processed/winner_context_quality.json) retain source/codebook digests and coverage.

UFC Data's source notes attribute weekly rankings through June 2026 to [martj42/ufc-rankings](https://www.kaggle.com/datasets/martj42/ufc-rankings), with UFCalendar's ranking API thereafter. The retained wide file contains dated, bout-linked ranking observations rather than a full current board archive. Boards must strictly predate a modeled bout; evidence older than 35 days is masked. Absent/stale evidence is unknown, not proof of being unranked. DOB supplies scenario age; height/reach are retrospectively observed UFCStats profile measurements, not verified historical measurement availability.

## Full dated weekly ranking archive

Research Winner V3/V4 additionally retain Mart Jürisoo (`martj42`), [UFC Rankings, version 47](https://www.kaggle.com/datasets/martj42/ufc-rankings/versions/47), labeled [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/) by its author. The author describes recovering official UFC media rankings through Internet Archive, with missing weeks. This is separate from the bout-linked V2 context and canonical completed-fight manifest; it does not modify either frozen artifact.

The pinned archive contains 100,065 rows across 532 dated boards from February 4, 2013 through June 2, 2026. Normalization maps 99,900 rows by exact canonical identities/documented source-anchored aliases. Of 6,643 division/date boards, 4,350 satisfy conservative completeness checks. Partial, ambiguous or contradictory boards cannot establish unranked status by absence. Retrieval after historical events supports reconstruction, not proof of real-time archival availability. Source version, original metadata, hashes, observation time and licensing are retained in [historical provenance](rankings/history/provenance.json), with [detailed attribution and identity policy](rankings/history/ATTRIBUTION.md).

## Announced schedule

[ESPN's public UFC scoreboard](https://site.api.espn.com/apis/site/v2/sports/mma/ufc/scoreboard) and athlete APIs supply the observed October 22–December 31, 2026 schedule. Exact original responses, observation timestamps, URLs, and SHA256 hashes are retained under [schedules](schedules/). These are published bookings at observation time, subject to changes/cancellations. Card and segment starts are not individual walkout times.

## Portraits

Images remain at original ESPN/UFC media URLs. No image is generated, downloaded for rehosting, claimed as project artwork, or used as a model feature. ESPN/UFC names and media retain their respective owners' rights. A photo does not itself establish a canonical model identity. Entries retain profile/source links, observation times, SHA evidence, and credit in [fighter_photos.json](fighter_photos.json); unavailable images use initials.

The ESPN player-search/profile workflow establishes **1,844 verified portraits**, preserving the initial 134 entries. It requires a unique exact normalized canonical name, ESPN MMA UID/profile ID, an observed individual CDN URL, and a successful image-header check. Its original source observations remain in `fighter_profiles/espn_player_search.jsonl.gz`, with the [original ESPN coverage audit](fighter_profiles/coverage.json). That audit's 230 missing cases describe the ESPN-only stage, not final coverage.

Official [UFC athlete profiles](https://www.ufc.com/athletes) supply **192 additional retained portraits** after the asset identity audit. Final coverage is **2,036/2,074 (98.17%)**. The other 38 comprise 30 unresolved primary-profile cases, six asset-identity quarantines, and two ambiguous canonical athletes named Bruno Silva. Unavailable means no qualifying verified image was established, not that no photograph exists elsewhere.

UFC profile identity requires an exact canonical/source-linked H1, or primary structured name plus the exact DOB from a retained same-ID UFCStats profile. Explicit Latin transliteration is allowed; fuzzy name matching is not. The decoded image filename must contain the canonical/source-verified name tokens, and its named athlete asset must be observed in the HTML and return HTTP 200 image headers. This second check matters: official athlete pages can embed opponent headshots.

The complete audit removed **96 unverified/opponent/placeholder assets across 90 entries**, promoted three correct observed alternatives, and quarantined six unsupported primaries. Exact previous/replacement metadata is in [ufc_image_identity_audit.json](fighter_profiles/ufc_image_identity_audit.json). Primary HTML, hashes, observed UTC, and identity evidence remain in `fighter_profiles/ufc_portraits.jsonl.gz`; the [UFC coverage audit](fighter_profiles/ufc_coverage.json) records final counts. Prior photo-index JSON bytes are retained under `fighter_profiles/photo_index_archive/`. All original 1,844 ESPN entries remain unchanged.

`python scripts/refresh_fighter_photos.py --ufc-fallback` resumes missing-profile recovery; `--audit-ufc-images` repeats asset identity checks; `--retry-failures` explicitly revisits unresolved queries. Cached replay preserves actual source observation time; `generated_at` records only the new summary run. Schedule refresh preserves verified broad coverage. No photo refresh rewrites an issued forecast.

## Current ranking display

The [official UFC ranking page](https://www.ufc.com/rankings) is frozen separately in [rankings/current.json](rankings/current.json): 206 entries across 13 boards, mapped to canonical IDs with unique exact names/verified aliases and explicit Latin transliteration. The snapshot records original HTML SHA, source URL, observation UTC, rank/champion/P4P fields, official athlete links, and a compressed raw cache reference.

The retained media table has an explicit **yearless** footer, “Last updated: Tuesday, Oct. 20”. The separately parsed [media snapshot](rankings/current_media.json) records `published_label` and `published_date_inferred=2026-10-20`, with the year explicitly inferred from the unique nearby weekday/date match. `published_board_date` remains null because the page omits the year. Actual source availability begins at `2026-10-22T20:17:16.420435+00:00`, never at the inferred October 20 date. The media poll is distinct from the page's algorithmic Meta table, which is excluded.

Neither current snapshot backfills historical model inputs. In the complete, fully identity-resolved media board, absence from the relevant division may establish explicitly observed unranked status; partial/unresolved boards and missing or stale history remain unknown. `python scripts/refresh_rankings.py` is an explicit refresh; previous JSON bytes are archived.

## Optional language model

The prepared local explanation runtime is [Ollama 0.40.8](https://github.com/ollama/ollama/releases/tag/v0.40.8) with [Qwen3:4b](https://ollama.com/library/qwen3:4b), Q4_K_M. Qwen model weights use Apache 2.0; the model's original license is retained in its downloaded Ollama layer. Runtime/weights reside outside the application repository, not in the Git checkout or container. See [local setup](../docs/LOCAL_LLM.md) for verified digests and server configuration.

Generated explanations and experimental guidance are derived outputs, not source ground truth or validated probability estimates. This independent project is not affiliated with UFC, ESPN, the data authors, Ollama, or the model providers.
