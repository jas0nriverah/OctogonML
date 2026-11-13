# Model evidence and historical snapshots

This page preserves the detailed evidence previously carried by the README. Documentation reviewed November 12, 2026. The packaged completed-fight history ends **October 17, 2026**; the announced-card and portrait examples reflect their retained observation dates. They are historical snapshots, not a live schedule, current ranking feed, or fresh evaluation.

The default winner remains **V2**, strike inference uses **V3**, and elapsed duration uses its independent classifier. Failed winner V3/V4 studies remain research artifacts. A successful LLM workflow does not validate its experimental probability overlay.

## Historical data and interface snapshots

The pinned completed-fight snapshot contains **6,639 bouts, 13,278 athlete-bout observations, and 2,074 fighters**, spanning February 2013–October 17, 2026. It combines Noah Bair's UFC Data with an exact-ID UFCStats increment from Greco1899. [Attribution](../data/ATTRIBUTION.md) documents licenses, commits, hashes, retrieval times, and transformations. Raw source downloads are Git-ignored and excluded from source releases. An explicit ingestion command can recover pinned sources for an independent refit; offline serving uses the packaged normalized snapshot.

The historical October 22, 2026 schedule observation covers six announced events and 65 bouts through December 31. It produces **118 athlete projections for 59 matched bouts**; six unresolved bouts remain visible without forecasts. Times are published card/segment starts, not individual walkout times. Bookings can change.

- [Upcoming projections](../reports/upcoming_predictions.csv) and [complete forecast payloads](../reports/upcoming_predictions.json)
- [Preserved V2 issuance](../reports/issued_forecasts.json) and [separate V3 issuance](../reports/issued_forecasts_v3.json)

Issued pre-event forecasts and reconstructed historical predictions are different evidence. The retained October issuance reports contain no settled outcomes; they do not establish their settlement status as of November 12, 2026; a new database begins with its own empty ledger. Reported strike thresholds are synthetic, with no live market feed.

**Matchup Lab** presents winner, duration, and strike models separately. Observed comparisons cover striking output, absorbed strikes, net differential, accuracy, defense, duration, and sample sizes for last-five/all-covered windows. Rates use ratio-of-sums aggregation. Covered UFC history since 2013 is not a complete MMA career record. More projected strikes do not determine who wins; completing scheduled time does not prove the bout ends in a decision.

Hypothetical scenarios require known last covered UFC divisions within the same competition category, separated by at most one class. Wider gaps are marked **Unrealistic matchup**; unknown classes remain unavailable. This is an application policy, not official matchmaking eligibility. Division ceilings are not measured body weights.

**Fighters** provides searchable, division-filtered profiles for all 2,074 canonical athletes. **2,036 have verified portraits**: 1,844 ESPN and 192 UFC. The other 38 use initials: 36 lack a qualifying verified portrait and two have ambiguous identities. A filename identity audit removed 96 unverified, opponent, or placeholder assets. [Photo coverage](../data/fighter_profiles/ufc_coverage.json) and the [asset audit](../data/fighter_profiles/ufc_image_identity_audit.json) retain the evidence; no images are generated or rehosted.

A separate [official UFC rankings observation](../data/rankings/current.json) contains 206 mapped entries across 13 boards. The media footer gives “Tuesday, Oct. 20” without a year; the [media snapshot](../data/rankings/current_media.json) explicitly labels its inferred 2026 date and retains the actual retrieval timestamp, with `published_board_date` null. Display observations never backfill historical model inputs. Missing or stale dated ranking evidence means **unknown**, not unranked.

## Model evidence

All results below are retrospective. Candidate selection uses declared development periods; 2024 was previously inspected in this project and is not an untouched holdout. These retained reports do not establish prospective performance.

### Decisive winner

The gated **99-feature Winner V2** uses lagged striking, grappling, control, prior-result strength/form, age, physical measurements, division contrasts, and dated ranking evidence. It is separate from the strike and duration estimators. Training/calibration exclude draws; opposite fighter perspectives produce complementary probabilities.

| 2024 comparison · 507 decisive bouts | Accuracy | Log loss ↓ | Brier ↓ |
| --- | ---: | ---: | ---: |
| Preserved 46-feature winner baseline | 57.79% | 0.674632 | 0.240965 |
| Winner V2 | **63.91%** | **0.642838** | **0.225529** |

V2 was chosen from four fixed candidates using **2021–2023 only**, not the 2024 results. Development log loss improves from 0.674881 to 0.653719. Its paired event-bootstrap difference is −0.021162, with a 95% interval of [−0.030922, −0.011478] across 128 events. The stored promotion gates also check Brier nonregression, calibration stability, and annual consistency. These checks do not establish future accuracy or justify an extreme probability for an individual matchup.

See [Winner V2 evaluation](../reports/winner_v2_evaluation.json), [selection](../reports/winner_v2_selection.json), [prediction rows](../reports/winner_v2_predictions.csv), and [metadata](../artifacts/winner_v2/metadata.json). The default independent winner channel is `auto`, with `CAGEMETRICS_WINNER_CHANNEL=v1` for rollback.

Two additional studies tested exposure shrinkage, opponent-adjusted striking, full dated weekly ranking boards, and ranked-opponent history. Neither passed the predeclared replacement gates; V2 remains the default. Their candidates, failed checks, subgroup results, and retrospective comparisons are preserved in [winner research](WINNER_RESEARCH.md).

### Elapsed rounds

The independent duration model estimates a distribution over 20 time categories: half-round interiors plus exact boundary atoms through five rounds, with unsupported categories masked in three-round scenarios. Over/under/push probabilities preserve exact boundary mass; interior seconds follow an explicit uniform assumption.

| 2024 comparison · 507 bouts | Scheduled empirical baseline | Duration HGB |
| --- | ---: | ---: |
| Category log loss ↓ | 1.544579 | **1.534827** |
| Survival Brier ↓ | 0.188263 | **0.180496** |
| Completion Brier ↓ | 0.250764 | **0.238192** |

Nominal 80% duration intervals cover 93.7% of these outcomes: conservative coverage, not evidence of calibrated 80% intervals. See [duration evaluation](../reports/rounds_evaluation.json) and [row-level predictions](../reports/rounds_predictions.csv).

### Significant strikes

The V3 strike channel uses 33 pre-fight inputs and a duration-conditioned negative-binomial mixture. A symmetric gate weights full-duration and early-stop conditional count models; realized duration is a training label, never an inference input.

Development NLL improves from 4.75050 to 4.72267 on 2021–2023. The separately reported, previously seen 2024 comparison improves from 4.78220 to 4.73010; its paired event-bootstrap NLL difference is −0.05210 (95% interval −0.06948 to −0.03475). **No mean-count MAE improvement is established.**

The preserved V1 historical report covers 1,870 athlete forecasts during January 2025–October 7, 2026: boosting MAE **25.68** versus **28.24** for the last-five baseline. These are metrics for that historical evaluation, not a new test of the current V3 artifact. The October 8–22 replay has 50 outcomes, MAE 20.85, and +12.74 forecast-minus-actual bias; it was reconstructed after the events.

See the [model card](../MODEL_CARD.md), [V1 report](../reports/evaluation.json), [V2 feature study](../reports/experiments.json), [V3 study](../reports/research.json), and [recent replay](../reports/recent_predictions.json).

### Experimental AI guidance

The narrator receives server-authored evidence tied to saved forecast IDs. It selects sourced advantages, an exact comparison, and a supported conditional route from an evidence-linked catalogue; missing support permits explicit abstention. Rationale and the bounded adjustment proposal remain LLM output. It can propose **at most ±5 percentage points** to the winner probability, with complementary probabilities displayed separately from the unchanged ML baseline. This bound is a policy, not a learned ensemble weight. The adjustment has **no established backtest or calibration performance**; the 63.91% figure above evaluates ML alone.

Analyses retain all server evidence, the IDs of the bounded subset used by the narrator, cited claims, provider/model identity, prompt version, original prediction IDs, and experimental guidance in immutable SQL records. Repeated identical requests reuse the saved result. Client-supplied facts and mismatched athlete, round, date, or source snapshots are rejected.

Matchup reports add a factual comparison, recent results, dated strength-of-schedule evidence, opponent-adjusted striking, and the separate duration outlook. Context-only statistics are labeled separately from inputs actually used by the saved model. Unknown ranks do not become unranked observations, and no finish method or exact winning round is invented.

For an observed unranked athlete with at most three covered UFC bouts facing a verified current top-five/champion opponent with at least eight, the interface withholds a published favorite and collapses the numerical research output. This is a stated publication policy, not a learned confidence score or an altered probability. Full forecast records remain inspectable.

The retained [Riley–Lopes example](../reports/matchup-analysis-riley-lopes.json) retains 68 facts, with 24 supplied to the local model, including both latest results and dated opponent-quality context. The compact ID-selection protocol passed live checks in both fighter orders in 14.8 and 15.0 seconds; saved GET and repeated POST results matched exactly. Both proposed zero probability adjustment. See [verification details](../reports/matchup-analysis-riley-lopes-verification.json). These are workflow observations on Apple M4, not a capacity benchmark or predictive improvement claim. [Earlier enriched](../reports/matchup-analysis-local-enriched.json) and [original](../reports/matchup-analysis-example.json) responses preserve their previous prompt versions.


## Reproducing and interpreting reports

Use the [model card](../MODEL_CARD.md), [engineering decisions](ENGINEERING.md), and [winner research protocol](WINNER_RESEARCH.md) for definitions and limitations. A frozen report can be inspected offline. A new fit must retain its own source/protocol fingerprint and fresh artifact/report directory; replacing a report does not turn a previously inspected year into an untouched holdout.

Model-quality reports and [runtime benchmarks](benchmarking.md) answer different questions. Chronological selection, reserved calibration, paired event bootstraps, and prospective issuance each have distinct scopes. The retained forecasts demonstrate issuance and lineage; they do not by themselves establish profitable decisions or future accuracy.
