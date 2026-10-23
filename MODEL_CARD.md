# OctogonML model card

## Scope

OctogonML serves three independently trained objectives:

| Objective | Output | Current artifact |
| --- | --- | --- |
| Significant strikes | One athlete's full-bout count distribution, intervals, and over/under/push probabilities | V3 duration-conditioned negative-binomial mixture |
| Decisive winner | Complementary probabilities conditional on a decisive result | `winner-v2-d04e8f09a8abcbb4` |
| Elapsed time | Round-threshold survival, boundary mass, scheduled-time completion, and approximate mean duration | `rounds-8f9fc4ebaff2963e` |

These outputs support research and prospective evaluation. **Research only — not betting advice.** More expected strikes do not imply a greater win probability. Completing scheduled time includes a stoppage recorded at the final bell and does not establish a decision. No calibrated joint distribution over strikes, duration, and winner is claimed.

An optional LLM produces evidence-linked explanations and a separate experimental probability overlay. It is not the winner classifier, and its output is not included in the reported classifier metrics.

## Data, identity, and time

The normalized pinned snapshot contains **6,639 bouts, 13,278 athlete-bout observations, and 2,074 fighters**, from February 2013 through October 17, 2026. Noah Bair's UFC Data supplies the historical base; Greco1899 supplies an exact-ID UFCStats increment. Unsupported bout formats, no contests, and overturned results are excluded. See [source attribution](data/ATTRIBUTION.md), [provenance](data/processed/provenance.json), and [normalization quality](data/processed/quality.json).

Features use completed events **strictly before** the forecast's history cutoff. Same-day outcomes cannot influence each other. The October 8–22 replay additionally freezes history before October 8. No current-bout result, realized duration, final round, or current-bout performance statistic is an input. Opponent and bout IDs are canonical source identities; ambiguous matches are quarantined rather than guessed.

Raw sources were observed after the historical evaluation events. Event-date reconstruction does not prove contemporaneous source availability. Recorded DOB supplies age at the scenario date. Heights and reaches are retrospectively observed profile measurements, with no claim that each historical measurement was published at that time.

Auxiliary grappling normalization covers every canonical athlete-bout pair: takedowns landed/attempted, submission attempts, control seconds, and knockdowns. Career/last-five rates divide aggregate counts by aggregate observed minutes. Unknown values and sparse history are explicitly represented.

Historical ranking inputs come from **dated, bout-linked** weekly-board observations, not a complete current ranking feed. A board must strictly predate the forecast, and evidence older than 35 days is masked. Missing or stale evidence is unknown, never proof of being unranked. A separate [official ranking display snapshot](data/rankings/current.json) contains 206 mapped entries across 13 boards observed on October 22. Its media footer explicitly says “Last updated: Tuesday, Oct. 20” without a year. The [separate media snapshot](data/rankings/current_media.json) records October 20, 2026 as an explicitly inferred date; `published_board_date` stays null. Live eligibility starts at its actual October 22 observation timestamp, not the inferred date. That display observation does not backfill historical predictors or silently update frozen model inputs.

## Winner V2

The binary target uses **6,590 decisive bouts** and excludes 49 canonical draws. No contests and overturned bouts are already excluded by normalization. All 6,626 labels overlapping the two raw sources agree, including the draws. [The label audit](reports/winner_labels_audit.json) records exact source identities and the narrowly resolved source-name typos.

The preserved winner baseline uses 46 signed lagged contrasts: striking history, prior-result Elo and win rate, covered experience, prior division ceilings, and scheduled-round interactions. V2 extends this to **99 inputs** with grappling/control, age/physical observations, dated ranking evidence and missingness, and recency-weighted Elo, opponent strength, win quality, streak, and finish-win summaries. These are learned inputs, not manual probability bonuses. No betting odds or requested example probability enters training.

Training mirrors both fighter perspectives. Inference symmetrizes opposite contrasts, then applies zero-intercept temperature calibration on a disjoint temporal sample. Reversing fighter order complements the probability. Draw/no-contest probabilities are not modeled. The application permits only known same/adjacent last covered UFC divisions within a competition category; this is a scenario policy, not official eligibility or measured body weight.

### Selection and promotion

Four fixed candidates—preserved HGB baseline, enriched logistic regression, enriched HGB, and regularized enriched HGB—are evaluated on **2021–2023 only**. Each annual fold trains before July 1 of the preceding year, calibrates during its July–December period, and scores the next calendar year. The selected candidate is `enriched_hgb_regularized`.

Promotion requires:

- Development log-loss improvement of at least 0.002.
- No pooled Brier regression and no more than 0.015 ECE regression.
- Improvement in at least two development years and no annual log-loss regression above 0.005.
- A paired event-bootstrap 95% upper bound below zero for the log-loss difference.

All stored gates pass. Development log loss falls from **0.674881 to 0.653719** across 1,497 decisive bouts. The difference is −0.021162, with a 95% event-bootstrap interval **[−0.030922, −0.011478]**, using 2,000 resamples over 128 events. This interval conditions on fitted models and excludes training/selection uncertainty.

### Retrospective 2024 comparison

| Model · 507 decisive bouts | Log loss ↓ | Brier ↓ | Accuracy |
| --- | ---: | ---: | ---: |
| Preserved 46-feature winner baseline | 0.674632 | 0.240965 | 57.79% |
| Winner V2 | **0.642838** | **0.225529** | **63.91%** |

2024 was previously inspected during other project research. It was **not used to select or promote V2**, and it is not an untouched project-wide holdout. Exact 50/50 forecasts receive expected tie accuracy of 0.5. Accuracy describes this collection of forecasts; it does not validate an individual requested confidence level or an AI adjustment.

Deployment fits before July 1, 2026 and calibrates on July 1–October 17. Its artifact freezes source, outcome-label and context hashes, implementation fingerprints, feature names, outcome history, and cutoffs. `CAGEMETRICS_WINNER_CHANNEL=auto` prefers verified V2 only when promotion passes; `v1` restores the baseline and `v2` explicitly requests V2. Missing/invalid winner artifacts do not turn strike predictions into outcome forecasts.

Evidence: [V2 evaluation](reports/winner_v2_evaluation.json), [selection](reports/winner_v2_selection.json), [protocol](reports/winner_v2_protocol.json), [predictions](reports/winner_v2_predictions.csv), [metadata](artifacts/winner_v2/metadata.json), and [baseline evaluation](reports/winner_evaluation.json).

## Unpromoted winner research and publication policy

Winner V3 and V4 both failed their unchanged, predeclared development gates on 1,497 bouts scored in 2021–2023. Selected V3 log loss was 0.652766 versus V2's 0.653719: the 0.000952 gain was below the required 0.002, and the paired event-bootstrap interval crossed zero. V4 selected log loss was 0.653430, also below the required gain; its Brier score regressed slightly and its bootstrap interval crossed zero. Better-looking already-seen 2024 results do not authorize promotion. `auto` retains V2; explicit `v3`/`v4` channels are research opt-ins. No validated accuracy improvement is claimed. [Full methods, gates, source provenance and reports](docs/WINNER_RESEARCH.md).

Those research artifacts freeze a separate CC0 full weekly ranking archive through June 2, 2026. Historical lookups require strictly prior board dates and a 35-day freshness limit; unknown rank differs from explicitly unranked on a complete resolved board. Current official media observations are eligible only after their actual retrieval timestamp and never backfill training.

The publication policy `sparse-top5-comparison-v1` withholds a displayed favorite when one athlete has at most three covered UFC bouts and is explicitly observed unranked on a complete current division board, while the opponent has at least eight covered bouts and is verified current champion or top five. Both current rankings must be eligible, known media observations. This conservative presentation rule does not change the stored ML probabilities, fit a confidence score, or substitute a hand-chosen probability. Missing or stale rank alone does not trigger that specific rule. The comparison can still present supported factual context without publishing a favorite.

## Independent elapsed-time model

A separate symmetric HGB classifier uses 33 pre-fight summary inputs. Its target has **20 categories**: each of ten half-round intervals has an interior category and an exact 150-second boundary atom. Three-round scenarios mask categories beyond their scheduled limit. Interior categories distribute probability uniformly over their 149 integer seconds; this is a resolution assumption, not second-by-second empirical calibration.

The scheduled-duration empirical model is the baseline. Candidate selection uses pooled 2021–2023 categorical log loss, with separate annual training and temperature calibration. Development categorical log loss is 1.635408 for HGB versus 1.644593 for the empirical baseline. Deployment uses the same July 1, 2026 fitting cutoff and October 18 exclusive calibration cutoff as the winner models.

| Retrospective 2024 · 507 bouts | Empirical baseline | Duration HGB |
| --- | ---: | ---: |
| Category log loss ↓ | 1.544579 | **1.534827** |
| Survival Brier ↓ | 0.188263 | **0.180496** |
| Scheduled-time completion Brier ↓ | 0.250764 | **0.238192** |
| Mean absolute error, rounds | 1.01544 | **0.97193** |
| Nominal 80% interval coverage | 93.10% | 93.69% |

Survival Brier pools half-round thresholds strictly below each bout's scheduled limit. The broad nominal 80% intervals are conservative; **93.69% coverage is not evidence of calibrated 80% intervals**. All 2024 comparisons remain retrospective.

The API preserves P(time > line), P(time < line), and P(time = line), including integer-round atoms and the full-time atom. Mean duration uses interior-bin midpoint approximation. Round-reach probabilities describe reaching a round's start, not a win or finish method. The service rejects nonmonotone, nonfinite, nonpartitioning, or out-of-clock model output before writing SQL.

Evidence: [evaluation](reports/rounds_evaluation.json), [protocol](reports/rounds_protocol.json), [selection](reports/rounds_selection.json), [predictions](reports/rounds_predictions.csv), and [metadata](artifacts/rounds/metadata.json).

## Significant-strike distribution

V3 retains 33 strictly pre-fight inputs: career/last-five strike summaries, absorbed/attempted pace, accuracy, historical duration/finish share, layoff, experience, opponent equivalents, and scheduled rounds. A shared gate estimates full-duration versus early-stop probability by averaging both perspectives. Two conditional boosted means and separately calibrated negative-binomial dispersions form the actual mixture. Realized duration supplies a training/calibration regime label only. The shared gate does not create a joint distribution over both athletes.

Eight bounded candidates are compared on 2021–2023 annual folds, then the candidate is frozen before the separately reported 2024 comparison. Training precedes the prior July 1; calibration uses the next July–December period. Deployment fits before July 1, 2026 and calibrates on July 1–October 17.

| Period | Global NB NLL | Duration-mixture NLL |
| --- | ---: | ---: |
| 2021–2023 selection | 4.75050 | 4.72267 |
| 2024, previously seen | 4.78220 | 4.73010 |

The declared promotion gates require at least 0.005 lower development NLL, at most 1% worse MAE, and lower confirmation NLL with the same MAE tolerance. They pass. The 2024 event-bootstrap NLL difference is −0.05210 (95% interval −0.06948 to −0.03475). **No MAE improvement is established**: corresponding MAE intervals include zero. Already-seen 2025–2026 diagnostics do not select candidates or decide promotion.

The preserved V1 report covers 1,870 athlete forecasts over 935 bouts/76 events from January 2025–October 7, 2026. Boosting MAE is 25.68 versus 28.24 for last-five prediction and 25.83 for Poisson. Its MAE difference versus Poisson has a 95% interval [−0.49, +0.15]; superiority is not established. These figures evaluate that historical artifact, not current V3 serving.

The October 8–22 replay contains 25 bouts/50 outcomes over two events, with MAE 20.85, bias +12.74 strikes, and nominal 80% interval coverage 90%. History is frozen before October 8, but predictions were reconstructed after the outcomes. Two events cannot establish stable calibration; positive bias indicates recent overprediction.

Evidence: [V1 evaluation](reports/evaluation.json), [V2 study](reports/experiments.json), [V3 study](reports/research.json), [compressed V3 rows](reports/research_predictions.csv.gz), and [recent replay](reports/recent_predictions.json).

## Experimental AI explanation and guidance

`POST /api/v1/matchup-analysis` accepts **saved prediction IDs only**. Server-authored evidence includes compatible forecast outputs, prior observations, provenance, and separately dated ranking display facts. The LLM selects sourced advantages, an exact comparison, and a conditional tactical route from a server-authored catalogue linked to original fact IDs. Eligible routes depend on observed metrics; zero takedowns do not establish takedown defense, and insufficient metrics allow explicit abstention. These catalogue selections still require a valid actual LLM response; they are not a fabricated failure fallback. Rationale and the bounded adjustment proposal remain LLM outputs. Validation checks JSON structure, identities, evidence references, numbers, and prohibited unsupported claims; it is not a complete proof of semantic truth.

The LLM may propose a finite adjustment of zero to five percentage points in either fighter's favor. The server applies the signed shift to a separate experimental probability, clipping to [0, 1] and preserving complements. **The saved ML forecast remains unchanged.** The five-point limit is a policy bound, not a learned ensemble coefficient. No accuracy, calibration, or prospective performance is established for this overlay. The 63.91% Winner V2 figure excludes AI guidance.

Evidence, original IDs, source hash, narrator/model identity, prompt version, output, and guidance are stored in immutable SQL analyses. Equivalent requests reuse the result. Provider timeout/unavailability yields an explicit error; invalid grounded output is not replaced with fabricated analysis. Local Ollama is the default; explicit server-side OpenAI configuration is optional. The basic JSON check and complete local analyses in both fighter orientations passed. Those complete checks took 37.5 and 56.6 seconds; both proposed zero adjustment, preserving the ML baseline. These checked examples do not establish universal semantic correctness or predictive performance. See [local setup](docs/LOCAL_LLM.md).

## Serving, maintenance, and prospective evidence

Models load trusted prebuilt joblib artifacts; startup never fits them. Data, label, implementation, and artifact checksums protect the inference contract. Explicit model channels permit rollback. Integer strike thresholds retain push mass; finite rendered count charts retain an explicit tail probability and do not impose a hard count maximum.

SQL persists feature snapshots and separate forecast kinds. Source mismatches require restart/reload, while conflicting corrected historical records require an explicit new data/version strategy. Today's history cutoff and a future scenario date remain distinct. Sparse histories, hypothetical scenarios, stale data, and limited cross-division evidence warrant caution. Injuries, training camps, and comprehensive non-UFC records are not available.

The [forward ledger](docs/FORWARD_GUIDE.md) accepts real verified future fixtures only before scheduled time, freezes issuance provenance, and later settles exact identities. V2 and V3 batches remain separate. No settled prospective performance is claimed. Synthetic thresholds are not historical market lines, and no betting ROI evaluation is supported.

After source changes, validate identities and corrections, rebuild lagged features, fit/calibrate with explicit cutoffs, and preserve prior artifacts/reports. Verified artifact reuse is different from refitting; independent refits belong in fresh versioned directories. A genuinely later prospective period remains the strongest next validation step.
