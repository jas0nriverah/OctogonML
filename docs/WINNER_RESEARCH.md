# Winner research: V3 and V4

Both studies failed their predeclared promotion gates. The default `CAGEMETRICS_WINNER_CHANNEL=auto` therefore retains the promoted 99-feature Winner V2 artifact (`winner-v2-d04e8f09a8abcbb4`). V3 and V4 are reproducible research channels available through explicit `v3` or `v4` configuration; neither is an approved replacement. These studies establish no validated accuracy gain.

## Study design

Each study declared five fixed candidates before fitting. Selection used pooled log loss on 1,497 decisive bouts scored in 2021–2023. For each scoring year, training ended before July 1 of the preceding year, followed by July–December calibration. Candidate rows had to agree exactly on bout, event, date, athletes, outcome and strictly prior history counts. Features exclude outcomes on or after the forecast event day.

Promotion required all of the following:

- A pooled log-loss improvement of at least 0.002 over V2, with no Brier-score regression.
- Expected calibration error no more than V2 plus 0.015.
- Improvement in at least two scoring years, with no annual log-loss regression above 0.005.
- A paired event-bootstrap 95% upper confidence bound below zero for selected-minus-V2 log loss: 2,000 resamples across 128 events.
- Nonregression in the sparse-versus-veteran subgroup: one athlete with at most three covered prior bouts and the other with at least eight. With at least 30 examples, permitted regressions were at most 0.005 log loss and 0.003 Brier score.

The 2024 comparison was computed only after selection was locked. It is an already-seen retrospective check, not an independent holdout or a second selection opportunity. Deployment artifacts use training before July 1, 2026 and calibration before October 18, 2026. No gates were relaxed after inspecting results or individual matchups.

## Development results

Lower log loss and Brier score are better. Values below describe the selected candidate in each study, not every candidate.

| Model | Selected candidate | Log loss | Brier score | ECE | Promotion |
|---|---|---:|---:|---:|---|
| V2 reference | `baseline_v2` | 0.653719 | 0.230933 | 0.044492 | Retained |
| V3 | `weekly_blend_50` | 0.652766 | 0.230531 | 0.042101 | Failed |
| V4 | `curated_blend50` | 0.653430 | 0.230947 | 0.043480 | Failed |

V3 added exposure reliability, opponent-adjusted striking, dated weekly ranks and ranked-opponent history. Its selected model blends V2 and the weekly-rank classifier equally. Its 0.000952 log-loss improvement fell short of 0.002; the paired difference interval was **[−0.003436, +0.001560]**, crossing zero. Both required gates failed.

V4 used a curated 99-feature schema that removes raw striking/count/accuracy/duration differences and their duplicates while retaining reliability, opposition, ranking, biography, grappling and rating information. Its selected model equally blends V2 and a constrained curated classifier. Its 0.000289 log-loss improvement also fell short; Brier score regressed slightly, and the paired difference interval **[−0.004009, +0.003473]** crossed zero. Three required gates failed.

The sparse-versus-veteran subgroup contained 187 bouts. Its log loss was 0.660025 for V2, 0.657152 for V3 and 0.656271 for V4; both subgroup nonregression gates passed. This local improvement does not override failed overall gates. The both-novice diagnostic (351 bouts) worsened from V2 log loss 0.654915 to V3 0.656077 and V4 0.659815. V4's additional any-sparse diagnostic (835 bouts) worsened from 0.662028 to 0.663373. Covered UFC history is not a fighter's complete professional record.

## Retrospective 2024 check

| Model | Bouts | Log loss | Brier score | Accuracy |
|---|---:|---:|---:|---:|
| V2 | 507 | 0.642838 | 0.225529 | 63.91% |
| V3 selected | 507 | 0.638899 | 0.223802 | 63.91% |
| V4 selected | 507 | 0.635376 | 0.222289 | 64.10% |

These retrospective figures do not authorize promotion. Choosing V4 because its 2024 result looks favorable would reuse already-seen outcomes for selection.

## Ranking provenance and availability

The historical source is the pinned [martj42 UFC Rankings dataset, version 47](https://www.kaggle.com/datasets/martj42/ufc-rankings/versions/47), licensed CC0. Its author describes recovery of official UFC media-poll boards through Internet Archive, with gaps. The retained source has 100,065 rows across 532 board dates from February 4, 2013 through June 2, 2026. It supplies 6,643 division/date boards; 4,350 pass the conservative completeness rule. Exact canonical identities and documented source-anchored aliases resolve 99,900 rows. Twelve contradictory same-athlete slots remain unsuitable for completeness claims.

A dated lookup requires a board strictly earlier than the bout day and no more than 35 days old. Absence means explicitly unranked only on a fully resolved, complete relevant division board; partial boards and stale or missing records remain unknown. Historical opponent quality uses the opponent's ranking before that historical bout, rather than today's rank. Unknown ranks do not become invented unranked observations. Ratios and opposition summaries include documented shrinkage and must not be presented as raw observed rates.

The separate [official UFC media ranking snapshot](https://www.ufc.com/rankings) contains 206 exact entries across 13 boards and was observed at **2026-10-22T20:17:16.420435+00:00**. Its footer says “Last updated: Tuesday, Oct. 20” without a year. The inferred October 20, 2026 date is recorded explicitly as an inference; `published_board_date` remains null. Live availability begins at the actual observation timestamp. This snapshot never backfills historical training examples, and the separate algorithmic Meta rankings table is excluded.

The source archive was retrieved retrospectively: temporal ordering of historical board dates does not establish that this application had collected those boards in real time. Current live observations and retrospectively recovered dated records retain distinct provenance.

## Reproducibility and audit records

V3 artifact version: `winner-v3-0913d6afdfb128e4`. V4 artifact version: `winner-v4-ce245eac7d9bf525`. Both freeze the same V2 baseline, outcome labels and ranking-context artifact. Engine loading validates implementation, artifact and source hashes. Post-selection descriptive probes checked exact feature schemas, finite inputs and complementary probabilities under fighter-order reversal; these examples were not used to select candidates or alter promotion decisions.

- V3: [protocol](../reports/winner_v3_protocol.json), [selection and gates](../reports/winner_v3_selection.json), [evaluation](../reports/winner_v3_evaluation.json), [row-level predictions](../reports/winner_v3_predictions.csv), [artifact metadata](../artifacts/winner_v3/metadata.json).
- V4: [protocol](../reports/winner_v4_protocol.json), [selection and gates](../reports/winner_v4_selection.json), [evaluation](../reports/winner_v4_evaluation.json), [row-level predictions](../reports/winner_v4_predictions.csv), [artifact metadata](../artifacts/winner_v4/metadata.json).
- Rankings: [historical attribution](../data/rankings/history/ATTRIBUTION.md), [historical provenance](../data/rankings/history/provenance.json), [current media snapshot](../data/rankings/current_media.json).

Historical CSV SHA-256: `a31032fc11db4ec2ef3b43f940937118f07dbd0b3bd622d27bf22f672fad2437`. Current media HTML SHA-256: `2990cc2e5fe968dcfd9076f7a578c01198850e32821b4c6345740bb3b7fd08e0`.
