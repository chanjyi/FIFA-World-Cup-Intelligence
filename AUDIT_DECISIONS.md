# Audit decisions — user-raised concerns (v7.2)

| Concern | Verdict | Action |
|---|---|---|
| Pandas 3 / unpinned dependencies | **Valid bug / reproducibility risk** | Replace `np.issubdtype` with pandas dtype API; pin all direct dependencies and recommend Python 3.13. Fresh Streamlit installation still needs to be tested on your Windows PC. |
| Wrong 2026 champion / title globe | **Valid**, but different-champion count remains **8** | Read ET/penalty winner; sync 2026 tournament row from raw JSON; Spain's count becomes 2. |
| Weak majority-class baseline | **Valid** | Add higher-Elo rule and Elo-only logistic benchmark; headline 61/104 vs 61/104, paired McNemar p=1.0; report small uncertain log-loss benefit. |
| Impurity importance / experience proxy | **Valid risk**, not proven spurious | Held-out 2022 permutation importance; compare accuracy/log loss after dropping experience features. Keep fitted model unchanged. |
| Low draw recall | **Valid** | Call out actual 29, predicted 14, recalled 4; defer calibration or Poisson replacement pending true chronological comparison. |
| Over-reading stages / giant killers | **Valid** | Add stage Wilson error intervals and `n`, soften conclusions and identify small-count leaderboards. |
| Row-level independence tests | **Valid limitation** | Team-cluster multiplier-bootstrap sensitivity intervals for model-vs-Elo comparison, no formal tournament-level significance with only one holdout. Retain old row-level permutation outputs only as historic exploratory reports, not inferential evidence. |
| Three practical findings/actions | **Useful and implemented** | Executive Summary at top of Insights. |
| Monte Carlo title odds | **Interesting but unjustified with current timing** | Not added; would need explicit bracket, dated feature snapshot, ET/shootout mechanism. |
| Squad, stadium, hosts | **Useful descriptively** | Read 48 2026 squad records, 16 venues and 15 host-national-team matches; not used in trained model. |
| Brier/calibration uncertainty/ablation intervals | **Useful and implemented** | Multiclass Brier, class-specific calibration Wilson intervals and `n`, 2022 ablation counts and approximate intervals. |
| End-of-2026 ratings | **Valid forecasting caveat** | Label hypothetical scenario and date limitation; not disguised as pre-2026 prediction. |
| Knockout ET/pens option | **Possible only as a transparent what-if** | Optional post-draw advancement-allocation assumption; not a learned advancement probability. |
| Long Insights | **Valid UI concern** | Advanced analysis sections collapsed inside expanders, Method & Limits expander at end. |
| First-listed labels | **Awkward but structurally true** | Rename to “First-listed team”/“Second-listed team”, use selected country names in Predictor. |

**Note on uncertainty:** A tournament-level significance test for 2026 generalization cannot be estimated from only one held-out tournament. The team-cluster sensitivity bootstrap is not a substitute for independent tournament replication.

## v7.3 follow-up
- Confirmed `streamlit==1.50.0` does not support a named `width` parameter for st.plotly_chart, causing warnings. Pinned `streamlit==1.66.1` and removed all `use_container_width` usage.
- The Pandas `dark_table` fix was **already present** in v7.2; the actual pinned Pandas version was 2.2.3.
- Removed stale `model_test_results.csv` (0.955) and `calibration.csv`, replaced by audited `evaluation_audit.json` and `calibration_with_intervals.csv`.
- Consolidated checks into `verify_project.py`: static/dependency checks, source-data and model checks, and real Streamlit AppTest (when installed).
- Licence on code is MIT. Exact Kaggle dataset redistribution permission was unverified; public artifact excludes Kaggle raw and derived CSVs, recreating summaries from OpenFootball.
