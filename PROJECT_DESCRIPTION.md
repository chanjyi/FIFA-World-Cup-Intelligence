# FIFA World Cup Intelligence — Project Description

**Hackathon:** AQX Sports Analytics Data Bowl 4.0  
**Project type:** Interactive sports-analytics dashboard and explainable match-outcome modeling prototype  
**Technology:** Streamlit · Python · Pandas · Scikit-learn · Plotly · Three.js

## The problem

World Cup records are scattered across tournaments, while conventional match predictions often return a single answer without showing historical context, uncertainty or failure cases. Analysts and football fans need an accessible way to explore team performance, compare opponents and understand when a statistical estimate can—and cannot—be trusted.

## Our solution

FIFA World Cup Intelligence brings men's World Cup match history (1930–2026) into an interactive scouting dashboard. It transforms openly licensed match data into visual comparisons, team-level trends and a retrospective machine-learning evaluation. Instead of presenting a black-box “winner forecast,” it combines a transparent World Cup-only Elo benchmark with a Random Forest model for 90-minute win/draw/loss probabilities, and exposes the model's limitations.

## Core features

- Overview: Interactive 3D globe, tournament history, champion distribution and scoring trends.
- Team Lab: Side-by-side national-team comparison, normalized performance radar, Elo histories and illustrative historical Starting XI visualizations.
- Match Predictor: Explore 90-minute result probability scenarios for two selected teams, using end-of-2026 team state; not a live or pre-2026 forecast.
- Model Report: Compare model results with higher-Elo and Elo-only baselines, inspect calibration, feature importance, errors and uncertainty intervals.
- Insights: Three actionable findings, upset patterns, team archetypes and defensive trends.

## Evidence and actionable impact

On the 104-match 2026 retrospective holdout, the selected Random Forest classified 61/104 matches correctly (58.7%), the same accuracy as selecting the higher-Elo team. Its multiclass log loss was 0.928 against 0.951 for the Elo-only logistic model, but the advantage is uncertain. Draws remained difficult: 4 of 29 actual draws were correctly classified (13.8% recall).

This translates into three practical analyst actions: (1) use Elo as a simple, interpretable first-pass scouting benchmark; (2) prepare multiple scenarios for closely matched teams rather than relying on a single draw/win label; and (3) compare tournament scoring patterns using per-match rates and stage context before making claims about format changes. The intended impact is more transparent, responsible match preparation and sports-data exploration, not demonstrated improvements in real-world coaching outcomes.

## Data and limitations

The public release uses OpenFootball's CC0 World Cup JSON and derived data. An earlier Kaggle dataset is credited but not redistributed because its license is unclear. Historical match features are tested retrospectively; selected-team scenarios use **end-of-2026** values and do **not** predict pre-tournament outcomes, live matches or betting results. See [`DATA_LICENSE_NOTICE.md`](DATA_LICENSE_NOTICE.md) and [`README.md`](README.md).
