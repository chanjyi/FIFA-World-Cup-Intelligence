"""Train + evaluate the match-outcome model (replaces notebooks 03-04 logic).
Run from the project root AFTER notebook 01:  python train_model.py
"""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.elo import ELO_FEATURES, add_elo, canonical, team_state
from src.worldcup_data import FEATURE_COLUMNS, create_prematch_features

ROOT = Path(__file__).resolve().parent
PROC, REP, MOD = ROOT / 'data' / 'processed', ROOT / 'data' / 'reports', ROOT / 'models'
for p in (PROC, REP, MOD):
    p.mkdir(parents=True, exist_ok=True)
ALL = FEATURE_COLUMNS + ELO_FEATURES


def pipe(model, scale=False):
    steps = [('impute', SimpleImputer(strategy='median'))]
    if scale:
        steps.append(('scale', StandardScaler()))
    return Pipeline(steps + [('model', model)])


def rf():
    return RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=5, random_state=42)


CANDIDATES = {
    'Dummy prior': (pipe(DummyClassifier(strategy='prior')), ALL),
    'Logistic (no Elo)': (pipe(LogisticRegression(max_iter=2000), True), FEATURE_COLUMNS),
    'Random Forest (no Elo)': (pipe(rf()), FEATURE_COLUMNS),
    'Logistic + Elo': (pipe(LogisticRegression(max_iter=2000), True), ALL),
    'Random Forest + Elo': (pipe(rf()), ALL),
}

# ---------- 1. Features (leakage-safe) ----------
matches = pd.read_csv(PROC / 'matches_clean.csv', parse_dates=['date'])
matches['ml_eligible'] = matches['ml_eligible'].astype(str).str.lower().eq('true')
feats, elo_hist = add_elo(create_prematch_features(matches))
df = feats[(feats['year'] >= 2002) & feats['ml_eligible']].dropna(subset=['result_90']).copy()
df['year'] = df['year'].astype(int)
df.to_csv(PROC / 'prematch_features.csv', index=False, date_format='%Y-%m-%d')
elo_hist.to_csv(PROC / 'elo_history.csv', index=False, date_format='%Y-%m-%d')
state = team_state(matches)
last_elo = elo_hist.sort_values('date').groupby('team')['elo'].last()
state['elo'] = [float(last_elo.get(canonical(t), 1500.0)) for t in state.index]
state.to_csv(PROC / 'team_state.csv')

# ---------- 2. Walk-forward validation (train on all earlier World Cups) ----------
rows = []
for name, (est, cols) in CANDIDATES.items():
    for y in (2014, 2018, 2022):
        tr, te = df[df['year'] < y], df[df['year'] == y]
        m = clone(est).fit(tr[cols], tr['result_90'])
        rows.append({'model': name, 'year': y,
                     'accuracy': accuracy_score(te['result_90'], m.predict(te[cols])),
                     'log_loss': log_loss(te['result_90'], m.predict_proba(te[cols]), labels=m.classes_)})
cv = pd.DataFrame(rows)
summary = cv.groupby('model', as_index=False)[['accuracy', 'log_loss']].mean().sort_values('log_loss')
summary.to_csv(REP / 'model_validation_results.csv', index=False)
best = summary[summary['model'] != 'Dummy prior'].iloc[0]['model']  # chosen by log loss, never by the 2026 test
est, cols = CANDIDATES[best]

# ---------- 3. Final fit through 2022, test on 2026 ONCE ----------
train, test = df[df['year'] <= 2022], df[df['year'] == 2026]
model = clone(est).fit(train[cols], train['result_90'])
pred, proba = model.predict(test[cols]), model.predict_proba(test[cols])
y = test['result_90'].values
correct = pred == y
boot = [np.random.default_rng(i).choice(correct, len(correct)).mean() for i in range(2000)]
majority = train['result_90'].mode()[0]
bt = test[['date', 'year', 'team_1', 'team_2', 'goals_1', 'goals_2', 'result_90', 'elo_diff']].copy()
bt['predicted_result'] = pred
bt['stage'] = np.where(test['round'].str.lower().str.startswith('matchday'), 'Group stage', 'Knockout')
for j, c in enumerate(model.classes_):
    bt['prob_' + c] = proba[:, j]
bt['prob_actual'] = [proba[i, list(model.classes_).index(v)] for i, v in enumerate(y)]
bt.to_csv(MOD / 'worldcup_2026_backtest.csv', index=False, date_format='%Y-%m-%d')

# Permutation / calibration reports are generated on the validated audit path.
# Run `python audit_model.py` after training; do not regenerate legacy
# `calibration.csv` (pooled over classes without uncertainty intervals).
last = model.named_steps['model']
imp = last.feature_importances_ if hasattr(last, 'feature_importances_') else np.abs(last.coef_).mean(axis=0)

metrics = {
    'selected_model': best, 'test_matches': int(len(test)), 'accuracy': float(correct.mean()),
    'ci_low': float(np.percentile(boot, 2.5)), 'ci_high': float(np.percentile(boot, 97.5)),
    'baseline_accuracy': float((y == majority).mean()), 'baseline_label': majority,
    'log_loss': float(log_loss(y, proba, labels=model.classes_)),
    'draw_recall': float(((pred == 'draw') & (y == 'draw')).sum() / max((y == 'draw').sum(), 1)),
    'stage_accuracy': {s: float(correct[(bt['stage'] == s).values].mean()) for s in bt['stage'].unique()},
    'walkforward': summary.round(4).to_dict('records'),
    'importance': dict(sorted(zip(cols, map(float, imp)), key=lambda kv: -kv[1])),
}
(REP / 'model_metrics.json').write_text(json.dumps(metrics, indent=2))
joblib.dump({'model': model, 'features': cols, 'classes': list(model.classes_), 'trained_through': 2022,
             'selected': best}, MOD / 'match_outcome_model.joblib')

print(summary.round(3).to_string(index=False))
print(f"\nSelected: {best}\n2026 accuracy {metrics['accuracy']:.1%} "
      f"(95% CI {metrics['ci_low']:.1%}-{metrics['ci_high']:.1%}) vs simple first-listed baseline {metrics['baseline_accuracy']:.1%} (not a fair strength baseline; run audit_model.py)")
print('By stage:', {k: f'{v:.1%}' for k, v in metrics['stage_accuracy'].items()}, '| draw recall', f"{metrics['draw_recall']:.0%}")
