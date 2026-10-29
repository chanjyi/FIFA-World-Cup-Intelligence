"""Elo ratings + latest team state. Ratings for a match use ONLY earlier dates (no leakage)."""
from collections import defaultdict
import numpy as np
import pandas as pd

ELO_FEATURES = ['elo_1', 'elo_2', 'elo_diff', 'elo_expected_1']

# Successor-state assumption so rating history continues (e.g. West Germany -> Germany).
# East Germany is kept separate. Document this in the README.
LINEAGE = {'West Germany': 'Germany', 'Soviet Union': 'Russia', 'Czechoslovakia': 'Czech Republic',
           'Yugoslavia': 'Serbia', 'Serbia and Montenegro': 'Serbia'}


def canonical(team):
    return LINEAGE.get(team, team)


def _expected(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))


def add_elo(features, k=40, start=1500):
    """Return (features + Elo columns, long Elo history). Venue/host advantage is not modelled."""
    df = features.sort_values(['date', 'match_key']).reset_index(drop=True)
    rating = defaultdict(lambda: float(start))
    pre, hist = [], []
    for date, day in df.groupby('date', sort=True):
        for idx, m in day.iterrows():  # 1) snapshot ratings BEFORE any same-day result
            r1, r2 = rating[canonical(m['team_1'])], rating[canonical(m['team_2'])]
            pre.append((idx, r1, r2, r1 - r2, _expected(r1, r2)))
        for _, m in day.iterrows():  # 2) update after the whole day is featurised
            if not bool(m['ml_eligible']):
                continue
            a, b = canonical(m['team_1']), canonical(m['team_2'])
            g1, g2 = m['goals_1'], m['goals_2']
            s1 = 1.0 if g1 > g2 else 0.5 if g1 == g2 else 0.0
            d = abs(g1 - g2)
            mult = 1.0 if d <= 1 else 1.5 if d == 2 else (11 + d) / 8
            delta = k * mult * (s1 - _expected(rating[a], rating[b]))
            rating[a] += delta
            rating[b] -= delta
            hist += [{'date': date, 'year': int(m['year']), 'team': a, 'elo': rating[a]},
                     {'date': date, 'year': int(m['year']), 'team': b, 'elo': rating[b]}]
    elo = pd.DataFrame(pre, columns=['idx'] + ELO_FEATURES).set_index('idx')
    return df.join(elo), pd.DataFrame(hist)


def team_state(matches):
    """Each team's latest last-5 World Cup form (used by the match predictor)."""
    m = matches[matches['ml_eligible'].astype(bool)].sort_values(['date', 'match_key'])
    log = defaultdict(list)
    for _, r in m.iterrows():
        log[r['team_1']].append((float(r['goals_1'] > r['goals_2']), r['goals_1'], r['goals_2']))
        log[r['team_2']].append((float(r['goals_2'] > r['goals_1']), r['goals_2'], r['goals_1']))
    rows = []
    for team, h in log.items():
        last = np.array(h[-5:])
        rows.append({'team': team, 'prev_matches': len(h), 'win_rate_last5': last[:, 0].mean(),
                     'goals_for_last5': last[:, 1].mean(), 'goals_against_last5': last[:, 2].mean()})
    return pd.DataFrame(rows).set_index('team')
