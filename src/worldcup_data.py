"""Parsers and leakage-aware feature engineering for OpenFootball + Kaggle WC datasets."""
from pathlib import Path
import json
import re
from collections import defaultdict
import numpy as np
import pandas as pd

# Historical political entities are intentionally NOT collapsed into modern countries.
TEAM_ALIASES = {
    'USA': 'United States',
    'United States of America': 'United States',
    'Korea Republic': 'South Korea',
    'Korea, Republic of': 'South Korea',
    'IR Iran': 'Iran',
    'Islamic Republic of Iran': 'Iran',
}

FEATURE_COLUMNS = [
    'team_1_prev_matches', 'team_2_prev_matches',
    'team_1_win_rate_last5', 'team_2_win_rate_last5',
    'team_1_goals_for_last5', 'team_2_goals_for_last5',
    'team_1_goals_against_last5', 'team_2_goals_against_last5',
    'win_rate_diff', 'goals_for_diff', 'goals_against_diff',
    'head_to_head_matches', 'head_to_head_team1_win_rate',
]


def canonical_team(value):
    if pd.isna(value):
        return pd.NA
    s = re.sub(r'\s+', ' ', str(value)).strip()
    return TEAM_ALIASES.get(s, s)


def parse_score(score):
    """Return two scores, provenance, and ET/penalty flags.
    In this source, score.ft means 90-minute regulation score, while
    a flat score is used for many normal matches, particularly group matches.
    Flat scores in knockout matches require review before labeling.
    """
    is_et = isinstance(score, dict) and 'et' in score
    is_penalties = isinstance(score, dict) and 'p' in score
    if isinstance(score, dict):
        ft = score.get('ft')
        basis = 'ft_object'
    elif isinstance(score, list):
        ft = score
        basis = 'flat_score'
    else:
        ft = None
        basis = 'missing'
    if not (isinstance(ft, list) and len(ft) == 2):
        return np.nan, np.nan, 'missing', bool(is_et), bool(is_penalties)
    try:
        return int(ft[0]), int(ft[1]), basis, bool(is_et), bool(is_penalties)
    except (TypeError, ValueError):
        return np.nan, np.nan, 'missing', bool(is_et), bool(is_penalties)


def load_openfootball(folder: Path):
    rows=[]
    files=sorted(folder.glob('????/worldcup.json'))
    if not files:
        raise FileNotFoundError(f'No yearly worldcup.json found in {folder}')
    for path in files:
        year=int(path.parent.name)
        with path.open('r',encoding='utf-8') as f:
            data=json.load(f)
        for i, m in enumerate(data.get('matches', [])):
            s1,s2,score_format,et,pen=parse_score(m.get('score'))
            round_text=str(m.get('round') or '')
            group_text=m.get('group')
            is_group = bool(group_text) or round_text.lower().startswith('matchday')
            ambiguous = (score_format=='flat_score' and not is_group)
            row={
                'year':year, 'date':m.get('date'),
                'team_1_raw':m.get('team1'), 'team_2_raw':m.get('team2'),
                'team_1':canonical_team(m.get('team1')),
                'team_2':canonical_team(m.get('team2')),
                'goals_1':s1,'goals_2':s2,
                'round':round_text,'group':group_text,
                'is_group':is_group, 'score_format':score_format,
                'ambiguous_flat_knockout':ambiguous,
                'extra_time':et, 'penalty_shootout':pen,
                'source_file':str(path.relative_to(folder)),
                'source_row':i,
            }
            rows.append(row)
    df=pd.DataFrame(rows)
    df['date']=pd.to_datetime(df['date'],errors='coerce')
    df['year']=pd.to_numeric(df['year'],errors='coerce').astype('Int64')
    for col in ['goals_1','goals_2']:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    # A stable per-match identifier independent of home/away team ordering.
    df['match_key']=df.apply(lambda r:
        f"{r['year']}|{r['date'].date() if pd.notna(r['date']) else 'NA'}|"
        + '|'.join(sorted((str(r['team_1']),str(r['team_2'])))),axis=1)
    valid=df['goals_1'].notna() & df['goals_2'].notna()
    df['result_90']=pd.NA
    df.loc[valid & (df['goals_1']>df['goals_2']),'result_90']='team_1_win'
    df.loc[valid & (df['goals_1']==df['goals_2']),'result_90']='draw'
    df.loc[valid & (df['goals_1']<df['goals_2']),'result_90']='team_2_win'
    # Only matches with an interpretable 90-minute score are usable for ML.
    df['ml_eligible']= valid & (~df['ambiguous_flat_knockout']) & df['date'].notna()
    return df.sort_values(['date','match_key']).reset_index(drop=True)


def load_kaggle_standings(folder: Path):
    dfs=[]
    for path in sorted(folder.glob('FIFA - [0-9][0-9][0-9][0-9].csv')):
        year=int(path.stem.split(' - ')[-1]); df=pd.read_csv(path,encoding='utf-8-sig')
        df=df.rename(columns={
            'Position':'position','Team':'team_raw','Games Played':'games_played',
            'Win':'wins','Draw':'draws','Loss':'losses',
            'Goals For':'goals_for','Goals Against':'goals_against',
            'Goal Difference':'goal_difference','Points':'points'
        })
        df['team']=df['team_raw'].map(canonical_team)
        df['year']=year
        for col in ['position','games_played','wins','draws','losses',
                    'goals_for','goals_against','goal_difference','points']:
            df[col]=pd.to_numeric(
                df[col].astype(str).str.replace('−','-',regex=False).str.replace('–','-',regex=False),
                errors='coerce')
        df['valid_game_count']=(df['wins']+df['draws']+df['losses']==df['games_played'])
        df['valid_goal_difference']=(df['goals_for']-df['goals_against']==df['goal_difference'])
        df['source_file']=path.name
        dfs.append(df)
    if not dfs: raise FileNotFoundError(f'No Kaggle yearly CSVs in {folder}')
    return pd.concat(dfs,ignore_index=True).sort_values(['year','position'])


def load_kaggle_summary(folder: Path):
    p=folder/'FIFA - World Cup Summary.csv'
    df=pd.read_csv(p,encoding='utf-8-sig')
    df.columns=df.columns.str.strip().str.lower().str.replace(' ','_',regex=False)
    df['year']=pd.to_numeric(df['year'],errors='coerce').astype('Int64')
    return df


def create_prematch_features(matches):
    """Pre-match World Cup form, strictly from EARLIER DATES (no match-outcome leakage).

    World Cup games only, NOT all international fixtures or current FIFA rankings.
    For ML, initial years before 2002 are useful for calculating prior form.
    """
    matches=matches.sort_values(['date','match_key']).copy()
    history=defaultdict(list)
    h2h=defaultdict(list)
    rows=[]
    def form(team):
        recent=history[team][-5:]
        if not recent: return (0,np.nan,np.nan,np.nan)
        return (len(history[team]),float(np.mean([r[0] for r in recent])),
                float(np.mean([r[1] for r in recent])),float(np.mean([r[2] for r in recent])))
    for date,day in matches.groupby('date',sort=True,dropna=True):
        for _,m in day.iterrows():
            a,b=m['team_1'],m['team_2']
            if pd.isna(a) or pd.isna(b): continue
            n1,w1,gf1,ga1=form(a); n2,w2,gf2,ga2=form(b)
            past_h2h=h2h[frozenset((a,b))]
            h2h_w=np.mean([v == a for v in past_h2h]) if past_h2h else np.nan
            rows.append({
                'match_key':m['match_key'], 'year':m['year'],'date':date,
                'team_1':a,'team_2':b, 'round':m['round'],
                'goals_1':m['goals_1'],'goals_2':m['goals_2'],
                'result_90':m['result_90'],'ml_eligible':m['ml_eligible'],
                'team_1_prev_matches':n1,'team_2_prev_matches':n2,
                'team_1_win_rate_last5':w1,'team_2_win_rate_last5':w2,
                'team_1_goals_for_last5':gf1,'team_2_goals_for_last5':gf2,
                'team_1_goals_against_last5':ga1,'team_2_goals_against_last5':ga2,
                'win_rate_diff':w1-w2, 'goals_for_diff':gf1-gf2,
                'goals_against_diff':ga1-ga2,
                'head_to_head_matches':len(past_h2h),
                'head_to_head_team1_win_rate':h2h_w,
            })
        # Batch updates only AFTER creating features for ALL games on the date.
        for _,m in day.iterrows():
            if not bool(m['ml_eligible']): continue
            a,b=m['team_1'],m['team_2']; g1,g2=m['goals_1'],m['goals_2']
            history[a].append((float(g1>g2),int(g1),int(g2)))
            history[b].append((float(g2>g1),int(g2),int(g1)))
            winner=a if g1>g2 else b if g2>g1 else None
            h2h[frozenset((a,b))].append(winner)
    out=pd.DataFrame(rows)
    if out.empty: raise ValueError('No matches with valid dates for feature engineering')
    return out.sort_values(['date','match_key']).reset_index(drop=True)
