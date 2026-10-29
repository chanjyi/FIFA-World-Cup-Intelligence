"""Independently reconstruct tournament summaries and 90-minute team tables from OpenFootball.

No Kaggle raw or derived CSV is redistributed. A score with ET reflects regulation
and extra-time scores; penalty shootout goals are not included in tournament totals.
"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from .worldcup_data import canonical_team

HOSTS = {
    1930: 'Uruguay', 1934: 'Italy', 1938: 'France', 1950: 'Brazil',
    1954: 'Switzerland', 1958: 'Sweden', 1962: 'Chile', 1966: 'England',
    1970: 'Mexico', 1974: 'West Germany', 1978: 'Argentina', 1982: 'Spain',
    1986: 'Mexico', 1990: 'Italy', 1994: 'United States', 1998: 'France',
    2002: 'South Korea, Japan', 2006: 'Germany', 2010: 'South Africa',
    2014: 'Brazil', 2018: 'Russia', 2022: 'Qatar',
    2026: 'Canada, Mexico, United States',
}

def score_pair(record):
    s=record.get('score')
    if isinstance(s, list): return s
    if isinstance(s, dict):
        for key in ('et','ft'):
            if isinstance(s.get(key), list): return s[key]
    return None

def decided_winner(record):
    s=record.get('score')
    if isinstance(s,list):
        pairs=[s]
    elif isinstance(s,dict):
        pairs=[s.get(x) for x in ('p','pen','pens','penalties','et','ft')]
    else: pairs=[]
    for pair in pairs:
        if isinstance(pair,list) and len(pair)==2 and pair[0] != pair[1]:
            return record['team1'] if pair[0]>pair[1] else record['team2']
    return None

def rebuild_tournament_summary(folder: Path):
    rows=[]
    for file in sorted(folder.glob('????/worldcup.json')):
        year=int(file.parent.name)
        games=json.loads(file.read_text(encoding='utf-8'))['matches']
        if not games:raise ValueError(f'No matches in {year}')
        if year==1950:
            # The 1950 title was decided by a final group, not a standalone final.
            deciding=[m for m in games if m['team1']=='Uruguay' and m['team2']=='Brazil']
            if len(deciding)!=1 or decided_winner(deciding[0])!='Uruguay':
                raise ValueError('1950 final-group decision cannot be verified')
            champion,runner='Uruguay','Brazil'
        else:
            finals=[g for g in games if str(g.get('round','')).strip().lower()=='final']
            if len(finals)!=1:raise ValueError(f'Expected exactly one {year} final, got {len(finals)}')
            champion=decided_winner(finals[0])
            if champion is None:raise ValueError(f'Unresolved {year} champion')
            runner=finals[0]['team2'] if champion==finals[0]['team1'] else finals[0]['team1']
        third_games=[g for g in games if 'third place' in str(g.get('round','')).lower()]
        third=decided_winner(third_games[-1]) if third_games else None
        final_goals=[score_pair(g) for g in games]
        # 1938 includes Austria's withdrawn fixture with Sweden (no played match).
        # Exclude unplayed/scoreless records rather than inventing a score.
        scored=[x for x in final_goals if isinstance(x,list) and len(x)==2]
        if not scored: raise ValueError(f'No scored matches in {year}')
        goals=sum(sum(x) for x in scored)
        played=[g for g in games if score_pair(g) is not None]
        teams={canonical_team(g[k]) for g in played for k in ('team1','team2')}
        rows.append({'year':year, 'host':HOSTS.get(year), 'champion':canonical_team(champion),
            'runner_up':canonical_team(runner), 'third_place':canonical_team(third) if third else None,
            'teams':len(teams),'matches_played':len(scored),'goals_scored':goals,
            'avg_goals_per_game':round(goals/len(scored), 3)})
    return pd.DataFrame(rows).sort_values('year').reset_index(drop=True)

def derive_team_year_results(matches):
    """90-minute observed results; NOT FIFA official group/knockout standings."""
    usable=matches[matches['ml_eligible'].astype(bool)].copy()
    first=usable[['year','team_1','goals_1','goals_2']].rename(columns={
        'team_1':'team','goals_1':'goals_for','goals_2':'goals_against'})
    second=usable[['year','team_2','goals_2','goals_1']].rename(columns={
        'team_2':'team','goals_2':'goals_for','goals_1':'goals_against'})
    rows=pd.concat([first,second],ignore_index=True)
    rows['wins_90']=(rows.goals_for>rows.goals_against).astype(int)
    rows['draws_90']=(rows.goals_for==rows.goals_against).astype(int)
    rows['losses_90']=(rows.goals_for<rows.goals_against).astype(int)
    result=rows.groupby(['year','team'], as_index=False).agg(
        games_observed=('wins_90','size'),wins_90=('wins_90','sum'),
        draws_90=('draws_90','sum'),losses_90=('losses_90','sum'),
        goals_for_90=('goals_for','sum'),goals_against_90=('goals_against','sum'))
    result['goal_difference_90']=result.goals_for_90-result.goals_against_90
    result['points_3_1_0_90']=3*result.wins_90+result.draws_90
    return result.sort_values(['year','team']).reset_index(drop=True)
