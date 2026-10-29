"""Refresh 2026 tournament summary from detailed match source; never modify raw JSON.
Run after notebook 01 or raw data updates: python sync_champions.py
"""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'data/raw/openfootball/2026/worldcup.json'
TABLE=ROOT/'data/processed/tournament_summary_clean.csv'

def match_scores(match):
    score=match.get('score') or {}
    for key in ('et','ft'):
        pair=score.get(key)
        if isinstance(pair,list) and len(pair)==2:
            return pair,key
    raise ValueError('No usable match score: '+str(match.get('date')))

def decide(match):
    score=match.get('score') or {}
    for key in ('p','pen','pens','penalties','et','ft'):
        pair=score.get(key)
        if isinstance(pair,list) and len(pair)==2 and pair[0]!=pair[1]:
            return match['team1'] if pair[0]>pair[1] else match['team2'],key
    return None,None

def main():
    games=json.loads(SOURCE.read_text(encoding='utf8'))['matches']
    finals=[x for x in games if x.get('round','').strip().lower()=='final']
    if len(finals)!=1: raise ValueError(f'2026 final ambiguity: {len(finals)}')
    final=finals[0];winner,method=decide(final)
    if not winner: raise ValueError('Cannot resolve winner from ET or penalties; refusing to add champion')
    loser=final['team2'] if winner==final['team1'] else final['team1']
    third_matches=[x for x in games if 'third place' in x.get('round','').lower()]
    third=decide(third_matches[0])[0] if len(third_matches)==1 else None
    scores=[match_scores(x)[0] for x in games]
    goals=int(sum(sum(s) for s in scores))
    names=set(x['team1'] for x in games)|set(x['team2'] for x in games)
    summary=pd.read_csv(TABLE)
    row={'year':2026,'host':'Canada, Mexico, United States','champion':winner,'runner_up':loser,
         'third_place':third,'teams':len(names),'matches_played':len(games),
         'goals_scored':goals,'avg_goals_per_game':round(goals/len(games),2)}
    summary=summary[summary['year']!=2026]
    summary=pd.concat([summary,pd.DataFrame([row])],ignore_index=True).sort_values('year')
    summary.to_csv(TABLE,index=False)
    print('Updated 2026 champion:',winner,'method:',method,'matches:',len(games),'goals:',goals)
if __name__=='__main__':main()
