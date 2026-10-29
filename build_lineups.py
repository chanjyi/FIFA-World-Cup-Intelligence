"""Extract real World Cup Starting XI, listed bench players and substitutions.

Run from the repository root: python build_lineups.py
No player positions or tactical formations are claimed: OpenFootball source does not
consistently include those attributes.
"""
from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw' / 'openfootball'
OUT = ROOT / 'data' / 'processed'


def extract_lineups() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    match_rows, players, subs = [], [], []
    for file in sorted(RAW.glob('*/worldcup-full.json')):
        year = int(file.parent.name)
        obj = json.loads(file.read_text(encoding='utf-8'))
        for idx, match in enumerate(obj.get('matches', [])):
            blocks = match.get('lineup')
            if not isinstance(blocks, list) or len(blocks) < 2:
                continue
            teams = [match.get('team1'), match.get('team2')]
            key = f"{year}|{idx:03d}|{match.get('date','')}|{teams[0]}|{teams[1]}"
            match_rows.append({
                'lineup_key': key, 'year': year, 'date': match.get('date'),
                'team_1': teams[0], 'team_2': teams[1], 'round': match.get('round', ''),
                'source_file': str(file.relative_to(ROOT)).replace('\\','/'),
            })
            for side, block in enumerate(blocks):
                if not isinstance(block, dict):
                    continue
                for role, source_field in [('starter', 'starter'), ('bench', 'bench')]:
                    for n, player in enumerate(block.get(source_field, []), start=1):
                        if isinstance(player, str):
                            player = {'name': player}
                        if not isinstance(player, dict):
                            continue
                        players.append({
                            'lineup_key': key, 'year': year, 'date': match.get('date'),
                            'team': teams[side], 'side': side+1, 'role': role,
                            'order': n, 'player': player.get('name','Unknown'),
                            'captain': bool(player.get('captain', False)),
                        })
                for n, item in enumerate(block.get('subs', []), start=1):
                    if isinstance(item, dict):
                        subs.append({'lineup_key': key, 'team': teams[side], 'order': n,
                                     'player_on': item.get('on',''), 'player_off':item.get('off',''),
                                     'minute': item.get('minute','')})
    return pd.DataFrame(match_rows), pd.DataFrame(players), pd.DataFrame(subs)


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    matches, players, subs = extract_lineups()
    matches.to_csv(OUT / 'lineup_matches.csv', index=False, encoding='utf-8-sig')
    players.to_csv(OUT / 'lineup_players.csv', index=False, encoding='utf-8-sig')
    subs.to_csv(OUT / 'lineup_substitutions.csv', index=False, encoding='utf-8-sig')
    print(f'Lineup matches: {len(matches):,} | player records: {len(players):,} | substitutions: {len(subs):,}')
