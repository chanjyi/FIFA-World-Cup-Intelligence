"""Rebuild the descriptive clean-sheet and blowout counts used in Insights.

Run: python build_defensive_trends.py
This report is based on eligible 90-minute match outcomes, not shootout wins.
"""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'/'processed'/'matches_clean.csv'
OUT=ROOT/'data'/'reports'/'clean_sheet_blowout_trends.csv'


def calculate(df:pd.DataFrame)->pd.DataFrame:
    df=df.copy()
    eligible=df['ml_eligible'].astype(str).str.lower().eq('true')
    df=df[eligible & df['year'].between(2002,2026) & df['goals_1'].notna() & df['goals_2'].notna()].copy()
    df['clean_sheet_match']=df['goals_1'].eq(0)|df['goals_2'].eq(0)
    df['blowout_match']=df['goals_1'].sub(df['goals_2']).abs().ge(3)
    df['scoreless_draw']=df['goals_1'].eq(0)&df['goals_2'].eq(0)
    return df.groupby('year',as_index=False).agg(
        matches=('year','size'),
        clean_sheet_matches=('clean_sheet_match','sum'),
        blowout_matches=('blowout_match','sum'),
        scoreless_draws=('scoreless_draw','sum'),
        clean_sheet_rate=('clean_sheet_match','mean'),
        blowout_rate=('blowout_match','mean'),
        scoreless_draw_rate=('scoreless_draw','mean'),
    )


if __name__=='__main__':
    t=calculate(pd.read_csv(DATA))
    OUT.parent.mkdir(parents=True,exist_ok=True)
    t.to_csv(OUT,index=False)
    print(t.to_string(index=False))
    print('Saved:',OUT)
