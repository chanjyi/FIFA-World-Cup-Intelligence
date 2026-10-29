"""FIFA World Cup Intelligence - globe-centred command dashboard.
Run: python -m pip install -r requirements.txt && python -m streamlit run app.py
"""
from __future__ import annotations

import json
from html import escape
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.elo import canonical
from src.globe_component import show_cinematic_globe
from src.lineup_pitch import pitch_figure
from src.three_views import show_pitch_3d

ROOT = Path(__file__).resolve().parent
PROC = ROOT / 'data' / 'processed'
REP = ROOT / 'data' / 'reports'
MOD = ROOT / 'models'

CYAN = '#18E0FF'
CYAN_SOFT = '#6EEBFF'
GOLD = '#FFC857'
PINK = '#FF5AA5'
GREEN = '#3DDC97'
TEXT = '#E8F4FF'
TEXT_MUTED = '#98B7D8'
PANEL = 'rgba(14, 29, 61, 0.88)'
BORDER = 'rgba(24, 224, 255, 0.34)'

PRETTY = {
    'team_1_win': 'First-listed team wins',
    'draw': 'Draw',
    'team_2_win': 'Second-listed team wins',
}

ISO = {
    'Brazil': 'BRA', 'Italy': 'ITA', 'West Germany': 'DEU', 'Germany': 'DEU', 'Argentina': 'ARG',
    'Uruguay': 'URY', 'France': 'FRA', 'England': 'GBR', 'Spain': 'ESP'
}

COUNTRY_ALIASES = {
    'Bosnia & Herzegovina': 'Bosnia and Herzegovina',
    'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
    "Côte d'Ivoire": 'Ivory Coast',
    'Ivory Coast': 'Ivory Coast',
    'DR Congo': 'Democratic Republic of the Congo',
    'Iran': 'Iran',
    'South Korea': 'South Korea',
    'North Korea': 'North Korea',
    'United States': 'United States',
    'Cape Verde': 'Cape Verde',
    'Curaçao': 'Curacao',
    'Serbia and Montenegro': 'Serbia',
    'West Germany': 'Germany',
    'England': 'United Kingdom',
    'Scotland': 'United Kingdom',
    'Wales': 'United Kingdom',
}

st.set_page_config(page_title='FIFA World Cup Intelligence', page_icon='⚽', layout='wide')

st.markdown(f"""
<style>
:root {{
    --bg-1:#040A18;
    --bg-2:#091C3F;
    --panel:{PANEL};
    --text:{TEXT};
    --muted:{TEXT_MUTED};
    --cyan:{CYAN};
    --cyan-soft:{CYAN_SOFT};
    --gold:{GOLD};
    --pink:{PINK};
    --green:{GREEN};
    --border:{BORDER};
}}

html, body, [data-testid="stAppViewContainer"], .stApp {{
    color: var(--text);
    background:
        radial-gradient(circle at 82% 16%, rgba(24,224,255,0.16) 0%, rgba(24,224,255,0.02) 18%, rgba(0,0,0,0) 34%),
        radial-gradient(circle at 18% -5%, rgba(32,118,255,0.22) 0%, rgba(32,118,255,0.04) 28%, rgba(0,0,0,0) 48%),
        linear-gradient(180deg, #07132d 0%, #030816 100%);
}}


/* The Streamlit deployment toolbar must not cover the banner at any browser zoom. */
[data-testid="stHeader"] {{ background: #07132d !important; color: #E8F4FF !important; height: 3.1rem !important; }}
[data-testid="stToolbar"] {{ background: transparent !important; color: #E8F4FF !important; }}
[data-testid="stHeader"] button {{ color:#E8F4FF !important; }}
.block-container {{padding-top: 4.3rem !important; padding-bottom: 2.5rem; max-width: 100rem;}}


h1, h2, h3, h4 {{color: var(--text) !important;}}
/* Chart containers now use true visible navy cards, not fake HTML wrappers. */
div[data-testid="stPlotlyChart"] {{
 background:linear-gradient(145deg,rgba(14,31,64,.89),rgba(7,18,42,.93)) !important;
 border:1px solid rgba(30,221,255,.24) !important;
 border-radius:18px !important;
 padding:.55rem .55rem .35rem .55rem !important;
 box-shadow:inset 0 0 22px rgba(24,224,255,.035),0 8px 28px rgba(0,0,0,.14);
}}
/* native Streamlit expandable panels and controls */
[data-testid="stExpander"], [data-testid="stExpander"] details,
[data-testid="stExpander"] summary,
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {{
  background: #0D2044 !important; color:#E8F4FF !important;
  border-color:rgba(24,224,255,.28)!important;
}}
[data-testid="stExpander"] summary:hover {{ background: #15315B !important; }}
[data-testid="stExpander"] summary *, [data-testid="stExpander"] svg {{color:#E8F4FF !important;fill:#E8F4FF !important;}}
[data-testid="stExpander"] details {{border-radius:12px;overflow:hidden;}}
[data-testid="stExpander"] div {{color:#E8F4FF;}}

p, li, .stMarkdown, label {{color: var(--text);}}
div[data-testid="stCaptionContainer"] p {{color: var(--muted) !important; font-size: 1.05rem;}}

.hero-wrap {{display:flex; align-items:flex-start; gap:1rem; margin-bottom: 0.8rem;}}
.hero-icon {{font-size: 2.25rem; line-height: 1; position:relative; top:.50rem; flex-shrink:0; filter: drop-shadow(0 0 12px rgba(24,224,255,.25));}}
.hero-title {{font-size: clamp(2.1rem, 3.6vw, 3.7rem); font-weight: 800; letter-spacing: .2px; line-height: 1.04;
    background: linear-gradient(90deg, #8AEFFF 0%, #17E0FF 36%, #47C7FF 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;}}
.hero-subtitle {{margin-top: .28rem; color: var(--muted); font-size: 1.02rem;}}
.hero-subtitle b {{color: var(--cyan-soft); font-weight: 600;}}

.section-chip {{display:inline-block; padding: .3rem .72rem; border: 1px solid var(--border); border-radius: 999px;
    font-size: .76rem; letter-spacing: .08em; text-transform: uppercase; color: var(--cyan-soft);
    background: rgba(24,224,255,.08); box-shadow: inset 0 0 15px rgba(24,224,255,.05); margin-bottom: .65rem;}}

.panel, .kpi {{
    background: linear-gradient(180deg, rgba(15,29,60,.96) 0%, rgba(10,21,47,.9) 100%);
    border: 1px solid var(--border);
    box-shadow: 0 0 0 1px rgba(255,255,255,.015), 0 10px 34px rgba(0,0,0,.22), inset 0 0 34px rgba(24,224,255,.04);
    border-radius: 18px;
}}
.panel {{padding: 1rem 1.05rem;}}
.kpi {{padding: 1.05rem 1.15rem; min-height: 122px;}}
.kpi-label {{font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .11em;}}
.kpi-value {{font-size: 2.25rem; font-weight: 800; color: var(--gold); margin-top: .45rem;}}
.kpi-help {{font-size: .8rem; color: var(--muted); margin-top: .25rem;}}
.panel-title {{font-size: 1.05rem; font-weight: 700; color: var(--text); margin: 0 0 .7rem;}}
.panel-subtitle {{font-size: .9rem; color: var(--muted); margin-top: -.1rem; margin-bottom: .65rem;}}
.panel-note {{border-left: 3px solid var(--gold); padding: .7rem .9rem; border-radius: 8px;
    background: rgba(255,200,87,.08); color: var(--text); margin-top: .6rem;}}
.small-note {{font-size: .84rem; color: var(--muted); line-height: 1.45;}}
.mini-list {{margin:0; padding-left:1rem;}}
.mini-list li {{margin: .35rem 0; color: var(--text);}}

[data-testid="stTabs"] [role="tablist"] {{gap: 1.1rem; border-bottom: 1px solid rgba(255,255,255,.06);}}
[data-testid="stTabs"] [role="tab"] {{
    color: var(--muted); background: transparent; padding: .6rem .1rem .8rem .1rem;
    border-bottom: 2px solid transparent; font-weight: 600;
}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{color: var(--text) !important; border-bottom-color: #FF6262;}}

label, .stSelectbox label, .stSlider label, .stRadio label {{color: var(--text) !important; font-weight: 600 !important;}}

/* Dark dropdowns — selectors cover both Streamlit's themed wrapper and BaseWeb internals.
   The .streamlit/config.toml dark theme is the primary fallback if DOM details change. */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stSelectbox"] div[role="combobox"],
.stSelectbox div[data-baseweb="select"] > div {{
    background: #0E1D3D !important;
    background-color: #0E1D3D !important;
    border: 1px solid rgba(24, 224, 255, .42) !important;
    border-radius: 12px !important;
    min-height: 46px;
    color: #E8F4FF !important;
    box-shadow: inset 0 0 24px rgba(24,224,255,.06), 0 0 0 1px rgba(255,255,255,.02) !important;
}}
[data-testid="stSelectbox"] div[data-baseweb="select"]:hover > div,
[data-testid="stSelectbox"] div[role="combobox"]:hover {{
    border-color: var(--cyan) !important;
}}
[data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within > div,
[data-testid="stSelectbox"] div[role="combobox"]:focus-visible {{
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 2px rgba(24,224,255,.18) !important;
}}
[data-testid="stSelectbox"] div[data-baseweb="select"] *,
[data-testid="stSelectbox"] div[role="combobox"] *,
[data-testid="stSelectbox"] div[data-baseweb="select"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] svg {{
    color: #E8F4FF !important;
    fill: #E8F4FF !important;
    -webkit-text-fill-color: #E8F4FF !important;
}}
[data-testid="stSelectbox"] input::placeholder {{color: #98B7D8 !important;}}
/* The option list is often portalled to document.body, outside stSelectbox. */
[data-baseweb="popover"] [role="listbox"],
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="popover"] [role="option"],
[data-baseweb="popover"] li {{
    background: #0C1B3C !important;
    color: #E8F4FF !important;
}}
[data-baseweb="popover"] [role="option"] *,
[data-baseweb="popover"] li * {{color: #E8F4FF !important;}}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [role="option"][aria-selected="true"],
[data-baseweb="popover"] li:hover {{
    background: #17436B !important;
    color: #6EEBFF !important;
}}

div[role="radiogroup"] {{gap: .5rem;}}
div[role="radiogroup"] > label {{
    background: rgba(15,29,60,.9); border: 1px solid var(--border); border-radius: 999px; padding: .45rem .85rem;
}}
div[role="radiogroup"] > label:has(input:checked) {{
    background: rgba(24,224,255,.14); box-shadow: 0 0 0 1px rgba(24,224,255,.2), inset 0 0 18px rgba(24,224,255,.08);
}}

[data-baseweb="slider"] > div > div {{background: rgba(255,255,255,.15) !important;}}
[data-baseweb="slider"] [role="slider"] {{background: #FF6262 !important; box-shadow: 0 0 0 4px rgba(255,98,98,.18);}}

.table-card {{
    background: linear-gradient(180deg, rgba(14,29,61,.96) 0%, rgba(9,18,38,.96) 100%);
    border: 1px solid rgba(24,224,255,.22); border-radius: 16px; overflow: hidden;
}}
.table-card table {{width:100%; border-collapse: collapse; font-size: .94rem;}}
.table-card thead th {{
    background: rgba(24,224,255,.1); color: var(--cyan-soft); text-transform: uppercase; letter-spacing: .08em;
    font-size: .76rem; padding: .78rem .85rem; text-align:left; border-bottom: 1px solid rgba(24,224,255,.18);
}}
.table-card tbody td {{padding: .76rem .85rem; border-bottom: 1px solid rgba(255,255,255,.06); color: var(--text);}}
.table-card tbody tr:nth-child(odd) td {{background: rgba(255,255,255,.015);}}
.table-card tbody tr:hover td {{background: rgba(24,224,255,.06);}}
.table-muted td:first-child {{color: var(--muted); font-weight: 600; width: 34%;}}

div[data-testid="stPlotlyChart"] > div {{background: transparent !important; border-radius: 18px;}}
[data-testid="stVerticalBlockBorderWrapper"]:has(.globe-panel-marker) {{
    background:linear-gradient(140deg,rgba(5,25,58,.92),rgba(3,11,30,.95));
    border:1px solid rgba(30,221,255,.28);
    box-shadow: 0 0 42px rgba(13,141,255,.13), inset 0 0 32px rgba(17,210,255,.05);
    border-radius:20px;
}}
[data-testid="stVerticalBlockBorderWrapper"] {{border-radius:17px;}}
@media(max-width:800px){{.kpi{{min-height:100px;padding:.75rem}}.kpi-value{{font-size:1.6rem}}.hero-subtitle{{font-size:.88rem}}}}


/* v7.1: use native Streamlit gaps; global gap overrides forced KPI columns to wrap. */
[data-testid="stPlotlyChart"] {{
  padding: .48rem !important;
  margin-block: .10rem .35rem !important;
  min-width:0;
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
  margin-block: .16rem .65rem;
  box-shadow: 0 8px 24px rgba(0,0,0,.16);
}}
/* Tighten chart cards nested inside native Streamlit bordered containers. */
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stPlotlyChart"] {{ border:0 !important; box-shadow:none !important; background:transparent !important; }}

/* Hero KPI grid does not share st.columns breakpoints or spacing with Plotly charts. */
.kpi-grid {{
  display:grid;
  grid-template-columns:repeat(5,minmax(0,1fr));
  gap:1.05rem;
  align-items:stretch;
  width:100%;
  margin:.65rem 0 1.2rem;
}}
.kpi-grid .kpi {{
  min-width:0;
  min-height:126px;
  height:100%;
  padding:.88rem 1.02rem;
  box-sizing:border-box;
}}
.kpi-grid .kpi-value {{font-size:clamp(1.75rem,2.2vw,2.2rem);margin-top:.40rem;}}
.kpi-grid .kpi-label {{font-size:.73rem;}}
.kpi-grid .kpi-help {{font-size:.77rem;line-height:1.35;}}
@media(max-width:1050px){{.kpi-grid{{grid-template-columns:repeat(3,minmax(0,1fr));}}}}
@media(max-width:650px){{.kpi-grid{{grid-template-columns:repeat(2,minmax(0,1fr));gap:.72rem;}} .hero-icon{{top:.25rem;font-size:1.85rem;}}}}
/* EDA cards live in their own columns and cannot change hero KPI sizing. */
[data-testid="stPlotlyChart"] {{box-sizing:border-box;max-width:100%;overflow:hidden;}}
hr {{border-color: rgba(255,255,255,.06);}}
</style>
""", unsafe_allow_html=True)


def style(fig: go.Figure, h: int = 380) -> go.Figure:
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=h,
        margin=dict(l=12, r=12, t=54, b=16),
        colorway=[CYAN, GOLD, PINK, GREEN, '#7AA2FF'],
        font=dict(color=TEXT, size=12),
        title=dict(font=dict(color=TEXT, size=19), x=0.02),
        legend=dict(font=dict(color=TEXT_MUTED)),
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,.10)', zeroline=False, color=TEXT_MUTED)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,.10)', zeroline=False, color=TEXT_MUTED)
    return fig


def kpi(col, label: str, value: str, help_text: str | None = None) -> None:
    help_html = f"<div class='kpi-help'>{help_text}</div>" if help_text else ''
    col.markdown(
        f"<div class='kpi'><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div>{help_html}</div>",
        unsafe_allow_html=True,
    )


def panel(title: str, subtitle: str | None = None) -> None:
    st.markdown(
        f"<div class='panel-title'>{title}</div>" +
        (f"<div class='panel-subtitle'>{subtitle}</div>" if subtitle else ''),
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f"<div class='panel-note'>{text}</div>", unsafe_allow_html=True)


def fmt_num(v):
    if isinstance(v, (int, np.integer)):
        return f'{int(v):,}'
    if isinstance(v, (float, np.floating)):
        if abs(v) >= 100:
            return f'{v:,.0f}'
        if abs(v) >= 10:
            return f'{v:,.1f}'
        return f'{v:,.2f}'
    return str(v)


def dark_table(df: pd.DataFrame, title: str | None = None, first_col_muted: bool = False) -> None:
    if df.empty:
        st.info('No records available.')
        return
    show = df.copy()
    for c in show.columns:
        if pd.api.types.is_datetime64_any_dtype(show[c]):
            show[c] = show[c].dt.strftime('%Y-%m-%d')
        else:
            show[c] = show[c].map(lambda x: fmt_num(x) if pd.notna(x) else '—')
    if title:
        st.markdown(f"<div class='panel-title' style='margin-bottom:.55rem'>{title}</div>", unsafe_allow_html=True)
    cls = 'table-card table-muted' if first_col_muted else 'table-card'
    html = [f"<div class='{cls}'><table><thead><tr>"]
    for c in show.columns:
        html.append(f'<th>{escape(str(c))}</th>')
    html.append('</tr></thead><tbody>')
    for _, row in show.iterrows():
        html.append('<tr>')
        for val in row:
            html.append(f'<td>{escape(str(val))}</td>')
        html.append('</tr>')
    html.append('</tbody></table></div>')
    st.markdown(''.join(html), unsafe_allow_html=True)


def team_to_map_name(team: str) -> str:
    return COUNTRY_ALIASES.get(team, team)


@st.cache_data
def advanced_reports():
    paths = ['advanced_metrics.json','feature_ablation_2022.csv','error_by_elo_gap.csv',
             'error_by_experience.csv','error_by_stage.csv','misjudged_teams_2026.csv',
             'upset_by_elo_gap.csv','giant_killers.csv','biggest_upsets.csv',
             'knockout_decisions_by_year.csv','knockout_metadata_coverage.csv','format_2022_2026.csv',
             'team_archetypes.csv','archetype_centroids.csv']
    if not all((REP/x).exists() for x in paths):
        return None
    out = {}
    for name in paths:
        if name.endswith('.json'):
            out[name] = json.loads((REP/name).read_text(encoding='utf8'))
        else:
            out[name] = pd.read_csv(REP/name)
    return out


NEEDED = [
    PROC / 'matches_clean.csv',
    PROC / 'tournament_summary_clean.csv',
    PROC / 'elo_history.csv',
    PROC / 'team_state.csv',
    MOD / 'worldcup_2026_backtest.csv',
    MOD / 'match_outcome_model.joblib',
    REP / 'model_metrics.json',
]
missing = [p.name for p in NEEDED if not p.exists()]
if missing:
    st.error(f'Missing files: {missing}. Run notebook 01, then `python train_model.py`.')
    st.stop()


@st.cache_data
def load_data():
    matches = pd.read_csv(PROC / 'matches_clean.csv', parse_dates=['date'])
    matches['ml_eligible'] = matches['ml_eligible'].astype(str).str.lower().eq('true')
    matches = matches[matches['goals_1'].notna() & matches['goals_2'].notna()].copy()
    matches['goals_total'] = matches['goals_1'] + matches['goals_2']
    matches['map_team_1'] = matches['team_1'].map(team_to_map_name)
    matches['map_team_2'] = matches['team_2'].map(team_to_map_name)

    summary = pd.read_csv(PROC / 'tournament_summary_clean.csv')
    backtest = pd.read_csv(MOD / 'worldcup_2026_backtest.csv', parse_dates=['date'])
    elo = pd.read_csv(PROC / 'elo_history.csv', parse_dates=['date'])
    team_state = pd.read_csv(PROC / 'team_state.csv', index_col='team')
    metrics = json.loads((REP / 'model_metrics.json').read_text())
    return matches, summary, backtest, elo, team_state, metrics


@st.cache_resource
def load_model():
    return joblib.load(MOD / 'match_outcome_model.joblib')


@st.cache_data
def load_lineup_tables():
    matches_path = PROC / 'lineup_matches.csv'
    players_path = PROC / 'lineup_players.csv'
    subs_path = PROC / 'lineup_substitutions.csv'
    if not matches_path.exists() or not players_path.exists():
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    return (pd.read_csv(matches_path, parse_dates=['date']),
            pd.read_csv(players_path),
            pd.read_csv(subs_path) if subs_path.exists() else pd.DataFrame())


m, summ, bt, elo, state, mx = load_data()
lineup_matches, lineup_players, lineup_subs = load_lineup_tables()
bundle = load_model()
adv = advanced_reports()
audit_file = REP / 'evaluation_audit.json'
audit = json.loads(audit_file.read_text(encoding='utf8')) if audit_file.exists() else None

e = m[m['ml_eligible']].copy()
modern = e[e['year'] >= 2002].copy()
teams = sorted(set(modern['team_1']) | set(modern['team_2']))

# Derived tables
appearances_all = pd.concat([m[['year', 'team_1']].rename(columns={'team_1': 'team'}),
                             m[['year', 'team_2']].rename(columns={'team_2': 'team'})], ignore_index=True)
champions = summ['champion'].replace({'West Germany': 'Germany'})


def final_match(year: int) -> pd.Series | None:
    pool = m[m['year'] == year].copy()
    if pool.empty:
        return None
    finals = pool[pool['round'].astype(str).str.contains('Final', case=False, na=False)]
    if finals.empty:
        return None
    finals = finals.sort_values('date')
    return finals.iloc[-1]


def winner_from_row(row: pd.Series) -> str:
    if row is None:
        return 'Unknown'
    y=int(row['year'])
    source=ROOT/'data'/'raw'/'openfootball'/str(y)/'worldcup.json'
    if source.exists():
        games=json.loads(source.read_text(encoding='utf8')).get('matches',[])
        for match in games:
            if (str(match.get('round','')).strip().lower()=='final' and
                match.get('team1')==row['team_1'] and match.get('team2')==row['team_2']):
                score=match.get('score') or {}
                for field in ('p','pen','pens','penalties','et','ft'):
                    result=score.get(field)
                    if isinstance(result,list) and len(result)==2 and result[0]!=result[1]:
                        return match['team1'] if result[0]>result[1] else match['team2']
    if row['goals_1']!=row['goals_2']:
        return row['team_1'] if row['goals_1']>row['goals_2'] else row['team_2']
    return 'Unresolved — check extra time / penalties in source'


def build_globe_values(metric: str, lo: int, hi: int) -> pd.DataFrame:
    subset = m[m['year'].between(lo, hi)].copy()
    if metric == 'Titles (all-time)':
        df = champions.value_counts().rename_axis('team').reset_index(name='value')
        df['label'] = 'Titles'
    elif metric == 'World Cup appearances':
        tmp = pd.concat([subset[['team_1']].rename(columns={'team_1': 'team'}),
                         subset[['team_2']].rename(columns={'team_2': 'team'})], ignore_index=True)
        df = tmp['team'].value_counts().rename_axis('team').reset_index(name='value')
        df['label'] = 'Match appearances'
    elif metric == 'Goals scored':
        a = subset.groupby('team_1', as_index=False)['goals_1'].sum().rename(columns={'team_1': 'team', 'goals_1': 'value'})
        b = subset.groupby('team_2', as_index=False)['goals_2'].sum().rename(columns={'team_2': 'team', 'goals_2': 'value'})
        df = pd.concat([a, b]).groupby('team', as_index=False)['value'].sum()
        df['label'] = 'Goals scored'
    else:  # Elo rating (end of 2026)
        df = state.reset_index()[['team', 'elo']].rename(columns={'elo': 'value'})
        df['label'] = 'Elo rating'
    df['map_name'] = df['team'].map(team_to_map_name)
    df = df.dropna(subset=['map_name']).copy()
    group_aggregation = 'max' if metric == 'Elo rating' else 'sum'
    df = df.groupby(['map_name', 'label'], as_index=False)['value'].agg(group_aggregation)
    return df.sort_values('value', ascending=False)


st.markdown("""
<div class='hero-wrap'>
  <div class='hero-icon'>⚽</div>
  <div>
    <div class='hero-title'>FIFA World Cup Intelligence</div>
    <div class='hero-subtitle'>
      Globe-first analytics dashboard for <b>historical performance</b>, <b>team scouting</b>,
      <b>Elo-based prediction</b> and an <b>honest model report</b>.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Compact five-card KPI grid, independent of Streamlit column breakpoints.
hero_kpis = [
    ('Matches analysed', f'{len(m):,}', 'All parsed World Cup match records'),
    ('90-minute goals', f"{int(m['goals_total'].sum()):,}", 'Regulation-time goal records; excludes extra-time goals'),
    ('Different champions', f"{champions.nunique()}", 'World Cup champions, 1930–2026'),
    ('Avg goals / match', f"{m['goals_total'].mean():.2f}", 'Historical average across matches'),
    ('2026 model accuracy', f"{mx['accuracy']:.0%}", 'Same as higher-Elo pick; retrospective only'),
]
hero_html = '<div class="kpi-grid">' + ''.join(
    '<div class="kpi"><div class="kpi-label">{}</div><div class="kpi-value">{}</div>'
    '<div class="kpi-help">{}</div></div>'.format(escape(label), escape(value), escape(help_text))
    for label, value, help_text in hero_kpis
) + '</div>'
st.markdown(hero_html, unsafe_allow_html=True)

st.write('')

t_overview, t_lab, t_pred, t_model, t_insights = st.tabs([
    '🌍 Overview', '🧪 Team Lab', '🎯 Match Predictor', '📊 Model Report', '💡 Insights'
])

# ---------------- Overview ----------------
with t_overview:
    st.markdown("<div class='section-chip'>Command centre overview</div>", unsafe_allow_html=True)
    control_left, control_right = st.columns([1.35, 1])
    years = sorted(m['year'].astype(int).unique())
    with control_left:
        year_range = st.select_slider('Tournament range', options=years, value=(2002, years[-1]))
    with control_right:
        map_metric = st.selectbox('Globe metric', ['World Cup appearances', 'Goals scored', 'Titles (all-time)', 'Elo rating'], index=0, key='globe_metric')

    globe_mode = st.radio('Globe view', ['Cinematic 3D globe', 'Interactive data globe'], horizontal=True,
                          help='3D mode uses Three.js from a CDN. The Plotly mode requires no external JavaScript library.')
    lo, hi = year_range
    metric_df = build_globe_values(map_metric, lo, hi)
    top_metric = metric_df.head(8).copy()

    # Globe chart
    globe = go.Figure()
    globe.add_trace(go.Choropleth(
        locations=metric_df['map_name'],
        locationmode='country names',
        z=metric_df['value'],
        text=metric_df['map_name'],
        customdata=np.stack([metric_df['label']], axis=-1),
        colorscale=[
            [0.0, '#0B2447'],
            [0.25, '#114E8B'],
            [0.55, CYAN],
            [0.8, '#7FE8FF'],
            [1.0, GOLD],
        ],
        marker_line_color='rgba(165, 227, 255, 0.42)',
        marker_line_width=0.55,
        colorbar=dict(title=dict(text=map_metric, font=dict(color=TEXT)), tickfont=dict(color=TEXT)),
        hovertemplate='<b>%{location}</b><br>%{customdata[0]}: %{z:.0f}<extra></extra>',
    ))
    globe.update_geos(
        projection_type='orthographic',
        projection_rotation=dict(lon=18, lat=17),
        showframe=False,
        showcoastlines=True,
        coastlinecolor='rgba(127, 232, 255, 0.35)',
        showcountries=True,
        countrycolor='rgba(127, 232, 255, 0.32)',
        showocean=True,
        oceancolor='rgba(6, 17, 40, 1)',
        showland=True,
        landcolor='rgba(10, 29, 61, 1)',
        bgcolor='rgba(0,0,0,0)',
        lataxis_showgrid=False,
        lonaxis_showgrid=False,
    )
    globe.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=16, b=0),
        height=680
    )

    left, right = st.columns([1.55, 0.95])
    with left:
        with st.container(border=True):
            st.markdown("<span class='globe-panel-marker'></span>", unsafe_allow_html=True)
            panel('WORLD CUP • GLOBAL INTELLIGENCE',
                  f'Cinematic Earth with country-level {map_metric.lower()} statistics • {lo}–{hi}')
            if globe_mode == 'Cinematic 3D globe':
                show_cinematic_globe(metric_df, map_metric, height=604)
                st.caption('Drag to rotate, scroll to zoom, click a country for data. Three.js requires access to cdnjs; switch to Interactive data globe if it does not load.')
            else:
                st.plotly_chart(globe, width='stretch', config={'displayModeBar': False})
                st.caption('Rotate to explore country-level data; hover for details. England, Scotland and Wales share a geography on sovereign-state maps.')

    with right:
        with st.container(border=True):
            panel(f'Top countries by {map_metric.lower()}', 'Fast scouting snapshot for the selected globe layer')
            rank_table = top_metric.rename(columns={'map_name': 'Country', 'value': top_metric['label'].iloc[0] if not top_metric.empty else 'Value'})
            rank_table['Country'] = rank_table['Country'].map(str)
            keep_cols = ['Country', top_metric['label'].iloc[0] if not top_metric.empty else 'Value']
            dark_table(rank_table[keep_cols], first_col_muted=True)

        latest_final = final_match(years[-1])
        latest_winner = winner_from_row(latest_final)
        fact_html = f"""
        <div class='panel' style='margin-top:1rem'>
          <div class='panel-title'>Tournament quick facts</div>
          <ul class='mini-list'>
            <li><b>Range selected:</b> {lo}–{hi}</li>
            <li><b>Latest champion:</b> {champions.iloc[-1] if len(champions) else 'Unknown'} ({int(summ['year'].max())})</li>
            <li><b>2026 final at 90 min:</b> {latest_final['team_1']} {int(latest_final['goals_1'])}-{int(latest_final['goals_2'])} {latest_final['team_2']} ({int(latest_final['year'])})</li>
            <li><b>2026 champion after extra time:</b> {latest_winner} (1–0 AET)</li>
            <li><b>2026 format signal:</b> 104 matches vs 64 matches in each 2002–2022 tournament</li>
          </ul>
        </div>
        """
        st.markdown(fact_html, unsafe_allow_html=True)

        stat_panel = modern[modern['year'].between(lo, hi)]
        draw_rate = (stat_panel['result_90'] == 'draw').mean() if len(stat_panel) else 0
        avg_goals_range = stat_panel['goals_total'].mean() if len(stat_panel) else 0
        st.markdown(
            f"<div class='panel' style='margin-top:1rem'><div class='panel-title'>Quick reads</div>"
            f"<div class='small-note'><b style='color:{TEXT}'>Draw share:</b> {draw_rate:.0%}<br>"
            f"<b style='color:{TEXT}'>Matches in range:</b> {len(stat_panel):,}<br>"
            f"<b style='color:{TEXT}'>Average goals:</b> {avg_goals_range:.2f}<br>"
            f"<b style='color:{TEXT}'>Highest current Elo:</b> {state['elo'].idxmax()} ({state['elo'].max():.0f})</div></div>",
            unsafe_allow_html=True,
        )

    # Three distinct, breathing EDA cards; Streamlit containers own their borders.
    st.markdown("<div class='section-chip' style='margin-top:1.0rem'>Tournament performance • EDA</div>", unsafe_allow_html=True)
    row_a, row_b, row_c = st.columns([1.08, 1.02, 1.10], gap='medium')
    with row_a:
        with st.container(border=True):
            g = m[m['year'].between(lo, hi)].groupby('year', as_index=False)['goals_total'].mean()
            f1 = px.area(g, x='year', y='goals_total', markers=True, title='Average goals per match')
            if not g.empty:
                pk = g.loc[g['goals_total'].idxmax()]
                f1.add_annotation(x=pk['year'], y=pk['goals_total'], text=f"Peak {pk['goals_total']:.1f} ({int(pk['year'])})",
                                  showarrow=True, arrowcolor=GOLD, font=dict(color=GOLD))
            f1.update_layout(showlegend=False, margin=dict(l=48,r=24,t=68,b=36))
            st.plotly_chart(style(f1, 372), width='stretch')
    with row_b:
        with st.container(border=True):
            rr = e[e['year'].between(lo, hi)]['result_90'].map(PRETTY).value_counts()
            f2 = px.pie(values=rr.values, names=rr.index, hole=.63, title='90-minute result mix')
            f2.update_layout(legend=dict(orientation='h',y=-.13,x=0,font=dict(size=10)), margin=dict(l=26,r=18,t=66,b=45))
            st.plotly_chart(style(f2, 372), width='stretch')
    with row_c:
        with st.container(border=True):
            selected_modern = modern[modern['year'].between(lo, hi)]
            mix = pd.crosstab(selected_modern['year'], selected_modern['result_90'], normalize='index').mul(100).rename(columns=PRETTY)
            mix = mix.reset_index().melt('year', var_name='Result', value_name='%')
            f3 = px.bar(mix, x='year', y='%', color='Result', title='Result mix per tournament (%)')
            f3.update_layout(barmode='stack', legend=dict(orientation='h',y=-.20,x=0,font=dict(size=10)), margin=dict(l=40,r=18,t=68,b=48))
            st.plotly_chart(style(f3, 372), width='stretch')

    st.markdown("<div class='section-chip' style='margin-top:.75rem'>Goal distribution • statistical analysis</div>", unsafe_allow_html=True)
    box_source = m[m['year'].between(lo, hi)].copy()
    box_source['Tournament'] = box_source['year'].astype(str)
    bp = px.box(box_source, x='Tournament', y='goals_total', points='outliers',
                color_discrete_sequence=[CYAN], title='Goals per match by World Cup tournament')
    bp.update_traces(fillcolor='rgba(24,224,255,.22)', line_color=CYAN,
                     marker=dict(color=GOLD, opacity=.82, size=6))
    bp.update_layout(yaxis_title='Goals in match', xaxis_title='Tournament', showlegend=False)
    st.plotly_chart(style(bp, 385), width='stretch')
    st.caption('Each box shows the spread of match goal totals for one tournament. Outlier points identify unusually high-scoring matches, not errors.')
    note('The globe provides the visual entry point; the box plot and supporting charts provide testable, interpretable historical comparisons.')

# ---------------- Team Lab ----------------
with t_lab:
    st.markdown("<div class='section-chip'>Interactive scouting lab</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    ta = c1.selectbox('Team A', teams, index=teams.index('Argentina') if 'Argentina' in teams else 0, key='lab_a')
    tb = c2.selectbox('Team B', teams, index=teams.index('France') if 'France' in teams else 1, key='lab_b')

    def stats(team: str) -> dict:
        s = modern[(modern['team_1'] == team) | (modern['team_2'] == team)]
        gf = np.where(s['team_1'] == team, s['goals_1'], s['goals_2'])
        ga = np.where(s['team_1'] == team, s['goals_2'], s['goals_1'])
        n = max(len(s), 1)
        return {
            'Matches': len(s),
            'Wins': int((gf > ga).sum()),
            'Draws': int((gf == ga).sum()),
            'Points / game': round(((gf > ga) * 3 + (gf == ga)).sum() / n, 2),
            'Goals for / game': round(gf.mean(), 2),
            'Goals against / game': round(ga.mean(), 2),
            'Elo (end of 2026)': round(state.loc[team, 'elo']) if team in state.index else np.nan,
        }

    sa, sb = stats(ta), stats(tb)
    emin, emax = state['elo'].min(), state['elo'].max()

    def radar_values(s: dict):
        return [
            s['Points / game'] / 3,
            min(s['Goals for / game'] / 3, 1),
            max(1 - s['Goals against / game'] / 3, 0),
            (s['Elo (end of 2026)'] - emin) / (emax - emin) if pd.notna(s['Elo (end of 2026)']) else 0,
            min(s['Matches'] / 45, 1),
        ]

    axes = ['Results', 'Attack', 'Defence', 'Elo strength', 'Experience']
    radar = go.Figure()
    for nm, s, col in ((ta, sa, CYAN), (tb, sb, GOLD)):
        values = radar_values(s)
        radar.add_trace(go.Scatterpolar(r=values + values[:1], theta=axes + axes[:1], fill='toself', name=nm,
                                        line=dict(color=col), opacity=0.78))
    radar.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor='rgba(255,255,255,.10)'),
            angularaxis=dict(gridcolor='rgba(255,255,255,.08)', color=TEXT),
        ),
        title='Team profile (2002–2026, normalized 0–1 scale)',
    )

    lab_left, lab_right = st.columns([1.02, 1.2])
    with lab_left:
        st.plotly_chart(style(radar, 440), width='stretch')
    with lab_right:
        hist = elo[elo['team'].isin({canonical(ta), canonical(tb)})].copy()
        line = px.line(hist, x='date', y='elo', color='team', title='Elo rating over time')
        st.plotly_chart(style(line, 440), width='stretch')

    compare_df = pd.DataFrame({ta: pd.Series(sa), tb: pd.Series(sb)}).reset_index().rename(columns={'index': 'Metric'})
    dark_table(compare_df, title='Side-by-side team comparison', first_col_muted=True)

    meet = e[((e['team_1'] == ta) & (e['team_2'] == tb)) | ((e['team_1'] == tb) & (e['team_2'] == ta))].copy()
    meet['Score'] = meet['goals_1'].astype(int).astype(str) + '-' + meet['goals_2'].astype(int).astype(str)
    meet = meet[['date', 'team_1', 'Score', 'team_2', 'round']].sort_values('date', ascending=False)
    if not meet.empty:
        meet['team_1'] = meet['team_1'].map(str)
        meet['team_2'] = meet['team_2'].map(str)
    dark_table(meet.head(12), title='Head-to-head World Cup meetings')

    st.markdown("<div class='section-chip' style='margin-top:1.2rem'>Team comparison • attack vs defence</div>", unsafe_allow_html=True)
    scatter_l, scatter_r = st.columns([1.65, 0.9])
    with scatter_r:
        min_games = st.slider('Minimum matches per team', min_value=1, max_value=15, value=5,
                              help='Filter out small-sample teams that could look unusually strong after only one or two games.')
        st.markdown(f"""<div class='panel'><div class='panel-title'>How to read this chart</div>
        <div class='small-note'>Each point is a country in 2002–2026 World Cup matches.
        <b style='color:{TEXT}'>Further right</b> means more goals scored per game;
        <b style='color:{TEXT}'>lower down</b> means fewer goals conceded per game.
        Lower-right teams combine attack and defence effectively.</div></div>""", unsafe_allow_html=True)
    attack_rows = []
    for nm in teams:
        ms = modern[(modern['team_1'] == nm) | (modern['team_2'] == nm)]
        if len(ms) < min_games:
            continue
        gf = np.where(ms['team_1'] == nm, ms['goals_1'], ms['goals_2'])
        ga = np.where(ms['team_1'] == nm, ms['goals_2'], ms['goals_1'])
        attack_rows.append({'Country':str(nm), 'Goals scored / game':float(gf.mean()),
                            'Goals conceded / game':float(ga.mean()), 'Matches':len(ms),
                            'Selected':nm in (ta,tb)})
    with scatter_l:
        sc = pd.DataFrame(attack_rows)
        if not sc.empty:
            scat = px.scatter(sc, x='Goals scored / game', y='Goals conceded / game',
                              size='Matches', hover_name='Country', color='Selected',
                              color_discrete_map={True:GOLD,False:CYAN}, size_max=27,
                              title='Team attacking efficiency vs defensive resilience')
            scat.update_layout(showlegend=False)
            scat.update_traces(marker=dict(line=dict(width=1,color='#B2F3FF'),opacity=.84))
            st.plotly_chart(style(scat,385), width='stretch')
            st.caption(f'{ta} and {tb} are highlighted in gold (when they meet the selected minimum-match threshold).')

    st.markdown("<div class='section-chip' style='margin-top:1.1rem'>Historical match explorer • Starting XI</div>", unsafe_allow_html=True)
    if lineup_matches.empty:
        st.info('Historical lineup CSVs are missing. Run `python build_lineups.py` once to generate them from the already-included worldcup-full.json files.')
    else:
        year_choices = sorted(lineup_matches['year'].dropna().astype(int).unique(), reverse=True)
        default_year = year_choices.index(2022) if 2022 in year_choices else 0
        l_year = st.selectbox('World Cup year for Starting XI', year_choices, index=default_year, key='lineup_year')
        year_games = lineup_matches[lineup_matches['year']==l_year].copy().sort_values('date',ascending=False)
        game_opts = year_games['lineup_key'].tolist()
        game_labels = {row.lineup_key:f"{row.date.strftime('%Y-%m-%d')} • {str(row.team_1)} vs {str(row.team_2)} • {row['round']}"
                       for _,row in year_games.iterrows()}
        l_key = st.selectbox('Match with recorded lineups', game_opts,
                             format_func=lambda k: game_labels.get(k,k),key='lineup_match')
        l_game = year_games[year_games['lineup_key']==l_key].iloc[0]
        lineup_schematic = st.selectbox('Illustrative pitch layout (not verified historical formation)',
                                         ['4-3-3', '4-4-2'], key='lineup_schematic')
        pitch_mode = st.radio('Historical lineup visualization', ['Interactive 3D pitch', '2D data pitch'], horizontal=True,
                              help='3D uses Three.js within a Streamlit iframe. Switch to 2D if WebGL or the CDN is unavailable.', key='pitch_mode')
        note('Pitch placement is illustrative, NOT verified formation or player positions. Player names and captain status come from the source.')
        pitch_l,pitch_r = st.columns(2)
        for col, team, tint in [(pitch_l,l_game['team_1'],CYAN),(pitch_r,l_game['team_2'],GOLD)]:
            persons = lineup_players[(lineup_players['lineup_key']==l_key)&
                                     (lineup_players['team']==team)]
            starters = persons[persons['role']=='starter'].sort_values('order')
            with col:
                if pitch_mode == 'Interactive 3D pitch':
                    show_pitch_3d(starters, team, lineup_schematic, tint, height=490)
                else:
                    st.plotly_chart(pitch_figure(starters, team, tint, lineup_schematic, height=420), width='stretch',
                                    config={'displayModeBar':False})
                st.caption('Numbers refer to source-list order, not shirt number or on-field position.')
                roster = starters[['order','player','captain']].copy()
                roster['captain'] = roster['captain'].map({True:'Yes',False:'—'})
                roster.columns=['List #','Player','Captain']
                dark_table(roster,title=f'{str(team)} • verified starter names')
                bench = persons[persons['role']=='bench'].sort_values('order')
                with st.expander('Listed bench players and substitutions'):
                    if len(bench):
                        st.markdown('**Source-listed bench players** (may not be full matchday bench)')
                        dark_table(bench[['order','player']].rename(columns={'order':'List #','player':'Player'}))
                    else:
                        st.caption('No bench entries were supplied in this match record.')
                    if not lineup_subs.empty:
                        events = lineup_subs[(lineup_subs['lineup_key']==l_key)&
                                             (lineup_subs['team']==team)]
                        if not events.empty:
                            st.markdown('**Recorded substitutions**')
                            dark_table(events[['minute','player_on','player_off']].rename(columns={
                                'minute':'Minute','player_on':'On','player_off':'Off'}))

# ---------------- Match Predictor ----------------
with t_pred:
    st.markdown("<div class='section-chip'>Match simulator</div>", unsafe_allow_html=True)
    note('<b>Analyst interpretation:</b> This model predicts 90-minute results using World Cup form and Elo. The team selectors use end-of-2026 state; they do not represent any pre-2026 forecast. Draw recall was low in testing. This is not a live forecast or a betting tool.')
    c1, c2 = st.columns(2)
    pa = c1.selectbox('Team A', teams, index=teams.index('Spain') if 'Spain' in teams else 0, key='pr_a')
    pb = c2.selectbox('Team B', teams, index=teams.index('Brazil') if 'Brazil' in teams else 1, key='pr_b')

    def predictor_row(x: str, y: str) -> dict:
        s1, s2 = state.loc[x], state.loc[y]
        hh = e[((e['team_1'] == x) & (e['team_2'] == y)) | ((e['team_1'] == y) & (e['team_2'] == x))]
        wins_x = int((((hh['team_1'] == x) & (hh['goals_1'] > hh['goals_2'])) |
                      ((hh['team_2'] == x) & (hh['goals_2'] > hh['goals_1']))).sum())
        row = {
            'team_1_prev_matches': s1['prev_matches'],
            'team_2_prev_matches': s2['prev_matches'],
            'team_1_win_rate_last5': s1['win_rate_last5'],
            'team_2_win_rate_last5': s2['win_rate_last5'],
            'team_1_goals_for_last5': s1['goals_for_last5'],
            'team_2_goals_for_last5': s2['goals_for_last5'],
            'team_1_goals_against_last5': s1['goals_against_last5'],
            'team_2_goals_against_last5': s2['goals_against_last5'],
            'win_rate_diff': s1['win_rate_last5'] - s2['win_rate_last5'],
            'goals_for_diff': s1['goals_for_last5'] - s2['goals_for_last5'],
            'goals_against_diff': s1['goals_against_last5'] - s2['goals_against_last5'],
            'head_to_head_matches': len(hh),
            'head_to_head_team1_win_rate': wins_x / len(hh) if len(hh) else np.nan,
            'elo_1': s1['elo'],
            'elo_2': s2['elo'],
            'elo_diff': s1['elo'] - s2['elo'],
            'elo_expected_1': 1 / (1 + 10 ** ((s2['elo'] - s1['elo']) / 400)),
        }
        p = bundle['model'].predict_proba(pd.DataFrame([row])[bundle['features']])[0]
        return dict(zip(bundle['classes'], p))

    if pa == pb:
        st.warning('Pick two different teams.')
    else:
        p12, p21 = predictor_row(pa, pb), predictor_row(pb, pa)
        probs = {
            pa: (p12['team_1_win'] + p21['team_2_win']) / 2,
            'Draw': (p12['draw'] + p21['draw']) / 2,
            pb: (p12['team_2_win'] + p21['team_1_win']) / 2,
        }
        prob_df = pd.DataFrame({'Outcome': list(probs.keys()), 'Probability': list(probs.values())})
        prob_df['Display'] = (prob_df['Probability'] * 100).round(1).astype(str) + '%'
        gap = state.loc[pa, 'elo'] - state.loc[pb, 'elo']
        favourite = pa if gap >= 0 else pb

        pl, pr = st.columns([1, 1], gap='large')
        with pl:
            donut = go.Figure(go.Pie(
                labels=prob_df['Outcome'],
                values=prob_df['Probability'],
                hole=.64,
                marker=dict(colors=[CYAN, '#667996', GOLD]),
                sort=False,
            ))
            donut.update_layout(title='Predicted 90-minute outcome split')
            st.plotly_chart(style(donut, 400), width='stretch')
            st.caption('The 2D chart shows the estimated 90-minute probabilities; hover over a segment for exact values.')
        with pr:
            with st.container(border=True):
                panel('Prediction briefing', 'A quick readable summary for judges and analysts')
                top_outcome = prob_df.sort_values('Probability', ascending=False).iloc[0]
                st.markdown(
                    f"<div class='small-note'>"
                    f"<b style='color:{TEXT}'>Most likely outcome:</b> {escape(str(top_outcome['Outcome']))} ({top_outcome['Display']})<br><br>"
                    f"<b style='color:{TEXT}'>Elo gap:</b> {abs(gap):.0f} points favouring <b style='color:{CYAN_SOFT if favourite == pa else GOLD}'>{escape(favourite)}</b><br>"
                    f"<b style='color:{TEXT}'>Last-5 World Cup win rate:</b> {escape(pa)} {state.loc[pa, 'win_rate_last5']:.0%} vs {escape(pb)} {state.loc[pb, 'win_rate_last5']:.0%}<br>"
                    f"<b style='color:{TEXT}'>Caution:</b> only {mx['draw_recall']:.0%} draw recall on the 2026 backtest; probability is uncertain, not a guaranteed result."
                    f"</div>", unsafe_allow_html=True
                )

        view = st.radio('How should this matchup be presented?',
                        ['90-minute outcome', 'Knockout advancement — scenario only'], horizontal=True,
                        key='predictor_target')
        if view == 'Knockout advancement — scenario only':
            note('Knockout advancement has NOT been trained or validated. The assumptions below are a ' 
                       'what-if scenario for extra time and penalties, not a model probability.')
            a_share = st.slider(f'If tied after 90 minutes, assumed probability of {pa} advancing',
                                min_value=0.05,max_value=0.95,value=0.50,step=0.05)
            st.markdown(f'**Illustrative advancement under this assumption:** '
                        f'{pa}: **{(probs[pa]+probs["Draw"]*a_share):.1%}**, '
                        f'{pb}: **{(probs[pb]+probs["Draw"]*(1-a_share)):.1%}**. '
                        'The 90-minute draw probability is allocated using your chosen assumption; '
                        'no extra-time or penalty outcome model is used.')

        compare = pd.DataFrame({
            'Metric': ['Elo', 'Prev matches', 'Win rate last 5', 'Goals for last 5', 'Goals against last 5'],
            pa: [state.loc[pa, 'elo'], state.loc[pa, 'prev_matches'], state.loc[pa, 'win_rate_last5'], state.loc[pa, 'goals_for_last5'], state.loc[pa, 'goals_against_last5']],
            pb: [state.loc[pb, 'elo'], state.loc[pb, 'prev_matches'], state.loc[pb, 'win_rate_last5'], state.loc[pb, 'goals_for_last5'], state.loc[pb, 'goals_against_last5']],
        })
        dark_table(compare, title='Model input snapshot', first_col_muted=True)

        st.markdown("<div class='section-chip' style='margin-top:1.15rem'>Strength over time</div>", unsafe_allow_html=True)
        p_hist = elo[elo['team'].isin({canonical(pa),canonical(pb)})].copy()
        p_hist = p_hist[p_hist['date'].dt.year >= 2002].sort_values('date')
        if not p_hist.empty:
            trend = px.line(p_hist,x='date',y='elo',color='team',
                            color_discrete_sequence=[CYAN,GOLD],
                            title='Historical Elo rating trends: how team strength evolved')
            trend.update_layout(xaxis_title='Match date',yaxis_title='World Cup-only Elo rating',
                                legend_title_text='National team')
            st.plotly_chart(style(trend,385),width='stretch')
            st.caption('Ratings are updated after World Cup matches, not continuously during the calendar year. The model uses a historical snapshot through 2026, not current real-world squad quality.')
        note('**For judges:** the probability donut answers **what** the model estimates. The Elo trend and input comparison help explain **why**, while the Model Report quantifies **how reliable** the estimates actually are.')

# ---------------- Model Report ----------------
with t_model:
    st.markdown("<div class='section-chip'>Model accountability</div>", unsafe_allow_html=True)
    kk = st.columns(4)
    kpi(kk[0], f"Accuracy ({mx['test_matches']} matches)", f"{mx['accuracy']:.1%}")
    kpi(kk[1], '95% match-bootstrap interval', f"{mx['ci_low']:.0%}–{mx['ci_high']:.0%}")
    kpi(kk[2], 'Higher-Elo baseline', f"{audit['higher_elo_accuracy']:.1%}" if audit else 'Run audit_model.py')
    kpi(kk[3], 'Draw recall', f"{mx['draw_recall']:.0%}")
    st.write('')
    if audit:
        note('<b>Fair comparison:</b> both the selected Random Forest and a simple higher-Elo pick made '
             f'<b>{audit["model_accuracy"]:.1%}</b> correct predictions on the 104 held-out 2026 matches. '
             'The older 44.2% first-listed baseline was weaker and is not the headline comparison. '
             "The model\'s probability quality is slightly better than Elo-only logistic regression, "
             'but this improvement is uncertain.')
        c1,c2,c3,c4=st.columns(4)
        kpi(c1,'RF log loss',f'{audit["model_log_loss"]:.3f}','Lower is better')
        kpi(c2,'Elo-logistic log loss',f'{audit["elo_logistic_log_loss"]:.3f}','Same 104 matches')
        kpi(c3,'RF multiclass Brier',f'{audit["model_brier"]:.3f}','Lower is better (0 to 2)')
        kpi(c4,'Elo-logistic Brier',f'{audit["elo_logistic_brier"]:.3f}','Same three outcome classes')
        mc=audit['mcnemar_higher_elo']
        st.caption(f'Paired exact McNemar vs higher-Elo PICK: p={mc["p_value"]:.3f} '
                   f'(RF-only correct {mc["model_only_correct"]}; Elo-only correct {mc["baseline_only_correct"]}). '
                   'This does not show an accuracy advantage. Team-cluster bootstrap interval for '
                   f'log-loss improvement: {audit["team_cluster_bootstrap_logloss_improvement_ci"][0]:+.3f} '
                   f'to {audit["team_cluster_bootstrap_logloss_improvement_ci"][1]:+.3f}; includes zero.')
        st.caption(f'Draws: {audit["actual_draws"]}/{audit["n_test"]} actual, '
                   f'{audit["predicted_draws"]}/{audit["n_test"]} predicted, '
                   f'draw recall {audit["draw_recall"]:.0%}. '
                   'An alternative Poisson or calibrated-draw model requires separate chronological validation; '
                   'we have not modified the existing model without such evidence.')

    wf = pd.DataFrame(mx['walkforward'])
    sacc = pd.Series(mx['stage_accuracy']).rename_axis('stage').reset_index(name='accuracy')
    m1, m2 = st.columns(2)
    with m1:
        f = px.bar(wf, x='log_loss', y='model', orientation='h', color='log_loss',
                   color_continuous_scale=[GREEN, GOLD, PINK], title='Walk-forward log loss 2014–2022 (lower is better)')
        st.plotly_chart(style(f, 340), width='stretch')
    with m2:
        stage_file=REP/'stage_accuracy_intervals.csv'
        if stage_file.exists():
            sacc=pd.read_csv(stage_file)
            sacc['upper_error']=sacc['ci_high']-sacc['accuracy']
            sacc['lower_error']=sacc['accuracy']-sacc['ci_low']
            sacc['label']=sacc.apply(lambda r:f"{r['accuracy']:.0%} (n={int(r['n'])})",axis=1)
            f=px.bar(sacc,x='stage',y='accuracy',color='stage',text='label',
                     error_y='upper_error',error_y_minus='lower_error',
                     color_discrete_sequence=[CYAN,GOLD],title='2026 stage accuracy ± 95% Wilson CI')
            f.update_layout(showlegend=False,yaxis_tickformat='.0%',yaxis_range=[0,1.10])
        else:
            f=px.bar(sacc,x='stage',y='accuracy',color='stage',
                     color_discrete_sequence=[CYAN,GOLD],title='2026 stage accuracy')
        st.plotly_chart(style(f,340),width='stretch')

    cal_ci_file=REP/'calibration_with_intervals.csv'
    cal_ci=pd.read_csv(cal_ci_file) if cal_ci_file.exists() else None
    if cal_ci is None:
        raise FileNotFoundError('Missing current calibration_with_intervals.csv. Run python audit_model.py.')
    calib = go.Figure([go.Scatter(x=[0,1], y=[0,1], mode='lines',
        line=dict(dash='dash', color='#6b7a99'), name='Perfect')])
    for label, chunk in cal_ci.groupby('class'):
        chunk = chunk.sort_values('predicted')
        calib.add_trace(go.Scatter(
            x=chunk['predicted'], y=chunk['observed'], mode='markers+lines',
            name=label, customdata=chunk[['n']],
            hovertemplate='Predicted %{x:.2f}<br>Observed %{y:.2f}<br>n=%{customdata[0]}<extra></extra>',
            error_y=dict(type='data', array=chunk['ci_high']-chunk['observed'],
                         arrayminus=chunk['observed']-chunk['ci_low'], visible=True)))
    calib.update_layout(title='Calibration: predicted vs observed frequency', xaxis_title='Predicted probability', yaxis_title='Observed frequency')
    cm = pd.crosstab(bt['result_90'].map(PRETTY), bt['predicted_result'].map(PRETTY))
    mm1, mm2 = st.columns(2)
    with mm1:
        st.plotly_chart(style(calib, 360), width='stretch')
    with mm2:
        heat = px.imshow(cm, text_auto=True, color_continuous_scale=['#0d1b36', CYAN], title='Confusion matrix (rows = actual)')
        st.plotly_chart(style(heat, 360), width='stretch')

    pi_path = REP / 'permutation_importance_2022.csv'
    if pi_path.exists():
        pi = pd.read_csv(pi_path).head(12).iloc[::-1]
        impf = px.bar(pi, x='mean_log_loss_increase', y='feature', orientation='h',
                      error_x='std', color_discrete_sequence=[CYAN],
                      title='2022 holdout permutation importance (log-loss impact)')
        impf.update_layout(showlegend=False,xaxis_title='Increase in log loss when shuffled',yaxis_title='')
        st.plotly_chart(style(impf,420),width='stretch')
        st.caption('Permutation importance is measured on 2022 validation after training through 2018. '
                   'Correlated variables can share importance. This is not evidence of causal importance.')
        if audit:
            ex=audit['experience_2022_validation']
            st.caption(f'Experience bias check (2022, n={ex["n"]}): full model log loss '
                       f'{ex["full_model"]["log_loss"]:.3f}; without previous-match-count features '
                       f'{ex["without_experience"]["log_loss"]:.3f}. Neither result proves that experience causes wins.')
    else:
        st.info('Run `python audit_model.py` for held-out permutation importance.')

    up = bt.sort_values('prob_actual').head(8).copy()
    up['Score'] = up['goals_1'].astype(int).astype(str) + '-' + up['goals_2'].astype(int).astype(str)
    up['Chance of actual result'] = (up['prob_actual'] * 100).round(1).astype(str) + '%'
    dark_table(up[['date', 'team_1', 'Score', 'team_2', 'Chance of actual result', 'stage']], title='Biggest surprises of 2026')
    note('This section stays in the dashboard because it makes the project more credible: you are not only showing wins, you are showing uncertainty, calibration and known weaknesses.')

    # ---------------- v6: Statistical evidence / ablation / error slicing ----------------
    st.markdown("<div class='section-chip' style='margin-top:1.3rem'>Evidence beyond accuracy</div>", unsafe_allow_html=True)
    if adv is None:
        st.info('Advanced reports are missing. Run `python build_advanced_analytics.py` once after `train_model.py`.')
    else:
        cmc = adv['advanced_metrics.json']['comparison']
        with st.expander('Historical comparison: first-listed majority-class baseline (weak)'):
            st.subheader('Legacy weak-baseline comparison (for transparency)')
            sc1, sc2, sc3 = st.columns(3)
            kpi(sc1, 'Exact McNemar p-value', f"{cmc['mcnemar_exact_p_two_sided']:.4f}", 'Legacy first-listed baseline; not a strong comparator')
            kpi(sc2, 'Model-only correct', str(cmc['model_only_correct']), 'Baseline missed; model right')
            kpi(sc3, 'Baseline-only correct', str(cmc['baseline_only_correct']), 'Model missed; baseline right')
            note(f"<b>Interpretation:</b> p = {cmc['mcnemar_exact_p_two_sided']:.4f} for a paired test against a fixed "
                 f"training-majority baseline ({cmc['baseline_label']}). The result is exploratory: it does not "
                 'establish reliable performance in future World Cups, or superiority over higher-Elo picks.')

        st.subheader('Feature ablation — 2022 chronological validation')
        ab = adv['feature_ablation_2022.csv'].copy()
        # Approximate Wilson intervals for each accuracy; match dependencies are not resolved by this CI.
        z=1.96
        nval=ab['Validation matches'].astype(float)
        phat=ab['Accuracy'].astype(float)
        denom=1+z*z/nval
        centre=(phat+z*z/(2*nval))/denom
        width=z*np.sqrt(phat*(1-phat)/nval+z*z/(4*nval*nval))/denom
        ab['upper_err']=(centre+width-phat).clip(lower=0)
        ab['lower_err']=(phat-centre+width).clip(lower=0)
        c1, c2 = st.columns(2)
        with c1:
            f = px.bar(ab, x='Features', y='Accuracy', text=ab.apply(lambda r:f"{r['Accuracy']:.1%} (n={int(r['Validation matches'])})",axis=1),
                       error_y='upper_err',error_y_minus='lower_err',color='Features', color_discrete_sequence=[CYAN, GOLD, GREEN],
                       title='Ablation: 2022 holdout accuracy')
            f.update_layout(showlegend=False, yaxis_tickformat='.0%', yaxis_range=[0,0.8])
            st.plotly_chart(style(f,340), width='stretch')
        with c2:
            f = px.bar(ab, x='Features', y='Log loss', color='Features', text_auto='.3f',
                       color_discrete_sequence=[CYAN, GOLD, GREEN], title='Ablation: log loss (lower is better)')
            f.update_layout(showlegend=False)
            st.plotly_chart(style(f,340), width='stretch')
        st.caption('Ablation uses 64 matches from the 2022 chronological holdout; error bars are approximate binomial Wilson intervals and ignore repeated teams. The held-out 2026 tournament is not used for feature selection.')

        st.subheader('2026 error analysis — where does the model struggle?')
        e1, e2, e3 = st.columns(3)
        for container, file, key, title in [
            (e1,'error_by_elo_gap.csv','elo_gap_bucket','Accuracy by pre-match Elo gap'),
            (e2,'error_by_experience.csv','experience_bucket','Accuracy by team experience'),
            (e3,'error_by_stage.csv','stage','Accuracy by match stage'),
        ]:
            with container:
                d = adv[file].copy(); d[key] = d[key].astype(str)
                f = px.bar(d, x=key, y='accuracy', color=key,
                           text=d['matches'].map(lambda n:f'n={n}'),
                           title=title, color_discrete_sequence=[CYAN, GOLD, GREEN, PINK, '#95A9FF'])
                f.update_layout(yaxis_tickformat='.0%', yaxis_range=[0,1],
                                xaxis_title='', showlegend=False)
                f.update_traces(textposition='outside')
                st.plotly_chart(style(f,350),width='stretch')
        st.caption('Experience means the smaller of the two teams’ previous World Cup match counts, measured before kickoff. Small buckets should not be overinterpreted.')
        bad=adv['misjudged_teams_2026.csv'].head(10).copy()
        bad['Team']=bad['team'].map(str)
        bad['Error rate']=bad['error_rate'].map(lambda x:f'{x:.0%}')
        dark_table(bad[['Team','matches','errors','Error rate']].rename(columns={'matches':'Matches','errors':'Wrong picks'}),
                   title='Most frequently misjudged teams (2026 sample; minimum 3 appearances)')
        note('Error counts are descriptive: matches contribute to both participating teams, so these totals must not be summed as independent errors.')

# ---------------- Insights ----------------
with t_insights:
    st.markdown("<div class='section-chip'>Executive takeaways</div>", unsafe_allow_html=True)
    st.subheader('Executive summary — three findings and actions')
    st.markdown('''
**Finding 1 — Strength matters more than complexity.** The model and a higher-Elo pick have identical accuracy in 2026.
**Action:** Use Elo as a transparent first-pass scouting benchmark; interpret the model probability as supplemental.

**Finding 2 — Draws remain a weakness.** Only 4 of 29 actual draws were correctly classified.
**Action:** For evenly matched teams, plan multiple match scenarios instead of relying on a single outcome label.

**Finding 3 — Tournament structure changed.** 2026 contains 104 matches compared with 64 in 2022; scoring patterns differ, but this comparison cannot isolate the impact of expansion.
**Action:** Compare goal margins and team strength across stages before claiming a systematic format effect.
''')
    early, late = modern[modern['year'] <= 2010], modern[modern['year'] >= 2014]
    draw_rate = lambda x: (x['result_90'] == 'draw').mean() if len(x) else 0
    g26 = m[m['year'] == 2026]['goals_total'].mean()
    gold = m[m['year'].between(2002, 2022)]['goals_total'].mean()
    worst = bt.sort_values('prob_actual').iloc[0]

    notes = st.columns(2)
    with notes[0]:
        note(f"<b>Draws are fading:</b> {draw_rate(early):.0%} of 2002–2010 matches ended level vs {draw_rate(late):.0%} from 2014 onward.")
        note(f"<b>2026 was high-scoring:</b> {g26:.2f} goals per match vs {gold:.2f} for 2002–2022, but whether tournament expansion explains the difference requires testing.")
    with notes[1]:
        note(f"<b>Knockout accuracy is lower in this sample (not proven generally):</b> {mx['stage_accuracy'].get('Knockout', 0):.0%} accuracy in knockouts vs {mx['stage_accuracy'].get('Group stage', 0):.0%} in groups.")
        note(f"<b>Biggest upset in the test set:</b> {worst['team_1']} {int(worst['goals_1'])}-{int(worst['goals_2'])} {worst['team_2']} (actual result probability {worst['prob_actual']:.0%}).")

    st.subheader('2026 roster, stadium & host context — exploratory')
    squads_file=ROOT/'data/raw/openfootball/2026/worldcup.squads.json'
    stadiums_file=ROOT/'data/raw/openfootball/2026/worldcup.stadiums.json'
    if squads_file.exists() and stadiums_file.exists():
        squads=json.loads(squads_file.read_text(encoding='utf8'))
        venues=json.loads(stadiums_file.read_text(encoding='utf8')).get('stadiums',[])
        position_counts={}
        for team_item in squads:
            for person in team_item.get('players',[]):
                code=person.get('pos','Unspecified')
                position_counts[code]=position_counts.get(code,0)+1
        hosts={'United States','Canada','Mexico'}
        host_matches=m[(m.year==2026)&(m.team_1.isin(hosts)|m.team_2.isin(hosts))]
        c1,c2,c3=st.columns(3)
        kpi(c1,'2026 squad entries',str(len(squads)),'Source roster entries (not confirmed starters)')
        kpi(c2,'Stadium entries',str(len(venues)),'Venue metadata; not trained features')
        kpi(c3,'Host-nation matches',str(len(host_matches)),'Any match featuring Canada, Mexico or United States')
        st.caption('Positional roster coverage: '+', '.join(f'{k} {v}' for k,v in sorted(position_counts.items()))+
                   '. Host involvement is not the same as home advantage. Stadium and squad metadata are '
                   'provided for descriptive scouting, but they are not inputs to the current trained model.')

    st.subheader('How an analyst would use this dashboard')
    st.markdown(
        "- **First glance:** use the globe-centric Overview to identify football power centres and major shifts in scoring and outcome patterns.\n"
        "- **Scouting:** compare two teams in Team Lab to understand attack, defence, experience and Elo trend.\n"
        "- **Match preparation:** use Match Predictor as a baseline discussion tool rather than a final answer.\n"
        "- **Model trust check:** use the Model Report to explain when the system is weak, especially on draws and knockouts."
    )

    with st.expander('Upsets, bracket decisions, tournament effect & archetypes', expanded=False):
        # ---------------- v6: Exploratory football insights ----------------
        st.markdown("<div class='section-chip' style='margin-top:1.5rem'>Exploratory football intelligence</div>", unsafe_allow_html=True)
        if adv is None:
            st.info('Run `python build_advanced_analytics.py` to generate the extended insight charts.')
        else:
            st.subheader('Upset risk by pre-match Elo gap')
            ups = adv['upset_by_elo_gap.csv'].copy()
            upc1, upc2 = st.columns([1.3,1])
            with upc1:
                f = px.bar(ups, x='elo_gap_bucket', y='upset_rate', color='year', barmode='group',
                           text=ups['matches'].map(lambda x:f'n={x}'),
                           title='Underdog win rate: 2022 vs 2026',
                           color_discrete_sequence=[CYAN,GOLD])
                f.update_layout(yaxis_tickformat='.0%',xaxis_title='Absolute pre-match Elo gap',legend_title='Tournament')
                st.plotly_chart(style(f,360),width='stretch')
            with upc2:
                giant=adv['giant_killers.csv'].head(7).copy()
                giant['Team']=giant['underdog'].map(str)
                dark_table(giant[['Team','upset_wins','mean_gap']].rename(columns={
                    'upset_wins':'Underdog wins','mean_gap':'Mean Elo gap'}),title='Giant-killers (small-sample counts, not ranks)')
            st.caption('An upset here means the lower-rated team WON in 90 minutes and Elo gap was at least 25. Draws excluded; 2022/2026 only. This Elo is derived from World Cup results, not an external official strength ranking.')
            biggest=adv['biggest_upsets.csv'].head(8).copy()
            biggest['Underdog']=biggest['underdog'].map(str)
            biggest['Favourite']=biggest['favorite'].map(str)
            dark_table(biggest[['year','Underdog','Favourite','elo_gap']].rename(columns={'year':'Year','elo_gap':'Pre-match Elo gap'}),title='Largest sampled underdog wins')

            st.subheader('How were knockout games decided across tournament years?')
            et = adv['knockout_decisions_by_year.csv'].copy()
            fl=px.bar(et, x='year',y='matches',color='decision',barmode='stack',
                      color_discrete_map={'Penalties':PINK,'Extra time':GOLD,'No ET/penalty flagged':CYAN},
                      title='Knockout decisions in the recorded match metadata')
            fl.update_layout(xaxis_title='Tournament year',yaxis_title='Match records',legend_title='Recorded outcome')
            st.plotly_chart(style(fl,365),width='stretch')
            st.caption('Important missing-data caveat: “No ET/penalty flagged” does NOT prove a regulation-time decision. Some historical records use flat scores without reliable extra-time metadata; consult knockout_metadata_coverage.csv before comparing eras.')

            st.subheader('Did the 48-team tournament cause more goals?')
            form = adv['format_2022_2026.csv'].copy()
            cmp1,cmp2=st.columns(2)
            with cmp1:
                f=px.bar(form,x=form['year'].astype(str),y='mean_goals',text_auto='.2f',
                         title='Goals per match: 2022 vs 2026',color_discrete_sequence=[GOLD])
                f.update_layout(xaxis_title='World Cup year',yaxis_title='Average match goals')
                st.plotly_chart(style(f,330),width='stretch')
            with cmp2:
                f=px.bar(form,x=form['year'].astype(str),y='mean_elo_gap',text_auto='.1f',
                         title='Average pre-match Elo gap',color_discrete_sequence=[CYAN])
                f.update_layout(xaxis_title='World Cup year',yaxis_title='Mean absolute Elo gap')
                st.plotly_chart(style(f,330),width='stretch')
            sample = adv['advanced_metrics.json']['format_exploration']
            note(f"<b>Exploratory result:</b> average goals increased by {sample['goals']['difference_2026_minus_2022']:+.2f} "
                 f"per match; "
                 f"the average Elo gap changed by {sample['elo_gap']['difference_2026_minus_2022']:+.1f} points "
                 f". "
                 'This dataset does not support a simple claim that wider Elo mismatches caused the goal change. ' 
                 'The old row-permutation test treats fixtures as independent and is not used for inferential claims. Two tournaments cannot identify the causal effect of expansion.')

            st.subheader('Team archetypes — exploratory K-means')
            clusters = adv['team_archetypes.csv'].copy()
            archetypes=adv['archetype_centroids.csv'].copy()
            archetypes['Group']=archetypes['archetype']
            ac1,ac2=st.columns([1.45,1])
            with ac1:
                f=px.scatter(clusters,x='attack',y='defence_conceded',color='archetype',size='matches',
                             hover_name='team',hover_data=['elo','draw_rate'],
                             color_discrete_sequence=[CYAN,GOLD,PINK],
                             title='Team profiles: goals scored vs goals conceded')
                f.update_layout(xaxis_title='Goals scored per match',yaxis_title='Goals conceded per match',legend_title='Cluster')
                st.plotly_chart(style(f,400),width='stretch')
            with ac2:
                archetypes['Mean goals for']=archetypes['attack'].round(2)
                archetypes['Mean goals against']=archetypes['defence_conceded'].round(2)
                archetypes['Mean Elo']=archetypes['elo'].round(0)
                archetypes['Mean draw rate']=archetypes['draw_rate'].map(lambda x:f'{x:.0%}')
                dark_table(archetypes[['Group','Mean goals for','Mean goals against','Mean Elo','Mean draw rate']],
                           title='Cluster centroids (not tactical labels)')
            st.caption('K=3, standardized attack, defence (goals conceded), World Cup-only Elo and draw rate; teams must have at least five matches from 2002–2026. Groups are descriptive statistical similarities, not measured formations or proven styles.')
            for cluster, df_group in clusters.groupby('archetype'):
                examples=', '.join(str(t) for t in df_group.sort_values('matches',ascending=False)['team'].head(6))
                st.markdown(f'**{cluster}:** {examples}')

    with st.expander('Clean sheets, blowouts & 0–0 draws', expanded=False):
        # ---------------- v7: defensive / blowout trends ----------------
        st.markdown("<div class='section-chip' style='margin-top:1.6rem'>Defensive resilience • score margins</div>", unsafe_allow_html=True)
        st.subheader('Clean-sheet and blowout trends')
        st.caption('Based on the cleaned, ML-eligible World Cup matches since 2002. A clean-sheet match has at least one team scoring 0; a blowout is a regulation-time winning margin of at least 3 goals. The former includes 0–0 draws.')
        recent_scores = e[e['year'].between(2002, 2026)].copy()
        recent_scores['clean_sheet_match'] = (recent_scores['goals_1'] == 0) | (recent_scores['goals_2'] == 0)
        recent_scores['blowout_match'] = (recent_scores['goals_1'] - recent_scores['goals_2']).abs() >= 3
        recent_scores['zero_zero'] = (recent_scores['goals_1'] == 0) & (recent_scores['goals_2'] == 0)
        yearly_trends = recent_scores.groupby('year', as_index=False).agg(
            matches=('year','size'),
            clean_sheet_rate=('clean_sheet_match','mean'),
            blowout_rate=('blowout_match','mean'),
            scoreless_draw_rate=('zero_zero','mean'),
            clean_sheet_matches=('clean_sheet_match','sum'),
            blowout_matches=('blowout_match','sum'),
        )
        clean_left, clean_right = st.columns([1.2, 1], gap='large')
        with clean_left:
            with st.container(border=True):
                melted = yearly_trends.melt('year', value_vars=['clean_sheet_rate','blowout_rate','scoreless_draw_rate'],
                                            var_name='Indicator',value_name='Match share')
                label_map = {'clean_sheet_rate':'At least one clean sheet', 'blowout_rate':'3+ goal margin',
                             'scoreless_draw_rate':'0–0 draw'}
                melted['Indicator'] = melted['Indicator'].map(label_map)
                fig = px.line(melted,x='year',y='Match share',color='Indicator',markers=True,
                              color_discrete_sequence=[CYAN,GOLD,PINK],title='Share of matches by tournament')
                fig.update_layout(yaxis_tickformat='.0%',xaxis_title='World Cup year',
                                  legend=dict(orientation='h',y=-.22,x=0,font=dict(size=11)))
                st.plotly_chart(style(fig,390),width='stretch')
        with clean_right:
            with st.container(border=True):
                counts=yearly_trends.melt('year',value_vars=['clean_sheet_matches','blowout_matches'],
                                         var_name='Type',value_name='Matches')
                counts['Type']=counts['Type'].map({'clean_sheet_matches':'Clean-sheet matches',
                                                   'blowout_matches':'3+ goal-margin matches'})
                fig=px.bar(counts,x='year',y='Matches',color='Type',barmode='group',
                           color_discrete_sequence=[CYAN,GOLD],title='Counts (different tournament sizes)')
                fig.update_layout(legend=dict(orientation='h',y=-.22,x=0,font=dict(size=11)))
                st.plotly_chart(style(fig,390),width='stretch')
        recent_overview = yearly_trends[['year','matches','clean_sheet_rate','blowout_rate']].copy()
        recent_overview['clean_sheet_rate']=(recent_overview['clean_sheet_rate']*100).round(1).astype(str)+'%'
        recent_overview['blowout_rate']=(recent_overview['blowout_rate']*100).round(1).astype(str)+'%'
        recent_overview.columns=['Year','Matches','At least one clean sheet','3+ goal-margin matches']
        dark_table(recent_overview, title='Defensive trends — exact tournament percentages')
        st.caption('Rates are more comparable than raw counts when tournament sizes change. These are descriptive trends and do not establish the cause of a change.')

with t_insights:
    with st.expander('Method & limits — read before drawing conclusions'):
        st.markdown('''
- **Sample dependence:** Teams play multiple matches. Per-match tests and standard confidence intervals can be optimistic. Team-cluster sensitivity estimates are in Model Report; an independent tournament-cluster analysis is impossible with a single 2026 holdout.
- **Predictions are retrospective:** Form and Elo update during a tournament. The selection dropdown uses end-of-2026 state, which is not a pre-2026 forecast.
- **Knockout interpretation:** All probabilistic scores predict 90-minute win, draw or loss, not the team advancing after extra time and penalties.
- **Missing variables:** No coach lineups, injuries, travel, host advantage, match venue or official external pre-match ratings are incorporated into the trained predictor. Historical squad and stadium files are for optional exploratory descriptions, not validated predictive features.
- **No bracket Monte Carlo:** A meaningful title-odds simulator needs an explicit bracket, date-specific ratings and a validated extra-time/shootout mechanism. Generating odds from end-of-tournament ratings would be misleading.
- **Debutants & experience:** Matches-played history is not a causal strength variable. Held-out permutation importance and experience-ablation results appear under Model Report.
- **Repeated match metadata:** Older tournaments can have uncertain shootout and extra-time score formats; do not infer absent ET flags mean a 90-minute decision.
''')
