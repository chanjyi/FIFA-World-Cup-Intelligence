"""Optional cinematic Three.js globe for the Streamlit Overview page.

Three.js is loaded from cdnjs by the viewer's browser. The local simplified
Natural Earth boundaries and Python-provided World Cup statistics are embedded
in the iframe HTML; no remote statistics API is used.
"""
import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / 'assets' / 'cinematic_globe.html'
GEO = ROOT / 'assets' / 'world_lowres.json'

# Natural Earth names can differ from the match dataset's country names.
TO_WORLD = {
    'United States': 'United States of America',
    'Democratic Republic of the Congo': 'Dem. Rep. Congo',
    'Bosnia and Herzegovina': 'Bosnia and Herz.',
    'Dominican Republic': 'Dominican Rep.',
    'Czech Republic': 'Czechia',
    'Curacao': 'Curaçao',
    'Ivory Coast': "Côte d'Ivoire",
    'South Korea': 'South Korea',
    'North Korea': 'North Korea',
}


def show_cinematic_globe(metric_df, metric: str, height: int = 600):
    features = json.loads(GEO.read_text(encoding='utf-8'))
    vals = {}
    for row in metric_df.itertuples(index=False):
        name = getattr(row, 'map_name', '')
        if not name:
            continue
        vals[TO_WORLD.get(name, name)] = {'value': float(getattr(row, 'value')), 'label': metric}
    data = json.dumps({'features': features, 'values': vals, 'metric': metric}, ensure_ascii=False).replace('</', '<\\/')
    # These are shipped with the project, not user-supplied HTML.
    page = TEMPLATE.read_text(encoding='utf-8').replace('__GLOBE_DATA__', data)
    if hasattr(st, "iframe"):
        st.iframe(page, height=height, alt="Interactive 3D FIFA World Cup globe")
    else:
        components.html(page, height=height, scrolling=False)
