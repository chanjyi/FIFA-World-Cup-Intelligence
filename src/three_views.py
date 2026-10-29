"""Interactive, illustrative historical Starting XI pitch powered by Three.js.

JavaScript runs in a sandboxed Streamlit iframe and player names are escaped
before embedding. No real historical formation or pitch positions are implied.
"""
from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

ASSETS = Path(__file__).resolve().parents[1] / 'assets'


def _embed(template: str, payload: dict, height: int) -> None:
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    page = (ASSETS / template).read_text(encoding='utf-8').replace('__VIEW_DATA__', data)
    if hasattr(st, "iframe"):
        st.iframe(page, height=height, alt="Interactive Three.js football visualization")
    else:
        components.html(page, height=height, scrolling=False)


def show_pitch_3d(players, team: str, formation: str, colour: str, height: int = 490) -> None:
    roster = [
        {'order': int(row['order']), 'name': str(row['player']), 'captain': str(row['captain']).lower() in ('true', '1', 'yes')}
        for _, row in players.sort_values('order').head(11).iterrows()
    ]
    _embed('lineup_3d.html', {'team':team, 'formation':formation, 'colour':colour, 'players':roster}, height)
