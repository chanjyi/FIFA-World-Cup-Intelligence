"""One verification entry point, default includes an actual Streamlit AppTest.

Usage (after pip install -r requirements.txt):
    python verify_project.py
Offline/CI static-data validation without Streamlit (NOT a full app test):
    python verify_project.py --data-only
"""
from __future__ import annotations
import argparse
import ast
import json
import sys
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

import pandas as pd

ROOT=Path(__file__).resolve().parent

def check_data_and_static():
    errors=[]
    app=(ROOT/'app.py').read_text(encoding='utf8')
    req=(ROOT/'requirements.txt').read_text(encoding='utf8')
    if 'streamlit==1.66.1' not in req:errors.append('Streamlit pin mismatch')
    if 'pandas==2.3.3' not in req:errors.append('Pandas pin mismatch')
    if 'np.issubdtype(show[c]' in app:errors.append('Obsolete datetime detection')
    if 'pd.api.types.is_datetime64_any_dtype(show[c])' not in app:errors.append('Pandas datetime compatibility code missing')
    for old in ('use_container_width=', "REP / 'calibration.csv'", "REP/'calibration.csv'"):
        if old in app:errors.append(f'Legacy call in app: {old}')
    # AST rejects missing/incorrect width in ANY Plotly chart call, catching regressions
    tree=ast.parse(app)
    charts=[]
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call):continue
        f=node.func
        if isinstance(f,ast.Attribute) and f.attr=='plotly_chart':
            charts.append(node)
            width=[kw.value for kw in node.keywords if kw.arg=='width']
            if len(width)!=1 or not isinstance(width[0],ast.Constant) or width[0].value!='stretch':
                errors.append(f'Plotly chart line {node.lineno} missing width="stretch"')
            if any(kw.arg=='use_container_width' for kw in node.keywords):
                errors.append(f'Deprecated use_container_width at {node.lineno}')
    if len(charts)<10:errors.append('Unexpectedly few Plotly charts detected')
    for file in ('verify_v71.py','verify_v72.py','data/reports/model_test_results.csv',
                 'data/reports/calibration.csv','data/processed/team_standings_clean.csv'):
        if (ROOT/file).exists():errors.append('Stale release file exists: '+file)
    if list((ROOT/'data/raw').rglob('FIFA - *.csv')):errors.append('Unverified Kaggle raw CSVs included')
    # Submission-specific checks: dark widget theme and public-suitable docs.
    theme_path=ROOT/'.streamlit/config.toml'
    if not theme_path.exists():
        errors.append('Missing native dark theme config')
    else:
        import tomllib
        theme=tomllib.loads(theme_path.read_text(encoding='utf8')).get('theme',{})
        if theme.get('base') != 'dark':errors.append('Selectbox theme must have dark base')
        if theme.get('secondaryBackgroundColor','').lower() != '#0e1d3d':
            errors.append('Selectbox theme secondary background mismatch')
    if '[data-testid="stSelectbox"]' not in app:
        errors.append('Dark selectbox CSS missing')
    for doc in ('PROJECT_DESCRIPTION.md','README.md','DATA_LICENSE_NOTICE.md'):
        if not (ROOT/doc).exists():errors.append('Submission document missing: '+doc)
    if (ROOT/'DEMO_SCRIPT.md').exists():errors.append('Obsolete demo script still included')
    for f in ('LICENSE','DATA_LICENSE_NOTICE.md','data/raw/openfootball/LICENSE.md'):
        if not (ROOT/f).exists():errors.append('Missing licensing/provenance: '+f)
    report=json.loads((ROOT/'data/reports/evaluation_audit.json').read_text())
    metrics=json.loads((ROOT/'data/reports/model_metrics.json').read_text())
    if report['n_test'] != 104:errors.append('Unexpected test count')
    if abs(report['model_log_loss']-metrics['log_loss'])>1e-8:errors.append('Model log-loss inconsistency')
    if abs(report['model_accuracy']-metrics['accuracy'])>1e-8:errors.append('Model accuracy inconsistency')
    if abs(report['higher_elo_accuracy']-report['model_accuracy'])>1e-8:errors.append('Higher-Elo comparison changed')
    summ=pd.read_csv(ROOT/'data/processed/tournament_summary_clean.csv')
    if len(summ)!=23:errors.append('Expected 23 OpenFootball tournament summaries')
    row=summ[summ.year==2026]
    if len(row)!=1 or row.iloc[0]['champion']!='Spain':errors.append('2026 champion not Spain')
    if summ['champion'].replace({'West Germany':'Germany'}).nunique()!=8:errors.append('Wrong champion count')
    if not (ROOT/'data/reports/calibration_with_intervals.csv').exists():errors.append('Current calibration report missing')
    if errors:raise AssertionError('\n'.join(errors))
    print(f'PASS: static API + dependency + licensing + model data checks ({len(charts)} Plotly calls).')

def check_app():
    try:
        import streamlit as st
        from streamlit.testing.v1 import AppTest
    except ImportError as exc:
        raise RuntimeError('A real AppTest requires installed Streamlit: run pip install -r requirements.txt') from exc
    if st.__version__!='1.66.1':
        raise AssertionError(f'Expected pinned Streamlit 1.66.1, found {st.__version__}')
    app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=120).run()
    def assert_clean(at, route):
        issues=[]
        for key in ('exception','error','warning'):
            entries=list(getattr(at,key))
            if entries:issues.append(f'{key} on {route}: '+ '; '.join(str(x.message)[:180] for x in entries))
        if issues:raise AssertionError('\n'.join(issues))
    assert_clean(app,'default cinematic route')
    # Switch to 2D Plotly globe to execute that code path too.
    for rb in app.radio:
        if rb.label=='Globe view':
            rb.set_value('Interactive data globe').run()
            assert_clean(app,'Plotly globe route')
            break
    else:
        raise AssertionError('Globe view radio control not found')
    print('PASS: Streamlit AppTest executes both globe routes without warning/error/exception elements.')
    print('NOTE: Browser-only/Three.js/CSS visual rendering requires a separate manual test.')

def main():
    parser=argparse.ArgumentParser(description='Project verification')
    parser.add_argument('--data-only',action='store_true',help='Run static/data checks only, not the Streamlit app')
    args=parser.parse_args()
    check_data_and_static()
    if not args.data_only:check_app()
    print('Verification complete.')

if __name__=='__main__':main()
