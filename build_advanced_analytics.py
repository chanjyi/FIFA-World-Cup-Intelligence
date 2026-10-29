"""Reproducible exploratory reports for v6.0.
Run after train_model.py: python build_advanced_analytics.py
Statistical tests are descriptive and DO NOT establish causality or generalization.
"""
from pathlib import Path
from math import comb
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import log_loss, accuracy_score
from src.worldcup_data import FEATURE_COLUMNS
from src.elo import ELO_FEATURES, canonical

ROOT=Path(__file__).resolve().parent
PROC=ROOT/'data'/'processed'; REP=ROOT/'data'/'reports'; MOD=ROOT/'models'
REP.mkdir(exist_ok=True)

match=pd.read_csv(PROC/'matches_clean.csv',parse_dates=['date'])
feat=pd.read_csv(PROC/'prematch_features.csv',parse_dates=['date'])
bt=pd.read_csv(MOD/'worldcup_2026_backtest.csv',parse_dates=['date'])
state=pd.read_csv(PROC/'team_state.csv').set_index('team')
match['ml_eligible']=match['ml_eligible'].astype(str).str.lower().eq('true')
feat['ml_eligible']=feat['ml_eligible'].astype(str).str.lower().eq('true')

# A. Fair paired 2026 test-set comparison against baseline, exact two-sided McNemar p-value.
# Baseline was fixed as majority class of historical training data through 2022 (team_1_win).
train=feat[(feat['year']<=2022) & feat['ml_eligible']]
label=train['result_90'].mode().iloc[0]
merged=bt.merge(feat[['match_key','date','team_1','team_2','team_1_prev_matches','team_2_prev_matches','elo_1','elo_2']],on=['date','team_1','team_2'],how='left',validate='one_to_one')
merged['model_correct']=merged['predicted_result'].eq(merged['result_90'])
merged['baseline_correct']=merged['result_90'].eq(label)
b=int((merged['model_correct'] & ~merged['baseline_correct']).sum())
c=int((~merged['model_correct'] & merged['baseline_correct']).sum())
n=b+c
pval=(min(1.0,2*sum(comb(n,k) for k in range(min(b,c)+1))/2**n) if n else 1.0)

# B. Same RF configuration, 2022 chronological validation: three feature sets.
abl=[]
tr=feat[(feat['year']<2022)&(feat['year']>=2002)&feat['ml_eligible']]
va=feat[(feat['year']==2022)&feat['ml_eligible']]
for name, cols in [('Form only',FEATURE_COLUMNS),('Elo only',ELO_FEATURES),('Form + Elo',FEATURE_COLUMNS+ELO_FEATURES)]:
    pipe=make_pipeline(SimpleImputer(strategy='median'),RandomForestClassifier(n_estimators=300,max_depth=5,min_samples_leaf=5,random_state=42))
    pipe.fit(tr[cols],tr['result_90'])
    proba=pipe.predict_proba(va[cols]); labels=pipe.classes_
    abl.append({'Features':name,'Accuracy':float(accuracy_score(va['result_90'],pipe.predict(va[cols]))),'Log loss':float(log_loss(va['result_90'],proba,labels=labels)),'Validation matches':int(len(va)),'Training matches':int(len(tr))})
pd.DataFrame(abl).to_csv(REP/'feature_ablation_2022.csv',index=False)

# C. Out-of-time error analysis (2026 only)
merged['abs_elo_gap']=merged['elo_diff'].abs()
merged['elo_gap_bucket']=pd.cut(merged['abs_elo_gap'],[-0.01,75,150,300,np.inf],labels=['0–75','75–150','150–300','300+'])
merged['min_team_experience']=merged[['team_1_prev_matches','team_2_prev_matches']].min(axis=1)
merged['experience_bucket']=pd.cut(merged['min_team_experience'],[-0.01,5,20,np.inf],labels=['0–5 prior','6–20 prior','21+ prior'])
for key,file in [('elo_gap_bucket','error_by_elo_gap.csv'),('experience_bucket','error_by_experience.csv'),('stage','error_by_stage.csv')]:
    out=merged.groupby(key,observed=True).agg(matches=('model_correct','size'),accuracy=('model_correct','mean'),mean_elo_gap=('abs_elo_gap','mean')).reset_index()
    out.to_csv(REP/file,index=False)
wrong=[]
for t in sorted(set(merged['team_1'])|set(merged['team_2'])):
    s=merged[(merged['team_1']==t)|(merged['team_2']==t)]
    if len(s)>=3: wrong.append({'team':t,'matches':int(len(s)),'errors':int((~s['model_correct']).sum()),'error_rate':float((~s['model_correct']).mean())})
pd.DataFrame(wrong).sort_values(['errors','error_rate'],ascending=False).to_csv(REP/'misjudged_teams_2026.csv',index=False)

# D. Elo-gap upsets from prior-match ratings only: underdog wins by regulation, non-draw 2022+2026.
# Because rating comes from World-Cup-only history, it's a surrogate, not real FIFA ranking.
pool=feat[(feat['year'].isin([2022,2026])) & feat['ml_eligible']].copy()
pool=pool[pool['result_90'].isin(['team_1_win','team_2_win'])].copy()
pool['abs_elo_gap']=pool['elo_diff'].abs()
pool['underdog_win']=np.where(pool['elo_diff']>0,pool['result_90'].eq('team_2_win'),pool['result_90'].eq('team_1_win'))
pool=pool[pool['abs_elo_gap']>=25].copy()  # very close ratings are not useful underdogs
pool['elo_gap_bucket']=pd.cut(pool['abs_elo_gap'],[25,75,150,300,np.inf],labels=['25–75','75–150','150–300','300+'],include_lowest=True)
upsets=pool.groupby(['year','elo_gap_bucket'],observed=True).agg(matches=('underdog_win','size'),upset_rate=('underdog_win','mean')).reset_index()
upsets.to_csv(REP/'upset_by_elo_gap.csv',index=False)
pool['underdog']=np.where(pool['elo_diff']>0,pool['team_2'],pool['team_1'])
pool['favorite']=np.where(pool['elo_diff']>0,pool['team_1'],pool['team_2'])
pool['elo_gap']=pool['abs_elo_gap'].round(0)
pool[pool['underdog_win']].groupby('underdog',as_index=False).agg(upset_wins=('underdog_win','sum'),mean_gap=('elo_gap','mean')).sort_values(['upset_wins','mean_gap'],ascending=False).to_csv(REP/'giant_killers.csv',index=False)
pool[pool['underdog_win']][['year','date','underdog','favorite','elo_gap','goals_1','goals_2']].sort_values('elo_gap',ascending=False).to_csv(REP/'biggest_upsets.csv',index=False)

# E. 90 minutes / extra time / penalties; missing/no-flags may mean unavailable metadata.
match['extra_time']=match['extra_time'].astype(str).str.lower().eq('true')
match['penalty_shootout']=match['penalty_shootout'].astype(str).str.lower().eq('true')
match['is_group']=match['is_group'].astype(str).str.lower().eq('true')
ko=match[(~match['is_group']) & match['year'].between(1986,2026) & match['goals_1'].notna()].copy()
ko['decision']=np.select([ko['penalty_shootout'],ko['extra_time']],['Penalties','Extra time'],default='No ET/penalty flagged')
ko.groupby(['year','decision']).size().rename('matches').reset_index().to_csv(REP/'knockout_decisions_by_year.csv',index=False)
coverage=ko.groupby('year').agg(all_knockout_matches=('decision','size'),flat_score_records=('score_format',lambda s:int(s.eq('flat_score').sum()))).reset_index()
coverage.to_csv(REP/'knockout_metadata_coverage.csv',index=False)

# F. 2022 vs 2026 descriptive comparison; Elo gap from prior-match features and goals for matching eligible fixtures.
two=feat[(feat['year'].isin([2022,2026])) & feat['ml_eligible']].copy()
two['elo_gap']=two['elo_diff'].abs(); two['goals']=two['goals_1']+two['goals_2']
two.groupby('year').agg(matches=('goals','size'),mean_goals=('goals','mean'),mean_elo_gap=('elo_gap','mean'),median_elo_gap=('elo_gap','median')).reset_index().to_csv(REP/'format_2022_2026.csv',index=False)
# Randomisation test: descriptive difference, not a causal effect of expanding field size.
rng=np.random.default_rng(2026)
perm={}
for col in ['goals','elo_gap']:
    a=two[two['year']==2022][col].to_numpy(); z=two[two['year']==2026][col].to_numpy()
    obs=float(z.mean()-a.mean()); both=np.concatenate([a,z]); count=0; B=5000
    for _ in range(B):
        order=rng.permutation(both);delta=order[len(a):].mean()-order[:len(a)].mean()
        if abs(delta)>=abs(obs)-1e-12:count+=1
    perm[col]={'difference_2026_minus_2022':obs,'permutation_p_two_sided':float((count+1)/(B+1))}

# G. descriptive Kmeans: teams 2002-2026, min 5 valid matches, goals for/against per match, regulation draw rate, current World-Cup-only Elo.
modern=match[(match['year'].between(2002,2026))&match['ml_eligible']].copy()
rows=[]
for t in sorted(set(modern['team_1'])|set(modern['team_2'])):
    s=modern[(modern['team_1']==t)|(modern['team_2']==t)]
    if len(s)<5:continue
    gfor=np.where(s['team_1']==t,s['goals_1'],s['goals_2']); gagainst=np.where(s['team_1']==t,s['goals_2'],s['goals_1'])
    elo_val=float(state.loc[t,'elo']) if t in state.index else 1500.0
    rows.append({'team':t,'matches':int(len(s)),'attack':float(gfor.mean()),'defence_conceded':float(gagainst.mean()),'elo':elo_val,'draw_rate':float((gfor==gagainst).mean())})
ar=pd.DataFrame(rows)
features=['attack','defence_conceded','elo','draw_rate']
scaled=StandardScaler().fit_transform(ar[features])
km=KMeans(n_clusters=3,random_state=42,n_init=20)
ar['cluster']=km.fit_predict(scaled)
cent=ar.groupby('cluster')[features].mean()
# cluster IDs intentionally descriptive only, no labels implying tactical causation.
labels={k: f'Statistical profile {k+1}' for k in cent.index}
ar['archetype']=ar['cluster'].map(labels)
ar.to_csv(REP/'team_archetypes.csv',index=False)
cent.reset_index().assign(archetype=lambda d:d['cluster'].map(labels)).to_csv(REP/'archetype_centroids.csv',index=False)

result={'comparison':{'baseline_label':label,'n_test':int(len(merged)),'model_accuracy':float(merged['model_correct'].mean()),'baseline_accuracy':float(merged['baseline_correct'].mean()),'model_only_correct':b,'baseline_only_correct':c,'discordant_pairs':n,'mcnemar_exact_p_two_sided':float(pval),'test':'two-sided exact McNemar'},'format_exploration':perm,'notes':['All p-values exploratory: dataset coverage and football-format changes prevent causal interpretation.','McNemar uses 2026 model predictions vs fixed training-set-majority baseline on exactly the same fixtures.','Ablation uses 2022 holdout; do not choose model using 2026 test results.','Clustering descriptive, not discovered tactical truths.']}
(REP/'advanced_metrics.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print('Wrote advanced reports'); print('McNemar exact p',pval,'ablation',abl,'format',perm)
