"""Reproducible holdout audit; makes NO retraining decisions based on the 2026 test.
Run after train_model.py: python audit_model.py
"""
import json
from pathlib import Path
from math import comb
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parent
P=ROOT/'data/processed'; R=ROOT/'data/reports'; M=ROOT/'models'
R.mkdir(parents=True,exist_ok=True)
f=pd.read_csv(P/'prematch_features.csv',parse_dates=['date'])
bt=pd.read_csv(M/'worldcup_2026_backtest.csv',parse_dates=['date'])
f['ml_eligible']=f['ml_eligible'].astype(str).str.lower().eq('true')
test=f[(f.year==2026)&f.ml_eligible].copy().sort_values(['date','match_key'])
train=f[(f.year<=2022)&(f.year>=2002)&f.ml_eligible].copy()
train_old=f[(f.year<2022)&(f.year>=2002)&f.ml_eligible].copy()
val=f[(f.year==2022)&f.ml_eligible].copy()
merged=bt.merge(test[['date','team_1','team_2','match_key','elo_1','elo_2','team_1_prev_matches','team_2_prev_matches']],on=['date','team_1','team_2'],how='left',validate='one_to_one')
assert len(merged)==len(bt)==104
labels=['draw','team_1_win','team_2_win']
p_model=merged[['prob_'+v for v in labels]].to_numpy()
y=merged['result_90'].to_numpy()
yidx=np.array([labels.index(v) for v in y]); n=len(y)
model_preds=merged['predicted_result'].to_numpy()
elo_preds=np.where(merged['elo_1']>=merged['elo_2'],'team_1_win','team_2_win')
# Unlike the majority-class baseline, this is a strength benchmark, and ties go to first-listed.
elo_correct=(elo_preds==y); model_correct=(model_preds==y)
# More meaningful probabilistic baseline: logistic trained ONLY through 2022 on Elo columns.
elo_features=['elo_1','elo_2','elo_diff','elo_expected_1']
elo_logit=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler()),('model',LogisticRegression(max_iter=2500))])
elo_logit.fit(train[elo_features],train['result_90'])
p_elo=elo_logit.predict_proba(test[elo_features])
# Align to saved test order using merge key; not rely on coincidentally identical CSV row sequence.
raw_p=pd.DataFrame(p_elo,columns=elo_logit.classes_)
raw_p['match_key']=test['match_key'].to_numpy()
p_elo=merged[['match_key']].merge(raw_p,on='match_key',validate='one_to_one')[labels].to_numpy()

def brier(probs):
    target=np.zeros_like(probs);target[np.arange(n),yidx]=1
    return float(((probs-target)**2).sum(axis=1).mean())

def exact_mcnemar(a,b):
    only_a=int((a & ~b).sum());only_b=int((~a & b).sum()); d=only_a+only_b
    p=min(1.,2*sum(comb(d,k) for k in range(min(only_a,only_b)+1))/2**d) if d else 1.
    return {'model_only_correct':only_a,'baseline_only_correct':only_b,'discordant_matches':d,'p_value':float(p)}

# Paired TEAM-cluster multiplier bootstrap: both teams' weights jointly influence each match.
# One year -> cannot perform a useful tournament-level cluster bootstrap.
rng=np.random.default_rng(2610)
teams=np.array(sorted(set(merged.team_1)|set(merged.team_2)))
index={t:i for i,t in enumerate(teams)}
ix1=np.array([index[t] for t in merged.team_1]);ix2=np.array([index[t] for t in merged.team_2])
delta_acc=model_correct.astype(float)-elo_correct.astype(float)
loss_model=-np.log(np.clip(p_model[np.arange(n),yidx],1e-12,1.))
loss_elo=-np.log(np.clip(p_elo[np.arange(n),yidx],1e-12,1.))
delta_loss=loss_elo-loss_model  # positive -> model beats Elo-only logistic
boot_acc=[];boot_log=[]
for _ in range(3000):
    w=rng.exponential(size=len(teams)); weights=(w[ix1]+w[ix2])/2
    boot_acc.append(float(np.average(delta_acc,weights=weights)))
    boot_log.append(float(np.average(delta_loss,weights=weights)))

# Evaluate correlated importance via held-out 2022 permutation, not training-set Gini importance.
bundle=joblib.load(M/'match_outcome_model.joblib')
estimator=clone(bundle['model']);features=bundle['features']
estimator.fit(train_old[features],train_old['result_90'])
pi=permutation_importance(estimator,val[features],val['result_90'],scoring='neg_log_loss',n_repeats=12,random_state=2601,n_jobs=1)
importance=pd.DataFrame({'feature':features,'mean_log_loss_increase':pi.importances_mean,'std':pi.importances_std})
importance.sort_values('mean_log_loss_increase',ascending=False).to_csv(R/'permutation_importance_2022.csv',index=False)

# Model dependence on experience: train 2022 validation without previous match counts and compare.
without_experience=[x for x in features if x not in ('team_1_prev_matches','team_2_prev_matches')]
reduced=clone(bundle['model']).fit(train_old[without_experience],train_old['result_90'])
def scoreval(est,columns):
    p=est.predict_proba(val[columns]); return {'accuracy':float(accuracy_score(val.result_90,est.predict(val[columns]))),
                                               'log_loss':float(log_loss(val.result_90,p,labels=est.classes_))}
experience_2022={'full_model':scoreval(estimator,features),'without_experience':scoreval(reduced,without_experience),'n':len(val)}

# Stage-wise uncertainty: Wilson intervals, no claim that stages differ significantly.
def wilson(k,n,z=1.96):
    if not n:return [None,None]
    p=k/n; denom=1+z*z/n;mid=(p+z*z/(2*n))/denom
    plus=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [float(mid-plus),float(mid+plus)]
stage=[]
for s,g in merged.groupby('stage'):
    hits=int((g.predicted_result==g.result_90).sum())
    stage.append({'stage':s,'n':int(len(g)),'correct':hits,'accuracy':hits/len(g),'ci_low':wilson(hits,len(g))[0],'ci_high':wilson(hits,len(g))[1]})
pd.DataFrame(stage).to_csv(R/'stage_accuracy_intervals.csv',index=False)

# one-vs-rest calibration bins with n and Wilson bars
cal=[]
for j,label in enumerate(labels):
    for lo,hi in [(0,.15),(.15,.30),(.30,.45),(.45,.60),(.60,1.01)]:
        mask=(p_model[:,j]>=lo)&(p_model[:,j]<hi)
        count=int(mask.sum())
        if not count:continue
        success=int((y[mask]==label).sum()); ci=wilson(success,count)
        cal.append({'class':label,'bin':f'{lo:.2f}–{min(hi,1):.2f}','n':count,
                    'predicted':float(p_model[mask,j].mean()),'observed':success/count,
                    'ci_low':ci[0],'ci_high':ci[1]})
pd.DataFrame(cal).to_csv(R/'calibration_with_intervals.csv',index=False)

# Baseline score table, bracket comparison and uncertainty
rows=[
 {'model':'Random Forest + Elo','accuracy':float(model_correct.mean()),'log_loss':float(log_loss(y,p_model,labels=labels)),'brier_multiclass':brier(p_model)},
 {'model':'Higher-Elo pick','accuracy':float(elo_correct.mean()),'log_loss':None,'brier_multiclass':None},
 {'model':'Elo-only logistic','accuracy':float((np.asarray(labels)[p_elo.argmax(axis=1)]==y).mean()),'log_loss':float(log_loss(y,p_elo,labels=labels)),'brier_multiclass':brier(p_elo)},
]
pd.DataFrame(rows).to_csv(R/'fair_baselines_2026.csv',index=False)

summary={
 'n_test':n,'model_accuracy':rows[0]['accuracy'],'higher_elo_accuracy':rows[1]['accuracy'],
 'elo_logistic_accuracy':rows[2]['accuracy'],'model_log_loss':rows[0]['log_loss'],
 'elo_logistic_log_loss':rows[2]['log_loss'],'model_brier':rows[0]['brier_multiclass'],
 'elo_logistic_brier':rows[2]['brier_multiclass'],
 'mcnemar_higher_elo':exact_mcnemar(model_correct,elo_correct),
 'team_cluster_bootstrap_accuracy_delta_ci':list(map(float,np.percentile(boot_acc,[2.5,97.5]))),
 'team_cluster_bootstrap_logloss_improvement_ci':list(map(float,np.percentile(boot_log,[2.5,97.5]))),
 'accuracy_delta_model_minus_higher_elo':float(delta_acc.mean()),
 'logloss_improvement_over_elo_logistic':float(delta_loss.mean()),
 'actual_draws':int((y=='draw').sum()),'predicted_draws':int((model_preds=='draw').sum()),
 'draw_recall':float(((model_preds=='draw')&(y=='draw')).sum()/max((y=='draw').sum(),1)),
 'experience_2022_validation':experience_2022,
 'limitations':[
     'All 2026 scores are retrospective tests, never pre-tournament odds.',
     'Team-level multiplier bootstrap is a sensitivity analysis; it is not tournament-level replication.',
     'Only one 2026 tournament was held out; estimates are not guaranteed to transfer to other tournaments.',
     'Draw recalibration was not applied without independent validation.',
 ]
}
(R/'evaluation_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
print(json.dumps(summary,indent=2))
