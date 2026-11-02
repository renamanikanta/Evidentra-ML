"""Synthetic-data demonstration model. Not a clinical or coverage decision system."""
from functools import lru_cache
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

FEATURES = ['charge','prior_denials','authorization','documentation','coding_complexity','payer_friction','days_since_visit']

@lru_cache(maxsize=1)
def get_model():
    rng=np.random.default_rng(42)
    n=18000
    charge=rng.lognormal(7.6,.72,n).clip(100,25000)
    prior=rng.poisson(.9,n).clip(0,7)
    authorization=rng.binomial(1,.72,n)
    documentation=rng.uniform(0,1,n)
    complexity=rng.integers(1,6,n)
    friction=rng.uniform(0,1,n)
    days=rng.integers(0,100,n)
    X=np.column_stack([charge,prior,authorization,documentation,complexity,friction,days])
    logit=-3.25 + .000045*charge + .34*prior - 1.45*authorization - 1.4*documentation + .36*complexity + 1.55*friction + .008*days
    p=1/(1+np.exp(-logit))
    y=rng.binomial(1,p)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
    model=HistGradientBoostingClassifier(max_iter=105,max_leaf_nodes=12,learning_rate=.075,l2_regularization=3,random_state=42)
    model.fit(Xtr,ytr)
    preds=model.predict_proba(Xte)[:,1]
    return model,{'roc_auc':round(float(roc_auc_score(yte,preds)),3),'pr_auc':round(float(average_precision_score(yte,preds)),3),'n_samples':n,'positive_rate':round(float(y.mean()),3)}

def score(payload):
    model,metrics=get_model()
    row=np.array([[payload[k] for k in FEATURES]],dtype=float)
    risk=float(model.predict_proba(row)[0,1])
    explanations=[]
    if not payload['authorization']: explanations.append('Prior authorization is missing')
    if payload['documentation']<.5: explanations.append('Documentation completeness is low')
    if payload['prior_denials']>=2: explanations.append('Elevated prior-denial history')
    if payload['coding_complexity']>=4: explanations.append('Higher coding complexity')
    if payload['payer_friction']>.65: explanations.append('Higher simulated payer friction')
    if not explanations: explanations.append('No prominent input-level risk flags')
    return {'risk':round(risk,4),'band':'High' if risk>=.55 else 'Moderate' if risk>=.25 else 'Low','signals':explanations,'model_metrics':metrics,'disclaimer':'Synthetic training data. Demonstration only; not a medical or reimbursement decision.'}
