from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import numpy as np
from .ml import score, get_model
from .knowledge import answer

ROOT=Path(__file__).resolve().parents[1]
app=FastAPI(title='Thanmmay AI Lab',version='1.0.0',description='Synthetic AI/ML portfolio demonstrations')
app.mount('/static',StaticFiles(directory=ROOT/'static'),name='static')

class Claim(BaseModel):
    charge:float=Field(2450,ge=100,le=25000)
    prior_denials:int=Field(1,ge=0,le=7)
    authorization:int=Field(1,ge=0,le=1)
    documentation:float=Field(.82,ge=0,le=1)
    coding_complexity:int=Field(3,ge=1,le=5)
    payer_friction:float=Field(.35,ge=0,le=1)
    days_since_visit:int=Field(14,ge=0,le=100)
class Ask(BaseModel):
    question:str=Field(min_length=3,max_length=1000)

@app.get('/')
def home():return FileResponse(ROOT/'static'/'index.html')
@app.get('/api/health')
def health():return {'status':'ok','mode':'synthetic-demo'}
@app.get('/api/model')
def model():return get_model()[1]
@app.post('/api/predict')
def predict(claim:Claim):return score(claim.model_dump())
@app.post('/api/ask')
def ask(q:Ask):return answer(q.question)

# Histogram-based PSI calculated against fixed simulated baseline and candidate samples.
def psi(a,b,bins=10):
    edges=np.quantile(a,np.linspace(0,1,bins+1)); edges=np.unique(edges)
    if len(edges)<3:return 0.0
    edges[0]=-np.inf;edges[-1]=np.inf
    pa=np.histogram(a,edges)[0]/len(a)
    pb=np.histogram(b,edges)[0]/len(b)
    pa=np.maximum(pa,1e-6);pb=np.maximum(pb,1e-6)
    return float(np.sum((pb-pa)*np.log(pb/pa)))
@app.get('/api/monitor')
def monitor(shift:float=0):
    if not 0<=shift<=100:raise HTTPException(422,'shift must be 0–100')
    rng=np.random.default_rng(19)
    baseline=rng.normal(.43,.15,5000).clip(0,1)
    candidate=rng.normal(.43+shift/260,.15,5000).clip(0,1)
    val=psi(baseline,candidate)
    hist_a,edges=np.histogram(baseline,bins=np.linspace(0,1,11));hist_b,_=np.histogram(candidate,bins=edges)
    return {'psi':round(val,3),'status':'High shift' if val>=.25 else 'Investigate' if val>=.1 else 'Stable', 'baseline':hist_a.tolist(),'current':hist_b.tolist(),'labels':[f'{i/10:.1f}–{(i+1)/10:.1f}' for i in range(10)],'notes':'Synthetic reference and current samples. PSI thresholds are heuristics, not diagnostic rules.'}
