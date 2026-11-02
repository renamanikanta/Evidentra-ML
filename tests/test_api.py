from fastapi.testclient import TestClient
from app.main import app
client=TestClient(app)
def test_health():assert client.get('/api/health').json()['status']=='ok'
def test_predict():
    r=client.post('/api/predict',json={'charge':2500,'prior_denials':1,'authorization':1,'documentation':.8,'coding_complexity':3,'payer_friction':.3,'days_since_visit':12})
    assert r.status_code==200 and 0<=r.json()['risk']<=1
def test_ask():assert client.post('/api/ask',json={'question':'How can we monitor drift?'}).json()['grounded']
def test_monitor():assert client.get('/api/monitor?shift=25').status_code==200
