"""Local curated examples; extractive retrieval, not an LLM or production RAG."""
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS = [
 ('D01','Claim denial workflow','A claim denial review begins with identifying the remittance reason, checking authorization requirements, validating eligibility and coverage dates, and examining coding completeness. Escalate uncertain or conflicting records for human review.'),
 ('D02','RAG architecture','A retrieval-augmented generation pipeline ingests source documents, chunks them, generates embeddings, retrieves candidate passages, reranks for relevance, and passes grounded context to a language model. Track retrieval recall, faithfulness and citations.'),
 ('D03','Model monitoring','Track input feature distributions, missingness, prediction score distributions, latency, and delayed ground-truth performance. Population Stability Index is a heuristic for distribution shift, not proof of model degradation.'),
 ('D04','MLOps release workflow','Register versioned models and datasets. Evaluate on held-out examples, run bias and regression checks, canary-deploy, monitor drift and errors, and define rollback conditions. Separate reproducible training from serving.'),
 ('D05','LLM evaluation','LLM-as-a-judge can help evaluate relevance and formatting, but may have verbosity bias or shared blind spots. Compare judge ratings with expert annotations and deterministic checks; agreement does not establish truth.'),
 ('D06','Data governance','Never put protected health information or credentials into unapproved public AI services. Apply least-privilege access, audit logging, minimization and retention policies. Synthetic data should be clearly labeled.'),
 ('D07','Industrial anomaly detection','Predictive maintenance combines timestamped sensor streams, engineered rolling-window features, anomaly scores, and technician feedback. Validation should separate equipment and time periods to prevent leakage.'),
 ('D08','Classification metrics','For rare outcomes, precision, recall, PR-AUC and operating-threshold costs can be more informative than accuracy. Select a threshold using review capacity and the relative costs of false positives and false negatives.'),
]
@lru_cache(maxsize=1)
def index():
    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2))
    matrix=vectorizer.fit_transform([title+' '+text for _,title,text in DOCS])
    return vectorizer,matrix

def answer(question):
    vectorizer,matrix=index()
    sim=cosine_similarity(vectorizer.transform([question]),matrix)[0]
    hits=sorted(range(len(DOCS)),key=lambda i:sim[i],reverse=True)[:3]
    if not hits or sim[hits[0]]<.035:
        return {'answer':'I could not find enough supporting information in the local example knowledge base. Try asking about claim denials, RAG, monitoring, or MLOps.','sources':[],'grounded':False}
    sources=[{'id':DOCS[i][0],'title':DOCS[i][1],'excerpt':DOCS[i][2],'similarity':round(float(sim[i]),3)} for i in hits if sim[i]>=.035]
    return {'answer':sources[0]['excerpt'],'sources':sources,'grounded':True,'note':'Extractive answer from a small local demonstration corpus; no generative model is called.'}
