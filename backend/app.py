from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import fitz

app = FastAPI()

# CORS settings: allow frontend (you can restrict origins later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and embeddings once at startup
model = SentenceTransformer("./model")
job_embeddings = torch.load("job_embeddings.pt")
jobs_df = pd.read_csv("jobs.csv")

# Helper to extract text from PDF bytes
def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "".join(page.get_text() for page in doc)

@app.post("/match-jobs/")
async def match_jobs(file: UploadFile = File(...)):
    content = await file.read()
    resume_text = extract_text_from_pdf(content)
    embedding = model.encode(resume_text, convert_to_tensor=True)
    scores = util.cos_sim(embedding, job_embeddings)[0]
    top_k = torch.topk(scores, k=5)
    results = [
        {
            "title": jobs_df.iloc[idx]["Job Title"],
            "description": jobs_df.iloc[idx]["Job Description"][:400],
            "score": round(score.item(), 4),
        }
        for score, idx in zip(top_k[0], top_k[1])
    ]
    return {"matches": results}
