# import os
# import urllib.request

# @app.on_event("startup")
# def download_and_load():
#     def download_if_missing(url, filename):
#         if not os.path.exists(filename):
#             print(f"Downloading {filename}...")
#             urllib.request.urlretrieve(url, filename)
#             print(f"Downloaded {filename}.")

# # Download files if needed
# download_if_missing(
#     "https://github.com/snehab03/resume_matcher/releases/download/v1.0/job_embeddings.pt",
#     "job_embeddings.pt"
# )

# download_if_missing(
#     "https://github.com/snehab03/resume_matcher/releases/download/v1.0/Jobs.csv",
#     "Jobs.csv"
# )

# from fastapi import FastAPI, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from sentence_transformers import SentenceTransformer, util
# import torch
# import pandas as pd
# import fitz

# app = FastAPI()

# # CORS settings: allow frontend (you can restrict origins later)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Load model and embeddings once at startup
# model = SentenceTransformer("./model")
# job_embeddings = torch.load("job_embeddings.pt")
# jobs_df = pd.read_csv("jobs.csv")

# # Helper to extract text from PDF bytes
# def extract_text_from_pdf(file_bytes):
#     doc = fitz.open(stream=file_bytes, filetype="pdf")
#     return "".join(page.get_text() for page in doc)

# @app.post("/match-jobs/")
# async def match_jobs(file: UploadFile = File(...)):
#     content = await file.read()
#     resume_text = extract_text_from_pdf(content)
#     embedding = model.encode(resume_text, convert_to_tensor=True)
#     scores = util.cos_sim(embedding, job_embeddings)[0]
#     top_k = torch.topk(scores, k=5)
#     results = [
#         {
#             "title": jobs_df.iloc[idx]["Job Title"],
#             "description": jobs_df.iloc[idx]["Job Description"][:400],
#             "score": round(score.item(), 4),
#         }
#         for score, idx in zip(top_k[0], top_k[1])
#     ]
#     return {"matches": results}
import os
import urllib.request
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import fitz

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URL = "https://github.com/snehab03/resume_matcher/releases/download/v1.0/job_embeddings.pt"
CSV_URL = "https://github.com/snehab03/resume_matcher/releases/download/v1.0/Jobs.csv"

@app.on_event("startup")
def download_and_load():
    def download_if_missing(url, filename):
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename}.")

    # Download files if needed
    download_if_missing(MODEL_URL, "job_embeddings.pt")
    download_if_missing(CSV_URL, "Jobs.csv")

    # Load model and embeddings
    global model
    global job_embeddings
    global jobs_df

    model = SentenceTransformer("./model")
    job_embeddings = torch.load("job_embeddings.pt")
    jobs_df = pd.read_csv("Jobs.csv")

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
