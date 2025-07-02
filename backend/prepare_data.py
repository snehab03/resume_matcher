import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import os

# 1️⃣ Load your dataset
input_csv = "Jobs.csv"  # adjust if your filename is different
df = pd.read_csv(input_csv)

# 2️⃣ Display columns to help you verify
print("Columns found:", df.columns)

# 3️⃣ Clean and combine columns
# If your dataset uses different column names, adjust here
df["Full_Text"] = df["Job Title"].fillna("") + ". " + df["Job Description"].fillna("")

# 4️⃣ Remove rows with empty text
df = df[df["Full_Text"].str.strip().astype(bool)]

# 5️⃣ Save cleaned CSV
df[["Job Title", "Job Description", "Full_Text"]].to_csv("jobs.csv", index=False)
print("✅ Cleaned jobs.csv saved.")

# 6️⃣ Load model
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)
print(f"✅ Model '{model_name}' loaded.")

# 7️⃣ Encode embeddings
print("⏳ Encoding embeddings...")
embeddings = model.encode(df["Full_Text"].tolist(), convert_to_tensor=True, show_progress_bar=True)

# 8️⃣ Save embeddings
torch.save(embeddings, "job_embeddings.pt")
print("✅ job_embeddings.pt saved.")

# 9️⃣ Save model
if not os.path.exists("model"):
    os.makedirs("model")
model.save("model")
print("✅ Model saved to ./model/")
