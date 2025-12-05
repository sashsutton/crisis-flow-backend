# CrisisFlow AI Engine (FastAPI) 🤖

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-brightgreen)
![Machine Learning](https://img.shields.io/badge/ML-Semantic%20Embeddings-orange)
![Deployment](https://img.shields.io/badge/Deploy-Docker%20|%20HF%20Spaces-yellow)

---

## 📌 Overview

The **CrisisFlow AI Engine** powers real-time disaster intelligence by transforming raw communication data into structured, meaningful insights. Using semantic embeddings, unsupervised clustering, and dimensionality reduction, CrisisFlow turns chaotic text streams into actionable data for crisis response.

The backend is built with **FastAPI** and exposes endpoints used by the CrisisFlow dashboard.

- **Frontend Repository:** [GitHub - CrisisFlow Frontend](https://github.com/sashsutton/crisis-flow-frontend)
- **Live Website:** [CrisisFlow Dashboard](https://crisis-flow-frontend-mhy4qore7-sashasuttons-projects.vercel.app/)

---

## 📁 Dataset Source

The dataset used for semantic analysis and clustering comes from the **[Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started/data)** Kaggle competition.

> [!NOTE]
> This project uses the dataset for **demonstration purposes** to show semantic clustering capabilities. It does **not** use the `target` column to classify tweets as "Real Disaster" or "Fake".

It is stored locally in the project as:

`data/train.csv`

If using your own dataset from Kaggle, ensure it has similar fields (e.g., "message", "genre", etc.).

---

## 🧠 Technologies

### Core Stack
- **Python 3.10.19**
- **FastAPI**
- **Uvicorn** (ASGI Server)

### Machine Learning & Data
- **Sentence Transformers** (`all-MiniLM-L6-v2`)
- **scikit-learn** (K-Means, PCA)
- **NumPy**
- **Pandas**

---

## 🏗️ Architecture

The AI Engine processes data in 5 stages:

### 1) Load Disaster Messages
Reads the Kaggle dataset from `data/train.csv` and loads a random sample of up to **200 rows** for processing.

```python
# From app/ai_engine.py
self.df = pd.read_csv('data/train.csv')
sample_n = min(200, len(self.df))
self.df = self.df.sample(n=sample_n, random_state=42).reset_index(drop=True)
self.df['location'] = self.df['location'].fillna('Unknown')
```

### 2) Semantic Embeddings
Converts text messages into high-dimensional vectors (384 dimensions).

**Model**: `all-MiniLM-L6-v2`

```python
# From app/ai_engine.py
self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=cache_path)
self.vectors = self.model.encode(self.df['text'].tolist(), show_progress_bar=False)
```

### 3) Unsupervised Clustering
Uses K-Means with `n_clusters=5` to group messages by **semantic similarity**.

> [!IMPORTANT]
> This is an **unsupervised** approach. The engine groups similar messages (e.g., fires, floods, medical needs) but does **not** predict whether a message is a real disaster or a false alarm.

The resulting clusters (`cluster_id`: 0 to 4) are mapped to human-readable concepts in the frontend.

### 4) Visualization (PCA)
Principal Component Analysis (PCA) is used to reduce the 384-dimensional vectors to 2-dimensional coordinates (`pca_x`, `pca_y`). This allows the frontend dashboard to plot the messages in a semantic vector space.

```python
# From app/ai_engine.py
self.kmeans = KMeans(n_clusters=5, random_state=42)
self.clusters = self.kmeans.fit_predict(self.vectors)
self.pca = PCA(n_components=2)
self.coords = self.pca.fit_transform(self.vectors)
```

### 5) API Delivery
FastAPI exposes the processed data with a RESTful endpoint:

`GET /data`

Which returns a list of dictionaries with the required fields: `id`, `text`, `location`, `cluster_id`, `pca_x`, and `pca_y`.

```python
# From app/main.py
@app.get("/data")
def get_dashboard_data():
    return engine.get_dashboard_data()
```

---

## 📦 Project Structure

```powershell
CrisisFlow-AI-Engine/
├── app/
│   ├── main.py           # FastAPI app & routes
│   └── ai_engine.py      # Embeddings, PCA, clustering
│
├── data/
│   └── train.csv         # Kaggle dataset
│
├── requirements.txt      # Dependencies
├── Dockerfile            # Deployment config
└── test.py               # Local testing script
```

---

## ▶️ Running Locally

### Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies include: `fastapi`, `uvicorn`, `pandas`, `numpy`, `scikit-learn`, `sentence-transformers`.

### Start the API

```bash
uvicorn app.main:app --reload
```

### Test endpoints
The default URL is `http://127.0.0.1:8000`.

- **Root**: `http://127.0.0.1:8000` (Returns `{"status": "online", "message": "Crisis Flow API is running !"}`)
- **Data**: `http://127.0.0.1:8000/data`

---

## 🐳 Docker Deployment (HuggingFace, Render, Railway)

### Dockerfile

```dockerfile
FROM python:3.10.19

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app
COPY ./data /code/data

RUN mkdir -p /code/model_cache && chmod 777 /code/model_cache

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
```

### Notes

The `CMD` exposes port **7860** for compatibility with cloud hosting platforms like Hugging Face Spaces.

### Start command for deployment services (Render / Railway)

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

---

## 🧪 Example Python Test Script

`test.py`

```python
print("--- Starting Test Script ---")

try:
    from app.ai_engine import CrisisEngine
    print("Import successful. Initializing Engine...")
    
    engine = CrisisEngine()
    print("Engine Initialized.")
    
    data = engine.get_dashboard_data()
    print(f"Data Retrieved. First Item: {data[0]}")

except Exception as e:
    print(f"❌ CRASHED: {e}")
```
