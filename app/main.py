from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.ai_engine import CrisisEngine

app = FastAPI()
# Allow Frontend to talk to Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#Initialise the AI Engine
engine = CrisisEngine()

@app.get("/")
def home():
    return {"status": "online", "message": "Crisis Flow API is running !"}

@app.get("/data")
def get_dashboard_data():
    return engine.get_dashboard_data()

