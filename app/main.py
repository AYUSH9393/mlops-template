from fastapi import FastAPI
from pydantic import BaseModel
import torch
from app.ml_model import LogisticRegressionModel

app = FastAPI()

#load model
model = LogisticRegressionModel(3)
model.load_state_dict(torch.load("model.pth"))
model.eval()

class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

@app.get("/")
def home():
    return {"message": "PyTorch model FastAPI integration successful"}

@app.post("/predict")
def predict(data: InputData,threshold:float = 0.5):
    # Convert input to tensor
    inputs = torch.tensor([[data.feature1, data.feature2,data.feature3]])
    with torch.no_grad():
        prob = model(inputs).item()
    
    prediction = 1 if prob >=  threshold else 0

    return {
        "threshold_used": threshold,
        "probability": round(prob, 4),
        "prediction": prediction
    }

@app.get("/version")
def version():
    return {
        "model_version": "1.0.0",
        "framework": "PyTorch",
        "author": "Ayush Patel"
    }


# @app.get("/")
# def read_root():
#     return {"message": "Hello, Ayush! Welcome to your MLOps journey 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

# @app.get("/square/{num}")
# def square(num: int):
#     return {"number": num, "square": num ** 2}

# @app.get("/greet/")
# def greet(name: str = "Ayush"):
#     return {"greeting": f"Hello, {name}! 👋"}

