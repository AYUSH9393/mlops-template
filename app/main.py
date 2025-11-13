from fastapi import FastAPI
from pydantic import BaseModel
import torch
from app.ml_model import LogisticRegressionModel

app = FastAPI()

#load model
model = LogisticRegressionModel(2)
model.load_state_dict(torch.load("model.pth"))
model.eval()

class InputData(BaseModel):
    feature1: float
    feature2: float

@app.get("/")
def home():
    return {"message": "PyTorch model FastAPI integration successful"}

@app.post("/predict")
def predict(data: InputData):
    # Convert input to tensor
    inputs = torch.tensor([[data.feature1, data.feature2]])
    with torch.no_grad():
        output = model(inputs)
        prediction = (output.item() > 0.5)
    return {"prediction": int(prediction), "probability": round(output.item(), 3)}

# @app.get("/")
# def read_root():
#     return {"message": "Hello, Ayush! Welcome to your MLOps journey 🚀"}

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# @app.get("/square/{num}")
# def square(num: int):
#     return {"number": num, "square": num ** 2}

# @app.get("/greet/")
# def greet(name: str = "Ayush"):
#     return {"greeting": f"Hello, {name}! 👋"}

