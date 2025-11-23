import time

import torch
from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from app.logger import get_logger
from app.metrics import INFERENCE_LATENCY
from app.middleware import metrics_middleware
from app.ml_model import LogisticRegressionModel

app = FastAPI()

# Register metrics middleware
app.middleware("http")(metrics_middleware)

# Load PyTorch model
model = LogisticRegressionModel(3)
model.load_state_dict(torch.load("model.pth"))
model.eval()

logger = get_logger("ml-api")


# -----------------------------
# Input Schema
# -----------------------------
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    logger.info("API started successfully")
    return {"message": "PyTorch model FastAPI integration successful"}


@app.post("/predict")
def predict(data: InputData, threshold: float = 0.5):
    # Convert to tensor
    inputs = torch.tensor([[data.feature1, data.feature2, data.feature3]])

    # -----------------------------
    # Measure inference latency
    # -----------------------------
    with torch.no_grad():
        start = time.time()
        with INFERENCE_LATENCY.time():  # Prometheus histogram
            prob = model(inputs).item()
        latency = round(time.time() - start, 4)

    logger.info(f"Inference latency: {latency}s")

    # Count request
    # REQUEST_COUNT.inc()

    # Generate classification
    prediction = 1 if prob >= threshold else 0

    return {
        "threshold_used": threshold,
        "probability": round(prob, 4),
        "prediction": prediction,
        "latency_seconds": latency,
    }


@app.get("/version")
def version():
    return {"model_version": "1.0.0", "framework": "PyTorch", "author": "Ayush Patel"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/live")
def live():
    return {"status": "alive"}


@app.get("/ready")
def ready():
    try:
        _ = model
        return {"status": "ready"}
    except Exception:
        return {"status": "not ready"}, 503


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# @app.get("/square/{num}")
# def square(num: int):
#     return {"number": num, "square": num ** 2}

# @app.get("/greet/")
# def greet(name: str = "Ayush"):
#     return {"greeting": f"Hello, {name}! 👋"}
