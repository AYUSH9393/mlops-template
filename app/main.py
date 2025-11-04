from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, Ayush! Welcome to your MLOps journey 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}
