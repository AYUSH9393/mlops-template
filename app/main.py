from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, Ayush! Welcome to your MLOps journey 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/square/{num}")
def square(num: int):
    return {"number": num, "square": num ** 2}

@app.get("/greet/")
def greet(name: str = "Ayush"):
    return {"greeting": f"Hello, {name}! 👋"}

