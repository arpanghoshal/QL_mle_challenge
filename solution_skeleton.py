from typing import List
import torch
import time
from fastapi import FastAPI, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from gmm import GMMSimple, train_model, extract_gmm_parameters

# Define data point structure
class Point(BaseModel):
    x: float
    y: float

# Define multiple points structure
class Points(BaseModel):
    coordinates: List[Point]

# Define main model class
class Model:
    def __init__(self):
        self.data = []
        self.new_data = []
        self.last_data_length = 0
        self.init_model()

    # Initialize the model and the optimizer
    def init_model(self, n_components: int = None):
        self.model = GMMSimple(n_components) if n_components else None
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01) if self.model else None

    # Train the model and extract its parameters
    def train_and_extract(self, x: torch.Tensor, epochs: int):
        train_model(self.model, self.optimizer, x, n_epochs=epochs)
        return extract_gmm_parameters(self.model, x.mean().numpy(), x.std().numpy())

    # Update data points
    def update_data(self, point: Points):
        for p in point.coordinates:
            self.new_data.append([p.x])
        self.data.extend(self.new_data)

# Create the FastAPI application and mount static files
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize model
model = Model()

# Get model
def get_model() -> Model:
    return model

# Read and return the HTML file content
def read_html_file() -> str:
    with open("./index.html") as f:
        return f.read()

# Return an HTMLResponse with the HTML file content
@app.get("/")
async def root():
    return HTMLResponse(content=read_html_file(), status_code=200)

# Adjust the number of epochs based on the number of data points
def adjust_epochs_based_on_points(n_points: int) -> int:
    if n_points < 5:
        return 200
    elif n_points < 10:
        return 150
    elif n_points < 40:
        return 100
    elif n_points < 50:
        return 50
    else:
        return 10

# Update the model with new data points, re-train it and return the updated parameters
@app.post("/update")
async def online_train(point: Points, n_components: int, epochs: int, model: Model = Depends(get_model)):
    start_time = time.time()
    model.update_data(point)
    x = torch.Tensor(model.new_data)
    if model.model is None:
        model.init_model(n_components)
        epochs = 500
    else:
        epochs = adjust_epochs_based_on_points(len(model.new_data))
    pi, mu, sigma = model.train_and_extract(x, epochs=epochs)
    model.new_data = []  # Clear new data after training
    execution_time = time.time() - start_time
    return {"pi": pi.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "time": execution_time}

# Train the model with data points and return the parameters
@app.post("/train")
async def train(point: Points, n_components: int, epochs: int, model: Model = Depends(get_model)):
    start_time = time.time()
    x = torch.Tensor([p.x for p in point.coordinates])
    model = GMMSimple(n_components)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    train_model(model, optimizer, x, n_epochs=500)
    pi, mu, sigma = extract_gmm_parameters(model, x.mean().numpy(), x.std().numpy())
    execution_time = time.time() - start_time
    return {"pi": pi.tolist(), "mu": mu.tolist(), "sigma": sigma.tolist(), "time": execution_time}

# Clear the model and optimizer
@app.post("/clear")
async def clear_model(model: Model = Depends(get_model)):
    model.data = []
    model.init_model()
    return {"status": "Model and optimizer cleared"}
