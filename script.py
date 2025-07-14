from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import xgboost as xgb
import pandas as pd
import uvicorn
import pickle
from pathlib import Path

model = xgb.XGBClassifier()
model.load_model("personality_model.json")

with open("model_scaler.pkl", "rb") as f:
    preprocess = pickle.load(f)

scaler = preprocess["scaler"]
numeric_cols = preprocess["numeric_cols"]
binary_cols = preprocess["binary_cols"]
feature_order = preprocess["feature_order"]

binary_map = {'Yes': 1, 'No': 0}

app = FastAPI()

# Mount the static directory
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())

class InputData(BaseModel):
    Time_spent_Alone: float
    Stage_fear: str
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: str
    Friends_circle_size: float
    Post_frequency: float

@app.post("/predict")
def predict(data: InputData):
    input_dict = data.dict()
    df = pd.DataFrame([input_dict], columns=feature_order)

    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    df[numeric_cols] = scaler.transform(df[numeric_cols])

    prediction = model.predict(df)[0]

    if(prediction == 1):
        return {'result': 'You are an Extrovert!'}
    else:
        return {'result': 'You are an Introvert!'}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
