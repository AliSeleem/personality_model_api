# Personality Predictor API

This project implements a machine learning model that predicts whether a person is an introvert or extrovert based on various behavioral features. The model is served through a FastAPI web service.

## Project Structure

```
├── model_scaler.pkl       # Saved preprocessing components (scaler and column information)
├── personality_model.json  # Trained XGBoost model
├── requirements.txt       # Project dependencies
├── script.py             # FastAPI application code
└── README.md            # Project documentation
```

## Features

The model takes into account the following features to make predictions:

- Time spent alone
- Stage fear
- Social event attendance
- Going outside frequency
- Whether the person feels drained after socializing
- Friends circle size
- Social media post frequency

## Technology Stack

- **FastAPI**: Modern web framework for building APIs
- **XGBoost**: Machine learning model implementation
- **scikit-learn**: For data preprocessing
- **pandas**: For data handling
- **uvicorn**: ASGI server for running the FastAPI application

## Installation

1. Clone this repository
2. Create a virtual environment (recommended)
3. Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

1. Start the server:

```bash
python script.py
```

2. The API will be available at `http://127.0.0.1:8000`

3. Send POST requests to `/predict` endpoint with JSON data in the following format:

```json
{
	"Time_spent_Alone": 7.5,
	"Stage_fear": "Yes",
	"Social_event_attendance": 3.0,
	"Going_outside": 4.0,
	"Drained_after_socializing": "Yes",
	"Friends_circle_size": 5.0,
	"Post_frequency": 2.0
}
```

4. The API will return a prediction indicating whether the person is an introvert or extrovert:

```json
{
	"result": "You are an Introvert!"
}
```

## API Documentation

Once the server is running, you can access:

- Interactive API documentation at `http://127.0.0.1:8000/docs`
- Alternative API documentation at `http://127.0.0.1:8000/redoc`

## Input Features Description

- **Time_spent_Alone**: Float value representing hours spent alone per day
- **Stage_fear**: String ("Yes" or "No") indicating presence of stage fear
- **Social_event_attendance**: Float value representing frequency of attending social events
- **Going_outside**: Float value representing frequency of going outside
- **Drained_after_socializing**: String ("Yes" or "No") indicating whether person feels drained after socializing
- **Friends_circle_size**: Float value representing number of close friends
- **Post_frequency**: Float value representing frequency of social media posts

## Model Details

The system uses an XGBoost classifier model trained on personality data. The model processes both numeric and binary features:

- Numeric features are scaled using a pre-fitted scaler
- Binary features ("Yes"/"No") are mapped to 1/0 respectively
- The model outputs a binary prediction (0 for Introvert, 1 for Extrovert)

## Dependencies

Key dependencies and their versions are listed in `requirements.txt`. The project uses Python packages including:

- fastapi==0.116.1
- xgboost==3.0.2
- pandas==2.3.1
- uvicorn==0.35.0
- scikit-learn==1.7.0
- numpy==2.3.1

## License

This project is open-sourced under the MIT License.
