from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize VADER
vader = SentimentIntensityAnalyzer()

# Load trained ML model
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Request body format
class Review(BaseModel):
    text: str


# -------------------------------
# VADER Sentiment
# -------------------------------
def analyze_sentiment(text):

    score = vader.polarity_scores(text)
    compound = score["compound"]

    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    confidence = round(abs(compound) * 100, 2)

    return sentiment, confidence


# -------------------------------
# Recursive Sentiment Logic
# -------------------------------
def recursive_sentiment(text):

    sentences = text.split(".")
    results = []

    for s in sentences:
        if s.strip() != "":
            sentiment, confidence = analyze_sentiment(s)
            results.append(sentiment)

    if len(results) == 0:
        return "Neutral", 0

    final = max(set(results), key=results.count)

    sentiment, confidence = analyze_sentiment(text)

    return final, confidence


# -------------------------------
# ML Model Sentiment
# -------------------------------
def ml_sentiment(text):

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    return prediction


# -------------------------------
# Home API
# -------------------------------
@app.get("/")
def home():
    return {"message": "PolarityNet Hybrid AI Running"}


# -------------------------------
# Predict API
# -------------------------------
@app.post("/predict")
def predict(review: Review):

    text = review.text

    # VADER + Recursive result
    vader_sentiment, confidence = recursive_sentiment(text)

    # ML prediction
    ml_prediction = ml_sentiment(text)

    # Hybrid decision
    if vader_sentiment == ml_prediction:
        final_sentiment = vader_sentiment
    else:
        final_sentiment = ml_prediction

    return {
        "review": text,
        "vader_sentiment": vader_sentiment,
        "ml_sentiment": ml_prediction,
        "final_sentiment": final_sentiment,
        "confidence": confidence
    }
