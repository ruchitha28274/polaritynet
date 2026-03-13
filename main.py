from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vader = SentimentIntensityAnalyzer()

class Review(BaseModel):
    text: str


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


@app.get("/")
def home():
    return {"message": "PolarityNet Hybrid AI Running"}


@app.post("/predict")
def predict(review: Review):

    sentiment, confidence = recursive_sentiment(review.text)

    return {
        "review": review.text,
        "sentiment": sentiment,
        "confidence": confidence
    }