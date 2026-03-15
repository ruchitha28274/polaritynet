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

# Load ML model
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


class Review(BaseModel):
    text: str


# -------------------------
# Highlight sentiment words
# -------------------------
def highlight_words(text):

    words = text.split()
    highlighted = []

    for word in words:

        score = vader.polarity_scores(word)["compound"]

        if score >= 0.05:
            highlighted.append(f"<span style='color:green'>{word}</span>")

        elif score <= -0.05:
            highlighted.append(f"<span style='color:red'>{word}</span>")

        else:
            highlighted.append(f"<span style='color:black'>{word}</span>")

    return " ".join(highlighted)


# -------------------------
# VADER sentiment
# -------------------------
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


# -------------------------
# Recursive logic
# -------------------------
def recursive_sentiment(text):

    sentences = text.split(".")
    results = []

    for s in sentences:
        if s.strip() != "":
            sentiment, _ = analyze_sentiment(s)
            results.append(sentiment)

    if len(results) == 0:
        return "Neutral", 0

    final = max(set(results), key=results.count)

    sentiment, confidence = analyze_sentiment(text)

    return final, confidence


# -------------------------
# ML model prediction
# -------------------------
def ml_sentiment(text):

    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]

    return prediction


# -------------------------
# Home API
# -------------------------
@app.get("/")
def home():
    return {"message": "PolarityNet Hybrid AI Running"}


# -------------------------
# Predict API
# -------------------------
@app.post("/predict")
def predict(review: Review):

    text = review.text

    vader_sentiment, confidence = recursive_sentiment(text)

    ml_prediction = ml_sentiment(text)

    # Hybrid logic
    if vader_sentiment == ml_prediction:
        final_sentiment = vader_sentiment
    else:
        final_sentiment = ml_prediction

    highlighted = highlight_words(text)

    return {
        "review": text,
        "highlighted_text": highlighted,
        "vader_sentiment": vader_sentiment,
        "ml_sentiment": ml_prediction,
        "final_sentiment": final_sentiment,
        "confidence": confidence
    }
