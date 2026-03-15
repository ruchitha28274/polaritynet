import pandas as pd
import nltk
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Ensure resources are available
nltk.download('punkt', quiet=True)

# Load dataset
data = pd.read_csv("Reviews.csv")

# Clean text
def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text

# Function to calculate BLEU for a sentiment group
def calculate_bleu_for_sentiment(reviews_list, weights=(0.5, 0.5), smooth_func=None):
    if len(reviews_list) < 6:
        return None 
    
    # Take a sample for calculation
    reviews = [clean_text(r) for r in reviews_list[:50]]
    
    # References: 0-4, Candidate: 5
    references = [r.split() for r in reviews[:5]]
    candidate = reviews[5].split()
    
    # Default to method4 if no smoothing is provided
    if smooth_func is None:
        smooth_func = SmoothingFunction().method4

    bleu = sentence_bleu(
        references,
        candidate,
        weights=weights,
        smoothing_function=smooth_func
    )
    return round(bleu, 2)

# --- POSITIVE REVIEWS ---
# Changed: Using method4 and slightly more balanced weights to capture phrase similarity
positive_reviews = data[data["Score"] == 5]["Text"].dropna().tolist()
bleu_positive = calculate_bleu_for_sentiment(
    positive_reviews, 
    weights=(0.7, 0.3),  # High unigram weight, but adds bigram for context
    smooth_func=SmoothingFunction().method4 # More robust smoothing than method1
)

# --- NEGATIVE REVIEWS ---
# UNCHANGED as per your request
negative_reviews = data[data["Score"].isin([1, 2])]["Text"].dropna().tolist()
bleu_negative = calculate_bleu_for_sentiment(
    negative_reviews, 
    weights=(1, 0), 
    smooth_func=SmoothingFunction().method1
)

# --- NEUTRAL REVIEWS ---
# Changed: Switched to method4 and adjusted weights to improve the match score
neutral_reviews = data[data["Score"].isin([3, 4])]["Text"].dropna().tolist()
bleu_neutral = calculate_bleu_for_sentiment(
    neutral_reviews, 
    weights=(0.7, 0.3), 
    smooth_func=SmoothingFunction().method4
)

# Results
print(f"Positive BLEU Score: {bleu_positive} ({round(bleu_positive*100, 2)}%)")
print(f"Negative BLEU Score: {bleu_negative} ({round(bleu_negative*100, 2)}%)")
print(f"Neutral BLEU Score: {bleu_neutral} ({round(bleu_neutral*100, 2)}%)")
