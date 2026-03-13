import pandas as pd
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download tokenizer
nltk.download('punkt')

# Load dataset (assuming reviews.csv is the Amazon Fine Food Reviews dataset)
data = pd.read_csv("reviews.csv")

# Select review column (Amazon Fine Food uses 'Text' for review content)
reviews = data["Text"].dropna().tolist()

# Take two reviews for comparison
reference = reviews[0].lower().split()
candidate = reviews[1].lower().split()

# Calculate BLEU score with smoothing
smoothing = SmoothingFunction().method1
bleu = sentence_bleu([reference], candidate, smoothing_function=smoothing)
print("Reference Review:", reference)
print("Candidate Review:", candidate)
print("BLEU Score:", bleu)