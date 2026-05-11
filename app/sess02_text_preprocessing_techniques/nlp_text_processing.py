"""
Natural Language Processing pipeline for Restaurant Reviews
The Preprocessing pipeline includes:
1. Lowercasing
2. Slang and Abbreviation normalization
3. Contraction expansion
4. Repeated character normalization
5. Emoji removal
6. Punctuation cleaning
7. Tokenization
8. Stopwords removal
9. Optional Lemmatization

Author: Temple
Date: 06 May 2026
"""

# --------------------------------------------------------------------
# 0. Import the required modules
# --------------------------------------------------------------------
import matplotlib.pyplot as plt
import nltk
import re
import ssl
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List

# --------------------------------------------------------------------
# 1. Download the required data
# --------------------------------------------------------------------
try:
    # SSL fix
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# --------------------------------------------------------------------
# 2. Raw Data (restaurant reviews)
# --------------------------------------------------------------------
REVIEWS = [
    "The food was 2die4! Best burgers in town 😋🍔",
    "Service was 10/10, loved the vibe too! Totally recommend this place 🔥👌",
    "2 hrs wait for a table... not worth it. Food was ok, but not great.",
    "Tbh, 4 the price, I expected way better quality. Disappointed.",
    "Great place for brunch! Had the eggs benedict, 1 of my faves 💯🍳",
    "The ambience was gr8! Luvd it :) !!!",
    "Had a blast at this place!! Will come again soon 😋",
    "Food was OK, but could be better, meh...",
    "This pizza was absolutely amazing, best I’ve had!!",
    "Service was horrible... never coming back!",
    "I loved the pasta! But the portion was so small :(",
    "The dessert was soooooo good!! 😍",
    "So disappointed, the steak was overcooked...",
    "Great experience, but the music was a little loud tbh.",
    "Good food, but they forgot my drink. :(",
    "Superb food! Totally worth the price! Will return!",
    "Was okay, nothing special. Meh.",
    "The chicken was so dry, I couldn't finish it :( too bad!",
    "Fantastic, I can't wait to visit again! :) :)",
    "Not worth the price, won't be coming back :( 😔"
]

# --------------------------------------------------------------------
# 3. Normalization Rules
# --------------------------------------------------------------------
SLANG_DICT = {
    r'\b2die4\b': 'to die for',
    r'\bgr8\b': 'great',
    r'\bluvd\b': 'loved',
    r'\bfaves\b': 'favourites',
    r'\btbh\b': 'to be honest',
    r'\bthx\b': 'thanks',
    r'\bplz\b': 'please',
    r'\bu\b': 'you',
    r'\b4\b': 'for',
    r'\b2\b': 'to',
    r'\b1\b': 'one',
    r'\b10/10\b': 'perfect',
    r'\bok\b': 'okay',
    r'\bmeh\b': 'mediocre',
    r'\btotally\s*recommend\b': 'highly recommend',
    r'\bambiance\b': 'ambience',
}

CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'ve": " have",
    "'m": " am",
}

# --------------------------------------------------------------------
# 4. Processing Functions
# --------------------------------------------------------------------
def normalise_text(text: str) -> str:
    """
    Apply basic text normalization: lowercasing, slang replacement,
    and contraction expansion.
    """
    text = text.lower()

    # Replace slang patterns
    for pattern, replacement in SLANG_DICT.items():
        text = re.sub(pattern, replacement, text)

    # Expand contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = text.replace(contraction, expansion)

    return text


def remove_emojis(text: str) -> str:
    """Remove emojis and non-ASCII characters."""
    return re.sub(r'[^\x00-\x7F]+', '', text)


def normalise_repeated_characters(text: str) -> str:
    """
    Reduce repeated characters (e.g., 'soooooo' -> 'so').
    Keeps only one occurrence of a character repeated three or more times.
    """
    return re.sub(r'(.)\1{2,}', r'\1', text)


def clean_text(text: str) -> str:
    """
    Remove punctuation and extra whitespace, keep only letters and spaces.
    """
    # Keep only lowercase letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Collapse multiple spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenise_and_filter(text: str) -> List[str]:
    """
    Tokenize text and remove stopwords. Also discards words shorter than 3 letters.
    """
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    return [word for word in tokens if word not in stop_words and len(word) > 2]


def lemmatise_tokens(tokens: List[str]) -> List[str]:
    """Apply lemmatization to each token."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


# --------------------------------------------------------------------
# 5. Pipeline Function
# --------------------------------------------------------------------
def preprocess_review(text: str) -> List[str]:
    """
    Apply full preprocessing pipeline to a single review.
    Returns a list of cleaned and lemmatized tokens.
    """
    text = normalise_text(text)
    text = normalise_repeated_characters(text)
    text = remove_emojis(text)
    text = clean_text(text)

    tokens = tokenise_and_filter(text)
    tokens = lemmatise_tokens(tokens)

    return tokens


# --------------------------------------------------------------------
# 6. Visualization
# --------------------------------------------------------------------
def plot_word_frequencies(reviews: List[str], processed_reviews: List[List[str]]) -> None:
    """
    Plot top 8 word frequencies before and after preprocessing.
    """
    # Original: simple split on whitespace after lowercasing
    original_words = " ".join(reviews).lower().split()
    # Processed: flatten list of token lists
    processed_words = [word for review in processed_reviews for word in review]

    orig_counts = Counter(original_words)
    proc_counts = Counter(processed_words)

    top_orig = dict(orig_counts.most_common(8))
    top_proc = dict(proc_counts.most_common(8))

    # Create two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(top_orig.keys(), top_orig.values(), color='skyblue')
    ax1.set_title('Before Preprocessing (Raw Words)')
    ax1.set_xlabel('Words')
    ax1.set_ylabel('Frequency')
    ax1.tick_params(axis='x', rotation=45)

    ax2.bar(top_proc.keys(), top_proc.values(), color='lightgreen')
    ax2.set_title('After Preprocessing (Lemmas, No Stopwords)')
    ax2.set_xlabel('Lemmas')
    ax2.set_ylabel('Frequency')
    ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------
# 7. Main Execution Function
# --------------------------------------------------------------------
def main():
    """Execute preprocessing pipeline and display results."""
    print(f"\nOriginal review (first one):\n{REVIEWS[0]}\n")

    # Process all reviews
    processed_reviews = [preprocess_review(review) for review in REVIEWS]

    print(f"Processed review (first one):\n{processed_reviews[0]}\n")

    # Visualize frequency changes
    plot_word_frequencies(REVIEWS, processed_reviews)


# --------------------------------------------------------------------
# 8. Run the script
# --------------------------------------------------------------------
if __name__ == "__main__":
    main()