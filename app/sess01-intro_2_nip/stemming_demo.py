# python script to demonstrate stemming with visualization

# --------------------------------------------------------------------
# 0. import the required modules
# --------------------------------------------------------------------
import matplotlib.pyplot as plt
import nltk
import re
import ssl
from collections import Counter
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

# --------------------------------------------------------------------
# 1. Download the required NLTK data
# --------------------------------------------------------------------
try:
    # Try to create an unverified SSL context
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download the necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# --------------------------------------------------------------------
# 2. Sample Text to be stemmed
# --------------------------------------------------------------------
TEXT = """
The researchers were studying the running patterns of various animals.
They observed that the faster animals consistently outperformed the
slower one. The studies showed the interesting running behaviours.
"""

# initialise the stemmer
stemmer = SnowballStemmer(language='english')

# --------------------------------------------------------------------
# 3. Text Preparation Function
# --------------------------------------------------------------------
def preprocess_text(text: str) -> list:
    """
    Tokenize and clean the input text.
    This function converts text to lowercase, removes punctuation,
    and returns a list of valid word tokens.

    :param text: Raw input text
    :return: List of valid word tokens
    """
    tokens = word_tokenize(text.lower())
    cleaned_tokens = [re.sub(r'[^a-z]', '', token) for token in tokens]
    return [token for token in cleaned_tokens if token]

# --------------------------------------------------------------------
# 4. Stemming Function
# --------------------------------------------------------------------
def apply_stemming(tokens: list) -> list:
    """
    Apply stemming to a list of tokens.
    :param tokens: List of word tokens
    :return: List of stemmed word tokens
    """
    return [stemmer.stem(token) for token in tokens]

# --------------------------------------------------------------------
# 5. Visualization Function
# --------------------------------------------------------------------
def plot_frequencies(original: list, stemmed: list) -> None:
    """
    Plot frequency comparison between original and stemmed word tokens.

    This helps illustrate how stemming groups similar words together.
    :param original: List of original word tokens
    :param stemmed: List of stemmed word tokens
    """
    original_counts = Counter(original)
    stemmed_counts = Counter(stemmed)

    # select top items for clarity
    top_original = dict(original_counts.most_common(5))
    top_stemmed = dict(stemmed_counts.most_common(5))

    # plot original word frequencies
    plt.figure(figsize=(12, 8))
    plt.bar(top_original.keys(), top_original.values())
    plt.title('Top 5 original words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')

    # plot stemmed word frequencies
    plt.figure()
    plt.bar(top_stemmed.keys(), top_stemmed.values())
    plt.title('Top 5 stemmed words')
    plt.xlabel('Stems')
    plt.ylabel('Frequency')

    plt.show()

# --------------------------------------------------------------------
# 6. Main Execution Function
# --------------------------------------------------------------------
def main():
    """
    Execute the Stemming Demo.
    This demonstrates text processing, text stemming and visual comparison.
    """
    print(f"\nOriginal text: \n{TEXT}")

    # preprocess the text
    tokens = preprocess_text(TEXT)
    print(f"\nTokenized text: \n{tokens}")

    # apply stemming
    stemmed_tokens = apply_stemming(tokens)
    print(f"\nStemmed tokens: \n{stemmed_tokens}")

    # show visual comparison
    plot_frequencies(tokens, stemmed_tokens)

# --------------------------------------------------------------------
# 7. Run the script by invoking its main() function
# --------------------------------------------------------------------
if __name__ == "__main__":
    main()