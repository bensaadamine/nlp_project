import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import unicodedata


STOPWORDS = set(stopwords.words("french"))

def remove_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return unicodedata.normalize("NFC", text)



def preprocess_text(text: str) -> str:
    text = text.lower()
    text = remove_accents(text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    cleaned_tokens = [
        token for token in tokens
        if token not in STOPWORDS and len(token) > 2
    ]
    #Reconstruct text
    return " ".join(cleaned_tokens)


if __name__ == "__main__":
    sample_text = """
    premièrement, c'est un exemple de texte en français avec des accents.
    Deuxièmement, il contient des mots courants qui devraient être supprimés.
    """

    print("Original text:")
    print(sample_text)

    print("\nPreprocessed text:")
    print(preprocess_text(sample_text))
