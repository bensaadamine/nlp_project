import pdfplumber
from pathlib import Path
from preprocessing import preprocess_text
import unicodedata
import re


def clean_pdf_text(text: str) -> str:
    """Nettoie le texte extrait du PDF en normalisant les accents mal positionnés"""
    # Normalise d'abord le texte
    text = unicodedata.normalize("NFD", text)
    
    # Supprime les accents orphelins (accents sans caractère avant)
    text = re.sub(r"[\´`¨^~]", "", text)
    
    # Recompose les caractères
    text = unicodedata.normalize("NFC", text)
    
    return text


def extract_text_from_pdf(pdf_path: Path) -> str:
    extracted_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()

            if text:
                # Nettoie le texte extrait
                text = clean_pdf_text(text)
                extracted_text.append(text)
            else:
                print(f"[WARNING] No text found on page {page_number}")

    return "\n".join(extracted_text)


if __name__ == "__main__":
    pdf_file = Path("data/cvs/cv_test.pdf")

    text = extract_text_from_pdf(pdf_file)

    print("===== EXTRACTED TEXT =====")
    clean_text = preprocess_text(text)
    print(clean_text[:1500])  
