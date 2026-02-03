import pdfplumber
from pathlib import Path


def extract_text_from_pdf(pdf_path: Path) -> str:
    extracted_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()

            if text:
                extracted_text.append(text)
            else:
                print(f"[WARNING] No text found on page {page_number}")

    return "\n".join(extracted_text)


if __name__ == "__main__":
    pdf_file = Path("data/cvs/cv_test.pdf")

    text = extract_text_from_pdf(pdf_file)

    print("===== EXTRACTED TEXT =====")
    print(text[:1500])  # afficher seulement le début
