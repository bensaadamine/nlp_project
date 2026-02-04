from pathlib import Path

from pdf_extraction import extract_text_from_pdf
from preprocessing import preprocess_text
from vectorization import compute_similarity


def load_job_description(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def rank_cvs(
    cvs_dir: Path,
    job_description_path: Path,
    acceptance_threshold: float = 0.25,
):
    job_raw = load_job_description(job_description_path)
    job_clean = preprocess_text(job_raw)

    cv_texts = []
    cv_files = []

    for pdf_file in cvs_dir.glob("*.pdf"):
        raw_text = extract_text_from_pdf(pdf_file)
        clean_text = preprocess_text(raw_text)

        cv_texts.append(clean_text)
        cv_files.append(pdf_file.name)

    scores = compute_similarity(cv_texts, job_clean)

    results = []
    for filename, score in zip(cv_files, scores):
        results.append(
            {
                "cv": filename,
                "score": round(score, 4),
                "decision": "ACCEPTED" if score >= acceptance_threshold else "REJECTED",
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


if __name__ == "__main__":
    cvs_directory = Path("data/cvs")
    job_path = Path("data/job_description.txt")

    ranking = rank_cvs(cvs_directory, job_path)

    print("===== CV RANKING =====")
    for i, item in enumerate(ranking, start=1):
        print(
            f"{i}. {item['cv']} | score={item['score']} | {item['decision']}"
        )
