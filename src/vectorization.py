from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(cv_texts: list[str], job_description: str) -> list[float]:
    documents = cv_texts + [job_description]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    cv_vectors = tfidf_matrix[:-1]
    job_vector = tfidf_matrix[-1]

    scores = cosine_similarity(cv_vectors, job_vector)

    return scores.flatten().tolist()


if __name__ == "__main__":
    cvs = [
        "python machine learning data analysis pandas numpy",
        "network security firewall routing switching",
        "web development html css javascript react"
    ]

    job_desc = "looking for a python data scientist with machine learning skills"

    scores = compute_similarity(cvs, job_desc)

    for i, score in enumerate(scores, start=1):
        print(f"CV {i} score: {score:.4f}")
