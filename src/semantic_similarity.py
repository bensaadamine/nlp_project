from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSimilarity:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]):
        return self.model.encode(texts, normalize_embeddings=True)

    def compute_similarity(self, cv_texts: list[str], job_description: str) -> list[float]:
        embeddings = self.encode(cv_texts + [job_description])

        cv_embeddings = embeddings[:-1]
        job_embedding = embeddings[-1].reshape(1, -1)

        scores = cosine_similarity(cv_embeddings, job_embedding)

        return scores.flatten().tolist()


if __name__ == "__main__":
    cvs = [
        "I have worked on regression and classification models, including supervised learning techniques.",
        "Experienced network engineer working on firewall and routing.",
        "Frontend developer specialized in HTML, CSS and React."
    ]

    job_desc = "We are looking for a machine learning engineer with experience in supervised learning."


    semantic = SemanticSimilarity()
    scores = semantic.compute_similarity(cvs, job_desc)

    for i, score in enumerate(scores, start=1):
        print(f"CV {i} semantic score: {score:.4f}")
