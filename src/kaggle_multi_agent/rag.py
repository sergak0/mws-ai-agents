from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class RetrievedChunk:
    source: str
    score: float
    text: str


class LocalKnowledgeBase:
    def __init__(
        self,
        vectorizer: TfidfVectorizer,
        matrix,
        chunks: list[RetrievedChunk],
    ) -> None:
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.chunks = chunks

    @classmethod
    def build(cls, source_dir: Path) -> "LocalKnowledgeBase":
        paths = sorted(source_dir.rglob("*.md"))
        chunks: list[RetrievedChunk] = []
        for path in paths:
            raw_text = path.read_text(encoding="utf-8")
            for chunk in _split_markdown(raw_text):
                chunks.append(RetrievedChunk(source=str(path), score=0.0, text=chunk))
        corpus = [chunk.text for chunk in chunks] or [""]
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        matrix = vectorizer.fit_transform(corpus)
        return cls(vectorizer=vectorizer, matrix=matrix, chunks=chunks)

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        ranked = sorted(enumerate(scores.tolist()), key=lambda item: item[1], reverse=True)[:k]
        return [
            RetrievedChunk(
                source=self.chunks[index].source,
                score=score,
                text=self.chunks[index].text,
            )
            for index, score in ranked
            if score > 0
        ]

    def save(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.vectorizer, output_dir / "vectorizer.joblib")
        joblib.dump(self.matrix, output_dir / "matrix.joblib")
        joblib.dump(self.chunks, output_dir / "chunks.joblib")

    @classmethod
    def load(cls, output_dir: Path) -> "LocalKnowledgeBase":
        vectorizer = joblib.load(output_dir / "vectorizer.joblib")
        matrix = joblib.load(output_dir / "matrix.joblib")
        chunks = joblib.load(output_dir / "chunks.joblib")
        return cls(vectorizer=vectorizer, matrix=matrix, chunks=chunks)


def _split_markdown(text: str) -> list[str]:
    parts = [part.strip() for part in text.split("\n\n")]
    return [part for part in parts if len(part) >= 20]
