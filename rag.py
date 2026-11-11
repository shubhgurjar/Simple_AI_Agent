import json
import logging
import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from utils import configure_logging


class SimpleRAGPipeline:
    """Create text embeddings and compare their semantic similarity."""

    def __init__(self, debug_enabled=False):
        # Load API settings from the project's .env file before creating the client.
        load_dotenv()
        self.debug_enabled = debug_enabled
        self.client = OpenAI(
            base_url=os.environ["BASE_URL"],
            api_key=os.environ["API_KEY"],
        )
        self.embedding_model = os.environ["EMBEDDING_MODEL_NAME"]
        self.logger = logging.getLogger(__name__)

    def _load_documents(self, directory, filenames=None):
        """Read selected .txt files and return filename/text records."""
        documents = []
        if filenames is None:
            filenames = sorted(os.listdir(directory))
        for filename in filenames:
            if filename.endswith(".txt"):
                filepath = os.path.join(directory, filename)
                with open(filepath, "r", encoding="utf-8") as file:
                    documents.append(
                        {
                            "filename": filename,
                            "text": file.read(),
                        }
                    )
                    if self.debug_enabled:
                        self.logger.info("Loaded document: %s", filename)
        return documents

    def _chunk_documents(self, documents):
        """Split documents at blank lines and retain each source filename."""
        chunks = []
        for document in documents:
            chunk_id = 0
            for section in document["text"].split("\n\n"):
                chunk_text = section.strip()
                if not chunk_text:
                    continue
                chunk_id += 1
                chunks.append(
                    {
                        "filename": document["filename"],
                        "chunk_id": chunk_id,
                        "chunk_text": chunk_text,
                    }
                )
        return chunks

    def _load_or_create_chunks(self, directory, vector_file):
        """Load cached chunks or read, chunk, and embed files missing from cache."""
        try:
            with open(vector_file, "r", encoding="utf-8") as file:
                vector_cache = json.load(file)
        except FileNotFoundError:
            vector_cache = {}
        except json.JSONDecodeError as exc:
            self.logger.exception("Invalid vector cache: %s", vector_file)
            raise ValueError(f"Vector cache is not valid JSON: {vector_file}") from exc

        if not isinstance(vector_cache, dict):
            raise ValueError(f"Vector cache must contain an object: {vector_file}")

        filenames = [
            filename
            for filename in sorted(os.listdir(directory))
            if filename.endswith(".txt")
        ]
        uncached_filenames = [
            filename for filename in filenames if filename not in vector_cache
        ]
        new_documents = (
            self._load_documents(directory, uncached_filenames)
            if uncached_filenames
            else []
        )
        documents_by_filename = {
            document["filename"]: document for document in new_documents
        }

        chunks = []
        cache_changed = False
        for filename in filenames:
            if filename in vector_cache:
                cached_chunks = vector_cache[filename]
                if not isinstance(cached_chunks, list):
                    raise ValueError(
                        f"Cached chunks for {filename!r} must be a list."
                    )
                chunks.extend(cached_chunks)
                if self.debug_enabled:
                    self.logger.info("Loaded cached chunks for %s", filename)
                continue

            new_chunks = self._chunk_documents(
                [documents_by_filename[filename]]
            )
            self._embed_chunks(new_chunks)
            vector_cache[filename] = new_chunks
            chunks.extend(new_chunks)
            cache_changed = True

        if cache_changed:
            with open(vector_file, "w", encoding="utf-8") as file:
                json.dump(vector_cache, file, indent=2)

        return chunks

    def _get_embedding(self, text):
        """Return the embedding vector for one text from the configured model."""
        try:
            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=text,
            )
            embedding = response.data[0].embedding
            if self.debug_enabled:
                self.logger.info(
                    "Generated embedding with %d dimensions.",
                    len(embedding),
                )
            return embedding
        except OpenAIError as exc:
            self.logger.exception("Failed to get embeddings")
            raise RuntimeError(f"Failed to get embeddings: {exc}") from exc

    def _cosine_similarity(self, embedding1, embedding2):
        """Return cosine similarity, or zero if either vector has zero magnitude."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot_product / (norm1 * norm2)

    def _embed_chunks(self, chunks):
        """Add missing embedding vectors to chunk records."""
        for chunk in chunks:
            if "embedding" not in chunk:
                chunk["embedding"] = self._get_embedding(chunk["chunk_text"])
        return chunks

    def _top_k_chunks(self, query_embedding, chunks, k=3):
        """Score chunk records and return the k most similar records."""
        for chunk in chunks:
            chunk["similarity_score"] = float(
                self._cosine_similarity(
                    query_embedding,
                    chunk["embedding"],
                )
            )
        return sorted(
            chunks,
            key=lambda chunk: chunk["similarity_score"],
            reverse=True,
        )[:k]

    def search(self, user_query, chunks, k=3):
        """Return the top-k chunks relevant to a query.

        Each chunk has ``filename``, ``chunk_id``, and ``chunk_text``. Existing
        embeddings are reused; missing embeddings are generated before ranking.
        """
        if not chunks:
            return []

        query_embedding = self._get_embedding(user_query)
        embedded_chunks = self._embed_chunks(chunks)
        return self._top_k_chunks(
            query_embedding,
            embedded_chunks,
            k=k,
        )

    def _print_results(self, chunks):
        """Print ranked source names, chunk IDs, scores, and text."""
        for rank, chunk in enumerate(chunks, start=1):
            print(
                f"{rank}. {chunk['filename']} "
                f"(chunk {chunk['chunk_id']}, "
                f"similarity: {chunk['similarity_score']:.3f})"
            )
            print(chunk["chunk_text"])

    def run(self):
        """Load and chunk documents, then print the three best matches."""
        try:
            user_query = input("Enter a query to compare with the knowledge base: ")

            dir_path = os.path.join(os.path.dirname(__file__), "knowledge_base")
            vector_file = os.path.join(dir_path, "vectors.json")
            chunks = self._load_or_create_chunks(dir_path, vector_file)
            print(f"Loaded {len(chunks)} sections from the knowledge base.")
            if not chunks:
                self.logger.warning("No text chunks found in %s", dir_path)
                return []

            top_chunks = self.search(
                user_query,
                chunks,
                k=3,
            )
            self._print_results(top_chunks)
            return top_chunks

        except KeyboardInterrupt:
            self.logger.info("Agent session interrupted by user.")
            print("\n🤖 RAG: Interrupted. Goodbye.")


def main():
    load_dotenv()
    debug_enabled = configure_logging()
    try:
        SimpleRAGPipeline(debug_enabled=debug_enabled).run()
    except KeyError as exc:
        logging.error("Missing required environment variable: %s", exc.args[0])
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
