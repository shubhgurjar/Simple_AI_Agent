import json
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from rag import SimpleRAGPipeline


class RAGPipelineTests(unittest.TestCase):
    def make_pipeline(self):
        with patch.dict(
            "os.environ",
            {
                "BASE_URL": "https://example.test/v1",
                "API_KEY": "test-key",
                "EMBEDDING_MODEL_NAME": "test-embedding-model",
            },
        ), patch("rag.load_dotenv"), patch("rag.OpenAI", return_value=Mock()):
            return SimpleRAGPipeline()

    def test_load_documents_returns_sorted_filename_and_text_records(self):
        pipeline = self.make_pipeline()
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "b.txt"), "w", encoding="utf-8") as file:
                file.write("Second document")
            with open(os.path.join(directory, "a.txt"), "w", encoding="utf-8") as file:
                file.write("First document")
            with open(os.path.join(directory, "ignored.md"), "w", encoding="utf-8") as file:
                file.write("Not a text document")

            documents = pipeline._load_documents(directory)

        self.assertEqual(
            documents,
            [
                {"filename": "a.txt", "text": "First document"},
                {"filename": "b.txt", "text": "Second document"},
            ],
        )

    def test_top_k_returns_full_chunk_records_sorted_by_similarity(self):
        pipeline = self.make_pipeline()
        chunks = [
            {"filename": "diagonal.txt", "chunk_id": 1, "chunk_text": "diagonal", "embedding": [0.8, 0.6]},
            {"filename": "aligned.txt", "chunk_id": 1, "chunk_text": "aligned", "embedding": [1.0, 0.0]},
            {"filename": "orthogonal.txt", "chunk_id": 1, "chunk_text": "orthogonal", "embedding": [0.0, 1.0]},
            {"filename": "opposite.txt", "chunk_id": 1, "chunk_text": "opposite", "embedding": [-1.0, 0.0]},
        ]

        top_chunks = pipeline._top_k_chunks([1.0, 0.0], chunks)

        self.assertEqual(
            [chunk["filename"] for chunk in top_chunks],
            ["aligned.txt", "diagonal.txt", "orthogonal.txt"],
        )
        self.assertAlmostEqual(top_chunks[0]["similarity_score"], 1.0)
        self.assertAlmostEqual(top_chunks[1]["similarity_score"], 0.8)
        self.assertIn("embedding", top_chunks[0])
        self.assertIn("chunk_text", top_chunks[0])

    def test_chunk_documents_splits_sections_and_numbers_each_file(self):
        pipeline = self.make_pipeline()
        documents = [
            {
                "filename": "guide.txt",
                "text": " First section \n\n\nSecond section\n\n",
            },
            {"filename": "policy.txt", "text": "Only section"},
        ]

        chunks = pipeline._chunk_documents(documents)

        self.assertEqual(
            chunks,
            [
                {
                    "filename": "guide.txt",
                    "chunk_id": 1,
                    "chunk_text": "First section",
                },
                {
                    "filename": "guide.txt",
                    "chunk_id": 2,
                    "chunk_text": "Second section",
                },
                {
                    "filename": "policy.txt",
                    "chunk_id": 1,
                    "chunk_text": "Only section",
                },
            ],
        )

    def test_load_or_create_chunks_persists_embeddings_and_reuses_them(self):
        pipeline = self.make_pipeline()

        with tempfile.TemporaryDirectory() as directory:
            vector_file = os.path.join(directory, "vectors.json")
            with open(
                os.path.join(directory, "guide.txt"),
                "w",
                encoding="utf-8",
            ) as file:
                file.write("First section\n\nSecond section")
            pipeline._get_embedding = Mock(
                side_effect=[[0.1, 0.2], [0.3, 0.4]]
            )
            load_documents = Mock(wraps=pipeline._load_documents)
            pipeline._load_documents = load_documents

            chunks = pipeline._load_or_create_chunks(directory, vector_file)

            self.assertEqual(pipeline._get_embedding.call_count, 2)
            load_documents.assert_called_once_with(directory, ["guide.txt"])
            with open(vector_file, "r", encoding="utf-8") as file:
                saved_vectors = json.load(file)
            self.assertEqual(saved_vectors["guide.txt"], chunks)

            pipeline._get_embedding.reset_mock()
            load_documents.reset_mock()
            cached_chunks = pipeline._load_or_create_chunks(directory, vector_file)

        pipeline._get_embedding.assert_not_called()
        load_documents.assert_not_called()
        self.assertEqual(cached_chunks, chunks)

    def test_load_or_create_chunks_embeds_only_uncached_files(self):
        pipeline = self.make_pipeline()

        with tempfile.TemporaryDirectory() as directory:
            vector_file = os.path.join(directory, "vectors.json")
            with open(
                os.path.join(directory, "existing.txt"),
                "w",
                encoding="utf-8",
            ) as file:
                file.write("Cached text")
            pipeline._get_embedding = Mock(return_value=[0.5, 0.5])
            pipeline._load_or_create_chunks(directory, vector_file)
            pipeline._get_embedding.reset_mock()

            with open(
                os.path.join(directory, "new.txt"),
                "w",
                encoding="utf-8",
            ) as file:
                file.write("New text")
            load_documents = Mock(wraps=pipeline._load_documents)
            pipeline._load_documents = load_documents
            chunks = pipeline._load_or_create_chunks(
                directory,
                vector_file,
            )

        pipeline._get_embedding.assert_called_once_with("New text")
        load_documents.assert_called_once_with(directory, ["new.txt"])
        self.assertEqual(
            [chunk["filename"] for chunk in chunks],
            ["existing.txt", "new.txt"],
        )

    def test_embed_chunks_adds_vectors_to_existing_records(self):
        pipeline = self.make_pipeline()
        chunks = [
            {
                "filename": "guide.txt",
                "chunk_id": 1,
                "chunk_text": "Useful guidance",
            }
        ]
        pipeline._get_embedding = Mock(return_value=[0.25, 0.75])

        result = pipeline._embed_chunks(chunks)

        self.assertIs(result, chunks)
        self.assertEqual(
            chunks[0],
            {
                "filename": "guide.txt",
                "chunk_id": 1,
                "chunk_text": "Useful guidance",
                "embedding": [0.25, 0.75],
            },
        )

    def test_search_embeds_chunks_and_returns_top_k_records(self):
        pipeline = self.make_pipeline()
        chunks = [
            {"filename": "first.txt", "chunk_id": 1, "chunk_text": "First"},
            {"filename": "second.txt", "chunk_id": 1, "chunk_text": "Second"},
            {"filename": "third.txt", "chunk_id": 1, "chunk_text": "Third"},
        ]
        pipeline._get_embedding = Mock(
            side_effect=[
                [1.0, 0.0],
                [0.0, 1.0],
                [0.8, 0.6],
                [-1.0, 0.0],
            ]
        )

        results = pipeline.search("query", chunks, k=2)

        self.assertEqual(
            [chunk["filename"] for chunk in results],
            ["second.txt", "first.txt"],
        )
        self.assertEqual(pipeline._get_embedding.call_count, 4)
        self.assertIn("embedding", results[0])
        self.assertIn("similarity_score", results[0])

    def test_search_reuses_existing_chunk_embeddings(self):
        pipeline = self.make_pipeline()
        chunks = [
            {
                "filename": "ready.txt",
                "chunk_id": 1,
                "chunk_text": "Already embedded",
                "embedding": [1.0, 0.0],
            }
        ]
        pipeline._get_embedding = Mock(return_value=[1.0, 0.0])

        results = pipeline.search("query", chunks)

        pipeline._get_embedding.assert_called_once_with("query")
        self.assertEqual(results[0]["similarity_score"], 1.0)


if __name__ == "__main__":
    unittest.main()
