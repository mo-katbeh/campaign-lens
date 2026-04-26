from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from app import rag_service
from app.embeddings import embed_query
from app.pinecone_store import translate_filters
from app.schemas import RetrievedChunk


class FilterTranslationTests(unittest.TestCase):
    def test_translate_filters_supports_friendly_keys_and_numeric_operators(self) -> None:
        filters = {
            "theme": "medical",
            "year": 2025,
            "funding_ratio": {"lt": 1, "gte": 0.1},
            "quality": {"gte": 80},
            "beneficiary": None,
        }

        translated = translate_filters(filters)

        self.assertEqual(
            translated,
            {
                "campaign_theme": "medical",
                "year_label": "2025",
                "funding_ratio": {"$lt": 1.0, "$gte": 0.1},
                "record_quality_score": {"$gte": 80.0},
            },
        )


class ApiMiddlewareTests(unittest.TestCase):
    def test_cors_allows_local_vite_dev_server(self) -> None:
        client = TestClient(app)

        response = client.options(
            "/retrieve",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "POST",
            },
        )

        self.assertIn(response.status_code, {200, 204})
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://127.0.0.1:5173")


class EmbeddingTests(unittest.TestCase):
    @patch("app.embeddings.embed_texts", return_value=[[0.1, 0.2, 0.3]])
    def test_embed_query_returns_first_embedding(self, mocked_embed_texts) -> None:
        result = embed_query("find medical campaigns")

        self.assertEqual(result, [0.1, 0.2, 0.3])
        mocked_embed_texts.assert_called_once_with(["find medical campaigns"])


class ContextBuilderTests(unittest.TestCase):
    def test_build_context_skips_blank_text_and_respects_limit(self) -> None:
        chunks = [
            RetrievedChunk(chunk_id="1-0", campaign_id=1, title="A", theme="medical", year="2025", text=""),
            RetrievedChunk(chunk_id="2-0", campaign_id=2, title="B", theme="food", year="2024", text="short text"),
            RetrievedChunk(
                chunk_id="3-0",
                campaign_id=3,
                title="C",
                theme="water",
                year="2023",
                text="x" * 300,
            ),
        ]

        context = rag_service.build_context(chunks, max_context_chars=140)

        self.assertIn("[Chunk 2-0]", context)
        self.assertNotIn("[Chunk 1-0]", context)
        self.assertNotIn("[Chunk 3-0]", context)


class RetrievalAndAnswerTests(unittest.TestCase):
    @patch("app.rag_service.query_index")
    @patch("app.rag_service.embed_query", return_value=[0.1, 0.2])
    def test_retrieve_chunks_queries_pinecone_and_normalizes_results(self, mocked_embed_query, mocked_query_index) -> None:
        mocked_query_index.return_value = {
            "matches": [
                {
                    "id": "10-0",
                    "score": 0.91,
                    "metadata": {
                        "campaign_id": 10,
                        "title": "Medical Campaign",
                        "campaign_theme": "medical",
                        "beneficiary_group": "patients",
                        "year_label": "2025",
                        "funding_ratio": 0.8,
                        "record_quality_score": 88,
                        "text": "Campaign text",
                    },
                }
            ]
        }

        result = rag_service.retrieve_chunks("help patients", filters={"theme": "medical"}, top_k=4)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].campaign_id, 10)
        self.assertEqual(result[0].theme, "medical")
        mocked_embed_query.assert_called_once_with("help patients")
        mocked_query_index.assert_called_once_with(vector=[0.1, 0.2], top_k=4, filters={"theme": "medical"})

    @patch("app.rag_service.get_gemini_client")
    @patch("app.rag_service.retrieve_chunks")
    @patch("app.rag_service.answer_funding_ratio_ranking_question", return_value=None)
    def test_answer_question_returns_grounded_result_with_sources(
        self,
        mocked_analytics_answer,
        mocked_retrieve_chunks,
        mocked_get_gemini_client,
    ) -> None:
        mocked_retrieve_chunks.return_value = [
            RetrievedChunk(
                chunk_id="10-0",
                campaign_id=10,
                title="Medical Campaign",
                theme="medical",
                year="2025",
                text="The campaign funds cancer treatment.",
            )
        ]
        mocked_model = mocked_get_gemini_client.return_value.models
        mocked_model.generate_content.return_value.text = "The retrieved campaign is funding cancer treatment."

        result = rag_service.answer_question("What is this campaign funding?")

        self.assertEqual(result.source_campaign_ids, [10])
        self.assertEqual(result.source_chunk_ids, ["10-0"])
        self.assertIn("cancer treatment", result.answer)
        mocked_analytics_answer.assert_called_once()

    @patch("app.rag_service.get_gemini_client")
    @patch("app.rag_service.load_campaign_analytics_rows")
    def test_answer_question_returns_ranked_lowest_funded_campaigns_in_syria(
        self,
        mocked_load_campaign_analytics_rows,
        mocked_get_gemini_client,
    ) -> None:
        mocked_load_campaign_analytics_rows.return_value = [
            {
                "campaign_id": 613,
                "title": "Until the Last Tent",
                "year": "2024",
                "theme": "housing",
                "beneficiary": "families",
                "funding_ratio": 0.097436,
                "quality": 100.0,
                "funding_status": "in_progress",
                "location_mentions": ["Syria"],
                "search_chunks": '[{"chunk_id":"613-0","text":"Housing support in Syria."}]',
            },
            {
                "campaign_id": 781,
                "title": "Our Countryside Deserves Better",
                "year": "2025",
                "theme": "community",
                "beneficiary": "families",
                "funding_ratio": 0.0242922,
                "quality": 95.0,
                "funding_status": "in_progress",
                "location_mentions": ["Damascus", "Syria"],
                "search_chunks": '[{"chunk_id":"781-0","text":"Community support in Damascus countryside."}]',
            },
            {
                "campaign_id": 999,
                "title": "Outside Syria",
                "year": "2024",
                "theme": "community",
                "beneficiary": "families",
                "funding_ratio": 0.001,
                "quality": 90.0,
                "funding_status": "in_progress",
                "location_mentions": ["Lebanon"],
                "search_chunks": '[{"chunk_id":"999-0","text":"Outside Syria."}]',
            },
        ]

        result = rag_service.answer_question("what the lowest funding campaigns in syria", top_k=2)

        self.assertEqual(result.source_campaign_ids, [781, 613])
        self.assertEqual(result.source_chunk_ids, ["781-0", "613-0"])
        self.assertIn("Lowest-funded campaigns in Syria", result.answer)
        self.assertIn("2.43% funded", result.answer)
        self.assertIn("9.74% funded", result.answer)
        mocked_get_gemini_client.assert_not_called()


class RecommendationTests(unittest.TestCase):
    @patch("app.rag_service.query_index")
    @patch("app.rag_service.get_seed_vectors_for_campaign")
    def test_recommendations_exclude_source_and_group_by_campaign(self, mocked_seed_vectors, mocked_query_index) -> None:
        mocked_seed_vectors.return_value = [
            {"chunk_id": "1-0", "values": [0.1, 0.2], "metadata": {"campaign_id": 1}},
            {"chunk_id": "1-1", "values": [0.2, 0.3], "metadata": {"campaign_id": 1}},
        ]
        mocked_query_index.side_effect = [
            {
                "matches": [
                    {
                        "id": "1-9",
                        "score": 0.99,
                        "metadata": {"campaign_id": 1, "title": "Source", "text": "same campaign"},
                    },
                    {
                        "id": "2-0",
                        "score": 0.93,
                        "metadata": {"campaign_id": 2, "title": "Other A", "text": "alpha"},
                    },
                ]
            },
            {
                "matches": [
                    {
                        "id": "2-1",
                        "score": 0.91,
                        "metadata": {"campaign_id": 2, "title": "Other A", "text": "beta"},
                    },
                    {
                        "id": "3-0",
                        "score": 0.89,
                        "metadata": {"campaign_id": 3, "title": "Other B", "text": "gamma"},
                    },
                ]
            },
        ]

        result = rag_service.recommend_similar_campaigns(campaign_id=1, top_k=3, seed_chunk_limit=2)

        self.assertEqual([item.campaign_id for item in result], [2, 3])
        self.assertEqual(result[0].score, 0.93)
        self.assertEqual(result[0].supporting_chunk_count, 2)
        self.assertEqual(sorted(result[0].source_chunk_ids), ["2-0", "2-1"])


if __name__ == "__main__":
    unittest.main()
