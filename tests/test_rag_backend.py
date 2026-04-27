from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from app.main import app
from app import rag_service
from app import reranker as reranker_module
from app.embeddings import BGE_QUERY_INSTRUCTION, embed_query
from app.pinecone_store import translate_filters
from app.schemas import RetrievedChunk
from scripts import prepare_campaigns_dataset, upload_campaign_chunks_to_pinecone


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
        mocked_embed_texts.assert_called_once_with(
            [f"{BGE_QUERY_INSTRUCTION}find medical campaigns"]
        )


class UploadEmbeddingPipelineTests(unittest.TestCase):
    def tearDown(self) -> None:
        upload_campaign_chunks_to_pinecone.load_local_model.cache_clear()

    def test_embed_text_batch_normalizes_and_returns_plain_lists(self) -> None:
        fake_embeddings = Mock()
        fake_embeddings.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
        fake_model = Mock()
        fake_model.encode.return_value = fake_embeddings

        result = upload_campaign_chunks_to_pinecone.embed_text_batch(fake_model, ["alpha", "beta"])

        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])
        fake_model.encode.assert_called_once_with(
            ["alpha", "beta"],
            batch_size=2,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    @patch("scripts.upload_campaign_chunks_to_pinecone.SentenceTransformer")
    def test_load_local_model_caches_singleton(self, mocked_sentence_transformer) -> None:
        mocked_sentence_transformer.return_value = Mock(name="cached-model")
        upload_campaign_chunks_to_pinecone.load_local_model.cache_clear()

        first = upload_campaign_chunks_to_pinecone.load_local_model()
        second = upload_campaign_chunks_to_pinecone.load_local_model()

        self.assertIs(first, second)
        mocked_sentence_transformer.assert_called_once_with(upload_campaign_chunks_to_pinecone.LOCAL_EMBEDDING_MODEL)


class ScrapedTextCleaningTests(unittest.TestCase):
    def test_clean_scraped_text_strips_html_and_preserves_readable_text(self) -> None:
        raw_text = (
            '<p><strong>You are not alone.</strong><br>Your donation helps.</p>'
            '<ul><li>Chair</li><li>Hope</li></ul>'
        )

        cleaned = prepare_campaigns_dataset.clean_scraped_text(raw_text)

        self.assertIsInstance(cleaned, str)
        self.assertNotIn("<p>", cleaned)
        self.assertNotIn("<strong>", cleaned)
        self.assertNotIn("<li>", cleaned)
        self.assertIn("You are not alone.", cleaned)
        self.assertIn("Your donation helps.", cleaned)
        self.assertIn("- Chair", cleaned)
        self.assertIn("- Hope", cleaned)

    def test_clean_scraped_text_repairs_common_mojibake_sequences(self) -> None:
        raw_text = "<p>Support \xe2\u0153\u2026 now \xf0\u0178\u2019\u2014</p>"
        cleaned = prepare_campaigns_dataset.clean_scraped_text(raw_text)

        self.assertEqual(cleaned, "Support ✅ now 💗")

    def test_build_search_text_ignores_empty_parts_after_cleaning(self) -> None:
        cleaned = prepare_campaigns_dataset.build_search_text(["Title", "<p>Body</p>", "", None])

        self.assertEqual(cleaned, "Title\n\nBody")


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


class RerankerTests(unittest.TestCase):
    def tearDown(self) -> None:
        reranker_module.get_reranker.cache_clear()

    @patch("app.reranker._is_fp16_available", return_value=True)
    def test_get_reranker_caches_singleton(self, mocked_fp16_available) -> None:
        factory = Mock(side_effect=[Mock(name="reranker-instance")])
        fake_module = types.SimpleNamespace(FlagReranker=factory)
        reranker_module.get_reranker.cache_clear()

        with patch.dict(sys.modules, {"FlagEmbedding": fake_module}):
            first = reranker_module.get_reranker()
            second = reranker_module.get_reranker()

        self.assertIs(first, second)
        factory.assert_called_once_with("BAAI/bge-reranker-base", use_fp16=True)
        mocked_fp16_available.assert_called_once()

    def test_compatible_reranker_falls_back_when_prepare_for_model_is_missing(self) -> None:
        class FakeTensor:
            def __init__(self, values):
                self._values = values

            def view(self, *_args):
                return self

            def float(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return list(self._values)

            def to(self, _device):
                return self

        class FakeTokenizer:
            def __call__(self, queries, passages, **_kwargs):
                self.queries = queries
                self.passages = passages
                return {"input_ids": FakeTensor([[1], [2]]), "attention_mask": FakeTensor([[1], [1]])}

        class FakeModel:
            def to(self, _device):
                return self

            def eval(self):
                return self

            def __call__(self, **_kwargs):
                return types.SimpleNamespace(logits=FakeTensor([0.25, 0.75]))

        backend = types.SimpleNamespace(
            tokenizer=FakeTokenizer(),
            model=FakeModel(),
            batch_size=8,
            max_length=512,
            normalize=False,
            target_devices=["cpu"],
            get_detailed_inputs=lambda pairs: pairs,
            compute_score=Mock(side_effect=AttributeError("XLMRobertaTokenizer has no attribute prepare_for_model")),
        )

        compatible = reranker_module._CompatibleFlagReranker(backend)
        scores = compatible.compute_score([("q1", "d1"), ("q2", "d2")])

        self.assertEqual(scores, [0.25, 0.75])
        backend.compute_score.assert_not_called()
        self.assertEqual(backend.tokenizer.queries, ["q1", "q2"])
        self.assertEqual(backend.tokenizer.passages, ["d1", "d2"])

    @patch("app.reranker.get_reranker")
    def test_rerank_chunks_filters_dedupes_and_scores(self, mocked_get_reranker) -> None:
        mocked_reranker = Mock()
        mocked_reranker.compute_score.return_value = [0.1, 0.9, 0.5]
        mocked_get_reranker.return_value = mocked_reranker
        chunks = [
            RetrievedChunk(chunk_id="1-0", campaign_id=1, score=0.9, quality=80, text="Alpha campaign update"),
            RetrievedChunk(chunk_id="1-1", campaign_id=1, score=0.7, quality=70, text="Alpha campaign update   "),
            RetrievedChunk(chunk_id="2-0", campaign_id=2, score=0.2, quality=90, text="Beta needs urgent care"),
            RetrievedChunk(chunk_id="3-0", campaign_id=3, score=0.4, quality=60, text="   "),
            RetrievedChunk(chunk_id="4-0", campaign_id=4, score=0.3, quality=40, text="Gamma school support"),
        ]

        result = reranker_module.rerank_chunks("help patients", chunks, top_k=3)

        self.assertEqual([chunk.chunk_id for chunk in result], ["2-0", "4-0", "1-0"])
        mocked_reranker.compute_score.assert_called_once_with(
            [
                ("help patients", "Alpha campaign update"),
                ("help patients", "Beta needs urgent care"),
                ("help patients", "Gamma school support"),
            ]
        )
        self.assertAlmostEqual(result[0].rerank_score or 0.0, 0.9)
        self.assertAlmostEqual(result[0].final_score or 0.0, 0.85 + 0.09 + 0.01)
        self.assertAlmostEqual(result[1].final_score or 0.0, 0.425 + 0.04 + 0.015)
        self.assertAlmostEqual(result[2].final_score or 0.0, 0.08 + 0.045)

    @patch("app.reranker.get_reranker")
    def test_rerank_chunks_handles_equal_scores_with_tie_breakers(self, mocked_get_reranker) -> None:
        mocked_reranker = Mock()
        mocked_reranker.compute_score.return_value = [5.0, 5.0]
        mocked_get_reranker.return_value = mocked_reranker
        chunks = [
            RetrievedChunk(chunk_id="1-0", campaign_id=1, score=0.2, quality=90, text="Alpha"),
            RetrievedChunk(chunk_id="2-0", campaign_id=2, score=0.8, quality=20, text="Beta"),
        ]

        result = reranker_module.rerank_chunks("campaign", chunks, top_k=2)

        self.assertEqual([chunk.chunk_id for chunk in result], ["1-0", "2-0"])
        self.assertAlmostEqual(result[0].final_score or 0.0, 0.09 + 0.01)
        self.assertAlmostEqual(result[1].final_score or 0.0, 0.02 + 0.04)

    @patch("app.reranker.get_reranker")
    def test_rerank_chunks_caps_candidates_and_trims_top_k(self, mocked_get_reranker) -> None:
        mocked_reranker = Mock()
        mocked_reranker.compute_score.return_value = list(range(50))
        mocked_get_reranker.return_value = mocked_reranker
        chunks = [
            RetrievedChunk(
                chunk_id=f"{index}-0",
                campaign_id=index,
                score=float(index) / 100,
                quality=50,
                text=f"Campaign text {index}",
            )
            for index in range(60)
        ]

        result = reranker_module.rerank_chunks("campaign", chunks, top_k=3)
        pairs = mocked_reranker.compute_score.call_args.args[0]

        self.assertEqual(len(pairs), 50)
        self.assertEqual(len(result), 3)
        self.assertEqual([chunk.chunk_id for chunk in result], ["49-0", "48-0", "47-0"])

    def test_build_rerank_debug_scores_limits_to_top_ten_and_excludes_text(self) -> None:
        chunks = [
            RetrievedChunk(
                chunk_id=f"{index}-0",
                score=0.1 * index,
                rerank_score=1.0 * index,
                final_score=2.0 * index,
                text=f"text {index}",
            )
            for index in range(12)
        ]

        result = reranker_module.build_rerank_debug_scores(chunks)

        self.assertEqual(len(result), 10)
        self.assertEqual(result[0], {"chunk_id": "0-0", "vector_score": 0.0, "rerank_score": 0.0, "final_score": 0.0})
        self.assertNotIn("text", result[0])


class QueryUnderstandingTests(unittest.TestCase):
    def tearDown(self) -> None:
        rag_service.load_campaign_analytics_rows.cache_clear()
        rag_service.get_known_locations.cache_clear()
        rag_service.get_known_themes.cache_clear()
        rag_service.get_known_beneficiaries.cache_clear()

    def test_parse_query_matches_supported_phrases_with_word_boundaries(self) -> None:
        self.assertEqual(rag_service.parse_query("medical campaigns"), {"theme": "medical"})
        self.assertEqual(rag_service.parse_query("MEDICAL campaigns"), {"theme": "medical"})
        self.assertEqual(rag_service.parse_query("biomedical campaigns"), {})
        self.assertEqual(rag_service.parse_query("education support"), {"theme": "education"})
        self.assertEqual(
            rag_service.parse_query("low funding medical campaigns"),
            {"funding_ratio": {"lt": 1}, "theme": "medical"},
        )
        self.assertEqual(rag_service.parse_query("under funded campaigns"), {"funding_ratio": {"lt": 1}})
        self.assertEqual(rag_service.parse_query("underfunded campaigns"), {"funding_ratio": {"lt": 1}})

    @patch("app.rag_service.datetime")
    def test_parse_query_computes_recent_year_dynamically(self, mocked_datetime) -> None:
        mocked_datetime.now.return_value.year = 2026

        self.assertEqual(
            rag_service.parse_query("high quality recent medical campaigns"),
            {
                "quality": {"gte": 80},
                "theme": "medical",
                "year": {"gte": "2024"},
            },
        )

    def test_clean_query_removes_only_detected_phrases_and_falls_back(self) -> None:
        self.assertEqual(rag_service.clean_query("low funding medical campaigns"), "campaigns")
        self.assertEqual(rag_service.clean_query("need high quality education support"), "need support")
        self.assertEqual(rag_service.clean_query("biomedical campaigns"), "biomedical campaigns")
        self.assertEqual(rag_service.clean_query("under funded"), "campaign")
        self.assertEqual(rag_service.clean_query("recent"), "campaign")

    def test_build_question_plan_detects_low_funding_before_year(self) -> None:
        plan = rag_service.build_question_plan("give me low funding campaigns before 2024")

        self.assertEqual(plan.answer_mode, "structured_ranking")
        self.assertEqual(plan.sort_direction, "asc")
        self.assertEqual(plan.funding_intent, "underfunded")
        self.assertEqual(plan.year_constraint, rag_service.YearConstraint(operator="lt", year=2024))

    def test_build_question_plan_detects_failed_to_reach_goal_and_count(self) -> None:
        plan = rag_service.build_question_plan("2 campaigns that failed to reach their goal")

        self.assertEqual(plan.answer_mode, "structured_ranking")
        self.assertEqual(plan.funding_intent, "failed_to_reach_goal")
        self.assertEqual(plan.sort_direction, "asc")
        self.assertEqual(plan.requested_count, 2)

    @patch("app.rag_service.get_known_locations", return_value={"syria": "Syria"})
    def test_build_question_plan_detects_highest_funded_location_and_year(self, mocked_locations) -> None:
        plan = rag_service.build_question_plan("highest funded campaigns in syria before 2025")

        self.assertEqual(plan.answer_mode, "structured_ranking")
        self.assertEqual(plan.sort_direction, "desc")
        self.assertEqual(plan.location_filters, ("Syria",))
        self.assertEqual(plan.year_constraint, rag_service.YearConstraint(operator="lt", year=2025))
        mocked_locations.assert_called_once()

    def test_build_question_plan_leaves_open_ended_question_for_rag(self) -> None:
        plan = rag_service.build_question_plan("what is this campaign funding?")

        self.assertEqual(plan.answer_mode, "rag_generation")


class RetrievalAndAnswerTests(unittest.TestCase):
    def tearDown(self) -> None:
        rag_service.load_campaign_analytics_rows.cache_clear()
        rag_service.get_known_locations.cache_clear()
        rag_service.get_known_themes.cache_clear()
        rag_service.get_known_beneficiaries.cache_clear()

    @patch("app.rag_service.rerank_chunks")
    @patch("app.rag_service.query_index")
    @patch("app.rag_service.embed_query", return_value=[0.1, 0.2])
    def test_retrieve_chunks_queries_pinecone_and_normalizes_results(
        self,
        mocked_embed_query,
        mocked_query_index,
        mocked_rerank_chunks,
    ) -> None:
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
        mocked_rerank_chunks.side_effect = lambda query, chunks, top_k: chunks[:top_k]

        result = rag_service.retrieve_chunks("help patients", filters={"theme": "medical"}, top_k=4)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].campaign_id, 10)
        self.assertEqual(result[0].theme, "medical")
        mocked_embed_query.assert_called_once_with("help patients")
        mocked_query_index.assert_called_once_with(vector=[0.1, 0.2], top_k=20, filters={"theme": "medical"})
        rerank_query, rerank_chunks = mocked_rerank_chunks.call_args.args
        rerank_top_k = mocked_rerank_chunks.call_args.kwargs["top_k"]
        self.assertEqual(rerank_query, "help patients")
        self.assertEqual(rerank_top_k, 4)
        self.assertEqual(len(rerank_chunks), 1)
        self.assertEqual(rerank_chunks[0].chunk_id, "10-0")

    @patch.dict(os.environ, {"RAG_DEBUG": "true"}, clear=False)
    @patch("app.rag_service.logger")
    @patch("app.rag_service.rerank_chunks", return_value=[])
    @patch("app.rag_service.query_index")
    @patch("app.rag_service.embed_query", return_value=[0.3, 0.4])
    def test_retrieve_chunks_merges_query_filters_and_logs_when_debug_enabled(
        self,
        mocked_embed_query,
        mocked_query_index,
        mocked_rerank_chunks,
        mocked_logger,
    ) -> None:
        mocked_query_index.return_value = {"matches": []}

        with patch("app.rag_service.datetime") as mocked_datetime:
            mocked_datetime.now.return_value.year = 2026
            result = rag_service.retrieve_chunks(
                "high quality recent underfunded medical campaigns",
                filters={"theme": "education", "funding_ratio": {"gte": 0.2}},
                top_k=3,
            )

        self.assertEqual(result, [])
        mocked_embed_query.assert_called_once_with("campaigns")
        mocked_query_index.assert_called_once_with(
            vector=[0.3, 0.4],
            top_k=20,
            filters={
                "theme": "education",
                "funding_ratio": {"gte": 0.2, "lt": 1},
                "quality": {"gte": 80},
                "year": {"gte": "2024"},
            },
        )
        mocked_rerank_chunks.assert_called_once_with("campaigns", [], top_k=3)
        mocked_logger.info.assert_called_once_with(
            {
                "raw_query": "high quality recent underfunded medical campaigns",
                "cleaned_query": "campaigns",
                "initial_k": 20,
                "extracted_filters": {
                    "funding_ratio": {"lt": 1},
                    "quality": {"gte": 80},
                    "theme": "medical",
                    "year": {"gte": "2024"},
                },
                "final_filters": {
                    "theme": "education",
                    "funding_ratio": {"gte": 0.2, "lt": 1},
                    "quality": {"gte": 80},
                    "year": {"gte": "2024"},
                },
                "top_k": 3,
                "scores": [],
            }
        )

    @patch.dict(os.environ, {"RAG_DEBUG": "false"}, clear=False)
    @patch("app.rag_service.logger")
    @patch("app.rag_service.rerank_chunks", return_value=[])
    @patch("app.rag_service.query_index")
    @patch("app.rag_service.embed_query", return_value=[0.5, 0.6])
    def test_retrieve_chunks_uses_campaign_fallback_and_skips_logging_when_debug_disabled(
        self,
        mocked_embed_query,
        mocked_query_index,
        mocked_rerank_chunks,
        mocked_logger,
    ) -> None:
        mocked_query_index.return_value = {"matches": []}

        result = rag_service.retrieve_chunks("recent", filters=None, top_k=2)

        self.assertEqual(result, [])
        mocked_embed_query.assert_called_once_with("campaign")
        mocked_query_index.assert_called_once_with(
            vector=[0.5, 0.6],
            top_k=20,
            filters={"year": {"gte": str(rag_service.datetime.now().year - 2)}},
        )
        mocked_rerank_chunks.assert_called_once_with("campaign", [], top_k=2)
        mocked_logger.info.assert_not_called()

    @patch("app.rag_service.rerank_chunks")
    @patch("app.rag_service.query_index")
    @patch("app.rag_service.embed_query", return_value=[0.7, 0.8])
    def test_retrieve_chunks_preserves_user_lt_filter_over_query_inference(
        self,
        mocked_embed_query,
        mocked_query_index,
        mocked_rerank_chunks,
    ) -> None:
        mocked_query_index.return_value = {"matches": []}
        mocked_rerank_chunks.return_value = []

        result = rag_service.retrieve_chunks(
            "underfunded campaigns",
            filters={"funding_ratio": {"lt": 0.5}},
            top_k=1,
        )

        self.assertEqual(result, [])
        mocked_embed_query.assert_called_once_with("campaigns")
        mocked_query_index.assert_called_once_with(
            vector=[0.7, 0.8],
            top_k=20,
            filters={"funding_ratio": {"lt": 0.5}},
        )
        mocked_rerank_chunks.assert_called_once_with("campaigns", [], top_k=1)

    @patch("app.rag_service.rerank_chunks", return_value=[])
    @patch("app.rag_service.query_index", return_value={"matches": []})
    @patch("app.rag_service.embed_query", return_value=[0.9, 1.0])
    def test_retrieve_chunks_uses_adaptive_initial_k_for_medium_top_k(
        self,
        mocked_embed_query,
        mocked_query_index,
        mocked_rerank_chunks,
    ) -> None:
        rag_service.retrieve_chunks("campaigns", top_k=8)

        mocked_embed_query.assert_called_once_with("campaigns")
        mocked_query_index.assert_called_once_with(vector=[0.9, 1.0], top_k=30, filters=None)
        mocked_rerank_chunks.assert_called_once_with("campaigns", [], top_k=8)

    @patch("app.rag_service.rerank_chunks", return_value=[])
    @patch("app.rag_service.query_index", return_value={"matches": []})
    @patch("app.rag_service.embed_query", return_value=[1.1, 1.2])
    def test_retrieve_chunks_uses_adaptive_initial_k_for_large_top_k(
        self,
        mocked_embed_query,
        mocked_query_index,
        mocked_rerank_chunks,
    ) -> None:
        rag_service.retrieve_chunks("campaigns", top_k=12)

        mocked_embed_query.assert_called_once_with("campaigns")
        mocked_query_index.assert_called_once_with(vector=[1.1, 1.2], top_k=50, filters=None)
        mocked_rerank_chunks.assert_called_once_with("campaigns", [], top_k=12)

    @patch("app.rag_service.logger")
    @patch("app.rag_service.rerank_chunks", side_effect=RuntimeError("reranker unavailable"))
    @patch("app.rag_service.query_index")
    @patch("app.rag_service.embed_query", return_value=[0.11, 0.22])
    def test_retrieve_chunks_falls_back_to_vector_ranking_when_reranker_fails(
        self,
        mocked_embed_query,
        mocked_query_index,
        mocked_rerank_chunks,
        mocked_logger,
    ) -> None:
        mocked_query_index.return_value = {
            "matches": [
                {"id": "1-0", "score": 0.9, "metadata": {"campaign_id": 1, "text": "alpha"}},
                {"id": "2-0", "score": 0.8, "metadata": {"campaign_id": 2, "text": "beta"}},
            ]
        }

        result = rag_service.retrieve_chunks("campaigns", top_k=1)

        self.assertEqual([chunk.chunk_id for chunk in result], ["1-0"])
        mocked_embed_query.assert_called_once_with("campaigns")
        mocked_query_index.assert_called_once_with(vector=[0.11, 0.22], top_k=20, filters=None)
        mocked_rerank_chunks.assert_called_once()
        mocked_logger.exception.assert_called_once_with("Reranking failed; falling back to vector-ranked results.")

    @patch("app.rag_service.get_gemini_client")
    @patch("app.rag_service.retrieve_chunks")
    @patch("app.rag_service.answer_structured_campaign_question", return_value=None)
    def test_answer_question_returns_grounded_result_with_sources(
        self,
        mocked_structured_answer,
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
        mocked_structured_answer.assert_called_once()

    @patch("app.rag_service.get_gemini_client")
    @patch("app.rag_service.retrieve_chunks")
    @patch("app.rag_service.answer_structured_campaign_question", return_value=None)
    def test_answer_question_includes_question_interpretation_in_prompt(
        self,
        mocked_structured_answer,
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

        rag_service.answer_question("What is this campaign funding?")

        prompt = mocked_model.generate_content.call_args.kwargs["contents"]
        self.assertIn("Question interpretation:", prompt)
        self.assertIn("Answer mode: rag_generation", prompt)
        mocked_structured_answer.assert_called_once()

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

    @patch("app.rag_service.get_known_beneficiaries", return_value={})
    @patch("app.rag_service.get_known_themes", return_value={})
    @patch("app.rag_service.get_known_locations", return_value={})
    @patch("app.rag_service.get_gemini_client")
    @patch("app.rag_service.load_campaign_analytics_rows")
    def test_answer_question_returns_low_funding_campaigns_before_2024_from_structured_data(
        self,
        mocked_load_campaign_analytics_rows,
        mocked_get_gemini_client,
        mocked_locations,
        mocked_themes,
        mocked_beneficiaries,
    ) -> None:
        mocked_load_campaign_analytics_rows.return_value = [
            {
                "campaign_id": 101,
                "title": "Before 2024 Lowest",
                "year": "2023",
                "theme": "medical",
                "beneficiary": "patients",
                "funding_ratio": 0.10,
                "quality": 88.0,
                "funding_status": "in_progress",
                "location_mentions": [],
                "search_chunks": '[{"chunk_id":"101-0","text":"Support in 2023."}]',
            },
            {
                "campaign_id": 102,
                "title": "Before 2024 Higher",
                "year": "2022",
                "theme": "medical",
                "beneficiary": "patients",
                "funding_ratio": 0.20,
                "quality": 90.0,
                "funding_status": "in_progress",
                "location_mentions": [],
                "search_chunks": '[{"chunk_id":"102-0","text":"Support in 2022."}]',
            },
            {
                "campaign_id": 103,
                "title": "Year 2024 Excluded",
                "year": "2024",
                "theme": "medical",
                "beneficiary": "patients",
                "funding_ratio": 0.05,
                "quality": 95.0,
                "funding_status": "in_progress",
                "location_mentions": [],
                "search_chunks": '[{"chunk_id":"103-0","text":"Support in 2024."}]',
            },
        ]

        result = rag_service.answer_question("give me low funding campaigns before 2024", top_k=5)

        self.assertEqual(result.source_campaign_ids, [101, 102])
        self.assertNotIn(103, result.source_campaign_ids)
        self.assertIn("Lowest-funded campaigns before 2024 by funding ratio:", result.answer)
        self.assertIn("Before 2024 Lowest", result.answer)
        self.assertIn("Before 2024 Higher", result.answer)
        mocked_get_gemini_client.assert_not_called()
        mocked_locations.assert_called_once()
        mocked_themes.assert_called_once()
        mocked_beneficiaries.assert_called_once()

    @patch("app.rag_service.get_known_beneficiaries", return_value={})
    @patch("app.rag_service.get_known_themes", return_value={})
    @patch("app.rag_service.get_known_locations", return_value={})
    @patch("app.rag_service.get_gemini_client")
    @patch("app.rag_service.load_campaign_analytics_rows")
    def test_answer_question_failed_to_reach_goal_uses_underfunded_and_count(
        self,
        mocked_load_campaign_analytics_rows,
        mocked_get_gemini_client,
        mocked_locations,
        mocked_themes,
        mocked_beneficiaries,
    ) -> None:
        mocked_load_campaign_analytics_rows.return_value = [
            {
                "campaign_id": 201,
                "title": "Under Goal 1",
                "year": "2023",
                "theme": "community",
                "beneficiary": "families",
                "funding_ratio": 0.12,
                "quality": 88.0,
                "funding_status": "in_progress",
                "location_mentions": [],
                "search_chunks": '[{"chunk_id":"201-0","text":"Support 1."}]',
            },
            {
                "campaign_id": 202,
                "title": "Under Goal 2",
                "year": "2022",
                "theme": "community",
                "beneficiary": "families",
                "funding_ratio": 0.20,
                "quality": 90.0,
                "funding_status": "in_progress",
                "location_mentions": [],
                "search_chunks": '[{"chunk_id":"202-0","text":"Support 2."}]',
            },
            {
                "campaign_id": 203,
                "title": "Fully Funded Excluded",
                "year": "2021",
                "theme": "community",
                "beneficiary": "families",
                "funding_ratio": 1.10,
                "quality": 95.0,
                "funding_status": "overfunded",
                "location_mentions": [],
                "search_chunks": '[{"chunk_id":"203-0","text":"Support 3."}]',
            },
        ]

        result = rag_service.answer_question("2 campaigns that failed to reach their goal", top_k=5)

        self.assertEqual(result.source_campaign_ids, [201, 202])
        self.assertEqual(len(result.retrieved_chunks), 2)
        self.assertIn("Campaigns that failed to reach their goal by funding ratio:", result.answer)
        self.assertNotIn("Fully Funded Excluded", result.answer)
        mocked_get_gemini_client.assert_not_called()
        mocked_locations.assert_called_once()
        mocked_themes.assert_called_once()
        mocked_beneficiaries.assert_called_once()

    @patch("app.rag_service.get_known_beneficiaries", return_value={})
    @patch("app.rag_service.get_known_themes", return_value={})
    @patch("app.rag_service.get_known_locations", return_value={"syria": "Syria"})
    @patch("app.rag_service.get_gemini_client")
    @patch("app.rag_service.load_campaign_analytics_rows")
    def test_answer_question_highest_funded_in_location_before_year_enforces_filters(
        self,
        mocked_load_campaign_analytics_rows,
        mocked_get_gemini_client,
        mocked_locations,
        mocked_themes,
        mocked_beneficiaries,
    ) -> None:
        mocked_load_campaign_analytics_rows.return_value = [
            {
                "campaign_id": 301,
                "title": "Syria Winner",
                "year": "2024",
                "theme": "community",
                "beneficiary": "families",
                "funding_ratio": 0.90,
                "quality": 91.0,
                "funding_status": "in_progress",
                "location_mentions": ["Syria"],
                "search_chunks": '[{"chunk_id":"301-0","text":"Support Syria."}]',
            },
            {
                "campaign_id": 302,
                "title": "Syria Excluded By Year",
                "year": "2025",
                "theme": "community",
                "beneficiary": "families",
                "funding_ratio": 0.95,
                "quality": 92.0,
                "funding_status": "in_progress",
                "location_mentions": ["Syria"],
                "search_chunks": '[{"chunk_id":"302-0","text":"Support Syria 2025."}]',
            },
            {
                "campaign_id": 303,
                "title": "Lebanon Excluded",
                "year": "2024",
                "theme": "community",
                "beneficiary": "families",
                "funding_ratio": 0.99,
                "quality": 90.0,
                "funding_status": "in_progress",
                "location_mentions": ["Lebanon"],
                "search_chunks": '[{"chunk_id":"303-0","text":"Support Lebanon."}]',
            },
        ]

        result = rag_service.answer_question("highest funded campaigns in syria before 2025", top_k=5)

        self.assertEqual(result.source_campaign_ids, [301])
        self.assertIn("Highest-funded campaigns in Syria before 2025 by funding ratio:", result.answer)
        self.assertNotIn("Syria Excluded By Year", result.answer)
        self.assertNotIn("Lebanon Excluded", result.answer)
        mocked_get_gemini_client.assert_not_called()
        mocked_locations.assert_called_once()
        mocked_themes.assert_called_once()
        mocked_beneficiaries.assert_called_once()

    @patch("app.rag_service.get_known_beneficiaries", return_value={})
    @patch("app.rag_service.get_known_themes", return_value={})
    @patch("app.rag_service.get_known_locations", return_value={})
    @patch("app.rag_service.get_gemini_client")
    @patch("app.rag_service.load_campaign_analytics_rows", return_value=[])
    def test_answer_question_returns_no_match_message_for_structured_query(
        self,
        mocked_load_campaign_analytics_rows,
        mocked_get_gemini_client,
        mocked_locations,
        mocked_themes,
        mocked_beneficiaries,
    ) -> None:
        result = rag_service.answer_question("give me low funding campaigns before 2024", top_k=5)

        self.assertEqual(result.answer, "No matching campaigns were found in the campaign data for the requested criteria.")
        self.assertEqual(result.retrieved_chunks, [])
        self.assertEqual(result.source_campaign_ids, [])
        self.assertEqual(result.source_chunk_ids, [])
        mocked_get_gemini_client.assert_not_called()
        mocked_load_campaign_analytics_rows.assert_called_once()
        mocked_locations.assert_called_once()
        mocked_themes.assert_called_once()
        mocked_beneficiaries.assert_called_once()


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
