import unittest

from rag_utils import (
    SYSTEM_PROMPT,
    build_context,
    expand_source_citations,
    rerank_search_results,
)


AUTOMATIC_SECTION = "1-1. Automatic startup procedure"
RESULTS = [
    {
        "id": "toc-chunk",
        "content": (
            "1-1. Automatic startup procedure "
            "................................................ 3"
        ),
        "source_file": "document1.pdf",
        "source_url": "https://sharepoint.example/document1.pdf",
        "location": "Page 2",
        "section": AUTOMATIC_SECTION,
        "chunk_id": "3",
        "score": 0.04,
    },
    {
        "id": "procedure-chunk",
        "content": (
            "Type number change: enter the new type number on the "
            "operation screen.\n"
            "7) Confirm the selected type number.\n"
            "8) Press the automatic operation start switch.\n"
            "9) Confirm that the machine starts automatically."
        ),
        "source_file": "document1.pdf",
        "source_url": "https://sharepoint.example/document1.pdf",
        "location": "Page 3",
        "section": AUTOMATIC_SECTION,
        "chunk_id": "31",
        "score": 0.03,
    },
    {
        "id": "manual-chunk",
        "content": "1) Set manual mode and press the operation button.",
        "source_file": "document1.pdf",
        "source_url": "https://sharepoint.example/document1.pdf",
        "location": "Page 8",
        "section": "3-2. Startup procedure of manual operation",
        "chunk_id": "52",
        "score": 0.05,
    },
    {
        "id": "return-chunk",
        "content": "1) Press the return-start switch.",
        "source_file": "document1.pdf",
        "source_url": "https://sharepoint.example/document1.pdf",
        "location": "Page 4",
        "section": "2-1. Return startup procedure",
        "chunk_id": "40",
        "score": 0.06,
    },
]


class RetrievalBehaviorTests(unittest.TestCase):
    def test_exact_automatic_startup_title_ranks_procedure_over_toc(self):
        ranked = rerank_search_results(
            "Automatic startup procedure",
            RESULTS,
            top_k=4,
        )
        self.assertEqual(ranked[0]["id"], "procedure-chunk")
        self.assertLess(
            next(i for i, hit in enumerate(ranked) if hit["id"] == "procedure-chunk"),
            next(i for i, hit in enumerate(ranked) if hit["id"] == "toc-chunk"),
        )

    def test_step_by_step_question_retrieves_numbered_source_steps(self):
        ranked = rerank_search_results(
            "Explain the automatic startup procedure step by step",
            RESULTS,
            top_k=3,
        )
        self.assertEqual(ranked[0]["id"], "procedure-chunk")
        self.assertIn("7)", ranked[0]["content"])
        self.assertIn("8)", ranked[0]["content"])
        self.assertIn("9)", ranked[0]["content"])

    def test_manual_operation_query_prefers_manual_section(self):
        ranked = rerank_search_results(
            "The startup procedure of manual operation",
            RESULTS,
            top_k=3,
        )
        self.assertEqual(ranked[0]["id"], "manual-chunk")

    def test_unsupported_fact_policy_and_actual_metadata_citations(self):
        source = RESULTS[1]
        context = build_context([source])
        self.assertIn("[Source 1: document1.pdf; Page 3", context)
        self.assertIn(f"Section: {AUTOMATIC_SECTION}", context)
        self.assertNotIn(source["source_url"], context)
        self.assertIn(
            "available excerpts do not establish it",
            SYSTEM_PROMPT,
        )
        self.assertIn("never claim the entire document lacks", SYSTEM_PROMPT)
        self.assertIn("never renumber steps", SYSTEM_PROMPT)
        self.assertIn("do not mix in manual-operation", SYSTEM_PROMPT)

        answer, citations = expand_source_citations(
            "The excerpt does not establish the maximum temperature. "
            "[Source 1] [Source 1]",
            [source],
        )
        self.assertEqual(
            answer,
            "The excerpt does not establish the maximum temperature. "
            "[Source 1] [Source 1]",
        )
        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["id"], "procedure-chunk")
        self.assertEqual(citations[0]["chunk_id"], "31")
        self.assertEqual(citations[0]["location"], "Page 3")
        self.assertEqual(citations[0]["source_file"], "document1.pdf")
        self.assertEqual(citations[0]["section"], AUTOMATIC_SECTION)
        self.assertEqual(citations[0]["source_url"], source["source_url"])
        self.assertEqual(citations[0]["content"], source["content"])

        with self.assertRaisesRegex(ValueError, "unavailable source 2"):
            expand_source_citations("Unsupported [Source 2]", [source])


if __name__ == "__main__":
    unittest.main()
