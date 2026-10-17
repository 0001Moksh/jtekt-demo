import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import agent_graph
from agent_graph import SYSTEM_PROMPT, agent_graph, prepare_query, rag_search, run_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class AgentThreadMemoryTests(unittest.TestCase):
    def test_run_agent_includes_history_before_current_question(self):
        history = [
            HumanMessage(content="My machine is model X."),
            AIMessage(content="Understood."),
        ]
        final_messages = [
            SystemMessage(content="system"),
            *history,
            HumanMessage(content="What model did I mention?"),
            AIMessage(content="You mentioned model X."),
        ]

        with patch.object(
            agent_graph,
            "invoke",
            return_value={"messages": final_messages},
        ) as invoke:
            result = run_agent("What model did I mention?", history=history)

        actual_messages = invoke.call_args.args[0]["messages"]
        self.assertEqual(
            [(message.type, message.content) for message in actual_messages],
            [
                ("system", SYSTEM_PROMPT),
                ("human", "My machine is model X."),
                ("ai", "Understood."),
                ("human", "What model did I mention?"),
            ],
        )
        self.assertEqual(result["answer"], "You mentioned model X.")

    @patch("agent_graph.completion")
    def test_prepare_query_uses_follow_up_context(self, completion):
        completion.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content="automatic startup procedure steps after the initial setup"
            ))]
        )

        query = prepare_query.invoke({
            "question": "What happens after that?",
            "conversation_context": (
                "User: Explain the automatic startup procedure.\n"
                "Assistant: First select the operation mode."
            ),
        })

        self.assertEqual(
            query,
            "automatic startup procedure steps after the initial setup",
        )
        self.assertIn(
            "Explain the automatic startup procedure.",
            completion.call_args.kwargs["messages"][1]["content"],
        )
        self.assertIn(
            "What happens after that?",
            completion.call_args.kwargs["messages"][1]["content"],
        )

    @patch("agent_graph.search_documents")
    def test_rag_search_exposes_source_metadata_for_preview(self, search_documents):
        search_documents.return_value = [{
            "id": "chunk-1",
            "content": "Step one.",
            "source_file": "procedure.pdf",
            "source_url": "https://example.test/procedure.pdf",
            "location": "Page 3",
            "section": "Startup",
            "chunk_id": "1",
            "document_id": "doc-1",
        }]

        result = json.loads(rag_search.invoke({"query": "startup procedure"}))

        chunk = result["chunks"][0]
        for key in (
            "source_file",
            "source_url",
            "location",
            "section",
            "chunk_id",
            "document_id",
        ):
            self.assertIn(key, chunk)
        self.assertEqual(chunk["source_file"], "procedure.pdf")
        self.assertEqual(chunk["content"], "Step one.")


if __name__ == "__main__":
    unittest.main()
