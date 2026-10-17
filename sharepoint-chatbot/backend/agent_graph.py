"""
LangGraph agent: start -> agent <-> tools -> end

Tools:
  - rag_search: retrieve document chunks (your existing search_documents)
  - general_query: optional lightweight tool for non-document answers / clarifications

LLM goes through LiteLLM (Groq by default). Agent can call any tool any number of times.
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Any, Literal, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from litellm import completion

# ---------------------------------------------------------------------------
# Optional import of your existing search layer (keeps graph independent)
# ---------------------------------------------------------------------------
try:
    from search_service import search_documents
except ImportError:
    def search_documents(query: str, top_k: int = 5) -> list[dict]:
        """Fallback stub when search_service is not available."""
        return [
            {
                "content": f"[stub] No index available for query: {query}",
                "source": "stub",
                "score": 0.0,
            }
        ]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def rag_search(query: str, top_k: int = 5) -> str:
    """
    Retrieve relevant document chunks from the indexed knowledge base
    (local documents + SharePoint cache). Use this whenever the user
    asks about company documents, policies, presentations, PDFs, etc.

    Args:
        query: Natural language search query.
        top_k: Number of chunks to return (1-10).
    """
    top_k = max(1, min(10, int(top_k)))
    results = search_documents(query, top_k=top_k)
    if not results:
        return json.dumps({"found": False, "chunks": []})

    # Normalize to a stable shape for the LLM
    chunks = []
    for i, r in enumerate(results):
        chunks.append(
            {
                "id": i + 1,
                "content": r.get("content") or r.get("text") or str(r),
                "source": r.get("source") or r.get("filename") or r.get("path") or "unknown",
                "score": r.get("score"),
                "metadata": {
                    k: v
                    for k, v in r.items()
                    if k not in ("content", "text", "score")
                },
            }
        )
    return json.dumps({"found": True, "chunks": chunks}, ensure_ascii=False)


@tool
def general_query(question: str) -> str:
    """
    Answer general knowledge or reasoning questions that do NOT require
    looking up indexed documents. Use only when the question is clearly
    outside the document corpus (e.g. math, definitions, pure reasoning).

    Args:
        question: The general question to answer briefly.
    """
    model = os.getenv("LITELLM_MODEL", "groq/openai/gpt-oss-20b")
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("LITELLM_API_KEY")
    resp = completion(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Answer briefly and accurately. No document lookup.",
            },
            {"role": "user", "content": question},
        ],
        api_key=api_key,
        temperature=0.2,
        max_tokens=512,
    )
    return resp.choices[0].message.content or ""


TOOLS = [rag_search, general_query]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    sources: list[dict]
    citations: list[Any]


# ---------------------------------------------------------------------------
# LiteLLM-backed chat model (tool-calling)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a helpful document assistant.

You have two tools:
1. rag_search – use this to fetch relevant chunks from the company document index
   (PDFs, PowerPoints, SharePoint, etc.). Always prefer this for any question
   that might be answered from documents.
2. general_query – only for pure general knowledge / reasoning that clearly
   does not need documents.

Rules:
- Call tools when you need information. You may call the same tool multiple times
  with different queries if needed.
- After you have enough context, answer the user clearly and cite sources by
  filename or source name when you used rag_search.
- If documents do not contain the answer, say so honestly.
- Do not invent document content.
"""


def _lc_messages_to_openai(messages: Sequence[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to OpenAI/LiteLLM format."""
    out: list[dict] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            msg: dict[str, Any] = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"])
                            if isinstance(tc["args"], dict)
                            else tc["args"],
                        },
                    }
                    for tc in m.tool_calls
                ]
            out.append(msg)
        elif isinstance(m, ToolMessage):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content if isinstance(m.content, str) else json.dumps(m.content),
                }
            )
        else:
            role = getattr(m, "type", "user")
            out.append({"role": role, "content": getattr(m, "content", str(m))})
    return out


def _tool_parameters(tool) -> dict:
    """Extract JSON Schema parameters from a LangChain tool (Pydantic v1/v2 safe)."""
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is None:
        return {"type": "object", "properties": {}}
    if hasattr(args_schema, "model_json_schema"):
        params = args_schema.model_json_schema()
    elif hasattr(args_schema, "schema"):
        params = args_schema.schema()
    else:
        return {"type": "object", "properties": {}}
    params.pop("title", None)
    params.pop("$defs", None)
    params.pop("definitions", None)
    if "properties" in params:
        for prop in params["properties"].values():
            if isinstance(prop, dict):
                prop.pop("title", None)
    return params


def _openai_tools_schema() -> list[dict]:
    """Build OpenAI-style tool schemas from LangChain tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": _tool_parameters(t),
            },
        }
        for t in TOOLS
    ]


def call_model(state: AgentState) -> dict:
    """Agent node: call LLM via LiteLLM with tools bound."""
    model = os.getenv("LITELLM_MODEL", "groq/openai/gpt-oss-20b")
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("LITELLM_API_KEY")

    messages = list(state["messages"])
    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    openai_messages = _lc_messages_to_openai(messages)
    tools_schema = _openai_tools_schema()

    resp = completion(
        model=model,
        messages=openai_messages,
        tools=tools_schema,
        tool_choice="auto",
        api_key=api_key,
        temperature=0.2,
        max_tokens=2048,
    )

    choice = resp.choices[0].message
    content = choice.content or ""
    tool_calls = []

    if getattr(choice, "tool_calls", None):
        for tc in choice.tool_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            tool_calls.append(
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": args,
                }
            )

    ai_msg = AIMessage(content=content, tool_calls=tool_calls)
    return {"messages": [ai_msg]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Route: if last AI message has tool_calls -> tools, else -> end."""
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


# ---------------------------------------------------------------------------
# Build graph: START -> agent <-> tools -> END
# ---------------------------------------------------------------------------

def build_agent_graph():
    tool_node = ToolNode(TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")  # loop back

    return graph.compile()


agent_graph = build_agent_graph()


# ---------------------------------------------------------------------------
# Public runner used by FastAPI
# ---------------------------------------------------------------------------

def run_agent(question: str) -> dict:
    """
    Execute the agent graph for a single user question.

    Returns:
        {
            "answer": str,
            "sources": list[dict],
            "citations": list
        }
    """
    initial: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ],
        "sources": [],
        "citations": [],
    }

    final_state = agent_graph.invoke(initial)

    answer = ""
    sources: list[dict] = []

    for msg in final_state["messages"]:
        if isinstance(msg, AIMessage) and not msg.tool_calls:
            answer = msg.content or answer
        if isinstance(msg, ToolMessage) and msg.name == "rag_search":
            try:
                payload = json.loads(msg.content)
                if payload.get("found") and payload.get("chunks"):
                    sources.extend(payload["chunks"])
            except (json.JSONDecodeError, TypeError):
                pass

    seen = set()
    unique_sources = []
    for s in sources:
        key = (s.get("source"), (s.get("content") or "")[:120])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return {
        "answer": answer or "I could not produce an answer.",
        "sources": unique_sources,
        "citations": [],
    }