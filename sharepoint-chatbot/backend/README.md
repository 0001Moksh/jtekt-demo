# Backend configuration and operations

The FastAPI backend preserves the LangGraph `agent -> tools -> agent` flow and
the existing Azure AI Search retrieval implementation. All model calls pass
through `llm_gateway.py`; no gateway or search credential belongs in the
frontend.

## Configuration

Copy `.env.example` to `.env` and replace the safe placeholders. The checked-in
example contains no usable credentials.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LITELLM_GATEWAY_MODE` | `sdk` | `sdk` uses LiteLLM directly; `proxy` requires `LITELLM_PROXY_URL` and uses that URL for primary and fallback routes. |
| `LITELLM_MODEL` | `groq/openai/gpt-oss-20b` | Primary tool-calling model. The provider must support the LiteLLM tool-call format and its corresponding provider credential must be configured. |
| `LITELLM_FALLBACK_MODELS` | empty | Up to three comma-separated model names attempted only after retryable primary failures. Availability and credentials are the operator's responsibility; routing decisions are logged. |
| `GROQ_API_KEY`, `LITELLM_API_KEY` | unset | Server-side provider credential; LiteLLM's shared key is accepted as a fallback. Other providers use their standard `<PROVIDER>_API_KEY` environment variable. |
| `LITELLM_PROXY_URL`, `LITELLM_PROXY_API_KEY` | unset | Optional Proxy URL and server-side virtual key. |
| `LITELLM_TIMEOUT` | `30` | LiteLLM read/write timeout per completion in seconds. |
| `LITELLM_CONNECT_TIMEOUT` | `10` | LiteLLM connection/pool acquisition timeout in seconds. |
| `LITELLM_MAX_RETRIES` | `2` | Bounded retries for timeouts, transient failures, and rate-limit/quota responses. Authentication and invalid-request errors are not retried. |
| `LITELLM_MAX_OUTPUT_TOKENS` | `2048` | Maximum completion token request. |
| `LITELLM_MAX_CONCURRENT_REQUESTS` | `10` | Gateway semaphore limit; read when the process starts. |
| `LITELLM_ENABLE_COST_TRACKING` | `false` | Logs LiteLLM response cost metadata when the SDK returns it. |
| `AGENT_MAX_INPUT_CHARS` | `2000` | Maximum question length. |
| `AGENT_MAX_HISTORY_MESSAGES` | `12` | Maximum recent user/assistant history messages retained. |
| `AGENT_MAX_CONTEXT_TOKENS` | `6000` | Approximate combined question/history budget (estimated at four characters per token). |
| `AGENT_MAX_TOTAL_CONTEXT_TOKENS` | `12000` | Approximate total question/history/retrieved-text budget. |
| `AGENT_RESERVED_SYSTEM_TOKENS` | `2000` | Portion reserved for trusted instructions and tool schemas when limiting retrieval context. |
| `AGENT_MAX_GRAPH_ITERATIONS` | `8` | Maximum agent model turns; the LangGraph recursion limit is also bounded. |
| `AGENT_MAX_TOOL_CALLS` | `10` | Maximum attempted tool calls per request, including invalid calls. |
| `AGENT_MAX_RAG_SEARCHES` | `3` | Maximum RAG calls per request. |
| `AGENT_MAX_RETRIEVED_CHUNKS` | `5` | Maximum chunks returned by one retrieval call. |
| `AGENT_MAX_RETRIEVED_CONTEXT_TOKENS` | `4000` | Approximate total retrieved text budget. |
| `AGENT_MAX_QUERY_CHARS` | `2000` | Prepared and submitted search-query limit. |
| `AGENT_MAX_TOOL_DURATION_SECONDS` | `20` | Maximum time the graph waits for one tool. |
| `AGENT_MAX_OUTPUT_CHARS` | `12000` | Maximum answer size. |
| `API_REQUEST_TIMEOUT_SECONDS` | `180` | HTTP chat request wait limit. |
| `API_MAX_CONCURRENT_CHATS` | `10` | Concurrent `/chat` request limit per application process. |
| `API_RATE_LIMIT_REQUESTS` | `20` | Chat requests allowed for a client IP per window. |
| `API_RATE_LIMIT_WINDOW_SECONDS` | `60` | Fixed-window rate limit duration. |
| `REDIS_URL` | unset | Shared rate-limit store; required with `APP_ENV=production`. Without it, development uses a process-local limiter only. |
| `PII_MODE` | `off` | `off`, `redact`, or `reject` for supported email, phone, and token patterns. |
| `LOG_LEVEL` | `INFO` | Standard Python logging level used by the application. Configure retention in the deployment's logging sink. |
| `AZURE_SEARCH_TIMEOUT_SECONDS` | `15` | Server-side Azure AI Search query timeout, also capped by the tool wait limit. |

The current search backend returns hybrid relevance fields (`@search.score`
and optionally `@search.reranker_score`). No numeric cutoff is applied because
their semantics are not a universal confidence scale. Retrieval ordering and
original citation metadata are preserved.

## Windows PowerShell

From the repository's `sharepoint-chatbot` directory:

```powershell
Set-Location .\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env and provide the required Azure Search and model-provider settings.
python -m uvicorn main:app --host 0.0.0.0 --port 8003
```

Run the backend unit tests from the same directory:

```powershell
python -m unittest discover -s . -p "test*.py" -v
```

`/health` is a liveness check; `/ready` checks model credential/routing and
search-service availability. In production, configure Redis and set
`APP_ENV=production` before starting multiple workers.

## Operational limitations

- Provider model availability and fallback credentials cannot be established
  offline; operators must verify every configured route and that it supports
  tool calls.
- The gateway was validated against LiteLLM 1.106.0. The project requirement
  remains `litellm>=1.50.0`; the gateway uses LiteLLM's `completion`,
  `supports_function_calling`, disables LiteLLM internal retries with
  `num_retries=0` in favor of its bounded retry loop, and uses OpenAI timeout
  features. The capability checker is optional at runtime for SDK builds that
  do not export it; those cases are logged as unverified.
- Python cannot forcibly stop a synchronous tool running in a worker thread.
  The graph stops waiting after the tool limit, and Azure Search has its own
  server-side query timeout, but an unusually stuck SDK call may continue in
  the bounded worker pool.
- Cancelling the HTTP wait does not terminate an already-running Python worker
  thread. Keep model/tool timeouts bounded and configure the reverse proxy's
  request timeout consistently.
- Prompt-injection detection is a logged risk signal, not a classifier or
  blocker. PII matching covers only the documented patterns. LangSmith tracing
  is not enabled or required by this project.
