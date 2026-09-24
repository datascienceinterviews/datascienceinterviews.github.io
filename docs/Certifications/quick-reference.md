---
title: "Claude Certification Quick Reference and Glossary"
description: Final-week cram tables for the four Claude certification exams, covering exam numbers, the API, Claude Code, the Agent SDK, MCP, decision rules and a glossary.
last_reviewed: 2026-09-23
---

# Quick Reference and Glossary

Use this page in the final days before your exam. It condenses the numbers, facts, decision rules and terms that other pages teach in full: the cheat sheets and decision rules link to the page that teaches each topic, and glossary rows name the objectives that use each term. Product facts are as of September 2026; where the documentation changed after the July 2026 exam guides, the page gives both versions (usually in a warning box) and the wording the exam expects.

## Exam numbers

The logistics numbers for the four exams, from the four exam guides (Version 1.0, effective July 2026), the Anthropic Certification Exam Policy, and the live program pages (the Anthropic Partner Academy FAQ and Policies pages, and Pearson VUE's Anthropic pages) as of September 2026. Where the guides and the live pages disagree, the row gives both.

| Number | Value | Taught in |
|---|---|---|
| Exam codes and study guides | [CCAO-F](claude-certified-associate.md): Associate, Foundations · [CCDV-F](claude-certified-developer.md): Developer, Foundations · [CCAR-F](claude-certified-architect-foundations.md): Architect, Foundations · [CCAR-P](claude-certified-architect-professional.md): Architect, Professional | [Pick your exam](index.md#pick-your-exam) |
| Guide version | 1.0, effective July 2026, for all four; each guide says it is subject to change without notice | [Pick your exam](index.md#pick-your-exam) |
| Items | CCAO-F 60 · CCDV-F 53 · CCAR-F 60 · CCAR-P 63 | [How the exams work](index.md#how-the-exams-work) |
| Item format | Guides: multiple-choice and multiple-response; each item states how many responses to select. The [FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says all four exams use "multiple choice and scenario-based multiple response questions" | [How the exams work](index.md#how-the-exams-work) |
| Delivery | Proctored: online proctored and/or test center, per program policy (guide wording) | [How the exams work](index.md#how-the-exams-work) |
| Time | 120 minutes to answer on all four; plan for about 135 minutes of seat time (check-in, instructions, survey) | [How the exams work](index.md#how-the-exams-work) |
| Average time per item (our arithmetic: 120 minutes ÷ items) | CCAO-F 2:00 · CCDV-F about 2:16 · CCAR-F 2:00 · CCAR-P about 1:54 (minutes:seconds) | [How the exams work](index.md#how-the-exams-work) |
| Exam structure | CCAR-F only: 4 scenarios drawn from a bank of 6, picked at random. The CCAO-F, CCDV-F and CCAR-P guides' exam tables have no exam-structure row | [The six scenarios](claude-certified-architect-foundations.md#the-six-scenarios) |
| Domains and weights (D1 first) | [CCAO-F, 7 domains](claude-certified-associate.md#blueprint): 14, 21, 12, 16, 12, 15, 10% · [CCDV-F, 8](claude-certified-developer.md#blueprint): 14.7, 33.1, 3.1, 2.6, 16.8, 11.0, 8.1, 10.6% · [CCAR-F, 5](claude-certified-architect-foundations.md#blueprint): 27, 18, 20, 20, 15% · [CCAR-P, 7](claude-certified-architect-professional.md#blueprint): 17, 13, 19, 16, 14, 14, 7% | Each exam's Blueprint (links in the value) |
| Largest domain | CCAO-F Output Evaluation and Validation (21%) · CCDV-F Applications and Integration (33.1%) · CCAR-F Agentic Architecture & Orchestration (27%) · CCAR-P Integration (19%) | [Pick your exam](index.md#pick-your-exam) |
| Official sample questions | CCAO-F 3 · CCDV-F 3 · CCAR-F 12 · CCAR-P 3; every published sample has one correct answer | [How the exams work](index.md#how-the-exams-work) |
| Pass mark | Scaled score of 720 on a scale of 100 to 1,000, all four exams | [How the exams work](index.md#how-the-exams-work) |
| Score report | Pass or fail plus the scaled score; percent correct per domain is feedback only and does not decide pass or fail. Score on screen at the end; test centers also print a report; a copy arrives by email | [Scoring, results and badges](index.md#scoring-results-and-badges) |
| Credly badge | Claim email usually arrives within minutes for online exams, a little later for test-center exams | [Scoring, results and badges](index.md#scoring-results-and-badges) |
| List fee (USD) | CCAO-F &#36;99 · CCDV-F &#36;125 · CCAR-F &#36;125 · CCAR-P &#36;175 | [Fees and partner discounts](index.md#fees-and-partner-discounts) |
| Partner-tier discount | Registered tier: full price. Select, Preferred and Global Premier: 50% off at checkout. Global Premier: 100% off through December 31, 2026, then 50% | [Fees and partner discounts](index.md#fees-and-partner-discounts) |
| Counts toward Claude Partner Network tier | CCAO-F no · CCDV-F, CCAR-F and CCAR-P yes | [Pick your exam](index.md#pick-your-exam) |
| Eligibility | People at Claude Partner Network organizations, registering with a partner email on a recognized company domain; minimum age 18, checked against government ID | [Who can sit the exams](index.md#who-can-sit-the-exams) |
| Recommended experience (not required) | CCAO-F: regular hands-on use of Claude at work · CCDV-F: 1 to 5 years of software engineering plus 6+ months with Claude or comparable LLM systems · CCAR-F (typical-candidate profile; no prerequisites line in its guide): 6+ months building with Claude APIs, Agent SDK, Claude Code and MCP · CCAR-P: 3+ years in systems architecture or platform engineering plus 6+ months with Claude or comparable LLM systems in production | [Who can sit the exams](index.md#who-can-sit-the-exams) |
| Registration shelf life | FAQ: a registration is valid for 5 years, and you can schedule and sit the exam at any time in that window. Policies page: no deadline for sitting the exam. Plan to the 5-year limit | [Registration, step by step](index.md#registration-step-by-step) |
| Cancel or reschedule | At least 48 hours before the appointment (Policies page and FAQ; Pearson's Anthropic page says the same for test center appointments; September 2026). The July 2026 guides still say 24 hours. Plan on 48 | [Policies that cost candidates money](index.md#policies-that-cost-candidates-money) |
| No-show or late arrival | Fee forfeited; you buy a new attempt | [Policies that cost candidates money](index.md#policies-that-cost-candidates-money) |
| Refund | Policies page: canceling at least 48 hours ahead gets a full refund. FAQ: canceling in Pearson only frees the slot, so request the refund by email to certifications-support@anthropic.com | [Policies that cost candidates money](index.md#policies-that-cost-candidates-money) |
| Retakes | Wait 14 days after the 1st failed attempt, 30 after the 2nd, 90 after the 3rd. At most 4 attempts per exam in a rolling 12 months. Every attempt costs the full fee (tier discount applies). Counters reset when the exam moves to a new version. A pass cannot be retaken to raise the score | [Policies that cost candidates money](index.md#policies-that-cost-candidates-money) |
| Name on ID | Pearson profile name must match your government ID exactly. FAQ: correct it at least 24 hours before the exam; corrections typically take 24 to 48 business hours. The guides and the Policies page say to email the correction before you schedule, which is the safe plan | [Registration, step by step](index.md#registration-step-by-step) |
| Accommodations | Must be approved before you schedule; cannot be added to a booked exam. FAQ: request 10 days or more ahead. Pearson: allow 10 business days | [Policies that cost candidates money](index.md#policies-that-cost-candidates-money) |
| Online (OnVUE) setup | Check-in opens 30 minutes before the appointment. Windows 10 or macOS 14 or higher; at least 6 Mbps down and 2 Mbps up; no VPNs, corporate or public networks; digital whiteboard only | [How the exams work](index.md#how-the-exams-work) |
| Exam-room rules | Closed book, English only, no browser translation tools, no AI products or services during the exam | [How the exams work](index.md#how-the-exams-work) |
| Appeals | Within 14 days of the decision notice, or of the exam date for a result dispute (guides, FAQ and Policies page). Policies page: a confirmed faulty question that affected your result earns a free retake, not a changed score. The July 2026 guides say the content of individual exam items is not subject to appeal | [Scoring, results and badges](index.md#scoring-results-and-badges) |
| Validity | 12 months from the date the credential is awarded | [Renewal](index.md#renewal) |
| Renewal | Free online assessment on Anthropic Partner Academy before expiry (guides and FAQ: non-proctored; Policies page: open-book, retakable as often as needed, and adds 12 months from the current expiration date). After a lapse: the full exam at the full fee. FAQ: full renewal details will be shared before the first certifications come up for renewal | [Renewal](index.md#renewal) |

## API cheat sheet

CCDV-F covers Claude API mechanics (messages, tools, streaming, vision, thinking, caching, third-party vendors and batch use), cost and token management, and error handling. The CCAR-F appendix names `stop_reason` values, `tool_choice` options, `max_tokens`, system prompts and the Message Batches API among the technologies that "might appear on the exam", explicitly tests stop-reason loop control, `tool_choice` configuration and batch appropriateness, and puts streaming, rate limits and pricing calculations, vision, token counting algorithms, cloud provider configuration and caching implementation details beyond knowing caching exists out of scope ([CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Model facts are as of September 2026.

### Request fields

| Field | What to remember | Go deeper |
|---|---|---|
| `model` | From the 4.6 generation on, IDs are dateless (`claude-opus-5-5`) yet fixed: one ID, one snapshot, and updates ship under new IDs. Earlier models have dated IDs plus aliases such as `claude-sonnet-4-5` that resolve to the most recent dated snapshot for that minor version; the guarantee that the model stays constant covers IDs, not aliases. CCDV-F tests model version pinning | [Model versions](knowledge/claude-api.md#model-versions-deprecation-and-migration) |
| `max_tokens` | Hard ceiling on generated tokens, thinking included; the model may stop earlier. `0` pre-warms the prompt cache and returns no content | [Messages API call](knowledge/claude-api.md#how-a-messages-api-call-works) |
| `messages` | Alternating `user` and `assistant` turns; consecutive same-role turns are merged into one. Stateless: send the full history on every call. At most 100,000 messages per request | [Messages API call](knowledge/claude-api.md#how-a-messages-api-call-works) |
| Final `assistant` turn (prefill) | The reply continues from it on older models; returns 400 on Claude 4.6 and later models and Claude Mythos Preview (the error says the conversation must end with a user message). Use structured outputs or system prompt instructions instead | [Messages API call](knowledge/claude-api.md#how-a-messages-api-call-works) |
| `system` | Top-level string or array of text blocks (a block can carry `cache_control`). The API reference still says there is no `"system"` role for input messages, but on Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Opus 5.5, Opus 4.8 and Opus 5 you can append a `"role": "system"` message mid-conversation (never as the first entry; not on Sonnet 5) without invalidating the cached prefix | [Messages API call](knowledge/claude-api.md#how-a-messages-api-call-works) |
| `tools` | Each client tool has `name`, `description` (optional, strongly recommended) and `input_schema` (JSON Schema). Top-level `"strict": true` on a tool guarantees schema-valid inputs | [Defining a tool](knowledge/tool-use-and-mcp.md#defining-a-tool) |
| `tool_choice` | `auto`, `any`, `tool`, `none` (next table) | [Controlling tool choice](knowledge/tool-use-and-mcp.md#controlling-tool-choice) |
| `stop_sequences` | Custom strings; when one fires, `stop_reason` is `"stop_sequence"` and the `stop_sequence` field holds the match | [Stop reasons](knowledge/claude-api.md#stop-reasons-and-the-agent-loop) |
| `stream` | `true` sends server-sent events: `message_start`, then per block `content_block_start`, `content_block_delta`, `content_block_stop`, then `message_delta` (with `stop_reason` and cumulative usage) and `message_stop` | [Streaming](knowledge/claude-api.md#streaming) |
| `thinking` | `{"type": "adaptive"}`, `{"type": "enabled", "budget_tokens": N}` (N at least 1,024 and below `max_tokens`) or `{"type": "disabled"}`. Extended (`enabled`) is deprecated on Claude 4.6 models and rejected on 4.7 and later. Thinking tokens bill as output | [Thinking and effort](knowledge/claude-api.md#extended-thinking-adaptive-thinking-and-effort) |
| `output_config.effort` | `low`, `medium`, `high`, `xhigh`, `max`. Default `high` on most models, `medium` on Opus 5.5. A behavioral signal, not a token budget: use `max_tokens` for a hard cap | [Thinking and effort](knowledge/claude-api.md#extended-thinking-adaptive-thinking-and-effort) |
| `output_config.format` | `{"type": "json_schema", "schema": {...}}`: JSON outputs through constrained decoding. Replaces the older `output_format`. Returns 400 if combined with citations | [Structured outputs](knowledge/claude-api.md#structured-outputs) |
| `cache_control` | Top level: automatic caching on the last cacheable block. On a block: an explicit breakpoint. `{"type": "ephemeral", "ttl": "1h"}` for the 1-hour lifetime | [Prompt caching](knowledge/claude-api.md#prompt-caching) |
| `temperature`, `top_p`, `top_k` | Deprecated from Claude Opus 4.7: non-default values return 400 on Claude 4.7 and later models and Claude Mythos Preview; the replacement is prompting. The Python SDK v1.0 and later removes them (`TypeError`). Even `temperature` 0 is not fully deterministic | [Messages API call](knowledge/claude-api.md#how-a-messages-api-call-works) |
| `metadata.user_id` | Opaque ID (uuid or hash) for abuse detection; never a name, email or phone number | [Messages API call](knowledge/claude-api.md#how-a-messages-api-call-works) |
| `service_tier` | `"auto"` (default) or `"standard_only"`; the response's `usage.service_tier` reports `standard`, `priority` or `batch` | [Cost and usage](knowledge/claude-api.md#cost-and-usage-tracking) |
| `content` (response) | Array of typed blocks (`text`, `thinking`, `tool_use` and more). Select blocks by `type`, not position, because a reply can start with a thinking block | [Messages API call](knowledge/claude-api.md#how-a-messages-api-call-works) |
| `usage` (response) | Total input = `input_tokens` + `cache_creation_input_tokens` + `cache_read_input_tokens`; `output_tokens` is non-zero even for an empty reply | [Cost and usage](knowledge/claude-api.md#cost-and-usage-tracking) |

### Stop reasons

| `stop_reason` | Meaning | What your code does | Go deeper |
|---|---|---|---|
| `end_turn` | Claude finished naturally | Use the reply. An empty `end_turn` (2 to 3 tokens) typically comes after tool results; a common cause is text blocks added right after `tool_result` blocks: don't add text there, and don't resend the empty reply unchanged | [Stop reasons](knowledge/claude-api.md#stop-reasons-and-the-agent-loop) |
| `tool_use` | Claude is calling a tool | Run it; send back one user message holding only `tool_result` blocks, one per `tool_use`, matched by `tool_use_id`, with the same `tools` array; loop | [Returning tool results](knowledge/tool-use-and-mcp.md#returning-tool-results-and-errors) |
| `max_tokens` | Hit your `max_tokens` | Raise the limit or continue; a truncated `tool_use` block needs a retry with a higher limit | [Stop reasons](knowledge/claude-api.md#stop-reasons-and-the-agent-loop) |
| `stop_sequence` | One of your `stop_sequences` fired | Read the `stop_sequence` field | [Stop reasons](knowledge/claude-api.md#stop-reasons-and-the-agent-loop) |
| `pause_turn` | The server-side loop for server tools (web search and similar) hit its iteration limit, 10 by default | Append the assistant content as is and call again. A response waiting on a client tool is `tool_use`, never `pause_turn` | [Stop reasons](knowledge/claude-api.md#stop-reasons-and-the-agent-loop) |
| `refusal` | A safety classifier declined; this is an HTTP 200 response, not an error | Read `stop_details.category`; reset or rephrase the context, or retry on another model. Continuing unchanged gets more refusals | [Stop reasons](knowledge/claude-api.md#stop-reasons-and-the-agent-loop) |
| `model_context_window_exceeded` | The reply filled the context window | Treat it as truncated. Returned without a beta header on Sonnet 4.5 and newer | [Tokens and context windows](knowledge/claude-api.md#tokens-context-windows-and-counting) |

- **Exam loop rule (CCAR-F 1.1):** continue while `stop_reason` is `"tool_use"`, finish on `"end_turn"`; the anti-patterns the guide names are in [Reliability rules](#reliability-rules). The API docs describe the same loop: while `stop_reason` is `tool_use`, run the tools and continue; exit on any other stop reason. Loops that use server tools must also handle `pause_turn`. Answer in the guide's `tool_use` versus `end_turn` terms.
- **Not errors:** stop reasons arrive in successful responses; failures are HTTP 4xx and 5xx.
- **Streaming:** `stop_reason` is null in `message_start` and arrives in `message_delta`.
- **Beta extra:** with on-demand compaction (beta header `compact-2026-09-04`), or threshold compaction (`compact-2026-01-12`) with `pause_after_compaction` enabled, a response can also stop with `compaction`.

### tool_choice

| Value | Behavior | Go deeper |
|---|---|---|
| `{"type": "auto"}` | Claude decides whether to call a tool; the default when `tools` are provided | [Controlling tool choice](knowledge/tool-use-and-mcp.md#controlling-tool-choice) |
| `{"type": "any"}` | Must call one of the tools, but not a particular one | [Controlling tool choice](knowledge/tool-use-and-mcp.md#controlling-tool-choice) |
| `{"type": "tool", "name": "get_weather"}` | Must call the named tool | [Controlling tool choice](knowledge/tool-use-and-mcp.md#controlling-tool-choice) |
| `{"type": "none"}` | No tool calls; the default when no `tools` are provided | [Controlling tool choice](knowledge/tool-use-and-mcp.md#controlling-tool-choice) |
| `"disable_parallel_tool_use": true` (inside `tool_choice`, not top level) | With `auto`: at most one tool call. With `any` or `tool`: exactly one | [Parallel tool calls](knowledge/tool-use-and-mcp.md#parallel-tool-calls) |

- With `any` or `tool`, the API prefills the assistant turn, so Claude writes no explanation before the call. For an explanation plus a call, use `auto` and ask for the tool in the user message.
- Forced choice (`any` or `tool`) errors with manual extended thinking and works with adaptive thinking. Opus 5.5, Fable 5.1 and Mythos 5.1 reject `any` and `tool` with a 400 `invalid_request_error` whatever the thinking setting (the token counting endpoint too); `auto` and `none` still work there.
- Changing `tool_choice` invalidates cached message blocks; tool definitions and the system prompt stay cached.

!!! warning "Exam guide vs current docs"

    - **Forced tool use.** The CCAR-F guide (July 2026) teaches `tool_choice: "any"` to guarantee a tool call and forced selection (`{"type": "tool", "name": "extract_metadata"}`) to make a specific tool run first. As of September 2026, Opus 5.5, Fable 5.1 and Mythos 5.1 reject both with a 400 (other models still accept them, except with manual extended thinking); the docs' substitute there is `auto` plus `strict: true`, or structured outputs (`output_config.format`). On the exam, answer in the guide's terms: `any` guarantees a call, forced selection guarantees which one.
    - **Structured output.** The guide calls tool use with JSON schemas the most reliable route to schema-compliant output and says strict schemas remove syntax errors but not semantic ones, such as line items that do not sum to the total. The docs now add JSON outputs (`output_config.format`) and strict tool use (`strict: true`), both enforced by constrained decoding, and warn that without strict mode Claude might return incompatible types or omit required fields. Our reading: constrained decoding enforces the schema's shape, so semantic checks such as totals that must add up remain your code's job either way.

### Prompt caching numbers

| Item | Value | Go deeper |
|---|---|---|
| Turn it on | Automatic: one top-level `"cache_control": {"type": "ephemeral"}`. Explicit: `cache_control` on individual blocks (breakpoints). `ephemeral` is the only cache type | [Prompt caching](knowledge/claude-api.md#prompt-caching) |
| Prefix order | `tools`, then `system`, then `messages`, up to and including the marked block. A change at one level invalidates that level and every later one | [Prompt caching](knowledge/claude-api.md#prompt-caching) |
| Breakpoints | Up to 4 per request; automatic caching uses one of the 4 slots. Writes happen only at breakpoints; on a read, each breakpoint checks at most 20 positions (counting itself) for an entry an earlier request wrote | [Prompt caching](knowledge/claude-api.md#prompt-caching) |
| Lifetime | 5 minutes by default, refreshed free on every hit; `"ttl": "1h"` for 1 hour. Measured from the start of the request. 1-hour entries must come before 5-minute entries. No manual clear | [Prompt caching](knowledge/claude-api.md#prompt-caching) |
| Price (multiple of base input) | 5-minute write 1.25x; 1-hour write 2x; read 0.1x (0.05x on Opus 5.5; 0.025x on Fable 5.1 and Mythos 5.1). At the 0.1x read rate, caching pays off after 1 read (5-minute) or 2 reads (1-hour). Stacks with the batch discount | [Cost and usage](knowledge/claude-api.md#cost-and-usage-tracking) |
| Minimum cacheable prompt | 512 tokens: Fable 5.1, Mythos 5.1, Opus 5.5, Opus 5, Fable 5, Mythos 5. 1,024: Opus 4.8, Sonnet 5, Sonnet 4.6, Sonnet 4.5. 2,048: Opus 4.7, Mythos Preview. 4,096: Opus 4.6, Opus 4.5, Haiku 4.5. Shorter prompts run uncached with no error | [Prompt caching](knowledge/claude-api.md#prompt-caching) |
| Usage fields | `cache_creation_input_tokens` (written), `cache_read_input_tokens` (read), `input_tokens` (after the last breakpoint). Both cache fields 0 means nothing cached | [Prompt caching](knowledge/claude-api.md#prompt-caching) |
| What breaks the cache | Editing tool definitions (whole cache); toggling web search or citations (system and messages); changing `tool_choice`, images or `disable_parallel_tool_use` (messages); changing thinking settings or `output_config.effort` (messages, plus tools and system on models that render that configuration ahead of them; setting effort explicitly to the model's default is the same as omitting it and does not invalidate); switching `speed` between fast and standard (system and messages). Hits need a 100% identical prefix | [Prompt caching](knowledge/claude-api.md#prompt-caching) |
| Rate limits | On most models only uncached input (`input_tokens` + `cache_creation_input_tokens`) counts toward input tokens per minute | [Errors and rate limits](knowledge/claude-api.md#errors-retries-and-rate-limits) |
| Context | Cached prefixes still occupy the context window: caching changes the price, not the token count | [Tokens and context windows](knowledge/claude-api.md#tokens-context-windows-and-counting) |
| Concurrency | An entry exists only after the first response begins; send parallel requests after that | [Prompt caching](knowledge/claude-api.md#prompt-caching) |
| Pre-warm | `max_tokens: 0` with an explicit breakpoint: empty `content`, `stop_reason: "max_tokens"`, zero output tokens billed | [Prompt caching](knowledge/claude-api.md#prompt-caching) |
| Exam rule | CCAR-P Sample 2 (static content first, then caching): see [Cost and latency rules](#cost-and-latency-rules) | [CCAR-P sample questions](claude-certified-architect-professional.md#official-sample-questions) |

!!! note "Terms and scope"

    The CCDV-F guide lists "cache check-pointing" among its caching techniques ([CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). The [prompt caching documentation](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) does not use that term; its closest concept is the cache breakpoint (`cache_control`), which is our mapping. CCAR-F lists "Prompt caching implementation details (beyond knowing it exists)" as out of scope ([CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)), while CCDV-F (cost and token management) and CCAR-P (prompt reuse strategies, Sample 2) go further. Treat the numbers above as CCDV-F and CCAR-P depth.

### Message Batches numbers

| Item | Value | Go deeper |
|---|---|---|
| Cost | 50% off input and output tokens; stacks with prompt caching discounts | [Message Batches](knowledge/claude-api.md#message-batches) |
| Timing | Most batches finish in under 1 hour. Results open when every request is done or after 24 hours; unfinished requests expire at 24 hours. No latency SLA | [Message Batches](knowledge/claude-api.md#message-batches) |
| Size | 100,000 requests or 256 MB per batch, whichever comes first; an oversized batch creation request may get a 413 `request_too_large` | [Message Batches](knowledge/claude-api.md#message-batches) |
| Results | `.jsonl`, in any order: match on `custom_id`. Downloadable for 29 days from `created_at` | [Message Batches](knowledge/claude-api.md#message-batches) |
| `custom_id` | 1 to 64 characters, letters, digits, hyphens and underscores (`^[a-zA-Z0-9_-]{1,64}$`); unique per request | [Message Batches](knowledge/claude-api.md#message-batches) |
| `processing_status` | `in_progress`, `canceling`, `ended` | [Message Batches](knowledge/claude-api.md#message-batches) |
| Result types | `succeeded`, `errored`, `canceled`, `expired`; the last three are not billed. An `errored` `invalid_request_error` needs a fixed body before resending | [Message Batches](knowledge/claude-api.md#message-batches) |
| Not allowed in a batch request | `stream: true`, `speed` (fast mode), `max_tokens: 0`; every request needs `max_tokens` of at least 1 | [Message Batches](knowledge/claude-api.md#message-batches) |
| Validation | Asynchronous: errors appear only when the whole batch ends, so dry-run one request shape on the Messages API first | [Batch processing design](knowledge/prompt-engineering.md#batch-processing-design) |
| Changes | A submitted batch cannot be edited; cancel and resubmit. One failed request does not affect the others | [Message Batches](knowledge/claude-api.md#message-batches) |
| Rate limits (Start tier) | 1,000 requests per minute, 200,000 batch requests in the processing queue, 100,000 per batch; shared across all models, and batch usage does not affect Messages API limits | [Message Batches](knowledge/claude-api.md#message-batches) |
| Caching inside batches | Best effort, typical hit rates 30% to 98%; batches can outlast 5 minutes, so consider the 1-hour lifetime | [Prompt caching](knowledge/claude-api.md#prompt-caching) |
| Exam rules | CCAR-F 4.5, Q11 and CCDV-F Sample 1 (what to batch, what to keep real time, resubmitting by `custom_id`, SLA arithmetic): see [Cost and latency rules](#cost-and-latency-rules) | [Batch processing design](knowledge/prompt-engineering.md#batch-processing-design) |

!!! warning "Exam guide vs current docs"

    - **Tool calling in batches.** CCAR-F says the batch API does not support multi-turn tool calling within a single request. The docs say server tools (web search, web fetch, code execution and others) run the same server-side loop inside a batch request as in a synchronous call, with more iterations per turn, though a result can still return `pause_turn` and need a follow-up request; multi-turn conversation history can also be batched. Our reading, which the docs do not state outright: both hold, because server tools run on Anthropic's side while a client tool needs your code to run between requests, which a single batch request cannot do. Answer in the guide's terms.
    - **Batch size.** The October 2024 launch post allowed 10,000 queries per batch. Current docs allow 100,000 requests or 256 MB; material quoting 10,000 is out of date.

### Error codes

| HTTP | `error.type` | What to do | Go deeper |
|---|---|---|---|
| 400 | `invalid_request_error` | Fix the request; it is not in the SDKs' retry list. Also returned when the input alone exceeds the context window (`prompt is too long`), for prefill on Claude 4.6 and later, for `tool_choice` `any` or `tool` on Opus 5.5, Fable 5.1 and Mythos 5.1, and when usage reaches a spend limit you set yourself | [Errors and rate limits](knowledge/claude-api.md#errors-retries-and-rate-limits) |
| 401 | `authentication_error` | Check the API key (malformed, revoked or expired) | [Errors and rate limits](knowledge/claude-api.md#errors-retries-and-rate-limits) |
| 402 | `billing_error` | Fix billing or payment details | [Errors and rate limits](knowledge/claude-api.md#errors-retries-and-rate-limits) |
| 403 | `permission_error` | The key lacks permission for the resource | [Errors and rate limits](knowledge/claude-api.md#errors-retries-and-rate-limits) |
| 404 | `not_found_error` | The requested resource does not exist; check the ID | [Errors and rate limits](knowledge/claude-api.md#errors-retries-and-rate-limits) |
| 409 | `conflict_error` | Resolve the resource-state conflict, then retry | [Errors and rate limits](knowledge/claude-api.md#errors-retries-and-rate-limits) |
| 413 | `request_too_large` | Shrink the request: 32 MB for Messages and token counting, 256 MB for batches, 500 MB for Files | [Errors and rate limits](knowledge/claude-api.md#errors-retries-and-rate-limits) |
| 429 | `rate_limit_error` | Your organization hit a rate limit: back off and honor `retry-after`. A sharp traffic ramp can also trigger it (acceleration limits), so ramp gradually. The usage tier's monthly spend cap also returns 429, with `error.details.error_code` `enforced_spend_limit_reached` and no `retry-after`. Claude Code requests over a Claude Code workspace spend limit can get a 429 instead of the 400 for spend limits you set, and that 429 carries a `retry-after` header | [Errors and rate limits](knowledge/claude-api.md#errors-retries-and-rate-limits) |
| 500 | `api_error` | Internal error: retry with exponential backoff; contact support with the request ID if it persists | [Errors and rate limits](knowledge/claude-api.md#errors-retries-and-rate-limits) |
| 504 | `timeout_error` | Use streaming or batches for long requests | [Errors and rate limits](knowledge/claude-api.md#errors-retries-and-rate-limits) |
| 529 | `overloaded_error` | API-wide high traffic, not your quota; retry with backoff. Mid-stream it arrives as an `error` event after a 200 | [Errors and rate limits](knowledge/claude-api.md#errors-retries-and-rate-limits) |

- **SDK retries:** the official SDKs retry transient failures (connection errors, rate limits and 5xx) twice by default with exponential backoff, honoring `retry-after` when present. The Python SDK's list is connection errors, 408, 409, 429 and 500 or above; set `max_retries` (0 disables).
- **Error body:** JSON with a top-level `error` object (`type`, `message`) plus `request_id`; catch the SDK's typed exceptions (such as Python `RateLimitError`) instead of matching message strings.
- **Refusals are not errors:** `stop_reason: "refusal"` arrives with HTTP 200.

### Key headers and endpoints

| Header or endpoint | Value | Go deeper |
|---|---|---|
| `anthropic-version` | Required on every request, for example `2023-06-01`; the SDKs send it for you | [Messages API call](knowledge/claude-api.md#how-a-messages-api-call-works) |
| `content-type` | `application/json` | [Messages API call](knowledge/claude-api.md#how-a-messages-api-call-works) |
| API key | `Authorization: Bearer <key>` (required unless `x-api-key` is set) or `x-api-key: <key>`, which the API overview calls a legacy fallback that is still supported. The docs' quickstart curl example still sends `x-api-key`; both work. SDKs read `ANTHROPIC_API_KEY` and send the auth, version and content-type headers for you. Keys start with `sk-ant-` and are shown only once | [Secrets and API keys](knowledge/security-and-governance.md#secrets-and-api-keys) |
| `anthropic-beta` | Beta names look like `feature-name-YYYY-MM-DD`; several go comma-separated (or as repeated headers); an invalid or inaccessible name returns 400. SDKs take a `betas` parameter | [Messages API call](knowledge/claude-api.md#how-a-messages-api-call-works) |
| `anthropic-workspace-id` | Required with a multi-workspace API key, optional for other keys | [Messages API call](knowledge/claude-api.md#how-a-messages-api-call-works) |
| `request-id` (response) | Unique ID such as `req_...`; quote it to support. Python and TypeScript expose `_request_id` | [Errors and rate limits](knowledge/claude-api.md#errors-retries-and-rate-limits) |
| `retry-after`, `anthropic-ratelimit-*` (response) | Seconds to wait before retrying; `anthropic-ratelimit-{requests,tokens,input-tokens,output-tokens}-{limit,remaining,reset}` | [Errors and rate limits](knowledge/claude-api.md#errors-retries-and-rate-limits) |
| Endpoints | `POST /v1/messages`, `POST /v1/messages/batches`, `POST /v1/messages/count_tokens` (free, own rate limits, estimate only), `GET /v1/models` | [Tokens and context windows](knowledge/claude-api.md#tokens-context-windows-and-counting) |

### Context, tokens, vision and platforms

CCDV-F names vision and invoking Claude through third-party vendors; CCAR-F lists vision, token counting algorithms and cloud provider configurations as out of scope.

| Item | Value | Go deeper |
|---|---|---|
| Context window | Everything counts: the system prompt, every message (tool results, images, documents included), tool definitions and the output, thinking included. 1M tokens by default (no beta header, standard pricing) on Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Opus 4.6, Sonnet 5, Sonnet 4.6 and Mythos Preview; 200K on the others, such as Sonnet 4.5 and Haiku 4.5. More is not automatically better: accuracy and recall degrade as the token count grows (context rot) | [Tokens and context windows](knowledge/claude-api.md#tokens-context-windows-and-counting) |
| Too long | Input alone over the window: 400 `invalid_request_error` (`prompt is too long`) on every model. A reply that fills the window stops with `model_context_window_exceeded` | [Tokens and context windows](knowledge/claude-api.md#tokens-context-windows-and-counting) |
| Token counting | `POST /v1/messages/count_tokens` takes the same inputs as message creation (system, tools, images, PDFs) and returns the total input tokens. Free, with its own rate limits; the count is an estimate; it does not use prompt caching. Recount against the model you plan to use | [Tokens and context windows](knowledge/claude-api.md#tokens-context-windows-and-counting) |
| Images | JPEG, PNG, GIF or WebP (animations: first frame only), as `base64`, `url` or `file` (Files API) sources; Bedrock and Google Cloud accept base64 only. Up to 600 images or PDF pages per request (100 on 200K-window models); at most 8000x8000 px; more than 20 images triggers a stricter per-image limit; 10 MB base64 per image on the Claude API, 5 MB on Bedrock and Google Cloud. Images work best before text | [Vision and PDF input](knowledge/claude-api.md#vision-and-pdf-input) |
| PDFs | `document` blocks by URL, base64 or `file_id`; 32 MB per request, 600 pages (100 when the context window is under 1M); no passwords or encryption. Each page arrives as extracted text plus a page image, typically 1,500 to 3,000 text tokens per page plus image tokens. `.docx` and `.xlsx` must be converted to text or PDF first | [Vision and PDF input](knowledge/claude-api.md#vision-and-pdf-input) |
| Fast mode | Research preview: `speed: "fast"` plus the `fast-mode-2026-02-01` beta header gives up to 2.5x output tokens per second on Opus 5.5, Opus 5 and Opus 4.8 (same weights; it speeds output, not time to first token). Opus 5.5 fast pricing &#36;8 / &#36;40 per MTok. Claude API only; not with the Batch API; switching speeds invalidates the prompt cache | [Thinking and effort](knowledge/claude-api.md#extended-thinking-adaptive-thinking-and-effort) |
| Cloud platforms | Anthropic-operated: Claude Platform on AWS and Microsoft Foundry. Partner-operated: Amazon Bedrock and Google Cloud. Features vary: the Bedrock, Google Cloud and Foundry pages each list Message Batches as unsupported. On Google Cloud, `model` goes in the endpoint URL and `anthropic_version` (`vertex-2023-10-16`) in the body. For FedRAMP High, IL4, IL5 or HIPAA-ready compliance, or AWS as the sole data processor, the docs point to Bedrock | [Claude on cloud platforms](knowledge/claude-api.md#claude-on-cloud-platforms) |

### Models at a glance (as of September 2026)

| Model | API ID | Latency | Input / output per MTok | Context | Max output | Thinking | Default effort |
|---|---|---|---|---|---|---|---|
| [Claude Fable 5.1](knowledge/claude-api.md#models-and-how-to-choose-one) | `claude-fable-5-1` | Slower | &#36;10 / &#36;50 | 1M | 128K | Adaptive, always on | `high` |
| [Claude Opus 5.5](knowledge/claude-api.md#models-and-how-to-choose-one) | `claude-opus-5-5` | Moderate | &#36;4 / &#36;20 | 1M | 128K | Adaptive, always on | `medium` |
| [Claude Sonnet 5](knowledge/claude-api.md#models-and-how-to-choose-one) | `claude-sonnet-5` | Fast | &#36;2 / &#36;10 | 1M | 128K | Adaptive | `high` |
| [Claude Haiku 4.5](knowledge/claude-api.md#models-and-how-to-choose-one) | `claude-haiku-4-5-20251001` (alias `claude-haiku-4-5`) | Fastest | &#36;1 / &#36;5 | 200K | 64K | Extended | Not supported |

Batch requests are 50% off. The pricing page's rule of thumb is Haiku for simple tasks, Sonnet for most production workloads and Opus for the most complex reasoning; the models overview says that if you are unsure, start with Claude Opus 5.5 for most workloads, and it points to Claude Fable 5.1 for demanding reasoning and long-horizon agentic work. The guides frame model choice as Opus versus Sonnet versus Haiku trade-offs in cost, speed and quality, so learn the reasoning rather than this month's lineup.

## Claude Code cheat sheet

Claude Code Configuration & Workflows is 20% of CCAR-F. On CCDV-F, Domain 3 (Claude Code) is 3.1%, and the Domain 2 skill Configuration Management (4.1%) separately covers CLAUDE.md files, settings.json, model version pinning, prompt versioning and plugin dependencies. CCAR-P Domain 7 (7%) includes configuring Claude tools and environments for teams, with Claude Code as the guide's example. Product values below are as of September 2026 (changelog v2.1.280, September 22, 2026). The exam decision rules for Claude Code are collected under [Claude Code layer rules](#claude-code-layer-rules).

### Paths and files

| Path | Scope and sharing | What it holds | Go deeper |
|---|---|---|---|
| `~/.claude/CLAUDE.md` | You, every project; not shared through version control | Personal instructions | [CLAUDE.md and memory](knowledge/claude-code-configuration.md#claudemd-and-the-memory-hierarchy) |
| `./CLAUDE.md` or `./.claude/CLAUDE.md` | Project; committed and shared with the team | Build and test commands, coding standards, architectural decisions, naming conventions, common workflows | [CLAUDE.md and memory](knowledge/claude-code-configuration.md#claudemd-and-the-memory-hierarchy) |
| `./CLAUDE.local.md` | You, this project; add it to `.gitignore` | Private project notes; loads after `CLAUDE.md` in the same directory | [CLAUDE.md and memory](knowledge/claude-code-configuration.md#claudemd-and-the-memory-hierarchy) |
| `<subdirectory>/CLAUDE.md` | Directory level | Not loaded at launch; loads when Claude reads a file in that directory with the Read tool (not when it writes or creates one there) | [CLAUDE.md and memory](knowledge/claude-code-configuration.md#claudemd-and-the-memory-hierarchy) |
| `./AGENTS.md` | Project | Newer than the exam guides (v2.1.277 or later): by default read as project instructions only when no `CLAUDE.md` or `CLAUDE.local.md` exists in the working directory or above; where direct reading is unavailable, import it from a CLAUDE.md with `@AGENTS.md` | [CLAUDE.md and memory](knowledge/claude-code-configuration.md#claudemd-and-the-memory-hierarchy) |
| Managed `CLAUDE.md` | Every user on the machine; set by IT; cannot be excluded | macOS `/Library/Application Support/ClaudeCode/CLAUDE.md`, Linux and WSL `/etc/claude-code/CLAUDE.md`, Windows `C:\Program Files\ClaudeCode\CLAUDE.md` | [Managed settings](knowledge/claude-code-configuration.md#managed-settings-for-organizations) |
| `.claude/rules/*.md`, `~/.claude/rules/*.md` | Project (shared) or user (every project) | Topic rules; with `paths` frontmatter they load only for matching files | [Path-scoped rules](knowledge/claude-code-configuration.md#path-scoped-rules) |
| `~/.claude/settings.json` | You, every project | Personal settings | [Settings files](knowledge/claude-code-configuration.md#settings-files-and-precedence) |
| `.claude/settings.json` | Project; commit it | Team permissions, hooks, env, plugins | [Settings files](knowledge/claude-code-configuration.md#settings-files-and-precedence) |
| `.claude/settings.local.json` | You, this project; Claude Code keeps it out of git | Personal overrides; Bash approvals you save from a permission prompt land here | [Settings files](knowledge/claude-code-configuration.md#settings-files-and-precedence) |
| `managed-settings.json` | Organization; nothing you set overrides it, apart from a few security-sensitive keys where a stricter lower-level value still counts | macOS `/Library/Application Support/ClaudeCode/managed-settings.json`, Linux and WSL `/etc/claude-code/managed-settings.json`, Windows `C:\Program Files\ClaudeCode\managed-settings.json` | [Managed settings](knowledge/claude-code-configuration.md#managed-settings-for-organizations) |
| `~/.claude.json` | You | Sign-in session, MCP server configurations, per-project trust state. Not the place for `permissions`, `hooks` or `env` | [Settings files](knowledge/claude-code-configuration.md#settings-files-and-precedence) |
| `.mcp.json` (repository root) | Project; shared | Team MCP servers; `settings.json` does not read an `mcpServers` key. More in [MCP in Claude Code](#mcp-in-claude-code) | [MCP in Claude Code](knowledge/claude-code-configuration.md#mcp-servers-in-claude-code) |
| `.claude/skills/<name>/SKILL.md`, `~/.claude/skills/<name>/SKILL.md` | Project or personal | A skill: `/name` workflow or on-demand knowledge. Needs a folder with `SKILL.md`, not a loose `name.md` | [Agent Skills](knowledge/claude-code-configuration.md#agent-skills) |
| `.claude/commands/*.md`, `~/.claude/commands/*.md` | Project or user | Single-file custom commands: the older format, still supported and the same mechanism as skills; the docs prefer a skill for new work | [Slash commands](knowledge/claude-code-configuration.md#slash-commands) |
| `.claude/agents/*.md`, `~/.claude/agents/*.md` | Project (check it in) or user | Subagent definitions | [Subagents](knowledge/claude-code-configuration.md#subagents) |
| `.claude/output-styles/`, `~/.claude/output-styles` | Project or user | Custom output styles | [Models and output styles](knowledge/claude-code-configuration.md#models-and-output-styles) |
| `~/.claude/projects/<project>/memory/MEMORY.md` | You, per repository (shared across worktrees), this machine only | Auto memory Claude writes; the first 200 lines or 25KB load each session | [CLAUDE.md and memory](knowledge/claude-code-configuration.md#claudemd-and-the-memory-hierarchy) |
| `.claude/agent-memory/<name>/` | Project (`memory: project`; `user` and `local` scopes also exist) | A subagent's persistent memory | [Subagents](knowledge/claude-code-configuration.md#subagents) |
| `~/.claude/projects/<project>/<session-id>.jsonl` | You | Session transcripts, kept 30 days by default (`cleanupPeriodDays`) | [Sessions](knowledge/claude-code-workflows.md#sessions-continue-resume-fork-and-rewind) |
| `.claude-plugin/plugin.json`, `.claude-plugin/marketplace.json` | Plugin or marketplace root | Plugin manifest (optional; if present, `name` is the only required field); marketplace catalog (required: `name`, `owner`, `plugins`) | [Plugins](knowledge/claude-code-configuration.md#plugins-and-marketplaces) |

### Loading and precedence

| Rule | Value | Go deeper |
|---|---|---|
| CLAUDE.md lookup | Loads `CLAUDE.md` and `CLAUDE.local.md` from the working directory and every directory above it. Files concatenate, none overrides another; root comes first, so the closest file is read last | [CLAUDE.md and memory](knowledge/claude-code-configuration.md#claudemd-and-the-memory-hierarchy) |
| Imports | `@path/to/file` in CLAUDE.md; resolved relative to the importing file; up to four hops deep. Imported files load at launch, so they organize but do not save context | [CLAUDE.md and memory](knowledge/claude-code-configuration.md#claudemd-and-the-memory-hierarchy) |
| CLAUDE.md size | Target under 200 lines per file; a file over 4 MiB is skipped | [CLAUDE.md and memory](knowledge/claude-code-configuration.md#claudemd-and-the-memory-hierarchy) |
| CLAUDE.md authority | Delivered as a user message after the system prompt: context, not enforcement. To block an action, use permissions or a PreToolUse hook | [Configuration layers](knowledge/claude-code-configuration.md#the-configuration-layers-at-a-glance) |
| Rules | Rules without `paths` load at launch like `.claude/CLAUDE.md`; path-scoped rules load when Claude reads a matching file | [Path-scoped rules](knowledge/claude-code-configuration.md#path-scoped-rules) |
| Settings precedence | Managed, then command-line arguments, then `.claude/settings.local.json`, then `.claude/settings.json`, then `~/.claude/settings.json`. List keys such as `permissions.allow` merge across files | [Settings files](knowledge/claude-code-configuration.md#settings-files-and-precedence) |
| Permission rules | Evaluated deny, then ask, then allow; first match wins. A deny at any level cannot be overridden by an allow at another | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| Same-name skills | Enterprise over personal over project; a skill beats a `.claude/commands/` file of the same name; plugin skills are namespaced `/plugin-name:skill-name` | [Agent Skills](knowledge/claude-code-configuration.md#agent-skills) |
| Same-name subagents | Managed settings, then `--agents`, then `.claude/agents/`, then `~/.claude/agents/`, then a plugin's `agents/` | [Subagents](knowledge/claude-code-configuration.md#subagents) |
| Model choice | `/model`, then `--model`, then `ANTHROPIC_MODEL`, then the `model` setting, then `ANTHROPIC_DEFAULT_MODEL` | [Models and output styles](knowledge/claude-code-configuration.md#models-and-output-styles) |
| Hooks | Merge across settings files; managed hooks cannot be removed by other files | [Hooks](knowledge/claude-code-workflows.md#hooks) |
| After compaction | Project-root CLAUDE.md and unscoped rules are re-injected from disk. Path-scoped rules and nested CLAUDE.md files return only when a matching file is read again. Invoked skills come back capped at 5,000 tokens each, 25,000 in total | [Compaction](knowledge/context-engineering.md#compaction-context-editing-and-memory) |

### Frontmatter keys

| File | Keys | What to know | Go deeper |
|---|---|---|---|
| `SKILL.md` | `name`, `description`, `when_to_use`, `argument-hint`, `arguments`, `disable-model-invocation`, `user-invocable`, `allowed-tools`, `disallowed-tools`, `model`, `effort`, `context`, `agent`, `background`, `hooks`, `paths`, `shell`, `metadata`, `license`, `compatibility`. In Claude Code all are optional and `description` is recommended; the platform Agent Skills spec requires `name` and `description` | `context: fork` runs the skill in an isolated subagent that does not see the conversation history (not a fork of the conversation); since v2.1.218 it runs in the background by default, and `background: false` waits for the result. `agent`: `Explore`, `Plan`, `general-purpose` (default) or a custom agent. `disable-model-invocation: true`: only you can invoke it. `user-invocable: false`: only Claude can. The guide and the docs describe `allowed-tools` and `argument-hint` differently: see [Exam guide vs current docs](#exam-guide-vs-current-docs) | [Agent Skills](knowledge/claude-code-configuration.md#agent-skills) |
| `.claude/commands/*.md` | The skill keys except `name` and `paths` | `$ARGUMENTS` receives the text typed after the command | [Slash commands](knowledge/claude-code-configuration.md#slash-commands) |
| `.claude/rules/*.md` | `paths` only (YAML list or comma-separated globs); any other key is ignored | `paths: ["terraform/**/*"]` style scoping; unparseable YAML loads the rule unscoped | [Path-scoped rules](knowledge/claude-code-configuration.md#path-scoped-rules) |
| `.claude/agents/*.md` | `name` and `description` required; optional `tools`, `disallowedTools`, `model`, `permissionMode`, `maxTurns`, `skills`, `mcpServers`, `hooks`, `memory`, `background`, `effort`, `isolation`, `color`, `initialPrompt`, `omitClaudeMd`, `experimental` | `description` tells Claude when to delegate. Omitting `tools` inherits every tool available to subagents. `model`: `sonnet`, `opus`, `haiku`, `fable`, a full ID or `inherit`. `skills` preloads full skill content | [Subagents](knowledge/claude-code-configuration.md#subagents) |
| Output style `.md` | `name`, `description`, `keep-coding-instructions` (default `false`), `force-for-plugin` (plugin styles only) | Without `keep-coding-instructions: true`, a custom style drops Claude Code's software engineering instructions | [Models and output styles](knowledge/claude-code-configuration.md#models-and-output-styles) |
| `.claude-plugin/plugin.json` | `name` (the only required field), `version`, `description`, `dependencies` and more | Setting `version` means users update only when you bump it; a `dependencies` entry can pin a semver range such as `^2.0` | [Plugins](knowledge/claude-code-configuration.md#plugins-and-marketplaces) |

A user-only skill from the [skills docs](https://code.claude.com/docs/en/skills). Saved as `.claude/skills/deploy/SKILL.md`, it becomes `/deploy` (the directory name sets the command), runs in a forked subagent, and Claude never loads it on its own.

```yaml
---
name: deploy
description: Deploy the application to production
context: fork
disable-model-invocation: true
---

Deploy the application:
1. Run the test suite
2. Build the application
3. Push to the deployment target
```

### Built-in subagents

The three built-ins a `context: fork` skill can name as its `agent`. The docs also list helper agents such as `statusline-setup` (used for `/statusline`) and `claude-code-guide` (questions about Claude Code features).

| Subagent | What it does | Details | Go deeper |
|---|---|---|---|
| `Explore` | Fast, read-only search and analysis of a codebase (Write and Edit denied). Claude delegates to it to keep exploration results out of the main conversation | Claude sets a thoroughness level: quick, medium or very thorough. Since v2.1.198 it inherits the main conversation's model (capped at Opus on the Claude API) instead of always running on Haiku. Skips CLAUDE.md files and the git status snapshot. One-shot: it cannot be resumed | [Subagents](knowledge/claude-code-configuration.md#subagents) |
| `Plan` | Read-only research agent used during plan mode to gather context before presenting a plan | Skips CLAUDE.md files and the git status snapshot; one-shot | [Plan mode](knowledge/claude-code-workflows.md#plan-mode-or-direct-execution) |
| `general-purpose` | Complex, multi-step tasks that need both exploration and action, with every tool available to subagents | The default `agent` for a `context: fork` skill | [Subagents](knowledge/claude-code-configuration.md#subagents) |

- **Exam framing:** CCAR-F 3.4 uses the Explore subagent to isolate verbose discovery output and return summaries, so a multi-phase task does not exhaust the main context window.
- **Inheritance and nesting:** built-in subagents inherit the parent conversation's permissions. By default a subagent can spawn its own subagents up to three layers below the main conversation (`CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH` changes this).
- **Turning them off:** deny `Agent(Explore)` to disable one subagent; deny the `Agent` tool itself to stop all delegation.

### Exam guide vs current docs

!!! warning "Answer in the guide's terms"

    The CCAR-F guide (Version 1.0, effective July 2026) describes several features differently from the September 2026 docs. Answer in the guide's terms; know the current behavior for real work.

    | Topic | CCAR-F guide says | Docs say today |
    |---|---|---|
    | `allowed-tools` | Restricts tool access during a skill | Pre-approves the listed tools for the invoking turn and restricts nothing; `disallowed-tools`, deny rules or `--tools` remove tools |
    | `argument-hint` | Prompts developers for required parameters when invoked without arguments | A hint shown during autocomplete |
    | `/memory` | Verifies which memory files are loaded | Lists and opens memory files; `/context` shows what actually loaded |
    | Custom commands | Team commands go in `.claude/commands/` (Q4) | Still works, but slash commands and skills merged in v2.1.3; a same-named skill wins over a command file, and the docs prefer `.claude/skills/<name>/SKILL.md` for new work |
    | Path-scoped rules | Load when editing matching files | Load when Claude reads a matching file; rules without `paths` load unconditionally |
    | Subagent tool | The Task tool spawns subagents; `allowedTools` must include `"Task"` | Renamed Agent in v2.1.63; `Task(...)` still works as an alias |

### Commands

| Command | What it does | Go deeper |
|---|---|---|
| `/init` | Generates a starter CLAUDE.md from the codebase; suggests improvements if one exists | [CLAUDE.md and memory](knowledge/claude-code-configuration.md#claudemd-and-the-memory-hierarchy) |
| `/memory` | Lists and opens CLAUDE.md, CLAUDE.local.md and other memory files; toggles auto memory | [CLAUDE.md and memory](knowledge/claude-code-configuration.md#claudemd-and-the-memory-hierarchy) |
| `/context` | Colored grid of context usage, including which CLAUDE.md, rules and auto memory files loaded | [Large codebases](knowledge/claude-code-workflows.md#managing-context-in-large-codebases) |
| `/compact [instructions]` | Summarizes the conversation to free context, with optional focus (`/compact focus on the auth bug fix`) | [Large codebases](knowledge/claude-code-workflows.md#managing-context-in-large-codebases) |
| `/clear` (aliases `/reset`, `/new`) | New conversation with empty context; project memory stays | [Sessions](knowledge/claude-code-workflows.md#sessions-continue-resume-fork-and-rewind) |
| `/rewind` (aliases `/checkpoint`, `/undo`; or `Esc` twice on empty input) | Restores code, conversation or both to a checkpoint, or summarizes from a message. Does not capture files changed by Bash commands | [Sessions](knowledge/claude-code-workflows.md#sessions-continue-resume-fork-and-rewind) |
| `/resume [session]` (alias `/continue`), `/rename`, `/branch [name]` | Switch to, name, or branch a conversation | [Sessions](knowledge/claude-code-workflows.md#sessions-continue-resume-fork-and-rewind) |
| `/plan [description]` | Switches into plan mode; with a description, starts planning that task (`/plan fix the auth bug`) | [Plan mode](knowledge/claude-code-workflows.md#plan-mode-or-direct-execution) |
| `/model`, `/effort` | Switch the model (saved as your default) and the effort level | [Models and output styles](knowledge/claude-code-configuration.md#models-and-output-styles) |
| `/permissions` (alias `/allowed-tools`) | Manage allow, ask and deny rules by scope | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `/hooks` | Read-only browser of configured hooks; edit the settings JSON to change them | [Hooks](knowledge/claude-code-workflows.md#hooks) |
| `/mcp` | Manage MCP server connections and OAuth | [MCP in Claude Code](knowledge/claude-code-configuration.md#mcp-servers-in-claude-code) |
| `/agents` | Since v2.1.198 only prints a reminder: ask Claude, or edit `.claude/agents/` | [Subagents](knowledge/claude-code-configuration.md#subagents) |
| `/skills`, `/plugin`, `/reload-plugins` | List skills; install, enable or disable plugins; apply plugin changes without a restart | [Plugins](knowledge/claude-code-configuration.md#plugins-and-marketplaces) |
| `/config` (alias `/settings`), `/status` | Settings interface; Status shows version, model, account and the settings sources loaded | [Settings files](knowledge/claude-code-configuration.md#settings-files-and-precedence) |
| `/code-review` (alias `/review`), `/security-review` | Bundled skill that reviews the current diff for correctness bugs; security review of the branch's changes | [Automated code review](knowledge/claude-code-workflows.md#automated-code-review-that-engineers-trust) |
| `/batch <instruction>` | Splits a large change into 5 to 30 units, each run by a background subagent in its own git worktree | [Practices](knowledge/claude-code-workflows.md#practices-from-the-claude-code-documentation) |
| `/btw [question]` | A side question whose answer never enters the conversation history | [Large codebases](knowledge/claude-code-workflows.md#managing-context-in-large-codebases) |
| `/usage` (aliases `/cost`, `/stats`) | Session cost, plan usage limits and activity | [Monitoring and cost](knowledge/claude-code-workflows.md#monitoring-usage-and-cost) |
| `/add-dir <path>` | Adds a working directory for this session | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `/install-github-app` | Installs the Claude GitHub App, adds the secret and prepares a workflow (github.com only) | [Claude Code in CI/CD](knowledge/claude-code-workflows.md#claude-code-in-cicd) |

- **Input prefixes and keys:** `!` runs a shell command in the session, `@` mentions a file, `Shift+Tab` cycles permission modes, `Esc` interrupts, `Ctrl+B` backgrounds a running task, `Ctrl+G` opens a proposed plan in your editor.
- **Gone:** the `#` quick-memory shortcut (removed in v2.0.70; ask Claude to edit CLAUDE.md instead) and `/vim` (removed in v2.1.92).

### CLI flags and environment variables

| Flag or variable | Effect | Go deeper |
|---|---|---|
| `-p`, `--print` | Non-interactive: process the prompt, print the result, exit. Without it a CI job waits for input | [Headless mode](knowledge/claude-code-workflows.md#headless-mode-and-the-cli) |
| `--output-format` `text`, `json`, `stream-json` | Print-mode output; `json` returns `result`, `session_id` and metadata such as `total_cost_usd` | [Headless mode](knowledge/claude-code-workflows.md#headless-mode-and-the-cli) |
| `--json-schema '<schema>'` | With `--output-format json`, schema-validated output in the `structured_output` field; an invalid schema now exits with an error | [Headless mode](knowledge/claude-code-workflows.md#headless-mode-and-the-cli) |
| `--max-turns`, `--max-budget-usd` | Cap agentic turns or dollar spend (print mode only) | [Headless mode](knowledge/claude-code-workflows.md#headless-mode-and-the-cli) |
| `--allowedTools` | Listed tools run without a prompt, for example `--allowedTools "Bash(npm test)" "Read"` | [Headless mode](knowledge/claude-code-workflows.md#headless-mode-and-the-cli) |
| `--disallowedTools`, `--tools` | `--disallowedTools` takes deny rules: a bare tool name removes the tool, a scoped rule such as `Bash(rm *)` denies only matching calls. `--tools` restricts which built-in tools Claude can use (`""` disables all; MCP tools are unaffected) | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `--permission-mode` | Start in `default`, `acceptEdits`, `plan`, `auto`, `dontAsk` or `bypassPermissions` (`manual` is an alias of `default`) | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `--dangerously-skip-permissions` | Same as `--permission-mode bypassPermissions`; on Linux and macOS it refuses to start as root or under `sudo`, except inside a recognized sandbox | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `-c`, `--continue` | Most recent conversation in this directory; skips sessions created with `-p` or the Agent SDK unless you also pass `-p` | [Sessions](knowledge/claude-code-workflows.md#sessions-continue-resume-fork-and-rewind) |
| `-r`, `--resume` | A session by ID or name, or a picker; name sessions with `-n` (`--name`) or `/rename` | [Sessions](knowledge/claude-code-workflows.md#sessions-continue-resume-fork-and-rewind) |
| `--fork-session` | With `--resume` or `--continue`: a new session ID with the same history (the guide's `fork_session`) | [Sessions](knowledge/claude-code-workflows.md#sessions-continue-resume-fork-and-rewind) |
| `--append-system-prompt`, `--system-prompt` | Append to the default system prompt, or replace it (replacing drops the default tool guidance and safety instructions) | [Headless mode](knowledge/claude-code-workflows.md#headless-mode-and-the-cli) |
| `--model`, `--fallback-model` | Session model; comma-separated fallbacks when the primary is overloaded or unavailable | [Models and output styles](knowledge/claude-code-configuration.md#models-and-output-styles) |
| `--bare` | Skips auto-discovery of hooks, skills, commands, subagents, plugins, MCP servers, auto memory and CLAUDE.md; recommended for scripted and SDK calls, and the docs say it will become the default for `-p` in a future release. The exam guides do not mention it | [Headless mode](knowledge/claude-code-workflows.md#headless-mode-and-the-cli) |
| `--settings`, `--setting-sources` | Session settings (applied above user, project and local, below managed); choose which of `user`, `project`, `local` load | [Settings files](knowledge/claude-code-configuration.md#settings-files-and-precedence) |
| `--agent`, `--agents` | Run the whole session as a named subagent (its system prompt, tool restrictions and model; overrides the `agent` setting); define session-only subagents in JSON | [Subagents](knowledge/claude-code-configuration.md#subagents) |
| `--mcp-config`, `--strict-mcp-config` | Load MCP servers from JSON; use only those | [MCP in Claude Code](knowledge/claude-code-configuration.md#mcp-servers-in-claude-code) |
| `--add-dir`, `--worktree` (`-w`) | `--add-dir` adds working directories for file access; skills, command files and subagents in them also load, most other `.claude/` configuration does not (a `permissions.additionalDirectories` entry grants file access only). `-w` starts in an isolated git worktree at `<repo>/.claude/worktrees/<name>` | [Large codebases](knowledge/claude-code-workflows.md#managing-context-in-large-codebases) |
| `ANTHROPIC_API_KEY` | API key; used instead of a subscription login when set, and always used in `-p` | [Headless mode](knowledge/claude-code-workflows.md#headless-mode-and-the-cli) |
| `ANTHROPIC_MODEL`, `ANTHROPIC_DEFAULT_OPUS_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Session model; pin the version an alias resolves to | [Models and output styles](knowledge/claude-code-configuration.md#models-and-output-styles) |
| `CLAUDE_CODE_USE_BEDROCK`, `CLAUDE_CODE_USE_VERTEX` | Route to Amazon Bedrock or Google Cloud's Agent Platform | [Cloud platforms](knowledge/claude-api.md#claude-on-cloud-platforms) |
| `DISABLE_AUTO_COMPACT=1`, `DISABLE_COMPACT=1` | Turn off automatic compaction (manual `/compact` stays), or all compaction | [Large codebases](knowledge/claude-code-workflows.md#managing-context-in-large-codebases) |
| Traps | The CCAR-F Q10 rationale calls a `CLAUDE_HEADLESS` environment variable and a `--batch` flag non-existent features; `/batch` is a real slash command, but neither a `--batch` flag nor a `CLAUDE_HEADLESS` variable appears in the [CLI reference](https://code.claude.com/docs/en/cli-reference) or the [environment variables reference](https://code.claude.com/docs/en/env-vars) | [CCAR-F sample questions](claude-certified-architect-foundations.md#official-sample-questions) |

From the [permission modes](https://code.claude.com/docs/en/permission-modes), [headless](https://code.claude.com/docs/en/headless) and [CLI reference](https://code.claude.com/docs/en/cli-reference) pages:

```bash
# CI run: never prompts; any call outside the allowlist that would prompt is denied
claude -p "run the test suite" --permission-mode dontAsk --allowedTools "Bash(npm test)" "Read"

# Machine-readable result for the next pipeline step
claude -p "Summarize this project" --output-format json | jq -r '.result'

# Branch an earlier session into a new session ID
claude --resume abc123 --fork-session
```

### Hook events

A selection; the hooks reference lists more events, such as `PostToolBatch` and `MessageDisplay`.

| Event | Fires | Matcher and blocking | Go deeper |
|---|---|---|---|
| `SessionStart` | A session begins or resumes | Matcher `startup`, `resume`, `clear`, `compact` or `fork`. Exit 2 only shows stderr to the user; plain stdout becomes context. Only `command` and `mcp_tool` handlers | [Hooks](knowledge/claude-code-workflows.md#hooks) |
| `UserPromptSubmit` | You submit a prompt, before Claude processes it | No matcher. Exit 2 blocks and erases the prompt; plain stdout becomes context | [Hooks](knowledge/claude-code-workflows.md#hooks) |
| `PreToolUse` | Before a tool call runs | Matcher: tool name. Exit 2 blocks the call; JSON can allow, deny, ask or defer, or rewrite the input | [Hooks](knowledge/claude-code-workflows.md#hooks) |
| `PermissionRequest` | A tool call needs a permission decision | Matcher: tool name. Exit 2 is not honored; decide with `hookSpecificOutput.decision.behavior` (`allow` or `deny`) | [Hooks](knowledge/claude-code-workflows.md#hooks) |
| `PostToolUse`, `PostToolUseFailure` | After a tool call succeeds, or fails | Matcher: tool name. PostToolUse exit 2 cannot block (the tool already ran); stderr goes to Claude | [Hooks](knowledge/claude-code-workflows.md#hooks) |
| `Stop` | Claude finishes responding (not on a user interrupt; API errors fire `StopFailure`) | No matcher. Exit 2 keeps Claude working. Check `stop_hook_active`; Claude Code overrides after 8 consecutive blocks | [Hooks](knowledge/claude-code-workflows.md#hooks) |
| `SubagentStart`, `SubagentStop` | A subagent is spawned, or finishes | Matcher: agent type (`general-purpose`, `Explore`, `Plan`, custom names). SubagentStop exit 2 keeps the subagent working; SubagentStart cannot block | [Hooks](knowledge/claude-code-workflows.md#hooks) |
| `PreCompact`, `PostCompact` | Before and after compaction | Matcher `manual` or `auto`. PreCompact exit 2 blocks compaction | [Hooks](knowledge/claude-code-workflows.md#hooks) |
| `InstructionsLoaded` | A CLAUDE.md or `.claude/rules/*.md` file loads, at start or lazily | Use it to log which instruction files load, when and why | [Hooks](knowledge/claude-code-workflows.md#hooks) |
| `Notification` | Claude Code sends a notification, such as a permission prompt | The hooks guide's desktop-alert example uses it | [Hooks](knowledge/claude-code-workflows.md#hooks) |
| `SessionEnd` | A session terminates | Matcher `clear`, `resume`, `logout`, `prompt_input_exit`, `other`. Cannot block; cleanup only | [Hooks](knowledge/claude-code-workflows.md#hooks) |

- **Where hooks live:** under `"hooks"` in a settings file, nested event, then matcher group, then handlers. Plugins use `hooks/hooks.json`, and skill or subagent frontmatter can carry hooks too; there is no standalone hooks file for user or project config.
- **Handler types:** `command`, `http`, `mcp_tool`, `prompt`, `agent` (agent hooks are experimental; the docs prefer command hooks for production). Default timeouts: 600 seconds for command, http and mcp_tool; 30 for prompt; 60 for agent.
- **Exit codes:** 0 is success (stdout is parsed as JSON only if it starts with `{` and ends with `}`). 2 is a blocking error on events that can block, and even a JSON `permissionDecision` of `allow` cannot override it. For most events, 1 and other codes are non-blocking errors, so the action proceeds. Use exit 2 to enforce a policy.
- **PreToolUse JSON:** `hookSpecificOutput.permissionDecision` of `allow`, `deny`, `ask` or `defer`; when hooks disagree, deny beats defer beats ask beats allow. `updatedInput` replaces the whole input object. The old top-level `decision` and `reason` for PreToolUse are deprecated (`approve` and `block` map to `allow` and `deny`); events such as PostToolUse, Stop and UserPromptSubmit still use a top-level `decision`, whose only value is `block`.
- **Limits:** hooks can tighten restrictions but not loosen them past what permission rules allow; a PreToolUse deny blocks even in `bypassPermissions`. A PermissionRequest `allow` does not override a matching deny rule.
- **Matchers:** case-sensitive. `"*"`, `""` or no matcher matches every occurrence. A matcher of only letters, digits, `_`, `-`, spaces, commas and `|` is an exact name or list (`Edit|Write`); any other character makes it an unanchored JavaScript regex (`Edit.*` also matches `NotebookEdit`). Match a whole MCP server with `mcp__server__.*`.
- **Trust:** all matching hooks run in parallel. Interactive sessions hold back settings-file hooks until you accept the workspace trust dialog; `-p` and SDK sessions treat the folder as trusted, so a repository's committed hooks run there without a prompt.

Both snippets are from the [hooks reference](https://code.claude.com/docs/en/hooks). In a settings file, the `matcher` narrows to Bash calls and the `if` field (permission rule syntax) narrows to commands matching `rm *`, so the script only spawns when both match. The script, saved as `.claude/hooks/block-rm.sh` and made executable, blocks by writing the reason to stderr and exiting 2. The reference pairs the same settings block with a script that prints a JSON `permissionDecision` of `deny` when the command contains `rm -rf`; exit 2 and a JSON deny are two ways to block a call.

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "if": "Bash(rm *)",
            "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/block-rm.sh",
            "args": []
          }
        ]
      }
    ]
  }
}
```

```bash
#!/bin/bash
# Reads JSON input from stdin, checks the command
input=$(cat)
command=$(jq -r '.tool_input.command' <<<"$input")

if [[ "$command" == rm* ]]; then
  echo "Blocked: rm commands are not allowed" >&2
  exit 2  # Blocking error: tool call is prevented
fi

exit 0  # No decision: the normal permission flow applies
```

### Permission modes

| Mode | Runs without asking | Notes | Go deeper |
|---|---|---|---|
| `default` (labeled Manual) | Reads only | Best for reviewing every action yourself and sensitive work | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `acceptEdits` | Reads, file edits, and `mkdir`, `touch`, `rm`, `rmdir`, `mv`, `cp`, `sed` inside the working directories | Protected paths and critical-path removals still prompt | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `plan` | Reads (plus classifier-approved commands when auto mode is available) | Research and propose; no source edits until you approve the plan. Best for exploring a codebase before changing it | [Plan mode](knowledge/claude-code-workflows.md#plan-mode-or-direct-execution) |
| `auto` | Everything, with background safety checks | A separate classifier model reviews actions; it reduces prompts but does not guarantee safety | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `dontAsk` | Reads and pre-approved tools; anything that would prompt is denied | Best for locked-down CI and scripts; never appears in the `Shift+Tab` cycle | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `bypassPermissions` | Everything | Isolated containers and VMs only; same as `--dangerously-skip-permissions`; no protection against prompt injection | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |

- **Deny always wins:** deny rules block in every mode, including `bypassPermissions`, where allow rules have no effect.
- **No mode approves everything:** even `bypassPermissions` does not auto-approve tools matched by an explicit ask rule, `AskUserQuestion`, or `rm` and `rmdir` removals that target a critical path such as the filesystem root or your home directory.
- **Starting mode:** `--permission-mode` (or `--dangerously-skip-permissions`), then `permissions.defaultMode` in a settings file, then the built-in default. As of September 2026 the built-in default is auto on Pro, Max and Team plans, and Manual (`default`) for `-p` runs, the Agent SDK, Enterprise plans, Console API keys and the cloud platforms.
- **Switching:** `Shift+Tab` cycles `default`, `acceptEdits`, `plan` (optional modes slot in after `plan`; `dontAsk` never appears). A `defaultMode` of `auto` or `bypassPermissions` does not take effect from project or local settings.
- **Plan mode:** enter with `Shift+Tab`, `/plan` or `claude --permission-mode plan`; make it a project default by setting `permissions.defaultMode` to `"plan"` in `.claude/settings.json`.

| Permission rule | Matches | Go deeper |
|---|---|---|
| `Bash` (bare, in `deny`) | Removes the tool from Claude's context entirely | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `Bash(npm run build)` | Exactly that command | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `Bash(git log *)` | `git log` and its variants only; `Bash(git *)` allows every git command | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `Bash(ls *)` versus `Bash(ls*)` | The space matters: `ls *` matches `ls -la` but not `lsof`; `ls*` matches `lsof` too | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `Read(./.env)`, `Read(//Users/alice/secrets/**)` | File rules use gitignore syntax: `//` anchors at the filesystem root, `~/` at your home directory, a single `/` at the settings source (the primary working directory for project settings), and `./` or a bare path at the current directory. `/Users/alice/file` is therefore not an absolute path | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `WebFetch(domain:example.com)`, `WebFetch(domain:*.example.com)` | Fetches to `example.com`; the `*.` form matches any subdomain at any depth but not `example.com` itself | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `mcp__puppeteer`, `mcp__puppeteer__*` | Every tool from the `puppeteer` MCP server | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `Agent(Explore)` | The built-in Explore subagent (deny it to disable) | [Subagents](knowledge/claude-code-configuration.md#subagents) |
| `Skill(name)`, `Skill(name *)` | One skill, exactly or with any arguments | [Agent Skills](knowledge/claude-code-configuration.md#agent-skills) |

- Each part of a compound command (`&&`, `;`, pipes and similar) must match a rule on its own.
- A Bash deny rule is not a security boundary: `Bash(rm *)` does not stop `/bin/rm -rf build/`. For enforcement, add the sandbox or a PreToolUse hook.

A `permissions` block from the [settings reference](https://code.claude.com/docs/en/settings-reference): `npm run` scripts run without asking, `git push` commands ask first, a deny rule covers reads of `./.env` (a rule, not an OS-level block), and sessions start in `acceptEdits`.

```json
{
  "permissions": {
    "allow": ["Bash(npm run *)"],
    "ask": ["Bash(git push *)"],
    "deny": ["Read(./.env)"],
    "defaultMode": "acceptEdits"
  }
}
```

### settings.json keys

| Key | What it does | Go deeper |
|---|---|---|
| `permissions` | `allow`, `ask`, `deny`, `additionalDirectories`, `blockReadsOutsideWorkingDirectories`, `defaultMode`, `disableBypassPermissionsMode`, `disableAutoMode` | [Permissions](knowledge/claude-code-configuration.md#permissions-and-permission-modes) |
| `hooks` | Hook configuration keyed by event; merges across files | [Hooks](knowledge/claude-code-workflows.md#hooks) |
| `env` | Environment variables for every session and its subprocesses; beats the same variable exported in your shell | [Settings files](knowledge/claude-code-configuration.md#settings-files-and-precedence) |
| `model` | Default model for new sessions; `--model` and `ANTHROPIC_MODEL` beat it for one session | [Models and output styles](knowledge/claude-code-configuration.md#models-and-output-styles) |
| `availableModels` (enforced when set in managed settings) | Allowlist of models users can pick, applied to `/model`, `--model`, subagents and skill frontmatter | [Managed settings](knowledge/claude-code-configuration.md#managed-settings-for-organizations) |
| `enabledPlugins`, `extraKnownMarketplaces` | `"plugin-name@marketplace-name": true`; team marketplaces apply after each teammate trusts the folder | [Plugins](knowledge/claude-code-configuration.md#plugins-and-marketplaces) |
| `outputStyle` | Style name, case-sensitive (`Explanatory`, not `explanatory`) | [Models and output styles](knowledge/claude-code-configuration.md#models-and-output-styles) |
| `sandbox` | OS-level isolation for Bash commands on macOS, Linux and WSL2 (`enabled`, `failIfUnavailable`, `allowUnsandboxedCommands` and more); native Windows is not supported | [Security controls](knowledge/security-and-governance.md#claude-code-security-controls) |
| `claudeMd` (managed only), `claudeMdExcludes` | Organization CLAUDE.md text inside managed settings; skip CLAUDE.md files by glob (managed ones cannot be excluded) | [Managed settings](knowledge/claude-code-configuration.md#managed-settings-for-organizations) |
| `disableAllHooks`, `allowManagedHooksOnly` (managed) | Turn hooks off (cannot disable managed hooks outside managed settings); run only managed hooks | [Hooks](knowledge/claude-code-workflows.md#hooks) |
| `apiKeyHelper` | Shell command whose output is sent as both the `X-Api-Key` and `Authorization: Bearer` headers; cached for five minutes by default | [Settings files](knowledge/claude-code-configuration.md#settings-files-and-precedence) |
| `cleanupPeriodDays` | Transcript retention in days; default 30 | [Sessions](knowledge/claude-code-workflows.md#sessions-continue-resume-fork-and-rewind) |

### Claude Code in CI/CD

| Item | Value | Go deeper |
|---|---|---|
| GitHub Action | `anthropics/claude-code-action@v1`; inputs `prompt` and `claude_args` (any CLI flags). No `prompt`: Claude waits for `@claude` in a comment. With a `prompt`: runs without waiting for a mention. Upgrading from beta: `prompt` replaces `direct_prompt`, `mode` is removed, and `max_turns` and `model` move into `claude_args` | [Claude Code in CI/CD](knowledge/claude-code-workflows.md#claude-code-in-cicd) |
| Credentials | Repository secret `ANTHROPIC_API_KEY`, or `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token` for a subscription; never commit keys | [Claude Code in CI/CD](knowledge/claude-code-workflows.md#claude-code-in-cicd) |
| Non-interactive runs | `claude -p` so the job does not wait for input; `--output-format json` with `--json-schema` for machine-parseable findings to post as inline PR comments | [Headless mode](knowledge/claude-code-workflows.md#headless-mode-and-the-cli) |
| GitLab CI/CD | Beta, maintained by GitLab; one job in `.gitlab-ci.yml` plus a masked variable | [Claude Code in CI/CD](knowledge/claude-code-workflows.md#claude-code-in-cicd) |
| Code Review (managed service) | Research preview for Team and Enterprise (not with Zero Data Retention); the check run always completes neutral, so it never blocks merging through branch protection; tune it with `REVIEW.md` | [Automated code review](knowledge/claude-code-workflows.md#automated-code-review-that-engineers-trust) |
| Exam rules (CCAR-F 3.6) | CLAUDE.md as project context for CI, test generation and re-review after new commits: see [Claude Code layer rules](#claude-code-layer-rules); independent review instances: see [Reliability rules](#reliability-rules) | [Automated code review](knowledge/claude-code-workflows.md#automated-code-review-that-engineers-trust) |

## Agent SDK cheat sheet

The Claude Agent SDK is a Python and TypeScript library that runs the Claude Code binary inside a process you operate, so your agent gets Claude Code's built-in tools, agent loop, context management, permissions, sessions and hooks. The full treatment is in [The Claude Agent SDK](knowledge/agents-and-agent-sdk.md#the-claude-agent-sdk); this section is the revision sheet.

On the exams, the CCAR-F appendix lists the SDK's agent definitions, agentic loops, `stop_reason` handling, hooks, subagent spawning via the Task tool and `allowedTools` configuration, and three of its six scenarios (1, 3 and 4) build with the SDK. On CCDV-F, the Agent Construction with Claude skill (5.3%) covers the Agent SDK, custom agent loops and harnesses, managed agent deployment models and hooks for deterministic actions.

### Four ways to build an agent

| Option | Who runs the loop | Pick it when |
|---|---|---|
| Client SDK (Messages API) | Your code: you write the tool loop, or let the beta tool runner drive it | You need custom agent loops and fine-grained control. For human-in-the-loop approval, custom logging or conditional execution, use the manual loop rather than the tool runner |
| Agent SDK | The SDK, in your process; one agent session maps to one `claude` subprocess | You want Claude Code's tools, permissions, sessions and hooks inside your own application |
| CLI `claude -p` as a subprocess | Claude Code | You drive the same loop from a language other than Python or TypeScript (`-p` with `--output-format json`) |
| Claude Managed Agents | An Anthropic-hosted harness, with sessions in an Anthropic-managed cloud sandbox or a self-hosted sandbox | Long-running or asynchronous work where you do not want to build the loop, sandbox or tool execution layer |

[CCDV-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) names "managed agent deployment models (self-hosted vs. Anthropic-hosted)". In current docs that maps to where Managed Agents sessions run: an Anthropic-managed cloud sandbox, or a self-hosted sandbox on your own infrastructure. Keep the two meanings of self-hosted apart:

- **Self-hosting the Agent SDK**: the library and its `claude` subprocess run in a process you operate, so the loop, tools and session files are all on your side.
- **A Managed Agents self-hosted sandbox**: orchestration stays on Anthropic's side and tool execution moves into infrastructure you control (the agent's code, filesystem and network egress stay in your environment), but tool inputs and outputs still flow to Anthropic's control plane so Claude can see results. The docs call it a good fit for data that cannot leave your network boundary, internal services that are not publicly routable, or your own compliance and audit controls.

### Names that changed

| Older name | Current name | Note |
|---|---|---|
| Claude Code SDK | Claude Agent SDK | Renamed in Anthropic's September 29, 2025 announcement |
| `@anthropic-ai/claude-code` | `@anthropic-ai/claude-agent-sdk` | TypeScript package |
| `claude-code-sdk` (`claude_code_sdk`) | `claude-agent-sdk` (`claude_agent_sdk`) | Python package and import |
| `ClaudeCodeOptions` | `ClaudeAgentOptions` | Python options type |
| Task tool | Agent tool | Renamed in Claude Code v2.1.63; `Task` is still accepted as an alias |
| Headless mode | Non-interactive mode | Same `-p` flag, same behavior |

!!! warning "Exam guide vs current docs: Task or Agent"

    [CCAR-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) (July 2026) names "The Task tool as the mechanism for spawning subagents" and says `allowedTools` must include "Task". Current docs call the tool `Agent`: it appears as `"Agent"` in `tool_use` blocks but still as `"Task"` in the `system:init` tools list, and `Task(...)` references keep working. Expect "Task" on the exam, write `"Agent"` in new code, and match both names when you detect subagent calls. The docs examples list `"Agent"` in `allowed_tools`, although `allowed_tools` only auto-approves and `Agent` does not ask before running in `default` mode. On the exam, treat including the spawn tool in `allowedTools` as the requirement the guide states.

### Minimal agent

=== "Python"

    ```python
    import asyncio
    from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, ResultMessage

    async def main():
        # Agentic loop: streams messages as Claude works
        async for message in query(
            prompt="Review utils.py for bugs that would cause crashes. Fix any issues you find.",
            options=ClaudeAgentOptions(
                allowed_tools=["Read", "Edit", "Glob"],  # Auto-approve these tools
                permission_mode="acceptEdits",  # Auto-approve file edits
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if hasattr(block, "text"):
                        print(block.text)
                    elif hasattr(block, "name"):
                        print(f"Tool: {block.name}")
            elif isinstance(message, ResultMessage):
                print(f"Done: {message.subtype}")

    asyncio.run(main())
    ```

=== "TypeScript"

    ```typescript
    import { query } from "@anthropic-ai/claude-agent-sdk";

    for await (const message of query({
      prompt: "Review utils.py for bugs that would cause crashes. Fix any issues you find.",
      options: {
        allowedTools: ["Read", "Edit", "Glob"], // Auto-approve these tools
        permissionMode: "acceptEdits" // Auto-approve file edits
      }
    })) {
      if (message.type === "assistant" && message.message?.content) {
        for (const block of message.message.content) {
          if ("text" in block) {
            console.log(block.text);
          } else if ("name" in block) {
            console.log(`Tool: ${block.name}`);
          }
        }
      } else if (message.type === "result") {
        console.log(`Done: ${message.subtype}`);
      }
    }
    ```

### Options that appear in questions

| Python (`ClaudeAgentOptions`) | TypeScript | What it does |
|---|---|---|
| `allowed_tools` | `allowedTools` | Auto-approves the listed tools. It does not restrict Claude to them: unlisted tools stay available and fall through to the permission mode and `canUseTool` |
| `disallowed_tools` | `disallowedTools` | Blocks the listed tools regardless of other settings. A bare name (`"Bash"`) removes the tool from context; a scoped rule (`Bash(rm *)`) denies matching calls in every mode, `bypassPermissions` included |
| `tools` | `tools` | Availability: `{"type": "preset", "preset": "claude_code"}` for Claude Code's default tools, a list to restrict built-ins, `[]` to remove every built-in |
| `permission_mode` | `permissionMode` | One of the six modes below |
| `can_use_tool` | `canUseTool` | Your approval callback, invoked only when the permission flow falls through to a prompt |
| `hooks` | `hooks` | Callback hooks per event, with matchers |
| `agents` | `agents` | Programmatic subagents (`AgentDefinition`) |
| `mcp_servers` | `mcpServers` | MCP servers; each key becomes the `{server_name}` in `mcp__{server_name}__{tool_name}` |
| `system_prompt` | `systemPrompt` | Default is a minimal prompt; `{"type": "preset", "preset": "claude_code"}` (optionally with `append`) gives Claude Code's prompt; or pass your own string |
| `setting_sources` | `settingSources` | `"user"`, `"project"`, `"local"`. Omitted means all three, like the CLI; `[]` means programmatic configuration only. Managed policy is read regardless |
| `max_turns` | `maxTurns` | Caps tool-use turns only. Default: no limit |
| `max_budget_usd` | `maxBudgetUsd` | Spend cap. Subagent spend counts toward it, and at the cap Claude Code refuses new subagents (`Budget limit reached`) and stops background ones; these subagent enforcement behaviors need Claude Code v2.1.217 or later. Default: no limit; the docs call a budget a good production default |
| `effort` | `effort` | `low`, `medium`, `high`, `xhigh`, `max`, per session or per subagent; independent of extended thinking |
| `resume`, `continue_conversation`, `fork_session` | `resume`, `continue`, `forkSession` | Resume a session by ID, continue the most recent one, or branch into a new session ID |
| `output_format` | `outputFormat` | `{"type": "json_schema", "schema": {...}}`; the result then carries `structured_output` |
| `env` | `env` | Python merges it over the inherited environment; TypeScript replaces it, so spread `process.env` |

The minimal default system prompt leaves out the Claude Code preset's safety instructions and environment context, and it differs from `claude -p`, which uses the Claude Code system prompt. CLAUDE.md loading is controlled by `setting_sources`, not by the preset.

### Permission modes and evaluation order

The six modes (`default`, `dontAsk`, `acceptEdits`, `plan`, `auto`, `bypassPermissions`) are defined in the [Claude Code cheat sheet](#claude-code-cheat-sheet). What changes when the SDK runs them:

| Mode | SDK behavior to remember |
|---|---|
| `default` | The starting mode for `claude -p` and Agent SDK sessions. Calls that need approval go to `canUseTool`; calls that need none (reads in the working directories, read-only Bash, the `Agent` tool) run whether or not you list them |
| `dontAsk` | Pair it with `allowedTools` for a locked-down agent: anything not pre-approved is denied instead of prompting |
| `acceptEdits` | Auto-approves file edits and filesystem Bash commands, but not MCP tools: list those in `allowedTools` |
| `plan` | Edits are never auto-approved; they prompt through your `canUseTool` callback |
| `auto` | A model classifier approves or denies permission prompts |
| `bypassPermissions` | `allowed_tools` does not narrow it: `allowed_tools=["Read"]` still approves `Bash`, `Write` and `Edit`. Deny rules and hook denies still block, and a matching ask rule still sends the call to `canUseTool`. TypeScript also needs `allowDangerouslySkipPermissions: true`. Reserve it for CI, containers or other isolated environments |

A tool request passes through these steps in order:

1. Hooks
2. Deny rules
3. Ask rules
4. Permission mode
5. Allow rules
6. The `canUseTool` callback

- A hook that returns `allow` does not skip the deny and ask rules.
- Auto-approved calls never reach `canUseTool`. A check that must run on every call belongs in a `PreToolUse` hook, whose deny applies even in `bypassPermissions`.
- `allowed_tools=["*"]` or `["mcp__*"]` is ignored with a startup warning; globs work only after a literal `mcp__<server>__` prefix, as in `mcp__puppeteer__*`.
- `set_permission_mode()` / `setPermissionMode()` changes the mode mid-session.

### Messages and results

| Item | What to know |
|---|---|
| Message types | `SystemMessage`, `AssistantMessage`, `UserMessage`, `StreamEvent`, `ResultMessage` |
| `SystemMessage` subtypes | `init` (session metadata), `compact_boundary` (fires after compaction), `informational` (plain-text status banners) and `worker_shutting_down`. In TypeScript every subtype other than `init` is its own message type |
| `ResultMessage` subtypes | `success`, `error_max_turns`, `error_max_budget_usd`, `error_during_execution`, `error_max_structured_output_retries` |
| On every result | `total_cost_usd`, `usage`, `num_turns`, `session_id`, so you can track cost and resume even after errors. After a session crash the final `error_during_execution` may carry zeroed cost fields |
| Only on `success` | `result`, the final text: check `subtype` before reading it |
| `stop_reason` on the result | Common values `end_turn`, `max_tokens`, `refusal` |
| Reading content | Python uses `isinstance()`; TypeScript checks `message.type` and reads `message.message.content` |
| Loop hygiene | Iterate the stream to completion, because trailing system events can follow the result. A single-shot `query()` yields the error result and then raises |

A *turn* in the SDK is one tool-use round trip, and `max_turns` counts only those. The Claude Code glossary uses *turn* for one complete response to a user message, however many tool calls it contains.

| | `query()` | `ClaudeSDKClient` (Python) |
|---|---|---|
| Session | New session per call unless you pass `resume` or `continue_conversation=True` | Each `client.query()` continues the same session |
| Interrupts | Not supported | Supported |
| Use for | One-off tasks | Chat interfaces, or when the next action depends on Claude's response |

TypeScript has no session-holding client: pass `continue: true` on later `query()` calls. Streaming input mode is the SDK's preferred mode; single-message input suits stateless environments such as a lambda function.

### Hooks

| Topic | What to know |
|---|---|
| Python events (10) | `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `UserPromptSubmit`, `Stop`, `SubagentStop`, `PreCompact`, `Notification`, `SubagentStart`, `PermissionRequest` |
| TypeScript extras | Many more, including `SessionStart`, `SessionEnd`, `PostCompact` and `PermissionDenied`. In Python, `SessionStart` and `SessionEnd` exist only as shell hooks in settings files |
| Matcher | The tool name; MCP tools are `mcp__<server>__<action>`. Matchers never see file paths, so filter on `tool_input` inside the callback. No matcher means every event of that type |
| `PreToolUse` output | Inside `hookSpecificOutput`: `permissionDecision` (`allow`, `deny`, `ask`, `defer`), `permissionDecisionReason`, `updatedInput` |
| `PostToolUse` output | `additionalContext` appends information to the tool result; `updatedToolOutput` replaces the output before Claude sees it, for any tool in both SDKs. `updatedMCPToolOutput` (MCP output only) is deprecated |
| Allow unchanged | Return `{}` |
| Several matching hooks | They run in parallel; `deny` beats `defer`, which beats `ask`, which beats `allow` |
| Who reads what | A deny's `permissionDecisionReason` reaches Claude, so it stops retrying; `systemMessage` reaches the user, not the model |
| Timeouts | 600 seconds for most events; 30 for `UserPromptSubmit`, `PreModelSwitch` and `PostModelSwitch`; 10 for `MessageDisplay`. On `PreToolUse`, a timed-out SDK callback blocks the tool call (Claude gets an error result naming the timeout), while a timed-out `command`, `http` or `mcp_tool` hook does not block |
| Context cost | None: hooks run in your process, not in the context window |
| Shell hooks in settings | Exit codes and JSON decisions work as in the [Claude Code cheat sheet](#hook-events): a policy hook must exit 2 |

[CCAR-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)'s "tool call interception" is the `PreToolUse` event, and its data-normalization hooks (1.5-S1) are `PostToolUse`. The guide's rule is to choose hooks over prompt-based enforcement when business rules require guaranteed compliance (1.5-S3). The block below enforces the guide's refund example. It is adapted from the SDK hooks docs' `block_etc_writes` example; the server name `billing`, the tool `process_refund` and the `amount` field are illustrative, and the limit of 500 comes from the "refunds exceeding &#36;500" example in CCAR-F 1.5-S2.

```python
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

REFUND_LIMIT = 500

async def enforce_refund_limit(input_data, tool_use_id, context):
    amount = input_data["tool_input"].get("amount", 0)
    if amount > REFUND_LIMIT:
        return {
            # Top-level field: message shown to the user
            "systemMessage": "Refund above limit routed to a human agent.",
            "hookSpecificOutput": {
                "hookEventName": input_data["hook_event_name"],
                "permissionDecision": "deny",
                # Shown to Claude for "deny", so it can switch to the escalation workflow
                "permissionDecisionReason": "Refunds over $500 require human approval. Escalate to a human agent with a structured handoff summary instead.",
            },
        }
    return {}

options = ClaudeAgentOptions(
    hooks={"PreToolUse": [HookMatcher(matcher="mcp__billing__process_refund", hooks=[enforce_refund_limit])]}
)
```

More hook patterns, including a prerequisite gate and timestamp normalization: [Hooks in the SDK](knowledge/agents-and-agent-sdk.md#hooks-in-the-sdk).

### Subagents

| Topic | What to know |
|---|---|
| Spawn tool | `Agent` (formerly `Task`); the docs examples add `"Agent"` to `allowed_tools` |
| Three ways to define one | The `agents` option (recommended), markdown files in `.claude/agents/`, or the built-in `general-purpose` subagent |
| `AgentDefinition` required fields | `description` (when to use it; drives automatic delegation) and `prompt` (its system prompt) |
| `AgentDefinition` optional fields | `tools` (omit it and the subagent inherits every tool available to subagents), `model` (an alias such as `fable`, `opus`, `sonnet`, `haiku`, `inherit`, or a full model ID), plus `disallowedTools`, `skills`, `memory`, `mcpServers`, `maxTurns`, `effort`, `permissionMode` and others. Field names stay camelCase even in Python |
| Context in (non-fork subagent) | Its own system prompt (`prompt`) and the Agent tool's prompt string, plus project CLAUDE.md (loaded through `setting_sources`; TypeScript `omitClaudeMd` skips it) and its tool definitions. Never the parent's conversation history, tool results or system prompt, so put file paths, findings and decisions in the Agent tool's prompt. A fork is the exception: it inherits the parent conversation |
| Context out | Only the subagent's final message, returned as the Agent tool result; the parent may summarize it, so ask for verbatim output if you need it |
| Parallel work | Emit several spawn calls in one response; parallel subagents finish in the time of the slowest one |
| Foreground or background | Subagents run in the background by default; Claude sets `run_in_background: false` when it needs the result first |
| Tracing | Look for `tool_use` blocks named `Agent` (or `Task`); messages from inside a subagent carry `parent_tool_use_id` |
| Growth limits | Depth `CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH` (default 3 layers since Claude Code v2.1.219), concurrency `CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS` (default 20), spend `max_budget_usd`. Set the two variables through the `env` option |
| Name clash | Programmatic agents beat filesystem agents with the same name |
| Not available, or off by default | `AskUserQuestion` is not currently available in subagents spawned through the Agent tool; agent teams are a CLI feature, not an SDK option; fork mode is off by default in the SDK and in `-p` mode |

```python
from claude_agent_sdk import ClaudeAgentOptions, AgentDefinition

options = ClaudeAgentOptions(
    # Auto-approve these tools
    allowed_tools=["Read", "Grep", "Glob", "Agent"],
    agents={
        "code-reviewer": AgentDefinition(
            # description tells Claude when to use this subagent
            description="Expert code review specialist. Use for quality, security, and maintainability reviews.",
            # prompt defines the subagent's behavior and expertise
            prompt="You are a code review specialist with expertise in security, performance, and best practices. ...",  # shortened here
            # tools restricts what the subagent can do (read-only here)
            tools=["Read", "Grep", "Glob"],
            # model overrides the default model for this subagent
            model="sonnet",
        ),
    },
)
```

### Sessions

| Need | Python | TypeScript | CLI |
|---|---|---|---|
| Continue the most recent session | `continue_conversation=True` | `continue: true` | `claude --continue` |
| Resume a specific session | `resume=session_id` | `resume: sessionId` | `claude --resume <name>` |
| Branch without changing the original | `resume=...` plus `fork_session=True` | `resume` plus `forkSession: true` | `--fork-session` with `--resume` or `--continue`, or `/branch` in a session |

- Capture the ID from `session_id` on the result message; it is present on every result subtype.
- Sessions persist the conversation, not the filesystem. Forking branches history only, so a forked agent's file edits are real. File checkpointing tracks Write, Edit and NotebookEdit changes (not Bash), and rewinding files does not rewind the conversation.
- Transcripts live in `~/.claude/projects/<encoded-cwd>/*.jsonl`. TypeScript `persistSession: false` turns persistence off, and such sessions cannot be resumed.
- `resume_session_at` (`resumeSessionAt` in TypeScript) loads a session only up to a given message UUID, usually together with `fork_session`.
- Sessions created by `claude -p` or the Agent SDK are left out of the session picker and out of `claude --continue` (`claude -p --continue` includes them); resume them by ID.
- Name a CLI session at startup with `claude -n auth-refactor` or mid-session with `/rename`.
- To move work between hosts, passing captured results into a fresh session's prompt is often more reliable than shipping transcript files.

```python
# Fork: branch from session_id into a new session; the original is untouched
async for message in query(
    prompt="Instead of JWT, outline how OAuth2 would work for the auth module",
    options=ClaudeAgentOptions(
        resume=session_id,
        fork_session=True,
        max_turns=5,
    ),
):
    if isinstance(message, ResultMessage):
        forked_id = message.session_id  # The fork's ID, distinct from session_id
```

When to resume and when to start fresh is a decision rule: see [Context rules](#context-rules).

### SDK custom tools

Custom tools are functions you expose through the SDK's in-process MCP server, which runs inside your application rather than as a separate process. A tool has four parts (name, description, input schema, handler). TypeScript schemas are Zod; Python takes a dict of names to types or a full JSON Schema dict. The blocks below condense the weather-tool examples from the custom tools docs (handler bodies elided) and include the docs' `readOnlyHint` annotation.

=== "Python"

    ```python
    from typing import Any
    from claude_agent_sdk import tool, create_sdk_mcp_server, query, ClaudeAgentOptions, ToolAnnotations

    @tool(
        "get_temperature",
        "Get the current temperature at a location",
        {"latitude": float, "longitude": float},
        annotations=ToolAnnotations(readOnlyHint=True),  # Lets Claude batch this with other read-only calls
    )
    async def get_temperature(args: dict[str, Any]) -> dict[str, Any]:
        ...
        return {"content": [{"type": "text", "text": "Temperature: ..."}]}

    weather_server = create_sdk_mcp_server(
        name="weather",
        version="1.0.0",
        tools=[get_temperature],
    )

    options = ClaudeAgentOptions(
        mcp_servers={"weather": weather_server},
        allowed_tools=["mcp__weather__get_temperature"],
    )
    ```

=== "TypeScript"

    ```typescript
    import { tool, createSdkMcpServer, query } from "@anthropic-ai/claude-agent-sdk";
    import { z } from "zod";

    const getTemperature = tool(
      "get_temperature",
      "Get the current temperature at a location",
      { latitude: z.number(), longitude: z.number() },
      async (args) => ({ content: [{ type: "text", text: "..." }] }),
      { annotations: { readOnlyHint: true } } // Lets Claude batch this with other read-only calls
    );

    const weatherServer = createSdkMcpServer({ name: "weather", version: "1.0.0", tools: [getTemperature] });

    for await (const message of query({
      prompt: "What's the temperature in San Francisco?",
      options: { mcpServers: { weather: weatherServer }, allowedTools: ["mcp__weather__get_temperature"] }
    })) { /* ... */ }
    ```

- The handler returns `content` (required) plus optional `structuredContent` and `isError`. The Python `@tool` decorator forwards only `content` and `is_error`.
- A handler error does not stop the loop: an uncaught exception becomes an error result, and returning `isError: true` lets you write the message Claude reads.
- Custom tools run sequentially by default. `readOnlyHint: true` lets the tool run in parallel with other read-only tools; the other annotations are informational. Annotations are metadata, not enforcement.
- `tools: []` removes every built-in, so Claude can use only your MCP tools.

### Structured output, cost and hosting

- **Structured output**: pass a JSON Schema in `output_format` / `outputFormat`. The SDK validates with JSON Schema draft-07 and re-prompts on a mismatch; failure ends with `error_max_structured_output_retries`, and a `success` without `structured_output` also counts as failure. The CLI equivalent is `--json-schema` (print mode only).
- **Cost fields**: `total_cost_usd` and `costUSD` are client-side estimates, not billing data, so do not bill end users from them. `usage` excludes subagent activity; `modelUsage` (`model_usage` in Python) covers the whole tree. A resumed call reports the whole session's spend, while `max_budget_usd` counts only the call's own spend.
- **Hosting**: the SDK supervises a `claude` subprocess that owns a shell, a working directory and on-disk sessions. Transcripts, CLAUDE.md memory and working files do not survive a container restart; a `SessionStore` mirrors transcripts only. A starting size is 1 GiB RAM, 5 GiB disk and 1 CPU per agent.
- **Multi-tenant isolation**: `setting_sources=[]`, `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`, and a per-tenant `CLAUDE_CONFIG_DIR` and `cwd`. Keep credentials out of the agent's environment: a proxy outside it injects them, so the agent never sees them.
- **Managed Agents, as of September 2026**: four concepts (Agent, Environment, Session, Events), the `managed-agents-2026-04-01` beta header, permission policies (`always_allow`, `always_ask`, `auto`) set per toolset with per-tool overrides (the agent toolset defaults to `always_allow`, MCP toolsets to `always_ask`, and custom tools are not governed by policies), no server-side `max_turns`, and no eligibility for Zero Data Retention or HIPAA BAA coverage.

Deeper coverage: [Subagents in the SDK](knowledge/agents-and-agent-sdk.md#subagents-in-the-sdk), [Permissions and enforcement](knowledge/agents-and-agent-sdk.md#permissions-and-enforcement), [Sessions, resumption and forking](knowledge/agents-and-agent-sdk.md#sessions-resumption-and-forking), [Custom tools in the SDK](knowledge/agents-and-agent-sdk.md#custom-tools-in-the-sdk) and [Deployment models](knowledge/agents-and-agent-sdk.md#deployment-models).

## MCP cheat sheet

The Model Context Protocol is the open standard Anthropic [open-sourced](https://www.anthropic.com/news/model-context-protocol) on November 25, 2024 "for connecting AI assistants to the systems where data lives". All MCP messages are JSON-RPC 2.0. As of September 2026 the current specification revision is 2026-07-28; the one before it was 2025-11-25. The full treatment starts at [MCP architecture](knowledge/tool-use-and-mcp.md#mcp-architecture).

On the exams, MCP sits in CCAR-F Domain 2 (Tool Design & MCP Integration, 18%) and CCDV-F Domain 8 (Tools and MCPs, 10.6%), whose MCP Server Development skill covers authoring, deploying and integrating servers, their resources, tools and prompts, and communication patterns. CCAR-P asks you to choose among integration mechanisms (MCP, API/CLI, agent-to-agent).

### Architecture

| Role | What it is | Rules to remember |
|---|---|---|
| Host | The LLM application that initiates connections (the MCP docs' example is VS Code) | Creates one MCP client per server; enforces security policies and consent; handles authorization decisions; aggregates context. The full conversation history stays with the host |
| Client | A connector inside the host | Talks to exactly one server |
| Server | A service that provides context and capabilities | Receives only the context it needs and cannot see into other servers |

MCP has two layers: a data layer (the JSON-RPC protocol, primitives and notifications) and a transport layer (connection setup, message framing and authorization). An application gathers the tools of every connected server into one registry for the model.

### Server primitives: who controls what

| Primitive | Controlled by | Use it for | Methods |
|---|---|---|---|
| Tools | The model, which decides when to call them | Actions: writing to databases, calling APIs, modifying files | `tools/list`, `tools/call` |
| Resources | The application, which decides how to use them | Passive, read-only context such as file contents, database schemas and API documentation. Each resource has a URI; templates such as `travel://activities/{city}/{category}` use RFC 6570 | `resources/list`, `resources/read`, `resources/templates/list` |
| Prompts | The user, who selects them explicitly (often as slash commands) | Pre-built instruction templates for a workflow | `prompts/list`, `prompts/get` |

The [CCAR-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) in-scope list sums up the design rule as "resources for content catalogs, tools for actions": expose issue summaries, documentation hierarchies or database schemas as resources so the agent does not burn tool calls exploring.

| Client feature | What it does | Status in 2026-07-28 |
|---|---|---|
| Elicitation | The server asks the user for input: form mode (flat object of primitive fields) or URL mode (an out-of-band URL). Passwords, API keys, access tokens and payment credentials must go through URL mode | Current |
| Sampling | The server requests an LLM completion through the client, with no server API keys | Deprecated: integrate directly with LLM provider APIs |
| Roots | `file://` URIs telling servers which directories matter; informational guidance, not access control | Deprecated: pass paths as tool parameters, resource URIs or configuration |

### Transports

| Transport | How it works | Typical use |
|---|---|---|
| stdio | The client launches the server as a subprocess; newline-delimited JSON-RPC over stdin and stdout; stderr may carry logs; nothing but valid MCP messages may go to stdout | Local processes on the same machine, usually one client; credentials come from the environment |
| Streamable HTTP | One MCP endpoint (for example `https://example.com/mcp`); every client message is a new HTTP POST; the `Accept` header lists `application/json` and `text/event-stream`; the server answers with one JSON object or an SSE stream | Remote servers with many clients; bearer tokens, API keys or custom headers, with OAuth recommended for tokens |
| HTTP+SSE (2024-11-05) | The older transport that Streamable HTTP replaced in revision 2025-03-26 | Deprecated; migrate to Streamable HTTP |
| Custom (Unix domain sockets, TCP) | Any bidirectional channel; byte-stream transports SHOULD reuse the stdio framing | Custom setups; the 2026-07-28 transports page mentions sockets only as custom transports |

- An HTTP server MUST validate the `Origin` header (403 when invalid) to stop DNS rebinding, SHOULD bind to localhost rather than `0.0.0.0` when local, and SHOULD authenticate every connection.
- The [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) lists "communication patterns (stdio, sockets, client vs. server)"; in the spec, sockets fall under the custom transports in the table above. Claude Code also accepts WebSocket servers (`type: "ws"`), configured only through `.mcp.json` or `claude mcp add-json`.

### Two protocol eras

| | Legacy (2025-11-25 and earlier) | Modern (2026-07-28) |
|---|---|---|
| Connection start | `initialize` request (protocol version, capabilities, `clientInfo`), a server response (capabilities, `serverInfo`, optional `instructions`), then `notifications/initialized` | No handshake: every request carries `io.modelcontextprotocol/protocolVersion` and `io.modelcontextprotocol/clientCapabilities` in `_meta`; servers MUST implement `server/discover` |
| State | Stateful connections; Streamable HTTP may use an `Mcp-Session-Id` | Stateless requests; protocol sessions and `Mcp-Session-Id` are gone; cross-call state uses explicit handles passed as tool arguments |
| Server asks the client for something | The server sends its own requests: `sampling/createMessage`, `roots/list`, `elicitation/create` | Multi Round-Trip Requests: the server returns `InputRequiredResult` and the client retries with `inputResponses` |
| Resource not found | `-32002` | `-32602`; clients should accept both |

!!! warning "Exam guide vs current docs: which MCP revision?"

    All four guides are dated effective July 2026, the same month as revision 2026-07-28, and none of them names a specification revision. Claude Code's v2 MCP runtime asks HTTP servers whether they support the newer revision but connects to stdio servers the older way unless `MCP_PROTOCOL_NEGOTIATION=auto` is set. Learn both eras; the guides' own vocabulary (`isError`, resources, `.mcp.json`, environment variable expansion) works in either.

### Tool definitions and results

| Field | Rule |
|---|---|
| `name` | SHOULD be 1 to 128 characters, case-sensitive, using A-Z, a-z, 0-9, `_`, `-` and `.`; unique only per server, so aggregating clients should disambiguate, for example by prefixing a server identifier |
| `title`, `description`, `icons` | Optional display title, the description, optional icons; the display name order is `title`, then `annotations.title`, then `name` |
| `inputSchema` | MUST be a valid JSON Schema object, never `null`; a tool with no parameters should use `{ "type": "object", "additionalProperties": false }` |
| `outputSchema` | Optional; when present, the server MUST return conforming structured results and clients SHOULD validate them |
| `annotations` | Hints only (table below) |

| Annotation | Default | Meaning |
|---|---|---|
| `readOnlyHint` | `false` | The tool does not modify its environment |
| `destructiveHint` | `true` | May perform destructive updates; meaningful only when `readOnlyHint` is false |
| `idempotentHint` | `false` | Repeated calls with the same arguments have no additional effect; meaningful only when `readOnlyHint` is false |
| `openWorldHint` | `true` | Interacts with an open world of external entities (a web search tool) rather than a closed domain (a memory tool) |

Clients MUST treat annotations as untrusted unless they come from a trusted server. A tool result carries unstructured `content` (text, image, audio, resource links, embedded resources) and/or `structuredContent`, which in 2026-07-28 may be any JSON value. A tool that returns structured content SHOULD also return the serialized JSON in a text block. `structuredContent` is server data and has nothing to do with the Claude API's structured outputs feature.

### Errors

| Kind | How it is returned | Examples | Reaches the model? |
|---|---|---|---|
| Protocol error | A JSON-RPC `error` object with `code`, `message` and optional `data` | Unknown tool (`-32602`), malformed request, server errors | Clients MAY pass it on; models are less likely to fix these |
| Tool execution error | A normal result with `isError: true` and explanatory content | API failures, input validation errors, business-logic errors | Clients SHOULD pass it on, so the model can self-correct |

Codes to recognize: `-32700` and `-32600` to `-32603` are the standard JSON-RPC codes; `-32602` (Invalid params) covers an unknown tool, missing prompt arguments and invalid cursors; `-32603` is an internal error; `-32022` is `UnsupportedProtocolVersion` in 2026-07-28. If a tool reports its own failure as a protocol error instead of inside the result, the model cannot see that an error occurred.

The spec's tool execution error is `isError: true` plus explanatory content; the spec's tools page has no field named `errorCategory`, `isRetryable` or `retriable`. CCAR-F adds application-level error metadata: an `errorCategory` (transient, validation, permission), an `isRetryable` flag and a human-readable description (the guide does not say where these fields go; the example below carries them in the `isError` result's text content), with `retriable: false` and a customer-friendly explanation for business-rule violations. The guide's knowledge bullet names business errors as a fourth category, so the example below uses `business` as a category value; that is a reconciliation of the guide's two lists, not guide wording. The example is illustrative: its envelope follows the spec's tool execution error example (`resultType`, `content`, `isError: true`), and the text payload is ours.

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "resultType": "complete",
    "content": [
      {
        "type": "text",
        "text": "{\"errorCategory\": \"business\", \"isRetryable\": false, \"description\": \"This order is outside the return window, so a refund cannot be issued. Explain the return policy to the customer and offer a human agent.\"}"
      }
    ],
    "isError": true
  }
}
```

Servers on 2025-11-25 or earlier do not send `resultType`; a 2026-07-28 client MUST treat an absent `resultType` as `"complete"`.

### MCP in Claude Code

| Scope | Stored in | Shared with the team? | Use it for |
|---|---|---|---|
| Local (the default) | `~/.claude.json`, under the project's path | No | Personal development servers, experiments, servers whose credentials stay out of git |
| Project | `.mcp.json` at the project root | Yes, commit it | Team-shared tooling |
| User | `~/.claude.json`, top-level `mcpServers` key | No | Your servers across every project |

```bash
claude mcp add --transport http notion https://mcp.notion.com/mcp                 # remote HTTP (local scope by default)
claude mcp add --env AIRTABLE_API_KEY=YOUR_KEY --transport stdio airtable \
  -- npx -y airtable-mcp-server                                                   # stdio: server command after --
claude mcp add --transport http shared-server --scope project https://example.com/mcp   # writes .mcp.json
claude mcp add --transport http hubspot --scope user https://mcp.hubspot.com/anthropic  # all projects
claude mcp list
claude mcp reset-project-choices # reset .mcp.json approvals
```

```json
{
  "mcpServers": {
    "api-server": {
      "type": "http",
      "url": "${API_BASE_URL:-https://api.example.com}/mcp",
      "headers": {
        "Authorization": "Bearer ${API_KEY}"
      }
    }
  }
}
```

| Topic | What to know |
|---|---|
| Same server defined in several places | Local beats project, which beats user (the three scopes match duplicates by name); plugin servers and then claude.ai connectors come next and match by endpoint (same URL or command). A server your organization provides through the `managedMcpServers` managed setting ranks above all of these (Claude Code v2.1.259 or later). The winning entry is used whole, with no field merging |
| Environment variables | `${VAR}` and `${VAR:-default}` expand in `command`, `args`, `env`, `url` and `headers`. In a remote server's `url` and `headers`, credential variables such as `ANTHROPIC_API_KEY` read as empty |
| Project server approval | Interactive sessions ask before using `.mcp.json` servers; `claude -p`, Agent SDK and cloud sessions load them without asking |
| `url` without `type` | A configuration error: an entry with no `type` is read as stdio |
| Output size | Warning above 10,000 tokens; default maximum 25,000 tokens; raise it with `MAX_MCP_OUTPUT_TOKENS` |
| Tool search | On by default: only tool names and server instructions load at session start; full schemas load when needed |
| Resources and prompts | Reference a resource as `@server:protocol://resource/path`; run a prompt as `/mcp__servername__promptname` |
| Names in hooks and permissions | `mcp__<server>__<tool>`. A hook matcher needs `mcp__memory__.*` to catch a whole server; a permission rule can use `mcp__puppeteer__*` |
| Authentication | OAuth 2.0 for remote servers through `/mcp`, or `claude mcp login <name>` from the shell (v2.1.186 and later) |

!!! warning "Exam guide vs current docs: scopes and discovery"

    [CCAR-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) contrasts project `.mcp.json` (shared team tooling) with user-level `~/.claude.json` (personal or experimental servers), and says tools "are discovered at connection time and available simultaneously". The docs list three scopes, and local, the default, is the one they recommend for experimental configurations. Local and user scope are both stored in `~/.claude.json`, so the guide's file-level contrast still holds (our reading). With tool search on by default, Claude Code loads only tool names at session start and defers full definitions until they are needed. Answer in the guide's terms: shared team servers go in project `.mcp.json`, personal or experimental servers in user-scoped `~/.claude.json`, and all configured servers' tools are available to the agent at once.

### MCP in the API and the Agent SDK

| | Messages API MCP connector | Agent SDK |
|---|---|---|
| Configure | An `mcp_servers` array (`type: "url"`, an `https://` `url`, a unique `name`, optional `authorization_token`) plus one `mcp_toolset` per server in `tools` | The `mcp_servers` / `mcpServers` option, or `.mcp.json` loaded through the `project` setting source |
| Turn it on | Beta header `mcp-client-2025-11-20`. As of September 2026, `mcp-client-2026-09-15` (available on the Claude API) includes everything it does and adds tool-list pinning; send it in place of the older header | List the tools, or `mcp__github__*`, in `allowedTools`: without permission Claude sees MCP tools but cannot call them, and `acceptEdits` does not cover them |
| Transports | Public HTTP servers only (Streamable HTTP or SSE); no local stdio servers | A command means stdio, a URL means HTTP or SSE, tools in your own code mean an in-process SDK server |
| What is supported | Tool calls only, not resources or prompts; responses add `mcp_tool_use` and `mcp_tool_result` blocks | Tools, plus resources through the built-in `ListMcpResourcesTool` and `ReadMcpResourceTool`; the `init` message reports each server's status: `pending`, `connected`, `failed`, `needs-auth`, `disabled` |
| OAuth | You run the OAuth flow, pass the token and refresh it | No interactive flow: finish OAuth in your app and pass the token in `headers` |
| Limits (as of September 2026) | Beta on the Claude API, Claude Platform on AWS and Microsoft Foundry; not on Amazon Bedrock or Google Cloud; not covered by ZDR arrangements | 30-second connection timeout by default (`MCP_TIMEOUT`); the same 25,000-token output limit as Claude Code |

=== "Messages API (cURL)"

    ```bash
    curl https://api.anthropic.com/v1/messages \
      -H "Content-Type: application/json" \
      -H "X-API-Key: $ANTHROPIC_API_KEY" \
      -H "anthropic-version: 2023-06-01" \
      -H "anthropic-beta: mcp-client-2025-11-20" \
      -d '{
        "model": "claude-opus-5-5",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": "What tools do you have available?"}],
        "mcp_servers": [
          { "type": "url", "url": "https://example-server.modelcontextprotocol.io/sse",
            "name": "example-mcp", "authorization_token": "YOUR_TOKEN" }
        ],
        "tools": [
          { "type": "mcp_toolset", "mcp_server_name": "example-mcp",
            "default_config": { "enabled": false },
            "configs": { "search_events": { "enabled": true } } }
        ]
      }'
    ```

=== "Agent SDK (Python)"

    ```python
    from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

    options = ClaudeAgentOptions(
        mcp_servers={
            "claude-code-docs": {"type": "http", "url": "https://code.claude.com/docs/mcp"}
        },
        allowed_tools=["mcp__claude-code-docs__*"],
    )
    ```

The cURL tab joins the connector docs' basic request to their allowlist toolset; `search_events` is the tool name from the docs' calendar allowlist example, shown here only to illustrate the shape. An allowlist sets `default_config.enabled: false`, then enables named tools in `configs`, and tool-specific `configs` override `default_config`. The opposite, a denylist of write or destructive tools, is what the docs recommend for read-only assistants or when you want a human confirmation step before state changes. The Python tab is the Agent SDK MCP docs' example (the `query()` loop omitted). In-process SDK tools are covered under [SDK custom tools](#sdk-custom-tools).

### Building a server

The Python tab is the v2 Python SDK README example plus the `mcp.run(transport="stdio")` entry point from the 2026-07-28 build-server tutorial; the TypeScript tab is the v2 TypeScript SDK README example. In Python the type hints are the schema, so no hand-written JSON Schema is needed.

=== "Python (SDK v2)"

    ```python
    from mcp.server import MCPServer

    mcp = MCPServer("Demo")

    @mcp.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @mcp.resource("greeting://{name}")
    def greeting(name: str) -> str:
        """Greet someone by name."""
        return f"Hello, {name}!"

    if __name__ == "__main__":
        mcp.run(transport="stdio")
    ```

=== "TypeScript (SDK v2)"

    ```typescript
    import { McpServer } from '@modelcontextprotocol/server';
    import { StdioServerTransport } from '@modelcontextprotocol/server/stdio';
    import * as z from 'zod/v4';

    const server = new McpServer({ name: 'greeting-server', version: '1.0.0' });

    server.registerTool(
        'greet',
        {
            description: 'Greet someone by name',
            inputSchema: z.object({ name: z.string() })
        },
        async ({ name }) => ({
            content: [{ type: 'text', text: `Hello, ${name}!` }]
        })
    );

    async function main() {
        const transport = new StdioServerTransport();
        await server.connect(transport);
    }
    main();
    ```

- Older tutorials use the v1 names: `from mcp.server.fastmcp import FastMCP` in Python and the `@modelcontextprotocol/sdk` package in TypeScript. `pip install mcp` now installs the 2.x line.
- A stdio server must never write to stdout: `print()` in Python or `console.log()` in TypeScript corrupts the JSON-RPC stream. Log to stderr.
- Test with the MCP Inspector: `npx @modelcontextprotocol/inspector` for the web UI, `--cli` for scripts and CI.
- CCAR-F lists deploying or hosting MCP servers as out of scope; CCDV-F's MCP Server Development skill includes server authoring and deployment.

### Security quick facts

- Tools represent arbitrary code execution. Hosts must obtain explicit user consent before invoking any tool, and there SHOULD always be a human in the loop who can deny an invocation.
- Authorization is optional. HTTP transports SHOULD follow the spec's OAuth 2.1-based authorization; stdio servers SHOULD NOT and take credentials from the environment instead.
- A server that calls upstream APIs must not pass through the token it received from the MCP client, and must not accept tokens that were not issued for it.
- Organizations control servers with `managed-mcp.json` (exclusive control) or `allowedMcpServers` and `deniedMcpServers`. A denylist match always wins, and a `serverName` entry is not a security control: use `serverCommand` or `serverUrl`.
- Servers that fetch external content can expose you to prompt injection, so verify you trust each server before connecting it.
- CCAR-F excludes OAuth and authentication protocol details; CCDV-F covers identity, secrets and key management.

Choosing between MCP, a built-in tool, a custom tool or a Skill: see [Tool rules](#tool-rules) and [Built-in tools, custom tools, Skills or MCP](knowledge/tool-use-and-mcp.md#built-in-tools-custom-tools-skills-or-mcp).

## Decision rules

All four exams use multiple-choice and scenario-based multiple-response items, and the scenarios turn on a small set of recurring judgments. The first table collects the reasoning patterns the official rationales use; after it, each rule is one table row: the rule, the wrong answers it rules out, and in the Basis column a pointer to the objective, sample question or anti-pattern in the official guides ([CCAO-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf), [CCDV-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf), [CCAR-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf), [CCAR-P](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). Q numbers are CCAR-F sample questions; Sample numbers belong to the other three guides. All of them are reproduced with Anthropic's rationales on the exam pages ([CCAO-F](claude-certified-associate.md#official-sample-questions), [CCDV-F](claude-certified-developer.md#official-sample-questions), [CCAR-F](claude-certified-architect-foundations.md#official-sample-questions), [CCAR-P](claude-certified-architect-professional.md#official-sample-questions)). "Prep course" in the Basis column means a learning objective of the free Claude Certified Architect - Professional Prep Course; which CCAR-P objective it sits beside is our mapping, not Anthropic's. The CCAO-F and CCAR-P guides list their objectives as unnumbered bullets; labels such as CCAO-F D3.4 and CCAR-P 3.7 are this site's numbering, counted in each guide's order. CCAR-F task statement numbers and CCDV-F skill names are the guides' own. Where current docs differ from a guide, a warning box here or in the cheat sheet a rule links to gives both sides; answer exam items in the guide's terms.

### How the official rationales pick an answer

| Pattern | Official wording | Source |
|---|---|---|
| Fix the problem the stem describes, not a neighboring one | "Option D addresses tool availability rather than tool ordering, which is not the actual problem." | [CCAR-F Q1 rationale](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) |
| Take the proportionate first step before adding infrastructure | "This is the proportionate first response before adding infrastructure." | [CCAR-F Q3 rationale](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) |
| Distrust extra machinery offered as a first fix | "A routing layer (C) is over-engineered and bypasses the LLM's natural language understanding." | [CCAR-F Q2 rationale](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) |
| When a rule must always hold, code beats prompts | "programmatic enforcement provides deterministic guarantees that prompt-based approaches cannot" | [CCAR-F Q1 rationale](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) |
| Remove capability rather than watch it | "Least privilege means removing capabilities the role does not require, eliminating the attack surface rather than monitoring or guarding it." | [CCAR-P Sample 1 rationale](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf) |
| A model's own confidence is not evidence | "Self-reported confidence (A, C) is not a reliable accuracy signal, and reformatting (D) does not address correctness." | [CCAO-F Sample 1 rationale](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) |
| A request to the model is not a control | "a polite request (C) is not an enforceable control" | [CCDV-F Sample 2 rationale](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) |
| Match each workload to the mechanism built for it | "the simpler solution is matching each API to its appropriate use case" | [CCAR-F Q11 rationale](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) |

### Enforcement rules

| Rule | Rules out | Basis |
|---|---|---|
| If a step must happen every time (verify the customer before any refund), block the downstream tool call in code until the prerequisite has succeeded | A firmer system prompt; few-shot examples of the right order; a classifier that changes which tools exist | CCAR-F 1.4, Q1 |
| If a business limit must never be crossed (the guide's example: refunds above &#36;500), intercept the outgoing call, deny it, and redirect to the escalation workflow | Asking the agent to respect the limit | CCAR-F 1.5, Exercise 1 |
| If tools return mixed formats (Unix timestamps, ISO 8601, numeric status codes), normalize the results in a PostToolUse hook before the model reads them | Leaving the model to reconcile formats | CCAR-F 1.5 |
| If behavior should usually happen, a prompt or CLAUDE.md instruction is proportionate; if it must never fail, use a hook or a permission rule | Treating CLAUDE.md as enforcement | CCAR-F 1.4, 1.5; Claude Code docs |
| If a guardrail must stop destructive actions, the hook has to deny: from a command hook, exit code 2 or JSON `permissionDecision: "deny"` on stdout; from an SDK callback, `permissionDecision: "deny"` | Exit code 1 with no JSON decision, which does not block | CCDV-F Claude Hooks |
| If a check must run on every Agent SDK tool call, put it in a PreToolUse hook (a hook deny applies even in `bypassPermissions` mode) | `canUseTool` (`can_use_tool` in Python), which auto-approved calls never reach | CCDV-F Agent Construction with Claude |

Go deeper: the refund-limit hook in the [Agent SDK cheat sheet](#agent-sdk-cheat-sheet), policy blocks and result normalization in [Hooks in the SDK](knowledge/agents-and-agent-sdk.md#hooks-in-the-sdk), prerequisite gates in [Permissions and enforcement](knowledge/agents-and-agent-sdk.md#permissions-and-enforcement), and Claude Code hooks in [Hooks](knowledge/claude-code-workflows.md#hooks).

### Tool rules

| Rule | Rules out | Basis |
|---|---|---|
| If Claude picks the wrong one of two similar tools, first expand each description: input formats, example queries, edge cases and when to use it instead of the other | Few-shot routing examples, a keyword routing layer, or merging the tools as the first step | CCAR-F 2.1, Q2 |
| If two tools overlap, rename and redescribe them so each has one distinct job; split a generic tool into purpose-specific tools with defined inputs and outputs | Near-identical names and descriptions | CCAR-F 2.1 |
| If good descriptions are still ignored, look for keyword-sensitive wording in the system prompt that overrides them | Rewriting the descriptions again | CCAR-F 2.1 |
| If an agent holds many tools (18 rather than 4 to 5) and picks badly, give each agent only the tools its role needs | One agent with every tool | CCAR-F 2.3; CCAR-P 3.1 |
| If one agent often needs a narrow piece of another's capability (simple fact checks), give it a scoped cross-role tool and send complex cases through the coordinator | Handing it all the web search tools, batching the requests, speculative caching | CCAR-F 2.3, Q9 |
| If a generic tool invites misuse (`fetch_url`), replace it with a constrained one that validates its input (`load_document`) | Prompt warnings about misuse | CCAR-F 2.3 |
| If the model must call a tool rather than answer in prose, set `tool_choice` to `{"type": "any"}` (the guide writes `tool_choice: "any"`); if a specific tool must run first, force it with `{"type": "tool", "name": "extract_metadata"}` and handle later steps in follow-up turns | `{"type": "auto"}`, where the model may return text instead of calling a tool | CCAR-F 2.3, 4.3 |
| If an MCP tool fails, return `isError: true` with an error category, a retryable flag and a plain description; mark business-rule violations non-retryable with a customer-friendly explanation | A uniform, generic failure message; a JSON-RPC protocol error, which clients are not required to pass to the model (the MCP spec says clients SHOULD pass tool execution errors and MAY pass protocol errors) | CCAR-F 2.2; MCP spec |
| If a query fails, say so; if it succeeded and found nothing, return a valid empty result | Reporting both the same way | CCAR-F 2.2, 5.3 |
| If a standard integration exists (Jira), use the community MCP server; build custom servers only for team-specific workflows | Writing your own server first | CCAR-F 2.4 |
| If the agent keeps using Grep instead of a more capable MCP tool, describe that tool's capabilities and outputs in detail | Leaving the MCP tool's description thin | CCAR-F 2.4 |
| If the agent spends calls discovering what data exists, expose the catalog as MCP resources and keep tools for actions | More exploratory tool calls | CCAR-F 2.4, In-Scope list |
| If several Claude applications need the same capability maintained independently, build an MCP server that exposes it as tools | Logic hard-coded in system prompts; pasting data into context; assuming built-in tools reach internal APIs | CCDV-F Sample 3 |
| If the need is a connection to a service, use MCP; if Claude must also know how to use it well (schema, query patterns), add a Skill | Using a Skill alone to connect to an external service; using MCP alone when Claude also needs your team's schema and query patterns | CCDV-F Agentic Customization |
| If one agent calls one service, a direct API call is enough; for quick, permissive local integrations, a CLI; for production cloud agents that must run across web, mobile and cloud platforms, an MCP server; for independent agents collaborating across team or partner boundaries, agent-to-agent (A2A) between the agents with MCP inside each agent | One mechanism for every case | CCAR-P 3.7 (criteria from Anthropic's MCP production blog and the A2A project docs; our mapping) |
| Search file contents with Grep, find files by name pattern with Glob, change a unique span with Edit, and fall back to Read plus Write when Edit's anchor text is not unique | Reading every file up front | CCAR-F 2.5 |
| In an unfamiliar codebase, Grep for entry points, then Read to follow imports and trace flows | Loading all files before starting | CCAR-F 2.5 |

!!! warning "Exam guide vs current docs: tool rules"

    - **Forced tool choice.** As of September 2026, Claude Opus 5.5, Fable 5.1 and Mythos 5.1 reject `"any"` and forced selection; both versions and the documented substitutes are in the warning under [tool_choice](#tool_choice). Answer exam items with the guide's logic.
    - **Split or consolidate.** The guide splits a generic tool into purpose-specific ones; the define-tools page groups related operations into one tool with an `action` parameter. Both rest on one rule: each tool needs a clear, distinct purpose.
    - **Error fields.** The guide's `errorCategory` and `isRetryable` are application-level conventions carried in an `isError: true` result, not MCP spec fields: see [Errors](#errors) in the MCP cheat sheet.
    - **Grep, Glob and Edit.** Current Claude Code leaves Glob and Grep out of the default tool set on macOS, Linux and WSL and searches with `find` and `grep` through Bash, and it resolves a non-unique Edit match with a longer `old_string` or `replace_all: true`. The guide's selection logic and its Read plus Write fallback are still the exam answers.

Go deeper: [Writing tool descriptions that steer selection](knowledge/tool-use-and-mcp.md#writing-tool-descriptions-that-steer-selection), [Designing a tool set](knowledge/tool-use-and-mcp.md#designing-a-tool-set) and [Built-in tools, custom tools, Skills or MCP](knowledge/tool-use-and-mcp.md#built-in-tools-custom-tools-skills-or-mcp).

### Context rules

| Rule | Rules out | Basis |
|---|---|---|
| If a long conversation carries amounts, dates, order numbers and statuses, keep them in a persistent case-facts block included in every prompt, outside the summarized history | Progressive summarization that blurs the numbers | CCAR-F 5.1 |
| If tool results carry many irrelevant fields (40+ per order lookup when 5 matter), trim them to the relevant fields before they accumulate | Passing whole payloads through | CCAR-F 5.1 |
| If findings from the middle of a long aggregated input go missing, put a key-findings summary first and organize the details under explicit section headers | Hoping a bigger window fixes it (compare Q12's rationale on multi-file review: larger context windows do not solve attention quality) | CCAR-F 5.1 |
| If a downstream agent has a small context budget, have upstream agents return structured facts, citations and relevance scores | Verbose content and reasoning chains | CCAR-F 5.1 |
| If a subagent needs earlier findings, pass them in its prompt, in a structured format that keeps content apart from metadata such as URLs, document names and page numbers | Assuming it inherits the coordinator's context | CCAR-F 1.2, 1.3 |
| If exploration output is verbose, delegate it to subagents (the Explore subagent in Claude Code) that return summaries while the main agent coordinates | Exploring in the main conversation until the window fills | CCAR-F 3.4, 5.4 |
| If a long session starts citing typical patterns instead of the specific classes it found, keep a scratchpad file of key findings and reread it; run `/compact` when discovery output fills the window | Carrying on in a degraded context | CCAR-F 5.4 |
| Before the next exploration phase, summarize what the last phase found and inject that summary into the new subagents' initial context | Starting each phase from nothing | CCAR-F 5.4 |
| Resume a session when its context is mostly valid; start a new one with an injected structured summary when its tool results are stale; when resuming after code changes, name the files that changed | Resuming on stale results; forcing a full re-exploration | CCAR-F 1.7 |
| If you want to compare two approaches from one analysis, fork the session from that shared baseline | Rerunning the analysis twice | CCAR-F 1.3, 1.7 |
| If a multi-agent run must survive a crash, have each agent export its state to a known location and let the coordinator load a manifest on resume | Restarting the whole run | CCAR-F 5.4 |
| Restart, summarize or persist: if a chat has gone off track or hits its length limit, open a new conversation with a clearer prompt (or use a project); if long material will not fit, summarize or extract its key sections before sending; persist stable facts in memory or project knowledge, and look volatile details up fresh | Steering the same long chat further | CCAO-F D3.4 |
| If a system has many tools or a large knowledge base, load definitions and detail on demand (tool search, Skills' progressive disclosure) and keep what is always needed small | Loading everything into context up front | CCAR-P 3.8 |

Go deeper: [Preserving critical information in long conversations](knowledge/context-engineering.md#preserving-critical-information-in-long-conversations), [Subagents as context isolation](knowledge/context-engineering.md#subagents-as-context-isolation) and [Sessions, resumption and forking](knowledge/agents-and-agent-sdk.md#sessions-resumption-and-forking).

### Reliability rules

#### Agent loops and orchestration

| Rule | Rules out | Basis |
|---|---|---|
| Keep looping while `stop_reason` is `"tool_use"`, append each tool result to the history, and stop on `"end_turn"` | Parsing the reply text for a completion signal; an iteration cap as the main stop; stopping when assistant text appears | CCAR-F 1.1 |
| If every subagent succeeded but the report misses whole areas, fix the coordinator's decomposition | Blaming the search, analysis or synthesis agents | CCAR-F 1.2, Q7 |
| Have the coordinator pick only the subagents a query needs, give each a distinct slice of scope, route all their communication through itself, and re-delegate until coverage is sufficient | Always running the full pipeline; overlapping assignments | CCAR-F 1.2 |
| For independent subtasks, emit several Task tool calls in one coordinator response (the guide's name; since Claude Code v2.1.63 the tool is called Agent, and `Task` still works as an alias) | Spawning them across separate turns | CCAR-F 1.3 |
| Write coordinator prompts that set research goals and quality criteria, so subagents can adapt | Step-by-step procedural instructions | CCAR-F 1.3 |
| Use prompt chaining for predictable multi-aspect work; use dynamic decomposition for open-ended investigation, mapping structure first and adapting the plan | One fixed plan for open-ended tasks | CCAR-F 1.6 |
| If a message raises several issues, split them, investigate in parallel with shared context, and answer with one resolution | Handling only the first issue | CCAR-F 1.4 |

The guide frames loop control as `"tool_use"` versus `"end_turn"`; the other stop reasons real code must handle, `pause_turn` included, are in [Stop reasons](#stop-reasons). Answer with the guide's pair.

#### Errors and escalation

| Rule | Rules out | Basis |
|---|---|---|
| If a subagent fails, send the coordinator the failure type, the attempted query, any partial results and possible alternatives | A generic unavailable status; an empty result marked successful; ending the whole workflow | CCAR-F 5.3, Q8 |
| Retry transient failures inside the subagent; propagate only what it cannot resolve, with what was tried and the partial results | Escalating every timeout; a generic "search unavailable" status after the retries run out (Q8's option B fails on the status, not on the retry) | CCAR-F 2.2, 5.3 |
| If escalation is miscalibrated, add explicit escalation criteria with few-shot examples to the system prompt first | Self-rated confidence thresholds; a separate trained classifier; sentiment triggers | CCAR-F 5.2, Q3 |
| If the customer asks for a human, escalate at once without investigating first | Trying to resolve it anyway | CCAR-F 5.2 |
| If the customer is frustrated but the issue is straightforward, acknowledge it and offer to resolve; escalate if they repeat the request | Escalating on frustration alone | CCAR-F 5.2 |
| If policy is silent or ambiguous on the request (a competitor price match), escalate | Improvising a policy exception | CCAR-F 5.2 |
| If a lookup returns several customers, ask for another identifier | Choosing one by heuristic | CCAR-F 5.2 |
| When escalating to a human who cannot see the transcript, send a structured summary: customer ID, root cause, refund amount and recommended action | Forwarding the raw conversation | CCAR-F 1.4 |
| If a request returns HTTP 200 but the output is wrong, check `stop_reason` first (a refusal arrives as HTTP 200 with `stop_reason: "refusal"`; API errors arrive as 4xx or 5xx), then check how your code reads the response and what context it gave the model before blaming the model | Monitoring only HTTP errors | CCDV-F Debugging and Error Handling |
| If API calls fail with a rate limit (429) or a server error (500, or 529 `overloaded_error`), retry with exponential backoff and honor `retry-after`; the official SDKs already retry these twice by default. A 429 whose `error.details.error_code` is `enforced_spend_limit_reached` is the monthly spend cap: it has no `retry-after`, every retry fails until access resumes at 00:00 UTC on the first day of the next month, and a higher tier restores access sooner | Retrying before `retry-after` has elapsed; treating a spend-cap 429 as a short rate-limit wait | CCDV-F Debugging and Error Handling |

#### Review, validation and oversight

| Rule | Rules out | Basis |
|---|---|---|
| Review generated code with a second, independent instance that lacks the generator's reasoning | Asking the same session to check itself; extended thinking as the reviewer | CCAR-F 3.6, 4.6 |
| If a large multi-file review is inconsistent, run per-file passes for local issues and a separate cross-file integration pass | A larger context window; requiring 2 of 3 runs to agree; making developers split PRs | CCAR-F 1.6, 4.6, Q12 |
| Before reducing human review, check accuracy by document type and field and keep stratified random sampling of high-confidence outputs | Trusting one aggregate figure such as 97% | CCAR-F 5.5 |
| Route reviews with field-level confidence calibrated on a labeled validation set; send low-confidence or contradictory-source items to people | Raw self-reported confidence | CCAR-F 5.5 |
| If a confident summary cites a specific subsection for a compliance audience, verify it against the official text before sharing | Sending it because Claude sounded sure; asking Claude to rate itself; polishing the wording | CCAO-F Sample 1 |
| If RAG answers turn confidently wrong right after a document refresh while model and latency are unchanged, investigate retrieval and indexing first | Silent weight changes, temperature, a shrunken window | CCAR-P Sample 3 |
| Gate every model or architecture change on the evaluation suite | Switching on intuition | CCAR-P 4.1, 4.2; prep course |
| Match human oversight to risk, and route each decision by confidence, reversibility and cost | One review level for everything | CCAR-P 5.3; prep course |

#### Synthesis and provenance

| Rule | Rules out | Basis |
|---|---|---|
| Have subagents output claim-source mappings (URL, document name, excerpt) that synthesis must preserve and merge | Summaries that drop the sources | CCAR-F 5.6 |
| If credible sources disagree, keep both values with attribution and separate well-established from contested findings | Picking one value | CCAR-F 5.6 |
| Require publication or collection dates so differences over time are not read as contradictions | Undated figures | CCAR-F 5.6 |
| Annotate coverage: say which findings are well supported and which areas have gaps from unavailable sources | Presenting partial research as complete | CCAR-F 5.3 |

!!! note "Self-reported confidence inside the guide"

    Task 5.2 and Q3 call self-reported confidence an unreliable proxy for escalation. Task 4.6 has the model self-report confidence to enable calibrated review routing, and Task 5.5 calibrates field-level confidence scores on labeled validation sets. Both hold: raw self-rating is not an escalation signal; field-level confidence calibrated on labeled data can route review.

Go deeper: [The agentic loop](knowledge/agents-and-agent-sdk.md#the-agentic-loop), [Error propagation in multi-agent systems](knowledge/evaluation-and-reliability.md#error-propagation-in-multi-agent-systems), [Escalation and ambiguity](knowledge/evaluation-and-reliability.md#escalation-and-ambiguity), [Human review and confidence calibration](knowledge/evaluation-and-reliability.md#human-review-and-confidence-calibration) and [Provenance and uncertainty in synthesis](knowledge/context-engineering.md#provenance-and-uncertainty-in-synthesis).

### Prompt rules

| Rule | Rules out | Basis |
|---|---|---|
| To cut false positives, state which issues to report and which to skip, category by category | General instructions to be conservative or to report only high-confidence findings | CCAR-F 4.1 |
| If one noisy category erodes trust in the rest, switch it off while you fix its prompt | Keeping every category live | CCAR-F 4.1 |
| If severity labels vary, define each level with concrete code examples | Vague severity labels | CCAR-F 4.1 |
| If instructions alone give inconsistent output, show the format in few-shot examples (location, issue, severity, suggested fix) | Longer instructions | CCAR-F 4.2 |
| For ambiguous cases, give 2 to 4 targeted examples that explain why one action beats the plausible alternatives | Examples that only show the easy path | CCAR-F 4.2 |
| If extraction misses fields that are present or struggles with varied layouts, add examples covering those structures | Retrying the same prompt | CCAR-F 4.2 |
| Start without examples (zero-shot) for simple tasks; add examples when a format is easier to show than describe or results stay inconsistent | Examples by default | CCAR-P 2.3 |
| For guaranteed schema-valid structure, define the extraction as a tool with a JSON schema and read the `tool_use` input | Asking for JSON in prose | CCAR-F 4.3 |
| Make a field optional (nullable) when the source may lack it; add `"unclear"` for ambiguous values and `"other"` plus a detail string for extensible categories | Required fields the model fills by inventing values | CCAR-F 4.3 |
| Strict schemas stop syntax errors, not semantic ones: extract `calculated_total` beside `stated_total` and flag `conflict_detected` | Assuming schema-valid means correct | CCAR-F 4.3, 4.4 |
| On validation failure, retry with the original document, the failed extraction and the specific errors; if the information is simply absent from the source, stop retrying | Blind retries | CCAR-F 4.4 |
| To learn why developers dismiss findings, add a `detected_pattern` field and analyze dismissals by pattern | Guessing from anecdotes | CCAR-F 4.4 |
| If source formatting is inconsistent, put normalization rules in the prompt beside the strict schema | Relying on the schema alone | CCAR-F 4.3 |
| If prose descriptions of a transformation are read inconsistently, give 2 or 3 concrete input and output examples | More adjectives | CCAR-F 3.5 |
| In an unfamiliar domain, have Claude interview you before it implements | Implementing on assumptions | CCAR-F 3.5 |
| Write the tests first and iterate by sharing failures; send interacting problems in one message and independent ones one at a time | Fixing interacting bugs piecemeal | CCAR-F 3.5 |
| Break a complex request into smaller subtasks or steps so each gets Claude's full attention | One sprawling prompt | CCAO-F D1.2 |

Two notes on current docs. The prompting best-practices page recommends 3 to 5 examples for best results, while the guide's figure is 2 to 4 targeted examples for ambiguous cases; answer with the guide's figure. The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) appendix phrase "strict mode for syntax error elimination" (listed under JSON Schema) corresponds, in our reading, to what the API now calls `strict: true`; how structured outputs and strict tool use relate to the guide's tool-use framing is in the warning under [tool_choice](#tool_choice). Answer exam items with the guide's framing.

Go deeper: [Explicit criteria and precision](knowledge/prompt-engineering.md#explicit-criteria-and-precision), [Few-shot examples](knowledge/prompt-engineering.md#few-shot-examples), [Structured output with tools and JSON schemas](knowledge/prompt-engineering.md#structured-output-with-tools-and-json-schemas) and [Validation, retry and feedback loops](knowledge/prompt-engineering.md#validation-retry-and-feedback-loops).

### Claude Code layer rules

| Rule | Rules out | Basis |
|---|---|---|
| If every teammate must get an instruction, put it in the project CLAUDE.md (`./CLAUDE.md` or `./.claude/CLAUDE.md`), committed to the repository | `~/.claude/CLAUDE.md`, which only you receive | CCAR-F 3.1 |
| If CLAUDE.md has grown large, split it into topic files in `.claude/rules/` and import the standards each package needs with `@path/to/import` syntax (the guide calls it "@import syntax") | One monolithic file | CCAR-F 3.1 |
| If conventions apply to files spread across directories (test files), use `.claude/rules/` files with `paths` globs such as `**/*.test.tsx` | Sections in the root CLAUDE.md; Skills; a CLAUDE.md in each subdirectory | CCAR-F 3.3, Q6 |
| Put always-on universal standards in CLAUDE.md and on-demand task workflows in Skills | Loading every workflow into every session | CCAR-F 3.2 |
| For a team slash command shared through git, create it in `.claude/commands/`; personal commands go in `~/.claude/commands/` | CLAUDE.md; a `.claude/config.json` commands array, which does not exist | CCAR-F 3.2, Q4 |
| If a skill's output is verbose or exploratory, set `context: fork` so it runs in an isolated subagent | Running it in the main conversation | CCAR-F 3.2 |
| For a personal variant of a team skill, create it in `~/.claude/skills/` under a different name | Editing the shared skill | CCAR-F 3.2 |
| If behavior differs between sessions, check which memory files actually loaded (the guide names `/memory`) | Assuming every file loaded | CCAR-F 3.1 |
| Use plan mode for architectural, multi-file work with several valid approaches (monolith to microservices, a library migration affecting 45+ files); use direct execution for a clear single-file fix; plan first, then execute the plan | Discovering the design while editing; switching to plan mode only if complexity appears; detailed upfront instructions without exploring the code | CCAR-F 3.4, Q5 |
| If a CI job hangs waiting for input, run `claude -p` | `CLAUDE_HEADLESS=true`, a `--batch` flag (neither exists), redirecting stdin | CCAR-F 3.6, Q10 |
| For machine-readable CI findings, combine `--output-format json` with `--json-schema` | Parsing free text | CCAR-F 3.6 |
| When re-reviewing after new commits, include the earlier findings and report only new or still-unaddressed issues | Reposting every comment | CCAR-F 3.6 |
| Give CI-invoked Claude Code its project context (testing standards, fixture conventions, review criteria) through CLAUDE.md; for useful test generation, also document valuable test criteria there and put existing test files in context | Tests that duplicate covered scenarios; low-value test output | CCAR-F 3.6 |
| Share team MCP servers in project `.mcp.json` with `${VAR}` expansion for tokens; keep personal or experimental servers in `~/.claude.json` | Committing secrets | CCAR-F 2.4 |
| If an organization rule must not be overridable by developers, set it in managed settings (for example `permissions.deny`); use managed CLAUDE.md for guidance | Project settings that developers can change | CCAR-P 7.1 |

Where the September 2026 docs describe these layers differently from the guide (commands merged into skills, `allowed-tools`, `argument-hint`, `/memory`, when path-scoped rules load), both versions are in [Exam guide vs current docs](#exam-guide-vs-current-docs); MCP scopes are in the warning under [MCP in Claude Code](#mcp-in-claude-code). Q4's answer still works, and exam items take the guide's terms.

Go deeper: [The configuration layers at a glance](knowledge/claude-code-configuration.md#the-configuration-layers-at-a-glance), [Path-scoped rules](knowledge/claude-code-configuration.md#path-scoped-rules), [Plan mode or direct execution](knowledge/claude-code-workflows.md#plan-mode-or-direct-execution) and [Claude Code in CI/CD](knowledge/claude-code-workflows.md#claude-code-in-cicd).

### Governance rules

| Rule | Rules out | Basis |
|---|---|---|
| If an agent holds capabilities its role never uses (refunds, account deletion for staff who only read and draft), remove them from its configuration | Logging them; confirmation prompts; a larger model | CCAR-P Sample 1 |
| Treat retrieved content as untrusted input, keep it apart from trusted instructions, and use guardrails or hooks so injected text cannot trigger sensitive actions | Raising temperature; a system-prompt line asking users to behave; a larger model | CCDV-F Sample 2 |
| If policy restricts regulated personal data, remove or anonymize the identifiers before uploading | Uploading because the analysis is internal; telling Claude not to retain the file; abandoning the task | CCAO-F Sample 3 |
| If the work needs enterprise-scale architecture or integration design, an Associate escalates it to Claude Architects and Developers | Building it anyway | CCAO-F guide scope |
| Place input screening, output screening and tool-call authorization so the system fails closed | A single guardrail layer | CCAR-P 5.1; prep course |
| Map each compliance obligation to a named control, owner and evidence artifact | Compliance by testing alone | CCAR-P 5.4; prep course |
| Keep secrets out of committed files with `${VAR}` expansion in `.mcp.json`; where the agent itself must never hold a credential, put a proxy outside its environment that injects the credential, so the agent can make the call but never sees the key | API keys in source control, client-side code or prompts | CCDV-F Identity, Secrets, and Key Management |
| If a workload needs Zero Data Retention or HIPAA coverage, check eligibility per feature: as of September 2026, Managed Agents is not eligible for ZDR or HIPAA BAA coverage, and Message Batches is not ZDR- or HIPAA-eligible (29-day retention). Under HIPAA readiness the API blocks a request that uses an ineligible feature with a 400 (unless the feature's details say otherwise); under ZDR it does not block it, and using it steps outside the arrangement for that data | Assuming every API feature inherits the arrangement | CCAR-P 5.4 |

Go deeper: [Prompt injection](knowledge/security-and-governance.md#prompt-injection), [Least privilege for tools and agents](knowledge/security-and-governance.md#least-privilege-for-tools-and-agents) and [Data retention, training and compliance](knowledge/security-and-governance.md#data-retention-training-and-compliance).

### Cost and latency rules

| Rule | Rules out | Basis |
|---|---|---|
| For non-urgent, high-volume work, use the Message Batches API: 50% lower cost, results within a 24-hour window, no guaranteed latency | Parallel synchronous calls (same per-token cost); lower `max_tokens`; the smallest model regardless of quality | CCDV-F Sample 1; CCAR-F 4.5 |
| Keep blocking workflows (a pre-merge check developers wait on) on the synchronous API and move overnight or weekly jobs to batches | Batching both with status polling or a real-time timeout fallback | CCAR-F 4.5, Q11 |
| Correlate batch results by `custom_id`, since results can arrive in any order; resubmit only the failed requests, changed as needed (chunk the documents that were too long) | Resubmitting the whole batch; keeping real-time calls to avoid result ordering issues, which Q11's rationale calls a misconception | CCAR-F 4.5, Q11 |
| Size the submission schedule to the deadline; the guide's example uses 4-hour submission windows to guarantee a 30-hour SLA with 24-hour batch processing (a document can wait up to 4 hours for the next submission, then up to 24 hours in processing: 28 hours, inside 30; our arithmetic) | Ignoring the processing window | CCAR-F 4.5 |
| Refine the prompt on a sample before batching large volumes | Discovering prompt problems at full volume | CCAR-F 4.5 |
| If the same large system prompt and policy precede every short request, put the static content first and enable prompt caching | Truncating the policy; the smallest model regardless of fit; moving the policy into few-shot examples | CCAR-P Sample 2 |
| For high volumes of short, straightforward drafts where speed and cost matter, use a faster, lower-cost model and keep the most capable model for complex reasoning | The top model for everything; disabling features; switching platforms | CCAO-F Sample 2 |
| Choose models by testing on your own prompts and data; tuning effort is often a better lever than switching models | Choosing from general impressions | CCDV-F Model Selection and Tradeoffs; CCAR-P 2.1 |

Batches and tool calling: the guide's rule (no multi-turn tool calling inside one batch request) and what the docs now say about server tools in batches are both in the warning under [Message Batches numbers](#message-batches-numbers).

Go deeper: [Message Batches](knowledge/claude-api.md#message-batches), [Prompt caching](knowledge/claude-api.md#prompt-caching) and [Models and how to choose one](knowledge/claude-api.md#models-and-how-to-choose-one).

## Glossary

Each term below appears in at least one of the four exam guides, unless its row says otherwise, and is defined briefly, with the objectives and official sample questions that use it. Where a guide's wording has drifted from current documentation, the row gives both; answer exam items in the guide's terms. CCAR-F numbers are the guide's own task statements; CCAO-F and CCAR-P numbers count the official bullets in order, as the exam pages do; CCDV-F is cited by skill name. Product details are as of September 2026.

**A**

| Term | Meaning | In the guides |
|---|---|---|
| A/B testing | Comparing variants on real user traffic to measure outcomes such as retention and task completion. It is slow (days or weeks to reach significance) and needs enough traffic; Anthropic's evals post places automated evals before launch and in CI/CD, production monitoring after launch, and A/B tests to validate significant changes once traffic is sufficient | CCAR-P 4.3 |
| Access failure versus valid empty result | An access failure (a timeout or an unavailable service) needs a retry decision; a valid empty result is a successful query that found no matches. Reporting them differently lets the coordinator decide what to do, and returning an empty result as success after a failure is an anti-pattern the guide names | CCAR-F 2.2, 5.3 |
| Adaptive thinking | A thinking mode in which Claude decides whether and how much to think on each request, with effort as the control. It is always on for Claude Opus 5.5 and Claude Fable 5.1; Claude Haiku 4.5 (like Sonnet 4.5 and Opus 4.5) supports only extended thinking | CCDV-F LLM Fundamentals, Model Selection and Tradeoffs |
| Agent | A system in which the LLM directs its own process and tool use, rather than following predefined code paths (compare Workflow) | CCDV-F Agent Architecture; CCAR-P 1.3 |
| Agent memory | CCDV-F lists Agent Memory as a Claude Code core component without defining it. The closest current feature (our mapping) is the subagent `memory` field, which gives a subagent a persistent directory that survives across conversations | CCDV-F Claude Code Operation |
| Agent tool (Task tool) | The tool that spawns subagents. Claude Code v2.1.63 renamed Task to Agent, and `Task(...)` still works as an alias; the guide says Task and requires it in `allowedTools` | CCAR-F 1.3 |
| AgentDefinition | The Agent SDK type for a programmatic subagent: `description` says when to use it and `prompt` is its system prompt (both required); `tools` restricts its tools, and leaving it out inherits every tool available to subagents | CCAR-F 1.3 |
| Agent-to-agent (A2A) | An open standard for communication between independent, potentially opaque agent systems that exchange information without access to each other's internal state, memory or tools. It is an open-source Linux Foundation project contributed by Google | CCAR-P 3.7 |
| Agentic frameworks | Libraries for building agents. CCDV-F names Strands (an open-source AWS SDK with a model-driven approach), LangGraph (a low-level orchestration framework that models agents as graphs of state, nodes and edges) and PydanticAI (agents that bundle instructions, tools, a structured output type and dependencies) | CCDV-F Agent Patterns and Frameworks |
| Agentic harness | The tools, context management and execution environment around the model; Claude Code is the harness and Claude the model. CCDV-F's harness dispatch is not a defined docs term; read it as the loop routing each tool call to the handler that matches its name (our reading), which is what the docs tell you to do with each `tool_use` block | CCDV-F Agent Construction with Claude, Tool Implementation |
| Agentic loop | Send a request, check `stop_reason`, run the requested tools, append the results and repeat until Claude finishes; the Claude Code glossary puts it as gather context, take action, verify results, repeat | CCAR-F 1.1; CCDV-F Agent Patterns and Frameworks |
| `allowed-tools` (skill frontmatter) | The guide says it restricts tool access while a skill runs; current docs say it pre-approves the listed tools and restricts nothing, and `disallowed-tools` removes tools. Expect the guide's meaning on the exam | CCAR-F 3.2 |
| `allowedTools` (Agent SDK) | Auto-approves the listed tools (`allowed_tools` in Python). It does not limit Claude to them: unlisted tools stay available, and calls that need approval fall through to the permission mode and `canUseTool`. The guide requires Task in it for a coordinator to spawn subagents | CCAR-F 1.3 |
| Approval patterns | Ways to put a person in front of tool calls: the Agent SDK's `canUseTool` callback and permission modes, PreToolUse hooks, and confirmation before hard-to-reverse actions. `canUseTool` never fires for auto-approved tools, so logic that must apply to every call belongs in a PreToolUse hook | CCDV-F Tool Implementation |
| `argument-hint` | SKILL.md frontmatter. The guide says it prompts developers for required parameters when they invoke the skill without arguments; the docs describe a hint shown during autocomplete, such as `[issue-number]` | CCAR-F 3.2 |
| Artifacts | Significant, self-contained things Claude makes that you would show someone (a design, document, deck, dashboard or small interactive tool), opened beside the chat and shareable by link. They now require Code execution and file creation to be enabled | CCAO-F D2.6, D3.1 |
| Attention dilution | CCAR-F's term for uneven depth when one pass covers many files at once. Per-file passes plus a separate cross-file integration pass fix it; the Q12 rationale adds that a larger context window does not | CCAR-F 1.6, 4.6, Q12 |
| Augmented LLM | The basic building block of agentic systems: an LLM enhanced with retrieval, tools and memory | CCAR-P 1.3 |
| Auto mode | A Claude Code permission mode (the guide writes auto-mode) in which a separate classifier model reviews actions before they run; it is the built-in starting mode on Pro, Max and Team plans | CCDV-F Claude Code Operation |

**B to C**

| Term | Meaning | In the guides |
|---|---|---|
| Bias | Skewed output, from stereotyping or political bias to subtler defaults such as better quality in some languages; it comes from patterns in the text models learn from | CCAO-F D2.2; CCAR-P 5.5 |
| Built-in tools | Claude Code's own tools. The guide's six: Read and Write for whole files, Edit for targeted changes by unique text match, Bash, Grep to search file contents and Glob to match file paths | CCAR-F 2.5 |
| Business value pillars | CCAR-P's list of what a solution should serve: efficiency, transformation, productivity, cost and performance SLAs | CCAR-P 1.6 |
| Cache check-pointing | CCDV-F's phrase, which is not a docs term. The closest docs concept (our mapping) is the cache breakpoint set with `cache_control`: up to 4 in a prompt, with a 5-minute or 1-hour lifetime | CCDV-F Cost and Token Management |
| Calibration | Checking confidence scores against labeled data before trusting them. CCAR-F has the model output field-level confidence scores, then calibrates review thresholds on labeled validation sets; the Q3 rationale warns that LLM self-reported confidence is poorly calibrated | CCAR-F 5.5, Q3 |
| Capability bloat | Tools or agents configured with more capability than the task needs. Anthropic's context-engineering post calls bloated tool sets with ambiguous decision points one of the most common failure modes, and CCAR-F says too many tools (18 instead of 4 to 5) degrade tool selection | CCAR-P 3.1; CCAR-F 2.3 |
| Case facts block | CCAR-F's pattern of keeping amounts, dates, order numbers and statuses in a persistent block included in each prompt, outside the summarized history | CCAR-F 5.1 |
| Chain of thought | Asking Claude to reason before answering; with thinking off, `<thinking>` and `<answer>` tags keep the reasoning apart from the final output | CCAR-P 2.3 |
| Chunking and indexing | Splitting a corpus into chunks (usually no more than a few hundred tokens) and indexing them for retrieval; chunk size, boundaries and overlap all affect results | CCAR-P 3.5 |
| Claim-source mapping | A structured record linking each claim to its source URL, document name and excerpt, kept through synthesis so attribution is not lost | CCAR-F 5.6 |
| Claude Agent SDK | A Python and TypeScript library that runs the Claude Code binary as a child process of your application, giving your agent Claude Code's tools, agent loop, context management, permissions, sessions and hooks. It was called the Claude Code SDK | CCAR-F Scenarios 1, 3 and 4; CCDV-F Agent Construction with Claude |
| Claude Code | In its own docs, the agentic harness: the layer around the model that provides tools and manages context | CCAR-F Domain 3; CCDV-F Claude Code; CCAR-P 7.1 |
| CLAUDE.md | Markdown instructions Claude Code loads at the start of every session as context, not enforced configuration. The guide's hierarchy runs user (`~/.claude/CLAUDE.md`), project (`./CLAUDE.md` or `./.claude/CLAUDE.md`) and directory level; the docs add a managed policy CLAUDE.md and `CLAUDE.local.md` | CCAR-F 3.1; CCDV-F Configuration Management, Claude Code Operation |
| Client-side and server-side tools | Tools differ by where the code runs. Client tools (your own, and Anthropic-schema tools that your code executes) run in your application and end the turn with `stop_reason: "tool_use"`; server tools such as web search, web fetch and code execution run on Anthropic's infrastructure, and you never build a `tool_result` for them | CCDV-F Tool Implementation |
| Code execution | The Claude apps capability named Code execution and file creation, which lets Claude run code and create Excel, PowerPoint, Word and PDF files; Skills and artifacts need it switched on | CCAO-F How to Prepare |
| Community MCP server | An existing server for a standard integration (the guide's example is Jira), preferred over writing a custom one | CCAR-F 2.4 |
| Compaction | Summarizing a conversation near the context limit and continuing from the summary; `/compact` does it on demand in Claude Code, optionally with focus instructions | CCAR-F 5.4; CCDV-F Context Engineering |
| Connectors | Integrations that let Claude read from and act in your apps and services, using each person's own permissions in the connected service | CCAO-F D5.2 |
| Content boundaries | Clear separation inside a prompt (instructions from data, code from documentation), usually with XML tags, and between trusted instructions and untrusted content | CCDV-F Claude Application Design |
| Context degradation | In long sessions the model starts giving inconsistent answers and citing typical patterns instead of the specific classes it found earlier; Anthropic's context-engineering post and context-window docs call the related effect context rot (recall degrades as token count grows) | CCAR-F 5.4 |
| Context drift and bloat | Quality loss as a long context fills with irrelevant material; CCDV-F names tool output pruning and compaction as preventions and subagents or multi-step workflows for isolation | CCDV-F Context Engineering |
| Context engineering | Curating and maintaining the best set of tokens during inference, including information outside the prompt itself | CCDV-F Context Engineering; CCAR-P Domain 2 |
| Context window | All the text the model can reference while generating, including its own response; the system prompt, messages, tool definitions and output all count | CCDV-F LLM Fundamentals; CCAR-P 2.4 |
| `context: fork` | SKILL.md setting that runs the skill in an isolated subagent that does not see the conversation history | CCAR-F 3.2 |
| Coordinator | The agent in a hub-and-spoke design that decomposes the task, picks which subagents to invoke, delegates, aggregates their results and routes all communication and error handling between them. Its decomposition sets coverage: in Q7 an overly narrow split leaves topics out, and the fault lies with the coordinator, not the subagents | CCAR-F 1.2, Q7 |
| Coverage annotations | Notes in a synthesis marking which findings are well supported and which areas have gaps because sources were unavailable | CCAR-F 5.3 |
| Criterion-referenced assessment | All four exams measure each candidate against a fixed standard, not against other candidates; the cut score is a scaled 720 on a 100 to 1,000 scale | All four guides |
| `custom_id` | The identifier on each Message Batches request; results can arrive in any order, so match them by `custom_id` | CCAR-F 4.5 |

**D to F**

| Term | Meaning | In the guides |
|---|---|---|
| Defensive parsing | Parsing model output so that format variation cannot break it: read tool inputs with a JSON parser rather than string matching, and select content blocks by type rather than position | CCDV-F Output Handling |
| `detected_pattern` | CCAR-F's field recording which code construct triggered a finding, so dismissals can be analyzed by pattern | CCAR-F 4.4 |
| Direct execution | Letting Claude Code make a simple, well-scoped change without a planning phase, such as a single-file bug fix with a clear stack trace | CCAR-F 3.4 |
| Dynamic decomposition | CCAR-F's dynamic adaptive decomposition: generating subtasks from what each step discovers, for open-ended investigation (for example, map the structure, find high-impact areas, then follow a prioritized plan that adapts). Prompt chaining is the choice for predictable multi-aspect reviews | CCAR-F 1.6 |
| Effort | The request setting `output_config.effort` (`low`, `medium`, `high`, `xhigh`, `max`) that governs how many tokens Claude spends on the whole response; most models default to `high`, Claude Opus 5.5 to `medium`, and not every model that supports `max` supports `xhigh` | CCDV-F LLM Fundamentals |
| Embeddings | The vectors used for embedding-based retrieval; Anthropic offers no embedding model of its own and its docs point to Voyage AI. The CCAR-P Sample 3 rationale lists mismatched embeddings as a retrieval fault | CCAR-P Sample 3 (related objectives: 3.5, 3.6) |
| Environment variable expansion | `${VAR}` and `${VAR:-default}` in `.mcp.json`, so credentials stay out of version control | CCAR-F 2.4 |
| `errorCategory` and `isRetryable` | Application-level fields the guide puts inside an MCP error result (transient, validation or permission, plus a retry flag, and `retriable: false` for business-rule violations). They are not MCP spec fields | CCAR-F 2.2 |
| Escalation | Handing a case to a human. Triggers: an explicit request for a human (honor it immediately), a policy exception or gap, or no meaningful progress. Sentiment and self-reported confidence are unreliable proxies for complexity, and multiple customer matches call for more identifiers, not a guess. In Managed Agents docs the word instead means consulting a more capable agent or model | CCAR-F 5.2, Q3 |
| Evaluation (eval) | A test for an AI system: an input plus grading logic applied to the output | CCDV-F Purpose and Value (area 5); CCAR-P Domain 4 |
| Evaluation metrics | What an eval measures; CCAR-P names accuracy, latency, cost, safety and security, and Anthropic's docs ask for criteria that are specific, measurable, achievable and relevant | CCAR-P 4.1 |
| Explore subagent | Claude Code's built-in fast, read-only subagent for searching and analyzing a codebase, which keeps exploration output out of the main conversation | CCAR-F 3.4 |
| Extended thinking | Thinking with a fixed `budget_tokens` budget (`thinking.type: "enabled"`). It is the only mode on Claude Haiku 4.5, Sonnet 4.5 and Opus 4.5, deprecated on the Claude 4.6 models and rejected with a 400 error on Claude 4.7 and later, where adaptive thinking replaces it. CCAR-F 4.6 says an independent review instance beats it for catching subtle issues | CCDV-F LLM Fundamentals; CCAR-F 4.6 |
| False positive | A reported finding that is not a real issue; categories with many false positives undermine trust in the accurate ones | CCAR-F 4.1 |
| Fast mode | A research-preview option that runs the same model with up to 2.5x higher output tokens per second on Claude Opus 5.5, Opus 5 and Opus 4.8, at premium pricing. Opt in with `speed: "fast"` and the `fast-mode-2026-02-01` beta header; it does not apply to batch processing | CCDV-F LLM Fundamentals |
| Few-shot (multishot) prompting | Putting several examples in the prompt; zero-shot means no examples and one-shot (the guide's single-shot) means one | CCDV-F LLM Fundamentals, Prompt Engineering; CCAR-F 4.2; CCAR-P 2.3 |
| Field-level confidence | A confidence score for each extracted field, calibrated on labeled data and used to route human review | CCAR-F 5.5 |
| First-contact resolution | The Scenario 1 success measure: the support agent targets 80% or more while still knowing when to escalate. In Q3 the agent sits at 55% while escalating straightforward cases and handling policy-exception cases itself; the fix is explicit escalation criteria with few-shot examples | CCAR-F Scenario 1, Q3 |
| `fork_session` | The Python Agent SDK option (`forkSession` in TypeScript, `--fork-session` in the CLI) that resumes a session under a new ID, so a copy of its history can explore a different approach. It branches the conversation, not the files | CCAR-F 1.3, 1.7 |

**G to L**

| Term | Meaning | In the guides |
|---|---|---|
| Guardrail layering | Combining safeguards (input screening, validation, prompt instructions, monitoring) so that no single layer carries the load | CCDV-F Guardrails and Safe Deployment; CCAR-P 5.1 |
| Hallucination | Fabricated, specific-looking detail such as a citation number; fabrication clusters in names, dates, statistics, citations, URLs and quotes | CCAO-F D2.2; CCAR-P 4.4 |
| Headless mode (non-interactive mode) | Running Claude Code with `-p` (`--print`): it processes the prompt, prints the result to stdout and exits without waiting for input, which is what CI needs. The docs now say non-interactive mode for the same flag | CCAR-F 3.6, Q10; CCDV-F Claude Code Operation |
| Hooks | Handlers that fire at fixed lifecycle points and so give deterministic control rather than relying on the model: PreToolUse runs on a tool call request and can block or modify it (the guide's tool call interception); PostToolUse runs after a tool succeeds and can add context or replace what Claude sees | CCAR-F 1.5; CCDV-F Claude Hooks, Agent Construction with Claude |
| Hub-and-spoke | A multi-agent design in which one coordinator manages all communication, error handling and information routing between subagents | CCAR-F 1.2 |
| Human-in-the-loop | Keeping people in the decision path in proportion to risk. The Claude and Accenture pilot-to-production guide uses four tiers: automated (no human review), sampled (a random subset reviewed on a cadence), reviewed (a human approves every output) and advisory (the AI analyzes, a human decides) | CCAR-P 5.3 |
| `@import` | `@path/to/import` syntax that pulls another file into CLAUDE.md to keep it modular. Imported files load at launch, so imports help organization but do not reduce context | CCAR-F 3.1 |
| Independent review instance | A second Claude instance without the generator's reasoning context, which catches subtle issues better than self-review instructions | CCAR-F 3.6, 4.6 |
| `/init` | Generates a starting CLAUDE.md by analyzing the codebase; the guide calls this repository initialization | CCDV-F Claude Code Operation |
| Input sanitization | Screening untrusted input before it reaches the main prompt, for example with pattern filters or a lightweight model, and JSON-encoding third-party strings | CCDV-F Prompt Engineering, AI Application Security |
| Interview pattern | Having Claude ask you questions that surface design considerations before it implements | CCAR-F 3.5 |
| `isError` | The MCP flag that marks a tool execution error inside a normal result, so the model can see it and recover; protocol errors travel as JSON-RPC errors instead. The Claude API's `tool_result` uses `is_error` | CCAR-F 2.2 |
| Jailbreak and prompt injection | In a jailbreak or direct injection the user of your application is the adversary; in indirect prompt injection the user is trusted but Claude processes third-party content (web pages, emails, tool results) that carries instructions. CCDV-F Sample 2's answer treats retrieved content as untrusted input kept apart from trusted instructions, with guardrails or hooks so injected text cannot trigger sensitive actions | CCDV-F AI Application Security, Sample 2 |
| JSON Schema | The schema language for tool inputs and structured output; the guide tests required versus optional fields, enums, nullable fields and `"other"` plus a detail string | CCAR-F 4.3 |
| Least privilege | Giving a role only the capabilities it needs and removing the rest, so a successful injection can do little damage. In CCAR-F Q9 it means giving the synthesis agent a scoped `verify_fact` tool for simple lookups rather than all web search tools | CCDV-F Guardrails and Safe Deployment; CCAR-P Sample 1; CCAR-F 2.3, Q9 |
| Lost in the middle | Models process the start and end of long inputs reliably but may miss findings in the middle; Anthropic's docs put long documents first and the query at the end | CCAR-F 5.1 |

**M to O**

| Term | Meaning | In the guides |
|---|---|---|
| Managed agent deployment | Claude Managed Agents: an Anthropic-hosted agent harness (beta header `managed-agents-2026-04-01`) whose sessions run in an Anthropic-managed cloud sandbox or a self-hosted sandbox. Self-hosting the Agent SDK instead runs the whole loop on your infrastructure | CCDV-F Agent Construction with Claude |
| Manager/supervisor hierarchy | A lead agent that breaks work down and delegates to worker agents; the closest Anthropic term (our mapping) is the orchestrator-workers pattern | CCDV-F Agent Architecture |
| Manifest | CCAR-F's crash-recovery record: each agent exports its state to a known location and the coordinator loads the manifest on resume | CCAR-F 5.4 |
| MCP (Model Context Protocol) | The open standard Anthropic created for connecting AI applications to tools and data. A host creates one client per server, and servers expose tools, resources and prompts. In CCDV-F Sample 3, a capability that several Claude applications must reuse is built as an MCP server | CCAR-F Domain 2; CCDV-F Tools and MCPs, Sample 3; CCAR-P 3.7 |
| MCP host, client and server | Host: the application that starts connections and enforces consent. Client: a connector inside the host that talks to exactly one server. Server: a service that provides context and capabilities | CCDV-F MCP Server Development |
| MCP prompts | User-controlled templates a server offers, usually run as slash commands | CCDV-F MCP Server Development |
| MCP resources | Application-controlled, read-only context identified by URI, such as files, schemas and documentation; the guide uses them to expose content catalogs | CCAR-F 2.4; CCDV-F MCP Server Development |
| MCP tools | Model-controlled functions the LLM decides to call to take actions such as writing to a database or calling an API | CCAR-F 2.1, 2.4; CCDV-F MCP Server Development |
| `.mcp.json` and `~/.claude.json` | `.mcp.json` at the repository root holds project-scoped servers, committed and shared; `~/.claude.json` holds user-scoped servers and local-scoped ones. The guide knows two scopes and puts personal or experimental servers in user scope; the docs add local scope, which is the default and is recommended for experiments | CCAR-F 2.4 |
| `/memory` | Lists and opens the memory files (CLAUDE.md and others); the guide uses it to check which memory files load, and the docs point to `/context` for what actually loaded | CCAR-F 3.1 |
| Memory (Claude apps) | Claude saves individual topics as you chat and each project keeps its own memory space; the guide asks when to restart, summarize or persist | CCAO-F D3.4, How to Prepare |
| Message Batches API | Asynchronous processing at 50% of standard cost; most batches finish within an hour, a batch expires if not done within 24 hours, and results stay available for 29 days. With no guaranteed latency SLA it suits overnight or weekly jobs, not blocking pre-merge checks, and a request cannot run client tools mid-request and continue. CCDV-F Sample 1 picks it for an overnight, cost-sensitive job | CCAR-F 4.5, Q11; CCDV-F Claude API Mechanics, Sample 1 |
| Messages API | Claude's core API. It is stateless, so every request carries the full conversation history | CCAR-F 5.1; CCDV-F Claude API Mechanics |
| Minimally qualified candidate (MQC) | The candidate profile an exam targets. CCAO-F, CCDV-F and CCAR-P each have an MQC profile section (CCAR-F has none), and all four guides say subject matter experts set the passing standard by judging the performance expected of this candidate | All four guides |
| Mixed methodologies | Combining evaluation methods; Anthropic's agent evals mix code-based grading (fastest, most reliable), model-based grading (flexible; test its reliability first) and human grading (most flexible, slow and expensive) | CCAR-P 4.2 |
| Model tiers | Haiku for the lowest latency and price, Sonnet for everyday speed plus capability, Opus for complex agentic coding and enterprise work. CCAO-F and CCDV-F name only these three; the current lineup adds Claude Fable 5.1 above Opus for the highest capability | CCAO-F D3.2; CCDV-F Model Selection and Tradeoffs; CCAR-P 2.1 |
| Model version pinning | Every Claude model ID is a pinned snapshot; from the 4.6 generation IDs carry no date but are still fixed, and updates ship under new IDs | CCDV-F Configuration Management |
| Multi-pass review | Per-file passes for local issues plus a separate pass for cross-file data flow | CCAR-F 1.6, 4.6 |
| Multiple-response item | An exam item with more than one correct option; each item states how many responses to select | All four guides |
| Next-token generation | Autoregressive models are pretrained to predict the next word from the preceding context | CCDV-F LLM Fundamentals |
| Non-determinism | Outputs vary between runs; even at temperature 0.0 results are not fully deterministic | CCDV-F LLM Fundamentals |
| Observability | Seeing what a system does in production. The Agent SDK can export OpenTelemetry traces, metrics and log events (each signal enabled separately; traces are beta) that show which tools ran, request latency, tokens spent and where failures occurred; Claude Code exports OpenTelemetry metrics and events once `CLAUDE_CODE_ENABLE_TELEMETRY` is set | CCAR-P 3.4, 4.6 |
| `--output-format json` and `--json-schema` | CLI flags for print mode that return JSON and, with a schema, validated output in the `structured_output` field | CCAR-F 3.6 |

**P to R**

| Term | Meaning | In the guides |
|---|---|---|
| Parallel subagents | Spawning several subagents at once by emitting multiple Task (now Agent) tool calls in a single coordinator response, rather than across separate turns | CCAR-F 1.3 |
| Path-scoped rules | Files in `.claude/rules/` with a `paths` glob list in YAML frontmatter, loaded only for matching files: the guide says when editing one, the docs say when Claude reads one. Rules without `paths` always load. Glob rules beat subdirectory CLAUDE.md files for conventions spread across directories, such as `**/*.test.tsx` | CCAR-F 3.3, Q6 |
| PII handling | Keeping personal identifiers such as customer names and account numbers out of prompts and outputs where policy requires it; CCAO-F Sample 3 anonymizes them before upload | CCDV-F AI Application Security; CCAO-F Sample 3 |
| Plan mode | Claude researches and proposes changes without making them; enter it with Shift+Tab or by starting a prompt with `/plan`, or start in it with `claude --permission-mode plan`. The guide says it is designed for complex tasks: large-scale changes, multiple valid approaches, architectural decisions and multi-file modifications | CCAR-F 3.4, Q5 |
| Plugins | Installable bundles of skills, agents, hooks, MCP servers and other components, distributed through marketplaces | CCDV-F Claude Application Design, Configuration Management |
| Prerequisite gate | Code that blocks a downstream tool call until a required step has completed, such as blocking `process_refund` until `get_customer` has returned a verified ID. When a tool order is critical business logic, the Q1 rationale prefers this deterministic enforcement over prompt instructions | CCAR-F 1.4, Q1 |
| Progressive discovery | Loading context on demand (skills whose full content loads only when relevant, tool search, just-in-time retrieval), as opposed to a monolithic context loaded up front | CCAR-P 3.8 |
| Progressive summarization | Repeatedly condensing history, which blurs numbers, percentages, dates and customer-stated expectations | CCAR-F 5.1 |
| Projects | Workspaces in the Claude apps with their own chat history, knowledge (uploaded files used as background in every chat) and instructions | CCAO-F D3.1, D5.1 |
| Prompt caching | Reusing a processed prompt prefix (tools, then system, then messages, up to a `cache_control` breakpoint). Cache reads cost 10% of base input on most models (5% on Claude Opus 5.5, 2.5% on Claude Fable 5.1 and Claude Mythos 5.1); 5-minute writes cost 1.25x base input and 1-hour writes 2x. CCAR-P Sample 2 puts static content before dynamic content so the prefix can be cached | CCDV-F Cost and Token Management; CCAR-P 2.5, Sample 2; CCAR-F out-of-scope list beyond knowing it exists |
| Prompt chaining | Splitting a task into fixed sequential steps where each call works on the previous output; it trades latency for accuracy | CCAR-F 1.6; CCAR-P 1.5 |
| Prompt versioning | Treating prompts and configuration as versioned artifacts with testing and documentation; an Anthropic Academy course recommends checking CLAUDE.md into Git at the repository root so changes are reviewed like code | CCDV-F Configuration Management |
| Pydantic | Listed in the guide's appendix for schema validation, semantic validation errors and validation-retry loops; Exercise 3 retries when Pydantic or JSON schema validation fails. The Python SDK's `client.messages.parse()` accepts Pydantic models for structured outputs | CCAR-F appendix, Exercise 3 |
| RAG (retrieval-augmented generation) | Retrieving relevant content from an external knowledge base at query time and passing it into the context window | CCAR-P 3.5, 3.6 |
| Research mode | The Claude apps Research feature: agentic, multi-step searching across the web and connected sources, suited to questions that need five or more tool calls | CCAO-F D3.1 |
| Retrieval strategies | Ways to find the right chunks. Lexical BM25 matches exact terms such as error codes that embeddings can miss, hybrid retrieval merges the two with rank fusion, and reranking re-scores the candidates at extra latency and cost | CCAR-P 3.6 |
| Retry with error feedback | A follow-up request carrying the original document, the failed extraction and the specific validation errors, so the model can correct itself. Retries fix format and structural errors; they cannot recover information that is absent from the source | CCAR-F 4.4 |

**S**

| Term | Meaning | In the guides |
|---|---|---|
| Sampling and temperature | `temperature` (0.0 to 1.0, default 1.0) controls randomness. The docs now mark it deprecated: Claude 4.7 and later models (Sonnet 5 included) and Claude Mythos Preview reject any non-default value with a 400 error and accept 1.0 only for backwards compatibility. The API reference's shorthand, models released after Claude Opus 4.6, is looser than that. In CCDV-F Sample 2 (a prompt injection), raising temperature is a wrong option because temperature is irrelevant to injection | CCDV-F LLM Fundamentals, Sample 2 |
| Scenario | CCAR-F frames items in production scenarios: each exam presents 4 scenarios drawn at random from a bank of 6 | CCAR-F |
| Scratchpad file | A file where an agent records key findings to reread for later questions, countering context degradation | CCAR-F 5.4 |
| Secrets management | Keeping keys and credentials out of source control, client code and prompts, setting expirations, and injecting credentials through a proxy so an agent never sees them | CCDV-F Identity, Secrets, and Key Management |
| Session | The conversation history an agent accumulates, saved to disk. Continue the most recent, resume a specific one by ID or name (`claude --resume <name>`), or fork it | CCAR-F 1.7; CCDV-F Claude Code Operation |
| Session hygiene | Keeping context clean: `/clear` between unrelated tasks, and a fresh session with a better prompt rather than a long one full of corrections | CCDV-F Claude Application Design |
| `settings.json` | Claude Code settings files: user `~/.claude/settings.json`, shared project `.claude/settings.json`, personal `.claude/settings.local.json`, and managed settings, which the other files cannot override (for a few restrictive keys, a stricter value from a lower scope still applies). Precedence, highest first: managed, command-line arguments, local project, shared project, user | CCDV-F Configuration Management, Claude Code Operation |
| Skills | Folders with a SKILL.md (YAML frontmatter plus instructions) and optional scripts and resources that Claude loads when relevant. Only names and descriptions load at startup; in Claude Code a skill also becomes `/skill-name` | CCAR-F 3.2; CCDV-F Agentic Customization; CCAR-P 2.5; CCAO-F How to Prepare |
| Slash commands | `/name` shortcuts. The guide puts project commands in `.claude/commands/` (shared through version control, the Q4 answer) and personal ones in `~/.claude/commands/`. Custom commands have since been merged into skills; `.claude/commands/` files still work, but the docs recommend skills for new work | CCAR-F 3.2, Q4 |
| stdio | The MCP transport in which the client launches the server as a subprocess and they exchange newline-delimited JSON-RPC over stdin and stdout. The spec's other standard transport is Streamable HTTP; the stdio framing also works over Unix domain sockets or TCP as a custom transport, one possible reading of the sockets that CCDV-F lists without defining | CCDV-F MCP Server Development |
| `stop_reason` | Why a response ended: `end_turn`, `max_tokens`, `stop_sequence`, `tool_use`, `pause_turn`, `refusal` or `model_context_window_exceeded`. The compaction betas (as of September 2026) also return `compaction`, a value the stop reasons page does not list | CCAR-F 1.1 |
| Stratified random sampling | CCAR-F's method for measuring error rates in high-confidence extractions and catching new error patterns, alongside accuracy checks by document type and field | CCAR-F 5.5 |
| Streamable HTTP | The MCP remote transport: one endpoint, every client message an HTTP POST, replies as JSON or an SSE stream. It was introduced in protocol version 2025-03-26 to replace the older HTTP+SSE transport | Not named in the guides; the spec's other standard transport, relevant to CCDV-F MCP Server Development (stdio, sockets, client vs. server) |
| Streaming | Receiving a response incrementally as server-sent events by setting `"stream": true`. CCDV-F also names a Claude Code streaming mode without defining it; the closest features are `--output-format stream-json` and the Agent SDK's streaming input mode | CCDV-F Claude API Mechanics, Claude Code Operation |
| Strict tool use | `strict: true` on a tool definition, which constrains sampling so tool inputs always match the schema; the guide's strict mode removes syntax errors but not semantic ones | CCAR-F 4.3 |
| Structured error context | What a subagent returns when it cannot recover: failure type, what was attempted, partial results and possible alternatives, so the coordinator can retry, try another approach or proceed with partial results. Generic statuses, suppressed errors and ending the whole workflow on one failure are the wrong options in Q8 | CCAR-F 5.3, Q8 |
| Structured handoff | The summary sent to a human who cannot see the transcript: customer ID, root cause, refund amount and recommended action | CCAR-F 1.4 |
| Structured outputs | API features that guarantee schema-compliant output through constrained decoding: JSON outputs (`output_config.format`) and strict tool use. The guide does not use this framing: it calls tool use with JSON schemas the most reliable approach, and its appendix's strict mode matches `strict: true`. Answer in its terms | CCAR-F 4.3; CCDV-F Output Handling |
| Subagent | A separate agent instance spawned for a focused subtask, with its own context window. Unless it is a fork, it starts without the parent's conversation, so the Agent tool's prompt string is the only content passed from the parent; only its final response returns to the parent | CCAR-F 1.2, 1.3; CCDV-F Agent Architecture |
| System prompt | Instructions in the API's top-level `system` field; a role there focuses behavior and tone. In the Claude apps the closest equivalents (our mapping) are profile instructions in Settings (all your conversations), project instructions (every chat in a project) and, on Team and Enterprise plans, organization instructions | CCAR-P 2.2; CCAO-F D5.3 |

**T to W**

| Term | Meaning | In the guides |
|---|---|---|
| Test-driven iteration | Writing the tests first, then iterating by sharing test failures with Claude | CCAR-F 3.5 |
| Third-party vendors | Claude on Amazon Bedrock and Google Cloud (partner-operated) and on Claude Platform on AWS and Microsoft Foundry (Anthropic-operated); feature availability varies by platform. Current docs call the Google Cloud offering Agent Platform, while its Python client is still `AnthropicVertex` | CCDV-F Claude API Mechanics |
| Token | The unit models read, generate and bill by. Anthropic's glossary puts one token at about 3.5 English characters and its pricing FAQ at about 4 characters or 0.75 words; treat both as rough estimates | CCDV-F LLM Fundamentals |
| Tool description | The primary mechanism the model uses to choose among tools; a good one covers input formats, example queries, edge cases and when to use it instead of similar tools. In Q2, improving minimal descriptions is the right first step | CCAR-F 2.1, Q2 |
| Tool set construction | Building a small set of tools with distinct purposes; wrapping every API endpoint as a tool is a common error | CCDV-F Tool Implementation |
| `tool_choice` | `auto` (Claude decides; the default when tools are provided), `any` (must call some tool), `tool` (must call the named tool) and `none` (no tools; the default when no tools are provided). The guide lists only the first three. As of September 2026, Claude Opus 5.5, Claude Fable 5.1 and Claude Mythos 5.1 reject `any` and `tool` with a 400 error, and the docs substitute `auto` with `strict: true` or structured outputs; on the exam, `any` and forced selection remain the guide's answers for guaranteeing a tool call | CCAR-F 2.3, 4.3 |
| `tool_use` and `tool_result` | A `tool_use` block (`id`, `name`, `input`) is Claude's request; your code answers in the next user message with a `tool_result` block (`tool_use_id`, content, optional `is_error`) | CCAR-F 1.1; CCDV-F Tool Implementation |
| Trace analysis | Reading the full record of a run (the transcript or trace; for the API, the full messages array) to identify failure modes | CCDV-F Debugging and Error Handling |
| Vision | Image understanding: images arrive as `image` content blocks (base64, URL or Files API) and Claude reads but does not generate them | CCDV-F Claude API Mechanics; out of scope for CCAR-F |
| WebSocket | A full-duplex protocol over one TCP connection, opened with an HTTP Upgrade handshake. The Messages API streams with server-sent events, not WebSockets, while Claude Code accepts WebSocket MCP servers (`type: "ws"`) | CCDV-F Technical Fundamentals |
| Workflow | A system that orchestrates LLMs and tools through predefined code paths, suited to well-defined tasks. Anthropic's workflow patterns are prompt chaining, routing, parallelization, orchestrator-workers and evaluator-optimizer | CCDV-F Agent Architecture; CCAR-P 1.3 |

??? info "Sources"

    - [Claude Certified Associate, Foundations Exam Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf): CCAO-F exam details, domain weights, retake, renewal and cancellation wording in the July 2026 guide; its objectives, scope and escalation wording, How to Prepare list and Samples 1 to 3 with rationales
    - [Claude Certified Developer, Foundations Exam Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf): CCDV-F exam details and domain weights; its skill descriptions (agents, API mechanics, caching with "cache check-pointing", configuration, Claude Code, hooks, security, tools and MCP) and Samples 1 to 3 with rationales
    - [Claude Certified Architect, Foundations Exam Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): CCAR-F exam details, scenario structure, task statements and anti-patterns, sample questions Q1 to Q12 and rationales, preparation exercises, appendix lists and out-of-scope list; the source of most decision rules and many glossary terms
    - [Claude Certified Architect, Professional Exam Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf): CCAR-P exam details, domain weights, recommended experience, objectives (capability bloat, integration mechanisms, progressive discovery, evaluation, governance) and Samples 1 to 3 with rationales, including Sample 2 on prompt caching
    - [Anthropic Partner Academy: Certifications FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications): fees, discounts, pass mark, seat time, cancellation, retakes, name corrections, accommodations lead time, age, results, Credly badges, validity, appeals, and the scenario-based multiple response format shared by all four exams
    - [Anthropic Partner Academy: Certification Policies](https://anthropic-partners.skilljar.com/page/policies-certifications): 48-hour cancellation window, retake rules and version reset, renewal assessment, appeals remedy
    - [Anthropic Partner Academy: Partner certifications](https://anthropic-partners.skilljar.com/page/partner-certifications): list prices by certification and the CPN tier-eligibility note for the Associate exam
    - [Anthropic Certification Exam Policy (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F34hhd92iyp94a0gtbr15cy5jk%2Fpublic%2F1782870704%2FAnthropic+Certification+Exam+Policy.pdf): accommodations approved before scheduling; no AI products or services during the exam
    - [Pearson VUE: Anthropic](https://www.pearsonvue.com/us/en/anthropic.html): exam codes, 48-hour test-center rescheduling window, attempts per 12 months
    - [Pearson VUE: Anthropic online proctored exams (OnVUE)](https://www.pearsonvue.com/us/en/anthropic/onvue.html): OnVUE check-in timing, system and network minimums, digital whiteboard
    - [Pearson VUE: Accommodations for Anthropic](https://www.pearsonvue.com/us/en/test-takers/accommodations/pearson_approve.anthropic.html): 10 business days for review; no accommodations added to a scheduled exam
    - [Claude Docs: API overview](https://platform.claude.com/docs/en/api/overview): base URL, endpoints, required headers, authentication header wording, cloud platforms and third-party access
    - [Claude Docs: Get started](https://platform.claude.com/docs/en/get-started): the quickstart curl example, which still sends `x-api-key`
    - [Claude Docs: Messages API reference](https://platform.claude.com/docs/en/api/messages/create): request fields (`max_tokens`, `messages`, `system`, `tools`, `tool_choice`, `thinking`, `output_config`, sampling parameters and their deprecation, `metadata`, `service_tier`, `cache_control`) and the response and usage shape
    - [Claude Docs: Working with messages](https://platform.claude.com/docs/en/build-with-claude/working-with-messages): statelessness and prefill support by model
    - [Claude Docs: Versioning](https://platform.claude.com/docs/en/api/versioning): the `anthropic-version` header
    - [Claude Docs: Beta headers](https://platform.claude.com/docs/en/api/beta-headers): `anthropic-beta` naming and use
    - [Claude Docs: Handling stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons): every `stop_reason` value and how to handle it; refusals as successful responses
    - [Claude Docs: Streaming](https://platform.claude.com/docs/en/build-with-claude/streaming): event flow, cumulative usage, mid-stream errors
    - [Claude Docs: Streaming refusals](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals): what to do after a refusal
    - [Claude Docs: Errors](https://platform.claude.com/docs/en/api/errors): HTTP error codes and types, request size limits, retries, request IDs, forced tool use errors on newer models
    - [Claude Docs: Rate limits](https://platform.claude.com/docs/en/api/rate-limits): spend-cap 429, `retry-after` and rate-limit headers, cache-aware input limits, Batches API limits
    - [Claude Docs: Service tiers](https://platform.claude.com/docs/en/api/service-tiers): `service_tier` values
    - [Claude Docs: Python SDK](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python): retried status codes and exception classes
    - [Claude Docs: Get an API key](https://platform.claude.com/docs/en/get-api-key): `sk-ant-` prefix and the `ANTHROPIC_API_KEY` variable
    - [Claude Docs: Models overview](https://platform.claude.com/docs/en/models/overview): the four headline models' IDs, latency, prices, context, output limits, thinking modes and default effort; the September 2026 lineup and pinned model IDs
    - [Claude Docs: Model IDs and versions](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions): dateless IDs, pinned snapshots and aliases
    - [Claude Docs: Claude Haiku 4.5 overview](https://platform.claude.com/docs/en/models/haiku-4-5/overview): Haiku 4.5 ID, alias and extended thinking
    - [Claude Docs: What's new in Claude Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5): forced tool use returns 400; select content blocks by type
    - [Claude Docs: Model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations): sampling parameters rejected on Claude 4.7 and later
    - [Claude Docs: Pricing](https://platform.claude.com/docs/en/about-claude/pricing): cache multipliers and break-even, batch discount, model-choice rule of thumb
    - [Claude Docs: Extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking) and [Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking): thinking modes, `budget_tokens` rules, deprecation on 4.6 models and 400 errors from 4.7 on, tool_choice limits with thinking
    - [Claude Docs: Effort](https://platform.claude.com/docs/en/build-with-claude/effort): effort levels and defaults
    - [Claude Docs: Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows): what counts toward the window; cached prefixes still count
    - [Claude Docs: Token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting): free counting endpoint with its own limits; counts are estimates
    - [Claude Docs: Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): enabling caching, breakpoints, lookback, TTLs, minimum lengths, invalidation, usage fields, pre-warming
    - [Claude Docs: Tool use with prompt caching](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching): `disable_parallel_tool_use` changes invalidate the messages cache
    - [Claude Docs: Batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing): batch limits, timing, 24-hour expiry, 29-day results, `custom_id`, unsupported parameters, caching in batches, server tools in batches
    - [Claude Docs: Message Batches API reference](https://platform.claude.com/docs/en/api/messages/batches): `processing_status` values
    - [Claude Docs: Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) and [Strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use): `output_config.format`, `strict: true`, constrained decoding, incompatibility with citations
    - [Claude Docs: Define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools) and [Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use): `tool_choice` values, prefill behavior, `disable_parallel_tool_use`, unsupported forced choice on the newest models, consolidating related operations, multiple tool calls in one response
    - [Claude Docs: Compaction at a token threshold](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold) and [Compaction on demand](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand): the `compaction` stop reason
    - [Claude Docs: Release notes](https://platform.claude.com/docs/en/release-notes/overview): forced `tool_choice` returning 400 on Fable 5.1, Mythos 5.1 and Opus 5.5
    - [Claude Code Docs: Memory](https://code.claude.com/docs/en/memory): CLAUDE.md locations, load order, imports, size, `/init`, auto memory, rules and `/memory` versus `/context`
    - [Claude Code Docs: Settings](https://code.claude.com/docs/en/settings) and [Settings reference](https://code.claude.com/docs/en/settings-reference): settings files, precedence, managed settings, key reference
    - [Claude Code Docs: Permissions](https://code.claude.com/docs/en/permissions) and [Permission modes](https://code.claude.com/docs/en/permission-modes): rule syntax (including MCP rules) and evaluation order, the six modes, plan mode, auto mode and the starting mode
    - [Claude Code Docs: Managed settings](https://code.claude.com/docs/en/managed-settings): managed-settings.json paths and precedence
    - [Claude Code Docs: Sandboxing](https://code.claude.com/docs/en/sandboxing): OS-level Bash isolation and supported platforms
    - [Claude Code Docs: Skills](https://code.claude.com/docs/en/skills): SKILL.md locations, frontmatter, commands merged into skills, `allowed-tools`, `disallowed-tools`, `argument-hint`, `context: fork`, dynamic context injection, precedence
    - [Claude Code Docs: Commands](https://code.claude.com/docs/en/commands): built-in commands and aliases
    - [Claude Code Docs: Interactive mode](https://code.claude.com/docs/en/interactive-mode): input prefixes and keyboard shortcuts
    - [Claude Code Docs: Subagents](https://code.claude.com/docs/en/sub-agents): subagent files, frontmatter, precedence, the Task-to-Agent rename, Explore, forks and nesting limits
    - [Claude Code Docs: The .claude directory](https://code.claude.com/docs/en/claude-directory): file map and frontmatter fields per file type
    - [Claude Code Docs: Debug your configuration](https://code.claude.com/docs/en/debug-your-config): common file-location mistakes and CLAUDE.md (guidance) versus permissions (enforcement)
    - [Claude Code Docs: Hooks reference](https://code.claude.com/docs/en/hooks) and [Hooks guide](https://code.claude.com/docs/en/hooks-guide): events, matchers, handler types, exit codes, JSON output and decision fields, `updatedToolOutput`, trust, deterministic control, the settings and script examples
    - [Claude Code Docs: CLI reference](https://code.claude.com/docs/en/cli-reference) and [Run Claude Code programmatically](https://code.claude.com/docs/en/headless): print mode, output formats, `--json-schema`, `--fork-session`, `--bare` and the other flags; `-p` and structured output in CI
    - [Claude Code Docs: Sessions](https://code.claude.com/docs/en/sessions) and [Checkpointing](https://code.claude.com/docs/en/checkpointing): continue, resume, name, branch, rewind and transcript retention
    - [Claude Code Docs: Context window](https://code.claude.com/docs/en/context-window): what survives compaction
    - [Claude Code Docs: Model configuration](https://code.claude.com/docs/en/model-config) and [Environment variables](https://code.claude.com/docs/en/env-vars): model selection order, alias pinning, provider and compaction variables
    - [Claude Code Docs: Output styles](https://code.claude.com/docs/en/output-styles): output style locations and frontmatter
    - [Claude Code Docs: Plugins reference](https://code.claude.com/docs/en/plugins-reference), [Plugin marketplaces](https://code.claude.com/docs/en/plugin-marketplaces), [Discover plugins](https://code.claude.com/docs/en/discover-plugins) and [Plugin dependencies](https://code.claude.com/docs/en/plugin-dependencies): plugin components, manifest fields, marketplace catalog, `enabledPlugins`, dependency ranges
    - [Claude Code Docs: Changelog](https://code.claude.com/docs/en/changelog): commands merged into skills (v2.1.3), removed `#` shortcut and `/vim`, v2.1.280 date
    - [Claude Code Docs: GitHub Actions](https://code.claude.com/docs/en/github-actions), [GitLab CI/CD](https://code.claude.com/docs/en/gitlab-ci-cd) and [Code Review](https://code.claude.com/docs/en/code-review): CI setup, secrets, trigger modes, managed review behavior
    - [Claude Code Docs: Best practices](https://code.claude.com/docs/en/best-practices): the `--output-format json` result shape, `/clear`, fresh sessions and CLI tools for external services
    - [Claude blog: Message Batches API launch post (October 2024)](https://claude.com/blog/message-batches-api): the October 2024 limit of 10,000 queries per batch, now superseded
    - [Anthropic Partner Academy: Claude Certified Architect, Professional prep course](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional): learning objectives on evaluations as gates, fail-closed safety stacks, routing decisions by confidence, reversibility and cost, and mapping compliance obligations to controls
    - [Claude Docs: Intro to Claude](https://platform.claude.com/docs/en/intro): API key hygiene
    - [Claude Docs: Glossary](https://platform.claude.com/docs/en/about-claude/glossary): definitions of tokens, temperature, next-token pretraining and RAG
    - [Claude Docs: Fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode): research-preview status, supported models and speed
    - [Claude Docs: Vision](https://platform.claude.com/docs/en/build-with-claude/vision): image input and the limits of image understanding
    - [Claude Docs: Embeddings](https://platform.claude.com/docs/en/build-with-claude/embeddings): Anthropic's pointer to Voyage AI for embeddings
    - [Claude Docs: Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback): monitoring refusals as their own signal
    - [Claude Docs: Choosing a model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model): selection criteria, effort as a lever, use-case benchmarks
    - [Claude Docs: Claude Opus 5.5 overview](https://platform.claude.com/docs/en/models/opus-5-5/overview): always-on adaptive thinking
    - [Claude Docs: Sonnet 5 migration guide](https://platform.claude.com/docs/en/models/sonnet-5/migration-guide): selecting content blocks by type
    - [Claude Docs: Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview): client tools, server tools and function calling
    - [Claude Docs: How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works): the `stop_reason` loop, exit conditions and `pause_turn`
    - [Claude Docs: Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls): `tool_use` and `tool_result` blocks and `is_error`
    - [Claude Docs: Tool runner](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner): when to use the manual loop instead
    - [Claude Docs: Troubleshooting tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/troubleshooting-tool-use): parsing tool inputs with a JSON parser
    - [Claude Docs: MCP connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector): `mcp_servers` and `mcp_toolset`, beta header, limitations, allowlists and denylists, platform availability
    - [Claude Docs: Mitigate jailbreaks and prompt injections](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks): the two threat models, untrusted content in `tool_result`, screens, JSON encoding, least privilege, guardrail layering
    - [Claude Docs: Increase output consistency](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/increase-consistency): breaking complex tasks into subtasks
    - [Claude Docs: Develop tests](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests): success criteria and the three grading methods
    - [Claude Docs: Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices): examples and their count, long-document placement, manual chain of thought, role prompting
    - [Claude Docs: API and data retention](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention): features outside ZDR and HIPAA eligibility, including batches
    - [Claude Docs: Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview): concepts, beta header, cloud or self-hosted environments, ZDR and HIPAA BAA ineligibility
    - [Claude Docs: Managed Agents self-hosted sandboxes](https://platform.claude.com/docs/en/managed-agents/self-hosted-sandboxes): orchestration on Anthropic's side, tool execution on yours
    - [Claude Docs: Managed Agents permission policies](https://platform.claude.com/docs/en/managed-agents/permission-policies): `always_allow`, `always_ask` and `auto`
    - [Claude Docs: Migrating from the Agent SDK to Managed Agents](https://platform.claude.com/docs/en/managed-agents/migration): `max_turns` and hooks move to the client
    - [Claude Code Docs: Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview): what the SDK is and how it compares with the Client SDK, the CLI and Managed Agents
    - [Claude Code Docs: Agent SDK quickstart](https://code.claude.com/docs/en/agent-sdk/quickstart): the minimal agent example
    - [Claude Code Docs: Migrate to the Claude Agent SDK](https://code.claude.com/docs/en/agent-sdk/migration-guide): the rename, package and type names, default system prompt and `settingSources`
    - [Claude Code Docs: Agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop): turns, message and result types, limits, effort, compaction, parallel tool execution
    - [Claude Code Docs: Agent SDK Python reference](https://code.claude.com/docs/en/agent-sdk/python): `ClaudeAgentOptions`, `query()` versus `ClaudeSDKClient`, hook events, AgentDefinition fields
    - [Claude Code Docs: Agent SDK TypeScript reference](https://code.claude.com/docs/en/agent-sdk/typescript): TypeScript-only options such as `allowDangerouslySkipPermissions` and `persistSession`
    - [Claude Code Docs: Agent SDK permissions](https://code.claude.com/docs/en/agent-sdk/permissions): evaluation order, permission modes, `allowedTools` behavior and locked-down agents
    - [Claude Code Docs: Agent SDK user input](https://code.claude.com/docs/en/agent-sdk/user-input): `canUseTool`, `AskUserQuestion` and `defer`
    - [Claude Code Docs: Agent SDK hooks](https://code.claude.com/docs/en/agent-sdk/hooks): hook events, matchers, outputs, precedence and timeouts; the base for the refund-limit example
    - [Claude Code Docs: Agent SDK subagents](https://code.claude.com/docs/en/agent-sdk/subagents): the Agent tool, AgentDefinition, context passing, growth limits and the Task naming quirk
    - [Claude Code Docs: Agent SDK sessions](https://code.claude.com/docs/en/agent-sdk/sessions): continue, resume and fork, storage and moving work between hosts
    - [Claude Code Docs: File checkpointing](https://code.claude.com/docs/en/agent-sdk/file-checkpointing): what file rewinding covers
    - [Claude Code Docs: Agent SDK custom tools](https://code.claude.com/docs/en/agent-sdk/custom-tools): in-process MCP servers, handler results, errors and annotations
    - [Claude Code Docs: Agent SDK MCP](https://code.claude.com/docs/en/agent-sdk/mcp): configuring servers, permissions, transports, status values, OAuth and timeouts
    - [Claude Code Docs: Agent SDK structured outputs](https://code.claude.com/docs/en/agent-sdk/structured-outputs): schema validation, draft-07 and failure subtypes
    - [Claude Code Docs: Modifying system prompts](https://code.claude.com/docs/en/agent-sdk/modifying-system-prompts): minimal default prompt, preset and CLAUDE.md loading
    - [Claude Code Docs: Claude Code features in the SDK](https://code.claude.com/docs/en/agent-sdk/claude-code-features): `settingSources` and agent teams
    - [Claude Code Docs: Streaming vs single mode](https://code.claude.com/docs/en/agent-sdk/streaming-vs-single-mode): streaming input as the preferred mode
    - [Claude Code Docs: Agent SDK cost tracking](https://code.claude.com/docs/en/agent-sdk/cost-tracking): client-side cost estimates, `usage` versus `modelUsage`, budgets on resume
    - [Claude Code Docs: Hosting the Agent SDK](https://code.claude.com/docs/en/agent-sdk/hosting): the subprocess model, persistence, resources and multi-tenant isolation
    - [Claude Code Docs: Secure deployment](https://code.claude.com/docs/en/agent-sdk/secure-deployment): credential-injecting proxies and least privilege
    - [Claude Code Docs: Agent SDK observability](https://code.claude.com/docs/en/agent-sdk/observability): OpenTelemetry traces, metrics and events
    - [Claude Code Docs: Glossary](https://code.claude.com/docs/en/glossary): agentic loop, harness, hooks, turn and non-interactive mode
    - [Claude Code Docs: How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works): Claude Code as the agentic harness
    - [Claude Code Docs: Features overview](https://code.claude.com/docs/en/features-overview): MCP versus Skills and the context cost of each extension
    - [Claude Code Docs: MCP](https://code.claude.com/docs/en/mcp): scopes, CLI commands, environment variable expansion, approvals, output limits, tool search, OAuth, WebSocket servers and client runtimes
    - [Claude Code Docs: MCP quickstart](https://code.claude.com/docs/en/mcp-quickstart): where local and user scopes are stored
    - [Claude Code Docs: Managed MCP](https://code.claude.com/docs/en/managed-mcp): `managed-mcp.json`, allowlists and denylists
    - [Claude Code Docs: Monitoring usage](https://code.claude.com/docs/en/monitoring-usage): OpenTelemetry export from Claude Code
    - [MCP: Specification overview, 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/index): hosts, clients, servers, primitives and security principles
    - [MCP: Architecture, 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/architecture/index): host responsibilities and design principles
    - [MCP: Base protocol, 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic/index): JSON-RPC rules, error codes, `resultType` and per-request metadata
    - [MCP: Versioning, 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning): legacy and modern eras
    - [MCP: Changelog, 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/changelog): statelessness, removed sessions, MRTR and deprecations
    - [MCP: Deprecated features](https://modelcontextprotocol.io/specification/2026-07-28/deprecated): Roots, Sampling and Logging timelines
    - [MCP: Transports, 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/index): standard transports and custom transports over sockets
    - [MCP: stdio transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio): framing, stdout rules and socket reuse
    - [MCP: Streamable HTTP transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http): endpoint, headers, Origin validation and history
    - [MCP: Authorization](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization/index): optional OAuth 2.1-based authorization and stdio credentials
    - [MCP: Authorization security considerations](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization/security-considerations): no token passthrough
    - [MCP: Server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover): `server/discover`
    - [MCP: Tools, 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/server/tools): tool definitions, names, results, `structuredContent`, protocol versus execution errors, handles
    - [MCP: Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources): URIs, templates and not-found errors
    - [MCP: Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts): user-controlled prompts and errors
    - [MCP: Pagination](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/pagination): invalid cursor errors
    - [MCP: Schema reference](https://modelcontextprotocol.io/specification/2026-07-28/schema): `isError` rationale and tool annotation defaults
    - [MCP: Elicitation](https://modelcontextprotocol.io/specification/2026-07-28/client/elicitation): form and URL modes
    - [MCP: Sampling](https://modelcontextprotocol.io/specification/2026-07-28/client/sampling): sampling and its deprecation
    - [MCP: Roots](https://modelcontextprotocol.io/specification/2026-07-28/client/roots): roots and their deprecation
    - [MCP: Specification overview, 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/index): the stateful, handshake-based revision
    - [MCP: Lifecycle, 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle): the `initialize` handshake
    - [MCP: Transports, 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports): sessions on Streamable HTTP
    - [MCP: Resources, 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources): the older not-found error code
    - [MCP: Changelog, 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/changelog): input validation errors as tool execution errors
    - [MCP: Architecture overview](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture): layers, one client per server, transports and a tool registry across servers
    - [MCP: Server concepts](https://modelcontextprotocol.io/docs/2026-07-28/learn/server-concepts): who controls tools, resources and prompts
    - [MCP: Client concepts](https://modelcontextprotocol.io/docs/2026-07-28/learn/client-concepts): roots as coordination, not security
    - [MCP: Versioning](https://modelcontextprotocol.io/docs/2026-07-28/learn/versioning): the current revision
    - [MCP: Build a server (2026-07-28)](https://modelcontextprotocol.io/docs/2026-07-28/develop/build-server): `MCPServer`, stdio and logging rules
    - [MCP: Build a server (2025-11-25)](https://modelcontextprotocol.io/docs/2025-11-25/develop/build-server): the older `FastMCP` and `@modelcontextprotocol/sdk` names
    - [MCP: Inspector](https://modelcontextprotocol.io/docs/2026-07-28/tools/inspector): testing servers
    - [MCP: Security best practices](https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices): token passthrough and local server risks
    - [MCP Python SDK README](https://raw.githubusercontent.com/modelcontextprotocol/python-sdk/main/README.md): the v2 Python server example
    - [MCP TypeScript SDK README](https://raw.githubusercontent.com/modelcontextprotocol/typescript-sdk/main/README.md): the v2 TypeScript server example
    - [Anthropic: Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol): MCP's open-sourcing on November 25, 2024
    - [Anthropic Engineering: Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): workflows versus agents, the augmented LLM and the workflow patterns
    - [Anthropic Engineering: How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system): production tracing for debugging agents
    - [Anthropic Engineering: Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): context engineering, context rot, compaction, just-in-time retrieval and bloated tool sets
    - [Anthropic Engineering: Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents): distinct tool purposes, consolidated tools and not wrapping every API endpoint
    - [Anthropic Engineering: Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval): chunking, BM25, hybrid retrieval and reranking
    - [Anthropic Engineering: Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents): evals, graders and A/B testing
    - [Anthropic Engineering: Equipping agents for the real world with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills): progressive disclosure in skills
    - [Anthropic Engineering: Advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use): when to use tool search
    - [Anthropic Engineering: Managed Agents](https://www.anthropic.com/engineering/managed-agents): the harness calling tools through one interface
    - [Anthropic: Building trusted AI in the enterprise (PDF)](https://www-cdn.anthropic.com/e5c9de22bc8884089970bd262ca0c8b952cb9136.pdf): prompts need version control, testing and documentation
    - [Claude blog: Building agents with the Claude Agent SDK](https://claude.com/blog/building-agents-with-the-claude-agent-sdk): the Claude Code SDK rename on September 29, 2025
    - [Claude blog: Harnessing Claude's intelligence](https://claude.com/blog/harnessing-claudes-intelligence): the agent harness and gating hard-to-reverse actions
    - [Claude blog: Building agents that reach production systems with MCP](https://claude.com/blog/building-agents-that-reach-production-systems-with-mcp): direct API calls, CLIs and MCP compared
    - [Claude blog: Best practices for prompt engineering](https://claude.com/blog/best-practices-for-prompt-engineering): when examples help and starting without them
    - [Claude docs: Skills overview](https://claude.com/docs/skills/overview): what skills are in the Claude apps
    - [Claude docs: How to create skills](https://claude.com/docs/skills/how-to): SKILL.md structure
    - [Claude Help Center: What are projects?](https://support.claude.com/en/articles/9517075-what-are-projects): projects, project knowledge and instructions
    - [Claude Help Center: How can I create and manage projects?](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects): project instructions
    - [Claude Help Center: What are artifacts and how do I use them?](https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them): artifacts and the code execution requirement
    - [Claude Help Center: Use Research on Claude](https://support.claude.com/en/articles/11088861-use-research-on-claude): how Research works
    - [Claude Help Center: When should I use web search, extended thinking and Research?](https://support.claude.com/en/articles/11095361-when-should-i-use-web-search-extended-thinking-and-research): Research for five or more tool calls
    - [Claude Help Center: Use connectors to extend Claude's capabilities](https://support.claude.com/en/articles/11176164-use-connectors-to-extend-claude-s-capabilities): connectors and inherited permissions
    - [Claude Help Center: Get started with custom connectors using remote MCP](https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp): MCP as an open standard created by Anthropic
    - [Claude Help Center: Use chat search and memory](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context): memory topics and project memory
    - [Claude Help Center: Create and edit files with Claude](https://support.claude.com/en/articles/12111783-create-and-edit-files-with-claude): code execution and file creation
    - [Claude Help Center: What are skills?](https://support.claude.com/en/articles/12512176-what-are-skills): skills and progressive disclosure in the apps
    - [Claude Help Center: Understanding Claude's personalization features](https://support.claude.com/en/articles/10185728-understanding-claude-s-personalization-features): account-wide instructions
    - [Claude Help Center: Set organization instructions](https://support.claude.com/en/articles/14546867-set-organization-instructions): PII-handling instruction example
    - [Claude Help Center: How do usage and length limits work?](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work): starting a new conversation or using projects at a length limit
    - [Anthropic Academy: Claude 101, getting better results](https://academy.claude.com/courses/claude-101/getting-better-results): knowing when to start over
    - [Anthropic Academy: AI Fluency, effective prompting techniques](https://academy.claude.com/courses/ai-fluency-framework-foundations/effective-prompting-techniques): breaking complex tasks into steps
    - [Anthropic Academy: AI Capabilities and Limitations, next token prediction](https://academy.claude.com/courses/ai-capabilities-and-limitations/next-token-prediction): where fabrication clusters
    - [Anthropic Academy: AI Capabilities and Limitations, when properties collide](https://academy.claude.com/courses/ai-capabilities-and-limitations/when-properties-collide): drift in long conversations
    - [Anthropic Academy: Parametric memory and context](https://academy.claude.com/tutorials/parametric-memory-and-context): what to persist between conversations
    - [Anthropic Academy: Why does bias exist in AI models?](https://academy.claude.com/tutorials/why-does-bias-exist-in-ai-models): bias definition and origin
    - [Anthropic Academy: AI-native SDLC playbook, CLAUDE.md](https://academy.claude.com/courses/ai-native-sdlc-playbook/claude-md): CLAUDE.md reviewed like code
    - [Anthropic Academy: Claude with the Anthropic API](https://anthropic.skilljar.com/claude-with-the-anthropic-api): content boundaries with XML tags
    - [Anthropic prompt engineering interactive tutorial, chapter 7](https://github.com/anthropics/prompt-eng-interactive-tutorial/blob/master/Anthropic%201P/07_Using_Examples_Few-Shot_Prompting.ipynb): zero-shot, one-shot and few-shot defined by the number of examples
    - [Claude and Accenture: Deploying AI from pilot to production (PDF)](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf): four oversight tiers and compliance built into the architecture
    - [A2A project README](https://raw.githubusercontent.com/a2aproject/A2A/main/README.md): A2A as an open standard and Linux Foundation project
    - [A2A specification](https://raw.githubusercontent.com/a2aproject/A2A/main/docs/specification.md): agents exchanging information without sharing internal state
    - [A2A and MCP](https://raw.githubusercontent.com/a2aproject/A2A/main/docs/topics/a2a-and-mcp.md): A2A between agents with MCP inside each agent
    - [AWS: Introducing Strands Agents](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/): Strands as an open-source, model-driven agent SDK
    - [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): LangGraph as a low-level orchestration framework for stateful agents
    - [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/graph-api): state, nodes and edges
    - [Pydantic AI: Agents](https://pydantic.dev/docs/ai/core-concepts/agent/index.md): what a Pydantic AI agent contains
    - [IETF RFC 6455: The WebSocket Protocol](https://www.rfc-editor.org/rfc/rfc6455.html): the Upgrade handshake and full-duplex messaging
