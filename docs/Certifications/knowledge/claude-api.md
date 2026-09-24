---
title: "Claude API Essentials for the Claude Certifications"
description: Messages API anatomy, the agent loop, streaming, errors, models, thinking, tokens, vision, caching, batches and cost for the four Claude certification exams.
last_reviewed: 2026-09-23
---

# Claude API Essentials

This page teaches the Claude API once, for all four certification exams: how a request and a response are built, how an agent loop reads `stop_reason`, how streaming, errors and rate limits behave, which model and thinking settings to pick, and how tokens, context windows, images and PDFs are counted, followed by caching, batches, structured outputs, files and citations, cost tracking, cloud platforms and model migration. Each section opens with the exam objectives it serves. Product facts are as of September 2026, and where the documentation has moved since the July 2026 exam guides, the section shows both versions and which wording to expect on the exam (the guide's).

!!! tip "Sitting CCAR-F only?"

    The CCAR-F guide's appendix names the agentic loop and `stop_reason` handling, `max_tokens`, system prompts, `tool_choice`, the Message Batches API and structured output via `tool_use` among the technologies and in-scope topics the exam tests. The same appendix says these related topics will not appear: API authentication, billing and account management; streaming and server-sent events; rate limiting, quotas and pricing calculations; vision; token counting algorithms and tokenization specifics; prompt caching implementation details beyond knowing it exists; and specific cloud provider configurations. Study [Stop reasons and the agent loop](#stop-reasons-and-the-agent-loop), [Message Batches](#message-batches) and [Structured outputs](#structured-outputs) closely, and read the other sections for context.

## How a Messages API call works

*Tested in: CCDV-F 2.3 Claude API Mechanics, 2.4 Software Engineering Foundations, 5.2 Technical Fundamentals · CCAR-F 1.1 (1.1-K2), 5.1 (5.1-K4), Appendix (max_tokens, system prompts); API authentication is out of scope for CCAR-F*

The Claude API is a REST API at `https://api.anthropic.com`, and its core is one endpoint: `POST /v1/messages`. You send a model ID, a `max_tokens` ceiling and the conversation so far. Claude returns one assistant message made of typed content blocks, plus a `stop_reason` and a `usage` report. The API keeps no conversation state between calls, so the history you send is the only context Claude has.

### Endpoint and headers

| Header | Value | Required |
|---|---|---|
| `Authorization` | `Bearer <token>`: your API key, or a short-lived access token from Workload Identity Federation | Yes, unless `x-api-key` is set |
| `x-api-key` | Your API key; the docs call it a legacy fallback for `Authorization` that is still supported | No |
| `anthropic-version` | The API version, for example `2023-06-01` | Yes |
| `content-type` | `application/json` | Yes |
| `anthropic-workspace-id` | The ID of the workspace the request runs in (IDs start `wrkspc_`) | Required with a multi-workspace API key, optional for other API keys, not used with Workload Identity Federation tokens (they select a workspace at token exchange) |
| `anthropic-beta` | Beta feature names, comma-separated, usually `feature-name-YYYY-MM-DD` | Only for beta features; an invalid name returns 400 |

The official client SDKs (Python, TypeScript, C#, Go, Java, PHP and Ruby; Anthropic also ships the `ant` command-line tool) send the authentication, `anthropic-version` and `content-type` headers for you and read the key from `ANTHROPIC_API_KEY`; you pass `anthropic-workspace-id` yourself when your key needs it. They also add type-safe request and response handling, built-in retries and error handling, streaming support and request timeouts. The same headers serve the neighboring endpoints: `POST /v1/messages/count_tokens`, `POST /v1/messages/batches` and `GET /v1/models`. Every response carries a `request-id` header; include it when you contact support.

The raw call, as the [Messages guide](https://platform.claude.com/docs/en/build-with-claude/working-with-messages) shows it. Its samples authenticate with `x-api-key`, which the API overview calls a still-supported legacy fallback for `Authorization: Bearer`; either header works.

```bash
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-opus-5-5",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Hello, Claude"}
    ]
  }'
```

### A complete request

An illustrative request body that adds the optional fields explained in the table below: a system prompt, a multi-turn history (the conversation is the Messages guide's own example, and its assistant turn can be written by you), a stop sequence and an opaque end-user ID.

```json
{
  "model": "claude-opus-5-5",
  "max_tokens": 1024,
  "system": "You are a concise tutor for a data science study group.",
  "messages": [
    {"role": "user", "content": "Hello, Claude"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "Can you describe LLMs to me?"}
  ],
  "stop_sequences": ["###"],
  "metadata": {"user_id": "5f1c2a9e-7d4b-4c1e-9a53-2b8e61f0c7d4"}
}
```

| Field | Required | What it does | What to know |
|---|---|---|---|
| `model` | Yes | The model to run | Every model ID is a pinned snapshot, dateless IDs included; only the convenience aliases for models before the 4.6 generation (such as `claude-sonnet-4-5`) move to a newer snapshot. See [Models and how to choose one](#models-and-how-to-choose-one) |
| `max_tokens` | Yes | Absolute maximum number of tokens to generate | The model may stop before it. Thinking counts toward it. `0` populates the prompt cache without generating. Reaching it ends the response with `stop_reason: "max_tokens"` |
| `messages` | Yes | The conversation so far | Alternating `user` and `assistant` turns; consecutive same-role turns are combined; at most 100,000 messages; `content` is a string or an array of content blocks |
| `system` | No | Instructions for the conversation | A string or an array of text blocks (each can carry `cache_control`) |
| `stop_sequences` | No | Custom strings that stop generation | A match sets `stop_reason: "stop_sequence"` and puts the matched string in `stop_sequence` |
| `stream` | No | Stream the response as server-sent events | See [Streaming](#streaming) |
| `tools`, `tool_choice` | No | Tools Claude may call, and how it must choose | `tool_choice` takes `auto` (the default when tools are provided), `any`, `tool` or `none` (the default when no tools are provided). `any` and `tool` return a 400 on Claude Opus 5.5, Fable 5.1 and Mythos 5.1, and on any model they are incompatible with manual extended thinking (`thinking.type: "enabled"`); see [Controlling tool choice](tool-use-and-mcp.md#controlling-tool-choice) |
| `thinking` | No | `adaptive`, `enabled` with `budget_tokens`, or `disabled` | Which values a model accepts varies; see [Extended thinking, adaptive thinking and effort](#extended-thinking-adaptive-thinking-and-effort) |
| `output_config` | No | `effort` (`low`, `medium`, `high`, `xhigh`, `max`) and `format` (a JSON schema) | For `format`, see [Structured outputs](#structured-outputs) |
| `metadata.user_id` | No | An opaque end-user ID Anthropic may use to detect abuse | A UUID or hash, never a name, email address or phone number; maximum length 512 |
| `service_tier` | No | `"auto"` or `"standard_only"` | Controls use of Priority Tier capacity; see [Errors, retries and rate limits](#errors-retries-and-rate-limits) |
| `inference_geo` | No | Region for inference | Defaults to the workspace's `default_inference_geo` |
| `temperature`, `top_p`, `top_k` | No | Sampling controls | Deprecated; non-default values return 400 on Claude 4.7 and later models and Claude Mythos Preview; see [sampling](#next-token-generation-sampling-and-non-determinism) |

### Roles and the conversation

- **The API is stateless.** In the words of the [Messages guide](https://platform.claude.com/docs/en/build-with-claude/working-with-messages): "The Messages API is stateless, which means that you always send the full conversational history to the API." Earlier assistant turns do not have to come from Claude; synthetic ones are allowed. The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) tests the same point as "The importance of passing complete conversation history in subsequent API requests to maintain conversational coherence".
- **Turns alternate.** Models are trained on alternating `user` and `assistant` turns. Two consecutive turns with the same role are combined into one turn, not rejected.
- **The system prompt is a top-level field.** There is no `system` role at the start of `messages`. On Claude Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Opus 5.5, Opus 4.8 and Opus 5 you may add a `"role": "system"` message after a user turn to give new instructions partway through. It has the same authority as the top-level `system` field and does not invalidate the cached prefix before it, but it cannot be the first entry, and Claude Sonnet 5 does not support it.
- **Untrusted text never goes in a system message.** Raw tool output, retrieved documents and web content placed there gain operator-level authority. Keep them inside `tool_result` blocks.
- **Tools live in the two ordinary roles.** Tool calls are `tool_use` blocks in assistant messages and results are `tool_result` blocks in user messages; the Claude API has no `tool` or `function` role.
- **Prefill is gone on new models.** Ending `messages` with an assistant turn used to make Claude continue from that text. On Claude 4.6 and later models (and Claude Mythos Preview) it returns a 400 whose message reads "This model does not support assistant message prefill. The conversation must end with a user message." ([Claude API errors](https://platform.claude.com/docs/en/api/errors)) The thinking guide adds that you cannot prefill while thinking is on. Use [structured outputs](#structured-outputs) or system prompt instructions instead.

### Content blocks

A message's `content` is a string (shorthand for one `text` block) or a list of typed blocks. You send `text`, `image` and `document` blocks (a PDF, for example), and `tool_result` blocks answering Claude's tool calls. When you replay Claude's earlier turns, its `tool_use` blocks go back as they came, and its `thinking` blocks must go back complete and unmodified within a tool-use turn (recommended across turns; outside tool use you may omit them). A response can contain `text`, `thinking`, `redacted_thinking`, `tool_use`, `server_tool_use` and server tool result blocks.

Read blocks by their `type`, never by position. On models with thinking on by default, a response can begin with a `thinking` block, so code that reads `content[0].text` breaks. The migration guides call this out as a breaking change.

### The response

An illustrative response from Claude Opus 5.5: the ID, text and token counts are examples, while the field names and shapes follow the API reference. Thinking is always on for that model, and adaptive thinking decides on each request whether to think. When it thinks, the thinking text is omitted by default (`display: "omitted"`), so the reasoning arrives as a `thinking` block with an empty `thinking` field and a `signature` that carries the encrypted full reasoning. For a simple request it can skip thinking, and then no `thinking` block is returned.

```json
{
  "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
  "type": "message",
  "role": "assistant",
  "model": "claude-opus-5-5",
  "content": [
    {"type": "thinking", "thinking": "", "signature": "EosnCkYICxIMMb3LzNrMu..."},
    {"type": "text", "text": "Sure, I'd be happy to provide..."}
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "stop_details": null,
  "usage": {
    "input_tokens": 30,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "output_tokens": 309,
    "output_tokens_details": {"thinking_tokens": 112},
    "service_tier": "standard",
    "inference_geo": "global"
  }
}
```

`stop_reason` is always set in a non-streaming response; the next section covers every value. `stop_details` is `null` unless the stop reason is `refusal`.

| `usage` field | Meaning |
|---|---|
| `input_tokens` | Input tokens neither read from nor written to the cache, that is, the tokens after the last cache breakpoint (all of the input only when `cache_creation_input_tokens` and `cache_read_input_tokens` are both 0) |
| `cache_creation_input_tokens` | Input tokens written to the prompt cache by this request |
| `cache_read_input_tokens` | Input tokens read from the prompt cache |
| `output_tokens` | Billed output tokens, thinking included; non-zero even for an empty reply |
| `output_tokens_details.thinking_tokens` | The part of `output_tokens` spent on internal reasoning (the raw reasoning, even when the returned thinking text is summarized or omitted); always no more than `output_tokens` |
| `service_tier` | `"standard"`, `"priority"`, `"batch"` or null |
| `cache_creation`, `server_tool_use`, `inference_geo` | Cache writes split by 5-minute and 1-hour TTL, web search and web fetch request counts, and where inference ran |

Total input for a request is `input_tokens + cache_creation_input_tokens + cache_read_input_tokens`. Token counts do not map one-to-one onto the text you can see. Tracking these fields across many requests is covered in [Cost and usage tracking](#cost-and-usage-tracking).

### The same call through the SDKs

An illustrative version of the request in the Python and TypeScript SDKs, assembled from the documentation's examples. Both clients read the key from `ANTHROPIC_API_KEY`, and both expose the `request-id` header as `_request_id` on the response object.

=== "Python"

    ```python
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    message = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=1024,
        system="You are a concise tutor for a data science study group.",
        messages=[{"role": "user", "content": "Hello, Claude"}],
    )

    for block in message.content:
        if block.type == "text":
            print(block.text)

    print(message.stop_reason, message.usage.output_tokens)
    print(message._request_id)  # from the request-id header
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";

    const client = new Anthropic(); // reads ANTHROPIC_API_KEY

    const message = await client.messages.create({
      model: "claude-opus-5-5",
      max_tokens: 1024,
      system: "You are a concise tutor for a data science study group.",
      messages: [{ role: "user", content: "Hello, Claude" }]
    });

    const textBlock = message.content.find(
      (block): block is Anthropic.TextBlock => block.type === "text"
    );
    console.log(textBlock?.text);

    console.log(message.stop_reason, message.usage.output_tokens);
    console.log(message._request_id); // from the request-id header
    ```

### Synchronous and asynchronous clients

This is where the engineering skills the CCDV-F guide lists (REST APIs, JSON, asynchronous programming, integrating with SDKs that wrap REST APIs) meet the Claude API. The Python SDK supports both synchronous and asynchronous operation: `AsyncAnthropic` has the same methods as `Anthropic`, awaited inside an `async def`, and installing `anthropic[aiohttp]` and passing `http_client=DefaultAioHttpClient()` swaps the default HTTP backend for aiohttp to improve async performance. In the TypeScript SDK every call returns a promise, and streams and paginated lists are consumed with `for await ... of`.

```python
import asyncio
from anthropic import AsyncAnthropic

client = AsyncAnthropic()  # reads ANTHROPIC_API_KEY

async def main() -> None:
    message = await client.messages.create(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello, Claude"}],
        model="claude-opus-5-5",
    )
    print(message.content)

asyncio.run(main())
```

Both clients also retry and time out by default; the defaults, the error classes and the retry strategy are in [What the SDKs already do](#what-the-sdks-already-do). Firing many requests at once needs a cap on how many are in flight, which [Integration patterns](solution-architecture.md#integration-patterns) shows with a semaphore.

### Decide

- If Claude "forgets" what was said two turns ago, check that every request carries the full history; not the model or its settings, because the API stores nothing between calls.
- If an instruction must hold from the first turn, put it in `system`. If it only becomes relevant later and the model supports it, a mid-conversation system message adds it without breaking the cache. Retrieved or tool-returned text goes in `tool_result` blocks, not in any system message.
- If you need a fixed output shape on Claude 4.6 or later, choose structured outputs or explicit instructions; not an assistant prefill, which returns a 400.

### Traps

- **Reading `content[0].text`.** The first block can be `thinking`. Filter by `type`.
- **Treating `max_tokens` as a target length.** It is only a ceiling, and it includes thinking tokens, so a small value can truncate a reply that thinks first.
- **Reading `input_tokens` as the whole prompt.** With caching it counts only the tokens after the last cache breakpoint.
- **Sending a system role as the first message.** Instructions that apply from the start belong in the top-level `system` field.

## Stop reasons and the agent loop

*Tested in: CCAR-F 1.1 (all six bullets), Exercise 1 step 2, Appendix (Claude API stop_reason values, tool_choice, max_tokens; Agent SDK stop_reason handling; agentic loop implementation) · CCDV-F 1.2 Agent Construction with Claude, 1.3 Agent Patterns and Frameworks, 2.3 Claude API Mechanics, 8.1 Tool Implementation*

Every successful response carries `stop_reason`, the API's structured record of why Claude stopped generating. In a non-streaming response it is never null; in a stream it is null in `message_start` and arrives on `message_delta`. A stop reason is not an error. Errors are failed requests with 4xx or 5xx statuses; `stop_reason` describes a request that succeeded. An agent loop is code that reads this one field and decides whether to call the API again.

### Every stop reason and how to handle it

The API defines seven values. The compaction betas add an eighth, `compaction`, which the stop reasons page does not list: an on-demand compaction request that produces its summary ends with it, and threshold compaction does when `pause_after_compaction` is on (see [Compaction, context editing and memory](context-engineering.md#compaction-context-editing-and-memory)).

| `stop_reason` | What happened | What your code does |
|---|---|---|
| `end_turn` | Claude finished its response naturally. The most common value. | Use the response. Read the `text` blocks by type. |
| `max_tokens` | The response reached the `max_tokens` you set. | Treat the output as truncated. Raise `max_tokens` or continue the response. If the cut-off block is an incomplete `tool_use`, retry with a higher `max_tokens` to get the full call. |
| `stop_sequence` | Claude emitted one of your `stop_sequences`. | Read the `stop_sequence` field to see which one fired. |
| `tool_use` | Claude is calling a client tool and expects you to run it. | Execute each `tool_use` block and send back one `tool_result` per call, then call the API again. |
| `pause_turn` | A server tool loop (for example web search) reached its iteration limit, 10 per request by default. | Send the assistant content back as-is and call again so Claude can finish. |
| `refusal` | Claude declined. Safety classifiers return this as a normal HTTP 200 response, not an error. | Read `stop_details` (`type`, `category`, `explanation`). Reset or rephrase the context, or retry on a fallback model. |
| `model_context_window_exceeded` | The response filled the model's context window. | Treat it as truncated, like `max_tokens`. |

Details behind the table that decide real integration bugs:

- **An empty `end_turn`.** Claude sometimes returns 2 to 3 tokens with no content and `stop_reason: "end_turn"`. The docs list two common causes: text blocks added immediately after `tool_result` blocks (Claude learns to expect user input after every tool use), and sending Claude's completed response back without adding anything (Claude has already decided it is done). Sending the empty response back unchanged does not help; fix the message structure, and as a last resort add a continuation prompt in a new user message.
- **`pause_turn` is only for server tools.** A response that leaves a client `tool_use` block waiting on you never has `stop_reason: "pause_turn"`: when Claude stops to call your tools, the value is `tool_use`, and you continue by sending `tool_result` blocks, not the response itself. The docs' own `pause_turn` handler stops after `max_continuations=5` and returns the last response.
- **Refusals.** `stop_details.category` is one of `cyber`, `bio`, `frontier_llm`, `reasoning_extraction` or `general_harms`, and both `category` and `explanation` are null when the refusal maps to no named category. A refusal that arrives before any output is not billed, though it counts against rate limits. Continuing the same conversation without resetting it produces more refusals. Server-side fallback (`fallbacks: "default"` with the `server-side-fallback-2026-07-01` beta header) can retry a classifier decline on another model; only a safety-classifier decline triggers it, so rate limits, overloads and server errors are returned as-is. It is in beta on the Claude API, is not available on Amazon Bedrock, Google Cloud or Microsoft Foundry, and is not supported on the Message Batches API.
- **`model_context_window_exceeded`** lets you request the maximum output without knowing the input size. Sonnet 4.5 and newer return it without a beta header; earlier models need `model-context-window-exceeded-2025-08-26`, and the SDKs type the value only in their beta namespace.

### The agent loop the exam tests

The CCAR-F guide describes the lifecycle as "sending requests to Claude, inspecting stop_reason ("tool_use" vs "end_turn"), executing requested tools, and returning results for the next iteration" ([CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf), 1.1-K1). Step by step:

1. Send `messages` with your `tools` (and `system`, `max_tokens`).
2. Read `stop_reason`. On `"tool_use"`, the response holds one or more `tool_use` blocks, each with an `id`, a `name` and an `input` that matches the tool's schema.
3. Run the matching tool in your code. Claude never executes anything itself: it emits a structured request, and your code (or Anthropic's servers, for server tools) runs it.
4. Append the assistant response exactly as received, thinking blocks included. Then append one user message holding one `tool_result` per `tool_use`, matched by `tool_use_id`. Keep that message to the results: text placed before a `tool_result` causes a 400, and the stop reasons guide says never to add text blocks immediately after tool results.
5. Call the API again with the whole conversation, and repeat while `stop_reason` is `"tool_use"`.
6. If you use server tools, `"pause_turn"` also means call again: append the assistant content as it is (replacing a paused assistant turn already at the end) and send the request.
7. Any other stop reason ends the loop. `"end_turn"` means Claude has finished.

This is model-driven control: Claude chooses the next tool from the conversation so far and each tool's description, instead of following a decision tree you wrote (1.1-K3). When a step must always happen in a fixed order, enforce it in code: the rationale to sample question 1 in the [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says programmatic enforcement "provides deterministic guarantees that prompt-based approaches cannot" when a tool sequence is required for critical business logic. See [Workflows or agents](agents-and-agent-sdk.md#workflows-or-agents).

### A correct loop in Python and TypeScript

This loop is minimally adapted from two documentation examples: the error-handling agent in [Build a tool-using agent](https://platform.claude.com/docs/en/agents-and-tools/tool-use/build-a-tool-using-agent) and the `pause_turn` guidance in [Stop reasons and fallback](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons). `run_tool` is your dispatcher: it runs the tool named `name` and raises on failure.

=== "Python"

    ```python
    import json
    import anthropic

    client = anthropic.Anthropic()

    def run_agent(user_query, tools, run_tool, model="claude-opus-5-5"):
        messages = [{"role": "user", "content": user_query}]

        while True:
            response = client.messages.create(
                model=model, max_tokens=4096, tools=tools, messages=messages
            )

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        try:
                            result = run_tool(block.name, block.input)
                            tool_results.append(
                                {"type": "tool_result", "tool_use_id": block.id,
                                 "content": json.dumps(result)}
                            )
                        except Exception as exc:
                            # Signal failure so Claude can retry or ask for clarification.
                            tool_results.append(
                                {"type": "tool_result", "tool_use_id": block.id,
                                 "content": str(exc), "is_error": True}
                            )
                # The whole assistant turn goes back unchanged, then only the results.
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
                continue

            if response.stop_reason == "pause_turn":
                # A server tool loop paused: send the response back and let Claude continue.
                # Replace a trailing paused turn instead of stacking a second assistant turn.
                if messages[-1]["role"] == "assistant":
                    messages[-1] = {"role": "assistant", "content": response.content}
                else:
                    messages.append({"role": "assistant", "content": response.content})
                continue

            # end_turn, max_tokens, stop_sequence, refusal, model_context_window_exceeded
            return response
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";

    const client = new Anthropic();

    type RunTool = (name: string, input: Record<string, unknown>) => unknown;

    async function runAgent(
      userQuery: string,
      tools: Anthropic.ToolUnion[],
      runTool: RunTool,
      model = "claude-opus-5-5"
    ): Promise<Anthropic.Message> {
      const messages: Anthropic.MessageParam[] = [{ role: "user", content: userQuery }];

      while (true) {
        const response = await client.messages.create({
          model,
          max_tokens: 4096,
          tools,
          messages
        });

        if (response.stop_reason === "tool_use") {
          const toolResults: Anthropic.ToolResultBlockParam[] = [];
          for (const block of response.content) {
            if (block.type === "tool_use") {
              try {
                const result = runTool(block.name, block.input as Record<string, unknown>);
                toolResults.push({
                  type: "tool_result",
                  tool_use_id: block.id,
                  content: JSON.stringify(result)
                });
              } catch (err) {
                // Signal failure so Claude can retry or ask for clarification.
                toolResults.push({
                  type: "tool_result",
                  tool_use_id: block.id,
                  content: String(err),
                  is_error: true
                });
              }
            }
          }
          // The whole assistant turn goes back unchanged, then only the results.
          messages.push({ role: "assistant", content: response.content });
          messages.push({ role: "user", content: toolResults });
          continue;
        }

        if (response.stop_reason === "pause_turn") {
          // A server tool loop paused: send the response back and let Claude continue.
          // Replace a trailing paused turn instead of stacking a second assistant turn.
          if (messages[messages.length - 1].role === "assistant") {
            messages[messages.length - 1] = { role: "assistant", content: response.content };
          } else {
            messages.push({ role: "assistant", content: response.content });
          }
          continue;
        }

        // end_turn, max_tokens, stop_sequence, refusal, model_context_window_exceeded
        return response;
      }
    }
    ```

Why each line is there:

- **The assistant turn goes back whole.** Within a tool-use turn you must pass thinking blocks back complete and unmodified. Filtering on `block.type == "thinking"` alone drops `redacted_thinking` blocks and breaks the protocol, and an edited thinking block in the latest assistant message returns a 400.
- **A failed tool still gets a result.** Put the error text in `content` and set `is_error: true`, so Claude can retry with corrected input, ask the user, or explain the limitation. Skipping the result fails with the error `tool_use ids were found without tool_result blocks immediately after`.
- **`pause_turn` gets its own branch.** The docs say to handle it in any agent loop that uses server tools, by adding the assistant's response to the messages and making another request. When two pauses arrive in a row, the code replaces the paused assistant turn rather than appending a second one, as the docs' multi-continuation handler does "to maintain alternating roles". A loop that treats it as final returns an unfinished answer. The request then ends with an assistant turn; the docs' own `pause_turn` example sends exactly that to `claude-opus-5-5`, even though a hand-written assistant prefill returns a 400 on Claude 4.6 and later models.

Outside a loop, or in any handler that receives a response, the [stop reasons guide](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons) recommends making a habit of checking `stop_reason`, with this pattern (the `handle_*` functions are yours to write). Its TypeScript version types the switch against `Anthropic.Beta.BetaMessage`, because the SDKs type `model_context_window_exceeded` only in their beta namespace.

```python
def handle_response(response):
    match response.stop_reason:
        case "tool_use":
            return handle_tool_use(response)
        case "max_tokens":
            return handle_truncation(response)
        case "model_context_window_exceeded":
            return handle_context_limit(response)
        case "pause_turn":
            return handle_pause(response)
        case "refusal":
            return handle_refusal(response)
        case _:
            return next(
                (block.text for block in response.content if block.type == "text"),
                "",
            )
```

You do not always have to write the loop yourself. For most tool use implementations the docs recommend the SDK tool runner (in beta), which handles tool execution, result formatting and conversation management; they point to the manual loop when you need human-in-the-loop approval, custom logging or conditional execution. The Claude Agent SDK runs the same loop until Claude produces a response with no tool calls; see [The agentic loop](agents-and-agent-sdk.md#the-agentic-loop). CCAR-F tests the loop itself (its Exercise 1 asks you to implement one that checks `stop_reason`), so know the manual version.

### The three anti-patterns CCAR-F 1.1 names

The guide asks for "Avoiding anti-patterns such as parsing natural language signals to determine loop termination, setting arbitrary iteration caps as the primary stopping mechanism, or checking for assistant text content as a completion indicator" ([CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf), 1.1-S3).

| Anti-pattern | Why it fails | Do instead |
|---|---|---|
| Parsing natural language to decide the loop is over ("done", "task complete") | Prose is not a contract. [How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works) puts it bluntly: "if you're writing a regex to extract a decision from model output, that decision should have been a tool call." | Loop on `stop_reason`, the structured field the API sets for exactly this purpose. |
| An arbitrary iteration cap as the primary stopping mechanism | It cuts off long legitimate tasks and says nothing about whether the work is finished. | Stop on `stop_reason`. A turn or budget cap is still sensible as a backstop: [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) says stopping conditions such as a maximum number of iterations are common for control, and the [Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/agent-loop) (options `max_turns` and `max_budget_usd`) call a budget "a good default for production agents". |
| Treating assistant text as the completion signal | Claude often writes explanatory text before a `tool_use` block in the same response, and an `end_turn` reply can come back empty. Neither the presence nor the absence of text tells you the state. | Read `stop_reason`, then read blocks by `type`. |

!!! warning "Exam guide vs current docs: which stop reasons end the loop"

    The guide frames loop control as two values: continue on `"tool_use"`, terminate on `"end_turn"`. The [tool use docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works) (as of September 2026) say the loop continues while `stop_reason` is `tool_use`, and "The loop exits on any other stop reason (`"end_turn"`, `"max_tokens"`, `"stop_sequence"`, or `"refusal"`)", with `pause_turn` added for server tools. On the exam, answer in the guide's terms: `tool_use` means continue, `end_turn` means done. In code, handle every value in the table above.

### Decide

- If an item asks how an agentic loop decides whether to continue, choose the option that continues while `stop_reason` is `tool_use` and stops on `end_turn`; not a check for assistant text, a match on completion phrases or an iteration count, because those are the three anti-patterns task 1.1 names.
- If an item offers an iteration cap, keep it only as a safety backstop behind `stop_reason`; not as the primary stopping mechanism.
- If a tool call fails, choose returning a `tool_result` with `is_error: true` so Claude can recover; not dropping the result or ending the loop.
- If a required step (for example, identity verification before a refund) must always run first, choose programmatic enforcement over prompt instructions, as the rationale to CCAR-F sample question 1 does.

### Traps beyond the guide's three

- **Dropping the assistant turn.** Appending only the tool results leaves Claude without its own call in the history; each request must contain the original messages, the assistant response and the results.
- **Text before or between results.** In the results message, `tool_result` blocks come first; text before them causes a 400, and text right after them invites empty `end_turn` replies.
- **Editing history on the newest models.** On Claude Fable 5.1 and Opus 5.5, changing the system prompt, the tools or an earlier message invalidates every replayed thinking block that comes after the change, which can return a 400. Keep the history append-only; the rules are in [Thinking inside a tool-use loop](#thinking-inside-a-tool-use-loop).
- **Handling a refusal as an exception.** It arrives as HTTP 200 with `stop_reason: "refusal"`; an error handler never sees it.
- **Forcing a tool call on the newest models.** `tool_choice` of `any` or `tool` returns a 400 on Claude Opus 5.5, Fable 5.1 and Mythos 5.1. The CCAR-F guide still teaches `any` and forced tool selection as the way to guarantee a tool call, so answer exam items in the guide's terms and use the docs' alternatives (strict tool use or structured outputs) in code; see [Controlling tool choice](tool-use-and-mcp.md#controlling-tool-choice).

## Streaming

*Tested in: CCDV-F 2.3 Claude API Mechanics (streaming), 5.2 Technical Fundamentals (websockets), Section 2 applied area (streaming, error handling, multi-format input) · CCAR-P 3.3, 4.5 (latency trade-offs) · Out of scope for CCAR-F (Appendix)*

Set `"stream": true` and the Messages API sends the response incrementally as server-sent events (SSE) instead of one JSON body at the end. Users see output as it is generated, which improves the perceived responsiveness of an application. The latency metric the docs single out for streaming is time to first token (TTFT): the time the model takes to generate the first token of the response, measured from when the prompt was sent.

### When to stream

| Situation | Choice | Why |
|---|---|---|
| A person is watching the answer appear (chat, IDE, support widget) | Stream | Output appears in real time, so the app feels faster even when total time is the same |
| A request may run long, especially over 10 minutes | Stream, or use [Message Batches](#message-batches) if nobody is waiting | Some networks drop idle connections, so a long non-streaming request can fail without a response |
| `max_tokens` above 21,333 through an SDK | Stream (or stream and collect the final message) | The SDKs require streaming above that value; it is a client-side check, not an API rule |
| Short back-end call whose result you only use whole | Either; a plain call is simpler | Nothing is shown to a user until the result is complete |

Both SDKs also refuse a non-streaming request they expect to take longer than about 10 minutes, and streaming is one way past that check; the rule and the timeout defaults are in [What the SDKs already do](#what-the-sdks-already-do).

### The event flow

Each event has an SSE event name (for example `event: message_stop`) and JSON data carrying the matching `type`. A stream follows this order:

1. `message_start`: a `Message` object with empty `content`; its `stop_reason` is null.
2. For each content block: `content_block_start`, one or more `content_block_delta` events, then `content_block_stop`. Each block's `index` is its position in the final `content` array. One documented exception: during server-side fallback, a `fallback` block arrives at each model boundary as a `content_block_start` and `content_block_stop` pair with no deltas in between.
3. One or more `message_delta` events with top-level changes, including `stop_reason` and `usage`. The token counts in its `usage` are cumulative, not per event.
4. `message_stop`.

Any number of `ping` events can appear anywhere. The API can also send an `error` event mid-stream. Under the versioning policy, new event types may be added, so your parser should handle unknown types gracefully (skip them) instead of failing on them. An abridged raw stream:

```text
event: message_start
data: {"type": "message_start", "message": {"id": "msg_...", "type": "message", "role": "assistant", "content": [], "model": "claude-opus-5-5", "stop_reason": null, "stop_sequence": null, "usage": {"input_tokens": 25, "output_tokens": 1}}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}

event: ping
data: {"type": "ping"}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence":null}, "usage": {"output_tokens": 15}}

event: message_stop
data: {"type": "message_stop"}
```

### Delta types

| Delta | Carries | Handling rule |
|---|---|---|
| `text_delta` | A fragment of a `text` block | Append to the text at that `index` |
| `input_json_delta` | A fragment of a `tool_use` block's `input`, as a partial JSON string in `partial_json` | Accumulate the strings and parse once at `content_block_stop`; the final `input` is always an object. Current models emit one complete key and value at a time, so expect pauses between tool events |
| `thinking_delta` | Thinking text | With `display: "omitted"`, a thinking block streams one `thinking_delta` with an empty string, then a single `signature_delta` |
| `signature_delta` | The thinking block's signature | Sent just before `content_block_stop` for each thinking block; keep it to pass the block back |
| `citations_delta` | A citation attached to text | See [Files, citations and search results](#files-citations-and-search-results) |

**Fine-grained tool streaming.** By default the API buffers and validates each parameter value before streaming it. Setting `eager_input_streaming: true` on a user-defined tool (on a streaming request) delivers that tool's input as Claude generates it, without server-side buffering or JSON validation, which reduces the time to the first fragment of a large parameter. All models support it on the Claude API, Amazon Bedrock, Claude Platform on AWS, Google Cloud and Microsoft Foundry. The trade-off is that the accumulated string can be partial or invalid JSON (for example when `max_tokens` cuts a parameter off), so guard the parse; the `input: {}` in `content_block_start` is only a placeholder. The per-tool field replaces the legacy `fine-grained-tool-streaming-2025-05-14` beta header. Tool definitions are covered in [Defining a tool](tool-use-and-mcp.md#defining-a-tool).

Two consequences for user interfaces. With thinking text omitted, the server skips streaming thinking tokens and delivers only the signature, so the visible answer starts sooner. And on Claude Fable 5.1, Mythos 5.1, Opus 5.5 and Fable 5, the short progress updates the model can write between tool calls come back as `thinking` blocks that are empty at the default `display: "omitted"`. An app that streams that text to users goes quiet between tool calls until it sets a `display` value that returns it: `"updates"` (beta, with the `thinking-display-updates-2026-08-18` header), which returns only the progress updates as text, or `"summarized"`, which also returns reasoning summaries in blocks you cannot tell apart from the updates.

### Streaming with the SDKs

The helper (`messages.stream`) gives you text events and can accumulate everything into the final `Message`: `.get_final_message()` in Python, `.finalMessage()` in TypeScript. The raw form (`create` with `stream` set to true) returns only the event iterator and uses less memory because it builds no final message.

=== "Python"

    ```python
    import anthropic

    client = anthropic.Anthropic()

    # Helper: print text as it arrives, then keep the complete Message.
    with client.messages.stream(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello"}],
        model="claude-opus-5-5",
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        message = stream.get_final_message()

    # Raw events: lower memory, you handle every event type yourself.
    stream = client.messages.create(
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello, Claude"}],
        model="claude-opus-5-5",
        stream=True,
    )
    for event in stream:
        print(event.type)
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";

    const client = new Anthropic();

    // Helper: print text as it arrives, then keep the complete Message.
    const stream = client.messages
      .stream({
        model: "claude-opus-5-5",
        max_tokens: 1024,
        messages: [{ role: "user", content: "Say hello there!" }]
      })
      .on("text", (text) => {
        console.log(text);
      });
    const message = await stream.finalMessage();

    // Raw events: lower memory; break, or call events.controller.abort(), to cancel.
    const events = await client.messages.create({
      max_tokens: 1024,
      messages: [{ role: "user", content: "Hello, Claude" }],
      model: "claude-opus-5-5",
      stream: true
    });
    for await (const messageStreamEvent of events) {
      console.log(messageStreamEvent.type);
    }
    ```

The Python SDK supports the same streaming calls on its async client (`AsyncAnthropic`, with `async for`). If you only stream to dodge timeouts on a large `max_tokens`, stream and collect the final message; you get the same `Message` object a non-streaming call returns.

### Errors and refusals inside a stream

Once a stream has started, the HTTP status is already 200. An error that happens later arrives as an `error` event in the stream, for example `overloaded_error`, the streaming counterpart of HTTP 529, and ordinary HTTP error handling does not see it:

```text
event: error
data: {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}
```

To recover from an interrupted stream without regenerating everything, capture the content received so far and send a continuation request. How depends on the model:

| Model generation | Continuation request |
|---|---|
| Claude 4.5 and earlier | Put the partial response at the start of a new assistant message, and Claude continues it |
| Claude 4.6 and later | Assistant prefill is not supported, so add a user message that contains the partial response and asks Claude to continue |

The docs' sample prompt for the second case is "Your previous response was interrupted and ended with [previous_response]. Continue from where you left off." ([Streaming messages](https://platform.claude.com/docs/en/build-with-claude/streaming)). Tool use and thinking blocks cannot be partially recovered; resume from the most recent text block.

A refusal in a stream arrives with its `stop_details` on the `message_delta` event, alongside `stop_reason: "refusal"`. A refusal that interrupts a stream bills the input and the output already streamed. Reset or rephrase the context, or retry on another model: continuing without a reset gets more refusals.

!!! warning "Exam guide vs current docs: websockets"

    CCDV-F's Technical Fundamentals skill mentions "integrating with SDKs that wrap REST APIs, websockets" ([CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)) without saying what role WebSockets play. The Messages API streaming documentation (as of September 2026) describes only server-sent events: a one-way channel from server to client over HTTP, served as `text/event-stream`, over which the client cannot send events back. A WebSocket (RFC 6455) is a full-duplex connection that starts as an HTTP Upgrade request answered with `101 Switching Protocols`, after which either side can send at any time. WebSockets do appear elsewhere in the Claude stack: Claude Code can connect to remote MCP servers of type `ws`, which suit servers that push events unprompted, while HTTP remains the choice for servers that only respond to requests (see [MCP transports](tool-use-and-mcp.md#mcp-transports)). Our reading, since the guide gives no definition: treat WebSockets as a general engineering concept, and if an item asks how the Messages API streams tokens to your application, answer SSE.

!!! note "API streaming is not Claude Code's streaming mode"

    The [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) also lists "streaming mode" among Claude Code features, again without a definition. Two documented features fit the name: Claude Code's `--output-format stream-json` (newline-delimited JSON for real-time streaming) and the Agent SDK's streaming input mode; no official page maps the exam term to either. Our reading: both are Claude Code and Agent SDK topics, not the Messages API's `stream` parameter; see [Headless mode and the CLI](claude-code-workflows.md#headless-mode-and-the-cli).

### Decide

- If a person waits on the answer, stream it, because users see output as it is generated. Model choice, effort and fast mode are separate latency levers; see [Models and how to choose one](#models-and-how-to-choose-one) and [fast mode](#fast-mode).
- If the job is long and nobody is waiting, use [Message Batches](#message-batches); not a long-held synchronous connection, because idle connections can be dropped and batches let you poll for results.
- If you only need the finished message but must stream to avoid timeouts, use the SDK's final-message helper; not a hand-written event parser.

### Traps

- **Parsing `input_json_delta` fragments as JSON one by one.** They are partial strings; parse after `content_block_stop`.
- **Summing `usage` across `message_delta` events.** The counts are cumulative.
- **Failing on an unknown event type.** New types can appear without a version change; ignore what you do not recognize.
- **Assuming a 200 means success.** An `error` event can still arrive mid-stream.

## Errors, retries and rate limits

*Tested in: CCDV-F 4.1 Debugging and Error Handling, 2.3 Claude API Mechanics, 5.2 Technical Fundamentals · CCAR-P 7.3 (debugging and operational issue resolution) · CCAR-F: rate limiting and quotas are out of scope (Appendix); error propagation between agents (5.3) is taught in [Error propagation in multi-agent systems](evaluation-and-reliability.md#error-propagation-in-multi-agent-systems)*

An API error means the request failed: you get an HTTP 4xx or 5xx status and an error body instead of a message. That is different from a `stop_reason`, which comes with a successful response. The [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) asks for "error type identification" and "recovery strategy selection", so learn each error by what it tells you to do next.

### The HTTP errors

| Status | `error.type` | Meaning | Retry? |
|---|---|---|---|
| 400 | `invalid_request_error` | The format or content of the request is wrong. Also used for other unlisted 4XX codes, and when usage reaches a spend limit you set yourself | No: fix the request (or raise your own limit) |
| 401 | `authentication_error` | API key problem: malformed, revoked or expired | No: fix the key |
| 402 | `billing_error` | Billing or payment problem | No: fix billing in the Console |
| 403 | `permission_error` | The key lacks permission for the resource | No: fix access or workspace settings |
| 404 | `not_found_error` | The resource was not found | No: check the endpoint path and IDs |
| 409 | `conflict_error` | The request conflicts with the resource's current state | After resolving the conflict |
| 413 | `request_too_large` | The request exceeds the endpoint's byte limit | No: shrink the payload |
| 429 | `rate_limit_error` | A rate limit, the usage tier's monthly spend cap, or a Claude Code workspace spend limit | Yes after `retry-after`, except at the tier spend cap |
| 500 | `api_error` | Unexpected error inside Anthropic's systems | Yes, with exponential backoff; if it persists, contact support with the request ID |
| 504 | `timeout_error` | The request timed out while processing | Yes; for long requests, stream instead |
| 529 | `overloaded_error` | The API is temporarily overloaded, which can happen under high traffic across all users | Yes, with backoff |

Two pairs are easy to confuse. **429 versus 529:** a 529 reflects load across all users, while a 429 is about your organization; a sharp increase in your own usage can also produce 429s from acceleration limits, so ramp traffic up gradually. **Tier spend cap versus your own spend limit:** reaching the tier's monthly cap returns a 429 `rate_limit_error` with no `retry-after` header and, on the Messages API, `error.details.error_code` set to `enforced_spend_limit_reached`, and usage pauses until 00:00 UTC on the first day of the next month unless you move to a higher tier. Reaching a limit you set yourself returns a 400 `invalid_request_error` whose message begins "You have reached your specified API usage limits" ([Rate limits](https://platform.claude.com/docs/en/api/rate-limits)).

### Error body and request ID

Errors are always JSON with a top-level `error` object holding `type` and `message`, plus a `request_id`:

```json
{
  "type": "error",
  "error": {
    "type": "not_found_error",
    "message": "The requested resource could not be found."
  },
  "request_id": "req_011CSHoEeqs5C35K2UUqR7Fy"
}
```

The set of `type` values may grow over time, so branch on the SDK's typed exceptions rather than on message text. Every response, success or failure, has a `request-id` header; the Python and TypeScript SDKs expose it as `_request_id` on response objects. Log it, and include it in any support ticket.

### Request size limits

| Endpoint | Maximum request size |
|---|---|
| Messages API | 32 MB |
| Token Counting API | 32 MB |
| Message Batches API | 256 MB |
| Files API | 500 MB |

Going over returns 413 `request_too_large`; on the direct Claude API, Cloudflare returns it before the request reaches the API servers. Partner-operated platforms set their own limits: Amazon Bedrock allows 20 MB and Google Cloud 30 MB. Claude Platform on AWS, which Anthropic operates, uses the same limits as the direct Claude API.

### What the SDKs already do

The official SDKs retry transient failures automatically: twice by default, with exponential backoff, honoring `retry-after` when present. The Python and TypeScript SDK pages list what they retry: connection errors, 408, 409, 429 and 500 or above. Set the count with `max_retries` (Python) or `maxRetries` (TypeScript), on the client or per request; `0` disables retries. Both SDKs time a request out after 10 minutes by default and retry a timed-out request too. One difference: the TypeScript SDK raises its default timeout for a large non-streaming `max_tokens`, up to 60 minutes. Both SDKs refuse a non-streaming request that is expected to take longer than about 10 minutes (Python raises a `ValueError`); stream the request or set `timeout` explicitly to avoid it.

Both SDKs raise typed exceptions. The HTTP-status classes have the same names in both languages; the timeout class does not:

| Status | Exception |
|---|---|
| 400 | `BadRequestError` |
| 401 | `AuthenticationError` |
| 403 | `PermissionDeniedError` |
| 404 | `NotFoundError` |
| 409 | `ConflictError` |
| 422 | `UnprocessableEntityError` |
| 429 | `RateLimitError` |
| 500 or above | `InternalServerError` |
| No response | `APIConnectionError` |
| Request timed out | Python `APITimeoutError`; TypeScript `APIConnectionTimeoutError` |

Catch the most specific class first. The retry and timeout values below are illustrative, not the defaults:

=== "Python"

    ```python
    import anthropic

    # Defaults: max_retries=2, timeout of 10 minutes. max_retries=0 disables retries.
    client = anthropic.Anthropic(max_retries=5, timeout=60.0)

    try:
        message = client.messages.create(
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello, Claude"}],
            model="claude-opus-5-5",
        )
    except anthropic.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)
    except anthropic.RateLimitError as e:
        print("A 429 status code was received; we should back off a bit.")
    except anthropic.APIStatusError as e:
        print("Another non-200-range status code was received")
        print(e.status_code)
        print(e.response)
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";

    // Defaults: maxRetries 2; timeout 10 minutes (scaled up to 60 for a large
    // non-streaming max_tokens). maxRetries: 0 disables retries.
    const client = new Anthropic({ maxRetries: 5, timeout: 60 * 1000 });

    const message = await client.messages
      .create({
        max_tokens: 1024,
        messages: [{ role: "user", content: "Hello, Claude" }],
        model: "claude-opus-5-5"
      })
      .catch(async (err) => {
        if (err instanceof Anthropic.APIError) {
          console.log(err.status); // 400
          console.log(err.name); // BadRequestError
          console.log(err.headers); // {server: 'nginx', ...}
        } else {
          throw err;
        }
      });
    ```

Retrying cannot fix the tier spend cap: the SDKs' automatic retries fail until access resumes. A stream that has already returned 200 reports later failures as `error` events instead; see [Streaming](#streaming). When you want to handle a 429 yourself instead of waiting, set retries to `0` for that request: the fast mode docs use this pattern to fall back to standard speed as soon as the fast mode limit is hit. Because `0` also switches off retries for overloaded and 5xx errors, their examples reissue those failures with default retries.

Claude Code applies its own retry policy: it retries transient failures up to 10 times with exponential backoff before showing an error (`CLAUDE_CODE_MAX_RETRIES` defaults to 10). That policy is covered in [Reliability engineering for Claude applications](evaluation-and-reliability.md#reliability-engineering-for-claude-applications).

### How rate limits work

There are two kinds of limit. **Spend limits** cap what an organization can spend on the API each month; **rate limits** cap how many requests and tokens it can use over time. Both are set by usage tier. Organizations are placed on a tier automatically, based on usage history and account standing, and can move up over time; new organizations may start in an Evaluation tier with lower limits. The published limits are maximums, not guaranteed minimums.

- **Three measures per model class:** requests per minute (RPM), input tokens per minute (ITPM) and output tokens per minute (OTPM). Exceeding any one returns a 429 that says which limit was hit, with a `retry-after` header.
- **Token bucket.** Capacity refills continuously up to the maximum rather than resetting on the minute, so a limit can bite over shorter intervals: 60 RPM might be enforced as 1 request per second, and a short burst can trigger 429s.
- **Acceleration limits.** A sharp increase in your organization's usage can also produce 429s; ramp traffic up gradually and keep usage patterns consistent.
- **Cache-aware ITPM.** For most models only uncached input counts toward ITPM: `input_tokens` and `cache_creation_input_tokens` count, `cache_read_input_tokens` does not (the exception is Claude Haiku 3.5, now retired except on Amazon Bedrock and Google Cloud). The docs' example: with a 2,000,000 ITPM limit and an 80% cache hit rate you could process 10,000,000 total input tokens per minute.
- **OTPM counts real output.** It is evaluated as tokens are generated, and `max_tokens` does not factor in, so a generous `max_tokens` costs nothing in rate-limit terms.
- **Per model.** Limits apply separately to each model, so different models can each run up to their own limits at once. Some models share a combined bucket: Fable 5.1 and Fable 5 share one, Mythos 5.1 and Mythos 5 share another, Sonnet 4.6 and 4.5 share the Sonnet 4.x limit, and the Opus 4.x limit covers Opus 4.8, 4.7, 4.6 and 4.5 together. Opus 5.5, Opus 5 and Sonnet 5 each have their own.
- **Separate pools.** Message Batches, token counting, the Files API, Managed Agents endpoints and fast mode each have limits of their own, separate from the Messages API limits. File-related calls are limited to approximately 500 requests per minute (shared across upload, list, retrieve, download and delete), and Managed Agents allows 300 requests per minute on create endpoints and 1,200 on read endpoints, per organization. Batch limits per tier are in [Message Batches](#message-batches) and token-counting limits in [Tokens, context windows and counting](#tokens-context-windows-and-counting). Rate limits are shared across all `inference_geo` values.
- **Workspaces.** You can set lower spend and rate limits per workspace (not on the default workspace) to protect other workspaces; unset workspace limits match the organization's, and organization-wide limits always apply.

Standard limits as of September 2026 for two model classes:

| Tier | Claude Opus 5.5 (RPM / ITPM / OTPM) | Claude Fable 5.x (RPM / ITPM / OTPM) | Monthly spend cap |
|---|---|---|---|
| Start | 1,000 / 2,000,000 / 400,000 | 1,000 / 500,000 / 100,000 | &#36;500 |
| Build | 5,000 / 5,000,000 / 1,000,000 | 2,000 / 1,500,000 / 300,000 | &#36;1,000 |
| Scale | 10,000 / 10,000,000 / 2,000,000 | 4,000 / 4,000,000 / 800,000 | &#36;200,000 |
| Custom | Higher than Scale, through sales | Higher than Scale, through sales | None; limits arranged with the account team |

At every tier, Claude Opus 5, Opus 4.x, Sonnet 5, Sonnet 4.x and Haiku 4.5 carry the same numbers as Opus 5.5, each as its own limit. To raise limits or the spend cap, use **Request rate limit increase** on the Console's Rate limits page (on Claude Platform on AWS, contact your account representative or support instead).

Every response carries rate-limit headers you can use to pace traffic before you hit a 429:

| Header | Meaning |
|---|---|
| `retry-after` | Seconds to wait before retrying; earlier retries fail. Not sent with the spend-cap 429 |
| `anthropic-ratelimit-requests-limit` / `-remaining` / `-reset` | The request limit, what is left, and when it is fully replenished (RFC 3339) |
| `anthropic-ratelimit-input-tokens-*`, `anthropic-ratelimit-output-tokens-*` | The same three values for input and output tokens (remaining rounded to the nearest thousand) |
| `anthropic-ratelimit-tokens-*` | The most restrictive token limit currently in effect (for example a workspace limit) |
| `anthropic-priority-input-tokens-*`, `anthropic-priority-output-tokens-*` | The same values for Priority Tier capacity (Priority Tier only) |
| `anthropic-fast-input-tokens-*`, `anthropic-fast-output-tokens-*` | Fast mode's separate limits |

!!! note "Priority Tier, as of September 2026"

    `service_tier: "auto"` (the default) uses Priority Tier capacity when you have it, and `"standard_only"` never does. Priority Tier prioritizes requests to minimize server-overloaded errors and targets 99.5% uptime, and a request it serves draws on both the commitment and your regular rate limits. New commitments are no longer available for purchase (existing ones run to their contract end date), and it is not supported on Claude Fable 5.1, Mythos 5.1, Mythos 5, Mythos Preview, Opus 5.5, Opus 5 or Sonnet 5. Server-side fallback (the beta `fallbacks` parameter) does not help with capacity either, because only a safety classifier decline triggers it; see the refusals note in [Every stop reason and how to handle it](#every-stop-reason-and-how-to-handle-it).

### Recovery strategy by symptom

| Symptom | Cause | Recovery |
|---|---|---|
| 400 "prompt is too long" ([Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows)) | The input alone exceeds the context window | Trim, compact or split the input; see [Tokens, context windows and counting](#tokens-context-windows-and-counting) |
| 400 on `thinking: {"type": "enabled"}` | Extended thinking was removed on Claude 4.7 and later | Use `thinking: {"type": "adaptive"}` and `output_config.effort` |
| 400 on `thinking: {"type": "disabled"}` | Thinking is always on (Opus 5.5, Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Mythos Preview), or Opus 5 at effort `xhigh` or `max` | Omit `thinking` (on Opus 5, lower effort to `high` or below to keep thinking off); to hide the reasoning, set `display: "omitted"` instead |
| 400 "adaptive thinking is not supported on this model" ([errors](https://platform.claude.com/docs/en/api/errors)) | Adaptive thinking on Claude 4.5 or earlier | Use `thinking: {"type": "enabled", "budget_tokens": N}` |
| 400 "This model does not support assistant message prefill" ([errors](https://platform.claude.com/docs/en/api/errors)) | Prefill on Claude 4.6 and later, or Mythos Preview | Structured outputs or system prompt instructions |
| 400 on `tool_choice` `any` or `tool` | Forced tool use on Opus 5.5, Fable 5.1 or Mythos 5.1, or forced tool use combined with manual extended thinking on any model | `auto` with strict tool use, or structured outputs (or switch to adaptive thinking where the model supports it) |
| 400 saying thinking blocks "cannot be modified" ([errors](https://platform.claude.com/docs/en/api/errors)) | Your code edited, reordered, filtered or rebuilt thinking blocks | Pass the assistant turn back exactly as received |
| 401 | Malformed, revoked or expired key | Replace the key; retries cannot help |
| 413 | Payload over the endpoint limit | Upload large files once with the Files API and reference them by `file_id`, or split the work across requests |
| 429 with `retry-after` | A rate limit (RPM, ITPM or OTPM), often hit by a burst | Wait as told, smooth bursts, cache repeated input, move non-urgent volume to [Message Batches](#message-batches) |
| 429s after a sudden jump in traffic | Acceleration limits | Ramp traffic up gradually and keep usage consistent |
| 429 without `retry-after`, `enforced_spend_limit_reached` | Tier spend cap | Request a higher limit; retries fail until access resumes |
| 500 persisting after retries | Internal error | Contact support with the request ID |
| 529, or an `overloaded_error` event mid-stream | High load across all users | Back off and retry |
| 504, or long calls dropped | Long request on a plain connection | Stream, or use Message Batches |

Two rows conflict with the July 2026 guides: the CCAR-F guide still teaches `tool_choice` `"any"` and forced tool selection as the way to guarantee a tool call, and CCDV-F lists extended thinking as a model option. On the exam, answer in the guides' terms, as the freshness notes in [Structured outputs](#structured-outputs) and [Extended thinking, adaptive thinking and effort](#extended-thinking-adaptive-thinking-and-effort) explain, and use the model-specific 400s when you debug real code.

An HTTP error is an integration-layer failure: the request failed before Claude produced a message. A successful response with a wrong or weak answer is a different problem, covered in [Debugging: model or integration](evaluation-and-reliability.md#debugging-model-or-integration). Errors that your own tools return to Claude inside an agent loop (CCAR-F 2.2 distinguishes transient, validation, business and permission errors) are covered in [Returning tool results and errors](tool-use-and-mcp.md#returning-tool-results-and-errors).

### Decide

These rules are ours, each drawn from the documented behavior above.

- If the status is 400, 401, 403, 404 or 413, fix the request, key, permissions or payload; not a retry loop, because the same request fails the same way.
- If you get a 429 with `retry-after`, wait that long and smooth your traffic; not an immediate tight retry, because earlier retries fail. If the 429 has no `retry-after` and carries `enforced_spend_limit_reached`, raise the limit; not a retry at all.
- If ITPM is the binding limit and many requests share a long prefix, cache it; not a bigger tier first, because cache reads do not count toward ITPM on most models.
- If a high-volume job can wait, use Message Batches, which have their own limits; not more parallel synchronous calls, which burst into 429s and, in the words of the [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)'s Sample 1 rationale, "Sending requests synchronously in parallel (A) does not reduce per-token cost".

### Traps

- **Wrapping every call in your own retry loop on top of the SDK's.** The SDKs already retry twice, so an outer loop multiplies the attempts made for each failed call (our reasoning: three outer attempts around an SDK that makes up to three can send nine requests). Tune `max_retries` instead.
- **Setting a small `max_tokens` to stay under OTPM.** OTPM counts generated tokens, not the `max_tokens` value.
- **Reading a 529 as your fault or a 429 as Anthropic's.** 529 reflects high traffic across all users; 429 is your organization's limits (or a Claude Code workspace spend limit).
- **Treating every 429 as retryable.** The spend-cap 429 has no `retry-after` and keeps failing until access resumes.
- **String-matching error messages.** Types can be added and messages change; use the typed exceptions.
- **Retrying, then hiding the failure.** In CCAR-F sample Question 8, the wrong option B retries with exponential backoff inside a subagent and then returns a generic "search unavailable" status; the [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) rejects it because "Option B's generic status hides valuable context from the coordinator, preventing informed decisions." Backoff suits 429 and 529; what an agent reports upward after its retries still decides whether the system can recover. See [Error propagation in multi-agent systems](evaluation-and-reliability.md#error-propagation-in-multi-agent-systems).

## Models and how to choose one

*Tested in: CCAO-F D3.2, D3.3 · CCDV-F 5.3 Model Selection and Tradeoffs, 2.6 Configuration Management (model version pinning) · CCAR-P 2.1, 3.3, 4.5 · CCAR-F: model comparison metrics are out of scope (Appendix)*

The guides frame model choice as a trade-off: [CCAO-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) asks you to align it with "cost, speed, quality", [CCDV-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) names "quality/latency/cost" trade-offs, and [CCAR-P](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf) asks you to select models "based on trade-offs". The CCAO-F guide states the rule in its Sample 2 rationale: "Aligning model selection with task requirements means matching a faster, lower-cost model to straightforward, high-volume work, reserving the most capable model for complex reasoning." ([CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)).

### The current lineup (as of September 2026)

| | Claude Fable 5.1 | Claude Opus 5.5 | Claude Sonnet 5 | Claude Haiku 4.5 |
|---|---|---|---|---|
| Anthropic's description | For demanding reasoning and long-horizon agentic work | For long-running agentic coding and knowledge work | The best combination of speed and intelligence | The fastest model with near-frontier intelligence |
| Claude API ID | `claude-fable-5-1` | `claude-opus-5-5` | `claude-sonnet-5` | `claude-haiku-4-5-20251001` (alias `claude-haiku-4-5`) |
| Comparative latency | Slower | Moderate | Fast | Fastest |
| Price per million tokens (input / output) | &#36;10 / &#36;50 | &#36;4 / &#36;20 | &#36;2 / &#36;10 | &#36;1 / &#36;5 |
| Context window | 1M tokens | 1M tokens | 1M tokens | 200K tokens |
| Max output (synchronous) | 128K tokens | 128K tokens | 128K tokens | 64K tokens |
| Thinking | Adaptive (always on) | Adaptive (always on) | Adaptive | Extended |
| Default effort | `high` | `medium` | `high` | Not supported |
| Reliable knowledge cutoff | Jun 2026 | Jun 2026 | Jan 2026 | Feb 2025 |
| Retirement not sooner than | September 1, 2027 | September 22, 2027 | June 30, 2027 | October 15, 2026 |

All current models take text and image input, produce text output, and support multilingual use, vision and tool use. Claude Opus 5.5 was released on September 22, 2026. Claude Mythos 5.1 offers Fable 5.1's capabilities to Project Glasswing participants only and is invite only. Batch API requests are 50% off these prices; caching discounts are in [Prompt caching](#prompt-caching).

Three footnotes to the table. Max output is the synchronous Messages API limit: on the Message Batches API, Opus 5.5, Opus 5, Sonnet 5, Opus 4.8, Opus 4.7, Opus 4.6 and Sonnet 4.6 go up to 300k output tokens with the `output-300k-2026-03-24` beta header. Comparative latency is relative to the current lineup, and actual latency depends on prompt length, output length and thinking effort. The retirement dates are Anthropic's commitment for Anthropic-operated platforms (Claude API, Claude Platform on AWS, Microsoft Foundry); Amazon Bedrock and Google Cloud set their own.

Legacy models are still available:

| Legacy model | Claude API ID | Context window | Max output | Price per million tokens (input / output) |
|---|---|---|---|---|
| Claude Fable 5 | `claude-fable-5` | 1M | 128K | &#36;10 / &#36;50 |
| Claude Opus 5 | `claude-opus-5` | 1M | 128K | &#36;5 / &#36;25 |
| Claude Opus 4.8 | `claude-opus-4-8` | 1M | 128K | &#36;5 / &#36;25 |
| Claude Opus 4.7 | `claude-opus-4-7` | 1M | 128K | &#36;5 / &#36;25 |
| Claude Opus 4.6 | `claude-opus-4-6` | 1M | 128K | &#36;5 / &#36;25 |
| Claude Sonnet 4.6 | `claude-sonnet-4-6` | 1M | 128K | &#36;3 / &#36;15 |
| Claude Opus 4.5 | `claude-opus-4-5-20251101` | 200K | 64K | &#36;5 / &#36;25 |
| Claude Sonnet 4.5 | `claude-sonnet-4-5-20250929` | 200K | 64K | &#36;3 / &#36;15 |

To check limits and features in code instead of from a table, call `GET /v1/models`: it lists models newest first and returns `max_input_tokens`, `max_tokens` and a `capabilities` object (batch, citations, effort levels, image and PDF input, structured outputs, thinking types) for each.

### What each tier is for

| When you need | Start with | Anthropic's example uses |
|---|---|---|
| The highest available capability | Claude Fable 5.1 | Agent sessions that run for hours, multistep deep research, analysis carried through to a finished document, spreadsheet or deck |
| Complex agentic coding and enterprise work | Claude Opus 5.5 | Multihour autonomous coding agents, large-scale refactoring, complex systems engineering, vision-heavy workflows, computer use |
| Speed and capability for everyday coding, agent and enterprise workloads | Claude Sonnet 5 | Code generation, data analysis, content creation, visual understanding, agentic tool use |
| The lowest latency and price, with extended thinking | Claude Haiku 4.5 | Real-time applications, high-volume intelligent processing, cost-sensitive deployments needing strong reasoning, sub-agent tasks |

The pricing page's cost tips for building agents give a simpler three-family rule: "Choose Haiku for simple tasks, Sonnet for most production workloads, and Opus for the most complex reasoning" ([Pricing](https://platform.claude.com/docs/en/about-claude/pricing)). Its everyday default (Sonnet) differs from the [Models overview](https://platform.claude.com/docs/en/models/overview)'s default when unsure, which is Claude Opus 5.5 for most workloads, moving to Fable 5.1 when evals on Opus 5.5 at higher effort still fall short. For speed-critical applications, the latency guide points to Claude Haiku 4.5 as the fastest.

### Two ways to pick a starting model

| | Efficiency-first | Capability-first |
|---|---|---|
| Start with | Claude Haiku 4.5 | Claude Opus 5.5 |
| Then | Test thoroughly, and upgrade only for specific capability gaps | Optimize prompts, then lower effort or downgrade models as the workflow matures; move to Claude Fable 5.1 if `xhigh` or `max` effort still falls short |
| Best for | Prototyping, tight latency requirements, cost-sensitive implementations, high-volume straightforward tasks | Complex reasoning, scientific or mathematical work, nuanced understanding, accuracy that outweighs cost, advanced coding and high-autonomy agentic work |

Either way, the decision to change models rests on your own evidence. The docs list four criteria (capabilities, speed, cost, effort), say "having a good evaluation set is the most important step in the process", and add that "Tuning effort is often a better lever than switching models." ([Choosing a model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model)).

### Cost is per task, not per token

- **Compare cost per completed task.** Anthropic's cost guidance says "Compare on cost per completed task, not per token" ([Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence)) and tells you to price every candidate that way on your own traffic. The Academy's effort tutorial adds that cost per task "can diverge significantly from cost per token" ([Claude Academy](https://academy.claude.com/tutorials/how-to-select-the-right-effort-setting-for-claude-cowork-and-chat)).
- **A stronger model at lower effort can beat a weaker model at high effort.** Anthropic's post on reducing cost: "A stronger model at low effort can be cheaper than a weaker model working hard (high effort)." ([Reducing cost and improving performance](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform)). The Academy's effort tutorial makes the same point for app users: "A frontier model at medium or low effort often outperforms an older model at high or maximum effort." ([Claude Academy](https://academy.claude.com/tutorials/how-to-select-the-right-effort-setting-for-claude-cowork-and-chat)).
- **Token prices are not comparable across tokenizers.** Claude 4.7 and later models use a newer tokenizer that produces about 30% more tokens for the same text. Claude Sonnet 5's lower per-token price therefore does not lower per-request cost in proportion.
- **Combine models.** Multi-model strategies pair a lower-cost model with a frontier model so that most tokens bill at the lower rate: an executor that escalates hard decisions to an advisor, or an orchestrator that delegates bulk work to cheaper workers.

### Pinning the version

Every Claude model ID, dateless ones included, is a pinned snapshot; a pre-4.6 alias such as `claude-sonnet-4-5` is not. Why that holds, and the ID formats, are in [A model ID is a pinned snapshot](#a-model-id-is-a-pinned-snapshot).

For the [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)'s "model version pinning", our rule follows from those facts: put a model ID, not a pre-4.6 alias, in configuration (`claude-sonnet-4-5-20250929` rather than `claude-sonnet-4-5`; from 4.6 on the dateless ID is itself the pin), and change it only as a deliberate, tested upgrade. CCDV-F 5.3 also names "breaking behavior changes across model releases"; those, with lifecycle states, retirement notice and migration steps, are in [Model versions, deprecation and migration](#model-versions-deprecation-and-migration).

!!! warning "Exam guide vs current docs: three tiers or four"

    The July 2026 guides name three families: [CCAO-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) asks you to "Differentiate between Claude model types (Haiku, Sonnet, Opus)" and [CCDV-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) tests "Opus vs. Sonnet vs. Haiku use cases". As of September 2026 the lineup also has Claude Fable 5.1, which [Choosing a model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model) calls "Anthropic's most capable widely released model" (and the invite-only Mythos line), Claude Opus 5.5 is the docs' default starting point, and Claude Haiku 4.5 is the only Haiku in the current lineup, with retirement not sooner than October 15, 2026. On the exam, answer in the guides' terms and reason in tiers: Haiku, the fastest and cheapest, for straightforward high-volume work; Sonnet for most production workloads; Opus for the most complex reasoning (the split the pricing page also uses). The guides name families, not model versions, so the tier logic is what carries over.

### Decide

These rules are ours, built on the official rationales and the docs cited above.

- If the task is high-volume and straightforward and speed or cost matter more than deep reasoning, choose the fastest, lowest-cost tier; not the most capable model for every call, which in the [CCAO-F rationale](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) "wastes the cost and latency budget".
- If accuracy on complex reasoning or long agentic work outweighs cost, start capability-first and optimize down with effort; not the smallest model regardless of fit, an option the official samples mark wrong twice.
- If a model is close but slightly too slow or costly, lower effort before switching models, because [Choosing a model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model) calls tuning effort "often a better lever than switching models".
- If the requirement is faster output from an Opus model rather than a cheaper model, consider fast mode (research preview, premium pricing, on Opus 5.5, Opus 5 and Opus 4.8); see [fast mode](#fast-mode).

### Traps

- **A larger model as a security fix.** The official rationales reject it: a more instruction-following model "can be more susceptible, not less" to prompt injection ([CCDV-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) Sample 2), and model size "is unrelated to authorization scope" ([CCAR-P](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf) Sample 1).
- **Downsizing blindly.** "Switch to the smallest available model regardless of output quality." is a wrong option in [CCDV-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) Sample 1, and "Switch to the smallest available model regardless of task fit." is one in [CCAR-P](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf) Sample 2.
- **Comparing per-token prices across generations** without recounting tokens on the target model: Claude 4.7 and later produce about 30% more tokens for the same text.
- **Assuming a dateless ID tracks the newest model.** Dateless IDs are pinned snapshots, not evergreen pointers.
- **Cutting features or changing platforms instead of changing the model.** In [CCAO-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) Sample 2, "Disable all product features to reduce cost." and "Switch to a different AI platform." are wrong options; the rationale says disabling features or switching platforms "does not address the trade-off".

## Extended thinking, adaptive thinking and effort

*Tested in: CCDV-F 5.1 LLM Fundamentals (fast mode, extended thinking, adaptive thinking, effort levels), 5.3 Model Selection and Tradeoffs (adaptive thinking support), 2.3 Claude API Mechanics (thinking) · CCAR-P 2.1, 4.5 (our mapping: effort is a model-configuration trade-off) · CCAR-F 4.6-K2 (extended thinking versus an independent reviewer) · CCAO-F D1.4, adapting to task type (our mapping: the CCAO-F guide does not mention effort or thinking; the app settings are covered below)*

Three controls decide how much work Claude does on a request, and the exams expect you to keep them apart. **Thinking** decides whether Claude reasons in separate thinking blocks before answering. **Effort** sets how many tokens Claude spends on the whole response: text, tool calls and thinking. **`max_tokens`** is the hard ceiling on output. A fourth option, **fast mode**, changes output speed on some Opus models without changing the model.

### How thinking works

- Thinking arrives as one or more `thinking` blocks before the `text` blocks. Each carries a `signature`, an encrypted copy of the full reasoning that you pass back unchanged in tool-use and multi-turn conversations.
- What you can read is never the raw chain of thought: it is a summary, produced by a different model from the one you called.
- Thinking tokens are billed as output tokens even when the text is not returned, and they count toward `max_tokens`. `usage.output_tokens_details.thinking_tokens` shows how many billed output tokens were reasoning.
- Safety-redacted reasoning arrives as a `redacted_thinking` block with an encrypted `data` field. Pass it back unchanged too.

### Adaptive or extended

| | Adaptive thinking | Extended (manual) thinking |
|---|---|---|
| Request | `thinking: {"type": "adaptive"}` | `thinking: {"type": "enabled", "budget_tokens": N}` |
| Who decides how much to think | Claude decides whether and how much, per request, steered by effort; it may skip thinking on easy inputs | A fixed budget; Claude thinks on every request |
| Depth control | `output_config.effort` | `budget_tokens`: at least 1,024 and less than `max_tokens` (except with interleaved thinking); a target, not a strict cap; not allowed with `max_tokens: 0`. On Opus 4.5, the only extended-only model that supports effort, set both |
| Thinking between tool calls (interleaved) | Automatic, no beta header | Beta header `interleaved-thinking-2025-05-14` on Opus 4.5, Sonnet 4.5 and earlier Claude 4 models (still functional but deprecated on Sonnet 4.6; none in Opus 4.6's manual mode); not supported on Haiku 4.5 |
| Forced tool use (`any`, `tool`) | Allowed, except on Opus 5.5, Fable 5.1 and Mythos 5.1 | Not allowed: only `auto` or `none` |
| Turn structure | No assistant turn needs to start with a thinking block | The final assistant turn must begin with a thinking block |
| Status | The current mode | Deprecated on the Claude 4.6 models (still succeeds), rejected with a 400 on Claude 4.7 and later, the only mode on Claude 4.5 and earlier; Mythos Preview accepts both |

Which configuration each model accepts, as of September 2026:

| Model | Thinking types | Default | Rejected with 400 |
|---|---|---|---|
| Claude Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Opus 5.5 | Adaptive only | Always on | `"enabled"`, `"disabled"` |
| Claude Mythos Preview | Adaptive, extended | Always on | `"disabled"` |
| Claude Opus 5 | Adaptive only | On | `"enabled"`; `"disabled"` at effort `xhigh` or `max` |
| Claude Sonnet 5 | Adaptive only | On | `"enabled"` |
| Claude Opus 4.8, Opus 4.7 | Adaptive only | Off | `"enabled"` |
| Claude Opus 4.6, Sonnet 4.6 | Adaptive, extended (deprecated) | Off | None |
| Claude Opus 4.5, Haiku 4.5, Sonnet 4.5 | Extended only | Off | `"adaptive"` |

Models marked **Always on** cannot turn thinking off; models marked **On** default to thinking but accept `{"type": "disabled"}`. Two consequences matter for the exams (our reading). Claude Haiku 4.5 is the current-lineup model without adaptive thinking, which bears on the [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)'s "adaptive thinking support". And on Opus 5.5 you cannot turn thinking off at all: omit the `thinking` field or send `{"type": "adaptive"}` (the two are equivalent), use effort to control depth, and set `display: "omitted"` if the goal was only to keep thinking text out of responses.

### Showing or hiding the thinking

The `display` field sets what comes back: `"summarized"` returns readable summaries, `"omitted"` returns thinking blocks with an empty `thinking` field (the signature is still there), and `"updates"` (beta, header `thinking-display-updates-2026-08-18`) keeps reasoning empty but returns the short progress updates some models write between tool calls. `"omitted"` is the default on Claude Opus 4.7 and later, Sonnet 5 and the Fable and Mythos models, and `"summarized"` on Opus 4.6, Sonnet 4.6 and earlier, so on newer models you opt in to see the thinking. `display` works with `"adaptive"` and `"enabled"` alike and is invalid with `"disabled"`. Omitting speeds up the first visible text when streaming but does not cut cost: "You're still charged for the full thinking tokens. Omitting reduces latency, not cost." ([Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking)).

=== "Python"

    ```python
    import anthropic

    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=16000,
        thinking={"type": "adaptive", "display": "summarized"},
        messages=[{"role": "user", "content": "What is the greatest common divisor of 1071 and 462?"}],
    )
    for block in response.content:
        match block.type:
            case "thinking":
                print(f"\nThinking: {block.thinking}")
            case "text":
                print(f"\nResponse: {block.text}")
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";

    const client = new Anthropic();

    const response = await client.messages.create({
      model: "claude-opus-4-8",
      max_tokens: 16000,
      thinking: {
        type: "adaptive",
        display: "summarized"
      },
      messages: [
        { role: "user", content: "What is the greatest common divisor of 1071 and 462?" }
      ]
    });
    for (const block of response.content) {
      switch (block.type) {
        case "thinking":
          console.log(`\nThinking: ${block.thinking}`);
          break;
        case "text":
          console.log(`\nResponse: ${block.text}`);
          break;
      }
    }
    ```

On Opus 4.8 thinking is off until you set `{"type": "adaptive"}`, which is why the example sets it. On Opus 5.5 thinking is already on, and the docs' opt-in to see it is this same request with only the model string changed (`display: "summarized"` is what makes the text visible).

Moving off extended thinking is a small mapping, from the [extended thinking guide](https://platform.claude.com/docs/en/build-with-claude/extended-thinking): remove `budget_tokens`, set adaptive, and move depth control to effort.

```json
{"model": "claude-sonnet-4-6", "max_tokens": 16000,
 "thinking": {"type": "enabled", "budget_tokens": 10000}}
```

becomes

```json
{"model": "claude-sonnet-4-6", "max_tokens": 16000,
 "thinking": {"type": "adaptive"},
 "output_config": {"effort": "high"}}
```

`effort: "high"` matches the API default on this model, so omitting it behaves identically; it is there to show where depth control now lives. Expect a behavioral change, not just a syntax change: a fixed budget thinks on every request, while adaptive thinking may skip thinking on easy inputs at lower effort. The `interleaved-thinking-2025-05-14` header can go too, and the first request after the switch invalidates prompt-cache breakpoints.

If you stay on an extended-only model such as Haiku 4.5, tune the budget from about 1,024 tokens for simple tasks to 16,000 or more for complex ones, use batch processing for budgets above 32k to avoid networking issues, and track the real spend in `usage.output_tokens_details.thinking_tokens`.

### Thinking inside a tool-use loop

- **A tool-use loop is one assistant turn.** Keep the thinking configuration fixed until the turn ends. If you toggle thinking mid-turn, the API does not error; it silently disables thinking for that request.
- **Pass thinking blocks back.** Within a tool-use turn this is required, complete and unmodified. Across turns it is recommended, and outside tool use you may omit old thinking; the API filters old blocks itself and bills only the ones shown to Claude. Filtering on `type == "thinking"` alone drops `redacted_thinking` blocks and breaks the protocol.
- **Keep history append-only on the newest models.** On Claude Fable 5.1 and Opus 5.5, a replayed thinking block stays valid only while the top-level system prompt, the tools and the messages before it are unchanged. The `thinking-binding-controls-2026-08-01` beta adds `thinking.block_binding.prefix_mismatch_behavior`: `"error"` rejects a broken prefix with a 400, and `"drop_block"` drops each failing block and every thinking block after it so the request succeeds. Accounts created on or after August 31, 2026, 00:00 UTC get the 400 by default (in the Message Batches API, an item that leaves the field unset has its failing blocks dropped instead of failing); older accounts are checked only on requests that set the field. Build append-only either way.
- **Which prior thinking stays in context depends on the model**; the per-model rule is in [The context window](#the-context-window).
- **Config changes restart the cache.** Changing the thinking type, `budget_tokens` or the top-level effort invalidates prompt-cache breakpoints; see [What breaks the cache](#what-breaks-the-cache).

### Effort

Set `output_config.effort`; the top-level parameter is generally available with no beta header. It supports five levels, though not every model that supports `max` supports `xhigh`. Effort is supported on Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Mythos Preview, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Opus 4.6, Opus 4.5, Sonnet 5 and Sonnet 4.6; Haiku 4.5 and Sonnet 4.5 do not support it.

| Level | What it does | Typical use |
|---|---|---|
| `max` | Absolute maximum capability, no constraint on token spend (Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Mythos Preview, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Opus 4.6, Sonnet 5, Sonnet 4.6) | The deepest reasoning and most thorough analysis |
| `xhigh` | Extended capability for long-horizon work (Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Sonnet 5) | Long-running agentic and coding tasks (over 30 minutes) with token budgets in the millions |
| `high` | Spends as many tokens as the task needs; the default on every effort-supporting model except Opus 5.5 | Complex reasoning, hard coding problems, agentic tasks |
| `medium` | Balanced, with moderate token savings; the default on Opus 5.5 | Agentic tasks that balance speed, cost and performance |
| `low` | Most efficient, with some capability reduction | Simple tasks that need speed and low cost, such as subagents |

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=4096,
    messages=[{"role": "user", "content": "Analyze the trade-offs between microservices and monolithic architectures"}],
    output_config={"effort": "medium"},
)
```

On Opus 5.5, `"medium"` is the default, so this request behaves exactly like one that omits `effort`; on Opus 5 the same line would lower effort by one level.

- Effort affects every output token (text, tool calls and arguments, and thinking when active), so it works with or without thinking. Lower effort also means fewer, terser tool calls with less preamble; higher effort may mean more tool calls, a plan explained before acting and more detailed summaries.
- It is guidance, not a budget: "Effort is a behavioral signal, not a strict token budget." ([Effort](https://platform.claude.com/docs/en/build-with-claude/effort)). At lower levels Claude still thinks on hard problems, just less, and can skip thinking entirely on simpler ones.
- Setting the default explicitly behaves exactly like omitting the parameter. Most models default to `high`; Claude Opus 5.5 defaults to `medium`, so a request that omits effort runs one level lower than it did on Opus 5. For Opus 5.5 the docs advise an effort sweep on your own evals rather than carrying settings over from an earlier model.
- At high effort levels, set a large `max_tokens`: it is the hard limit on thinking plus response text. The docs suggest starting at 64k when running Opus 4.7, 4.8 or 5 at `xhigh` or `max`, and recommend starting at `xhigh` for coding and agentic work on Opus 4.7 and 4.8.
- On Opus 4.5, effort works alongside `budget_tokens`: set effort for the task, then the budget for reasoning depth.
- Changing top-level effort between requests invalidates the prompt cache. On Fable 5.1, Mythos 5.1, Opus 5.5 and Opus 5, per-message effort (beta, header `mid-conversation-output-config-2026-07-01`) changes it through a `role: "system"` message with empty `content` and the new level in `output_config.effort`, and keeps the cache.

Anthropic's cost analysis shows why effort is worth tuning: on Claude Fable 5.1, Humanity's Last Exam (without tools) scores about 53% at low effort and about 61% at max, but "the last step up to max adds about half a point for 46% more cost" ([Reducing cost and improving performance](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform)), a gain inside the benchmark's run-to-run noise. The same post notes that on an evaluation that is not saturated, a flat performance-cost curve across effort levels suggests the task is not bound by thinking compute, so raising effort is not beneficial.

### Which control to reach for

| Goal | Control |
|---|---|
| Lower cost or latency on a thinking workload | Lower effort first; it scales the whole response down, thinking included |
| Claude thinks too rarely or too shallowly | Raise effort |
| No thinking at all | `thinking: {"type": "disabled"}`, only on models that accept it |
| A hard ceiling on spend per request | `max_tokens`; the [thinking guide](https://platform.claude.com/docs/en/build-with-claude/thinking) says "Effort is soft guidance. `max_tokens` is a strict limit." |
| A budget across a whole agentic loop | Task budgets: `output_config.task_budget` (beta, header `task-budgets-2026-03-13`, minimum `total` of 20,000 tokens) on Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Opus 5.5, Opus 5, Opus 4.8 and Opus 4.7. A soft hint the model paces itself against, while `max_tokens` stays the hard per-request limit |
| Faster output from the same Opus model | Fast mode (below) |

### Fast mode

Fast mode is a research preview that gives up to 2.5x higher output tokens per second on Claude Opus 5.5, Opus 5 and Opus 4.8. It runs the same model with a faster inference configuration, so capability does not change, and the gain is in output tokens per second, not time to first token.

```python
import anthropic

client = anthropic.Anthropic()

response = client.beta.messages.create(
    model="claude-opus-5-5",
    max_tokens=4096,
    speed="fast",
    betas=["fast-mode-2026-02-01"],
    messages=[{"role": "user", "content": "Refactor this module to use dependency injection"}],
)
```

- **Price:** Opus 5.5 fast mode costs &#36;8 input and &#36;40 output per million tokens; Opus 5 and Opus 4.8 cost &#36;10 and &#36;50.
- **Where:** Claude API only (including Managed Agents); not on Amazon Bedrock, Claude Platform on AWS, Google Cloud or Microsoft Foundry.
- **Model quirks:** `speed: "fast"` returns an error on Opus 4.7, while Opus 4.6 silently runs at standard speed and bills standard rates.
- **Limits and cache:** fast mode has its own rate limits (a 429 with `retry-after` when exceeded, and `anthropic-fast-*` headers), switching speeds invalidates the prompt cache, and it is not available with the Batch API or Priority Tier.
- **Access and checking:** access is by request (account manager or waitlist), and `usage.speed` in the response reports `"fast"` or `"standard"`.

### In the Claude apps

CCAO-F candidates meet the same ideas as settings. The help center describes Low and Medium effort for routine tasks, High as the best overall balance of quality and speed, Extra high (xhigh) for long-running coding and agentic tasks on Opus 4.7 and newer models, and Max for the deepest reasoning; thinking and effort are separate settings you can combine, and in the apps thinking cannot be turned off on Claude Opus 5.5, Fable 5.1 or Opus 5. An Academy tutorial lists the signs of too little effort (missed instructions, long work that stops before it is finished) and of too much (more verbose answers without better quality, scope that grows past the request). More on app settings is in [Choosing a model in the apps](claude-for-work.md#choosing-a-model-in-the-apps).

!!! warning "Exam guide vs current docs: extended thinking and effort"

    CCDV-F lists "model options (fast mode, extended thinking, adaptive thinking, effort levels)" ([CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). As of September 2026, extended thinking (`budget_tokens`) is deprecated on the 4.6 models and returns a 400 on Claude 4.7 and later (Claude Opus 4.7 was released April 16, 2026), adaptive thinking is always on for Opus 5.5 and Fable 5.1, effort has five levels with Opus 5.5 defaulting to `medium`, and fast mode is still a research preview behind a beta header. On the exam, treat extended thinking and adaptive thinking as two of the four model options the guide lists. The docs define them: extended thinking is the manual mode with a fixed `budget_tokens`, and adaptive thinking lets Claude decide how much to think, steered by effort ([Models overview](https://platform.claude.com/docs/en/models/overview)). Use the current-docs picture (adaptive plus effort, with a 400 for `budget_tokens` on Claude 4.7 and later) when you write real code.

### Decide

These rules are ours, each resting on the documented behavior above.

- If you want lower cost or latency and thinking is on, lower effort; not a switch to extended thinking with a small budget, which Claude 4.7 and later reject.
- If you need a guaranteed ceiling on tokens per request, set `max_tokens`; not effort, which is a behavioral signal.
- If users never see the reasoning, use `display: "omitted"` (already the default on Opus 4.7 and later); it trims time to the first visible text, but expect the same bill.
- If only output speed matters on a supported Opus model and the budget allows premium pricing, try fast mode; not a smaller model, which changes capability.
- If generated work needs a critical review, use an independent review instance; not extended thinking in the same session, which [CCAR-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) 4.6-K2 ranks below it: "Independent review instances (without prior reasoning context) are more effective at catching subtle issues than self-review instructions or extended thinking".

### Traps

- **`output_config.effort: "adaptive"`.** Invalid: `adaptive` belongs in `thinking.type`.
- **`thinking: {"type": "disabled"}` on Opus 5.5 or Fable 5.1.** A 400; those models are always on.
- **`thinking: {"type": "adaptive"}` on Haiku 4.5.** A 400 that reads "adaptive thinking is not supported on this model" ([Claude API errors](https://platform.claude.com/docs/en/api/errors)).
- **Editing, reordering or filtering thinking blocks before sending them back.** A 400 saying the blocks cannot be modified; the most common cause is filtering on `type == "thinking"`, which drops `redacted_thinking` blocks.
- **Toggling thinking in the middle of a tool-use loop.** No error: the API silently disables thinking for that request.
- **Asking Claude to write out its internal reasoning in the answer** on Fable 5.1, Opus 5.5 or Fable 5. It can be refused with `stop_details.category: "reasoning_extraction"`; read the thinking blocks instead.

## Tokens, context windows and counting

*Tested in: CCDV-F 5.1 LLM Fundamentals (tokens, context windows, sampling, non-determinism, next-token generation), 5.4 Cost and Token Management (token budgeting and usage tracking), 6.1 Context Engineering · CCAR-P 2.4, 4.5 · CCAO-F D3.4 · CCAR-F 5.1 (5.1-K3, 5.1-K4), Appendix (context window management is in scope; token counting algorithms and tokenization specifics are out of scope)*

Tokens are the unit of everything on the API: what fits in the context window, what counts toward rate limits, and what you pay for.

### What a token is, roughly

A token is the model's smallest unit of text: a word, part of a word, a character or a byte. The docs give rules of thumb that do not agree exactly, so treat them as estimates:

| Source | Rule of thumb |
|---|---|
| Glossary | A token is about 3.5 English characters |
| Pricing FAQ | A token is about 4 characters or 0.75 words in English |
| Models overview | On the current tokenizer (introduced with Claude Opus 4.7), 1M tokens is roughly 555k words or 2.5M Unicode characters; earlier models fit about 750k words in 1M tokens. The page also gives 200k tokens as roughly 150k words without naming a tokenizer, which matches the earlier ratio (our arithmetic) |

The tokenizer changed with Claude Opus 4.7. Claude 4.7 and later models and Claude Mythos Preview use the newer tokenizer, and it produces about 30% more tokens for the same text; Sonnet 5 counts about 30% more than Sonnet 4.6, and Fable 5.1, Mythos 5.1, Fable 5 and Mythos 5 share the same tokenizer. The Opus 5 migration guide gives the range as roughly 1x to 1.35x, varying by content. Sonnet 4.6 and earlier models use the previous tokenizer. Recount a prompt on the model you plan to use rather than reusing an old count. When you need a real number, count it (below).

### Next-token generation, sampling and non-determinism

Claude's underlying model is autoregressive: it was pretrained to predict the next word given the text before it. Anthropic's Academy breaks generation into three steps: tokenization, prediction (a probability for each possible next token) and sampling (choosing one token from those probabilities). The model picks one token, then repeats the whole process to build the response. A separate Academy lesson (Next Token Prediction, in the AI Capabilities and Limitations course) compares generative AI to a very sophisticated autocomplete and says this one property gives both the fluency and the hallucination. It adds that fabrication concentrates in specifics (names, dates, statistics, citations, URLs and quotes), so the more precise a claim, the more it needs verifying.

The sampling step is what the classic parameters control:

| Parameter | What it does | Status as of September 2026 |
|---|---|---|
| `temperature` | Amount of randomness, 0.0 to 1.0, default 1.0. Classic guidance: nearer 0.0 for analytical or multiple-choice work, nearer 1.0 for creative work | Deprecated. On Claude 4.7 and later models (and Mythos Preview), any value other than 1.0 returns a 400 |
| `top_k` | Sample only from the top K options for each next token | Deprecated. Any value returns a 400 on those models |
| `top_p` | Nucleus sampling: cut the cumulative probability distribution off at `top_p` | Deprecated. Values below 0.99 return a 400 on those models |

The recommended replacement is prompting. Version 1.0 and later of the Python SDK removes the three parameters, so passing them raises a `TypeError`. On older models the rules differ: Claude Haiku 4.5 returns a 400 if you set both `temperature` and `top_p`, and while thinking is on, `temperature` and `top_k` are incompatible and `top_p` must be between 0.95 and 1.

Output is not deterministic even at the lowest temperature. The glossary is explicit: "Even with temperature set to 0, the results will not be fully deterministic and identical inputs may produce different outputs across API calls." ([Glossary](https://platform.claude.com/docs/en/about-claude/glossary)). That holds on Anthropic's own API and on third-party cloud platforms alike. Design for variation: validate outputs, and evaluate on more than one run.

!!! warning "Exam guide vs current docs: sampling"

    [CCDV-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) still lists sampling and non-determinism under LLM Fundamentals, so know what `temperature`, `top_k` and `top_p` do. Two things changed or conflict. First, Claude 4.7 and later models reject non-default sampling values, so a current-model code answer omits them. Second, the [Academy's temperature lesson](https://academy.claude.com/courses/building-with-the-claude-api/temperature) calls temperature 0 "completely deterministic", while the API reference and glossary say it is not; answer with the docs. In the four exam guides, temperature appears only as a wrong option: CCDV-F Sample 2 option A (the [rationale](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) says "Temperature (A) is irrelevant to injection") and CCAR-P Sample 3 option C.

### The context window

The context window is everything the model can reference when it generates a response, including the response itself. It works as the model's working memory, and bigger is not automatically better: "As token count grows, accuracy and recall degrade, a phenomenon known as *context rot*." ([Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows)). Curating what goes in matters as much as the size.

**What counts.** The system prompt, every message (including tool results, images and documents), your tool definitions, and the output Claude generates for the turn, thinking included. Cached input still counts: caching changes what you pay for those tokens, not whether they occupy the window. Because the API is stateless, every turn resends the whole history, so a short question late in a long conversation can cost more than the same question in a fresh one. The [Academy](https://academy.claude.com/tutorials/parametric-memory-and-context) draws the practical lesson for app users: starting a new chat "can be the cheapest and fastest way to get an answer."

**Sizes, as of September 2026.**

| Context window | Models |
|---|---|
| 1M tokens (the default, no beta header, standard pricing) | Claude Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Opus 4.6, Sonnet 5, Sonnet 4.6, Mythos Preview |
| 200k tokens | Other models, including Claude Sonnet 4.5, Opus 4.5 and Haiku 4.5 |

A request can carry up to 600 images or PDF pages (100 on 200k-window models), and large payloads can hit the 32 MB request limit before the token limit.

**When it overflows.**

| Situation | What the API does |
|---|---|
| The input alone exceeds the window | 400 `invalid_request_error` with "prompt is too long" ([Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows)), on every model |
| Input plus `max_tokens` exceeds the window, on Claude 4.5 and newer | Accepts the request; if generation reaches the limit, stops with `stop_reason: "model_context_window_exceeded"` |
| The same on earlier models | Validation error, unless you send the `model-context-window-exceeded-2025-08-26` beta header |

**Thinking and the window.** Thinking generated in the current turn counts toward `max_tokens` and the window. Whether earlier turns' thinking stays depends on the model: Opus 4.5 and later Opus, Sonnet 4.6 and later Sonnet, and the Fable and Mythos models keep all prior turns (billed as input), while earlier Opus and Sonnet models and every Haiku through 4.5 keep only the last turn and strip older thinking blocks automatically when you pass them back.

**Context awareness.** Claude Sonnet 5, Sonnet 4.6, Sonnet 4.5 and Haiku 4.5 track their remaining budget: the API injects `<budget:token_budget>` into the system prompt and a `<system_warning>` usage update after each tool call. Claude Opus 4.7 and later Opus models, Fable 5.1, Mythos 5.1, Fable 5 and Mythos 5 get no injected tags; on those models you can give an explicit budget with task budgets (beta) instead. Image tokens are included in the injected budgets.

**Keeping it under control.** For long conversations and agentic work, the docs name server-side compaction (beta, on Claude 4.6 and later) as the primary strategy, with context editing (tool result clearing, thinking block clearing) for finer control. The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) adds the design side: tool results "accumulate in context and consume tokens disproportionately to their relevance (e.g., 40+ fields per order lookup when only 5 are relevant)" (5.1-K3), so trim them before they pile up. Both are taught in [Compaction, context editing and memory](context-engineering.md#compaction-context-editing-and-memory) and [Why context is a budget](context-engineering.md#why-context-is-a-budget).

### Counting tokens before you send

`POST /v1/messages/count_tokens` takes the same input as a message request (system prompt, messages, tools, images, PDFs) and returns the input token count.

=== "Python"

    ```python
    import anthropic

    client = anthropic.Anthropic()

    response = client.messages.count_tokens(
        model="claude-opus-5-5",
        system="You are a scientist",
        messages=[{"role": "user", "content": "Hello, Claude"}],
    )
    print(response.json())  # { "input_tokens": 14 }
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";

    const client = new Anthropic();

    const response = await client.messages.countTokens({
      model: "claude-opus-5-5",
      system: "You are a scientist",
      messages: [{ role: "user", content: "Hello, Claude" }]
    });
    console.log(response); // { input_tokens: 14 }
    ```

| Rule | Detail |
|---|---|
| It is an estimate | The real count can differ slightly. Tokens the system adds for optimization may be included, and you are not billed for them |
| Count on the target model | The endpoint counts with the tokenizer of the `model` you pass |
| It is free, with its own limits | Start 5,000, Build 10,000, Scale 20,000 requests per minute, independent of message-creation limits |
| It ignores caching | `cache_control` is accepted but no caching logic runs |
| Some inputs are rejected | Server tools (except the advisor tool), the MCP connector, and `image` or `document` blocks with a `url` or `file` source: send images and PDFs as base64 to count them |
| Forced tool use still fails | `tool_choice` `any` or `tool` returns a 400 on Opus 5.5, Fable 5.1 and Mythos 5.1 here too |

After the request, the response's `usage` object is the record of what was consumed; see the field table in [The response](#the-response). For requests that use server tools or MCP servers, which the counting endpoint rejects, `usage` is where the numbers come from.

### Decide

- If you must know whether a prompt fits, or what it will cost, before sending, count it on the target model; not a characters-divided-by-four estimate, which is only a rule of thumb and shifts with the tokenizer.
- If a long conversation nears the window, compact or clear old tool results; not a bigger `max_tokens`, which does not free input space.
- If you need reproducible output, validate and constrain it; not `temperature: 0`, which is neither deterministic nor accepted as a non-default value on newer models.
- If a long chat in the Claude apps hits its length limit or nears the usage limit, attach fewer or smaller files, summarize or extract the key sections before sending, or start a new conversation; see [Preserving critical information in long conversations](context-engineering.md#preserving-critical-information-in-long-conversations).

### Traps

- **Assuming cached tokens do not count toward the window.** They still occupy it; caching changes only the price.
- **Reusing token counts from an older model.** Claude 4.7 and later models tokenize the same text into about 30% more tokens than earlier models.
- **Counting a request with a URL image or a `file_id` PDF.** The endpoint rejects them; send base64.
- **Believing a larger window always helps.** Context rot means recall degrades as the window fills.

## Vision and PDF input

*Tested in: CCDV-F 2.3 Claude API Mechanics (vision), Section 2 applied area (multi-format input) · Out of scope for CCAR-F (Appendix: vision and image analysis)*

Every current Claude model accepts images, and every active model can read PDFs. Both arrive as content blocks inside a user message, next to your text: `image` blocks for pictures and `document` blocks for PDFs. Claude understands images; it cannot generate or edit them.

### Sending an image

An `image` block has a `source` of one of three types. On Amazon Bedrock and Google Cloud only base64 is available.

| `source.type` | You send | Use it when (our rule of thumb) |
|---|---|---|
| `base64` | `media_type` and base64 `data` in the request body | The image is local and used once |
| `url` | A `url` pointing to an image hosted online | The image is already hosted online |
| `file` | A `file_id` from the Files API | The same image is reused across requests or turns (the docs: upload once, reference many times) |

```json
{
  "model": "claude-opus-5-5",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image",
          "source": {
            "type": "url",
            "url": "https://platform.claude.com/docs/images/vision-example.jpg"
          }
        },
        {"type": "text", "text": "Describe this image."}
      ]
    }
  ]
}
```

For base64, the source becomes `{"type": "base64", "media_type": "image/jpeg", "data": "..."}`. Supported formats are JPEG, PNG, GIF and WebP (`image/jpeg`, `image/png`, `image/gif`, `image/webp`); animated images use only the first frame. Put images before the text that asks about them: "Claude works best when images come before text." ([Vision](https://platform.claude.com/docs/en/build-with-claude/vision)).

Choose `file_id` for images you reuse. Because the API is stateless, every request resends the conversation; base64 images travel in full on every turn, which inflates payload size and latency, while a `file_id` keeps the request small however many images pile up. The Files API itself is covered in [Files, citations and search results](#files-citations-and-search-results).

### Image limits

| Limit | Value |
|---|---|
| Images per API request | 600, or 100 on models with a 200k-token context window |
| Images per message on claude.ai | 20 |
| Maximum dimensions | 8000x8000 px |
| More than 20 images in one request | A stricter per-image dimension limit applies to every image, including resent images from earlier turns and images inside `tool_result` content; keep each side at 2000 px or less, or send 20 or fewer image and document blocks |
| Maximum size per image | 10 MB base64 on the Claude API, 5 MB on Amazon Bedrock and Google Cloud |
| Request size | 32 MB on the Messages API, which many large images can reach before the image count |

### What an image costs

Claude sees an image as 28x28-pixel patches called visual tokens, so an image costs `⌈width / 28⌉ × ⌈height / 28⌉` visual tokens. Images larger than the model's limits are scaled down, preserving aspect ratio, which caps the cost. One exception: a screenshot or zoom image returned in a `tool_result` to the computer use or browser use toolsets that exceeds the model's limits is rejected with a validation error instead of downscaled, so resize those images in your application before returning them.

| Resolution tier | Models | Max long edge | Max visual tokens |
|---|---|---|---|
| High-resolution | Claude 4.7 and later | 2576 px | 4784 |
| Standard | All other models | 1568 px | 1568 |

Worked examples from the docs: a 1000x1000 image costs 1296 tokens on either tier; a 1920x1080 image costs 1560 tokens on the standard tier (downsized to 1456x819) and 2691 on the high-resolution tier (not resized). High-resolution images can use up to roughly three times the visual tokens, so downsample when you do not need the extra fidelity for computer use, screenshots or dense documents. Images under 200 px are more error-prone.

### What vision cannot do

- Name people in images: under the Acceptable Use Policy Claude cannot be used for this, and it refuses.
- Guarantee accuracy on low-quality, rotated or very small images (under 200 px).
- Give exact spatial answers or counts: coordinates and object counts are approximate, especially for many small objects.
- Tell whether an image is AI-generated.
- Replace professional diagnosis: it can analyze general medical images but is not designed to interpret complex diagnostic scans such as CTs or MRIs.
- Process inappropriate or explicit images that violate the Acceptable Use Policy.
- Read image metadata, or generate and edit images.

For high-stakes image work, the docs say to review Claude's interpretations and not to rely on it for tasks needing perfect precision without human oversight.

### Sending a PDF

A PDF goes in a `document` block. Its `source` can be a `url`, `base64` data with `media_type: "application/pdf"`, or a Files API `file_id`. On Amazon Bedrock and Google Cloud only base64 is available, and on Microsoft Foundry the Files API is not supported for deployments hosted on Azure.

```json
{
  "model": "claude-opus-5-5",
  "max_tokens": 1024,
  "messages": [{
    "role": "user",
    "content": [
      {"type": "document", "source": {"type": "url", "url": "https://assets.anthropic.com/m/1cd9d098ac3e6467/original/Claude-3-Model-Card-October-Addendum.pdf"}},
      {"type": "text", "text": "What are the key findings in this document?"}
    ]
  }]
}
```

| Requirement | Limit |
|---|---|
| Request size | 32 MB, lower on some platforms; the limit covers the whole payload, not just the PDF |
| Pages per request | 600, or 100 when the request's context window is under 1M tokens |
| Format | Standard PDF, with no password or encryption |
| Other file types | `.docx`, `.xlsx` and other binary formats are not supported in document blocks; convert them to text or PDF first. Plain text files (`.txt`, `.csv`, `.md`) can go in document blocks directly: upload them to the Files API with MIME type `text/plain` and reference the `file_id` |

How it works: the system converts each page into an image and extracts the page's text, and Claude receives both. That is why Claude can answer questions about charts, diagrams and tables, and why a PDF costs text tokens (typically 1,500 to 3,000 per page, depending on density) plus image tokens for every page, with no separate PDF fee. Because PDF reading relies on vision, it inherits the vision limitations above. Dense PDFs can fill the context window before reaching the page limit.

The docs' practices for better results: place PDFs before your text, use standard fonts and legible text, rotate pages upright, refer to logical page numbers (as a PDF viewer shows them), split large PDFs into chunks, and enable prompt caching for repeated analysis of the same document. For high volumes, process documents with [Message Batches](#message-batches). To estimate a PDF's cost before sending, count it with the token counting endpoint, sending the PDF as base64.

Platform note: on Amazon Bedrock's Converse API (part of the legacy Bedrock integration for Opus 4.6 and earlier), full visual PDF analysis requires citations to be enabled; without them, only text is extracted. See [Claude on cloud platforms](#claude-on-cloud-platforms).

### Decide

- If the same image or PDF appears in many requests or turns, upload it once and reference the `file_id`; not base64 on every turn, which resends the bytes each time.
- If a request fails on many images, check the more-than-20-images dimension rule and the 32 MB request limit before the 600-image count.
- If a Word or Excel file must be analyzed, convert it to text or PDF; not a `document` block with the raw file.
- If costs are high on Claude 4.7 and later and you do not need fine detail, downsample images before sending.

### Traps

- **Expecting Claude to create or edit an image.** It only understands images.
- **Asking Claude to identify a person in a photo.** It refuses.
- **Counting a URL or `file_id` image with the token counting endpoint.** Send base64 instead.
- **Assuming a PDF is read as text only.** On the Claude API each page is also seen as an image, and billed as one. (The exception is Bedrock's Converse API without citations enabled, which falls back to text extraction.)

## Prompt caching

*Tested in: CCDV-F 2.3 Claude API Mechanics, 5.4 Cost and Token Management, Section 2 applied area (managing tokens, caching and batch processing) · CCAR-P 2.4, 2.5, 4.5, Sample 2 (Domain 2) · CCAR-F: know that it exists; implementation details are out of scope · CCAO-F: not listed*

Prompt caching lets a request reuse the already-processed start of a prompt instead of processing it again. The API checks whether the prompt prefix up to a cache breakpoint was cached by a recent request. If it was, that prefix is read from the cache; if not, the full prompt is processed and the prefix is cached once the response begins. The payoff is lower cost and a faster first token. Two things do not change: the output, and the size of the context. In the docs' words, "Prompt caching doesn't reduce the number of tokens in context, but it reduces what you pay for them on subsequent requests." ([Manage tool context](https://platform.claude.com/docs/en/agents-and-tools/tool-use/manage-tool-context))

### How a cache hit works

- **Order is fixed.** The cached prefix is the whole prompt in the order `tools`, `system`, `messages`, up to and including the block marked with `cache_control`.
- **One write per breakpoint.** Marking a block writes exactly one cache entry: a hash of the prefix ending at that block. Keys are cryptographic hashes, so only an identical prompt can read an entry.
- **Byte-identical or nothing.** A hit needs a 100% identical prefix, including all text and images, up to and including the marked block.
- **Per model.** A router, A/B test or fallback that sends the request to a different model misses the cache.
- **Per workspace.** Caches are isolated per workspace on the Claude API, Claude Platform on AWS and Microsoft Foundry, and per organization on Amazon Bedrock and Google Cloud. They are never shared across organizations.
- **Not instant.** An entry becomes readable only after the first response begins, so if parallel requests need cache hits on a shared prefix, wait for the first response before sending the rest.
- **One cache type.** `"ephemeral"` is the only type. Caching works on all active models and no longer needs a beta prefix.

Illustration (ours), with a breakpoint on the last block of each request:

```text
Request 1:  [tools][system prompt + policy][turn 1][turn 2 <- cache_control]
            |<------------ processed, then written to the cache ----------->|

Request 2:  [tools][system prompt + policy][turn 1][turn 2][turn 3][turn 4 <- cache_control]
            |<------ lookback finds request 1's entry: read from cache ---->|<- processed, written ->|
```

### Turning it on: automatic or explicit

| | Automatic caching | Explicit breakpoints |
|---|---|---|
| How | One top-level `cache_control` field on the request | `cache_control` on individual content blocks |
| Where the breakpoint lands | On the last cacheable block, moving forward as the conversation grows | Exactly on the blocks you mark |
| Slots | Uses one of the 4 breakpoint slots when combined with explicit breakpoints | Up to 4 per request |
| Good first choice for | Multi-turn conversations | Sections that change at different frequencies |

Automatic caching works on every platform except the legacy Amazon Bedrock integration (Opus 4.6 and earlier), which returns 400 for a top-level `cache_control`. It shares pricing, minimum lengths, ordering rules and the 20-block lookback with explicit breakpoints. Its edge cases: an explicit marker with the same TTL on the last block makes it a no-op; a different TTL on the last block returns 400; four explicit breakpoints already in the request return 400 (no slot left); if the last block cannot take a breakpoint, the API walks backward to the nearest eligible block, and skips caching if none exists.

=== "Python"

    ```python
    import anthropic

    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=1024,
        cache_control={"type": "ephemeral"},
        system="You are a helpful assistant that remembers our conversation.",
        messages=[
            {"role": "user", "content": "My name is Alex. I work on machine learning."},
            {
                "role": "assistant",
                "content": "Nice to meet you, Alex! How can I help with your ML work today?",
            },
            {"role": "user", "content": "What did I say I work on?"},
        ],
    )
    print(response.usage.model_dump_json())
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";

    const client = new Anthropic();

    const response = await client.messages.create({
      model: "claude-opus-5-5",
      max_tokens: 1024,
      cache_control: { type: "ephemeral" },
      system: "You are a helpful assistant that remembers our conversation.",
      messages: [
        { role: "user", content: "My name is Alex. I work on machine learning." },
        {
          role: "assistant",
          content: "Nice to meet you, Alex! How can I help with your ML work today?"
        },
        { role: "user", content: "What did I say I work on?" }
      ]
    });
    console.log(response.usage);
    ```

An explicit breakpoint goes on the block that ends your reusable content. This request combines both methods and so uses 2 of the 4 slots:

```json
{
  "model": "claude-opus-5-5",
  "max_tokens": 1024,
  "cache_control": { "type": "ephemeral" },
  "system": [
    {
      "type": "text",
      "text": "You are a helpful assistant.",
      "cache_control": { "type": "ephemeral" }
    }
  ],
  "messages": [{ "role": "user", "content": "What are the key terms?" }]
}
```

To cache tool definitions, put the marker on the **last** tool in `tools`; that caches the whole tool-definition prefix. For an `mcp_toolset` entry, put the breakpoint on the entry itself and the API applies it to the final expanded tool.

```json
{
  "tools": [
    {
      "name": "get_weather",
      "description": "Get the current weather in a given location",
      "input_schema": {
        "type": "object",
        "properties": { "location": { "type": "string" } },
        "required": ["location"]
      }
    },
    {
      "name": "get_time",
      "description": "Get the current time in a given time zone",
      "input_schema": {
        "type": "object",
        "properties": { "timezone": { "type": "string" } },
        "required": ["timezone"]
      },
      "cache_control": { "type": "ephemeral" }
    }
  ]
}
```

### Where breakpoints go, and the 20-block lookback

The placement rule from the docs: "Place `cache_control` on the last block whose prefix is identical across the requests you want to share a cache." ([Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)) Put static content (tool definitions, system instructions, context, examples) at the start of the prompt and mark the end of it. The classic mistake is a breakpoint on a block that changes on every request, such as one containing a timestamp: "You pay for a fresh cache write on every request and never get a read." ([Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching))

Anthropic's cost guide measured what one small dynamic line does: a 25-token status line at the front of the system prompt made a run cost &#36;4.24 instead of &#36;0.59. Its fix is one sentence: "Keep per-request text in the newest user turn." ([Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence)) The guide describes its measurements as Anthropic-internal and directional.

How reads find earlier writes:

- Writes happen only at breakpoints. Reads walk backward one block at a time looking for an entry an earlier request wrote, checking at most 20 positions per breakpoint (the breakpoint itself counts as the first).
- On the Claude API a run of consecutive `tool_use` blocks counts as one position, and so does a run of consecutive `tool_result` blocks.
- Use more than one breakpoint when sections change at different frequencies, when you want finer control, or when a growing conversation pushes the breakpoint 20 or more blocks past the last write.
- Breakpoints themselves cost nothing, and in most cases a single breakpoint at the end of the static content is enough.
- The docs' lookback example: turn 1 has 10 blocks and writes at block 10; turn 2 has 15 blocks, walks back from block 15 and hits the block-10 entry, then writes at block 15; turn 3 has 35 blocks, checks blocks 35 to 16 and misses, because the block-15 entry is one position outside the window. A second breakpoint at block 15 would have found it.

### TTL: 5 minutes or 1 hour

| | 5-minute (default) | 1-hour |
|---|---|---|
| Marker | `{"type": "ephemeral"}` (`"ttl"` defaults to `"5m"`) | `{"type": "ephemeral", "ttl": "1h"}` |
| Write price | 1.25x base input | 2x base input |
| Pays for itself after | One cache read | Two cache reads |
| Latency | Same as 1-hour | Same as 5-minute |
| Use when | The prompt is reused more often than every 5 minutes | Reuse is less often than every 5 minutes but more often than hourly (side agents that run longer than 5 minutes, users who may not reply within 5 minutes); latency matters and follow-ups may come after 5 minutes; or you want better rate-limit utilization, because cache hits are not deducted against the rate limit |

Timing details that decide questions:

- The default 5-minute entry is refreshed at no additional cost each time the cached content is used, which is why frequent reuse should stay on it.
- The lifetime runs from the **start** of the request that writes or reads the entry, not from the end of its response. A 4-minute streamed response leaves about 1 minute of a 5-minute entry.
- There is no way to clear the cache manually; entries expire after at least 5 minutes of inactivity.
- When one request mixes TTLs, 1-hour entries must come before 5-minute entries. Billing then uses three positions: cache reads up to the highest cache hit (A), 1-hour write prices from A to the last 1-hour breakpoint (B), and 5-minute write prices from B to the last breakpoint (C).
- The 1-hour duration is available on the Claude API, Amazon Bedrock (both integrations), Claude Platform on AWS, Google Cloud and Microsoft Foundry.
- Anthropic's cost guide gives a rule of thumb: "More than about 1 gap in 20 falls between 5 minutes and an hour, and gaps over an hour are rare: use the 1-hour duration." Stay on the 5-minute default when turns are seconds apart or hour-plus gaps are common ([Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence)). The guide measured this on Claude Sonnet 5 and Claude Opus 5 and extends it to the models with the standard cache read price. Claude Fable 5.1, whose cache reads cost 0.025x, gets different advice: "On Claude Fable 5.1, keep the 5-minute cache warm while a person is away for minutes, and buy the 1-hour duration when pauses run toward an hour".

### Minimum cacheable length (as of September 2026)

| Minimum | Models |
|---|---|
| 512 tokens | Claude Fable 5.1, Mythos 5.1, Opus 5.5, Opus 5, Fable 5, Mythos 5 |
| 1,024 tokens | Claude Opus 4.8, Sonnet 5, Sonnet 4.6, Sonnet 4.5, Opus 4.1, Opus 4, Sonnet 4 (the last three retired except on some cloud platforms) |
| 2,048 tokens | Claude Mythos Preview, Opus 4.7, Haiku 3.5 (retired except on Bedrock and Google Cloud) |
| 4,096 tokens | Claude Opus 4.6, Opus 4.5, Haiku 4.5 |

A prompt shorter than the minimum is processed without caching and **no error is returned**. If both `cache_creation_input_tokens` and `cache_read_input_tokens` are 0, nothing was cached. If a prompt falls just short of its model's minimum, the docs note that expanding the cached content to reach it is often worthwhile, because reads cost much less than uncached input. The table's minimums are for the Claude API, Claude Platform on AWS, Google Cloud and Microsoft Foundry; on Bedrock, the per-model minimums and usage-field names follow AWS's own prompt caching documentation. The minimum has moved between releases (Opus 4.8 dropped to 1,024 from Opus 4.7's 2,048), so do not memorize one number for "Claude".

### What breaks the cache

The cache follows the hierarchy `tools`, then `system`, then `messages`, and a change at one level invalidates that level and every level after it.

| Change | What stays cached |
|---|---|
| Tool definitions (names, descriptions, parameters), or adding, removing or reordering a tool | Nothing |
| Turning web search, web fetch or citations on or off; switching between `speed: "fast"` and standard speed | Tools only |
| Changing `tool_choice`; adding or removing images anywhere; changing `disable_parallel_tool_use` | Tools and system |
| Changing thinking parameters or `output_config.effort` | Messages are always invalidated; tools and system too on models that render that configuration ahead of them |
| Setting or changing `output_config.format` (structured outputs) | Nothing, for that conversation |
| Changing the Skills list in the `container` | Not stated; the Skills guide says only that the change breaks the cache, because Skills render into the system prompt |
| Changing a task budget partway through a task | Invalidates any cached prefix that contains the budget value; the cost guide says to set it once, on the first request |
| Sending the request to a different model | Nothing (the cache is per model) |

Changes the docs say keep the cache intact:

- Setting `effort` explicitly to the model's default (equivalent to omitting it). On models that support per-message effort, an effort change carried in a `role: "system"` message inside `messages` also leaves the cached prefix intact.
- On Claude Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Opus 5.5, Opus 4.8 and Opus 5, appending a `{"role": "system"}` message to `messages` instead of editing the top-level `system` field. This is not available on Claude Sonnet 5.
- Tools discovered through tool search (`defer_loading`): they are appended inline as `tool_reference` blocks, not added to the prefix.
- With the `inline-tools-2026-09-15` beta header, adding or changing a tool through a `tool_addition` block in a mid-conversation system message, leaving `tools` unchanged. The one exception: if `tools` has no non-deferred tool, the first tool defined this way costs one full cache miss on that request.
- On Opus 4.5+ and Sonnet 4.6+, adding non-tool-result user content after thinking turns (thinking blocks are preserved). On earlier Opus and Sonnet models and all Haiku models, prior thinking blocks are stripped and the cache after them is invalidated.

Hygiene that prevents silent misses: keep the system prompt a byte-stable constant and move dynamic data into the first user message after the breakpoint; treat history as append-only and echo assistant content and tool results back verbatim; keep JSON key order stable in `tool_use` blocks (some languages, such as Swift and Go, randomize it).

### Prices and break-even

| Token type | Multiplier on the base input price |
|---|---|
| 5-minute cache write | 1.25x |
| 1-hour cache write | 2x |
| Cache read (hit) | 0.1x on most models; 0.05x on Claude Opus 5.5; 0.025x on Claude Fable 5.1 and Claude Mythos 5.1 |

These multipliers stack with the Batch API discount and with data residency pricing. The break-even follows from the numbers: two uncached sends of a prefix cost 2.0x, while a 5-minute write plus one read costs 1.25x + 0.1x = 1.35x; with the 1-hour duration, one read gives 2.1x (worse) but two reads give 2.2x against 3.0x (better). That is why the pricing page says caching pays off after one read for 5 minutes and after two reads for 1 hour (break-even arithmetic ours, at the standard 0.1x read rate).

!!! example "Worked example: an 8,000-token static prefix on Claude Sonnet 5"

    Claude Sonnet 5 prices (as of September 2026): &#36;2 per million input tokens, &#36;2.50 per million for 5-minute cache writes, &#36;0.20 per million for cache hits. Its minimum cacheable length is 1,024 tokens, so an 8,000-token prefix qualifies. For 1,000 requests that share the prefix and arrive while the entry stays warm (our arithmetic):

    - Without caching: 8,000 x 1,000 = 8,000,000 prefix tokens x &#36;2 / 1,000,000 = &#36;16.00.
    - With caching: one write, 8,000 x &#36;2.50 / 1,000,000 = &#36;0.02, plus 999 reads, 7,992,000 x &#36;0.20 / 1,000,000 = &#36;1.5984. Total about &#36;1.62, roughly 90% less for the prefix.
    - The short varying user message after the breakpoint is billed at the normal input rate in both cases.

    This is the shape of the CCAR-P Sample 2 scenario (a repeated 8,000-token system prompt and policy document followed by a short, varying user message).

Caching also stretches rate limits, because on most models cache reads do not count toward input tokens per minute; the rule and the docs' worked example are in [How rate limits work](#how-rate-limits-work).

!!! note "Older material says cache reads cost 10%"

    Earlier Anthropic material described cache reads as "costing only 10% of the base input token price" ([prompt caching announcement](https://claude.com/blog/prompt-caching)). As of September 2026 that still holds for most models, but Claude Opus 5.5 hits cost 5% and Claude Fable 5.1 and Mythos 5.1 hits cost 2.5% of the base input price.

### Reading the usage fields

The cache fields are defined in the `usage` table under [The response](#the-response). Two details matter here: the TTL split under `cache_creation` (`ephemeral_5m_input_tokens` and `ephemeral_1h_input_tokens`) sums to `cache_creation_input_tokens`, and when streaming the same fields arrive in the `message_start` event. The docs' mixed-TTL example response read 1,800 tokens, wrote 248 (148 with a 5-minute TTL, 100 with a 1-hour TTL) and processed 2,048 uncached, for 4,096 input tokens in total:

```json
{
  "usage": {
    "input_tokens": 2048,
    "cache_read_input_tokens": 1800,
    "cache_creation_input_tokens": 248,
    "output_tokens": 503,

    "cache_creation": {
      "ephemeral_5m_input_tokens": 148,
      "ephemeral_1h_input_tokens": 100
    }
  }
}
```

The docs' larger example: 100,000 tokens read from cache, 0 written and 50 new tokens is 100,050 input tokens processed. If you see 5-minute writes you did not ask for while using server tools such as web search, that is expected: when a request already carries at least one `cache_control` marker, the API places an automatic breakpoint on server tool results, always with the 5-minute TTL. [Worked example 1](#worked-example-1-pricing-a-response-from-its-usage-object) in Cost and usage tracking prices the response above line by line.

### Keeping the cache warm

A request with `max_tokens: 0` writes the cache at its breakpoints and returns an empty `content` array with `stop_reason: "max_tokens"` and a full `usage` block. It is billed a cache write if the prefix was not already cached, and zero output tokens. Use an explicit breakpoint on the last shared block (automatic caching would key the entry to the placeholder user message), and keep the same thinking configuration and effort as real traffic.

```python
import anthropic

client = anthropic.Anthropic()

# Fire this before users arrive to warm the shared system-prompt cache.
prewarm = client.messages.create(
    model="claude-opus-5-5",
    max_tokens=0,
    system=[
        {
            "type": "text",
            "text": "You are an expert software engineer with deep knowledge of distributed systems...",
            "cache_control": {"type": "ephemeral"},
        }
    ],
    messages=[{"role": "user", "content": "warmup"}],
)
print(prewarm.stop_reason)  # "max_tokens"
print(prewarm.content)  # []
print(prewarm.usage)
```

`max_tokens: 0` is rejected with `invalid_request_error` together with `stream: true`, extended thinking (`thinking.type: "enabled"`), structured outputs (`output_config.format`) or a `tool_choice` of type `tool` or `any`, and inside a Message Batches request. The docs prefer it over the older `max_tokens: 1` warm-up workaround: there is no single-token reply to discard and no output is billed. With the 5-minute default, the prompt caching docs say to re-send the pre-warm at least every 5 minutes; Anthropic's cost guide gives tighter timing, "within 4 minutes of the previous request's start, and every 4 minutes after that", because the lifetime runs from the start of the request, so generation time counts against it ([Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence)). For longer gaps use the 1-hour TTL.

### Finding out why you missed

A healthy baseline from Anthropic's cost guide: agent loops read a median 84% of their input from cache, and "Below about 80%, look for something breaking the cache" ([Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence)). The Console's "Rate Limit - Input Tokens" chart shows your cache rate. Before debugging, check the usual suspects: `tool_choice`, image use, the thinking configuration and `output_config.effort` must stay the same between calls, and the next call must land within the cache lifetime.

**Cache diagnostics** (beta header `cache-diagnosis-2026-04-07`, Claude API only) compares consecutive requests and reports where the prefix diverged. Pass the previous response's `id` as `diagnostics.previous_message_id` (on the first turn, pass `null` to opt in).

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = "You are an AI assistant analyzing a large document. <document>...</document>"

# Turn 1: opt in with previous_message_id=None
r1 = client.beta.messages.create(
    model="claude-opus-5-5",
    max_tokens=1024,
    cache_control={"type": "ephemeral"},
    system=SYSTEM,
    messages=[{"role": "user", "content": "Summarize section 1."}],
    diagnostics={"previous_message_id": None},
    betas=["cache-diagnosis-2026-04-07"],
)

# Turn 2: reference the previous response id
r2 = client.beta.messages.create(
    model="claude-opus-5-5",
    max_tokens=1024,
    cache_control={"type": "ephemeral"},
    system=SYSTEM,
    messages=[
        {"role": "user", "content": "Summarize section 1."},
        {"role": "assistant", "content": r1.content},
        {"role": "user", "content": "Now summarize section 2."},
    ],
    diagnostics={"previous_message_id": r1.id},
    betas=["cache-diagnosis-2026-04-07"],
)
diagnostics = r2.diagnostics
if diagnostics is None:
    print("No divergence detected.")
elif diagnostics.cache_miss_reason is None:
    print("Comparison still pending.")
else:
    print(f"cache_miss_reason: {diagnostics.cache_miss_reason.type}")
```

| `cache_miss_reason.type` | What to fix |
|---|---|
| `model_changed` | A router, A/B test or fallback picked a different model, and the cache is per model: hold the model constant within a cached conversation |
| `system_changed` | Make the system prompt a byte-stable constant; move dynamic data into the first user message after the breakpoint |
| `tools_changed` | Send the same tool list on every turn, in a fixed order, with deterministically serialized schemas (for example, sorted keys) |
| `messages_changed` | Treat history as append-only; echo assistant content and tool results back verbatim |
| `previous_message_not_found` | No comparison was produced (no stored fingerprint for that id): send the beta header on every turn and keep consecutive turns close together |
| `unavailable` | No comparison was produced, for example because `tool_choice`, `thinking`, `context_management`, `output_config` or the set of beta headers differs, or the change is beyond the comparison horizon: keep those parameters constant for the conversation |

Only the earliest divergence is reported, so fix it first; later ones may be hidden behind it. On a turn where you passed a real `previous_message_id`, diagnostics of `null` with low or zero reads means the requests matched but the entry was no longer available: shorten the gaps or use the 1-hour TTL. (On the first turn, with `previous_message_id: null`, diagnostics is always `null`.) Fingerprints hold only hashes and token-count estimates, never prompt content.

### Caching alongside other features

- **Thinking:** thinking blocks cannot carry `cache_control`, but they are cached with the rest of a previous assistant turn and count as input tokens when read.
- **Citations and documents:** citation sub-blocks cannot be cached directly; cache the top-level `document` block. Empty text blocks cannot be cached.
- **Message Batches:** discounts stack, but hits are best effort (typical batch hit rates range from 30% to 98%, depending on traffic patterns). See [Message Batches](#message-batches).
- **Context editing:** clearing tool results invalidates the cached prefix from the clearing point, so clear in a few large batches rather than many small ones. See [Compaction, context editing and memory](context-engineering.md#compaction-context-editing-and-memory).
- **Compaction:** put a breakpoint at the end of the system prompt so it stays cached when compaction happens.
- **Data retention:** prompt caching is ZDR eligible and HIPAA eligible; KV cache representations and hashes are held in memory only, not at rest, and deleted promptly after the TTL expires.

### How the exams frame it

CCAR-P Sample 2 describes an application that sends the same 8,000-token system prompt and policy document on every request, followed by a short, varying user message, and asks which optimization addresses both latency and cost. The answer is C, static content first plus prompt caching, and the rationale reads: "Ordering stable content first and enabling prompt caching lets repeated prefixes be reused, reducing both time-to-first-token and per-request cost without discarding required context." The wrong options fail for stated reasons: "Truncation (A) loses needed policy; downsizing blindly (B) risks quality; relocating to few-shot (D) does not create a cacheable, reusable prefix." ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf); full item on the [CCAR-P page](../claude-certified-architect-professional.md#official-sample-questions))

### Decide

- If the same large block goes out on every request, put it first and cache it; not truncation, a blindly smaller model or moving it into few-shot examples, because only a stable prefix is reusable.
- If something changes per request (a timestamp, a user name, retrieved snippets), put it after the breakpoint, in the newest user turn; not in the system prompt.
- If more than about 1 gap in 20 between requests falls between 5 minutes and an hour, and hour-plus gaps are rare, choose the 1-hour TTL; if requests are seconds apart or hour-plus gaps are common, stay on 5 minutes, which refreshes for free on every use. On Claude Fable 5.1, the cost guide says instead to keep the 5-minute cache warm with `max_tokens: 0` requests while pauses last minutes, and to buy the 1-hour duration when pauses run toward an hour.
- If you fan out many parallel requests over one prefix, let the first response begin (or pre-warm) before sending the rest.

### Traps

- Expecting caching to shrink the context window or change the output. It does neither.
- Assuming one global minimum (such as 1,024 tokens) for every model, then wondering why a short prompt reports zero cache tokens without an error.
- Switching `tool_choice`, effort or the output format mid-conversation and treating the resulting misses as a bug.
- Expecting a cache written by one model to serve another.

!!! warning "Exam guide vs current docs"

    The CCDV-F guide lists "caching techniques (prompt caching, cache check-pointing)" under Cost and Token Management ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). The guide does not define the term, and the prompt caching docs do not use it; the documented mechanism is the cache breakpoint (`cache_control`, up to 4 per request, automatic or explicit, 5-minute or 1-hour TTL). Our reading: because the guide lists it among caching techniques for cost optimization, treat "cache check-pointing" as choosing where the cache breakpoints go. Do not confuse it with Claude Code checkpointing, a different feature that captures code state before each prompt so you can rewind. CCAR-F lists "Prompt caching implementation details (beyond knowing it exists)" as out of scope, so for that exam know what caching is for and that static content goes first.

## Message Batches

*Tested in: CCAR-F 4.5 (4.5-K1 to 4.5-S4), Exercise 3 step 4, sample question 11; batch rate limits, cloud provider availability and caching details are out of scope for CCAR-F (Appendix) · CCDV-F 2.3 Claude API Mechanics (batch API use, realtime vs batch), Sample 1, Section 2 applied area (managing tokens, caching and batch processing) · CCAR-P 4.5, 1.6 · CCAO-F: not listed*

The Message Batches API takes many ordinary Messages requests in one call and processes them asynchronously. Most batches finish in less than 1 hour, all usage is charged at 50% of standard prices, and results become available when every request has finished or after 24 hours, whichever comes first. A batch that has not finished within 24 hours expires. The CCAR-F guide condenses this to "50% cost savings, up to 24-hour processing window, no guaranteed latency SLA" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)), and that sentence is the one to carry into the exam.

### Batch or realtime

| Workload | Use | Why |
|---|---|---|
| Blocking: a person, a merge gate or another system is waiting on the answer | Messages API (realtime) | A batch can take up to 24 hours and has no latency guarantee |
| Non-blocking and latency-tolerant: overnight reports, weekly audits, nightly test generation | Message Batches | Half the price, and nobody is waiting |
| The request needs streaming or fast mode (`speed`) | Messages API | Both are rejected inside a batch |
| Zero data retention is required | Messages API | Batch processing is not ZDR eligible (29-day retention, async storage required) |
| Your own tool must run mid-request and feed its result back | Messages API agent loop | A single batch request cannot execute your tools and continue |

Anthropic's cost guide gives the routing rule: "Route every request no one is waiting on through a batch, and keep the interactive path for the rest." ([Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence)) Where batching ranks among the guide's other cost levers is in [Where the savings are](#where-the-savings-are).

### Limits and rules (as of September 2026)

| Item | Value |
|---|---|
| Size of one batch | 100,000 requests or 256 MB, whichever is reached first; an oversized batch may get 413 `request_too_large` |
| `custom_id` | Required, unique, 1 to 64 characters, `^[a-zA-Z0-9_-]{1,64}$` |
| `params` | A normal Messages API request body |
| `max_tokens` | At least `1` per request. The `output-300k-2026-03-24` beta raises the cap to 300,000 for batch requests on Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Opus 4.6, Sonnet 5 and Sonnet 4.6 |
| Rejected in a batch | `stream: true`, `speed` (fast mode), `max_tokens: 0` (validation error). `max_tokens: 0` is the cache pre-warming pattern, and a cache entry written during batch processing would likely expire before the follow-up request runs |
| Processing | Most batches under 1 hour; `expires_at` is 24 hours after creation. Processing can slow down with current demand and your request volume, and then more requests may expire at 24 hours |
| Results kept | 29 days from `created_at` (not from `ended_at`); after that the batch is visible but results cannot be downloaded |
| Price | 50% of standard prices for input, output and any special tokens |
| Scope | One workspace; batches may go slightly over a workspace spend limit |
| Models | All active models |
| Rate limits | Separate from Messages API limits (batch usage does not count against them) and shared across models: an RPM limit on all Batches API endpoints plus a cap on batch requests waiting in the processing queue |
| Platforms | Claude API and Claude Platform on AWS; Bedrock, Google Cloud and Microsoft Foundry list the Message Batches API as unsupported |

The queue cap counts individual requests inside batches, not batches: a request stays in the processing queue until the model has successfully processed it.

| Usage tier | Batches API requests per minute (all endpoints) | Batch requests in the processing queue | Batch requests per batch |
|---|---|---|---|
| Start | 1,000 | 200,000 | 100,000 |
| Build | 2,000 | 300,000 | 100,000 |
| Scale | 4,000 | 500,000 | 100,000 |

A batch can carry almost anything a single request can: vision, tool use including all server tools, system messages, multi-turn conversations, extended thinking and most beta features. Each request is processed independently, so different kinds of request can share a batch, and each can set its own `inference_geo`.

### Lifecycle and endpoints

```text
POST /v1/messages/batches
        |
        v
processing_status: "in_progress" --- POST .../{id}/cancel ---> "canceling"
        |                                                          |
        | every request finished, or 24 hours passed              |
        v                                                          v
processing_status: "ended" <---------------------------------------+
        |
        v
GET .../{id}/results  (.jsonl, one result per line, in any order)
        |
        v
29 days after created_at: results can no longer be downloaded
```

| Operation | Endpoint | Notes |
|---|---|---|
| Create | `POST /v1/messages/batches` | Returns the batch object with its `id` |
| Retrieve | `GET /v1/messages/batches/{message_batch_id}` | Idempotent; use it to poll |
| List | `GET /v1/messages/batches` | Newest first |
| Cancel | `POST /v1/messages/batches/{message_batch_id}/cancel` | Moves to `canceling`; in-progress, non-interruptible requests may still complete |
| Results | `GET /v1/messages/batches/{message_batch_id}/results` | Streams the `.jsonl` file |
| Delete | `DELETE /v1/messages/batches/{message_batch_id}` | Only after processing has finished; cancel an in-progress batch first |

Create a batch:

=== "Python"

    ```python
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()

    message_batch = client.messages.batches.create(
        requests=[
            Request(
                custom_id="my-first-request",
                params=MessageCreateParamsNonStreaming(
                    model="claude-opus-5-5",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "Hello, world"}],
                ),
            ),
            Request(
                custom_id="my-second-request",
                params=MessageCreateParamsNonStreaming(
                    model="claude-opus-5-5",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": "Hi again, friend"}],
                ),
            ),
        ]
    )
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";

    const client = new Anthropic();

    const messageBatch = await client.messages.batches.create({
      requests: [
        {
          custom_id: "my-first-request",
          params: {
            model: "claude-opus-5-5",
            max_tokens: 1024,
            messages: [{ role: "user", content: "Hello, world" }]
          }
        },
        {
          custom_id: "my-second-request",
          params: {
            model: "claude-opus-5-5",
            max_tokens: 1024,
            messages: [{ role: "user", content: "Hi again, friend" }]
          }
        }
      ]
    });

    console.log(messageBatch);
    ```

The batch object right after creation shows the fields you poll on:

```json
{
  "id": "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d",
  "type": "message_batch",
  "processing_status": "in_progress",
  "request_counts": {
    "processing": 2,
    "succeeded": 0,
    "errored": 0,
    "canceled": 0,
    "expired": 0
  },
  "ended_at": null,
  "created_at": "2024-09-24T18:37:24.100435Z",
  "expires_at": "2024-09-25T18:37:24.100435Z",
  "cancel_initiated_at": null,
  "results_url": null
}
```

Poll until `processing_status` is `"ended"`:

=== "Python"

    ```python
    import time

    MESSAGE_BATCH_ID = "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d"

    message_batch = None
    while True:
        message_batch = client.messages.batches.retrieve(MESSAGE_BATCH_ID)
        if message_batch.processing_status == "ended":
            break

        print(f"Batch {MESSAGE_BATCH_ID} is still processing...")
        time.sleep(60)
    print(message_batch)
    ```

=== "TypeScript"

    ```typescript
    const client = new Anthropic();

    const messageBatchId = "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d";

    let messageBatch;
    while (true) {
      messageBatch = await client.messages.batches.retrieve(messageBatchId);
      if (messageBatch.processing_status === "ended") {
        break;
      }

      console.log(`Batch ${messageBatchId} is still processing... waiting`);
      await new Promise((resolve) => setTimeout(resolve, 60_000));
    }
    console.log(messageBatch);
    ```

### Results and failures

Results arrive as `.jsonl`, one JSON object per request. Stream them rather than loading the file at once, and match each result to its request by `custom_id`, because "Results are not guaranteed to be in the same order as requests." ([Message Batches API reference](https://platform.claude.com/docs/en/api/messages/batches))

The docs' own example results file lists the second request before the first:

```json
{"custom_id":"my-second-request","result":{"type":"succeeded","message":{"id":"msg_014VwiXbi91y3JMjcpyGBHX5","type":"message","role":"assistant","model":"claude-opus-5-5","content":[{"type":"text","text":"Hello again! It's nice to see you. How can I assist you today? Is there anything specific you'd like to chat about or any questions you have?"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":11,"output_tokens":36}}}}
{"custom_id":"my-first-request","result":{"type":"succeeded","message":{"id":"msg_01FqfsLoHwgeFbguDgpz48m7","type":"message","role":"assistant","model":"claude-opus-5-5","content":[{"type":"text","text":"Hello! How can I assist you today? Feel free to ask me any questions or let me know if there's anything you'd like to chat about."}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":34}}}}
```

| Result type | Meaning | Billed | What to do |
|---|---|---|---|
| `succeeded` | The request completed and the result holds the message | Yes | Process it |
| `errored` | The request hit an error and no message was created (invalid requests and internal server errors are the documented examples); `result.error` holds the standard error shape | No | If the error is `invalid_request_error`, fix the body before resending; other errors can be retried as they are |
| `canceled` | The batch was canceled before this request was sent to the model | No | Resubmit if still needed |
| `expired` | The batch hit its 24-hour expiration before this request was sent to the model | No | Resubmit; if demand or volume slowed processing, splitting the work into smaller batches is our suggestion, following the docs' advice to break very large datasets up |

```python
# Stream results file in memory-efficient chunks, processing one at a time
for result in client.messages.batches.results(
    "msgbatch_01HkcTjaV5uDC8jWR4ZsDV8d",
):
    outcome = result.result
    match outcome.type:
        case "succeeded":
            print(f"Success! {result.custom_id}")
        case "errored":
            if outcome.error.error.type == "invalid_request_error":
                # Request body must be fixed before re-sending request
                print(f"Validation error {result.custom_id}")
            else:
                # Request can be retried directly
                print(f"Server error {result.custom_id}")
        case "expired":
            print(f"Request expired {result.custom_id}")
```

Rules that shape a failure-handling design:

- One failed request does not affect the others in the batch.
- `params` are validated asynchronously, and validation errors come back only when the whole batch has ended. Dry-run one request shape against the Messages API before submitting thousands.
- A submitted batch cannot be modified. Cancel it and resubmit; cancellation may not take effect immediately, and a canceled batch ends as `ended` and may contain partial results for requests processed before the cancel.
- Use meaningful `custom_id` values, and split very large datasets into several batches.

The CCAR-F guide adds two skills on top of this. First, resubmit **only** the failed documents, identified by `custom_id`, with the change that fixes them, such as "chunking documents that exceeded context limits" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Second, refine the prompt on a sample set before batch-processing large volumes, so the first pass succeeds more often and resubmissions cost less. Step 4 of Preparation Exercise 3 (Build a Structured Data Extraction Pipeline) practices this: submit a batch of 100 documents, handle failures by `custom_id`, resubmit failed documents with modifications such as chunking oversized ones, and calculate total processing time relative to SLA constraints.

### Submission frequency from an SLA

The CCAR-F guide's example skill is "4-hour windows to guarantee 30-hour SLA with 24-hour batch processing". The arithmetic (ours): a document that arrives just after a submission waits up to 4 hours for the next window, then up to 24 hours for processing, 28 hours in the worst case, which leaves 2 hours inside a 30-hour SLA to collect results and escalate failures (a resubmitted request can take up to another 24 hours, so it is not guaranteed to finish inside the same SLA). The general rule (ours) is submission interval + 24 hours + handling time ≤ SLA. Plan on the 24-hour bound, not the typical "less than 1 hour": the guide describes batches as having "no guaranteed latency SLA", and 24 hours is the documented point at which results become available or unfinished requests expire.

### Tools inside a batch

!!! warning "Exam guide vs current docs"

    The CCAR-F guide says: "The batch API does not support multi-turn tool calling within a single request (cannot execute tools mid-request and return results)". The current batch docs add that server tools do work: "The batch worker runs the same server-side agentic loop as the synchronous Messages API." Web search, web fetch, code execution, MCP connectors, advisor and tool search all run inside batch requests. With no open connection to maintain, the batch loop runs more iterations per turn than a synchronous request before it returns `stop_reason: "pause_turn"`, and a result that comes back with `pause_turn` is continued by submitting the paused assistant content in a follow-up request (batch or synchronous). The two statements fit together for your own (client) tools: "The model can't run your code, so every tool call is a round trip: the model asks, you execute, you report back, the model continues." ([How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works)) Our reading, which the batch page does not state: a client tool call inside a batch request ends that result with `stop_reason: "tool_use"`, so continuing it takes a new request. On the exam, answer in the guide's terms: no multi-turn tool execution within one batch request.

The batch worker also throttles `web_search` per organization and retries throttled requests automatically.

### Caching inside batches

Batch and caching discounts stack, but cache hits in a batch are best effort; the docs report typical hit rates from 30% to 98% depending on traffic patterns. To raise them:

- Put identical `cache_control` blocks in every request of the batch, keep a steady stream of requests, and share as much cached content as possible.
- Consider the 1-hour cache duration for shared context, because batches can take longer than 5 minutes to process.
- The recipe the docs call the most cost effective way to use the 1-hour cache: submit a batch containing a single request with the shared prefix and a 1-hour cache block, monitor that job, and submit the rest of the requests as soon as it completes. The docs note it is common for batch requests to take between 5 minutes and 1 hour.

### How the exams frame it

**CCDV-F Sample 1**: 10,000 documents must be processed overnight for a non-urgent analytics report, cost is the main concern and results are not needed until morning. The answer is B, the Message Batches API: "The Message Batches API is designed for latency-tolerant, high-volume workloads at lower cost, which matches an overnight, non-urgent job." The distractors fail because "Sending requests synchronously in parallel (A) does not reduce per-token cost; lowering max_tokens (C) or blindly downsizing the model (D) does not address the batch-versus-realtime tradeoff." ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf); full item on the [CCDV-F page](../claude-certified-developer.md#official-sample-questions))

**CCAR-F Question 11**: a manager proposes moving both a blocking pre-merge check and an overnight technical debt report to batches for the 50% saving. The answer is A, batch the overnight report only, because "The Message Batches API offers 50% cost savings but has processing times up to 24 hours with no guaranteed latency SLA." The rationale rejects each distractor in turn: B because relying on "often faster" completion is not acceptable for blocking workflows, C because the ordering worry is a misconception (results are correlated by `custom_id`), and D, the timeout fallback to real-time, because "Option D adds unnecessary complexity when the simpler solution is matching each API to its appropriate use case." ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf); full item on the [CCAR-F page](../claude-certified-architect-foundations.md#official-sample-questions))

### Decide

- If nobody waits on the result and cost matters, choose Message Batches. Parallel synchronous calls do not reduce per-token cost, and lowering `max_tokens` or blindly downsizing the model does not address the batch-versus-realtime tradeoff (the CCDV-F Sample 1 rationale).
- If a workflow blocks a person or a pipeline, keep it on the realtime API, even though most batches finish in under an hour, because there is no latency SLA; do not add a timeout fallback to make batching fit.
- If some requests fail, resubmit only those, found by `custom_id`, after fixing the cause; not the whole batch.
- If the SLA is fixed, size the submission interval so that interval + 24 hours fits inside it.

### Traps

- Assuming results come back in request order.
- Putting `stream: true`, `speed` or `max_tokens: 0` in a batched request.
- Counting on the typical 1-hour completion for a guarantee.
- Using older study material that says a batch holds 10,000 requests: that was the October 2024 launch limit, and the current limit is 100,000 requests or 256 MB.
- Planning a batch job on Bedrock, Google Cloud or Foundry through Anthropic's Message Batches API.

Prompt design for batch jobs (sample-set refinement, chunking, SLA planning at pipeline level) is covered in [Batch processing design](prompt-engineering.md#batch-processing-design).

## Structured outputs

*Tested in: CCAR-F 4.3 (4.3-K1 to 4.3-K3, 4.3-S1 to 4.3-S4; the schema-design bullets 4.3-K4, 4.3-S5 and 4.3-S6 are taught on the prompt engineering page), 4.4-K4, 2.3-K4, Appendix (JSON Schema) · CCDV-F 6.3 Output Handling, 2.5 Claude Application Design (schema design) · CCAR-P 2.2 · CCAO-F: not listed*

Structured outputs is the API feature set that makes Claude's output follow a JSON Schema by construction rather than by persuasion. It has two parts that can be used separately or together: **JSON outputs** (`output_config.format`) shape Claude's own response, and **strict tool use** (`strict: true`) guarantees that tool names and tool inputs match your definitions. Both work by constrained decoding: "Structured outputs guarantee schema-compliant responses through constrained decoding" ([Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)); for tools the docs call it grammar-constrained sampling.

Without it, the docs warn, Claude can produce malformed JSON, miss required fields, use inconsistent types or violate the schema, even with careful prompting. Prompt-level structure and schema design (nullable fields, "other" plus detail enums, examples) are covered in [Structured output with tools and JSON schemas](prompt-engineering.md#structured-output-with-tools-and-json-schemas); this section covers the API mechanics and exactly what is guaranteed.

### Four routes, and what each guarantees

| Route | How | Guaranteed | Not guaranteed |
|---|---|---|---|
| Prompt only | An instruction to reply in JSON, plus examples | Nothing | Well-formed JSON, required fields, types |
| Tool use with a JSON schema (not strict) | A tool whose `input_schema` is your schema; read the `tool_use` block's `input` | The CCAR-F guide treats it as eliminating JSON syntax errors | The [strict tool use docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use): Claude "might return incompatible types (`"2"` instead of `2`) or omit required fields" |
| Strict tool use | `"strict": true` on the tool definition | Tool `input` follows `input_schema`; tool `name` is always valid | That a tool is called at all (that is `tool_choice`); semantic correctness; see the exceptions below |
| JSON outputs | `output_config.format` with `type: "json_schema"` | The response text is valid JSON that matches the schema | Semantic correctness; see the exceptions below |

The split between the two features is simple: JSON outputs control **what Claude says** (the response format); strict tool use validates **how Claude calls your functions**. Combined, you get valid tool calls during the loop and a structured final answer.

### JSON outputs: `output_config.format`

Add `output_config.format` with `type: "json_schema"` and the schema. The reply comes back as valid JSON in the text content block.

=== "Python"

    ```python
    import anthropic

    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Extract the key information from this email: John Smith (john@example.com) is interested in our Enterprise plan and wants to schedule a demo for next Tuesday at 2pm.",
            }
        ],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "plan_interest": {"type": "string"},
                        "demo_requested": {"type": "boolean"},
                    },
                    "required": ["name", "email", "plan_interest", "demo_requested"],
                    "additionalProperties": False,
                },
            }
        },
    )
    print(next(block.text for block in response.content if block.type == "text"))
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";

    const client = new Anthropic();

    const response = await client.messages.create({
      model: "claude-opus-5-5",
      max_tokens: 1024,
      messages: [
        {
          role: "user",
          content:
            "Extract the key information from this email: John Smith (john@example.com) is interested in our Enterprise plan and wants to schedule a demo for next Tuesday at 2pm."
        }
      ],
      output_config: {
        format: {
          type: "json_schema",
          schema: {
            type: "object",
            properties: {
              name: { type: "string" },
              email: { type: "string" },
              plan_interest: { type: "string" },
              demo_requested: { type: "boolean" }
            },
            required: ["name", "email", "plan_interest", "demo_requested"],
            additionalProperties: false
          }
        }
      }
    });

    for (const block of response.content) {
      if (block.type === "text") {
        console.log(block.text);
      }
    }
    ```

The SDKs can derive the schema from native types and parse the reply for you: Pydantic models with `client.messages.parse()` in Python, Zod schemas with `zodOutputFormat()` (or typed JSON Schema literals with `jsonSchemaOutputFormat()`) in TypeScript. Java, Ruby, PHP and C# also derive schemas from native classes; Go reflects structs into schemas automatically on the beta API, or takes raw JSON schemas through `output_config`. The Python `parse()` method still accepts `output_format` as a convenience parameter, translates it to `output_config.format` internally and puts the result in `response.parsed_output`; the other SDKs require `output_config` directly.

=== "Python"

    ```python
    from pydantic import BaseModel
    from anthropic import Anthropic

    class ContactInfo(BaseModel):
        name: str
        email: str
        plan_interest: str
        demo_requested: bool

    client = Anthropic()

    response = client.messages.parse(
        model="claude-opus-5-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": "Extract the key information from this email: John Smith (john@example.com) is interested in our Enterprise plan and wants to schedule a demo for next Tuesday at 2pm.",
            }
        ],
        output_format=ContactInfo,
    )

    print(response.parsed_output)
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";
    import { z } from "zod";
    import { zodOutputFormat } from "@anthropic-ai/sdk/helpers/zod";

    const ContactInfoSchema = z.object({
      name: z.string(),
      email: z.string(),
      plan_interest: z.string(),
      demo_requested: z.boolean()
    });

    const client = new Anthropic();

    const response = await client.messages.parse({
      model: "claude-opus-5-5",
      max_tokens: 1024,
      messages: [
        {
          role: "user",
          content:
            "Extract the key information from this email: John Smith (john@example.com) is interested in our Enterprise plan and wants to schedule a demo for next Tuesday at 2pm."
        }
      ],
      output_config: { format: zodOutputFormat(ContactInfoSchema) }
    });

    // Automatically parsed and validated
    console.log(response.parsed_output);
    ```

When a schema uses constraints the API does not support (such as `minimum` or `maxLength`), the SDK helpers remove them, move the constraint into the field description, add `additionalProperties: false`, filter string formats, and then validate the response against your original schema: "This means Claude receives a simplified schema, but your code still enforces all constraints through validation." ([Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs))

### Strict tool use: `strict: true`

`"strict": true` is a top-level property of the tool definition, next to `name`, `description` and `input_schema`. Objects in a strict schema must set `additionalProperties` to `false`. The docs' own illustration of why it matters: a booking system needs `passengers: int`; without strict mode Claude might provide `passengers: "two"` or `passengers: "2"`, and with `strict: true` the response always contains `passengers: 2`.

```json
{
  "name": "get_weather",
  "description": "Get the current weather in a given location",
  "strict": true,
  "input_schema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "The city and state, e.g. San Francisco, CA"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"]
      }
    },
    "required": ["location"],
    "additionalProperties": false
  }
}
```

- `strict` is available on every tool except `mcp_toolset` and the computer and browser toolset entries (`computer_toolset_20260801`, `browser_toolset_20260801`); a request that sets it on either toolset entry is rejected.
- Tools with `strict: true` are not supported with programmatic tool calling.
- On models that support forced tool use, pairing `tool_choice` `any` with `strict: true` guarantees both that one of your tools is called and that its inputs follow your schema.
- On Claude Opus 5.5, Claude Fable 5.1 and Claude Mythos 5.1, `tool_choice` of type `any` or `tool` returns a 400 `invalid_request_error`. There, keep `tool_choice: {"type": "auto"}`, set `strict: true`, say in the prompt when the tool applies, or use JSON outputs for a fixed-shape response.

The error those three models return for forced tool use:

```text
tool_choice: type "tool" and "any" are not supported for this model.
```

How to choose the tool-calling controls themselves is covered in [Controlling tool choice](tool-use-and-mcp.md#controlling-tool-choice).

### Schema features and limits

Both features share one set of JSON Schema rules.

| Supported | Not supported (400 error) |
|---|---|
| All basic types: object, array, string, integer, number, boolean, null | Recursive schemas |
| `enum` of strings, numbers, booleans or nulls; `const` | Complex types inside enums |
| `anyOf` and `allOf` (limited: no `allOf` with `$ref`) | External `$ref` (for example `http://...`) |
| `$ref`, `$def`, `definitions` (internal only) | Numerical constraints: `minimum`, `maximum`, `multipleOf` |
| `default`; `required`; `additionalProperties: false` | String constraints: `minLength`, `maxLength` |
| String formats `date-time`, `time`, `date`, `duration`, `email`, `hostname`, `uri`, `ipv4`, `ipv6`, `uuid` | Array constraints beyond `minItems` of 0 or 1; `additionalProperties` other than `false` |
| Simple regex `pattern` (anchors, basic quantifiers, character classes, groups) | Regex backreferences, lookahead or lookbehind, word boundaries, large `{n,m}` ranges (not supported; the structured outputs page says complex patterns may result in 400 errors, and for a strict tool the troubleshooting page gives the error as `Unsupported regex feature in pattern field`) |

| Complexity limit (per request, all strict schemas combined) | Value |
|---|---|
| Tools with `strict: true` (non-strict tools do not count) | 20 |
| Optional parameters across all strict tool schemas and JSON output schemas | 24 |
| Parameters using union types (`anyOf` or type arrays such as `["string", "null"]`) | 16 |

Even when every explicit limit is satisfied, internal grammar-size limits can return 400 "Schema is too complex for compilation.", and a 180-second compilation timeout is the final stop-gap. The docs' remedies, in order: mark only critical tools strict, cut optional parameters (each one roughly doubles part of the grammar's state space), flatten nesting, and split work across requests or sub-agents.

Output property order follows the schema with one twist: "required properties appear first, followed by optional properties" ([Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)). If order matters to a consumer, mark every property required.

### What the guarantee does not cover

| Case | What happens | Handle it by |
|---|---|---|
| Semantic errors | Schema-valid output can still be wrong: line items that do not sum to the total, values in the wrong field | Validating meaning in code and retrying with the specific error; see [Validation, retry and feedback loops](prompt-engineering.md#validation-retry-and-feedback-loops) |
| Refusal | `stop_reason: "refusal"`, HTTP 200, tokens billed, and the output may not match the schema because the refusal takes precedence | Checking `stop_reason` before parsing |
| Token limit | `stop_reason: "max_tokens"`; the output may be incomplete | Retrying with a higher `max_tokens` |
| Enum casing | A string `enum` or `const` value may differ from the schema only in capitalization (typically the first letter of a word after a space), with no error and no special `stop_reason`; this applies to JSON outputs and strict tool use alike | Comparing case-insensitively; avoiding enum values that differ only in case |

The CCAR-F guide states the first row as a knowledge point: "strict JSON schemas via tool use eliminate syntax errors but do not prevent semantic errors (e.g., line items that don't sum to total, values in wrong fields)" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Stop reasons in general are covered in [Stop reasons and the agent loop](#stop-reasons-and-the-agent-loop).

### Operational facts

- **First-use latency.** The first request with a new schema waits while the grammar compiles. Compiled grammars are cached for 24 hours from last use. Changing the schema's structure or the set of tools invalidates the grammar cache; changing only `name` or `description` fields does not.
- **Cost and caching.** Structured outputs add a system prompt, billed as input tokens, and changing `output_config.format` invalidates the prompt cache for that conversation.
- **Works with** batch processing (at the batch discount), token counting and streaming.
- **Does not work with** citations (400 when citations are enabled together with `output_config.format`), message prefilling (incompatible with JSON outputs), `max_tokens: 0` pre-warm requests, or an on-demand compaction request.
- **Scope of the grammar.** The `output_config.format` grammar applies to Claude's direct output, not to tool calls, tool results or thinking. Tool inputs are constrained separately when a tool is strict, through the same grammar pipeline.
- **Data handling.** The schema is cached for up to 24 hours since last use. Structured outputs are HIPAA eligible, but PHI must not appear in schema definitions (property names, enum or const values, patterns).

Availability (as of September 2026): the structured outputs page marks the feature generally available on the Claude API, Claude Platform on AWS, Amazon Bedrock, Google Cloud and Microsoft Foundry, and lists its supported models from `claude-fable-5-1`, `claude-mythos-5-1`, `claude-opus-5-5` and `claude-sonnet-5` back to `claude-sonnet-4-5-20250929`, `claude-opus-4-5-20251101` and `claude-haiku-4-5-20251001`. Bedrock is the exception to watch: the structured outputs page limits it to Claude Opus 4.6, Sonnet 4.6, Sonnet 4.5, Opus 4.5 and Haiku 4.5, and the page for the current Bedrock integration, "Claude in Amazon Bedrock (Opus 4.7 and later)", lists structured outputs among features not supported. See [Claude on cloud platforms](#claude-on-cloud-platforms).

!!! note "Renamed parameter"

    Beta-era code used `output_format` with the `structured-outputs-2025-11-13` header. The parameter is now `output_config.format` and needs no beta header; the API still accepts the old field and header for a transition period. The Python SDK v1.0 and later raises `TypeError` for `output_format={...}` on `client.beta.messages.create()` or `count_tokens()`, while `client.messages.parse()` keeps `output_format` as a convenience.

### How the exams frame it

!!! warning "Exam guide vs current docs"

    The CCAR-F guide (July 2026) teaches "Tool use (tool_use) with JSON schemas as the most reliable approach for guaranteed schema-compliant structured output, eliminating JSON syntax errors", with `tool_choice: "any"` when several extraction schemas exist and the document type is unknown, and a forced tool (`{"type": "tool", "name": "extract_metadata"}`) when one extraction must run first. Its appendix lists "strict mode for syntax error elimination" under JSON Schema ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

    The docs today add the dedicated features above, and say non-strict tool calls can still return wrong types or omit required fields, so a real guarantee needs `strict: true` or `output_config.format`. Since September 1, 2026 (Claude Fable 5.1 and Mythos 5.1) and September 22, 2026 (Claude Opus 5.5), `tool_choice` `any` and `tool` return 400 on those models. They still work on other models such as Claude Opus 5, but forced tool use is incompatible with manual extended thinking (it works with adaptive thinking).

    On the exam, answer with the guide's logic: `auto` may return text instead of a tool call, `any` forces some tool, a forced tool forces a specific one, and schemas remove syntax errors but not semantic ones. In code you ship, check the target model before relying on forced tool choice.

### Decide

- In code you ship (the docs' split): if the application needs Claude's final answer in a fixed shape, choose JSON outputs; if Claude must call your function with valid arguments, choose strict tool use; if both, combine them. On a CCAR-F item, the guide's answer for guaranteed structured extraction is tool use with a JSON schema.
- If several extraction schemas exist and the document type is unknown, the guide's answer is `tool_choice: "any"`; if one extraction must run before enrichment, force that tool.
- If a field may be missing from the source, make it optional or nullable rather than required, so the model is not pushed to invent a value.
- If totals or cross-field rules matter, validate them in code: schema compliance is not correctness.

### Traps

- `tool_choice: "auto"` when a tool call must happen (the guide's framing): the model may answer in text. On Claude Opus 5.5, Claude Fable 5.1 and Claude Mythos 5.1, `any` and `tool` return 400, so there keep `auto`, set `strict: true` and say in the prompt when the tool applies.
- Sending `minimum` or `maxLength` in a raw schema (the SDK helpers strip these constraints and validate them in code instead), or a recursive schema, and getting a 400.
- Enabling citations and `output_config.format` in the same request.
- Parsing the reply without checking for `refusal` or `max_tokens` first.
- Believing that schema-valid output needs no semantic validation.

## Files, citations and search results

*Tested in: CCDV-F 2.3 Claude API Mechanics ("Messages API data access patterns") · CCAR-P 3.5, 3.6 · CCAR-F 5.6 (claim-source mappings and provenance; the guide does not name these API features) · CCAO-F: not listed*

These three features decide how your own data reaches a request and how Claude points back to it. The Files API stores a file once and lets many requests reference it. Citations make Claude's answer point at exact spans of documents you supplied. Search result blocks let your retrieval system hand Claude snippets that it cites the same way it cites web search. The CCDV-F guide lists "Messages API data access patterns" without defining it; the choices below are our reading of the phrase.

### How data reaches a request

| Pattern | What you send | Fits |
|---|---|---|
| Inline | Text; images as `base64` or `url`; documents as `base64`, text or `url` | One-off inputs; the bytes travel with every request |
| Files API | `"source": {"type": "file", "file_id": "..."}` on an `image` or `document` block, or a `container_upload` block for code execution | The same file across many requests |
| Search results | `search_result` blocks in a user message or inside a `tool_result` | Your own retrieval (RAG) results, citable |
| Just in time | Lightweight identifiers (paths, stored queries, links) that tools resolve at runtime | Large or changing data |

The just-in-time pattern is a context-engineering choice rather than an API feature; see [Why context is a budget](context-engineering.md#why-context-is-a-budget).

### Files API

**Status (as of September 2026).** The Files API came out of beta on the Claude API on August 19, 2026 and needs no beta header; requests that still send `files-api-2025-04-14` keep working and get the beta response shapes. It is in beta on Claude Platform on AWS and Microsoft Foundry (on Foundry it requires a Hosted on Anthropic deployment) and not available on Amazon Bedrock or Google Cloud.

**The model.** Create once, use many times: upload a file, get a `file_id`, and reference that ID in Messages requests instead of re-uploading. Files created by skills or the code execution tool can be downloaded; files you uploaded cannot (downloading one returns 400).

=== "cURL"

    ```bash
    FILE_ID=$(curl -X POST https://api.anthropic.com/v1/files \
      -H "x-api-key: $ANTHROPIC_API_KEY" \
      -H "anthropic-version: 2023-06-01" \
      -F "file=@/path/to/document.pdf" | jq -r '.id')
    ```

=== "Python"

    ```python
    import anthropic

    client = anthropic.Anthropic()

    uploaded = client.files.upload(
        file=("document.pdf", open("/path/to/document.pdf", "rb"), "application/pdf"),
    )
    file_id = uploaded.id

    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please summarize this document for me."},
                    {"type": "document", "source": {"type": "file", "file_id": file_id}},
                ],
            }
        ],
    )
    ```

The upload response carries `id`, `filename`, `mime_type`, `size_bytes`, `created_at`, `downloadable` (always `false` for files you upload) and `expires_at`.

| File type | Content block |
|---|---|
| PDF, plain text | `document` |
| JPEG, PNG, GIF, WebP images | `image` |
| Datasets and other files | `container_upload` with the `file_id` (for the code execution tool) |
| .docx, .xlsx | Not supported by `document`: convert to plain text and include it in the message; convert a .docx that contains images to PDF |
| .csv, .md | Already plain text: include as text, or upload with an explicit `text/plain` content type; to analyze a dataset rather than read it, use `container_upload` |

| Limit or rule | Value |
|---|---|
| File size | 500 MB per file (larger returns 413) |
| Storage | 1 TB per organization (over it returns 400) |
| File name | 1 to 255 characters |
| Optional expiry | `expires_in_seconds` at upload, 3,600 (1 hour) to 7,776,000 (90 days); cannot be changed later |
| Lifetime | Until deleted with `DELETE /v1/files/{file_id}` or until `expires_at`; deleted files cannot be recovered |
| Changes | Files cannot be modified or renamed after upload |
| Listing | Paginated, newest first; `limit` 20 by default, at most 1,000; pass `next_page` back as `page`; up to 100 `ids[]` per lookup |
| Request rate | File-related calls are limited to approximately 500 requests per minute |
| Price | Upload, download, list, metadata and delete are free; file content used in a Messages request is priced as input tokens |

!!! danger "Files are workspace-scoped"

    "Uploaded files are accessible to your entire workspace, not scoped to an end user, conversation, or session." ([Files API](https://platform.claude.com/docs/en/build-with-claude/files)) The docs spell out the consequence: "Never accept `file_id` values from end users or other untrusted sources: a user-supplied file ID would let one user of your application read content that another user uploaded." Keep the user-to-file mapping in your application, and for multi-tenant products use one workspace per tenant as the isolation boundary (up to 100 workspaces per organization).

The Files API is not ZDR or HIPAA eligible: files are retained until you delete them or they reach their configured expiration. For large PDFs the PDF docs recommend uploading through the Files API and referencing the `file_id` to keep payloads small. Anthropic's cost guide also measured the value of keeping data files out of the prompt: with Claude Sonnet 5, a table pasted into the prompt (about 91,000 input tokens per request) answered 6 of 25 questions, while the same data uploaded and analyzed with code execution answered all 25 at about a twelfth of the cost; Claude Opus 5 showed the same pattern. The guide calls its results Anthropic-internal and directional, not guarantees.

### Citations

Set `"citations": {"enabled": true}` on each `document` block, and Claude's answer comes back with citation objects that point into those documents. Citations must be enabled on all of the documents in a request or none of them. As of September 2026 the feature is generally available on the Claude API, Claude Platform on AWS, Amazon Bedrock, Google Cloud and Microsoft Foundry, for all active models.

Why use the feature instead of asking for quotes in the prompt:

- **Cost:** `cited_text` does not count toward output tokens, and is not counted as input when passed back in later turns. Citations do add slightly to input tokens (system prompt additions and chunking).
- **Reliability:** "citations are guaranteed to contain valid pointers to the provided documents." ([Citations](https://platform.claude.com/docs/en/build-with-claude/citations))
- **Quality:** in Anthropic's evaluations the feature is significantly more likely to cite the most relevant quotes than purely prompt-based approaches.

| Document type | Chunking | Citation type | Location fields |
|---|---|---|---|
| Plain text | Sentences | `char_location` | `start_char_index`, `end_char_index`: 0-indexed characters, end exclusive |
| PDF | Sentences | `page_location` | `start_page_number`, `end_page_number`: 1-indexed pages, end exclusive |
| Custom content | None beyond the blocks you supply | `content_block_location` | `start_block_index`, `end_block_index`: 0-indexed blocks, end exclusive |

Every citation also carries `cited_text`, `document_index` and `document_title`. `document_index` counts from 0 across all document blocks in the request, spanning all messages. Only text inside a document's `source` can be cited: `title` and `context` reach the model but are never cited, which makes `context` a good place for metadata as text or stringified JSON. Only text citations exist; image citations are not yet possible, so a scanned PDF without extractable text cannot be cited.

A plain text document with citations on, and a custom content document (your own chunks, no further splitting):

```json
{
  "type": "document",
  "source": {
    "type": "text",
    "media_type": "text/plain",
    "data": "The grass is green. The sky is blue."
  },
  "title": "My Document",
  "context": "This is a trustworthy document.",
  "citations": {"enabled": true}
}
```

```json
{
  "type": "document",
  "source": {
    "type": "content",
    "content": [
      {"type": "text", "text": "First chunk"},
      {"type": "text", "text": "Second chunk"}
    ]
  },
  "title": "Document Title",
  "context": "Context about the document that will not be cited from",
  "citations": {"enabled": true}
}
```

A returned citation for a plain text document (the docs' example, with its inline comments removed):

```json
{
  "type": "char_location",
  "cited_text": "The exact text being cited",
  "document_index": 0,
  "document_title": "Document Title",
  "start_char_index": 0,
  "end_char_index": 50
}
```

Working rules:

- **RAG chunks:** to let Claude cite specific sentences, put each chunk in its own plain text document; to stop any further splitting, use a custom content document.
- **Streaming:** citations arrive as a `citations_delta` inside `content_block_delta` events, one citation per delta.
- **Caching:** put `cache_control` on the top-level document block; citation blocks in responses cannot be cached. Turning citations on or off changes the system prompt, which invalidates the system and message caches.
- **Structured outputs:** enabling citations on any `document` or `search_result` block together with `output_config.format` returns 400.
- **Works with** prompt caching, token counting and batch processing.

### Search result blocks

"Search result content blocks let Claude cite your own content the same way it cites web search results" ([Search results](https://platform.claude.com/docs/en/build-with-claude/search-results)), with the `source` and `title` you provide on each citation. They are part of the standard Messages API (no beta header) and work on all active models except Claude Haiku 3. There are two ways to supply them: returned from your own tool (dynamic RAG), or placed as top-level content in a user message (pre-fetched content).

| Field | Required | Notes |
|---|---|---|
| `type` | Yes | `"search_result"` |
| `source` | Yes | Any stable string: a URL or an internal identifier such as `kb://article-1234` |
| `title` | Yes | Shown on citations |
| `content` | Yes | Array of text blocks |
| `citations` | No | Off by default; set `{"enabled": true}` |
| `cache_control` | No | For caching the block |

```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
  "content": [
    {
      "type": "search_result",
      "source": "https://docs.company.com/pricing",
      "title": "Pricing Plans",
      "content": [
        {
          "type": "text",
          "text": "Acme Dashboard is available on the Starter plan at $10 per user per month and the Enterprise plan with custom pricing."
        }
      ],
      "citations": {"enabled": true}
    }
  ]
}
```

A citation of a search result has type `search_result_location` and carries `source`, `title`, `cited_text`, `search_result_index`, `start_block_index` and `end_block_index` (exclusive). `search_result_index` is 0-based and counts every `search_result` block in the request in order, across all messages and tool results. `cited_text` is the full text of the cited blocks joined together and does not count toward output tokens. Granularity is set by how you split the content: "The text block is the minimal citable unit: Claude cites whole blocks, not substrings within a block." ([Search results](https://platform.claude.com/docs/en/build-with-claude/search-results)) Split content into smaller text blocks for finer citations; one large block means every citation returns its full text.

The docs' example: a search result split into three text blocks, and a citation of the second block (index 1, end index 2 exclusive).

```json
{
  "type": "search_result",
  "source": "https://docs.company.com/api-guide",
  "title": "API Documentation",
  "content": [
    {
      "type": "text",
      "text": "Authentication: All API requests require an API key."
    },
    {
      "type": "text",
      "text": "Rate Limits: The API allows 1000 requests per hour per key."
    },
    {
      "type": "text",
      "text": "Error Handling: The API returns standard HTTP status codes."
    }
  ],
  "citations": { "enabled": true }
}
```

```json
{
  "type": "search_result_location",
  "cited_text": "Rate Limits: The API allows 1000 requests per hour per key.",
  "source": "https://docs.company.com/api-guide",
  "title": "API Documentation",
  "search_result_index": 0,
  "start_block_index": 1,
  "end_block_index": 2
}
```

Rules that produce validation errors or surprises:

- Citations are all or nothing across a request: every search result has them on, or every one has them off. When the web search tool is enabled in the same request, all `search_result` blocks must have citations on.
- Inside a `tool_result`, if any block is a `search_result`, every block must be one; mixing in other block types is a validation error.
- Search results hold text only and can appear only in user messages (tool results included).
- Availability (as of September 2026, per the search results page): Claude API, Amazon Bedrock and Google Cloud.

Good practice from the docs: when a search fails or finds nothing, return a plain text block describing the outcome (for example "No results found.") instead of raising an error, and return only the most relevant results to avoid overflowing the context.

### Choosing between them

| You need | Use |
|---|---|
| The same large file in many requests | Files API, referenced by `file_id` |
| Answers that quote exact spans of documents you pass in | `document` blocks with citations |
| Sentence-level citations from RAG chunks | Each chunk as its own plain text document |
| Citations at your own granularity (lists, transcripts) | A custom content document, or search results split into small text blocks |
| Retrieval results from your own search tool, cited like web search | `search_result` blocks returned in the `tool_result` |
| JSON-schema output and citations together | Not possible in one request: the combination returns 400 |

### How the exams frame it

- **CCDV-F 2.3** names "Messages API data access patterns" without defining it or giving an example. The decisions in this section that fit the phrase (our reading): inline data versus a `file_id`, workspace scoping of uploaded files, which platforms offer the Files API, and whether retrieved content goes in as documents or search results.
- **CCAR-F 5.6** asks for "structured claim-source mappings that the synthesis agent must preserve and merge when combining findings" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)), and 5.6-S1 lists what a mapping holds: source URLs, document names and relevant excerpts. The guide does not name the Citations feature; our mapping is that a citation object carries those same parts per claim (document title or search-result source, cited text, location), which a pipeline can carry into later steps. The multi-agent side is in [Provenance and uncertainty in synthesis](context-engineering.md#provenance-and-uncertainty-in-synthesis).
- **CCAR-P 3.5 and 3.6** ask you to "Design a RAG pipeline with appropriate chunking and indexing strategies" and "Apply retrieval strategies matched to data shape and query pattern" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). The guide names no API feature for this; search result blocks are the API mechanism the docs provide for handing retrieved chunks to Claude with their sources, and chunk granularity decides citation granularity. Pipeline design is in [Retrieval-augmented generation](solution-architecture.md#retrieval-augmented-generation).

### Traps

- Accepting a `file_id` from an end user, or assuming files are private to the user who uploaded them.
- Expecting to download a file you uploaded.
- Planning the Files API on Bedrock or Google Cloud.
- Enabling citations on some documents but not others, or combining citations with `output_config.format`.
- Expecting citations on a scanned PDF with no extractable text, or on a document's `title` or `context`.
- Mixing `search_result` blocks with other block types in one `tool_result`.

## Cost and usage tracking

*Tested in: CCDV-F 5.4 Cost and Token Management, Section 2 applied area (model tiers, tokens, caching and batch processing to optimize cost and latency) · CCAR-P 4.5, 4.1, 4.6 · CCAR-F: API pricing calculations and billing are out of scope (the 50% batch saving is in scope) · CCAO-F: not listed*

The CCDV-F guide describes this skill as "Token budgeting and cost management techniques for Claude applications, including token usage tracking, cost modeling, and caching techniques (prompt caching, cache check-pointing) for cost optimization." ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)) In practice that is four jobs: know what each token costs, price a workload before you build it, cap what a request or an account can spend, and read back what you actually spent. Caching mechanics, including what the guide's "cache check-pointing" corresponds to in the docs, are in [Prompt caching](#prompt-caching); this section is about the money.

### What you pay for

- **Input tokens**, including everything in the request: system prompt, messages, tool definitions (the `tools` parameter) and the tool-use system prompt the API adds when tools are present.
- **Output tokens**, including thinking: thinking tokens are part of `max_tokens` and billed as output.
- **Cache writes and cache reads**, at multiples of the input price.
- **Server tools**, on top of tokens: web search is &#36;10 per 1,000 searches plus standard token costs for the search-generated content (each search counts once whatever the number of results; errored searches are not billed); web fetch has no charge beyond the tokens of the fetched content (`max_content_tokens` caps it); code execution is free when used with `web_search_20260209` or `web_fetch_20260209` or later, and otherwise billed by execution time (5-minute minimum, 1,550 free hours per organization per month, then &#36;0.05 per hour per container). If files are included in a code execution request, execution time is billed even when the tool is not called, because the files are preloaded onto the container.
- **Claude Managed Agents** sessions: tokens at standard rates plus &#36;0.08 per session-hour while `running`; the Batch API discount does not apply, and session runtime replaces container-hour billing for code execution.
- **Free:** token counting, and Files API operations (upload, list, metadata, download, delete). File content is billed as input tokens when a request uses it.

The tool-use system prompt is not trivial, and it comes on top of the tokens in your `tools` definitions and in the `tool_use` and `tool_result` blocks. As of September 2026 it adds 286 tokens on Claude Opus 5.5 (`auto` or `none`), 354 tokens on Claude Sonnet 5 (`auto` or `none`; 474 with `any` or `tool`) and 496 tokens on Claude Haiku 4.5 (588 with `any` or `tool`); with no tools and `tool_choice` `none` it adds 0. Anthropic-defined tools add their own definitions, for example 700 tokens for `text_editor_20250429` (listed for Claude 4.x models) and, for the bash tool, 325 tokens on Claude Opus 5, Opus 4.8 and Opus 4.7 (244 on Opus 4.6, Sonnet 4.6 and earlier).

!!! note "Code execution allowance: two phrasings"

    The docs pricing page says each organization receives 1,550 free hours of code execution per month; the claude.com pricing page says "50 free hours of usage daily per organization." ([Claude pricing](https://claude.com/pricing)) The totals match only for a 31-day month (50 x 31 = 1,550, our arithmetic), and neither page says how a daily allowance maps onto the monthly pool. Use the docs figure, 1,550 hours per month, for API cost models.

### Prices per million tokens (as of September 2026)

| Model | Input | 5-min cache write | 1-hour cache write | Cache hit | Output | Batch input | Batch output |
|---|---|---|---|---|---|---|---|
| Claude Fable 5.1, Mythos 5.1 | &#36;10 | &#36;12.50 | &#36;20 | &#36;0.25 | &#36;50 | &#36;5 | &#36;25 |
| Claude Fable 5, Mythos 5 | &#36;10 | &#36;12.50 | &#36;20 | &#36;1 | &#36;50 | &#36;5 | &#36;25 |
| Claude Opus 5.5 | &#36;4 | &#36;5 | &#36;8 | &#36;0.20 | &#36;20 | &#36;2 | &#36;10 |
| Claude Opus 5, 4.8, 4.7, 4.6, 4.5 | &#36;5 | &#36;6.25 | &#36;10 | &#36;0.50 | &#36;25 | &#36;2.50 | &#36;12.50 |
| Claude Sonnet 5 | &#36;2 | &#36;2.50 | &#36;4 | &#36;0.20 | &#36;10 | &#36;1 | &#36;5 |
| Claude Sonnet 4.6, 4.5 | &#36;3 | &#36;3.75 | &#36;6 | &#36;0.30 | &#36;15 | &#36;1.50 | &#36;7.50 |
| Claude Haiku 4.5 | &#36;1 | &#36;1.25 | &#36;2 | &#36;0.10 | &#36;5 | &#36;0.50 | &#36;2.50 |

All prices are in USD. Sonnet 5's &#36;2 / &#36;10 was announced as introductory pricing through August 31, 2026 and is now the standard price; the increase to &#36;3 / &#36;15 that had been scheduled for September 1, 2026 will not happen. Retired models (Claude Opus 4.1, Sonnet 4 and Haiku 3.5, which the pricing page marks as retired except on Bedrock and Google Cloud, and Claude Opus 4, retired except on Google Cloud) are left out of the table. Claude Mythos 5.1 and Mythos 5 are listed with limited availability.

### Modifiers that change the bill

| Modifier | Effect |
|---|---|
| Message Batches | 50% off input and output tokens; stacks with caching |
| Cache writes and reads | 1.25x (5 min) or 2x (1 hour) for writes; 0.1x for reads on most models (0.05x Opus 5.5, 0.025x Fable 5.1 and Mythos 5.1); stack with batch and data residency |
| US-only inference (`inference_geo: "us"`, Claude 4.6 and later, on the Claude API and Claude Platform on AWS; the US Data Zone Standard deployment type on Microsoft Foundry) | 1.1x on every token category: input, output, cache writes, cache reads; global routing (the default) is standard price |
| Regional or multi-region endpoints on Bedrock and Google Cloud | 10% premium over global endpoints (Sonnet 4.5, Haiku 4.5, Opus 4.5 and later) |
| Fast mode (research preview, first-party Claude API only, not with batches) | Opus 5.5: &#36;8 / &#36;40; Opus 5 and Opus 4.8: &#36;10 / &#36;50 per million input / output tokens; stacks with caching and data residency |
| Long context | No premium on Claude 4.6 and later and Claude Mythos Preview; the [pricing page](https://platform.claude.com/docs/en/about-claude/pricing) puts it as "A 900k-token request is billed at the same per-token rate as a 9k-token request." |
| Tokenizer | Claude 4.7 and later (and Mythos Preview) produce approximately 30% more tokens for the same text than earlier models; the exact increase depends on the content |
| Marketplace billing (Claude Platform on AWS, Microsoft Foundry) | Billed in Claude Consumption Units at &#36;0.01 per CCU (100 CCU = &#36;1.00) |

The tokenizer row matters most when comparing models: a lower per-token price does not mean a lower per-request price if the same text becomes more tokens. Compare cost per completed task, as [Cost is per task, not per token](#cost-is-per-task-not-per-token) explains, and recount prompts on the target model with the [token counting endpoint](#counting-tokens-before-you-send).

### Worked example 1: pricing a response from its usage object

Take the mixed-TTL response from [Reading the usage fields](#reading-the-usage-fields) (2,048 uncached input tokens, 1,800 read from cache, 148 written with a 5-minute TTL, 100 written with a 1-hour TTL, 503 output tokens) and price it on Claude Opus 5.5. All arithmetic is ours; prices are from the table above.

| Line | Tokens | Price per million | Cost |
|---|---|---|---|
| Uncached input (`input_tokens`) | 2,048 | &#36;4 | &#36;0.008192 |
| Cache read (`cache_read_input_tokens`) | 1,800 | &#36;0.20 | &#36;0.000360 |
| 5-minute cache write (`ephemeral_5m_input_tokens`) | 148 | &#36;5 | &#36;0.000740 |
| 1-hour cache write (`ephemeral_1h_input_tokens`) | 100 | &#36;8 | &#36;0.000800 |
| Output (`output_tokens`) | 503 | &#36;20 | &#36;0.010060 |
| **Total** | | | **&#36;0.020152** |

- Per 1,000 identical responses: about &#36;20.15.
- The same 4,096 input tokens with no caching: 4,096 x &#36;4 / 1,000,000 = &#36;0.016384, plus the same &#36;0.01006 of output, &#36;0.026444 in total. Caching saved about 24% here even though this response paid for two writes.
- The same cached request sent through the Batch API: the discount applies to every token, cached ones included, so about &#36;0.010076.

The pricing page uses the same method in its Claude Managed Agents example: 40,000 cache-read tokens on Claude Opus 5 cost 40,000 x &#36;5 x 0.1 / 1,000,000 = &#36;0.02.

### Worked example 2: a monthly cost model

Take an illustrative workload: a team summarizes 20,000 documents a month on Claude Sonnet 5. Each request averages 2,000 input tokens (no shared prefix, so caching does not apply) and 300 output tokens. Nobody waits on the summaries. All arithmetic is ours.

| Option | Input cost | Output cost | Monthly total |
|---|---|---|---|
| Realtime Messages API | 40,000,000 x &#36;2 / 1M = &#36;80.00 | 6,000,000 x &#36;10 / 1M = &#36;60.00 | &#36;140.00 |
| Message Batches | 40,000,000 x &#36;1 / 1M = &#36;40.00 | 6,000,000 x &#36;5 / 1M = &#36;30.00 | &#36;70.00 |
| Message Batches with `inference_geo: "us"` | &#36;40.00 x 1.1 = &#36;44.00 | &#36;30.00 x 1.1 = &#36;33.00 | &#36;77.00 |

Three lessons carry into exam scenarios. The batch discount halves this workload because nobody is waiting (see [Message Batches](#message-batches)); each request in a batch can still set its own `inference_geo`. Residency is a 10% uplift, not a separate price list. And if someone proposes a cheaper model, rerun the cost model with that model's own token counts: the tokenizer difference between model generations changes the token totals as well as the rate.

### Tracking what you spent

**Per response.** Every response carries a `usage` object. Log it with the `request-id` response header and you can reconstruct the cost of each request.

| Field | What it tells you |
|---|---|
| `input_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens` | The three input categories; their sum is total input (`input_tokens` alone is only the uncached part) |
| `cache_creation.ephemeral_5m_input_tokens`, `ephemeral_1h_input_tokens` | Cache writes by TTL (priced differently); together they equal `cache_creation_input_tokens` |
| `output_tokens` | The authoritative billed output total, thinking included; `output_tokens_details.thinking_tokens` breaks out the thinking share |
| `server_tool_use.web_search_requests`, `web_fetch_requests` (the pricing page's code execution example also shows `code_execution_requests`) | Server tool calls: web searches are billed per search on top of tokens, web fetches add nothing beyond tokens |
| `service_tier` | `"standard"`, `"priority"`, `"batch"` or null |
| `inference_geo` | Where inference ran (the 1.1x residency price applies to `"us"`) |

Two cautions. Token counts will not match visible content one to one (`output_tokens` is non-zero even for an empty reply). And with server-side compaction (a beta, on demand or at a token threshold), the top-level `input_tokens` and `output_tokens` leave out the summarization step (on-demand compaction reports them as zero): sum the `usage.iterations` array to get the billed total.

**In the Console.** The Usage and Cost pages show organization history; the "Rate Limit - Input Tokens" chart includes your cache rate (the share of input tokens read from cache), and the Usage page exports a CSV broken down by API key and model, which is also how you find traffic still on a deprecated model.

**Programmatically: the Usage & Cost Admin API.** It needs Admin API credentials (an Admin API key `sk-ant-admin01-...`, an OAuth token with the `org:admin` scope, or a personal or service account key that is not scoped to a workspace); workspace API keys do not work, and the Admin API is unavailable for individual accounts.

| | Usage report | Cost report |
|---|---|---|
| Endpoint | `/v1/organizations/usage_report/messages` | `/v1/organizations/cost_report` |
| Granularity | `1m` (default 60 buckets, max 1,440), `1h` (default 24, max 168), `1d` (default 7, max 31) | Daily only (`1d`) |
| Measures | Uncached input, cached input, cache creation and output tokens, plus server tool usage such as web search | Token usage, web search and code execution costs in USD, as decimal strings in the lowest unit (cents) |
| Filter and group by | API key, workspace, model, service tier, context window, data residency (`inference_geo`: `global`, `us`, `not_available`), speed (beta; needs the `fast-mode-2026-02-01` header) | Workspace or description (grouping by `description` adds parsed fields such as `model` and `inference_geo`) |
| Watch for | Playground usage has `api_key_id` null; default-workspace usage has `workspace_id` null; models released before February 2026 report `inference_geo` as `"not_available"` | Priority Tier costs are excluded (track Priority Tier in the usage report with `service_tier`); code execution costs appear here under `Code Execution Usage`, not in the usage report |
| Pagination | `has_more` and `next_page` | Same |

The docs' daily-usage-by-model request:

```bash
curl "https://api.anthropic.com/v1/organizations/usage_report/messages?\
starting_at=2025-01-01T00:00:00Z&\
ending_at=2025-01-08T00:00:00Z&\
group_by[]=model&\
bucket_width=1d" \
  -H "anthropic-version: 2023-06-01" \
  -H "x-api-key: $ANTHROPIC_ADMIN_KEY"
```

Data typically appears within 5 minutes of request completion, and polling once per minute is supported for sustained use. The endpoints are not available on Claude Platform on AWS (use the Console pages there). Other products have their own reporting: Claude Enterprise organizations use the Claude Enterprise Analytics API with an Analytics API key, per-user Claude Code costs come from the Claude Code Analytics API (Admin API key), and the key types are not interchangeable: an Admin API key cannot call the Claude Enterprise Analytics API, and an Analytics API key cannot call the Admin API. Claude Code usage and OpenTelemetry are covered in [Monitoring, usage and cost](claude-code-workflows.md#monitoring-usage-and-cost).

### Controls that cap spend

| Control | Behavior |
|---|---|
| Tier spend cap (monthly) | Start &#36;500, Build &#36;1,000, Scale &#36;200,000; Custom has no cap. At the cap, usage pauses until 00:00 UTC on the first day of the next month unless a higher limit is granted sooner. Requests return 429 `rate_limit_error` with no `retry-after` header (on the Messages API, `error.details.error_code` is `enforced_spend_limit_reached`), and retries, including the SDKs' automatic retries, fail until access resumes |
| A spend limit you set (organization, or per workspace except the default workspace; it cannot exceed the tier cap) | Requests return HTTP 400 `invalid_request_error` with a message saying you reached your specified usage limits; raise or remove the limit to restore access sooner |
| Claude Code workspace spend limit | Checked separately: requests over it can receive a 429 that carries a `retry-after` header |
| Batches | May go slightly over a workspace's spend limit because of high throughput and concurrent processing |

How to tell these errors apart, and which of them to retry, is in [Errors, retries and rate limits](#errors-retries-and-rate-limits). For token budgeting inside a request, `max_tokens` is the enforced per-request limit; task budgets only tell the model its budget for a whole agentic loop and are a "soft hint, not a hard cap" in the docs' words ([Task budgets](https://platform.claude.com/docs/en/build-with-claude/task-budgets)). Their parameters are in [Which control to reach for](#which-control-to-reach-for).

### Where the savings are

Anthropic's cost guide sorts levers into free wins that do not touch quality (prompt caching, token hygiene, a prompt audit against the model you run, batch processing at 50% off for work that can wait up to 24 hours, and workspace spend limits as the backstop) and tradeoffs (model choice, effort, output caps and task budgets, multi-model architectures). Its measurements are Anthropic-internal and directional, not guarantees, but the findings are useful:

1. **Prompt caching** was the largest lever, cutting agent-loop cost by a factor of 2.7 to 5.3, because "A 40-turn task sends its first turn 40 times, so task cost grows with roughly the square of turn count." ([Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence))
2. **Batching** is the second-largest free lever after caching for unattended agent work (evaluation runs, backfills, scheduled jobs): 50% off every token, cached ones included, of any request nobody is waiting on.
3. **Keeping data files out of the prompt**: a table uploaded with the Files API and analyzed with code execution beat the same table pasted into the prompt on both accuracy and cost; the numbers are in [Files API](#files-api).
4. **Tool search** kept run cost flat as tool catalogs grew (45% less at 502 tools, with unchanged accuracy).
5. **Context editing** is not automatically a saving: on a short 20-issue run it cost 74% more, and on the long run it changed nothing while a hand-written prune saved 39% and compaction 32%.

The pricing page's own list is shorter: use the right model (its Haiku, Sonnet and Opus rule is quoted in [What each tier is for](#what-each-tier-is-for)), implement prompt caching, batch non-time-sensitive work, and monitor usage.

### How the exams frame it

- **CCDV-F 5.4** (Cost and Token Management, 2.8% of the exam) covers token budgeting, usage tracking, cost modeling and caching: know the `usage` fields and the Usage & Cost API, price a workload from the tables above (modifiers and the tokenizer caveat included), and know how caching and batching cut cost.
- **CCDV-F Sample 1** (tagged to Domain 2) is a cost scenario whose answer is the Message Batches API; the item and its rationale are taught under [Message Batches](#message-batches), and the full item is on the [CCDV-F page](../claude-certified-developer.md#official-sample-questions).
- **CCAR-P 4.5** ("Optimize token usage, latency, and cost-performance trade-offs", [CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)) frames cost against quality. CCAR-P Sample 2, which the guide tags to Domain 2 (the prompt-reuse objective 2.5 is the closest fit, our mapping), is the concrete case: static content first plus prompt caching; its answer and rationale are in [Prompt caching](#prompt-caching).
- **CCAR-P 4.1** lists cost among the evaluation metrics to define, and **CCAR-P 4.6** asks you to monitor performance with logging and observability tools. In API terms (our mapping): log `usage` with the `request-id` for each request, and use the Usage & Cost API or the Console for organization-level trends.
- **CCAR-F** lists "Rate limiting, quotas, or API pricing calculations", "Claude API authentication, billing, or account management" and prompt caching details beyond knowing it exists as out of scope ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)), but 4.5-K1 does test the Message Batches API's 50% cost savings, and sample question 11 turns on it. Know that fact and that caching exists, not the price tables.

### Decide

- If a cost question gives token counts, price each usage category separately (uncached input, cache reads, each write TTL, output), then apply batch and residency multipliers.
- If two models are compared on price, compare cost per completed task with each model's own token counts, not list price per token.
- If a bulk job can wait, choose the Message Batches API; not a lower `max_tokens` or a blindly smaller model, because those do not address the batch-versus-realtime choice.
- If spend must stay under a budget, set your own spend limit below the tier cap (for the organization or per workspace, though not on the default workspace; batches can overshoot a workspace limit slightly). A 400 `invalid_request_error` then means your limit was reached; a 429 with `enforced_spend_limit_reached` means the tier cap, and retrying will not help until access resumes.

### Traps

- Forgetting that tool definitions, the tool-use system prompt and thinking tokens are billed.
- Assuming cache reads always cost 10% of input on every model.
- Treating a 429 from the spend cap as a rate limit to retry.
- Reading top-level `input_tokens` as the whole input when caching is on (it is only the uncached part).
- Calling the Usage & Cost API with a workspace API key, or expecting it on Claude Platform on AWS.
- Cutting the cost of a non-urgent bulk job by lowering `max_tokens` or switching to the smallest model instead of batching it.

## Claude on cloud platforms

*Tested in: CCDV-F 2.3 Claude API Mechanics ("invoking Claude through third-party vendors"), Section 2 applied area (third-party integrations) · CCAR-P 5.4, 3.2 · CCAR-F: out of scope (specific cloud provider configurations) · CCAO-F: not listed*

Claude models are reachable through five platforms. Anthropic runs the Claude API itself, and also operates Claude Platform on AWS and Microsoft Foundry; Amazon Bedrock and Google Cloud are partner-operated. Feature availability differs by platform, and those differences are what this section teaches. The docs' framing of the choice: cloud platforms are best for existing cloud commitments, specific compliance requirements and consolidated cloud billing, while the direct API gives direct access to the latest models and features and suits new integrations that need full feature access.

### The five platforms

| Platform | Operated by | How you call it | Billing |
|---|---|---|---|
| Claude API | Anthropic | `https://api.anthropic.com`, Claude model IDs, API key | Direct |
| Claude Platform on AWS | Anthropic (AWS provides authentication, IAM and Marketplace billing) | Claude API endpoints and model IDs; SigV4 or API key; `anthropic-workspace-id` header required on inference requests | AWS Marketplace, in CCUs |
| Amazon Bedrock | AWS (partner-operated), with zero operator access for Anthropic personnel | Two integrations. Claude in Amazon Bedrock (Fable 5.1, Fable 5, Opus 5.5, Opus 5, Opus 4.8, Opus 4.7, Sonnet 5, Haiku 4.5, and Mythos Preview by invitation): the Messages API at `https://bedrock-mantle.{region}.api.aws/anthropic/v1/messages`, model IDs such as `anthropic.claude-opus-5-5`. The legacy integration: `InvokeModel`/`Converse` with ARN-versioned model IDs, which exist for Opus 4.6 and earlier (Fable, Opus 5.5, Opus 5, Sonnet 5, Opus 4.8 and Opus 4.7 are also reachable through `InvokeModel` on `bedrock-runtime`) | Through AWS |
| Google Cloud (Agent Platform, formerly Vertex AI) | Google (partner-operated) | `model` goes in the endpoint URL; `anthropic_version: "vertex-2023-10-16"` goes in the request body | Through Google Cloud |
| Microsoft Foundry | Anthropic-operated, with two hosting options: Hosted on Azure and Hosted on Anthropic | `https://{resource}.services.ai.azure.com/anthropic/v1/*`; your deployment name is the `model` value; API key (`api-key` or `x-api-key`) or Entra ID token | Azure Marketplace, in CCUs metered hourly |

Platform notes that come up in scenarios:

- **Claude Platform on AWS** typically gets new features the same day as the Claude API and uses a separate capacity pool from both the first-party API and Bedrock, so workloads can fail over between platforms. Its most common setup error is forgetting to enable outbound web identity federation once per AWS account. HIPAA readiness is not available there, and ZDR is opt-in.
- **Amazon Bedrock** authenticates with a Bedrock service role (recommended), IAM assumed roles (12-hour maximum session) or bearer tokens (least preferred). Default quota is 2 million input tokens per minute, raisable to 5 million input and 500,000 output TPM without extra Anthropic approval. Logs go to CloudWatch and CloudTrail, and Anthropic recommends keeping activity logs for at least 30 days. On the legacy integration, the request body carries `"anthropic_version": "bedrock-2023-05-31"`, and newer models need an inference profile (a prefix such as `global.`, `us.` or `eu.` on the base model ID) instead of the base model ID.
- **Google Cloud** offers global, multi-region (`us`, `eu`) and regional endpoints; regional and multi-region cost 10% more, and provisioned throughput requires a regional endpoint. The Python client is `AnthropicVertex`, with region `"global"` recommended.
- **Microsoft Foundry** does not return Anthropic's rate-limit headers (manage limits with Azure tools), and SDK support covers C#, Java, PHP, Python and TypeScript but not Go or Ruby. For Hosted on Azure deployments, prompts and completions stay in Azure; only usage metadata and content flagged by Anthropic's safety systems egress to Anthropic.

=== "Amazon Bedrock"

    ```python
    from anthropic import AnthropicBedrockMantle
    client = AnthropicBedrockMantle(aws_region="us-east-1")
    message = client.messages.create(
        model="anthropic.claude-opus-5-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello, Claude"}],
    )
    ```

=== "Google Cloud"

    ```python
    from anthropic import AnthropicVertex
    client = AnthropicVertex(project_id="MY_PROJECT_ID", region="global")
    message = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=100,
        messages=[{"role": "user", "content": "Hey Claude!"}],
    )
    ```

The Python SDK ships the platform clients as extras (`anthropic[bedrock]`, `anthropic[vertex]`, `anthropic[aws]`); Foundry support is in the base package.

### Feature availability (as of September 2026)

| Feature | Claude API | Claude Platform on AWS | Amazon Bedrock | Google Cloud | Microsoft Foundry |
|---|---|---|---|---|---|
| Message Batches API | Yes | Yes | No | No | No |
| Files API | Yes (GA) | Beta | No | No | Beta (not on Hosted on Azure deployments) |
| Structured outputs | Yes | Yes | Legacy integration only, five models (Opus 4.6, Sonnet 4.6, Sonnet 4.5, Opus 4.5, Haiku 4.5) | Yes | Yes |
| Citations | Yes | Yes | Yes | Yes | Yes |
| Search result blocks | Yes | Not listed | Yes | Yes | Not listed |
| 1-hour cache TTL | Yes | Yes | Yes (both integrations) | Yes | Yes |
| Automatic caching (top-level `cache_control`) | Yes | Yes | Not on the legacy integration | Yes | Yes |
| Prompt cache isolation | Per workspace | Per workspace | Per organization | Per organization | Per workspace |
| Fast mode | Yes (research preview) | No | No | No | No |
| `inference_geo` parameter | Yes | Yes | No (the endpoint or profile sets the region) | No (the endpoint sets the region) | No (Hosted on Azure deployments use the US Data Zone Standard deployment type instead) |
| Maximum request size | 32 MB for Messages and Token Counting (Message Batches 256 MB, Files API 500 MB) | Same as the Claude API | 20 MB | 30 MB | Not stated on the API overview page |
| Model retirement dates | Anthropic's | Anthropic's | Bedrock's own | Google Cloud's own | Anthropic's |

Beyond the table, as of September 2026:

- **Amazon Bedrock** (the current Messages API integration) also lacks server-side tools (code execution, web search, web fetch, advisor), Agent Skills, the MCP connector, programmatic tool calling, URL and Files input sources, Anthropic's Models, Admin, Compliance and Usage and Cost endpoints, Claude Managed Agents, server-side fallback, and the `computer_toolset_20260801` and `browser_toolset_20260801` toolsets. On-demand compaction is not available on Bedrock either.
- **Google Cloud** supports web search and structured outputs but not code execution, web fetch, advisor, Agent Skills, the MCP connector, programmatic tool calling, URL and Files input sources, Anthropic's Models, Admin, Compliance and Usage and Cost endpoints, Claude Managed Agents or server-side fallback.
- **Microsoft Foundry** does not offer the Admin API, Models API, Compliance API, advisor, Claude Managed Agents, server-side fallback or the computer and browser use toolsets. Hosted on Azure deployments additionally reject code execution, web search and web fetch versions later than `web_search_20250305` and `web_fetch_20250910`, Agent Skills, programmatic tool calling and the Files API with a `400 Bad Request`.
- **Claude Platform on AWS** has no programmatic Usage and Cost API endpoints; use the Console pages.

### Choosing a platform

| If the requirement is | Choose | Because |
|---|---|---|
| On AWS: FedRAMP High, IL4, IL5 or HIPAA-ready compliance, or AWS as the sole data processor | Amazon Bedrock, not Claude Platform on AWS | The Claude Platform on AWS docs direct these organizations to Bedrock (HIPAA readiness is also available on the Claude API itself) |
| AWS billing and identity, with Claude API features typically on the same day | Claude Platform on AWS | Anthropic operates it with Claude API endpoints and model IDs (fast mode and the computer and browser use toolsets are not available there) |
| The latest models and full feature access, including fast mode | Claude API | The docs position the direct API for the latest models and features; fast mode is first-party only |
| Anthropic's Message Batches API (50% off) | Claude API or Claude Platform on AWS | The other three list the Message Batches API as unsupported (Anthropic's 2024 launch post pointed Bedrock and Vertex AI users to those platforms' own batch mechanisms) |
| Prompts and completions must stay inside Azure | Microsoft Foundry, Hosted on Azure | Accepting that several features return 400 there, and that usage metadata and content flagged by Anthropic's safety systems still egress to Anthropic |
| A second capacity pool for failover | Claude Platform on AWS alongside another platform | It uses a separate capacity pool |
| Existing cloud commitment or consolidated cloud billing | The cloud platform you already use | The docs' stated reason for choosing a cloud platform |

Two operational consequences are easy to miss. Model IDs differ by platform (Bedrock uses an `anthropic.` prefix; Google Cloud writes dated models with `@` before the date), so code that hard-codes IDs is not portable; see [Model versions, deprecation and migration](#model-versions-deprecation-and-migration). And Claude Code running against a cloud provider does not send metrics back to Anthropic, so Anthropic's analytics do not cover it; OpenTelemetry does (see [Monitoring, usage and cost](claude-code-workflows.md#monitoring-usage-and-cost)).

Capacity planning across platforms (quotas, failover, provisioned throughput) is covered in [Deployment platforms and capacity](solution-architecture.md#deployment-platforms-and-capacity), and the compliance side (ZDR, HIPAA, residency) in [Data retention, training and compliance](security-and-governance.md#data-retention-training-and-compliance).

### How the exams frame it

- **CCDV-F 2.3** says "invoking Claude through third-party vendors" and nothing more, and the guide's list of applied areas mentions integrating Claude "through the API, client SDKs, and third-party integrations" ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). Our reading: expect scenarios that ask which platform supports a feature a workload needs, or why a request fails on one platform but not another.
- **CCAR-P 5.4** names GDPR, HIPAA and FedRAMP; the docs' platform answer for FedRAMP High, IL4, IL5 or HIPAA-ready requirements on AWS is Bedrock. **CCAR-P 3.2** ("Analyze authentication and authorization requirements to identify security gaps", [CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)) touches the per-platform identity models (our mapping): IAM roles and SigV4 on AWS, Entra ID on Foundry. The [CCAR-P certification page](https://anthropic-partners.skilljar.com/claude-certified-architect-professional-certification) also lists the catalog courses "Claude with Amazon Bedrock" and "Claude on Google Cloud" among its prep courses.
- **CCAR-F** lists "Specific cloud provider configurations (AWS, GCP, Azure)" as out of scope ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

### Traps

- Assuming every platform has every feature: Message Batches and the Files API are the clearest examples.
- Expecting Anthropic's deprecation dates to apply on Bedrock or Google Cloud.
- Sending `inference_geo` to Bedrock or Google Cloud, where the region comes from the endpoint.
- Assuming prompt caches are workspace-isolated on Bedrock and Google Cloud (they are organization-isolated there).
- Reading "Vertex AI" and "Agent Platform" as different products.

!!! note "Names and integrations that changed"

    Current Anthropic docs call Google's product "Google Cloud's Agent Platform, formerly Vertex AI" ([Claude Code third-party integrations](https://code.claude.com/docs/en/third-party-integrations)), while the SDK class is still `AnthropicVertex` and the docs page URL still ends in `claude-on-vertex-ai`; the CCAR-P prep course is titled "Claude on Google Cloud" ([CCAR-P certification page](https://anthropic-partners.skilljar.com/claude-certified-architect-professional-certification)). Treat Vertex AI, Agent Platform and Google Cloud as the same platform. On AWS, Bedrock now serves its current models (Claude Fable 5.1 and Fable 5, Opus 5.5, Opus 5, Opus 4.8 and Opus 4.7, Sonnet 5, Haiku 4.5, and Mythos Preview by invitation) through a Messages API endpoint (`bedrock-mantle`). The legacy `InvokeModel`/`Converse` integration with ARN-versioned model IDs is documented for Opus 4.6 and earlier; Fable 5.1, Fable 5, Opus 5.5, Opus 5, Sonnet 5, Opus 4.8 and Opus 4.7 are also reachable through `InvokeModel` on `bedrock-runtime`, but the docs point to the Messages API endpoint for full feature parity. Feature support differs between the two integrations, as the structured outputs row shows.

## Model versions, deprecation and migration

*Tested in: CCDV-F 5.3 Model Selection and Tradeoffs (breaking behavior changes across model releases), 2.6 Configuration Management (model version pinning), 2.2 Systems Life Cycle · CCAR-P 4.4, 4.3, 6.5 · CCAR-F: not listed · CCAO-F: not listed*

Models change on a schedule you do not control: new ones launch, old ones are deprecated and then retired, and each generation changes a few request rules. This section covers how a model ID pins behavior, how the lifecycle works, which breaking changes to expect, and a safe migration routine. Choosing between current models is in [Models and how to choose one](#models-and-how-to-choose-one).

### A model ID is a pinned snapshot

"When you use a model ID in an API request, the underlying model remains constant for the lifetime of that ID." ([Model IDs and versions](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions)) The guarantee covers model IDs, not convenience aliases. Updates never change an existing ID: "Anthropic does not update the weights or configuration of an existing model ID. When an updated version is available, it ships under a new model ID."

| ID style | Examples | Behavior |
|---|---|---|
| Dateless, from the Claude 4.6 generation on: `claude-{name}-{major}[-{minor}]` | `claude-opus-5-5`, `claude-sonnet-5`, `claude-sonnet-4-6` | Each maps to one fixed snapshot; major releases such as Sonnet 5 omit the minor segment |
| Dated, before 4.6: `claude-{name}-{major}-{minor}-{YYYYMMDD}` | `claude-sonnet-4-5-20250929`, `claude-haiku-4-5-20251001` | Pinned to that snapshot |
| Alias, before 4.6 | `claude-sonnet-4-5`, `claude-haiku-4-5` | A convenience pointer to the most recent dated snapshot for that minor version |
| Amazon Bedrock | `anthropic.claude-opus-5-5` | `anthropic.` prefix; Opus 4.6 (`anthropic.claude-opus-4-6-v1`) was the last ID with a `-v1` suffix |
| Google Cloud | Dated models use `@` before the date | Platform-specific format |

Two consequences follow. First, a dateless ID is not a moving target: "A common misconception is that dateless model IDs such as `claude-sonnet-4-6` behave as evergreen pointers that route to the latest or best-performing version. That is not the case." ([Model IDs and versions](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions)) Second, a fixed ID can still show small behavior changes, because the serving infrastructure (router, safety classifiers, sampling logic) can change: "If you notice unexpected behavioral differences on a previously stable model ID, an infrastructure update is the most likely cause." Every ID, dated or dateless, has its own deprecation and retirement schedule.

What pinning means in practice:

- **In API code**, send a full model ID, not a pre-4.6 alias such as `claude-sonnet-4-5`, and treat changing it as a release.
- **In Claude Code on a cloud provider**, pin with the `ANTHROPIC_DEFAULT_*_MODEL` variables: "Pinning lets you control when your users move to a new model." ([Claude Code third-party integrations](https://code.claude.com/docs/en/third-party-integrations))
- **The API contract** is versioned separately with the `anthropic-version` header (for example `2023-06-01`, which the SDKs send for you). Within a version Anthropic keeps existing inputs and outputs working but may add optional inputs, add output values, change error conditions and add enum variants such as new streaming event types, so parse defensively.
- **Beta features** are typically dated too, in the `anthropic-beta` header described in [Endpoint and headers](#endpoint-and-headers); the SDKs take a `betas` parameter.
- **Feature detection:** check what a model supports with `GET /v1/models` (see [The current lineup](#the-current-lineup-as-of-september-2026)) instead of assuming it.

### The lifecycle: Active, Legacy, Deprecated, Retired

The [model deprecations page](https://platform.claude.com/docs/en/about-claude/model-deprecations) defines four states:

| State | Definition |
|---|---|
| Active | "The model is fully supported and recommended for use." |
| Legacy | "The model will no longer receive updates and may be deprecated in the future." |
| Deprecated | "The model is still functional but no longer recommended. Anthropic provides a recommended replacement and assigns a retirement date." |
| Retired | "The model is no longer available for use. Requests to retired models will fail." |

Rules around the states:

- Anthropic gives at least 60 days' notice before retiring publicly released models, and notifies affected customers by email and in the documentation.
- The dates apply to Anthropic-operated platforms (the Claude API, Claude Platform on AWS, Microsoft Foundry). Amazon Bedrock and Google Cloud set their own retirement schedules, so a model's status and dates can differ there.
- Deprecated models are likely to be less reliable than active ones.
- To find traffic still on a deprecated model, export the Console Usage page to CSV, which breaks usage down by API key and model.
- Anthropic has committed to long-term preservation of model weights to reduce the downsides of retirement.

A worked timeline: on June 5, 2026 Anthropic notified developers using Claude Opus 4.1 of its retirement on the Claude API; `claude-opus-4-1-20250805` was retired on August 5, 2026, with `claude-opus-4-8` as the replacement. That is 61 days (our count), just over the 60-day minimum.

| Model ID (as of September 2026) | State | Retirement |
|---|---|---|
| `claude-fable-5-1` | Active | Not sooner than September 1, 2027 |
| `claude-opus-5-5` | Active | Not sooner than September 22, 2027 |
| `claude-sonnet-5` | Active | Not sooner than June 30, 2027 |
| `claude-haiku-4-5-20251001` | Active | Not sooner than October 15, 2026 |
| `claude-sonnet-4-5-20250929` | Active | Not sooner than September 29, 2026 |
| `claude-mythos-preview` | Deprecated June 9, 2026 | To be announced (replacement: `claude-mythos-5`) |
| `claude-opus-4-1-20250805` | Retired | August 5, 2026 |
| `claude-sonnet-4-20250514`, `claude-opus-4-20250514` | Retired | June 15, 2026 |

The Haiku 4.5 and Sonnet 4.5 dates are earliest possible retirements, not scheduled ones: as of September 2026 neither model is deprecated, and Anthropic gives at least 60 days' notice before retiring a publicly released model. Because the dates are close, check the deprecations page before relying on either model for new work.

### Breaking changes to expect

Each generation changes a few request rules. As of September 2026 these are the changes that break or silently alter working code:

| Change | Affects | What to do |
|---|---|---|
| Manual extended thinking (`thinking: {type: "enabled", budget_tokens: N}`) returns 400 | Claude 4.7 and later; deprecated on Opus 4.6 and Sonnet 4.6; Sonnet 5 removed it | Remove `budget_tokens`, set `thinking: {type: "adaptive"}`, control depth with `output_config.effort` |
| Thinking cannot be disabled | Claude Opus 5.5, Fable 5.1, Mythos 5.1, Fable 5, Mythos 5 and Mythos Preview (`{"type": "disabled"}` returns 400); on Opus 5, disabling is allowed only at effort `high` or below | Omit `thinking` or send `{"type": "adaptive"}`; choose an effort level instead |
| Adaptive thinking returns 400 | Claude Opus 4.5, Sonnet 4.5 and Haiku 4.5 (extended thinking only) | Keep `budget_tokens` when you move to one of these models, for example to cut cost |
| Thinking on by default, so responses can start with thinking blocks | Opus 5.5, Opus 5, Sonnet 5, and the Fable and Mythos models | Select content blocks by `type`, never `content[0].text`; pass thinking blocks back unmodified in tool loops |
| Default effort changed | Opus 5.5 runs at `medium` when `effort` is omitted (Opus 5 ran at `high`) | Set `effort` explicitly and re-baseline |
| `temperature`, `top_p`, `top_k` at non-default values return 400 | Claude 4.7 and later, Mythos Preview; the Python SDK v1.0 removed the parameters (`TypeError`) | Delete them; steer with the prompt |
| Assistant prefill returns 400 | Claude 4.6 and later, Mythos Preview | Use structured outputs or system prompt instructions |
| `tool_choice` `any` and `tool` return 400 | Claude Opus 5.5, Fable 5.1, Mythos 5.1 | `auto` with `strict: true` (strict tool use), or structured outputs; say in the prompt when the tool applies |
| `computer_20251124` tool returns 400 | Claude Opus 5.5 on the Claude API and Google Cloud | Declare the `computer_toolset_20260801` toolset instead |
| Thinking blocks are bound to the model and the conversation | Claude Opus 5.5 does not read Fable or Mythos thinking blocks, and only Fable 5.1 and Mythos 5.1 (on the Claude API) read its blocks; unreadable blocks are dropped without an error. On accounts created on or after August 31, 2026, replaying an Opus 5.5 thinking block after the system prompt, tools or an earlier message changed returns 400 | Expect lost reasoning after a model switch; keep conversations append-only and change instructions with mid-conversation system messages |
| New tokenizer: approximately 30% more tokens for the same text | Claude 4.7 and later, Mythos Preview | Recount prompts with the token counting endpoint; revisit `max_tokens` and budgets |
| `output_format` moved to `output_config.format` | Structured outputs everywhere (the old form will be removed in a future model release) | Update the request field |
| Beta headers that are no longer needed | `effort-2025-11-24`, `fine-grained-tool-streaming-2025-05-14`, `interleaved-thinking-2025-05-14` (the features no longer need them); `token-efficient-tools-2025-02-19` and `output-128k-2025-02-19` have no effect on Claude 4 and later | Remove them |
| Tool string parameters keep trailing newlines (4.5+); JSON strings may be escaped differently (4.6+) | Tool-call parsing | Use a real JSON parser, not string matching |

Behavior changes do not throw errors but still change results. Claude Opus 4.7 "interprets prompts more literally and explicitly than Claude Opus 4.6, particularly at lower effort levels" ([Opus 5 migration guide](https://platform.claude.com/docs/en/models/opus-5/migration-guide)); on Opus 4.7 and later the thinking display defaults to omitted, a silent change from Opus 4.6's summarized thinking; on Opus 5.5, text between tool calls arrives as progress-update thinking blocks, so an app that streamed that text goes quiet until it sets a `display` value. Thinking modes and effort are covered in [Extended thinking, adaptive thinking and effort](#extended-thinking-adaptive-thinking-and-effort).

Cache minimums also move between models (Opus 4.8 is 1,024 tokens, down from Opus 4.7's 2,048; Opus 5.5, Opus 5 and the Fable and Mythos 5 and 5.1 models are 512, while Sonnet 5 is 1,024), so a prompt that cached on one model may silently stop caching, or start caching, on another. See [Prompt caching](#prompt-caching).

### A migration routine

1. **Read the target model's migration guide.** The index covers Claude Fable 5.1 and Mythos 5.1, Mythos 5 and Fable 5, Opus 5.5, Sonnet 5 and Haiku 4.5.
2. **Apply the mechanical changes.** Swap the model ID, remove rejected parameters and beta headers, move `output_format` to `output_config.format`. In Claude Code, `/claude-api migrate` invokes the bundled Claude API skill to apply model ID swaps and breaking parameter changes.
3. **Recount tokens** against the new model, because the tokenizer may have changed.
4. **Run your evaluation set** on your own prompts and data. Anthropic calls a good evaluation set "the most important step in the process." ([Choosing a model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model))
5. **Re-baseline cost and latency** at the effort level you choose; the Opus 5.5 checklist ends with "Re-baseline cost and latency at your chosen effort level." ([Opus 5.5 migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide))
6. **Roll out gradually.** For its own stateful multi-agent research system, Anthropic describes rainbow deployments, "gradually shifting traffic from old to new versions while keeping both running simultaneously" ([Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)); applying the same pattern to a model switch is our suggestion.
7. **Do it early.** "Test your applications with newer models well before the retirement date of your current model." ([Model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations))

How to run the comparison (A/B tests, regression sets, rollback criteria) is covered in [Upgrading models safely](evaluation-and-reliability.md#upgrading-models-safely).

### How the exams frame it

- **CCDV-F 5.3** asks about "breaking behavior changes across model releases when selecting models for tasks", and **CCDV-F 2.6** names "model version pinning" ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). Our reading: expect scenarios where a request that worked on one model returns 400 on its successor, or where behavior changed after a model switch.
- **CCAR-P 4.4** lists "model mismatch" among system issues to diagnose ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)), and **CCAR-P 6.5** covers lifecycle phases through monitoring and iteration. CCAR-P Sample 3 (a RAG system turns confidently wrong after a document refresh, with latency and model version unchanged) offers "The model weights have silently changed." as a distractor. The official answer is the retrieval or indexing step, and the rationale says "The other options would not be triggered specifically by a document refresh." ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)); the docs add that weights never change under an existing model ID ([Model IDs and versions](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions)). The full item is on the [CCAR-P page](../claude-certified-architect-professional.md#official-sample-questions).

### Decide

- If behavior shifts on an unchanged model ID, suspect a serving-infrastructure update or your own inputs (prompt, retrieval, data) before suspecting the model; weights for an ID never change.
- If a model is deprecated, move before the retirement date, after running your evaluations on the replacement; not on the day requests start failing.
- If you need stable behavior, pin a full model ID and change it deliberately; not an alias, and not a "latest" pointer.

### Traps

- Treating dateless IDs as "latest" pointers.
- Assuming Anthropic's retirement dates also apply on Bedrock and Google Cloud.
- Moving code between models without checking the target model's breaking changes: prefill (Claude 4.6 and later), non-default `temperature`/`top_p`/`top_k` and `budget_tokens` (Claude 4.7 and later), forced `tool_choice` (Opus 5.5, Fable 5.1 and Mythos 5.1). On the exam, the CCAR-F guide still treats `tool_choice` `"any"` and forced tool selection as correct.
- Reading `content[0].text` on a model that can open with a thinking block.
- Comparing model costs with token counts measured on a different tokenizer.

!!! warning "Exam guide vs current docs"

    Where the July 2026 guides name model tiers (CCDV-F 5.3 and CCAO-F D3.2), they name only Opus, Sonnet and Haiku. The current lineup beyond those three tiers is covered in [Models and how to choose one](#models-and-how-to-choose-one). Several of the breaking changes above (forced `tool_choice` returning 400, thinking that cannot be disabled on Opus 5.5) arrived after the guides' July 2026 effective date. Answer tier questions in the guides' terms. The CCAR-F guide (2.3 and 4.3) also still teaches `tool_choice` `"any"` and forced tool selection as the way to guarantee a tool call, so answer those items in the guide's terms even though Opus 5.5, Fable 5.1 and Mythos 5.1 now reject them (see [Structured outputs](#structured-outputs)). Treat per-model facts on this page as dated.

## Exam map

*Tested in: all four exams, section by section (CCAO-F, CCDV-F, CCAR-F, CCAR-P)*

Which official objective each section of this page serves, taken from the four July 2026 exam guides. How to read the labels: CCAO-F objectives are numbered D1.1 to D7.3 in the order the guide lists them; CCDV-F numbers such as 2.3 are the skill's position within its domain; CCAR-F numbers are the guide's own task statement numbers, with K (knowledge) and S (skill) bullets numbered in guide order, plus appendix items (APPX), preparation exercise steps (EX) and sample questions (Q); CCAR-P numbers such as 2.5 are the objective's position within its domain. Apart from the CCAR-F task statement, exercise and question numbers, these are reference labels: the guides list the skills, objectives and knowledge and skill bullets without numbers. Some sections also name a CCDV-F applied area (Section 2) in their own Tested-in line; this table leaves the applied areas out.

| Section | CCAO-F | CCDV-F | CCAR-F | CCAR-P |
|---|---|---|---|---|
| [How a Messages API call works](#how-a-messages-api-call-works) | Not listed | 2.3, 2.4, 5.2 | 1.1-K2, 5.1-K4, APPX-TECH-5; API authentication out of scope (APPX-OUTSCOPE-2) | Not listed |
| [Stop reasons and the agent loop](#stop-reasons-and-the-agent-loop) | Not listed | 1.2, 1.3, 2.3, 8.1 | 1.1-K1 to 1.1-K3, 1.1-S1 to 1.1-S3, EX1-STEP2, APPX-TECH-5, APPX-TECH-1, APPX-INSCOPE-1 | Not listed |
| [Streaming](#streaming) | Not listed | 2.3, 5.2 | Out of scope (APPX-OUTSCOPE-10) | 3.3, 4.5 |
| [Errors, retries and rate limits](#errors-retries-and-rate-limits) | Not listed | 4.1, 2.3, 5.2 | Rate limits out of scope (APPX-OUTSCOPE-11) | 7.3 |
| [Models and how to choose one](#models-and-how-to-choose-one) | D3.2, D3.3 (Sample 2) | 5.3, 2.6 | Not listed; model comparison metrics out of scope (APPX-OUTSCOPE-14) | 2.1, 3.3, 4.5 |
| [Extended thinking, adaptive thinking and effort](#extended-thinking-adaptive-thinking-and-effort) | D1.4 (adapting to task type; the link to effort and thinking settings is our mapping) | 5.1, 5.3, 2.3 | 4.6-K2 (extended thinking versus an independent reviewer) | 2.1, 4.5 |
| [Tokens, context windows and counting](#tokens-context-windows-and-counting) | D3.4 | 5.1, 5.4, 6.1 | 5.1-K3, 5.1-K4, APPX-TECH-12; tokenization specifics out of scope (APPX-OUTSCOPE-16) | 2.4, 4.5 |
| [Vision and PDF input](#vision-and-pdf-input) | Not listed | 2.3 | Out of scope (APPX-OUTSCOPE-9) | Not listed |
| [Prompt caching](#prompt-caching) | Not listed | 2.3, 5.4 | Awareness only (APPX-OUTSCOPE-15) | 2.4, 2.5 (Sample 2, tagged Domain 2), 4.5 |
| [Message Batches](#message-batches) | Not listed | 2.3 (Sample 1) | 4.5-K1 to 4.5-K4, 4.5-S1 to 4.5-S4, APPX-TECH-6, APPX-INSCOPE-15, EX3-STEP4, Q11 | 4.5, 1.6 |
| [Structured outputs](#structured-outputs) | Not listed | 6.3, 2.5 | 4.3-K1 to 4.3-K3, 4.3-S1 to 4.3-S4 (4.3-K4, 4.3-S5 and 4.3-S6 are taught on the prompt engineering page), 4.4-K4, 2.3-K4, APPX-TECH-7, APPX-INSCOPE-13 | 2.2 |
| [Files, citations and search results](#files-citations-and-search-results) | Not listed | 2.3 | 5.6-K2, 5.6-S1, APPX-INSCOPE-18 | 3.5, 3.6 |
| [Cost and usage tracking](#cost-and-usage-tracking) | Not listed | 5.4 | Pricing calculations and billing out of scope (APPX-OUTSCOPE-11, APPX-OUTSCOPE-2); the 50% batch saving is in scope (4.5-K1) | 4.5, 4.1, 4.6 |
| [Claude on cloud platforms](#claude-on-cloud-platforms) | Not listed | 2.3 | Out of scope (APPX-OUTSCOPE-13) | 5.4, 3.2 |
| [Model versions, deprecation and migration](#model-versions-deprecation-and-migration) | Not listed | 5.3, 2.6, 2.2 | Not listed | 4.4, 4.3, 6.5 |

Where this page carries the most weight (skill weights are shares of the whole exam, as the CCDV-F guide states them):

- **CCDV-F:** most of 2.3 Claude API Mechanics (6.8%), all of 5.4 Cost and Token Management (2.8%), and large parts of 5.1 LLM Fundamentals (5.2%), 5.3 Model Selection and Tradeoffs (2.7%) and 6.3 Output Handling (2.6%).
- **CCAR-F:** Task Statements 4.3 (structured output) and 4.5 (batch processing), and the agent-loop mechanics behind 1.1. The appendix places API authentication and billing, streaming, rate limits, pricing calculations, vision, cloud provider configuration, tokenization specifics and caching implementation details out of scope.
- **CCAR-P:** Domain 2, Claude Models, Prompting & Context Engineering (13%), especially 2.1, 2.4 and 2.5, and the cost-performance objective 4.5.
- **CCAO-F:** only model tiers (D3.2, D3.3), context limits (D3.4) and, by our mapping, effort and thinking settings as app users meet them when adapting to task type (D1.4); the rest of this page goes beyond that exam.

??? info "Sources"

    - [Claude Certified Architect, Foundations (CCAR-F) exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): Task 1.1 knowledge and skills, the three loop anti-patterns, Task Statements 4.3, 4.4, 4.5, 5.1 and 5.6, Exercises 1 and 3, sample questions 1, 8 and 11, the appendix's in-scope and out-of-scope technology lists
    - [Claude Certified Developer, Foundations (CCDV-F) exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf): skill descriptions and weights for Claude API Mechanics, Systems Life Cycle, Claude Application Design, Configuration Management, LLM Fundamentals, Technical Fundamentals, Model Selection and Tradeoffs, Cost and Token Management ("cache check-pointing"), Output Handling, Debugging and Error Handling; Sample 1 (Message Batches) and Sample 2 rationales
    - [Claude Certified Associate, Foundations (CCAO-F) exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf): objectives D1.4, D3.2, D3.3 and D3.4, and the Sample 2 rationale on model selection
    - [Claude Certified Architect, Professional (CCAR-P) exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf): objectives 1.6, 2.1 to 2.5, 3.2 to 3.6, 4.1 to 4.6, 5.4, 6.5 and 7.3; Sample 1, 2 and 3 options and rationales on model size, prompt caching, downsizing, temperature and model weights
    - [Claude API overview](https://platform.claude.com/docs/en/api/overview): base URL, endpoints, required headers, `Authorization` versus `x-api-key`, Anthropic-operated and partner-operated platforms, when to use cloud platforms, request size limits by platform
    - [API versions](https://platform.claude.com/docs/en/api/versioning): the `anthropic-version` header and what may change within a version, including new streaming event types
    - [Beta headers](https://platform.claude.com/docs/en/api/beta-headers): the `anthropic-beta` header, dated naming pattern and invalid-name errors
    - [Create a Message (API reference)](https://platform.claude.com/docs/en/api/messages/create): request fields, required parameters, roles, deprecated sampling parameters, `output_config`, `cache_control` TTL values, response and `usage` fields, stop reason values, delta types
    - [Using the Messages API](https://platform.claude.com/docs/en/build-with-claude/working-with-messages): statelessness, multi-turn requests, mid-conversation system messages, prefill removal, image source types, the basic request and response
    - [Mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages): Sonnet 5 exclusion and the rule against untrusted text in system messages
    - [Stop reasons and fallback](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons): every stop reason and its handling, empty responses, `pause_turn` handling, the manual tool loop and dispatcher examples
    - [Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback): refusal categories, billing and server-side fallback
    - [Handle streaming refusals](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/handle-streaming-refusals): `stop_details` on `message_delta` and resetting context after a refusal
    - [How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works): the canonical loop, exit conditions, `pause_turn` for server tools, the regex rule, client tools needing a round trip
    - [Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview): how Claude decides to call a tool under `auto`
    - [Define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools): the four `tool_choice` options, explanation text before tool calls, forced tool use support by model, `any` plus `strict` guarantees
    - [Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls): tool calls in assistant messages, `tool_result` ordering rules, `is_error`
    - [Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use): one `tool_result` per `tool_use`, all in the next user message; models that do not support `tool_choice` `any` or `tool`
    - [Troubleshooting tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/troubleshooting-tool-use): the missing `tool_result` error text
    - [Build a tool-using agent](https://platform.claude.com/docs/en/agents-and-tools/tool-use/build-a-tool-using-agent): the Python and TypeScript error-handling loop adapted in the agent loop example
    - [Tool runner](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner): when to use the Tool Runner and when to write the loop by hand
    - [Streaming messages](https://platform.claude.com/docs/en/build-with-claude/streaming): event flow, delta types, SDK helpers, mid-stream errors, error recovery by model generation
    - [Fine-grained tool streaming](https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming): `eager_input_streaming`, platform support, partial JSON and the legacy beta header
    - [Reducing latency](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-latency): time to first token, streaming for perceived responsiveness, Haiku 4.5 for speed
    - [Claude API errors](https://platform.claude.com/docs/en/api/errors): HTTP error types, request size limits, error shape, request IDs, SDK retries, long requests, common validation errors, the forced `tool_choice` error message and accepted substitutes
    - [Rate limits](https://platform.claude.com/docs/en/api/rate-limits): spend caps and spend-limit errors, tiers, token bucket, cache-aware ITPM, OTPM, per-model limits, Message Batches rate limits per tier, response headers, the cache-rate chart
    - [Service tiers](https://platform.claude.com/docs/en/api/service-tiers): Priority Tier availability and `service_tier` values
    - [Python SDK](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python): exceptions, retries, timeouts, streaming helpers, async client, platform extras for Bedrock, Google Cloud and AWS
    - [TypeScript SDK](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/typescript): exceptions, retries, dynamic timeouts, streaming helpers and cancellation
    - [Client SDKs overview](https://platform.claude.com/docs/en/cli-sdks-libraries/overview): the seven official SDKs and the higher-level agent products
    - [Get an API key](https://platform.claude.com/docs/en/get-api-key): SDKs reading `ANTHROPIC_API_KEY`
    - [Authentication](https://platform.claude.com/docs/en/manage-claude/authentication): Workload Identity Federation access tokens
    - [Workspaces](https://platform.claude.com/docs/en/manage-claude/workspaces): workspace ID format and the Claude Code workspace
    - [Models overview](https://platform.claude.com/docs/en/models/overview): the current lineup table, IDs, prices, context windows, output limits, thinking, default effort, cutoffs, retirement commitments, pinned snapshots, starting recommendation, tokenizer word counts
    - [Choosing a model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model): selection criteria, efficiency-first and capability-first approaches, the selection matrix, the evaluation set as the most important step, effort as a lever, multi-model strategies
    - [Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence): cost per completed task, free wins, caching and batching as the largest levers, the 84% cache-read benchmark, the TTL rule of thumb, the status-line cache breaker, context editing, file and tool search measurements
    - [Model IDs and versions](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions): pinned snapshots, dateless IDs, aliases, platform ID formats, infrastructure changes
    - [List Models (API reference)](https://platform.claude.com/docs/en/api/models/list): the Models API response and capabilities returned per model
    - [Pricing](https://platform.claude.com/docs/en/about-claude/pricing): model prices, cache multipliers and break-even, batch prices, residency, regional endpoint premium, fast mode, long context, tokenizer change, tool-use system prompt tokens, server tool pricing, Managed Agents runtime, CCUs, the cache-read worked example, the Haiku/Sonnet/Opus rule of thumb, tokens per character
    - [Model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations): lifecycle state definitions, 60-day notice, platform scope of dates, usage audit, status table, Opus 4.1 retirement, sampling parameters deprecated from Claude 4.7, Python SDK v1.0 removal
    - [Claude Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/overview): release date and always-on adaptive thinking
    - [What's new in Claude Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/whats-new-opus-5-5): breaking changes, default effort, progress-update thinking blocks
    - [Claude Fable 5.1](https://platform.claude.com/docs/en/models/fable-5-1/overview): release date, prices and the Mythos 5.1 relationship
    - [Claude Mythos 5.1](https://platform.claude.com/docs/en/models/mythos-5-1/overview): invite-only status
    - [Claude Sonnet 5](https://platform.claude.com/docs/en/models/sonnet-5/overview): release date and behavior changes from Sonnet 4.6
    - [What's new in Claude Sonnet 5](https://platform.claude.com/docs/en/models/sonnet-5/whats-new-sonnet-5): about 30% more tokens than Sonnet 4.6
    - [Claude Haiku 4.5](https://platform.claude.com/docs/en/models/haiku-4-5/overview): release date, ID, alias and manual extended thinking
    - [Migrating to Claude Haiku 4.5](https://platform.claude.com/docs/en/models/haiku-4-5/migration-guide): the rule against setting both `temperature` and `top_p`
    - [Claude Opus 5](https://platform.claude.com/docs/en/models/opus-5/overview): legacy model ID, limits, price, thinking and default effort
    - [Migrating to Claude Opus 5](https://platform.claude.com/docs/en/models/opus-5/migration-guide): thinking-on-by-default breakage, removed parameters, tokenizer range, legacy beta headers, tool string changes, Opus 4.7 behavior
    - [Claude Fable 5](https://platform.claude.com/docs/en/models/fable-5/overview): legacy model ID, limits and price
    - [Claude Opus 4.8](https://platform.claude.com/docs/en/models/opus-4-8/overview): legacy model ID, limits and price
    - [Claude Opus 4.7](https://platform.claude.com/docs/en/models/opus-4-7/overview): release date, legacy model ID, limits and price
    - [Claude Opus 4.6](https://platform.claude.com/docs/en/models/opus-4-6/overview): legacy model ID, limits and price
    - [Claude Opus 4.5](https://platform.claude.com/docs/en/models/opus-4-5/overview): legacy model ID, limits and price
    - [Claude Sonnet 4.6](https://platform.claude.com/docs/en/models/sonnet-4-6/overview): legacy model ID, limits and price
    - [Claude Sonnet 4.5](https://platform.claude.com/docs/en/models/sonnet-4-5/overview): legacy model ID, limits and price
    - [Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking): thinking blocks, signatures, display modes, tool-use rules, preservation, caching interaction, output limits, sampling restrictions, forced tool use with adaptive thinking
    - [Preserved thinking](https://platform.claude.com/docs/en/build-with-claude/preserved-thinking): the prefix check on replayed thinking blocks, `prefix_mismatch_behavior` and batch behavior
    - [Troubleshooting thinking](https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting): the per-model thinking configuration table
    - [Extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking): budget rules and tuning, interleaved thinking header, rejection on Claude 4.7 and later, migration to adaptive thinking and effort
    - [Effort](https://platform.claude.com/docs/en/build-with-claude/effort): levels, defaults, per-model recommendations, per-message effort, cache interaction
    - [Fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode): availability, pricing, platforms, model-specific behavior, rate limits
    - [Task budgets](https://platform.claude.com/docs/en/build-with-claude/task-budgets): the beta header, 20,000-token minimum and soft-hint behavior
    - [Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows): what counts, cached prefixes still in the window, sizes by model, context rot, thinking and the window, context awareness, overflow behavior, compaction
    - [Token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting): the free counting endpoint, estimates, unsupported inputs, rate limits, caching, tokenizer differences
    - [Glossary](https://platform.claude.com/docs/en/about-claude/glossary): tokens, pretraining, temperature and non-determinism, time to first token
    - [Vision](https://platform.claude.com/docs/en/build-with-claude/vision): image source types, formats, limits, visual-token cost and resolution tiers, limitations, FAQ
    - [PDF support](https://platform.claude.com/docs/en/build-with-claude/pdf-support): PDF requirements, how pages are processed, cost, best practices, large PDFs through the Files API, Bedrock Converse behavior, unsupported binary formats
    - [Claude Code: MCP](https://code.claude.com/docs/en/mcp): WebSocket MCP servers and when to use HTTP instead
    - [Claude Code: headless mode](https://code.claude.com/docs/en/headless): `--output-format stream-json`
    - [Agent SDK: streaming input](https://code.claude.com/docs/en/agent-sdk/streaming-vs-single-mode): the Agent SDK's streaming input mode
    - [Claude Code: errors](https://code.claude.com/docs/en/errors): Claude Code's retry policy and `CLAUDE_CODE_MAX_RETRIES`
    - [Agent SDK: the agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop): the SDK loop, turn and budget limits
    - [Building effective agents (Anthropic engineering)](https://www.anthropic.com/engineering/building-effective-agents): stopping conditions such as maximum iterations
    - [Reducing cost and improving performance (Claude blog)](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform): stronger model at lower effort, diminishing returns at max effort, flat effort curves
    - [Claude Academy: Temperature](https://academy.claude.com/courses/building-with-the-claude-api/temperature): tokenization, prediction and sampling steps, and the lesson's claim that temperature 0 is deterministic
    - [Claude Academy: Next token prediction](https://academy.claude.com/courses/ai-capabilities-and-limitations/next-token-prediction): fluency, fabrication and where fabrication concentrates
    - [Claude Academy: Selecting the right effort setting](https://academy.claude.com/tutorials/how-to-select-the-right-effort-setting-for-claude-cowork-and-chat): signs of too little or too much effort, frontier model at lower effort, cost per task
    - [Claude Academy: Parametric memory and context](https://academy.claude.com/tutorials/parametric-memory-and-context): why a fresh chat can be cheaper and faster
    - [Claude Help Center: Change the model, effort and thinking settings](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings): effort levels and thinking settings in the Claude apps
    - [Claude Help Center: Troubleshoot Claude error messages](https://support.claude.com/en/articles/12466728-troubleshoot-claude-error-messages): length-limit remedies in the apps
    - [Claude Help Center: How usage and length limits work](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work): starting a new conversation near usage limits
    - [RFC 6455: The WebSocket Protocol](https://www.rfc-editor.org/rfc/rfc6455.html): WebSocket handshake and full-duplex messaging
    - [MDN: Server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events): server push over SSE
    - [MDN: Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events): SSE is one-way and uses `text/event-stream`
    - [Claude Certified Architect, Professional certification page (Anthropic Partner Academy)](https://anthropic-partners.skilljar.com/claude-certified-architect-professional-certification): the "Claude with Amazon Bedrock" and "Claude on Google Cloud" prep courses
    - [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): automatic and explicit breakpoints, prefix order, 4 slots, 20-block lookback, TTLs, minimum lengths, invalidation table, usage fields, pre-warming, isolation, multipliers
    - [Tool use with prompt caching](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching): caching tool definitions, `mcp_toolset` breakpoints, deferred tools, automatic breakpoints on server tool results
    - [Cache diagnostics](https://platform.claude.com/docs/en/build-with-claude/cache-diagnostics): beta header, `previous_message_id`, `cache_miss_reason` values and fixes
    - [Manage tool context](https://platform.claude.com/docs/en/agents-and-tools/tool-use/manage-tool-context): caching does not reduce tokens in context
    - [Batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing): batch size, 24-hour expiry, 29-day results, `custom_id` rules, result types, unsupported parameters, server tools and `pause_turn`, caching in batches, 300k output beta
    - [Message Batches API reference](https://platform.claude.com/docs/en/api/messages/batches): endpoints, `processing_status` values, cancellation and deletion rules, result ordering
    - [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs): JSON outputs, SDK helpers and schema transformation, supported and unsupported schema features, complexity limits, invalid outputs, grammar caching, incompatibilities, platform availability, migration from `output_format`
    - [Strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use): `strict: true` placement and guarantees, the non-strict type warning, toolsets that reject it, PHI in schemas
    - [Tool reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference): which tools accept `strict`
    - [Programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling): strict tools not supported with programmatic calling
    - [Files API](https://platform.claude.com/docs/en/build-with-claude/files): status, upload and reference flow, block mapping, limits, expiry, listing, pricing, workspace scoping and the `file_id` warning
    - [Citations](https://platform.claude.com/docs/en/build-with-claude/citations): enabling citations, document types and location fields, indexing rules, token costs, streaming deltas, caching, incompatibility with structured outputs
    - [Search results](https://platform.claude.com/docs/en/build-with-claude/search-results): `search_result` fields, citation location type, all-or-nothing and tool-result rules, availability, error handling
    - [Effective context engineering for AI agents (Anthropic engineering)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): the "just in time" retrieval pattern
    - [Claude pricing](https://claude.com/pricing): the "50 free hours of usage daily" phrasing for code execution
    - [Usage and Cost API](https://platform.claude.com/docs/en/manage-claude/usage-cost-api): credentials, usage and cost endpoints, bucket limits, grouping, data latency, platform availability, other analytics APIs
    - [Analytics APIs](https://platform.claude.com/docs/en/manage-claude/analytics-api): Admin and Analytics API keys are not interchangeable
    - [API and data retention](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention): ZDR and HIPAA eligibility of batches, Files API, prompt caching and structured outputs
    - [Data residency](https://platform.claude.com/docs/en/manage-claude/data-residency): `inference_geo` pricing, supported models and platforms
    - [Features overview](https://platform.claude.com/docs/en/build-with-claude/overview): batch processing availability and ZDR status
    - [Claude in Amazon Bedrock](https://platform.claude.com/docs/en/build-with-claude/claude-in-amazon-bedrock): Messages API endpoint, operator access, authentication, quotas, logging, unsupported features
    - [Claude on Amazon Bedrock (legacy)](https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy): `InvokeModel` body version and inference profiles for Opus 4.6 and earlier
    - [Claude on Google Cloud's Agent Platform](https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai): request format, endpoint types and premium, `AnthropicVertex` client, unsupported features, payload limit
    - [Claude in Microsoft Foundry](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry): hosting options, endpoints and authentication, CCU billing, missing rate-limit headers, SDK support, unsupported features
    - [Claude Platform on AWS](https://platform.claude.com/docs/en/build-with-claude/claude-platform-on-aws): Anthropic-operated, authentication and headers, capacity pool, compliance guidance pointing to Bedrock, ZDR opt-in
    - [Migrating to Claude Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide): `/claude-api migrate`, re-baselining cost and latency
    - [Migration guides](https://platform.claude.com/docs/en/about-claude/models/migration-guide): which migration guides exist
    - [Release notes](https://platform.claude.com/docs/en/release-notes/overview): Files API general availability date, structured outputs rename, forced `tool_choice` changes on the newest models
    - [Agent Skills guide](https://platform.claude.com/docs/en/build-with-claude/skills-guide): changing the Skills list breaks the prompt cache
    - [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing): tool result clearing invalidates cached prefixes
    - [Compaction](https://platform.claude.com/docs/en/build-with-claude/compaction): the two server-side compaction modes
    - [Compaction at a token threshold](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold): cache breakpoint placement with compaction, `usage.iterations` billing
    - [Compaction on demand](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand): platform availability, request restrictions
    - [Claude Code third-party integrations](https://code.claude.com/docs/en/third-party-integrations): Agent Platform naming, model pinning variables
    - [Claude Code costs](https://code.claude.com/docs/en/costs): cloud-provider metrics not sent to Anthropic, OpenTelemetry coverage
    - [Claude Code checkpointing](https://code.claude.com/docs/en/checkpointing): checkpoints that capture code state before each prompt
    - [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system): rainbow deployments
    - [Prompt caching with Claude](https://claude.com/blog/prompt-caching): older "10% of the base input token price" description
    - [Introducing the Message Batches API](https://claude.com/blog/message-batches-api): the 2024 limit of 10,000 queries per batch, and pointers for Bedrock and Vertex AI users
