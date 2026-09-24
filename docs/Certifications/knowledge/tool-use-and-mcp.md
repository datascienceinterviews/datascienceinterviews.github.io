---
title: "Tool Use and the Model Context Protocol for the Claude Certifications"
description: "How Claude tool use and the Model Context Protocol work for the Claude certification exams: exact parameters, correct examples, decision rules and exam traps."
last_reviewed: 2026-09-23
---

# Tool Use and the Model Context Protocol

Tool use is how Claude acts outside the conversation: Claude emits a structured call, and your application or Anthropic's servers run it. The Model Context Protocol (MCP) turns that connection into a shared protocol, so one server can expose tools, resources and prompts to any compatible client. This page teaches both for the Claude certification exams that test them, with an exam map at the end; product facts are as of September 2026, and where the documentation has moved since the July 2026 exam guides, the section shows both versions and which wording to expect on the exam (the guide's).

## How tool use works

*Tested in: CCAR-F 1.1, EX1-STEP2, APPX-TECH-5, APPX-INSCOPE-1 · CCDV-F D8.1 Tool Implementation, D1.2 Agent Construction with Claude, D1.3 Agent Patterns and Frameworks, D2.3 Claude API Mechanics*

Tool use (also called function calling) lets Claude call functions that you define or that Anthropic provides. Claude decides when to call a tool from the user's request and the tool's description, then returns a structured call. Your application executes client tools; Anthropic executes server tools. The docs state the contract in one line: "The model never executes anything on its own." ([How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works))

### Where the code runs: three buckets

The docs call where the code executes "The primary axis along which tools differ": every tool falls into one of three buckets, and the bucket decides what your application is responsible for ([How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works)).

| Bucket | Examples | Who publishes the schema | Who runs the code | Your application's job |
|---|---|---|---|---|
| User-defined (client-executed) | Your own tools, such as the guide's `get_customer` and `lookup_order` | You | Your application | Execute each call and return a `tool_result` |
| Anthropic-schema (client-executed) | `memory`, `bash`, `text_editor`, `computer`, `browser` | Anthropic | Your application | Execute each call and return a `tool_result` |
| Server-executed | `web_search`, `web_fetch`, `code_execution`, `tool_search` | Anthropic | Anthropic | Enable the tool and read the final answer; never build a `tool_result` for it |

Three consequences follow. Claude never sees a user-defined tool's implementation, only the schema you supplied and the result you returned, so what the tool does, when to use it and what its inputs mean has to be stated in the definition. An Anthropic-schema tool's schema is trained in, which is why [Anthropic-defined tools](#anthropic-defined-tools) recommends it over an equivalent of your own; the individual tools are listed there too. Server-executed tools usually finish before you see the response; the main exceptions are a paused server loop (`pause_turn`) and a turn that also calls a client tool. The server-side set also covers MCP connectors and the advisor tool.

### The client-tool round trip

The Claude API has no separate `tool` or `function` role. Tool calls arrive inside assistant messages and tool results go back inside user messages. Because the model cannot run your code, every client tool call is a round trip:

1. Send a request with your `tools` array and the user message.
2. Claude responds with `stop_reason: "tool_use"` and one or more `tool_use` blocks. Each block carries an `id` (used to match the result), the tool `name`, and an `input` object conforming to your `input_schema` (only `strict: true` guarantees that; see [Defining a tool](#defining-a-tool)).
3. Execute each tool and format each output as a `tool_result` block whose `tool_use_id` matches the call's `id`.
4. Send a new request containing the original messages, the assistant's response, and a user message holding the `tool_result` blocks.
5. Repeat while `stop_reason` is `"tool_use"`. Any other stop reason (`"end_turn"`, `"max_tokens"`, `"stop_sequence"` or `"refusal"`) exits the loop: Claude has either produced a final answer or stopped for another reason that your application should handle.

The Messages API is stateless, so the harness re-sends everything on each turn: past actions, tool descriptions and instructions. Appending each result to that history is what lets Claude reason about its next action, which is what CCAR-F 1.1-K2 and 1.1-S2 test, and the guide's first preparation exercise asks you to build this loop and handle both `"tool_use"` and `"end_turn"` correctly (EX1-STEP2).

### Agentic harness dispatch

Anthropic defines an agent harness as "the software scaffolding around a model: the loop, tools, context management, and guardrails that turn raw intelligence into a working agent" ([Agent harness design](https://claude.com/blog/harnessing-claudes-intelligence)). The [Claude Code glossary](https://code.claude.com/docs/en/glossary) puts it the same way: "Claude Code is the harness; Claude is the model inside it." Dispatch is the harness step that routes each call to real code: run the tool in your codebase that corresponds to the tool `name`, passing the tool `input`. In Anthropic's Managed Agents design, the harness is the loop that calls Claude and routes its tool calls to the relevant infrastructure, and every tool gets one interface, `execute(name, input) → string` (a name and input go in, a string comes back); that interface supports custom tools, MCP servers and Anthropic's own tools.

The excerpt below is "Ring 4: Error handling" of Anthropic's [build a tool-using agent](https://platform.claude.com/docs/en/agents-and-tools/tool-use/build-a-tool-using-agent) tutorial. `run_tool` is the dispatcher: it maps a name to a handler and raises on an unknown name. The loop sends any exception back as an `is_error: true` result instead of crashing, so Claude can retry with corrected input, ask the user for clarification, or explain the limitation.

=== "Python"

    ```python
    # Excerpt from Ring 4: the imports (json, anthropic), client and tools
    # (create_calendar_event, list_calendar_events) are defined above this
    # point, and messages just after run_tool, in the tutorial's full example.
    def run_tool(name, tool_input):
        if name == "create_calendar_event":
            if "attendees" in tool_input and len(tool_input["attendees"]) > 10:
                raise ValueError("Too many attendees (max 10)")
            return {"event_id": "evt_123", "status": "created", "title": tool_input["title"]}
        if name == "list_calendar_events":
            return {"events": [{"title": "Existing meeting", "start": "14:00", "end": "15:00"}]}
        raise ValueError(f"Unknown tool: {name}")

    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=1024,
        tools=tools,
        messages=messages,
    )

    while response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                try:
                    result = run_tool(block.name, block.input)
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
                    )
                except Exception as exc:
                    # Signal failure so Claude can retry or ask for clarification.
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(exc),
                            "is_error": True,
                        }
                    )

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model="claude-opus-5-5",
            max_tokens=1024,
            tools=tools,
            messages=messages,
        )
    ```

=== "TypeScript"

    ```typescript
    // Excerpt from Ring 4: the import, client and tools (create_calendar_event,
    // list_calendar_events) are defined above this point, and messages just
    // after runTool, in the tutorial's full example.
    function runTool(name: string, input: Record<string, unknown>) {
      if (name === "create_calendar_event") {
        const attendees = input.attendees as string[] | undefined;
        if (attendees && attendees.length > 10) {
          throw new Error("Too many attendees (max 10)");
        }
        return { event_id: "evt_123", status: "created", title: input.title };
      }
      if (name === "list_calendar_events") {
        return {
          events: [{ title: "Existing meeting", start: "14:00", end: "15:00" }],
        };
      }
      throw new Error(`Unknown tool: ${name}`);
    }

    let response = await client.messages.create({
      model: "claude-opus-5-5",
      max_tokens: 1024,
      tools,
      messages,
    });

    while (response.stop_reason === "tool_use") {
      const toolResults: Anthropic.ToolResultBlockParam[] = [];
      for (const block of response.content) {
        if (block.type === "tool_use") {
          try {
            const result = runTool(block.name, block.input as Record<string, unknown>);
            toolResults.push({
              type: "tool_result",
              tool_use_id: block.id,
              content: JSON.stringify(result),
            });
          } catch (err) {
            // Signal failure so Claude can retry or ask for clarification.
            toolResults.push({
              type: "tool_result",
              tool_use_id: block.id,
              content: String(err),
              is_error: true,
            });
          }
        }
      }

      messages.push({ role: "assistant", content: response.content });
      messages.push({ role: "user", content: toolResults });

      response = await client.messages.create({
        model: "claude-opus-5-5",
        max_tokens: 1024,
        tools,
        messages,
      });
    }
    ```

Three details in that loop are worth memorizing. All results for one assistant turn go back together in a single user message. Each result carries the `tool_use_id` of its call. A failure becomes `is_error: true` rather than an exception that kills the agent; the tutorial notes that the flag is the only difference from a successful result. The formatting rules are in [Returning tool results and errors](#returning-tool-results-and-errors).

Computer use and browser use toolsets add one dispatch rule. A member `tool_use` block also carries a `toolset_name` (`"computer"` or `"browser"`), and its `name` is the member being called, such as `screenshot` or `navigate`. Dispatch on the pair (`toolset_name`, `name`), because a custom tool in the same request can share a member's name, and echo the same `toolset_name` on the `tool_result`: a member result that omits it is rejected. CCAR-F lists computer use (browser automation, desktop interaction) as out of scope.

### Tool Runner or a manual loop

The client SDKs ship a Tool Runner (beta, in all seven SDKs: Python, TypeScript, C#, Go, Java, PHP and Ruby) that runs your tools, handles the request and response cycle, and manages conversation state. It loops until Claude returns a message without a tool use, or until `max_iterations` if you set it. In Python, `@beta_tool` derives the JSON schema from the function's arguments and docstring. In TypeScript, `betaZodTool()` defines a type-safe tool from a Zod schema (Zod 3.25.0 or higher), and `betaTool()` takes JSON Schema instead. When a tool throws, the runner returns the error to Claude as an `is_error: true` result.

=== "Python"

    ```python
    import json
    from anthropic import Anthropic, beta_tool

    client = Anthropic()

    @beta_tool
    def get_weather(location: str, unit: str = "fahrenheit") -> str:
        """Get the current weather in a given location.

        Args:
            location: The city and state, e.g. San Francisco, CA
            unit: Temperature unit, either 'celsius' or 'fahrenheit'
        """
        return json.dumps({"temperature": "20°C", "condition": "Sunny"})

    runner = client.beta.messages.tool_runner(
        model="claude-opus-5-5",
        max_tokens=1024,
        tools=[get_weather],
        messages=[{"role": "user", "content": "What's the weather like in Paris?"}],
    )
    for message in runner:
        print(message)
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";
    import { betaZodTool } from "@anthropic-ai/sdk/helpers/beta/zod";
    import { z } from "zod";

    const client = new Anthropic();

    const getWeatherTool = betaZodTool({
      name: "get_weather",
      description: "Get the current weather in a given location",
      inputSchema: z.object({
        location: z.string().describe("The city and state, e.g. San Francisco, CA"),
        unit: z.enum(["celsius", "fahrenheit"]).default("fahrenheit").describe("Temperature unit")
      }),
      run: async (input) => {
        return JSON.stringify({ temperature: "20°C", condition: "Sunny" });
      }
    });

    const finalMessage = await client.beta.messages.toolRunner({
      model: "claude-opus-5-5",
      max_tokens: 1024,
      tools: [getWeatherTool],
      messages: [{ role: "user", content: "What's the weather like in Paris?" }]
    });

    for (const block of finalMessage.content) {
      if (block.type === "text") {
        console.log(block.text);
      }
    }
    ```

**Decide.** If some calls need human-in-the-loop approval, custom logging or conditional execution, write the manual loop, because the docs point to it for exactly those cases. Otherwise the Tool Runner removes the boilerplate. In Python and TypeScript you can still inspect results before they reach Claude with `generate_tool_call_response()` or `generateToolResponse()`.

### When a tool is the right design

| Use a tool when the task needs | Skip the tool when |
|---|---|
| An action with side effects (send an email, write a file, update a record) | Claude can answer from training alone (summarizing, translating, general knowledge) |
| Fresh or external data (current prices, today's weather, a database's contents) | The exchange is one-shot Q&A with no side effects: nothing to execute |
| A guaranteed-shape output: a JSON object with specific fields, not prose that happens to contain them | Tool-calling latency would dominate a trivial response: every call is at least one extra round trip, and for lightweight tasks the overhead can exceed the work |
| A call into an existing system (database, internal API, filesystem) | |

The docs give a sharp test: "if you're writing a regex to extract a decision from model output, that decision should have been a tool call." ([How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works))

When the existing system must also be reusable across several Claude applications and maintained independently of any one app, CCDV-F Sample 3 keys an MCP server that exposes the operations as tools; the item and its rationale are worked in [How the official sample frames it](#how-the-official-sample-frames-it). MCP is taught from [MCP architecture](#mcp-architecture) onward.

### Server tools run their own loop

Server tools run a loop inside Anthropic's infrastructure, so one request can trigger several searches or code runs before you see a response. That loop has an iteration limit, 10 iterations per request by default; a turn that hits it returns `stop_reason: "pause_turn"` instead of `"end_turn"`, meaning the work is not finished, and you re-send the conversation (including the paused response) to let Claude continue. When Claude calls a server tool and a client tool in the same parallel group, control comes back to you with `"tool_use"` before the server tool runs. Both cases are covered in [Anthropic-defined tools](#anthropic-defined-tools) and [Parallel tool calls](#parallel-tool-calls).

### How the exams frame it

**Decide.** CCAR-F 1.1-S1 asks for loop control that continues when `stop_reason` is `"tool_use"` and terminates when it is `"end_turn"`. Objective 1.1-K3 tests the distinction between model-driven decision-making, where Claude reasons about which tool to call next based on context, and pre-configured decision trees or tool sequences. In the agentic loop Claude chooses the next tool. When critical business logic requires a specific sequence, enforce it in code with a programmatic prerequisite (1.4-S1 and sample question 1; see [Controlling tool choice](#controlling-tool-choice)).

**Traps.** The guide names three anti-patterns (1.1-S3), and each is a wrong-answer shape:

- Parsing natural language signals, such as Claude saying it has finished, to decide when to stop.
- Arbitrary iteration caps as the *primary* stopping mechanism. A cap as a safety limit is normal: Anthropic's own agent guidance calls stopping conditions such as a maximum number of iterations common, the Tool Runner has `max_iterations`, and the Agent SDK has `max_turns` and `max_budget_usd`. The trap is using the cap instead of `stop_reason`.
- Treating assistant text as a completion signal. Claude often writes explanatory text before its `tool_use` blocks, so text in a response does not mean the work is finished.

!!! warning "Exam guide vs current docs"

    The guide frames loop control as `"tool_use"` versus `"end_turn"`. The docs (as of September 2026) say the loop continues while `stop_reason` is `"tool_use"` and exits on any other value, including `"max_tokens"`, `"stop_sequence"` and `"refusal"`, and they add `"pause_turn"` for server-tool loops. On the exam, continue on `"tool_use"` and stop on `"end_turn"`; in real code, handle the others too. Stop reasons in depth: [Stop reasons and the agent loop](claude-api.md#stop-reasons-and-the-agent-loop). Loops in agent frameworks: [The agentic loop](agents-and-agent-sdk.md#the-agentic-loop).

## Defining a tool

*Tested in: CCAR-F 2.1-K2, 4.3-K1, 4.3-K3, 4.3-S1, 4.4-K4, EX3-STEP1, APPX-TECH-5, APPX-TECH-7, APPX-INSCOPE-13 · CCDV-F D8.1 Tool Implementation, D2.3 Claude API Mechanics, D6.3 Output Handling*

Client tools go in the top-level `tools` parameter of the request. A user-defined tool is a `name`, a `description` and an `input_schema`. Optional properties compose on the same tool, so one definition can set `defer_loading`, `cache_control` and `strict` together. Anthropic-schema tools are declared differently, by a date-versioned `type` (see [Anthropic-defined tools](#anthropic-defined-tools)).

### The core fields

| Field | What it holds | Exact rule |
|---|---|---|
| `name` | The identifier Claude calls | Must match `^[a-zA-Z0-9_-]{1,128}$`: letters, digits, `_` and `-`, 1 to 128 characters, no dots or spaces |
| `description` | Plain text: what the tool does, when it should be used, and how it behaves | An extremely detailed description is "by far the most important factor in tool performance" ([Define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools)); see [Writing tool descriptions that steer selection](#writing-tool-descriptions-that-steer-selection) |
| `input_schema` | A JSON Schema object defining the expected parameters | Claude's `input` is meant to conform to it; only `strict: true` guarantees that it does |
| `input_examples` | Optional array of example input objects | Each example must be valid against `input_schema`, or the request returns a 400 |

A definition with a required parameter, an optional enum parameter and examples (the docs' own example):

```json
{
  "name": "get_weather",
  "description": "Get the current weather in a given location",
  "input_schema": {
    "type": "object",
    "properties": {
      "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
      "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature"}
    },
    "required": ["location"]
  },
  "input_examples": [
    {"location": "San Francisco, CA", "unit": "fahrenheit"},
    {"location": "Tokyo, Japan", "unit": "celsius"},
    {"location": "New York, NY"}
  ]
}
```

The one-line description here is the docs' minimal example. A production description should be several sentences long, as the next section explains.

### Optional properties on any tool

| Property | Effect | Notes |
|---|---|---|
| `strict` | Guarantees schema validation on tool names and inputs | Not on `mcp_toolset`, `computer_toolset_20260801` or `browser_toolset_20260801` |
| `cache_control` | Sets a prompt-cache breakpoint at this tool definition | All tools (on the computer and browser toolsets, on the toolset entry itself); put it on the last tool to cache the whole tools prefix |
| `defer_loading` | Leaves the tool out of the initial system prompt; it loads when tool search returns a `tool_reference` for it | All tools (per member inside `configs` on the computer and browser toolsets); see [Progressive discovery with tool search](#progressive-discovery-with-tool-search) |
| `allowed_callers` | An array restricting who may call the tool: `"direct"` (the default when omitted) is a normal `tool_use` block; `"code_execution_20260120"` (or the interchangeable `"code_execution_20260521"`) lets code in a code execution sandbox call it | All tools except `mcp_toolset`; the computer and browser toolsets accept only `["direct"]`. Not a security boundary; see [Programmatic tool calling](#programmatic-tool-calling) |
| `input_examples` | Example inputs | User-defined and Anthropic-schema client tools, except the computer and browser toolsets; not server tools |
| `eager_input_streaming` | `true` streams this tool's input as it is generated, without server-side buffering or JSON validation; `false` keeps standard buffered streaming | User-defined tools only |

### Strict tool use

`strict: true` is a top-level property of the tool definition, next to `name`, `description` and `input_schema`. It constrains token sampling to schema-valid output (grammar-constrained sampling). Without it, Claude can return an incompatible type (`"2"` instead of `2`) or omit a required field. With it, the input always follows `input_schema` and the tool `name` is always valid.

```json
{
  "name": "search_flights",
  "strict": true,
  "input_schema": {
    "type": "object",
    "properties": {
      "destination": {"type": "string"},
      "departure_date": {"type": "string", "format": "date"},
      "passengers": {"type": "integer", "enum": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    },
    "required": ["destination", "departure_date"],
    "additionalProperties": false
  }
}
```

Strict mode and JSON outputs (`output_config.format`) are the two structured-outputs features. They share the same JSON Schema limits and can be used together: JSON outputs control what Claude says, strict tool use validates how Claude calls your functions. JSON outputs are taught in [Structured outputs](claude-api.md#structured-outputs).

**Schema rules under strict mode**

| Rule | Detail |
|---|---|
| Objects | `additionalProperties` must be `false` |
| Not supported | Recursive schemas; numerical constraints (`minimum`, `maximum`, `multipleOf`); string length constraints (`minLength`, `maxLength`); array constraints beyond `minItems` of 0 or 1 |
| String formats supported | `date-time`, `time`, `date`, `duration`, `email`, `hostname`, `uri`, `ipv4`, `ipv6`, `uuid` |
| Regex `pattern` | No backreferences, lookaround, word boundaries or large `{n,m}` ranges; the error reads `Unsupported regex feature in pattern field` |

**Per-request limits (as of September 2026)**

| Limit | Value |
|---|---|
| Tools with `strict: true` | 20 (non-strict tools do not count) |
| Optional parameters, across all strict tool schemas and JSON output schemas | 24 |
| Parameters using union types (`anyOf` or type arrays), across all strict schemas | 16 |
| Internal grammar-size limits | Exceeding them returns a 400 with `Schema is too complex for compilation`, even when every limit above is met; a 180-second compilation timeout is the final stop-gap |
| Compiled grammar cache | 24 hours from last use; changing only `name` or `description` does not invalidate it |

When a design hits these limits, the docs list four strategies to try in order: mark only critical tools as strict; reduce optional parameters (make parameters `required` where possible); simplify nested structures; and split many strict tools across separate requests or subagents.

Behaviors to design around:

- Enum and `const` capitalization is not guaranteed, even under strict tool use: Claude may return a value that differs from your schema only in capitalization, with no error and no special `stop_reason`. Compare enum values case-insensitively and avoid values that differ only in case.
- Besides enum casing, the schema guarantee has two more documented exceptions. A refusal (`stop_reason: "refusal"`, HTTP 200, billed) takes precedence over schema constraints, and a response cut off at `max_tokens` may be incomplete; retry the latter with a higher `max_tokens`.
- Required properties are emitted first, then optional ones, whatever order the schema lists them in.
- Strict tool use is HIPAA eligible, but protected health information must not appear in the schema itself (property names, enum or `const` values, patterns), because compiled schemas are cached separately.
- Strict tools cannot be called through programmatic tool calling, and the computer and browser toolset entries reject `strict: true`.

**What strict mode does not fix.** The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) tests (4.3-K3) "That strict JSON schemas via tool use eliminate syntax errors but do not prevent semantic errors (e.g., line items that don't sum to total, values in wrong fields)". The docs are consistent with this: strict mode guarantees the shape, not the truth of the values. If an answer choice claims a schema alone will catch a total that does not add up, it is wrong. The guide's own remedies are validation plus a retry that appends the specific validation error (4.4-K1), and self-check fields such as a `calculated_total` extracted alongside the `stated_total` so a discrepancy can be flagged (4.4-S4); both are taught in [Validation, retry and feedback loops](prompt-engineering.md#validation-retry-and-feedback-loops). Schema idioms for extraction (nullable fields, an `"other"` value plus a detail field) are in [Structured output with tools and JSON schemas](prompt-engineering.md#structured-output-with-tools-and-json-schemas); in a large strict schema, budget them against the 24 optional and 16 union-typed parameter limits above.

!!! warning "Exam guide vs current docs"

    The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) calls "Tool use (tool_use) with JSON schemas as the most reliable approach for guaranteed schema-compliant structured output, eliminating JSON syntax errors" (4.3-K1). As of September 2026 the docs offer two dedicated structured-outputs features backed by constrained decoding: JSON outputs (`output_config.format`) and strict tool use (`strict: true`). A plain, non-strict tool call can still return a wrong type or omit a required field. On the exam, answer in the guide's terms (tool use with a JSON schema is the reliable route to schema-compliant output); its appendix phrase "strict mode for syntax error elimination" matches `strict: true`. In code you ship, set `strict: true` to get the guarantee.

### Input examples

`input_examples` help most for complex tools with nested objects, optional parameters or format-sensitive inputs. They cost prompt tokens: roughly 20 to 50 per simple example and 100 to 200 for a complex nested object. In Anthropic's internal testing, tool use examples raised accuracy on complex parameter handling from 72% to 90%. Anthropic's guidance is realistic data (real city names, plausible prices), a mix of minimal, partial and full specification patterns, and 1 to 5 examples per tool, added only where correct usage is not obvious from the schema. The description still comes first: the docs' advice is to prioritize descriptions and consider `input_examples` for complex tools.

### Extraction tools

For extraction (CCAR-F 4.3-S1), the tool is a schema rather than an action: define a tool whose `input_schema` is the record you want and read the structured data from the `tool_use` block's `input`. Anthropic's cookbook ["Extracting Structured JSON using Claude and Tool Use"](https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/extracting_structured_json.ipynb) works this way, with tools such as `print_summary` and `print_entities`, and it also shows an open-ended `input_schema` for keys you cannot know up front, with the prompt explaining how to use the tool. To make sure the extraction call happens at all, pair the tool with a `tool_choice` of `any` or a forced tool on models that support forcing (the guide's answer, 4.3-S2 and 4.3-S3). The newest models reject forcing; the replacement is in [Where forced tool use is rejected](#where-forced-tool-use-is-rejected-as-of-september-2026).

### Reading what Claude sends back

- Claude often writes explanatory text before its `tool_use` blocks. Treat that text like any other assistant text and do not rely on its formatting.
- Never string-match serialized tool input: Unicode and forward-slash escaping differs between model versions. Parse it with `json.loads()` or `JSON.parse()`.
- When streaming, a tool's input arrives as `input_json_delta` events carrying `partial_json` strings; the final `tool_use.input` is always an object. With `eager_input_streaming` the input is not validated before it streams, so you can receive partial or invalid JSON. Guard the parse, and if it fails, return the raw string wrapped as `{"INVALID_JSON": "..."}` in an `is_error: true` result. Details: [Streaming](claude-api.md#streaming).

The per-tool `eager_input_streaming` field replaces the legacy `fine-grained-tool-streaming-2025-05-14` beta header. A request that still sends the header turns fine-grained streaming on for tools that leave the field unset, and an explicit `false` keeps buffered streaming for that tool. CCAR-F lists streaming API implementation as out of scope; CCDV-F D2.3 includes streaming.

Agent SDK custom tools are defined differently, as MCP tools on an in-process server; see [The Claude Agent SDK](#the-claude-agent-sdk) below and [Custom tools in the SDK](agents-and-agent-sdk.md#custom-tools-in-the-sdk).

## Writing tool descriptions that steer selection

*Tested in: CCAR-F 2.1, 2.4-S3, EX1-STEP1, APPX-INSCOPE-4 · CCDV-F D8.1 Tool Implementation*

The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) treats tool descriptions as "the primary mechanism LLMs use for tool selection" and warns that minimal descriptions make selection among similar tools unreliable (2.1-K1). The [Define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools) page agrees, calling extremely detailed descriptions "by far the most important factor in tool performance." With the default `tool_choice` of `auto`, Claude calls a tool when the request maps to that tool's described capability and the answer is not already in context; it responds directly for stable knowledge, creative tasks and conversational turns. The description is therefore what Claude matches the request against (our inference from the docs' wording).

When you pass `tools`, the API builds a special system prompt from your tool definitions, the tool configuration and your own system prompt. The docs publish its template, below. Your tool definitions and your own system prompt land in the same system prompt; our reading is that this is one reason system prompt wording can change tool selection:

```text
In this environment you have access to a set of tools you can use to answer the user's question.
{{ FORMATTING INSTRUCTIONS }}
String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular expressions.
Here are the functions available in JSONSchema format:
{{ TOOL DEFINITIONS IN JSON SCHEMA }}
{{ USER SYSTEM PROMPT }}
{{ TOOL CONFIGURATION }}
```

### What a description must say

| Element | Anthropic guidance | CCAR-F wording |
|---|---|---|
| Purpose | What the tool does | Each tool's purpose (2.1-S1) |
| When to use it, and when not to | When it should be used (and when it shouldn't); clear boundaries from other tools | Boundary explanations (2.1-K2); when to use it versus similar alternatives (2.1-S1) |
| Inputs | What each parameter means and how it affects the tool's behavior; input format requirements | Input formats (2.1-K2); expected inputs (2.1-S1) |
| Outputs | What data it returns (one of the four points the docs credit to their good example) | Outputs (2.1-S1) |
| Limits | Important caveats or limitations, such as what information the tool does not return; edge cases | Edge cases (2.1-K2) |
| Examples | Example usage | Example queries (2.1-K2) |

The "Anthropic guidance" column combines the [Define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools) checklist with [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents), which says "A good tool definition often includes example usage, edge cases, input format requirements, and clear boundaries from other tools."

Aim for at least 3 to 4 sentences per description, more for a complex tool. Anthropic's [tool-design post](https://www.anthropic.com/engineering/writing-tools-for-agents) suggests describing the tool as you would to a new hire on your team, making implicit context explicit (specialized query formats, niche terminology, relationships between resources), and naming parameters unambiguously: `user_id` rather than `user`.

The docs contrast a good and a poor definition of the same tool (both verbatim from [Define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools)):

=== "Good"

    ```json
    {
      "name": "get_stock_price",
      "description": "Retrieves the current stock price for a given ticker symbol. The ticker symbol must be a valid symbol for a publicly traded company on a major US stock exchange like NYSE or NASDAQ. The tool will return the latest trade price in USD. It should be used when the user asks about the current or most recent price of a specific stock. It will not provide any other information about the stock or company.",
      "input_schema": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string",
            "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
          }
        },
        "required": ["ticker"]
      }
    }
    ```

=== "Poor"

    ```json
    {
      "name": "get_stock_price",
      "description": "Gets the stock price for a ticker.",
      "input_schema": {
        "type": "object",
        "properties": {
          "ticker": {
            "type": "string"
          }
        },
        "required": ["ticker"]
      }
    }
    ```

The good version says what the tool does, when to use it, what it returns and what `ticker` means, and it states what the tool will not provide. In the docs' words, the poor one "is too brief and leaves Claude with many open questions about the tool's behavior and usage." ([Define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools))

### Separate similar tools by when, not only what

Writing about agents connected to many MCP servers, Anthropic's [advanced tool use post](https://www.anthropic.com/engineering/advanced-tool-use) reports that the most common failures are wrong tool selection and incorrect parameters, especially when tools have similar names such as `notification-send-user` and `notification-send-channel`. The troubleshooting guide traces "Claude calls tool A when you wanted tool B" to description ambiguity, and its fix is: "Sharpen descriptions. Differentiate tools by WHEN to use them, not only WHAT they do." ([Troubleshooting tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/troubleshooting-tool-use))

CCAR-F builds its objectives on the same idea, with three concrete moves:

- **Diagnose overlap.** Near-identical descriptions cause misrouting, as with `analyze_content` versus `analyze_document` (2.1-K3).
- **Rename and re-describe.** Remove functional overlap by renaming `analyze_content` to `extract_web_results` and giving it a web-specific description (2.1-S2).
- **Split a generic tool.** Break a generic `analyze_document` into `extract_data_points`, `summarize_content` and `verify_claim_against_source`, each with a defined input and output contract (2.1-S3). When to split and when to consolidate is covered in [Designing a tool set](#designing-a-tool-set).

The guide's first preparation exercise asks you to build this deliberately: 3 to 4 MCP tools, including at least two with similar functionality that need careful descriptions to avoid selection confusion (EX1-STEP1).

**MCP tools competing with built-in tools.** CCAR-F 2.4-S3 tests enhancing MCP tool descriptions to explain capabilities and outputs in detail, so the agent does not prefer a built-in tool such as Grep over a more capable MCP tool. Claude Code's default tool set now handles Grep and Glob differently from the guide (see the note under [What each option is](#what-each-option-is)), but the objective's point still holds. Under Claude Code's tool search, MCP tool descriptions load on demand and are truncated at a default length; those rules, and how to answer the guide's related 2.4-K3 wording, are in [What Claude sees once servers connect](#what-claude-sees-once-servers-connect).

### Worked case: CCAR-F sample question 2

In the guide's sample question 2 (scenario: Customer Support Resolution Agent), production logs show the agent calling `get_customer` when users ask about orders instead of `lookup_order`; both tools have minimal descriptions ("Retrieves customer information" / "Retrieves order details") and accept similar identifier formats. The keyed answer, B, is to expand each description with the input formats it handles, example queries, edge cases, and boundaries explaining when to use it versus similar tools. Anthropic's rationale: "Tool descriptions are the primary mechanism LLMs use for tool selection. When descriptions are minimal, models lack the context to differentiate between similar tools." ([CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)) The full item is reproduced on the [CCAR-F page](../claude-certified-architect-foundations.md#official-sample-questions).

The same rationale explains each distractor, and those explanations generalize ([CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)):

| Tempting fix | Why the guide rejects it as a first step |
|---|---|
| Few-shot examples of correct routing in the system prompt | They "add token overhead without fixing the underlying issue" |
| A routing layer that pre-selects tools by keyword | It "is over-engineered and bypasses the LLM's natural language understanding" |
| Consolidating both tools into one `lookup_entity` | A valid architectural choice, but it requires more effort than a first step warrants when the immediate problem is inadequate descriptions |

**Decide.** If Claude picks the wrong tool and the descriptions are thin, fix the descriptions first: the rationale in the [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says option B "directly addresses this root cause" with a low-effort fix. If the descriptions are already complete and misrouting persists, restructure the tool set: the rationale calls consolidation "a valid architectural choice", and renaming or splitting overlapping tools is what 2.1-S2 and 2.1-S3 test. A keyword routing layer and few-shot routing examples are the shapes the rationale rejects (our rule, derived from that rationale and those objectives).

### System prompt wording steers selection too

The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) warns (2.1-K4) that "keyword-sensitive instructions can create unintended tool associations", and 2.1-S4 asks you to review system prompts for keyword-sensitive instructions that might override well-written descriptions. The docs show how strongly wording moves tool use ([Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)):

| System prompt line | Effect |
|---|---|
| `Use the tools to investigate before responding.` | A light push toward more tool use |
| `Always call a tool first before responding.` | A stronger push |
| `Use your judgment about whether to call a tool or respond directly.` | Keeps triggering behavior conservative |

The same page adds that to require a tool call rather than rely on prompting, you set `tool_choice` (next section).

Anthropic's [prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) say Claude Opus 4.5 and Claude Opus 4.6 are more responsive to the system prompt than previous models: "If your prompts were designed to reduce undertriggering on tools or skills, these models may now overtrigger. The fix is to dial back any aggressive language." Where you might have written "CRITICAL: You MUST use this tool when...", the page suggests more normal prompting such as "Use this tool when...". Phrasing also decides whether Claude acts at all: "Can you suggest some changes to improve this function?" can get suggestions only, while "Change this function to improve its performance." gets Claude to make the change with its tools.

Stale instructions cost money on newer models: a newer model follows over-specific instructions to the letter, adding tool rounds. Anthropic's [cost guide](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence) describes a support-desk prompt whose mandatory procedure forced four tool calls on a simple ticket, and reports that, on a support-desk evaluation, prompts written for Claude Opus 4.8 cost 36% more per ticket on Claude Opus 5 with no change in accuracy. The same patterns tend to appear in tool descriptions and skills, so the cost guide advises auditing prompts against the model you run now and again whenever you change models, with descriptions and skills included.

### Missing and guessed parameters

When a required parameter is missing from the prompt, Claude Opus is much more likely to recognize that and ask for it; Claude Sonnet might ask, especially when prompted to think before outputting a tool request, but it might also infer a reasonable value. The docs show a `get_weather` call where Claude guessed "New York, NY" for a location the user never gave, and say the behavior is not guaranteed, especially for more ambiguous prompts and less capable models. Anthropic's sample prompt for parallel calls also tells Claude never to use placeholders or guess missing parameters. Our rule: if a wrong guess is costly, say so in the description and the system prompt, and validate the input in code before acting on it.

### Symptom and first fix

| Symptom | Likely cause | First fix |
|---|---|---|
| Claude calls tool A when you wanted tool B | Description ambiguity | Sharpen descriptions; differentiate the tools by when to use them |
| Claude never calls your tool | Tool name collision or overly generic schema | Check for duplicate names in the tool list; add `input_examples` |
| Wrong parameter types | Model guessing at an ambiguous schema | Add `strict: true` (if the schema is in the supported subset) or `input_examples` |
| A parameter your schema does not have | Over-generation without strict mode | Add `strict: true` (if the schema is in the supported subset) |
| Values outside your enum | Missing strict mode or too large an enum | Shrink the enum or add `input_examples` showing valid choices |
| Missing required parameters | The description lacked information | During development, retry with more detailed `description` values |

What tool-call metrics (invalid-parameter errors, redundant calls) say about a description is covered in [Evaluate the tool set](#evaluate-the-tool-set).

Anthropic's own results show the payoff. Claude Sonnet 3.5 reached state-of-the-art SWE-bench Verified performance after precise refinements to tool descriptions. In the multi-agent research system, a tool-testing agent that rewrote a flawed MCP tool's description produced a 40% decrease in task completion time for later agents using the new description. When Claude's web search tool launched, Claude was needlessly appending 2025 to the query parameter until Anthropic improved the tool description.

## Controlling tool choice

*Tested in: CCAR-F 2.3-K4, 2.3-S4, 2.3-S5, 4.3-K2, 4.3-S2, 4.3-S3, 1.4-S1 (contrast with enforced order), Q1, APPX-TECH-5, APPX-INSCOPE-13 · CCDV-F D2.3 Claude API Mechanics, D8.1 Tool Implementation, D5.3 Model Selection and Tradeoffs, D5.4 Cost and Token Management*

`tool_choice` decides whether Claude may, must, or must not call a tool on a request. The API accepts four values.

| Value | What Claude does | Default? |
|---|---|---|
| `auto` | Decides whether to call any tool; may answer in plain text instead | Yes, when `tools` are provided |
| `any` | Must call one of the provided tools, but chooses which | No |
| `tool` (with `name`) | Must call the named tool | No |
| `none` | Cannot call any tool | Yes, when no `tools` are provided |

The request forms, including the parallel-call switch covered in [Parallel tool calls](#parallel-tool-calls):

```json
{"tool_choice": {"type": "auto"}}
{"tool_choice": {"type": "any"}}
{"tool_choice": {"type": "tool", "name": "get_weather"}}
{"tool_choice": {"type": "none"}}
{"tool_choice": {"type": "auto", "disable_parallel_tool_use": true}}
{"tool_choice": {"type": "any", "disable_parallel_tool_use": true}}
```

A forced call on a model that supports it. This is the docs' own example, and it uses `claude-opus-5`: the newest models reject forcing (see below).

=== "Python"

    ```python
    client = anthropic.Anthropic()

    tools = [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ]

    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=1024,
        tools=tools,
        tool_choice={"type": "tool", "name": "get_weather"},
        messages=[{"role": "user", "content": "What's the weather like in San Francisco?"}],
    )
    ```

=== "TypeScript"

    ```typescript
    const client = new Anthropic();

    const response = await client.messages.create({
      model: "claude-opus-5",
      max_tokens: 1024,
      tools: [
        {
          name: "get_weather",
          description: "Get the current weather in a given location",
          input_schema: {
            type: "object",
            properties: {
              location: {
                type: "string",
                description: "The city and state, e.g. San Francisco, CA"
              }
            },
            required: ["location"]
          }
        }
      ],
      tool_choice: { type: "tool", name: "get_weather" },
      messages: [{ role: "user", content: "What's the weather like in San Francisco?" }]
    });
    ```

### What forcing changes

- **No preamble.** With `any` or `tool`, the API prefills the assistant message to force a tool call, so Claude writes no natural-language explanation before the `tool_use` block, even if explicitly asked to (the docs say testing shows this should not reduce performance). If you want an explanation and a specific tool, keep `auto` and put an explicit instruction in the user message, as in the docs' example `What's the weather like in London? Use the get_weather tool in your response.`
- **Guarantee a call and its shape.** On models that support forced tool use, `any` plus `strict: true` on your tools guarantees both that a tool is called and that its input follows your schema. `any` alone guarantees the call, not the shape.
- **Caching.** Changing `tool_choice` invalidates cached message blocks; tool definitions and the system prompt stay cached. If `tool_choice` must change mid-conversation, place cache breakpoints before the change.

### Where forced tool use is rejected (as of September 2026)

Where forced tool use is unsupported, `any` and `tool` fail while `auto` and `none` keep working.

| Model or setting | `any` and `tool` | Use instead |
|---|---|---|
| Manual extended thinking (`thinking: {type: "enabled"}`) | Not supported; the request errors, because forcing is incompatible with manual extended thinking | `auto` or `none`. Adaptive thinking does not block forcing (Claude Opus 5 supports forced tool use with thinking on) |
| Claude Opus 5.5, Claude Fable 5.1, Claude Mythos 5.1 | 400 `invalid_request_error` with the message `tool_choice: type "tool" and "any" are not supported for this model.`, also from the token counting endpoint, regardless of thinking settings | `auto` with strict tool use for schema-valid inputs, or structured outputs for a response in a fixed JSON shape; prompting still influences which tool `auto` picks, and `none` is also accepted |

The Claude Opus 5.5 migration guide shows the replacement for a forced call: strict tools, `auto`, and the instruction moved into the user message. The snippet assumes an existing `tools` list that contains a `get_weather` tool, like the forced example above.

```python
client.messages.create(
    model="claude-opus-5-5",
    max_tokens=1024,
    # strict tool use: every call matches the tool's input_schema
    tools=[{**tool, "strict": True} for tool in tools],
    tool_choice={"type": "auto"},
    messages=[
        {
            "role": "user",
            "content": "What's the weather in Paris? Use the get_weather tool.",
        }
    ],
)
```

The release notes date the change: Claude Fable 5.1 and Claude Mythos 5.1 on September 1, 2026, and Claude Opus 5.5, which launched on September 22, 2026. Manual extended thinking itself is deprecated on Claude Opus 4.6 and Claude Sonnet 4.6 and rejected with a 400 by Claude 4.7 and later models; Claude Sonnet 4.5, Claude Opus 4.5 and Claude Haiku 4.5 support only extended thinking. On the newest models thinking is already on and needs no configuration. Thinking modes are taught in [Extended thinking, adaptive thinking and effort](claude-api.md#extended-thinking-adaptive-thinking-and-effort).

Other places a forced choice is refused:

- A `tool` choice cannot name the computer use or browser use toolset or one of its members; use `auto`, `any` or `none`.
- `tool_choice` cannot name a tool whose `allowed_callers` omits `"direct"`. The fix the docs give is to add `"direct"` to that tool's `allowed_callers`, or remove the tool from `tool_choice` and let Claude invoke it from code. Nor can you force programmatic calling of a specific tool through `tool_choice`.
- An on-demand compaction request must leave out a `tool_choice` of `any` or `tool`, and a `max_tokens: 0` request is rejected when combined with `any` or `tool`.

### What each choice costs

Supplying tools makes the API add a tool-use system prompt, whose size depends on the model and on `tool_choice` (the table assumes at least one tool is provided). Those tokens come on top of the other tool tokens: the `tools` parameter (names, descriptions and schemas), `tool_use` blocks and `tool_result` blocks, all billed as normal input and output tokens. Server tools can add usage-based charges as well (web search, for example, charges per search). With no `tools`, `none` adds 0 system prompt tokens.

| Model | `auto`, `none` | `any`, `tool` |
|---|---|---|
| Claude Opus 5.5 | 286 tokens | Not supported (the docs list only `auto` and `none`) |
| Claude Opus 5 | 286 tokens | 406 tokens |
| Claude Opus 4.8 | 290 tokens | 410 tokens |
| Claude Opus 4.7 | 675 tokens | 804 tokens |
| Claude Opus 4.6, Claude Sonnet 4.6 | 497 tokens | 589 tokens |
| Claude Opus 4.5, Claude Sonnet 4.5, Claude Haiku 4.5 | 496 tokens | 588 tokens |
| Claude Sonnet 5 | 354 tokens | 474 tokens |

Our reading of the models in this table: forcing adds about 90 to 130 system prompt tokens per request (406 minus 286 is 120 on Claude Opus 5; 588 minus 496 is 92 on the 4.5 models; our arithmetic). The docs' table also lists older retired models, not shown here: Claude Opus 4.1, Claude Sonnet 4 and Claude Haiku 3.5 (retired except on Bedrock and Google Cloud) and Claude Opus 4 (retired except on Google Cloud). This table serves CCDV-F D5.4 Cost and Token Management; CCAR-F lists rate limiting, quotas and API pricing calculations as out of scope.

### Recipes the exams test

| Goal | Setting | Where it comes from |
|---|---|---|
| Guarantee structured output when several extraction schemas exist and the document type is unknown | `"any"` | CCAR-F 4.3-S2 |
| Guarantee a tool call rather than conversational text | `"any"` | CCAR-F 2.3-S5 |
| Make one extraction run first, such as `extract_metadata` before enrichment tools | `{"type": "tool", "name": "extract_metadata"}`, then the later steps in follow-up turns | CCAR-F 2.3-S4, 4.3-S3 |
| Get an explanation and still steer to one tool | `auto` plus an explicit instruction in the user message | Define tools docs |
| Guarantee both a call and schema-valid input | `any` plus `strict: true` | Define tools docs |
| Schema-valid tool inputs on Claude Opus 5.5, Fable 5.1 or Mythos 5.1 | `auto` plus `strict: true`, with the prompt saying when the tool applies; or JSON outputs for a fixed response shape | Define tools, errors and Opus 5.5 migration docs |

**Traps.**

- `auto` does not guarantee a tool call; the model may return text instead (4.3-K2). On the exam, and on models that support forcing, if the task needs a call every time the answer is `any` or a forced tool, not stronger prompt wording: the docs say a system prompt line steers triggering, and that to require a tool call rather than rely on prompting you set `tool_choice`. On Claude Opus 5.5, Claude Fable 5.1 and Claude Mythos 5.1, use `auto` with strict tool use and say in the prompt when the tool applies (table above).
- `any` does not pick the tool for Claude. To make a particular tool run, force it by name.
- A forced call arrives with no explanatory text, so code that expects prose before the tool call will not find any.
- Forcing a tool while manual extended thinking is on is an error, not a silent downgrade.

**Forcing a first call is not the same as enforcing an order.** `tool_choice` is set per request, so forcing `extract_metadata` makes it run on that request, and the guide has you process the later steps in follow-up turns (2.3-S4). When a sequence must hold for critical business logic, CCAR-F sample question 1 keys a programmatic prerequisite that blocks `lookup_order` and `process_refund` until `get_customer` has returned a verified customer ID because, in Anthropic's rationale for that question in the [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf), "programmatic enforcement provides deterministic guarantees that prompt-based approaches cannot". The objective behind it is 1.4-S1. Our rule: use `tool_choice` to make one step happen on one request; use a code gate or hook when a rule must hold across the whole conversation (see [Hooks in the SDK](agents-and-agent-sdk.md#hooks-in-the-sdk)).

!!! warning "Exam guide vs current docs"

    **Options.** The CCAR-F guide lists three `tool_choice` options: `"auto"`, `"any"` and forced tool selection (2.3-K4). The docs list four, adding `none`.

    **Forcing on the newest models.** The guide teaches `any` and forced selection as the way to guarantee a tool call (2.3-S4, 2.3-S5, 4.3-S2, 4.3-S3). As of September 2026 both return a 400 on Claude Opus 5.5, Claude Fable 5.1 and Claude Mythos 5.1, and error with manual extended thinking; the docs point to `auto` with strict tool use, or structured outputs. They still work on models such as Claude Opus 5. On the exam, answer in the guide's terms: `any` guarantees a call, a forced tool guarantees a specific call. In code you ship, check the model first. Our reading: this is an example of what the [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) lists in D5.3 as "breaking behavior changes across model releases"; see [Model versions, deprecation and migration](claude-api.md#model-versions-deprecation-and-migration).

## Parallel tool calls

*Tested in: CCAR-F 1.3-S3, EX4-STEP2 · CCDV-F D8.1 Tool Implementation, D2.3 Claude API Mechanics*

By default, Claude may call several tools in a single response, and Claude 4 and later models make parallel calls by default when a request benefits from more than one tool. The response then has `stop_reason: "tool_use"` and can contain several `tool_use` blocks in one assistant turn.

### Running the calls

The API does not prescribe an execution order. You can run the calls concurrently (`asyncio.gather`, `Promise.all`), one by one in the order they appear, or in any mix that suits your tools.

**Decide.** Independent, read-only operations are usually safe to run in parallel for lower latency. Tools with side effects, shared state or ordering requirements may be better run sequentially. Whatever you choose, the reply format below does not change.

The Agent SDK makes this choice for you: read-only tools such as `Read`, `Glob`, `Grep` and MCP tools marked read-only can run concurrently, while state-modifying tools such as `Edit`, `Write` and `Bash` run sequentially. SDK custom tools run sequentially unless you set `readOnlyHint: true` in their annotations (the fifth argument to `tool()` in TypeScript, the `annotations` keyword of `@tool` in Python). The annotation is metadata, not enforcement: a tool marked read-only can still write if its handler does, so keep the hint accurate.

=== "Python"

    ```python
    from claude_agent_sdk import tool, ToolAnnotations

    @tool(
        "get_temperature",
        "Get the current temperature at a location",
        {"latitude": float, "longitude": float},
        annotations=ToolAnnotations(readOnlyHint=True),  # Lets Claude batch this with other read-only calls
    )
    async def get_temperature(args):
        return {"content": [{"type": "text", "text": "..."}]}
    ```

=== "TypeScript"

    ```typescript
    import { tool } from "@anthropic-ai/claude-agent-sdk";
    import { z } from "zod";

    tool(
      "get_temperature",
      "Get the current temperature at a location",
      { latitude: z.number(), longitude: z.number() },
      async (args) => ({ content: [{ type: "text", text: `...` }] }),
      { annotations: { readOnlyHint: true } } // Lets Claude batch this with other read-only calls
    );
    ```

### Returning the results

Return one `tool_result` for each `tool_use` block, all together in the next user message, matched by `tool_use_id` and placed before any text in that message. The docs name incorrect result formatting as the most common reason Claude stops making parallel calls: sending each result in its own user message "teaches" Claude to avoid parallel calls ([Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use)).

```text
// Wrong: separate user messages reduce parallel tool use
[
  {"role": "assistant", "content": [tool_use_1, tool_use_2]},
  {"role": "user", "content": [tool_result_1]},
  {"role": "user", "content": [tool_result_2]}  // Separate message
]
// Correct: one user message with all results maintains parallel tool use
[
  {"role": "assistant", "content": [tool_use_1, tool_use_2]},
  {"role": "user", "content": [tool_result_1, tool_result_2]}  // Single message
]
```

If you decide not to run one of the calls, for example because you ran the batch sequentially and an earlier call failed, you still owe it a result: `is_error: true` and a short explanation.

```json
{
  "type": "tool_result",
  "tool_use_id": "toolu_02",
  "is_error": true,
  "content": "Not executed: the preceding write_file call failed."
}
```

The loop in [How tool use works](#how-tool-use-works) already follows both rules: it collects every result for a turn into one list, including `is_error` results for calls that failed, and sends one user message.

### Encouraging, limiting and disabling parallel calls

To encourage parallel calls, the [parallel tool use docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use) suggest this system prompt line for Claude 4 and later models: "For maximum efficiency, whenever you need to perform multiple independent operations, invoke all relevant tools simultaneously rather than sequentially." Anthropic's [prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) say parallel calling is steerable and that their sample prompt can boost it to about 100%. That longer version also tells Claude to keep dependent calls sequential and never to guess missing parameters:

```text
<use_parallel_tool_calls>
If you intend to call multiple tools and there are no dependencies between the tool
calls, make all of the independent tool calls in parallel. Prioritize calling tools
simultaneously whenever the actions can be done in parallel rather than sequentially.
For example, when reading 3 files, run 3 tool calls in parallel to read all 3 files into
context at the same time. Maximize use of parallel tool calls where possible to increase
speed and efficiency. However, if some tool calls depend on previous calls to inform
dependent values like the parameters, do NOT call these tools in parallel and instead
call them sequentially. Never use placeholders or guess missing parameters in tool
calls.
</use_parallel_tool_calls>
```

If dependent calls still show up in the same batch, the [parallel tool use docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use) suggest adding "Only batch tool calls that are independent of each other." to the system prompt. If a parallel call fails because its prerequisite had not run yet, return `is_error: true` with the natural error message and Claude reissues the call on the next turn. To tone parallel execution down in general, the [best-practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) sample is "Execute operations sequentially with brief pauses between each step to ensure stability."

To turn parallel calls off, set `disable_parallel_tool_use: true` inside the `tool_choice` object. It is not a top-level request parameter.

| `tool_choice` type | With `disable_parallel_tool_use: true` |
|---|---|
| `auto` | At most one tool call per response; Claude may still answer in text |
| `any` or `tool` | Exactly one tool call |

The second row needs forced tool choice to be accepted: as of September 2026, Claude Opus 5.5, Claude Fable 5.1 and Claude Mythos 5.1 reject `any` and `tool`, and so does any request with manual extended thinking (`thinking: {type: "enabled"}`) (see [Controlling tool choice](#controlling-tool-choice)).

Three rules catch people out:

- The flag must be set on the request that returns the `tool_use`. Setting it on a later request has no effect on calls already made.
- Changing it invalidates the messages cache.
- Programmatic tool calling does not support `disable_parallel_tool_use: true`.

Model behavior varies (as of September 2026): Claude Fable 5.1 may issue fewer parallel calls than earlier models in long agent loops, such as custom coding agents, bash or text editor harnesses and computer use, where the next reads are only implied. Standard function calling is unaffected. For that case the best-practices page says to send the parallel-calls instruction as a turn-scoped system message after each round of tool results (turn-scoped system messages are in beta and need the `mid-conversation-system-clear-at-2026-08-21` header).

### Toolsets are stricter

When Claude returns several computer use or browser use member calls in one turn (a batch action), run them sequentially in the order they appear and stop at the first failure. Every block still gets a result, and each member result echoes its `toolset_name`: the normal result for each action that succeeded, `is_error: true` with a description for the one that failed, and `is_error: true` with the exact text the tool defines for each skipped call: `Not executed: an earlier computer action in this turn failed.` for computer use, `Not executed: an earlier action in this turn failed.` for browser use. A request that leaves any block in the batch unanswered is rejected. If a person must confirm consequential actions, check before each block runs, because one batch can complete a multistep action within a single turn.

### A server tool and a client tool in one group

When Claude calls a server tool and a client tool in the same parallel group, the API does not run the server tool yet. The response comes back with `stop_reason: "tool_use"`, not `"pause_turn"`, and the server tool appears as a `server_tool_use` block with no matching result block. The docs say there is no other marker: detect the state by that unmatched `id`. An `mcp_tool_use` block from the MCP connector behaves the same way.

```json
{
  "stop_reason": "tool_use",
  "content": [
    {"type": "text", "text": "I'll fetch the article and check your system at the same time."},
    {"type": "server_tool_use", "id": "srvtoolu_01HxbWnMRmbWyMfUtJKC45rA", "name": "web_fetch", "input": { "url": "https://example.com/article" }},
    {"type": "tool_use", "id": "toolu_01PjgRJLbXrXEMZwDNYLnBqk", "name": "run_command", "input": { "command": "uname -a" }}
  ]
}
```

The only valid follow-up is a user message containing nothing but the client `tool_result` blocks, one per client `tool_use`, sent with the same `tools` array; the API then runs the deferred server tool at the start of that request. Text after those results ends the assistant turn and fails with a 400 that names the unresolved server tool. To give Claude more input, send it as a separate user message after the turn completes.

```json
{
  "role": "user",
  "content": [
    {"type": "tool_result", "tool_use_id": "toolu_01PjgRJLbXrXEMZwDNYLnBqk", "content": "Linux demo-host 6.8.0-52-generic x86_64 GNU/Linux"}
  ]
}
```

### Parallel subagents

The same mechanism drives multi-agent fan-out. The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) tests (1.3-S3) spawning parallel subagents "by emitting multiple Task tool calls in a single coordinator response rather than across separate turns", and preparation exercise 4 asks you to measure the latency gain against sequential execution (EX4-STEP2). In Anthropic's multi-agent research system, two kinds of parallelism (the lead agent starting 3 to 5 subagents at once, and each subagent using 3 or more tools at once) cut research time by up to 90% for complex queries. Orchestration design is in [Multi-agent orchestration](agents-and-agent-sdk.md#multi-agent-orchestration).

!!! warning "Exam guide vs current docs"

    The guide calls the subagent-spawning tool "Task". Current Claude Code and Agent SDK releases name it `Agent`; the rename came in Claude Code v2.1.63, and `Task` still works as an alias. In the Agent SDK the tool appears as `"Agent"` in `tool_use` blocks but as `"Task"` in the `system:init` tools list, so in current releases a coordinator's parallel fan-out shows up as several `Agent` blocks in one response (our reading of those two facts). Expect "Task" on the exam.

**Traps.** The plausible wrong answers follow from the rules above:

- Spawning independent subagents one per turn. 1.3-S3 wants multiple Task calls in a single coordinator response.
- Returning parallel results in separate user messages. The docs say this teaches Claude to avoid parallel calls.
- Putting `disable_parallel_tool_use` at the top level of the request, or on a later request. It belongs inside `tool_choice` on the request that returns the `tool_use`.
- Running a computer use or browser use batch concurrently. Those batches run in order and stop at the first failure.

## Returning tool results and errors

*Tested in: CCAR-F 1.1-K2, 1.1-S2, 2.2, 5.1-K3, 5.1-S3, 5.3-K2, 5.3-K4, 5.3-S2, APPX-INSCOPE-7, EX1-STEP3, Q8 · CCDV-F D8.1 Tool Implementation, D4.1 Debugging and Error Handling, D7.1 AI Application Security*

A client tool's output goes back to Claude as a `tool_result` content block inside a new message with role `user`. Why that growing history matters (CCAR-F 1.1-K2 and 1.1-S2) is covered in [The client-tool round trip](#the-client-tool-round-trip); this section covers the result's fields, the ordering rules the API enforces, how to report a failure, and what a result should contain.

| Field | Required | Rule |
|---|---|---|
| `type` | Yes | `"tool_result"` |
| `tool_use_id` | Yes | The `id` of the `tool_use` block this answers |
| `content` | No | A string, or a list of content blocks of type `text`, `image`, `document` or `search_result`; it can be omitted entirely |
| `is_error` | No | `true` when the tool execution failed |
| `toolset_name` | Only for computer use or browser use members | Must echo the `toolset_name` of the member's `tool_use` block; a member result that omits it is rejected |

### Ordering rules the API enforces

1. **Adjacent.** `tool_result` blocks must immediately follow their `tool_use` blocks in the message history. No message may sit between the assistant's `tool_use` message and your `tool_result` message.
2. **Results first.** In that user message, the docs put it bluntly: `tool_result` blocks "must come FIRST in the content array. Any text must come AFTER all tool results." ([Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)) Text before a result returns a 400.
3. **Results only, in mixed turns.** If the same assistant turn also called a server tool that has no result yet, the user message may contain only `tool_result` blocks (the full case is in [A server tool and a client tool in one group](#a-server-tool-and-a-client-tool-in-one-group)).

Skip a result, put content before the results, or send a next message with no `tool_result` blocks at all, and the API answers with `tool_use ids were found without tool_result blocks immediately after`. The fix is one result per `tool_use`, placed before any text.

Text after the results is accepted for a client-only turn, but Anthropic's [stop-reason guide](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons) lists "Never add text blocks immediately after tool results" as a best practice: it teaches Claude to expect user input after every tool use, and it is a common cause of empty `end_turn` responses. Programmatic tool calls and mixed server and client turns forbid the text entirely.

Two more continuation rules:

- **Thinking blocks.** When thinking is on, send the whole assistant message back unchanged and then append your `tool_result`. The whole tool-use loop counts as one assistant turn, and within it passing thinking blocks back is required; a modified thinking block gets a 400.
- **Truncated calls.** If a response hits `max_tokens` with an incomplete `tool_use` block, retry the request with a higher `max_tokens` to get the full call.

### Reporting a failure with is_error

For an execution error such as a network failure, put the error message in `content` and set `is_error: true`. Claude folds it into its reply, and can retry with corrected input, ask the user for clarification, or explain the limitation.

```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
      "content": "ConnectionError: the weather service API is not available (HTTP 500)",
      "is_error": true
    }
  ]
}
```

Write the message for Claude, not for a log file. The [Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls) page says: "Instead of generic errors like `"failed"`, include what went wrong and what Claude should try next (for example, `"Rate limit exceeded. Retry after 60 seconds."`)." Anthropic's [tool-design post](https://www.anthropic.com/engineering/writing-tools-for-agents) makes the same point: "you can prompt-engineer your error responses to clearly communicate specific and actionable improvements, rather than opaque error codes or tracebacks."

When a call is invalid, for example missing a required parameter, you can answer it with an `is_error` result such as `Error: Missing required 'location' parameter`; Claude retries 2 to 3 times with corrections before apologizing to the user. During development, an invalid call usually means the description lacked information, so the docs' best bet is a more detailed `description`. To eliminate invalid calls, the docs point to `strict: true`, which prevents missing parameters and type mismatches; the caveats on that guarantee are in [Defining a tool](#defining-a-tool).

Who sets the flag depends on the layer, and the spelling changes with it:

| Layer | Field | Who sets it |
|---|---|---|
| Claude API `tool_result` | `is_error` | Your loop |
| Tool Runner (SDK) | `is_error` | The runner, when your tool throws; Claude gets the error message, not the full stack trace |
| Agent SDK custom tool | `isError` (TypeScript), `"is_error"` (Python) | Your handler, when it catches the failure and composes the message; an uncaught exception still becomes an error result carrying the raw message, and the agent loop continues |
| MCP tool result | `isError` | The MCP server; see [MCP errors and structured results](#mcp-errors-and-structured-results) |
| Server tools (`web_search` and others) | Not your job | Claude handles server tool errors itself; see [How a server-tool turn comes back](#how-a-server-tool-turn-comes-back) |

**Decide.** Catch the error yourself whenever the raw exception text would not tell Claude what to do next; the Agent SDK docs recommend exactly that, and the example under [Structured error metadata: the exam guide's convention](#structured-error-metadata-the-exam-guides-convention) shows how.

### Error categories the exam expects

CCAR-F 2.2 asks you to return errors the agent can act on. The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) distinguishes four kinds of error (2.2-K2), wants retryable and non-retryable errors told apart so the agent does not waste retries (2.2-K4), and rejects uniform responses such as a generic "Operation failed", which leave the agent unable to choose a recovery (2.2-K3).

| Category (2.2-K2) | Guide's examples | Recovery the sources describe |
|---|---|---|
| Transient | Timeouts, service unavailability | Retry (EX1-STEP3); a subagent recovers locally and escalates only what it cannot resolve, with partial results and what it tried (2.2-S3) |
| Validation | Invalid input | Correct the input and retry: the MCP spec reports input validation errors as tool execution errors so the model can self-correct |
| Business | Policy violations | Do not retry: flag `retriable: false` and give a customer-friendly explanation the agent can pass on (2.2-S2, EX1-STEP3) |
| Permission | None given | Listed in the guide's `errorCategory` values; the guide sets no separate recovery rule for it |

The metadata the guide asks for in 2.2-S1 (`errorCategory`, `isRetryable` and a human-readable description) is its own convention, not a field of the Claude API or the MCP spec, and it goes inside the error content. The guide is also inconsistent about how many categories there are and how the retry flag is spelled. Both points, with worked examples, are in [Structured error metadata: the exam guide's convention](#structured-error-metadata-the-exam-guides-convention).

### Access failure or valid empty result

A query that ran and matched nothing is a success; a query that could not run is a failure. CCAR-F names the difference in three places (2.2-S4, 5.3-K2, 5.3-S2): an access failure needs a retry decision, while a valid empty result is a successful query with no matches, and reporting them differently lets the coordinator decide what to do.

| What happened | Return |
|---|---|
| The query ran and found nothing | A normal result that says so. The server-side web search tool, for example, returns an empty `content` list, not an error |
| The query could not run (timeout, outage, access denied) | `is_error: true`, with what failed and whether a retry makes sense |

For your own search tools, Anthropic's [search results guide](https://platform.claude.com/docs/en/build-with-claude/search-results) says that when a search fails or returns nothing, you should return a plain text block describing the outcome, such as `No results found.`, instead of raising an error. That advice covers both cases and is about the content format, so the text itself must say which case happened; for an execution failure the Messages API still expects `is_error: true` (our reading of the two pages together).

MCP writes the same principle into its resources spec; see [A failure is not an empty answer](#a-failure-is-not-an-empty-answer).

**Traps.** CCAR-F sample question 8 has a web search subagent that times out. The correct option (A) returns structured error context to the coordinator: the failure type, the attempted query, any partial results and potential alternative approaches. Catching the timeout and returning an empty result marked as successful (option C) is wrong; the guide's rationale: "Option C suppresses the error by marking failure as success, which prevents any recovery and risks incomplete research outputs." ([CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)) Retrying and then returning only a generic "search unavailable" status (option B) hides the context the coordinator needs, and terminating the whole workflow (option D) is the opposite anti-pattern, which 5.3-K4 also names. How errors travel between agents is taught in [Error propagation in multi-agent systems](evaluation-and-reliability.md#error-propagation-in-multi-agent-systems).

### Results carry untrusted content

Content that tools return (files, web pages, search results) is untrusted data and can carry an indirect prompt injection. Anthropic's guidance:

- Deliver third-party content inside `tool_result` blocks, never in `system` prompts or plain user `text` blocks.
- Tell Claude in the system prompt that content returned from tools, documents or searches is untrusted data and must never override the system prompt or the user's original request.
- Do not put your own instructions in tool results. The [jailbreak mitigation guide](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks) explains: "Because Claude treats tool-result content as untrusted data, instructions you place there may be ignored or flagged as a potential injection." Send them in a user turn that follows the `tool_result` block or, on models that support it (as of September 2026, not Claude Sonnet 5), in a mid-conversation system message.
- JSON-encode untrusted strings inside the result, so escaping gives unambiguous delimiters between the payload and the surrounding structure.
- Screen tool output with a small classifier call, for example Claude Haiku 4.5 with structured outputs, before returning it.

The defenses around this (least privilege, guardrail layers) are in [Prompt injection](security-and-governance.md#prompt-injection).

### Keep results small

The API does not truncate tool results; an oversized request is rejected, so truncate large outputs in your application. Anthropic's [tool-design post](https://www.anthropic.com/engineering/writing-tools-for-agents) says Claude Code restricts tool responses to 25,000 tokens by default, and the current Claude Code docs give the same 25,000-token default maximum for MCP tool output (the controls are in [Output size, timeouts and reconnection](#output-size-timeouts-and-reconnection)). Results also pile up: CCAR-F 5.1-K3 describes order lookups returning 40+ fields when only 5 are relevant, and 5.1-S3 asks you to trim verbose outputs to the relevant fields before they accumulate in context. Designing lean responses is covered in [Shape tool responses](#shape-tool-responses); managing what is already in a long conversation is covered in [Preserving critical information in long conversations](context-engineering.md#preserving-critical-information-in-long-conversations).

## Anthropic-defined tools

*Tested in: CCDV-F D8.1 Tool Implementation (client-side vs. server-side tools), D2.3 Claude API Mechanics · CCAR-F 4.5-K3, APPX-TECH-6; computer use is out of scope for CCAR-F (APPX-OUTSCOPE-8)*

Anthropic provides two kinds of ready-made tools. Anthropic-schema client tools have a published schema, but your application executes them. Server tools run on Anthropic's infrastructure, and you never build a `tool_result` for them. Most tools of both kinds carry a `_YYYYMMDD` suffix in their `type` string, and older versions stay available so existing integrations keep working.

### The catalog (as of September 2026)

**Server tools**

| Tool | `type` values | `name` | Beta header | Key facts |
|---|---|---|---|---|
| Web search | `web_search_20260318`, `web_search_20260209`, `web_search_20250305` | `web_search` | None | &#36;10 per 1,000 searches plus token costs; errored searches are not billed |
| Web fetch | `web_fetch_20260318`, `web_fetch_20260309`, `web_fetch_20260209`, `web_fetch_20250910` | `web_fetch` | None | No additional cost beyond tokens |
| Code execution | `code_execution_20260521`, `code_execution_20260120`, `code_execution_20250825` | `code_execution` (fixed) | None | Sandbox with no internet access |
| Tool search | `tool_search_tool_regex_20251119`, `tool_search_tool_bm25_20251119` (undated aliases accepted) | `tool_search_tool_regex`, `tool_search_tool_bm25` | None | Loads deferred tools on demand; see [Progressive discovery with tool search](#progressive-discovery-with-tool-search) |
| Advisor | `advisor_20260301` | `advisor` | `advisor-tool-2026-03-01` | A faster executor model consults a higher-intelligence advisor model mid-generation |
| MCP connector | `mcp_toolset` (not date-versioned) | `mcp_server_name` points to an entry in `mcp_servers` | `mcp-client-2025-11-20` | Remote MCP servers from the Messages API; see [MCP in the API and the Agent SDK](#mcp-in-the-api-and-the-agent-sdk) |

**Anthropic-schema client tools** (your application executes every call)

| Tool | `type` values | `name` | Beta header | Commands or inputs |
|---|---|---|---|---|
| Bash | `bash_20250124` | `bash` | None | `command`; `restart: true` restarts the session |
| Text editor | `text_editor_20250728` (Claude 4 and later), `text_editor_20250124` (earlier models) | `str_replace_based_edit_tool` for the Claude 4 versions (the name changed with `text_editor_20250429`) | None | `view`, `str_replace`, `create`, `insert` (`text_editor_20250124` also has `undo_edit`) |
| Memory | `memory_20250818` | `memory` | None | `view`, `create`, `str_replace`, `insert`, `delete`, `rename` under `/memories` |
| Computer use | `computer_toolset_20260801`; beta `computer_20251124`, `computer_20250124` | None for the toolset | None for the toolset; `computer-use-2025-11-24` and `computer-use-2025-01-24` for the older types | 17 member tools such as `screenshot`, `left_click`, `type`, `zoom` |
| Browser use | `browser_toolset_20260801` | None | None | 27 member tools by default, plus four optional ones off by default |

Declaring them is one short entry each:

```json
{"type": "web_search_20260318", "name": "web_search", "response_inclusion": "excluded"}
{"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 10, "citations": {"enabled": true}, "max_content_tokens": 100000}
{"type": "code_execution_20260120", "name": "code_execution"}
{"type": "bash_20250124", "name": "bash"}
{"type": "text_editor_20250728", "name": "str_replace_based_edit_tool", "max_characters": 10000}
{"type": "memory_20250818", "name": "memory"}
{"type": "computer_toolset_20260801", "configs": {"zoom": {"enabled": false}}, "cache_control": {"type": "ephemeral"}}
{"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}
```

With `web_search_20260209`, `web_fetch_20260209` or later, you do not need to declare code execution: the API provisions it for dynamic filtering. If you do declare it, use `code_execution_20260120` or later, because the API rejects older code execution versions alongside those web tool versions. Those web tool versions also change the `allowed_callers` default; see [Programmatic tool calling](#programmatic-tool-calling).

**Why prefer an Anthropic-schema tool.** When you need memory, shell commands, file editing, or desktop or browser control, prefer the Anthropic-schema tool over defining your own equivalent. Its schema is trained in: Claude has been optimized on thousands of successful trajectories that use those exact signatures, so it calls them more reliably and recovers from errors more gracefully than it would with a custom tool that does the same thing.

### How a server-tool turn comes back

A server tool call appears as a `server_tool_use` block whose `id` starts with `srvtoolu_`, followed in the same assistant turn by its result block (for example `web_search_tool_result`), paired by `tool_use_id`. You do not answer it with a `tool_result`. Claude handles server tool errors itself; for web search, web fetch and tool search the API still returns HTTP 200 with the error inside the result block. One exception to watch: if an administrator has disabled web search for the organization in the Console, the request fails with a 400 `invalid_request_error` instead of an in-result error code.

On a long-running turn the server-side loop can pause and return `stop_reason: "pause_turn"`. Pass the paused response back as-is in a follow-up request, include the same tools (a paused turn can end on a server tool call that has not run yet), and cap the number of continuations like any retry loop, because a continued turn can pause again. Any agent loop that uses server tools should handle `pause_turn`.

```python
def handle_server_tool_conversation(client, user_query, tools, max_continuations=5):
    messages = [{"role": "user", "content": user_query}]

    for _ in range(max_continuations):
        response = client.messages.create(
            model="claude-opus-5-5", max_tokens=4096, messages=messages, tools=tools
        )

        if response.stop_reason != "pause_turn":
            # Claude finished processing - return the final response
            return response

        # pause_turn: replace the full message list to maintain alternating roles
        messages = [
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": response.content},
        ]

    # Reached max continuations - return the last response
    return response
```

Do not confuse the two stops. A `pause_turn` response never leaves a client `tool_use` waiting; you continue it by re-sending the content. A `tool_use` stop is continued by sending `tool_result` blocks, including the mixed case where a server tool is still pending (see [Parallel tool calls](#parallel-tool-calls)). In both cases the API runs the pending server tool at the start of the next request.

**Batches.** All server tools, including MCP connectors, the advisor and tool search, work in Message Batches requests, and the batch loop runs more iterations per turn than a synchronous request before it returns `pause_turn`. A paused batch result is continued by a follow-up request, batch or synchronous.

!!! warning "Exam guide vs current docs"

    The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says in 4.5-K3 that the batch API "does not support multi-turn tool calling within a single request (cannot execute tools mid-request and return results)". The docs (as of September 2026) say the batch worker runs the same server-side agentic loop as the synchronous API, so server tools do run several steps inside one batched request. The two fit if you read the guide as being about client tools: every client tool call is a round trip through your application, which a batch request cannot make mid-request. Expect the guide's statement on the exam. Batch design is in [Message Batches](claude-api.md#message-batches).

### Server tool details worth knowing

| Topic | Fact |
|---|---|
| Web search versions | `_20250305` is basic, `_20260209` adds dynamic filtering (Claude writes and runs code to filter results before they reach the context window; Claude 4.6 and later models), `_20260318` adds `response_inclusion` |
| Web search limits | `max_uses` is the hard cap on searches per request; going over produces the error code `max_uses_exceeded` |
| Web search results | Citations are always on; pass `encrypted_content` back unchanged in later turns or the request fails with a 400 |
| Web fetch reach | Fetches only URLs already in the conversation (user messages, client tool results, earlier search or fetch results), not URLs that appear only in Claude's output or the system prompt |
| Web fetch options | Citations off by default; `max_content_tokens` truncates text (not PDFs); `use_cache` (default `true`) needs `web_fetch_20260309` or later; no JavaScript-rendered pages |
| Domain filters | Use `allowed_domains` or `blocked_domains`, not both; no `http://` or `https://` scheme; subdomains are included; wildcards only in the path, and web fetch matches on the domain only, so an entry with a path never matches a fetch URL; if the organization has an allowed list in the Claude Console, request-level `allowed_domains` must be a subset of it |
| Code execution runtime | Python 3.11, 1 CPU, 5 GiB RAM, internet disabled, so no package installs at runtime; `code_execution_20250825` has Bash and file operations, `code_execution_20260120` adds REPL state persistence and programmatic tool calling, `code_execution_20260521` is the same runtime but tells Claude about the 90-second wall-clock limit on each Python cell in programmatic tool calling |
| Code execution cost | Free when the request also includes `web_search_20260209`, `web_fetch_20260209` or later; otherwise billed by execution time with a 5-minute minimum, 1,550 free hours per organization per month, then &#36;0.05 per hour per container |
| Containers | Reuse one by passing the earlier response's container `id`; an idle container is checkpointed after about 5 minutes, and a request with its ID restores it until the container expires 30 days after creation |
| Zero data retention | The basic `web_search_20250305` and `web_fetch_20250910` are ZDR-eligible; the `_20260209` and later versions are not by default, unless you set `"allowed_callers": ["direct"]`, which turns off dynamic filtering |
| Cloud platforms | Web search is not on Amazon Bedrock, and Google Cloud has only the basic version; web fetch and code execution are on neither |

Two security notes. Enabling web fetch where Claude reads untrusted input alongside sensitive data creates an exfiltration risk; mitigations include disabling the tool, capping `max_uses`, and restricting `allowed_domains`. And if you give Claude both server-side code execution and your own client-side bash tool, Claude can confuse the two environments, so make clear in the system prompt that state does not carry over between them.

### Client tools you execute: your security job

Anthropic publishes the schema, but the blast radius is yours:

- **Bash.** Validate commands with an allowlist, not a blocklist, and treat that check as a tripwire: the real control is running the session isolated, in a container or VM, as a least-privileged user, treating every command as untrusted input. The session cannot run interactive programs such as `vim`, `less` or password prompts.
- **Text editor.** Make each `str_replace` match exactly one location, validate paths against traversal, and keep backups. `text_editor_20250429` removed `undo_edit`.
- **Memory.** Validate every path in every command so a request like `/memories/../../secrets.env` cannot escape the directory. The API adds a memory protocol to the system prompt when the tool is present, and the tool works on all Claude 4 and later models.
- **Output size.** Truncate large command output yourself; see [Keep results small](#keep-results-small).

**Computer use and browser use.** The current toolsets are single `tools` entries with no `name`; each member call carries a `toolset_name`, which your `tool_result` must echo. Claude Opus 5.5 on the Claude API and Google Cloud accepts only `computer_toolset_20260801` and rejects `computer_20251124`. The toolset entries reject `strict`, `input_examples`, entry-level `defer_loading`, code execution callers, the legacy fine-grained streaming header, and a `tool_choice` of type `tool` that names the toolset or a member. Declaring the computer use toolset with its default members adds about 4,500 input tokens. Anthropic's precautions: a dedicated minimal-privilege VM or container, no sensitive data, domain allowlists, and human confirmation for consequential actions; Anthropic also runs classifiers over what the tools return, such as screenshots, to flag potential prompt injections. CCAR-F lists computer use as out of scope.

!!! note "Older write-ups"

    Many write-ups describe computer use as `computer_20250124` or `computer_20251124` with a beta header and an `action` input field. The current `computer_toolset_20260801` needs no beta header, has no `name`, and exposes member tools instead of `input.action`.

    Anthropic's November 24, 2025 advanced-tool-use post, which introduced tool search and programmatic tool calling as beta features, enables them with the beta header `advanced-tool-use-2025-11-20` and shows `allowed_callers: ["code_execution_20250825"]`. The current docs say tool search needs no beta header and that programmatic tool calling requires `code_execution_20260120` or later.

### Programmatic tool calling

Programmatic tool calling (PTC) lets Claude write code that calls your tools inside a code execution container, instead of a round trip through the model for every call. It needs `code_execution_20260120` or later; you opt a tool in with `allowed_callers: ["code_execution_20260120"]`, choosing that or `["direct"]` per tool rather than both. `"code_execution_20260521"` is also accepted in `allowed_callers` and is interchangeable with `"code_execution_20260120"`; response blocks always tag the caller as `code_execution_20260120`. For your own tools and the earlier web tool versions, omitting `allowed_callers` means `["direct"]`. The `_20260209` and later web search and web fetch tools instead default to the code execution caller only. On models that do not support programmatic tool calling, such as Claude Haiku 4.5, they need `"allowed_callers": ["direct"]`, or the request returns a 400.

```json
"tools": [
  {"type": "code_execution_20260120", "name": "code_execution"},
  {
    "name": "query_database",
    "description": "Execute a SQL query against the sales database. Returns a list of rows as JSON objects.",
    "input_schema": {
      "type": "object",
      "properties": {"sql": {"type": "string", "description": "SQL query to execute"}},
      "required": ["sql"]
    },
    "allowed_callers": ["code_execution_20260120"]
  }
]
```

When Claude's code calls the tool, execution pauses and the API returns an ordinary `tool_use` block whose `caller` field says `{"type": "code_execution_20260120", "tool_id": "srvtoolu_..."}` (a direct call says `{"type": "direct"}`). Your result resumes the code, and the intermediate results never enter Claude's context. Tools are exposed to the code as async Python functions, so Claude can fan out with `asyncio.gather`.

| Rule | Detail |
|---|---|
| Continuation | Include the paused response's container `id` (required while a call is pending), the same `tools`, and a user message of only `tool_result` blocks whose content is a string or text blocks |
| Timeouts | A pending call times out after about 4 minutes with a `TimeoutError` inside the code; idle containers are reclaimed after about 5 minutes |
| Incompatible with | `strict: true` tools; `disable_parallel_tool_use: true`; and you cannot force programmatic calling of a specific tool through `tool_choice` |
| Cannot be called this way | MCP connector tools, the computer and browser toolsets, tools whose `input_schema` has a recursive `$ref` |
| Billing | Results of programmatic calls do not count toward token usage; only the final code output and Claude's response do |
| Models | Claude Haiku 4.5 accepts `code_execution_20260120` but does not support PTC |

**Decide.** PTC pays off for fan-out across many items, large results that code can filter, and agentic search. It does not help strictly sequential calls where each depends on Claude's reasoning, a few small calls, or calls that need user feedback in between. Anthropic's measurements: roughly 38% fewer billed input tokens on a 75-tool project-management agent benchmark with no change in accuracy; typical token savings of 20% to 40% across production traffic for requests with 10 to 49 tool definitions; but on τ²-bench, where each turn makes one or two sequential calls, unchanged scores at roughly 8% more cost. Because Claude parses the results in code, document each tool's output format (JSON structure and field types) in its description.

`allowed_callers` is not a security boundary: be ready to handle a direct `tool_use` for any tool you define.

### The think tool: prefer extended thinking

An Anthropic engineering post introduced a "think" tool that obtains no new information and only appends a thought to a log. The post now carries a December 15, 2025 update recommending extended thinking instead of a dedicated think tool in most cases, so reach for extended thinking first and do not present the think tool as current best practice.

## Designing a tool set

*Tested in: CCAR-F 2.1-S3, 2.3-K1, 2.3-K2, 2.3-K3, 2.3-S1, 2.3-S2, 2.3-S3, APPX-INSCOPE-4 · CCDV-F D8.1 Tool Implementation (tool set construction, approval patterns), D7.2 Guardrails and Safe Deployment · CCAR-P 3.1, 3.8, 5.1*

Anthropic calls tools "a new kind of software which reflects a contract between deterministic systems and non-deterministic agents" ([Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)). Its [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) post adds a rule of thumb: plan to invest as much effort in the agent-computer interface (ACI) as teams put into human-computer interfaces. Designing a tool set means making the ACI decisions: how many tools an agent sees, how they overlap, what they are called, what they return and what they are allowed to do.

### Fewer, higher-level tools

More tools do not always mean better results. A common error Anthropic reports is tools that merely wrap existing software functions or API endpoints, whether or not those suit an agent. Its recommendation is "building a few thoughtful tools targeting specific high-impact workflows", then scaling up from there. Anthropic's own before-and-after examples ([Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents), plus the define-tools docs for the last row):

| Instead of | Build | Why |
|---|---|---|
| `list_contacts` | `search_contacts` or `message_contact` | Returning every contact wastes the agent's limited context |
| `list_users`, `list_events`, `create_event` | `schedule_event`, which finds availability and schedules | One tool per real task |
| `read_logs` | `search_logs`, which returns only relevant lines with some surrounding context | Less noise in context |
| `get_customer_by_id`, `list_transactions`, `list_notes` | `get_customer_context`, which compiles recent, relevant customer information at once | One call instead of three |
| `create_pr`, `review_pr`, `merge_pr` | One tool with an `action` parameter | Fewer, more capable tools reduce selection ambiguity |

Anthropic's post on [building agents that reach production systems with MCP](https://claude.com/blog/building-agents-that-reach-production-systems-with-mcp) sums it up: "Fewer, well-described tools consistently outperform exhaustive API mirrors." For very large APIs it points to a thin surface that accepts code; its example is Cloudflare's MCP server, where two tools (search and execute) cover about 2,500 endpoints in roughly 1K tokens.

### Split or consolidate

The advice can look contradictory. CCAR-F 2.1-S3 splits a generic `analyze_document` into `extract_data_points`, `summarize_content` and `verify_claim_against_source`, while the docs group `create_pr`, `review_pr` and `merge_pr` into one tool. Both rest on the same principle: every tool needs a clear, distinct purpose, because overlapping or vague tools confuse agents.

**Decide.** Our reconciliation of the guide and the docs (neither rule is absolute):

- **Consolidate** when several low-level calls or CRUD variants serve one purpose: fewer, higher-level tools.
- **Split** when one vague tool hides several jobs with different inputs and outputs, or overlaps another tool.
- **Constrain** when a tool is more general than the job: CCAR-F 2.3-S2 replaces a generic `fetch_url` with `load_document`, which validates document URLs. Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) post names the general practice poka-yoke: change the arguments so mistakes are harder. In its SWE-bench agent, changing the tool to always require absolute file paths fixed the model's relative-path errors.

### Namespacing

When tools span services or resources, prefix their names: by service (`asana_search`, `jira_search`) or by resource (`asana_projects_search`, `asana_users_search`). The docs' examples are `github_list_prs` and `slack_send_message`, and they stress it for tool search, where one consistent prefix lets a single search match the whole group. Anthropic found that choosing prefix or suffix namespacing had non-trivial effects on its tool-use evaluations, and that the effects vary by model, so choose by your own evaluations. Claude Code and the Agent SDK namespace MCP tools for you; the pattern is in [Names, permissions and hooks](#names-permissions-and-hooks).

### How many tools per agent

Several numbers circulate. They are different claims, not one threshold:

| Source | Number | Claim |
|---|---|---|
| CCAR-F 2.3-K1 | 18 tools instead of 4 to 5 | Too many tools degrades selection reliability by increasing decision complexity |
| Tool search docs | 10 or more tools | One of five signals to use tool search (all five are under [Progressive discovery with tool search](#progressive-discovery-with-tool-search)) |
| Managing tool context docs | Roughly 20 tools | Add tool search once the tool set grows past this, or once baseline context use becomes noticeable |
| Anthropic multi-agent blog | Often 20+ tools | An agent struggles to pick the right one; a signal for specialization |
| Anthropic multi-agent blog | 15 to 20+ tools | Consider the tool search tool before adopting a multi-agent architecture |
| Tool search docs | 30 to 50 tools | Claude's ability to pick the right tool degrades once you exceed this range |

For the exam, carry the guide's lesson, which is about selection reliability rather than a numeric cutoff: each agent should see the few tools its role needs.

### Scope tools to each agent's role

The CCAR-F guide builds 2.3 around role-scoped access:

- Agents with tools outside their specialization tend to misuse them, as when a synthesis agent attempts web searches (2.3-K2).
- Give each agent only the tools its role needs, with limited cross-role tools for specific high-frequency needs (2.3-K3, 2.3-S1).
- For example, give the synthesis agent a scoped `verify_fact` tool, while complex cases still route through the coordinator (2.3-S3).

CCAR-F sample question 9 tests exactly this. A synthesis agent hands every verification back to the coordinator, which adds 2 to 3 round trips per task and 40% latency, although 85% of the verifications are simple fact-checks. The keyed answer (A) gives it a scoped `verify_fact` tool for simple lookups while complex verifications still go through the coordinator; the rationale: "Option A applies the principle of least privilege by giving the synthesis agent only what it needs for the 85% common case (simple fact verification) while preserving the existing coordination pattern for complex cases." Option C, access to all web search tools, is the over-provisioning trap: "Option C over-provisions the synthesis agent, violating separation of concerns." ([CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf))

In the Agent SDK, the scope lives in each subagent's definition:

- `AgentDefinition.tools` lists the allowed tools. If you omit it, the subagent inherits every tool available to subagents, so (our inference) leaving it out over-provisions by default.
- A tool left out of the list is simply absent from the subagent's session: no permission prompt, no error.
- The docs' common combinations: read-only analysis gets `Read`, `Grep`, `Glob`; test execution gets `Bash`, `Read`, `Grep`; code modification gets `Read`, `Edit`, `Write`, `Grep`, `Glob`.
- A bare-name deny rule such as `"Bash"` in `disallowed_tools` removes the tool definition so Claude cannot even attempt it; a scoped rule such as `"Bash(rm *)"` only denies matching calls. `allowed_tools` does not constrain `bypassPermissions` mode, so block tools there with `disallowed_tools`. (These are the Python option names; TypeScript uses `allowedTools` and `disallowedTools`.)

Mechanics are in [Subagents in the SDK](agents-and-agent-sdk.md#subagents-in-the-sdk) and [Permissions and enforcement](agents-and-agent-sdk.md#permissions-and-enforcement). Specialization that improves tool selection or task focus is also one of the three situations where Anthropic finds multiple agents consistently outperform a single agent, alongside context pollution and work that can run in parallel ([Building multi-agent systems](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)); the same post's advice to consider tool search before splitting is in the table above.

### Least privilege: remove before you guard

CCAR-P sample question 1 describes a support agent that can read tickets, draft replies, issue refunds and delete user accounts, when staff only ever need the first two. The keyed answer removes the refund and delete tools. The rationale from the [CCAR-P guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf): "Least privilege means removing capabilities the role does not require, eliminating the attack surface rather than monitoring or guarding it. Logging (A) and confirmations (C) are detective/compensating controls, not removal of unnecessary privilege; model size (D) is unrelated to authorization scope."

Anthropic's injection guidance gives the same reason from the security side: apply least privilege so a successful injection does minimal damage, by withholding secrets Claude does not need, sandboxing tools, and scoping permissions as narrowly as possible.

**Decide.** Our ordering, derived from the CCAR-P rationale and the guide's constrained-tool example. Start at the top; the lower rungs are for capabilities the role genuinely keeps, and they add to the upper rungs rather than replace them:

1. Remove tools the role never needs.
2. Constrain the tools it keeps: narrower inputs, validated arguments, typed purpose-built tools.
3. Gate hard-to-reverse actions behind a confirmation or a hook.
4. Log for audit.

The rationale calls logging and confirmations "detective/compensating controls", so an answer that keeps an unneeded capability and adds either one is the trap. The wider treatment is in [Least privilege for tools and agents](security-and-governance.md#least-privilege-for-tools-and-agents).

### Approval patterns for consequential actions

When an agent keeps a risky capability, the harness decides when a person must approve it. Anthropic's [agent harness design post](https://claude.com/blog/harnessing-claudes-intelligence) covers this under its third pattern, "Set boundaries carefully in your harness design", in the subsection on declarative tools. Three of its points bear on approvals:

- **Promote actions to dedicated tools.** A generic bash tool hands the harness only a command string; a dedicated, typed tool gives it an action-specific hook it can intercept, gate, render or audit. Actions that require a security boundary are natural candidates.
- **Gate by reversibility.** Hard-to-reverse actions, such as external API calls, can be gated by user confirmation.
- **Let a tool ask.** A tool can render as a modal that shows the user a question, offers options, or blocks the agent loop until the user responds.

The MCP spec sets the same human-in-the-loop expectation for clients; its list is under [MCP primitives](#mcp-primitives).

Where the gate lives:

| Stack | Approval mechanism |
|---|---|
| Messages API | Your manual loop checks each `tool_use` before executing it; the docs send human-in-the-loop approval to the manual loop rather than the Tool Runner |
| Agent SDK | The `canUseTool` callback (Python: `can_use_tool`) pauses execution until it returns allow (optionally with modified input) or deny with a message Claude sees; a denied call comes back to Claude as a rejection message in the tool result, and Claude typically tries a different approach or reports that it could not proceed |
| Agent SDK, every call | The callback never fires for auto-approved tools, so logic that must run on every call belongs in a `PreToolUse` hook |
| Headless, locked down | Pair `allowedTools` with `permissionMode: "dontAsk"` (TypeScript names), which denies instead of prompting |
| Computer use | Confirm consequential actions before each block runs, because one batch can finish a multistep action in a single turn |

The SDK checks permissions in a fixed order: hooks, deny rules, ask rules, permission mode, allow rules, then the `canUseTool` callback. Details: [Permissions and enforcement](agents-and-agent-sdk.md#permissions-and-enforcement) and [Hooks in the SDK](agents-and-agent-sdk.md#hooks-in-the-sdk).

### Shape tool responses

What a tool returns is part of its design:

- **High signal, meaningful names.** Anthropic advises prioritizing contextual relevance over flexibility and avoiding low-level technical identifiers such as `uuid`, `256px_image_url` and `mime_type`; fields like `name`, `image_url` and `file_type` are more likely to inform the agent's next action. Resolving arbitrary alphanumeric UUIDs to meaningful language, or even a 0-indexed ID scheme, significantly improved Claude's retrieval precision by reducing hallucinations. The define-tools docs still want stable, semantic identifiers (slugs or UUIDs) rather than opaque internal references, so (our reconciliation) keep the IDs Claude needs for follow-up calls and make the rest readable.
- **Let the agent choose verbosity.** A `response_format` enum with `"concise"` and `"detailed"` lets the agent control detail; in Anthropic's Slack example, concise responses used about a third of the tokens (72 versus 206).
- **Bound the size.** Use some combination of pagination, range selection, filtering and truncation with sensible default parameter values. If you truncate, steer the agent with helpful instructions, for example to make several small targeted searches instead of one broad one. The default output limits are in [Keep results small](#keep-results-small) and [Output size, timeouts and reconnection](#output-size-timeouts-and-reconnection).
- **Pick the format by evaluation.** XML, JSON or Markdown can each score differently; there is no one-size-fits-all format.
- **Label risk.** For MCP servers, tool annotations disclose which tools need open-world access or make destructive changes; see [MCP primitives](#mcp-primitives).

### Progressive discovery with tool search

CCAR-P asks you to "Evaluate progressive discovery vs. monolithic context strategy" (3.8) and to evaluate tool and agent configuration for capability bloat (3.1) ([CCAR-P guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). In our mapping of the guide's terms onto the docs, monolithic means every tool definition is loaded into Claude's context window up front, and progressive discovery means tool search. A typical five-server setup (GitHub, Slack, Sentry, Grafana, Splunk) can take about 55K tokens of definitions, and Anthropic has seen definitions consume 134K tokens before optimization. With tool search, Claude searches your catalog and loads only the tools it needs; the current docs say the search covers tool names, descriptions, argument names and argument descriptions (Anthropic's November 2025 [advanced tool use post](https://www.anthropic.com/engineering/advanced-tool-use) mentioned names and descriptions only). Tool search typically cuts that definition load by over 85 percent, loading only the 3 to 5 tools Claude needs for a given request.

A deferred tool alongside the regex tool search tool (definition as in the [tool search docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)):

```json
{
  "tools": [
    {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"},
    {
      "name": "get_weather",
      "description": "Get current weather for a location",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": { "type": "string" },
          "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
        },
        "required": ["location"]
      },
      "defer_loading": true
    }
  ]
}
```

The rules, as of September 2026:

| Rule | Detail |
|---|---|
| What you send | Every tool's full definition, on every request; `defer_loading` controls what enters the context window, not what you send |
| Non-deferred tools | At least one tool must stay non-deferred, and the tool search tool itself must never be deferred; an all-deferred request returns a 400. Keep your 3 to 5 most-used tools non-deferred |
| Variants | Regex (`tool_search_tool_regex_20251119`): Claude writes Python `re.search()` patterns, matched case-insensitively, up to 200 characters. BM25 (`tool_search_tool_bm25_20251119`): natural-language queries, up to 500 characters |
| Results | Matches come back as `tool_reference` blocks, up to 5 by default (Claude can set a `limit` from 1 to 10,000 in its search input), and the API expands them into full definitions; never send a `tool_result` for the search's `srvtoolu_` id |
| Scale | Up to 10,000 deferred tools per request |
| Caching | Deferred tools stay out of the system-prompt prefix and are expanded inline, so tool search does not break the cache; a deferred tool cannot also carry `cache_control` (400) |
| Custom search | Your own tool (embeddings, for example) can return `tool_reference` blocks in a normal `tool_result` |
| MCP connector | Deferral is set on the `mcp_toolset` entry, not on individual tools; see [Many MCP tools: deferral and progressive discovery](#many-mcp-tools-deferral-and-progressive-discovery) |
| Billing and models | Not metered as a separate server tool (loaded definitions are ordinary input tokens); not supported on Claude Opus 4.1 and earlier |
| Defaults elsewhere | On by default in the Agent SDK (it defers SDK MCP tools) and in Claude Code |

In Anthropic's internal MCP evaluations, enabling tool search raised Claude Opus 4 from 49% to 74% and Claude Opus 4.5 from 79.5% to 88.1%.

**Decide.** The tool search docs list when to use it: 10 or more tools, more than 10K tokens of definitions, selection accuracy dropping as the tool set grows, several aggregated MCP servers (200+ tools), or a library that grows over time. They say standard tool calling (monolithic, in our mapping) fits better when you have fewer than 10 tools, every tool is used in every request, or the definitions are small (under 100 tokens total).

### Context levers by bottleneck

Tool definitions and accumulated results both consume context. Anthropic's guidance matches each lever to a bottleneck and suggests an order for a high-volume agent:

| Bottleneck | Lever | When to add it |
|---|---|---|
| Re-sending the same definitions on every request | Prompt caching: `cache_control` on the last tool caches the whole tools prefix; 5-minute cache writes cost 25% more than base input and pay off after one cache read (1-hour writes cost 2x base input and pay off after two reads) | From day one |
| Definitions crowding the context | Tool search | Past roughly 20 tools, or when baseline context use becomes noticeable |
| Old tool results filling a long conversation | Context editing with `clear_tool_uses_20250919` (beta header `context-management-2025-06-27`): by default it triggers at 100,000 input tokens and keeps the 3 most recent tool uses | Long conversations |
| Large intermediate results in chains of small calls | [Programmatic tool calling](#programmatic-tool-calling) | Repetitive chains of small calls |
| Parameter errors on complex inputs | Tool use examples (`input_examples`; see [Input examples](#input-examples)) | Only where usage is ambiguous |

Tools sit at the front of the cached prefix (tools, then system, then messages), so modifying a tool definition invalidates the entire cache; Anthropic's [agent harness design post](https://claude.com/blog/harnessing-claudes-intelligence) puts it plainly: "Tools sit in the cached prefix. Adding or removing one invalidates it." To add tools without that cost, defer them and let tool search append them inline instead of modifying the head of the `tools` array, or use the beta mid-conversation tool changes (`mid-conversation-tool-changes-2026-07-01`). Declare the full tool set in `tools` up front, marking tools that are not yet needed `defer_loading: true`, then offer or withdraw them with `tool_addition` and `tool_removal` blocks in a `system` message, so the `tools` array never changes. A `tool_addition` that names a tool not declared in `tools` returns a 400; carrying a tool's full definition inside a `tool_addition` block needs a separate beta header, `inline-tools-2026-09-15`. Caching mechanics are in [Prompt caching](claude-api.md#prompt-caching); context editing is in [Compaction, context editing and memory](context-engineering.md#compaction-context-editing-and-memory).

### Evaluate the tool set

Test tools the way the agent will use them: realistic, multi-step tasks, measured on accuracy plus runtime per call and per task, number of tool calls, total tokens and tool errors. Read the metrics as design feedback: many redundant calls suggest adjusting pagination or token limits, and many invalid-parameter errors suggest clearer descriptions or better examples. Evaluation design is in [Evaluating agents](evaluation-and-reliability.md#evaluating-agents).

## MCP architecture

*Tested in: CCAR-F 2.4-K3, APPX-TECH-2, APPX-INSCOPE-6 · CCDV-F D8.2 MCP Server Development (client vs. server) · CCAR-P 3.7 · CCAO-F D5.2 (connectors)*

The Model Context Protocol is an open standard for connecting AI applications to external data and actions, meant to replace fragmented, per-source integrations with a single protocol. Anthropic open-sourced it on November 25, 2024, calling it "a new standard for connecting AI assistants to the systems where data lives" ([Anthropic announcement](https://www.anthropic.com/news/model-context-protocol)). The current protocol revision is 2026-07-28 ([MCP versioning](https://modelcontextprotocol.io/docs/2026-07-28/learn/versioning)); the one before it was 2025-11-25 ([2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)).

### Hosts, clients and servers

The [MCP specification](https://modelcontextprotocol.io/specification/2026-07-28) defines three roles that exchange JSON-RPC 2.0 messages. Hosts are "LLM applications that initiate connections", clients are "Connectors within the host application", and servers are "Services that provide context and capabilities".

| Role | Responsibilities in the spec | Examples |
|---|---|---|
| Host | Creates and manages the clients, one for each server; enforces security policies and consent requirements; handles user authorization decisions; manages context aggregation across its clients | Claude Code and Claude Desktop (the MCP architecture overview's examples of hosts), VS Code (its worked example). Claude Code runs its own MCP client runtimes |
| Client | Created by the host; communicates with exactly one server | The client object a host creates for each configured server |
| Server | Exposes resources, tools and prompts; receives only the context it needs | A local filesystem server, a remote Sentry server (the overview's examples) |

An illustrative host with three servers (the transports and fan-out follow the architecture overview's typical cases):

```text
+------------------------------ Host (for example Claude Code) ------------------------------+
|  full conversation history . consent prompts . one merged tool list for the model          |
|                                                                                            |
|     MCP client A                  MCP client B                    MCP client C             |
+--------|-----------------------------|-----------------------------------|-----------------+
         | stdio                       | Streamable HTTP                   | Streamable HTTP
         v                             v                                   v
   filesystem server             GitHub server (remote)             Sentry server (remote)
   (local subprocess)            typically serves many clients      typically serves many clients
```

Five facts from the architecture explain most exam questions:

1. **One client per server.** Each client "Communicates with exactly one server" ([MCP architecture specification](https://modelcontextprotocol.io/specification/2026-07-28/architecture)), and the host creates a separate client for each server it connects to ([MCP architecture overview](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture)).
2. **The host owns the conversation.** The full conversation history stays with the host. Servers receive only the context they need, and by the spec's design principles they should not be able to read the whole conversation or "see into" other servers.
3. **The host merges tools.** The application fetches tools from every connected server and combines them into one registry the model can use. This matches the [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) wording that tools from all configured MCP servers are "discovered at connection time and available simultaneously to the agent" (2.4-K3). Claude Code now defers full definitions by default; how that squares with the guide's wording is in [What Claude sees once servers connect](#what-claude-sees-once-servers-connect).
4. **Fan-out differs by transport.** Local stdio servers typically serve one client; remote Streamable HTTP servers typically serve many.
5. **MCP stops at context exchange.** The protocol does not dictate how the application uses the model or manages the context it provides.

### Two layers: data and transport

MCP separates a data layer (the JSON-RPC protocol: discovery, primitives and notifications) from a transport layer (connection establishment, message framing and authorization). The transport layer abstracts communication details from the data layer, so the same JSON-RPC 2.0 message format is used across all transports; transports are covered in [MCP transports](#mcp-transports).

All MCP messages MUST follow JSON-RPC 2.0 and be UTF-8 encoded. Three message kinds exist:

| Kind | Carries | Rules |
|---|---|---|
| Request | `id`, `method`, `params` | `id` is a string or integer and, unlike base JSON-RPC, MUST NOT be `null`; it MUST NOT match the ID of any other request the same sender has issued and not yet received a response for |
| Response | the request's `id`, plus `result` or `error` | An error carries an integer `code`, a `message`, and optional `data`. In 2026-07-28 every `result` also carries a `resultType` (`"complete"` or `"input_required"`) |
| Notification | `method`, `params`, no `id` | The receiver must not respond |

Each primitive follows the same method pattern: `*/list` for discovery, a retrieval method (`prompts/get`, `resources/read`), and for tools an execution method (`tools/call`). The full method list is in [MCP primitives](#mcp-primitives).

### Two protocol eras

MCP versions are date strings (`YYYY-MM-DD`) marking the last backwards-incompatible change. The [2026-07-28 versioning rules](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning) call the handshake-based revisions (2025-11-25 and earlier) "Legacy", the per-request-metadata revisions (2026-07-28 and later) "Modern", and an implementation that speaks both "Dual-era". The two eras differ in ways that exam items and real configurations both touch:

| Topic | Legacy (2025-11-25 and earlier) | Modern (2026-07-28) |
|---|---|---|
| Connection setup | `initialize` MUST be the first interaction; the client then sends `notifications/initialized` | No handshake: every request carries `io.modelcontextprotocol/protocolVersion` and `io.modelcontextprotocol/clientCapabilities` in `_meta` |
| Capabilities | Negotiated once at initialization; both sides use only what was negotiated | Declared on each request; a server that needs an undeclared client capability returns `MissingRequiredClientCapability` (`-32021`) |
| Server identity and `instructions` | Returned in the `initialize` result | Returned by `server/discover`, which servers MUST implement and clients may call |
| State | Stateful connection | Servers MUST NOT rely on earlier requests on the same connection; cross-request state travels as an explicit identifier |
| HTTP sessions | Optional `MCP-Session-Id` header | Removed |
| Server needs input (sampling, elicitation, roots) | Server sends its own request to the client | Multi Round-Trip Requests: the server returns an `InputRequiredResult` (`resultType: "input_required"`) and the client retries the request with `inputResponses` |
| Version mismatch | Server answers with another version it supports; the client should disconnect if it cannot use it | `UnsupportedProtocolVersion` (`-32022`) lists supported versions and the client retries |

A legacy connection opens with this handshake (trimmed from the [2025-11-25 lifecycle page](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle), which also shows `tasks` capabilities and extra `clientInfo`/`serverInfo` fields such as `title`, `description` and `icons`). Client to server:

```json
{ "jsonrpc": "2.0", "id": 1, "method": "initialize",
  "params": { "protocolVersion": "2025-11-25",
              "capabilities": { "roots": { "listChanged": true }, "sampling": {}, "elicitation": { "form": {}, "url": {} } },
              "clientInfo": { "name": "ExampleClient", "version": "1.0.0" } } }
```

Server to client, then the client's notification:

```json
{ "jsonrpc": "2.0", "id": 1,
  "result": { "protocolVersion": "2025-11-25",
              "capabilities": { "logging": {}, "prompts": { "listChanged": true },
                                "resources": { "subscribe": true, "listChanged": true },
                                "tools": { "listChanged": true } },
              "serverInfo": { "name": "ExampleServer", "version": "1.0.0" },
              "instructions": "Optional instructions for the client" } }
```

```json
{ "jsonrpc": "2.0", "method": "notifications/initialized" }
```

A modern request needs no handshake because it describes itself ([MCP architecture overview](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture)):

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "weather_current",
    "arguments": { "location": "San Francisco", "units": "imperial" },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": { "name": "example-client", "version": "1.0.0" },
      "io.modelcontextprotocol/clientCapabilities": { "elicitation": {} }
    }
  }
}
```

!!! warning "Exam guide vs current docs"

    All four exam guides are marked "Effective July 2026" and none of them names an MCP revision, while the current revision, 2026-07-28, removed the `initialize` handshake and deprecated sampling. Claude Code speaks both eras (as of September 2026): its v2 client runtime, built on MCP TypeScript SDK 2.0, asks HTTP servers whether they support the newer revision but connects to stdio servers the older way unless `MCP_PROTOCOL_NEGOTIATION` is set to `auto`. Learn both. If an item mentions `initialize`, capability negotiation at connection time, or server-initiated sampling, read it as describing the handshake era rather than as an error, and answer in the guide's terms.

### Security principles the host enforces

The spec puts consent and safety on the host, because the protocol cannot enforce them itself:

- Users must explicitly consent to and understand all data access and operations. Hosts must obtain explicit user consent before exposing user data to servers and before invoking any tool.
- Tools represent arbitrary code execution and must be treated with appropriate caution. Descriptions of tool behavior, such as annotations, should be considered untrusted unless obtained from a trusted server.
- MCP "cannot enforce these security principles at the protocol level" ([MCP specification](https://modelcontextprotocol.io/specification/2026-07-28)); implementors SHOULD build consent and authorization flows.
- `clientInfo` and `serverInfo` are self-reported by the sender and not verified by the protocol. They are meant for display, logging and debugging, and implementations SHOULD NOT rely on them for security decisions.

Prompt injection through tool results and malicious servers is covered in [Security, Safety and Governance](security-and-governance.md#prompt-injection).

### Where MCP runs in Claude products

The same protocol appears across Claude's surfaces, with different ways to add a server and different transport limits (as of September 2026):

| Surface | How servers are added | Transports | Key limit |
|---|---|---|---|
| Claude Code | `claude mcp add`, project `.mcp.json`, `~/.claude.json` (local and user scopes) | stdio, HTTP, SSE (deprecated), WebSocket (`type: "ws"`, JSON config only) | Project-scoped servers need approval in interactive sessions; see [MCP in Claude Code](#mcp-in-claude-code) |
| Claude Desktop, local servers | `claude_desktop_config.json` under `mcpServers`, or a `.mcpb` desktop extension | stdio | Local servers are not available in Cowork or claude.ai |
| Claude, Cowork and Claude Desktop custom connectors | Customize > Connectors on Pro and Max; on Team and Enterprise only Owners add them, then each user connects individually | Remote MCP, reached from Anthropic's cloud | The server must be reachable over the public internet from Anthropic's IP ranges; Free users are limited to one custom connector |
| Messages API (MCP connector, beta header `mcp-client-2025-11-20`) | `mcp_servers` plus an `mcp_toolset` entry in `tools` | Remote HTTP: Streamable HTTP or SSE, publicly exposed; MCP tunnels (research preview) reach servers in a private network | Tool calls only; no local stdio servers |
| Claude Agent SDK | `mcpServers` (TypeScript) or `mcp_servers` (Python) option, `.mcp.json`, or in-process SDK servers | stdio, HTTP, SSE, in-process | MCP tools need explicit permission before Claude can call them |
| Claude Managed Agents | Declared by name and URL at agent creation; credentials supplied per session from a vault | Remote; MCP tunnels (research preview) reach servers in a private network | Secrets stay out of agent definitions |

For the CCAO-F objective on connectors (D5.2 names Google Drive and Gmail), the facts that matter are these: connectors inherit each person's permissions from the connected service, and they work across Claude, Claude Desktop, Claude Code and the API through the MCP connector. On Team and Enterprise plans, Owners can restrict connector actions organization-wide by setting each permission category or individual permission to Always allow, Needs approval, or Blocked. The day-to-day workflow is in [Claude Apps for Work](claude-for-work.md#search-research-and-connectors). Choosing MCP versus an API or CLI, or agent-to-agent (CCAR-P 3.7), is in [Built-in tools, custom tools, Skills or MCP](#built-in-tools-custom-tools-skills-or-mcp).

### Decide and avoid

Our decision rules, derived from the spec's roles and security principles:

- If an item asks who obtains the user's consent before a tool runs, choose the host; not the server, because the spec puts consent on the host and the server sees only the context the host sends, and not the protocol, because MCP cannot enforce these principles at the protocol level.
- If an item describes several servers, expect several clients inside one host; not one client multiplexing servers.
- If an item says a server can read the whole conversation or see into another server, it contradicts the spec's design principles: the full conversation history stays with the host, and servers get only the context they need.
- If an item uses `serverInfo` or `clientInfo` to authorize access, reject it: both are self-reported and unverified.

## MCP primitives

*Tested in: CCAR-F 2.4-K4, 2.4-S5, APPX-INSCOPE-5, APPX-TECH-2 · CCDV-F D8.2 MCP Server Development (MCP resources, tools, and prompts) · CCAR-P 3.7*

A server primitive is a kind of thing a server offers (clients offer primitives too; see below). The question that sorts the three server primitives is who decides when each one is used: the model, the application, or the user.

| Primitive | Controlled by | What it is | Methods | Examples in the MCP docs |
|---|---|---|---|---|
| Tools | Model | Functions the model decides to call; they can write to databases, call APIs, modify files | `tools/list`, `tools/call` | Search flights, send messages, create calendar events |
| Resources | Application | Passive, read-only data for context, each identified by a URI | `resources/list`, `resources/templates/list`, `resources/read` | Retrieve documents, access knowledge bases, read calendars |
| Prompts | User | Pre-built instruction templates the user chooses | `prompts/list`, `prompts/get` | Plan a vacation, summarize my meetings, draft an email |

The [MCP architecture overview](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture) gives one server all three: a database server "can expose tools for querying the database, a resource that contains the schema of the database, and a prompt that includes few-shot examples for interacting with the tools." The CCAR-F in-scope list compresses the same rule to "resources for content catalogs, tools for actions" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf), Appendix).

### Tools

A server that offers tools declares the `tools` capability (with `listChanged` if it will announce list changes). Clients discover tools with a paginated `tools/list` and run one with `tools/call`.

| Field | What the spec requires or recommends |
|---|---|
| `name` | SHOULD be 1 to 128 characters, case-sensitive, using only A-Z, a-z, 0-9, underscore, hyphen and dot. SHOULD be unique within a server (uniqueness across servers is not required) |
| `title` | Optional display name. Display precedence is `title`, then `annotations.title`, then `name` |
| `description` | "Human-readable description of functionality" in the spec. The CCAR-F guide (2.1-K1) treats descriptions as the primary mechanism LLMs use for tool selection |
| `inputSchema` | MUST be a valid JSON Schema object, never `null`. For a tool with no parameters the spec recommends `{ "type": "object", "additionalProperties": false }`. With no `$schema`, the dialect is JSON Schema 2020-12 |
| `outputSchema` | Optional. When present, results MUST conform; see [MCP errors and structured results](#mcp-errors-and-structured-results) |
| `annotations` | Optional behavior hints (table below) |
| `icons` | Optional |

Because names are unique only within a server, a client that aggregates servers should apply its own disambiguation, such as prefixing tool names with a server identifier. Claude Code and the Agent SDK name MCP tools `mcp__<server>__<tool>` (plugin-bundled servers get `mcp__plugin_<plugin-name>_<server-name>__<tool-name>`), which is also the string you use in permission rules, `allowedTools` entries and hook matchers. How to write the description itself is covered in [Writing tool descriptions that steer selection](#writing-tool-descriptions-that-steer-selection).

**Annotations** describe behavior to the client. Besides an optional `title`, they carry four hints:

| Hint | Default | Meaning in the [MCP schema](https://modelcontextprotocol.io/specification/2026-07-28/schema) |
|---|---|---|
| `readOnlyHint` | `false` | The tool does not modify its environment |
| `destructiveHint` | `true` | "If true, the tool may perform destructive updates to its environment." Meaningful only when `readOnlyHint` is false |
| `idempotentHint` | `false` | Repeated calls with the same arguments have no additional effect on the environment. Meaningful only when `readOnlyHint` is false |
| `openWorldHint` | `true` | The tool may reach an open world of external entities (a web search); false means a closed domain (a memory tool) |

An illustrative tool definition (our example, not from the spec) using only the spec's field names:

```json
{
  "name": "delete_file",
  "description": "Delete a file from the project workspace",
  "inputSchema": { "type": "object", "properties": { "path": { "type": "string" } }, "required": ["path"] },
  "annotations": {
    "title": "Delete file",
    "readOnlyHint": false,
    "destructiveHint": true,
    "idempotentHint": true,
    "openWorldHint": false
  }
}
```

Annotations are not a security control. Clients MUST treat them as untrusted unless the server is trusted, and the schema says "Clients should never make tool use decisions based on ToolAnnotations" received from untrusted servers ([MCP schema](https://modelcontextprotocol.io/specification/2026-07-28/schema)). In the Agent SDK, `readOnlyHint: true` has one practical effect: the tool can run in parallel with other read-only tools; the other hints are informational, and the SDK docs call annotations "metadata, not enforcement" ([Agent SDK custom tools](https://code.claude.com/docs/en/agent-sdk/custom-tools)). Anthropic's [desktop extension guide](https://claude.com/docs/connectors/building/mcpb) lists mandatory tool annotations for all tools among its directory submission requirements.

For safety the spec says there SHOULD always be a human in the loop able to deny a tool invocation, and clients SHOULD prompt for confirmation on sensitive operations, show tool inputs before calling the server, validate tool results before passing them to the model, implement timeouts, and log tool use for audit. Servers, for their part, MUST validate all tool inputs, implement proper access controls, rate limit invocations, and sanitize outputs.

**Stateful tools under a stateless protocol.** In 2026-07-28 a server keeps cross-call state by returning an explicit handle from a creation tool (the spec's example is `basket_id`) and accepting it as an argument on later calls; the model carries the handle forward. A call with an expired or unknown handle should return a tool execution error so the model can recover. The MCP [security best practices](https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices) add that servers MUST NOT treat possession of a state handle as authentication, and SHOULD bind handles to the authenticated user.

### Resources

Resources are application-driven: the host decides how to bring them into context, whether by selecting portions, searching them with embeddings, or passing everything to the model. Each resource is identified by a URI.

| Concept | Detail |
|---|---|
| Direct resource | Fixed URI, for example `calendar://events/2024` |
| Resource template | Parameterized RFC 6570 URI template, for example `travel://activities/{city}/{category}`; arguments can be auto-completed through the completion API |
| Contents | Text (`text`) or binary (`blob`, base64); one `resources/read` may return several contents, such as the files in a directory |
| Annotations | `audience` (`"user"`, `"assistant"`), `priority` (0.0 to 1.0), `lastModified` (ISO 8601) |
| Capability | `resources`, with optional `listChanged` and `subscribe` that a server may advertise independently |
| URI schemes | `https://` only when the client can fetch it directly, `file://`, `git://`, or a custom scheme that follows RFC 3986 |
| Server duties | MUST validate URIs and sanitize file paths against directory traversal |

A resource template (one entry of a `resources/templates/list` result), then a read of one concrete URI, trimmed from the [MCP resources spec](https://modelcontextprotocol.io/specification/2026-07-28/server/resources). Like the spec's own examples, the request omits the `_meta` fields (protocol version, client info, client capabilities). Every 2026-07-28 request MUST carry the required ones, protocol version and client capabilities, and clients SHOULD also include client info:

```json
{ "uriTemplate": "file:///{path}", "name": "Project Files", "description": "Access files in the project directory", "mimeType": "application/octet-stream" }
```

```json
{ "jsonrpc": "2.0", "id": 2, "method": "resources/read", "params": { "uri": "file:///project/src/main.rs" } }
```

**How the exam frames resources.** CCAR-F Task 2.4 treats resources as the way to expose content catalogs, with the guide's examples "issue summaries, documentation hierarchies, database schemas", so an agent can see what data exists "without requiring exploratory tool calls" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). In Claude Code a user references a resource with an @ mention in the form `@server:protocol://resource/path`, and Claude Code automatically gives Claude tools to list and read resources when servers support them; see [MCP in Claude Code](#mcp-in-claude-code).

### Prompts

Prompts are user-controlled: a server exposes them so the user can pick one explicitly, typically as a slash command, a command palette entry, a button or a context-menu item. The spec is precise about the word: user-controlled "refers to who decides when the prompt is used, not who authors its content" ([MCP prompts spec](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)). The server writes the prompt; the user triggers it.

- `prompts/list` (paginated) returns what is available; `prompts/get` takes a `name` and `arguments` and returns `messages`, each with a `role` of `"user"` or `"assistant"`.
- Servers SHOULD answer an invalid prompt name or a missing required argument with `-32602` (Invalid params), and an internal failure with `-32603` (Internal error).
- Prompt arguments can be auto-completed with `completion/complete`, which returns at most 100 values per response.
- Claude Code lists each server prompt as a command `/servername:promptname (MCP)`; typing `/mcp__servername__promptname` also runs it.

### Client primitives: elicitation, sampling and roots

Clients offer primitives to servers too. Two of the three are deprecated in 2026-07-28, with removal no earlier than the first revision released on or after 2027-07-28.

| Primitive | What it lets a server do | Status in 2026-07-28 | Rules worth knowing |
|---|---|---|---|
| Elicitation | Ask the user for information, in form mode (a flat schema of strings, numbers, booleans and enums) or URL mode (send the user to an external URL) | Current; delivered inside an `InputRequiredResult` instead of a server-initiated request | Passwords, API keys, access tokens and payment credentials MUST use URL mode, never form mode. The user answers accept, decline or cancel |
| Sampling | Request an LLM completion through the client, so the server needs no API key of its own | Deprecated; new implementations SHOULD NOT adopt it, and existing ones SHOULD migrate to calling LLM provider APIs directly | There SHOULD always be a human in the loop able to deny sampling requests |
| Roots | Tell a server which `file://` directories or files are relevant | Deprecated; pass paths through tool parameters, resource URIs or server configuration | Informational only, not a security boundary |

The [MCP client concepts page](https://modelcontextprotocol.io/docs/2026-07-28/learn/client-concepts) is explicit that roots serve as "a coordination mechanism between clients and servers, not a security boundary." As of September 2026, Claude Code still answers `roots/list` with the session's launch directory plus every additional working directory you have granted, and it shows elicitation dialogs in both form and URL modes (an `Elicitation` hook can auto-respond instead).

### Utilities

| Utility | How it works |
|---|---|
| Progress | The client puts a `progressToken` in the request's `_meta`; the server MAY send `notifications/progress`, and `progress` MUST increase each time |
| Cancellation | Transport-specific in 2026-07-28. Streamable HTTP: closing the SSE response stream cancels. stdio: the client MUST send `notifications/cancelled` |
| Pagination | Opaque cursors with a server-determined page size; a missing `nextCursor` means the last page; an invalid cursor SHOULD return `-32602` (Invalid params) |
| Completion | `completion/complete` suggests values for prompt and resource-template arguments (at most 100 per response) |
| Caching hints | In 2026-07-28, complete results from `server/discover`, the `*/list` methods and `resources/read` MUST carry `ttlMs` and `cacheScope` (`"public"` or `"private"`) |
| Logging | Deprecated in 2026-07-28; log to stderr or OpenTelemetry instead |

### Decide and avoid

Our decision rules, derived from the guide's resource objectives and the spec's control model:

- If the agent keeps calling list or search tools only to learn what exists (open issues, the documentation tree, table schemas), publish that catalog as a resource; not another tool, because the guide ties resources to fewer exploratory calls.
- If the operation changes state or needs parameters the model chooses at run time, make it a tool.
- If a user should trigger a canned workflow on demand, make it a prompt; prompts are designed for the user to select explicitly.
- Do not rely on annotations or roots for security. Annotations are untrusted hints and roots are informational guidance, not access control; enforce access in the server and in the host's permission system.
- The Messages API MCP connector currently supports only tool calls, not resources or prompts (as of September 2026); see [MCP in the API and the Agent SDK](#mcp-in-the-api-and-the-agent-sdk).

## MCP transports

*Tested in: CCDV-F D8.2 MCP Server Development (communication patterns: stdio, sockets, client vs. server) · CCAR-P 3.2, 3.7 · CCAR-F background only (deploying or hosting MCP servers is out of scope, APPX-OUTSCOPE-4)*

A transport carries JSON-RPC messages between a client and a server. The spec defines two standard transports, stdio and Streamable HTTP, and allows custom ones as long as they preserve the JSON-RPC message format, the message patterns and the per-request metadata model. Protocol semantics are identical on every transport; the transport only defines framing, delivery, how request metadata travels, and how cancellation and termination are signaled.

| | stdio | Streamable HTTP |
|---|---|---|
| Where the server runs | A local subprocess the client launches | Anywhere reachable over HTTP, behind one endpoint such as `https://example.com/mcp` |
| Clients per server | Typically one | Typically many |
| Framing | Newline-delimited JSON-RPC on stdin and stdout | Each client message is a new HTTP POST; the reply to a request is one JSON object or an SSE stream |
| Authentication | Credentials from the environment; MCP authorization is optional, and when it is supported the authorization spec SHOULD NOT be applied | MCP authorization is optional; when it is supported, the server SHOULD follow the MCP authorization spec (OAuth 2.1); bearer tokens, API keys and custom headers also work |
| Best for | Local processes on the same machine, with no network overhead | Remote servers |
| Status | Standard | Standard since 2025-03-26, when it replaced the 2024-11-05 HTTP+SSE transport |

Two other options exist. The old HTTP+SSE transport has been deprecated since 2025-03-26 and should be migrated to Streamable HTTP. A custom transport may run over any channel that supports bidirectional messages; one that runs over a reliable byte stream, such as a Unix domain socket or a TCP connection, SHOULD reuse the stdio framing rather than invent its own.

### stdio

The client starts the server as a subprocess. The server reads JSON-RPC messages from stdin and writes them to stdout, one message per line; messages MUST NOT contain embedded newlines ([stdio transport spec](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)).

- **stdout is reserved.** The server MUST NOT write anything to stdout that is not a valid MCP message. Logs go to stderr, which the client may capture, forward or ignore, and clients should not treat stderr output as an error. A stray `print()` or `console.log()` corrupts the stream; see [Building an MCP server](#building-an-mcp-server).
- **Environment.** What a stdio server inherits depends on the client. The [MCP debugging guide](https://modelcontextprotocol.io/docs/2026-07-28/tools/debugging) says servers launched over stdio inherit only a limited, platform-dependent subset of environment variables (its examples use Claude Desktop). Claude Code passes stdio servers its own environment minus the variables it strips from subprocesses, or only a safe baseline when `CLAUDE_CODE_MCP_ALLOWLIST_ENV=1`. Either way, pass what the server needs through the `env` key of its configuration. Use absolute paths, because the working directory may be undefined.
- **Lifecycle.** If the server exits unexpectedly the client SHOULD restart it; in-flight requests are lost and can be retried. Shutdown uses no protocol message in either era: the client SHOULD close the server's stdin, wait for it to exit, and only then force termination (on POSIX, typically `SIGTERM` escalating to `SIGKILL`; on Windows, `TerminateProcess` or Job Objects). Servers SHOULD exit promptly when stdin closes.
- **Security.** A server meant to run locally SHOULD guard against use by other local processes: use stdio, which limits access to the MCP client, or, if it listens on HTTP, require an authorization token or use Unix domain sockets or another IPC mechanism with restricted access.

### Streamable HTTP

The server exposes a single MCP endpoint. Every JSON-RPC message from the client MUST be a new HTTP POST whose `Accept` header lists both `application/json` and `text/event-stream`. For a request, the server answers with one JSON object or with an SSE stream, and clients must handle both; an accepted notification gets `202 Accepted` with no body (transport mechanics only: the 2026-07-28 core protocol defines no client-to-server notifications over Streamable HTTP).

A 2026-07-28 `tools/call` over Streamable HTTP, as the [Streamable HTTP spec](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http) shows it (the spec's example leaves out the `Accept` header listing `application/json` and `text/event-stream`, which a real client MUST also send):

```text
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: get_weather

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": { "location": "Seattle, WA" },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": { "name": "ExampleClient", "version": "1.0.0" },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Rules from the 2026-07-28 Streamable HTTP spec. The `Origin`, local-binding and authentication rules also appear in 2025-11-25, which already required `MCP-Protocol-Version` on requests after initialization; the `Mcp-Method`, `Mcp-Name` and `Mcp-Param-{name}` headers and the header-to-body match (`HeaderMismatch`) are new in 2026-07-28.

| Rule | Detail |
|---|---|
| `MCP-Protocol-Version` header | Required on every POST and must match the version in `_meta`, otherwise `400` with `HeaderMismatch` (`-32020`) |
| `Mcp-Method`, `Mcp-Name` headers | `Mcp-Method` on all requests; `Mcp-Name` on `tools/call`, `resources/read` and `prompts/get` |
| `Origin` validation | Servers MUST validate `Origin` on all incoming connections, to stop DNS rebinding, and MUST answer `403` when it is present and invalid |
| Local binding | A server running locally SHOULD bind only to `127.0.0.1`, not all interfaces (`0.0.0.0`) |
| Authentication | Every Streamable HTTP server SHOULD implement proper authentication for all connections |
| Proxies | Servers SHOULD send `X-Accel-Buffering: no` when opening an SSE stream so reverse proxies such as nginx do not buffer it |
| Sensitive parameters | The `x-mcp-header` value on a tool parameter supplies the `{name}` in an `Mcp-Param-{name}` header that mirrors the parameter's value: `"x-mcp-header": "Region"` gives `Mcp-Param-Region`. Header values are visible to network intermediaries, so sensitive parameters (passwords, API keys, tokens, PII) SHOULD NOT be marked |
| Cancellation | Closing a request's SSE response stream is the cancellation signal (on stdio the client sends `notifications/cancelled` instead) |

### What changed between the eras on HTTP

| Feature | 2025-11-25 | 2026-07-28 |
|---|---|---|
| Endpoint methods | POST and GET; GET opens an SSE stream for server-initiated messages, or returns `405` | POST only: the GET stream endpoint is removed, and a server supporting only this revision SHOULD answer GET or DELETE with `405` |
| Sessions | Server MAY assign `MCP-Session-Id` in the initialize response; a `404` on a request with that ID means start a new session; clients SHOULD send DELETE to end one | Protocol-level sessions and `Mcp-Session-Id` removed (a modern-only server ignores the header); cross-call state uses server-minted handles passed as tool arguments |
| Server-to-client requests | Servers could send JSON-RPC requests (sampling, elicitation, roots) on SSE streams | Not allowed; the server embeds them in an `InputRequiredResult` and the client retries (see [MCP errors and structured results](#mcp-errors-and-structured-results)) |
| Resuming a dropped stream | SSE event IDs plus a GET with `Last-Event-ID` | Not supported |
| Missing version header | Server SHOULD assume `2025-03-26` when it has no other way to tell | A server that still supports clients older than `2025-06-18` MAY treat it as `2025-03-26`; any other server MUST reject the request |
| Request headers | `MCP-Protocol-Version` on requests after initialization | `MCP-Protocol-Version` matching `_meta`, plus `Mcp-Method` on all requests and `Mcp-Name` on `tools/call`, `resources/read`, `prompts/get`; a mismatch gets `400` with `HeaderMismatch` |

A dual-era client talking to an unknown stdio server SHOULD probe with `server/discover` and fall back to `initialize` on any error that is not a recognized modern error, or on a timeout; a recognized modern error such as `UnsupportedProtocolVersion` means the server is modern, so the client picks a supported version instead of falling back. Over Streamable HTTP, a dual-era client MAY send a modern request first; on a `400` it inspects the body and falls back to `initialize` only if the body is empty or not a recognized modern error. For very old HTTP+SSE servers, a client that gets `400`, `404` or `405` without a modern error body issues a GET and expects an SSE stream whose first event is `endpoint`.

### Authorization on HTTP transports

Authorization is optional in MCP. When an HTTP server uses it, the server is an OAuth 2.1 resource server and the MCP client is an OAuth 2.1 client; a separate or co-hosted authorization server issues tokens ([MCP authorization spec](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)). The points that decide architecture questions (CCAR-P 3.2):

| Requirement | Detail |
|---|---|
| Discovery | Servers MUST publish OAuth 2.0 Protected Resource Metadata (RFC 9728), advertised in the `WWW-Authenticate` header of a `401` or at `/.well-known/oauth-protected-resource`; it names at least one authorization server |
| Client registration | Client ID Metadata Documents when there is no prior relationship (the most common case), pre-registration when there is one, or Dynamic Client Registration for backwards compatibility (deprecated in 2026-07-28) |
| PKCE | Clients MUST implement PKCE and use `S256` when technically capable |
| Audience | Clients send the RFC 8707 `resource` parameter; servers MUST accept only tokens issued for them and MUST NOT pass a client's token through to upstream APIs |
| Token placement | `Authorization: Bearer` on every request; never in the URI query string |
| Status codes | `401` authorization required or token invalid; `403` invalid scopes or insufficient permissions (step-up via `error="insufficient_scope"`); `400` malformed authorization request |
| Scope | Start minimal and elevate incrementally; avoid wildcard scopes such as `*`, `all`, `full-access` |

Confused deputy, token passthrough, SSRF and state handle hijacking, with their mitigations, are covered in [Security, Safety and Governance](security-and-governance.md#the-threat-model-for-claude-applications).

### Transports on Claude surfaces (as of September 2026)

| Surface | stdio | Streamable HTTP | SSE | Other |
|---|---|---|---|---|
| Claude Code | Yes: `claude mcp add ... -- <command>` | Yes, the recommended option for remote servers (`--transport http`) | Deprecated but accepted; from Claude Code v2.1.265, `--transport http` switches to SSE when the server does not accept HTTP | WebSocket (`"type": "ws"`) only through `.mcp.json` or `claude mcp add-json`, with header-only authentication and no OAuth |
| Claude Desktop | Yes, local servers and `.mcpb` extensions | Through custom connectors | A connector URL ending in `/sse` selects SSE | |
| Messages API MCP connector | No | Yes, publicly exposed servers (a private-network server through an MCP tunnel, research preview) | Yes | |
| Agent SDK | Yes (`command`) | Yes (`"type": "http"` in code) | Yes (`"type": "sse"`) | In-process SDK servers |

Reachability trips people up. A custom connector connects from Anthropic's cloud, not from the user's device, on every Claude client (claude.ai, Claude Desktop, Cowork and the mobile apps), so the server must be reachable from the public internet (or allowlist Anthropic's IP ranges). The Messages API connector documentation likewise requires a publicly exposed HTTP server and rules out local stdio servers. For a server inside a private network, MCP tunnels (research preview) carry traffic over an outbound-only connection, so no inbound firewall ports are opened; you attach a tunnel hostname to a Claude Managed Agents session (Managed Agents declare MCP servers by name and URL) or pass it to the Messages API through the MCP connector.

!!! warning "Exam guide vs current docs"

    The CCDV-F guide lists the communication patterns as "stdio, sockets, client vs. server" and does not define "sockets" ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). The current spec names two standard transports, stdio and Streamable HTTP, and neither is a socket transport by name. In current MCP and Claude Code documentation, sockets come up as custom transports (one over a reliable byte stream such as a Unix domain socket or TCP SHOULD reuse the stdio framing), as a way to restrict access to a local HTTP server (Unix domain sockets or another IPC mechanism with restricted access, per the MCP security best practices), and as Claude Code's WebSocket servers (`"type": "ws"`), which hold a persistent bidirectional connection. WebSocket is not a standard MCP transport, and Claude Code connects SSE and WebSocket servers on the earlier, handshake-based protocol.

    Answer in the guide's terms. Our reading, since the guide gives no definition: "stdio" is a local server the client launches as a subprocess and talks to over standard streams; "sockets" is a server the client reaches over a connection instead of launching it; "client vs. server" is the host's one-client-per-server structure in [MCP architecture](#mcp-architecture). The guide names no MCP revision, so expect wording from either era. Do not treat an option that says SSE as the modern choice: the standalone HTTP+SSE transport is the deprecated 2024-11-05 design, not Streamable HTTP.

### Decide and avoid

- If the server needs the local filesystem, local tools, or must work offline, choose stdio (or a `.mcpb` extension in Claude Desktop); not a remote server, which runs on your servers rather than on the user's machine (as a remote connector in claude.ai, Claude Desktop or Cowork, it is also reached through Anthropic's infrastructure).
- If several applications, users or cloud agents must share one server, choose Streamable HTTP with OAuth; not stdio, which typically serves a single client on the same machine.
- If a Messages API application must call the server, it has to be reachable over HTTP (publicly, or from a private network through an MCP tunnel); a local stdio server will not work with the MCP connector.
- If an item offers HTTP+SSE for a new build, it is the deprecated option.
- If a Claude Code server must push events to Claude unprompted, a WebSocket (`"type": "ws"`) server fits; if it only responds to requests, use HTTP, which supports OAuth and the `--transport` flag.
- If a stdio server breaks its connection for no visible reason, suspect writes to stdout.

## MCP errors and structured results

*Tested in: CCAR-F 2.2 (2.2-K1 to 2.2-S4), APPX-TECH-2, APPX-INSCOPE-7, Exercise 1 step 3 · CCDV-F D8.1 Tool Implementation (error handling), D4.1 Debugging and Error Handling (error type identification, recovery strategy selection)*

A tool can fail in two ways in MCP, and the choice decides whether Claude reliably sees what went wrong and can act on it. CCAR-F Task 2.2 is built on this: its first knowledge bullet is "The MCP isError flag pattern for communicating tool failures back to the agent" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

### Two error channels

| | Protocol error | Tool execution error |
|---|---|---|
| How it is sent | A JSON-RPC `error` object (`code`, `message`, optional `data`) in place of a `result` | A normal `result` whose content explains the failure, with `isError: true` |
| Causes the spec lists | Unknown tool, malformed request, server errors | API failures, input validation errors (a date in the wrong format, a value out of range), business logic errors |
| Can the model fix it? | Less likely: the problem is the request structure itself | Often: the content carries actionable feedback the model can use to self-correct and retry |
| What the client should do | MAY pass it to the model | SHOULD pass it to the model |

The [MCP tools spec](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) says tool execution errors "contain actionable feedback that language models can use to self-correct and retry with adjusted parameters". `isError` defaults to false, and the [MCP schema](https://modelcontextprotocol.io/specification/2026-07-28/schema) explains why errors from inside the tool belong in the result: otherwise "the LLM would not be able to see that an error occurred". Revision 2025-11-25 clarified that input validation errors should be returned as tool execution errors rather than protocol errors, to enable model self-correction.

=== "Tool execution error"

    ```json
    {
      "jsonrpc": "2.0",
      "id": 4,
      "result": {
        "resultType": "complete",
        "content": [
          { "type": "text", "text": "Invalid departure date: must be in the future. Current date is 08/08/2025." }
        ],
        "isError": true
      }
    }
    ```

=== "Protocol error"

    ```json
    {
      "jsonrpc": "2.0",
      "id": 3,
      "error": { "code": -32602, "message": "Unknown tool: invalid_tool_name" }
    }
    ```

Both examples come from the 2026-07-28 spec. The `resultType` field is new in that revision: every result MUST carry one, and the core protocol defines `"complete"` and `"input_required"` (extensions MAY add values). A client MUST treat a missing `resultType` from an older server as `"complete"`, so a 2025-11-25 server, which never sends it, still works.

### JSON-RPC error codes

| Code | Meaning | Revision |
|---|---|---|
| `-32700`, `-32600` to `-32603` | Standard JSON-RPC 2.0 codes for general protocol failures | Both |
| `-32602` | Invalid params: unknown tool, unknown prompt or missing required prompt argument; the 2026-07-28 spec also uses it for an invalid pagination cursor, a request missing a required `_meta` field, and a resource that does not exist | Both for the tool and prompt uses; 2026-07-28 for the `_meta` and missing-resource uses |
| `-32603` | Internal error (the code the prompts and resources specs give for internal failures) | Both |
| `-32002` | Resource not found | 2025-11-25 and earlier; 2026-07-28 servers MUST NOT emit it, but clients SHOULD still accept it |
| `-32042` | URL elicitation required | 2025-11-25 only |
| `-32020` | `HeaderMismatch` | 2026-07-28 |
| `-32021` | `MissingRequiredClientCapability` | 2026-07-28 |
| `-32022` | `UnsupportedProtocolVersion` | 2026-07-28 |
| `-32000` to `-32019` | Legacy, implementation-defined sub-range; new implementations SHOULD NOT use it | Partitioned in 2026-07-28 |
| `-32020` to `-32099` | Reserved for codes the MCP spec defines | 2026-07-28 |

### A failure is not an empty answer

CCAR-F skill 2.2-S4 separates access failures from valid empty results; the general rule, with the Q8 anti-pattern, is in [Returning tool results and errors](#returning-tool-results-and-errors). MCP writes the same rule into the protocol for resources: a server MUST NOT return an empty `contents` array for a resource that does not exist, because "An empty array is ambiguous" ([MCP resources spec](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)); it returns a JSON-RPC error instead (`-32602` in 2026-07-28, `-32002` in earlier revisions). For tools, the equivalent is to return a normal result that says nothing matched when the query ran, and `isError: true` when it could not run.

The same thinking covers a stateful tool called with an expired or unknown handle: it returns a tool execution error, as described under [MCP primitives](#mcp-primitives).

### Structured error metadata: the exam guide's convention

The spec defines one flag, `isError`, plus free-form content. The CCAR-F guide asks for more structure inside that content: "errorCategory (transient/validation/permission), isRetryable boolean, and human-readable descriptions" (2.2-S1), and "retriable: false flags and customer-friendly explanations for business rule violations" (2.2-S2) ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The reason is 2.2-K3: a uniform "Operation failed" leaves the agent unable to choose a recovery, and 2.2-K4: a retryable flag prevents wasted retries.

The guide's knowledge bullet 2.2-K2 names four error types (transient, validation, business, permission); the recovery each one calls for is tabulated in [Returning tool results and errors](#returning-tool-results-and-errors). What matters at the MCP layer is where the metadata goes.

The payload travels as text in the `content` of an `isError: true` result. A business-rule refusal at the protocol level (illustrative; the category value `"business"` is the fourth type from 2.2-K2, which 2.2-S1's list of `errorCategory` values leaves out):

```json
{
  "jsonrpc": "2.0",
  "id": 7,
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

In the Agent SDK, a custom tool is served by an in-process MCP server, and the flag is spelled differently in each language. Adapted from the error-handling example in the [Agent SDK custom tools docs](https://code.claude.com/docs/en/agent-sdk/custom-tools); the order-service URL is a placeholder and the field names are the guide's:

=== "Python"

    ```python
    import json
    from typing import Any

    import httpx
    from claude_agent_sdk import tool

    @tool("lookup_order", "Look up one order by its order number", {"order_id": str})
    async def lookup_order(args: dict[str, Any]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"https://orders.example.com/orders/{args['order_id']}")
        except Exception as e:
            # A network error or timeout would otherwise reach Claude as the raw message.
            error = {
                "errorCategory": "transient",
                "isRetryable": True,
                "description": f"Could not reach the order service: {e}. Retry the same call once.",
            }
            return {"content": [{"type": "text", "text": json.dumps(error)}], "is_error": True}
        if response.status_code == 503:
            error = {
                "errorCategory": "transient",
                "isRetryable": True,
                "description": "The order service is temporarily unavailable. Retry the same call once.",
            }
            # is_error marks this as a failed call rather than odd-looking data.
            return {"content": [{"type": "text", "text": json.dumps(error)}], "is_error": True}
        if response.status_code == 404:
            error = {
                "errorCategory": "validation",
                "isRetryable": False,
                "description": f"No order {args['order_id']} exists. Ask the customer to confirm the order number.",
            }
            return {"content": [{"type": "text", "text": json.dumps(error)}], "is_error": True}
        if response.status_code != 200:
            # Any other failure status is still a failed call, never a normal result.
            return {
                "content": [{"type": "text", "text": f"Order service error: {response.status_code} {response.reason_phrase}"}],
                "is_error": True,
            }
        return {"content": [{"type": "text", "text": response.text}]}
    ```

=== "TypeScript"

    ```typescript
    import { tool } from "@anthropic-ai/claude-agent-sdk";
    import { z } from "zod";

    const lookupOrder = tool(
      "lookup_order",
      "Look up one order by its order number",
      { order_id: z.string() },
      async (args) => {
        let response: Response;
        try {
          response = await fetch(`https://orders.example.com/orders/${args.order_id}`);
        } catch (error) {
          // A network error or timeout would otherwise reach Claude as the raw message.
          const reason = error instanceof Error ? error.message : String(error);
          const failure = {
            errorCategory: "transient",
            isRetryable: true,
            description: `Could not reach the order service: ${reason}. Retry the same call once.`
          };
          return { content: [{ type: "text", text: JSON.stringify(failure) }], isError: true };
        }
        if (response.status === 503) {
          const error = {
            errorCategory: "transient",
            isRetryable: true,
            description: "The order service is temporarily unavailable. Retry the same call once."
          };
          // isError marks this as a failed call rather than odd-looking data.
          return { content: [{ type: "text", text: JSON.stringify(error) }], isError: true };
        }
        if (response.status === 404) {
          const error = {
            errorCategory: "validation",
            isRetryable: false,
            description: `No order ${args.order_id} exists. Ask the customer to confirm the order number.`
          };
          return { content: [{ type: "text", text: JSON.stringify(error) }], isError: true };
        }
        if (!response.ok) {
          // Any other failure status is still a failed call, never a normal result.
          return {
            content: [{ type: "text", text: `Order service error: ${response.status} ${response.statusText}` }],
            isError: true
          };
        }
        return { content: [{ type: "text", text: await response.text() }] };
      }
    );
    ```

If a handler throws instead, the SDK's in-process server converts the exception into an error result carrying the raw message, and the agent loop continues. In both cases Claude can retry, try another tool, or explain the failure. Catch errors yourself when the raw message would not tell Claude what to do next: the `try`/`except` (`try`/`catch`) block above is what gives a network failure or timeout the full metadata.

Where the retry happens is a design decision of its own. CCAR-F skill 2.2-S3 asks for local recovery of transient failures inside a subagent, and for propagating to the coordinator only what cannot be resolved locally, with partial results and what was attempted. How the coordinator uses that is covered in [Evaluation, Debugging and Reliability](evaluation-and-reliability.md#error-propagation-in-multi-agent-systems).

!!! warning "Exam guide vs current docs"

    `errorCategory`, `isRetryable` and `retriable` are the exam guide's application-level vocabulary. The [MCP tools spec](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) defines `isError: true` plus free-form content and contains no field with those names. The guide is also inconsistent with itself: 2.2-K2 names four error types while 2.2-S1 and Exercise 1 list three `errorCategory` values, and the retry flag is written `isRetryable` in 2.2-S1 but `retriable: false` in 2.2-S2. On the exam, answer in the guide's terms: a category, a retryable flag, and a human-readable description, carried in an `isError` result.

### Structured results: `outputSchema` and `structuredContent`

A successful result can carry machine-readable data as well as text. A tool that declares an `outputSchema` MUST return `structuredContent` that conforms to it, and clients SHOULD validate it. In 2026-07-28 `structuredContent` can be any JSON value (object, array, string, number, boolean or null); the 2025-11-25 text described it as a JSON object, the 2026-07-28 changelog lists the loosening, and the Agent SDK custom tools page still calls it an optional JSON object. For backwards compatibility, a tool that returns structured content SHOULD also return the serialized JSON in a text block.

=== "Tool definition"

    ```json
    {
      "name": "get_weather_data",
      "title": "Weather Data Retriever",
      "description": "Get current weather data for a location",
      "inputSchema": {
        "type": "object",
        "properties": {
          "location": { "type": "string", "description": "City name or zip code" }
        },
        "required": ["location"]
      },
      "outputSchema": {
        "type": "object",
        "properties": {
          "temperature": { "type": "number", "description": "Temperature in celsius" },
          "conditions": { "type": "string", "description": "Weather conditions description" },
          "humidity": { "type": "number", "description": "Humidity percentage" }
        },
        "required": ["temperature", "conditions", "humidity"]
      }
    }
    ```

=== "Result"

    ```json
    {
      "jsonrpc": "2.0",
      "id": 5,
      "result": {
        "resultType": "complete",
        "content": [
          { "type": "text", "text": "{\"temperature\": 22.5, \"conditions\": \"Partly cloudy\", \"humidity\": 65}" }
        ],
        "structuredContent": { "temperature": 22.5, "conditions": "Partly cloudy", "humidity": 65 }
      }
    }
    ```

Three details decide questions about structured results:

- **It is not "structured outputs".** The spec says `structuredContent` "is server-produced result data and is unrelated to LLM" structured outputs, which it glosses as schema-constrained model generation ([MCP tools spec](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)). Constraining Claude's own response format is a different feature; see [Structured outputs](claude-api.md#structured-outputs).
- **Claude may not see the text copy.** In the Agent SDK, when `structuredContent` is set, Claude receives the JSON plus any image or resource blocks, and text blocks in `content` are not forwarded because they are assumed to duplicate it. The Python `@tool` decorator forwards only `content` and `is_error`, so returning `structuredContent` from Python needs a standalone MCP server.
- **Other content types.** Unstructured `content` can hold text, images and audio (base64 `data` with a `mimeType`), `resource_link` items (not guaranteed to appear in `resources/list`), and embedded resources, whose servers SHOULD implement the `resources` capability.

### How an MCP failure reaches Claude

The flag changes spelling by layer (`isError` in MCP, `is_error` on a Messages API `tool_result`); the full list is in [Returning tool results and errors](#returning-tool-results-and-errors). Two MCP-specific cases complete it. With the Messages API MCP connector, a failed call comes back as an `mcp_tool_result` block with `is_error`, paired with the `mcp_tool_use` block that names the `server_name`. In Claude Code, whether Claude hears about a server that failed to connect depends on tool search, which is on by default: with it, Claude Code tells Claude which server failed and its connection error, so Claude reports the connection failure in its response; in any configuration without tool search, Claude Code does not report failed server connections to Claude.

### When the server needs input: not an error

In 2026-07-28 a server that needs input from the client mid-call (an elicitation from the user, an LLM completion through sampling, or the client's roots) returns a result with `resultType: "input_required"` instead of failing or sending its own request. This is allowed only on `tools/call`, `resources/read` and `prompts/get`. The client collects the input and retries with `inputResponses` under a new JSON-RPC `id`, echoing the opaque `requestState` exactly; the server MUST treat `requestState` as attacker-controlled.

Server reply to `tools/call` with `id` 2, then the client's retry ([MCP tools spec](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)):

```json
{ "jsonrpc": "2.0", "id": 2,
  "result": { "resultType": "input_required",
    "inputRequests": { "github_login": { "method": "elicitation/create",
      "params": { "mode": "form", "message": "Please provide your GitHub username",
        "requestedSchema": { "type": "object", "properties": { "name": { "type": "string" } }, "required": ["name"] } } } },
    "requestState": "eyJsb2NhdGlvbiI6Ik5ldyBZb3JrIn0..." } }
```

```json
{ "jsonrpc": "2.0", "id": 3, "method": "tools/call",
  "params": { "name": "get_weather", "arguments": { "location": "New York" },
    "inputResponses": { "github_login": { "action": "accept", "content": { "name": "octocat" } } },
    "requestState": "eyJsb2NhdGlvbiI6Ik5ldyBZb3JrIn0..." } }
```

### Decide and avoid

- If the input was invalid, return an `isError` result that says what was wrong and how to fix it; not a JSON-RPC protocol error, which the model is less able to act on.
- If a timeout or an unavailable service caused the failure, mark it transient and retryable and recover locally; if that fails, propagate structured context (the failure type, the attempted query, any partial results, possible alternatives) so the coordinator can decide. Not a generic "search unavailable" status after exhausted retries, which is option B of CCAR-F sample question 8 in the [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) and hides that context.
- If a business rule blocked the action, mark it non-retryable and include a customer-friendly explanation; retrying cannot change a policy (our reasoning from 2.2-S2 and 2.2-K4).
- If the query succeeded with no matches, return a normal result; not `isError`, which would send the agent into a retry decision it does not need. The reverse is the sample-question trap: a timeout returned as an empty result marked successful (option C of question 8) prevents any recovery.
- If a stateful call names an expired or unknown handle, return a tool execution error that says so.
- If the tool name does not exist or the request is malformed, a JSON-RPC error (`-32602` for an unknown tool) is correct.
- Do not describe `errorCategory` or `isRetryable` as MCP spec fields, and do not confuse `structuredContent` with structured outputs.

## Building an MCP server

*Tested in: CCDV-F D8.2 MCP Server Development (server authoring, deployment, integration with Claude applications), Sample 3 · CCAR-F 2.4-S4, Exercise 1 step 1 (hosting is out of scope, APPX-OUTSCOPE-4) · CCAR-P 3.7*

CCDV-F D8.2 covers server authoring, deployment and integration with Claude applications, and CCAR-F 2.4-S4 asks the question that comes first: whether to build a server at all. This section follows that order: build or reuse, the official SDKs, a minimal server, testing with the MCP Inspector, connecting to a Claude host, and a design checklist.

### Build or reuse?

Before writing a server, check whether one exists. CCAR-F skill 2.4-S4 prefers "existing community MCP servers over custom implementations for standard integrations (e.g., Jira), reserving custom servers for team-specific workflows" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Where to look, and what each source does and does not promise:

| Source | What it is | Caveat |
|---|---|---|
| Anthropic Directory | Reviewed connectors; any remote server listed there can be added to Claude Code with `claude mcp add` | Anthropic reviews listings against its criteria but does not security-audit or manage any MCP server |
| MCP Registry | The official centralized metadata repository for publicly accessible MCP servers (in preview) | Hosts metadata, not code; relies on the wider ecosystem for security scanning; private servers belong in your own private registry |
| `modelcontextprotocol/servers` repository | Reference implementations | "educational examples", not production-ready solutions |

The quoted phrase is from the [servers repository README](https://github.com/modelcontextprotocol/servers). Build your own when the capability is specific to your team, or when the CCDV-F Sample 3 situation applies: an internal service that several Claude applications must share and that should be "maintained independently" of any one app ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)).

### SDKs

Official MCP SDKs exist in three tiers (2026-07-28 docs): Tier 1 TypeScript, Python, C#, Go and Rust; Tier 2 Java and Ruby; Tier 3 Swift, PHP and Kotlin. Every SDK can build servers that expose tools, resources and prompts, build clients, and use local and remote transports.

| | Python SDK v2 (current) | TypeScript SDK v2 (current) |
|---|---|---|
| Install | `uv add "mcp[cli]"` (or pip); the `cli` extra adds `mcp dev`, `mcp run`, `mcp install` | `npm install @modelcontextprotocol/server` (clients use `@modelcontextprotocol/client`) |
| Runtime | Python 3.10+ | Node.js 20 or higher in the official tutorial |
| Server class | `from mcp.server import MCPServer` | `import { McpServer } from '@modelcontextprotocol/server'` |
| Register a tool | `@mcp.tool()` decorator | `server.registerTool(name, config, handler)` |
| Input schema | Type hints are the schema; type hints and the docstring generate the tool definition | Standard Schema: Zod v4, Valibot, ArkType or any compatible library |
| Spec support | 2026-07-28 and every earlier revision | Released alongside the 2026-07-28 spec |

### A minimal server in each language

Both servers below run over stdio, which is what Claude Desktop and a local Claude Code configuration launch. The Python version is the SDK README's 15-line server plus the stdio entry point from the official tutorial; the TypeScript version is the TypeScript SDK README's example.

=== "Python"

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

    ```bash
    uv add "mcp[cli]"
    uv run mcp dev server.py                               # open the server in the MCP Inspector
    uv run mcp run server.py --transport streamable-http   # serve it over Streamable HTTP instead
    ```

=== "TypeScript"

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

    ```bash
    npm install @modelcontextprotocol/server zod
    npx @modelcontextprotocol/inspector node build/index.js   # after compiling with tsc
    ```

What the Python version did not need is the point of the SDK: "no JSON Schema (`a: int, b: int` _is_ the schema)", in the words of the [Python SDK README](https://github.com/modelcontextprotocol/python-sdk), and no request parsing, validation code or protocol handling. The SDK builds the tool definition from the type hints and the docstring, so write the docstring for the model.

!!! warning "Older tutorials vs current SDKs"

    The official build-server tutorial for the 2025-11-25 revision uses the v1 SDK names: `from mcp.server.fastmcp import FastMCP` with `mcp = FastMCP("weather")` in Python, and the single package `@modelcontextprotocol/sdk` with imports such as `@modelcontextprotocol/sdk/server/mcp.js` in TypeScript. The current v2 lines use `MCPServer` and the split `@modelcontextprotocol/server` and `@modelcontextprotocol/client` packages, and `pip install mcp` now installs 2.x. The TypeScript v1 line keeps receiving bug fixes and security updates for at least 6 months after v2's release; the Python v1 line lives on a `v1.x` branch and still gets critical bug fixes and security patches. The exam guides name no SDK version, so recognize both: `FastMCP` in an item means the v1 Python API.

### The stdout rule

For a stdio server, stdout belongs to the protocol. The official tutorial is blunt: "Writing to stdout will corrupt the JSON-RPC messages and break your server" ([build-server tutorial](https://modelcontextprotocol.io/docs/2026-07-28/develop/build-server)). In Python keep `print()` out of the server and use the `logging` module, which writes to stderr; in TypeScript use `console.error()`, never `console.log()`. HTTP servers may log to stdout. The MCP debugging guide says the host application captures a stdio server's stderr automatically; a Streamable HTTP server's stderr is not captured by the client, so use your own server-side log aggregation or OpenTelemetry. Logging over the protocol itself (`notifications/message`) is deprecated as of 2026-07-28.

### Test it with the MCP Inspector

The MCP Inspector is the reference tool for testing and debugging servers. One npm package, `@modelcontextprotocol/inspector`, provides a web UI (the default, on port `6274`), a scriptable CLI (`--cli`) and a TUI (`--tui`). It needs Node 22.19.0 or newer and runs through `npx` with no install, and it negotiates the legacy or modern protocol era the same way in all three clients.

```bash
npx @modelcontextprotocol/inspector node path/to/server/index.js            # web UI
npx @modelcontextprotocol/inspector --cli node path/to/server/index.js --method tools/list
npx @modelcontextprotocol/inspector --server-url https://api.example.com/mcp --transport http
```

### Connect it to a Claude host

The Claude Desktop entries below are the official 2026-07-28 build-server tutorial's configurations for its `weather` server; substitute your own absolute path and file name. The Claude Code commands are our illustration: the documented stdio syntax `claude mcp add [options] <name> -- <command> [args...]` wrapped around the same launch command.

=== "Python server"

    Claude Desktop, in `claude_desktop_config.json` (macOS `~/Library/Application Support/Claude/`, Windows `%APPDATA%\Claude\`):

    ```json
    {
      "mcpServers": {
        "weather": {
          "command": "uv",
          "args": ["--directory", "/ABSOLUTE/PATH/TO/PARENT/FOLDER/weather", "run", "weather.py"]
        }
      }
    }
    ```

    Claude Code, with everything after `--` passed to the server untouched:

    ```bash
    claude mcp add --transport stdio weather -- uv --directory /ABSOLUTE/PATH/TO/PARENT/FOLDER/weather run weather.py
    ```

=== "TypeScript server"

    Claude Desktop, in `claude_desktop_config.json`:

    ```json
    {
      "mcpServers": {
        "weather": {
          "command": "node",
          "args": ["/ABSOLUTE/PATH/TO/PARENT/FOLDER/weather/build/index.js"]
        }
      }
    }
    ```

    Claude Code:

    ```bash
    claude mcp add --transport stdio weather -- node /ABSOLUTE/PATH/TO/PARENT/FOLDER/weather/build/index.js
    ```

Use absolute paths: a stdio server's working directory may be undefined (for example `/` on macOS). Claude Desktop writes MCP logs to `~/Library/Logs/Claude` on macOS or `%APPDATA%\Claude\logs` on Windows, with connection failures in `mcp.log`.

**Going remote.** Local servers configured in Claude Desktop are not available in Cowork or claude.ai, and the Messages API MCP connector cannot connect to a local stdio server. To serve those surfaces, or several applications at once, run the server over HTTP: the Python SDK serves the same file with `uv run mcp run server.py --transport streamable-http`, and the README's client example then connects to `http://localhost:8000/mcp`. Where each Claude surface connects from, and why a custom connector's server must be reachable from the public internet, is in [Transports on Claude surfaces](#transports-on-claude-surfaces-as-of-september-2026). Before exposing it, apply the Streamable HTTP security rules and, if the server supports authorization (optional in MCP), the authorization spec that HTTP servers SHOULD then follow (OAuth 2.1), both in [MCP transports](#mcp-transports). For CCAR-F, hosting details stop there (see the last item under Decide and avoid below).

**Distributing a local server.** A desktop extension (`.mcpb`, formerly `.dxt`) is a zip archive holding a local MCP server and a `manifest.json`, installable in Claude Desktop with a single click. It runs locally over stdio, bundles its dependencies and needs no OAuth; values marked `"sensitive": true` in `user_config` are kept in the operating system's secret store. Build one with `npm install -g @anthropic-ai/mcpb`, then `mcpb init` and `mcpb pack`. Anthropic calls MCPB the secondary distribution path and recommends remote servers for directory listing.

A scaffolding shortcut exists in Claude Code: the official `mcp-server-dev` plugin (`/plugin install mcp-server-dev@claude-plugins-official`, then `/mcp-server-dev:build-mcp-server`) asks about the use case and scaffolds a remote HTTP or local stdio server.

### Server design checklist

The protocol is the easy part; these rules decide whether Claude uses the server well and safely.

| Area | Rule | Source |
|---|---|---|
| Inputs and outputs | Validate inputs, enforce access control, rate limit invocations, sanitize outputs | MCP tools spec (MUST) |
| Resources | Validate URIs; sanitize file paths against directory traversal | MCP resources spec (MUST) |
| Tool surface | Group tools around intent; "Fewer, well-described tools consistently outperform exhaustive API mirrors." | Anthropic's MCP production post |
| Descriptions | Claude Code truncates each tool description and each server's instructions at 2,048 characters by default, so put critical details first | Claude Code MCP docs |
| Server instructions | Tell Claude when to search for your tools: task categories, when to search, key capabilities | Claude Code MCP docs |
| Schemas | The Claude API does not accept `anyOf`, `oneOf` or `allOf` at the root of an input schema (Claude Code flattens such schemas before sending, or skips the tool if it cannot); Claude Code also excludes a tool whose top-level property names are not 1 to 64 characters of ASCII letters, digits, `_`, `.` or `-` | Claude Code MCP docs |
| Listing order | Return `tools/list` in a deterministic order, which helps client caching and prompt-cache hit rates | 2026-07-28 changelog (SHOULD) |
| State | Return explicit handles instead of relying on connection state; check that a handle belongs to the caller | MCP tools spec, security best practices |
| Annotations | Set `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint` honestly; unset, they default to the cautious side (`destructiveHint` and `openWorldHint` true, `readOnlyHint` and `idempotentHint` false); they are hints, not controls | MCP schema |
| Large results | `_meta["anthropic/maxResultSizeChars"]` raises one tool's Claude Code threshold for text content, up to 500,000 characters; it has no effect on image content | Claude Code MCP docs |
| Always-confirm tools | `_meta["anthropic/requiresUserInteraction"]: true` makes Claude Code prompt on every call, even in `acceptEdits`, `auto` and `bypassPermissions` modes | Claude Code MCP docs |

The quoted tool-surface rule is from [Building agents that reach production systems with MCP](https://claude.com/blog/building-agents-that-reach-production-systems-with-mcp); the post's advice for very large APIs is in [Fewer, higher-level tools](#fewer-higher-level-tools). Writing the descriptions themselves is covered in [Writing tool descriptions that steer selection](#writing-tool-descriptions-that-steer-selection), and sizing the tool set in [Designing a tool set](#designing-a-tool-set).

### Decide and avoid

- If a standard integration already has an existing community server (Jira is the guide's example), use it; reserve custom servers for team-specific workflows.
- If one internal service must serve several Claude applications and be maintained independently of any one app, build an MCP server that exposes its operations as tools (CCDV-F Sample 3; the distractors are worked in [How the official sample frames it](#how-the-official-sample-frames-it)).
- If an item shows a stdio server calling `print()` or `console.log()`, that is the bug.
- If an item asks how to test a server before connecting it to Claude, choose the MCP Inspector.
- For CCAR-F, stop at configuration and design: deploying or hosting MCP servers (infrastructure, networking, container orchestration) is listed as out of scope.

## MCP in Claude Code

*Tested in: CCAR-F 2.4 (2.4-K1 to 2.4-K4, 2.4-S1 to 2.4-S3, 2.4-S5), APPX-TECH-2, APPX-INSCOPE-6, Exercise 2 step 4 · CCDV-F D8.2 (integration with Claude applications), D7.4 Identity, Secrets, and Key Management · CCAR-P 7.1*

CCAR-F Task 2.4 is mostly about this section: where a server's configuration lives, how secrets stay out of git, what Claude sees once servers connect, and how resources reduce exploratory calls. The configuration-layer view of the same files (how `.mcp.json` sits beside `settings.json`, `CLAUDE.md` and plugins) is in [Claude Code Configuration](claude-code-configuration.md#mcp-servers-in-claude-code).

### Adding servers from the CLI

```bash
# Remote HTTP server (the recommended transport for remote servers); local scope by default
claude mcp add --transport http notion https://mcp.notion.com/mcp

# Remote server with a static bearer token header
claude mcp add --transport http secure-api https://api.example.com/mcp \
  --header "Authorization: Bearer your-token"

# Local stdio server: Claude Code's options first, then --, then the server command
claude mcp add --env AIRTABLE_API_KEY=YOUR_KEY --transport stdio airtable \
  -- npx -y airtable-mcp-server

# Scopes
claude mcp add --transport http shared-server --scope project https://example.com/mcp    # writes .mcp.json
claude mcp add --transport http hubspot --scope user https://mcp.hubspot.com/anthropic   # all your projects

# From JSON, and management
claude mcp add-json weather-api '{"type":"http","url":"https://api.weather.com/mcp","headers":{"Authorization":"Bearer token"}}'
claude mcp list
claude mcp get notion
claude mcp remove notion
```

| Flag or rule | Detail |
|---|---|
| `-s`, `--scope` | `local` (default), `project`, `user` |
| `-e`, `--env` | Set an environment variable for the server, for example `-e KEY=value`. It accepts several `KEY=value` pairs, so put at least one other option (such as `--transport stdio`) between `--env` and the server name, or the CLI reads the name as another pair and rejects it |
| `-t`, `-H` | Short forms of `--transport` and `--header` |
| `--` | Separates Claude Code's options from the server command; everything after it goes to the server untouched |
| Server names | Letters, numbers, hyphens and underscores only; names of built-in servers such as `workspace`, `claude-in-chrome` and `computer-use` are reserved |
| `type` in JSON | `streamable-http` is accepted as an alias for `http`. An entry with a `url` but no `type` is a configuration error, because an entry with no `type` is read as stdio |
| SSE | `--transport sse` still exists but the SSE transport is deprecated; from v2.1.265 `--transport http` falls back to SSE when a server does not accept HTTP |
| WebSocket | `"type": "ws"` works only through `.mcp.json` or `claude mcp add-json`; `--transport` does not accept `ws` |
| Inside a session | `/mcp` shows server status, handles OAuth, and reconnects, enables or disables servers |

### The three scopes

| Scope | Loads in | Shared with team | Stored in | Use it for |
|---|---|---|---|---|
| Local (default) | The current project only | No | `~/.claude.json`, under that project's path | Personal development servers, experiments, servers whose credentials must stay out of version control |
| Project | The current project | Yes, through version control | `.mcp.json` at the project root | Team tooling everyone should get when they clone the repository |
| User | All your projects | No | `~/.claude.json`, under the top-level `mcpServers` key | Personal servers you want everywhere |

What a local-scope entry looks like inside `~/.claude.json` ([Claude Code MCP docs](https://code.claude.com/docs/en/mcp)):

```json
{
  "projects": {
    "/path/to/your/project": {
      "mcpServers": {
        "stripe": { "type": "http", "url": "https://mcp.stripe.com" }
      }
    }
  }
}
```

Three rules settle most scope questions:

- **One winner, no merging.** When the same server is defined in more than one place, Claude Code connects to it once, using the highest-precedence source in this order: local, project, user, plugin-provided servers, claude.ai connectors. It takes the whole entry from the winning source; "fields are not merged across scopes" ([Claude Code MCP docs](https://code.claude.com/docs/en/mcp)). The three scopes match duplicates by name; plugins and connectors match by endpoint (the same URL or command). A server an organization provides through the `managedMcpServers` managed setting ranks above all of these.
- **MCP local scope is not `settings.local.json`.** MCP local scope lives in `~/.claude.json` in your home directory; general local settings live in `.claude/settings.local.json` in the project.
- **`settings.json` does not read an `mcpServers` key.** Servers added there never appear. Project servers go in `.mcp.json` at the repository root, not inside `.claude/`, and under the `mcpServers` key (a VS Code style top-level `servers` key is not read).

!!! warning "Exam guide vs current docs"

    The CCAR-F guide contrasts two levels: "project-level (.mcp.json) for shared team tooling vs user-level (~/.claude.json) for personal/experimental servers" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). As of September 2026, the docs describe three scopes, and the default is local, not user: a plain `claude mcp add` makes a server private to the current project, not available in all projects. Local scope is also stored in `~/.claude.json`, and the docs say to use it for "personal development servers, experimental configurations, or servers with credentials you don't want in version control" ([Claude Code MCP docs](https://code.claude.com/docs/en/mcp)). The guide's contrast still holds (shared goes in `.mcp.json`, personal goes in `~/.claude.json`), so answer scope items in the guide's terms and remember `--scope user` when a server must follow you across projects.

### Secrets: environment variable expansion in `.mcp.json`

A committed `.mcp.json` should hold references to credentials, never the credentials themselves. Claude Code expands environment variables in it, "allowing teams to share configurations while maintaining flexibility for machine-specific paths and sensitive values like API keys" ([Claude Code MCP docs](https://code.claude.com/docs/en/mcp)), so each developer supplies secrets from their own environment. This is CCAR-F knowledge 2.4-K2, whose example is `${GITHUB_TOKEN}`.

| Syntax | Result |
|---|---|
| `${VAR}` | The value of environment variable `VAR` |
| `${VAR:-default}` | `VAR` if it is set, otherwise `default` |

Expansion works in `command`, `args`, `env`, `url` and `headers`. The docs' example, shared safely because it holds only references:

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

A stdio server takes its secret through `env` instead (illustrative entry assembled from the documented fields; an entry with no `type` is a stdio server):

```json
{
  "mcpServers": {
    "airtable": {
      "command": "npx",
      "args": ["-y", "airtable-mcp-server"],
      "env": { "AIRTABLE_API_KEY": "${AIRTABLE_API_KEY}" }
    }
  }
}
```

Two behaviors to know. An unset variable with no default does not stop the config from loading: Claude Code reports a missing-variable warning for that server in `claude mcp list` (and in `/mcp`) and passes the unexpanded `${VAR}` text as-is. The fix is to set the variable or add a `:-default` fallback.

Second, in a remote server's `url` and `headers`, Claude Code reads credential variables as empty rather than expanding them, with no warning, so a project's `.mcp.json` or a plugin cannot send your Claude Code or cloud-provider credentials to a server it names. The docs list the covered names by example: Claude Code's own (`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`), cloud-provider ones (`AWS_BEARER_TOKEN_BEDROCK`) and other credentials your environment carries (`HTTPS_PROXY`, `NPM_TOKEN`). A covered name reads as empty even when it is set, and a `:-default` on it is ignored; a header such as `Bearer ${ANTHROPIC_AUTH_TOKEN}` reaches the server as `Bearer ` and the request is usually rejected with a `401`. A name outside the set, such as `API_KEY`, expands as written. To give a server one of the covered credentials, copy it into a variable with a name of your own and reference that.

!!! note "The guide's `${GITHUB_TOKEN}` example"

    The Claude Code MCP page gives the covered names as examples ("such as"), not as a complete list, and does not mention `GITHUB_TOKEN`, so whether the guide's example variable reads as empty in a remote server's `headers` is not stated. The exam point is unaffected: 2.4-K2 and 2.4-S1 test that credentials reach `.mcp.json` through environment variable expansion instead of being committed. If a remote server receives an empty credential in practice, rename the variable as described above.

### Approval of project servers

Because a repository can ship a `.mcp.json`, Claude Code does not trust it automatically:

- In interactive sessions Claude Code asks for approval before using project-scoped servers from `.mcp.json`; `claude mcp reset-project-choices` resets those choices.
- Settings can decide in advance: `enableAllProjectMcpServers`, `enabledMcpjsonServers`, and `disabledMcpjsonServers`, where a rejection takes precedence over both approval settings.
- "A cloned repository can't approve its own servers" ([Claude Code MCP docs](https://code.claude.com/docs/en/mcp)): approvals committed to the project are ignored until the workspace trust dialog is accepted.
- In `claude -p` runs, Agent SDK sessions and cloud sessions there is no one to ask, so Claude Code loads project-scoped servers without asking. Block unwanted ones with `disabledMcpjsonServers`, `--setting-sources`, or `--strict-mcp-config` (which uses only the servers passed with `--mcp-config`).
- Toggling a server off in `/mcp` is recorded per project in `~/.claude.json`, separately from the `.mcp.json` approval settings.

### Authentication for remote servers

| Need | How |
|---|---|
| OAuth 2.0 sign-in | `/mcp` inside a session, or `claude mcp login <name>` from the shell (v2.1.186+); `claude mcp logout <name>` clears credentials |
| When sign-in is triggered | Claude Code marks a remote server as needing authentication when it answers `401 Unauthorized` or `403 Forbidden`. Exception: if you configured the server's `Authorization` header (in `headers` or through `headersHelper`), a `401` or `403` while connecting is reported as a failed connection instead, because the credential to fix is the one you configured |
| Pre-registered OAuth client | `--client-id`, `--client-secret` (masked prompt or `MCP_CLIENT_SECRET`), and `--callback-port` to match a redirect URI of the form `http://localhost:PORT/callback`; the secret goes to the system keychain or a credentials file, not the config |
| Restrict scopes | `oauth.scopes` pins the requested scopes and overrides discovered ones |
| Dynamic headers | `headersHelper` runs a command at connection time that prints a JSON object of headers; it times out after 10 seconds and overrides static headers with the same name |
| Registration | Client ID Metadata Document servers work automatically; Dynamic Client Registration is supported |

### What Claude sees once servers connect

After a server connects, Claude Code sends discovery requests such as `tools/list`, `prompts/list` and `resources/list`, retrying up to three times on transient errors. Tools from every configured server are then available together, as CCAR-F 2.4-K3 says. What changed is how much of each definition sits in context:

| Setting | Behavior |
|---|---|
| Tool search (default) | Only tool names and server instructions load at session start; full definitions load on demand, so extra servers cost little context |
| `ENABLE_TOOL_SEARCH` | Unset: defer all MCP tools, but load them upfront when `ANTHROPIC_BASE_URL` points to a non-first-party host, on a Microsoft Foundry deployment hosted on Azure, or on Google Cloud Agent Platform models earlier than the Claude 4.5 generation. `true`: defer all MCP tools even behind a proxy, where requests fail if the proxy does not support `tool_reference` blocks (the Foundry and older Agent Platform exceptions still apply). `auto`: load upfront until definitions reach 10% of the context window, then defer. `auto:N`: the same with a custom percentage. `false`: load everything upfront |
| Model requirement | Tool search needs a model that supports `tool_reference` blocks: Claude Sonnet 4.5, Claude Haiku 4.5, Claude Opus 4.5 and later |
| Exempt a server or tool | `"alwaysLoad": true` on the server, or `"anthropic/alwaysLoad": true` in a tool's `_meta` |
| Description length | Tool descriptions and server instructions are truncated at 2,048 characters by default (`CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH` changes it, v2.1.280+) |
| List changes | Claude Code honors `list_changed` notifications and refreshes tools, prompts and resources without reconnecting |

CCAR-F skill 2.4-S3 (richer descriptions so the agent stops preferring built-in tools like Grep over a more capable MCP tool) still applies under tool search, with one addition from the docs. At session start Claude sees only tool names and server instructions, and "Server instructions help Claude understand when to search for your tools" ([Claude Code MCP docs](https://code.claude.com/docs/en/mcp)). The docs ask server authors to explain in those instructions what category of tasks the tools handle, when Claude should search for them, and the server's key capabilities. Descriptions and instructions are each truncated at 2,048 characters by default, so put critical details near the start. Our takeaway: write the description (capabilities and outputs in detail) and the server instructions (when to search) together. The drafting technique is in [Writing tool descriptions that steer selection](#writing-tool-descriptions-that-steer-selection).

!!! warning "Exam guide vs current docs"

    CCAR-F 2.4-K3 says tools from all configured servers "are discovered at connection time and available simultaneously to the agent" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). That is still true of availability. As of September 2026, though, Claude Code defers full tool definitions by default and loads them when needed. Where the discovery cache is turned on (Claude Code v2.1.221 or later; it is off by default unless a gradual rollout has enabled it for the account, and `MCP_DISCOVERY_CACHE=1` turns it on), a remote HTTP or SSE server used in an earlier session can show a `cached` status: its tool list comes from the cache, the tools are available from the first message, and the server itself connects the first time Claude calls one of its tools. For the exam, keep the guide's statement: all configured servers' tools are available to the agent at the same time.

### Resources and prompts in a session

**Resources.** Type `@` to see resources from all connected servers alongside files, and reference one as `@server:protocol://resource/path`, for example `Can you analyze @github:issue://123 and suggest a fix?` ([Claude Code MCP docs](https://code.claude.com/docs/en/mcp)). Referenced resources are fetched and attached automatically, and when servers support resources Claude Code also gives Claude tools to list and read them. That second point is how CCAR-F skill 2.4-S5 plays out: a server that publishes its catalog (issue summaries, a documentation tree, a schema) as resources lets Claude see what exists without exploratory tool calls.

**Prompts.** Each server prompt appears in the `/` menu as `/servername:promptname (MCP)`; typing `/mcp__servername__promptname` also runs it, with arguments separated by spaces (the docs' example is `/mcp__github__pr_review 456`). The prompt's result is injected directly into the conversation.

**Elicitation.** When a server asks the user for input, Claude Code shows a dialog in form or URL mode. An `Elicitation` hook can answer automatically instead.

### Names, permissions and hooks

Every MCP tool is named `mcp__<server>__<tool>` (plugin-bundled servers use `mcp__plugin_<plugin-name>_<server-name>__<tool-name>`). That name is what permission rules and hook matchers see, and the two use different wildcard syntax:

| Target | Permission rule | Hook matcher |
|---|---|---|
| Every tool on one server | `mcp__puppeteer` or `mcp__puppeteer__*` | `mcp__memory__.*` (a regex; bare `mcp__memory` matches no tool) |
| One tool | `mcp__puppeteer__puppeteer_navigate` | `mcp__memory__create_entities` |
| Every MCP tool | `mcp__*` in deny or ask rules only; allow globs need a literal `mcp__<server>__` prefix | `mcp__.*` |

```json
{
  "permissions": { "allow": ["mcp__github__get_*", "mcp__puppeteer__*"] },
  "hooks": {
    "PreToolUse": [
      { "matcher": "mcp__memory__.*", "hooks": [ { "type": "command", "command": "./log-memory-call.sh" } ] }
    ]
  }
}
```

The hook command path is illustrative. A server author can also force a prompt on every call with `_meta["anthropic/requiresUserInteraction"]: true`, which applies even in `bypassPermissions` mode. Permission modes and hooks in full: [Claude Code Configuration](claude-code-configuration.md#permissions-and-permission-modes) and [Hooks](claude-code-workflows.md#hooks).

### Output size, timeouts and reconnection

| Behavior | Default and control |
|---|---|
| Output warning | Shown when a tool result exceeds 10,000 tokens (fixed) |
| Output limit | 25,000 tokens; raise with `MAX_MCP_OUTPUT_TOKENS`. A text result over the limit is saved to a file in the session's `tool-results` directory and replaced by a message naming the path |
| Per-tool threshold | A server can raise one tool's threshold with `_meta["anthropic/maxResultSizeChars"]`, up to 500,000 characters (not for image content) |
| Startup timeout | `MCP_TIMEOUT`, in milliseconds (default 30000) |
| Tool timeout | A per-server `timeout` in milliseconds in `.mcp.json` overrides `MCP_TOOL_TIMEOUT` for that server |
| Idle timeout | A call with no response and no progress notification for the idle window aborts: five minutes for HTTP, SSE, WebSocket and connector servers, 30 minutes for stdio (`CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT`) |
| Long calls | A main-conversation call still running after two minutes moves to a background task (v2.1.212+; `CLAUDE_CODE_MCP_AUTO_BACKGROUND_MS`) |
| Dropped connections | Remote servers reconnect with exponential backoff, up to five attempts starting at one second; stdio servers are not reconnected automatically |

### Other sources of servers

- **claude.ai connectors.** Servers added at claude.ai (on Team and Enterprise, by admins only) appear in Claude Code automatically when you sign in with a claude.ai account. They are not loaded when `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `apiKeyHelper` or a third-party provider is active; turn them off with `disableClaudeAiConnectors: true` or `ENABLE_CLAUDEAI_MCP_SERVERS=false`.
- **Claude Desktop.** `claude mcp add-from-claude-desktop` imports Claude Desktop's servers (macOS and WSL only).
- **Plugins.** A plugin defines servers in `.mcp.json` at its root or inline in `plugin.json`, using `${CLAUDE_PLUGIN_ROOT}` for paths.
- **Organization policy.** `managed-mcp.json` gives administrators exclusive control; `allowedMcpServers` and `deniedMcpServers` filter what users may add, and nothing overrides a denylist match. A `serverName` entry is not a security control, because users choose names; match on `serverCommand` or `serverUrl`. Details: [Managed settings for organizations](claude-code-configuration.md#managed-settings-for-organizations).
- **Claude Code as a server.** `claude mcp serve` exposes Claude Code's own tools over stdio; the connecting client is responsible for confirming individual tool calls.

### Decide and avoid

- If the whole team should get a server when they clone the repository, put it in project-scoped `.mcp.json` and reference credentials as `${VAR}`; not in `~/.claude.json`, which nobody else sees.
- If the server is personal or experimental, keep it in `~/.claude.json` (local scope by default, `--scope user` for every project); not in the committed `.mcp.json`.
- If a secret appears as a literal in `.mcp.json`, that is the defect; the fix is environment variable expansion, not a `.gitignore` entry for a file the team needs.
- If Claude keeps using Grep instead of a more capable MCP search tool, improve that tool's description so it explains capabilities and outputs in detail; this is what skill 2.4-S3 prescribes. Under Claude Code's tool search, also make the server instructions say when to search for the server's tools.
- If the agent makes many calls just to discover what issues, documents or tables exist, expose the catalog as MCP resources.
- If a headless CI run picks up a repository's MCP servers unexpectedly, remember that `claude -p` loads project servers without asking; restrict them explicitly (`disabledMcpjsonServers`, `--setting-sources` or `--strict-mcp-config`).

## MCP in the API and the Agent SDK

*Tested in: CCAR-F 2.4-K3, 1.5-S1, 1.5-S2, APPX-TECH-1 (allowedTools configuration) · CCDV-F D8.2 (integration with Claude applications), D2.3 Claude API Mechanics, D1.2 Agent Construction with Claude · CCAR-P 3.7, 3.8*

Outside Claude Code, an application you build reaches MCP servers through one of three routes covered here: the Messages API's MCP connector (no MCP client in your code; the API connects to the server), MCP client helpers in the Anthropic SDKs (your code is the client), or the Claude Agent SDK (the agent runtime is the client). Claude Managed Agents adds a hosted variant.

### The Messages API MCP connector

The connector lets a Messages API request "connect to remote MCP servers directly from the Messages API without a separate MCP client" ([MCP connector docs](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector)). Facts to know, as of September 2026:

| Item | Value |
|---|---|
| Status and header | Beta; `anthropic-beta: mcp-client-2025-11-20`. A newer `mcp-client-2026-09-15` header (available on the Claude API) includes everything in `2025-11-20` and adds recording and pinning of each server's tool list |
| Deprecated version | `mcp-client-2025-04-04`, which kept tool configuration on the server definition; it now lives in `mcp_toolset` entries in `tools` |
| Platforms | Claude API, Claude Platform on AWS and Microsoft Foundry (all beta); not available on Amazon Bedrock or Google Cloud |
| What is supported | Tool calls only, not resources or prompts |
| Which servers | Publicly exposed HTTP servers (Streamable HTTP or SSE); local stdio servers cannot be connected |
| Data retention | Not covered by ZDR arrangements |
| Batches | `mcp_servers` works in Message Batches requests |
| Several servers | Allowed: one `mcp_servers` entry and one matching toolset per server |

A request has two parts. `mcp_servers` says how to reach each server; an `mcp_toolset` entry in `tools` says which of its tools Claude may use:

| `mcp_servers` field | Required | Notes |
|---|---|---|
| `type` | Yes | Only `"url"` |
| `url` | Yes | Must start with `https://` |
| `name` | Yes | Unique; referenced by exactly one toolset |
| `authorization_token` | No | OAuth bearer token; you obtain and refresh it |

| `mcp_toolset` field | Notes |
|---|---|
| `type` | `"mcp_toolset"` |
| `mcp_server_name` | Must match a server in `mcp_servers` |
| `default_config` | Applies to every tool on the server |
| `configs` | Per-tool overrides keyed by tool name |
| `cache_control` | Optional |
| Per-tool options | `enabled` (default `true`) and `defer_loading` (default `false`, used with tool search) |

Precedence runs from tool-specific `configs`, to the set-level `default_config`, to system defaults. Every server must be used by exactly one toolset, and a tool name in `configs` that the server does not have only logs a backend warning, because servers may change their tool lists.

=== "Python"

    ```python
    import anthropic

    client = anthropic.Anthropic()

    response = client.beta.messages.create(
        model="claude-opus-5-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": "What tools do you have available?"}],
        mcp_servers=[
            {
                "type": "url",
                "url": "https://example-server.modelcontextprotocol.io/sse",
                "name": "example-mcp",
                "authorization_token": "YOUR_TOKEN",
            }
        ],
        tools=[
            {
                "type": "mcp_toolset",
                "mcp_server_name": "example-mcp",
                "default_config": {"enabled": False},
                "configs": {"echo": {"enabled": True}},
            }
        ],
        betas=["mcp-client-2025-11-20"],
    )
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";

    const anthropic = new Anthropic();

    const response = await anthropic.beta.messages.create({
      model: "claude-opus-5-5",
      max_tokens: 1000,
      messages: [{ role: "user", content: "What tools do you have available?" }],
      mcp_servers: [
        {
          type: "url",
          url: "https://example-server.modelcontextprotocol.io/sse",
          name: "example-mcp",
          authorization_token: "YOUR_TOKEN"
        }
      ],
      tools: [
        {
          type: "mcp_toolset",
          mcp_server_name: "example-mcp",
          default_config: { enabled: false },
          configs: { echo: { enabled: true } }
        }
      ],
      betas: ["mcp-client-2025-11-20"]
    });
    ```

The example is the docs' basic request with the docs' allowlist pattern applied: `default_config.enabled: false` turns every tool off, and `configs` turns back on only the tools you name, here `echo`, the tool the docs' examples show on `example-mcp`. Name a tool the server actually has: a name in `configs` that the server lacks only logs a backend warning, so with everything else off Claude would get no tools from that server. The opposite, a denylist, leaves the defaults and disables named tools; the docs recommend denylisting write or destructive tools for read-only assistants or when a human should confirm state changes:

```json
{
  "type": "mcp_toolset",
  "mcp_server_name": "google-calendar-mcp",
  "configs": {
    "delete_all_events": { "enabled": false },
    "share_calendar_publicly": { "enabled": false }
  }
}
```

The response adds two content block types ([MCP connector docs](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector)). The MCP connector counts among the server tools, which Anthropic executes; unlike client tools, your code does not run the call or send a `tool_result` back, and the result arrives in the same response:

```json
{
  "type": "mcp_tool_use",
  "id": "mcptoolu_014Q35RayjACSWkSj4X2yov1",
  "name": "echo",
  "server_name": "example-mcp",
  "input": { "param1": "value1", "param2": "value2" }
}
```

```json
{
  "type": "mcp_tool_result",
  "tool_use_id": "mcptoolu_014Q35RayjACSWkSj4X2yov1",
  "is_error": false,
  "content": [{ "type": "text", "text": "Hello" }]
}
```

Other behaviors to know: Claude calls an MCP tool when the request maps to a tool's described capability; it does not call one for a general knowledge question about the connected service. For an OAuth-protected server, your application runs the OAuth flow, passes the access token as `authorization_token` and refreshes it; the MCP Inspector can obtain a token for testing. And connector tools cannot be called programmatically, and `strict` schema validation does not apply to `mcp_toolset`.

**Connector or client-side helpers?** The docs draw the line: use `mcp_servers` for remote servers where tools are all you need; use the SDK's client-side helpers "when you need local servers, prompts, resources, or more control over the connection" ([MCP connector docs](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector)). In TypeScript the helpers (`mcpTools`, `mcpMessages`, `mcpResourceToContent`, `mcpResourceToFile`) come from `@anthropic-ai/sdk/helpers/beta/mcp`; in Python, install `anthropic[mcp]`.

### Many MCP tools: deferral and progressive discovery

Connecting several servers quickly fills the context. The token numbers, how tool search works, when to use it and when to stay monolithic (CCAR-P 3.8) are covered in [Progressive discovery with tool search](#progressive-discovery-with-tool-search). What is specific to MCP:

- **With the MCP connector,** you do not set `defer_loading` on individual tool definitions; set it on the toolset's `default_config` for the whole server, or per tool in `configs`. Tool search also needs a tool search tool (`tool_search_tool_regex_20251119` or `tool_search_tool_bm25_20251119`) in the same `tools` array, for example `{"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}`. The usual deferral rules apply: the search tool itself is never deferred, and a request with every tool deferred fails with a 400.
- **In the Agent SDK and Claude Code,** tool search is on by default. In the SDK, the agent starts with a summary of available tools and searches when it needs a capability; up to five of the most relevant tools are loaded into context by default. Claude Code loads only tool names and server instructions at session start.

```json
{
  "type": "mcp_toolset",
  "mcp_server_name": "google-calendar-mcp",
  "default_config": { "enabled": false, "defer_loading": true },
  "configs": {
    "search_events": { "enabled": true, "defer_loading": false },
    "list_events": { "enabled": true }
  }
}
```

In that docs example, `search_events` stays loaded, `list_events` is enabled but deferred (it inherits `defer_loading: true`), and every other tool on the server is off.

The [MCP client best practices](https://modelcontextprotocol.io/docs/2026-07-28/develop/clients/client-best-practices) describe the same idea from the host's side: the host fetches tool definitions but defers injecting them, gives the model a lightweight `search_tools` meta-tool, and loads definitions on demand, switching at a threshold such as 1% to 5% of the context window. A further step is programmatic tool calling ("code mode"), where the model writes code that calls tools in a sandbox and only the final result returns to the model. In Anthropic's [Code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) example, presenting MCP servers as code APIs cut token use from 150,000 to 2,000, at the cost of running agent-generated code in a secure environment with sandboxing, resource limits and monitoring.

!!! note "Thresholds differ by source"

    Claude Code's `auto` mode defers once definitions reach 10% of the context window (its default, with the variable unset, defers all MCP tools), while the MCP client best practices give 1% to 5% as an example threshold. Our reading: both are tuning choices for one trade-off, context cost against an extra search round trip, not protocol requirements.

### The Claude Agent SDK

In the Agent SDK, "MCP servers can run as local processes, connect over HTTP, or execute directly within your SDK application" ([Agent SDK MCP docs](https://code.claude.com/docs/en/agent-sdk/mcp)). Pass them in the `mcpServers` option (TypeScript) or `mcp_servers` in `ClaudeAgentOptions` (Python), or load a `.mcp.json` through the `project` setting source. The docs' quickstart connects to the Claude Code documentation server over HTTP:

=== "Python"

    ```python
    import asyncio
    from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

    async def main():
        options = ClaudeAgentOptions(
            mcp_servers={
                "claude-code-docs": {
                    "type": "http",
                    "url": "https://code.claude.com/docs/mcp",
                }
            },
            allowed_tools=["mcp__claude-code-docs__*"],
        )

        async for message in query(
            prompt="Use the docs MCP server to explain what hooks are in Claude Code",
            options=options,
        ):
            if isinstance(message, ResultMessage) and message.subtype == "success":
                print(message.result)

    asyncio.run(main())
    ```

=== "TypeScript"

    ```typescript
    import { query } from "@anthropic-ai/claude-agent-sdk";

    for await (const message of query({
      prompt: "Use the docs MCP server to explain what hooks are in Claude Code",
      options: {
        mcpServers: {
          "claude-code-docs": {
            type: "http",
            url: "https://code.claude.com/docs/mcp"
          }
        },
        allowedTools: ["mcp__claude-code-docs__*"]
      }
    })) {
      if (message.type === "result" && message.subtype === "success") {
        console.log(message.result);
      }
    }
    ```

| Rule | Detail |
|---|---|
| Choosing the transport | A command to run means stdio; a URL means HTTP or SSE; tools you write in your own code mean an in-process SDK MCP server |
| `"http"` in code | The SDKs' `McpHttpServerConfig` declares only `"http"`; `"streamable-http"` is accepted only in JSON config files |
| Permission | MCP tools need explicit permission. Without it, "Claude will see that tools are available but won't be able to call them" |
| Tool names | `mcp__<server-name>__<tool-name>`; allow a whole server with `mcp__github__*`. An unanchored `"*"` or `"mcp__*"` in allowed tools is ignored with a warning |
| Permission modes | `acceptEdits` does not auto-approve MCP tools; `bypassPermissions` does but also disables most other prompts, so prefer `allowedTools` |
| OAuth | The SDK runs no interactive OAuth flow; a server that needs it reports `needs-auth` and the run continues without that server's tools. Complete OAuth in your app and pass the token in `headers` |
| Status | The `system` init message reports each server as `pending`, `connected`, `failed`, `needs-auth` or `disabled` |
| Limits | Connections time out after 30 seconds by default (`MCP_TIMEOUT`); the 25,000-token output limit matches Claude Code |
| Isolation | `strictMcpConfig: true` (TypeScript) or `strict_mcp_config=True` (Python) uses only the servers you pass, ignoring project `.mcp.json`, user settings, plugin-provided servers and claude.ai connectors |

The permission quote is from the [Agent SDK MCP docs](https://code.claude.com/docs/en/agent-sdk/mcp). Check server status before trusting a run: if a server is `failed` or `needs-auth`, the docs warn that Claude can fall back to built-in tools when the server is unavailable. The init message alone is not enough. A server that needs OAuth may still read `pending` when the init message is emitted, so confirm with `mcpServerStatus()` (TypeScript) or `ClaudeSDKClient.get_mcp_status()` (Python). A server with a cached tool list can also read `pending` and connect on its first tool call.

**Your own tools as an in-process server.** Custom Agent SDK tools are MCP tools too: define them with `tool()` (TypeScript) or `@tool` (Python), wrap them with `createSdkMcpServer` or `create_sdk_mcp_server`, and pass the result in `mcpServers`. The server runs inside your application, not as a separate process, and the key you give it becomes the `{server_name}` in `mcp__{server_name}__{tool_name}`. Setting `readOnlyHint: true` lets the tool run in parallel with other read-only tools. The full treatment is in [Custom tools in the SDK](agents-and-agent-sdk.md#custom-tools-in-the-sdk).

**Hooks on MCP tools.** SDK hook matchers use the same `mcp__<server>__<action>` names, where `<server>` is the key in `mcpServers`. A `PostToolUse` hook can normalize data from different MCP tools before the agent reasons about it (CCAR-F 1.5-S1); to replace a tool's output before Claude sees it, set `updatedToolOutput`, which works for any tool (the MCP-only `updatedMCPToolOutput` is deprecated). A `PreToolUse` hook, the guide's "tool call interception", can block a policy-violating call with `permissionDecision: "deny"` (CCAR-F 1.5-S2). See [Hooks in the SDK](agents-and-agent-sdk.md#hooks-in-the-sdk).

### Claude Managed Agents

In Managed Agents, MCP servers are declared when the agent is created (a name and a URL) and authentication is supplied when a session is created, from a vault; the docs say this "keeps secrets out of reusable agent definitions while letting each session authenticate with its own credentials" ([Managed Agents MCP connector](https://platform.claude.com/docs/en/managed-agents/mcp-connector)). MCP toolsets default to the `always_ask` permission policy (the agent toolset defaults to `always_allow`). For a server inside a private network, attach an MCP tunnel (research preview) to the session; how tunnels work is in [Transports on Claude surfaces](#transports-on-claude-surfaces-as-of-september-2026).

### Which surface fits

| Situation | Choose | Why |
|---|---|---|
| A Messages API app needs tools from a public remote server | MCP connector (`mcp_servers` + `mcp_toolset`) | No MCP client code; tool calls only |
| The app needs a local stdio server, or resources and prompts | Anthropic SDK client-side MCP helpers | The connector supports neither |
| An autonomous agent with built-in tools, permissions and hooks | Agent SDK with `mcpServers` and `allowedTools` | The agent loop is the MCP client |
| A Managed Agents session or Messages API app must reach a server inside a private network | An MCP tunnel (research preview) | Outbound-only connection; no inbound ports or public endpoint |
| Dozens of servers or hundreds of tools | Any of the above plus tool search | Accuracy and context cost |

### Decide and avoid

- If an item puts a local stdio server behind the Messages API connector, it is wrong: the connector needs a server publicly exposed over HTTP (Streamable HTTP or SSE). A server in a private network is reached through an MCP tunnel; a local stdio server calls for the SDK's client-side helpers.
- If an item asks the connector to read MCP resources or prompts, it is wrong: tool calls only.
- If an Agent SDK agent can see an MCP tool but never calls it, check permissions first: add it to `allowedTools`; `acceptEdits` will not help.
- If an Agent SDK agent must use an OAuth-protected server, complete OAuth in the application and pass the token in `headers`; the SDK will not open a browser.
- If a read-only assistant should never change state, denylist the write and destructive tools in the toolset.

## Built-in tools, custom tools, Skills or MCP

*Tested in: CCDV-F D8.3 Agentic Customization, Sample 3 · CCAR-F 2.4-S3, 2.4-S4, 2.5-K1 to 2.5-K3, 2.5-S1, 2.5-S2 · CCAR-P 2.5, 3.7*

CCDV-F skill D8.3 is worded as a trade-off: "Tradeoffs among built-in Tools, custom Tools, Skills, and MCPs for selecting and applying the appropriate approach for a given use case" ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). The four options answer different questions: what Claude can already do, what your code does for one application, what Claude knows how to do, and which external systems Claude can reach. Claude Code's feature guide puts the last two in one sentence: "MCP connects Claude to external services. Skills extend what Claude knows, including how to use those services effectively." ([Claude Code features overview](https://code.claude.com/docs/en/features-overview))

### What each option is

| Option | What it adds | Where it runs | Who can reuse it |
|---|---|---|---|
| Built-in tools | Capabilities that ship with the product. In Claude Code and the Agent SDK, the six that CCAR-F Task 2.5 names: Read, Write, Edit, Bash, Grep, Glob. In the API: Anthropic-defined tools, either server tools (`web_search`, `web_fetch`, `code_execution`, `tool_search`) or Anthropic-schema client tools (`bash`, `text_editor`) | Server tools on Anthropic's infrastructure; the rest in your application or on your machine | Anyone using that product |
| Custom tools | Your own function behind a schema: a Messages API client tool, or an Agent SDK tool served by an in-process SDK MCP server | Your application | The application that defines it (sharing one capability across several applications is the CCDV-F Sample 3 case for an MCP server) |
| Skills | Instructions, scripts and resources in one directory with a `SKILL.md` at its root, loaded progressively when relevant | API: in the code execution container, with no network access. Claude Code: on the filesystem, with the machine's network access | API: the workspace. Claude Code: you, the project, or a plugin. The platform overview says custom Skills do not sync across surfaces; the Claude Code docs add a one-way sync of claude.ai account skills into Claude Code (v2.1.273 or later) |
| MCP servers | A standard connection to an external system: tools, resources and prompts | A local process (typically stdio, one client) or a remote server (typically Streamable HTTP, many clients) | Any compatible MCP client: several Claude applications and other hosts |

!!! warning "Exam guide vs current docs: Grep and Glob"

    CCAR-F Task 2.5 and Scenario 4 name built-in Grep and Glob tools. As of September 2026, the [Claude Code tools reference](https://code.claude.com/docs/en/tools-reference) says that on macOS, Linux and WSL, Claude Code leaves Glob and Grep out of the default tool set, "and Claude searches with `find` and `grep` through the Bash tool instead"; `--tools` gives you the tools you list, and naming either one in `--allowedTools` restores both. The guide's selection logic (Grep for file contents, Glob for file paths) still matches the tool descriptions. Answer in the guide's tool names.

The combination matters as much as the choice. The features overview's "Skill + MCP" row reads: "MCP provides the connection; a skill teaches Claude how to use it well", with the example "MCP connects to your database, a skill documents your schema and query patterns" ([Claude Code features overview](https://code.claude.com/docs/en/features-overview)).

### Decision table

| If the need is... | Choose | Not | Why |
|---|---|---|---|
| Search file contents, find files, read and edit code in a repository | Built-in Grep, Glob, Read, Edit | A custom file tool | CCAR-F Task 2.5: the built-ins already cover content search, path matching and targeted edits |
| A public web lookup inside an API application | An Anthropic server tool such as `web_search` | A home-made scraper tool | Server tools run on Anthropic's infrastructure: you add them to `tools` and Anthropic executes them |
| One application calling one internal API, with no reuse planned | A custom tool | An MCP server | Direct API calls: "it works fine for one agent talking to one service" |
| Several Claude applications sharing one internal capability, maintained on its own | An MCP server exposing the operations as tools | Logic in each app's prompt, pasted data, or a built-in tool | CCDV-F Sample 3, answer B |
| A standard integration such as Jira | An existing community MCP server | A custom server | CCAR-F skill 2.4-S4 reserves custom servers for team-specific workflows |
| Production agents in the cloud, on web or mobile, reaching your system | A remote MCP server | A CLI | Remote MCP is "the only configuration that runs across web, mobile, and cloud-hosted agents" |
| Claude Code on a machine with a good CLI (`gh`, `aws`, `gcloud`) | The CLI through Bash | An MCP server for the same thing | "CLI tools are the most context-efficient way to interact with external services" |
| Your conventions, schema knowledge or a repeatable procedure | A Skill | An MCP server | Skills extend what Claude knows |
| Access to a system and the know-how to use it well | MCP plus a Skill | Either alone | The docs' Skill + MCP pattern |
| A rule that must hold every time | A hook | A Skill or a prompt instruction | "If a rule must hold every time, make it a hook rather than a prompt instruction." |
| Collaboration with another team's or vendor's autonomous agent | Agent-to-agent (A2A) | Wrapping that agent as a tool | A2A connects agents; MCP connects each agent to its own tools |

The decision rules in the table are ours, each resting on the source named in its last column. The quotations in the table come, in order, from [Building agents that reach production systems with MCP](https://claude.com/blog/building-agents-that-reach-production-systems-with-mcp) (the first two), the [Claude Code best practices](https://code.claude.com/docs/en/best-practices), and the [Claude Code features overview](https://code.claude.com/docs/en/features-overview).

The rest of CCAR-F Task 2.5 (the Read then Write fallback when Edit finds no unique anchor text, incremental exploration from Grep entry points, tracing usage across wrapper modules) is taught in [The Claude Agent SDK](agents-and-agent-sdk.md#the-claude-agent-sdk) and [Exploring large codebases](context-engineering.md#exploring-large-codebases).

### Choosing an integration mechanism (CCAR-P 3.7)

CCAR-P objective 3.7 asks candidates to "select the appropriate integration mechanism (MCP, API/CLI, agent-to-agent)" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). Anthropic's post on production MCP compares direct API calls, CLIs and MCP, and says "The key distinction is whether there's a common layer between agents and services, and how far that layer reaches" ([Building agents that reach production systems with MCP](https://claude.com/blog/building-agents-that-reach-production-systems-with-mcp)).

| Mechanism | How it works | Fits | Limits |
|---|---|---|---|
| Direct API calls | The agent calls the API itself, from code in a sandbox or through a generic function-calling tool | One agent and one service, or a few integrations that will not be reused | Scales badly: every agent and service pair needs its own auth, descriptions and edge cases, the M×N problem |
| CLI | The agent runs a command-line tool in a shell | Local environments and sandboxed containers; quick, permissive local integrations | No reach where there is no container (mobile, web, hosted clients); auth usually relies on a credential file on disk |
| MCP | A protocol layer standardizes auth, discovery and semantics; one remote server reaches any compatible client | Production agents in the cloud reaching a system behind auth | Cost: a little more upfront investment, repaid by a portable integration |
| Agent-to-agent (A2A) | An open protocol between independent, possibly opaque agents that do not share state, memory or tools | Stateful, multi-turn collaboration with another organization's agent | A well-defined operation that can be called in a stateless, tool-like way fits a tool better; the A2A project describes MCP as agents using capabilities and A2A as agents partnering on tasks |

The same post ([Building agents that reach production systems with MCP](https://claude.com/blog/building-agents-that-reach-production-systems-with-mcp)) expects mature integrations "will ship all three: the API as the foundation, a CLI for local-first environments, and MCP for cloud-based agents." For agents that are all your own, Claude Managed Agents' multiagent orchestration (beta) coordinates several agents inside one session, an in-platform option to weigh against A2A (our comparison): the agents share the sandbox, filesystem and vault credentials, but not tools, MCP servers or context, and the coordinator delegates only one level deep. The A2A project's own summary of the split is "A2A connects the agents to each other; MCP connects each agent to its own tools" ([A2A and MCP](https://github.com/a2aproject/A2A/blob/main/docs/topics/a2a-and-mcp.md)).

### Context cost

Each option loads differently, which is often the deciding factor in an agent with many capabilities:

| Option | What loads, and when | Cost |
|---|---|---|
| `CLAUDE.md` (for comparison) | Full content, every request | Always paid |
| Skills | Name and description on every request (about 100 tokens per Skill, per the Agent Skills overview); instructions, under 5k tokens, when triggered; files as needed | Low until used |
| MCP servers in Claude Code | Tool names and server instructions at session start; full schemas on demand | Low until a tool is used |
| MCP or custom tools with tool search off | All definitions loaded upfront; 50 tools can use 10 to 20K tokens | High, and selection accuracy degrades beyond 30 to 50 tools |
| Hooks | Nothing; they run outside the model | Zero unless the hook returns context |

That is why [Designing a tool set](#designing-a-tool-set) keeps each agent's tools few, and why [MCP in the API and the Agent SDK](#mcp-in-the-api-and-the-agent-sdk) turns on tool search for large catalogs. For prompt reuse across requests (CCAR-P objective 2.5 groups caching, modular prompts and Skills), see [Agent Skills](claude-code-configuration.md#agent-skills) and [Prompt caching](claude-api.md#prompt-caching).

### How the official sample frames it

CCDV-F Sample 3 describes a team whose internal inventory service is exposed as a REST API and must be reusable across several Claude applications and maintained independently. The keyed answer, B, is to build an MCP server that exposes the inventory operations as tools, because "An MCP server exposes reusable tools that multiple Claude applications can share and that can be maintained independently." The guide's rationale for the distractors: "Hard-coding logic into prompts (A) is neither reusable nor maintainable; pasting data (C) gives no live access and wastes context; built-in tools (D) do not automatically reach arbitrary internal APIs" ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). The full item is reproduced on the [CCDV-F exam page](../claude-certified-developer.md#official-sample-questions).

The pattern to recognize (our generalization from the rationale and the docs quoted above): a stem about reuse across several applications and independent maintenance points to a server boundary, which is what MCP provides. A capability needed by one application only points to a custom tool. Know-how about doing something well points to a Skill. A rule that must always hold points to a hook.

### Decide and avoid

- Built-in tools do not reach arbitrary internal APIs; an option that claims they do is the Sample 3 distractor.
- On the API, a Skill is not a live connection: it runs with no network access, so it cannot fetch current data; pair it with a tool or an MCP server. In Claude Code a Skill has the machine's network access, but the docs' division of labor still holds: MCP connects, a Skill teaches how to use the connection.
- A Skill or an MCP server is not an enforcement mechanism; a hook is.
- Custom Skills uploaded in claude.ai are not available through the API, and API uploads do not reach claude.ai or Claude Code; plan to upload per surface. The one bridge, per the Claude Code docs: a Claude Code session signed in with a claude.ai account syncs that account's skills one way (v2.1.273 or later).
- Pasting reference data into every request is neither live nor cheap; expose it through a tool, or as MCP resources if it is a catalog.
- When Claude prefers a built-in such as Grep over a more capable MCP tool, fix the MCP tool's description before anything else (CCAR-F skill 2.4-S3).

## Exam map

*Tested in: CCAO-F, CCDV-F, CCAR-F and CCAR-P, as mapped below from the four July 2026 exam guides*

Each row lists the objectives and official sample questions a section serves. The mapping is ours, made from the objective wording in the four guides; a cell reading None means we mapped no objective from that guide to the section. Tools and MCP form a whole domain in two exams: CCAR-F Domain 2, Tool Design & MCP Integration, is 18% of the exam, and CCDV-F Domain 8, Tools and MCPs, is 10.6%.

| Section | CCAO-F | CCDV-F | CCAR-F | CCAR-P |
|---|---|---|---|---|
| [How tool use works](#how-tool-use-works) | None | D8.1, D1.2, D1.3, D2.3 | 1.1-K1 to 1.1-S3, EX1-STEP2, APPX-TECH-5, APPX-INSCOPE-1 | None |
| [Defining a tool](#defining-a-tool) | None | D8.1, D2.3, D6.3 | 2.1-K2, 4.3-K1, 4.3-K3, 4.3-S1, 4.4-K4, EX3-STEP1, APPX-TECH-5, APPX-TECH-7, APPX-INSCOPE-13 | None |
| [Writing tool descriptions that steer selection](#writing-tool-descriptions-that-steer-selection) | None | D8.1 | 2.1-K1 to 2.1-S4, 2.4-S3, EX1-STEP1, APPX-INSCOPE-4, Q2 | None |
| [Controlling tool choice](#controlling-tool-choice) | None | D2.3, D8.1, D5.3, D5.4 | 2.3-K4, 2.3-S4, 2.3-S5, 4.3-K2, 4.3-S2, 4.3-S3, 1.4-S1 (contrast with an enforced order), APPX-TECH-5, APPX-INSCOPE-13, Q1 | None |
| [Parallel tool calls](#parallel-tool-calls) | None | D8.1, D2.3 | 1.3-S3, EX4-STEP2 | None |
| [Returning tool results and errors](#returning-tool-results-and-errors) | None | D8.1, D4.1, D7.1 | 1.1-K2, 1.1-S2, 2.2-K1 to 2.2-S4, 5.1-K3, 5.1-S3, 5.3-K2, 5.3-K4, 5.3-S2, APPX-INSCOPE-7, EX1-STEP3, Q8 | None |
| [Anthropic-defined tools](#anthropic-defined-tools) | None | D8.1, D2.3 | 4.5-K3, APPX-TECH-6; computer use is out of scope (APPX-OUTSCOPE-8) | None |
| [Designing a tool set](#designing-a-tool-set) | None | D8.1, D7.2 | 2.1-S3, 2.3-K1, 2.3-K2, 2.3-K3, 2.3-S1, 2.3-S2, 2.3-S3, APPX-INSCOPE-4, Q9 | 3.1, 3.8, 5.1, Sample 1 |
| [MCP architecture](#mcp-architecture) | D5.2 | D8.2 | 2.4-K3, APPX-TECH-2, APPX-INSCOPE-6 | 3.7 |
| [MCP primitives](#mcp-primitives) | None | D8.2 | 2.4-K4, 2.4-S5, APPX-TECH-2, APPX-INSCOPE-5 | 3.7 |
| [MCP transports](#mcp-transports) | None | D8.2 | Background only; hosting MCP servers is out of scope (APPX-OUTSCOPE-4) | 3.2, 3.7 |
| [MCP errors and structured results](#mcp-errors-and-structured-results) | None | D8.1, D4.1 | 2.2-K1 to 2.2-S4, APPX-TECH-2, APPX-INSCOPE-7, EX1-STEP3, Q8 | None |
| [Building an MCP server](#building-an-mcp-server) | None | D8.2, Sample 3 | 2.4-S4, EX1-STEP1, APPX-OUTSCOPE-4 | 3.7 |
| [MCP in Claude Code](#mcp-in-claude-code) | None | D8.2, D7.4 | 2.4-K1 to 2.4-K4, 2.4-S1, 2.4-S2, 2.4-S3, 2.4-S5, APPX-TECH-2, APPX-INSCOPE-6, EX2-STEP4 | 7.1 |
| [MCP in the API and the Agent SDK](#mcp-in-the-api-and-the-agent-sdk) | None | D8.2, D2.3, D1.2 | 2.4-K3, 1.5-S1, 1.5-S2, APPX-TECH-1 | 3.7, 3.8 |
| [Built-in tools, custom tools, Skills or MCP](#built-in-tools-custom-tools-skills-or-mcp) | None | D8.3, Sample 3 | 2.4-S3, 2.4-S4, 2.5-K1 to 2.5-K3, 2.5-S1, 2.5-S2 | 2.5, 3.7 |

### How to read the IDs

The CCAO-F and CCAR-P guides list objectives as unnumbered bullets under numbered domains; the CCDV-F guide lists named, weighted skills under each numbered domain without numbering them; CCAR-F numbers its task statements (such as 2.4) but not the bullets under them. The objective IDs on these pages are therefore our labels, numbered in the order each guide lists them.

- **CCAR-F** ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)): `2.4-K1` is Task Statement 2.4, first "Knowledge of" bullet; `S` marks a "Skills in" bullet. `APPX-TECH`, `APPX-INSCOPE` and `APPX-OUTSCOPE` are the appendix lists of technologies, in-scope topics and out-of-scope topics. `EX1-STEP3` is Preparation Exercise 1, step 3. `Q2` is the guide's sample Question 2.
- **CCDV-F** (skill and weight): D1.2 Agent Construction with Claude (5.3%), D1.3 Agent Patterns and Frameworks (4.9%), D2.3 Claude API Mechanics (6.8%), D4.1 Debugging and Error Handling (2.6%), D5.3 Model Selection and Tradeoffs (2.7%), D5.4 Cost and Token Management (2.8%), D6.3 Output Handling (2.6%), D7.1 AI Application Security (3.2%), D7.2 Guardrails and Safe Deployment (2.3%), D7.4 Identity, Secrets, and Key Management (1.6%), D8.1 Tool Implementation (4.4%), D8.2 MCP Server Development (2.1%), D8.3 Agentic Customization (4.1%). Sample 3 is the guide's Domain 8 sample question.
- **CCAR-P** (objective): 2.5 prompt reuse strategies (caching, modular prompts, Skills); 3.1 tool and agent configuration for capability bloat; 3.2 authentication and authorization gaps; 3.7 choosing an integration mechanism (MCP, API/CLI, agent-to-agent); 3.8 progressive discovery vs. monolithic context; 5.1 guardrails and safety controls; 7.1 configuring Claude tools and environments for teams. Sample 1 is the guide's least-privilege sample in Domain 3.
- **CCAO-F** ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): D5.2 is "Manage uploaded knowledge and connectors (e.g., Google Drive, Gmail)", the only CCAO-F objective we map to this page.

The exam pages teach each objective in exam framing and link back here: [CCAR-F Domain 2](../claude-certified-architect-foundations.md#domain-2-tool-design-mcp-integration), [CCDV-F Domain 8](../claude-certified-developer.md#domain-8-tools-and-mcps), [CCAR-P Domain 3](../claude-certified-architect-professional.md#domain-3-integration) and [CCAO-F Domain 5](../claude-certified-associate.md#domain-5-configuration-and-knowledge-management).

??? info "Sources"

    - [Claude Certified Architect, Foundations exam guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): CCAR-F Domain 2 weight, objectives 1.1, 1.3-S3, 1.4-S1, 1.5-S1, 1.5-S2, 2.1 to 2.5, 4.3, 4.4, 4.5-K3, 5.1 and 5.3, the appendix technology, in-scope and out-of-scope lists, preparation exercises 1 to 4, and sample questions 1, 2, 8 and 9 with their rationales
    - [Claude Certified Developer, Foundations exam guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf): CCDV-F Domain 8 weight, skills D1.2, D1.3, D2.3, D4.1, D5.3, D5.4, D6.3, D7.1, D7.2, D7.4 and D8.1 to D8.3 with their weights and descriptions, and Sample 3 with its rationale
    - [Claude Certified Architect, Professional exam guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf): CCAR-P objectives 2.5, 3.1, 3.2, 3.7, 3.8, 5.1 and 7.1, and Sample 1 on least privilege with its rationale
    - [How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works): the tool-use contract, the three execution buckets, the client-tool loop, the server-side loop, and when to use tools
    - [Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview): when Claude calls a tool, client tools versus server tools, system-prompt steering lines, missing parameters, pricing, and tool-use system prompt token counts per model
    - [Define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools): tool fields and name regex, the tool-use system prompt template, description best practices and the get_stock_price example, input_examples, tool_choice values and forced tool use restrictions
    - [Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls): tool_use and tool_result fields, ordering rules, is_error, instructive error messages, retries on invalid calls, untrusted content in tool results
    - [Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use): execution semantics, returning all results in one message, disable_parallel_tool_use, prompts for parallel calls, toolset batch rules, model notes
    - [Troubleshooting tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/troubleshooting-tool-use): wrong-tool and never-called fixes, enum and type fixes, the missing tool_result error, raw string matching, disable_parallel_tool_use timing
    - [Tool reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference): Anthropic tool types, versions and beta headers, optional tool properties, toolset restrictions, `strict` not applying to `mcp_toolset`
    - [Strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use): grammar-constrained sampling, guarantees, placement of strict, HIPAA note, toolset rejection
    - [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs): strict schema rules, per-request complexity limits, grammar caching, enum capitalization and property ordering
    - [Extracting Structured JSON using Claude and Tool Use (Anthropic cookbook)](https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/extracting_structured_json.ipynb): extraction tools such as print_summary and print_entities, and an open-ended input schema
    - [Fine-grained tool streaming](https://platform.claude.com/docs/en/agents-and-tools/tool-use/fine-grained-tool-streaming): eager_input_streaming, the replaced beta header, invalid JSON handling
    - [Streaming messages](https://platform.claude.com/docs/en/build-with-claude/streaming): input_json_delta and partial_json for tool input
    - [Tool Runner](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner): the SDK tool runner, @beta_tool and betaZodTool examples, max_iterations, error handling, when to use the manual loop
    - [Build a tool-using agent](https://platform.claude.com/docs/en/agents-and-tools/tool-use/build-a-tool-using-agent): the error-handling agent loop in Python and TypeScript
    - [Server tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools): server_tool_use blocks, pause_turn handling, mixed server and client turns, domain filters, ZDR eligibility
    - [Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool): versions, max_uses, citations, encrypted_content, empty results, pricing, platform availability, error responses
    - [Web fetch tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool): versions, URL rules, options, cost, exfiltration risk, platform availability
    - [Code execution tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool): versions, runtime, containers, pricing, platform availability, environment confusion with client bash
    - [Bash tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool): definition, inputs, version, allowlist and isolation guidance, result truncation
    - [Text editor tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool): tool name and versions, commands, removed undo_edit, safe replacement guidance
    - [Memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool): configuration, commands, path validation, memory protocol, model support
    - [Computer use tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool): the computer toolset, older versions and beta headers, dispatch on toolset_name, batch actions, token cost, precautions
    - [Browser use tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/browser-use-tool): member tool counts and the batch halt text
    - [Advisor tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool): purpose, type, name and beta header
    - [Tool search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool): deferred loading rules, variants, limits, the multi-server token example and 85 percent reduction, caching, custom search, `defer_loading` with the MCP connector, when to use it
    - [Programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling): allowed_callers, the caller field, continuation and timeouts, incompatibilities (including MCP connector tools), measured savings, fit
    - [MCP connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector): beta headers, platform availability, limitations, `mcp_servers` and `mcp_toolset` fields, allowlist and denylist patterns, response blocks, OAuth handling, client-side helpers, ZDR status
    - [Manage tool context](https://platform.claude.com/docs/en/agents-and-tools/tool-use/manage-tool-context): the four context levers and their suggested order, cache write markup
    - [Tool use with prompt caching](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching): caching tool definitions and what invalidates the cache
    - [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): tool_choice changes and cache breakpoints, max_tokens 0 restrictions
    - [Handling stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons): the server-side iteration limit, pause_turn, text after tool results, truncated tool_use blocks
    - [Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking): forced tool_choice with manual and adaptive thinking, passing thinking blocks back, always-on thinking on the newest models
    - [Extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking): models that support only extended thinking, and rejection on Claude 4.7 and later
    - [Models overview](https://platform.claude.com/docs/en/models/overview): extended thinking deprecation on the 4.6 models
    - [API errors](https://platform.claude.com/docs/en/api/errors): the forced tool_choice error message and the computer tool type rejection on Claude Opus 5.5
    - [Release notes](https://platform.claude.com/docs/en/release-notes/overview): dates of the forced tool_choice change and the Claude Opus 5.5 launch, memory tool beta header removal
    - [Claude Opus 5.5 migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide): replacing a forced tool_choice with strict tools, auto and an instruction in the user message
    - [Batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing): server tools (including MCP connectors) in batches, batch loop iterations, continuing paused batch results
    - [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing): clear_tool_uses_20250919, its beta header and defaults
    - [Compaction on demand](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand): forced tool_choice not allowed in compaction requests
    - [Mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages): tool_addition and tool_removal without changing the tools array
    - [Search results](https://platform.claude.com/docs/en/build-with-claude/search-results): returning a text block when a search finds nothing
    - [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices): overtriggering on newer models, action-oriented phrasing, the parallel tool calls prompt
    - [Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence): stale prompts that add tool rounds, and prompt audits on model changes
    - [Mitigate jailbreaks and prompt injections](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks): third-party content in tool_result blocks, no instructions in tool results, JSON encoding, output screening, least privilege
    - [Agent SDK: the agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop): concurrent read-only tools, sequential custom tools, denied tool results
    - [Agent SDK: custom tools](https://code.claude.com/docs/en/agent-sdk/custom-tools): in-process SDK MCP servers, handler return fields, readOnlyHint and other annotations, error handling with isError and is_error, `structuredContent` forwarding, mcp__ tool names, tool search default
    - [Agent SDK: permissions](https://code.claude.com/docs/en/agent-sdk/permissions): evaluation order, permission modes, deny rules, bypassPermissions and allowed_tools, unanchored allowed-tool wildcards being ignored, locked-down agents
    - [Agent SDK: user input and approvals](https://code.claude.com/docs/en/agent-sdk/user-input): canUseTool behavior and approval response patterns
    - [Agent SDK: subagents](https://code.claude.com/docs/en/agent-sdk/subagents): AgentDefinition tools, omitted tools, common tool combinations
    - [Agent SDK: Python reference](https://code.claude.com/docs/en/agent-sdk/python): the Agent tool name and the Task alias, `strict_mcp_config`
    - [Claude Code: MCP](https://code.claude.com/docs/en/mcp): CLI commands and flags, scopes and precedence, environment variable expansion, credential variables, project approval, OAuth options, tool search and discovery at session start, description truncation at 2,048 characters, resources, prompts, elicitation, output limits, timeouts, reconnection, claude.ai connectors, client runtimes, reserved names
    - [Claude Code: subagents](https://code.claude.com/docs/en/sub-agents): the rename of the Task tool to Agent in v2.1.63
    - [Claude Code glossary](https://code.claude.com/docs/en/glossary): the harness definition
    - [MCP specification 2026-07-28: tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools): tool fields, naming, annotations trust, isError for tool execution errors, input validation errors, structured content, stateful handles, human-in-the-loop expectations
    - [MCP specification 2026-07-28: resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources): URIs, templates, contents, annotations, no empty contents array for a missing resource
    - [MCP architecture 2026-07-28](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture): data and transport layers, method families, tools, resources and prompts as server primitives, unified tool registry, transport fan-out, worked database example
    - [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents): tools as a contract, consolidation examples, namespacing, response shaping, error responses, tool evaluation, the 25,000-token default
    - [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): agent-computer interfaces, good tool definitions, poka-yoke, stopping conditions
    - [Introducing advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use): common tool failures, tool definition token counts, tool search and tool use example results
    - [Anthropic engineering: the think tool](https://www.anthropic.com/engineering/claude-think-tool): the think tool definition and the update recommending extended thinking instead
    - [Anthropic engineering: Managed Agents](https://www.anthropic.com/engineering/managed-agents): the harness routing tool calls and the execute(name, input) interface
    - [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system): tool description rewrites, parallelism results
    - [Agent harness design](https://claude.com/blog/harnessing-claudes-intelligence): harness definition, stateless re-packaging, dedicated tools as approval hooks, reversibility, tools in the cached prefix
    - [Building agents that reach production systems with MCP](https://claude.com/blog/building-agents-that-reach-production-systems-with-mcp): fewer, well-described tools, thin code-accepting tool surfaces, direct API versus CLI versus MCP, MCP as a common protocol layer, remote MCP reach
    - [Building multi-agent systems: when and how to use them](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them): when multiple agents win, tool counts as a specialization signal, tool search before splitting
    - [Claude Certified Associate, Foundations: Exam Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf): objective D5.2 on connectors
    - [Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol): open-sourcing date and description of MCP, Claude Desktop local server support at launch
    - [Agent Skills overview (Claude API docs)](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview): progressive disclosure costs, sharing scope and runtime by surface, Skills not syncing across surfaces
    - [Skills guide (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/skills-guide): Skills running in the code execution container without network access
    - [Skills API: create a Skill](https://platform.claude.com/docs/en/api/skills/create): Skill directory structure with `SKILL.md` at the root
    - [Managed Agents MCP connector](https://platform.claude.com/docs/en/managed-agents/mcp-connector): servers declared at agent creation, credentials from a vault per session
    - [Managed Agents permission policies](https://platform.claude.com/docs/en/managed-agents/permission-policies): MCP toolsets defaulting to `always_ask`
    - [Managed Agents multiagent orchestration](https://platform.claude.com/docs/en/managed-agents/multiagent): shared sandbox and credentials, separate tools and MCP servers, one level of delegation
    - [MCP tunnels overview](https://platform.claude.com/docs/en/agents-and-tools/mcp-tunnels/overview): outbound-only connection to private-network servers
    - [MCP quickstart (Claude Code docs)](https://code.claude.com/docs/en/mcp-quickstart): where local and user scope entries are stored in `~/.claude.json`
    - [Settings reference (Claude Code docs)](https://code.claude.com/docs/en/settings-reference): `enableAllProjectMcpServers`, `enabledMcpjsonServers`, `disabledMcpjsonServers`
    - [Managed MCP configuration (Claude Code docs)](https://code.claude.com/docs/en/managed-mcp): `managed-mcp.json`, allowlists and denylists, `serverName` not being a security control
    - [Hooks reference (Claude Code docs)](https://code.claude.com/docs/en/hooks): `mcp__<server>__<tool>` naming and hook matchers for MCP tools
    - [Permissions (Claude Code docs)](https://code.claude.com/docs/en/permissions): permission rule syntax for MCP servers and tools
    - [Permission modes (Claude Code docs)](https://code.claude.com/docs/en/permission-modes): `requiresUserInteraction` tools never auto-approved
    - [Commands (Claude Code docs)](https://code.claude.com/docs/en/commands): `/mcp` and MCP prompts as commands
    - [Debug your configuration (Claude Code docs)](https://code.claude.com/docs/en/debug-your-config): `settings.json` not reading an `mcpServers` key
    - [Environment variables (Claude Code docs)](https://code.claude.com/docs/en/env-vars): `MCP_TIMEOUT` default, `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH`
    - [Security (Claude Code docs)](https://code.claude.com/docs/en/security): Anthropic reviews directory listings but does not security-audit MCP servers
    - [Features overview (Claude Code docs)](https://code.claude.com/docs/en/features-overview): MCP versus Skills, the Skill + MCP pattern, hooks for rules that must always hold, context cost by feature
    - [Best practices (Claude Code docs)](https://code.claude.com/docs/en/best-practices): CLI tools as the most context-efficient route to external services
    - [Tools reference (Claude Code docs)](https://code.claude.com/docs/en/tools-reference): Glob and Grep left out of the default tool set on macOS, Linux and WSL, and how `--tools` and `--allowedTools` bring them back
    - [Connect to external tools with MCP (Agent SDK docs)](https://code.claude.com/docs/en/agent-sdk/mcp): configuration options, transport choice, permissions, OAuth, status values, timeouts, output limit
    - [Tool search (Agent SDK docs)](https://code.claude.com/docs/en/agent-sdk/tool-search): default tool search, tool count and token guidance
    - [Agent SDK hooks](https://code.claude.com/docs/en/agent-sdk/hooks): MCP tool matchers and `PostToolUse` output replacement
    - [Get started with custom connectors using remote MCP (Claude Help Center)](https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp): plan availability, connections from Anthropic's cloud, Owner-only setup on Team and Enterprise, local servers not available in claude.ai or Cowork
    - [Use connectors to extend Claude's capabilities (Claude Help Center)](https://support.claude.com/en/articles/11176164-use-connectors-to-extend-claude-s-capabilities): connectors across surfaces, permission inheritance, organization-level tool permissions
    - [Getting started with local MCP servers on Claude Desktop (Claude Help Center)](https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop): sensitive extension fields in OS secure storage
    - [Remote MCP custom connectors (Claude docs)](https://claude.com/docs/connectors/custom/remote-mcp): `/sse` URLs selecting the SSE transport
    - [Desktop extensions, MCPB (Claude docs)](https://claude.com/docs/connectors/building/mcpb): `.mcpb` format, stdio and local characteristics, CLI commands, directory submission requirements
    - [Claude Desktop Extensions (Anthropic engineering)](https://www.anthropic.com/engineering/desktop-extensions): `.dxt` to `.mcpb` rename, `manifest.json`, sensitive `user_config`
    - [Code execution with MCP (Anthropic engineering)](https://www.anthropic.com/engineering/code-execution-with-mcp): 150,000 to 2,000 token example, sandboxing overhead
    - [MCP specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28): hosts, clients, servers, primitives, security principles
    - [MCP architecture (specification)](https://modelcontextprotocol.io/specification/2026-07-28/architecture): one client per server, host responsibilities, design principles
    - [MCP base protocol (specification)](https://modelcontextprotocol.io/specification/2026-07-28/basic): JSON-RPC rules, `_meta` fields, error codes, `resultType`
    - [MCP versioning (specification)](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning): Legacy, Modern and Dual-era, version negotiation
    - [MCP versioning (docs)](https://modelcontextprotocol.io/docs/2026-07-28/learn/versioning): 2026-07-28 as the current protocol version
    - [MCP lifecycle, 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle): `initialize` handshake, capabilities, shutdown
    - [MCP transports, 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports): sessions, GET streams, resumability, default version header behavior
    - [MCP transports overview (specification)](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports): standard and custom transports
    - [stdio transport (specification)](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio): framing, stdout and stderr rules, restarts, dual-era probing, socket reuse of stdio framing
    - [Streamable HTTP transport (specification)](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http): POST rules, headers, Origin validation, removed sessions, legacy detection
    - [MCP authorization (specification)](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization): OAuth 2.1 roles, metadata discovery, client registration, token rules, status codes
    - [Prompts (specification)](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts): user control, `prompts/get`, errors
    - [MCP schema (specification)](https://modelcontextprotocol.io/specification/2026-07-28/schema): `isError` rationale, annotation defaults and meanings
    - [MCP changelog 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/changelog): statelessness, MRTR, subscriptions, deprecations, deterministic tool order, schema loosening
    - [MCP changelog 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/changelog): input validation errors as tool execution errors
    - [Deprecated features (specification)](https://modelcontextprotocol.io/specification/2026-07-28/deprecated): Roots, Sampling, Logging and their earliest removal
    - [Multi Round-Trip Requests (specification)](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr): `InputRequiredResult`, `requestState`, retry rules
    - [Elicitation (specification)](https://modelcontextprotocol.io/specification/2026-07-28/client/elicitation): form and URL modes, sensitive data rule, response actions
    - [Sampling (specification)](https://modelcontextprotocol.io/specification/2026-07-28/client/sampling): purpose and deprecation
    - [Roots (specification)](https://modelcontextprotocol.io/specification/2026-07-28/client/roots): `file://` roots, deprecation
    - [Server discovery (specification)](https://modelcontextprotocol.io/specification/2026-07-28/server/discover): `server/discover` and `instructions`
    - [Progress, cancellation, pagination, completion, logging and caching (specification)](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/pagination): utility rules, with the sibling pages for progress, cancellation, completion, logging and caching
    - [Server concepts (docs)](https://modelcontextprotocol.io/docs/2026-07-28/learn/server-concepts): who controls tools, resources and prompts, examples
    - [Client concepts (docs)](https://modelcontextprotocol.io/docs/2026-07-28/learn/client-concepts): roots as a coordination mechanism, not a security boundary
    - [Build an MCP server (docs)](https://modelcontextprotocol.io/docs/2026-07-28/develop/build-server): `MCPServer`, stdio entry point, stdout rules, Node.js requirement, Claude Desktop configuration
    - [Build an MCP server, 2025-11-25 version (docs)](https://modelcontextprotocol.io/docs/2025-11-25/develop/build-server): `FastMCP` and `@modelcontextprotocol/sdk` names
    - [MCP SDKs (docs)](https://modelcontextprotocol.io/docs/2026-07-28/sdk): SDK tiers and capabilities
    - [MCP Inspector (docs)](https://modelcontextprotocol.io/docs/2026-07-28/tools/inspector): web, CLI and TUI clients, Node requirement, commands
    - [Debugging (docs)](https://modelcontextprotocol.io/docs/2026-07-28/tools/debugging): absolute paths, environment inheritance, stderr capture
    - [Connect to local MCP servers (docs)](https://modelcontextprotocol.io/docs/2026-07-28/develop/connect-local-servers): `claude_desktop_config.json` locations and logs
    - [MCP client best practices (docs)](https://modelcontextprotocol.io/docs/2026-07-28/develop/clients/client-best-practices): progressive tool discovery, thresholds, code mode
    - [Security best practices (docs)](https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices): local server risks, token passthrough, scope minimization, handle hijacking
    - [MCP Registry](https://modelcontextprotocol.io/registry/about): official metadata registry, preview status, no private servers, no code scanning
    - [MCP Python SDK README](https://github.com/modelcontextprotocol/python-sdk): v2 status, install, 15-line server, type hints as schema, `mcp dev` and `mcp run`
    - [MCP TypeScript SDK README](https://github.com/modelcontextprotocol/typescript-sdk): v2 packages, Standard Schema, minimal stdio server, v1 support window
    - [MCP reference servers README](https://github.com/modelcontextprotocol/servers): reference servers as educational examples, not production-ready
    - [A2A and MCP (A2A project documentation)](https://github.com/a2aproject/A2A/blob/main/docs/topics/a2a-and-mcp.md): agent-to-agent versus agent-to-tool framing
    - [A2A specification](https://github.com/a2aproject/A2A/blob/main/docs/specification.md): A2A as an open protocol between independent, opaque agents
