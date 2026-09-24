---
title: "Context Engineering and Long-Running Work for the Claude Certifications"
description: Context budgets, compaction, context editing, memory, subagent isolation, codebase exploration, handoffs and provenance, taught for all four Claude exams.
last_reviewed: 2026-09-23
---

# Context Engineering and Long-Running Work

Everything Claude reads competes for the same limited attention, so long conversations, long-running agent tasks and multi-agent pipelines fail in predictable ways when nobody decides what stays in the context window. This page teaches those decisions: what fills the window, how critical facts survive summarization, how compaction, context editing and memory work on the Claude API and in Claude Code, how subagents isolate noisy work, how to explore a large codebase without flooding the window, what must cross each handoff between agents, and how findings keep their sources through synthesis. It is the depth reference for Domain 5 of the Architect, Foundations exam (weighted 15%), the Context Engineering skill of the Developer exam (3.8%), the context objectives of the Architect, Professional exam, and the Associate objective on context limits and memory; each section opens with the objectives it serves, and the [Exam map](#exam-map) collects them.

## Why context is a budget

*Tested in: CCAR-F 5.1 (5.1-K3, 5.1-K4), 1.6-S2, Appendix technology list ("Context window management"), sample question 12 · CCDV-F D6.1 Context Engineering, D5.1 LLM Fundamentals (context windows), D5.4 Cost and Token Management · CCAR-P 2.4, 3.8, 4.5, sample 2 · CCAO-F D3.4*

Anthropic's engineering post [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) defines context as "the set of tokens included when sampling from a large-language model (LLM)" and context engineering as "the set of strategies for curating and maintaining the optimal set of tokens (information) during LLM inference, including all the other information that may land there outside of the prompts." The API docs define the window itself as "all the text a language model can reference when generating a response, including the response itself" ([Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows)).

### More tokens, less attention

Three documented effects explain why the window is treated as a budget to spend, not a bucket to fill. The quotations come from the [engineering post](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) unless marked otherwise:

- **Context rot.** "as the number of tokens in the context window increases, the model’s ability to accurately recall information from that context decreases." The API's [Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows) page uses the same name: "As token count grows, accuracy and recall degrade, a phenomenon known as *context rot*."
- **An attention budget.** "LLMs have an “attention budget” that they draw on when parsing large volumes of context. Every new token introduced depletes this budget by some amount". The architectural reason given is that every token can attend to every other token, which "results in n² pairwise relationships for n tokens."
- **A gradient, not a cliff.** "These factors create a performance gradient rather than a hard cliff: models remain highly capable at longer contexts but may show reduced precision for information retrieval and long-range reasoning".

The post adds that some models degrade more gently than others, but "this characteristic emerges across all models", and draws its conclusion: "Context, therefore, must be treated as a finite resource with diminishing marginal returns." Its guiding principle is "good context engineering means finding the smallest possible set of high-signal tokens that maximize the likelihood of some desired outcome." Smallest is not the same as shortest: "minimal does not necessarily mean short; you still need to give the agent sufficient information up front to ensure it adheres to the desired behavior."

Claude Code's documentation states the same constraint in working terms: "Most best practices are based on one constraint: Claude's context window fills up fast, and performance degrades as it fills." When the window is getting full, the same page warns, Claude may start "forgetting" earlier instructions or making more mistakes ([Best practices for Claude Code](https://code.claude.com/docs/en/best-practices)).

### A bigger window is not the fix

The Architect, Foundations exam tests this directly. In [sample question 12](../claude-certified-architect-foundations.md#official-sample-questions) a single-pass review of a 14-file pull request gives detailed feedback on some files, superficial comments on others, missed bugs and contradictory verdicts. Option C, "Switch to a higher-tier model with a larger context window to give all 14 files adequate attention in one pass", is wrong, and Anthropic's rationale says why: "Option C misunderstands that larger context windows don't solve attention quality issues". Anthropic's [context engineering cookbook](https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools) gives the mechanism: "Context rot and prefill latency scale with how much is in the window, not with the window's limit". The [engineering post](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) expects that "for the foreseeable future, context windows of all sizes will be subject to context pollution and information relevance concerns", at least "for situations where the strongest agent performance is desired".

**Decide (our rule, from the sample question 12 rationale).** If an item describes missed details, uneven depth or contradictions on a large input, choose the option that shrinks, splits or restructures what is in the window: focused passes (CCAR-F 1.6-S2 splits reviews "to avoid attention dilution"), trimmed tool output, subagents or compaction. Do not choose an option that keeps the same input in one pass and only adds window size, even when it also offers a higher-tier model as option C does: the rationale says larger windows "don't solve attention quality issues", and the engineering post says the degradation "emerges across all models".

### Where the tokens go

The first four rows come from the API's [Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows) page (as of September 2026), the fifth from [Using the Messages API](https://platform.claude.com/docs/en/build-with-claude/working-with-messages), and the last from Claude Code's [Extend Claude Code](https://code.claude.com/docs/en/features-overview) and the Agent SDK's [How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) pages.

| What occupies the window | What to know |
|---|---|
| System prompt, every message (tool results, images and documents included) and tool definitions | "Everything in the request counts toward the context window" |
| The response, including extended thinking | The window includes "the response itself"; "The output Claude generates for the turn, including its extended thinking, counts too." |
| Cached prompt prefixes | Caching "changes what you pay for those tokens, not whether they count"; `input_tokens`, `cache_read_input_tokens` and `cache_creation_input_tokens` all count toward the window |
| Thinking blocks from earlier turns | Kept by default, and counted like other input, on Opus 4.5 and later Opus models, Sonnet 4.6 and later Sonnet models, Fable 5.1, Mythos 5.1, Fable 5, Mythos 5 and Mythos Preview; stripped automatically on earlier Opus and Sonnet models and all Haiku models. [Thinking block clearing](#context-editing-clearing-tool-results-and-thinking) (context editing) overrides the default in either direction |
| The whole conversation, on every turn | "The Messages API is stateless, which means that you always send the full conversational history to the API." |
| Hooks in Claude Code and the Agent SDK | Zero, unless a hook returns additional context: in the SDK, hooks "run in your application process, not inside the agent's context window" |

Because the API is stateless, CCAR-F 5.1-K4 expects you to know "The importance of passing complete conversation history in subsequent API requests to maintain conversational coherence." The price is that every turn re-sends everything before it. Claude Academy's tutorial [How context affects Claude's performance and cost](https://academy.claude.com/tutorials/parametric-memory-and-context) makes the same point for chat: "That's why a short question late in a long conversation may cost more than the same question in a fresh one, and why starting a new chat can be the cheapest and fastest way to get an answer." Agentic work multiplies the effect; in Anthropic's research-system measurements "agents typically use about 4× more tokens than chat interactions, and multi-agent systems use about 15× more tokens than chats" ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)).

Tools are the other large consumer, twice over:

- **Definitions load before any work.** The [Tool search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool) docs describe a typical multiserver setup (GitHub, Slack, Sentry, Grafana and Splunk) that can consume about 55k tokens of definitions before Claude does any work, and note that "Tool search typically reduces this by over 85 percent". They also warn that Claude's ability to pick the right tool degrades once more than 30 to 50 tools are available.
- **Results pile up.** CCAR-F 5.1-K3 names "How tool results accumulate in context and consume tokens disproportionately to their relevance (e.g., 40+ fields per order lookup when only 5 are relevant)". Trimming before results accumulate is taught in [Preserving critical information in long conversations](#preserving-critical-information-in-long-conversations); clearing them afterwards is in [Compaction, context editing and memory](#compaction-context-editing-and-memory); sizing the tool set itself is in [Designing a tool set](tool-use-and-mcp.md#designing-a-tool-set).

Model window sizes, output limits and token counting are in [Tokens, context windows and counting](claude-api.md#tokens-context-windows-and-counting). Two overflow behaviors matter here. If the input alone is larger than the window, the API returns a 400 `invalid_request_error` ("prompt is too long") on every model. On Claude 4.5 models and newer, a request whose input plus `max_tokens` exceeds the window is accepted, and if generation reaches the limit it stops with `stop_reason: "model_context_window_exceeded"`, which the [stop-reason reference](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons) says to "Treat the response as truncated." On earlier models the same request returns a validation error by default; the `model-context-window-exceeded-2025-08-26` beta header opts them in to the `model_context_window_exceeded` behavior.

### Up front or just in time

CCAR-P 3.8 asks you to "Evaluate progressive discovery vs. monolithic context strategy." Anthropic's [engineering post](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) describes both ends. Many AI-native applications "employ some form of embedding-based pre-inference time retrieval to surface important context for the agent to reason over." Agents built with the "just in time" approach instead "maintain lightweight identifiers (file paths, stored queries, web links, etc.) and use these references to dynamically load data into context at runtime using tools", which lets an agent incrementally discover relevant context through exploration.

In the table below, quoted cells are the post's words; unquoted Gains and Costs cells are our summary of the trade-off.

| Strategy | How it works | Gains | Costs and fit |
|---|---|---|---|
| Up front (monolithic) | Load or retrieve everything that might matter before the first call | No exploration step; stable content can be ordered first and cached | Every token draws on the attention budget whether it is used or not; a pre-built index can go stale |
| Just in time (progressive discovery) | Hold identifiers; load content with tools when needed | Only what the task touches enters the window; Claude Code's glob and grep retrieval bypasses "the issues of stale indexing and complex syntax trees" | "runtime exploration is slower than retrieving pre-computed data" |
| Hybrid | Retrieve "some data up front for speed" and explore the rest | Claude Code works this way: "CLAUDE.md files are naively dropped into context up front, while primitives like glob and grep allow it to navigate its environment and retrieve files just-in-time" | The post suggests the hybrid "might be better suited for contexts with less dynamic content, such as legal or finance work" |

Progressive discovery shows up across the products:

- **Agent Skills** load in stages: name and description metadata at startup, the full SKILL.md when relevant, bundled files only as needed. Per Anthropic's [Agent Skills post](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills), that metadata "is the first level of progressive disclosure".
- **Tool search** (`defer_loading: true`) discovers tool definitions on demand. The [Manage tool context](https://platform.claude.com/docs/en/agents-and-tools/tool-use/manage-tool-context) page suggests adding it "once your toolset grows past roughly 20 tools or your baseline context usage becomes noticeable."
- **Claude Code** loads CLAUDE.md in full, and it costs context on every request; skills load only their descriptions until used; MCP servers load tool names, with full schemas deferred until a tool is needed; subagents run in a context isolated from the main session, fresh unless the subagent is a fork of the current conversation ([Extend Claude Code](https://code.claude.com/docs/en/features-overview)).

**Decide (our rule, from the sources cited).** Content every request needs (role, policies, a style guide) goes up front, ordered before the dynamic content, and is cached. CCAR-P [sample 2](../claude-certified-architect-professional.md#official-sample-questions) rewards exactly that for a repeated 8,000-token system prompt and policy: "Place the static system prompt and policy before the dynamic content and enable prompt caching." See [Prompt caching](claude-api.md#prompt-caching). Content that is large, changing or rarely needed (a codebase, a document store, a large tool catalog) is discovered just in time. When both apply, use the hybrid. In the passage that describes the hybrid, the engineering post also says "do the simplest thing that works" will likely remain Anthropic's best advice for teams building agents on top of Claude.

### How Claude tracks its own budget

Some models see their remaining budget. Per the [Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows) page (as of September 2026), Claude Sonnet 5, Sonnet 4.6, Sonnet 4.5 and Haiku 4.5 have context awareness: they track their remaining context window, which the docs call their "token budget", throughout a conversation. Nothing needs enabling, and you never send the tags yourself: "The API injects them." The injected formats, shown for a 200k-token window, are:

```text
<budget:token_budget>200000</budget:token_budget>
<system_warning>Token usage: 35000/200000; 165000 remaining</system_warning>
```

The first tag goes in the system prompt of every request; the second arrives after each tool call, and image tokens are included in these budgets. Opus 4.7 and later Opus models, Fable 5.1, Mythos 5.1, Fable 5 and Mythos 5 "don't receive these injected tags." On those models the docs point to **task budgets** instead (beta, header `task-budgets-2026-03-13`), which tell Claude how many tokens it has for a whole agentic loop, including thinking, tool calls, tool results and output ([Task budgets](https://platform.claude.com/docs/en/build-with-claude/task-budgets)). The request body from the docs' example, sent with `anthropic-beta: task-budgets-2026-03-13`:

```json
{
  "model": "claude-opus-5-5",
  "max_tokens": 128000,
  "stream": true,
  "messages": [{
    "role": "user",
    "content": "Review the codebase and propose a refactor plan."
  }],
  "output_config": {
    "effort": "high",
    "task_budget": {"type": "tokens", "total": 64000}
  }
}
```

| Task budget fact (as of September 2026) | Value |
|---|---|
| Fields | `type` (always `"tokens"`), `total`, optional `remaining` (defaults to `total`) |
| Scope | One agentic turn: everything Claude does in response to one user message that carries no tool results, which can span several requests |
| Minimum `total` | 20,000 tokens; smaller values return a 400 error |
| Enforcement | "a **soft hint, not a hard cap**"; `max_tokens` is still the enforced per-request output limit |
| Visibility | "The countdown is visible only to the model."; responses carry no remaining-budget field |
| Too small a budget | "can cause refusal-like behavior": declining, aggressive scoping down or stopping early |
| Your own compaction | Pass `remaining` so the countdown continues instead of resetting to `total` |
| Server-side compaction | Compaction during a turn does not reset the budget |
| Where it works | The Messages API, in beta on Fable 5.1, Mythos 5.1, Opus 5.5, Opus 5, Fable 5, Mythos 5, Opus 4.8 and Opus 4.7; not supported on Claude Code or Cowork surfaces |

!!! warning "Exam guide vs current docs"

    None of the four July 2026 exam guides names context awareness or task budgets. They test budgets as a concept: the CCAR-F technology list pairs "token budgets" with progressive summarization and scratchpad files, and the CCDV-F Cost and Token Management skill covers "Token budgeting". In today's docs, whether a model sees a countdown depends on the model, and a visible countdown can backfire: the [Fable 5 prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5) says that in very long sessions Fable 5 can occasionally suggest a new session, offer to summarize and hand off, or trim its own work, most often when the harness shows it a remaining-token countdown, and advises: "Avoid surfacing explicit context-budget counts where possible." Answer exam items in the guides' terms: what to do with the budget (trim, summarize, persist, isolate), not which tag carries it.

### Traps

- **Choosing prompt caching to free up context.** It does not: "Prompt caching doesn't reduce the number of tokens in context, but it reduces what you pay for them on subsequent requests" ([Manage tool context](https://platform.claude.com/docs/en/agents-and-tools/tool-use/manage-tool-context)). Caching is the right answer when the problem is the cost and latency of a repeated, stable prefix (CCAR-P sample 2), not when the window is full.
- **Truncating content the task needs.** CCAR-P sample 2's rationale rejects cutting the policy document because "Truncation (A) loses needed policy". Shrink what is irrelevant, not what is required.
- **Sending only the latest message to save tokens.** The API is stateless; dropping history breaks coherence (5.1-K4). Shrink history deliberately with compaction or a summary instead.
- **Adding tools to make the agent more capable.** Definitions cost tokens up front, and selection accuracy falls as the tool count grows.
- **Buying a bigger window to fix missed details.** See sample question 12 above.

## Preserving critical information in long conversations

*Tested in: CCAR-F 5.1 (5.1-K1, 5.1-K2, 5.1-S1 to 5.1-S4), 1.5-K1, Appendix in-scope topic "Context window optimization", Scenario 1 (Customer Support Resolution Agent) · CCDV-F D6.1 Context Engineering (context drift and bloat, tool output pruning) · CCAR-P 2.4 · CCAO-F D3.4*

Long conversations get summarized, by you, by the API or by Claude Code, and summaries lose detail. The [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) names the risk in 5.1-K1: "Progressive summarization risks: condensing numerical values, percentages, dates, and customer-stated expectations into vague summaries". Anthropic's own sources agree on what goes first:

- The [context engineering cookbook](https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools): "The summary preserves key decisions and facts but may drop specific numbers or exact phrasing." In its probe, high-level facts central to the task usually survived, but "Obscure specifics (a single cell in an appendix table, a heterogeneity statistic) usually don't."
- Claude Code's [How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works): "Your requests and key code snippets are preserved; detailed instructions from early in the conversation may be lost. Put persistent rules in CLAUDE.md rather than relying on conversation history."
- Claude Academy's [How context affects Claude's performance and cost](https://academy.claude.com/tutorials/parametric-memory-and-context): compaction "is still a summary, and can still occasionally result in lost details."

Our rule for the whole section: **anything that must be exact should not depend on a summary.** Keep it somewhere that is re-sent or re-injected verbatim, and write summary instructions that name it.

### Pin transactional facts outside the summary

CCAR-F 5.1-S1: "Extracting transactional facts (amounts, dates, order numbers, statuses) into a persistent "case facts" block included in each prompt, outside summarized history". For sessions that handle several issues, 5.1-S2 adds "Extracting and persisting structured issue data (order IDs, amounts, statuses) into a separate context layer for multi-issue sessions".

The exam guide's "case facts" block is a pattern you build yourself; the products supply the building blocks listed below. The mechanics are simple because the API is stateless: your code owns the request, so it can rebuild the facts block on every turn and keep it out of the range that gets compacted. That placement matters. On-demand compaction removes the summarized messages and puts one compaction block first in `messages`, and `role: "system"` messages inside the summarized range "are summarized too", so a fact that lives only in an old message ends up in the summary, where it can go vague. An illustrative block for Scenario 1's support agent (field names and values are ours):

```xml
<case_facts>
customer_id: C-4471 (verified by get_customer)
issue_1: order #12345, status: delivered damaged, refund requested: 89.99 USD
issue_2: order #12377, status: in transit, customer expects delivery by 2026-09-26
agreed_so_far: replacement offered for issue_1; customer declined and asked for a refund
</case_facts>
```

Each surface documents mechanisms that come closest to the case-facts block. Some re-send or re-inject exact text; others only steer what a summary keeps, and that summary is still lossy:

| Surface | Verbatim re-injection, and summary steering | Where it is taught |
|---|---|---|
| Messages API | Your code re-sends the facts block each turn, outside the compacted range (verbatim); compaction `instructions` "say what the summary must retain" (steers the summary; still lossy) | [API compaction](#server-side-compaction-on-the-claude-api) |
| Claude Code | A `SessionStart` hook with the `compact` matcher re-injects critical context after every compaction, and the plan-mode plan file is re-injected from disk (verbatim); a "Compact Instructions" section in CLAUDE.md or `/compact` with a focus (steer the summary; still lossy) | [Compaction in Claude Code](#compaction-in-claude-code-and-the-agent-sdk) |
| Agent SDK | CLAUDE.md loaded through `settingSources` is re-injected on every request (verbatim); a section there tells the compactor what to preserve, and the header is free-form (steers the summary; still lossy) | [Compaction in Claude Code](#compaction-in-claude-code-and-the-agent-sdk) |
| Any long-running agent | The memory tool (API) or a scratchpad file (CCAR-F 5.4-K2) holds what must survive summarization, outside the window | [The memory tool](#the-memory-tool) |

### Write summary instructions that keep specifics

When you control the summarization prompt, name what must survive. Both server-side compaction kinds on the Messages API accept an `instructions` string, and in both it replaces the default summarization prompt completely instead of adding to it. For on-demand compaction (beta header `compact-2026-09-04`), the [Compaction on demand](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand) page caps `instructions` at 16,384 characters and says to "say what the summary must retain and tell the model not to call tools." Its example:

```json
{
  "compaction": {
    "type": "summarize",
    "instructions": "Summarize this recipe app design conversation. Preserve every entity and field name agreed so far, and the user's latest open request. Do not call tools; respond with the summary text only."
  }
}
```

The [Prompting Claude Fable 5.1](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-1) guide notes that server-side compaction already tells the model what to retain, and publishes a summary instruction for client-side compaction that lists six things to preserve: difficulties and how they were handled; options raised, tried or set aside; anything asked for, decided, agreed or ruled out, stated exactly; where things stand now; what is still open; and hard-to-reconstruct details such as names, numbers, dates, exact wording and links, kept exactly. It weights the two voices differently: "keep what the user said, asked for, shared, or established carefully and close to their own words", while Claude's own explanations can be condensed.

Two tuning rules from the [engineering post](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): "Start by maximizing recall to ensure your compaction prompt captures every relevant piece of information from the trace, then iterate to improve precision by eliminating superfluous content." And the reason for caution: "overly aggressive compaction can result in the loss of subtle but critical context whose importance only becomes apparent later." The [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing) page lists "Tasks requiring precise recall of early conversation details" and "Tasks that need to maintain exact state across many variables" as less ideal use cases for client-side SDK compaction (`compaction_control`, now deprecated; see the warning in [Server-side compaction on the Claude API](#server-side-compaction-on-the-claude-api)); those are exactly the facts to pin.

### Trim tool output before it accumulates

CCAR-F 5.1-S3: "Trimming verbose tool outputs to only relevant fields before they accumulate in context (e.g., keeping only return-relevant fields from order lookups)". The guide's example is an order lookup that returns 40+ fields when 5 matter (5.1-K3). Trim at the point where the result is produced:

```python
# Illustrative: a returns agent needs a handful of fields from a 40+ field order record.
RETURN_FIELDS = ("order_id", "status", "delivered_at", "total", "return_window_ends")

def trim_order(order: dict) -> dict:
    return {key: order[key] for key in RETURN_FIELDS if key in order}
```

The table draws on Anthropic's [Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents), Claude Code's [Manage costs effectively](https://code.claude.com/docs/en/costs) and [Environment variables](https://code.claude.com/docs/en/env-vars) pages, and the API's [Manage tool context](https://platform.claude.com/docs/en/agents-and-tools/tool-use/manage-tool-context) page.

| Technique | Documented detail |
|---|---|
| Return only high-signal fields | Tools should "eschew low-level technical identifiers (for example: uuid , 256px_image_url , mime_type )" |
| Let the agent choose verbosity | A `response_format` enum "allowing your agent to control whether tools return “concise” or “detailed” responses"; in Anthropic's Slack example concise used about one third of the tokens |
| Paginate, filter, truncate | Recommended with sensible default parameter values; "For Claude Code, we restrict tool responses to 25,000 tokens by default." |
| Search instead of dump | "consider implementing a search_logs tool which only returns relevant log lines and some surrounding context" instead of `read_logs` |
| Pre-filter in a hook (Claude Code) | "a hook can grep for `ERROR` and return only matching lines, reducing context from tens of thousands of tokens to hundreds" |
| Keep intermediate results out of history | With programmatic tool calling, "The intermediate results never enter the conversation history." |
| Claude Code output caps | `MAX_MCP_OUTPUT_TOKENS` (default 25000; a warning shows when output exceeds 10,000 tokens); `BASH_MAX_OUTPUT_LENGTH` (default 30000 characters, maximum 150000) |

**Decide (our rule, from the sources cited).** If you own the tool, trim in the tool. If you do not (a third-party MCP server), transform the result before the model sees it: CCAR-F 1.5-K1 names "Hook patterns (e.g., PostToolUse) that intercept tool results for transformation before the model processes them", and current Claude Code and Agent SDK docs let a `PostToolUse` hook replace any tool's output with `updatedToolOutput` (the Agent SDK docs call the older MCP-only `updatedMCPToolOutput` deprecated; the Claude Code reference says to prefer `updatedToolOutput`). The replacement only changes what Claude sees; the tool has already run. See [Hooks in the SDK](agents-and-agent-sdk.md#hooks-in-the-sdk). If old results have already piled up and can be fetched again, clear them with [context editing](#context-editing-clearing-tool-results-and-thinking).

### Order aggregated input for position effects

CCAR-F 5.1-K2 defines the effect the exam calls "lost in the middle": "models reliably process information at the beginning and end of long inputs but may omit findings from middle sections". The matching skill, 5.1-S4, is "Placing key findings summaries at the beginning of aggregated inputs and organizing detailed results with explicit section headers to mitigate position effects".

Anthropic's [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) give the documented layout for large documents or data-rich inputs (20k+ tokens): "Place your long documents and inputs near the top of your prompt, above your query, instructions, and examples." The page reports that "Queries at the end can improve response quality by up to 30 percent in tests, especially with complex, multidocument inputs," recommends wrapping each document in `<document>` tags with `<document_content>` and `<source>` subtags when using multiple documents, and suggests asking Claude to quote relevant parts first. An illustrative layout of our own that follows the docs' order (material first, query last) and the guide's skill (summary first, explicit sections):

```xml
<aggregated_findings>
  <key_findings_summary>
    The three findings that matter most, each with its source ID.
  </key_findings_summary>
  <section title="Pricing data">...</section>
  <section title="Regulatory news">...</section>
  <section title="Technical benchmarks">...</section>
</aggregated_findings>

Using only the findings above, recommend a Q3 focus and cite the source ID for each claim.
```

!!! warning "Exam guide vs Anthropic's documentation"

    The exam guide defines "lost in the middle", and Anthropic's post [Best practices for prompt engineering for 2026](https://claude.com/blog/best-practices-for-prompt-engineering) uses the same term: Claude 4.x models "have significantly improved context awareness capabilities" that help address historical "lost-in-the-middle" issues, yet the post still advises: "When working with long contexts, structure your information clearly with the most critical details at the beginning or end." Anthropic's September 2023 post [Prompt engineering for Claude's long context window](https://www.anthropic.com/news/prompting-long-context) reports that "Claude 2 performance on 95K sees a small dip in the middle", with a footnote citing a paper that "found a U-shaped relationship between performance and location in the context", and today's prompting guide puts long material first and the query last.

    The guide's skill (key findings summary at the start, explicit section headers) and the docs' order (long material first, query last) are separate pieces of advice, and neither source states the other's rule. In our reading, the two can be combined: put the aggregated material above the query and open it with its key-findings summary, as the block above does. The blog's advice to put the most critical details at the beginning or end is consistent with that summary-first placement. Answer exam items with the guide's wording.

    The December 2023 trick from [Long context prompting for Claude 2.1](https://claude.com/blog/claude-2-1-prompting), adding "Here is the most relevant sentence in the context:" to the start of Claude's response, was a prefill. Starting with Claude 4.6 models and Claude Mythos Preview, prefilling the last assistant turn is no longer supported and returns a 400 error. The quote-first technique survives; the prefill mechanism does not.

Full long-document prompting technique is in [Long documents](prompt-engineering.md#long-documents).

### Traps

- **Summarizing everything, including the numbers.** 5.1-K1 is the trap itself: amounts, percentages, dates and what the customer was promised go vague. Pin them.
- **Relying on an instruction given once, early in the chat.** Claude Code's docs say such instructions can be lost at compaction; put standing rules in CLAUDE.md.
- **Passing raw 40-field tool results "just in case".** They crowd out what matters (5.1-K3). Trim first.
- **Answering a position-effect item with a bigger window.** The fix is ordering and structure (5.1-S4).

## Compaction, context editing and memory

*Tested in: CCAR-F 5.4-K2, 5.4-S2, 5.4-S5, Appendix technology list (`/compact`, `/memory`, "Context window management") · CCDV-F D6.1 Context Engineering (compaction, tool output pruning), D1.3 Agent Patterns and Frameworks (memory, context-window management), D3.1 Claude Code Operation (Agent Memory) · CCAR-P 2.4, 4.5 · CCAO-F D3.4*

Anthropic's [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) names three techniques for work that outgrows one window: "compaction, structured note-taking, and multi-agent architectures." Its [context engineering cookbook](https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools) adds clearing and gives one mental model for compaction, clearing and memory: "compaction compresses the whole window when it grows too large, clearing drops stale re-fetchable data inside the window, and memory moves information out of the window so it survives across sessions." Subagents are covered in [Subagents as context isolation](#subagents-as-context-isolation).

| Technique | What it loses | Choose it when |
|---|---|---|
| Compaction | Detail, in a controlled way: "The summary preserves key decisions and facts but may drop specific numbers or exact phrasing." | "Compaction maintains conversational flow for tasks requiring extensive back-and-forth;" |
| Clearing (context editing) | Nothing that matters, if the tool can be called again: "Clearing is lossless as long as the tool is re-callable." | Old tool results are stale and re-fetchable |
| Memory and structured notes | Nothing that was written down; the rest is gone | "Note-taking excels for iterative development with clear milestones;" |
| Multi-agent | The subagents' working detail; only their summaries return | "Multi-agent architectures handle complex research and analysis where parallel exploration pays dividends." |

### Server-side compaction on the Claude API

"Compaction replaces the older turns of a conversation with a summary that Claude writes on the server, so you need no summarization code of your own." The [Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows) page calls server-side compaction "the primary strategy for context management" for long-running conversations and agentic workflows. There are two kinds, both in beta, and the [compaction overview](https://platform.claude.com/docs/en/build-with-claude/compaction) says: "Use on-demand compaction wherever it is available."

| | Compaction on demand | Compaction at a token threshold |
|---|---|---|
| Beta header | `compact-2026-09-04` | `compact-2026-01-12` |
| How you ask | A separate request with top-level `"compaction": {"type": "summarize"}` | `{"type": "compact_20260112"}` in `context_management.edits` on ordinary requests |
| Who decides when | Your application | The API, when input tokens reach the trigger |
| Trigger | Your code's own limit | Default `{"type": "input_tokens", "value": 150000}`; `input_tokens` is the only type; `value` at least 50,000 |
| What comes back | The `compaction` block alone, `stop_reason: "compaction"`, no reply | The block, then the reply continues; with `pause_after_compaction` the API stops with `stop_reason: "compaction"` so you can add content first |
| What you send next | The block first in `messages`, summarized messages removed, exactly one block on every later request | Append the response as usual; "The API automatically drops all content blocks prior to the `compaction` block" |
| Custom prompt | `instructions`, up to 16,384 characters, replaces the default | `instructions` replaces the default; it does not supplement it |
| Recent turns kept word for word | Yes, with keep-tail compaction | Yes, by pausing after compaction and re-inserting them |
| Runs in the background | Yes, with background compaction | No: it runs inside the request that reaches the threshold |
| Platforms | Claude API, Claude Platform on AWS, Google Cloud, Microsoft Foundry (beta); not Amazon Bedrock | Beta on all five platforms |

An on-demand request and the swap that follows it, combined from the request sample and the compaction loop in the [on-demand compaction docs](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand) (`history` is your message list; send the same `system` prompt and `tools` as the rest of the conversation when you have them):

=== "Python"

    ```python
    response = client.beta.messages.create(
        model="claude-opus-5-5",
        # max_tokens caps the whole call, including any thinking, so allow several thousand tokens.
        max_tokens=4096,
        betas=["compact-2026-09-04"],
        messages=history,
        compaction={"type": "summarize"},
    )
    if response.stop_reason == "compaction":
        history = [{"role": "assistant", "content": response.content}]
    ```

=== "TypeScript"

    ```typescript
    const response = await client.beta.messages.create({
      model: "claude-opus-5-5",
      // max_tokens caps the whole call, including any thinking, so allow several thousand tokens.
      max_tokens: 4096,
      betas: ["compact-2026-09-04"],
      messages: history,
      compaction: { type: "summarize" }
    });
    if (response.stop_reason === "compaction") {
      history = [{ role: "assistant", content: response.content }];
    }
    ```

The response carries only the block, and the summarization tokens appear under `usage.iterations`:

```json
{
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "compaction",
      "content": "Summary of the conversation: ...",
      "signature": "EuYBCkQY..."
    }
  ],
  "stop_reason": "compaction",
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "iterations": [{ "type": "compaction", "input_tokens": 144, "output_tokens": 276 }]
  }
}
```

Rules that production bugs turn on (current-docs material from [Compaction on demand](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand) and [Compaction at a token threshold](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold); the guides test compaction only as a concept and as `/compact`):

- **Compact early.** "The conversation must still fit the model's context window, so compact before you outgrow it, not after."
- **Check `stop_reason` first.** A summary comes back only when the call ends normally; "Otherwise, the response is still a 200 with empty `content`, so check `stop_reason` before you look for the block."
- **Swap carefully.** "Send it in future requests exactly as it came," signature included; a changed `signature` or `content` returns a 400. Summarized messages left in front of the block return a 400 (`compaction_block_misplaced`), but "Two mistakes in the swap raise no error": summarized messages left after the block are sent to Claude again, and a later request that leaves the block out gives Claude no summary. Send the `compact-2026-09-04` header on every request that carries the block.
- **Restate what the summary cannot carry.** "Images, documents, `container_upload` blocks, and fetched URLs inside the summarized messages are gone once the block replaces them." `role: "system"` messages inside the summarized range are summarized too, so restate instructions that must keep applying.
- **One kind per request.** "You can't combine `compaction` with `context_management` on one request."
- **Same model.** Both kinds summarize with the request's model; the threshold docs put it plainly: "There is no option to use a different (for example, cheaper) model for the summary."
- **Tools during summarization.** With tools defined, the model occasionally calls a tool instead of summarizing. Threshold compaction then yields a `compaction` block with `content: null`; an on-demand request returns empty `content` with `stop_reason: "tool_use"`. In both cases, custom `instructions` that tell the model not to call tools are the documented fix.
- **Billing.** Compaction adds a sampling step that is billed and rate-limited; "To calculate total tokens consumed and billed for a request, sum across all entries in the `usage.iterations` array." Sending a block back on later requests adds no compaction cost.
- **Caching.** For threshold compaction, the docs advise: "To maximize cache hit rates, add a `cache_control` breakpoint at the end of your system prompt," so only the summary needs a new cache write. For on-demand compaction, "`cache_control` on the block places a breakpoint after the summary."

!!! note "`compaction` as a stop reason"

    `stop_reason: "compaction"` is documented on the compaction pages (every successful on-demand request, and threshold compaction with `pause_after_compaction`). The quick-reference table on the [stop reasons page](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons) does not mention it, so treat it as a beta-feature stop reason. Our advice: a loop that branches on `stop_reason` needs its own `compaction` branch, or a compaction pause can be mistaken for a finished turn.

Variants on the on-demand loop: **keep-tail** compaction ([Compaction that keeps recent turns](https://platform.claude.com/docs/en/build-with-claude/compaction-keep-recent-turns)) keeps the last few turns word for word, and "No parameter sets which turns are kept": you pick a cut point where no tool call is left open. **Background** (async) compaction lets the conversation continue on its full history while the summary is written, then swaps the block in. The SDK tool runners can do the request for you ([Compaction on demand](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand)): "When you decide to compact, call `compact_before_next_turn()` on the runner" (`compactBeforeNextTurn()` in TypeScript and Java); create the runner with the `compact-2026-09-04` beta, "because the runner doesn't add it."

With a [task budget](#how-claude-tracks-its-own-budget), the three compaction routes behave differently ([Task budgets](https://platform.claude.com/docs/en/build-with-claude/task-budgets)):

| How you compact | What to do with the budget |
|---|---|
| Your own code summarizes or rewrites history | "Pass `remaining` on the next request so the countdown continues from where you left off rather than resetting to `total`" |
| Threshold compaction during a turn | Nothing: it does not reset the budget, and tokens the turn used before the compaction still count against it |
| On-demand compaction | Do not send `remaining` with `compaction` or on requests that carry the block; the docs say doing so returns a 400 error |

!!! warning "Exam guide vs current docs"

    The July 2026 guides test compaction as a concept or a Claude Code command: CCDV-F D6.1 lists "prevention of context drift and bloat (tool output pruning, compaction)", and CCAR-F 5.4-S5 and its technology list name `/compact`. None of the four guides names a compaction API parameter, beta header or context-editing strategy. The current docs (as of September 2026) describe two server-side kinds, both beta: threshold compaction (`compact-2026-01-12`) and on-demand compaction (`compact-2026-09-04`), and the [compaction overview](https://platform.claude.com/docs/en/build-with-claude/compaction) says "Use on-demand compaction wherever it is available." Per the [context editing docs](https://platform.claude.com/docs/en/build-with-claude/context-editing), client-side SDK compaction (`compaction_control`) "is deprecated in the TypeScript and Ruby SDKs" and "The Python SDK removed it in v1.0"; "Anthropic recommends server-side compaction over SDK compaction." Tutorials that show `compaction_control` in Python are out of date. On the exam, answer in the guides' terms: compaction is a lossy summary that frees context, and `/compact` is the tool for an exploration session full of verbose discovery output.

### Context editing: clearing tool results and thinking

[Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing) removes content server-side before the prompt reaches Claude, on the reasoning that "context is a finite resource with diminishing returns, and irrelevant content degrades model focus." It needs the beta header `context-management-2025-06-27` and works on all supported Claude models. "Your client application maintains the full, unmodified conversation history," so there is nothing to sync. Current docs call server-side compaction the primary strategy for most cases and context editing the tool for finer-grained control.

The two strategies combined, as in the docs' example (request fields only; the call also sends the `context-management-2025-06-27` beta header):

```json
{
  "context_management": {
    "edits": [
      {"type": "clear_thinking_20251015", "keep": {"type": "thinking_turns", "value": 2}},
      {
        "type": "clear_tool_uses_20250919",
        "trigger": {"type": "input_tokens", "value": 50000},
        "keep": {"type": "tool_uses", "value": 5}
      }
    ]
  }
}
```

| `clear_tool_uses_20250919` option | Default | Effect |
|---|---|---|
| `trigger` | 100,000 input tokens | When clearing starts; set in `input_tokens` or `tool_uses` |
| `keep` | 3 tool uses | Recent tool use/result pairs kept; the oldest are cleared first |
| `clear_at_least` | None | Minimum tokens cleared each time, so the cache break is worth it; if the API can't clear that much, the strategy is not applied |
| `exclude_tools` | None | Tools whose uses and results are never cleared |
| `clear_tool_inputs` | false | `true` also clears the tool call parameters |

Per the [context editing docs](https://platform.claude.com/docs/en/build-with-claude/context-editing), cleared results are replaced with "placeholder text indicating to Claude that it was removed." `clear_thinking_20251015` takes `keep` as `{"type": "thinking_turns", "value": N}` (N greater than 0) or `"all"`; its default is model-specific (all turns on Opus 4.5 and later, Sonnet 4.6 and later, and the Fable and Mythos models; the last turn only on earlier Opus and Sonnet models and all Haiku models). When both strategies are used, `clear_thinking_20251015` "must be listed first in the `edits` array." The response's `context_management.applied_edits` reports what was applied (for example `cleared_tool_uses`, `cleared_thinking_turns`, `cleared_input_tokens`), in the final `message_delta` event when streaming. Tool result clearing invalidates cached prompt prefixes, which is why `clear_at_least` exists; kept thinking blocks preserve the cache, and cleared ones invalidate it at the clearing point. Combined with the memory tool, Claude "receives an automatic warning to preserve important information" as the clearing threshold approaches. In the launch post ([Managing context on the Claude Developer Platform](https://claude.com/blog/context-management), September 29, 2025), context editing plus the memory tool improved Anthropic's internal agentic-search evaluation by 39% over baseline, and context editing alone by 29%.

### The memory tool

The [memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool) is how an API agent writes structured notes outside the window. "Claude can create, read, update, and delete files that persist between sessions, building up knowledge over time without keeping everything in the context window." It is client-side: "Claude requests file operations, and your application executes them." Setup is one tools entry, available on all Claude 4 and later models with no beta header for the tool itself (the SDK helpers live in beta namespaces):

```json
{
  "tools": [{"type": "memory_20250818", "name": "memory"}]
}
```

```python
import anthropic
from anthropic.tools import BetaLocalFilesystemMemoryTool

client = anthropic.Anthropic()
memory = BetaLocalFilesystemMemoryTool(base_path="./memory")
runner = client.beta.messages.tool_runner(
    model="claude-opus-5-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Remember that customer Acme Corp prefers email follow-ups."}],
    tools=[memory],
)
final_message = runner.until_done()
```

| Fact | Detail |
|---|---|
| Commands your handler implements | `view`, `create`, `str_replace`, `insert`, `delete`, `rename` |
| Default behavior | Claude "automatically checks its memory directory before starting a task" and stores what it learns under `/memories` |
| Injected protocol | Includes "ASSUME INTERRUPTION: Your context window might be reset at any moment, so you risk losing any progress that is not recorded in your memory directory." |
| Security | "Your implementation must validate every path in every command to prevent directory traversal attacks." |
| SDK helpers | Subclass `BetaAbstractMemoryTool` (Python, C#), use `betaMemoryTool` (TypeScript) or implement `BetaMemoryToolHandler` (Java) to back memory with your own storage; Python and TypeScript also ship the ready-made `BetaLocalFilesystemMemoryTool` used above |
| Pairing | "compaction keeps the active context small without client-side bookkeeping, and memory preserves the information that must survive summarization." |

For multi-session work the [memory tool docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool) describe an initializer session that sets up "a progress log (tracking what has been done and what comes next), a feature checklist (defining the scope of work), and a reference to any startup or initialization script the project needs," with one rule for later sessions: "Mark a feature complete only after end-to-end verification confirms it works, not when the code is written." Do not confuse the memory tool with [Managed Agents memory stores](https://platform.claude.com/docs/en/managed-agents/memory). A store is mounted as a directory in the session's sandbox, every change creates an immutable memory version, individual memories are capped at 100 kB (about 25k tokens), and "A store holds a maximum of 10,000 memories." When a store reaches that limit, writes to new memories fail while existing memories stay readable and editable, which is why the docs advise small, purpose-built stores.

### Compaction in Claude Code and the Agent SDK

When the window fills, Claude Code "clears older tool outputs first, then summarizes the conversation if needed" ([How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)). What that summary keeps and loses is quoted in [Preserving critical information in long conversations](#preserving-critical-information-in-long-conversations); the short rule is that persistent rules belong in CLAUDE.md. Automatic compaction does not end the session. The [commands](https://code.claude.com/docs/en/commands) that manage context:

| Command | What it does |
|---|---|
| `/compact [instructions]` | "Free up context by summarizing the conversation so far. Optionally pass focus instructions for the summary." |
| `/context` | Shows current context usage "as a colored grid" with optimization suggestions, including which CLAUDE.md and auto memory files loaded |
| `/clear [name]` | Starts a new conversation with empty context (aliases `/reset`, `/new`); "To free up context while continuing the same conversation, use `/compact` instead." |
| `/autocompact` | Sets "how full the context window gets before Claude Code compacts automatically" (Claude Code v2.1.221 or later) |
| `/rewind` | "Summarize from here" or "Summarize up to here" compresses part of the conversation ([Checkpointing](https://code.claude.com/docs/en/checkpointing)) |
| `/btw` | Asks a side question "without adding to the conversation" |
| `/memory` | Edits CLAUDE.md files, turns auto memory on or off, shows auto memory entries |

CCDV-F D3.1 lists "Agent Memory" among Claude Code's core components. The closest match in current docs is the [subagent `memory` field](https://code.claude.com/docs/en/sub-agents#enable-persistent-memory) (the guide does not define the term), which "gives the subagent a persistent directory that survives across conversations"; it is covered in [Subagent housekeeping](#subagent-housekeeping).

Auto-compaction settings, from [Model configuration](https://code.claude.com/docs/en/model-config) and [Environment variables](https://code.claude.com/docs/en/env-vars):

| Setting (as of September 2026) | Behavior |
|---|---|
| Default threshold | With no auto-compact window set, Claude Code compacts when the conversation reaches the model's context limit, with exceptions that include these: models running a native 1M window (for example Sonnet 5, the Fable models, and Opus 4.7 and later on the Anthropic API) compact "at about 967K tokens by default", and Sonnet 4.6 and Opus 4.6 without extended context compact at the 200K boundary |
| `/autocompact` and `--autocompact` | Accept a window from 100K to 1M tokens (for example `/autocompact 500k`); the command saves it to user settings as `autoCompactWindow`, the flag applies to one launch |
| `CLAUDE_CODE_AUTO_COMPACT_WINDOW` | 100000 to 1000000, plain integers; "Takes precedence over the `/autocompact` command, the `--autocompact` flag, and the `autoCompactWindow` setting." |
| `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` | Percentage (1 to 100) of the auto-compact window at which auto-compaction triggers; can only lower the threshold; applies only in sessions that compact before the model's context limit; also applies to subagents |
| `DISABLE_AUTO_COMPACT=1` | Turns off automatic compaction; manual `/compact` still works |
| `DISABLE_COMPACT=1` | Turns off all compaction, `/compact` included |

What survives a compaction ([Context window](https://code.claude.com/docs/en/context-window)):

| Item | After compaction |
|---|---|
| System prompt and output style | Still apply |
| Project-root CLAUDE.md, unscoped rules, auto memory | Re-injected from disk |
| Git status snapshot | A fresh one is read from the repository |
| The plan Claude wrote in plan mode | Re-injected from disk |
| Rules with `paths:` frontmatter, nested CLAUDE.md files | Summarized away; reload only when Claude reads a matching file again |
| Files Claude read or edited | Up to five re-read, most recently modified first; a file over 5,000 tokens returns only as a path reference (`Referenced file`) |
| Invoked skill bodies | Re-injected, capped at 5,000 tokens per skill and 25,000 tokens total; oldest dropped first (truncation keeps the start of `SKILL.md`) |
| Background commands and background subagents | Keep running; Claude is reminded which ones are still running |
| Context that hooks added earlier | Summarized with the rest |
| `SessionStart` hooks matching `compact` | Run again; their output is added |

"If a rule must persist across compaction, drop the `paths:` frontmatter or move it to the project-root CLAUDE.md" ([Context window](https://code.claude.com/docs/en/context-window)). Three ways to control what the summary keeps:

1. A compact-instructions section in the CLAUDE.md at the project root, for example (from the costs docs):

    ```markdown
    # Compact instructions

    When you are using compact, please focus on test output and code changes
    ```

2. A focus argument: `/compact focus on the auth bug fix`, run "before starting a long new task" ([Context window](https://code.claude.com/docs/en/context-window)).
3. A `SessionStart` hook with the `compact` matcher, which re-injects critical context after every automatic or manual compaction; Claude Code adds the text the command writes to stdout. The hooks guide's example for `.claude/settings.json`:

    ```json
    {
      "hooks": {
        "SessionStart": [
          {
            "matcher": "compact",
            "hooks": [
              {
                "type": "command",
                "command": "echo 'Reminder: use Bun, not npm. Run bun test before committing. Current sprint: auth refactor.'"
              }
            ]
          }
        ]
      }
    }
    ```

Two more hook events bracket compaction ([Hooks reference](https://code.claude.com/docs/en/hooks)): `PreCompact` runs before it, matches `manual` (`/compact`) or `auto`, and can block it ("Exit with code 2 to block compaction."; returning JSON with `"decision": "block"` also works); `PostCompact` receives `compact_summary` and has no decision control. Hook mechanics are in [Hooks](claude-code-workflows.md#hooks).

Failure mode: if one large file or output refills the window after each summary, "Claude Code stops auto-compacting after a few attempts and shows an error instead of looping" ([How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)). The error begins `Autocompact is thrashing: the context refilled to the limit...`. Recovery, per the [troubleshooting docs](https://code.claude.com/docs/en/troubleshooting): read the file in smaller chunks, run `/compact` with a focus that drops the large output, move the work to a subagent, or `/clear` if the earlier conversation is no longer needed.

Cost notes: `/compact` on a large context is itself a large request, while "When you want a fresh start instead of continuity, `/clear` costs nothing" ([Manage costs effectively](https://code.claude.com/docs/en/costs)). Compaction also invalidates the conversation layer of the prompt cache, "since the next request has a new, shorter history that doesn't share a prefix with the old one" ([How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching)).

**The Agent SDK** runs the same machinery ([How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop)). Its context window "does not reset between turns within a session"; near the limit it compacts automatically and emits a system message with `subtype: "compact_boundary"` (a `SystemMessage` in Python, an `SDKCompactBoundaryMessage` in TypeScript). Persistent rules belong in CLAUDE.md, loaded through `settingSources`, rather than in the initial prompt, "because CLAUDE.md content is re-injected on every request." The compactor reads CLAUDE.md, and "The compactor matches on intent, so the section header is free-form":

```markdown
# Summary instructions

When summarizing this conversation, always preserve:
- The current task objective and acceptance criteria
- File paths that have been read or modified
- Test results and error messages
- Decisions made and the reasoning behind them
```

A `PreCompact` hook in the SDK receives `trigger` (`manual` or `auto`) and is the place to archive the full transcript before it is summarized; like every hook it runs in your process and costs no context ([Where the tokens go](#where-the-tokens-go)). To compact on demand from the SDK, send `/compact` as the prompt string; commands sent this way are ordinary SDK inputs.

### Long-running agents: notes, resets and harness files

Anthropic's [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) frames the core problem: "The core challenge of long-running agents is that they must work in discrete sessions, and each new session begins with no memory of what came before." Compaction alone was not enough; even Opus 4.5 on the Agent SDK "will fall short of building a production-quality web app if it’s only given a high-level prompt". Two failure modes: running out of context mid-feature, "leaving the next session to start with a feature half-implemented and undocumented", and a later instance that would "look around, see that progress had been made, and declare the job done."

The fix was structured note-taking in files the next session reads first:

| Artifact | Purpose |
|---|---|
| `init.sh` | A script "that can run the development server" |
| `claude-progress.txt` | "a log of what agents have done" |
| Initial git commit, then commits "with descriptive commit messages" | The first commit "shows what files were added"; later commits let the model "use git to revert bad code changes and recover working states" |
| Feature list in JSON (over 200 features in the example, all initially marked failing) | Coding agents may only change each feature's `passes` field; JSON because "the model is less likely to inappropriately change or overwrite JSON files compared to Markdown files" |

```json
{
  "category": "functional",
  "description": "New chat button creates a fresh conversation",
  "steps": [
    "Navigate to main interface",
    "Click the 'New Chat' button",
    "Verify a new conversation is created",
    "Check that chat area shows welcome state",
    "Verify conversation appears in sidebar"
  ],
  "passes": false
}
```

In the [harness post](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents), each coding session starts the same way: run `pwd` to see the working directory, "Read the git logs and progress files to get up to speed on what was recently worked on," then read the feature list and take the highest-priority unfinished feature, one at a time, committing with descriptive messages. The post also has the agent run a basic end-to-end test before implementing a new feature, and reports that, absent explicit prompting, Claude tended to mark a feature as complete without proper end-to-end testing. Anthropic's [prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) generalizes the pattern: "Use the first context window to set up a framework (write tests, create setup scripts), then use future context windows to iterate on a todo-list"; track structured state in JSON (such as `tests.json`), progress notes as free text, and checkpoints in git ("Git provides a log of what's been done and checkpoints that can be restored."); and tell Claude "It is unacceptable to remove or edit tests because this could lead to missing or buggy functionality."

If your harness compacts context or lets Claude save it to external files, say so in the prompt; "Otherwise, Claude may sometimes naturally try to wrap up work as it approaches the context limit." The [prompting guide's](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) example prompt:

```text
Your context window will be automatically compacted as it approaches its limit, allowing
you to continue working indefinitely from where you left off. Therefore, do not stop
tasks early due to token budget concerns. As you approach your token budget limit, save
your current progress and state to memory before the context window refreshes. Always be
as persistent and autonomous as possible and complete tasks fully, even if the end of
your budget is approaching. Never artificially stop any task early regardless of the
context remaining.
```

Guidance for unattended runs on current models (as of September 2026): on Opus 5.5 ([Prompting Claude Opus 5.5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5-5)), "Treat a text-only end of turn as a report rather than as proof the task is done," and stop after two or three automatic continuations on the same task; on Fable 5 ([Prompting Claude Fable 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5)), "Before reporting progress, audit each claim against a tool result from this session." On Fable 5.1 ([Prompting Claude Fable 5.1](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-1)), cheaper cache reads mean "compacting early to save cost may no longer be the right cost-intelligence tradeoff on Claude Fable 5.1, so experiment with later compaction points."

!!! warning "Compaction or a clean reset: it depends on the model generation"

    The difference, in Anthropic's March 2026 post [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps): "While compaction preserves continuity, it doesn't give the agent a clean slate", whereas a reset clears the window and starts a fresh agent with a structured handoff. The post found that Sonnet 4.5 showed "context anxiety" ("they begin wrapping up work prematurely as they approach what they believe is their context limit") strongly enough "that compaction alone wasn't sufficient", so context resets became essential. With Opus 4.5 the behavior largely disappeared, resets were dropped, and the Agent SDK's automatic compaction handled context growth; the [Managed Agents post](https://www.anthropic.com/engineering/managed-agents) calls those resets "dead weight". The [prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) still suggests, when a window is cleared, "consider starting with a brand new context window rather than using compaction", because "Claude's latest models are extremely effective at discovering state from the local filesystem." Treat reset versus compaction as a trade-off, not a fixed rule. Reset handoffs are covered in [Session to session: resets with a structured handoff](#session-to-session-resets-with-a-structured-handoff).

### Restart, summarize or persist

The Associate exam asks the same question for everyday use: "Understand and manage context limitations and memory considerations (when to restart, summarize, or persist)" (CCAO-F D3.4). Claude Academy's [How context affects Claude's performance and cost](https://academy.claude.com/tutorials/parametric-memory-and-context) draws the line: "Compaction is how Claude can continue a single conversation beyond the context limit. Written memory is how Claude keeps important details in context across multiple conversations."

| Situation | Action | Mechanism by surface |
|---|---|---|
| A new, unrelated task | **Restart** | New chat in the apps; `/clear` in Claude Code ("Old conversation crowds out the files you need next and costs tokens on every message.") |
| Corrections are piling up | **Restart** with a better prompt | After two failed corrections, `/clear`: "A clean session with a better prompt almost always outperforms a long session with accumulated corrections." |
| Same task, long conversation, the thread matters more than exact wording | **Summarize** ([pin exact figures first](#pin-transactional-facts-outside-the-summary), or persist them) | Automatic context management in the apps (paid plans, code execution on); `/compact` or auto-compaction in Claude Code; API compaction |
| Facts needed in future conversations | **Persist** | Project knowledge and memory in the apps; CLAUDE.md and auto memory in Claude Code; the memory tool on the API |
| Details that change often (prices, statuses) | **Look up fresh**, do not persist | Academy: "Claude can and should look those things up next time they come up, rather than writing them down and hoping they haven't changed." |
| A length-limit error in the apps | **Restart or reduce** | "start a new conversation or use features like projects"; or summarize or extract key sections before sending |

App-specific facts (as of September 2026), from the [context window help article](https://support.claude.com/en/articles/8606394-how-large-is-the-context-window-on-paid-claude-plans): "Code execution must be enabled for automatic context management to work," and "Longer conversations that trigger automatic context management use more of your usage limit." Projects "use retrieval-augmented generation (RAG)", loading only relevant content. From the [projects help article](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects): "Context is not shared across chats within a project unless the information is added into the project knowledge base." Each project has its own memory space and summary; memory is on by default on Free, Pro and Max plans, while on Team and Enterprise plans "memory is off by default and can be turned on by an owner" ([Use Claude's chat search and memory](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context)); incognito chats are not saved to memory or history. The app features are taught in [Projects](claude-for-work.md#projects) and [Memory, styles and personalization](claude-for-work.md#memory-styles-and-personalization); Claude Code's CLAUDE.md and auto memory in [CLAUDE.md and the memory hierarchy](claude-code-configuration.md#claudemd-and-the-memory-hierarchy).

### Traps

- **Treating compaction as lossless.** It is a summary that can drop numbers and exact phrasing (first table above); [pin exact facts](#pin-transactional-facts-outside-the-summary) or write them to memory.
- **Avoiding clearing because it loses data.** Not if the tool can be called again; Anthropic's [engineering post](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) calls tool result clearing "One of the safest lightest touch forms of compaction".
- **Expecting a path-scoped rule to survive compaction.** It reloads only when a matching file is read again; a rule that must persist belongs in the project-root CLAUDE.md, which is re-injected (the same fix as for early instructions, a trap in [Preserving critical information in long conversations](#preserving-critical-information-in-long-conversations)).
- **Combining `compaction` and `context_management` in one call.** The API rejects it.
- **Keeping key findings only in the conversation.** CCAR-F 5.4-K2 and 5.4-S2 answer with a scratchpad file the agent records findings in and refers back to (see [Keep a scratchpad file](#keep-a-scratchpad-file)).
- **Using a deprecated mechanism in new code.** `compaction_control` is deprecated in the TypeScript and Ruby SDKs and gone from the Python SDK; server-side compaction is the recommended route.

## Subagents as context isolation

*Tested in: CCAR-F 1.2-K2, 1.3-K1, 1.3-K2, 3.2-K3, 3.2-S2, 3.4-K4, 3.4-S3, 5.4-K3, 5.4-S1, Appendix technology list (Explore subagent, subagent spawning via Task tool) · CCDV-F D6.1 Context Engineering ("context isolation through subagents or multi-step agentic workflows"), D1.1 Agent Architecture (role of subagents), D1.3 Agent Patterns and Frameworks · CCAR-P 1.4, 3.8*

A subagent protects the main context window because it spends its own window instead of yours. Anthropic's [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) describes the trade: "Each subagent might explore extensively, using tens of thousands of tokens or more, but returns only a condensed, distilled summary of its work (often 1,000-2,000 tokens)." The [multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) relies on the same effect: subagents "facilitate compression by operating in parallel with their own context windows". CCAR-F 5.4-K3 puts it in exam terms: "Subagent delegation for isolating verbose exploration output while the main agent coordinates high-level understanding".

The shape of one delegation (the file count and token figure are illustrative):

```text
Main conversation (coordinator)                 Subagent (fresh window)
------------------------------                  -----------------------
goal, decisions, key findings                   prompt string from the coordinator
        |                                               |
        |---- Agent tool call: task + context --------->|  reads 40 files, runs grep,
        |                                               |  runs tests, 60k tokens of output
        |<--- final message only (a short summary) -----|
        |
grows by the summary, not the transcript
```

The mechanics, in one line each (configuration is taught in [Subagents in the SDK](agents-and-agent-sdk.md#subagents-in-the-sdk) and [Subagents](claude-code-configuration.md#subagents)):

- **In:** "The only content you pass from parent to subagent is the Agent tool's prompt string, so include any file paths, error messages, or decisions the subagent needs directly in that prompt" ([Subagents in the SDK](https://code.claude.com/docs/en/agent-sdk/subagents)). A non-fork subagent does not see the parent's conversation history, tool results or system prompt (Agent SDK docs; CCAR-F 1.2-K2 and 1.3-K2 state the conversation-history part).
- **Out:** per the same page, "intermediate tool calls and results stay inside the subagent; only its final message returns to the parent." The parent "may summarize it in its own response", so ask for verbatim output if you need it.
- **Project rules reach most subagents, not all.** Per the [Claude Code subagents docs](https://code.claude.com/docs/en/sub-agents), a non-fork Claude Code subagent loads the same CLAUDE.md hierarchy as the main conversation, but the built-in Explore and Plan agents skip CLAUDE.md, and a definition with `omitClaudeMd` loads only managed policy files (or none). The main conversation still applies its full CLAUDE.md when it reads their results; if a rule must reach one of these subagents (the docs' example is ignoring the `vendor/` directory), restate it in the delegation prompt.

### Kinds of isolation

What each mechanism starts with and returns (from the [Claude Code subagents docs](https://code.claude.com/docs/en/sub-agents) unless a row links another page):

| Mechanism | Starts with | What returns to the main context |
|---|---|---|
| Non-fork subagent (Claude Code, Agent SDK) | A fresh window: its own system prompt, the delegation prompt, CLAUDE.md files (unless it opts out), tool definitions, any preloaded skills (the `skills` field in Claude Code, `AgentDefinition.skills` in the SDK) and, in Claude Code, a git status snapshot; not the conversation, invoked skills or files already read | The final message |
| Fork | "the entire conversation so far" | Only the final result: "The fork's own tool calls still stay out of your conversation" |
| Built-in Explore subagent | Read-only tools ("Write and Edit are denied"); skips CLAUDE.md and the git status snapshot "to keep research fast and inexpensive"; Claude sets a thoroughness level of quick, medium or very thorough | Findings only |
| Plan subagent in plan mode | Research delegated "so that exploration output stays in a separate context window while the main conversation remains read-only" | The research summary |
| [Skill](https://code.claude.com/docs/en/skills) with `context: fork` | An isolated subagent that "doesn't see your conversation history, so the skill's instructions have to stand on their own" (despite the name, not a fork of the conversation) | The skill's result |
| [Workflow script](https://code.claude.com/docs/en/workflows) (Claude Code) | Agents orchestrated by a script | "Claude's context holds only the final answer" |
| [Programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/manage-tool-context) (API) | Code that calls tools in a sandbox | Only what the code returns: "The intermediate results never enter the conversation history." |
| [Managed Agents multiagent session](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration) | Each agent in its own session thread | "Tools, MCP servers, and context are not shared." |

CCAR-F 3.4-K4 and 3.4-S3 name the Explore subagent specifically: it isolates "verbose discovery output" and returns summaries "to preserve main conversation context", and it is the tool "for verbose discovery phases to prevent context window exhaustion during multi-phase tasks." CCAR-F 3.2-K3 describes the skill option in the same terms: `context: fork` is "for running skills in an isolated sub-agent context, preventing skill outputs from polluting the main conversation", and 3.2-S2 applies it to "skills that produce verbose output (e.g., codebase analysis)". Skill configuration is taught in [Agent Skills](claude-code-configuration.md#agent-skills).

### When to isolate and when not to

The [Claude Code subagents docs](https://code.claude.com/docs/en/sub-agents) give the split:

| Use the main conversation when | Use a subagent when |
|---|---|
| The task needs frequent back-and-forth or iterative refinement | "The task produces verbose output you don't need in your main context" |
| Multiple phases share significant context (planning, implementation, testing) | You want to enforce specific tool restrictions or permissions |
| It is a quick, targeted change | The work is self-contained and can return a summary |
| Latency matters: a non-fork subagent starts fresh and may need time to gather context | |

Limits on the other side of the ledger:

- **Isolation is not free.** "Running many subagents that each return detailed results can consume significant context" ([Claude Code subagents docs](https://code.claude.com/docs/en/sub-agents)), and multi-agent work multiplies token use; Anthropic's measured multipliers are in [Where the tokens go](#where-the-tokens-go).
- **Shared-context work fits badly.** Anthropic's [multi-agent research post](https://www.anthropic.com/engineering/multi-agent-research-system): "some domains that require all agents to share the same context or involve many dependencies between agents are not a good fit for multi-agent systems today."
- **Over-delegation.** Anthropic's [prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) says Claude Opus 4.6 "may spawn subagents for code exploration when a direct grep call is faster and sufficient", and the [Agent SDK subagent docs](https://code.claude.com/docs/en/agent-sdk/subagents) note that Claude Opus 5 "delegates to subagents more readily than earlier models". The prompting guide's remedy is explicit guidance in the prompt (below).
- **For a question about what is already in context**, `/btw` fits better than a subagent: it sees the full context but has no tool access, and the answer is not added to history.
- **For a reusable prompt or workflow that should run in the main conversation**, the Claude Code docs point to Skills instead of a subagent.

The prompting guide's sample prompt for excessive subagent use:

```text
Use subagents when tasks can run in parallel, require isolated context, or involve
independent workstreams that don't need to share state. For simple tasks, sequential
operations, single-file edits, or tasks where you need to maintain context across steps,
work directly rather than delegating.
```

**Decide.** If the work would flood the main window with output you will not reread (file reads, test logs, search results), isolate it and ask for a bounded summary. If the next step depends on the details the work produces, or the user needs to steer each step, keep it in the main conversation. If the task is a single lookup, run the lookup.

### Delegation prompts that keep the return small

CCAR-F 5.4-S1 frames delegation as "Spawning subagents to investigate specific questions (e.g., "find all test files," "trace refund flow dependencies") while the main agent preserves high-level coordination". The Claude Code docs' own examples ask a specific question and request only what the parent needs:

```text
Use subagents to investigate how our authentication system handles token
refresh, and whether we have any existing OAuth utilities I should reuse.
```

```text
Use a subagent to run the test suite and report only the failing tests with their error messages
```

What to put in the prompt string (objective, output format, tool guidance, boundaries, complete prior findings) is taught in [Multi-agent orchestration](agents-and-agent-sdk.md#multi-agent-orchestration); what the return should look like when the parent's budget is tight is in [Up: subagent to coordinator](#up-subagent-to-coordinator).

### Isolation for judgment, not only for tokens

A fresh window also removes the author's bias. [Claude Code's best practices](https://code.claude.com/docs/en/best-practices) note that a reviewer running in a fresh subagent context "sees only the diff and the criteria you give it, not the reasoning that produced the change". Anthropic's [harness design post](https://www.anthropic.com/engineering/harness-design-long-running-apps) found that agents tend to praise their own output, and that "Separating the agent doing the work from the agent judging it proves to be a strong lever to address this issue." Managed Agents applies the same idea to [outcome grading](https://platform.claude.com/docs/en/managed-agents/define-outcomes): "The grader uses a separate context window to avoid being influenced by the main agent's implementation choices." Multi-pass review design is taught in [Multi-instance and multi-pass review](prompt-engineering.md#multi-instance-and-multi-pass-review).

### Subagent housekeeping

From the [Claude Code subagents docs](https://code.claude.com/docs/en/sub-agents):

- **Compaction.** Subagents "support automatic compaction using the same logic as the main conversation", and their transcripts are stored in separate files, so "when the main conversation compacts, subagent transcripts are unaffected."
- **Agent memory.** A Claude Code subagent's `memory` frontmatter field "gives the subagent a persistent directory that survives across conversations": `user` scope at `~/.claude/agent-memory/<name-of-agent>/`, `project` at `.claude/agent-memory/<name-of-agent>/` (shareable via version control), `local` at `.claude/agent-memory-local/<name-of-agent>/`; "`project` is the recommended default scope." It is part of auto memory, so turning auto memory off makes the field have no effect.
- **Growth caps (as of September 2026).** Nesting defaults to three layers below the main conversation (`CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH`), and spawning fails with `Concurrent subagent limit reached` once 20 subagents are running (`CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS`; Claude Code v2.1.217 or later, and not enforced in sessions with [ultracode](https://code.claude.com/docs/en/model-config#adjust-effort-level) active); there is no limit on the total spawned over a session. Configuration is in [Subagents](claude-code-configuration.md#subagents).

!!! note "Do subagents share memory? Answer in the guide's terms"

    CCAR-F 1.3-K2 says "subagents do not automatically inherit parent context or share memory between invocations." The `memory` field does not contradict it: persistence is opt-in, per subagent definition, and lives in files the subagent reads, not in a shared conversation. For an exam item, the default holds: whatever a subagent needs from the coordinator must be in its prompt (our reading of 1.3-K2 alongside the current docs).

!!! warning "Exam guide vs current docs"

    The CCAR-F guide says "The Task tool as the mechanism for spawning subagents, and the requirement that allowedTools must include "Task" for a coordinator to invoke subagents" (1.3-K1). Current [Claude Code subagents docs](https://code.claude.com/docs/en/sub-agents): "In version 2.1.63, the Task tool was renamed to Agent. Existing `Task(...)` references in settings and agent definitions still work as aliases." The [Agent SDK](https://code.claude.com/docs/en/agent-sdk/subagents) shows the tool as `"Agent"` in `tool_use` blocks but as `"Task"` in the `system:init` tools list, and its examples put `"Agent"` in `allowed_tools`. Expect "Task" on the exam, and recognize "Agent" as the same tool. Also changed since older material: "As of v2.1.198, Explore inherits the main conversation's model instead of always running on Haiku" (capped at Opus on the Claude API).

### Traps

- **Assuming the subagent can see what the coordinator found.** Not unless it is in the prompt string (1.3-K2). Forks are the exception.
- **Returning everything so nothing is lost.** Verbose returns re-flood the coordinator; ask for summaries or structured findings.
- **Delegating every exploration.** Quick, targeted lookups and tightly coupled phases belong in the main conversation.
- **Reading `context: fork` as a conversation fork.** It runs the skill in an isolated subagent that does not see the history.
- **Assuming Explore follows your CLAUDE.md.** Explore and Plan skip it; restate any rule they must follow in the delegation prompt.
- **Treating "Task" and "Agent" as different tools.** They are the same subagent-spawning tool under its old and current names.

## Exploring large codebases

*Tested in: CCAR-F 5.4 (5.4-K1 to 5.4-S5), 2.5-S4, 1.7-K3, 1.7-K4, 1.7-S3, 1.7-S4, Scenario 2 (Code Generation with Claude Code) and Scenario 4 (Developer Productivity with Claude) · CCDV-F D3.1 Claude Code Operation (session management), D2.5 Claude Application Design (session hygiene) · CCAR-P 7.1, 7.2*

CCAR-F Task 5.4 is "Manage context effectively in large codebase exploration". Codebase work fills context quickly: every file read, grep result and test log lands in the window. The guide describes the symptom precisely in 5.4-K1: "Context degradation in extended sessions: models start giving inconsistent answers and referencing "typical patterns" rather than specific classes discovered earlier". When an agent that named a specific class an hour ago starts describing generic patterns instead, that is the degradation the guide describes, and the same drift Claude Code's best practices warn about as the window fills ([More tokens, less attention](#more-tokens-less-attention)). Our reading, for the reason given in [A bigger window is not the fix](#a-bigger-window-is-not-the-fix): the remedy is less and better-organized context, not more.

The CCAR-F guide's five skills for this task form one workflow:

| Skill | Wording in the guide | Tool that implements it |
|---|---|---|
| 5.4-S1 | "Spawning subagents to investigate specific questions (e.g., "find all test files," "trace refund flow dependencies") while the main agent preserves high-level coordination" | Explore or a custom subagent |
| 5.4-S2 | "Having agents maintain scratchpad files recording key findings, referencing them for subsequent questions to counteract context degradation" | A notes file in the repo or scratch directory |
| 5.4-S3 | "Summarizing key findings from one exploration phase before spawning sub-agents for the next phase, injecting summaries into initial context" | The delegation prompt |
| 5.4-S4 | "Designing crash recovery using structured agent state exports (manifests) that the coordinator loads on resume and injects into agent prompts" | State files plus a manifest; see [Multi-agent handoffs](#multi-agent-handoffs) |
| 5.4-S5 | "Using /compact to reduce context usage during extended exploration sessions when context fills with verbose discovery output" | `/compact` with a focus |

### Explore incrementally, not all at once

CCAR-F 2.5-S4: "Building codebase understanding incrementally: starting with Grep to find entry points, then using Read to follow imports and trace flows, rather than reading all files upfront". This is the just-in-time strategy from [Up front or just in time](#up-front-or-just-in-time) applied to code; Anthropic's [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) gives Claude Code's glob and grep retrieval as its example of loading files just in time. Answer exam items with the guide's grep-then-read pattern. Related documented advice includes the [common-workflows](https://code.claude.com/docs/en/common-workflows) tip "Start with broad questions, then narrow down to specific areas". The anti-pattern has a name in the [best practices](https://code.claude.com/docs/en/best-practices), "The infinite exploration": you ask Claude to investigate without scoping it, and "Claude reads hundreds of files, filling the context." The fix: "Scope investigations narrowly or use subagents so the exploration doesn't consume your main context."

!!! warning "Exam guide vs current docs: the Grep tool"

    CCAR-F 2.5-S4 names the built-in Grep tool, and Scenario 4 lists both Grep and Glob. As of September 2026, per the [tools reference](https://code.claude.com/docs/en/tools-reference), on macOS, Linux and WSL Claude Code leaves Glob and Grep out of the default tool set, "and Claude searches with `find` and `grep` through the Bash tool instead"; `--tools` gives you the tools you list, and naming either one in `--allowedTools` restores both. The strategy is unchanged (search for entry points, then read along the flow), and the guide's selection logic (Grep for file contents, Glob for paths) still matches the tool descriptions. Answer in the guide's tool names.

Ways to cut the cost of each lookup, from the [large codebases docs](https://code.claude.com/docs/en/large-codebases) (as of September 2026):

- **Code intelligence.** "In a large codebase, finding where a symbol is defined or used can cost many file reads and grep calls." A code intelligence plugin gives Claude "go to definition" and "find references" navigation ([Common workflows](https://code.claude.com/docs/en/common-workflows)) through the LSP tool, which stays inactive until such a plugin is installed (and stays inactive in cloud sessions, where plugin language servers are not started).
- **An existing index.** If your organization runs code search or a RAG index, "expose it as an MCP tool so Claude queries it instead of reading files directly".
- **Keep generated and vendored code out.** `Read` deny rules in `.claude/settings.json` block Claude from opening checked-in generated and vendored files (Claude's content searches respect `.gitignore`, so gitignored paths such as `dist/` and `build/` already stay out of content-search results). A Bash search such as `grep -r` or `find` over a directory that contains denied files still includes them in its output. The directory patterns end with `/**/*` rather than `/**`, so each rule covers everything inside the directory but not the directory itself, and Claude can still list it or change into it:

```json
{
  "permissions": {
    "deny": [
      "Read(./**/dist/**/*)",
      "Read(./**/build/**/*)",
      "Read(./**/*.generated.*)",
      "Read(./**/vendor/**/*)"
    ]
  }
}
```

Monorepo configuration (where to start Claude, layered CLAUDE.md files, worktrees) is taught in [Managing context in large codebases](claude-code-workflows.md#managing-context-in-large-codebases).

### Keep a scratchpad file

CCAR-F 5.4-K2 names "The role of scratchpad files for persisting key findings across context boundaries". A scratchpad is structured note-taking applied to exploration: the findings live on disk, so compaction or a fresh session cannot blur them. Anthropic's examples include "your custom agent maintaining a NOTES.md file" ([Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)), a [Managed Agents cookbook](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/managed_agents/CMA_explore_unfamiliar_codebase.ipynb) system prompt that says "Write notes to /tmp/NOTES.md as you go.", and, in the [C compiler project](https://www.anthropic.com/engineering/building-c-compiler), instructions "to maintain extensive READMEs and progress files that should be updated frequently with the current status." An illustrative scratchpad for a refund-flow investigation:

```markdown
**Refund flow investigation (scratchpad)**

### Confirmed
- Entry point: api/refunds.py -> RefundService.create() (services/refund_service.py:41)
- RefundService calls PaymentGateway.reverse() only after OrderRepo.lock() succeeds
- Tests: tests/services/test_refund_service.py covers the happy path only

### Open questions
- Where are partial refunds rounded? (not in RefundService; check money/utils.py)

### Next
- Trace PaymentGateway.reverse() error handling
```

Two habits make scratchpads and logs searchable instead of noisy. From [Building a C compiler with a team of parallel Claudes](https://www.anthropic.com/engineering/building-c-compiler): "The test harness should not print thousands of useless bytes. At most, it should print a few lines of output and log all important information to a file so Claude can find it when needed." And: "if there are errors, Claude should write ERROR and put the reason on the same line so grep will find it."

### Delegate, summarize between phases, compact

- **Delegate the reading.** "Exploring a large codebase fills your context with file reads. Delegate the exploration so only the findings come back" ([Common workflows](https://code.claude.com/docs/en/common-workflows)). Isolation mechanics are in [Subagents as context isolation](#subagents-as-context-isolation).
- **Carry phase summaries forward.** Before the next wave of subagents, condense what the last wave found and put it in their initial prompts (5.4-S3); the handoff itself is taught in [Down: coordinator to subagent](#down-coordinator-to-subagent).
- **Compact with a focus.** `/compact focus on the auth bug fix` style instructions keep what the next step needs (mechanics in [Compaction in Claude Code and the Agent SDK](#compaction-in-claude-code-and-the-agent-sdk)); a plan written in plan mode is re-injected after each compaction, "so the plan survives where conversation history may not" ([large codebases docs](https://code.claude.com/docs/en/large-codebases)). Use `/context` to see what is consuming the window.
- **Clear between unrelated tasks.** The [best practices'](https://code.claude.com/docs/en/best-practices) fix for the "kitchen sink session" is "`/clear` between unrelated tasks."

A worked sequence for the guide's example question, "trace refund flow dependencies" (illustrative):

1. Plan mode or a read-only start; ask a scoped question, not *investigate the payments code*.
2. Delegate with a scoped prompt such as *Use a subagent to find every caller of RefundService.create and report file paths and line numbers only.*
3. Record the returned findings in the scratchpad file.
4. Delegate the next phase with the scratchpad's confirmed findings pasted into the prompt.
5. When `/context` shows the window filling with discovery output, run `/compact` with a focus on the confirmed findings and the open questions.
6. Answer later questions from the scratchpad, not from memory of earlier turns.

### Resuming after the code changed

Exploration can span several sessions. CCAR-F 1.7-K3 stresses "informing the agent about changes to previously analyzed files when resuming sessions after code modifications", and 1.7-K4 asks "Why starting a new session with a structured summary is more reliable than resuming with stale tool results". In the [Agent SDK sessions docs'](https://code.claude.com/docs/en/agent-sdk/sessions) words, "Sessions persist the conversation, not the filesystem", so a resumed session carries its earlier tool results exactly as they were read. Claude Code's [checkpoints](https://code.claude.com/docs/en/checkpointing) do not reliably capture outside edits either: "Manual changes you make to files outside of Claude Code and edits from other concurrent sessions are normally not captured, unless they happen to modify the same files as the current session." Switching branches changes the files Claude sees, "but your conversation history stays the same" ([How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)). The safety net is partial: when Claude edits a file that changed on disk since its last read, the [tools reference](https://code.claude.com/docs/en/tools-reference) says "the result notes that the file carries other changes so Claude re-reads it before edits that depend on surrounding content", but that happens only when an edit is attempted, not when Claude reasons from an earlier read. Telling the agent what changed stays your job.

| Situation | Choose (CCAR-F wording) |
|---|---|
| Prior context mostly still valid | Resume (1.7-S3) |
| A few files changed | Resume and name the changed files "for targeted re-analysis rather than requiring full re-exploration" (1.7-S4) |
| Earlier tool results are stale | Start fresh "with injected summaries" (1.7-S3); your scratchpad is that summary |

Claude Code builds a version of this choice into resuming (as of September 2026; [sessions docs](https://code.claude.com/docs/en/sessions)). On a Pro or Max plan, resuming a session that has been inactive for more than about an hour and is over 100,000 tokens opens a dialog: **Resume from summary** runs `/compact` immediately and keeps the summary, your most recent exchanges and up to five recently read files, while resuming as-is "keeps every detail of the conversation available, at a per-request cost that scales with the conversation's size." The dialog trades detail against cost; it is not a check for changed files. Our reading: when files changed, still name them (1.7-S4) or start fresh with a summary (1.7-S3).

Session commands (`--resume`, `--fork-session`, `/rewind`) are in [Sessions: continue, resume, fork and rewind](claude-code-workflows.md#sessions-continue-resume-fork-and-rewind).

### Traps

- **Reading everything up front "for completeness".** 2.5-S4 rejects it: find entry points first, then follow the flow.
- **Trusting the model's memory of turn 5 at turn 80.** Degradation shows as generic answers (5.4-K1); reference the scratchpad instead.
- **Delegating without the previous phase's findings.** The next subagent starts blind (5.4-S3, 1.3-K2).
- **Resuming a week-old session after a refactor without saying what changed.** Tell the agent, or start fresh with a summary (1.7-K3, 1.7-K4).

## Multi-agent handoffs

*Tested in: CCAR-F 1.3-K2, 1.3-S1, 1.3-S2, 1.4-K3, 1.4-S3, 1.7-K4, 5.1-S5, 5.1-S6, 5.4-K4, 5.4-S3, 5.4-S4, Appendix in-scope topic "Subagent context management", Exercise 4 steps 1 and 3 · CCDV-F D1.1 Agent Architecture, D1.3 Agent Patterns and Frameworks · CCAR-P 1.4, 1.5*

A handoff is any point where work crosses a context boundary: coordinator to subagent, subagent back to coordinator, one phase to the next, one session to the next, a crashed run to its restart, or an agent to a human. The receiver sees only what is handed over, so the question behind each crossing is the same (our framing of the guide objectives below): what must cross, in what shape, and where does it live if the receiver cannot hold it all?

| Handoff | What must cross | Guide objective |
|---|---|---|
| Coordinator to subagent | Complete findings from prior agents, directly in its prompt (the Agent SDK docs add any file paths, error messages or decisions the subagent needs) | 1.3-K2, 1.3-S1 |
| Phase to phase | A summary of the last phase's key findings, injected into the next subagents' initial context | 5.4-S3 |
| Subagent to coordinator | Structured data (key facts, citations, relevance scores) with metadata (dates, source locations, methodological context) kept separate from content | 5.1-S5, 5.1-S6, 1.3-S2 |
| Crashed run to its restart | Each agent's state, exported to a known location, plus a manifest the coordinator loads on resume and injects into agent prompts | 5.4-K4, 5.4-S4 |
| Session to session | A structured summary instead of stale tool results | 1.7-K4 |
| Agent to human | Customer ID, root cause, refund amount, recommended action | 1.4-K3, 1.4-S3 |

### Down: coordinator to subagent

The mechanism and what to write in the delegation prompt are taught in [Multi-agent orchestration](agents-and-agent-sdk.md#multi-agent-orchestration). The rule in one line: "subagents do not automatically inherit parent context or share memory between invocations" (1.3-K2), so 1.3-S1 asks for "Including complete findings from prior agents directly in the subagent's prompt (e.g., passing web search results and document analysis outputs to the synthesis subagent)". CCAR-F Exercise 4 step 1 asks you to ensure that "each subagent receives its research findings directly in its prompt rather than relying on automatic context inheritance."

Current [Agent SDK subagent docs](https://code.claude.com/docs/en/agent-sdk/subagents) (as of September 2026) agree that the prompt string is the only content passed from parent to subagent, and add one exception the guide does not mention: a fork inherits the parent conversation, while any other subagent's "context window starts fresh, with no parent conversation, but isn't empty." What that fresh window holds, forks, and the renaming of Task to Agent are covered in [Subagents as context isolation](#subagents-as-context-isolation); on the exam, answer in the guide's terms (context is passed explicitly in the prompt).

Between phases, send the distilled state, not the transcript. 5.4-S3: "Summarizing key findings from one exploration phase before spawning sub-agents for the next phase, injecting summaries into initial context". Anthropic's [multi-agent research post](https://www.anthropic.com/engineering/multi-agent-research-system) lists the same pattern among its tips for long-horizon conversation management: "We implemented patterns where agents summarize completed work phases and store essential information in external memory before proceeding to new tasks." And near the limit: "When context limits approach, agents can spawn fresh subagents with clean contexts while maintaining continuity through careful handoffs."

### Up: subagent to coordinator

Two CCAR-F 5.1 skills define the return shape:

- 5.1-S5: "Requiring subagents to include metadata (dates, source locations, methodological context) in structured outputs to support accurate downstream synthesis"
- 5.1-S6: "Modifying upstream agents to return structured data (key facts, citations, relevance scores) instead of verbose content and reasoning chains when downstream agents have limited context budgets"

And 1.3-S2 keeps content and metadata apart: "Using structured data formats to separate content from metadata (source URLs, document names, page numbers) when passing context between agents to preserve attribution". Exercise 4 step 3 names the fields: "each finding should include a claim, evidence excerpt, source URL/document name, and publication date". An illustrative return that meets all three skills and the exercise (field names and values are ours):

```json
{
  "subtask": "EU retail pricing trends, 2025 to 2026",
  "findings": [
    {
      "claim": "Average basket prices rose in 2025.",
      "evidence_excerpt": "verbatim sentence from the source",
      "relevance": 0.9,
      "source": {
        "url": "https://example.org/report.pdf",
        "document_name": "Retail Price Monitor 2025",
        "page": 14,
        "publication_date": "2026-02-10",
        "method": "survey of retailers"
      }
    }
  ],
  "gaps": [{"topic": "2026 Q2 data", "status": "source unavailable", "detail": "statistics portal timed out twice (access failure, not an empty result)"}],
  "conflicts": []
}
```

The `gaps` and `conflicts` fields are there for synthesis: coverage gaps feed the coverage annotations of 5.3-S4, and conflicting values travel annotated rather than resolved, as 5.6-S3 asks. Both are taught in [Provenance and uncertainty in synthesis](#provenance-and-uncertainty-in-synthesis).

Why not return prose? The receiver has its own budget. A coordinator that collects ten verbose reports has recreated the context problem that delegation was meant to solve (our reasoning; Claude Code's subagent docs give the same warning, quoted in [When to isolate and when not to](#when-to-isolate-and-when-not-to)). Keep reasoning chains inside the subagent, and send facts, citations and scores up (5.1-S6). How findings then survive synthesis is in [Provenance and uncertainty in synthesis](#provenance-and-uncertainty-in-synthesis); how failures are reported (failure type, attempted query, partial results, alternative approaches) is in [Error propagation in multi-agent systems](evaluation-and-reliability.md#error-propagation-in-multi-agent-systems).

### By value or by reference

1.3-S1 says to pass complete findings in the prompt. The appendix of Anthropic's [multi-agent research post](https://www.anthropic.com/engineering/multi-agent-research-system), a list of "additional miscellaneous tips for multi-agent systems", recommends a second channel for some outputs: "Subagents call tools to store their work in external systems, then pass lightweight references back to the coordinator." The post's tip opens with "Subagent output to a filesystem to minimize the ‘game of telephone.’" and says the pattern "prevents information loss during multi-stage processing and reduces token overhead from copying large outputs through conversation history." It names where it fits: "The pattern works particularly well for structured outputs like code, reports, or data visualizations where the subagent's specialized prompt produces better results than filtering through a general coordinator." In Anthropic's [three-agent harness](https://www.anthropic.com/engineering/harness-design-long-running-apps) (planner, generator, evaluator), the generator and evaluator negotiated each sprint contract through files: "Communication was handled via files: one agent would write a file, another agent would read it and respond either within that file or with a new file that the previous agent would read in turn."

**Decide (our rule, built from the sources above).** Pass by value (in the prompt) when the receiver needs the content to reason with: findings, decisions, error messages, file paths. Pass by reference (a path or ID plus a short summary) when the output is large or is itself the deliverable, such as code, a report or a data visualization. Either way the receiver gets enough in its prompt to know what the reference contains; a bare path is not a handoff. For exam items about what the synthesis subagent receives, the guide's answer is by value: complete findings directly in its prompt (1.3-S1).

### Crash recovery: state exports and a manifest

CCAR-F 5.4-K4: "Structured state persistence for crash recovery: each agent exports state to a known location, and the coordinator loads a manifest on resume". 5.4-S4 completes it: "Designing crash recovery using structured agent state exports (manifests) that the coordinator loads on resume and injects into agent prompts". "Manifest" as a crash-recovery artifact is the guide's term, and none of the posts or docs in this table uses the word in that sense. They describe related ways to keep state outside a single context window or process (product details as of September 2026):

| Source | How state is kept outside the window or process |
|---|---|
| [Research system](https://www.anthropic.com/engineering/multi-agent-research-system) | "Instead, we built systems that can resume from where the agent was when the errors occurred." The lead agent saved its plan to Memory because, in the system the post describes, "if the context window exceeds 200,000 tokens it will be truncated" |
| [Managed Agents](https://www.anthropic.com/engineering/managed-agents) | The session log lives outside the harness, so a new harness "can be rebooted with wake(sessionId) , use getSession(id) to get back the event log, and resume from the last event." |
| [Long-running harness](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) | A fresh window recovers state from "the claude-progress.txt file alongside the git history" |
| [Claude Code workflows](https://code.claude.com/docs/en/workflows) | Runs are resumable in the same session because "The runtime tracks each agent's result as the run progresses"; on relaunch a completed agent "returns its saved result" unless its prompt or an earlier agent's prompt changed, or an agent that started before it failed |
| [Agent SDK](https://code.claude.com/docs/en/agent-sdk/sessions) (moving work to another host) | Instead of relying on session resume: "Capture the results you need (analysis output, decisions, file diffs) as application state and pass them into a fresh session's prompt." The docs rate this as often more dependable than shipping transcript files around |

An illustrative manifest (field names are ours):

```json
{
  "run_id": "research-2026-09-23-a",
  "phase": "analysis",
  "agents": [
    {"name": "web_search", "status": "completed", "state_file": "state/web_search.json"},
    {"name": "doc_analysis", "status": "failed", "state_file": "state/doc_analysis.json", "last_error": "timeout"},
    {"name": "synthesis", "status": "not_started"}
  ]
}
```

On resume the coordinator loads the manifest and injects the exported state into agent prompts, which is what 5.4-S4 describes; skipping completed agents and restarting only the failed one is this example's design, not the guide's wording. ([Claude Code workflows](https://code.claude.com/docs/en/workflows) replay less selectively: a completed agent returns its saved result only if its prompt and every earlier agent's prompt are unchanged, and a failed agent "runs again, and so does every agent that started after it, even ones that completed.") Keep state files structured, in JSON as Anthropic's long-running harness does, for the reason quoted in [Long-running agents](#long-running-agents-notes-resets-and-harness-files). One product caveat: an Agent SDK `SessionStore` mirrors transcripts only, "not `CLAUDE.md` memory files or other working-directory artifacts"; the [hosting docs](https://code.claude.com/docs/en/agent-sdk/hosting) say to "Mount a shared volume or sync those separately."

### Session to session: resets with a structured handoff

A context reset is a deliberate handoff to a fresh agent. Anthropic's [harness design post](https://www.anthropic.com/engineering/harness-design-long-running-apps) describes it as clearing the window entirely and starting a fresh agent, "combined with a structured handoff that carries the previous agent's state and the next steps". The trade-off against compaction, and how it changed between model generations, is in the warning in [Long-running agents](#long-running-agents-notes-resets-and-harness-files). The price of a reset: "A reset provides a clean slate, at the cost of the handoff artifact having enough state for the next agent to pick up the work cleanly." The same post notes that resets add "orchestration complexity, token overhead, and latency to each harness run."

The exam frames the same choice at the session level: 1.7-K4 prefers a new session with a structured summary over resuming with stale tool results, and the Agent SDK gives matching advice for work that moves between hosts (crash-recovery table above). When to resume and when to start fresh (1.7-S3) is decided in [Resuming after the code changed](#resuming-after-the-code-changed). A good session handoff reads like the harness files (our summary): what is done and verified, what is in progress, what is next, and where the evidence lives. Anthropic's prompting guide gives this example of free-text progress notes:

```text
// Progress notes (progress.txt)
Session 3 progress:
- Fixed authentication token validation
- Updated user model to handle edge cases
- Next: investigate user_management test failures (test #2)
- Note: Do not remove tests as this could lead to missing functionality
```

### Agent to human

A human who takes over an escalation has not seen the transcript. CCAR-F 1.4-K3 asks for "Structured handoff protocols for mid-process escalation that include customer details, root cause analysis, and recommended actions", and 1.4-S3 names the fields: "Compiling structured handoff summaries (customer ID, root cause, refund amount, recommended action) when escalating to human agents who lack access to the conversation transcript". That four-field schema is the guide's own; the closest first-party structure is Anthropic's [customer-escalation skill](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/customer-support/skills/customer-escalation/SKILL.md), which packages an issue into "a structured escalation brief for engineering, product, or leadership", with a section for what has been tried ("Any troubleshooting or workarounds attempted") and the rule "Always quantify impact". When to escalate, and how the guide's fields map onto that brief, are in [Escalation and ambiguity](evaluation-and-reliability.md#escalation-and-ambiguity).

### Traps

- **Assuming the synthesis agent can look up what the search agent found.** It sees only what you pass in its prompt (1.3-K2, Exercise 4 step 1).
- **Returning verbose content and reasoning chains to a budget-limited coordinator.** 5.1-S6 says return structured data: key facts, citations, relevance scores.
- **Flattening metadata into prose.** Keep source URLs, document names and page numbers separate from content to preserve attribution (1.3-S2), and include dates and methodological context (5.1-S5).
- **Restarting a crashed pipeline from zero.** Each agent exports state to a known location, and the coordinator resumes from a manifest (5.4-K4, 5.4-S4).
- **Escalating with a pointer to the conversation.** The human lacks access to the transcript; send the structured summary (1.4-S3).

## Provenance and uncertainty in synthesis

*Tested in: CCAR-F 5.6 (5.6-K1 to 5.6-S5), 5.3-S4, 5.5-S4, 1.3-S2, Appendix in-scope topic "Information provenance", Exercise 4 steps 3 to 5, sample question 8, Scenario 3 (Multi-Agent Research System) · CCDV-F D6.3 Output Handling ("skepticism toward confident output") · CCAR-P 4.4 (sample 3), 5.3 · CCAO-F D2.1, D2.3 (sample 1)*

CCAR-F Task 5.6 is "Preserve information provenance and handle uncertainty in multi-source synthesis". The failure it guards against is in 5.6-K1: "How source attribution is lost during summarization steps when findings are compressed without preserving claim-source mappings". Every compression step in [Multi-agent handoffs](#multi-agent-handoffs) is a chance to drop the link between a claim and its evidence; once dropped, it can be rebuilt only by going back to the source documents (our reasoning; that is the job Anthropic gives a separate citations agent, below). The guide's Appendix in-scope topic list names four parts: "Information provenance: claim-source mappings, temporal data handling, conflict annotation, coverage gap reporting".

### Rule 1: every claim keeps its source, through every step

5.6-S1: "Requiring subagents to output structured claim-source mappings (source URLs, document names, relevant excerpts) that downstream agents preserve through synthesis". 5.6-K2 adds that the synthesis agent "must preserve and merge" those mappings when combining findings. Exercise 4 asks you to "Verify that the synthesis subagent preserves source attribution when combining findings." The record shape is in [Up: subagent to coordinator](#up-subagent-to-coordinator). Anthropic's open-source [research subagent prompt](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_subagent.md) asks for the same bookkeeping at the source: "For important facts, especially numbers and dates:" it says to "Keep track of findings and sources".

Anthropic's [multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) shows one production answer: once enough information is gathered, the system exits the research loop and "passes all findings to a CitationAgent, which processes the documents and research report to identify specific locations for citations." The post points readers to "the open-source prompts in our Cookbook for example prompts from our system", and those prompts ([lead agent](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_lead_agent.md), [citations agent](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/citations_agent.md)) make the division explicit:

- Lead agent: "Do not include ANY Markdown citations, a separate agent will be responsible for citations."
- Citations agent: "ONLY add citations where the source documents directly support claims in the text"
- Citations agent: "Do NOT modify the `<synthesized_text>` in any way - keep all content 100% identical, only add citations"

The [research system's](https://www.anthropic.com/engineering/multi-agent-research-system) LLM-judge rubric checked both directions: "factual accuracy (do claims match sources?), citation accuracy (do the cited sources match the claims?)".

**API features that carry provenance.** The [Citations](https://platform.claude.com/docs/en/build-with-claude/citations) feature returns "the exact passages that support each claim", and "citations are guaranteed to contain valid pointers to the provided documents." `search_result` blocks let Claude cite your own retrieval results with the `source` and `title` you supply. Both are taught in [Files, citations and search results](claude-api.md#files-citations-and-search-results). Three constraints matter for pipeline design (the consequences column is our reading):

| Constraint (as of September 2026) | Consequence |
|---|---|
| Citations and structured outputs cannot be combined: with citations enabled on any `document` or `search_result` block, also sending `output_config.format` (or the deprecated `output_format`) means "the API returns a 400 error" | A JSON claim-source record produced with structured outputs must carry its own source fields (URL, excerpt, date) in the schema; it cannot lean on API citations in the same call |
| Citations are all or nothing: "citations must be enabled on all or none of the documents within a request" | Decide per request, not per document |
| Granularity depends on the input. Plain text and PDF documents "are automatically chunked into sentences"; custom content documents use your blocks "as-is"; for search results, "The text block is the minimal citable unit" | Split retrieved content into smaller blocks when you need finer citations. On a `document`, put metadata (date, method) in the optional `context` field: `title` and `context` "are passed to the model but not used toward cited content" |

### Rule 2: annotate conflicts; do not pick a winner

5.6-K3: "How to handle conflicting statistics from credible sources: annotating conflicts with source attribution rather than arbitrarily selecting one value". 5.6-S3 sets who decides: "Completing document analysis with conflicting values included and explicitly annotated, letting the coordinator decide how to reconcile before passing to synthesis". Exercise 4 tests it with "two credible sources with different statistics" and checks that synthesis "preserves both values with source attribution rather than arbitrarily selecting one".

Anthropic's own prompts differ from the guide on who reconciles first. The cookbook [research subagent](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_subagent.md) reconciles on its own (recency, consistency, source quality) and hands the conflict up only if it cannot: "When encountering conflicting information, prioritize based on recency, consistency with other facts, the quality of the sources used, and use your best judgment and reasoning. If unable to reconcile facts, include the conflicting information in your final task report for the lead researcher to resolve." 5.6-S3, by contrast, has document analysis include and annotate the conflicting values and leave reconciliation to the coordinator. On the exam, a subagent that settles the conflict itself by silently keeping the newer value is the first trap below. The [lead agent](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_lead_agent.md) is told to "Note any discrepancies you observe between sources or issues with the quality of sources." The next line of its prompt has it resolve them: "When encountering conflicting information, prioritize based on recency, consistency with other facts, and use best judgment." So in Anthropic's system the coordinator reconciles whatever reaches it, which is the part that matches 5.6-S3 ("letting the coordinator decide how to reconcile"). For the exam, the synthesis output still keeps both values with source attribution (Exercise 4 step 5). Anthropic's [knowledge-synthesis skill](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/enterprise-search/skills/knowledge-synthesis/SKILL.md) (enterprise-search plugin) states the rule flatly: "Always surface conflicts rather than silently picking one version."

An illustrative annotation (field names are ours):

```json
{
  "metric": "2025 market growth",
  "values": [
    {"value": "12%", "source": "Industry Association annual report", "published": "2026-01-15", "method": "member survey"},
    {"value": "8%", "source": "National statistics office release", "published": "2026-03-02", "method": "census of firms"}
  ],
  "conflict": true,
  "possible_explanation": "different methods and populations",
  "status": "unresolved; passed to coordinator"
}
```

### Rule 3: dates prevent false contradictions

5.6-K4: "Temporal data: requiring publication/collection dates in structured outputs to prevent temporal differences from being misinterpreted as contradictions". Two figures a year apart may both be right. 5.6-S4 makes it a subagent requirement: "Requiring subagents to include publication or data collection dates in structured outputs to enable correct temporal interpretation". The required date field is the guide's prescription; no other Anthropic doc, post or prompt cited on this page requires subagents to output one. Anthropic's own material supports the same habit: the cookbook research prompts ([subagent](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_subagent.md), [lead agent](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_lead_agent.md)) inject the current date ("The current date is {{.CurrentDate}}."), the lead agent is told to "Note any temporal or contextual constraints on the question.", the [knowledge-synthesis skill's](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/enterprise-search/skills/knowledge-synthesis/SKILL.md) attribution rules say "Include the date or relative time" and it keeps items separate rather than deduplicating them when "Different time periods are represented", and [web search](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool) results carry `page_age`, "When the site was last updated" (as of September 2026).

### Rule 4: separate the established from the contested

5.6-S2: "Structuring reports with explicit sections distinguishing well-established findings from contested ones, preserving original source characterizations and methodological context". In practice (our reading), "preserving original source characterizations" means a source that said "may" stays "may". The cookbook [subagent prompt](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_subagent.md) makes the same demand for speculation (predictions, "could" or "may" language, financial projections): "you should make sure to note this explicitly in the final report, rather than accepting these events as having happened", and asks subagents to "flag these issues when returning your report to the lead researcher rather than blindly presenting all results as established facts."

### Rule 5: report coverage gaps

5.3-S4: "Structuring synthesis output with coverage annotations indicating which findings are well-supported versus which topic areas have gaps due to unavailable sources". In Exercise 4 the coordinator must "proceed with partial results and annotate the final output with coverage gaps" after a simulated subagent timeout. This depends on errors arriving as structured context rather than as empty success. In [sample question 8](../claude-certified-architect-foundations.md#official-sample-questions) the correct design returns "structured error context to the coordinator including the failure type, the attempted query, any partial results, and potential alternative approaches", and the rationale rejects the empty-result option because it "suppresses the error by marking failure as success, which prevents any recovery and risks incomplete research outputs." Error reporting itself is taught in [Error propagation in multi-agent systems](evaluation-and-reliability.md#error-propagation-in-multi-agent-systems).

"Coverage annotation" is the guide's term. Closest product behavior (as of September 2026): Claude Code's bundled `/deep-research` [workflow](https://code.claude.com/docs/en/workflows) "votes on each claim, and returns a cited report with claims that didn't survive cross-checking filtered out", and "When the verifier agents can't check a claim, such as after a rate limit or API error, the report lists that claim as unverified instead of counting it as refuted." Unverified and refuted are different statuses. So are *no source found* and *source unavailable*: the guide separates "access failures (timeouts needing retry decisions)" from "valid empty results (successful queries with no matches)" (5.3-K2). Anthropic's [customer-research skill](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/customer-support/skills/customer-research/SKILL.md) gives its research brief a "Gaps & Unknowns" section (what could not be confirmed, and what might need verification from a subject matter expert), and Anthropic's [legal summarization guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/legal-summarization) tells Claude to note information that is not explicitly stated as "Not specified".

### Rule 6: render each content type in its own form

5.6-S5: "Rendering different content types appropriately in synthesis outputs" (financial data as tables, news as prose, technical findings as structured lists) "rather than converting everything to a uniform format". The pairing of content types with formats is the guide's; the Anthropic sources cited here do not spell it out. Anthropic's cookbook [lead agent](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_lead_agent.md) asks the same underlying question: "Would it need to be a detailed report, a list of entities, an analysis of different perspectives, a visual report, or something else?"

Putting the rules together, an illustrative report skeleton:

```markdown
### Summary (key findings first, each with source IDs)
### Well-established findings
### Contested findings (both values, sources, dates, methods)
### Coverage gaps (topics with no data, sources that were unavailable)
### Financial data (table)
### Sources (ID, title, URL, publication date)
```

### Uncertainty that should reach a human

When sources contradict each other and the pipeline cannot reconcile them, the decision may not belong to a model at all. CCAR-F 5.5-S4: "Routing extractions with low model confidence or ambiguous/contradictory source documents to human review, prioritizing limited reviewer capacity". Review design (sampling, calibration) is in [Human review and confidence calibration](evaluation-and-reliability.md#human-review-and-confidence-calibration).

Be careful with the "low model confidence" half. The guide calls "self-reported confidence scores" "unreliable proxies for actual case complexity" (5.2-K3), and two official rationales reject raw self-rating: CCAR-F [sample question 3](../claude-certified-architect-foundations.md#official-sample-questions) ("LLM self-reported confidence is poorly calibrated") and CCAO-F sample 1 ("Self-reported confidence (A, C) is not a reliable accuracy signal"). Two other objectives route on confidence, and only one says how to calibrate: 4.6-S3 has the model self-report confidence "alongside each finding to enable calibrated review routing", while 5.5-S3 calibrates first: "Having models output field-level confidence scores, then calibrating review thresholds using labeled validation sets". The consistent reading (ours): confidence is a routing signal only after it has been checked against labeled outcomes for your task, and it never replaces checking claims against sources.

Prompt-level techniques that keep synthesis honest, from Anthropic's [Reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations) page:

- "Explicitly give Claude permission to admit uncertainty."
- For long documents (over 20k tokens), extract word-for-word quotes first, then do the task.
- Verify with citations: find a supporting quote for each claim; "If it can't find a quote, it must retract the claim."
- "Explicitly instruct Claude to only use information from provided documents and not its general knowledge."

For research-style work the [prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) adds "Ask Claude to verify information across multiple sources.", and its sample prompt for complex research asks Claude to "develop several competing hypotheses" and "Track your confidence levels in your progress notes to improve calibration." More in [Reducing hallucinations](prompt-engineering.md#reducing-hallucinations).

Bad sources produce confident errors too. The Architect, Professional exam's [sample 3](../claude-certified-architect-professional.md#official-sample-questions) (tagged to Domain 4; in our mapping it exercises objective 4.4, "Diagnose system issues (prompt failure, hallucinations, model mismatch)") describes a RAG system that "suddenly returns confident but incorrect answers after a document refresh"; the answer points to "retrieval feeding the model poor context, for example a broken re-index or mismatched embeddings."

**For business users (CCAO-F).** The same discipline applies in the Claude apps: "Evaluate Claude-generated outputs for accuracy and completeness" (D2.1) and "Apply fact-checking and validation techniques" (D2.3). The Associate exam's [sample 1](../claude-certified-associate.md#official-sample-questions) tests it: Claude produces a confident summary of a new regulation that cites a specific subsection number, and the answer is "Verify the cited subsection against the official regulation text before sharing." The rationale: "Language models can fabricate specific-looking details such as citation numbers, a hallucination." Claude Academy's tutorial [Why do AI models hallucinate?](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate) gives a practical check: "If you have an answer you're unsure about, start a new chat and ask the AI to find errors in the answer, and to confirm that the sources support the statements." See [Verifying Claude's output](claude-for-work.md#verifying-claudes-output).

!!! note "Where the guide's wording comes from"

    Several prescriptions in this section are the exam guide's own wording rather than named product features: coverage annotations (5.3-S4), required publication or collection dates in subagent outputs (5.6-K4, 5.6-S4), and rendering financial data as tables, news as prose and technical findings as structured lists (5.6-S5). The Anthropic sources above support related behaviors (unverified versus refuted claims, current-date injection, recency and temporal constraints, choosing the answer's form). Learn the guide's phrasing for the exam; use the product features to build it.

### Traps

- **Silently keeping one of two conflicting values** (the newest, or the one from the more official-looking source). 5.6-K3 and 5.6-S3: include both, annotate them with their sources, and let the coordinator decide how to reconcile.
- **Treating a year-apart difference as a contradiction.** Require publication or collection dates (5.6-K4, 5.6-S4).
- **A polished summary with the sources stripped out.** This is how attribution is lost (5.6-K1); keep claim-source mappings through every step (5.6-S1).
- **Returning an empty result as success when a source failed.** Sample question 8's rationale: it "prevents any recovery and risks incomplete research outputs", and the gap never reaches the coverage annotations (5.3-S4).
- **Trusting a confident answer or a high self-rating.** Self-reported confidence "is not a reliable accuracy signal" ([CCAO-F sample 1](../claude-certified-associate.md#official-sample-questions)) and is an unreliable proxy for case complexity (5.2-K3); verify claims against sources.
- **Converting every content type to one format.** 5.6-S5 rejects a uniform format.

## Exam map

Which objective of which exam each section serves, taken from the four exam guides (each Version 1.0, effective July 2026); the section-to-objective mapping is ours. The CCAR-F guide numbers its task statements (5.1 is "Task Statement 5.1") but not the bullets under them, so in IDs such as 5.1-K3 the suffix is our numbering: the third "Knowledge of" bullet (S for "Skills in"), counted in the order the guide prints them. The CCDV-F guide lists its skills as unnumbered headings under each domain, and the CCAR-P and CCAO-F guides list their objectives as unnumbered bullets, so IDs such as D6.1, 2.4 and D3.4 are also our numbering, in printed order. Appendix topics and exercises are named as the CCAR-F guide names them.

| Section | CCAO-F | CCDV-F | CCAR-F | CCAR-P |
|---|---|---|---|---|
| [Why context is a budget](#why-context-is-a-budget) | D3.4 | D5.1, D5.4, D6.1 | 5.1-K3, 5.1-K4, 1.6-S2; sample question 12; Appendix technology list ("Context window management") | 2.4, 3.8, 4.5; sample 2 |
| [Preserving critical information in long conversations](#preserving-critical-information-in-long-conversations) | D3.4 | D6.1 | 5.1-K1, 5.1-K2, 5.1-S1, 5.1-S2, 5.1-S3, 5.1-S4, 1.5-K1; Appendix in-scope topic "Context window optimization"; Scenario 1 | 2.4 |
| [Compaction, context editing and memory](#compaction-context-editing-and-memory) | D3.4 | D1.3, D3.1, D6.1 | 5.4-K2, 5.4-S2, 5.4-S5; Appendix technology list (`/compact`, `/memory`, "Context window management") | 2.4, 4.5 |
| [Subagents as context isolation](#subagents-as-context-isolation) | Not listed | D1.1, D1.3, D6.1 | 1.2-K2, 1.3-K1, 1.3-K2, 3.2-K3, 3.2-S2, 3.4-K4, 3.4-S3, 5.4-K3, 5.4-S1; Appendix technology list (Task tool, Explore subagent) | 1.4, 3.8 |
| [Exploring large codebases](#exploring-large-codebases) | Not listed | D2.5, D3.1 | 5.4-K1 to 5.4-S5, 2.5-S4, 1.7-K3, 1.7-K4, 1.7-S3, 1.7-S4; Scenarios 2 and 4 | 7.1, 7.2 |
| [Multi-agent handoffs](#multi-agent-handoffs) | Not listed | D1.1, D1.3 | 1.3-K2, 1.3-S1, 1.3-S2, 1.4-K3, 1.4-S3, 1.7-K4, 5.1-S5, 5.1-S6, 5.4-K4, 5.4-S3, 5.4-S4; Appendix in-scope topic "Subagent context management"; Exercise 4 steps 1 and 3 | 1.4, 1.5 |
| [Provenance and uncertainty in synthesis](#provenance-and-uncertainty-in-synthesis) | D2.1, D2.3; sample 1 | D6.3 | 5.6-K1 to 5.6-S5, 5.3-S4, 5.5-S4, 1.3-S2; Appendix in-scope topic "Information provenance"; Exercise 4 steps 3 to 5; sample question 8; Scenario 3 | 4.4 (sample 3), 5.3 |

The objectives referenced above, by exam:

| Exam | ID | Objective (as printed in the guide) |
|---|---|---|
| CCAO-F | D2.1 | Evaluate Claude-generated outputs for accuracy and completeness |
| CCAO-F | D2.3 | Apply fact-checking and validation techniques |
| CCAO-F | D3.4 | Understand and manage context limitations and memory considerations (when to restart, summarize, or persist) |
| CCDV-F | D1.1 | Agent Architecture (4.5%) |
| CCDV-F | D1.3 | Agent Patterns and Frameworks (4.9%) |
| CCDV-F | D2.5 | Claude Application Design (8.6%) |
| CCDV-F | D3.1 | Claude Code Operation (3.1%) |
| CCDV-F | D5.1 | LLM Fundamentals (5.2%) |
| CCDV-F | D5.4 | Cost and Token Management (2.8%) |
| CCDV-F | D6.1 | Context Engineering (3.8%) |
| CCDV-F | D6.3 | Output Handling (2.6%) |
| CCAR-F | 1.2 | Task Statement 1.2: Orchestrate multi-agent systems with coordinator-subagent patterns |
| CCAR-F | 1.3 | Task Statement 1.3: Configure subagent invocation, context passing, and spawning |
| CCAR-F | 1.4 | Task Statement 1.4: Implement multi-step workflows with enforcement and handoff patterns |
| CCAR-F | 1.5 | Task Statement 1.5: Apply Agent SDK hooks for tool call interception and data normalization |
| CCAR-F | 1.6 | Task Statement 1.6: Design task decomposition strategies for complex workflows |
| CCAR-F | 1.7 | Task Statement 1.7: Manage session state, resumption, and forking |
| CCAR-F | 2.5 | Task Statement 2.5: Select and apply built-in tools (Read, Write, Edit, Bash, Grep, Glob) effectively |
| CCAR-F | 3.2 | Task Statement 3.2: Create and configure custom slash commands and skills |
| CCAR-F | 3.4 | Task Statement 3.4: Determine when to use plan mode vs direct execution |
| CCAR-F | 5.1 | Task Statement 5.1: Manage conversation context to preserve critical information across long interactions |
| CCAR-F | 5.3 | Task Statement 5.3: Implement error propagation strategies across multi-agent systems |
| CCAR-F | 5.4 | Task Statement 5.4: Manage context effectively in large codebase exploration |
| CCAR-F | 5.5 | Task Statement 5.5: Design human review workflows and confidence calibration |
| CCAR-F | 5.6 | Task Statement 5.6: Preserve information provenance and handle uncertainty in multi-source synthesis |
| CCAR-P | 1.4 | Design multi-agent systems and orchestration strategies |
| CCAR-P | 1.5 | Apply decomposition techniques for complex problem solving |
| CCAR-P | 2.4 | Optimize context windows and manage token usage |
| CCAR-P | 3.8 | Evaluate progressive discovery vs. monolithic context strategy |
| CCAR-P | 4.4 | Diagnose system issues (prompt failure, hallucinations, model mismatch) |
| CCAR-P | 4.5 | Optimize token usage, latency, and cost-performance trade-offs |
| CCAR-P | 5.3 | Apply human-in-the-loop validation strategies |
| CCAR-P | 7.1 | Configure Claude tools and environments for teams (e.g., Claude Code) |
| CCAR-P | 7.2 | Improve developer workflows using AI-assisted tooling |

CCAR-F Domain 5, "Context Management & Reliability", carries 15% of that exam; its other task statements are taught mainly on the reliability page: 5.2 in [Escalation and ambiguity](evaluation-and-reliability.md#escalation-and-ambiguity), 5.3 in [Error propagation in multi-agent systems](evaluation-and-reliability.md#error-propagation-in-multi-agent-systems) and 5.5 in [Human review and confidence calibration](evaluation-and-reliability.md#human-review-and-confidence-calibration). CCDV-F's Context Engineering skill sits in Domain 6, "Prompt and Context Engineering" (11.0%). CCAR-P objective 2.4 sits in Domain 2, "Claude Models, Prompting & Context Engineering" (13%), and 3.8 in Domain 3, "Integration" (19%). CCAO-F D3.4 sits in Domain 3, "Product and Model Selection" (12%).

??? info "Sources"

    - [CCAR-F exam guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): Domain 5 and task statements 1.2 to 1.7, 2.5, 3.4, 5.1 to 5.6, sample questions 8 and 12, Exercise 4, the Appendix technology and in-scope lists, the Task tool wording
    - [CCDV-F exam guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf): skills D1.1, D1.3, D2.5, D3.1, D5.1, D5.4, D6.1, D6.3 and their weights
    - [CCAR-P exam guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf): objectives 1.4, 1.5, 2.4, 3.8, 4.4, 4.5, 5.3, 7.1, 7.2 and domain weights
    - [CCAO-F exam guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf): objectives D2.1, D2.3, D3.4 and the Domain 3 weight
    - [Context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows): definition of the window, context rot, what counts, cached prefixes, thinking blocks, overflow behavior, context awareness tags, compaction as the primary strategy
    - [Task budgets](https://platform.claude.com/docs/en/build-with-claude/task-budgets): beta header, `task_budget` fields, 20,000-token minimum, soft limit, countdown visibility, interaction with compaction
    - [Stop reasons and fallback](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons): treating `model_context_window_exceeded` as truncated
    - [Using the Messages API](https://platform.claude.com/docs/en/build-with-claude/working-with-messages): the API is stateless and takes the full history on every request
    - [Compaction overview](https://platform.claude.com/docs/en/build-with-claude/compaction): the two kinds of server-side compaction and the advice to use on-demand compaction where available
    - [Compaction on demand](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand): request and response shapes, Python and TypeScript examples, swap rules, silent mistakes, content lost after the swap, platforms, tool runner method, custom instructions example
    - [Compaction at a token threshold](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold): `compact_20260112`, trigger defaults, `pause_after_compaction`, dropping blocks before the compaction block, billing, caching, same-model and tool-call limitations
    - [Compaction that keeps recent turns](https://platform.claude.com/docs/en/build-with-claude/compaction-keep-recent-turns): keep-tail compaction and where to cut
    - [Compaction in the background](https://platform.claude.com/docs/en/build-with-claude/compaction-background): async compaction
    - [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing): tool result and thinking clearing, option defaults, ordering rule, response reporting, cache interaction, SDK compaction deprecation, cases less suited to compaction
    - [Memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool): configuration, commands, injected protocol, path validation, SDK helpers, pairing with compaction, multi-session pattern
    - [Using agent memory (Managed Agents)](https://platform.claude.com/docs/en/managed-agents/memory): memory stores, versions and the 10,000-memory limit
    - [Multiagent orchestration (Managed Agents)](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration): context isolation between session threads
    - [Define outcomes (Managed Agents)](https://platform.claude.com/docs/en/managed-agents/define-outcomes): grader in a separate context window
    - [Manage tool context](https://platform.claude.com/docs/en/agents-and-tools/tool-use/manage-tool-context): caching does not reduce tokens in context, when to add tool search, programmatic tool calling keeps intermediate results out of history
    - [Tool search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool): definition token cost of a five-server MCP setup, reduction from tool search, selection accuracy beyond 30 to 50 tools
    - [Web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool): the `page_age` field
    - [Citations](https://platform.claude.com/docs/en/build-with-claude/citations): exact supporting passages, valid pointers, all-or-none enablement, the 400 error with `output_config.format`, the `context` field
    - [Search results](https://platform.claude.com/docs/en/build-with-claude/search-results): `search_result` blocks and the minimal citable unit
    - [Reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations): permission to admit uncertainty, quote extraction, citation verification, restricting to provided documents
    - [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices): long-document placement and the 30 percent result, document tags, quote first, prefill no longer supported, multi-window harness guidance, context-limit prompt, state files, research tips, over-spawning subagents
    - [Prompting Claude Fable 5.1](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-1): client-side compaction summary instruction and later compaction points
    - [Prompting Claude Fable 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5): auditing progress claims, avoiding budget counts
    - [Prompting Claude Opus 5.5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5-5): unattended runs and continuation caps
    - [Legal summarization use case guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/legal-summarization): marking absent information "Not specified"
    - [Best practices for Claude Code](https://code.claude.com/docs/en/best-practices): the context constraint, forgetting as the window fills, the infinite-exploration and kitchen-sink failure patterns, `/clear` between tasks and after repeated corrections, subagents for investigation, fresh-context reviewers
    - [How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works): clearing tool outputs before summarizing, Compact Instructions, what early instructions lose, stopping repeated auto-compaction, branch switching
    - [Explore the context window (Claude Code)](https://code.claude.com/docs/en/context-window): what survives compaction, focused `/compact`, `/context`, clearing between tasks
    - [Commands](https://code.claude.com/docs/en/commands): `/compact`, `/context`, `/clear`, `/autocompact`, `/btw`, `/memory`
    - [Checkpointing](https://code.claude.com/docs/en/checkpointing): rewind summarize options, outside edits not captured
    - [Manage sessions (Claude Code)](https://code.claude.com/docs/en/sessions): the resume-from-summary dialog for long, inactive sessions
    - [Model configuration](https://code.claude.com/docs/en/model-config): default auto-compact point for native 1M models, the allowed window range, ultracode
    - [Environment variables](https://code.claude.com/docs/en/env-vars): auto-compact variables, `MAX_MCP_OUTPUT_TOKENS`, `BASH_MAX_OUTPUT_LENGTH`
    - [Hooks reference](https://code.claude.com/docs/en/hooks): `PreCompact`, `PostCompact`, `SessionStart` matchers
    - [Automate actions with hooks](https://code.claude.com/docs/en/hooks-guide): re-injecting context after compaction
    - [Troubleshooting](https://code.claude.com/docs/en/troubleshooting): the autocompact thrashing error and recovery steps
    - [Manage costs effectively](https://code.claude.com/docs/en/costs): compact instructions heading, hook pre-filtering, cost of `/compact` versus `/clear`
    - [How Claude Code uses prompt caching](https://code.claude.com/docs/en/prompt-caching): compaction invalidates the conversation cache layer
    - [How Claude remembers your project (Claude Code)](https://code.claude.com/docs/en/memory): CLAUDE.md persists across compaction; conversation-only instructions do not
    - [Create custom subagents (Claude Code)](https://code.claude.com/docs/en/sub-agents): fresh context, forks, Explore and Plan, choosing between main conversation and subagents, depth and concurrency limits, subagent compaction and memory, the Task to Agent rename, Explore model change
    - [Extend Claude with skills (Claude Code)](https://code.claude.com/docs/en/skills): `context: fork` runs a skill in an isolated subagent
    - [Common workflows](https://code.claude.com/docs/en/common-workflows): start broad then narrow, delegate exploration to subagents
    - [Set up Claude Code in a monorepo or large codebase](https://code.claude.com/docs/en/large-codebases): `Read` deny rules, code intelligence, code search as an MCP tool, plan file re-injection
    - [Tools reference](https://code.claude.com/docs/en/tools-reference): Glob and Grep left out of the default tool set, the LSP tool and code intelligence plugins, the note when an edited file changed on disk
    - [Orchestrate subagents at scale with dynamic workflows](https://code.claude.com/docs/en/workflows): intermediate results kept out of context, resumable runs, `/deep-research` verification behavior
    - [Extend Claude Code](https://code.claude.com/docs/en/features-overview): context cost of each extension type
    - [How the agent loop works (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/agent-loop): context accumulation, automatic compaction and `compact_boundary`, CLAUDE.md re-injection, summary instructions, `PreCompact`, hooks outside the window
    - [Subagents in the SDK](https://code.claude.com/docs/en/agent-sdk/subagents): the prompt string as the only input, the final message as the only output, Agent and Task naming
    - [Work with sessions (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/sessions): passing captured results into a fresh session
    - [Hosting the Agent SDK](https://code.claude.com/docs/en/agent-sdk/hosting): `SessionStore` mirrors transcripts only
    - [Intercept and control agent behavior with hooks (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/hooks): replacing tool output with `updatedToolOutput`
    - [How large is the context window on paid Claude plans?](https://support.claude.com/en/articles/8606394-how-large-is-the-context-window-on-paid-claude-plans): automatic context management, the code execution requirement, usage impact, projects and RAG
    - [How do usage and length limits work?](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work): what to do at a length limit
    - [Use Claude's chat search and memory](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context): project memory spaces, Team and Enterprise defaults, incognito chats
    - [How can I create and manage projects?](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects): context is not shared across chats in a project
    - [Troubleshoot Claude error messages](https://support.claude.com/en/articles/12466728-troubleshoot-claude-error-messages): summarizing or extracting key sections at a length limit
    - [How context affects Claude's performance and cost (Claude Academy)](https://academy.claude.com/tutorials/parametric-memory-and-context): compaction versus written memory, why a fresh chat can be cheaper, what not to persist
    - [Why do AI models hallucinate? (Claude Academy)](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate): checking an answer in a new chat
    - [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): definitions, context rot, attention budget, smallest high-signal token set, just-in-time and hybrid retrieval, compaction, note-taking and multi-agent techniques
    - [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system): token multipliers, compression by subagents, fit for multi-agent work, CitationAgent, phase summaries, filesystem outputs and lightweight references, resuming after errors, evaluation rubric
    - [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents): initializer artifacts, progress file and git, JSON feature list, session start routine
    - [Harness design for long-running application development](https://www.anthropic.com/engineering/harness-design-long-running-apps): context anxiety, context resets versus compaction, file-based agent communication, separating doer from judge
    - [Scaling Managed Agents: Decoupling the brain from the hands](https://www.anthropic.com/engineering/managed-agents): session log and crash recovery, resets as dead weight on Opus 4.5
    - [Building a C compiler with a team of parallel Claudes](https://www.anthropic.com/engineering/building-c-compiler): progress files and READMEs, keeping test output short, grep-friendly error lines
    - [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents): high-signal tool responses, `response_format`, pagination and the 25,000-token default, search tools instead of dump tools
    - [Introducing advanced tool use on the Claude Developer Platform](https://www.anthropic.com/engineering/advanced-tool-use): tool search with `defer_loading`
    - [Equipping agents for the real world with Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills): progressive disclosure in Skills
    - [Managing context on the Claude Developer Platform](https://claude.com/blog/context-management): context editing and memory tool launch results
    - [Prompt engineering for Claude's long context window](https://www.anthropic.com/news/prompting-long-context): the 2023 position study
    - [Best practices for prompt engineering for 2026 (Claude blog)](https://claude.com/blog/best-practices-for-prompt-engineering): improved context awareness in Claude 4.x models, critical details at the beginning or end of long contexts
    - [Long context prompting for Claude 2.1](https://claude.com/blog/claude-2-1-prompting): the historical prefill technique
    - [Context engineering: memory, compaction, and tool clearing (Claude Cookbook)](https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools): what summaries drop, the three-technique mental model, lossiness, context rot scaling with window contents
    - [Research subagent prompt (Claude Cookbooks)](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_subagent.md): conflict handling, flagging speculation, current-date injection
    - [Research lead agent prompt (Claude Cookbooks)](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_lead_agent.md): discrepancies, temporal constraints, no citations from the lead, choosing the answer's form
    - [Citations agent prompt (Claude Cookbooks)](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/citations_agent.md): cite only direct support, leave text unchanged
    - [Explore: grounding in an unfamiliar codebase (Claude Cookbooks)](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/managed_agents/CMA_explore_unfamiliar_codebase.ipynb): writing notes to a file while exploring
    - [Knowledge synthesis skill (knowledge-work-plugins)](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/enterprise-search/skills/knowledge-synthesis/SKILL.md): surfacing conflicts, dates and time periods
    - [Customer research skill (knowledge-work-plugins)](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/customer-support/skills/customer-research/SKILL.md): the Gaps and Unknowns section
    - [Customer escalation skill (knowledge-work-plugins)](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/customer-support/skills/customer-escalation/SKILL.md): structured escalation briefs
