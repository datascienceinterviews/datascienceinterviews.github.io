---
title: "Prompt Engineering and Structured Output for the Claude Certifications"
description: Prompting and structured output for all four Claude exams, from criteria, examples and XML tags to schemas, validation, batches, review and hallucinations.
last_reviewed: 2026-09-23
---

# Prompt Engineering and Structured Output

This page teaches prompting and structured output once, for all four Claude certification exams: the principles that decide most prompt questions, explicit criteria, examples, prompt structure, reasoning and chaining, and long documents, followed by structured output, validation, batch design, review architectures, hallucination control and prompt iteration. Each section opens with the exam objectives it serves. Product facts are as of September 2026, and where the documentation changed after the July 2026 exam guides, the section shows both versions and which wording to expect on the exam (the guide's).

| Exam | Where prompting sits in the blueprint | Weight |
|---|---|---|
| CCAO-F | Domain 1: Prompting and Task Execution | 14% |
| CCDV-F | Domain 6: Prompt and Context Engineering (mainly D6.2 Prompt Engineering, 4.6%, and D6.3 Output Handling, 2.6%; D6.1 Context Engineering, 3.8%, is taught on the [context engineering page](context-engineering.md) apart from long-document prompting). D5.1 LLM Fundamentals (5.2%, in Domain 5, not part of the 11.0%) adds zero-, single- and multi-shot prompting and thinking, taught in [Few-shot examples](#few-shot-examples) and [Chain of thought and prompt chaining](#chain-of-thought-and-prompt-chaining) | 11.0% |
| CCAR-F | Domain 4: Prompt Engineering & Structured Output | 20% |
| CCAR-P | Domain 2: Claude Models, Prompting & Context Engineering | 13% |

!!! info "One reference page, as of September 2026"

    Anthropic's separate technique pages (be clear and direct, multishot prompting, chain of thought, XML tags, system prompts, chain prompts, long-context tips) now redirect to a single page, [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices). The [overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) calls it the place to begin: "That's the living reference; start there." Older study notes may still link the retired per-technique URLs; the techniques themselves live on in the consolidated page.

## Principles that decide most prompt questions

*Tested in: CCAO-F D1.1, D1.3, D1.4, D7.1 · CCDV-F D6.2 Prompt Engineering · CCAR-F 1.4, 1.5, 2.1 · CCAR-P 2.2, 2.3, 4.4*

The official sample questions treat prompting as a series of decisions more than a test of wording. CCAR-F Question 1 asks whether a prompt can guarantee a tool sequence, Question 2 whether examples or better tool descriptions fix misrouting, and Question 3 whether explicit criteria or a confidence score fixes escalation. Anthropic's [prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) sets the frame: before prompting you need "A clear definition of the success criteria for your use case", a way to test against those criteria, and a first draft. It adds a warning that several official sample questions echo: "Not every success criteria or failing eval is best solved by prompt engineering." Latency and cost, for example, can sometimes be fixed more easily by choosing a different model.

### Seven principles at a glance (our synthesis)

| # | Principle | The rule in one line | Where the exams lean on it |
|---|---|---|---|
| 1 | Prompts guide; code guarantees | If it must happen every time, enforce it in code | CCAR-F 1.4, 1.5, Q1; CCDV-F Sample 2; CCAO-F Sample 3; CCAR-P Sample 1 |
| 2 | Fix the layer that is broken | Tool descriptions, criteria, placement, retrieval, integration or model before more prompt text | CCAR-F Q2, Q3; CCAR-P Samples 2 and 3; CCDV-F Sample 3 |
| 3 | Be explicit and complete | Claude lacks context about your task, organization and norms unless the prompt gives it | CCAO-F D1.1; CCDV-F instruction clarity |
| 4 | Give the reason | Explain why, and state the goal as well as the format | CCAO-F D1.1, D7.1 |
| 5 | Say what to do, in units Claude can follow | Positive instructions; sentence or paragraph counts, not word counts | CCDV-F output constraints |
| 6 | Minimal, calm and current | Right altitude, no shouting, audit prompts on every model change | CCAR-F 2.1; CCAR-P 2.2 |
| 7 | Iterate against evidence | Test, change one thing, test again | CCAO-F D1.3; CCDV-F iterative refinement |

### 1. Prompts guide; code guarantees

The CCAR-F guide separates "programmatic enforcement (hooks, prerequisite gates)" from "prompt-based guidance for workflow ordering", and says that when deterministic compliance is required, "prompt instructions alone have a non-zero failure rate". Sample Question 1 applies it: a programmatic prerequisite beats a stronger system prompt or few-shot examples, because "Options B and C rely on probabilistic LLM compliance, which is insufficient when errors have financial consequences." CCDV-F Sample 2 says the same about prompt injection: "a polite request (C) is not an enforceable control", and it adds that "a more instruction-following model (D) can be more susceptible, not less." The other exams' samples make the same point from other angles. CCAO-F Sample 3 rejects uploading regulated personal data with an instruction not to keep it, because "instructing the model not to retain data (C) does not satisfy the policy control." CCAR-P Sample 1 removes the refund and delete tools that support staff never need, rather than adding a confirmation prompt, because "Least privilege means removing capabilities the role does not require, eliminating the attack surface rather than monitoring or guarding it." Claude Code draws the same line for its memory files: CLAUDE.md is delivered as a user message after the system prompt, and "Claude treats them as context, not enforced configuration"; a PreToolUse hook blocks an action regardless of what Claude decides ([Claude Code memory](https://code.claude.com/docs/en/memory)).

**Decide:** if a rule must hold on every run (identity verified before a refund, no refund above a policy threshold, no refund or delete capability for a role that never needs it), choose a hook, a prerequisite gate or removal of the capability. If a behavior should usually happen (tone, format, when to offer escalation), a prompt instruction is proportionate. See [Hooks in the SDK](agents-and-agent-sdk.md#hooks-in-the-sdk) and [Permissions and enforcement](agents-and-agent-sdk.md#permissions-and-enforcement).

### 2. Fix the layer that is broken

Before rewording anything, name the layer that fails. Read together, the official rationales reward the fix that addresses the root cause with the least new machinery (the Q2 rationale says its answer "directly addresses this root cause" with little effort):

| Symptom in the stem | Layer at fault | Proportionate fix | Official source |
|---|---|---|---|
| Similar tools with minimal descriptions get confused | Tool definitions | Expand each description: input formats, example queries, edge cases, boundaries | CCAR-F Q2 |
| Agent escalates easy cases and handles hard ones itself | Unclear decision boundaries | Explicit escalation criteria with few-shot examples in the system prompt | CCAR-F Q3 |
| Same 8,000-token system prompt and policy on every request; latency and cost both matter | Prompt layout | Static content first, dynamic content after it, prompt caching on | CCAR-P Sample 2 |
| Confident but wrong answers right after a document refresh, with model and latency unchanged | Retrieval | Investigate the retrieval or indexing step first (irrelevant or stale chunks) | CCAR-P Sample 3 |
| Several Claude applications need the same internal capability | Integration | An MCP server that exposes the operations as tools; logic hard-coded into prompts "is neither reusable nor maintainable" | CCDV-F Sample 3 |
| Output quality is fine but latency or cost is not | Model choice | Consider a different model: the overview says this can sometimes be easier than more prompt work | Prompt engineering overview |
| A tool sequence must never be skipped | Enforcement | Programmatic prerequisite, not prompt text | CCAR-F Q1 |

Two rationale lines are worth knowing by heart. On few-shot examples as a fix for poor descriptions: "Few-shot examples (A) add token overhead without fixing the underlying issue." On explicit criteria for escalation: "This is the proportionate first response before adding infrastructure." Tool-description fixes are covered in [Writing tool descriptions that steer selection](tool-use-and-mcp.md#writing-tool-descriptions-that-steer-selection). When the problem may sit in your code rather than in Claude's output, isolate it first; see [Debugging: model or integration](evaluation-and-reliability.md#debugging-model-or-integration).

### 3. Be explicit and complete

Claude starts every request without context about you, your task or your organization beyond what the prompt gives it. The docs put it in one image: "Think of Claude as a brilliant but new employee who lacks context on your norms and workflows." Their test is the golden rule: "Show your prompt to a colleague with minimal context on the task and ask them to follow it. If they'd be confused, Claude will be too." The help center uses the same idea for app users: "Think of Claude as a newly-hired contractor." ([Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices); [Introduction to prompt design](https://support.claude.com/en/articles/7996853-introduction-to-prompt-design))

What "explicit" means in practice:

- **Ask for more than the minimum when you want it.** The docs rate "Create an analytics dashboard" as less effective than the same request followed by "Include as many relevant features and interactions as possible. Go beyond the basics to create a fully-featured implementation." The docs' rule: if you want "above and beyond" behavior, request it rather than relying on inference.
- **Number the steps** when order or completeness matters.
- **Say whether you want action or advice.** "Can you suggest some changes" may produce suggestions only; "Change this function to improve its performance." produces the change.
- **State the scope.** Claude Opus 4.8 and Claude Sonnet 5 interpret prompts literally, particularly at lower effort: Opus 4.8 "does not silently generalize an instruction from one item to another". The docs' fix is to write the scope out, as in "Apply this formatting to every section, not just the first one".
- **Front-load the whole task.** For interactive coding, the Sonnet 5 guide says to give the task, intent and constraints in the first human turn; ambiguous or underspecified prompts conveyed progressively over multiple user turns "tend to relatively reduce token efficiency and sometimes performance".
- **Use a repeatable structure for everyday prompts.** Claude 101 teaches three parts: setting the stage (your role, objectives and context), defining the task (what action you want Claude to take) and specifying rules (style, tone and examples). For recurring tasks, Anthropic's AI Fluency course suggests a template prompt with placeholders for the information that changes each time.

### 4. Give the reason

"NEVER use ellipses" is a bare rule. The better version in [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) carries its reason: "Your response will be read aloud by a text-to-speech engine, so never use ellipses since the text-to-speech engine will not know how to pronounce them." The payoff is transfer: "Claude is smart enough to generalize from the explanation." [Anthropic's AI capabilities course](https://academy.claude.com/courses/ai-capabilities-and-limitations/steerability) makes the same point for everyday users by separating a goal ("Convince my team this timeline is realistic") from a format ("Three bullet points"). When Claude honors the letter of an instruction but misses its intent, re-prompt with the goal attached: "Make this shorter. My goal is to keep the executive's attention through the key finding on page two."

### 5. Say what to do, in units Claude can follow

- **Positive instructions beat prohibitions.** Instead of "Do not use markdown in your response", the docs suggest "Your response should be composed of smoothly flowing prose paragraphs." The model guides for Claude Opus 5, Claude Sonnet 5 and Claude Opus 4.8 add that positive examples of the communication style you want tend to work better than instructions about what not to do (the [Opus 5 guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5) says it of progress narration; the Sonnet 5 and Opus 4.8 guides say it of concision).
- **Your prompt's style leaks into the output.** Removing markdown from the prompt can reduce markdown in the response.
- **Constrain length in sentences or paragraphs.** Models count tokens, not words, so "asking for an exact word count or a word count limit is not as effective a strategy as asking for paragraph or sentence count limits." Word limits still appear in official app guidance: for a response of the wrong length, [Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results) suggests "Keep this under 100 words" as one explicit option. For API prompts, prefer sentence or paragraph counts. `max_tokens` is a hard limit that can cut a response mid-sentence; the docs call it a blunt technique best kept for short answers.
- **Machine-read output needs a schema, not prose rules.** For JSON that code consumes, use the features in [Structured output with tools and JSON schemas](#structured-output-with-tools-and-json-schemas).

!!! warning "Older prompting advice vs current docs: prefill"

    Older material recommends prefilling, writing the first characters of Claude's reply (for example an opening `{` to force JSON). The [claude.com prompt engineering post](https://claude.com/blog/best-practices-for-prompt-engineering) still suggests it for enforcing output formats and skipping preambles. As of September 2026 the [docs](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) say that prefilling the last assistant turn is not supported starting with the Claude 4.6 models (and Claude Mythos Preview): "Requests with prefilled assistant messages to these models return a 400 error", with the error message "This model does not support assistant message prefill. The conversation must end with a user message." ([API errors](https://platform.claude.com/docs/en/api/errors)) Earlier models still accept prefill. The documented replacements: structured outputs, or tools with an enum field for classification; a system-prompt instruction such as "Respond directly without preamble. Do not start with phrases like 'Here is...', 'Based on...', etc."; and, for continuations, moving the continuation into the user message with the final text of the interrupted response. None of the four exam guides mentions prefill. Our rule: if an answer option relies on prefill for a Claude 4.6 or later model, treat it as wrong, because that request fails.

### 6. Minimal, calm and current

Anthropic's context engineering post describes the target as "the right altitude": between engineers "hardcoding complex, brittle logic in their prompts" and "vague, high-level guidance that fails to give the LLM concrete signals for desired outputs or falsely assumes shared context" ([Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)). Aim for the minimal set of information that fully outlines the expected behavior, noting that "minimal does not necessarily mean short": start with a minimal prompt on the best available model, then add instructions and examples for the failure modes you observe. The claude.com post agrees: "Longer, more complex prompts are NOT always better."

Three habits keep prompts calm and current:

- **Drop the shouting.** Claude Opus 4.5 and Opus 4.6 are more responsive to the system prompt, so prompts written to fight undertriggering can now overtrigger. Replace "CRITICAL: You MUST use this tool when..." with "Use this tool when...", remove lines like "If in doubt, use [tool]", and replace blanket defaults with targeted instructions about when the tool helps.
- **Watch for keyword traps.** System-prompt wording can create unintended tool associations that override good tool descriptions (CCAR-F 2.1; see [System prompts and roles](#system-prompts-and-roles)). Anthropic's [cost guidance](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence) gives a support-desk case where a mandatory 6-step procedure plus an instruction to investigate fully even when the ticket looks simple forced four tool calls on a "where's my package" ticket.
- **Audit on every model change.** The same page: "Auditing prompts against the model you run now, and again whenever you change models, is a free win." On a support-desk evaluation, "prompts written for Claude Opus 4.8 cost 36% more per ticket on Claude Opus 5 for no change in accuracy"; after the audit, Opus 5 was 14% cheaper than with the unaudited prompts and more accurate (97% of tickets, up from 92%). Prompting best practices lists six model-specific prompting pages (Fable 5.1 with Mythos 5.1, Fable 5 with Mythos 5, Sonnet 5, Opus 5.5, Opus 5 and Opus 4.8) and says to read the one for your model first; the habits that changed are tabulated in [Prompt versioning and iteration](#prompt-versioning-and-iteration).

### 7. Iterate against evidence

[Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results) tells app users: "Think of your initial prompt as the start of a conversation, not a one-shot request." It also asks for specific feedback: "Make it shorter" is fine, but "Cut the first two paragraphs and make the conclusion more action-oriented" is better. Anthropic's [Claude Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/validating-skills-for-plugins) adds, for revising a skill: "Change one thing at a time." When a review finds several problems, pick the one that matters more, fix it, re-run, then review again. In Claude Code, the best-practices page says that if you have corrected Claude more than twice on the same issue in one session, the context is cluttered with failed approaches: run `/clear` and start fresh with a more specific prompt that incorporates what you learned. For production prompts, the latency guide says to first engineer a prompt that works well without model or prompt constraints, then apply latency reduction. Versioning and regression testing are in [Prompt versioning and iteration](#prompt-versioning-and-iteration); Claude Code refinement loops are in [Iterative refinement](claude-code-workflows.md#iterative-refinement).

### Diagnosing an underperforming prompt

CCAO-F D7.1 and CCAR-P 4.4 ask you to diagnose poor output. Start from the symptom:

| Symptom | Likely cause | First fix |
|---|---|---|
| Too generic | Not enough context about your situation | Add audience, role or constraints |
| Too long or too short | Claude is guessing at length | State the length explicitly; in API prompts, prefer sentence or paragraph counts to word counts |
| Right content, wrong format | Claude understood what, not how | Show an example of the format |
| Wrong tone | Claude defaults to helpful and professional | Name the tone, or share 2 to 3 samples of the voice you want |
| Suggestions when you wanted changes | Advisory phrasing | Use an imperative, such as "Change this function to improve its performance." |
| Inconsistent format across runs | Instructions alone underdetermine the output | Add few-shot examples |
| Complex task gives unreliable results | Too much in one prompt | Split it into a chain of focused prompts |
| Invented facts | No permission to be uncertain | Explicitly allow "I don't know" |
| Confident but wrong | Plausible but incorrect specifics, especially on niche topics | Verify key facts against an authoritative source, ask Claude to cite sources, enable web search (CCAO-F Sample 1: a self-rated confidence is not a reliable accuracy signal) |
| Instruction honored, intent missed | Letter over spirit | Re-prompt with the goal alongside the instruction |
| Tools overtrigger after a model upgrade | Emphatic legacy instructions | Plain "Use this tool when..." wording |

### Adapting the prompt to the task type

CCAO-F D1.4 names four task types. The moves below come from Anthropic's courses, help articles and use-case pages; grouping them by task type is our mapping:

| Task type | Prompting moves that fit it |
|---|---|
| Analysis | Define the question and give all relevant data in one well-structured message; add "flag anything that surprises you" to get interpretation alongside a chart; test Claude on past data whose answer you already know before delegating the task |
| Research | Say who you are, who you serve and what you need to know; give clear success criteria; ask Claude to verify across multiple sources and track competing hypotheses; flag specific claims (regulations, prices, deadlines) for checking against primary sources |
| Drafting | Outline requirements, audience and key points up front; upload 2 to 3 pieces that show your voice; make revision requests specific instead of "make this more compelling" |
| Brainstorming | Work turn by turn, because each answer changes the next question; ask for 2 to 3 options or deliberately different variants; invite pushback ("Push back if my assumptions are wrong"); let Claude interview you when you are unsure what you need |

Effort is a separate lever from wording. The [help center](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings) says simple questions and general writing "don't need extra effort or thinking", while complex tasks call for raising effort, turning on thinking, or both. See [Choosing a model in the apps](claude-for-work.md#choosing-a-model-in-the-apps).

### Traps

- **A stronger prompt offered as a guarantee.** All-caps MUST, a mandatory-step sentence, few-shot examples, or an instruction not to retain data, for a rule that must always hold. The [CCAR-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) Q1 rationale says the stronger-prompt and few-shot options "rely on probabilistic LLM compliance, which is insufficient when errors have financial consequences"; [CCAO-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) Sample 3 says an instruction not to retain data "does not satisfy the policy control."
- **A bigger or more obedient model offered as a control.** CCAR-P Sample 1 says model size "is unrelated to authorization scope"; CCDV-F Sample 2 says a more instruction-following model "can be more susceptible, not less."
- **More prompt text for a problem in another layer.** Examples for bad tool descriptions, moving a policy into a few-shot block to fix a caching problem, logic hard-coded into each system prompt instead of a shared MCP server, prompt tuning for a latency target that a different model might meet more easily.
- **Prohibition-only instructions** ("Do not use markdown in your response", or the "avoid a generic AI look" that the Opus 5.5 guide says "mostly swaps one default for another") where a positive target or specific named patterns work better.
- **Legacy techniques on current models:** prefill, emphatic tool instructions, prompts tuned for an older model.

## Explicit criteria and precision

*Tested in: CCAR-F 4.1, 4.2, 5.2, 3.6-S3 · CCDV-F D6.2 Prompt Engineering · CCAR-P 2.2 · CCAO-F D1.1, D5.3*

A criterion is explicit when someone with no context could apply it and reach the same verdict as you. In the [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf), Task Statement 4.1 is "Design prompts with explicit criteria to improve precision and reduce false positives", and Scenario 5 (Claude Code for Continuous Integration) asks you to "design prompts that provide actionable feedback and minimize false positives". In this section, precision means the share of reported findings that are real, and recall means the share of real issues that get reported. Anthropic's model guides ([Prompting Claude Sonnet 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5), [Prompting Claude Opus 4.8](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-4-8)) describe review prompts moving one at the expense of the other: "Precision typically rises, but measured recall can fall even though the model's underlying bug-finding ability has improved."

### Vague instruction, explicit criterion

| Vague | Explicit | Source |
|---|---|---|
| "check that comments are accurate" | "flag comments only when claimed behavior contradicts actual code behavior" | CCAR-F 4.1 |
| "be conservative" or "only report high-confidence findings" | Named categories to report (bugs, security) and to skip (minor style, local patterns) | CCAR-F 4.1 |
| Report what is "important" | "report any bugs that could cause incorrect behavior, a test failure, or a misleading result; only omit nits like pure style or naming preferences." | Opus 4.8 and Sonnet 5 prompting guides (both use this wording) |
| Severity labels with no definitions | Each severity level defined, with a concrete code example | CCAR-F 4.1 |
| Unsafe-content category names only | Definitions and example phrases for each category | Content moderation guide |
| "be professional" | "Respond in formal English. Don’t use contractions, slang, or emojis." | Organization instructions article |
| Escalate "complex" cases | Escalate when the customer asks for a human, when policy is silent or needs an exception, or when no meaningful progress is possible | CCAR-F 5.2 |

The pattern in every row: the explicit version names the condition that makes something reportable, or lists the categories, so the verdict no longer depends on the model's sense of what "conservative" or "complex" means.

### Why confidence words do not buy precision

The CCAR-F guide states it directly: general instructions like "be conservative" or "only report high-confidence findings" fail to improve precision compared with specific categorical criteria, and the skill it tests is writing criteria "rather than relying on confidence-based filtering". Self-assessed confidence is a weak filter on its own: the guides reject it as the gate for escalation (CCAR-F Task Statement 5.2 and Question 3) and as a reason to trust a summary (CCAO-F Sample 1), and [Verify; do not trust confidence](#verify-do-not-trust-confidence) sets out that evidence. Confidence can travel with each finding as extra data for routing once it is calibrated (CCAR-F 4.6-S3), but it does not replace a definition of what counts.

!!! warning "Exam guide vs current docs: be conservative"

    The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says such instructions fail to improve precision. The current guides for Claude Opus 4.8 and Claude Sonnet 5 (as of September 2026) describe what happens on those models: they may follow "only report high-severity issues", "be conservative" or "don't nitpick" more faithfully than earlier models did, investigate the code just as thoroughly, and then not report findings below the stated bar, so measured recall can fall. The Sonnet 5 guide says a harness tuned for an earlier model showing lower recall is "likely a harness effect, not a capability regression" ([Prompting Claude Sonnet 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5)). The [Opus 5 guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5) describes the same behavior on that model and advises: "ask it to report everything and filter in a separate pass instead." Both framings favor concrete criteria over qualitative words. On the exam, answer in the guide's terms: explicit categorical criteria, not general caution.

### Two designs from the model guides

| Design | How it works | When to choose it (our guidance) |
|---|---|---|
| Single pass with a concrete bar | One prompt states exactly what to report and what to omit (the concrete-bar wording from the Opus 4.8 and Sonnet 5 guides in the table above) | Cost or latency allows one call; the categories are stable |
| Find broadly, then filter | A finding stage reports everything with a confidence level and an estimated severity; a separate verification step applies the explicit criteria (the Sonnet 5 guide adds that the finding prompt works even without a real second step, but "moving confidence filtering out of the finding step often helps") | Recall matters; you can afford a second pass |

The Sonnet 5 guide's finding-stage prompt ("Your goal here is coverage") is reproduced in [Prompting the passes](#prompting-the-passes), together with the verification pass that applies your criteria. Claude Code's managed Code Review service runs the same find-then-verify shape in production; see [Four review architectures](#four-review-architectures). Whichever design you choose, the model guides say to "Iterate on prompts against a subset of your evals or test cases to validate recall or F1 score gains."

### What explicit criteria look like in Anthropic's own review prompts

The `code-review` plugin published in Anthropic's public [claude-code repository](https://raw.githubusercontent.com/anthropics/claude-code/main/plugins/code-review/commands/code-review.md) writes its bar as categories, not adjectives (excerpt, verbatim):

```text
Flag issues where:
- The code will fail to compile or parse (syntax errors, type errors, missing imports, unresolved references)
- The code will definitely produce wrong results regardless of inputs (clear logic errors)
- Clear, unambiguous CLAUDE.md violations where you can quote the exact rule being broken

Do NOT flag:
- Code style or quality concerns
- Potential issues that depend on specific inputs or state
- Subjective suggestions or improvements
```

The same prompt then adds a certainty bar and its reason: "If you are not certain an issue is real, do not flag it. False positives erode trust and waste reviewer time." It does not rely on that line alone: later steps launch subagents to validate flagged issues, then "Filter out any issues that were not validated in step 5."

The managed Code Review feature reads review-only criteria from a `REVIEW.md` file at the repository root; the agents that find and verify findings receive its contents as the repository's review instructions. Its documented levers put CCAR-F 4.1 into practice (the mapping to 4.1 is ours):

| Lever | Example rule from the Code Review docs | What it fixes |
|---|---|---|
| Severity | Reserve Important for findings that "would break behavior, leak data, or block a rollback"; style and naming are Nit at most | Vague severity |
| Nit volume | "report at most five nits, mention the rest as a count in the summary" | Noise per review |
| Skip rules | No findings on listed paths, branch patterns or finding categories; common candidates are generated code, lockfiles, vendored dependencies, machine-authored branches and anything CI already enforces | Categories that only produce false positives |
| Higher bar instead of skipping | "in `scripts/`, only report if near-certain and severe." | Areas that need light scrutiny |
| Verification bar | "behavior claims need a `file:line` citation in the source, not an inference from naming" | Claims inferred from names |
| Re-review convergence | "after the first review, suppress new nits and post Important findings only" | Churn on each new commit |

Explicit does not mean long. The same page warns: "Length has a cost: a long `REVIEW.md` dilutes the rules that matter most." Setup and CI wiring are in [Automated code review that engineers trust](claude-code-workflows.md#automated-code-review-that-engineers-trust).

### Severity anchored in code

CCAR-F 4.1 asks for "explicit severity criteria with concrete code examples for each severity level". A minimal rubric in that shape (illustrative, ours; adapt the levels to your repository):

```text
<severity_levels>
<level name="critical">
Data can be corrupted or exposed. Example:
    query = "SELECT * FROM users WHERE id = " + request.args["id"]
</level>
<level name="high">
A realistic input produces a wrong result, or a failure is reported as success. Example:
    except TimeoutError:
        return []
</level>
<level name="low">
A real defect with no user-visible effect today. Example:
    return total
    log.info("done")  # unreachable
</level>
</severity_levels>
Anything the linter or formatter already enforces is not a finding at any level.
```

Pair a rubric like this with few-shot examples that set an acceptable pattern beside a genuine issue (CCAR-F 4.2); see [Few-shot examples](#few-shot-examples).

### Protecting developer trust

The guide's reason for caring about precision is human: "high false positive categories undermine" confidence in the accurate categories. Its remedy is operational: temporarily disable a high false-positive category to restore trust while you improve the prompt for it. With the managed Code Review service, a `REVIEW.md` skip rule for that finding category is one way to do that (our mapping). To find which categories are noisy, record the code construct behind each finding (the guide's `detected_pattern` field) and analyze what developers dismiss; see [Validation, retry and feedback loops](#validation-retry-and-feedback-loops). When a review re-runs after new commits, CCAR-F 3.6 says to include the prior findings and ask for only new or still-unaddressed issues, so developers do not see duplicate comments.

### The same discipline outside code review

- **Classification.** List the valid labels and the rule for choosing among them: the [ticket-routing](https://platform.claude.com/docs/en/about-claude/use-case-guides/ticket-routing) prompt says "A request may have ONLY ONE applicable intent." When Claude weighs the wrong signal, say what to ignore (the guide's example begins "Ignore all customer emotions."), and when tickets raise several issues, state how to prioritize them.
- **Moderation.** Category names alone are underspecified; the content moderation guide adds definitions and phrases for each category, then tracks precision and recall in production.
- **Escalation.** CCAR-F 5.2 lists the triggers (a request for a human, policy exceptions or gaps, inability to make progress) and says to put "explicit escalation criteria with few-shot examples" in the system prompt. Sentiment and self-reported confidence are named as unreliable proxies. Details are in [Escalation and ambiguity](evaluation-and-reliability.md#escalation-and-ambiguity).
- **Standing instructions in the Claude apps.** The [organization-instructions article](https://support.claude.com/en/articles/14546867-set-organization-instructions) lists seven best practices, among them: be specific, avoid instructions that contradict each other ("Claude may not follow either one reliably"), and test in a new conversation across several types of question. The same habits suit any standing instruction, such as a project's instructions (our extension).

### Decide

- If a reviewer's output is mostly noise, choose explicit report and skip categories with a code example per severity level; not "be conservative" or a confidence cut-off.
- If one category is noisy and the rest are accurate, disable that category while you rework it; not keep posting it, because its noise lowers trust in the accurate ones.
- If recall matters, choose a broad finding stage plus a separate filter with explicit criteria; not a single prompt told to report only what matters.
- If escalation is miscalibrated, choose explicit criteria with few-shot examples; not a confidence threshold, sentiment trigger or new classifier as the first step.

### Traps

- Options that add "be conservative", "only report high-confidence findings" or "don't nitpick". The guide names the first two as failing to improve precision.
- Options that filter by the model's own confidence score instead of defining categories.
- Severity scales defined only by adjectives (minor, moderate, serious).
- Options that escalate on sentiment or on a self-reported confidence score. CCAR-F 5.2 calls both "unreliable proxies for actual case complexity".
- A longer, more detailed policy file offered as the fix for missed rules. The Code Review docs warn that length dilutes the rules that matter most.

## Few-shot examples

*Tested in: CCAR-F 4.2, 3.5, 5.2 · CCDV-F D5.1 LLM Fundamentals, D6.2 Prompt Engineering · CCAR-P 2.3 · CCAO-F D1.1, D7.1*

Examples show Claude what instructions struggle to describe. [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) calls them "one of the most reliable ways to steer Claude's output format, tone, and structure." The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) goes further and ranks few-shot examples as "the most effective technique for achieving consistently formatted, actionable output when detailed instructions alone produce inconsistent results". Read the condition at the end of that sentence: examples are the fix once instructions have been tried and the output still varies.

### Vocabulary: counting shots

[Anthropic's interactive prompting tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial/blob/master/Anthropic%201P/07_Using_Examples_Few-Shot_Prompting.ipynb) defines the term: the number of "shots" "refers to how many examples are used within the prompt."

| Term | Examples in the prompt | Where you meet the term |
|---|---|---|
| Zero-shot | None; instructions only | CCDV-F D5.1, CCAR-P 2.3, Anthropic's GitHub tutorial |
| One-shot (CCDV-F writes "single-shot") | One | CCDV-F D5.1, the claude.com prompting post, Anthropic's GitHub tutorial |
| Few-shot, multishot (CCDV-F writes "multi-shot") | Several | The docs: "few-shot or multishot prompting"; CCAR-F 4.2; CCAR-P 2.3 |

### When examples are the right fix

| Reach for examples when | Look elsewhere when |
|---|---|
| The format is easier to show than to describe, a specific tone is needed, or the task has subtle conventions | The task is simple: be specific, be clear, give context, and stop there |
| Detailed instructions still produce inconsistent output | The stem shows minimal tool descriptions: expand the descriptions first (CCAR-F Q2) |
| The model must judge ambiguous cases, such as which tool fits an ambiguous request or whether a branch lacks test coverage | The rule must hold every time: enforce it in code (CCAR-F Q1) |
| Extraction invents values from informal measurements or varied document layouts | The goal is caching a long policy: a few-shot block "does not create a cacheable, reusable prefix" (CCAR-P Sample 2) |
| Prose descriptions of a transformation are read inconsistently | Output must match a schema exactly: use [structured output](#structured-output-with-tools-and-json-schemas) |

### How many examples

| Source | Count | Context |
|---|---|---|
| CCAR-F 4.2 | 2 to 4 targeted examples | Ambiguous scenarios, each showing its reasoning |
| CCAR-F 3.5 | 2 to 3 concrete input/output examples | Transformations in Claude Code that prose leaves unclear |
| Prompting best practices | 3 to 5 examples | "for best results" |
| claude.com prompting post | Start with one, add more only if needed | General advice |
| CCAR-F Q2, option A (a wrong answer) | 5 to 8 examples | Offered as a fix for weak tool descriptions |

Our reading: the counts agree more than they differ, pointing to a handful of well-chosen examples, grown from one when the output still misses. The docs also suggest asking Claude "to evaluate your examples for relevance and diversity, or to generate additional ones based on your initial set", and Anthropic's API course suggests taking examples from your highest-scoring eval outputs.

### What a good example contains

- **Relevant, diverse, structured.** Mirror the real use case; "Cover edge cases and vary enough that Claude doesn't pick up unintended patterns" ([Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)); wrap each example in `<example>` tags and a set in `<examples>`, so Claude can tell examples from instructions.
- **Canonical, not exhaustive.** Anthropic's [context engineering post](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) advises teams to "curate a set of diverse, canonical examples" instead of stuffing a laundry list of edge cases into the prompt.
- **Nothing you do not want copied.** The [claude.com post](https://claude.com/blog/best-practices-for-prompt-engineering) warns: "Claude 4.x and similar advanced models pay very close attention to details in examples." For instance (our illustration), if every example opens the same way or has the same length, expect the output to as well.
- **The reasoning, not only the answer.** [CCAR-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) 4.2 asks for examples "that show reasoning for why one action was chosen over plausible alternatives", and the same task statement credits examples with letting the model "generalize judgment to novel patterns rather than matching only pre-specified cases". The [ticket-routing guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/ticket-routing) agrees: include "a classification rationale for particularly nuanced ticket intents". For examples used with thinking, [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) suggests `<thinking>` tags inside each example to show the reasoning pattern; Claude "will generalize that style to its own extended thinking blocks."
- **Positive models of the target.** Show the style you want instead of listing what to avoid (see [principle 5](#5-say-what-to-do-in-units-claude-can-follow)). When Fable 5.1 summarizes documents, it is more likely than Fable 5 to reproduce source passages without marking them as quotations; the documented fix is one complete example in the system prompt: the user's request, the response, and a sentence explaining why the response is correct.
- **Current.** A [claude.com post on reducing cost](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform) warns that "Few-shot examples tuned to an older model's failure modes can teach a frontier model to imitate long reasoning chains on requests that don't need them."

### Examples by job

| Job | What each example must show | Source |
|---|---|---|
| Consistent findings | The exact output fields (location, issue, severity, suggested fix) | CCAR-F 4.2 |
| Fewer false positives | An acceptable code pattern beside a genuine issue | CCAR-F 4.2 |
| Ambiguous routing or tool choice | The chosen action and why the alternatives lost | CCAR-F 4.2 |
| Extraction across layouts | Inline citations vs bibliographies, methodology sections vs embedded details, narrative vs tables | CCAR-F 4.2, Exercise 3 |
| Required fields coming back empty | Correct extraction from the layouts that fail | CCAR-F 4.2 |
| Transformations in Claude Code | Concrete input and expected output; for edge cases such as null values in migration scripts, specific test cases | CCAR-F 3.5 |
| Escalation calibration | Cases to escalate and cases to resolve, next to explicit criteria | CCAR-F 5.2, Q3 |
| Voice in the Claude apps | 2 to 3 pieces in your style; "Provide an example of writing in the style you want." | [Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results), Academy use cases |
| Tool inputs | Schema-valid `input_examples` on the tool definition | Define tools docs |

### An official example format

The ticket-routing guide's classification prompt shows the example input, then the exact output format with reasoning before the label (excerpt, verbatim, from [Ticket routing](https://platform.claude.com/docs/en/about-claude/use-case-guides/ticket-routing)):

```text
As an example, consider the following request:
<request>Hello! I had high-speed fiber internet installed on Saturday and my installer, Kevin, was absolutely fantastic! Where can I send my positive review? Thanks for your help!</request>

Here is an example of how your output should be formatted (for the above example request):
<reasoning>The user seeks information in order to leave positive feedback.</reasoning>
<intent>Support, Feedback, Complaint</intent>
```

### Worked example: extraction from varied layouts

This prompt targets the two CCAR-F 4.2 skills on varied document structures and empty required fields: one inline citation and one bibliography entry, each with a one-line reason, and nulls where the source is silent (illustrative, ours; the names and titles are invented):

```text
<instructions>
Extract each cited source in the excerpt. Citations appear either inline
(author and year in the text) or as numbered bibliography entries.
Use null for any field the excerpt does not state. Never infer a value.
</instructions>

<examples>
<example>
<input>...as shown by Okafor and Lind (2021), latency falls when...</input>
<output>{"authors": "Okafor and Lind", "year": 2021, "title": null, "venue": null}</output>
<why>Inline citation: only authors and year are stated, so title and venue are null.</why>
</example>
<example>
<input>[4] M. Silva. Streaming joins at scale. Proc. of DataSys, 2019.</input>
<output>{"authors": "M. Silva", "year": 2019, "title": "Streaming joins at scale", "venue": "Proc. of DataSys"}</output>
<why>Bibliography entry: all four fields are stated, so none is null.</why>
</example>
</examples>

<excerpt>
{{PAPER_EXCERPT}}
</excerpt>
```

The examples teach layout handling; the guarantee that the output parses belongs to the schema. In production, send the fields through [structured output](#structured-output-with-tools-and-json-schemas) with nullable types, and check the values in [Validation, retry and feedback loops](#validation-retry-and-feedback-loops).

### Examples at scale

- **Too many classes.** "As the number of classes grows, the number of examples required also expands, potentially making the prompt unwieldy." For 20+ intent categories, the ticket-routing guide suggests a taxonomic hierarchy of classifiers instead, at some cost in latency.
- **Inputs too varied for a fixed set.** Retrieve the most relevant examples per request with vector similarity search; the [ticket-routing guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/ticket-routing) says this approach, detailed in Anthropic's classification recipe, "has been shown to improve performance from 71% accuracy to 93% accuracy."
- **A large, stable set.** Cache it. The prompt caching page notes that "with prompt caching you can get even better performance by including 20+ diverse examples of high quality answers." Put static content, examples included, at the start of the prompt so the prefix can be reused; see [Prompt caching](claude-api.md#prompt-caching).
- **Examples for tool inputs.** A tool definition accepts an optional `input_examples` array, meant for tools with complex inputs, nested objects or format-sensitive parameters; the [define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools) page still says "Clear descriptions are most important", which matches CCAR-F Q2. Each example must be valid against the tool's `input_schema` (an invalid one returns a 400 error), and the field is not supported on server tools such as web search or code execution, or on the computer use and browser use toolsets. When Claude returns parameter values outside an enum, the troubleshooting guide names missing strict mode or a too-large enum as the causes, and suggests shrinking the enum or adding `input_examples` that show valid choices. See [Defining a tool](tool-use-and-mcp.md#defining-a-tool).

!!! warning "Exam guide vs current docs: how many, and what to call them"

    The CCAR-F guide says 2 to 4 targeted examples for ambiguous scenarios and 2 to 3 input/output examples for transformations; [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) recommends three to five examples "for best results"; the [claude.com post](https://claude.com/blog/best-practices-for-prompt-engineering) says to start with one. CCDV-F says "single-shot" and "multi-shot", while the docs say "few-shot or multishot prompting". On the exam, use each guide's own numbers and words. Our practical rule: start small and add examples while the eval scores keep improving.

### Decide

- If instructions are clear but output format or judgment still varies, add targeted examples (for ambiguous judgment calls, 2 to 4 that show why one action beat the alternatives; for format, examples showing the exact output fields); not another paragraph of rules.
- If the stem's root cause is outside the prompt (tool descriptions, a must-always rule, caching), fix that layer; not add examples.
- If extraction leaves required fields empty on some layouts, add examples showing correct extraction from those layouts, and make a field nullable where the source may genuinely lack the value; not a rule that forces a value.
- If the example set would be huge or the inputs vary widely, retrieve examples per request or cache a stable set; not a longer prompt.

### Traps

- Examples offered as the fix for minimal tool descriptions (CCAR-F Q2) or for a rule that must never be skipped (CCAR-F Q1).
- Examples that share an accidental feature, which the model then copies.
- Answer-only examples for judgment calls; without the reason, the model has less to generalize from.
- A few-shot block proposed as a caching or cost fix (CCAR-P Sample 2, option D).

## XML tags, system prompts and roles

*Tested in: CCDV-F D6.2 Prompt Engineering, D2.5 Claude Application Design · CCAR-F 2.1, APPX-TECH-5 · CCAR-P 2.2, 2.5 · CCAO-F D5.1, D5.3*

Two structural decisions sit under every prompt: how to separate its parts, and where each instruction lives. CCDV-F names both. Its Prompt Engineering skill lists "system versus user placement" and "prompt and instruction placement across components", and its Claude Application Design skill covers "how Claude interprets instructions across interfaces" and "content boundaries". CCAR-P 2.2 asks you to "Design system prompts, templates, and guardrails" and 2.5 to "Implement prompt reuse strategies (caching, modular prompts, Skills)". CCAR-F 2.1 tests how system prompt wording affects tool selection. CCAO-F D5.3, in the Configuration and Knowledge Management domain, asks you to "Create effective system-level instructions".

### XML tags mark content boundaries

"XML tags help Claude parse complex prompts unambiguously, especially when your prompt mixes instructions, context, examples, and variable inputs." Wrapping each type of content in its own tag (for example `<instructions>`, `<context>`, `<input>`) "reduces misinterpretation" ([Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)). The docs give two rules: use consistent, descriptive tag names across your prompts, and nest tags when content has a natural hierarchy (documents inside `<documents>`, each inside `<document index="n">`). Anthropic Academy's API course notes use "content boundaries" for exactly this: tags are essential when large blocks could be confused (code vs documentation, data vs instructions), and useful even for short content "to explicitly mark it as external input".

A reusable skeleton built only from the tag names the docs use (illustrative, ours). Long material sits at the top and the request at the end, as [Long documents](#long-documents) explains; `{{DOUBLE_BRACE}}` marks the slots a template fills, the convention in the docs' own examples:

```text
<documents>
  <document index="1">
    <source>{{SOURCE_NAME}}</source>
    <document_content>
    {{DOCUMENT_TEXT}}
    </document_content>
  </document>
</documents>

<context>
{{WHO_THIS_IS_FOR_AND_WHY}}
</context>

<examples>
{{EXAMPLES}}
</examples>

<instructions>
1. {{STEP_ONE}}
2. {{STEP_TWO}}
</instructions>

<input>
{{USER_REQUEST}}
</input>
```

Markdown headers work as section markers too. Anthropic's context engineering post recommends distinct sections such as `<background_information>`, `<instructions>`, `## Tool guidance` and `## Output description`, delimited "using techniques like XML tagging or Markdown headers", and adds that exact formatting "is likely becoming less important as models become more capable."

!!! danger "Tags separate content; they do not make it safe"

    A tag tells Claude where untrusted text starts and ends, but the text keeps whatever instructions it carries. For third-party content, [Mitigate jailbreaks and prompt injections](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks) says: "Deliver third-party content to Claude inside `tool_result` blocks, never in `system` prompts or plain user `text` blocks", JSON-encode it so an attacker cannot close a quote or tag and "break out" into an instruction context, and state in the system prompt that such content is untrusted. Even the `<pasted_content id="...">` marking in [Prompting Claude Opus 5.5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5-5) comes with the warning "The tags are plain text and can be imitated". CCDV-F Sample 2 tests this: the correct answer treats retrieved page content "as untrusted input", keeps it "separate from trusted instructions", and uses "guardrails or hooks so injected instructions cannot trigger sensitive actions"; the rationale credits "enforcing least-privilege guardrails so injected text cannot invoke sensitive tools" and dismisses a system-prompt line asking users not to send malicious instructions because "a polite request (C) is not an enforceable control". Full treatment: [Prompt injection](security-and-governance.md#prompt-injection).

### Tags in the output

Asking Claude to write parts of its answer inside named tags lets code split the response. The ticket-routing guide has Claude write `<reasoning>` then `<intent>` and extracts each "independently" with regular expressions; the legal summarization guide puts each summary section in its own tags so it "can easily be parsed out as a post-processing step". The docs also list XML format indicators as a way to steer format, for example asking for prose inside `<smoothly_flowing_prose_paragraphs>` tags.

Know where this stops. The tool-use docs say that a decision you would pull out of prose with a regex should have been a tool call. Tags suit text that people read or pipelines log; a decision your code acts on belongs in a tool call or a structured output, covered in [Structured output with tools and JSON schemas](#structured-output-with-tools-and-json-schemas). If you do parse tags, parse defensively: the ticket-routing code falls back to an empty string when a tag is missing.

### System prompts and roles

In the Messages API the system prompt is the top-level `system` parameter, not a message. The docs' role example, as a call (Python and TypeScript from [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices); import lines added):

=== "Python"

    ```python
    import anthropic

    client = anthropic.Anthropic()

    message = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=1024,
        system="You are a helpful coding assistant specializing in Python.",
        messages=[
            {"role": "user", "content": "How do I sort a list of dictionaries by key?"}
        ],
    )

    print(message.content)
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";

    const client = new Anthropic();

    const message = await client.messages.create({
      model: "claude-opus-5-5",
      max_tokens: 1024,
      system: "You are a helpful coding assistant specializing in Python.",
      messages: [{ role: "user", content: "How do I sort a list of dictionaries by key?" }]
    });

    console.log(message.content);
    ```

Guidance on roles and system prompts from Anthropic's docs and posts, and from the CCAR-F guide:

- **A role focuses behavior and tone.** "Even a single sentence makes a difference". The prompt-leak page calls a predominantly role-based system prompt the "most effective way to use system prompts".
- **Do not over-constrain the role.** The claude.com post: "You are a helpful assistant" is often better than an overly specific persona, and "being explicit about what perspective you want is more effective" in many cases, for example "Analyze this investment portfolio, focusing on risk tolerance and long-term growth potential" instead of assigning a role.
- **Personas need detail.** For a consistent character, describe personality, background and quirks in the system prompt, and prepare Claude with common scenarios and expected responses.
- **Tools change the system prompt.** When you pass `tools`, "the API constructs a special system prompt from the tool definitions, tool configuration, and any user-specified system prompt."
- **System prompt wording steers tool selection.** CCAR-F 2.1 tests that "keyword-sensitive instructions can create unintended tool associations" and asks you to review system prompts for instructions "that might override well-written tool descriptions". Emphatic or blanket tool instructions written for older models cause the related problem of overtriggering; see [Minimal, calm and current](#6-minimal-calm-and-current). Tool descriptions themselves: [Writing tool descriptions that steer selection](tool-use-and-mcp.md#writing-tool-descriptions-that-steer-selection).
- **Build long prompts from modules.** The customer-support guide writes named blocks (identity, static context, examples, guardrails) one at a time, then combines them "into a single string"; the split "also makes debugging your prompt easier". Anthropic's context engineering post adds the altitude test from [Principles](#principles-that-decide-most-prompt-questions): specific enough to guide behavior, yet flexible enough to give the model strong heuristics.
- **On Opus 5, repeat a length instruction near the end of a long system prompt.** Opus 5's default user-facing responses run longer than earlier Opus models', and [Prompting Claude Opus 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5) says: "In a long system prompt, pair the instruction with a short reminder near the end of the prompt" (the instruction being a conciseness instruction; its example reminder is a `<tone_preference>` block).

### Where each instruction goes

| Instruction | Put it in | Why |
|---|---|---|
| Role and standing policy from the first turn | Top-level `system` | Applies from the start and stays in the stable, cacheable prefix |
| Task data, documents, the specific request | The first user turn | The customer-support guide says Claude "works best with the bulk of its prompt content written inside the first" user turn, role prompting excepted |
| A standing rule that becomes relevant mid-session | A mid-conversation `{"role": "system"}` message (supported models only) | Same authority as the top-level `system` field, without invalidating the cached prefix before it |
| A reminder for one turn only | A turn-scoped system message with `clear_at: "next_user_message"` (beta header `mid-conversation-system-clear-at-2026-08-21`) | Once a later user message exists it renders nothing and costs no input tokens; the docs use it to resend a parallel-calls reminder after each round of tool results on Fable 5.1 |
| An instruction that follows tool results | A user turn after the `tool_result` (or a mid-conversation system message) | Instructions inside tool results "may be ignored or flagged as a potential injection" |
| Web pages, emails, retrieved files | `tool_result` blocks | Placing that text in a system message would give it operator-level authority |
| Stable text plus changing text | Static first, dynamic last, with caching on | The cache hashes `tools`, then `system`, then `messages`; any edit to `system` misses the cache for everything after it |

The first two rows reconcile official pages that differ. The [customer-support guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/customer-support-chat) puts the bulk of the content in the first user turn, "with the only exception being role prompting"; [Using the Messages API](https://platform.claude.com/docs/en/build-with-claude/working-with-messages) puts "instructions that should apply from the very first turn" in the top-level `system` field; and [Prompting Claude Opus 5.5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5-5) adds its standing instruction "at the end of your system prompt". Treat neither view as a universal rule. A workable split (ours) is role and standing policy in `system`, task material in the user turn. CCAR-P Sample 2 turns placement into cost: with an 8,000-token system prompt and policy sent on every request, the right move is to "Place the static system prompt and policy before the dynamic content and enable prompt caching", because that "lets repeated prefixes be reused". The rationale also rules out moving the policy into a few-shot block, which "does not create a cacheable, reusable prefix". Keep volatile values out of that prefix: Anthropic's cost post warns that "A dynamic timestamp or ID in the system prompt can change across model calls, and break the cache."

A mid-conversation system message, from [Mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages) (code comments and the output lines omitted). The feature is available on Fable 5.1, Mythos 5.1, Fable 5, Mythos 5, Opus 5.5, Opus 4.8 and Opus 5, needs no beta header, and is not available on Claude Sonnet 5:

=== "Python"

    ```python
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=1024,
        cache_control={"type": "ephemeral"},
        system="You are a code review assistant. Be concise.",
        messages=[
            {
                "role": "user",
                "content": "Review process() in utils.py for performance issues.",
            },
            {
                "role": "assistant",
                "content": "The list comprehension is fine for small inputs. For large inputs, consider a generator to avoid materializing the full list.",
            },
            {
                "role": "user",
                "content": "Now review the calling code that invokes process().",
            },
            {
                "role": "system",
                "content": "From now on, every suggestion must include explicit type annotations.",
            },
        ],
    )
    ```

=== "TypeScript"

    ```typescript
    const client = new Anthropic();

    const response = await client.messages.create({
      model: "claude-opus-5-5",
      max_tokens: 1024,
      cache_control: { type: "ephemeral" },
      system: "You are a code review assistant. Be concise.",
      messages: [
        {
          role: "user",
          content: "Review process() in utils.py for performance issues."
        },
        {
          role: "assistant",
          content:
            "The list comprehension is fine for small inputs. For large inputs, consider a generator to avoid materializing the full list."
        },
        {
          role: "user",
          content: "Now review the calling code that invokes process()."
        },
        {
          role: "system",
          content: "From now on, every suggestion must include explicit type annotations."
        }
      ]
    });
    ```

The rules that decide questions about this feature, from [the same page](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages):

- **Placement.** A `system` message that carries content cannot be the first entry in `messages`. It must immediately follow a `user` turn (a `user` turn that carries `tool_result` blocks counts) or an `assistant` turn ending in a server tool result, and it must either end the array or be followed by an `assistant` turn. Any other position for such a message, including between a `tool_use` block and its `tool_result`, returns a 400 error. The exception is an empty message that only sets `output_config.effort` (per-message effort, beta header `mid-conversation-output-config-2026-07-01`): it is accepted anywhere in `messages`, including first.
- **Precedence.** Later system messages take precedence over earlier ones, and a mid-conversation system message takes precedence over the top-level `system` field for the turns that follow it.
- **Operator, not user.** A `user` message is treated as coming from the end user and a `system` message as coming from the application operator; when the two conflict, system instructions take precedence. Use the `system` role for constraints that should hold even if the end user asks for something different, and phrase them as context: Claude "is trained to resist instructions that appear to work against the user, and that protection still applies to the system role", so state what changed rather than telling Claude to ignore the user.
- **Caching is opt-in.** The example turns on automatic caching with the top-level `cache_control` field. Without `cache_control` nothing is cached, so there is no prefix to preserve.
- **Edits cost more than appends.** Editing or removing a system message already sent invalidates the cache from that point; on Fable 5.1 and Opus 5.5 it also invalidates the thinking blocks in later assistant turns. Append a new system message instead.

The top-level field has the same sensitivity: the Opus 5.5 guide says to add its standing "unattended" instruction at the end of the system prompt from the first request, because adding it partway through "invalidates the conversation's earlier thinking blocks". Full request mechanics are in [How a Messages API call works](claude-api.md#how-a-messages-api-call-works).

!!! warning "Exam guide vs current docs: system prompts"

    The July 2026 guides speak of "system prompts" (CCAR-F lists "system prompts" among Claude API topics; CCAR-P says "Design system prompts, templates, and guardrails"), and none of the four guides mentions system-role messages. As of September 2026, the API reference still says "there is no `"system"` role for input messages in the Messages API", while the [Using the Messages API](https://platform.claude.com/docs/en/build-with-claude/working-with-messages) page documents `"role": "system"` messages after a user turn on seven models. For the exam, read "system prompt" as the top-level `system` parameter; know mid-conversation system messages as the current way to add operator-level instructions later without breaking the cache.

### Instructions across interfaces

The same instruction reaches Claude differently depending on the surface:

| Surface | Where standing instructions live | How they apply |
|---|---|---|
| Claude API | Top-level `system`; you re-send the whole conversation on every call | The API is stateless. Updates to Anthropic's claude.ai system prompt "do not apply to the Claude API" |
| Claude apps | Instructions for Claude (every conversation), project instructions (chats in that project), organization instructions (Owners and Primary Owners on Team and Enterprise) | Organization instructions take precedence over individual ones, but this "relies on prompt-level instructions" |
| Claude Code | CLAUDE.md files, concatenated from broadest to most specific; `--append-system-prompt` for system-level text | CLAUDE.md arrives "as a user message after the system prompt", so it is context, not enforcement |
| Agent SDK | `systemPrompt` or `system_prompt` option | Default is a minimal prompt without the `claude_code` preset's safety instructions; the preset plus `append` is "the lowest-risk customization" |

Two details are worth knowing, one from the Claude Code CLI and one from the Agent SDK. `--system-prompt` and `--system-prompt-file` replace the default prompt and are mutually exclusive; `--append-system-prompt` and `--append-system-prompt-file` add to it and can be combined with either replacement flag. In the Agent SDK, CLAUDE.md levels are additive with "no hard precedence rule", so conflicting rules resolve however Claude interprets them; write rules that do not conflict. Configuration details: [CLAUDE.md and the memory hierarchy](claude-code-configuration.md#claudemd-and-the-memory-hierarchy).

For app users (CCAO-F D5.1 and D5.3), the help center says Claude performs best with project instructions "for general context around your project, key guidelines, and Claude's role", with task-specific instructions kept for the chat itself. Claude Academy's [Claude 101 course](https://academy.claude.com/courses/claude-101/introduction-to-projects) lists what good project instructions typically include: context about the work, process instructions, tone and style preferences, and specific requirements such as "Always include a call-to-action at the end of marketing copy." The name and description you give a project do not reach Claude: the help center notes that "Claude will not have access to these details", so anything Claude needs belongs in the instructions or the project knowledge. See [Projects](claude-for-work.md#projects).

!!! warning "Docs vs the claude.com post: are XML tags and roles outdated?"

    The docs recommend XML tags and a role in the system prompt. The [claude.com prompting post](https://claude.com/blog/best-practices-for-prompt-engineering) (dated November 10, 2025) says "XML tags and heavy role prompting are less necessary with modern models", while granting that tags "can still be useful in specific situations". Both are official. Read them together: tags for complex prompts that mix instructions, data and examples; a light role rather than an elaborate persona; plain clear instructions for simple tasks.

### Decide

- If a prompt mixes instructions, documents, examples and user input, choose one descriptive tag per part; not one undifferentiated block.
- If code must act on part of the output, choose a tool call or structured output; not a regex over tagged prose.
- If an instruction becomes relevant mid-session on a supported model, append a system message; not an edit to `system`, which misses the cache for everything after it.
- If content comes from a third party, put it in a `tool_result`; not in `system` or a system message, where it gains operator-level authority.
- If the same large system prompt and policy go out on every request, order static content first and enable prompt caching (CCAR-P Sample 2); not truncation, a smaller model, or a few-shot block.
- If a tool order or business rule must always hold, enforce it in code, as CCAR-F Question 1 does with a programmatic prerequisite; not a stronger system-prompt instruction, which the rationale says relies on "probabilistic LLM compliance".
- If a Claude Code rule must be enforced, use a hook or a permission rule in settings; not a line in CLAUDE.md.
- If tool selection goes wrong after a system-prompt edit, review the prompt for keyword-sensitive instructions that override the tool descriptions (CCAR-F 2.1).

### Traps

- Treating XML tags around untrusted content as an injection defense.
- A system-prompt line asking users not to send malicious instructions, offered as an injection control (CCDV-F Sample 2, option C).
- Putting your own instructions inside `tool_result` content, where Claude may ignore them or flag them.
- A mid-conversation system message proposed for Claude Sonnet 5, which does not support it, or placed as the first entry in `messages`.
- A timestamp or request ID in the system prompt of a cached application.
- An over-specified persona (the claude.com post's "world-renowned expert who only speaks in technical jargon and never makes mistakes") presented as better prompting than a clear, light role.

## Chain of thought and prompt chaining

*Tested in: CCAR-F 1.6, APPX-TECH-11 · CCDV-F D5.1 LLM Fundamentals · CCAR-P 1.5, 2.3 · CCAO-F D1.2*

Both techniques give a hard problem more room, in different places. Chain of thought gives the model room to reason inside one call before it answers. Prompt chaining splits the task across several calls, so each call does one thing and your code sees every intermediate result. CCAR-P 2.3 names chain-of-thought as a technique to apply, and CCAR-P 1.5 asks you to "Apply decomposition techniques for complex problem solving"; CCAR-F's appendix lists prompt chaining as sequential task decomposition into focused passes, and CCAR-F 1.6 tests when to chain; CCAO-F D1.2 asks you to "Apply task decomposition techniques to structure complex requests"; CCDV-F D5.1 covers the model options behind reasoning (extended thinking, adaptive thinking, effort levels).

### Chain of thought on current models: thinking is built in

The term predates built-in reasoning. On current models, the reasoning step is a model feature you configure, not text you coax out of the response:

| Models | Thinking modes | When you omit `thinking` | Depth control |
|---|---|---|---|
| Opus 5.5, Fable 5.1, Mythos 5.1, Fable 5, Mythos 5 | Adaptive only | Always on; cannot be turned off | `effort` |
| Opus 5, Sonnet 5 | Adaptive only | On; `{type: "disabled"}` is accepted (on Opus 5 only at effort `high` or lower) | `effort` |
| Opus 4.8, Opus 4.7 | Adaptive only (`"enabled"` returns a 400 error) | Off until you set `thinking: {type: "adaptive"}` | `effort` |
| Opus 4.6, Sonnet 4.6 | Adaptive; extended thinking still works but is deprecated | Off until you set `thinking: {type: "adaptive"}` | `effort` (`budget_tokens` deprecated) |
| Opus 4.5, Haiku 4.5, Sonnet 4.5 | Extended only (`"adaptive"` returns a 400 error) | Off | `budget_tokens`; Opus 4.5 also accepts `effort` |

Four facts change how you design around it. Thinking tokens "are billed as output tokens, even when the thinking text isn't returned to you". The thinking text you can read is a summary, not the raw reasoning. `display` defaults to `"omitted"` on Opus 4.7, Opus 4.8, Opus 5, Opus 5.5, Sonnet 5 and the Fable and Mythos models, and to `"summarized"` on Opus 4.6, Sonnet 4.6 and earlier; omitting the text reduces latency, not cost. Setting `budget_tokens` returns a 400 error on Claude 4.7 and later. The request shape after migrating from a token budget, from [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices):

=== "Python"

    ```python
    client.messages.create(
        model="claude-opus-4-8",
        max_tokens=16000,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": "..."}],
    )
    ```

=== "TypeScript"

    ```typescript
    await client.messages.create({
      model: "claude-opus-4-8",
      max_tokens: 16000,
      thinking: { type: "adaptive" },
      output_config: { effort: "high" },
      messages: [{ role: "user", content: "..." }]
    });
    ```

Opus 4.8 defaults to `"omitted"`, so add `"display": "summarized"` inside `thinking` when you want to read the reasoning. Parameter details and per-model limits are in [Extended thinking, adaptive thinking and effort](claude-api.md#extended-thinking-adaptive-thinking-and-effort).

### Prompting the thinking

Quotes in this list come from [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) unless a bullet names another page:

- **General beats prescriptive.** "A prompt like "think thoroughly" often produces better reasoning than a hand-written step-by-step plan."
- **Effort first, wording second.** [Steering thinking](https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost) says lowering effort is "usually the better first lever, since it is a calibrated control rather than a wording-sensitive instruction." If reasoning is shallow on hard problems, [Prompting Claude Sonnet 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5) says to raise effort to `high` or `xhigh` "rather than prompting around it".
- **Per-turn nudges.** Steering thinking also says that appending "Please think hard before responding." to a user message encourages thinking on that turn; "Answer directly without deliberating." suppresses it.
- **Reflection after tools.** "After receiving tool results, carefully reflect on their quality and determine optimal next steps before proceeding."
- **Less thinking where it adds nothing.** Triggering is promptable: "Thinking adds latency and should only be used when it will meaningfully improve answer quality". For Opus 5.5 in a chat product, [Prompting Claude Opus 5.5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5-5) reports that removing a "think carefully" line "made replies start sooner, with no clear decline in the quality of the reply."
- **Examples can carry the reasoning pattern.** `<thinking>` tags inside few-shot examples show Claude how to reason; see [What a good example contains](#what-a-good-example-contains).
- **Self-checks, model by model.** "Before you finish, verify your answer against [test criteria]." catches errors, especially in coding and math, except on Opus 5, where such instructions cause over-verification and should be removed. CCAR-F goes further: its objective 4.6 says an independent review instance catches subtle issues better than self-review instructions or extended thinking; see [Multi-instance and multi-pass review](#multi-instance-and-multi-pass-review).

### Manual chain of thought: the fallback

When thinking is off, you can still ask for reasoning in the response. [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) calls this a fallback and recommends `<thinking>` and `<answer>` tags "to cleanly separate reasoning from the final output." The [claude.com prompting post](https://claude.com/blog/best-practices-for-prompt-engineering) (dated November 10, 2025) agrees that "When available, extended thinking is generally preferable to manual chain of thought prompting", but frames manual chain of thought more broadly: its cases include "Extended thinking isn't available (i.e. the free Claude.ai plan)" and "You need transparent reasoning that you can review", and it calls explicit chain of thought and extended thinking "complementary, not mutually exclusive." It describes three levels, shown here with its donor-email example:

| Level | What you add | Example wording from the post |
|---|---|---|
| Basic | A request to reason first | "Think step-by-step before you write the email." |
| Guided | Named reasoning stages | "First, think through what messaging might appeal to this donor given their donation history." |
| Structured | Tags that separate reasoning from the answer | "Think before you write the email in `<thinking>` tags." |

Reasoning before the verdict is also how Anthropic structures classification and grading. The ticket-routing prompt says "Remember to always include your classification reasoning before your actual intent output." For LLM graders, the evaluation guide says to "reason first before producing an evaluation score, and then discard the reasoning." Two cautions on current models: on Opus 5, prefer thinking at a lower effort to manual chain of thought, because with thinking disabled the model "can occasionally emit internal XML tags into its visible output"; and on Opus 4.5 with thinking disabled, the word "think" is a sensitive trigger, so "consider," "evaluate," or "reason through" are the suggested alternatives.

!!! warning "Exam guide vs current docs: chain-of-thought"

    CCAR-P 2.3 says "Apply prompt engineering techniques (zero-shot, few-shot, chain-of-thought)", and none of the four guides mentions the `reasoning_extraction` refusal category. As of September 2026 the docs treat manual chain of thought as a fallback for when thinking is off; the November 2025 [claude.com post](https://claude.com/blog/best-practices-for-prompt-engineering) also calls extended thinking generally preferable when available, yet still calls the two approaches complementary. The Fable 5 and Opus 5.5 guides go further: prompts that tell the model to reproduce its internal reasoning as response text can be declined under the `reasoning_extraction` refusal category, so read `thinking` blocks (with `display: "summarized"`) instead. Anthropic's cost post (September 8, 2026) lists fixed scaffolds such as "think step by step in a scratchpad" among the rituals frontier models do not need, and the engineering post on the "think" tool now recommends extended thinking over a dedicated think tool "in most cases". Our reading for the exam: answer in the guide's terms, so an option that gives the model room to reason on a multistep problem is the chain-of-thought answer; if the options also name current features, thinking at a suitable effort level is the current form of it, and chained calls are the form to pick when you must inspect the steps.

### Prompt chaining

Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) defines it: "Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one." Programmatic checks, which the post calls gates, can sit between steps to keep the process on track. It fits tasks that decompose cleanly into fixed subtasks, and "The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task." The post's examples: write marketing copy, then translate it; or write an outline, check it meets criteria, then write the document.

The current docs narrow the case for it. [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) says "With adaptive thinking and subagent orchestration, Claude handles most multistep reasoning internally", and that explicit chaining "is still useful when you need to inspect intermediate outputs or enforce a specific pipeline structure." The most common pattern is self-correction: generate a draft, have Claude review it against criteria, then refine it. "Each step is a separate API call so you can log, evaluate, or branch at any point." Two related points from Anthropic's material: in its parallelization section, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) says complex tasks go better "when each consideration is handled by a separate LLM call", and the [claude.com post](https://claude.com/blog/best-practices-for-prompt-engineering) says a focused task with clear boundaries "consistently produces higher quality results than trying to accomplish multiple objectives in a single prompt." Building effective agents also supplies the counterweight: "For many applications, however, optimizing single LLM calls with retrieval and in-context examples is usually enough."

A three-step self-correction chain with a gate (illustrative and ours, built from the documented `messages.create` call and response-text pattern; the three prompts adapt the claude.com post's research-summary chain, and the gate is a programmatic check of the kind Building effective agents describes):

=== "Python"

    ```python
    import anthropic

    client = anthropic.Anthropic()

    def ask(prompt):
        # Thinking is always on for this model, and thinking tokens count toward max_tokens
        response = client.messages.create(
            model="claude-opus-5-5",
            max_tokens=16000,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "max_tokens":
            raise RuntimeError("Response hit max_tokens; the output is incomplete.")
        return next(block.text for block in response.content if block.type == "text")

    paper = open("paper.txt").read()

    # Step 1: draft
    summary = ask(
        f"<paper>\n{paper}\n</paper>\n\n"
        "Summarize this medical paper covering methodology, findings, and clinical "
        "implications. Put each part in <methodology>, <findings> and <implications> tags."
    )

    # Gate: stop the chain if a required section is missing
    for tag in ("<methodology>", "<findings>", "<implications>"):
        if tag not in summary:
            raise ValueError(f"Draft is missing {tag}; not sending it to review.")

    # Step 2: review against criteria
    review = ask(
        f"<paper>\n{paper}\n</paper>\n<summary>\n{summary}\n</summary>\n\n"
        "Review the summary for accuracy, clarity, and completeness. Provide graded feedback."
    )

    # Step 3: refine
    final = ask(
        f"<paper>\n{paper}\n</paper>\n<summary>\n{summary}\n</summary>\n"
        f"<feedback>\n{review}\n</feedback>\n\n"
        "Improve the summary based on this feedback."
    )
    print(final)
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";
    import { readFileSync } from "node:fs";

    const client = new Anthropic();

    async function ask(prompt: string): Promise<string> {
      // Thinking is always on for this model, and thinking tokens count toward max_tokens
      const response = await client.messages.create({
        model: "claude-opus-5-5",
        max_tokens: 16000,
        messages: [{ role: "user", content: prompt }]
      });
      if (response.stop_reason === "max_tokens") {
        throw new Error("Response hit max_tokens; the output is incomplete.");
      }
      const textBlock = response.content.find(
        (block): block is Anthropic.TextBlock => block.type === "text"
      );
      return textBlock?.text ?? "";
    }

    const paper = readFileSync("paper.txt", "utf8");

    // Step 1: draft
    const summary = await ask(
      `<paper>\n${paper}\n</paper>\n\n` +
        "Summarize this medical paper covering methodology, findings, and clinical " +
        "implications. Put each part in <methodology>, <findings> and <implications> tags."
    );

    // Gate: stop the chain if a required section is missing
    for (const tag of ["<methodology>", "<findings>", "<implications>"]) {
      if (!summary.includes(tag)) {
        throw new Error(`Draft is missing ${tag}; not sending it to review.`);
      }
    }

    // Step 2: review against criteria
    const review = await ask(
      `<paper>\n${paper}\n</paper>\n<summary>\n${summary}\n</summary>\n\n` +
        "Review the summary for accuracy, clarity, and completeness. Provide graded feedback."
    );

    // Step 3: refine
    const final = await ask(
      `<paper>\n${paper}\n</paper>\n<summary>\n${summary}\n</summary>\n` +
        `<feedback>\n${review}\n</feedback>\n\n` +
        "Improve the summary based on this feedback."
    );
    console.log(final);
    ```

The claude.com post states the cost plainly: "Chaining increases latency (multiple API calls) but often dramatically improves accuracy and reliability for complex tasks."

### Fixed chains or adaptive decomposition

CCAR-F 1.6 asks when to use "fixed sequential pipelines (prompt chaining) versus dynamic adaptive decomposition based on intermediate findings":

| Pattern | Choose it when | The guide's example |
|---|---|---|
| Prompt chaining (fixed pipeline) | The steps are known in advance, as in predictable multi-aspect reviews | Analyze each file individually, then run a cross-file integration pass |
| Dynamic decomposition | The task is open-ended and each finding changes the next step | Adding tests to a legacy codebase: map the structure, identify high-impact areas, then follow a prioritized plan that adapts as dependencies appear |

CCAR-F sample Question 12 tests the fixed-chain case on a 14-file pull request; the item, its rationale and the review architecture are in [What sample question 12 teaches](#what-sample-question-12-teaches). Routing, parallelization, orchestrator-workers and evaluator-optimizer are in [Workflow patterns](agents-and-agent-sdk.md#workflow-patterns).

### Decomposition without code

In the Claude apps, the person is the orchestrator. The claude.com post notes that chaining is usually built with workflows or code, "but you could manually provide the prompts after receiving responses." The help center's advice for weak answers includes "Break down complex requests into substeps." The "AI capabilities and limitations" course on Claude Academy adds a checkpoint habit against reasoning drift, where small errors compound: ask Claude "to stop and show you the result of step 2 before continuing".

### Decide

- If a problem needs multistep reasoning on a current model, rely on thinking and tune `effort`; not a hand-written step-by-step scaffold.
- If you must log, inspect or gate the intermediate results, or enforce a fixed pipeline, chain separate calls; not one prompt that asks Claude to print its reasoning.
- If the steps are predictable, use a fixed chain; if each finding changes the next step, use adaptive decomposition.
- If a large review gives shallow or contradictory feedback, split it into per-file passes plus an integration pass; not a higher-tier model with a larger context window.
- If one well-built call with retrieval and examples already meets the success criteria, stop there; chaining adds latency.

### Traps

- "think step by step in a scratchpad" appended to every request to a frontier model.
- A system prompt that asks the model to write out its internal reasoning in the answer (risking a `reasoning_extraction` refusal on Fable 5.1, Opus 5.5 and Fable 5).
- A self-review instruction offered as the fix for missed issues when an independent reviewer is available (CCAR-F 4.6).
- A fixed pipeline for an investigation whose next step depends on what the last one found.
- For an uneven multi-file review, Question 12's distractors (developer-side pull request splitting, or two-of-three consensus across full passes); see [What sample question 12 teaches](#what-sample-question-12-teaches).
- Setting `budget_tokens` on Claude 4.7 or later (400 error).

## Long documents

*Tested in: CCAR-F 5.1 (5.1-K2, 5.1-S4 to 5.1-S6), APPX-TECH-12, APPX-INSCOPE-16 · CCDV-F D6.1 Context Engineering, D6.2 Prompt Engineering · CCAR-P 2.4 · CCAO-F D3.4*

A long input changes the prompt's job: help Claude find what matters in a lot of text, then answer from it. Anthropic's long-context guidance applies "When working with large documents or data-rich inputs (20k+ tokens)", and it comes down to three rules: long material at the top, one tagged block per document, and quotes before the task. This section covers prompting over long inputs. Keeping a long conversation's facts intact across turns is in [Preserving critical information in long conversations](context-engineering.md#preserving-critical-information-in-long-conversations).

### Three rules from the docs

| Rule | What the docs say |
|---|---|
| Documents first, question last | Put long documents and inputs "near the top of your prompt, above your query, instructions, and examples. This improves performance across all models." Queries at the end "can improve response quality by up to 30 percent in tests, especially with complex, multidocument inputs." |
| One tagged block per document | Wrap each document in `<document>` tags with `<document_content>` and `<source>` (and other metadata) subtags |
| Quotes before the task | "For long document tasks, ask Claude to quote relevant parts of the documents first before carrying out its task." It helps Claude focus on the relevant content and ignore the rest |

The docs' multidocument layout, verbatim from [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices):

```xml
<documents>
  <document index="1">
    <source>annual_report_2023.pdf</source>
    <document_content>
      {{ANNUAL_REPORT}}
    </document_content>
  </document>
  <document index="2">
    <source>competitor_analysis_q2.xlsx</source>
    <document_content>
      {{COMPETITOR_ANALYSIS}}
    </document_content>
  </document>
</documents>

Analyze the annual report and competitor analysis. Identify strategic advantages and recommend Q3 focus areas.
```

For the quote-first habit, the same page's diagnostic example ends its prompt with: "Place these in `<quotes>` tags. Then, based on these quotes, list all information that would help the doctor diagnose the patient's symptoms. Place your diagnostic information in `<info>` tags." In our reading, the quotes step also makes the answer checkable: a reviewer can compare each quote with the source.

### "Lost in the middle": the exam's wording and Anthropic's evidence

CCAR-F 5.1 defines the effect as: models "reliably process information at the beginning and end of long inputs but may omit findings from middle sections". The skill it tests is "Placing key findings summaries at the beginning of aggregated inputs and organizing detailed results with explicit section headers to mitigate position effects", and the appendix lists "position-aware input ordering" as in scope.

What that skill looks like in an aggregated input (illustrative and ours, following 5.1-S4 and the metadata that 5.1-S5 asks subagents to carry; the double-brace slots are placeholders). The question and instructions still go after it, as the docs' layout rule says:

```text
KEY FINDINGS (SUMMARY)
1. {{FINDING_1}} (source: {{SOURCE_A}}, {{DATE_A}})
2. {{FINDING_2}} (source: {{SOURCE_B}}, {{DATE_B}})

DETAILED RESULTS: {{SOURCE_A}}
{{RESULTS_A}}

DETAILED RESULTS: {{SOURCE_B}}
{{RESULTS_B}}
```

!!! warning "Exam guide vs current docs: lost in the middle"

    The phrase "lost in the middle" does not appear in [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices); what that page documents is long data at the top and the query at the end (up to 30 percent better in tests). The [claude.com prompting post](https://claude.com/blog/best-practices-for-prompt-engineering) says newer models' context awareness helps "address historical "lost-in-the-middle" issues", yet still advises: "structure your information clearly with the most critical details at the beginning or end." Anthropic's September 2023 study ([Prompt engineering for Claude's long context window](https://www.anthropic.com/news/prompting-long-context)) found that "Claude 2 performance on 95K sees a small dip in the middle", that pulling relevant quotes into a scratchpad "is helpful in all head-to-head comparisons", and stressed "putting the instructions at the end of the prompt". On the exam, answer with the guide's fix: key findings summarized at the beginning, detail organized under explicit section headers.

### Why a bigger context window is not the fix

Anthropic's context engineering post describes context rot: "as the number of tokens in the context window increases, the model’s ability to accurately recall information from that context decreases." It frames context as an attention budget that every token draws down, and the effect as "a performance gradient rather than a hard cliff". CCAR-F sample Question 12 applies this when it rejects a higher-tier model with a larger context window as the fix for a many-file review (see [What sample question 12 teaches](#what-sample-question-12-teaches)). Send less and structure it better: see [Why context is a budget](context-engineering.md#why-context-is-a-budget).

### When the document should not go in whole

| Situation | Documented approach |
|---|---|
| Longer than the context window, or many related documents | Meta-summarization: split into chunks, summarize each, then summarize the summaries |
| A batch job where some documents exceeded context limits | Resubmit only the failed documents, identified by `custom_id`, after chunking them (CCAR-F 4.5) |
| An agent pipeline where the next agent has a small context budget | Have upstream agents return structured key facts, citations and relevance scores instead of verbose content (CCAR-F 5.1) |
| A corpus too large to send whole, or one where data freshness or access control drives the design | Retrieve the relevant passages instead of sending everything. The 2026 Anthropic and Accenture pilot-to-production guide says "Retrieval layers still add genuine value when data freshness or access control are the primary drivers", while many pipelines solve a context-window limit that no longer applies. See [Retrieval-augmented generation](solution-architecture.md#retrieval-augmented-generation) |

The legal summarization guide's chunker splits text into 20,000-character pieces, summarizes each, and combines them with a final prompt. Two excerpts, verbatim from [Legal summarization](https://platform.claude.com/docs/en/about-claude/use-case-guides/legal-summarization): the chunking function, and the opening of the final prompt, which is a Python f-string (the `{"".join(chunk_summaries)}` slot joins the chunk summaries):

```python
def chunk_text(text, chunk_size=20000):
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
```

```text
You are looking at the chunked summaries of multiple documents that are all related.
Combine the following summaries of the document from different truthful sources into a coherent overall summary:

<chunked_summaries>
{"".join(chunk_summaries)}
</chunked_summaries>
```

The guide notes that even when the document fits, "this meta-summarization technique often captures additional important details in the final summary that were missed in the earlier single-summary approach." Batch mechanics are in [Batch processing design](#batch-processing-design).

### Grounding answers in the source

Two tools keep answers tied to the text. The prompt-level one is quote-first extraction; the hallucination guidance uses it for documents over 20k tokens and has Claude state "No relevant quotes found." when nothing matches (see [Reducing hallucinations](#reducing-hallucinations)). The API-level one is citations:

- Enable per document with `citations.enabled=true`; it must be on for all or none of the documents in a request.
- Locations come back as `char_location` (plain text, 0-indexed characters), `page_location` (PDF, 1-indexed pages) or `content_block_location` (custom content, 0-indexed blocks), with exclusive end indices.
- `cited_text` "does not count toward your output tokens", and citations "are guaranteed to contain valid pointers to the provided documents." In Anthropic's evaluations the feature was also more likely than prompt-based approaches to cite the most relevant quotes.
- A document's `title` and `context` are passed to the model but are not citable.
- Citations cannot be combined with `output_config.format`: the request returns a 400 error, because citations interleave citation blocks with text. When you need a fixed JSON shape and grounding together, one option (ours, not a documented pattern) is to carry source fields such as the quote and its location inside your own schema.

A citations request, with the document first and the question last (cURL body excerpt from [Citations](https://platform.claude.com/docs/en/build-with-claude/citations)):

```json
{
  "model": "claude-opus-5-5",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": [
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
        },
        {
          "type": "text",
          "text": "What color is the grass and sky?"
        }
      ]
    }
  ]
}
```

More on document inputs (PDFs, files, search results) is in [Files, citations and search results](claude-api.md#files-citations-and-search-results) and [Vision and PDF input](claude-api.md#vision-and-pdf-input).

### Long documents in the Claude apps

For CCAO-F D3.4, the app-side mechanics differ from the API. When a project's knowledge approaches the context window limit, Claude "will automatically enable RAG mode", which expands the project's capacity by up to 10x, and uses a project knowledge search tool to pull relevant passages; RAG for projects is on paid plans (Pro, Max, Team, Enterprise). Clear, descriptive file names help retrieval, and naming a specific document in your question helps Claude focus its search. If a single conversation hits its length limit, the help center's advice is to start a new conversation or use projects. See [Projects](claude-for-work.md#projects).

### Decide

- If a prompt carries long documents, put them first, tag each with its source, and put the question and instructions last; not the question first.
- If key findings get missed in aggregated inputs, put a summary of key findings at the beginning and organize detail under explicit headers; not a bigger context window.
- If the answer must be traceable, ask for quotes first or turn on citations; if you also need strict JSON, do not combine citations with `output_config.format` (400 error); carrying source fields in your own schema is our suggested workaround.
- If the text exceeds the window, chunk and meta-summarize or retrieve; in batch jobs, resubmit only the oversized documents after chunking.

### Traps

- Answer options that fix attention or position effects by switching to a higher-tier model with a larger context window (CCAR-F Question 12, option C).
- The Claude 2.1 technique of starting Claude's reply with "Here is the most relevant sentence in the context:". It was a prefill, and Claude 4.6 and later models (and Claude Mythos Preview) reject a prefilled final assistant turn with a 400 error; on those models, put the docs' quote-first instruction in the user turn instead.
- Citations enabled on some documents but not others in one request.
- Citations and structured outputs requested together.

## Structured output with tools and JSON schemas

*Tested in: CCAR-F 4.3 (4.3-K1 to 4.3-S6), 2.3-S4, 2.3-S5, 3.6-K2, 3.6-S2, Exercise 3 step 1, Scenario 6, APPX-TECH-5, APPX-TECH-7, APPX-INSCOPE-13 · CCDV-F D6.3 Output Handling, D2.5 Claude Application Design (schema design), D8.1 Tool Implementation · CCAO-F D2.6 (structured data as an output format)*

CCAR-F Scenario 6 describes the job this section serves: "The system extracts information from unstructured documents, validates the output using JavaScript Object Notation (JSON) schemas, and maintains high accuracy." ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)) This section is built around three design decisions: which mechanism produces the structure, how the schema is shaped so it never forces Claude to invent a value, and how your code reads the result. The API parameters, the supported JSON Schema features and the complexity limits are in [Structured outputs](claude-api.md#structured-outputs); the `tool_choice` mechanics are in [Controlling tool choice](tool-use-and-mcp.md#controlling-tool-choice).

### Choose the mechanism

| If you need | Use | Why |
|---|---|---|
| Claude's final answer as one fixed JSON shape (a report, a record of extracted fields) | JSON outputs: `output_config.format` with `type: "json_schema"` | Constrained decoding makes the reply valid JSON that matches the schema, returned in the text content block |
| Claude to call your function with valid arguments, or to pick one of several extraction schemas | One tool per schema, each with `"strict": true` | Strict mode guarantees the tool `input` follows `input_schema` and the tool `name` is valid |
| Valid tool calls during an agent loop and a structured final answer | Both features in the same request | JSON outputs shape what Claude says; strict tool use validates how Claude calls your functions |
| A classification label | A tool with an `enum` field, or structured outputs | The prompting guide's recommendation for classification in place of prefill, which returns a 400 error from Claude 4.6 models onward |
| Machine-parseable findings from Claude Code in CI | `claude -p` with `--output-format json` and `--json-schema`; read `structured_output` | The CLI validates the result against your schema after the agent finishes |
| Validated JSON at the end of a multi-step Agent SDK run | The `outputFormat` option (TypeScript) or `output_format` option (Python, set in `ClaudeAgentOptions`) passed to `query()`; read `structured_output` | The agent can use any tools on the way and still returns schema-valid data |

Two rules decide the rest. First, "if you're writing a regex to extract a decision from model output, that decision should have been a tool call" ([How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works)). Second, an instruction alone guarantees nothing: without structured outputs, Claude "can generate malformed JSON responses or invalid tool inputs that break your applications" ([Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)). Prompt-only JSON (an instruction to answer in JSON, with no schema enforcement) is still a documented option. The prompting best-practices page, in its advice on replacing prefill, says "Try asking the model to conform to your output structure first, as newer models can reliably match complex schemas when told to, especially if implemented with retries" ([Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)). So prompt-only JSON needs a validator and a retry behind it; when every reply must parse, use constrained decoding. Prefills "have been used to force specific output formats like JSON/YAML", but a prefilled last assistant turn returns a 400 error from Claude 4.6 models onward, and prefilling is incompatible with JSON outputs. On a CCAR-F item, the guide's answer for guaranteed schema-compliant output is tool use with a JSON schema (4.3-K1), not a prompt instruction.

### The extraction-tool pattern

CCAR-F skill 4.3-S1 is "Defining extraction tools with JSON schemas as input parameters and extracting structured data from the tool_use response". The tool is a schema carrier: your code reads its input and has nothing to execute. You define a tool whose `input_schema` is the record you want, tell Claude to use it, and read the `input` of the `tool_use` block, which conforms to the schema. Anthropic's cookbook "Extracting Structured JSON using Claude and Tool Use" does exactly this with tools such as `print_summary` and `print_entities`.

The extraction tool used in this section and the next two (adapted: built from documented schema features; the descriptions are ours; the field idioms `unclear`, `other` plus detail, `calculated_total` and `conflict_detected` are the CCAR-F guide's):

```json
{
  "name": "extract_invoice",
  "description": "Record the fields extracted from one invoice. Use null for any field the document does not state; never infer or invent a value. Exception: calculated_total and conflict_detected are computed by you.",
  "strict": true,
  "input_schema": {
    "type": "object",
    "properties": {
      "invoice_number": {"type": ["string", "null"]},
      "invoice_date": {
        "type": ["string", "null"],
        "description": "The invoice date written as YYYY-MM-DD"
      },
      "line_items": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "description": {"type": "string"},
            "amount": {"type": "number"}
          },
          "required": ["description", "amount"],
          "additionalProperties": false
        }
      },
      "stated_total": {"type": ["number", "null"]},
      "calculated_total": {
        "type": ["number", "null"],
        "description": "Your own sum of the line_items amounts, not a value copied from the document"
      },
      "conflict_detected": {
        "type": "boolean",
        "description": "true if the document contradicts itself, such as a stated total that differs from its line items"
      },
      "document_type": {"type": "string", "enum": ["invoice", "credit_note", "unclear", "other"]},
      "document_type_detail": {"type": ["string", "null"]}
    },
    "required": ["invoice_number", "invoice_date", "line_items", "stated_total", "calculated_total", "conflict_detected", "document_type", "document_type_detail"],
    "additionalProperties": false
  }
}
```

Call it and read the result (illustrative; the `max_tokens` value is our choice):

=== "Python"

    ```python
    import anthropic

    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,  # leaves room for thinking, which counts toward max_tokens
        tools=[EXTRACT_INVOICE_TOOL],  # the definition above
        messages=[
            {
                "role": "user",
                "content": f"<document>\n{document_text}\n</document>\n\nUse the extract_invoice tool.",
            }
        ],
    )

    extraction = None
    for block in response.content:
        if block.type == "tool_use" and block.name == "extract_invoice":
            extraction = block.input
            break
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";

    const client = new Anthropic();

    const response = await client.messages.create({
      model: "claude-opus-5-5",
      max_tokens: 16000, // leaves room for thinking, which counts toward max_tokens
      tools: [extractInvoiceTool], // the definition above, typed as Anthropic.Tool
      messages: [
        {
          role: "user",
          content: `<document>\n${documentText}\n</document>\n\nUse the extract_invoice tool.`
        }
      ]
    });

    // The tool_use block is not always first: find it in the content array.
    const toolUse = response.content.find(
      (block): block is Anthropic.ToolUseBlock =>
        block.type === "tool_use" && block.name === "extract_invoice"
    );
    const extraction = toolUse?.input;
    ```

Four details in that code are deliberate. `tool_choice` stays at its default, `auto` (the default whenever `tools` are provided), and the user message names the tool; that combination also works on the models that reject forced tool use. Both versions search by block type and name, because text can precede a `tool_use` block and, on Claude Opus 5.5, the first content block can be a `thinking` block. `max_tokens` is far above the size of one record because thinking tokens are a subset of `max_tokens` and thinking is always on for Claude Opus 5.5; the value stays under 21,333, above which the SDKs require streaming. And because `auto` lets Claude answer in text instead, the code must treat a missing extraction as a normal outcome, which [Validation, retry and feedback loops](#validation-retry-and-feedback-loops) does.

### Shape the schema so it cannot force invention

| Design choice | Rule | Where it comes from |
|---|---|---|
| Nullable fields for anything the source may lack | A required, non-null field pushes the model to fabricate a value to satisfy it; make it optional or nullable | CCAR-F 4.3-S4; [Agent SDK structured outputs](https://code.claude.com/docs/en/agent-sdk/structured-outputs) ("make those fields optional") |
| Required versus optional | Demand a non-null value only when every valid document has it; otherwise make the field nullable (still listed in `required`) or optional | CCAR-F 4.3-K4, 4.3-S4 (the rule is our reading); [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), tip for schemas that hit complexity limits ("Make parameters `required` where possible.") |
| An `"unclear"` enum value | Gives ambiguous cases an honest answer instead of a forced guess | CCAR-F 4.3-S5 |
| `"other"` plus a detail string | Keeps a closed enum extensible: `document_type: "other"` with `document_type_detail` saying what it is | CCAR-F 4.3-K4, 4.3-S5 |
| Normalization rules in the prompt | The schema fixes types; the prompt says how to convert inconsistent source formats | CCAR-F 4.3-S6 |
| `additionalProperties: false` on every object | Required for objects in strict schemas | Structured outputs docs |
| Focused, shallow schemas | Deeply nested schemas with many required fields are harder to satisfy | Agent SDK docs |

The guide's skill states the reason for the first row in full: "Designing schema fields as optional (nullable) when source documents may not contain the information, preventing the model from fabricating values to satisfy required fields" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Preparation Exercise 3 tests it: process documents where some fields are absent and verify that the model returns null rather than fabricating values. The `unclear` value, the `other` plus detail pattern, `calculated_total` and `conflict_detected` come from the CCAR-F guide; the structured outputs, strict tool use and tool definition pages and the structured JSON extraction cookbook do not describe them, so learn them as the exam's patterns.

**Nullable and required, or optional?** Both keep Claude from inventing values, and in strict schemas they cost different budgets. Required properties are emitted first in schema order and optional ones after them. A request may carry at most 24 optional parameters and 16 union-typed parameters (such as `"type": ["string", "null"]`) across all strict schemas, and each optional parameter roughly doubles part of the grammar's state space. Among its tips for schemas that hit complexity limits, [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) advises "Make parameters `required` where possible." The schema above makes every property required and lets five of them be null (our count), so it uses no optional parameters and 5 of the 16 union slots. Required-plus-nullable follows the docs' tip and the guide's 4.3-S4 (our reading), but it spends union slots, which the docs call "especially expensive because they create exponential compilation cost". Budget nullable fields in large schemas; do not drop them.

Normalization rules go in the prompt next to the schema (an illustrative example):

```text
Normalization rules:
- Write every date as YYYY-MM-DD. If the day and month cannot be told apart (for example 03/04/2026), return null for invoice_date.
- Write amounts as plain numbers with no currency symbol or thousands separator.
- If the document is clearly neither an invoice nor a credit note, set document_type to "other" and say what it is in document_type_detail. If you cannot tell, use "unclear".
```

Constraints the grammar cannot enforce belong in descriptions and validation code. Numerical and string-length constraints such as `minimum` and `maxLength` return a 400 in a strict schema; the SDK helpers strip them, move them into the field description (a `minimum: 100` becomes "Must be at least 100") and validate the response against the original constraint. String formats are the exception: `date-time`, `time`, `date`, `duration`, `email`, `hostname`, `uri`, `ipv4`, `ipv6` and `uuid` are supported. String `enum` values may come back with different capitalization, so compare them case-insensitively.

### Making sure a tool is called

The guide's three settings: `tool_choice: "auto"` means the model may return text instead of calling a tool; `"any"` means it must call a tool but chooses which, the guide's answer "when multiple extraction schemas exist and the document type is unknown"; a forced tool (`{"type": "tool", "name": "extract_metadata"}`) makes a particular extraction run before enrichment steps, with later steps in follow-up turns.

!!! warning "Exam guide vs current docs"

    The CCAR-F guide (July 2026) calls "Tool use (tool_use) with JSON schemas" "the most reliable approach for guaranteed schema-compliant structured output, eliminating JSON syntax errors", and teaches `"any"` and forced tool selection. The docs, as of September 2026, add two things. First, a plain tool schema is not a guarantee: "Without strict mode, Claude might return incompatible types (`"2"` instead of `2`) or omit required fields" ([Strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use)); the guarantee comes from `strict: true` or `output_config.format`, which matches the guide appendix's "strict mode for syntax error elimination". Second, Claude Opus 5.5, Claude Fable 5.1 and Claude Mythos 5.1 reject `tool_choice` `any` and `tool` with a 400 error, and on any model those two types error under manual extended thinking (`thinking: {type: "enabled"}`). They still work on other models with adaptive thinking, for example Claude Opus 5. Where they are rejected, use `auto` with strict tools and say in the prompt when the tool applies, or use JSON outputs for a fixed response shape. On the exam, answer in the guide's terms: `any` guarantees some tool call, a forced tool guarantees a specific one.

### Structured output from Claude Code in CI

CCAR-F 3.6-S2 uses `--output-format json` with `--json-schema` "to produce machine-parseable structured findings for automated posting as inline PR comments". The response carries run metadata (session ID, usage) plus the schema-conforming data in `structured_output`:

```bash
claude -p "Extract function names from auth.py" \
  --output-format json \
  --json-schema '{"type":"object","properties":{"functions":{"type":"array","items":{"type":"string"}}},"required":["functions"]}' \
  | jq '.structured_output'
```

`--json-schema` works in print mode only. An invalid schema makes `claude` exit with `Error: --json-schema is not a valid JSON Schema`; before v2.1.205 an invalid schema was silently ignored. Claude Code accepts the `format` keyword (such as `"format": "email"`) but treats it as an annotation and does not enforce it. For scripted calls the docs recommend `--bare`, which skips auto-discovery of hooks, skills, custom commands, subagents, plugins, MCP servers, auto memory and CLAUDE.md. Pipeline wiring is in [Claude Code in CI/CD](claude-code-workflows.md#claude-code-in-cicd).

### Decide

- If the answer itself is the data, choose JSON outputs; if Claude must pick among several record types or call your code, choose strict tools; if you are writing a regex over prose, switch to a tool call.
- If a field can be missing from the source, make it nullable (still listed in `required`) or optional; not a required non-null field plus "do your best", because a required non-null field invites a fabricated value.
- If categories are open-ended, add `"other"` with a detail field; if cases are ambiguous, add `"unclear"`; not a longer enum list.
- If source formats vary, write normalization rules into the prompt beside the schema; the schema alone only fixes types.
- If a guide-style item asks how to guarantee a tool call, answer `any` (some tool) or a forced tool (a specific one); not `auto`, which may return text.
- If a guide-style item asks for the most reliable way to get schema-compliant output, answer tool use with a JSON schema; not a prompt instruction to reply in JSON.

### Traps

- Believing a schema makes the content right. Strict schemas "eliminate syntax errors but do not prevent semantic errors (e.g., line items that don't sum to total, values in wrong fields)" (CCAR-F 4.3-K3).
- Every field required and non-null, then puzzling over invented invoice numbers and dates.
- Reading `response.content[0]` and assuming it is the tool call or the text.
- `tool_choice: "auto"` in a design that needs a record every time, with no handling for a text reply.
- Copying forced `tool_choice` code onto Claude Opus 5.5, Fable 5.1 or Mythos 5.1 (400 error).
- `minimum` or `maxLength` in a strict schema sent as it is, without an SDK helper that moves them into descriptions, or a recursive schema (400 error).
- A prompt instruction to "reply in JSON" with no schema enforcement and no validator, treated as a guarantee.
- A `max_tokens` sized for the record alone on a model that always thinks, so the reply is cut off at `max_tokens`.

## Validation, retry and feedback loops

*Tested in: CCAR-F 4.4 (4.4-K1 to 4.4-S4), 4.3-K3, Exercise 3 step 2, APPX-TECH-8 (Pydantic) · CCDV-F D6.3 Output Handling (response validation, defensive parsing), D4.1 Debugging and Error Handling · CCAO-F D2.3 (fact-checking and validation), D7.2 (adjusting to feedback and results) · CCAR-P 1.2 (feedback loops)*

A schema-valid extraction can still be wrong. The CCAR-F guide draws the line between "semantic validation errors (values don't sum, wrong field placement)" and "schema syntax errors (eliminated by tool use)" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Constrained decoding removes the second kind; your code has to catch the first, feed specific failures back to Claude, and recognize the failures no retry can fix. This section uses the `extract_invoice` tool from [Structured output with tools and JSON schemas](#structured-output-with-tools-and-json-schemas).

### Four checks, in order

| Check | What it catches | Who performs it |
|---|---|---|
| 1. Stop reason | `stop_reason: "refusal"` (HTTP 200, billed, and the output may not match the schema because the refusal takes precedence) and `"max_tokens"` (the output may be incomplete) | Your code, before parsing anything |
| 2. A result exists | With `tool_choice: "auto"`, a text reply instead of a tool call; in the Agent SDK, a result with subtype `success` but no `structured_output` | Your code |
| 3. Schema | Wrong types, missing required fields, invalid tool names | Constrained decoding (`strict: true` or `output_config.format`); the SDK helpers also validate against constraints the API cannot enforce, such as `minimum` |
| 4. Semantics | Line items that do not sum to the total, a value in the wrong field, a value that breaks a normalization rule the schema does not encode, cross-field rules | Your validator: Pydantic or plain code |

Parse defensively at every step: select content blocks by `type` rather than position (the first block can be `thinking`), compare string `enum` values case-insensitively, and treat an Agent SDK `success` without `structured_output` as a failure; the docs say "Treat that case as a failure as well." ([Agent SDK structured outputs](https://code.claude.com/docs/en/agent-sdk/structured-outputs))

Pydantic, which the CCAR-F appendix lists for "schema validation, semantic validation errors, validation-retry loops", fits two documented places. The Python SDK's `client.messages.parse()` takes a Pydantic model; "The `parse()` method automatically transforms your Pydantic model, validates the response, and returns a `parsed_output` attribute." ([Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs)) Anthropic's Pydantic tool-use cookbook does the same for tool calls: it constructs Pydantic models from the tool input to validate it. Either way, the cross-field rules in check 4 are still code you write, and Exercise 3 step 2 starts its retry loop "when Pydantic or JSON schema validation fails".

### Retry with error feedback

The guide defines the technique as "appending specific validation errors to the prompt on retry to guide the model toward correction", and the skill as follow-up requests "that include the original document, the failed extraction, and specific validation errors for model self-correction". Anthropic's Agent SDK post gives the principle behind it: "The best form of feedback is providing clearly defined rules for an output, then explaining which rules failed and why." ([Building agents with the Claude Agent SDK](https://claude.com/blog/building-agents-with-the-claude-agent-sdk)) A generic retry message with no specifics gives Claude nothing to correct.

A follow-up message with all three parts (illustrative, not an official example; the three-part content is the guide's 4.4-S1 pattern):

```json
{
  "role": "user",
  "content": "<document>...original invoice text...</document>\n<previous_extraction>{\"line_items\": [...], \"stated_total\": 500.00}</previous_extraction>\n<validation_errors>\n- line_items amounts sum to 450.00 but stated_total is 500.00: re-read the line items and correct any value copied into the wrong field\n- invoice_date \"15/01/2024\" does not match the required YYYY-MM-DD format\n</validation_errors>\nReturn a corrected extraction. If a value is not present in the document, return null for that field instead of guessing."
}
```

### When a retry works, and when it cannot

| Failure | Will a retry fix it? | Do this |
|---|---|---|
| Format mismatch (a date in the wrong format, a number written as text) | Yes | Retry with the specific error |
| Structural output error | Yes | Retry with the specific error; with constrained decoding these should not occur |
| Value in the wrong field, totals that do not add up because of a misread | Often, when the correct value is in the document (our reading) | Retry with the discrepancy spelled out, as above |
| Output cut off at `max_tokens` | Yes | Retry with a higher `max_tokens` |
| Information absent from the document | No | Return null, flag the record, route it to review |
| Information that exists only in an external document not provided | No | Supply that document, or route to review |
| The source contradicts itself (stated total differs from the line items) | No (our reading) | Record the conflict and route to human review |

The "Yes" verdicts for format and structural errors and the "No" verdicts for absent or external information are the guide's (4.4-S2, 4.4-K2); the `max_tokens` row is the structured outputs docs'. The other verdicts and the whole "Do this" column are our routing, built on those facts and on 4.4-S4.

The guide's wording on the limit is the one to remember: "retries are ineffective when the required information is simply absent from the source document (vs format or structural errors)". Preparation Exercise 3 asks you to track exactly this, which errors a retry resolved (format mismatches) and which it could not (information absent from the source).

### Self-correction fields

CCAR-F 4.4-S4 designs the check into the schema: extract `calculated_total` alongside `stated_total` so code can flag a discrepancy, and add a `conflict_detected` boolean for inconsistent source data. Our reading of why the pair helps: it separates two very different cases. If Claude's own `calculated_total` disagrees with the line items it extracted, the extraction is wrong and a retry with that error can fix it. If the document's printed total disagrees with its own line items and Claude says so with `conflict_detected: true`, the extraction is faithful and the document is inconsistent, so no retry can fix it; the record goes to a person. These field names are the guide's pattern, not fields from Anthropic's documentation.

### A validation loop in code

Illustrative Python built from the documented request, streaming and `tool_use` shapes; the retry cap, `max_tokens` values, helper names and tolerance are our choices, not official values.

```python
import json

import anthropic

client = anthropic.Anthropic()
MAX_ATTEMPTS = 3  # our choice: the guide sets no number

def request_extraction(document_text, feedback="", max_tokens=16000):
    # Streamed because the SDKs require streaming when max_tokens is above 21,333,
    # which the cut-off retry below uses.
    with client.messages.stream(
        model="claude-opus-5-5",
        max_tokens=max_tokens,  # thinking counts toward max_tokens
        tools=[EXTRACT_INVOICE_TOOL],
        messages=[
            {
                "role": "user",
                "content": f"<document>\n{document_text}\n</document>\n{feedback}\nUse the extract_invoice tool.",
            }
        ],
    ) as stream:
        response = stream.get_final_message()
    if response.stop_reason != "tool_use":
        # "refusal", "max_tokens", or "end_turn" (a text reply instead of a tool call)
        return None, response.stop_reason
    block = next(b for b in response.content if b.type == "tool_use")
    return block.input, response.stop_reason

def semantic_errors(data):
    errors = []
    items_sum = round(sum(item["amount"] for item in data["line_items"]), 2)
    calculated = data["calculated_total"]
    if calculated is not None and abs(items_sum - calculated) > 0.005:
        errors.append(f"calculated_total is {calculated} but line_items amounts sum to {items_sum}")
    stated = data["stated_total"]
    if stated is not None and abs(items_sum - stated) > 0.005 and not data["conflict_detected"]:
        errors.append(
            f"line_items amounts sum to {items_sum} but stated_total is {stated}: re-read the "
            "line items and correct any value copied into the wrong field, or set "
            "conflict_detected to true if the document itself disagrees"
        )
    return errors

def extract_with_retry(document_text):
    feedback = ""
    errors = []
    for _ in range(MAX_ATTEMPTS):
        data, stop_reason = request_extraction(document_text, feedback)
        if stop_reason == "max_tokens":
            # Cut off before the record was complete: rerun with more room.
            data, stop_reason = request_extraction(document_text, feedback, max_tokens=32000)
        if data is None:
            # "refusal", "end_turn" (a text reply instead of a tool call), or cut off again
            return {"status": "needs_review", "reason": stop_reason}
        errors = semantic_errors(data)
        if not errors:
            if data["conflict_detected"]:
                # The document disagrees with itself: a retry cannot fix that.
                return {"status": "needs_review", "reason": "conflict_detected", "data": data}
            return {"status": "ok", "data": data}
        feedback = (
            f"<previous_extraction>{json.dumps(data)}</previous_extraction>\n"
            "<validation_errors>\n"
            + "\n".join(f"- {error}" for error in errors)
            + "\n</validation_errors>\n"
            "Return a corrected extraction. If a value is not present in the document, "
            "return null for that field instead of guessing.\n"
        )
    return {"status": "needs_review", "reason": "retries_exhausted", "errors": errors}
```

Each retry is a fresh request that carries the document, the previous extraction and the errors, so no `tool_result` bookkeeping is needed. If you continue the same conversation instead, the user message must answer the `tool_use` block with a `tool_result` first, because `tool_result` blocks must come first in the content array.

### Retry budgets already built into Anthropic tools

| Layer | What happens on invalid output | Limit |
|---|---|---|
| Messages API, client tool call missing parameters, error returned in a `tool_result` with `"is_error": true` | Claude retries with corrections before apologizing to the user | 2 to 3 times, per the docs; `strict: true` removes invalid calls |
| Claude Code `-p` with `--json-schema`, and workflow subagents with a schema | Claude Code makes further attempts when the output fails validation; after the last one the run (or the workflow call) fails | `MAX_STRUCTURED_OUTPUT_RETRIES`, the number of attempts, default 5 (a first attempt plus four retries) |
| Agent SDK `outputFormat` / `output_format` | The SDK validates and re-prompts on mismatch | A retry limit exists (the [agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) page calls it "the configured retry limit"), but neither that page nor the Agent SDK structured outputs page states its value; when it is exhausted, or a model fallback retracts a completed output and no retry replaces it, the result subtype is `error_max_structured_output_retries` |
| Managed Agents outcomes | A grader evaluates the artifact against a rubric in a separate context window, and its feedback goes back to the agent for the next iteration | `max_iterations`, default 3, maximum 20 |

There is no single official retry count, so name the layer when you quote one. [Agent SDK structured outputs](https://code.claude.com/docs/en/agent-sdk/structured-outputs) lists why structured output fails: "the schema is too complex for the task, the task itself is ambiguous, or the agent hits its retry limit trying to fix validation errors". The first two are design problems that more retries do not solve (our reading); the docs' own tips are to keep schemas focused and to make fields optional when the task might not have all the information. How to shape the `is_error` message itself is in [Returning tool results and errors](tool-use-and-mcp.md#returning-tool-results-and-errors).

### Feedback loops from production: `detected_pattern`

Validation catches bad output before it ships; a feedback loop catches bad prompts after they ship. CCAR-F 4.4-K3 describes "tracking which code constructs trigger findings (detected_pattern field) to enable systematic analysis of dismissal patterns", and 4.4-S3 adds the field to structured review findings "to enable analysis of false positive patterns when developers dismiss findings". A finding shaped that way (illustrative; location, issue, severity and suggested fix follow the guide's 4.2-S2 format):

```json
{
  "file": "src/billing/refund.py",
  "line": 88,
  "severity": "high",
  "issue": "Refund amount is not compared with the original charge before process_refund is called",
  "suggested_fix": "Reject the refund when amount exceeds charge.amount",
  "detected_pattern": "unchecked_amount_before_side_effect"
}
```

The loop (our procedure, assembled from the guide's skills): log every dismissal with its `detected_pattern`, rank patterns by dismissal rate, and act on the worst. Tighten the criteria for that category, add few-shot examples that distinguish acceptable code patterns from genuine issues (4.2-S3), or temporarily disable the category to restore developer trust while you improve its prompt (4.1-S2), because high false positive categories undermine confidence in accurate ones (4.1-K3). Then measure again. Criteria writing is covered in [Explicit criteria and precision](#explicit-criteria-and-precision).

The field name `detected_pattern` is the guide's; Anthropic's own review tools implement the same idea under other names:

- Claude Code's ReportFindings tool can attach an optional `category` slug to each finding (v2.1.199 and later).
- The `claude-code-security-review` action gives every finding a `category` (for example `sql_injection`) and keeps an `exclusion_breakdown` count per exclusion reason.
- In Code Review, Anthropic collects thumbs-up and thumbs-down reaction counts after a PR merges and uses them to tune the reviewer (reactions change nothing on the PR); dismissing a finding means resolving its thread, and `REVIEW.md` skip rules can silence whole finding categories.

### Decide

- If validation fails on format or structure, retry with the specific errors plus the document and the failed output; not a bare "try again".
- If the needed value is not in the document, stop retrying: return null and route to review; not more attempts, because the information is absent.
- If the document contradicts itself, record the conflict (`conflict_detected`) and send it to a person; not a retry.
- If a finding category is dismissed far more often than others, analyze it by `detected_pattern` and tighten or pause that category; not a blanket "be conservative", which the [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says fails to improve precision compared with specific categorical criteria.

### Traps

- Assuming a strict schema means no validation is needed. It removes syntax errors only.
- Retrying until the model "finds" a value that the document never contained: the result is a fabrication.
- Parsing before checking `stop_reason`, so a refusal or a truncated reply reaches the parser.
- Checking only the Agent SDK `subtype` and missing a `success` with no `structured_output`.
- A retry message with no specifics, or one that leaves out the original document.
- Unbounded retry loops with no route to human review.
- Quoting one retry count as "the" official number; each layer sets its own limit, and the Agent SDK docs do not state the SDK's value.

## Batch processing design

*Tested in: CCAR-F 4.5 (4.5-K1 to 4.5-S4), Exercise 3 step 4, Q11, APPX-TECH-6, APPX-INSCOPE-15 · CCDV-F D2.3 Claude API Mechanics (batch API use, realtime versus batch), Sample 1 · CCAR-P 4.5, 1.6 (performance SLAs) · CCAO-F: no objective names batch processing*

The Message Batches API itself (limits, endpoints, result types, lifecycle, caching inside batches) is covered in [Message Batches](claude-api.md#message-batches). This section is the design around it: deciding what belongs in a batch, getting the prompt right before a batch multiplies any mistake, correlating and validating results, and fitting resubmissions inside an SLA. The facts to carry into the exam are the guide's: "50% cost savings, up to 24-hour processing window, no guaranteed latency SLA" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Most batches finish in less than 1 hour, but plan on the 24-hour bound: a batch that has not finished within 24 hours expires, and its unprocessed requests come back as `expired`.

### Three questions before you batch

1. **Is a person or a pipeline waiting on the answer?** If yes, stay synchronous. The guide's split: "synchronous API for blocking pre-merge checks, batch API for overnight/weekly analysis". Batch fits "non-blocking, latency-tolerant workloads (overnight reports, weekly audits, nightly test generation)".
2. **Can every result arrive up to 24 hours later and still meet the SLA?** If not, batching cannot be the only path; see the SLA arithmetic below.
3. **Does each item finish in one request?** The batch API "does not support multi-turn tool calling within a single request (cannot execute tools mid-request and return results)". Design each item so one request returns the data (JSON outputs, or a tool input you read directly), not as an agent loop that needs your own tools to run between turns. The docs add a nuance (as of September 2026): all server tools work in batch requests ([Batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing): "The batch worker runs the same server-side agentic loop as the synchronous Messages API"), and multi-turn conversation history can be batched. The guide's statement still holds for your own client tools, and it is the answer on the exam.

CCAR-F lists "Rate limiting, quotas, or API pricing calculations" as out of scope, so expect its batch items to turn on the 50% and 24-hour facts and the fit decision rather than price arithmetic (our reading).

### The pipeline

```text
1. Refine the prompt on a sample set (synchronous Messages API)
        |
2. Build one request per document: custom_id = document ID,
   params = the tuned prompt + the output schema
        |
3. Submit (split a very large set into several batches)
        |
4. Poll until processing_status is "ended"
        |
5. Stream the .jsonl results; match each one to its document by custom_id
        |
6. Validate every succeeded result (semantic checks; the schema is already enforced)
        |
7. Route each document:
     valid ..................................... store it
     errored, invalid_request_error ............ fix the request body, resubmit
     errored (other), expired or canceled ...... resubmit unchanged
     exceeded the context window ............... chunk the document, resubmit the chunks
     stop_reason "max_tokens" .................. resubmit with a higher max_tokens
     semantic failure .......................... resubmit with error feedback
     information absent, or conflicting source . null or conflict flag, human review
```

### Step 1: refine on a sample before you batch

CCAR-F 4.5-S4 is "Using prompt refinement on a sample set before batch-processing large volumes to maximize first-pass success rates and reduce iterative resubmission costs". Three batch properties make this step pay: the `params` of each request are validated asynchronously and errors come back only after the whole batch has ended; a submitted batch cannot be modified, only canceled and resubmitted; and results can take up to 24 hours. Our reasoning from those facts: a prompt flaw found on the sample costs a few synchronous calls; the same flaw found in the batch costs a full pass over every document plus the wait. The docs also recommend a dry run of one request shape against the Messages API before submitting.

### Step 2: requests with a custom_id and a schema

Every request carries a `custom_id` that is unique within the batch, 1 to 64 characters matching `^[a-zA-Z0-9_-]{1,64}$`. Results can come back in any order, so the `custom_id` is the only reliable join key: use the document's own ID, not a loop counter. Structured outputs work with batch processing at the batch discount, so each request can carry the same `output_config.format` schema (here, the `input_schema` object of the extraction tool from [Structured output with tools and JSON schemas](#structured-output-with-tools-and-json-schemas)). Batched requests need `max_tokens` of at least `1` and reject `stream: true`.

Adapted from the batch docs' create example, with a schema added to each request (the `max_tokens` value is our choice, sized to leave room for thinking):

=== "Python"

    ```python
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    client = anthropic.Anthropic()

    message_batch = client.messages.batches.create(
        requests=[
            Request(
                custom_id=f"inv-{doc_id}",
                params=MessageCreateParamsNonStreaming(
                    model="claude-opus-5-5",
                    max_tokens=16000,
                    system=EXTRACTION_INSTRUCTIONS,  # tuned on the sample set
                    messages=[{"role": "user", "content": f"<document>\n{text}\n</document>"}],
                    output_config={"format": {"type": "json_schema", "schema": INVOICE_SCHEMA}},
                ),
            )
            for doc_id, text in documents.items()
        ]
    )
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";

    const client = new Anthropic();

    const messageBatch = await client.messages.batches.create({
      requests: Object.entries(documents).map(([docId, text]) => ({
        custom_id: `inv-${docId}`,
        params: {
          model: "claude-opus-5-5",
          max_tokens: 16000,
          system: extractionInstructions, // tuned on the sample set
          messages: [{ role: "user" as const, content: `<document>\n${text}\n</document>` }],
          output_config: { format: { type: "json_schema" as const, schema: invoiceSchema } }
        }
      }))
    });
    ```

Keep the instructions identical across requests. Batch and prompt-caching discounts can be combined, and the docs' first tip for more cache hits is to include identical `cache_control` blocks in every request in the batch; hits are best-effort, with typical hit rates of 30% to 98%. The recipe is in [Message Batches](claude-api.md#message-batches).

### Steps 5 to 7: read, validate, route

Adapted from the batch docs' results example (the routing lists are ours); `semantic_errors` is the validator from [Validation, retry and feedback loops](#validation-retry-and-feedback-loops):

```python
import json

stored, fix_body, resend, raise_limit, retry_with_feedback, review = {}, [], [], [], {}, []

for result in client.messages.batches.results(message_batch.id):
    outcome = result.result
    match outcome.type:
        case "succeeded":
            message = outcome.message
            if message.stop_reason == "max_tokens":
                raise_limit.append(result.custom_id)
                continue
            if message.stop_reason == "refusal":
                review.append(result.custom_id)
                continue
            text = next(block.text for block in message.content if block.type == "text")
            data = json.loads(text)
            errors = semantic_errors(data)
            if data["conflict_detected"]:
                review.append(result.custom_id)
            elif errors:
                retry_with_feedback[result.custom_id] = (data, errors)
            else:
                stored[result.custom_id] = data
        case "errored":
            if outcome.error.error.type == "invalid_request_error":
                fix_body.append(result.custom_id)  # fix the request before re-sending
            else:
                resend.append(result.custom_id)  # can be retried as it is
        case "expired" | "canceled":
            resend.append(result.custom_id)
```

Then resubmit only what failed. The guide's skill is "resubmitting only failed documents (identified by custom_id) with appropriate modifications (e.g., chunking documents that exceeded context limits)". The docs back the economics: one failed request does not affect the others in the batch, and errored, canceled and expired requests are not billed. Retries with feedback carry the document, the failed extraction and the specific errors, exactly as in a synchronous loop. Preparation Exercise 3 rehearses the whole cycle with 100 documents, including calculating total processing time against the SLA.

### Fitting resubmissions inside an SLA

The guide's example skill is "4-hour windows to guarantee 30-hour SLA with 24-hour batch processing". Our arithmetic: a document that arrives just after a submission waits up to 4 hours, then up to 24 hours in the batch, 28 hours in the worst case. [Message Batches](claude-api.md#message-batches) works that example through. The design consequence for retries follows from the same numbers (also our arithmetic):

| Plan | Worst case for a document that needs one retry |
|---|---|
| Retry in the next batch window | 4 h wait + 24 h first pass + 4 h wait + 24 h retry pass = 56 h |
| Retry synchronously as soon as the first pass ends | 4 h + 24 h + minutes = about 28 h |

With a 30-hour SLA, a batched second pass cannot be guaranteed, so failures from the first pass have to be retried synchronously (at full price, for the few that failed) or escalated. If retries must also be batched, the SLA has to cover two full cycles of interval plus processing. The general rule (ours): worst case = (submission interval + 24 hours) for every batched pass + your own handling time, and it must fit inside the SLA.

### Decide

- If nobody waits on the result and cost matters, batch it; if a merge, a user or a downstream step blocks on it, keep it synchronous even though batches usually finish in under an hour.
- If a batched job fails for some documents, resubmit only those, found by `custom_id`, with the fix (chunking, a corrected body, a higher `max_tokens`, error feedback); not the whole batch.
- If the prompt is new, refine it on a sample synchronously first; not straight into a 10,000-document batch.
- If each item would need your own tools mid-request, it is not a batch item as designed; restructure it into a single request or keep it synchronous.
- If a retry must fit a tight SLA, send the few retries synchronously.

### Traps

- Q11 option D, "Switch both to batch processing with a timeout fallback to real-time if batches take too long": the rationale calls it "unnecessary complexity when the simpler solution is matching each API to its appropriate use case".
- Worrying that batch results arrive out of order: true, and irrelevant, because `custom_id` correlates them (Q11 option C keeps real-time calls "to avoid batch result ordering issues", and the rationale calls that a misconception).
- Planning on the typical sub-hour completion instead of the 24-hour bound; the Q11 rationale rejects option B because relying on "often faster" completion is not acceptable for blocking workflows.
- Using a loop index or a timestamp as `custom_id`, then being unable to find which document failed.
- Resubmitting an `invalid_request_error` unchanged; the docs say the request body must be fixed before re-sending.

## Multi-instance and multi-pass review

*Tested in: CCAR-F 4.6 (4.6-K1 to 4.6-S3), 3.6-K4, 1.6-K2, 1.6-S2, Q12, How to Prepare (multi-pass review architectures) · CCDV-F D2.4 Software Engineering Foundations (code review) · CCAO-F D2.4 (when human review or additional verification is required) is adjacent · CCAR-P: no objective names multi-instance or multi-pass review*

Task statement 4.6 combines two separate design choices. The first is **who reviews**: an independent instance that did not produce the work, rather than the generator checking itself. The second is **how the review is cut up**: focused per-file passes plus a separate integration pass, rather than one pass over everything. Code review is where both show up in the guide: Scenario 5 (Claude Code for Continuous Integration) runs automated code reviews, and sample question 12, one of that scenario's sample items, tests the second choice.

### Why self-review underperforms

The guide's knowledge bullet: "Self-review limitations: a model retains reasoning context from generation, making it less likely to question its own decisions in the same session". Its conclusion: "Independent review instances (without prior reasoning context) are more effective at catching subtle issues than self-review instructions or extended thinking" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The CI objective repeats it: the same session that generated code is less effective at reviewing its own changes than an independent review instance.

Anthropic's documentation and tools apply the same principle:

| Source | What it says or does |
|---|---|
| [Claude Code best practices](https://code.claude.com/docs/en/best-practices) | "A fresh context improves code review since Claude won't be biased toward code it just wrote." A reviewer in a fresh subagent sees only the diff and the criteria, not the reasoning behind the change |
| [Prompting Claude Fable 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5) | "Separate, fresh-context verifier subagents tend to outperform self-critique." |
| [Security guidance plugin](https://code.claude.com/docs/en/security-guidance) | "The plugin does not ask the same Claude instance that wrote the code to grade itself." Its reviewer starts from the diff with no investment in the original approach |
| Claude Security plugin | Findings appear in the report only after independent verifier agents analyze them, and each patch is reviewed by an agent independent of the one that wrote it |
| Managed Agents outcomes | The grader uses a separate context window so the main agent's implementation choices do not influence it |
| `/goal` in Claude Code | Completion is decided by a fresh model rather than the one doing the work |

### Four review architectures

| Architecture | How it works | What it fixes | Source |
|---|---|---|---|
| Independent reviewer | A second instance gets the output and the criteria, not the generator's reasoning | Self-review bias | CCAR-F 4.6-S1 |
| Multi-pass | Per-file local passes, then a separate integration pass for cross-file data flow | Attention dilution and contradictory findings | CCAR-F 4.6-K3, 4.6-S2, 1.6-S2 |
| Specialists plus verification | Several agents each look for a different class of issue; a verification step checks candidates against actual code behavior; results are deduplicated and ranked | False positives | Claude Code Code Review |
| Find, then filter | The finding pass reports every issue with confidence and severity; a separate pass filters | Lost recall from "be conservative" instructions | Prompting Claude Sonnet 5 |

A multi-pass review of a large pull request, with an independent verifier:

```text
Pull request, 14 files
   |
   +--> pass 1: file A, local issues ---+
   +--> pass 2: file B, local issues ---+
   +--> ...                             +--> integration pass: cross-file data flow
   +--> pass 14: file N, local issues --+              |
                                                       v
                                  verification pass (fresh context: diff + criteria)
                                                       |
                                                       v
                                      deduplicate, rank by severity, post
```

The guide lists multi-pass review under prompt chaining too: "Prompt chaining patterns that break reviews into sequential steps (e.g., analyze each file individually, then run a cross-file integration pass)". The chaining mechanics are in [Chain of thought and prompt chaining](#chain-of-thought-and-prompt-chaining).

### What sample question 12 teaches

A pull request touches 14 files and a single-pass review gives uneven depth, misses obvious bugs, and flags a pattern in one file while approving identical code in another. The answer is to split the review into per-file passes plus a separate integration pass. The rationale names the root cause, "attention dilution when processing many files at once", and rejects the distractors: "Option C misunderstands that larger context windows don't solve attention quality issues. Option D would actually suppress detection of real bugs by requiring consensus on issues that may only be caught intermittently." Requiring developers to split PRs (Option B) "shifts burden to developers without improving the system". The full item is on the [CCAR-F page](../claude-certified-architect-foundations.md#official-sample-questions).

Option D is itself a form of the "voting" pattern in Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) ("Running the same task multiple times to get diverse outputs"), set to a two-of-three threshold. The post's code-review example has "several different prompts review and flag the code if they find a problem", and the post notes that vote thresholds trade false positives against false negatives. Our reading, consistent with the Q12 rationale: flagging when any reviewer finds a problem favors recall, while requiring agreement across runs suppresses issues caught only intermittently, so what Q12 rejects is the consensus threshold, not parallel review as such.

### Prompting the passes

**The finding pass: coverage first.** The [Sonnet 5 prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5)'s finding-stage instruction:

```text
Report every issue you find, including ones you are uncertain about or consider low-severity. Do not filter for importance or confidence at this stage - a separate verification step will do that. Your goal here is coverage: it is better to surface a finding that later gets filtered out than to silently drop a real bug. For each finding, include your confidence level and an estimated severity so a downstream filter can rank them.
```

The reason is the recall that a "be conservative" style of instruction costs on current models; see [Why confidence words do not buy precision](#why-confidence-words-do-not-buy-precision).

**The verification pass: bounded scope.** [Claude Code best practices](https://code.claude.com/docs/en/best-practices) warn that "A reviewer prompted to find gaps will usually report some, even when the work is sound, because that is what it was asked to do." Tell the verifier to flag only correctness problems or gaps against stated requirements, and give it the criteria from [Explicit criteria and precision](#explicit-criteria-and-precision).

**Confidence for routing, calibrated.** CCAR-F 4.6-S3 runs verification passes "where the model self-reports confidence alongside each finding to enable calibrated review routing". How that squares with the same guide's rejection of raw self-reported confidence (5.2-K3, Q3) is in [Verify; do not trust confidence](#verify-do-not-trust-confidence).

### In Claude Code and CI

- `/code-review` reviews a diff in a subagent with its own context window (in the background by default; in the foreground under `-p` or the Agent SDK); at `low` and `medium` effort it reports only its most confident findings, and `high` through `max` broaden coverage.
- [Code Review](https://code.claude.com/docs/en/code-review) (a research preview for Team and Enterprise subscriptions as of September 2026) is the specialists-plus-verification row in the table above: "Each agent looks for a different class of issue, then a verification step checks candidates against actual code behavior to filter out false positives."
- Ultrareview (`/code-review ultra`, research preview) independently reproduces and verifies every reported finding.
- In a pipeline, run the reviewer as its own `claude -p` invocation, separate from the session that generated the code (our application of CCAR-F 3.6-K1 and 3.6-K4).

Setup and trust-building for automated review are in [Automated code review that engineers trust](claude-code-workflows.md#automated-code-review-that-engineers-trust).

!!! warning "Exam guide vs current docs"

    **"Extended thinking."** The guide contrasts independent review with "self-review instructions or extended thinking". As of September 2026 the term has moved on: Claude 4.7 and later models reject extended-thinking requests (`thinking.type: "enabled"` with `budget_tokens`) with a 400 error, and on Claude Opus 5.5 thinking is always on and adaptive thinking is the only mode. Our reading: treat the guide's phrase as more reasoning inside the same context; the point stands that more reasoning in the generator's own context is not an independent review.

    **Verification on Claude Opus 5 and self-check prompts.** The Opus 5 prompting guide says the model verifies its own work unprompted, that instructions such as "use a subagent to verify" cause over-verification and should be removed, and that "The same applies to legacy harness scaffolding that adds separate verification steps." ([Prompting Claude Opus 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5)) The same page says Claude Opus 5 coordinates teams of subagents well, "with effective writer-verifier patterns". The general [prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) page, for other models, recommends asking Claude to self-check and says "This catches errors reliably, especially for coding and math." Our reading: on Claude Opus 5 the page advises removing verification layered on top of the model's own checking, including harness steps and verifier subagents (its sample delegation instruction says "do not use subagents to verify or double-check your own work"), while the guide's 4.6 treats an independent reviewer as the architecture to choose. For Opus 5 deployments the two pull in different directions; neither universally supersedes the other. On the exam, answer in the guide's terms: an independent instance without the generator's reasoning context beats a self-review instruction.

### Decide

- If generated code or content needs review, use a second, independent instance with only the output and the criteria; not a self-review instruction or more thinking in the same session.
- If a review spans many files and results are uneven or contradictory, split into per-file passes plus an integration pass; not a bigger context window, and not consensus voting across runs.
- If precision matters, give the reviewer specific categorical criteria for what to report and what to skip (4.1-S1), and check candidates in a separate verification step; not "be conservative" or a confidence bar in the finding prompt, which the [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says fails to improve precision (4.1-K2) and the Sonnet 5 guide says can cut recall.
- If findings will be routed by confidence, calibrate the thresholds on labeled data first.

### Traps

- A self-review instruction in the generator's own prompt offered as the reliability fix.
- A larger or higher-tier model for a many-file review (Q12 Option C).
- Two-of-three consensus across identical runs (Q12 Option D): it hides bugs caught only some of the time.
- A reviewer prompt that asks for problems with no criteria, which will usually report some even when the work is sound.
- Treating raw self-reported confidence as calibrated.
- Requiring developers to split large PRs before review (Q12 Option B): it "shifts burden to developers without improving the system".

## Reducing hallucinations

*Tested in: CCAO-F D2.1, D2.2, D2.3, D2.4, Sample 1 · CCDV-F D6.3 Output Handling (skepticism toward confident output) · CCAR-F 4.3-S4, 4.2-K4, 5.2-K3, 5.5-S3, Exercise 3 step 1, APPX-INSCOPE-13, Q3 · CCAR-P 4.4, 5.2, Sample 3*

Anthropic's guardrails page defines the problem plainly: "Even the most advanced language models, like Claude, can sometimes generate text that is factually incorrect or inconsistent with the given context." ([Reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations)) Read against the objectives and sample questions above, the exams test it from three angles (our grouping): invented facts in prose (a citation number that does not exist), invented values in structured extraction (a field filled because the schema demanded it), and the judgment to verify rather than trust a confident answer. The API features used here are taught elsewhere: citations in [Grounding answers in the source](#grounding-answers-in-the-source) and [Files, citations and search results](claude-api.md#files-citations-and-search-results), provenance across agents in [Provenance and uncertainty in synthesis](context-engineering.md#provenance-and-uncertainty-in-synthesis).

### Where to expect them

Claude Academy's tutorial [Why do AI models hallucinate?](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate) lists the high-risk situations: "if you're asking for specific facts, statistics, or citations, or if the topic is obscure, niche, or very recent, if you're asking about real but not widely known people or places, or when you need exact details like dates, names, or numbers." Claude can also "hallucinate its capabilities", for example claiming to have sent an email or produced an external document. The [help center](https://support.claude.com/en/articles/8241188-claude-is-producing-links-that-don-t-work-and-falsely-claiming-that-it-has-sent-emails-or-produced-external-documents-what-s-going-on) is explicit: "Even if it claims otherwise, Claude does not have access to other tools or software that are not explicitly integrated, including email, word processors, or file transfers." And in extraction, a required field with no source value invites a fabricated one (CCAR-F 4.3-S4).

### Techniques, and when each fits

The first seven rows are the strategies on Anthropic's [Reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations) page. The rest come from the [citations](https://platform.claude.com/docs/en/build-with-claude/citations) docs, the claude.com [prompt engineering best practices](https://claude.com/blog/best-practices-for-prompt-engineering) post (the null instruction), the [Increase output consistency](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/increase-consistency) page (retrieval grounding), the [Claude Fable 5.1 guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-1) and the CCAR-F guide. The "Use it when" column is our guidance except where it cites a source.

| Technique | How | Use it when |
|---|---|---|
| Permission to say "I don't know" | State that uncertainty is acceptable and give the exact words to use, for example "I don't have enough information to confidently assess this." | Any task where the input may lack the answer; the docs say this "can drastically reduce false information" |
| Quotes first | Have Claude extract word-for-word quotes, then work only from those quotes; "No relevant quotes found." if none exist | Long documents (the docs say >20k tokens) |
| Verify with citations, retract | Have Claude cite quotes and sources for each claim; after drafting, find a supporting quote for each claim and retract any claim without one | Drafting from source material |
| Chain-of-thought verification | Ask for step-by-step reasoning before the final answer; the docs say this "can reveal faulty logic or assumptions" | Reasoning-heavy answers (see the caution below on current models) |
| Best-of-N verification | Run the same prompt several times and compare; "Inconsistencies across outputs could indicate hallucinations." | Facts you cannot check directly |
| Iterative refinement | Feed outputs back as inputs and ask Claude to verify or expand on earlier statements | Catching and correcting inconsistencies (the docs' stated purpose) |
| External knowledge restriction | Instruct Claude to use only the provided documents, not its general knowledge | Document Q&A, policy answers |
| Citations feature (API) | Turn on citations for the documents in the request; the mechanics are in [Grounding answers in the source](#grounding-answers-in-the-source) | Answers that must be traceable to source text |
| Nullable schema fields | Fields the source may lack are nullable, with "use null rather than guessing" in the prompt | Structured extraction (CCAR-F 4.3-S4) |
| Few-shot extraction examples | Examples of correct extraction from documents with varied formats | Extraction that returns empty or invented values (CCAR-F 4.2-K4, 4.2-S5) |
| Grounding with retrieval or search | Retrieval grounds answers "in a fixed information set"; for Claude Fable 5.1 at `low` effort, which is less likely than Claude Fable 5 to call a search or retrieval tool, the [Fable 5.1 guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-1) suggests raising effort for the affected turns, or telling Claude that recognizing a name is not the same as knowing its current state, so it searches before answering | Knowledge bases and chatbots; facts that change |

Two prompts from the same page, verbatim. Quotes first, then analysis only from the quotes:

```text
As our Data Protection Officer, review this updated privacy policy for GDPR and CCPA compliance.
<policy>
{{POLICY}}
</policy>

1. Extract exact quotes from the policy that are most relevant to GDPR and CCPA compliance. If you can't find relevant quotes, state "No relevant quotes found."

2. Use the quotes to analyze the compliance of these policy sections, referencing the quotes by number. Only base your analysis on the extracted quotes.
```

Restrict to the documents, then verify and retract after drafting:

```text
Draft a press release for our new cybersecurity product, AcmeSecurity Pro, using only information from these product briefs and market reports.
<documents>
{{DOCUMENTS}}
</documents>

After drafting, review each claim in your press release. For each claim, find a direct quote from the documents that supports it. If you can't find a supporting quote for a claim, remove that claim from the press release and mark where it was removed with empty [] brackets.
```

The same page states the limit, which matches the verify-first answer to CCAO-F Sample 1: "while these techniques significantly reduce hallucinations, they don't eliminate them entirely. Always validate critical information, especially for high-stakes decisions." ([Reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations))

For agents, two prompting-guide instructions carry the same idea into tool use. For coding, the `<investigate_before_answering>` block in [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) opens "Never speculate about code you have not opened." and tells Claude to read a referenced file before answering. For long autonomous runs, the [Claude Fable 5 guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5) recommends: "Before reporting progress, audit each claim against a tool result from this session. Only report work you can point to evidence for; if something is not yet verified, say so explicitly." It reports that "In Anthropic's testing, this nearly eliminated fabricated status reports even on tasks designed to elicit them".

A caution on chain-of-thought verification: on Claude Fable 5 and Claude Opus 5.5, a prompt that makes the model write out its internal reasoning as response text can be refused, so read the thinking blocks instead; see [Manual chain of thought: the fallback](#manual-chain-of-thought-the-fallback).

### Verify; do not trust confidence

CCAO-F Sample 1 is the reference case. Claude summarizes a new regulation and confidently cites a specific subsection; the correct action is to verify the cited subsection against the official regulation text before sharing. The rationale: "Language models can fabricate specific-looking details such as citation numbers, a hallucination." It adds that "Self-reported confidence (A, C) is not a reliable accuracy signal, and reformatting (D) does not address correctness." ([CCAO-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf); full item on the [CCAO-F page](../claude-certified-associate.md#official-sample-questions)) Question 3 in the [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) reaches the same verdict for an agent's escalation decisions: "LLM self-reported confidence is poorly calibrated" (full item on the [CCAR-F page](../claude-certified-architect-foundations.md#official-sample-questions)).

!!! warning "Self-reported confidence versus calibrated confidence (CCAR-F)"

    The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) takes two positions that only look contradictory. Knowledge item 5.2-K3 says "self-reported confidence scores are unreliable proxies for actual case complexity", and Q3 rejects a confidence threshold for escalation. Yet skill 5.5-S3 is "Having models output field-level confidence scores, then calibrating review thresholds using labeled validation sets", and skill 4.6-S3 has verification passes report a confidence level with each finding for review routing. Our reading: a raw self-rating is never the gate; a score per field or per finding, calibrated against labeled data, can decide where human reviewers look. The routing design is in [Human review and confidence calibration](evaluation-and-reliability.md#human-review-and-confidence-calibration).

Checks a user can run without special tools:

- Ask where each claim came from. The Academy [discernment toolkit](https://academy.claude.com/tutorials/discernment-toolkit) gives this prompt: "For each factual claim in your answer, tell me where it came from. Quote the exact passage from the source and include the link or page number. If a claim comes from your general knowledge, label it "unsourced" so I know what to check first."
- Ask the same question several ways and compare; for documents, ask for direct quotes rather than paraphrase ([writing an AI diligence statement](https://academy.claude.com/tutorials/writing-an-ai-diligence-statement)).
- Treat a specific date, name or quote you cannot trace to an input as "a flag, not a feature" ([Claude Cowork task loop](https://academy.claude.com/courses/introduction-to-claude-cowork/the-task-loop)).
- With web search results, open the cited sources: the [help center](https://support.claude.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on) notes that the originals may carry context the synthesis dropped.
- Be most careful with polished output. Anthropic's [AI Fluency Index](https://academy.claude.com/tutorials/the-ai-fluency-index) report found that in conversations where artifacts are created, users were less likely to identify missing context (-5.2pp), check facts (-3.7pp) or question the model's reasoning (-3.1pp). Its advice is to question polished outputs.

The help center's rule for everything above: "Users should not rely on Claude as a singular source of truth and should carefully scrutinize any high-stakes advice given by Claude." ([Claude is providing incorrect or misleading responses](https://support.claude.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on)) When human review or extra verification is required (CCAO-F D2.4), and how to check for inconsistencies and bias (the rest of D2.2), along with sycophancy, are covered in [Verifying Claude's output](claude-for-work.md#verifying-claudes-output).

### Diagnose where the hallucination comes from

CCAR-P 4.4 asks you to "Diagnose system issues (prompt failure, hallucinations, model mismatch)" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). Start from the symptom. The table is our mapping from symptom to origin, built from the sources linked in this section; the second row is the official answer to CCAR-P Sample 3.

| Symptom | Likely origin | First fix |
|---|---|---|
| Values for fields the document does not contain | The schema makes those fields required and non-null | Nullable fields plus "use null rather than guessing" |
| Confident but wrong answers right after a document refresh, with model and latency unchanged | Retrieval feeding the model poor context (for example a broken re-index or mismatched embeddings) | Investigate the retrieval and indexing step first |
| A precise-looking citation or section number that does not exist | Model fabrication | Verify against the authoritative source; use the citations feature for document answers |
| Claims that an email was sent or an external document was produced | Hallucinated capability: Claude has no access to tools that are not explicitly integrated | Integrate the tool, or tell users what Claude cannot do |
| Progress reports that claim unverified work | Claims not tied to evidence | Instruct Claude to audit each claim against a tool result from the session |
| Different answers to the same question across runs | Possible hallucination | Best-of-N comparison, then ground the answer in sources |

The rationale for Sample 3 also notes that the other options (silently changed weights, a low temperature, a shrunken context window) "would not be triggered specifically by a document refresh" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf); full item on the [CCAR-P page](../claude-certified-architect-professional.md#official-sample-questions)). The debugging method is in [Debugging: model or integration](evaluation-and-reliability.md#debugging-model-or-integration).

### Decide

- If a claim is specific and will be relied on (a regulation, a statistic, a citation), verify it against an authoritative source; not the model's stated confidence, and not a rewording.
- If the task is document-based, restrict Claude to the documents, have it quote first when the documents are long (the docs say >20k tokens), and use citations when answers must be traceable.
- If a structured field may be missing, make it nullable and say "null rather than guessing"; not a required field.
- If a design routes work on the model's own confidence, accept it only as a score (per field or per finding) calibrated on labeled validation data; not a raw self-rating used as the gate.
- If wrong answers appear after a data change, suspect retrieval before the model.

### Traps

- A confident tone treated as evidence of accuracy, or a confidence rating requested from Claude and used as the gate (CCAO-F Sample 1, options A and C; CCAR-F Q3, option B).
- Reformatting or polishing a summary as if that improved accuracy (CCAO-F Sample 1, option D).
- Believing any single technique eliminates hallucination.
- Required-but-unknowable schema fields, then blaming the model for filling them.
- Combining the citations feature with `output_config.format` in one request: the API returns a 400 error.

## Prompt versioning and iteration

*Tested in: CCDV-F D2.6 Configuration Management (prompt versioning, model version pinning), D6.2 Prompt Engineering (iterative refinement, prompt adjustment), D5.3 Model Selection and Tradeoffs (breaking behavior changes across model releases) · CCAO-F D1.3, D5.4, D7.2 · CCAR-P 4.3, 2.5, 6.5 · CCAR-F 4.5-S4, 4.1-S2*

Prompts are production artifacts. Anthropic's enterprise guide says it directly: "Like code, prompts need version control, testing, and proper documentation." ([Building trusted AI in the enterprise](https://www-cdn.anthropic.com/e5c9de22bc8884089970bd262ca0c8b952cb9136.pdf)) The [CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) puts "model version pinning, prompt versioning, and plugin dependencies" in D2.6 next to CLAUDE.md files and settings.json, and a CCDV-F prep course ([Accelerators & IP Contribution](https://anthropic-partners.skilljar.com/path/claude-certified-developer-foundations/accelerators-ip-contribution)) asks candidates to "version what ships so a model or prompt change does not silently break production". The everyday loop (test, change one thing, test again) is in [Principles that decide most prompt questions](#principles-that-decide-most-prompt-questions); this section makes iteration measurable and changes safe to ship.

### Make each iteration measurable

- **Criteria first.** The overview's three prerequisites (success criteria, a way to test against them, a first draft) open [Principles that decide most prompt questions](#principles-that-decide-most-prompt-questions). Good criteria are specific and measurable ([Define success criteria and build evaluations](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests)): not "Safe outputs" but "Less than 0.1% of outputs out of 10,000 trials flagged for toxicity by the content filter."
- **A test set that looks like production.** The same page says to design evals that "mirror your real-world task distribution" and to include edge cases: irrelevant or nonexistent input data, overly long input, and ambiguous cases where even humans would disagree. It prefers volume with automated grading: "More questions with slightly lower signal automated grading is better than fewer questions with high-quality human hand-graded evals."
- **Same questions, several trials, paired comparison.** "Because model outputs vary between runs, we run multiple trials to produce more consistent results." ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)) Anthropic's [statistical approach to model evaluations](https://www.anthropic.com/research/statistical-approach-to-model-evals) recommends a paired-differences test when two models answer the same question list, which removes question-difficulty variance, and recommends reporting the standard error of the mean with each score; the same logic applies to two prompt versions (our application). With a small eval, "small differences will likely go undetected."
- **Keep a held-out set.** Anthropic's [writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents) post reports relying on held-out test sets so the team did not overfit to its "training" evaluations.
- **Tune review prompts on recall or F1, not precision alone.** Why a stricter review prompt can raise precision while measured recall falls, and the model guides' advice to validate recall or F1 gains, are in [Explicit criteria and precision](#explicit-criteria-and-precision).
- **Protect trust while you iterate.** The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) lists "Temporarily disabling high false-positive categories to restore developer trust while improving prompts for those categories" (4.1-S2); see [Explicit criteria and precision](#explicit-criteria-and-precision).
- **Refine before scaling.** Before a large batch, refine the prompt on a sample set (CCAR-F 4.5-S4); see [Batch processing design](#batch-processing-design).

Grading methods and test-set design are taught in [Success criteria and test sets](evaluation-and-reliability.md#success-criteria-and-test-sets) and [Grading methods](evaluation-and-reliability.md#grading-methods).

### What to version

The "How to version it" column is our recommendation; the "Why" column carries the sourced facts.

| Artifact | How to version it | Why |
|---|---|---|
| System prompt, templates, few-shot examples | In a central repository with change tracking, a testing framework, and documentation of each prompt's purpose and expected behavior | Anthropic's enterprise guide asks for exactly these; the Academy SDLC playbook logs the spec, "the prompt that produced it", and the skill versions in force in version control |
| JSON schemas and tool definitions | In the repository beside the prompt that uses them | A schema change alters structured output; changing schema structure or the set of tools also invalidates the compiled-grammar cache, and changing `output_config.format` invalidates the prompt cache |
| Model ID | Pin an exact ID in code and config; for models before the 4.6 generation, use the dated ID, not the API alias | Every Claude model ID is a pinned snapshot, including dateless IDs from the 4.6 generation on; the guarantee does not cover the API's convenience aliases for earlier models (an alias such as `claude-sonnet-4-5` "resolves to the most recent dated snapshot for that minor version"); Claude Code aliases such as `opus` "update over time" |
| Effort | Set `effort` explicitly | Level names do not correspond to the same amount of thinking across models; the Claude Opus 5.5 guide says to set it explicitly and test several levels against your own evals |
| CLAUDE.md, skills, hooks | Git, with changes gated on eval results | Academy SDLC playbook: CLAUDE.md is checked into Git and reviewed like code, and configuration changes are gated on eval results |
| Skills called through the API | Pin a specific version in production (the version ID is a string) | Omitting `version` or using `"latest"` means a version uploaded by anyone in the workspace immediately changes what production agents run |
| Managed Agents agents | Versioned configurations; pin sessions to a version for staged rollouts | `version` starts at 1 and increments each time an update changes the agent; a bare agent ID gives the latest version |
| Plugin evals (`claude plugin eval`) | Pin both `--model` and `--judge-model` in CI | The CI guidance says to "pin both models so scores are comparable over time"; pinning `--model` means "a model rollout isn't mistaken for a plugin regression" |

Sources for the table: [Building trusted AI in the enterprise](https://www-cdn.anthropic.com/e5c9de22bc8884089970bd262ca0c8b952cb9136.pdf), the Academy [SDLC playbook](https://academy.claude.com/courses/ai-native-sdlc-playbook/requirements-and-design) with its [CLAUDE.md](https://academy.claude.com/courses/ai-native-sdlc-playbook/claude-md) and [Continuous evals in CI](https://academy.claude.com/courses/ai-native-sdlc-playbook/continuous-evals-in-ci) lessons, [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), [Model IDs and versioning](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions), [Claude Code model configuration](https://code.claude.com/docs/en/model-config), [Prompting Claude Opus 5.5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5-5), [Using Agent Skills with the API](https://platform.claude.com/docs/en/build-with-claude/skills-guide), [Managed Agents agent setup](https://platform.claude.com/docs/en/managed-agents/agent-setup), [Managed Agents sessions](https://platform.claude.com/docs/en/managed-agents/sessions) and [plugin evals](https://code.claude.com/docs/en/plugin-evals).

Our rule, extending the SDLC playbook's practice of logging the prompt and skill versions behind each spec: record the prompt version, schema version, model ID and effort together with each logged result. Without that record, a regression cannot be traced to the change that caused it.

### Gate changes, then roll out gradually

Our summary of the gate, assembled from the sources in the list below:

```text
one change (prompt, example, schema, model or effort)
   |
   v
offline evals: regression suite + capability set, same questions, several trials
   |
   v
compare with the pinned baseline (paired differences, standard error)
   |
   v
merge as a new version: prompt, schema, model ID and effort recorded together
   |
   v
staged rollout: pinned versions, then A/B test or canary on real traffic
   |
   v
production monitoring: new failure cases go back into the test set
```

- **Regression suite.** Regression evals ask whether the system still handles everything it used to and "should have a nearly 100% pass rate"; capability evals that reach high pass rates can graduate into a regression suite that runs continuously. Automated evals run on each agent change and each model upgrade ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).
- **The gate.** The Academy SDLC playbook ([Continuous evals in CI](https://academy.claude.com/courses/ai-native-sdlc-playbook/continuous-evals-in-ci)): "Gate configuration changes on the results. A skill change that drops the pass rate gets reviewed before it merges." Anthropic's [enterprise Skills guidance](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise): "Require the full evaluation suite to pass before promoting new versions." The [enterprise guide](https://www-cdn.anthropic.com/e5c9de22bc8884089970bd262ca0c8b952cb9136.pdf) also lists making "a decision based on a single evaluation test" among the things to avoid.
- **Online validation.** Anthropic's [success-criteria guidance](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests) defines A/B testing as comparing performance against a baseline model or a previous version; the [evals post](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) adds that it runs on real user traffic and measures real outcomes, but is slow ("days or weeks to reach significance") and needs enough traffic, so it "validates significant changes once you have sufficient traffic". The [CCAR-P prep course](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/enterprise-integration-production) expects you to set the hypothesis, pick metrics, estimate the required sample size, and read the result "without overclaiming".
- **Staged exposure.** The [enterprise guide](https://www-cdn.anthropic.com/e5c9de22bc8884089970bd262ca0c8b952cb9136.pdf)'s deployment advice lists "Progressively roll out your application" under Do and "Replace your previous system right away" under Don't. Pin Managed Agents sessions to an agent version to stage rollouts. Anthropic's [multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) uses "rainbow deployments", shifting traffic gradually from old to new versions while both run. In a [September 2025 postmortem](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues) of three issues, Anthropic describes its usual validation as benchmarks, safety evaluations and performance metrics, with spot checks and deployment to small "canary" groups first.
- **Close the loop.** "Update your offline evaluations based on production data", so failures seen in the field become test cases ([Building trusted AI in the enterprise](https://www-cdn.anthropic.com/e5c9de22bc8884089970bd262ca0c8b952cb9136.pdf)).

### A model change is a prompt change

A pinned model ID never changes under you: the underlying model "remains constant for the lifetime of that ID". An upgrade therefore happens only when you change the ID, and that is the moment to re-test the prompt. The prompting guide's rule: "Where a technique names a specific model, treat it as measured on that model and re-check it against your own evals before applying it to another." ([Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)) CCDV-F D5.3 names the same risk as "breaking behavior changes across model releases".

| Prompt habit | What changed (as of September 2026) |
|---|---|
| Prefilling the last assistant turn (to force JSON or to skip a preamble) | Returns a 400 error on Claude 4.6 and later models and on Claude Mythos Preview; earlier models still accept it. Use structured outputs or instructions |
| Forced `tool_choice` (`any` or `tool`) | Returns a 400 error on Claude Opus 5.5, Claude Fable 5.1 and Claude Mythos 5.1, regardless of thinking settings; `auto` and `none` still work |
| "CRITICAL: You MUST use this tool when..." | Claude Opus 4.5 and 4.6 are more responsive to the system prompt, so wording written to fix undertriggering may now overtrigger; write "Use this tool when..." |
| "include a final verification step" or "use a subagent to verify" | Causes over-verification on Claude Opus 5, which verifies its own work unprompted; remove it |
| Skills developed for earlier models | Often too prescriptive for Claude Fable 5 and can degrade output quality; review and consider removing older instructions |
| Omitted `effort` | Claude Opus 5.5 defaults to `medium`, Claude Opus 5 to `high` |
| Non-default `temperature`, `top_p` or `top_k` | Returns a 400 error on Claude Sonnet 5; guide tone and variety with system-prompt instructions |

Rows come from [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices), the [API errors](https://platform.claude.com/docs/en/api/errors) and [Define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools) pages and the model guides for [Claude Opus 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5), [Claude Fable 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5), [Claude Opus 5.5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5-5) and [Claude Sonnet 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5).

The Claude API skill bundled with Claude Code has two commands for this step. `/claude-api migrate` applies the model ID swap and breaking changes, then "produces a checklist of items to verify manually" ([Opus 5.5 migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide)). `/claude-api prompt-audit` "reads a project's prompts and request code and reports what was written for a different model." ([Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence)) In a customer support benchmark migrated from Claude Opus 4.8 to Claude Opus 5.5, where Anthropic planted one legacy anti-pattern in each of six otherwise clean prompts, running it once per prompt cut cost by a further 9% on top of the migration's own saving of around 18%, and raised accuracy by around 2 percentage points (averages across the six prompts) ([Reducing cost and improving performance with Claude Platform](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform)). The full upgrade procedure is in [Upgrading models safely](evaluation-and-reliability.md#upgrading-models-safely) and [Model versions, deprecation and migration](claude-api.md#model-versions-deprecation-and-migration).

### Iterating and maintaining instructions in the Claude apps

For CCAO-F candidates the same discipline applies at the scale of a conversation, a Project or a shared plugin:

- **Mostly right, or wrong at the core?** Claude Cowork's [task loop](https://academy.claude.com/courses/introduction-to-claude-cowork/the-task-loop): "If the draft is mostly right, tell Claude what to change rather than starting over." But "If the draft is wrong in a load-bearing way, the prompt was missing the load-bearing piece of context."
- **Branch instead of overwrite.** Editing a prior message creates a different version of the conversation, with its own artifacts, so you can explore a direction without losing earlier work ([What are artifacts](https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them)).
- **Checkpoints in multi-step tasks.** Ask Claude to stop and show an intermediate step before continuing, so small errors do not compound ([Steerability](https://academy.claude.com/courses/ai-capabilities-and-limitations/steerability)).
- **Keep instructions and knowledge current.** Revisit [organization instructions](https://support.claude.com/en/articles/14546867-set-organization-instructions) as needs change and remove outdated ones. Keep Project knowledge current too: "Outdated documents can lead to outdated responses." ([Introduction to projects](https://academy.claude.com/courses/claude-101/introduction-to-projects))
- **Own and gate shared plugins.** Give every shared plugin one named owner who reviews changes and runs the evals after edits, and do not push a change whose important cases fail. Quarterly is the Academy's "reasonable starting point" for a review rhythm ([Share what you build with your team](https://academy.claude.com/courses/introduction-to-claude-cowork/share-what-you-build-with-your-team)).

!!! note "Console prompt tools, as of September 2026"

    The old doc pages for the prompt generator, prompt templates and variables, and the prompt improver now redirect to Prompting best practices; the prompt improver and Evaluate descriptions below come from Anthropic's 2024 announcements. The [prompt improver](https://claude.com/blog/prompt-improver) (announced October 14, 2024) refines existing prompts and is positioned for "adapting prompts that were originally written for other AI models"; its listed methods include a chain-of-thought section, example standardization into XML, and "Prefill addition". Because prefill returns a 400 error on Claude 4.6 and later models, check any improved prompt for a prefilled assistant turn before using it. The [Evaluate](https://claude.com/blog/evaluate-prompts) feature (announced July 9, 2024) let you create new prompt versions, re-run a test suite and compare prompts side by side, with subject matter experts grading response quality on a 5-point scale. For a blank page, the Claude Cookbook's [metaprompt notebook](https://github.com/anthropics/claude-cookbooks/blob/main/misc/metaprompt.ipynb) produces a starting template for single-turn prompts, which its authors say is "not guaranteed to be optimal".

### Decide

- If a prompt, schema, model or effort setting changes, re-run the regression suite on the same questions and compare with the pinned baseline before merging; not a spot check of a few outputs.
- If a change must reach production safely, pin versions and stage the rollout (pinned sessions, A/B or canary); not an in-place edit of the only prompt.
- If you upgrade the model, audit the prompt for habits the new model rejects or over-follows; not the old prompt unchanged.
- If an eval difference is small and the test set is small, treat it as noise until a larger or paired comparison confirms it.

### Traps

- Using an alias or "latest" (an API alias for a pre-4.6 model such as `claude-sonnet-4-5`, a Claude Code model alias, an unpinned Skill version) in production and calling it versioned.
- Changing several things at once, so a better or worse score cannot be attributed.
- Judging a prompt change on one run of a non-deterministic system.
- Letting the eval harness's agent or judge model drift, then reading the drift as a prompt or plugin regression.
- Editing the production prompt in place, with no record of the version it replaced.

## Exam map

Which official objectives each section of this page serves. CCAR-F IDs follow the guide's task statements (4.3-K1 is task statement 4.3, first "Knowledge of" bullet; 4.3-S1 is its first "Skills in" bullet), plus its scenarios, sample questions (Q), preparation exercises (EX, with step numbers), "How to Prepare" list and appendix lists (APPX-TECH-n is the nth bullet of the appendix's "Technologies and Concepts" list, APPX-INSCOPE-n the nth bullet of its "In-Scope Topics" list; the K, S and APPX bullet numbers are ours). "Sample N" in another exam's column is that guide's official sample question N. The CCAO-F, CCDV-F and CCAR-P guides list their objectives (CCDV-F calls them skills) without numbers under each domain, so D2.3 means the third objective of Domain 2 in the guide's order, and CCAR-P 4.4 means the fourth objective of Domain 4; the numbers are ours. A dash means no objective in that guide names the topic. Exam items are written against these objectives in all four guides.

| Section | CCAO-F | CCDV-F | CCAR-F | CCAR-P |
|---|---|---|---|---|
| [Principles that decide most prompt questions](#principles-that-decide-most-prompt-questions) | D1.1, D1.3, D1.4, D7.1, Sample 3 | D6.2 Prompt Engineering, Samples 2 and 3 | 1.4-K1, 1.4-K2, 1.5-K3, 2.1-K4, 2.1-S4, Q1, Q2, Q3 | 2.2, 2.3, 4.4, Samples 1, 2 and 3 |
| [Explicit criteria and precision](#explicit-criteria-and-precision) | D1.1, D5.3 | D6.2 Prompt Engineering (instruction clarity, output constraints) | 4.1-K1 to 4.1-S3, 4.2-S3, 3.6-S3, 5.2-K1, 5.2-S1, Q3, Scenario 5 | 2.2 |
| [Few-shot examples](#few-shot-examples) | D1.1, D7.1 | D5.1 LLM Fundamentals (zero-, single-, multi-shot), D6.2 Prompt Engineering | 4.2-K1 to 4.2-S5, 3.5-K1, 3.5-S1, 3.5-S4, 5.2-S1, Q2, EX3-STEP3, APPX-TECH-10, APPX-INSCOPE-14 | 2.3 |
| [XML tags, system prompts and roles](#xml-tags-system-prompts-and-roles) | D5.1, D5.3 | D6.2 Prompt Engineering (system versus user placement), D2.5 Claude Application Design (content boundaries), Sample 2 | 2.1-K4, 2.1-S4, APPX-TECH-5 | 2.2, 2.5, Sample 2 |
| [Chain of thought and prompt chaining](#chain-of-thought-and-prompt-chaining) | D1.2 | D5.1 LLM Fundamentals (extended thinking, adaptive thinking, effort levels) | 1.6-K1 to 1.6-S3, APPX-TECH-11, Q12 | 1.5, 2.3 |
| [Long documents](#long-documents) | D3.4 | D6.1 Context Engineering, D6.2 Prompt Engineering | 5.1-K2, 5.1-S4, 5.1-S5, 5.1-S6, APPX-TECH-12, APPX-INSCOPE-16 | 2.4 |
| [Structured output with tools and JSON schemas](#structured-output-with-tools-and-json-schemas) | D2.6 | D6.3 Output Handling, D2.5 Claude Application Design (schema design), D8.1 Tool Implementation | 4.3-K1 to 4.3-S6, 2.3-S4, 2.3-S5, 3.6-K2, 3.6-S2, EX3-STEP1, APPX-TECH-5, APPX-TECH-7, APPX-INSCOPE-13, Scenario 6 | - |
| [Validation, retry and feedback loops](#validation-retry-and-feedback-loops) | D2.3, D7.2 | D6.3 Output Handling, D4.1 Debugging and Error Handling | 4.4-K1 to 4.4-S4, 4.3-K3, EX3-STEP2, APPX-TECH-8 | 1.2 |
| [Batch processing design](#batch-processing-design) | - | D2.3 Claude API Mechanics, Sample 1 | 4.5-K1 to 4.5-S4, EX3-STEP4, Q11, APPX-TECH-6, APPX-INSCOPE-15 | 1.6, 4.5 |
| [Multi-instance and multi-pass review](#multi-instance-and-multi-pass-review) | - | D2.4 Software Engineering Foundations (code review) | 4.6-K1 to 4.6-S3, 3.6-K4, 1.6-K2, 1.6-S2, Q12, How to Prepare | - |
| [Reducing hallucinations](#reducing-hallucinations) | D2.1, D2.2, D2.3, D2.4, Sample 1 | D6.3 Output Handling (skepticism toward confident output) | 4.3-S4, 4.2-K4, 5.2-K3, 5.5-K3, 5.5-S3, EX3-STEP1, APPX-INSCOPE-13, Q3 | 4.4, 5.2, Sample 3 |
| [Prompt versioning and iteration](#prompt-versioning-and-iteration) | D1.3, D5.4, D7.2 | D2.6 Configuration Management, D6.2 Prompt Engineering, D5.3 Model Selection and Tradeoffs | 4.5-S4, 4.1-S2 | 2.5, 4.3, 6.5 |

How to use the map by exam:

- **CCAR-F.** Domain 4 (20% of the exam) maps one task statement to each of six sections: 4.1 to [Explicit criteria and precision](#explicit-criteria-and-precision), 4.2 to [Few-shot examples](#few-shot-examples), 4.3 to [Structured output with tools and JSON schemas](#structured-output-with-tools-and-json-schemas), 4.4 to [Validation, retry and feedback loops](#validation-retry-and-feedback-loops), 4.5 to [Batch processing design](#batch-processing-design) and 4.6 to [Multi-instance and multi-pass review](#multi-instance-and-multi-pass-review). Scenario 5 (Claude Code for Continuous Integration) and Scenario 6 (Structured Data Extraction) both list Prompt Engineering & Structured Output as a primary domain. Preparation Exercise 3 (Build a Structured Data Extraction Pipeline) practices schema design, validation-retry loops, few-shot examples, batch processing and human review routing in one pipeline; its steps are on the [CCAR-F page](../claude-certified-architect-foundations.md#official-preparation-exercises).
- **CCDV-F.** Domain 6 (11.0%) holds Prompt Engineering (4.6%) and Output Handling (2.6%), both taught here, and Context Engineering (3.8%), which [Long documents](#long-documents) touches and [Context Engineering and Long-Running Work](context-engineering.md#why-context-is-a-budget) covers in depth; batch design sits in D2.3 Claude API Mechanics and prompt versioning in D2.6 Configuration Management, both in Domain 2 (33.1%). See [Domain 6](../claude-certified-developer.md#domain-6-prompt-and-context-engineering).
- **CCAO-F.** Domain 1 (14%) covers prompting and Domain 2 (21%, the largest CCAO-F domain) covers evaluating output, which is where [Reducing hallucinations](#reducing-hallucinations) carries most weight. See [Domain 2](../claude-certified-associate.md#domain-2-output-evaluation-and-validation).
- **CCAR-P.** Domain 2 (13%) includes prompting techniques, system prompts and templates, and prompt reuse; Domain 4 (16%) covers A/B testing, iterative improvement and diagnosing prompt failure and hallucinations. See [Domain 2](../claude-certified-architect-professional.md#domain-2-claude-models-prompting-context-engineering).

??? info "Sources"

    - [Claude Certified Architect, Foundations: Exam Guide (PDF, July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): Domain 4 weight; task statements 1.4 to 1.6, 2.1, 2.3, 3.5, 3.6, 4.1 to 4.6, 5.1, 5.2 and 5.5; Scenarios 5 and 6; Preparation Exercise 3; sample questions 1, 2, 3, 11 and 12; How to Prepare; appendix lists
    - [Claude Certified Developer, Foundations: Exam Guide (PDF, July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf): domain and skill weights; skills D2.3 to D2.6, D4.1, D5.1, D5.3, D6.1 to D6.3 and D8.1, including the wording "single-shot", "multi-shot", "content boundaries" and placement; Samples 1, 2 and 3 with their rationales
    - [Claude Certified Architect, Professional: Exam Guide (PDF, July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf): domain weights; objectives 1.2, 1.5, 1.6, 2.2 to 2.5, 4.3 to 4.5, 5.2 and 6.5; Samples 1, 2 and 3 with their rationales
    - [Claude Certified Associate, Foundations: Exam Guide (PDF, July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf): domain weights; objectives D1.1 to D1.4, D2.1 to D2.6, D3.4, D5.1, D5.3, D5.4, D7.1 and D7.2; Samples 1 and 3 with their rationales
    - [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices): clarity, context and motivation, examples, XML tags, roles, long-context layout, format control, prefill removal and its replacements (including classification with enum tools), thinking guidance, manual chain of thought as a fallback, prompt chaining, investigating before answering, re-checking model-specific techniques, redirects of the retired technique and prompt tool pages
    - [Prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview): prerequisites (success criteria, tests, first draft), when prompting is not the fix, the "living reference" note
    - [Prompting Claude Sonnet 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5): literal instruction following, first-turn specification, code review recall and the coverage-first finding prompt, effort guidance, sampling parameters rejected
    - [Prompting Claude Opus 4.8](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-4-8): literal interpretation and scope, concrete review bar, iterating review prompts against recall or F1
    - [Prompting Claude Opus 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5): report everything and filter in a separate pass, over-verification from verification instructions, positive examples, end-of-prompt length reminder
    - [Prompting Claude Opus 5.5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5-5): specific patterns to avoid, reasoning in the response text and the `reasoning_extraction` category, removing "think carefully" in chat, pasted-content tags, system prompt changes and thinking blocks, reading blocks by type, default effort
    - [Prompting Claude Fable 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5): fresh-context verifier subagents, auditing progress claims, the `reasoning_extraction` refusal category, prescriptive older prompts and skills
    - [Prompting Claude Fable 5.1](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-1): one complete example with an explanation of why it is correct, searching before answering about current state
    - [Steering thinking](https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost): effort as the first lever, per-message thinking nudges
    - [Thinking](https://platform.claude.com/docs/en/build-with-claude/thinking): defaults, billing, summarized thinking, `display` values, streaming required when `max_tokens` is above 21,333
    - [Thinking troubleshooting](https://platform.claude.com/docs/en/build-with-claude/thinking-troubleshooting): per-model thinking configuration table
    - [Extended thinking](https://platform.claude.com/docs/en/build-with-claude/extended-thinking): migration from `budget_tokens` to adaptive thinking and effort; Claude 4.7 and later reject extended-thinking requests
    - [Mid-conversation system messages](https://platform.claude.com/docs/en/build-with-claude/mid-conversation-system-messages): system-role messages, supported models, turn-scoped messages, cache order, code example
    - [Using the Messages API](https://platform.claude.com/docs/en/build-with-claude/working-with-messages): top-level `system` vs mid-conversation system messages, stateless API
    - [Messages API reference](https://platform.claude.com/docs/en/api/messages/create): the `system` parameter and role statement, `thinking` and `output_config` fields
    - [API errors](https://platform.claude.com/docs/en/api/errors): the prefill 400 error message; Claude Opus 5.5, Fable 5.1 and Mythos 5.1 rejecting `tool_choice` `any` and `tool`
    - [Citations](https://platform.claude.com/docs/en/build-with-claude/citations): enabling citations per document, location types, `cited_text`, guaranteed valid pointers, incompatibility with structured outputs, request example
    - [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs): JSON outputs and strict tool use, constrained decoding, schema features and limits, property ordering, complexity limits, SDK schema transformation and Pydantic `parse()`, refusal and max_tokens exceptions, enum casing, batch compatibility, the 400 error when citations are combined with `output_config.format`, grammar and prompt cache invalidation
    - [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): static content first, caching large example sets
    - [Define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools): `input_examples`, the tool-use system prompt, `tool_choice` options, naming a tool in the user message with `auto`, the replacement for forced tool use on the newest models, text before `tool_use` blocks
    - [How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works): a regex over model output should have been a tool call; client tool round trips and `stop_reason: "tool_use"`
    - [Troubleshooting tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/troubleshooting-tool-use): values outside an enum
    - [Mitigate jailbreaks and prompt injections](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks): third-party content in `tool_result` blocks, JSON encoding, no instructions inside tool results
    - [Reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations): definition, permission to say "I don't know", quotes first for long documents, the other basic and advanced techniques, the two example prompts, the limits of the techniques
    - [Reduce prompt leak](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-prompt-leak): role-based system prompts
    - [Reduce latency](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-latency): sentence and paragraph limits, `max_tokens` as a blunt limit, quality before latency
    - [Increase output consistency](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/increase-consistency): character prompts, structured outputs for guaranteed JSON, grounding responses with retrieval
    - [Define success criteria and build evaluations](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests): specific and measurable criteria, edge cases, volume over quality, A/B testing against a baseline, LLM graders that reason first
    - [Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence): prompt audits, stale over-specific instructions, the 36% cost measurement
    - [System prompts release notes](https://platform.claude.com/docs/en/release-notes/system-prompts/overview): the claude.ai system prompt and its scope
    - [Ticket routing](https://platform.claude.com/docs/en/about-claude/use-case-guides/ticket-routing): few-shot classification prompt, rationales, retrieved examples, hierarchical classifiers, tagged output, edge-case instructions
    - [Legal summarization](https://platform.claude.com/docs/en/about-claude/use-case-guides/legal-summarization): tagged summary sections, meta-summarization code
    - [Customer support agent](https://platform.claude.com/docs/en/about-claude/use-case-guides/customer-support-chat): content in the first user turn, modular prompt blocks
    - [Content moderation](https://platform.claude.com/docs/en/about-claude/use-case-guides/content-moderation): category definitions, precision and recall tracking
    - [Code Review](https://code.claude.com/docs/en/code-review): multi-agent review with a verification step, `/code-review` in a subagent and its effort levels, `REVIEW.md` levers and length warning, reactions used for tuning, dismissing findings, skip rules
    - [Claude Code memory](https://code.claude.com/docs/en/memory): CLAUDE.md delivery, load order, context not enforcement, `--append-system-prompt`
    - [Claude Code CLI reference](https://code.claude.com/docs/en/cli-reference): system prompt flags, `--json-schema` in print mode
    - [Claude Code best practices](https://code.claude.com/docs/en/best-practices): restarting after repeated corrections, fresh-context review, reviewers that always find gaps
    - [Modifying system prompts (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/modifying-system-prompts): default minimal prompt, `claude_code` preset with `append`
    - [Claude Code features in the Agent SDK](https://code.claude.com/docs/en/agent-sdk/claude-code-features): no hard precedence between CLAUDE.md levels
    - [Introduction to prompt design](https://support.claude.com/en/articles/7996853-introduction-to-prompt-design): the newly hired contractor framing
    - [My prompt isn't giving me a helpful answer](https://support.claude.com/en/articles/7996857-my-prompt-isn-t-giving-me-a-helpful-answer): breaking requests into substeps
    - [Set organization instructions](https://support.claude.com/en/articles/14546867-set-organization-instructions): specific instructions, conflicts, testing, precedence, reviewing and removing outdated instructions
    - [Understanding Claude's personalization features](https://support.claude.com/en/articles/10185728-understanding-claude-s-personalization-features): Instructions for Claude and project instructions scope
    - [How do usage and length limits work](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work): what project instructions should hold, length-limit advice
    - [Change the model, effort and thinking settings](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings): effort and thinking by task complexity
    - [Retrieval-augmented generation for projects](https://support.claude.com/en/articles/11473015-retrieval-augmented-generation-rag-for-projects): automatic RAG mode, file naming, naming documents in questions
    - [What are projects](https://support.claude.com/en/articles/9517075-what-are-projects): RAG on paid plans
    - [How can I create and manage projects](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects): project name and description not visible to Claude
    - [Usage limit best practices](https://support.claude.com/en/articles/9797557-usage-limit-best-practices): up-front requirements for drafting and analysis
    - [How large is the context window on paid Claude plans](https://support.claude.com/en/articles/8606394-how-large-is-the-context-window-on-paid-claude-plans): projects use retrieval
    - [Get started with Claude Design](https://support.claude.com/en/articles/14604416-get-started-with-claude-design): asking for 2 to 3 options
    - [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): the right altitude, minimal but sufficient prompts, section markers, canonical examples, context rot and attention budget
    - [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): prompt chaining definition, gates, when to chain, parallelization, the voting pattern for code review, single-call baseline
    - [The "think" tool](https://www.anthropic.com/engineering/claude-think-tool): built-in thinking preferred in most cases
    - [Prompt engineering for Claude's long context window](https://www.anthropic.com/news/prompting-long-context): the September 2023 position study, quotes in a scratchpad, instructions at the end
    - [Claude 2.1 prompting](https://claude.com/blog/claude-2-1-prompting): the historical prefill technique for long-context retrieval
    - [Best practices for prompt engineering](https://claude.com/blog/best-practices-for-prompt-engineering): troubleshooting table, one-shot first, attention to details in examples, roles, three chain-of-thought levels, chaining trade-offs, XML tags as less necessary, "lost-in-the-middle" and context awareness, "use null rather than guessing", prefill advice now superseded
    - [Reducing cost and improving performance with Claude Platform](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform): stale few-shot examples and scratchpad scaffolds as anti-patterns, volatile values in cached prefixes, the `/claude-api prompt-audit` result
    - [Claude 101: Getting better results](https://academy.claude.com/courses/claude-101/getting-better-results): symptom and fix table, iteration, testing analysis on known data
    - [Claude 101: Your first conversation with Claude](https://academy.claude.com/courses/claude-101/your-first-conversation-with-claude): the three-part prompt structure
    - [Claude 101: Introduction to projects](https://academy.claude.com/courses/claude-101/introduction-to-projects): what project instructions should contain, keeping project knowledge current
    - [Claude 101: Chat, Cowork and Code](https://academy.claude.com/courses/claude-101/claude-desktop-app-chat-cowork-code): brainstorming turn by turn
    - [AI capabilities and limitations: Steerability](https://academy.claude.com/courses/ai-capabilities-and-limitations/steerability): goal vs format, letter over spirit, checkpoints in multi-step tasks
    - [Building with the Claude API: Providing examples](https://academy.claude.com/courses/building-with-the-claude-api/providing-examples): examples from high-scoring eval outputs
    - [AI Fluency for nonprofits: Researching with AI](https://academy.claude.com/courses/ai-fluency-for-nonprofits/researching-with-ai): research prompts and discernment
    - [AI Fluency for nonprofits: Writing with AI](https://academy.claude.com/courses/ai-fluency-for-nonprofits/writing-with-ai): specific revision requests
    - [AI Fluency for small businesses: Researching with AI](https://academy.claude.com/courses/ai-fluency-for-small-businesses/researching-with-ai): flagging claims for primary-source checks
    - [AI Fluency framework foundations: Additional activities](https://academy.claude.com/courses/ai-fluency-framework-foundations/additional-activities): reusable prompt templates with placeholders
    - [Introduction to Claude Cowork: Validating skills for plugins](https://academy.claude.com/courses/introduction-to-claude-cowork/validating-skills-for-plugins): change one thing at a time
    - [The AI Fluency Index](https://academy.claude.com/tutorials/the-ai-fluency-index): inviting pushback, lower scrutiny of polished outputs
    - [Adapt content across platforms](https://academy.claude.com/use-cases/adapt-content-across-platforms): voice samples, deliberate variants
    - [Chart your data before you commit](https://academy.claude.com/use-cases/chart-your-data-before-you-commit): "flag anything that surprises you"
    - [Explore what Claude can do for you](https://academy.claude.com/use-cases/explore-what-claude-can-do-for-you): letting Claude interview you
    - [Claude with the Anthropic API (Anthropic Academy course notes)](https://anthropic.skilljar.com/claude-with-the-anthropic-api): "content boundaries" and XML tags
    - [Prompt engineering interactive tutorial, Chapter 7](https://github.com/anthropics/prompt-eng-interactive-tutorial/blob/master/Anthropic%201P/07_Using_Examples_Few-Shot_Prompting.ipynb): zero-shot, one-shot and n-shot definitions
    - [claude-code `code-review` plugin command](https://raw.githubusercontent.com/anthropics/claude-code/main/plugins/code-review/commands/code-review.md): categorical flag and do-not-flag criteria
    - [Strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use): what non-strict tool calls can get wrong; `strict: true` placement and guarantees
    - [Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls): `tool_use` block fields, `is_error` results, instructive error messages, Claude retrying invalid calls 2 to 3 times, `tool_result` ordering
    - [Claude Opus 5.5 migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide): strict tools with `auto` as the replacement for forced tool choice
    - [Batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing): 50% pricing, typical completion, 24-hour expiry, `custom_id` rules, asynchronous validation, result types and billing, ordering, request create and results examples, server tools in batches, caching in batches
    - [Message Batches API reference](https://platform.claude.com/docs/en/api/messages/batches): endpoints, processing status, results not guaranteed in request order
    - [Pricing](https://platform.claude.com/docs/en/about-claude/pricing): batch and caching discounts combine
    - [Models overview](https://platform.claude.com/docs/en/models/overview): every model ID is a pinned snapshot
    - [Model IDs and versions](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions): pinned IDs and dateless IDs
    - [Skills guide](https://platform.claude.com/docs/en/build-with-claude/skills-guide): pinning Skill versions in production
    - [Agent Skills for the enterprise](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise): the full evaluation suite before promoting a new version
    - [Managed Agents: agent setup](https://platform.claude.com/docs/en/managed-agents/agent-setup): versioned agent configurations
    - [Managed Agents: sessions](https://platform.claude.com/docs/en/managed-agents/sessions): pinning a session to an agent version for staged rollouts
    - [Managed Agents: define outcomes](https://platform.claude.com/docs/en/managed-agents/define-outcomes): the separate-context grader and `max_iterations`
    - [Agent SDK structured outputs](https://code.claude.com/docs/en/agent-sdk/structured-outputs): `outputFormat`/`output_format`, validation and re-prompting, result subtypes, success without `structured_output`, making fields optional
    - [Run Claude Code programmatically](https://code.claude.com/docs/en/headless): `--output-format json` with `--json-schema`, the `structured_output` field, invalid-schema error, `--bare`
    - [Environment variables](https://code.claude.com/docs/en/env-vars): `MAX_STRUCTURED_OUTPUT_RETRIES` default
    - [Workflows](https://code.claude.com/docs/en/workflows): schema-validated subagent output failing after five attempts
    - [Tools reference](https://code.claude.com/docs/en/tools-reference): the ReportFindings `category` slug
    - [Security guidance plugin](https://code.claude.com/docs/en/security-guidance): a separate Claude call reviews the diff
    - [Claude Security](https://code.claude.com/docs/en/claude-security): independent verification of patches
    - [Ultrareview](https://code.claude.com/docs/en/ultrareview): independently reproduced findings
    - [/goal](https://code.claude.com/docs/en/goal): completion judged by a fresh model
    - [Model configuration](https://code.claude.com/docs/en/model-config): Claude Code aliases that update over time
    - [Plugin evals](https://code.claude.com/docs/en/plugin-evals): three runs per case, pinning models in CI
    - [Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents): actionable error responses, held-out test sets
    - [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents): regression and capability evals, CI runs, A/B testing, multiple trials
    - [A statistical approach to model evaluations](https://www.anthropic.com/research/statistical-approach-to-model-evals): standard error, paired differences, small evals missing small differences
    - [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system): rainbow deployments
    - [A postmortem of three recent issues](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues): canary deployments
    - [Building agents with the Claude Agent SDK](https://claude.com/blog/building-agents-with-the-claude-agent-sdk): the best form of feedback names the rules that failed and why
    - [Building trusted AI in the enterprise](https://www-cdn.anthropic.com/e5c9de22bc8884089970bd262ca0c8b952cb9136.pdf): prompts need version control, testing and documentation; A/B testing infrastructure; updating offline evaluations with production data
    - [Prompt improver announcement](https://claude.com/blog/prompt-improver): prompt improver methods, including prefill addition
    - [Evaluate prompts in the developer console](https://claude.com/blog/evaluate-prompts): Console prompt versions and side-by-side comparison (2024)
    - [Extracting Structured JSON using Claude and Tool Use (Claude Cookbook)](https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/extracting_structured_json.ipynb): the extraction-tool pattern
    - [Metaprompt (Claude Cookbook)](https://github.com/anthropics/claude-cookbooks/blob/main/misc/metaprompt.ipynb): prompt templates for a blank page, not guaranteed optimal
    - [claude-code-security-review prompts](https://raw.githubusercontent.com/anthropics/claude-code-security-review/main/claudecode/prompts.py) and [findings filter](https://raw.githubusercontent.com/anthropics/claude-code-security-review/main/claudecode/findings_filter.py): finding categories and exclusion breakdowns
    - [AI-native SDLC playbook: continuous evals in CI (Anthropic Academy)](https://academy.claude.com/courses/ai-native-sdlc-playbook/continuous-evals-in-ci): gating configuration changes on eval results
    - [AI-native SDLC playbook: requirements and design (Anthropic Academy)](https://academy.claude.com/courses/ai-native-sdlc-playbook/requirements-and-design): logging prompts in version control
    - [AI-native SDLC playbook: CLAUDE.md (Anthropic Academy)](https://academy.claude.com/courses/ai-native-sdlc-playbook/claude-md): CLAUDE.md in Git, reviewed like code
    - [Why do AI models hallucinate? (Anthropic Academy)](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate): high-risk situations for hallucination
    - [Discernment toolkit (Anthropic Academy)](https://academy.claude.com/tutorials/discernment-toolkit): the source-grounding prompt
    - [Writing an AI diligence statement (Anthropic Academy)](https://academy.claude.com/tutorials/writing-an-ai-diligence-statement): asking the same question several ways, direct quotes
    - [Claude Cowork: the task loop (Anthropic Academy)](https://academy.claude.com/courses/introduction-to-claude-cowork/the-task-loop): mostly right versus wrong in a load-bearing way; untraceable details as flags
    - [Claude Cowork: share what you build with your team (Anthropic Academy)](https://academy.claude.com/courses/introduction-to-claude-cowork/share-what-you-build-with-your-team): one owner, evals before every publish, quarterly review
    - [Use artifacts to visualize and create AI apps (Anthropic Academy)](https://academy.claude.com/tutorials/use-artifacts-to-visualize-and-create-ai-apps-without-ever-writing-a-line-of-code): branching by editing a previous message
    - [What are artifacts and how do I use them? (Claude Help Center)](https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them): editing prior messages creates a new version of the conversation
    - [Claude is providing incorrect or misleading responses (Claude Help Center)](https://support.claude.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on): not a singular source of truth; review cited sources
    - [Claude is producing links that don't work and falsely claiming it has sent emails (Claude Help Center)](https://support.claude.com/en/articles/8241188-claude-is-producing-links-that-don-t-work-and-falsely-claiming-that-it-has-sent-emails-or-produced-external-documents-what-s-going-on): hallucinated capabilities
    - [Claude Certified Developer, Foundations prep path: accelerators and IP contribution (Anthropic Partner Academy)](https://anthropic-partners.skilljar.com/path/claude-certified-developer-foundations/accelerators-ip-contribution): versioning what ships
    - [Claude Certified Architect, Professional prep path: enterprise integration and production (Anthropic Partner Academy)](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/enterprise-integration-production): planning and reading an A/B test
    - [Effort](https://platform.claude.com/docs/en/build-with-claude/effort): the `effort` parameter and the models that support it
    - [Streaming messages](https://platform.claude.com/docs/en/build-with-claude/streaming): the `messages.stream()` helper and `get_final_message()`
    - [How the agent loop works (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/agent-loop): "the configured retry limit" and the `error_max_structured_output_retries` subtype
