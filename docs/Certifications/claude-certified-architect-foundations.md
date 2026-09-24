---
title: "Claude Certified Architect, Foundations (CCAR-F): Free Study Guide"
description: Free CCAR-F study guide covering the six exam scenarios, five weighted domains, all 30 task statements and the 12 official sample questions from Anthropic.
last_reviewed: 2026-09-23
---

# Claude Certified Architect, Foundations (CCAR-F)

According to the [official exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf), CCAR-F validates that you can "make informed decisions about tradeoffs when implementing real-world solutions with Claude", and it tests foundational knowledge across Claude Code, the Claude Agent SDK, the Claude API and the Model Context Protocol (MCP). Its 60 items are scenario-based: you see 4 scenarios drawn from a bank of 6, and each one presents a production context that frames a set of questions. This page follows the guide's Version 1.0 (effective July 2026) objective by objective, with the decision rule behind each one and Anthropic's own sample questions.

!!! abstract "The contract"

    **After this page you can:**

    - State the exam's numbers: 60 items, 120 minutes, 4 scenarios from a bank of 6, and a pass mark of 720 on a scaled range of 100 to 1,000.
    - Recognize each of the six scenarios and name the domains the guide lists as primary for it.
    - Apply the decision rule behind each of the 30 task statements across the five domains.
    - Work through all 12 official sample questions with the reasoning Anthropic gives for each answer.
    - Spot where current product documentation (as of September 2026) differs from the July 2026 guide, and answer exam items in the guide's terms.
    - Build your preparation around the guide's four hands-on exercises.

    **Who it is for:** the guide's ideal candidate is a solution architect who designs and implements production applications with Claude. That candidate typically has 6+ months of practical experience building with the Claude APIs, the Agent SDK, Claude Code and MCP. Details are in [Who this exam is for](#who-this-exam-is-for).

    **Who it is not for:**

    | If this describes you | Look at this instead |
    |---|---|
    | Your Claude work happens in the Claude apps (Projects, Artifacts, workflows), not in code you write against the API | [Claude Certified Associate, Foundations (CCAO-F)](claude-certified-associate.md) |
    | You mainly write application code against the Claude API: integration, streaming, model tiers, caching, security guardrails | [Claude Certified Developer, Foundations (CCDV-F)](claude-certified-developer.md) |
    | You lead enterprise-scale design, governance and stakeholder decisions, with 3+ years in systems architecture or platform engineering | [Claude Certified Architect, Professional (CCAR-P)](claude-certified-architect-professional.md) |
    | You do not work at a Claude Partner Network organization | As of September 2026, certification is available only to organizations in the Claude Partner Network: see [Who can sit the exams](index.md#who-can-sit-the-exams) |

## Exam at a glance

Start with the guide's own details table: 60 items, 120 minutes, 4 scenarios from a bank of 6, and a scaled pass mark of 720. This section then explains what those numbers mean on the day, adds the program rules the table leaves out, and lists where current product documentation (as of September 2026) differs from the guide.

### The official details table

This is Section 3 ("Exam Details at a Glance", page 2) of the [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf), row for row.

| Field | Value in the guide |
|---|---|
| Credential | Claude Certified Architect, Foundations |
| Exam code | CCAR-F |
| Number of items | 60 |
| Item format | Multiple-choice and multiple-response items; each item states how many responses to select |
| Exam structure | 4 scenarios drawn from a bank of 6 |
| Time limit | 120 minutes |
| Delivery | Proctored: online proctored and/or test center, per program policy |
| Passing score | Scaled score of 720 on a scale of 100 to 1,000 |
| Exam fee | &#36;125 USD |
| Validity period | 12 months from the date the credential is awarded |
| Result reporting | Pass/fail with scaled score (100 to 1,000), plus percent-correct by domain on the score report |

The "Exam structure" row does not appear in the other three Claude certification guides, and none of them publishes a bank of named scenarios. The draw is explained in [The six scenarios](#the-six-scenarios).

### What the numbers mean on the day

- **Pacing.** 120 minutes for 60 items is 2 minutes per item (our arithmetic). If the items were split evenly across the 4 scenarios, each scenario would carry 15 items and 30 minutes (our arithmetic; the guide does not state a per-scenario split).
- **Multiple response.** Some items ask for more than one answer, and each item tells you how many to select. All 12 official sample questions are single-answer items with four options (A to D), so the samples never show the multiple-response format.
- **The pass mark is scaled.** 720 is a point on a 100 to 1,000 scale, not a percentage. The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says scaled scoring helps "equate scores across multiple exam forms that might have slightly different difficulty levels", and it does not publish how many correct answers a 720 requires.
- **Domain percentages are feedback only.** The score report shows percent correct per domain, but pass or fail rests on the total scaled score. See [Scoring, results and badges](index.md#scoring-results-and-badges).
- **Seat time is longer than the clock.** The [program FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says: "Plan for about 135 minutes of total seat time, which includes check-in, instructions, and a brief post-exam survey."

### Beyond the guide's table (as of September 2026)

The guide's details table leaves out several things that decide whether and when you can sit the exam. Other sections of the guide and the program pages cover them.

| Topic | What the program documents say | Details |
|---|---|---|
| Who can register | Only people at Claude Partner Network organizations, using a partner email address on a recognized company domain; personal addresses do not work | [Who can sit the exams](index.md#who-can-sit-the-exams) |
| Price | &#36;125, raised from &#36;99 effective June 30, 2026. Registered-tier partners pay full price; Select, Preferred and Global Premier partners get 50% off, applied automatically at checkout; Global Premier partners get 100% off through December 31, 2026, and the standard 50% after that | [Fees and partner discounts](index.md#fees-and-partner-discounts) |
| Cancel or reschedule | The guide says up to 24 hours before the appointment, and changes within 24 hours forfeit the fee. The Skilljar policies page and the FAQ say 48 hours, and Pearson VUE's Anthropic page says 48 hours for test center appointments. Act at least 48 hours ahead | [Policies that cost candidates money](index.md#policies-that-cost-candidates-money) |
| Retakes | Waiting periods of 14 days after the first failed attempt, 30 after the second and 90 after the third; up to four attempts in a rolling twelve-month period; the fee applies to each attempt | [Policies that cost candidates money](index.md#policies-that-cost-candidates-money) |
| Where you sit it | Delivered by Pearson VUE, with online proctoring or a Pearson test center | [Registration, step by step](index.md#registration-step-by-step) |
| Language | The exam and prep content are in English only | [Program FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) |
| Validity and renewal | 12 months; the guide describes on-time renewal as a free, non-proctored assessment on the Anthropic Partner Academy, and a lapse means retaking the full exam at the full fee; the FAQ says full renewal details will be shared before the first certifications come up for renewal | [Renewal](index.md#renewal) |

### Guide version and document history

The [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) title block reads Version 1.0, effective July 2026, exam code CCAR-F, and adds: "This guide is subject to change without notice." It is a 39-page PDF linked from the exam's certification page on the [Anthropic Partner Academy](https://anthropic-partners.skilljar.com/claude-certified-architect-foundations-certification). The [program FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) calls the exam guide "the authoritative source for exam scope".

Section 18 of the guide ("Document Control") lists three versions:

| Version | Summary of change | Date |
|---|---|---|
| 1.0 | Formatting and layout updates | July 2026 |
| 0.2 | Draft revision | June 2026 |
| 0.1 | Initial draft | February 2026 |

The other three Claude certification guides (for example the [CCAR-P guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)) list a single row, "1.0 Initial publication July 2026". The CCAR-F table gives only these one-line summaries and does not list what changed between versions. Before you study, open the PDF from the exam page and check the title block: this page follows Version 1.0.

### Where current documentation differs from the guide

The guide's objectives do not always match the product documentation as of September 2026. Some differences predate the guide: the [subagents documentation](https://code.claude.com/docs/en/sub-agents.md) says the Task tool was renamed Agent in Claude Code version 2.1.63, and the [Claude Code changelog](https://code.claude.com/docs/en/changelog.md) dates that version February 28, 2026, yet Version 1.0 of the guide (July 2026) still says Task. Others are newer than the guide: the [API release notes](https://platform.claude.com/docs/en/release-notes/overview) entries for September 1, 2026 (the launch of Claude Fable 5.1 and Claude Mythos 5.1) and September 22, 2026 (the launch of Claude Opus 5.5) say that `tool_choice` types `any` and `tool` return a 400 error on those models.

Exam items are written against the guide's objectives, and neither the guide nor the program FAQ says whether the live item bank has been updated for any of these differences. Answer exam items in the guide's terms, and use the current behavior for real work.

| The guide says | Current documentation (as of September 2026) | Taught in |
|---|---|---|
| The agentic loop continues when `stop_reason` is "tool_use" and terminates when it is "end_turn" | The loop continues while `stop_reason` is `tool_use` and exits on any other stop reason (`end_turn`, `max_tokens`, `stop_sequence` or `refusal`); a server-tool loop that hits its iteration cap returns `pause_turn` instead of `end_turn` | [Domain 1](#domain-1-agentic-architecture-orchestration) |
| The Task tool spawns subagents, and allowedTools must include "Task" | The tool was renamed Agent in Claude Code 2.1.63; existing `Task(...)` references in settings and agent definitions still work as aliases. In the Agent SDK it appears as "Agent" in `tool_use` blocks and as "Task" in the `system:init` tools list | [Domain 1](#domain-1-agentic-architecture-orchestration) |
| Subagents do not inherit the coordinator's conversation history, so their context must be provided in the prompt | Consistent for ordinary subagents: the Agent SDK docs say the only content passed from parent to subagent is the Agent tool's prompt string. The exception is a fork, which the Claude Code docs describe as a subagent that inherits the entire conversation so far instead of starting fresh | [Domain 1](#domain-1-agentic-architecture-orchestration) |
| PostToolUse hooks transform tool results; tool call interception hooks block policy violations | Consistent, with names the guide does not use: interception is the `PreToolUse` event, which can deny a call with `permissionDecision`; a `PostToolUse` hook can replace any tool's output with `updatedToolOutput`, and the MCP-only `updatedMCPToolOutput` is deprecated | [Domain 1](#domain-1-agentic-architecture-orchestration) |
| `tool_choice` options are "auto", "any" and forced tool selection | A fourth option, `none`, prevents tool use. `any` and a forced `tool` return a 400 error on Claude Opus 5.5, Claude Fable 5.1 and Claude Mythos 5.1, where the docs point to `auto` with strict tool use, or to structured outputs, instead; with manual extended thinking (`thinking: {type: "enabled"}`) they are not supported either, and the docs list `auto` or `none` | [Domain 2](#domain-2-tool-design-mcp-integration), [Domain 4](#domain-4-prompt-engineering-structured-output) |
| MCP servers are project-level (`.mcp.json`) for shared tooling or user-level (`~/.claude.json`) for personal and experimental servers | There are three scopes. Local scope is the default, is also stored in `~/.claude.json`, and is the scope the docs suggest for experimental configurations | [Domain 2](#domain-2-tool-design-mcp-integration) |
| Tools from all configured MCP servers are discovered at connection time and available simultaneously | With tool search, which is on by default, only tool names and server instructions load at session start; full tool definitions are deferred until Claude needs them | [Domain 2](#domain-2-tool-design-mcp-integration) |
| Grep searches file contents and Glob matches file paths | The selection logic still matches the tool descriptions, but on macOS, Linux and WSL Claude Code leaves Glob and Grep out of the default tool set and searches with `find` and `grep` through Bash; naming either tool in `--allowedTools` restores both | [Domain 2](#domain-2-tool-design-mcp-integration) |
| When Edit fails on a non-unique match, fall back to Read + Write | The docs describe supplying a longer, unique string, or setting `replace_all: true` | [Domain 2](#domain-2-tool-design-mcp-integration) |
| Project slash commands live in `.claude/commands/` | Custom commands have been merged into skills; a file in `.claude/commands/` still works, and the docs prefer a skill for new work | [Domain 3](#domain-3-claude-code-configuration-workflows) |
| `allowed-tools` in skill frontmatter restricts tool access during the skill | `allowed-tools` pre-approves the listed tools and does not restrict which tools are available; `disallowed-tools` removes tools while the skill is active | [Domain 3](#domain-3-claude-code-configuration-workflows) |
| `argument-hint` prompts developers for required parameters | The docs describe it as a hint shown during autocomplete | [Domain 3](#domain-3-claude-code-configuration-workflows) |
| Path-scoped rules load only when editing matching files | Path-scoped rules trigger when Claude reads a matching file; rules without a `paths` field load unconditionally | [Domain 3](#domain-3-claude-code-configuration-workflows) |
| `/memory` verifies which memory files are loaded | The docs point to `/context` to check which CLAUDE.md and rules files loaded into the current session | [Domain 3](#domain-3-claude-code-configuration-workflows) |
| Tool use with JSON schemas is the most reliable approach for schema-compliant output | The docs also offer structured outputs: JSON outputs (`output_config.format`) and strict tool use (`strict: true`), both through constrained decoding | [Domain 4](#domain-4-prompt-engineering-structured-output) |
| The batch API does not support multi-turn tool calling within a single request | The docs say server tools work in batch requests, through the same server-side agentic loop as the synchronous Messages API. A tool your own code runs still cannot execute in the middle of a batch request (our reading; the docs do not state it in those words) | [Domain 4](#domain-4-prompt-engineering-structured-output) |

## Who this exam is for

The [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) is direct about its target: "The ideal candidate for this certification is a solution architect who designs and implements production applications with Claude." It also says candidates must show "not only conceptual knowledge but practical judgment about architecture, configuration, and tradeoffs in production deployments". The 12 official sample questions follow that pattern: each describes a situation inside a scenario and asks for a decision (a root cause, a fix, a file location, an approach), not a definition.

### The experience profile

The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) describes the typical candidate this way: "The candidate typically has 6+ months of practical experience building with Claude APIs, Agent SDK, Claude Code, and MCP, understanding both the capabilities and limitations of large language models in production environments."

Just before that sentence, the guide lists seven kinds of hands-on experience the ideal candidate has. The table pairs each with the task statements that test it and the official exercise that practices it. The pairings are our mapping from the guide's wording, not the guide's own.

| Hands-on experience the guide lists | Where the exam tests it (our mapping) | Official exercise that practices it (our mapping) |
|---|---|---|
| Building agentic applications with the Claude Agent SDK: multi-agent orchestration, subagent delegation, tool integration and lifecycle hooks | [Domain 1](#domain-1-agentic-architecture-orchestration): 1.1 to 1.5, 1.7 | Exercises 1 and 4 |
| Configuring Claude Code for team workflows: CLAUDE.md files, Agent Skills, MCP server integrations and plan mode | [Domain 3](#domain-3-claude-code-configuration-workflows): 3.1 to 3.4; also 2.4 | Exercise 2 |
| Designing MCP tool and resource interfaces for backend system integration | [Domain 2](#domain-2-tool-design-mcp-integration): 2.1, 2.2, 2.4 | Exercise 1 (steps 1 and 3) |
| Engineering prompts for reliable structured output with JSON schemas, few-shot examples and extraction patterns | [Domain 4](#domain-4-prompt-engineering-structured-output): 4.2, 4.3, 4.4 | Exercise 3 |
| Managing context windows across long documents, multi-turn conversations and multi-agent handoffs | [Domain 5](#domain-5-context-management-reliability): 5.1, 5.4, 5.6; also 1.3 | Exercise 4 (steps 1 and 3) |
| Integrating Claude into CI/CD pipelines for automated code review, test generation and pull request feedback | 3.6, 4.1, 4.5, 4.6 | None of the four exercises targets CI/CD |
| Making escalation and reliability decisions: error handling, human-in-the-loop workflows and self-evaluation patterns | 2.2, 5.2, 5.3, 5.5, 4.6 | Exercises 1, 3 and 4 |

A practical readiness rule: if a row describes work you have never done, do that row's exercise before you book. The CI/CD row has no official exercise, so build one yourself: run `claude -p` with `--output-format json` and `--json-schema` in a pipeline, as [Domain 3](#domain-3-claude-code-configuration-workflows) explains. The exercises themselves are in [Official preparation exercises](#official-preparation-exercises).

### No stated prerequisite in the guide

The CCAR-F guide has no "Prerequisites" line. The CCAO-F, CCDV-F and CCAR-P guides each say there are no mandatory prerequisites or required courses. The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) words its 6+ months as what the candidate "typically has", a description of the typical candidate rather than a stated condition of entry.

Foundations and Professional are separate architect exams. The [program FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) puts the difference in two sentences: "Foundations proves an architect can build with Claude. Professional proves they can design and govern Claude solutions at enterprise scale." It adds that Foundations is the natural place to start, that you can take Professional without holding Foundations, and that Foundations does not convert or upgrade to Professional automatically.

For partner organizations, CCAR-F counts toward Claude Partner Network eligibility, as do CCDV-F and CCAR-P. The Associate certification (CCAO-F) does not count toward partner tier eligibility.

### When another exam fits better

| Exam | Audience in its guide | Experience its guide describes | Choose it over CCAR-F when |
|---|---|---|---|
| [CCAO-F](claude-certified-associate.md#who-this-exam-is-for) | Professionals who use Claude as a productivity tool and build Claude Projects; limited to moderate technical expertise | Regular hands-on use of Claude at work; no software-development or API experience needed | Your Claude work happens in the Claude apps rather than in code |
| [CCDV-F](claude-certified-developer.md#who-this-exam-is-for) | AI and machine learning engineers, technical leads and senior software engineers who build, integrate and ship | One to five years of software engineering, at least six months with Claude or comparable LLM-based systems, Python and/or TypeScript | Your work centers on application code against the API: its largest domain is Applications and Integration (33.1%), and Claude Code is 3.1% |
| [CCAR-P](claude-certified-architect-professional.md#who-this-exam-is-for) | Mid- to senior-level solution architects, AI/ML engineers, technical leads and senior software engineers, often involved in stakeholder engagement and in leading architectural decisions | 3+ years in systems architecture or platform engineering and 6+ months with Claude or comparable LLM-based systems in production | You own enterprise-scale design and governance, including security, legal and executive discussions |

Two contrasts help place CCAR-F. Claude Code configuration and workflows is a full domain here, [Domain 3](#domain-3-claude-code-configuration-workflows) at 20%, while CCDV-F gives Claude Code 3.1%. In the other direction, the CCDV-F guide includes streaming in one of the areas of applied knowledge its holders can demonstrate (integrating Claude into application code through the API), while the CCAR-F guide lists streaming, vision, rate limits and API pricing calculations, and specific cloud provider configurations as out of scope (see [What is out of scope](#what-is-out-of-scope)). The [CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) draws the boundary from below: "Candidates are not expected to design enterprise-scale AI architectures or integrations; that scope belongs to the Claude Architect and Claude Developer credentials, to which Associates escalate more complex or technical work."

To compare all four exams side by side, including fees and item counts, see [Pick your exam](index.md#pick-your-exam).

## Blueprint

The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) defines the weights in its Section 4: "Weights reflect the relative importance of each domain to competent performance as determined through the job task analysis. The percentages indicate the approximate proportion of scored items drawn from each domain." Its Section 6 then lists the task statements, with the line "Exam items are written against these objectives."

### Domains, weights and task statements

| Domain | Weight | Approximate items (our arithmetic) | Task statements | Objective bullets (Knowledge + Skills) | Approximate items per task statement (our arithmetic) |
|---|---|---|---|---|---|
| [1. Agentic Architecture & Orchestration](#domain-1-agentic-architecture-orchestration) | 27% | about 16 (27% of 60 is 16.2) | 7 (1.1 to 1.7) | 48 (24 + 24) | about 2.3 |
| [2. Tool Design & MCP Integration](#domain-2-tool-design-mcp-integration) | 18% | about 11 (18% of 60 is 10.8) | 5 (2.1 to 2.5) | 43 (20 + 23) | about 2.2 |
| [3. Claude Code Configuration & Workflows](#domain-3-claude-code-configuration-workflows) | 20% | about 12 | 6 (3.1 to 3.6) | 49 (23 + 26) | about 2.0 |
| [4. Prompt Engineering & Structured Output](#domain-4-prompt-engineering-structured-output) | 20% | about 12 | 6 (4.1 to 4.6) | 47 (22 + 25) | about 2.0 |
| [5. Context Management & Reliability](#domain-5-context-management-reliability) | 15% | about 9 | 6 (5.1 to 5.6) | 53 (24 + 29) | about 1.5 |
| Total | 100% | 60 | 30 | 240 (113 + 127) | 2.0 |

The first item column multiplies each weight by 60; the last divides that figure by the domain's number of task statements. Both are planning figures, not counts: the guide calls the weights approximate, applies them to scored items, and does not say whether any of the 60 items are unscored.

The guide lists each task statement's "Knowledge of" and "Skills in" bullets without numbers. This page numbers them in the guide's order, so 1.3-K1 is the first Knowledge bullet of Task Statement 1.3 and 4.3-S2 is the second Skills bullet of Task Statement 4.3.

### How to read the weights

- **Domain 1 is the largest.** At 27% it carries about 16 items, and with Domain 2 (18%) it accounts for 45% of the blueprint weight (our arithmetic). Agent loops, orchestration, subagents, enforcement, hooks, decomposition, sessions, tool design, MCP integration and the built-in tools make up close to half of the scored items.
- **Domains 3 and 4 are tied** at 20% each, about 12 items apiece (our arithmetic).
- **Domain 5 is spread thin.** It has the lowest weight (15%) but the most objective bullets (53), and it is a primary domain in four of the six scenarios, more than any other domain. Expect its ideas (escalation, error propagation, context, human review, provenance) inside many scenarios, at about 1.5 items per task statement (our arithmetic).
- **Use the domain breakdown after the exam.** The score report shows the percentage of items you answered correctly in each domain, and the [program FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says: "You can use the section breakdown to decide what to review before a retake."

### Ideas the guide tests in more than one place

Several ideas appear in two or more objective bullets, sometimes in different domains, and some also appear in a sample question or an exercise. Learn each once and apply the same answer wherever it shows up. The first table lists ideas that recur across objective bullets; the second, collapsed because it gives away the answers, maps each official sample question to the objectives its correct answer rests on (our mapping, following the rationales' wording). Try the questions in [Official sample questions](#official-sample-questions) before you open it.

| Idea | Where the guide tests it |
|---|---|
| Enforce critical ordering in code (hooks, prerequisite gates), not in prompt instructions | 1.4-K1, 1.4-K2, 1.4-S1, 1.5-K3, 1.5-S3; Question 1 |
| Split large reviews into per-file passes plus a cross-file integration pass | 1.6-K2, 1.6-S2, 4.6-K3, 4.6-S2; Question 12 |
| Tool descriptions are the primary mechanism for tool selection | 2.1-K1, 2.1-K2, 2.1-S1, 2.4-S3; Question 2 |
| `tool_choice` "any" guarantees a tool call; a forced tool guarantees a specific one (the guide's logic; as of September 2026 some models reject both, see [Where current documentation differs from the guide](#where-current-documentation-differs-from-the-guide)) | 2.3-K4, 2.3-S4, 2.3-S5, 4.3-K2, 4.3-S2, 4.3-S3 |
| Subagents recover locally from transient failures and propagate only what they cannot resolve, with what was attempted and partial results | 2.2-S3, 5.3-S3 |
| An access failure is not the same as a valid empty result | 2.2-S4, 5.3-K2, 5.3-S2 |
| Subagents do not inherit the coordinator's conversation, so pass what they need in their prompt | 1.2-K2, 1.3-K2, 1.3-S1; Exercise 4 |
| Keep verbose exploration out of the main conversation: run it in a subagent (such as Explore) or in a skill with `context: fork` | 3.2-K3, 3.2-S2, 3.4-K4, 3.4-S3, 5.4-K3, 5.4-S1 |
| Fork a session to compare approaches from a shared baseline | 1.3-K4, 1.7-K2, 1.7-S2 |
| Concrete examples beat prose when descriptions produce inconsistent results | 3.5-K1, 3.5-S1, 4.2-K1 |
| A session reviewing its own output is weaker than an independent reviewer | 3.6-K4, 4.6-K1, 4.6-K2, 4.6-S1 |
| Keep claim-source mappings intact through every hand-off | 1.3-S2, 5.6-K1, 5.6-K2, 5.6-S1; Exercise 4 |

??? note "Which objectives each sample question rests on (reveals the answers)"

    | Sample question (from its stem) | Scenario | Objectives the correct answer rests on (our mapping) |
    |---|---|---|
    | Question 1: the agent skips `get_customer` in 12% of cases | Customer Support Resolution Agent | 1.4-K1, 1.4-K2, 1.4-S1, 1.5-S3 |
    | Question 2: the agent calls `get_customer` for order questions | Customer Support Resolution Agent | 2.1-K1, 2.1-K2, 2.1-S1 |
    | Question 3: 55% first-contact resolution against the 80% target | Customer Support Resolution Agent | 5.2-K3, 5.2-S1 |
    | Question 4: where to create a shared `/review` command | Code Generation with Claude Code | 3.2-K1, 3.2-S1 |
    | Question 5: restructuring a monolith into microservices | Code Generation with Claude Code | 3.4-K1, 3.4-K3, 3.4-S1 |
    | Question 6: conventions for test files spread throughout the codebase | Code Generation with Claude Code | 3.3-K1, 3.3-K3, 3.3-S2, 3.3-S3 |
    | Question 7: reports cover only visual arts | Multi-Agent Research System | 1.2-K3, 1.2-K4 |
    | Question 8: the web search subagent times out | Multi-Agent Research System | 5.3-K1, 5.3-K3, 5.3-K4, 5.3-S1 |
    | Question 9: verification round trips from the synthesis agent | Multi-Agent Research System | 2.3-K3, 2.3-S3 |
    | Question 10: a pipeline job hangs waiting for interactive input | Claude Code for Continuous Integration | 3.6-K1, 3.6-S1 |
    | Question 11: moving two workflows to the Message Batches API | Claude Code for Continuous Integration | 4.5-K1, 4.5-K2, 4.5-K4, 4.5-S1 |
    | Question 12: an inconsistent single-pass review of 14 files | Claude Code for Continuous Integration | 1.6-S2, 4.6-K3, 4.6-S2 |

One idea looks contradictory across domains: self-reported confidence is rejected as an escalation trigger (Task Statement 5.2 and Question 3) but used, once calibrated on labeled validation sets, to route human review (Task Statements 4.6 and 5.5). How the two fit is explained in the note under [Task 5.2](#task-52-design-effective-escalation-and-ambiguity-resolution-patterns).

The CCAR-F sample questions carry no domain label; the guide groups them by scenario instead. The other three guides label every sample with a domain. The samples are reproduced in [Official sample questions](#official-sample-questions).

## The six scenarios

CCAR-F questions are scenario-based. The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) explains the format in its Section 5: "The exam uses scenario-based questions. Each scenario presents a realistic production context that frames a set of questions. During the exam, 4 scenarios are presented and picked at random from the full set of the 6 scenarios below."

### How the draw works

- There are 15 possible sets of four scenarios drawn from six, and every exam leaves two scenarios out (our arithmetic).
- Any one scenario belongs to 10 of the 15 sets, so if every set is equally likely it appears in two exams out of three (our arithmetic; the [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says only that scenarios are "picked at random").
- Scenarios 4 and 6 have no official sample question. Of the 15 possible sets, 14 include at least one of them and 6 include both (our arithmetic). Skipping them is a poor bet.
- The guide does not say how many items each scenario carries, or whether a scenario's items come only from its listed primary domains. Its [blueprint section](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) does say the weights give "the approximate proportion of scored items drawn from each domain". Prepare every domain, whatever the draw.

### The scenarios at a glance

| # | Scenario | Primary domains (in the guide's order) | Official sample questions | Closest official exercise (our mapping) |
|---|---|---|---|---|
| 1 | [Customer Support Resolution Agent](#scenario-1-customer-support-resolution-agent) | 1, 2, 5 | 3 (Questions 1 to 3) | Exercise 1 |
| 2 | [Code Generation with Claude Code](#scenario-2-code-generation-with-claude-code) | 3, 5 | 3 (Questions 4 to 6) | Exercise 2 |
| 3 | [Multi-Agent Research System](#scenario-3-multi-agent-research-system) | 1, 2, 5 | 3 (Questions 7 to 9) | Exercise 4 |
| 4 | [Developer Productivity with Claude](#scenario-4-developer-productivity-with-claude) | 2, 3, 1 | None | None targets it directly |
| 5 | [Claude Code for Continuous Integration](#scenario-5-claude-code-for-continuous-integration) | 3, 4 | 3 (Questions 10 to 12) | None |
| 6 | [Structured Data Extraction](#scenario-6-structured-data-extraction) | 4, 5 | None | Exercise 3 |

Seen from the domain side, only two scenarios are left out of any draw, so a domain that is primary in three or more scenarios is primary in at least one scenario of every exam (our arithmetic, from the guide's "Primary domains" lines).

| Domain | Primary in scenarios | Primary in every possible draw? (our arithmetic) |
|---|---|---|
| [1. Agentic Architecture & Orchestration](#domain-1-agentic-architecture-orchestration) | 1, 3, 4 | Yes, in at least one drawn scenario |
| [2. Tool Design & MCP Integration](#domain-2-tool-design-mcp-integration) | 1, 3, 4 | Yes, in at least one |
| [3. Claude Code Configuration & Workflows](#domain-3-claude-code-configuration-workflows) | 2, 4, 5 | Yes, in at least one |
| [4. Prompt Engineering & Structured Output](#domain-4-prompt-engineering-structured-output) | 5, 6 | No: 1 of the 15 sets leaves out both Scenario 5 and Scenario 6 |
| [5. Context Management & Reliability](#domain-5-context-management-reliability) | 1, 2, 3, 6 | Yes, in at least two |

Each scenario below gives the guide's text, the domains it lists, what the text names, and our mapping to the task statements most likely in play. The mapping takes the task statements in the listed primary domains that the scenario's wording or its official sample questions point to. It is a study aid, not the guide's classification.

### Scenario 1: Customer Support Resolution Agent

> "You are building a customer support resolution agent using the Claude Agent SDK. The agent handles high-ambiguity requests like returns, billing disputes, and account issues. It has access to your backend systems through custom Model Context Protocol (MCP) tools (get_customer, lookup_order, process_refund, escalate_to_human). Your target is 80%+ first-contact resolution while knowing when to escalate." ([exam guide, page 3](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf))

- **Primary domains:** [Domain 1](#domain-1-agentic-architecture-orchestration), [Domain 2](#domain-2-tool-design-mcp-integration), [Domain 5](#domain-5-context-management-reliability)
- **Tools and systems it names:** the Claude Agent SDK; custom MCP tools `get_customer`, `lookup_order`, `process_refund` and `escalate_to_human`; a target of 80%+ first-contact resolution
- **Task statements most likely in play (our mapping):** 1.1, 1.4, 1.5, 2.1, 2.2, 5.1, 5.2
- **Official sample questions:** 3 (Questions 1 to 3)
- **Closest official exercise (our mapping):** Exercise 1, which reinforces the same three domains

**What the wording signals.** Several objective bullets are written in this scenario's own terms:

- Blocking `process_refund` until `get_customer` has returned a verified customer ID is the guide's example of a programmatic prerequisite (1.4-S1), and Question 1 tests it.
- Blocking refunds above a threshold is the guide's example of a hook that intercepts a tool call and redirects to human escalation (1.5-S2 names refunds exceeding &#36;500).
- The escalation half of the target ("knowing when to escalate") is [Task 5.2](#task-52-design-effective-escalation-and-ambiguity-resolution-patterns), which sets out the escalation rules situation by situation. Question 3 opens with 55% first-contact resolution against the 80% target.
- Order lookups that return 40+ fields when only 5 are relevant (5.1-K3), and a persistent "case facts" block for amounts, dates, order numbers and statuses (5.1-S1), use this setting's vocabulary.
- A human agent who cannot see the transcript needs a structured handoff: customer ID, root cause, refund amount and recommended action (1.4-S3).

Go deeper: [the agentic loop](knowledge/agents-and-agent-sdk.md#the-agentic-loop), [hooks in the SDK](knowledge/agents-and-agent-sdk.md#hooks-in-the-sdk), [escalation and ambiguity](knowledge/evaluation-and-reliability.md#escalation-and-ambiguity).

### Scenario 2: Code Generation with Claude Code

> "You are using Claude Code to accelerate software development. Your team uses it for code generation, refactoring, debugging, and documentation. You need to integrate it into your development workflow with custom slash commands, CLAUDE.md configurations, and understand when to use plan mode vs direct execution." ([exam guide, page 3](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf))

- **Primary domains:** [Domain 3](#domain-3-claude-code-configuration-workflows), [Domain 5](#domain-5-context-management-reliability)
- **Tools and systems it names:** Claude Code; custom slash commands; CLAUDE.md configurations; plan mode and direct execution
- **Task statements most likely in play (our mapping):** 3.1, 3.2, 3.3, 3.4, 3.5, 5.4
- **Official sample questions:** 3 (Questions 4 to 6)
- **Closest official exercise (our mapping):** Exercise 2, which configures CLAUDE.md, path-specific rules, a skill, MCP servers and plan mode for a team

**What the wording signals.**

- The three things the text names map directly to task statements: custom slash commands (3.2), CLAUDE.md configurations (3.1, with path-specific rules in 3.3) and plan mode versus direct execution (3.4). Questions 4, 5 and 6 turn on 3.2, 3.4 and 3.3 respectively (our reading of the rationales).
- Domain 5's share here most likely comes from 5.4 (our mapping): context degradation in extended sessions, scratchpad files, subagent delegation for verbose exploration, and `/compact`.
- As of September 2026, custom commands have been merged into skills. A file in `.claude/commands/` still works, so Question 4's answer still holds, although the docs prefer a skill for new work.

Go deeper: [CLAUDE.md and the memory hierarchy](knowledge/claude-code-configuration.md#claudemd-and-the-memory-hierarchy), [path-scoped rules](knowledge/claude-code-configuration.md#path-scoped-rules), [plan mode or direct execution](knowledge/claude-code-workflows.md#plan-mode-or-direct-execution).

### Scenario 3: Multi-Agent Research System

> "You are building a multi-agent research system using the Claude Agent SDK. A coordinator agent delegates to specialized subagents: one searches the web, one analyzes documents, one synthesizes findings, and one generates reports. The system researches topics and produces comprehensive, cited reports." ([exam guide, page 4](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf))

- **Primary domains:** [Domain 1](#domain-1-agentic-architecture-orchestration), [Domain 2](#domain-2-tool-design-mcp-integration), [Domain 5](#domain-5-context-management-reliability)
- **Tools and systems it names:** the Claude Agent SDK; a coordinator agent; four specialized subagents (web search, document analysis, synthesis, report generation); cited reports as the output
- **Task statements most likely in play (our mapping):** 1.2, 1.3, 2.1, 2.2, 2.3, 5.1, 5.3, 5.6
- **Official sample questions:** 3 (Questions 7 to 9)
- **Closest official exercise (our mapping):** Exercise 4, which reinforces the same three domains

**What the wording signals.**

- A coordinator that delegates to specialized subagents is the hub-and-spoke pattern of Task Statement 1.2. Question 7 traces missing coverage to a coordinator whose decomposition was too narrow (1.2-K4).
- Subagents do not inherit the coordinator's context, so the synthesis subagent must receive the search results and document analysis in its prompt (1.3-K2, 1.3-S1).
- Running searches in parallel is 1.3-S3: the coordinator emits multiple Task tool calls in a single response rather than across separate turns. Scope is partitioned across subagents to minimize duplication (1.2-S2), and the coordinator re-delegates targeted queries when the synthesis output shows gaps (1.2-S3).
- The synthesis agent is the guide's recurring example for tool scoping: a synthesis agent attempting web searches is misuse (2.3-K2), and a scoped `verify_fact` tool is the fix Question 9 rewards (2.3-S3).
- A web search subagent that times out is Question 8: return structured error context to the coordinator (5.3-S1).
- The requirement for cited reports brings in Task Statement 5.6: claim-source mappings, conflicting statistics annotated with their sources, and publication dates in subagent outputs.
- As of September 2026, the tool the guide calls Task is named Agent; the guide's objectives (1.3-K1, 1.3-S3) still say Task, so answer exam items in those terms. The rename is explained under [Task 1.3](#task-13-configure-subagent-invocation-context-passing-and-spawning).

Go deeper: [multi-agent orchestration](knowledge/agents-and-agent-sdk.md#multi-agent-orchestration), [error propagation in multi-agent systems](knowledge/evaluation-and-reliability.md#error-propagation-in-multi-agent-systems), [provenance and uncertainty in synthesis](knowledge/context-engineering.md#provenance-and-uncertainty-in-synthesis).

### Scenario 4: Developer Productivity with Claude

> "You are building developer productivity tools using the Claude Agent SDK. The agent helps engineers explore unfamiliar codebases, understand legacy systems, generate boilerplate code, and automate repetitive tasks. It uses the built-in tools (Read, Write, Bash, Grep, Glob) and integrates with Model Context Protocol (MCP) servers." ([exam guide, page 4](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf))

- **Primary domains:** [Domain 2](#domain-2-tool-design-mcp-integration), [Domain 3](#domain-3-claude-code-configuration-workflows), [Domain 1](#domain-1-agentic-architecture-orchestration), in that order
- **Tools and systems it names:** the Claude Agent SDK; the built-in tools Read, Write, Bash, Grep and Glob; MCP servers
- **Task statements most likely in play (our mapping):** 2.4, 2.5, 3.2, 3.4, 3.5, 1.6, 1.7
- **Official sample questions:** none
- **Closest official exercise (our mapping):** none targets it directly; Exercise 2 covers MCP server configuration (step 4) and plan mode (step 5)

**What the wording signals.**

- The named built-in tools are Task Statement 2.5: Grep for searching file contents, Glob for finding files by path pattern, Read then Write when Edit cannot find unique anchor text, and exploration that starts with Grep and follows imports with Read instead of reading every file up front.
- The MCP server integration is Task Statement 2.4: shared servers in project `.mcp.json` with environment variable expansion, personal servers in `~/.claude.json`, and MCP tool descriptions detailed enough that the agent does not prefer Grep over a more capable MCP tool.
- Exploring unfamiliar codebases and understanding legacy systems match (our mapping) 1.6-S3 (decomposing an open-ended task such as adding tests to a legacy codebase), 1.7-S2 (forking to compare refactoring approaches from a shared codebase analysis) and 3.4's Explore subagent. Automating repetitive tasks fits 3.2's skills for task-specific workflows.
- The scenario lists five built-in tools and leaves out Edit, which Task Statement 2.5 and the Appendix include. Study all six.
- Domain 5 is not listed here, although Task Statement 5.4 covers large codebase exploration. The guide does not say whether a scenario's items can come from domains outside its primary list.
- As of September 2026, the current docs differ from the guide on three points in this scenario's objectives (Glob and Grep defaults, the Edit fallback, MCP scopes); see [Where current documentation differs from the guide](#where-current-documentation-differs-from-the-guide). Answer in the guide's terms: Grep for content, Glob for paths, Read then Write when Edit cannot find a unique match, and project `.mcp.json` versus personal `~/.claude.json`.

Go deeper: [managing context in large codebases](knowledge/claude-code-workflows.md#managing-context-in-large-codebases), [MCP in Claude Code](knowledge/tool-use-and-mcp.md#mcp-in-claude-code), [exploring large codebases](knowledge/context-engineering.md#exploring-large-codebases).

### Scenario 5: Claude Code for Continuous Integration

> "You are integrating Claude Code into your Continuous Integration/Continuous Deployment (CI/CD) pipeline. The system runs automated code reviews, generates test cases, and provides feedback on pull requests. You need to design prompts that provide actionable feedback and minimize false positives." ([exam guide, page 4](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf))

- **Primary domains:** [Domain 3](#domain-3-claude-code-configuration-workflows), [Domain 4](#domain-4-prompt-engineering-structured-output)
- **Tools and systems it names:** Claude Code in a CI/CD pipeline; automated code reviews; generated test cases; pull request feedback
- **Task statements most likely in play (our mapping):** 3.6, 4.1, 4.2, 4.4, 4.5, 4.6
- **Official sample questions:** 3 (Questions 10 to 12)
- **Closest official exercise (our mapping):** none of the four exercises targets CI/CD

**What the wording signals.**

- Running in a pipeline is Task Statement 3.6: the `-p` (or `--print`) flag for non-interactive runs (Question 10), `--output-format json` with `--json-schema` for machine-parseable findings that can be posted as inline PR comments, CLAUDE.md for testing standards, fixture conventions and review criteria, prior findings in context so re-runs report only new or still-unaddressed issues, and existing test files in context so generated tests do not duplicate covered scenarios. As of September 2026, the docs put the schema-validated result in the `structured_output` field of the JSON response.
- Minimizing false positives is Task Statement 4.1: explicit categorical criteria rather than general instructions to be conservative, and temporarily disabling high false-positive categories. 4.2 adds few-shot examples that separate acceptable patterns from genuine issues, and 4.4 adds a `detected_pattern` field for analyzing dismissed findings.
- Actionable feedback maps (our mapping) to few-shot examples that show the desired output format, such as location, issue, severity and suggested fix (4.2-S2), and to explicit severity criteria with concrete code examples for each level (4.1-S3).
- Review at scale brings in 4.6 (a second, independent instance reviews the code without the generator's reasoning context, which 3.6-K4 also covers; per-file passes plus an integration pass, Question 12) and 4.5 (the synchronous API for blocking pre-merge checks, the Message Batches API for overnight work, Question 11).

Go deeper: [Claude Code in CI/CD](knowledge/claude-code-workflows.md#claude-code-in-cicd), [automated code review that engineers trust](knowledge/claude-code-workflows.md#automated-code-review-that-engineers-trust), [multi-instance and multi-pass review](knowledge/prompt-engineering.md#multi-instance-and-multi-pass-review).

### Scenario 6: Structured Data Extraction

> "You are building a structured data extraction system using Claude. The system extracts information from unstructured documents, validates the output using JavaScript Object Notation (JSON) schemas, and maintains high accuracy. It must handle edge cases gracefully and integrate with downstream systems." ([exam guide, page 4](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf))

- **Primary domains:** [Domain 4](#domain-4-prompt-engineering-structured-output), [Domain 5](#domain-5-context-management-reliability)
- **Tools and systems it names:** Claude; unstructured documents as input; JSON schemas for validating output; downstream systems that consume the results
- **Task statements most likely in play (our mapping):** 4.2, 4.3, 4.4, 4.5, 5.5
- **Official sample questions:** none
- **Closest official exercise (our mapping):** Exercise 3, which reinforces the same two domains

**What the wording signals.**

- Validating the output against JSON schemas is Task Statement 4.3: extraction tools with JSON schemas as input parameters, `tool_choice: "any"` when multiple extraction schemas exist and the document type is unknown, optional (nullable) fields so the model does not fabricate values for information the document lacks, and enum fields with an "unclear" value for ambiguous cases and an "other" value plus a detail string for extensible categories.
- Schema compliance is not correctness. Strict schemas via tool use eliminate syntax errors but not semantic errors such as line items that do not sum to the total (4.3-K3). 4.4-S4 flags them by extracting `calculated_total` alongside `stated_total` and adding a `conflict_detected` boolean for inconsistent source data.
- The demands for high accuracy and graceful handling of edge cases point to 4.4 (retry with the original document, the failed extraction and the specific validation errors, and knowing that retries are ineffective when the information is absent from the source) and 4.2 (few-shot examples for varied document structures).
- Deciding which extractions a human checks is Task Statement 5.5: field-level confidence calibrated on labeled validation sets, stratified random sampling of high-confidence extractions, and accuracy analyzed by document type and field, because an aggregate figure such as 97% overall may mask poor performance on specific document types or fields.
- Volume work brings in 4.5: the Message Batches API with 50% cost savings, a processing window of up to 24 hours, no guaranteed latency SLA, and `custom_id` to match requests to results and to resubmit only the documents that failed.
- As of September 2026, some models and settings reject `any` and a forced `tool`; the details and the documented replacement are under [Task 4.3](#task-43-enforce-structured-output-using-tool-use-and-json-schemas). Exam answers still follow the guide's logic: `"any"` guarantees a tool call, and a forced tool guarantees a specific one.

Go deeper: [structured output with tools and JSON schemas](knowledge/prompt-engineering.md#structured-output-with-tools-and-json-schemas), [validation, retry and feedback loops](knowledge/prompt-engineering.md#validation-retry-and-feedback-loops), [human review and confidence calibration](knowledge/evaluation-and-reliability.md#human-review-and-confidence-calibration).

## Domain 1: Agentic Architecture & Orchestration

**Official weight: 27%**, the approximate share of scored items, which is about 16 of the 60 items (27% of 60, our arithmetic; the guide gives weights as a share of scored items and does not say how many of the 60 are scored).

Domain 1 has seven task statements (1.1 to 1.7) carrying 24 "Knowledge of" and 24 "Skills in" bullets (our count of the guide's bullets). The guide lists it as a primary domain for three scenarios: Customer Support Resolution Agent, Multi-Agent Research System, and Developer Productivity with Claude (where it is listed third). If one of those three is among the four scenarios you draw, expect loop, orchestration, hook and session questions to be framed around it (our reading of the guide's primary-domain lists; see [The six scenarios](#the-six-scenarios)).

| Task | What it tests, in one line | Official sample question that exercises it |
|---|---|---|
| 1.1 | Loop on `stop_reason`, append tool results, avoid three named anti-patterns | none |
| 1.2 | Hub-and-spoke coordinator, dynamic subagent selection, coverage-driven refinement | Question 7 |
| 1.3 | Task tool, explicit context passing, AgentDefinition, parallel spawning | none |
| 1.4 | Code-enforced prerequisites, multi-concern requests, structured human handoff | Question 1 |
| 1.5 | PostToolUse normalization, tool call interception, hooks over prompts | Question 1 (shared with Task 1.4; its rationale applies the rule of 1.5-S3: code over prompts when compliance must be guaranteed) |
| 1.6 | Prompt chaining versus adaptive decomposition, per-file plus integration passes | Question 12 (shared with Task 4.6) |
| 1.7 | `--resume <session-name>`, `fork_session`, fresh start with a structured summary | none |

The guide does not label its sample questions by domain; the right-hand column is our mapping. The questions themselves, with Anthropic's rationales, are in [Official sample questions](#official-sample-questions).

### Task 1.1: Design and implement agentic loops for autonomous task execution

The six bullets cover one mechanism: send a request, read `stop_reason`, run the tools Claude asked for, put the results back into the conversation, and go round again until Claude is done. They also name three anti-patterns.

**Know**

- **Who executes tools.** Claude never runs a tool itself. The tool-use docs put it plainly: "The model never executes anything on its own." ([How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works)). Claude emits a structured request, your code (or Anthropic's servers, for server tools such as web search) runs it, and the result flows back into the conversation.

- **The lifecycle, step by step:**

| Step | What happens | What your code checks |
|---|---|---|
| 1 | Send `messages` with your `tools` (and `system`, `max_tokens`) | nothing yet |
| 2 | Claude replies with `stop_reason: "tool_use"` and one or more `tool_use` blocks, each carrying `id`, `name` and `input` | `stop_reason` |
| 3 | Run the tool that matches each `name`, passing its `input` | each block's `name` and `input` |
| 4 | Append the assistant response, then one user message holding one `tool_result` per `tool_use`, matched by `tool_use_id` | every `tool_use` has a result |
| 5 | Send the whole conversation again; repeat while `stop_reason` is `"tool_use"` | `stop_reason` |
| 6 | `stop_reason: "end_turn"` means Claude finished its response; use the final answer | `stop_reason` |

- **The loop in code.** This is the manual tool loop from the [stop reasons page](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons), with its comments removed. `execute_tools` is your function; it returns one `tool_result` block per `tool_use` block. The same page notes that for most use cases the SDK tool runner handles this loop with less code.

```python
def complete_tool_workflow(client, user_query, tools):
    messages = [{"role": "user", "content": user_query}]
    while True:
        response = client.messages.create(
            model="claude-opus-5-5", max_tokens=1024, messages=messages, tools=tools
        )
        if response.stop_reason == "tool_use":
            tool_results = execute_tools(response.content)
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
        else:
            return response
```

- **How tool results re-enter the conversation (1.1-K2, 1.1-S2).** The API keeps no state between calls: "The Messages API is stateless, which means that you always send the full conversational history to the API." ([Working with messages](https://platform.claude.com/docs/en/build-with-claude/working-with-messages)). Tool calls live in assistant messages and tool results in user messages; there is no `tool` or `function` role. Three ordering rules matter: results must immediately follow the `tool_use` blocks they answer, `tool_result` blocks come first in the user message with any text after them, and parallel calls get all their results together in one user message. A missing `tool_result`, or one that is not the first content block in the user message, returns the error `tool_use ids were found without tool_result blocks immediately after`.

- **A failed tool still returns a result.** Put the error text in `content` and set `is_error: true`, so Claude can retry, ask, or explain. The docs recommend saying what went wrong and what to try next rather than a generic "failed".

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

- **Model-driven versus pre-configured (1.1-K3).** Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) draws the line: "Workflows are systems where LLMs and tools are orchestrated through predefined code paths." and "Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks." In an agentic loop Claude picks the next tool from the conversation so far and each tool's description; with `tool_choice` at `auto` (the default when tools are provided), it calls a tool when the request maps to that tool's described capability and the answer is not already in context.

- **Where the loop is already written for you.** The Claude Agent SDK runs the same loop: Claude "continues calling tools and processing results until it produces a response with no tool calls" ([Agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop)). Its `max_turns` / `maxTurns` option counts tool-use turns, `max_budget_usd` / `maxBudgetUsd` caps spend, both default to no limit, and hitting either ends the run with a `ResultMessage` whose subtype is `error_max_turns` or `error_max_budget_usd`. The client SDKs also offer a Tool Runner (in beta as of September 2026) that automates the Messages API loop: it runs until Claude returns a message without a tool use, or until `max_iterations` if you set one. The docs point you to the manual loop when you need human-in-the-loop approval, custom logging, or conditional execution.

!!! warning "Exam guide vs current docs: which stop reasons end the loop"

    The guide frames loop control as two values: continue on `"tool_use"`, terminate on `"end_turn"`. The API docs (as of September 2026) say the loop continues while `stop_reason` is `tool_use` and "The loop exits on any other stop reason (`"end_turn"`, `"max_tokens"`, `"stop_sequence"`, or `"refusal"`)" ([How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works)). Loops that use server tools such as web search can also get `pause_turn` when the server-side loop hits its iteration limit (10 by default); you continue by sending the assistant content back. Answer exam items in the guide's terms: `tool_use` means continue, `end_turn` means done. In real code, handle the other values too.

**Decide**

- If the question asks what should drive the loop, choose inspecting `stop_reason` on every response; not scanning the reply text or counting turns, because `stop_reason` is the structured field the API returns to say why Claude stopped.
- If Claude needs a tool's output for its next step, append the assistant turn and the `tool_result` to the history and call again; not a fresh request that carries only the new result, because the API is stateless and Claude reasons only over what you send.
- If the right next step depends on what earlier tools returned, let Claude choose the next tool; not a hard-coded sequence or decision tree, because 1.1-K3 sets model-driven decision-making against pre-configured decision trees and tool sequences, and an agentic loop is the model-driven side. Code-enforced ordering is for rules that must never be broken (Task 1.4).
- If you need protection against a runaway loop, keep a turn or budget cap as a backstop; not as the primary stop. Anthropic's guidance keeps limits as a safety net: [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) says the task "often terminates upon completion" and that a maximum number of iterations is a common extra stopping condition "to maintain control", and the [Agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) page explains that "Without limits, the loop runs until Claude finishes on its own, which is fine for well-scoped tasks but can run long on open-ended prompts" before adding "Setting a budget is a good default for production agents." So a cap is fine as a safety net and wrong as the main exit condition.

**Traps**

- **Parsing natural language to decide termination**, for example stopping when the reply contains `done` or `task complete`. The tool-use docs are blunt: "if you're writing a regex to extract a decision from model output, that decision should have been a tool call." ([How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works)).
- **An arbitrary iteration cap as the primary stopping mechanism.** It can cut off a long legitimate task, and hitting it says nothing about whether the work is finished (our reasoning from 1.1-S3).
- **Treating assistant text as a completion signal.** The tool-use docs say "Claude often comments on what it's doing or responds naturally to the user before calling tools." ([Define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools)), so a response can hold text and a `tool_use` block together, and text does not mean done. (With `tool_choice` set to `any` or a forced tool, on the models and settings that support forced tool use, the API prefills the reply and Claude writes no text before the `tool_use` block.) The reverse also fails: an `end_turn` response can be empty (2 to 3 tokens, no content), particularly after tool results; one common cause is adding text blocks straight after the tool results.

**Go deeper:** [Stop reasons and the agent loop](knowledge/claude-api.md#stop-reasons-and-the-agent-loop)

### Task 1.2: Orchestrate multi-agent systems with coordinator-subagent patterns

This task is about the coordinator: how it splits work, which subagents it calls, how it checks coverage, and why every message goes through it. Scenario 3 (a coordinator delegating to search, document analysis, synthesis and report subagents) is its natural home.

**Know**

- **Hub-and-spoke (1.2-K1).** The coordinator manages all communication between subagents, all error handling and all information routing. In this model subagents do not talk to each other; each result returns to the coordinator, as in Claude Code, where "Subagents report results back to the conversation that spawned them" ([Agents](https://code.claude.com/docs/en/agents)). One current-product nuance (as of September 2026): the Claude Code docs add that subagents Claude named when it spawned them can also message each other. Exam items assume the guide's hub-and-spoke routing.

```text
                    +---------------+
   user request --> |  coordinator  | --> final answer
                    +---------------+
                   /    |       |    \
             search  analysis  synthesis  report
          (each subagent reports only to the coordinator)
```

- **Anthropic's name for the pattern.** [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) calls it orchestrator-workers: "In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results." Anthropic's own Research system uses it, with a lead agent delegating to specialized subagents that run in parallel.

- **Isolated context (1.2-K2).** Each subagent starts with a fresh context window and does not see the coordinator's conversation history. The one current-product exception, a fork, and what isolation means for delegation prompts are both covered in Task 1.3.

- **The coordinator's four jobs (1.2-K3):** decompose the task, delegate, aggregate results, and decide which subagents a query actually needs. Anthropic's Research system scaled effort to the query in its prompts: "Simple fact-finding requires just 1 agent with 3-10 tool calls, direct comparisons might need 2-4 subagents with 10-15 calls each, and complex research might use more than 10 subagents with clearly divided responsibilities." ([multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)). Early versions got this wrong and spawned 50 subagents for simple queries.

- **Why selection matters: cost.** In the data from Anthropic's [research post](https://www.anthropic.com/engineering/multi-agent-research-system), agents typically used about 4 times the tokens of chat interactions and multi-agent systems about 15 times. A January 2026 [Claude blog post](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them) uses a different baseline: multi-agent implementations typically use 3 to 10 times the tokens of a single-agent approach for equivalent tasks. It also reports teams that spent months building multi-agent architectures before finding that improved prompting on a single agent achieved equivalent results.

- **Narrow decomposition (1.2-K4).** If the coordinator splits a broad topic into subtasks that cover only part of it, every subagent can succeed and the report is still incomplete. The [research post](https://www.anthropic.com/engineering/multi-agent-research-system) describes a related coordinator failure, vague delegation: "Without detailed task descriptions, agents duplicate work, leave gaps, or fail to find necessary information." Question 7 is the narrow kind: its subtasks are specific (digital art, graphic design, photography) but cover only visual arts. The fix is still at the coordinator: widen the decomposition to cover the whole topic.

- **Partitioning scope (1.2-S2).** Give each subagent distinct subtopics or source types. The [research post](https://www.anthropic.com/engineering/multi-agent-research-system)'s delegation rule: "Each subagent needs an objective, an output format, guidance on the tools and sources to use, and clear task boundaries." Anthropic's open-source lead-agent prompt adds: "Define extremely clear, crisp, and understandable boundaries between sub-topics to prevent overlap." ([research_lead_agent.md](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_lead_agent.md)).

- **Iterative refinement (1.2-S3).** In Anthropic's Research system the lead agent synthesizes subagent results, decides whether more research is needed, creates more subagents or refines its strategy if so, and exits the loop once it has gathered sufficient information. The cookbook lead prompt tells the lead to find gaps in the collective research and deploy new subagents to fill them, and to stop at diminishing returns. The loop 1.2-S3 describes, with 1.2-S2's partitioning as step 1:

```text
1. Decompose into distinct subtopics; delegate to search and analysis subagents.
2. Invoke synthesis on everything returned.
3. Evaluate the synthesis for coverage gaps against the research goal.
4. Gaps found: re-delegate targeted queries, then re-invoke synthesis (back to 3).
5. Coverage sufficient: finish.
```

- **Routing through the coordinator (1.2-S4).** The guide's reasons are observability, consistent error handling and controlled information flow. The Claude Code and Agent SDK subagent mechanics fit this: subagents report back to the conversation that spawned them, only a subagent's final response returns to the parent as a tool result, messages produced inside a subagent carry a `parent_tool_use_id`, and `SubagentStart` / `SubagentStop` hooks let you track spawning and aggregate results. The contrasting design is Claude Code's agent teams, where teammates message each other directly and self-coordinate; agent teams are experimental and disabled by default (as of September 2026).

- **Timing (as of September 2026).** In the Agent SDK and in `-p` runs, subagents run in the background by default, and Claude sets `run_in_background: false` when it needs a result before continuing. In interactive Claude Code, where fork mode is on by default, Claude Code runs subagents in the background and Claude can't ask for the foreground. Anthropic's June 2025 research post described lead agents that waited for each set of subagents to finish, and named that synchronous execution as a bottleneck. The guide's parallel-spawning skill (1.3-S3) still holds; see Task 1.3.

**Decide**

- If a query is simple, choose a coordinator that analyzes it and invokes only the subagents it needs; not routing every request through the full pipeline, because the guide's skill is dynamic selection and multi-agent runs cost several times the tokens.
- If reports miss whole areas while every subagent reports success, examine the coordinator's decomposition first; not the search, analysis or synthesis agents, because they worked correctly within the scope they were given.
- If subagents duplicate each other's work, give each one a detailed task description with distinct subtopics or source types and explicit boundaries; not more subagents, because overlap comes from vague delegation.
- If the synthesis has gaps, have the coordinator re-delegate targeted queries and re-run synthesis until coverage is sufficient; not accept the first draft, because a single pass cannot see what it never retrieved.
- If two subagents need each other's output, pass it through the coordinator; not direct subagent-to-subagent calls, because the hub is where logging, error handling and information control happen.

**Traps**

- Blaming a downstream agent (synthesis lacks gap detection, search queries too narrow, analysis filters too strict) when the coordinator's own subtask list left topics out. Question 7's rationale rejects exactly these.
- Running every subagent on every request, presented as thoroughness.
- Peer-to-peer messaging between subagents presented as faster; it bypasses the coordinator's observability and error handling.
- Giving a subagent another role's whole tool set so it can skip the coordinator (Question 9, option C: all web search tools for the synthesis agent). Question 9's rationale calls that over-provisioning; it and [Domain 2](#domain-2-tool-design-mcp-integration) (Task 2.3) prefer a scoped cross-role tool for the common case and coordinator routing for the rest.
- For what a failing subagent should send back to the coordinator, see Task 5.3 in [Domain 5](#domain-5-context-management-reliability) and Question 8.

**Go deeper:** [Multi-agent orchestration](knowledge/agents-and-agent-sdk.md#multi-agent-orchestration)

### Task 1.3: Configure subagent invocation, context passing, and spawning

This task is the mechanics under 1.2: the tool that spawns a subagent, what the subagent receives, how it is configured, and how to spawn several at once.

**Know**

- **The spawning tool (1.3-K1).** In the guide's words, "The Task tool as the mechanism for spawning subagents, and the requirement that allowedTools must include "Task" for a coordinator to invoke subagents" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). In the Agent SDK you define subagents with the `agents` option, and Claude invokes them through the spawning tool.

!!! warning "Exam guide vs current docs: Task is now called Agent"

    The guide (July 2026) calls the tool `Task` and says `allowedTools` must include `"Task"`. The Claude Code docs say: "In version 2.1.63, the Task tool was renamed to Agent. Existing `Task(...)` references in settings and agent definitions still work as aliases." ([Subagents](https://code.claude.com/docs/en/sub-agents)). The Agent SDK examples now list `"Agent"` in `allowed_tools`, and the tool still shows as `"Task"` in the `system:init` tools list, so code that detects subagent calls should match both names.

    A second nuance: current SDK docs describe `allowed_tools` as auto-approval, not a restriction ("Tools not listed are still available, and calls to them that need approval fall through to the permission mode and `canUseTool`.", [Agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop)), and list `Agent` among tools that do not ask before running in `default` mode, so it runs whether or not it is listed. To stop delegation entirely, the Claude Code docs say to deny the `Agent` tool itself with `permissions.deny`.

    For the exam, use the guide's terms: the coordinator spawns subagents with the Task tool, and its `allowedTools` must include `"Task"`.

- **AgentDefinition (1.3-K3).** Each subagent type has a description, a system prompt and a tool restriction:

| Field | Required | What it does |
|---|---|---|
| `description` | Yes | Natural-language description of when to use this agent; Claude's automatic delegation is driven by it |
| `prompt` | Yes | The subagent's system prompt: its role and behavior |
| `tools` | No | Allowed tool names; if omitted, the subagent inherits every tool available to subagents. A tool left out is simply absent, with no prompt or error |
| `model` | No | A model alias such as `fable`, `opus`, `sonnet` or `haiku`, `inherit` (use the main model), or a full model ID. The Python reference's alias list omits `fable` |
| Others | No | `disallowedTools`, `skills`, `memory`, `mcpServers`, `initialPrompt`, `maxTurns`, `background`, `effort`, `permissionMode` (plus `omitClaudeMd` in TypeScript only) |

The Python `AgentDefinition` is a dataclass that keeps camelCase field names such as `disallowedTools` and `maxTurns`; passing snake_case keywords raises a `TypeError`. Here is a coordinator with two differently restricted subagents, from the [Agent SDK subagents page](https://code.claude.com/docs/en/agent-sdk/subagents), with the two prompts shortened:

```python
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AgentDefinition

async def main():
    async for message in query(
        prompt="Review the authentication module for security issues",
        options=ClaudeAgentOptions(
            # Auto-approve these tools
            allowed_tools=["Read", "Grep", "Glob", "Agent"],
            agents={
                "code-reviewer": AgentDefinition(
                    # description tells Claude when to use this subagent
                    description="Expert code review specialist. Use for quality, security, and maintainability reviews.",
                    # prompt defines the subagent's behavior and expertise
                    prompt="You are a code review specialist with expertise in security, performance, and best practices. ...",
                    # tools restricts what the subagent can do (read-only here)
                    tools=["Read", "Grep", "Glob"],
                    # model overrides the default model for this subagent
                    model="sonnet",
                ),
                "test-runner": AgentDefinition(
                    description="Runs and analyzes test suites. Use for test execution and coverage analysis.",
                    prompt="You are a test execution specialist. Run tests and provide clear analysis of results. ...",
                    # Bash access lets this subagent run test commands
                    tools=["Bash", "Read", "Grep"],
                ),
            },
        ),
    ):
        if hasattr(message, "result"):
            print(message.result)

asyncio.run(main())
```

- **Context must be passed explicitly (1.3-K2).** The SDK docs state the rule in one sentence: "The only content you pass from parent to subagent is the Agent tool's prompt string, so include any file paths, error messages, or decisions the subagent needs directly in that prompt." ([Agent SDK subagents](https://code.claude.com/docs/en/agent-sdk/subagents)). A non-fork subagent receives its own system prompt plus the Agent tool's prompt, the project CLAUDE.md (unless the agent sets `omitClaudeMd`, a TypeScript-only field) and its tool definitions; it does not receive the parent's conversation history, tool results or system prompt. Each invocation creates a new instance, so a later call starts without the earlier one's context unless you pass it in or explicitly resume that subagent. If a rule must reach the subagent, restate it in the delegation prompt.

- **The one exception is a fork.** "A fork is a subagent that inherits the entire conversation so far instead of starting fresh." (Claude Code [Subagents](https://code.claude.com/docs/en/sub-agents)). Claude starts one by requesting the `fork` subagent type through the Agent tool, which fork mode must allow. As of September 2026, fork mode is on by default in interactive Claude Code sessions and off by default in `-p` runs and the Agent SDK. Every other subagent still starts fresh, so exam answers about SDK coordinators should assume no inheritance.

- **Fork-based session management (1.3-K4).** A different fork: branching a whole session from a shared analysis baseline so two approaches can diverge. It is covered with sessions in Task 1.7.

- **Pass complete findings (1.3-S1).** The synthesis subagent needs the web search results and document analysis outputs themselves in its prompt, not a reference to earlier findings the subagent never saw. Two related facts: the parent receives a subagent's final message as the tool result but may summarize it in its own response, so the SDK docs say to instruct verbatim preservation in the main prompt when the subagent's exact output must reach the user; and for large outputs, Anthropic's [research post](https://www.anthropic.com/engineering/multi-agent-research-system) recommends having subagents store their work in external systems and pass lightweight references back to the coordinator, which avoids copying large outputs through conversation history.

- **Separate content from metadata (1.3-S2).** Keep source URLs, document names and page numbers in their own fields so attribution survives each handoff. An illustrative shape (not an official schema) built from the fields the guide names in 1.3-S2 and Exercise 4, step 3:

```json
{
  "findings": [
    {
      "claim": "...",
      "evidence_excerpt": "...",
      "source_url": "https://example.com/report",
      "document_name": "...",
      "page_number": 12,
      "publication_date": "..."
    }
  ]
}
```

Provenance through synthesis, including why such a record must carry its own source fields when it is produced as structured output, is [Task 5.6](#task-56-preserve-information-provenance-and-handle-uncertainty-in-multi-source-synthesis).

- **Parallel spawning (1.3-S3).** By default Claude may call multiple tools in a single response, so a coordinator spawns parallel subagents by emitting several Task calls in one turn. The payoff, per the [Agent SDK subagents page](https://code.claude.com/docs/en/agent-sdk/subagents): "multiple subagents can run concurrently, so independent subtasks finish in the time of the slowest one rather than the sum of all of them." Anthropic's lead-agent prompt makes it mandatory: "You MUST use parallel tool calls for creating multiple subagents (typically running 3 subagents at the same time) at the start of the research, unless it is a straightforward query." ([research_lead_agent.md](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_lead_agent.md)). In the Research system, spinning up 3 to 5 subagents in parallel (with subagents also using tools in parallel) cut research time by up to 90% for complex queries.

- **Goals, not procedures (1.3-S4).** Anthropic's [Research system](https://www.anthropic.com/engineering/multi-agent-research-system) worked the same way: "Our prompting strategy focuses on instilling good heuristics rather than rigid rules." The cookbook lead prompt's delegation checklist covers objectives, expected output format, background context, key questions, sources and scope boundaries. Explaining why an instruction exists also helps: the prompting docs say "Claude is smart enough to generalize from the explanation." ([Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)).

**Decide**

- If a subagent needs earlier findings, put them, complete, in its prompt; not reliance on the coordinator's history or on memory of an earlier invocation, because the Agent tool's prompt string is the only content a non-fork subagent gets from its parent.
- If attribution must survive the handoff, pass structured records with source metadata in separate fields; not a prose summary, because summaries drop the claim-to-source link.
- If subtasks are independent, emit several Task calls in a single coordinator response; not one per turn, because in the guide's model calls spread across turns run one after another (Exercise 4, step 2 asks you to measure the latency gain over sequential execution). Current SDK subagents run in the background by default, which (our reading) narrows that gap in practice; answer in the guide's terms.
- If subagents must adapt to what they discover, give the coordinator (and each subagent) research goals and quality criteria; not step-by-step procedures, because fixed steps cannot react to intermediate findings.
- If a subagent must not write or execute, set `tools` in its AgentDefinition; not a prompt line asking it to stay read-only, because an omitted `tools` field inherits every tool.

**Traps**

- Any option that assumes subagents automatically inherit coordinator context or share memory between invocations.
- A coordinator whose `allowedTools` leaves out `"Task"` (in the guide's model, it cannot spawn subagents).
- Subagents described as parallel but launched across separate coordinator turns (1.3-S3 names this as the wrong way).
- A coordinator prompt that scripts every search step instead of stating the goal and the bar for a sufficient result.
- Passing only the synthesis-ready conclusions and losing the URLs, document names and page numbers that downstream agents need to cite.

**Go deeper:** [Subagents in the SDK](knowledge/agents-and-agent-sdk.md#subagents-in-the-sdk)

### Task 1.4: Implement multi-step workflows with enforcement and handoff patterns

The customer support scenario lives here: a refund must never happen before identity is verified, a single message may raise several problems, and when the agent escalates, the human needs a summary that stands on its own.

**Know**

- **Two ways to order steps (1.4-K1).** Programmatic enforcement (hooks, prerequisite gates) makes the order a property of your code. Prompt-based guidance asks the model to follow it. The [Claude Code glossary](https://code.claude.com/docs/en/glossary): "Hooks are deterministic: they fire at fixed lifecycle points rather than at the model's discretion." The best-practices page: "Unlike CLAUDE.md instructions which are advisory, hooks are deterministic and guarantee the action happens." ([Claude Code best practices](https://code.claude.com/docs/en/best-practices)).

- **Why prompts are not enough for critical rules (1.4-K2).** When compliance must be deterministic, such as identity verification before a financial operation, the guide says "prompt instructions alone have a non-zero failure rate" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Anthropic's commerce-agent guide applies the same principle to approvals: every write the agent proposes is a staged change the operator approves outside the chat, and "An approval typed in chat approves nothing." ([Commerce agents](https://platform.claude.com/docs/en/about-claude/use-case-guides/commerce-agents)).

- **A prerequisite gate (1.4-S1).** Block `process_refund` until `get_customer` has returned a verified customer ID. In the Agent SDK, one way is a `PostToolUse` hook that records success (it fires after a tool call succeeds) and a `PreToolUse` hook that denies the refund until that record exists. The deny reason is shown to Claude, so it knows to call `get_customer` first. Hook inputs carry `session_id`, and MCP tools are matched as `mcp__<server>__<action>`. Illustrative code, adapted from the [Agent SDK hooks page](https://code.claude.com/docs/en/agent-sdk/hooks); the server and tool names are examples. This sketch treats any successful `get_customer` call as verification; a production gate would also check the call's `tool_response` for the verified customer ID (our note).

```python
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

verified_sessions: set[str] = set()

async def record_verification(input_data, tool_use_id, context):
    # PostToolUse only fires after the tool completed successfully
    verified_sessions.add(input_data["session_id"])
    return {}

async def require_verification(input_data, tool_use_id, context):
    if input_data["session_id"] not in verified_sessions:
        return {
            "hookSpecificOutput": {
                "hookEventName": input_data["hook_event_name"],
                "permissionDecision": "deny",
                "permissionDecisionReason": "Call get_customer and verify the customer before issuing a refund.",
            }
        }
    return {}

options = ClaudeAgentOptions(
    hooks={
        "PostToolUse": [HookMatcher(matcher="mcp__crm__get_customer", hooks=[record_verification])],
        "PreToolUse": [HookMatcher(matcher="mcp__billing__process_refund", hooks=[require_verification])],
    }
)
```

Hooks run before every other permission step, and a hook deny applies even in `bypassPermissions` mode, so a permissive setting elsewhere cannot let the refund slip through (our reading of the evaluation order). Question 1's correct option gates `lookup_order` as well; a matcher written as a `|`-separated list of exact tool names covers several tools.

- **Multi-concern requests (1.4-S2).** The guide's pattern: split the message into distinct items, investigate each in parallel using shared context, then synthesize one resolution. Anthropic's ticket-routing guide names a related failure: "When customers present multiple issues in a single interaction, Claude may have difficulty identifying the primary concern." ([Ticket routing](https://platform.claude.com/docs/en/about-claude/use-case-guides/ticket-routing)). Its own fix is to clarify how intents are prioritized so Claude can rank them and find the primary concern. The exam guide's pattern goes further: it investigates every item and synthesizes one resolution. Independent read-only lookups are usually safe to run in parallel, calls with side effects or ordering needs may be better run in sequence, and all the `tool_result` blocks for one turn go back together in the next user message. Exercise 1, step 5 asks you to test exactly this.

- **Structured handoff (1.4-K3, 1.4-S3).** When the agent escalates mid-process, the human agent cannot see the conversation transcript. The guide's handoff protocol includes customer details, root cause analysis and recommended actions; its list for the summary itself is customer ID, root cause, refund amount and recommended action. The guide names the fields but gives no schema. The nearest first-party structure is Anthropic's customer-escalation skill (written for human support staff who use Claude, not for an autonomous agent), whose brief has sections for Impact, Issue Description, What's Been Tried, Reproduction Steps, Customer Communication, What's Needed and Supporting Context. Its guidance: quantify impact, and state the exact ask (investigate, fix, or decide).

An illustrative `escalate_to_human` tool that forces a self-contained handoff (not an official Anthropic example; field names mirror the guide's list and the brief's sections):

```json
{
  "name": "escalate_to_human",
  "description": "Hand the case to a human support agent who cannot see this conversation. Call it immediately when the customer explicitly asks for a human, when the request needs a policy exception or the policy does not cover it, or when you cannot make progress. Do not call it only because the customer sounds upset. The summary must stand on its own.",
  "input_schema": {
    "type": "object",
    "properties": {
      "customer_id": {"type": "string", "description": "Verified customer ID returned by get_customer. Never a guess."},
      "reason": {"type": "string", "enum": ["customer_requested_human", "policy_exception_or_gap", "no_progress", "other"]},
      "reason_detail": {"type": "string"},
      "issue_summary": {"type": "string", "description": "What the customer needs, in 2-4 sentences."},
      "root_cause": {"type": "string", "description": "What you found, or 'unknown'."},
      "what_was_tried": {"type": "array", "items": {"type": "string"}},
      "amount_at_issue": {"type": ["number", "null"]},
      "recommended_action": {"type": "string"}
    },
    "required": ["customer_id", "reason", "issue_summary", "root_cause", "what_was_tried", "recommended_action"]
  }
}
```

When to escalate at all (explicit requests, policy gaps, no progress) is Task 5.2 in [Domain 5](#domain-5-context-management-reliability). If a human decision may take a while, the Agent SDK lets a `PreToolUse` hook return `defer` so the process can exit and resume later from the persisted session. Claude Code honors `defer` only in non-interactive `-p` mode (the hooks reference names an Agent SDK app as one such integration), and only when Claude makes a single tool call in the turn.

**Decide**

- If a step must always happen before another (identity before refund), enforce it in code with a prerequisite gate or hook; not a firmer system prompt or more few-shot examples, because prompt compliance is probabilistic and the errors cost money.
- If an ordering is merely good practice with no financial or safety consequence, a prompt instruction is proportionate; not a hook by default, because the guide ties programmatic enforcement to cases where deterministic compliance is required.
- If one message raises several issues, decompose it into items, investigate independent items in parallel, and answer once; not only the first issue, and not separate conversations, because the customer asked about all of them.
- If the agent escalates, send a structured summary with customer ID, root cause, amount and recommended action; not a pointer to the conversation or a bare request for help, because the human cannot read the transcript.

**Traps**

- Strengthening the system prompt, or adding few-shot examples of the right order, for a rule with financial consequences. Question 1's rationale rejects both as relying on probabilistic compliance.
- A routing classifier that enables a subset of tools per request: it changes which tools are available, not the order they are called in (Question 1, option D).
- A handoff that points the human to a transcript they cannot see, or omits the amount and the recommended next step.
- Treating a chat message that says an action is approved as the approval itself. Anthropic's [commerce-agent guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/commerce-agents) puts approval outside the conversation: "An approval typed in chat approves nothing."

**Go deeper:** [Permissions and enforcement](knowledge/agents-and-agent-sdk.md#permissions-and-enforcement)

### Task 1.5: Apply Agent SDK hooks for tool call interception and data normalization

Two hook jobs are tested: rewriting what a tool returns before Claude reads it, and stopping an outgoing tool call that breaks a business rule. The third bullet is the reason to use hooks at all.

**Know**

- **What an SDK hook is.** "Hooks are callback functions that run your code in response to agent events, like a tool being called, a session starting, or execution stopping." ([Agent SDK hooks](https://code.claude.com/docs/en/agent-sdk/hooks)). They run in your application process, not in the agent's context window, so they don't consume context. Both the Python and TypeScript SDKs support the two events this task needs (TypeScript supports more events overall).

- **The two events:**

| Event | When it fires | Exam use | Output that matters |
|---|---|---|---|
| `PreToolUse` | On a tool call request, before the tool runs ("can block or modify") | Tool call interception: block or redirect a policy-violating call | `hookSpecificOutput.permissionDecision` (`"allow"`, `"deny"`, `"ask"`, `"defer"`), `permissionDecisionReason`, `updatedInput` |
| `PostToolUse` | After a tool call succeeds | Normalize or trim a result before the model processes it | `hookSpecificOutput.updatedToolOutput` (replace what Claude sees), `additionalContext` (add a note next to the result) |

- **Matchers and inputs.** A matcher filters on the tool name only ([Agent SDK hooks](https://code.claude.com/docs/en/agent-sdk/hooks)): "Matchers only match tool names, not file paths or other arguments." MCP tools are named `mcp__<server>__<action>`, and the hooks page's regex example `^mcp__` matches every MCP tool. Callbacks receive `(input_data, tool_use_id, context)`; every input carries `session_id`, `cwd` and `hook_event_name`; a `PreToolUse` input adds `tool_name` and `tool_input`, and a `PostToolUse` input carries both `tool_input` and `tool_response`. Return `{}` to let the call through unchanged.

- **Normalization with PostToolUse (1.5-K1, 1.5-S1).** Different MCP tools return data in inconsistent formats: dates as Unix timestamps from one tool and ISO 8601 strings from another, and statuses as numeric codes from a third. A `PostToolUse` hook can convert them before the agent reads them ([Agent SDK hooks](https://code.claude.com/docs/en/agent-sdk/hooks)): "To replace the tool's output before Claude sees it, set `updatedToolOutput`, which works for any tool in both SDKs." Replacing a built-in tool's output must match that tool's output shape; MCP output is not schema-validated when replaced. Illustrative code, not from the docs (the `normalize_timestamps` helper is your own function):

```python
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

async def normalize_mcp_output(input_data, tool_use_id, context):
    normalized = normalize_timestamps(input_data["tool_response"])  # your code: Unix epoch / ISO 8601 -> one format
    return {
        "hookSpecificOutput": {
            "hookEventName": input_data["hook_event_name"],
            "updatedToolOutput": normalized,
        }
    }

options = ClaudeAgentOptions(
    hooks={"PostToolUse": [HookMatcher(matcher="^mcp__", hooks=[normalize_mcp_output])]}
)
```

- **Interception with PreToolUse (1.5-K2, 1.5-S2).** The guide's example blocks refunds above &#36;500 and redirects to human escalation. The deny reason reaches Claude ("`permissionDecisionReason` tells the model why, so it avoids retrying."), which is how the hook redirects the agent to the escalation workflow. A separate `systemMessage` goes to the user: "The `systemMessage` field shows a message to the user, not the model." The amount check happens inside the callback because matchers cannot see arguments. Illustrative code adapted from the [Agent SDK hooks page](https://code.claude.com/docs/en/agent-sdk/hooks):

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

- **Why the block holds.** When several hooks or rules apply, `deny` beats `defer`, which beats `ask`, which beats `allow`, so one deny is enough. A hook deny applies even in `bypassPermissions` mode. An Agent SDK callback that times out on `PreToolUse` blocks the tool call (a timed-out Claude Code command, HTTP or MCP-tool hook does not), so a slow policy callback fails closed.

- **Deterministic versus probabilistic (1.5-K3, 1.5-S3).** The hooks guide describes hooks as the mechanism "which gives you deterministic control: certain actions always happen rather than relying on the LLM to choose to run them." ([Hooks guide](https://code.claude.com/docs/en/hooks-guide)). A prompt instruction is followed most of the time; a hook runs every time its event fires.

!!! info "Current names for the guide's hook terms (as of September 2026)"

    The guide's "tool call interception" is the `PreToolUse` event, which blocks with `permissionDecision: "deny"`. The top-level `decision` / `reason` fields for `PreToolUse` are deprecated (old `"approve"` / `"block"` map to `"allow"` / `"deny"`). On `PostToolUse`, `updatedToolOutput` replaces any tool's output, and the older MCP-only `updatedMCPToolOutput` is deprecated. The guide's objectives and answers are unaffected.

**Decide**

- If different tools return data in inconsistent formats, normalize it in a `PostToolUse` hook; not a prompt asking Claude to convert formats as it reads, because the hook applies to every result before the model processes it.
- If a business rule must hold on every call (a refund cap, a blocked action), intercept the outgoing call in `PreToolUse`, deny it, and use the deny reason to redirect to escalation; not a system-prompt rule, because prompt compliance is probabilistic.
- If you must change a call's arguments or redact what goes out, use `PreToolUse`; if you must change or redact what comes back, use `PostToolUse`; not the other event, because `PreToolUse` runs before the tool and can return `updatedInput`, while by the time `PostToolUse` fires the tool has already run and `updatedToolOutput` only changes what Claude sees.

**Traps**

- **Blocking with PostToolUse.** By then the refund has happened: "`updatedToolOutput` only changes what Claude sees." ([Hooks reference](https://code.claude.com/docs/en/hooks)). Blocking belongs in `PreToolUse`.
- **Putting the threshold in the matcher.** Matchers see tool names only; check `amount` inside the callback.
- **Explaining the block only in `systemMessage`.** That text goes to the user. Claude learns why from `permissionDecisionReason` on a deny.
- **A firmer system-prompt instruction** offered as the guaranteed fix. For guaranteed compliance, the guide wants the hook.

**Go deeper:** [Hooks in the SDK](knowledge/agents-and-agent-sdk.md#hooks-in-the-sdk)

### Task 1.6: Design task decomposition strategies for complex workflows

The decision here is between a pipeline you fix in advance and a plan that changes as the work reveals things. The guide gives one example of each: multi-aspect code review (fixed) and adding tests to a legacy codebase (adaptive).

**Know**

- **Fixed pipelines: prompt chaining (1.6-K1).** "Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one." It suits tasks that split cleanly into fixed subtasks, and "The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task." ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)). Current prompting guidance adds that, with adaptive thinking and subagent orchestration, Claude handles most multistep reasoning internally, but "Explicit prompt chaining (breaking a task into sequential API calls) is still useful when you need to inspect intermediate outputs or enforce a specific pipeline structure." ([Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)).

- **Dynamic decomposition (1.6-K1, 1.6-K3).** In orchestrator-workers ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)) the "subtasks aren't pre-defined, but determined by the orchestrator based on the specific input." Agents fit open-ended problems where the number of steps cannot be predicted, and they need ground truth from the environment (tool results, code execution) at each step to judge progress. That is the value of an adaptive investigation plan: each step's findings generate the next subtasks. Anthropic's [lead-agent prompt](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_lead_agent.md) says it directly: "Update the search plan and your subagent delegation strategy based on findings from tasks."

- **Per-file passes plus an integration pass (1.6-K2, 1.6-S2).** A large review becomes a chain: analyze each file on its own for local issues, then run a separate pass for cross-file data flow. Anthropic's reasoning for splitting, from [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): "For complex tasks with multiple considerations, LLMs generally perform better when each consideration is handled by a separate LLM call, allowing focused attention on each specific aspect." The guide's name for the failure it avoids is attention dilution; Question 12 is the worked case, and Task 4.6 in [Domain 4](#domain-4-prompt-engineering-structured-output) covers the review architecture in more depth.

- **Choosing the pattern (1.6-S1):**

| Workflow shape | Choose | Why |
|---|---|---|
| Predictable, same aspects every time (a multi-aspect review checklist; write an outline, check it, then write the document) | Prompt chaining | Each step is easier and its output can be inspected |
| Large multi-file review | Per-file local passes, then one cross-file integration pass | Consistent depth per file; cross-file issues get their own pass |
| Open-ended investigation (legacy test coverage, an unfamiliar codebase) | Dynamic decomposition with an adaptive plan | The next subtasks depend on what each step finds |

- **The legacy-codebase example (1.6-S3).** The guide's order for an open-ended request to add tests across a legacy codebase:

```text
1. Map the structure first.
2. Identify the high-impact areas.
3. Create a prioritized plan.
4. Let the plan adapt as dependencies are discovered.
```

The Claude Code tooling for this is covered in [Domain 3](#domain-3-claude-code-configuration-workflows): plan mode for exploring before editing (Task 3.4) and the four-phase Explore, Plan, Implement, Commit workflow from the Claude Code best-practices page. For long runs, the prompting docs recommend structured formats such as JSON for state like test results, and Claude Code re-injects the plan file after each compaction.

**Decide**

- If the steps are known in advance and the same for every input, choose prompt chaining; not a dynamic orchestrator, because Anthropic's advice is that "you should consider adding complexity only when it demonstrably improves outcomes" ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)).
- If what to do next depends on what you find, choose dynamic decomposition with a plan that adapts; not a full step list written before you have looked, because in an adaptive plan each step's findings generate the next subtasks.
- If one pass over many files gives uneven depth and contradictory findings, split it into per-file passes plus a cross-file integration pass; not a larger context window, and not consensus voting across repeated full passes, because Question 12's rationale says larger context windows don't solve attention quality and consensus suppresses real bugs that are caught only intermittently.
- If the task is adding tests across a legacy codebase, map the structure and rank high-impact areas before writing tests; not tests written file by file in whatever order the files are listed, because the guide's order is structure first, then high-impact areas, then a prioritized plan that adapts as dependencies are discovered.

**Traps**

- For a large review, the alternatives Question 12 rejects (a bigger context window, consensus across repeated full passes, smaller PRs) are listed with the rationale's reasons under [Task 4.6](#task-46-design-multi-instance-and-multi-pass-review-architectures).
- A dynamic multi-agent orchestrator for a fixed checklist review, or a fixed script for an investigation whose shape is unknown.

**Go deeper:** [Workflow patterns](knowledge/agents-and-agent-sdk.md#workflow-patterns)

### Task 1.7: Manage session state, resumption, and forking

Sessions are how an investigation survives across days and how two approaches can grow from one analysis. The harder bullets are about staleness: what to tell a resumed session, and when not to resume at all.

**Know**

- **What a session holds.** "A session is the conversation history the SDK accumulates while your agent works." ([Agent SDK sessions](https://code.claude.com/docs/en/agent-sdk/sessions)). It is written to disk automatically and holds the conversation, not the filesystem. Claude Code stores sessions under `~/.claude/projects/<encoded-cwd>/*.jsonl`. A resumed session restores the full history, including tool calls and their results.

- **The commands and options:**

| Need | Claude Code CLI | Agent SDK (Python) | Agent SDK (TypeScript) |
|---|---|---|---|
| Name a session | `claude -n auth-refactor` at startup, `/rename` inside | `rename_session()` utility (sets a title) | `renameSession()` utility |
| Resume by name | `claude --resume auth-refactor` | By ID instead: `resume` takes a specific session ID (look it up with `list_sessions()`) | By ID instead: `resume` takes a session ID (look it up with `listSessions()`) |
| Resume by ID | `claude --resume <session-id>` | `resume=session_id` | `resume: sessionId` |
| Continue the most recent | `claude --continue` | `continue_conversation=True` | `continue: true` |
| Fork into a new session | `claude --resume abc123 --fork-session`, or `/branch` inside a session | `resume=session_id` plus `fork_session=True` | `resume: sessionId` plus `forkSession: true` |

- **Named resumption (1.7-K1, 1.7-S1).** `claude --resume <name>` "Resumes the named session directly" ([Sessions](https://code.claude.com/docs/en/sessions)); an ambiguous name opens the session picker with the name pre-filled. The best-practices page recommends descriptive names such as `oauth-migration`. Sessions created with `claude -p` or the Agent SDK are left out of the picker and `claude --continue`, but you can still resume them by session ID.

- **Forking (1.7-K2, 1.7-S2).** A fork is a new session that starts with a copy of the original's history and then diverges; the original stays unchanged and the fork gets its own session ID. Use it to compare, say, two testing strategies or two refactoring approaches from the same codebase analysis. One caution from the [Agent SDK sessions page](https://code.claude.com/docs/en/agent-sdk/sessions): "Forking branches the conversation history, not the filesystem." If a forked agent edits files, those edits are real for every session in that directory. In the SDK, capture the fork's ID from the result message:

```python
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

!!! info "One concept, several names (as of September 2026)"

    The guide's `fork_session` is the Python Agent SDK option. TypeScript spells it `forkSession: true`, the CLI flag is `--fork-session` (used with `--resume` or `--continue`), and inside an interactive session the command is `/branch`. Resuming the same session in two terminals without forking interleaves both conversations into one transcript, which is why comparisons need a fork. Answer exam items with the guide's term.

- **Resumed sessions do not know what changed (1.7-K3, 1.7-S4).** The history still holds the file contents Claude read before your edits. Claude Code's own safety nets only partly cover this: checkpoints normally do not capture "Manual changes you make to files outside of Claude Code and edits from other concurrent sessions" ([Checkpointing](https://code.claude.com/docs/en/checkpointing)), switching git branches changes the files Claude sees but not the conversation history, a `FileChanged` hook's message reaches the user, not Claude, and the Edit tool (Claude Code v2.1.208 or later) notices that a file changed on disk since Claude last read it only when Claude edits that file. Hook context added in earlier turns is replayed on resume rather than re-run, so values such as timestamps can be stale too. The guide's answer is to tell the resumed session which files changed and ask for targeted re-analysis. An illustrative prompt:

```text
Since our last session, src/billing/refund.py and src/billing/limits.py changed
(the refund cap logic moved into limits.py). Re-read those two files and update
your analysis of the refund flow. The rest of your earlier analysis still holds.
```

- **Fresh start with a structured summary (1.7-K4, 1.7-S3).** When much of the earlier tool output is stale, resuming carries those stale results forward. The [SDK sessions page](https://code.claude.com/docs/en/agent-sdk/sessions), discussing sessions that must move between hosts, describes the same pattern: "Capture the results you need (analysis output, decisions, file diffs) as application state and pass them into a fresh session's prompt." It rates that as often more dependable than moving transcript files around. For long tasks that span several context windows, the [prompting docs](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) make a related recommendation: "When a context window is cleared, consider starting with a brand new context window rather than using compaction." There, the new window rebuilds state from files the earlier work left behind, such as `progress.txt`, `tests.json` and the git logs, not from a summary.

Do not confuse either pattern with Claude Code's resume dialog. As of September 2026 it appears on a Pro or Max plan when a session idle for more than about an hour and over 100,000 tokens is resumed, and it is a cost decision: the [Sessions](https://code.claude.com/docs/en/sessions) page calls the choice between its options "a tradeoff between keeping every detail and sending fewer tokens per request". **Resume from summary** runs `/compact` on the same session; resuming as-is keeps every detail available at a per-request cost that grows with the conversation.

**Decide**

- If you will continue the same investigation in a later work session, name it and resume with `--resume <session-name>`; not a new session that re-explores the codebase, because a resumed session restores the full history, including earlier tool calls and results.
- If you want to compare two approaches from one shared analysis, use `fork_session` to create independent branches; not the same session resumed twice, and not two fresh sessions that each redo the analysis, because a fork starts from a copy of the shared history and leaves the original unchanged, while one session resumed in two terminals interleaves both conversations into one transcript.
- If a few files changed and most prior context is still valid, resume and name the changed files for targeted re-analysis; not a full re-exploration, because the rest of the prior analysis still holds and only the named files need re-reading.
- If much of the prior tool output is stale, start a new session and inject a structured summary of the conclusions; not a resume that keeps reasoning over outdated results, because a resumed session carries its old tool results forward and is not told what changed.

**Traps**

- Assuming a resumed session notices that the code changed.
- Resuming a session full of stale tool results when a fresh session with a summary would be more reliable.
- Treating a fork as a sandbox for the filesystem; it branches only the conversation.
- Answers that re-read the entire codebase after a small change instead of pointing the agent at the changed files.

**Go deeper:** [Sessions, resumption and forking](knowledge/agents-and-agent-sdk.md#sessions-resumption-and-forking)

## Domain 2: Tool Design & MCP Integration

**Official weight: 18%**, the approximate proportion of scored items drawn from this domain, which is about 11 of the 60 items (18% of 60 is 10.8, our arithmetic; the guide does not say how many of the 60 are scored, so treat the figure as a rough guide).

Domain 2 covers the surface between Claude and everything it can act on: how a tool is described, how it reports failure, which agent holds which tool, how MCP servers are configured in Claude Code and the Agent SDK, and which built-in tool (Read, Write, Edit, Bash, Grep or Glob) fits which job. The guide lists it as a primary domain in three of the six scenarios: Customer Support Resolution Agent, Multi-Agent Research System, and Developer Productivity with Claude (where it is listed first).

| Task | The decision it trains | Knowledge / skills bullets |
|---|---|---|
| 2.1 Tool interfaces | Write descriptions that separate similar tools; rename, split, and check the system prompt | 4 / 4 |
| 2.2 Structured MCP errors | Return failures the agent can act on: category, retryability, plain-language reason | 4 / 4 |
| 2.3 Tool distribution and `tool_choice` | Give each agent only its role's tools; require or force a tool call when the workflow needs it | 4 / 5 |
| 2.4 MCP integration | Put servers at the right scope, keep secrets out of git, use resources for catalogs, reuse community servers | 4 / 5 |
| 2.5 Built-in tools | Pick Grep, Glob, Read, Write, Edit or Bash for the job, and explore a codebase incrementally | 4 / 5 |

That is 20 knowledge bullets and 23 skill bullets in total (our count from the guide). The CCAR-F guide does not label its samples by domain, so the mapping here is ours. Two official sample questions test this domain directly: Question 2 (two tools with one-line descriptions) and Question 9 (a scoped `verify_fact` tool for a synthesis agent). Question 8 (a web search subagent that times out) belongs with Task 5.3, but its rationale also decides error-design questions under Task 2.2, so it is used below. All three appear with Anthropic's rationales under [Official sample questions](#official-sample-questions).

### Task 2.1: Design effective tool interfaces with clear descriptions and boundaries

The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) states the task as "Design effective tool interfaces with clear descriptions and boundaries". Its bullets cover four ideas: the description is what drives tool selection, what a description must contain, how overlapping descriptions cause misrouting, and how keyword-heavy system prompts can override good descriptions.

**Know**

- **Descriptions drive selection.** Per the guide, the description is the main signal Claude uses to choose among tools, and minimal descriptions make selection among similar tools unreliable. The [tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview) says that, with the default `tool_choice` of `auto`, Claude calls a tool when the request maps to that tool's described capability and the answer is not already in context. For a tool you define, Claude never sees your implementation, only the schema you provide and the result you return. When it chooses a tool, it works from the definition (name, description, input schema and any `input_examples`) together with whatever your system prompt says about tools.

- **What goes in a description.** The guide lists input formats, example queries, edge cases, and boundaries that explain when to use the tool instead of a similar one. The [define-tools page](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools) calls extremely detailed descriptions "by far the most important factor in tool performance" and asks for what the tool does, when it should (and should not) be used, what each parameter means, and caveats or limitations, in at least three to four sentences per tool (more for a complex tool). Anthropic's [tool-writing guidance](https://www.anthropic.com/engineering/writing-tools-for-agents) adds that parameters should be named so they cannot be misread: `user_id`, not `user`.

- **Definition fields in the Messages API:**

| Field | Rule |
|---|---|
| `name` | Must match `^[a-zA-Z0-9_-]{1,128}$`: letters, digits, underscore and hyphen, 1 to 128 characters, no dots |
| `description` | Plain text: what the tool does, when it should be used, how it behaves |
| `input_schema` | A JSON Schema object defining the parameters |
| `input_examples` | Optional array of example inputs; each must validate against `input_schema`, or the request returns a 400 error |

MCP servers declare the same thing with `inputSchema` (camelCase), and the MCP spec's naming guidance (a SHOULD, not a MUST) also allows dots in tool names.

The define-tools page contrasts a good and a poor description of the same tool ([define-tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools)). The good one says what the tool does, when to use it, what data it returns, and what the `ticker` parameter means; the poor one leaves Claude with open questions about the tool's behavior and usage. Both are copied from the docs:

=== "Good description (docs)"

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

=== "Poor description (docs)"

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

- **Overlap causes misrouting.** Near-identical descriptions (the guide's example is `analyze_content` versus `analyze_document`) send requests to the wrong tool. Anthropic's [tool-writing guidance](https://www.anthropic.com/engineering/writing-tools-for-agents) says it directly: "When tools overlap in function or have a vague purpose, agents can get confused about which ones to use."

- **Say when, not only what.** The [troubleshooting page](https://platform.claude.com/docs/en/agents-and-tools/tool-use/troubleshooting-tool-use) fix for a wrong-tool call is "Differentiate tools by WHEN to use them, not only WHAT they do."

- **Rename to remove overlap.** The guide renames `analyze_content` to `extract_web_results` and gives it a web-specific description. Namespacing works the same way: the define-tools page suggests service prefixes such as `github_list_prs` and `slack_send_message` once tools span several services.

- **Split or consolidate.** The guide splits a generic `analyze_document` into `extract_data_points`, `summarize_content` and `verify_claim_against_source`, each with a defined input/output contract. The define-tools page pushes the other way for families of actions: group `create_pr`, `review_pr` and `merge_pr` into one tool with an `action` parameter. Both follow one rule from Anthropic's [tool-writing guidance](https://www.anthropic.com/engineering/writing-tools-for-agents): "Make sure each tool you build has a clear, distinct purpose." Our reconciliation of the two, not a documented rule: split a tool that hides several jobs with different inputs and outputs, or that overlaps another tool; consolidate several tools that are variants of one job. The guide's appendix lists "splitting vs consolidating tools" as a tested topic, so expect to argue either direction from the purpose test.

- **The system prompt competes with descriptions.** When a request includes tools, the API builds one system prompt from the tool definitions, the tool configuration and your own system prompt, so your wording is read alongside every description. The docs say tool triggering can be steered from the system prompt. Anthropic's prompting best practices add that Claude Opus 4.5 and Claude Opus 4.6 are more responsive to the system prompt than earlier models, so prompts written to reduce undertriggering may now cause overtriggering; the fix is to dial back aggressive language. The guide's point (2.1-K4, 2.1-S4) is the keyword version of this: keyword-sensitive instructions can create unintended tool associations and can override well-written descriptions. As our illustration, a system prompt that tells Claude to analyze every document it sees can drag web-result requests toward any tool with `analyze` or `document` in its name. Anthropic's cost guidance, discussing stale prompt text carried over to newer models, adds that the same patterns tend to appear in tool descriptions and skills, which are worth auditing too.

**Decide**

- If Claude confuses two similar tools whose descriptions are one line each, choose expanding both descriptions (input formats, example queries, edge cases, when to use each instead of the other) as the first step; not few-shot routing examples, a keyword routing layer or merging the tools, because the missing context in the descriptions is the root cause (Question 2).
- If two tools overlap in purpose, choose renaming them and rewriting their descriptions so each has one job (the guide's `analyze_content` to `extract_web_results`); not a keyword rule in the system prompt that routes requests between them, because 2.1-K4 warns that keyword-sensitive instructions can create unintended tool associations.
- If one generic tool does several different jobs, choose splitting it into purpose-specific tools with defined input/output contracts; not a longer description of the generic tool, because (our reasoning) each split tool can then carry a precise contract of its own.
- If a well-described tool is still misrouted, choose reviewing the system prompt for keyword-sensitive instructions that might override the descriptions (2.1-S4); not yet another rewrite of the tool.

**Traps**

- The Question 2 rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) rejects three alternatives to better descriptions: few-shot examples "add token overhead without fixing the underlying issue"; a routing layer that pre-selects tools by keyword is "over-engineered and bypasses the LLM's natural language understanding"; merging the two tools is "a valid architectural choice" but more effort than a first step warrants.
- Adding more tools to cover more cases. The [tool-writing guidance](https://www.anthropic.com/engineering/writing-tools-for-agents) warns that too many or overlapping tools distract agents from efficient strategies.

**Go deeper:** [Writing tool descriptions that steer selection](knowledge/tool-use-and-mcp.md#writing-tool-descriptions-that-steer-selection). Exercise 1, step 1, practices this objective with two deliberately similar tools; see [Official preparation exercises](#official-preparation-exercises).

### Task 2.2: Implement structured error responses for MCP tools

The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) states the task as "Implement structured error responses for MCP tools". It tests the `isError` flag, four error types, why a generic error message blocks recovery, retryable versus non-retryable failures, structured error metadata (a category, a retry flag and a readable description), customer-friendly explanations for business-rule refusals, local recovery inside subagents, and the difference between a lookup that failed and a lookup that found nothing.

**Know**

- **Two error channels in MCP.** The [MCP tools spec](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) separates protocol errors (unknown tool, malformed request, server errors), returned as JSON-RPC errors that models are less able to fix, from tool execution errors (API failures, input validation errors, business logic errors), returned inside the tool result with `isError: true`. Execution errors "contain actionable feedback that language models can use to self-correct and retry with adjusted parameters".

- **Why the flag lives in the result.** `isError` defaults to false. The [MCP schema](https://modelcontextprotocol.io/specification/2026-07-28/schema) says errors that originate in the tool should be reported inside the result, because otherwise "the LLM would not be able to see that an error occurred". The tools spec says clients SHOULD pass tool execution errors to the model so it can self-correct (they MAY pass protocol errors). The 2025-11-25 revision clarified that input validation errors belong in this channel rather than in protocol errors, for the same reason.

- **Same idea, different spelling.** MCP uses `isError`. The Messages API `tool_result` block uses the optional `is_error`. Agent SDK custom tool handlers return `isError: true` in TypeScript and `"is_error": True` in Python.

- **The guide's four categories.** Knowledge bullet 2.2-K2 names transient errors (timeouts, service unavailability), validation errors (invalid input), business errors (policy violations) and permission errors. The table adds what each should make the agent do. It reconciles the guide's two lists (see the warning below) and is our summary, not guide wording.

| Category | Guide's examples | Retry the same call? | What the agent should do |
|---|---|---|---|
| Transient | Timeouts, service unavailability | Yes, locally | Retry inside the subagent; propagate only if it cannot be resolved, with partial results and what was attempted |
| Validation | Invalid input | Not with the same input | Correct the parameters from the error message, then call again |
| Business | Policy violations | No (`retriable: false`) | Explain the rule to the customer in plain language |
| Permission | None given | No (our reading) | Report it with a clear description; the guide gives no separate rule |

- **A structured error result.** Illustrative only: the guide's fields carried as JSON text inside the `content` of a result flagged with `isError`, with `business` as a fourth category value (our reconciliation). The names `errorCategory` and `isRetryable` are the guide's, not MCP fields. The shape follows the 2026-07-28 spec, which requires every result to include `resultType` (clients treat a missing `resultType` from older servers as `"complete"`).

```json
{
  "resultType": "complete",
  "content": [
    {
      "type": "text",
      "text": "{\"errorCategory\": \"business\", \"isRetryable\": false, \"description\": \"This order is outside the return window, so a refund cannot be issued. Explain the return policy to the customer and offer escalation to a human agent.\"}"
    }
  ],
  "isError": true
}
```

- **Generic errors block recovery.** A uniform "Operation failed" (the guide's example in 2.2-K3 of the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)) gives the agent nothing to choose between retrying, correcting its input, explaining, or escalating. The [handle-tool-calls page](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls) asks for errors that say what went wrong and what Claude should try next, with the example `Rate limit exceeded. Retry after 60 seconds.`, and Anthropic's tool-writing guidance recommends specific, actionable messages over opaque error codes or tracebacks.

- **Retryability saves wasted attempts.** The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says returning structured metadata "prevents wasted retry attempts" (2.2-K4). Claude also retries on its own: the Messages API docs say that when a tool request is invalid or missing parameters, Claude retries 2 to 3 times with corrections before apologizing to the user. That count is specific to that layer; other Anthropic products document different retry limits.

- **Recover locally, escalate what is left.** Subagents should handle transient failures themselves and pass to the coordinator only the errors they cannot resolve, together with partial results and what they attempted (2.2-S3). The same rule appears as 5.3-S3; local recovery and what the coordinator does with the error are taught under [Task 5.3](#task-53-implement-error-propagation-strategies-across-multi-agent-systems).

- **Failed lookup versus empty result.** An access failure needs a retry decision; a query that ran and matched nothing is a successful answer (2.2-S4). Anthropic's docs and the MCP spec keep the two apart. A web search that succeeds with no matches returns an empty content list, not an error. The MCP resources spec forbids returning empty contents for a resource that does not exist, because an empty array is ambiguous. The [Agent SDK custom tools page](https://code.claude.com/docs/en/agent-sdk/custom-tools) uses `is_error` so a failed call is read as a failure "rather than odd-looking data". One doc seems to blur the line: the search-results page says that when a search fails or returns nothing, the tool should return a plain text block describing the outcome. Our reading is that this governs the content format, so the text should still say which of the two happened. A real case of getting it wrong: before Claude Code v2.1.208, a Grep pattern, glob or file type that ripgrep rejected came back as `No files found` even when the text existed; current versions return an error that includes ripgrep's diagnostic, so Claude can correct the input and search again.

!!! warning "Exam guide vs current docs: error fields"

    The guide's `errorCategory`, `isRetryable` and `retriable` are application-level conventions that the guide recommends; in MCP they would travel inside the tool result's content. The [MCP tools spec](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) (revision 2026-07-28) reports tool execution errors, including input validation and business logic errors, with `isError: true` in the result, and the spec page has no field named `errorCategory`, `isRetryable` or `retriable`. The guide is also inconsistent with itself: 2.2-K2 names four categories, while 2.2-S1 and Exercise 1 list three `errorCategory` values (transient/validation/permission); Exercise 1 still asks the agent to explain business errors; and the appendix's in-scope list names "transient vs business vs permission" and leaves out validation. The retry flag is spelled `isRetryable` in 2.2-S1 and Exercise 1 but `retriable: false` in 2.2-S2. On the exam, answer in the guide's terms: all four categories, a retryable flag, and a human-readable description.

**Decide**

- If a timeout or an unavailable service caused the failure, choose retrying inside the subagent and propagating the error only if it cannot be resolved locally, as structured context (failure type, attempted query, partial results, possible alternatives); not a generic status after exhausted retries, because the Question 8 rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says a generic status "hides valuable context from the coordinator".
- If the input was invalid, choose an `isError` result whose message says what was wrong; not a JSON-RPC protocol error, because the schema's rationale is that otherwise the model would not see that an error occurred, and it cannot self-correct from an error it does not see.
- If a business rule blocked the action, choose a non-retryable flag plus a customer-friendly explanation; not a retry, because (our reasoning) the same rule blocks the same call again.
- If the query ran and matched nothing, choose a successful empty result; not `isError`, because (our reasoning) the coordinator would spend a retry or an escalation on a correct answer. The reverse also holds: never report a timeout as an empty success (Question 8, option C).

**Traps**

- One message for every failure, such as the guide's generic "Operation failed" ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)): the agent cannot tell a retryable timeout from a policy refusal.
- A timeout returned as an empty, successful result, or one timeout that ends the whole workflow: both are anti-patterns in 5.3-K4 and rejected options in Question 8 (see the traps under [Task 5.3](#task-53-implement-error-propagation-strategies-across-multi-agent-systems)).
- Retrying a policy violation, or sending every transient failure straight to the coordinator.
- Treating `errorCategory` or `isRetryable` as fields defined by the MCP spec.

**Go deeper:** [MCP errors and structured results](knowledge/tool-use-and-mcp.md#mcp-errors-and-structured-results). Exercise 1, step 3, practices these error types; see [Official preparation exercises](#official-preparation-exercises).

### Task 2.3: Distribute tools appropriately across agents and configure tool choice

The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) states the task as "Distribute tools appropriately across agents and configure tool choice". It has two halves: who gets which tools (few, role-specific, with narrow cross-role exceptions), and how `tool_choice` makes a tool call optional, required, or specific.

**Know**

- **Fewer tools, better choices.** The guide's example is an agent with 18 tools instead of 4 to 5, which selects less reliably because more tools increase decision complexity. Other Anthropic sources give figures from different contexts: Anthropic's [multi-agent guidance](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them), listing signals that tool specialization would help, says "An agent with too many tools (often 20+) struggles to select the appropriate one", and the [tool search docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool) place the drop in selection accuracy past 30 to 50 tools. Treat the guide's numbers as its example, not as a limit.

- **Specialists misuse foreign tools.** Per the guide, an agent given tools outside its specialization tends to misuse them (the guide's example is a synthesis agent running web searches). The [Agent SDK subagent docs](https://code.claude.com/docs/en/agent-sdk/subagents) list tool restrictions among the four benefits of subagents and suggest role sets: `Read`, `Grep`, `Glob` for read-only analysis; `Bash`, `Read`, `Grep` for test execution; `Read`, `Edit`, `Write`, `Grep`, `Glob` for code modification. Claude Code's built-in Explore subagent is a working example: it gets read-only tools, with Write and Edit denied.

- **Where tool scope is set (as of September 2026):**

| Setting | What it does |
|---|---|
| Agent SDK `AgentDefinition` `tools` | Allowed tool names for that subagent. Omit it and the subagent inherits every tool available to subagents. |
| Claude Code subagent `tools` and `disallowedTools` | `tools` lists what the subagent gets (omit it to inherit every tool available to subagents); `disallowedTools` removes tools. If both are set, `disallowedTools` is applied first, then `tools` is resolved against what remains, so a tool listed in both is removed. |
| Agent SDK `tools` option | Changes which tools are available; `{"type": "preset", "preset": "claude_code"}` gives Claude Code's default tools. |
| `allowedTools` / `allowed_tools` | Auto-approves the listed tools. It "does not restrict Claude to only these tools" ([Agent SDK Python reference](https://code.claude.com/docs/en/agent-sdk/python)); unlisted tools stay available, and calls to them that need approval fall through to the permission mode and `canUseTool`. |
| `disallowedTools` / `disallowed_tools` | A bare tool name such as `Bash` removes that tool from Claude's context. A scoped rule such as `Bash(rm *)` leaves the tool available and denies matching calls in every permission mode, including `bypassPermissions`, for the command as written. |

- **Constrained replacements.** The guide replaces a generic `fetch_url` with a `load_document` tool that validates document URLs. Anthropic's [agent-building guidance](https://www.anthropic.com/engineering/building-effective-agents) states the general principle behind it: "Poka-yoke your tools. Change the arguments so that it is harder to make mistakes." In Anthropic's SWE-bench agent, the model made mistakes with relative file paths once the agent had moved out of the root directory; after the tool was changed to always require absolute file paths, the post reports that the model used it flawlessly.

- **Scoped cross-role tools.** When a role needs another role's capability often, but only in a simple form, give it a narrow tool for that case and keep complex cases on the coordinator path. The guide's example is a `verify_fact` tool for the synthesis agent. In Question 9, 85% of verifications are simple fact-checks and 15% need deeper investigation, and the rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says the scoped tool "applies the principle of least privilege" by covering the common case while complex verifications keep going through the coordinator.

- **The `tool_choice` options.** The guide names three; the [define-tools page](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools) lists four. The [tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview) draws the same line as skill 2.3-S5: prompt wording can make Claude call tools more or less often, but to require a tool call rather than rely on prompting, you set `tool_choice`.

| `tool_choice` | Behavior (docs) | What the guide says |
|---|---|---|
| `{"type": "auto"}` | Claude decides whether to call any tool. Default when tools are provided. | The model may return text instead of calling a tool (4.3-K2) |
| `{"type": "any"}` | Claude must call one of the provided tools, but no particular one | Guarantees a tool call rather than conversational text (2.3-S5); used when several extraction schemas exist and the document type is unknown (4.3-S2) |
| `{"type": "tool", "name": "extract_metadata"}` | Claude must call the named tool | Ensures a specific tool is called first, such as `extract_metadata` before enrichment tools (2.3-S4, 4.3-S3) |
| `{"type": "none"}` | Claude uses no tools. Default when no tools are provided. | Not in the guide |

- **Details that decide edge cases.** With `any` or `tool`, the API prefills the assistant turn, so Claude writes no explanation before the `tool_use` block; for an explanation plus a specific call, use `auto` and ask for it in the user message. `disable_parallel_tool_use` sits inside the `tool_choice` object, not at the top level of the request: with `auto` it means at most one tool call per response, with `any` or `tool` exactly one.

- **Force first, then let go.** Skill 2.3-S4 forces one tool so it is called first and processes later steps in follow-up turns. An illustrative request sequence (tool name from the guide, `tool_choice` forms from the docs; the choice of `auto` for request 2 is ours):

```text
Request 1   "tool_choice": {"type": "tool", "name": "extract_metadata"}
            Claude must call extract_metadata; your code runs it and returns a tool_result
Request 2   "tool_choice": {"type": "auto"}
            Claude chooses which enrichment tools to call, if any
```

A cost note for real systems: with prompt caching, changing `tool_choice` between requests (as request 2 does) invalidates cached message blocks, so message content is reprocessed, while tool definitions and the system prompt stay cached. The guide's out-of-scope list excludes prompt caching implementation details beyond knowing that caching exists, so this is background rather than exam material.

!!! warning "Exam guide vs current docs: tool_choice"

    The guide (July 2026) lists three options: `"auto"`, `"any"` and forced selection. The [define-tools page](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools) lists four, adding `none`, and as of September 2026 some models and settings reject `any` and a forced `tool` with a 400 error. Which ones, and the documented replacement (`auto` with strict tool use, or structured outputs), are in the warning on forced tool choice under [Task 4.3](#task-43-enforce-structured-output-using-tool-use-and-json-schemas). On the exam, answer in the guide's terms: `"any"` guarantees a tool call, and a forced tool guarantees a specific one.

**Decide**

- If a subagent misuses tools outside its role, choose restricting its tool set to the tools its role needs (2.3-S1); not a stronger prompt instruction, because (our reasoning) a tool the agent does not hold cannot be misused.
- If a role needs another role's capability often but only in a simple form, choose a scoped tool for that case and route complex cases through the coordinator; not full access to the other role's tools, and not batching the requests until the end (Question 9).
- If a generic tool invites misuse, such as fetching any URL, choose a constrained replacement that validates its input.
- If one extraction must run before anything else, choose forced selection of that tool for the first turn and process the remaining steps in follow-up turns (with `auto`, or `any` if a tool call is still mandatory; our reading); not `any`, which guarantees some tool call but not that particular one.
- If the reply must be a tool call but the right tool depends on the input (for example, the document type is unknown), choose `"any"`; if a text answer is acceptable, keep `"auto"`.

**Traps**

- The Question 9 rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) rejects giving the synthesis agent every web search tool, which "over-provisions the synthesis agent, violating separation of concerns"; batching all verifications to the end of the pass, which "creates blocking dependencies"; and having the web search agent cache extra context in advance, which relies on speculative caching that cannot reliably predict what will need verifying.
- Using `allowedTools` to take tools away from a subagent in current SDK code. It pre-approves tools; restriction comes from `tools` or bare-name `disallowedTools` entries. The guide's model differs in one place: knowledge bullet 1.3-K1 says a coordinator's `allowedTools` must include "Task" for it to spawn subagents, and on that point answer in the guide's terms (see [Domain 1](#domain-1-agentic-architecture-orchestration)).
- Leaving `tool_choice` on `auto` when a tool call is mandatory: the model may return text instead.
- Expecting Claude to explain itself before a forced tool call: with `any` or `tool` it goes straight to the `tool_use` block.

**Go deeper:** [Controlling tool choice](knowledge/tool-use-and-mcp.md#controlling-tool-choice)

### Task 2.4: Integrate MCP servers into Claude Code and agent workflows

The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) states the task as "Integrate MCP servers into Claude Code and agent workflows". It tests where a server is configured, how secrets stay out of version control, what the agent sees once several servers connect, when to expose data as resources, how to make MCP tools win over built-in ones, and when to reuse a community server. Deploying or hosting MCP servers is outside the exam (see [What is out of scope](#what-is-out-of-scope)).

**Know**

- **Scopes in Claude Code** ([Claude Code MCP docs](https://code.claude.com/docs/en/mcp), as of September 2026):

| Scope | Loads in | Shared with the team | Stored in | Flag |
|---|---|---|---|---|
| Local (the default) | The current project only | No | `~/.claude.json`, under that project's path | none, or `--scope local` |
| Project | The current project only | Yes, via version control | `.mcp.json` in the project root | `--scope project` |
| User | All your projects | No | `~/.claude.json`, under the top-level `mcpServers` key | `--scope user` |

```bash
claude mcp add --transport http notion https://mcp.notion.com/mcp                       # local scope (default)
claude mcp add --transport http shared-server --scope project https://example.com/mcp   # writes .mcp.json
claude mcp add --transport http hubspot --scope user https://mcp.hubspot.com/anthropic  # all your projects
claude mcp list
```

All four commands are copied from the Claude Code MCP docs; the scope comments are ours.

- **Rules around `.mcp.json`.** Commit it so the whole team gets the same servers. In interactive sessions Claude Code asks for approval before using project-scoped servers (`claude mcp reset-project-choices` resets those choices); in `claude -p` runs, Agent SDK sessions and cloud sessions it cannot show that prompt and loads them without asking. A cloned repository cannot approve its own servers: approval settings such as `enableAllProjectMcpServers` committed to the project's `.claude/settings.json` are ignored in an untrusted folder, until you run `claude` there and accept the workspace trust dialog. When the same server name exists in several scopes, local wins over project, which wins over user (plugin-provided servers and claude.ai connectors rank below all three), and the winning entry is used whole rather than merged. `settings.json` does not read an `mcpServers` key.

- **Environment variable expansion.** `.mcp.json` supports `${VAR}` and `${VAR:-default}` in `command`, `args`, `env`, `url` and `headers`, so the file can be committed while tokens stay in each developer's environment (the guide's example is `${GITHUB_TOKEN}`). If a variable is unset and has no default, the config still loads, with a warning and the literal `${VAR}` text. In a remote server's `url` and `headers`, Claude Code reads certain credential variables as empty instead of expanding them, whether or not they are set, and ignores a `:-default` fallback on them. The docs describe three groups, each by example: Claude Code's own credentials, such as `ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN`; cloud-provider credentials, such as `AWS_BEARER_TOKEN_BEDROCK`; and other credentials the environment carries, such as `HTTPS_PROXY` and `NPM_TOKEN`. This keeps a project's `.mcp.json` from sending those credentials to a server it names. A name outside that set, such as `API_KEY` in the example below, expands as written. Because the groups are given by example rather than as a full list, the docs do not settle whether the guide's `${GITHUB_TOKEN}` is affected.

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

That is the docs' own variable-expansion example, copied verbatim: the URL falls back to a default when `API_BASE_URL` is unset, and the token comes from each developer's `API_KEY`.

- **All servers, one tool pool.** Per the guide, tools from every configured server are discovered at connection time and available to the agent simultaneously. The MCP architecture docs describe the same thing from the host's side: the application fetches tools from all connected servers and combines them into one registry for the model. In Claude Code each MCP tool is named `mcp__<server>__<tool>`, which is how hooks and permission rules match it.

- **In the Agent SDK.** Pass servers in `mcp_servers` (Python) or `mcpServers` (TypeScript) when calling `query()`, or put them in a `.mcp.json` at the project root, which loads when the `project` setting source is enabled (it is for default `query()` options). MCP tools need explicit permission: without it, Claude sees that they are available but cannot call them. `allowedTools` entries auto-approve MCP tools by their `mcp__<server-name>__<tool-name>` names, either one tool (the docs' example is `mcp__db__query`) or every tool from a server with a wildcard (`mcp__github__*`). The docs' quickstart, copied verbatim:

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

- **Tools for actions, resources for catalogs.** MCP servers offer three primitives, each controlled by a different party ([MCP server concepts](https://modelcontextprotocol.io/docs/2026-07-28/learn/server-concepts)):

| Primitive | Controlled by | Use it for |
|---|---|---|
| Tools | The model, which decides when to call them | Actions: write to databases, call APIs, modify files |
| Resources | The application | Passive, read-only context: file contents, database schemas, API documentation |
| Prompts | The user | Pre-built instruction templates; applications typically expose them as slash commands, command palettes, UI buttons or context menus |

The guide's in-scope list sums up the design rule as "resources for content catalogs, tools for actions" ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). A resource catalog of issue summaries, documentation hierarchies or database schemas shows the agent what exists without exploratory tool calls. In Claude Code you reference a resource with an @ mention in the form `@server:protocol://resource/path`, and Claude Code automatically provides tools to list and read resources when a server supports them. As of September 2026, the Messages API MCP connector supports only tool calls; for resources, the connector docs point to the SDK's client-side helpers, and Claude Code (above) handles them directly.

- **Make MCP tools win over built-ins.** The guide's fix for an agent that reaches for Grep when a more capable MCP tool exists is a richer MCP tool description that explains capabilities and outputs (2.4-S3). Under Claude Code's default tool search, only tool names and server instructions load at session start, and server instructions help Claude know when to search for your tools. As of September 2026, Claude Code truncates each tool description and each server's instructions at 2,048 characters by default (in v2.1.280 or later, `CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH` changes the limit), so put the critical details first; `alwaysLoad: true` on a server keeps its tools visible without a search step. In Anthropic's multi-agent research system, a tool-testing agent given a flawed MCP tool used it, then rewrote its description, and later agents using the new description completed tasks in 40% less time. The same system gave its agents the heuristic to prefer specialized tools over generic ones. The Claude Code MCP docs describe these levers but do not say how Claude weighs an MCP tool against a built-in such as Grep, so answer with the guide's pattern: a richer description. Claude Code's large-codebase guidance makes the same point from the other side: if you already run a code search or RAG index, expose it as an MCP tool so Claude queries it instead of reading files.

- **Reuse before you build.** For standard integrations the guide prefers an existing community server (its example is Jira) and keeps custom servers for team-specific workflows. Claude Code's docs point to reviewed connectors in the Anthropic Directory, and any remote server listed there can be added with `claude mcp add`. The MCP Registry, in preview as of September 2026, is the official centralized metadata repository for publicly accessible MCP servers. The [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) repository holds reference implementations meant as educational examples, "not as production-ready solutions". The Claude Code docs also say to verify that you trust each server before connecting it, because servers that fetch external content can expose you to prompt injection.

!!! warning "Exam guide vs current docs: MCP scopes and tool loading"

    - **Scopes.** The guide contrasts two scopes: project (`.mcp.json`) for shared team tooling and user (`~/.claude.json`) for personal or experimental servers. The [Claude Code MCP docs](https://code.claude.com/docs/en/mcp) (as of September 2026) describe three: local, project and user. Local is the default, the docs suggest it for personal development servers, experimental configurations and servers with credentials you do not want in version control, and it is also stored in `~/.claude.json`. The docs describe user scope as working well for personal utility servers, development tools or services you use across projects. The guide's contrast still holds (`.mcp.json` is shared, `~/.claude.json` is personal), so answer in the guide's terms: experimental and personal servers go in `~/.claude.json`.
    - **Tool loading.** The guide says tools from all configured servers are discovered at connection time. With tool search on, which is the default as of September 2026, Claude Code loads only tool names and server instructions at session start and defers full tool definitions until Claude needs them. Every configured server's tools are still available at once. Answer in the guide's terms.

**Decide**

- If the whole team needs a server, choose project scope: a committed `.mcp.json` with `${VAR}` references for tokens; not user scope, because teammates would not receive it, and not literal tokens, because the file lives in version control.
- If a server is personal or experimental, choose `~/.claude.json` (the guide's user scope); not `.mcp.json`, because that pushes it to everyone who pulls the repository.
- If the integration is a standard one such as Jira, choose an existing community server; reserve custom servers for team-specific workflows.
- If the agent makes many exploratory calls just to learn what data exists, choose exposing the catalog as MCP resources.
- If the agent keeps using Grep where an MCP tool would do better, choose rewriting the MCP tool's description to explain what it can do and what it returns.

**Traps**

- Committing real tokens in `.mcp.json`, or putting an `mcpServers` block in `settings.json`, which Claude Code does not read.
- Assuming `claude mcp add` without `--scope` shares the server or loads it in every project: the default is local scope.
- Confusing MCP local scope with Claude Code's general local settings: local-scoped MCP servers are stored in `~/.claude.json` in your home directory, while general local settings use `.claude/settings.local.json` in the project directory.
- Building a custom server for a standard integration.
- Leaving a content catalog reachable only through exploratory tool calls when a resource would show it up front.
- Believing the agent can use only one server's tools at a time.

**Go deeper:** [MCP in Claude Code](knowledge/tool-use-and-mcp.md#mcp-in-claude-code). Exercise 2, step 4, practices a project `.mcp.json` with variable expansion next to a personal experimental server in `~/.claude.json`, and checks that both are available at the same time; see [Official preparation exercises](#official-preparation-exercises).

### Task 2.5: Select and apply built-in tools (Read, Write, Edit, Bash, Grep, Glob) effectively

The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) states the task as "Select and apply built-in tools (Read, Write, Edit, Bash, Grep, Glob) effectively". It tests which tool fits which search or edit, the fallback when Edit cannot find a unique match, and how to explore a codebase incrementally, including tracing functions through wrapper modules. Scenario 4 names only five built-in tools and leaves out Edit; the task statement and the appendix list all six.

**Know**

- **The six tools** ([Claude Code tools reference](https://code.claude.com/docs/en/tools-reference), as of September 2026):

| Tool | What it does | Details worth knowing |
|---|---|---|
| Read | Returns a file's contents with line numbers | Claude is instructed to always pass absolute paths; a whole-file read over the token limit returns the first page with a `PARTIAL view` notice explaining how to read more with `offset` and `limit`; it reads files, not directories |
| Write | Creates a file, or overwrites one with the full content provided | It never appends or merges; for partial changes to an existing file Claude uses Edit |
| Edit | Replaces `old_string` with `new_string` by exact string match | No regex or fuzzy matching; `old_string` must appear exactly once unless `replace_all: true` is set |
| Bash | Runs shell commands (the Agent SDK lists it under Execution) | In the Agent SDK, Edit, Write and Bash run one at a time, while read-only tools can run concurrently |
| Grep | Searches file contents with ripgrep regex syntax | Escape metacharacters (`interface\{\}`); output modes `files_with_matches` (default), `content` and `count`; scope with the `glob` or `type` parameter; skips gitignored files |
| Glob | Finds files by path pattern | `**` matches at any depth (`**/*.js`, `src/**/*.ts`); results are sorted by modification time and capped at 100 files; it does not respect `.gitignore` by default |

- **The short version.** Glob finds files by name; Grep finds lines inside files.

- **Input shapes.** From the Agent SDK TypeScript reference (trimmed; the `GrepInput` type has more options than shown, such as `-i`, `-n`, `head_limit` and `multiline`):

```typescript
type FileEditInput = {
  file_path: string;
  old_string: string;
  new_string: string;
  replace_all?: boolean;
};

type FileReadInput = {
  file_path: string;
  offset?: number;
  limit?: number;
  pages?: string;
};

type FileWriteInput = {
  file_path: string;
  content: string;
};

type GlobInput = {
  pattern: string;
  path?: string;
};

type GrepInput = {
  pattern: string;
  path?: string;
  glob?: string;
  type?: string;
  output_mode?: "content" | "files_with_matches" | "count";
};
```

- **Edit's checks.** A single character of whitespace or indentation difference is enough for `old_string` to miss. The file must also have been read in the current conversation, and a read cut short with a `PARTIAL view` notice does not count. Claude Opus 4.6, Claude Haiku 4.5 and older models always require that read; newer models can edit an unread file when reading it would not need a permission prompt and the Read tool is available. That relaxed handling requires Claude Code v2.1.208 or later; before it, Claude Code refused any edit to a file it had not read in the conversation or that had changed on disk after the read.

- **Read, then Write.** The guide's fallback (2.5-K4, 2.5-S3) loads the full file with Read and writes the corrected whole file back, because Write replaces the entire file with the content given. The order matters in practice too: on Claude Opus 4.6, Claude Haiku 4.5 and older models, a Write to an existing file that has not been read in the conversation fails with an error, and a file read only partially (with a `PARTIAL view` notice) needs the read on every model. Before Claude Code v2.1.228, every model required the read before overwriting an existing file. New files need no prior read.

- **Incremental exploration (2.5-S4).** Start with Grep to find entry points, then Read to follow imports and trace the flow, instead of reading every file up front. This matches how Claude Code works: each tool call gives new information that informs the next step, and Anthropic's context-engineering write-up describes Claude Code loading CLAUDE.md up front while using glob and grep to retrieve files just in time. Claude Code's tips for a new codebase say to start with broad questions and then narrow down. When exploration would flood the main context, delegate it to a subagent (Task 5.4 in [Domain 5](#domain-5-context-management-reliability)).

- **Wrapper tracing (2.5-S5).** When code reaches a function through a wrapper module that re-exports it, first list every name the wrapper exports, then search the codebase for each name. The procedure is the guide's own: the Claude Code tools reference and common-workflows pages do not describe it, so learn it in the guide's words. An illustrative sequence (file and function names invented):

```text
1. Read   src/payments/index.ts                     list every exported name: charge, refund, voidCharge
2. Grep   pattern charge\(   output_mode content   glob **/*.ts
3. Grep   the same for refund\( and voidCharge\(
4. Read   each calling file, following imports only where the flow continues
```

- **Precise navigation.** Once a code intelligence plugin is installed for your language, Claude Code's LSP tool gives go-to-definition and find-references navigation, including call hierarchies.

!!! warning "Exam guide vs current docs: Edit fallback and default search tools"

    - **Edit fallback.** The guide's answer when Edit fails on a non-unique match is Read + Write. The [Claude Code tools reference](https://code.claude.com/docs/en/tools-reference) (as of September 2026) describes two other remedies: a longer `old_string` with enough surrounding context to match once, or `replace_all: true` when every occurrence should change. On the exam, choose Read + Write; in real work, try the documented remedies first.
    - **Grep and Glob availability.** As of September 2026, on macOS, Linux and WSL, Claude Code leaves Glob and Grep out of the default tool set and Claude searches with `find` and `grep` through Bash (embedded versions of `bfs` and `ugrep`); those searches reach hooks and permission rules as Bash calls. They come back when you name Glob or Grep in `--tools` or `--allowedTools` when you start the session, or in the equivalent Agent SDK options (with `--tools` you get the ones you list; naming either in `--allowedTools` restores both); when a deny rule, `--disallowedTools` or `--restricted` removes Bash from the session; or when a subagent's `tools` field lists Glob or Grep and leaves out Bash (they come back for that subagent only, or for the whole session when it runs as the main agent through `--agent` or the `agent` setting). An allow rule in a settings file does not bring them back. On Windows, Glob is part of the default set, and the Agent SDK's agent-loop page still lists Glob and Grep among its built-in search tools. The guide's selection logic (Grep for content, Glob for paths) is unchanged, so answer in its terms.

**Decide**

- If you need to find where a function is called, where an error message comes from, or which files import a module, choose Grep; not Glob, because Glob matches file paths, not file contents.
- If you need files by name or extension (for example `**/*.test.tsx`), choose Glob; not Grep.
- If the change is small and the anchor text is unique, choose Edit; if you are creating a file or replacing all of it, choose Write.
- If Edit fails because the anchor text is not unique, choose Read to load the full file and then Write the corrected version (the guide's answer).
- If the codebase is unfamiliar, choose Grep for entry points and Read along the imports; not reading every file up front.
- If calls go through a wrapper module, choose listing its exported names and searching for each; not searching only for the wrapper module's own name.

**Traps**

- Reading every file up front to understand a codebase: skill 2.5-S4 is the opposite pattern.
- Retrying Edit with the same non-unique `old_string`.
- Using Write for a one-line change. It rewrites the entire file, and Anthropic's prompting guide for Claude Fable 5.1 notes that, unless the file is short or most of it is changing, a rewrite costs more output tokens and time than a targeted edit.
- Expecting Grep to search gitignored files (it skips them unless given the file's path) or Glob to skip them (by default it does not).
- Swapping the two search tools: content search with Glob, or file-name search with Grep.

**Go deeper:** [Built-in tools, custom tools, Skills or MCP](knowledge/tool-use-and-mcp.md#built-in-tools-custom-tools-skills-or-mcp)

## Domain 3: Claude Code Configuration & Workflows

**Official weight: 20%**, which the guide defines as the approximate proportion of scored items drawn from this domain. That is about 12 of the 60 items (20% of 60, our arithmetic; the guide does not say how many of the 60 are scored).

Domain 3 has six task statements (3.1 to 3.6) carrying 23 "Knowledge of" and 26 "Skills in" bullets. It is a primary domain in three of the six scenarios: Scenario 2 (Code Generation with Claude Code), Scenario 4 (Developer Productivity with Claude) and Scenario 5 (Claude Code for Continuous Integration). The guide labels no sample question with a domain, but four of the twelve turn on this domain's content (our mapping): Question 4 (where a team command lives), Question 5 (plan mode), Question 6 (path-scoped rules) and Question 10 (`-p` in a pipeline). They are reproduced under [Official sample questions](#official-sample-questions). Exercise 2 in the guide lists Domain 3 first among the domains it reinforces and is the hands-on drill for this domain; its MCP step (step 4) belongs to Task 2.4 in [Domain 2](#domain-2-tool-design-mcp-integration).

Question 4 and Question 6 both turn on which file an instruction belongs in, so learn the paths exactly, including which ones reach teammates through version control.

### Paths, scopes and layers at a glance

| File | Exact path | Reaches teammates through version control? | When it loads |
|---|---|---|---|
| User CLAUDE.md | `~/.claude/CLAUDE.md` | No: it applies to you, in every project | Every session |
| Project CLAUDE.md | `./CLAUDE.md` or `./.claude/CLAUDE.md` | Yes, once committed | Every session |
| Directory CLAUDE.md | `<subdirectory>/CLAUDE.md` | Yes, once committed | When Claude reads a file in that subdirectory (at launch if you start Claude Code in that directory) |
| Local CLAUDE.md (docs, not in the guide) | `./CLAUDE.local.md` | No: add it to `.gitignore` | Every session, after `CLAUDE.md` at the same level |
| Managed policy CLAUDE.md (docs, not in the guide) | `/Library/Application Support/ClaudeCode/CLAUDE.md` (macOS), `/etc/claude-code/CLAUDE.md` (Linux, WSL), `C:\Program Files\ClaudeCode\CLAUDE.md` (Windows) | No: IT or DevOps deploys it to every user on the machine | Every session; individual settings cannot exclude it |
| Project rules | `.claude/rules/*.md` (subfolders allowed) | Yes, once committed | No `paths` field: every session. With `paths`: when Claude reads a matching file |
| User rules (docs, not in the guide) | `~/.claude/rules/*.md` | No | As above, in every project on your machine |
| Project commands | `.claude/commands/<name>.md` | Yes, once committed | Works like a skill: you invoke it as `/<name>`, and Claude can also invoke it |
| Personal commands | `~/.claude/commands/<name>.md` | No | Works like a skill: you invoke it as `/<name>`, and Claude can also invoke it |
| Project skills | `.claude/skills/<name>/SKILL.md` | Yes, once committed | Description in context by default; full body when invoked |
| Personal skills | `~/.claude/skills/<name>/SKILL.md` | No | Description in context by default; full body when invoked |

MCP server scopes (`.mcp.json` for shared team servers, `~/.claude.json` for personal ones) are Task 2.4 and are taught in Domain 2.

| If the instruction... | Put it in | Because |
|---|---|---|
| applies to every task in the project | the project `CLAUDE.md` | it loads in full every session |
| applies to one kind of file or one code area (test files anywhere, `terraform/**/*`, `src/api/**/*`) | a `.claude/rules/` file with `paths` globs | it loads only when matching files are read |
| is a task-specific workflow you run on demand | a skill (or a command file) | only its description sits in context until it is invoked |
| produces verbose or exploratory output | a skill with `context: fork`, or a subagent | the work runs in a separate context and only the result comes back |
| must happen every time, with no exceptions | a hook or a permission rule | Claude Code enforces these; CLAUDE.md and rules are guidance Claude reads |
| is yours alone | files under `~/.claude/`, or `CLAUDE.local.md` | none of them reach teammates through the repository |

### Task 3.1: Configure CLAUDE.md files with appropriate hierarchy, scoping, and modular organization

The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) (p. 12) words this task as "Configure CLAUDE.md files with appropriate hierarchy, scoping, and modular organization". Its bullets cover the three-level hierarchy, who sees user-level files, `@import`, `.claude/rules/`, and diagnosing load problems with `/memory`.

**Know**

- **The guide's three levels.** User-level `~/.claude/CLAUDE.md`, project-level `.claude/CLAUDE.md` or root `CLAUDE.md`, and directory-level `CLAUDE.md` files in subdirectories. The current [memory docs](https://code.claude.com/docs/en/memory) use the same project paths: "A project CLAUDE.md can be stored in either `./CLAUDE.md` or `./.claude/CLAUDE.md`."
- **User-level means you only.** Instructions in `~/.claude/CLAUDE.md` live in your home directory, not in the repository, so they never reach teammates through version control.
- **Files add up; they do not override.** Claude Code loads `CLAUDE.md` and `CLAUDE.local.md` from the working directory and every directory above it, concatenates them, and orders the content from the filesystem root down, so the file closest to where you launched is read last. If two files give different guidance for the same behavior, Claude may pick one arbitrarily. User-level rules load before project rules, and neither overrides the other either.
- **Subdirectory files load lazily.** A `CLAUDE.md` below the working directory is not loaded at launch. It loads when Claude reads a file in that directory with the Read tool, not when Claude writes or creates files there.
- **`@path/to/import`.** Pulls another file into a CLAUDE.md at launch. Relative paths resolve from the importing file, not the working directory. Imports can nest up to four hops. A path inside a code span or fenced code block is not imported. Imports organize content but do not reduce context, because imported files load at launch.
- **External imports.** An import in a project memory file that resolves outside the working directory triggers a one-time approval dialog; if you decline, those imports stay disabled.
- **`.claude/rules/`.** One topic per Markdown file (for example `testing.md` or `api-design.md`), discovered recursively. A rule file without `paths` frontmatter loads at launch with the same priority as `.claude/CLAUDE.md`. Path-scoped rules are Task 3.3.
- **Size.** The docs suggest keeping each CLAUDE.md under 200 lines, because longer files consume more context and reduce adherence, and suggest splitting into rules as a file approaches that length.
- **Context, not enforcement.** Claude treats CLAUDE.md as context, not enforced configuration.
- **Layers the guide does not list (as of September 2026).** Besides the docs-only rows in the table above, Claude Code v2.1.277 or later can read `AGENTS.md`. By default it does so only when there is no `CLAUDE.md`, `.claude/CLAUDE.md` or `CLAUDE.local.md` in the working directory or above it; if one exists, Claude reads the `CLAUDE.md` files only. Exam items are written against the guide's three levels.

An import-based CLAUDE.md, verbatim from the memory docs:

```markdown
See @README for project overview and @package.json for available npm commands for this project.

# Additional Instructions
- git workflow @docs/git-instructions.md
```

For skill 3.1-S2, each package's maintainers import only the standards files that apply to their package. An illustrative layout (the file names are ours):

```text
repo/
  CLAUDE.md                     universal standards for everyone
  standards/
    api-conventions.md
    react-conventions.md
  packages/
    billing-api/CLAUDE.md       contains the line: @../../standards/api-conventions.md
    web-app/CLAUDE.md           contains the line: @../../standards/react-conventions.md
```

Each import resolves from the file that contains it, so `../../standards/` points at the shared folder. If you launch Claude Code inside `packages/billing-api/`, that path resolves outside the working directory and triggers the one-time approval dialog for external imports (our reading of the external-imports rule).

**Decide**

- If instructions must reach every developer, choose the project `CLAUDE.md`, committed to the repository; not `~/.claude/CLAUDE.md`, because user-level files are never shared through version control. This is the fix for the guide's case of a new team member who does not receive the instructions.
- If instructions are personal, choose `~/.claude/CLAUDE.md` (all your projects) or `CLAUDE.local.md` (this project, gitignored); not the project file, because every teammate would inherit them.
- If packages need different standards, choose a package-level CLAUDE.md that `@import`s only the relevant standards files; not one root file that imports every standard, because imports load at launch and would put every package's rules in every session.
- If one CLAUDE.md has grown into a monolith, choose focused topic files in `.claude/rules/` (for example `testing.md`, `api-conventions.md`, `deployment.md`); not more text in the same file, because long files reduce adherence.
- If Claude behaves differently across sessions or teammates, first check which memory files actually loaded: `/memory` in the guide's terms (see the warning below).

**Traps**

- Team standards placed in `~/.claude/CLAUDE.md`: they work for the author and nobody else, which is exactly the hierarchy problem 3.1-S1 describes.
- Assuming a subdirectory CLAUDE.md overrides the root file. All levels are concatenated, and when two files contradict each other Claude may pick either one.
- Assuming `@import` saves tokens. It only reorganizes; imported files load at launch.
- Expecting a subdirectory CLAUDE.md to be active from the first prompt. It loads only once Claude reads a file in that directory.
- Treating CLAUDE.md as a guarantee. The [configuration debugging docs](https://code.claude.com/docs/en/debug-your-config) keep CLAUDE.md for how the project works and send security boundaries and anything that must never happen to permissions or hooks, "where you need a guarantee instead of guidance".

!!! warning "Exam guide vs current docs: /memory or /context"

    Skill 3.1-S4 in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) is "Using the /memory command to verify which memory files are loaded and diagnose inconsistent behavior across sessions". The current [memory docs](https://code.claude.com/docs/en/memory) describe `/memory` as a list of memory file locations (including entries for files that do not exist yet) that you can open and edit, and say: "To check which `CLAUDE.md` and rules files loaded into the current session, run `/context`." The `InstructionsLoaded` hook can also log which files load, when and why. The same page's troubleshooting steps use `/context` to verify that your CLAUDE.md files loaded and `/memory` to open and edit them. On the exam, choose `/memory` when the options follow the guide; in real work, use `/context`.

**Go deeper:** [CLAUDE.md and the memory hierarchy](knowledge/claude-code-configuration.md#claudemd-and-the-memory-hierarchy)

### Task 3.2: Create and configure custom slash commands and skills

The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) (p. 13) words this task as "Create and configure custom slash commands and skills". Its bullets cover project versus personal locations, three SKILL.md frontmatter keys (`context: fork`, `allowed-tools`, `argument-hint`), personal variants, and the choice between a skill and CLAUDE.md.

**Know**

| Mechanism | Project (shared through version control) | Personal (you only) | Invoked as |
|---|---|---|---|
| Command file | `.claude/commands/deploy.md` | `~/.claude/commands/deploy.md` | `/deploy` |
| Command file in a subfolder | `.claude/commands/frontend/component.md` | (the docs show the project form) | `/frontend:component` |
| Skill | `.claude/skills/deploy/SKILL.md` | `~/.claude/skills/deploy/SKILL.md` | `/deploy`, or loaded by Claude when relevant |

- **Command names.** A command file's name without the extension becomes the command; subfolders become `:` namespaces. A skill's command comes from its directory name; in a personal or project skill, the `name` field only sets the display label.
- **A skill is a folder.** `SKILL.md` must sit inside `.claude/skills/<name>/`. A file saved as `.claude/skills/name.md` is not a skill.
- **SKILL.md anatomy.** YAML frontmatter between `---` markers (read only when the opening `---` is the first line) followed by Markdown instructions. In Claude Code every frontmatter field is optional; `description` is the one the docs recommend, because Claude uses it to decide when to load the skill.
- **Command files take the same frontmatter** as skills, except `name` and `paths`, so `argument-hint` and `allowed-tools` also work in `.claude/commands/` files.
- **Loading cost.** In a regular session only skill descriptions sit in context; the full body loads when the skill is invoked, so long reference material costs almost nothing until needed. CLAUDE.md content loads in full every session.
- **Name clashes.** A skill beats a command file with the same name. Between skills: enterprise over personal, personal over project, so with `deploy` in both `~/.claude/skills/` and the project's `.claude/skills/`, `/deploy` runs the personal one. The guide's reason for giving personal variants in `~/.claude/skills/` different names is "to avoid affecting teammates"; the precedence rule adds a practical reason (our reading): a same-named personal skill would hide the team's version from you.
- **Arguments.** `$ARGUMENTS` in the body receives everything typed after the command. If no placeholder receives the arguments, Claude Code appends `ARGUMENTS: <your input>` to the end of the skill content.

A project command file of the kind Question 4 asks about, from the [`.claude` directory reference](https://code.claude.com/docs/en/claude-directory) (template escaping removed). Saved as `.claude/commands/fix-issue.md` and committed, it gives every developer `/fix-issue 123`; the `` !`...` `` line runs `gh issue view 123` and injects its output before Claude sees the prompt:

```markdown
---
argument-hint: <issue-number>
---

!`gh issue view $ARGUMENTS`

Investigate and fix the issue above.

1. Trace the bug to its root cause
2. Implement the fix
3. Write or update tests
4. Summarize what you changed and why
```

The frontmatter keys, side by side:

| Key | What the guide says | What the docs say (as of September 2026) |
|---|---|---|
| `context: fork` | runs the skill in an isolated sub-agent context so its output does not pollute the main conversation | starts a subagent (type set by `agent`: `Explore`, `Plan`, `general-purpose` or a custom agent; default `general-purpose`) with the skill content as its prompt; the subagent does not see your conversation history; since v2.1.218 it runs in the background by default, and `background: false` makes the turn wait; in a `-p` run Claude Code waits for the result anyway |
| `allowed-tools` | restricts tool access during skill execution | lets Claude use the listed tools without asking permission during the turn that invokes the skill (the grant clears with your next message); it does not restrict which tools are available, and permission settings still govern unlisted tools |
| `argument-hint` | prompts developers for required parameters when they invoke the skill without arguments | a hint shown during autocomplete, for example `[issue-number]` |
| `disallowed-tools` | not in the guide | removes tools from Claude's available pool while the skill is active; the restriction clears with your next message |

A forked skill that runs in the Explore agent with pre-approved `gh` commands, excerpted from the [skills docs](https://code.claude.com/docs/en/skills) (the section headings and the task line are left out). The `` !`command` `` lines run before the skill content is sent to Claude, and their output replaces the placeholder:

```markdown
---
name: pr-summary
description: Summarize changes in a pull request
context: fork
agent: Explore
allowed-tools: Bash(gh *)
---

- PR diff: !`gh pr diff`
- PR comments: !`gh pr view --comments`
- Changed files: !`gh pr diff --name-only`
```

Because the built-in Explore and Plan agents skip CLAUDE.md and the git status snapshot, a forked skill with `agent: Explore` sees only its SKILL.md content and the agent's own system prompt. When it finishes, the subagent summarizes its results and returns them to your main conversation.

An `argument-hint` example in a skill, excerpted from the [`.claude` directory reference](https://code.claude.com/docs/en/claude-directory) (only the frontmatter and the diff line are kept); `disable-model-invocation: true` means only you can invoke it, with `/name`, and Claude never invokes it on its own:

```markdown
---
description: Reviews code changes for security vulnerabilities, authentication gaps, and injection risks
disable-model-invocation: true
argument-hint: <branch-or-path>
---

!`git diff $ARGUMENTS`
```

**Decide**

- If every developer must get the command when they clone or pull, choose `.claude/commands/` in the repository (the answer to Question 4; a skill in `.claude/skills/` is shared the same way); not `~/.claude/commands/`, which is personal, and not CLAUDE.md, which holds instructions and context, not command definitions.
- If a skill produces verbose output (codebase analysis) or exploratory context (brainstorming alternatives), choose `context: fork`; not an inline skill, because inline output stays in the main conversation.
- If a skill must be limited to safe operations, the guide's answer is `allowed-tools` (for example, limiting it to file write operations to prevent destructive actions). In current releases, `disallowed-tools` removes tools while the skill is active, and the skills docs point to deny rules in your permission settings to block tools across all skills and prompts.
- If a skill needs a parameter, choose `argument-hint` in the frontmatter plus `$ARGUMENTS` in the body.
- If content is a universal standard that applies to every task, choose CLAUDE.md; if it is a task-specific workflow invoked on demand, choose a skill.
- If you want your own version of a team skill, create it in `~/.claude/skills/` under a different name, so the team's skill still runs for everyone, including you.

**Traps**

- Team commands in `~/.claude/commands/` (Question 4, option B): personal, never shared through version control.
- Command definitions inside CLAUDE.md (Question 4, option C).
- Invented configuration: Question 4's rationale says a `.claude/config.json` with a commands array "describes a configuration mechanism that doesn't exist in Claude Code" ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).
- Always-apply standards placed in a skill. Skills load on demand, which the guide contrasts with always-loaded CLAUDE.md.
- A forked skill written as bare guidelines. The docs say `context: fork` only makes sense for skills with explicit instructions: a skill that holds only guidelines such as "use these API conventions" gives the subagent no actionable prompt, and it returns without meaningful output. The forked subagent also cannot see the conversation, so the instructions must stand on their own.
- Outside the exam, reading `allowed-tools` in a repository you did not write as a safety limit. In current releases it grants approval, and workspace trust does not gate a project skill's `allowed-tools`, including in a `-p` run in a folder you have never trusted. On the exam, keep the guide's meaning (see the warning below).

!!! warning "Exam guide vs current docs: commands, allowed-tools and argument-hint"

    - **Commands.** The guide and Question 4 treat `.claude/commands/` as the home of project slash commands. The [skills docs](https://code.claude.com/docs/en/skills) now say "Custom commands have been merged into skills." A file at `.claude/commands/deploy.md` and a skill at `.claude/skills/deploy/SKILL.md` both create `/deploy`, existing command files keep working, and the docs say "Prefer a skill for new work". One current detail touches Question 4's example name: in current releases `/review` is an alias of the bundled `/code-review` skill. The skills docs' name-clash rule covers a skill that shares the bundled skill's own name: "A project `code-review` skill replaces `/code-review`, and the bundled alias `/review` never runs your skill". They do not say which one `/review` runs when your own command or skill is itself named `review`. Question 4 and its rationale do not mention the bundled command, so answer the item as written.
    - **`allowed-tools`.** Skill 3.2-S3 in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) reads "Configuring allowed-tools in skill frontmatter to restrict tool access during skill execution", and Exercise 2 step 3 asks for a skill "with context: fork and allowed-tools restrictions". The [skills docs](https://code.claude.com/docs/en/skills) say of `allowed-tools`: "It does not restrict which tools are available: every tool remains callable". The key that removes tools is `disallowed-tools`: "Tools removed from Claude's available pool while this skill is active."
    - **`argument-hint`.** The guide uses it to "prompt developers for required parameters when they invoke the skill without arguments"; the [skills docs](https://code.claude.com/docs/en/skills) define it as a "Hint shown during autocomplete to indicate expected arguments."

    Answer exam items in the guide's terms: `.claude/commands/` for shared commands, `allowed-tools` restricts, `argument-hint` prompts. Use the documented behavior in real projects.

**Go deeper:** [Agent Skills](knowledge/claude-code-configuration.md#agent-skills)

### Task 3.3: Apply path-specific rules for conditional convention loading

The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) (p. 13) words this task as "Apply path-specific rules for conditional convention loading". Its bullets cover the `paths` frontmatter field, what conditional loading saves, and why glob rules beat directory-level CLAUDE.md files for conventions spread across the codebase.

**Know**

- A rule file in `.claude/rules/` becomes conditional when its YAML frontmatter has a `paths` field holding glob patterns. The guide's example is `paths: ["terraform/**/*"]`; Exercise 2 uses `paths: ["src/api/**/*"]` for API conventions and `paths: ["**/*.test.*"]` for testing conventions.
- Loading only the rules that match keeps irrelevant conventions out of context and saves tokens. A rule without `paths` loads unconditionally and applies to all files.
- `paths` is the only frontmatter field Claude Code reads from a rule; any other field is ignored without an error. `paths` accepts a YAML list or a comma-separated string, and brace expansion such as `"src/**/*.{ts,tsx}"` works.
- If the YAML between the `---` markers does not parse, Claude Code ignores the frontmatter and loads the rule as if it had no `paths`, so it applies everywhere. `claude --debug` shows the parse error.
- Path-scoped rules are summarized away by compaction and reload only when a matching file is read again. If a rule must persist across compaction, the docs say to drop `paths` or move the rule into the project-root CLAUDE.md.
- Like CLAUDE.md, rules are guidance Claude reads, not configuration Claude Code enforces.

| Glob | Matches |
|---|---|
| `**/*.ts` | all TypeScript files in any directory |
| `src/**/*` | all files under `src/` |
| `*.md` | Markdown files in the project root |
| `**/*.test.tsx` | the guide's example for all test files, whatever their directory |
| `terraform/**/*` | the guide's example for everything under `terraform/` |

A test-file rule, verbatim from the [`.claude` directory reference](https://code.claude.com/docs/en/claude-directory):

```markdown
---
paths:
  - "**/*.test.ts"
  - "**/*.test.tsx"
---

# Testing Rules

- Use descriptive test names: "should [expected] when [condition]"
- Mock external dependencies, not internal modules
- Clean up side effects in afterEach
```

**Decide**

- If a convention applies to a file type spread across the codebase (for example test files that sit next to the code they test), choose a `.claude/rules/` file with a glob such as `**/*.test.tsx`; not subdirectory CLAUDE.md files, because those are bound to one directory. This is the reasoning behind Question 6.
- If a convention belongs to one directory that one team owns, a subdirectory CLAUDE.md is a reasonable home. The [large-codebases docs](https://code.claude.com/docs/en/large-codebases) point to central path-scoped rules when "You want all conventions in one place, or the same rule applies to many scattered paths".
- If a rule must apply everywhere, including after compaction, choose the project-root CLAUDE.md or a rule without `paths`; not a path-scoped rule, because compaction drops it until a matching file is read again.

**Traps** (the first three are Question 6's rejected options)

- One root CLAUDE.md with a section per code area, relying on Claude to infer which section applies: inference, not explicit matching.
- A skill per code type: Question 6's rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says this "requires manual skill invocation or relies on Claude choosing to load them", so in the guide's reasoning the conventions are not applied by explicit path matching. (As of September 2026 a skill can also take a `paths` field; the [skills docs](https://code.claude.com/docs/en/skills) say "When set, Claude loads the skill automatically only when working with files matching the patterns." Question 6's answer remains `.claude/rules/`.)
- A CLAUDE.md in every subdirectory: CLAUDE.md files are directory-bound and cannot easily cover files spread across many directories.
- Extra frontmatter on a rule (a `description`, say) does nothing, and malformed YAML turns a scoped rule into a global one (`claude --debug` shows the parse error).

!!! warning "Exam guide vs current docs: reading or editing"

    Knowledge bullet 3.3-K2 in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says path-scoped rules "load only when editing matching files". The [memory docs](https://code.claude.com/docs/en/memory) say: "Path-scoped rules trigger when Claude reads files matching the pattern, not on every tool use." Answer in the guide's words on the exam; in practice a scoped rule attaches as soon as Claude reads a matching file, before any edit.

**Go deeper:** [Path-scoped rules](knowledge/claude-code-configuration.md#path-scoped-rules)

### Task 3.4: Determine when to use plan mode vs direct execution

The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) (p. 14) words this task as "Determine when to use plan mode vs direct execution". Its bullets cover what plan mode is for, when direct execution is enough, the Explore subagent, and combining the two modes.

**Know**

- **What plan mode does.** The [permission modes docs](https://code.claude.com/docs/en/permission-modes) say: "Plan mode tells Claude to research and propose changes without making them." Claude reads files, runs shell commands to explore and writes a plan, but does not edit your source; except in interactive terminal sessions with bypass permissions available, edits stay blocked until you approve the plan.
- **How to enter it.** Press `Shift+Tab` (it cycles the permission modes) until the status bar shows `⏸ plan mode on`, prefix a single prompt with `/plan`, or start with `claude --permission-mode plan`. To make it the default for a project's terminal sessions, set `permissions.defaultMode` to `plan` in the project's `.claude/settings.json`.
- **Leaving it.** When the plan is ready you can approve and let Claude edit (with auto mode, or auto-accepted edits where auto mode is unavailable), approve with manual approval of each edit, or choose "No, keep planning" ([permission modes docs](https://code.claude.com/docs/en/permission-modes)). `Ctrl+G` opens the plan in your editor; pressing `Shift+Tab` again leaves plan mode without approving. The plan is re-injected from disk after compaction.
- **Why it pays off.** Planning explores the codebase and proposes an approach for approval before anything changes, which prevents expensive rework when the first direction is wrong.
- **The docs' rule of thumb.** "If you could describe the diff in one sentence, skip the plan." ([best practices](https://code.claude.com/docs/en/best-practices)). Planning is most useful when you are unsure of the approach, when the change touches multiple files, or when you do not know the code.
- **The recommended loop.** Explore (in plan mode), Plan, Implement (leave plan mode and code against the plan), Commit.
- **Model split.** The `opusplan` model alias uses `opus` during plan mode, then switches to `sonnet` for execution.
- **The Explore subagent.** A fast, read-only built-in subagent (Write and Edit are denied). Claude delegates to it to search or understand a codebase, which keeps exploration results out of the main conversation; it returns a summary. Claude sets its thoroughness to quick, medium or very thorough. It skips CLAUDE.md and the git status snapshot, and it is one-shot, so it cannot be resumed.
- **The Plan subagent.** When Claude needs to understand the codebase during plan mode, it delegates research to the built-in Plan subagent, so exploration output stays in a separate context window while the main conversation stays read-only. Like Explore, it is one-shot and cannot be resumed.

The project default as a settings file, adapted from the [permission modes docs](https://code.claude.com/docs/en/permission-modes) (their example sets `default` in `~/.claude/settings.json`; this one sets `plan` in `.claude/settings.json`, which the same page names as the way to make plan mode a project's default):

```json
{
  "permissions": {
    "defaultMode": "plan"
  }
}
```

| Task (the guide's own examples) | Choose |
|---|---|
| Restructure a monolith into microservices (dozens of files, service boundaries to decide) | Plan mode |
| Library migration affecting 45+ files | Plan mode to investigate, then direct execution of the agreed plan |
| Choose between integration approaches with different infrastructure requirements | Plan mode |
| New feature with multiple valid implementation approaches | Plan mode |
| Single-file bug fix with a clear stack trace | Direct execution |
| Add a date validation conditional, or a single validation check to one function | Direct execution |

**Decide**

- If the task already involves architectural decisions, service boundaries, many files or competing approaches, choose plan mode first; not starting in direct execution and switching only if it gets complicated, because the complexity is stated up front, not something that might emerge later (Question 5).
- If the change is a well-understood fix with clear scope, choose direct execution; not plan mode, because planning adds overhead when there is no design decision to make.
- If a multi-phase task needs a verbose discovery phase, choose the Explore subagent for discovery; not a sweep through files in the main conversation, because the verbose output can exhaust the context window.
- If investigation is hard but implementation is mechanical once decided, combine them: plan mode to investigate and agree on the approach, then direct execution of the plan.

**Traps** (the first three are Question 5's rejected options)

- Start with direct execution and let the implementation reveal the service boundaries: dependencies discovered late mean costly rework.
- Direct execution with detailed upfront instructions for every service: it assumes you already know the right structure without exploring the code.
- Begin in direct execution and switch to plan mode only if complexity appears: it ignores complexity the requirements already state.
- Plan mode for a typo, a log line or a rename: the [best practices docs](https://code.claude.com/docs/en/best-practices) call plan mode useful but say it "also adds overhead".

!!! note "Current product details (as of September 2026)"

    Since v2.1.198 the Explore subagent inherits the main conversation's model (capped at Opus on the Claude API) instead of always running on Haiku. When auto mode is available, a classifier reviews shell commands during planning instead of prompting you (`useAutoModeDuringPlan`, on by default). Neither detail changes the exam's decision rule.

**Go deeper:** [Plan mode or direct execution](knowledge/claude-code-workflows.md#plan-mode-or-direct-execution)

### Task 3.5: Apply iterative refinement techniques for progressive improvement

The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) (p. 14) words this task as "Apply iterative refinement techniques for progressive improvement". It names four techniques and when each one fits.

**Know**

| Technique | Use it when | What you give Claude |
|---|---|---|
| Concrete input/output examples | prose descriptions of a transformation are interpreted inconsistently | 2 to 3 input/output pairs (the guide writes "2-3"); for an edge case, a specific test case with example input and expected output (for example null values in a migration script) |
| Test-driven iteration | the behavior can be checked by tests | a test suite written first (expected behavior, edge cases, performance requirements), then the test failures on each round |
| Interview pattern | you are working in an unfamiliar domain | a request that Claude ask you questions before implementing, to surface considerations such as cache invalidation strategies and failure modes |
| One message or sequential | several issues need fixing | interacting issues together in one detailed message; independent issues one at a time |

- **Why examples work.** Anthropic's [prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) say: "Examples are one of the most reliable ways to steer Claude's output format, tone, and structure." Anthropic's [prompt engineering blog post](https://claude.com/blog/best-practices-for-prompt-engineering) lists "Simple instructions haven't produced consistent results" among the reasons to add examples.
- **Give Claude a check it can run.** The [best practices docs](https://code.claude.com/docs/en/best-practices) recommend tests, a build or a screenshot to compare; with a pass or fail signal, Claude runs the check, reads the result and iterates until it passes. Their example puts test cases for a `validateEmail` function directly in the prompt and ends with "run the tests after implementing".
- **The test-first recipe.** Anthropic's 2025 Claude Code best-practices post ([archived copy](https://web.archive.org/web/20250602202220/https://www.anthropic.com/engineering/claude-code-best-practices); the live URL now redirects to the docs) has Claude write tests based on expected input/output pairs, and says to be explicit that this is test-driven development so Claude avoids creating mock implementations. Then: "Tell Claude to run the tests and confirm they fail." Next, Claude writes code that passes them, "instructing it not to modify the tests", and keeps going until all tests pass, usually over a few iterations; the post adds that it can help to verify with independent subagents that the implementation is not overfitting to the tests. The guide's 3.5-S2 also puts performance requirements in the suite; that detail comes from the guide's own bullet. The [prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) add: "Tests are there to verify correctness, not to define the solution."
- **The interview pattern in the docs.** "For larger features, have Claude interview you first." ([best practices](https://code.claude.com/docs/en/best-practices)). Start with a minimal prompt, ask Claude to interview you with the `AskUserQuestion` tool (which asks multiple-choice questions to gather requirements or clarify ambiguity), have it write the spec to `SPEC.md`, then "Once the spec is complete, start a fresh session to execute it."
- **Interacting and independent issues.** In Anthropic's [failing-tests cookbook](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/managed_agents/CMA_iterate_fix_failing_tests.ipynb), `test_mean` starts passing once `add` and `divide` are fixed, with no direct edit to `mean()`, because `mean()` calls both: an example of failures that interact. The archived post's workflow for independent issues (a list of lint errors) has Claude address them one by one, fixing and verifying each before moving to the next. The rule itself (one detailed message for interacting problems, sequential fixes for independent ones) is the guide's; these two sources are illustrations of each side (our mapping).
- **When iteration stalls.** If you have corrected Claude more than twice on the same issue in one session, the docs say the context is cluttered with failed approaches: run `/clear` and start again with a better prompt.

An illustrative prompt for skill 3.5-S4 (our wording, not an official example): instead of re-describing the bug, it pins the edge cases down as input and expected output.

```text
The migration script mishandles missing values. Add these as test cases, run them,
and iterate until they pass without changing the tests:

Input row:  {"id": 7, "email": null, "signup_date": "2024-03-01"}
Expected:   inserted with email = NULL (not the string "null", not an empty string)

Input row:  {"id": 8, "email": "a@example.com", "signup_date": null}
Expected:   skipped and logged to migration_errors with reason "missing signup_date"
```

**Decide**

- If Claude keeps misreading a prose description of a transformation, choose 2 to 3 concrete input/output examples; not a longer or more emphatic rewrite of the prose, because the guide names examples as the most effective fix when prose is interpreted inconsistently.
- If the behavior is testable, choose tests first (expected behavior, edge cases, performance requirements) and feed back the failures; not repeated messages saying the output still looks wrong, because failing tests tell Claude exactly what to fix.
- If one edge case fails, choose a specific test case with its input and expected output.
- If the domain is unfamiliar, choose the interview pattern before any code is written.
- If the fixes interact, choose one detailed message that lists every issue; if they are independent, fix and verify them sequentially.

**Traps**

- Splitting interacting problems across separate turns, so each fix is made without the constraint the others impose.
- Letting Claude change the tests to make them pass.
- Implementing in an unfamiliar domain before surfacing considerations such as cache invalidation or failure modes.
- Correcting the same issue again and again in one long session instead of restarting with a better prompt.

**Go deeper:** [Iterative refinement](knowledge/claude-code-workflows.md#iterative-refinement)

### Task 3.6: Integrate Claude Code into CI/CD pipelines

The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) (p. 15) words this task as "Integrate Claude Code into CI/CD pipelines". Its bullets cover non-interactive mode, structured output flags, CLAUDE.md as CI context, independent review instances, and avoiding duplicate review comments and duplicate tests.

**Know**

- **`-p` (or `--print`).** Runs Claude Code in non-interactive mode: it processes the prompt, prints the result to stdout and exits. Without it, a pipeline job waits for interactive input (Question 10). Claude Code exits with code 0 on success and non-zero on failure, so scripts can branch on the exit status.
- **Output flags.** `--output-format` accepts `text` (the default), `json` (the result, session ID and metadata) or `stream-json` (newline-delimited JSON). `--json-schema` returns validated JSON matching a JSON Schema after the agent completes its workflow, in print mode only. With `--output-format json`, the schema-conforming result is in the `structured_output` field.
- **Invalid schemas.** Since v2.1.205, an invalid schema makes `claude` exit with `Error: --json-schema is not a valid JSON Schema`; earlier versions silently returned unstructured text.
- **Unattended controls.** `--allowedTools` lists tools that run without prompting. `--permission-mode dontAsk` auto-denies anything that would otherwise prompt; the [permission modes docs](https://code.claude.com/docs/en/permission-modes) list it for "Locked-down CI and scripts". `--max-turns` (print mode only) exits with an error when the limit is reached, and `--max-budget-usd` (print mode only) sets the maximum dollar amount to spend on API calls before stopping. For `-p`, the built-in starting permission mode is Manual (config value `default`) on every plan, so the headless docs say to pass the permission mode you want.
- **Project context in CI.** The guide names CLAUDE.md as the way to give CI-invoked Claude Code its testing standards, fixture conventions and review criteria. The [GitHub Actions docs](https://code.claude.com/docs/en/github-actions) likewise say to put code style guidelines, review criteria and project rules in a root `CLAUDE.md`, and the managed Code Review product can be tuned with `CLAUDE.md` or `REVIEW.md`.
- **Session context isolation.** The session that generated code is less effective at reviewing it than an independent review instance, because it keeps the reasoning from generation and is less likely to question its own decisions. The [best practices docs](https://code.claude.com/docs/en/best-practices) agree: "A fresh context improves code review since Claude won't be biased toward code it just wrote." The bundled `/code-review` skill reviews the current diff in a fresh subagent.

Schema-enforced output from a non-interactive run, verbatim from the [headless docs](https://code.claude.com/docs/en/headless):

```bash
claude -p "Extract function names from auth.py" \
  --output-format json \
  --json-schema '{"type":"object","properties":{"functions":{"type":"array","items":{"type":"string"}}},"required":["functions"]}' \
  | jq '.structured_output'
```

A locked-down CI run with an exact allowlist, verbatim from the [permission modes docs](https://code.claude.com/docs/en/permission-modes):

```bash
claude -p "run the test suite" --permission-mode dontAsk --allowedTools "Bash(npm test)" "Read"
```

- **Findings as PR comments (3.6-S2).** Define a schema for findings, run with `--output-format json --json-schema`, and let the pipeline post each item as an inline comment. In GitHub Actions, `claude_args` accepts `--json-schema`, and the validated result becomes the `structured_output` output of the step ([action usage docs](https://github.com/anthropics/claude-code-action/blob/main/docs/usage.md)).
- **Re-running a review after new commits (3.6-S3).** Put the prior review findings in context and instruct Claude to report only new or still-unaddressed issues, so the PR does not collect duplicate comments. This technique is the guide's own pattern. Anthropic's managed [Code Review](https://code.claude.com/docs/en/code-review) documents related behavior instead: in the "After every push" mode it catches new issues as the PR evolves and auto-resolves threads when you fix flagged issues; each run's results are deduplicated and ranked by severity before posting; and a `REVIEW.md` "Re-review convergence" rule tells Claude how to behave when a PR has already been reviewed.
- **Test generation (3.6-S4, 3.6-S5).** Put the existing test files in context so Claude does not suggest scenarios the suite already covers; Claude also examines existing tests to match their style, frameworks and assertion patterns. Document testing standards, what makes a test valuable, and the available fixtures in CLAUDE.md to raise quality and cut low-value tests.

**Decide**

- If a pipeline job hangs waiting for input, choose `-p`; not an environment variable, a stdin redirect or a `--batch` flag (Question 10).
- If findings must be posted automatically, choose `--output-format json` with `--json-schema` and read `structured_output`; not free-text output that a script has to scrape, because the guide asks for machine-parseable findings.
- If CI runs need project conventions, choose CLAUDE.md in the repository: testing standards, fixture conventions, review criteria.
- If Claude generated the code, choose an independent review instance (a separate session or a fresh subagent); not the generating session, because it is less likely to question its own decisions.
- If a review re-runs after new commits, choose to pass the prior findings and ask for new or still-open issues only; not a fresh full review that posts everything again.
- If Claude is generating tests, choose to include the existing test files; not generation from the source alone, which repeats covered scenarios.

**Traps**

- Question 10's distractors: a `CLAUDE_HEADLESS=true` environment variable and a `--batch` flag, which the rationale calls non-existent features, and redirecting stdin from `/dev/null`, which it calls a Unix workaround. Do not over-learn this: `/batch` does exist in current releases, as a bundled skill that decomposes a large change into 5 to 30 independent units, each run by a background subagent in its own git worktree; it has nothing to do with non-interactive mode, and it is a slash command, not a CLI flag.
- Asking the session that wrote the code to review its own changes.
- Re-runs that post the same findings again, and test generation that proposes cases the suite already has.
- Mixing this task up with Question 11, which is about the Message Batches API for overnight jobs: that is Task 4.5 in [Domain 4](#domain-4-prompt-engineering-structured-output).

!!! note "Current docs: scripted runs and trust (as of September 2026)"

    The guide does not mention `--bare`. The [headless docs](https://code.claude.com/docs/en/headless) say "`--bare` is the recommended mode for scripted and SDK calls, and will become the default for `-p` in a future release." Bare mode skips auto-discovery of hooks, skills, custom commands, subagents, plugins, MCP servers, auto memory and CLAUDE.md, which makes it useful for CI runs that must give the same result on every machine; project context then has to be passed explicitly, for example with `--append-system-prompt-file`. Bare mode never reads OAuth credentials or the system keychain, so a CI job on the Anthropic API sets `ANTHROPIC_API_KEY` (or supplies an `apiKeyHelper` in the `--settings` JSON). Without `--bare`, a `-p` run treats the folder as trusted: it runs the hooks in the repository's `.claude/settings.json` and connects the servers in its `.mcp.json` with no trust dialog. For exam answers, CLAUDE.md remains the mechanism for CI context; note that the guide's answer assumes a run that reads CLAUDE.md, which a `--bare` run does not (our reading).

**Go deeper:** [Claude Code in CI/CD](knowledge/claude-code-workflows.md#claude-code-in-cicd)

## Domain 4: Prompt Engineering & Structured Output

**Official weight: 20%**, which the guide defines as the approximate proportion of scored items drawn from this domain. That is about 12 items if applied to all 60 (20% of 60, our arithmetic); the guide does not say how many of the 60 are scored.

Domain 4 is a primary domain in two of [the six scenarios](#the-six-scenarios): Claude Code for Continuous Integration and Structured Data Extraction. Its six task statements hold 22 "Knowledge of" and 25 "Skills in" bullets, 47 in all. Two threads run through them (our grouping): review prompts whose findings engineers trust (4.1, 4.2, the feedback loop in 4.4, and 4.6) and extraction pipelines whose output downstream systems can trust (4.2 to 4.5). The guide's sample questions carry no domain labels, but by our mapping Question 11 turns on batch processing (Task 4.5) and Question 12 on multi-pass review (Task 4.6); both are reproduced under [Official sample questions](#official-sample-questions). Exercise 3 in [Official preparation exercises](#official-preparation-exercises) lists Domain 4 among the domains it reinforces, and the guide's How to Prepare advice asks you to practice three of these techniques: few-shot examples for ambiguous scenarios, explicit review criteria to reduce false positives, and multi-pass review for large code reviews.

### Task 4.1: Design prompts with explicit criteria to improve precision and reduce false positives

Task statement 4.1 in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): "Design prompts with explicit criteria to improve precision and reduce false positives". By our mapping it fits most directly in Scenario 5 (Claude Code for Continuous Integration), whose description says: "You need to design prompts that provide actionable feedback and minimize false positives."

**Know**

- **Explicit beats vague.** The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) contrasts "check that comments are accurate" (vague) with "flag comments only when claimed behavior contradicts actual code behavior" (explicit). The explicit version states the condition that makes something reportable.
- **Caution words do not buy precision.** The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says general instructions such as "be conservative" or "only report high-confidence findings" fail to improve precision compared with specific categorical criteria.
- **Write report and skip lists.** Define which issues to report (bugs, security) and which to skip (minor style, local patterns), instead of relying on confidence-based filtering.
- **Noise spreads.** High false-positive categories undermine developers' confidence in the categories that are accurate. The review prompt of the `code-review` plugin in Anthropic's [claude-code repository](https://raw.githubusercontent.com/anthropics/claude-code/main/plugins/code-review/commands/code-review.md) puts it plainly: "False positives erode trust and waste reviewer time."
- **Switch a noisy category off while you fix it.** Temporarily disabling a high false-positive category restores developer trust while you improve the prompt for that category. In Claude Code's managed [Code Review](https://code.claude.com/docs/en/code-review) (a research preview for Team and Enterprise subscriptions, as of September 2026), one way to do this is a `REVIEW.md` skip rule; skip rules "list paths, branch patterns, and finding categories where Claude should post no findings". The same page also describes a verification bar, which requires evidence before a class of finding is posted; using it as a gentler option than switching a category off is our suggestion, not the page's. The local `/code-review` command does not read `REVIEW.md`, so these rules have no effect there.
- **Anchor severity in code.** Define explicit criteria for each severity level, with concrete code examples for each level, to get consistent classification.

!!! example "Explicit criteria in a `REVIEW.md` file"

    Two sections of the `REVIEW.md` example in the [Code Review docs](https://code.claude.com/docs/en/code-review), verbatim. The full example also has a title line and sections that cap nits and list what to always check.

    ```markdown
    ## What Important means here

    Reserve Important for findings that would break behavior, leak data,
    or block a rollback: incorrect logic, unscoped database queries, PII
    in logs or error messages, and migrations that aren't backward
    compatible. Style, naming, and refactoring suggestions are Nit at
    most.

    ## Do not report

    - Anything CI already enforces: lint, formatting, type errors
    - Generated files under `src/gen/` and any `*.lock` file
    - Test-only code that intentionally violates production rules
    ```

The same [Code Review page](https://code.claude.com/docs/en/code-review) warns that "Length has a cost: a long `REVIEW.md` dilutes the rules that matter most." Explicit does not mean long: name the categories and the bar, then stop.

Task 4.1 also asks for severity criteria with a concrete code example for each level. Code Review tags findings with three levels, and the docs define each in one line. An illustrative rubric on those levels (the definitions are the docs'; the code examples are ours):

| Level | Definition in the Code Review docs | Concrete code example (ours) |
|---|---|---|
| Important | "A bug that should be fixed before merging" | New code in the diff: `except TimeoutError: return []`, so a failed lookup reads as "no orders" |
| Nit | "A minor issue, worth fixing but not blocking" | A new helper named `proc()` whose name does not say what it processes (the docs' `REVIEW.md` example above rates naming suggestions Nit at most) |
| Pre-existing | "A bug that exists in the codebase but was not introduced by this PR" | The same `except TimeoutError: return []` in a file the pull request does not change |

**Decide**

- If a reviewer's findings are mostly noise, choose explicit report and skip categories with examples of each; not a general caution instruction or a confidence cut-off, because the guide says general caution instructions do not improve precision the way categorical criteria do.
- If one category is noisy and the others are accurate, choose to disable that category temporarily while you rework its prompt; not to keep posting it, because its false positives erode trust in the accurate categories too.
- If severity labels drift between runs, choose written criteria with a concrete code example for each level; not a scale of adjectives such as minor or serious, because adjectives are the vague wording this task statement warns against. The current prompting guides make the same point about review bars: "be concrete about where the bar is", not qualitative terms such as *important* ([Prompting Claude Opus 4.8](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-4-8)).

**Traps**

- An option that adds "be conservative" or "only report high-confidence findings" to the prompt. The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) names both as instructions that fail to improve precision.
- An option that filters findings by the model's confidence instead of defining which categories to report and which to skip.
- An option that keeps a high false-positive category live because some of its findings are real. Its noise undermines confidence in the accurate categories too.

!!! warning "Exam guide vs current docs: caution instructions"

    The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says instructions like "be conservative" fail to improve precision. The current prompting guides for Claude Sonnet 5 and Claude Opus 4.8 (as of September 2026) describe a different effect on those models: they may follow such instructions more faithfully than earlier models did, and "Precision typically rises, but measured recall can fall even though the model's underlying bug-finding ability has improved" ([Prompting Claude Sonnet 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5)). Their remedy is to report every issue with a confidence level and an estimated severity and filter in a separate verification step, or, in a single pass, to state a concrete bar such as "report any bugs that could cause incorrect behavior, a test failure, or a misleading result; only omit nits like pure style or naming preferences." ([Prompting Claude Opus 4.8](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-4-8)). The [Claude Opus 5 guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5) gives the same advice for a prompt that says "be conservative": "ask it to report everything and filter in a separate pass instead." The guide and the docs agree that concrete criteria beat qualitative words. On the exam, answer in the guide's terms: explicit categorical criteria, not general caution.

**Go deeper:** [Explicit criteria and precision](knowledge/prompt-engineering.md#explicit-criteria-and-precision)

### Task 4.2: Apply few-shot prompting to improve output consistency and quality

Task statement 4.2 in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): "Apply few-shot prompting to improve output consistency and quality". Its bullets cover when examples beat detailed instructions, examples that show the reasoning behind ambiguous cases, examples that fix the output format or mark the line between acceptable code and a real issue, and examples for extraction from varied document structures.

**Know**

- **When examples beat instructions.** The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) calls few-shot examples "the most effective technique for achieving consistently formatted, actionable output when detailed instructions alone produce inconsistent results". Anthropic's [prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) agree: "Examples are one of the most reliable ways to steer Claude's output format, tone, and structure."
- **Ambiguous cases are where they pay off.** The guide's examples are tool selection for ambiguous requests and branch-level test coverage gaps. In both, the rule is hard to state and a worked case is easy to show (our illustrations): a request that could go to either of two tools, with the tool chosen and why; or a function whose tests exercise one branch of a condition and never the other, reported as a coverage gap. If the tool descriptions themselves are minimal, fix them first (see Question 2 under Decide).
- **Show the reasoning, not only the answer.** Write 2 to 4 targeted examples for ambiguous scenarios that show why one action was chosen over plausible alternatives. Few-shot examples let the model generalize its judgment to novel patterns instead of matching only the cases you listed, and the stated reasons give it the logic to carry over. The docs' [ticket-routing guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/ticket-routing) gives the same advice: add a classification rationale for nuanced cases "so that Claude can better generalize the logic to other tickets". The [prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) also suggest `<thinking>` tags inside few-shot examples to show the reasoning pattern, which Claude then generalizes to its own thinking.
- **Fix the output format by showing it.** Examples that demonstrate the exact fields you want (location, issue, severity, suggested fix) make findings consistent.
- **Teach the boundary.** Examples that set an acceptable code pattern beside a genuine issue reduce false positives while still letting the model generalize.
- **Extraction from varied documents.** Examples reduce hallucination in extraction tasks, such as handling informal measurements and varied document structures. Show how to handle inline citations versus bibliographies, and methodology sections versus details embedded in the text (Exercise 3 adds narrative descriptions versus structured tables). When required fields come back empty or null on documents with some formats, add examples of correct extraction from those formats.

How to build the examples, from the current docs:

- Make them relevant to your real inputs, diverse enough to cover edge cases, and structured: wrap each one in `<example>` tags, and a set in `<examples>`, so Claude can tell them apart from instructions.
- Expect details to be copied. The claude.com [prompt engineering post](https://claude.com/blog/best-practices-for-prompt-engineering) warns that "Claude 4.x and similar advanced models pay very close attention to details in examples." The best-practices page asks for examples that "vary enough that Claude doesn't pick up unintended patterns": examples that all share an incidental feature can teach that feature.
- Curate rather than pile up. Anthropic's [context engineering post](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) recommends "diverse, canonical examples" over a long list of edge cases.
- Tool definitions can carry examples too: `input_examples` holds example inputs that must be valid against the tool's `input_schema` (an invalid example returns a 400 error). It is not available on server tools such as web search or code execution, or on the computer use and browser use toolsets.

An illustrative pair for a code reviewer (ours, not an official example). One case is reported, one is skipped, each says why, and the severity uses the Important level from the rubric in 4.1:

```text
<examples>
<example>
<code>
def load_orders(customer_id):
    try:
        return db.fetch_orders(customer_id)
    except TimeoutError:
        return []
</code>
<finding>
location: orders/service.py:12
issue: A timeout is returned as an empty list, so callers cannot tell "no orders" from "lookup failed".
severity: Important
suggested_fix: Return an explicit error result and let the caller decide whether to retry.
</finding>
<why>Reported: a correctness bug (a failure disguised as a valid empty result), not a style preference.</why>
</example>
<example>
<code>
cust_id = request.args["customer_id"]
</code>
<finding>none</finding>
<why>Skipped: this module abbreviates customer to "cust" everywhere. A consistent local naming pattern is on the skip list.</why>
</example>
</examples>
```

**Decide**

- If detailed instructions still produce inconsistent format or judgment, choose 2 to 4 targeted examples of the ambiguous cases, each with its reasoning; not another paragraph of rules, because the guide ranks few-shot examples as the most effective fix for that symptom.
- If the stem shows a root cause that examples do not touch, fix the root cause. For minimal tool descriptions, expand the descriptions first: Question 2's rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says few-shot examples "add token overhead without fixing the underlying issue". For a sequence that must always happen, enforce it in code: Question 1's rationale says prompt instructions and few-shot examples "rely on probabilistic LLM compliance".
- If required fields come back empty or null on documents with some formats, although the information is in those documents, choose few-shot examples showing correct extraction from those formats; not a rule that forces a value, because required fields the source cannot fill push the model to fabricate (see 4.3). If the information is not in the document at all, no example can supply it (our reading, in line with the limits of retry in 4.4): make the field nullable so the model can return null.
- If escalation boundaries are unclear, choose explicit escalation criteria with few-shot examples; not a self-reported confidence threshold or a separate trained classifier, because Question 3's rationale calls criteria plus examples "the proportionate first response before adding infrastructure".

**Traps**

- Few-shot examples offered as the fix when the real problem is minimal tool descriptions (Question 2, option A).
- Few-shot examples offered as the guarantee for a rule that must hold every time, such as verifying the customer before a refund (Question 1, option C).
- Examples that all look alike, so the model learns their shared surface features instead of the boundary you meant to teach.
- Examples that show only the answer. Without the reason for each choice, the model has less to generalize from on a case the examples do not cover.

!!! warning "Exam guide vs current docs: how many examples"

    The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says 2 to 4 targeted examples for ambiguous scenarios (and, in Task 3.5, 2 to 3 concrete input/output examples to clarify transformation requirements). Anthropic's [prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) recommend 3 to 5 examples for best results, and the claude.com [prompt engineering post](https://claude.com/blog/best-practices-for-prompt-engineering) suggests starting with one example and adding more only if the output still does not match your needs. On the exam, use the guide's 2 to 4 for ambiguous-case examples.

**Go deeper:** [Few-shot examples](knowledge/prompt-engineering.md#few-shot-examples)

### Task 4.3: Enforce structured output using tool use and JSON schemas

Task statement 4.3 in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): "Enforce structured output using tool use and JSON schemas". By our mapping it is central to Scenario 6 (Structured Data Extraction), a system that "validates the output using JavaScript Object Notation (JSON) schemas".

**Know**

The pattern: define an extraction tool whose `input_schema` is the JSON shape you need, then read the structured data from the `input` of the `tool_use` block in the response. The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) calls tool use with JSON schemas "the most reliable approach for guaranteed schema-compliant structured output, eliminating JSON syntax errors". The [tool use docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works) state the principle ("When you need a JSON object with specific fields rather than prose that happens to contain the information, a tool schema enforces the shape.") and give a test: "if you're writing a regex to extract a decision from model output, that decision should have been a tool call."

Reading the result in Python, adapted from the [search results docs](https://platform.claude.com/docs/en/build-with-claude/search-results), which find the `tool_use` block the same way (the first comment line, the `next(...)` lookup and the `None` check are theirs; the variable names, the `extraction` line and its comment are ours):

```python
# The tool_use block is not always first: iterate to find it.
tool_block = next((block for block in response.content if block.type == "tool_use"), None)
if tool_block is not None:
    extraction = tool_block.input  # with "strict": true this follows input_schema; still check stop_reason and compare enum values case-insensitively
```

`tool_choice` decides whether the extraction tool has to be called:

| `tool_choice` | What it does | Use it in extraction when |
|---|---|---|
| `{"type": "auto"}` | The model may return text instead of calling a tool | A text reply is acceptable. It is the default when tools are provided |
| `{"type": "any"}` | The model must call a tool, but chooses which | Several extraction schemas exist and the document type is unknown |
| `{"type": "tool", "name": "extract_metadata"}` | The model must call the named tool | One extraction must run before enrichment steps; later steps run in follow-up turns |
| `{"type": "none"}` | Prevents tool use (not in the guide's list) | Never, for extraction. It is the default when no tools are provided |

With `any` or `tool`, the API prefills the assistant turn to force the call, so the model "will not emit a natural language response or explanation before `tool_use` content blocks, even if explicitly asked to do so" ([define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools)). On models that support forced tool use, the same page pairs `any` with strict tool use to guarantee both that a tool is called and that its inputs follow the schema.

**Syntax is not semantics.** Strict JSON schemas via tool use eliminate syntax errors but do not prevent semantic errors: line items that do not sum to the total, or values placed in the wrong fields. Catching those is Task 4.4.

**Schema design rules from the guide:**

- Make a field optional (nullable) when source documents may not contain the information. A required field that the document cannot fill pushes the model to fabricate a value. Exercise 3 asks you to run documents where some fields are absent and verify that the model returns null rather than fabricating values. The [Agent SDK docs](https://code.claude.com/docs/en/agent-sdk/structured-outputs) give the same advice: "If the task might not have all the information your schema requires, make those fields optional."
- Use enums for categories, with an `"unclear"` value for ambiguous cases and an `"other"` value plus a free-text detail field for categories you did not anticipate.
- Put format normalization rules in the prompt alongside the strict schema, so inconsistent source formatting lands in one canonical form.

An illustrative extraction tool, built only from documented schema features. The nullable, `"unclear"` and `"other"` plus detail patterns are the guide's; this is not an official example:

```json
{
  "name": "extract_invoice",
  "description": "Record the fields extracted from one invoice. Use null for any field the document does not state; never infer or invent a value.",
  "strict": true,
  "input_schema": {
    "type": "object",
    "properties": {
      "invoice_number": {"type": ["string", "null"]},
      "stated_total": {"type": ["number", "null"]},
      "calculated_total": {"type": ["number", "null"]},
      "document_type": {"type": "string", "enum": ["invoice", "credit_note", "unclear", "other"]},
      "document_type_detail": {"type": ["string", "null"]}
    },
    "required": ["invoice_number", "stated_total", "calculated_total", "document_type", "document_type_detail"],
    "additionalProperties": false
  }
}
```

In this example every key stays in `required`, but each nullable field allows `null` through a type array such as `["string", "null"]`, so a value the document lacks comes back as an explicit `null`. The other way to make a field optional is to leave it out of `required`. In strict schemas each optional field counts toward a limit of 24 optional parameters per request, and the [structured outputs docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) note that "Each optional parameter roughly doubles a portion of the grammar's state space." Nullable type arrays have their own budget: 16 parameters with union types per request. So use nullable or optional fields for data that is genuinely absent, but budget them in a large strict schema. `calculated_total` sits next to `stated_total` for the self-check in Task 4.4. To also return the `conflict_detected` flag described there, add it as a boolean property (and to `required`), because `additionalProperties: false` stops the tool from returning any field it does not declare.

Normalization rules that travel with that schema in the prompt (illustrative, ours):

```text
Normalization rules for extract_invoice:
- invoice_number: copy exactly as printed, without a leading "No." or "#".
- stated_total: the grand total printed on the document, as a plain number with no currency symbol or thousands separator ("1.234,50 EUR" becomes 1234.50).
- calculated_total: the sum of the line items you read, computed by you.
- document_type: "unclear" if it could be either an invoice or a credit note; "other" plus document_type_detail if it is neither.
- Any field the document does not state: null. Never infer a value.
```

Schema facts from the [structured outputs docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), as of September 2026:

- Objects in strict schemas must set `additionalProperties: false`, and `enum` values must be strings, numbers, booleans or nulls.
- Numerical and string constraints such as `minimum`, `maximum`, `minLength` and `maxLength` are not supported, and an unsupported feature returns a 400 error. The Python, TypeScript, Ruby and PHP SDKs transform such schemas automatically: they remove the unsupported constraints from the schema sent to Claude, add the constraint to the field description, and validate the response against your original schema.
- Limits per request with `output_config.format` or `strict: true`, combined across all strict schemas: 20 strict tools, 24 optional parameters, 16 parameters with union types. Each `["string", "null"]` field counts toward the 16.
- Required properties come first in the output, then optional ones.
- Capitalization of `enum` values is not guaranteed, so compare them case-insensitively.

For Claude Code in a CI pipeline, the equivalent is `--output-format json` with `--json-schema` (Task 3.6, covered in [Domain 3](#domain-3-claude-code-configuration-workflows)); the [headless docs](https://code.claude.com/docs/en/headless) example reads the result from the `structured_output` field with `jq '.structured_output'`.

**Decide**

- If downstream code must parse every response, choose an extraction tool with a JSON schema; not a prompt that says *respond only with JSON*, because prompt-only JSON can still arrive malformed or missing required fields.
- If several extraction schemas exist and the document type is unknown, choose `tool_choice: "any"`; not `"auto"`, because `auto` lets the model answer in text.
- If one extraction must run first (metadata before enrichment), force that tool by name; not `any`, because `any` lets the model pick another tool.
- If a field may be missing from the source, make it optional or nullable; not required, because required fields invite fabricated values.
- If categories will grow or some cases are ambiguous, add `"other"` plus a detail field and `"unclear"`; not a closed enum, which leaves no honest answer for a case that fits no listed value.
- If the JSON is valid but the numbers are wrong, add validation (Task 4.4); not a stricter schema, because a schema checks shape, not meaning.

**Traps**

- Treating schema-valid output as correct output.
- Leaving `tool_choice` on `auto` when the stem needs a guaranteed structured result.
- Making every field required for completeness, which produces invented values for data the document never contained.
- Relying on prompt wording alone to fix inconsistent source formats, or on the schema alone; the guide pairs normalization rules with a strict schema.

!!! warning "Exam guide vs current docs: where the guarantee comes from"

    The guide treats tool use with a JSON schema as the guaranteed route to schema-compliant output. The current [structured outputs docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) (as of September 2026) place the guarantee in two features backed by constrained decoding: JSON outputs (`output_config.format`) for the response itself, and strict tool use (`"strict": true` on a tool definition) for tool inputs. Without strict mode, the [strict tool use page](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use) warns, "Claude might return incompatible types (`"2"` instead of `2`) or omit required fields". The [guide's appendix](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) points the same way with "strict mode for syntax error elimination". On the exam, answer in the guide's terms (tool use with JSON schemas); in production, set `strict: true` or use `output_config.format` (on Amazon Bedrock, as of September 2026, structured outputs are available only for Claude Opus 4.6, Claude Sonnet 4.6, Claude Sonnet 4.5, Claude Opus 4.5 and Claude Haiku 4.5).

!!! warning "Exam guide vs current docs: forced tool choice on the newest models"

    The guide lists three `tool_choice` options and uses `any` and forced selection to guarantee a tool call. As of September 2026 the [define tools page](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools) lists four options (adding `none`) and says "Not every model and setting supports forced tool use."

    - The [release notes](https://platform.claude.com/docs/en/release-notes/overview) for Claude Fable 5.1 and Claude Mythos 5.1 (September 1, 2026) and Claude Opus 5.5 (September 22, 2026) say `tool_choice` types `any` and `tool` return a 400 error on those models; the define tools page adds that these models reject forced tool use regardless of thinking settings. The [errors page](https://platform.claude.com/docs/en/api/errors) gives the message as `tool_choice: type "tool" and "any" are not supported for this model.` (also on the token counting endpoint). `auto` and `none` still work.
    - Manual extended thinking (`thinking: {type: "enabled"}`) also rejects `any` and `tool`. Adaptive thinking does not block forced tool use on models that support it; the docs' example is Claude Opus 5 with thinking on.
    - The documented replacement on the restricted models is `auto` with strict tool use (or structured outputs), plus a sentence in the prompt saying when the tool applies; prompting still influences which tool `auto` picks.

    The replacement call from the [Opus 5.5 migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide), verbatim:

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

    Answer exam items in the guide's terms: `any` guarantees some tool call, and a forced tool guarantees that specific call. Code you run against Claude Opus 5.5, Claude Fable 5.1 or Claude Mythos 5.1 needs the `auto` pattern above.

**Go deeper:** [Structured output with tools and JSON schemas](knowledge/prompt-engineering.md#structured-output-with-tools-and-json-schemas)

### Task 4.4: Implement validation, retry, and feedback loops for extraction quality

Task statement 4.4 in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): "Implement validation, retry, and feedback loops for extraction quality". Its bullets cover retry with specific error feedback, the cases a retry cannot fix, the difference between semantic and syntax errors, self-check fields such as `calculated_total`, and a `detected_pattern` field for analyzing dismissed review findings.

**Know**

| Error type | Examples | What removes it |
|---|---|---|
| Schema syntax error | Malformed JSON, a wrong type, a missing required field | Tool use with a strict schema (the guide); strict tool use or `output_config.format` (current docs) |
| Semantic validation error | Line items that do not sum to the total; a value placed in the wrong field | Your own validation layer (the appendix names Pydantic) catches it. A retry with the specific error can fix a misread value (our reading), but not a document that contradicts itself: flag that with `conflict_detected` |

- **Retry with error feedback.** On retry, append the specific validation errors to the prompt. The follow-up request carries three things: the original document, the failed extraction, and the specific validation errors. Exercise 3 practices this loop with Pydantic or JSON schema validation. Anthropic's [Agent SDK post](https://claude.com/blog/building-agents-with-the-claude-agent-sdk) states the principle: "The best form of feedback is providing clearly defined rules for an output, then explaining which rules failed and why."

An illustrative follow-up message (adapted, not an official example):

```text
<document>...original invoice text...</document>
<previous_extraction>{"line_items": [...], "stated_total": 500.00}</previous_extraction>
<validation_errors>
- line_items amounts sum to 450.00 but stated_total is 500.00: re-read the line items and correct any value copied into the wrong field
- invoice_date "15/01/2024" does not match the required YYYY-MM-DD format
</validation_errors>
Return a corrected extraction. If a value is not present in the document, return null for that field instead of guessing.
```

Both errors above are the kind a retry can fix (our reading), as long as the correct values are in the document the model can re-read. If the document's own line items do not add up to its printed total, no retry will reconcile them: see the self-check fields below.

- **Know when a retry cannot help.** Retries succeed for format mismatches and structural output errors. They are ineffective when the required information is simply absent from the source, for example when it exists only in an external document that was not provided. Exercise 3 asks you to track which errors are resolvable via retry (format mismatches) and which are not (information absent from the source). For an absent value, stop retrying and return null for that field (Task 4.3); whether that record then also goes to a human reviewer is a separate routing decision (Task 5.5 in [Domain 5](#domain-5-context-management-reliability)).
- **Structured-output failures a schema cannot prevent.** The [structured outputs docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) list three under Invalid outputs: a refusal, a `max_tokens` cut-off, and enum value casing (covered in the schema facts in 4.3). A response that stops with `stop_reason: "max_tokens"` may be incomplete and not match the schema; retry with a higher `max_tokens`. A refusal (`stop_reason: "refusal"`) may not match the schema, because the refusal takes precedence over schema constraints.

- **Self-check fields.** Extract `calculated_total` alongside `stated_total` to flag discrepancies, and add a `conflict_detected` boolean for source data that contradicts itself. When the two totals disagree, the record shows it. Our reading of the two cases: either a line item was misread (retry with the specific error) or the document itself is inconsistent (keep both values and set `conflict_detected`). Task 5.5 routes extractions from ambiguous or contradictory source documents to human review.

An illustrative record for the second case (ours):

```json
{"stated_total": 500.00, "calculated_total": 450.00, "conflict_detected": true}
```

- **Feedback loop for review findings.** Add a `detected_pattern` field to each structured finding, recording which code construct triggered it. When developers dismiss findings, group the dismissals by `detected_pattern`. A pattern that keeps getting dismissed is a false-positive source to fix in the prompt (Task 4.1) or to switch off for now. The field name and the dismissal analysis are the guide's own pattern; Claude Code tooling has close analogues, not the same field. The `ReportFindings` tool in the [tools reference](https://code.claude.com/docs/en/tools-reference) can attach an optional `category` slug to a finding (v2.1.199 and later), Anthropic's claude-code-security-review action records a `category` per finding and counts exclusions by reason, and in managed [Code Review](https://code.claude.com/docs/en/code-review), Anthropic collects thumbs-up and thumbs-down reaction counts after the PR merges and "uses them to tune the reviewer" (reactions change nothing on the PR itself). In managed Code Review, a developer dismisses a finding without a code change by resolving its thread; replying does not dismiss it.

An illustrative finding with the tag (ours; the severity uses the Important level from the rubric in 4.1):

```json
{
  "location": "orders/service.py:12",
  "issue": "A timeout is returned as an empty list, so callers cannot tell a failed lookup from no orders.",
  "severity": "Important",
  "suggested_fix": "Return an explicit error result and let the caller decide whether to retry.",
  "detected_pattern": "exception_swallowed_as_empty_result"
}
```

- **Built-in retry behavior differs by layer** (as of September 2026), and the docs for each layer describe its retries separately. In the Messages API, when a tool request is invalid or missing parameters and you return the error in a `tool_result`, Claude retries 2 to 3 times with corrections before apologizing to the user. Claude Code's `-p` mode with `--json-schema` allows `MAX_STRUCTURED_OUTPUT_RETRIES` attempts, which [defaults to 5](https://code.claude.com/docs/en/env-vars), "a first attempt plus four retries". The Agent SDK validates structured output against your schema, re-prompts on a mismatch, and ends with the result subtype `error_max_structured_output_retries` when no valid output remains after its retry limit. All three react to invalid tool input or schema mismatches; a semantic check, such as whether the line items sum to the total, stays in your own code.

**Decide**

- If validation fails on format or structure, choose a follow-up that includes the document, the failed extraction and the specific errors; not a bare *try again*, because the model needs the error to correct it.
- If the missing value is not in the document you sent, stop retrying and return null for that field; not more retries, because a retry cannot recover information that is absent from the source.
- If totals can disagree, extract both `stated_total` and `calculated_total` and compare; not the stated total alone, because a single value hides the discrepancy.
- If developers keep dismissing one kind of finding, tag findings with `detected_pattern`, find the pattern, then fix or disable that category; not a global *be more careful* instruction.

**Traps**

- Retrying an extraction for data that exists only in an external document that was not provided.
- Assuming tool use or strict mode catches semantic errors such as totals that do not add up.
- A retry loop that resends the original prompt without the validation errors.
- Forcing a required field and retrying until it is filled, which trades an honest null for a fabricated value.

**Go deeper:** [Validation, retry and feedback loops](knowledge/prompt-engineering.md#validation-retry-and-feedback-loops)

### Task 4.5: Design efficient batch processing strategies

Task statement 4.5 in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): "Design efficient batch processing strategies". Its bullets cover what the Message Batches API offers, which workloads fit it, submission frequency against an SLA, `custom_id` for matching results and resubmitting failures, and refining the prompt on a sample before a large run.

**Know**

Message Batches API facts from the guide and from the [batch processing docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing) (docs as of September 2026):

| Fact | What to know |
|---|---|
| Cost | 50% savings: batch usage is billed at 50% of standard API prices, on input and output tokens |
| Processing window | Up to 24 hours. A batch that has not finished within 24 hours expires, and expired requests are not billed |
| Latency | No guaranteed latency SLA. The docs say most batches finish in less than 1 hour, which describes typical behavior, not a promise |
| Correlation | Each request carries a `custom_id` (1 to 64 characters: letters, digits, hyphens, underscores). Results can come back in any order, so match them on `custom_id` |
| Size | Up to 100,000 requests or 256 MB per batch, whichever comes first. Older material says 10,000, the limit in the 2024 launch post |
| Lifecycle | `processing_status` starts as `in_progress` and becomes `ended`. Poll until it ends, then read the `.jsonl` results, which stay available for 29 days after creation |
| Result types | `succeeded`, `errored`, `canceled`, `expired`. One failed request does not affect the others |
| Tool calling | The guide: no multi-turn tool calling within a single request; tools cannot be executed mid-request with their results returned. The docs now let server tools run inside a batched request (see the note below) |
| Compatibility | Structured outputs work with batch processing |

- **Fit.** Batch processing suits non-blocking, latency-tolerant work: overnight reports, weekly audits, nightly test generation. It does not suit blocking workflows such as pre-merge checks. Match the API to the latency requirement: the synchronous API for blocking pre-merge checks, the batch API for overnight or weekly analysis. Question 11 turns on exactly this split.

- **SLA arithmetic.** The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) example is "4-hour windows to guarantee 30-hour SLA with 24-hour batch processing". Our arithmetic: a document that arrives just after a submission waits up to 4 hours for the next one, then up to 24 hours in the batch, so the worst case is 28 hours, 2 hours inside the 30-hour SLA, provided nothing expires. The [batch processing docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing) say "Batches expire if processing does not complete within 24 hours." A request that has not reached the model by then comes back `expired`, with no extraction, and needs resubmitting; under heavy demand, the docs add, "you may see more requests expiring after 24 hours." The general rule (ours, from the same arithmetic): the submission interval plus the 24-hour processing window has to fit inside the SLA.

- **Failures.** Resubmit only the failed documents, identified by `custom_id`, with the change that fixes them, for example chunking a document that exceeded the context limit. In the docs' results loop, an `errored` result with `invalid_request_error` needs its request body fixed before it is resent, while other errors can be retried directly. The [context windows docs](https://platform.claude.com/docs/en/build-with-claude/context-windows) say input that alone exceeds the model's context window gets a 400 `invalid_request_error` ("prompt is too long"). That is the error type the results loop treats as needing a fixed request body, so by our reading that document needs a change (chunking), not a plain resend. Give each chunk a meaningful `custom_id` of its own (for example `doc-0042-part-1`, which fits the allowed characters) so the pieces can be matched back. Exercise 3 practices the whole loop: a batch of 100 documents, failures handled by `custom_id`, oversized documents chunked and resubmitted, and total processing time checked against the SLA.

The results loop from the [batch processing docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing), verbatim except that the client setup line is left out:

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

- **Refine on a sample first.** Tune the prompt on a sample set before batch-processing large volumes, to raise the first-pass success rate and cut the cost of resubmitting. The docs give a related reason to test first: each request's `params` are validated asynchronously and validation errors come back only when the whole batch has ended, so they recommend dry-running the request shape with the Messages API first.

- **No pricing math.** The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) lists "Rate limiting, quotas, or API pricing calculations" among its [out-of-scope topics](#what-is-out-of-scope). The 50% figure is what the exam uses.

**Decide**

- If someone waits on the result (a pre-merge check), choose the synchronous API; not the Message Batches API, because batches carry no latency guarantee and can take up to 24 hours.
- If the work can wait (an overnight technical-debt report, a weekly audit, nightly test generation), choose the Message Batches API for the 50% saving; not real-time calls.
- If you must meet a completion SLA, choose a submission interval so that the interval plus 24 hours fits inside it; not a schedule that assumes batches usually finish within an hour.
- If some requests fail, resubmit only those `custom_id` values with the fix; not the whole batch.
- If each document needs tool calls whose results your code must supply mid-conversation, choose synchronous calls in an agent loop; not a single batch request.

**Traps**

Question 11's rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) rejects three tempting designs:

- Moving a blocking check to batch processing with status polling. Typical completion times are not a guarantee, and a blocking workflow cannot rely on them.
- Keeping every workflow real-time to avoid batch result ordering issues. The rationale calls this a misconception, because `custom_id` correlates results.
- Adding a timeout fallback from batch to real-time. The rationale calls it unnecessary complexity when the simpler answer is to match each workflow to the right API.

A fourth trap, from Task 4.5 itself: resubmitting an entire batch because a few of its requests failed.

!!! warning "Exam guide vs current docs: tool calling in batches"

    The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says "The batch API does not support multi-turn tool calling within a single request (cannot execute tools mid-request and return results)". The current [batch processing docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing) (as of September 2026) list tool use, including all server tools, and multi-turn conversations as batchable, and say "The batch worker runs the same server-side agentic loop as the synchronous Messages API." Both hold once you separate tool types. Server tools such as web search and code execution run on Anthropic's side inside the request. A client tool needs your code to run it, and the [tool use docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works) describe every such call as "a round trip: the model asks, you execute, you report back, the model continues". The batch page does not say what a batched request that calls a client tool returns. Answer exam items in the guide's terms: no client-side tool round trips inside a batch request.

**Go deeper:** [Batch processing design](knowledge/prompt-engineering.md#batch-processing-design)

### Task 4.6: Design multi-instance and multi-pass review architectures

Task statement 4.6 in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): "Design multi-instance and multi-pass review architectures". Its bullets cover why a session reviewing its own output is weak, independent review instances, per-file passes plus a cross-file integration pass, and verification passes that report confidence for each finding.

**Know**

- **Why self-review is weak.** A model retains reasoning context from generation, which makes it less likely to question its own decisions in the same session. Task 3.6 makes the same point for CI: the session that generated code is less effective at reviewing its own changes than an independent review instance (see [Domain 3](#domain-3-claude-code-configuration-workflows)).
- **Use an independent instance.** A second Claude instance, without the generator's reasoning context, is more effective at catching subtle issues than self-review instructions or extended thinking. A practical checklist (ours, assembled from this domain): give it the artifact (the diff), the review criteria (4.1), examples (4.2) and the output schema (4.3), and do not give it the generator's conversation.
- **Anthropic's own tools are built this way.** The [security guidance plugin](https://code.claude.com/docs/en/security-guidance) "does not ask the same Claude instance that wrote the code to grade itself"; its reviewer "starts from the diff, has no investment in the original approach, and is instructed only to find problems". The [Claude Fable 5 prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5) says "Separate, fresh-context verifier subagents tend to outperform self-critique." In Claude Code, a subagent starts with a fresh, isolated context window and does not see your conversation history. The exception is a fork of the current conversation, which inherits it, so a fork is not an independent reviewer. The local `/code-review` command (v2.1.218 and later) reviews a diff, by default, in a background subagent with its own context window. The [Code Review docs](https://code.claude.com/docs/en/code-review) also call it a "forked subagent", but that phrase links to skills that run with `context: fork`, and the [skills docs](https://code.claude.com/docs/en/skills) say such a skill does not run in a fork of the current conversation: "The subagent doesn't see your conversation history".

- **Split big reviews into passes.** Run per-file passes for local issues, plus a separate integration pass for cross-file data flow. This avoids the attention dilution and contradictory findings of a single pass over many files: in Question 12, one pass over 14 files gave detailed feedback on some files and superficial comments on others, and flagged a pattern in one file while approving identical code in another. Anthropic's [agent design guidance](https://www.anthropic.com/engineering/building-effective-agents) gives the general reason: "LLMs generally perform better when each consideration is handled by a separate LLM call".

An illustrative layout for the Question 12 pull request (our sketch of the per-file, integration and verification passes that Task 4.6 names):

```text
Pull request: 14 changed files
  |
  |-- Passes 1 to 14: one file each, local issues only
  |
  |-- Integration pass: cross-file data flow
  |     (callers and callees, shared types, values passed between files)
  |
  '-- Verification pass: re-check each candidate finding,
        attach a confidence, route by a calibrated threshold
```

- **Verification pass with confidence.** Have the model report a confidence alongside each finding so review effort can be routed. The recommended finding-stage prompt in the [Sonnet 5 prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5) asks the model to "include your confidence level and an estimated severity so a downstream filter can rank them". Calibrate before you trust it: the guide rejects raw self-reported confidence as an escalation signal (Task 5.2 and Question 3) but uses confidence calibrated on labeled validation sets to route review (Task 5.5, in [Domain 5](#domain-5-context-management-reliability)).

**Decide**

- If the session that wrote the code also reviews it and misses bugs, choose a second, independent instance that sees only the diff and the criteria; not an instruction to *review your work carefully* in the same session or more thinking, because the generator's reasoning context is the problem.
- If a large multi-file review is inconsistent or contradicts itself, choose per-file passes plus an integration pass; not a bigger context window, not majority voting across repeated full passes, not forcing developers to split pull requests.
- If reviewer time is limited, choose a verification pass that attaches a confidence to each finding and route on it, with review thresholds calibrated on a labeled validation set (Task 5.5); not raw, uncalibrated self-reported confidence.

**Traps**

Question 12's rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) explains why the tempting alternatives fail: "larger context windows don't solve attention quality issues", and requiring agreement across repeated runs "would actually suppress detection of real bugs by requiring consensus on issues that may only be caught intermittently".

- A higher-tier model with a larger context window, to cover all files in one pass.
- Three independent full passes, flagging only issues that appear at least twice.
- Asking developers to split pull requests into a few files each, which shifts the burden to them without improving the system.
- Self-review instructions or extended thinking in place of an independent reviewer.
- Treating raw self-reported confidence as calibrated.

!!! warning "Exam guide vs current docs: extended thinking"

    The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) compares independent review with "extended thinking". The current [API errors page](https://platform.claude.com/docs/en/api/errors) (as of September 2026) says "Claude 4.7 and later models have removed extended thinking." (the manual `thinking: {type: "enabled"}` mode), and on Claude Opus 5.5, Claude Fable 5.1 and Claude Mythos 5.1, among others, thinking is always on and adaptive. Our reading: treat the guide's phrase as *the same model reasoning longer in the same context*; the exam's answer, an independent instance, does not change. A related nuance: the [Claude Opus 5 prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5) says Opus 5 verifies its own work unprompted and that explicit verification instructions, including "use a subagent to verify", cause over-verification on that model. The same section extends this to harness design: "The same applies to legacy harness scaffolding that adds separate verification steps." Its stated reason is wasted tokens on a model that already checks its own work, and the same page credits Opus 5 with "effective writer-verifier patterns" when it coordinates subagents. That is cost advice specific to one model, not an exam rule: answer in the exam guide's terms (an independent review instance).

**Go deeper:** [Multi-instance and multi-pass review](knowledge/prompt-engineering.md#multi-instance-and-multi-pass-review)

## Domain 5: Context Management & Reliability

**Official weight: 15%**, which the guide defines as the approximate proportion of scored items drawn from this domain. That is about 9 items if applied to all 60 (15% of 60, our arithmetic); the guide does not say how many of the 60 are scored.

Domain 5 carries the lowest weight on the blueprint, yet it is a primary domain in four of the six scenarios: Customer Support Resolution Agent, Code Generation with Claude Code, Multi-Agent Research System and Structured Data Extraction. Its six task statements hold 24 "Knowledge of" and 29 "Skills in" bullets, 53 in all, more than any other domain (our count: Domains 1 to 4 hold 48, 43, 49 and 47) ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The guide's sample questions carry no domain labels, but by our mapping Question 3 turns on escalation (Task 5.2) and Question 8 on error propagation (Task 5.3); both are reproduced under [Official sample questions](#official-sample-questions). Preparation Exercises 1, 3 and 4 each list Domain 5 among the domains they reinforce.

| Task | The decision the exam keeps asking for |
|---|---|
| 5.1 | Keep exact facts outside summarized history, trim tool output before it piles up, put key findings where the model reads reliably |
| 5.2 | Escalate on an explicit request, a policy gap or no progress; resolve what policy covers; ask for identifiers instead of guessing |
| 5.3 | Recover locally from transient failures, then return structured error context; never disguise a failure as an empty success |
| 5.4 | Isolate verbose exploration in subagents, persist findings in files, carry phase summaries forward, compact deliberately |
| 5.5 | Check accuracy by document type and field, sample high-confidence output, calibrate thresholds on labeled data |
| 5.6 | Carry claim-source mappings and dates through synthesis; annotate conflicts instead of choosing a value |

### Task 5.1: Manage conversation context to preserve critical information across long interactions

**What the guide tests:** four knowledge bullets on how long contexts lose or bury facts, and six skills for keeping exact values and findings intact as conversations and agent pipelines grow ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

**Know**

- **Summaries blur exact values.** The guide's first knowledge bullet warns about "condensing numerical values, percentages, dates, and customer-stated expectations into vague summaries" ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Anthropic's context engineering cookbook says the same of compaction: the summary "may drop specific numbers or exact phrasing" ([cookbook](https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools)). The context editing docs list tasks that "need to maintain exact state across many variables" among those less suited to compaction ([context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing)).

- **The API remembers nothing between calls.** "The Messages API is stateless, which means that you always send the full conversational history to the API" ([working with messages](https://platform.claude.com/docs/en/build-with-claude/working-with-messages)). The guide's bullet on passing complete conversation history in subsequent requests, which it ties to conversational coherence, follows from this (our reading): the model works only from what the request contains. Everything in the request counts toward the context window: the system prompt, every message including tool results, and the tool definitions ([context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows)).

```python
history.append({"role": "user", "content": question})
response = client.messages.create(
    model="claude-opus-5-5", max_tokens=8192, system=SYSTEM, messages=history,
)
history.append({"role": "assistant", "content": response.content})
```

The loop above is adapted from the on-demand compaction example in Anthropic's docs, with the compaction beta and the summarizing step taken out: every call sends the whole `history` list, and each reply is appended before the next question ([on-demand compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand)).

- **Tool results pile up.** The guide's example is an order lookup returning "40+ fields per order lookup when only 5 are relevant" ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Because the full history is resent, every one of those fields counts toward the context window again on each later request. Anthropic's tool-design advice is to return high-signal information and leave out low-level identifiers such as `uuid` or `mime_type`; Claude Code restricts tool responses to 25,000 tokens by default ([Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)). As of September 2026, the Claude Code docs state the same default for MCP tool output: a warning when output exceeds 10,000 tokens, a default maximum of 25,000 tokens, and `MAX_MCP_OUTPUT_TOKENS` to change it. A result with no image content that exceeds the limit is saved to a file and replaced in the conversation with a message naming the file path, so Claude reads the file only when it needs the content; an MCP server author can raise one tool's threshold with `_meta["anthropic/maxResultSizeChars"]`, up to a hard ceiling of 500,000 characters ([Claude Code MCP](https://code.claude.com/docs/en/mcp)).

- **Trim before the result enters history.** The guide's skill is keeping only the relevant fields (its example: return-relevant fields from order lookups) before they accumulate. You can do that in the tool itself, or with an Agent SDK `PostToolUse` hook that replaces the tool's output before Claude sees it by setting `updatedToolOutput` (as of September 2026; the hook mechanics are under [Task 1.5](#task-15-apply-agent-sdk-hooks-for-tool-call-interception-and-data-normalization)). Anthropic also suggests a `response_format` parameter with `concise` and `detailed` options; in its Slack example the concise responses used about a third of the tokens ([Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)).

- **Position matters.** The guide defines the "lost in the middle" effect: models "reliably process information at the beginning and end of long inputs but may omit findings from middle sections". Its matching skill places a key-findings summary "at the beginning of aggregated inputs" and organizes the detail under explicit section headers ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

!!! warning "Exam guide vs current docs: position effects"

    - **Exam guide (July 2026):** mitigate position effects with a key-findings summary at the start of aggregated inputs and explicit section headers for the detail.
    - **Anthropic's prompting docs today:** for long inputs (20k+ tokens), "Place your long documents and inputs near the top of your prompt, above your query, instructions, and examples", and "Queries at the end can improve response quality by up to 30 percent in tests" ([prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)). Anthropic's blog says improved context awareness in Claude 4.x helps with historical lost-in-the-middle issues, and still advises putting "the most critical details at the beginning or end" ([blog](https://claude.com/blog/best-practices-for-prompt-engineering)).
    - **On the exam, answer in the guide's terms:** summary first, detail under headers. The two pieces of advice can be followed together (our reading): the aggregated input, opening with its summary, sits above the question.

- **Case facts.** The guide's first skill extracts transactional facts (amounts, dates, order numbers, statuses) into a persistent case-facts block that is included in each prompt, outside the summarized history. Its second skill, for sessions that cover several issues, keeps structured issue data (order IDs, amounts, statuses) in a separate context layer. The name "case facts" comes from the guide ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The Anthropic docs cited below do not use that name; the product features closest to it (our mapping) are compaction instructions that say what to keep (a "Compact Instructions" section in CLAUDE.md, or the `instructions` string on API compaction, which replaces the default summarization prompt entirely and may be up to 16,384 characters on an on-demand compaction request), a `SessionStart` hook with the `compact` matcher that re-injects critical context after every compaction, and the project-root CLAUDE.md, which Claude Code re-injects from disk after compaction ([how Claude Code works](https://code.claude.com/docs/en/how-claude-code-works), [on-demand compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand), [hooks guide](https://code.claude.com/docs/en/hooks-guide), [context window](https://code.claude.com/docs/en/context-window)). For client-side compaction, Anthropic's prompting guide for Claude Fable 5.1 supplies a summarization instruction that asks for specific details ("names, numbers, dates, exact wording, links or references") to be kept exactly ([Prompting Claude Fable 5.1](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-1)).

```text
<case_facts>
customer_id: {{VERIFIED_CUSTOMER_ID}}
issue_1: order {{ORDER_1}} | amount {{AMOUNT_1}} | status {{STATUS_1}}
issue_2: order {{ORDER_2}} | amount {{AMOUNT_2}} | status {{STATUS_2}}
customer_expectation: {{STATED_EXPECTATION_IN_CUSTOMER_WORDS}}
</case_facts>
```

The block above is an illustrative layout of ours. Delimiting it with XML tags follows Anthropic's advice to organize prompts into distinct sections marked with XML tags or Markdown headers ([effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)).

- **Multi-agent handoffs are context too.** Two skills apply 5.1 to pipelines: require subagents to include metadata (dates, source locations, methodological context) in structured outputs, and, when downstream agents have limited context budgets, change the upstream agents to return structured data (key facts, citations, relevance scores) instead of verbose content and reasoning chains. In the Agent SDK only a subagent's final message returns to the parent, and the parent may summarize it further ([Agent SDK subagents](https://code.claude.com/docs/en/agent-sdk/subagents)). Anthropic describes subagents that explore with tens of thousands of tokens but return condensed summaries, often 1,000 to 2,000 tokens ([effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)).

**Decide**

- If exact values (amounts, dates, order numbers, statuses, what the customer said they expect) must survive a long session, choose a persistent case-facts block sent with every prompt outside the summarized history; not a better summary alone, because summarization is where those values blur.
- If one session covers several issues, choose structured per-issue data in its own context layer; not one running narrative, so each issue's order ID, amount and status stays attached to that issue.
- If a tool returns far more fields than the task needs, choose trimming to the relevant fields before the result is appended; not keeping the whole payload *just in case*, because every retained field is resent with every later request.
- If a long aggregated input feeds a decision, choose a key-findings summary at the top and explicit section headers; not trusting the model to surface a finding from the middle.
- If a downstream agent has a tight context budget, choose to change what the upstream agent returns (key facts, citations, relevance scores); not forwarding verbose content and reasoning chains.
- If a synthesis step depends on subagent findings, choose requiring subagents to include metadata (dates, source locations, methodological context) in their structured outputs; not bare conclusions the synthesizer cannot place or weigh.
- If a follow-up call is missing earlier turns, choose sending the complete history; the Messages API is stateless.

**Traps**

- *Move to a model with a larger context window.* The rationale for sample Question 12 rejects this because "larger context windows don't solve attention quality issues" ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)), and Anthropic expects context windows of all sizes to remain subject to context pollution and relevance concerns ([effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)).
- *Summarize the history more aggressively* as the cure for lost numbers: that is the failure the first knowledge bullet describes.
- Sending only the newest user message to save tokens.
- Passing a subagent's full transcript or reasoning chain downstream *for completeness* when the next agent's budget is limited.

**Go deeper:** [Preserving critical information in long conversations](knowledge/context-engineering.md#preserving-critical-information-in-long-conversations)

### Task 5.2: Design effective escalation and ambiguity resolution patterns

**What the guide tests:** when an agent hands a case to a human and when it resolves the case itself, plus what it does when a lookup is ambiguous. Scenario 1 frames it: an agent with an `escalate_to_human` tool and a target of "80%+ first-contact resolution while knowing when to escalate" ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

**Know**

The guide's escalation rules, situation by situation:

| Situation | What the guide wants |
|---|---|
| The customer explicitly asks for a human | Escalate immediately, without first attempting investigation |
| The customer is frustrated, but the issue is straightforward and within the agent's capability | Acknowledge the frustration and offer to resolve; escalate only if the customer reiterates the preference |
| Policy is ambiguous or silent on the request (the guide's example: competitor price matching when policy only addresses own-site adjustments) | Escalate |
| The case is complex, but policy covers it | Resolve; the trigger is a policy exception or gap, not complexity on its own |
| The agent cannot make meaningful progress | Escalate |
| A customer lookup returns several matches | Ask for additional identifiers; do not select one by heuristic |

- **The fix for miscalibrated escalation.** The guide's first skill adds explicit escalation criteria to the system prompt, with few-shot examples showing when to escalate and when to resolve autonomously. The rationale for sample Question 3 calls this the fix for the root cause, "unclear decision boundaries", and "the proportionate first response before adding infrastructure"; it rejects a separately trained classifier as over-engineered when prompt optimization has not been tried ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Anthropic's human-in-the-loop cookbook states the cost of getting it wrong in either direction: "an agent that escalates everything is exhausting to work with, and an agent that escalates nothing is dangerous" ([cookbook notebook](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/managed_agents/CMA_gate_human_in_the_loop.ipynb)). Anthropic's customer support guide sets an escalation-accuracy target of 95% or higher ([customer support guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/customer-support-chat)).

- **Sentiment measures something else.** Question 3's rationale: "sentiment doesn't correlate with case complexity, which is the actual issue" ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Anthropic's [customer support guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/customer-support-chat) tracks sentiment as a business metric (maintained or improved in 90% of interactions) and does not use it as an escalation trigger. Anthropic's ticket-routing guide warns that with dissatisfied customers "Claude may prioritize addressing the emotion over solving the underlying problem", and the documented fix is telling Claude when to prioritize sentiment and when not to ([ticket routing guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/ticket-routing)).

- **Ask, do not guess.** When a lookup returns several customers, the guide wants the agent to request additional identifiers rather than pick one. In the Agent SDK, Claude asks clarifying questions with the `AskUserQuestion` tool (1 to 4 questions, 2 to 4 options each), which is "not currently available in subagents spawned via the Agent tool" ([Agent SDK user input](https://code.claude.com/docs/en/agent-sdk/user-input)). A subagent therefore cannot use that tool to put the question to the user itself; in our reading, the clarification has to route through the parent agent that talks to the customer.

- **Judgment calls versus hard limits.** Escalation criteria in the prompt steer judgment. A rule that must hold every time, such as blocking operations above a threshold amount and redirecting to an escalation workflow (Preparation Exercise 1, step 4), belongs in a programmatic hook; the rationale for sample Question 1 explains that programmatic enforcement "provides deterministic guarantees that prompt-based approaches cannot" ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Enforcement patterns are taught under [Domain 1](#domain-1-agentic-architecture-orchestration).

- **"Escalation" has two meanings in Anthropic docs.** In the Managed Agents multiagent docs it means consulting a more capable agent or model for complex subtasks ([multiagent orchestration](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration)). On this exam it means handing the case to a human.

!!! note "One guide, two positions on model-reported confidence"

    The guide treats a model's own confidence score in two ways, and an item can come from either side ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

    - **Unreliable as an escalation trigger.** A Task 5.2 knowledge bullet covers "Why sentiment-based escalation and self-reported confidence scores are unreliable proxies for actual case complexity". Question 3's rationale rejects routing to humans on a self-reported 1 to 10 score because "LLM self-reported confidence is poorly calibrated" and "the agent is already incorrectly confident on hard cases".
    - **Usable for review routing once calibrated.** A Task 4.6 skill runs "verification passes where the model self-reports confidence alongside each finding to enable calibrated review routing". Task 5.5 has models output field-level confidence scores and then calibrate review thresholds "using labeled validation sets". Preparation Exercise 3 and the How to Prepare section also ask for confidence-based routing to human review.
    - **How the two fit** (our reading; it contradicts none of the bullets above): a raw self-reported score does not decide whether a case needs a human, because escalation follows explicit criteria. A field-level score that has been checked against labeled outcomes can decide where scarce review effort goes. Anthropic's research gives a reason to calibrate first: models "struggle with calibration of P(IK) on new tasks" ([Language models (mostly) know what they know](https://www.anthropic.com/research/language-models-mostly-know-what-they-know)).
    - **Exam rule:** reject an option that escalates on a raw confidence threshold or on sentiment; accept an option that routes review attention on field-level confidence calibrated against a labeled validation set.

**Decide**

- If the customer explicitly asks for a human, choose escalating now; not *let me look into it first*, because the guide's skill is honoring the request without first attempting investigation.
- If the customer is frustrated but the issue is straightforward and within the agent's capability, choose acknowledging the frustration and offering to resolve; escalate if they reiterate the request. Not escalating on negative sentiment alone.
- If policy is ambiguous or silent on what the customer asks for, choose escalation; not improvising a decision the policy does not cover.
- If a case is complex but squarely covered by policy, choose resolving it; not escalating by default, because the trigger is policy exceptions or gaps, not complexity alone.
- If escalation is miscalibrated in both directions, choose explicit criteria with few-shot examples in the system prompt as the first fix; not a confidence threshold, a sentiment trigger, or a new classifier.
- If a lookup returns multiple matching customers, choose asking for another identifier; not selecting the likeliest match.

**Traps**

- Escalating whenever sentiment turns negative, or whenever self-reported confidence drops below a threshold (both rejected in Question 3's rationale).
- A classifier trained on historical tickets as the first response, before the prompt has explicit criteria.
- Investigating before honoring an explicit request for a human.
- Treating *complex* as *escalate*: the trigger is a policy exception or gap.
- Choosing among several matching accounts by a heuristic instead of asking.

**Go deeper:** [Escalation and ambiguity](knowledge/evaluation-and-reliability.md#escalation-and-ambiguity)

### Task 5.3: Implement error propagation strategies across multi-agent systems

**What the guide tests:** what a subagent should send the coordinator when something fails, and which two failure-handling habits are anti-patterns: "silently suppressing errors (returning empty results as success)" and "terminating entire workflows on single failures" ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

**Know**

- **What the coordinator needs.** Structured error context: the failure type, what was attempted (for a search, the query), any partial results, and potential alternatives. The rationale for sample Question 8 says this lets the coordinator decide "whether to retry with a modified query, try an alternative approach, or proceed with partial results". A generic status such as "search unavailable" hides that context ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

- **Two outcomes that must never look alike.** An access failure (a timeout that needs a retry decision) and a valid empty result (the query succeeded and found nothing) call for different coordinator actions, so the report must say which one happened. The MCP resources spec applies the same logic: a server must not return an empty `contents` array for a resource that does not exist, because "An empty array is ambiguous" ([MCP resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources.md)). [Task 2.2](#task-22-implement-structured-error-responses-for-mcp-tools) shows a real case of the two getting confused: before Claude Code v2.1.208, a Grep pattern that ripgrep rejected came back as `No files found`.

- **How errors travel.** A failed tool call comes back inside the tool result with `isError: true` in MCP (`"is_error": true` on a Messages API `tool_result`), not as a protocol error, so the model can see it and self-correct. The two error channels, the per-layer spellings and the guide's `errorCategory` and `isRetryable` convention are taught under [Task 2.2](#task-22-implement-structured-error-responses-for-mcp-tools). This is the MCP spec's own example of a tool execution error ([MCP tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools.md)):

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "resultType": "complete",
    "content": [
      {
        "type": "text",
        "text": "Invalid departure date: must be in the future. Current date is 08/08/2025."
      }
    ],
    "isError": true
  }
}
```

The example comes from the 2026-07-28 revision; `resultType` is required from that revision on, and clients treat a missing `resultType` from older servers as `"complete"` ([MCP basic protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic/index.md)). A subagent's report to its coordinator can carry the four elements the guide names; the shape below is illustrative, with our field names:

```json
{
  "status": "failed",
  "failure_type": "timeout",
  "attempted": "<query or action the subagent ran, and any local retries>",
  "partial_results": ["<findings gathered before the failure>"],
  "alternatives": ["<other query, source or agent the coordinator could try>"]
}
```

- **Local recovery first.** The guide's third skill has subagents handle transient failures themselves and propagate only the errors they cannot resolve, together with what was attempted and the partial results. Task 2.2, on structured error responses for MCP tools, states the same rule for subagents (2.2-S3), so it can be tested under either domain. Anthropic's research-subagent prompt tells the subagent to try another tool or query when an approach is not working and never to repeat the exact same query with the same tool ([research subagent prompt](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_subagent.md)), and Anthropic's research post reports that "letting the agent know when a tool is failing and letting it adapt works surprisingly well" ([multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)).

- **What Claude Code does today (as of September 2026).** Since v2.1.199, a subagent whose run ends on an API error "reports that failure back to Claude instead of returning the error text as if it were the subagent's findings". What Claude receives depends on where the subagent ran. For a foreground subagent that already produced text, a rate limit, overload or server error makes the Agent tool return that partial output with a note that the subagent was cut off and did not finish; a foreground subagent that produced nothing, or only tool calls, fails with `Agent terminated early due to an API error` followed by the error detail (in v2.1.199 itself, a cut-off tool-calls-only run instead returned an empty partial result holding only the cut-off note). A background subagent is marked failed, and the message Claude receives names the API error and includes the subagent's last output, so partial work is not lost. Some local recovery is built in: when a response is cut off mid-stream with text but no tool calls, Claude Code prompts the subagent to continue, and with a fallback model chain configured, a failure the chain covers (such as the model being unavailable) switches the subagent to the first model in the chain that accepts the request ([Claude Code subagents](https://code.claude.com/docs/en/sub-agents)). In the Agent SDK, an API error that ends a subagent early "is never delivered as its result", and output at a subagent's `maxTurns` limit is marked partial so Claude knows the run is unfinished ([Agent SDK subagents](https://code.claude.com/docs/en/agent-sdk/subagents)).

- **Coverage annotations.** The guide's fourth skill structures synthesis output to show which findings are well supported and which topic areas have gaps because sources were unavailable. Preparation Exercise 4, step 4 simulates a subagent timeout and checks that the coordinator can proceed with partial results and annotate the final output with coverage gaps. As of September 2026, Claude Code's bundled `/deep-research` workflow shows the same distinction: when verifier agents cannot check a claim, "the report lists that claim as unverified instead of counting it as refuted" ([Claude Code workflows](https://code.claude.com/docs/en/workflows)).

**Decide**

- If a subagent hits a transient failure such as a timeout, choose local retry first, then, if it still fails, structured error context to the coordinator (failure type, attempted query, partial results, alternatives); not a generic status, not an empty result marked successful, not terminating the workflow.
- If a query ran and matched nothing, choose reporting a successful empty result; not an error that triggers pointless retries.
- If some sources stayed unavailable, choose proceeding with partial results and annotating the coverage gaps in the output; not presenting the report as complete, and not aborting the run.
- If a tool reports its outcome as plain text (Anthropic's [search results docs](https://platform.claude.com/docs/en/build-with-claude/search-results) suggest a plain text block instead of raising an error when a search "fails or returns nothing"), choose wording that still says which of the two happened (our reading, consistent with the guide's access-failure rule).

**Traps**

The rationale for sample Question 8 names the three wrong shapes ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)):

- Retry with backoff inside the subagent, then only a generic "search unavailable" status once retries run out. Local retry is not the problem (our reading), since the guide's own local-recovery skill asks for it; the trap is the bare status afterwards, because the "generic status hides valuable context from the coordinator".
- Catch the timeout and return an empty result marked successful: this "suppresses the error by marking failure as success".
- Propagate the exception to a top-level handler that ends everything: this "terminates the entire workflow unnecessarily when recovery strategies could succeed".

**Go deeper:** [Error propagation in multi-agent systems](knowledge/evaluation-and-reliability.md#error-propagation-in-multi-agent-systems)

### Task 5.4: Manage context effectively in large codebase exploration

**What the guide tests:** recognizing context degradation in long exploration sessions and the tools against it: subagents, scratchpad files, phase summaries, crash-recovery manifests and `/compact` ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

**Know**

- **The symptom.** In extended sessions the model starts giving inconsistent answers and talks about typical patterns instead of the specific classes it discovered earlier (the guide's first knowledge bullet). Claude Code's best practices rest on the same constraint: "Claude's context window fills up fast, and performance degrades as it fills" ([Claude Code best practices](https://code.claude.com/docs/en/best-practices)).

- **Subagents isolate exploration.** The guide's skill is spawning subagents for specific questions (its examples: "find all test files" and "trace refund flow dependencies") while the main agent keeps the high-level picture ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). In Claude Code, subagents "run in separate context windows and report back summaries" ([Claude Code best practices](https://code.claude.com/docs/en/best-practices)). The built-in Explore subagent is read-only (Write and Edit are denied), and "Claude delegates to Explore when it needs to search or understand a codebase without making changes. This keeps exploration results out of your main conversation context." ([Claude Code subagents](https://code.claude.com/docs/en/sub-agents)). The same page warns that "Running many subagents that each return detailed results can consume significant context."

- **Subagents start blank.** A non-fork subagent "doesn't see your conversation history, the skills you've already invoked, or the files Claude has already read" ([Claude Code subagents](https://code.claude.com/docs/en/sub-agents)); what an SDK subagent does receive is covered under [Task 1.3](#task-13-configure-subagent-invocation-context-passing-and-spawning). The guide's phase skill fills this gap (our reading): summarize the key findings from one exploration phase before spawning subagents for the next, and inject those summaries into their initial context. Anthropic's research post describes the same move: when context limits approach, agents can spawn fresh subagents with clean contexts "while maintaining continuity through careful handoffs" ([multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)).

- **Scratchpad files.** The guide's scratchpad bullets have agents write key findings to a file that persists across context boundaries and consult it for later questions. Anthropic calls this structured note-taking, or agentic memory: notes persisted outside the context window and pulled back in later, such as a custom agent's NOTES.md ([effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)). A Managed Agents cookbook tells its exploring agent: "Write notes to /tmp/NOTES.md as you go." ([cookbook notebook](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/managed_agents/CMA_explore_unfamiliar_codebase.ipynb)). On the API, the memory tool supports the same pattern. It operates client-side (Claude requests file operations and your application executes them), and its whole configuration is the `tools` entry `{"type": "memory_20250818", "name": "memory"}`. When it is present, the API automatically adds a memory protocol to the system prompt that includes "ASSUME INTERRUPTION: Your context window might be reset at any moment" ([memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)).

- **Crash recovery with manifests.** The guide's design: each agent exports its state to a known location, and on resume the coordinator loads a manifest and injects the saved state into the agents' prompts. "Manifest" is the guide's term ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Anthropic's own long-running harness recovers state in a fresh context window from a `claude-progress.txt` file plus the git history, and keeps its feature list in JSON because the model is less likely to inappropriately change or overwrite JSON than Markdown ([effective harnesses](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)). The related session rule, that a new session with a structured summary is more reliable than resuming with stale tool results, is tested under [Task 1.7](#task-17-manage-session-state-resumption-and-forking), with the Agent SDK's matching advice.

- **`/compact`.** Its description in the command reference: "Free up context by summarizing the conversation so far. Optionally pass focus instructions for the summary." In the same reference, `/clear` starts a new conversation with empty context instead, and `/context` visualizes what is using the window ([Claude Code commands](https://code.claude.com/docs/en/commands)). The docs suggest compacting with a focus, for example `/compact focus on the auth bug fix`, before starting a long new task ([context window](https://code.claude.com/docs/en/context-window)). As of September 2026, if one large file or output refills the context after every summary, Claude Code stops auto-compacting after a few attempts and shows an error instead of looping ([how Claude Code works](https://code.claude.com/docs/en/how-claude-code-works)).

What survives compaction in Claude Code, as of September 2026 ([context window](https://code.claude.com/docs/en/context-window)):

| Item | After compaction |
|---|---|
| System prompt and output style | Both still apply |
| Project-root CLAUDE.md and unscoped rules | Re-injected from disk |
| Auto memory | Re-injected from disk |
| The plan written in plan mode | Re-injected from disk |
| Files Claude read or edited | Up to five re-read, choosing the most recently modified; a file over 5,000 tokens comes back as a path reference without its content |
| Invoked skill bodies | Re-injected, capped at 5,000 tokens per skill and 25,000 tokens total; oldest dropped first |
| Path-scoped rules and nested CLAUDE.md files | Summarized away; they reload when a matching file is read again |
| Background commands and background subagents | Keep running; Claude Code reminds Claude which ones are still running so it does not start a duplicate |
| Context that hooks added earlier | Summarized with the rest of the conversation |
| `SessionStart` hooks that match the `compact` source | Claude Code runs them and adds their output to the compacted context |
| Instructions given only in the conversation | May be lost; add them to CLAUDE.md to make them persist ([memory](https://code.claude.com/docs/en/memory)) |

Subagent transcripts are stored in separate files and are unaffected when the main conversation compacts ([Claude Code subagents](https://code.claude.com/docs/en/sub-agents)).

!!! warning "Exam guide vs current docs: the Task tool"

    The guide calls the subagent-spawning tool "Task"; as of September 2026 it is named Agent (see the warning under [Task 1.3](#task-13-configure-subagent-invocation-context-passing-and-spawning)). On the exam, answer in the guide's terms (Task); in code you write today, use `Agent`.

**Decide**

- If a question needs verbose searching (find all test files, trace a flow's dependencies), choose a subagent that investigates and returns a summary; not running the search in the main context.
- If findings must outlive a context boundary (compaction, a new session, the next phase), choose writing them to a scratchpad file the agent consults for later questions; not trusting the conversation to keep them.
- If answers start drifting into generic patterns late in a session, choose re-grounding from the scratchpad or compacting with a focus; not continuing to question the degraded session.
- If the work moves to a new exploration phase, choose summarizing the phase's key findings and putting that summary into the next subagents' prompts; not assuming the subagents inherit what the coordinator learned.
- If a long multi-agent run can crash, choose per-agent state exports to a known location plus a manifest the coordinator loads and injects on resume; not relying on in-memory state or restarting from zero.
- If the context is filling with verbose discovery output and you still need continuity, choose `/compact`, ideally with a focus; if you need a clean start rather than continuity, `/clear`.

**Traps**

- Expecting a subagent to know what the parent already read or decided.
- Asking subagents to return everything they saw, which floods the coordinator's context.
- Relying on path-scoped rules or early chat instructions to survive compaction.
- Resuming a session whose tool results are stale when a fresh session with a structured summary would be more reliable (tested under [Domain 1](#domain-1-agentic-architecture-orchestration)).

**Go deeper:** [Exploring large codebases](knowledge/context-engineering.md#exploring-large-codebases)

### Task 5.5: Design human review workflows and confidence calibration

**What the guide tests:** deciding where limited human review goes in an extraction pipeline, and proving that automation is safe segment by segment before reducing review ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

**Know**

- **Aggregates hide segments.** The guide warns that aggregate accuracy metrics "(e.g., 97% overall) may mask poor performance on specific document types or fields" ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). A document type that makes up a small share of volume can fail often and still barely move the overall figure (our reasoning). So the guide requires validating accuracy by document type and by field before automating high-confidence extractions or reducing human review. Anthropic's eval guidance agrees that "Most use cases need multidimensional evaluation along several success criteria" ([develop tests](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests)), and its ticket-routing guide sets segment-level targets: at most a 5 to 10% accuracy drop for non-primary languages, at least 80% accuracy on a dedicated edge-case set, and routing accuracy within 2 to 3% across customer groups ([ticket routing guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/ticket-routing)).

- **Sample the confident output too.** The guide's sampling bullets call for stratified random sampling of high-confidence extractions, for ongoing error-rate measurement and to detect novel error patterns. In our reasoning, extractions accepted on high confidence are the ones no reviewer otherwise sees, so without a sample of them their error rate stays unknown, and stratifying by the segments above keeps small document types and fields in the sample. The guide gives no sample sizes or sampling rates. Anthropic's evals guidance points the same way for agents: production monitoring detects "distribution drift and unanticipated real-world failures", and teams should "sample transcripts to read weekly" ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).

- **Calibrate before you route.** The guide has the model output a confidence score per field, then sets review thresholds using a labeled validation set; in our reading, that makes the threshold reflect measured accuracy rather than the model's own sense of certainty. Anthropic's own workflow examples show the routing mechanism with fixed thresholds: a claude-code-action example returns `is_flaky` and `confidence` (0 to 1) and automatically starts a new run of the CI workflow on the same branch (`gh workflow run`) only when the failure is judged flaky with confidence of 0.7 or above; a flaky verdict below 0.7 prints "Not retrying automatically - manual review recommended" ([example workflow](https://raw.githubusercontent.com/anthropics/claude-code-action/main/examples/test-failure-analysis.yml)). The security-review action's prompt drops findings below 0.7 ([security-review prompt](https://raw.githubusercontent.com/anthropics/claude-code-security-review/main/claudecode/prompts.py)). Those examples document no calibration step; the exam's version chooses the threshold from labeled data.

```yaml
# Auto-retry only if flaky AND high confidence (>= 0.7)
- name: Retry flaky tests
  if: |
    fromJSON(steps.detect.outputs.structured_output).is_flaky == true &&
    fromJSON(steps.detect.outputs.structured_output).confidence >= 0.7
```

The excerpt shows confidence-based routing in an official example; it does not show calibration.

- **Spend scarce reviewer time where it matters.** The guide's routing skill sends extractions with low model confidence, or from ambiguous or contradictory source documents, to human review, prioritizing limited reviewer capacity. Anthropic's customer-research skill (part of its customer-support knowledge-work plugin) lists "Contradictory information found across sources" among its low-confidence signals ([customer research skill](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/customer-support/skills/customer-research/SKILL.md)). The content moderation guide gives a tiered example: you might automatically block queries judged high risk while flagging users with many medium-risk queries for human review ([content moderation guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/content-moderation)).

- **Exercise 3 puts it together.** Its final step has the model output field-level confidence scores, routes low-confidence extractions to human review, and analyzes accuracy by document type and field to verify consistent performance (see [Official preparation exercises](#official-preparation-exercises)). The confidence note under Task 5.2 above explains why this use of confidence does not contradict the guide's warning about self-reported confidence.

**Decide**

- If overall accuracy looks high and someone proposes cutting human review, choose accuracy analysis by document type and field first, and automate only the segments that hold up; not *97% is good enough*.
- If you need the error rate of extractions that skip review, choose stratified random sampling of the high-confidence output; not reviewing only the low-confidence items.
- If you are setting the confidence cutoff for review, choose a threshold calibrated on a labeled validation set; not the model's raw score or a round number.
- If reviewers are scarce, choose routing low-confidence extractions and those from ambiguous or contradictory sources first; not spreading review evenly across every document.

**Traps**

- Automating because the headline accuracy is high.
- Reviewing only what the model flags as uncertain, which leaves confident errors and new error patterns unmeasured.
- Treating model confidence as calibrated out of the box; Anthropic's research reports that models "struggle with calibration of P(IK) on new tasks" ([Language models (mostly) know what they know](https://www.anthropic.com/research/language-models-mostly-know-what-they-know)).
- Using a confidence score to decide escalation of a customer case; that is the Task 5.2 anti-pattern.

**Go deeper:** [Human review and confidence calibration](knowledge/evaluation-and-reliability.md#human-review-and-confidence-calibration)

### Task 5.6: Preserve information provenance and handle uncertainty in multi-source synthesis

**What the guide tests:** keeping every claim tied to its source through summarization and synthesis, and reporting disagreement between sources honestly: "annotating conflicts with source attribution rather than arbitrarily selecting one value" ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

**Know**

- **How attribution gets lost.** Source attribution disappears when a summarization step compresses findings without keeping claim-source mappings. The guide's answer is structured claim-source mappings (source URLs, document names, relevant excerpts) that subagents output and that the synthesis agent must preserve and merge. Preparation Exercise 4, step 3 specifies each finding as a claim, an evidence excerpt, a source URL or document name, and a publication date, then checks that the synthesis subagent keeps the attribution. Domain 1 tests the related skill of separating content from metadata when passing context between agents.

```json
{
  "claim": "<one factual statement>",
  "evidence_excerpt": "<the passage that supports it, quoted>",
  "source_url": "<URL>",
  "document_name": "<title of the source document>",
  "publication_date": "<date the source was published or the data collected>"
}
```

The record above is our illustration of the fields Exercise 4 lists.

- **Conflicts are reported, not resolved away.** When two credible sources give different statistics, annotate the conflict with both sources rather than picking one. The guide's matching skill finishes the document analysis with the conflicting values included and explicitly annotated, and lets the coordinator decide how to reconcile them before synthesis. Anthropic's research-subagent prompt goes only part of the way: it first has the subagent prioritize conflicting information "based on recency, consistency with other facts, the quality of the sources used", and only then says: "If unable to reconcile facts, include the conflicting information in your final task report for the lead researcher to resolve." ([research subagent prompt](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_subagent.md)). The guide goes further and leaves the reconciliation to the coordinator; on the exam, answer in the guide's terms. Anthropic's knowledge-synthesis skill puts it as a rule: "Always surface conflicts rather than silently picking one version." ([knowledge synthesis skill](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/enterprise-search/skills/knowledge-synthesis/SKILL.md)).

- **Dates prevent false contradictions.** The guide requires publication or data-collection dates in structured outputs so that a difference over time is not misread as a contradiction. Anthropic's knowledge-synthesis skill (part of its enterprise-search knowledge-work plugin) lists "Include the date or relative time" among its attribution rules and keeps items separate, rather than deduplicating them, when different time periods are represented ([knowledge synthesis skill](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/enterprise-search/skills/knowledge-synthesis/SKILL.md)). Anthropic's research-agent prompts in its cookbook inject the current date, and web search results carry a `page_age` field ([web search tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool)).

- **Established versus contested.** The guide's report structure has explicit sections that separate well-established findings from contested ones, preserving the original sources' characterizations and methodological context. Anthropic's research-subagent prompt asks the subagent to flag speculative claims explicitly rather than accept them as having happened, and to flag issues with results "rather than blindly presenting all results as established facts" ([research subagent prompt](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_subagent.md)).

- **Format follows content.** The guide's last skill renders each content type appropriately: "financial data as tables, news as prose, technical findings as structured lists", rather than converting everything to one uniform format ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

- **Citation tooling in Anthropic's own systems.** Anthropic's research system passes its findings to a separate CitationAgent that locates citations for each claim ([multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)); that agent's prompt says "ONLY add citations where the source documents directly support claims in the text" ([citations agent prompt](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/citations_agent.md)). The API's Citations feature returns the exact passages that support each claim ([Citations docs](https://platform.claude.com/docs/en/build-with-claude/citations)), and Anthropic's hallucination guidance has Claude find a supporting quote for each claim: "If it can't find a quote, it must retract the claim." ([reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations)).

!!! info "Current API constraint (as of September 2026)"

    Citations and structured outputs are incompatible: enabling citations on any user-provided document (`document` or `search_result` blocks) while also sending `output_config.format`, or the deprecated `output_format` parameter, returns a 400 error ([Citations docs](https://platform.claude.com/docs/en/build-with-claude/citations)). In our reading, a claim-source mapping produced as JSON through structured outputs therefore has to carry its own source fields (URL, excerpt, date) rather than rely on API citations in the same call.

**Decide**

- If findings pass through one or more summarization steps, choose structured claim-source mappings that every downstream agent must preserve and merge; not prose summaries with the sources listed at the end.
- If two credible sources report different figures, choose keeping both values with their attribution, dates and methodological context, and marking the conflict; not selecting one value.
- If a document-analysis subagent finds conflicting values, choose completing the analysis with both values annotated and letting the coordinator reconcile before synthesis; not having the subagent quietly settle the question.
- If two figures differ and come from different years, choose checking the publication or collection dates before calling it a contradiction.
- If the report mixes financial data, news and technical findings, choose a format per content type; not one uniform layout.

**Traps**

- A synthesis step that rewrites findings into clean prose and drops the claim-to-source links.
- Picking the most authoritative or most recent statistic without showing the other.
- Treating figures from different dates as contradictory.
- Presenting contested findings in the same voice as well-established ones.
- Rendering every content type as the same bullet list.

**Go deeper:** [Provenance and uncertainty in synthesis](knowledge/context-engineering.md#provenance-and-uncertainty-in-synthesis)

## Official sample questions

The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says its twelve sample questions "illustrate the format and difficulty level of the exam" and that "These are drawn from the practice test and include explanations to aid learning."

The [program FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) adds: "The practice exam available on the previous platform was retired in the move to Pearson. The exam guide includes sample questions that show the format and style of what's on the exam."

!!! info "Before you start"

    - All twelve items have four options and one correct answer. The live exam also uses multiple-response items, and the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says "each item states how many responses to select".
    - The CCAR-F samples carry no domain label, unlike the samples in the other three exam guides. The "What this teaches" line under each item is this page's mapping to a task statement.
    - Items are grouped under the guide's scenario labels. Scenarios 4 and 6 have no published samples.
    - Stems, options and rationales are reproduced word for word. Where the guide's text has an em dash, this page shows a colon. The code formatting on the glob pattern in Question 6's rationale is ours.
    - Commit to an answer before you open the answer block.

| Scenario | Sample items | Task statements the items test (our mapping) |
|---|---|---|
| 1. Customer Support Resolution Agent | Questions 1 to 3 | 1.4 and 1.5, 2.1, 5.2 |
| 2. Code Generation with Claude Code | Questions 4 to 6 | 3.2, 3.4, 3.3 |
| 3. Multi-Agent Research System | Questions 7 to 9 | 1.2, 5.3, 2.3 |
| 4. Developer Productivity with Claude | None published | See [Scenario 4](#scenario-4-developer-productivity-with-claude) |
| 5. Claude Code for Continuous Integration | Questions 10 to 12 | 3.6, 4.5, 4.6 and 1.6 |
| 6. Structured Data Extraction | None published | See [Scenario 6](#scenario-6-structured-data-extraction) |

### Customer Support Resolution Agent (Questions 1 to 3)

The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) labels these three items "Scenario: Customer Support Resolution Agent". The setup, including the four MCP tools, is in [Scenario 1](#scenario-1-customer-support-resolution-agent).

#### Question 1

Production data shows that in 12% of cases, your agent skips get_customer entirely and calls lookup_order using only the customer's stated name, occasionally leading to misidentified accounts and incorrect refunds. What change would most effectively address this reliability issue?

- **A.** Add a programmatic prerequisite that blocks lookup_order and process_refund calls until get_customer has returned a verified customer ID.
- **B.** Enhance the system prompt to state that customer verification via get_customer is mandatory before any order operations.
- **C.** Add few-shot examples showing the agent always calling get_customer first, even when customers volunteer order details.
- **D.** Implement a routing classifier that analyzes each request and enables only the subset of tools appropriate for that request type.

??? success "Answer and Anthropic's rationale"

    Anthropic's rationale, verbatim from the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf):

    > Correct Answer: A. When a specific tool sequence is required for critical business logic (like verifying customer identity before processing refunds), programmatic enforcement provides deterministic guarantees that prompt-based approaches cannot. Options B and C rely on probabilistic LLM compliance, which is insufficient when errors have financial consequences. Option D addresses tool availability rather than tool ordering, which is not the actual problem.

**What this teaches:** when a tool order protects money or identity, enforce it in code with a prerequisite gate or a hook that blocks the call; prompt instructions and examples alone still fail some of the time. Limiting which tools are available (Option D) does not fix the order they run in (Tasks 1.4 and 1.5 in [Domain 1](#domain-1-agentic-architecture-orchestration)).

#### Question 2

Production logs show the agent frequently calls get_customer when users ask about orders (e.g., "check my order #12345"), instead of calling lookup_order. Both tools have minimal descriptions ("Retrieves customer information" / "Retrieves order details") and accept similar identifier formats. What's the most effective first step to improve tool selection reliability?

- **A.** Add few-shot examples to the system prompt demonstrating correct tool selection patterns, with 5-8 examples showing order-related queries routing to lookup_order.
- **B.** Expand each tool's description to include input formats it handles, example queries, edge cases, and boundaries explaining when to use it versus similar tools.
- **C.** Implement a routing layer that parses user input before each turn and pre-selects the appropriate tool based on detected keywords and identifier patterns.
- **D.** Consolidate both tools into a single lookup_entity tool that accepts any identifier and internally determines which backend to query.

??? success "Answer and Anthropic's rationale"

    Anthropic's rationale, verbatim from the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf):

    > Correct Answer: B. Tool descriptions are the primary mechanism LLMs use for tool selection. When descriptions are minimal, models lack the context to differentiate between similar tools. Option B directly addresses this root cause with a low-effort, high-leverage fix. Few-shot examples (A) add token overhead without fixing the underlying issue. A routing layer (C) is over-engineered and bypasses the LLM's natural language understanding. Consolidating tools (D) is a valid architectural choice but requires more effort than a "first step" warrants when the immediate problem is inadequate descriptions.

**What this teaches:** fix misrouting between similar tools at the source. Each description should state the input formats the tool handles, example queries, edge cases, and when to use it instead of its neighbor (Task 2.1 in [Domain 2](#domain-2-tool-design-mcp-integration)).

#### Question 3

Your agent achieves 55% first-contact resolution, well below the 80% target. Logs show it escalates straightforward cases (standard damage replacements with photo evidence) while attempting to autonomously handle complex situations requiring policy exceptions. What's the most effective way to improve escalation calibration?

- **A.** Add explicit escalation criteria to your system prompt with few-shot examples demonstrating when to escalate versus resolve autonomously.
- **B.** Have the agent self-report a confidence score (1-10) before each response and automatically route requests to humans when confidence falls below a threshold.
- **C.** Deploy a separate classifier model trained on historical tickets to predict which requests need escalation before the main agent begins processing.
- **D.** Implement sentiment analysis to detect customer frustration levels and automatically escalate when negative sentiment exceeds a threshold.

??? success "Answer and Anthropic's rationale"

    Anthropic's rationale, verbatim from the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf):

    > Correct Answer: A. Adding explicit escalation criteria with few-shot examples directly addresses the root cause: unclear decision boundaries. This is the proportionate first response before adding infrastructure. Option B fails because LLM self-reported confidence is poorly calibrated: the agent is already incorrectly confident on hard cases. Option C is over-engineered, requiring labeled data and ML infrastructure when prompt optimization hasn't been tried. Option D solves a different problem entirely; sentiment doesn't correlate with case complexity, which is the actual issue.

    **Do not over-apply Option B's lesson.** The guide still uses calibrated, model-reported confidence to route human review (Tasks 4.6 and 5.5); see the note under [Task 5.2](#task-52-design-effective-escalation-and-ambiguity-resolution-patterns) for how the two positions fit together.

**What this teaches:** calibrate escalation with explicit criteria and few-shot examples in the system prompt. The triggers the guide names are a customer's request for a human, policy exceptions or gaps (not just complex cases), and inability to make meaningful progress. Self-reported confidence and sentiment are poor proxies for how hard a case is (Task 5.2 in [Domain 5](#domain-5-context-management-reliability)).

### Code Generation with Claude Code (Questions 4 to 6)

The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) labels these three items "Scenario: Code Generation with Claude Code". The setup is in [Scenario 2](#scenario-2-code-generation-with-claude-code).

#### Question 4

You want to create a custom /review slash command that runs your team's standard code review checklist. This command should be available to every developer when they clone or pull the repository. Where should you create this command file?

- **A.** In the .claude/commands/ directory in the project repository
- **B.** In ~/.claude/commands/ in each developer's home directory
- **C.** In the CLAUDE.md file at the project root
- **D.** In a .claude/config.json file with a commands array

??? success "Answer and Anthropic's rationale"

    Anthropic's rationale, verbatim from the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf):

    > Correct Answer: A. Project-scoped custom slash commands should be stored in the .claude/commands/ directory within the repository. These commands are version-controlled and automatically available to all developers when they clone or pull the repo. Option B (~/.claude/commands/) is for personal commands that aren't shared via version control. Option C (CLAUDE.md) is for project instructions and context, not command definitions. Option D describes a configuration mechanism that doesn't exist in Claude Code.

    **Freshness note (as of September 2026).** The [Claude Code skills documentation](https://code.claude.com/docs/en/skills) now says "Custom commands have been merged into skills." Command files keep working, so answer A is still correct, although the docs prefer a skill (a committed `.claude/skills/<name>/SKILL.md`) for new work. The current behavior, including `/review` now being an alias of a bundled skill, is in the warning under [Task 3.2](#task-32-create-and-configure-custom-slash-commands-and-skills).

**What this teaches:** a command the whole team should get belongs in the repository, in `.claude/commands/`, where version control shares it; `~/.claude/commands/` is personal (Task 3.2 in [Domain 3](#domain-3-claude-code-configuration-workflows)). The guide draws the same project-versus-personal line for MCP servers in Task 2.4 ([Domain 2](#domain-2-tool-design-mcp-integration)), which none of the twelve samples tests directly: `.mcp.json` for shared team tooling, `~/.claude.json` for personal or experimental servers.

#### Question 5

You've been assigned to restructure the team's monolithic application into microservices. This will involve changes across dozens of files and requires decisions about service boundaries and module dependencies. Which approach should you take?

- **A.** Enter plan mode to explore the codebase, understand dependencies, and design an implementation approach before making changes.
- **B.** Start with direct execution and make changes incrementally, letting the implementation reveal the natural service boundaries.
- **C.** Use direct execution with comprehensive upfront instructions detailing exactly how each service should be structured.
- **D.** Begin in direct execution mode and only switch to plan mode if you encounter unexpected complexity during implementation.

??? success "Answer and Anthropic's rationale"

    Anthropic's rationale, verbatim from the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf):

    > Correct Answer: A. Plan mode is designed for complex tasks involving large-scale changes, multiple valid approaches, and architectural decisions: exactly what monolith-to-microservices restructuring requires. It enables safe codebase exploration and design before committing to changes. Option B risks costly rework when dependencies are discovered late. Option C assumes you already know the right structure without exploring the code. Option D ignores that the complexity is already stated in the requirements, not something that might emerge later.

**What this teaches:** when the stem already announces architectural scope (dozens of files, service boundaries, module dependencies), start in plan mode, then switch to direct execution to carry out the plan. Do not over-apply it: the guide keeps direct execution for simple, well-scoped changes such as a single-file bug fix with a clear stack trace (Task 3.4 in [Domain 3](#domain-3-claude-code-configuration-workflows)).

#### Question 6

Your codebase has distinct areas with different coding conventions: React components use functional style with hooks, API handlers use async/await with specific error handling, and database models follow a repository pattern. Test files are spread throughout the codebase alongside the code they test (e.g., Button.test.tsx next to Button.tsx), and you want all tests to follow the same conventions regardless of location. What's the most maintainable way to ensure Claude automatically applies the correct conventions when generating code?

- **A.** Create rule files in .claude/rules/ with YAML frontmatter specifying glob patterns to conditionally apply conventions based on file paths
- **B.** Consolidate all conventions in the root CLAUDE.md file under headers for each area, relying on Claude to infer which section applies
- **C.** Create skills in .claude/skills/ for each code type that include the relevant conventions in their SKILL.md files
- **D.** Place a separate CLAUDE.md file in each subdirectory containing that area's specific conventions

??? success "Answer and Anthropic's rationale"

    Anthropic's rationale, verbatim from the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf):

    > Correct Answer: A. Option A is correct because .claude/rules/ with glob patterns (e.g., `**/*.test.tsx`) allows conventions to be automatically applied based on file paths regardless of directory location, essential for test files spread throughout the codebase. Option B relies on inference rather than explicit matching, making it unreliable. Option C requires manual skill invocation or relies on Claude choosing to load them, contradicting the need for deterministic "automatic" application based on file paths. Option D can't easily handle files spread across many directories since CLAUDE.md files are directory-bound.

    **Freshness note (as of September 2026).** Skills now accept a `paths` field too. The [skills documentation](https://code.claude.com/docs/en/skills) describes it as "Glob patterns that limit when this skill is activated" and says "When set, Claude loads the skill automatically only when working with files matching the patterns." The field narrows when a skill can activate; the same page still says "In a regular session, skill descriptions are loaded into context so Claude knows what's available, but full skill content only loads when invoked." Rules need no invocation. The [memory documentation](https://code.claude.com/docs/en/memory) says "Path-scoped rules trigger when Claude reads files matching the pattern, not on every tool use", and contrasts them with skills, which "only load when you invoke them or when Claude determines they're relevant to your prompt." The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) words it slightly differently: Task 3.3 says path-scoped rules "load only when editing matching files", where the docs say reading. Either way the rule loads without anyone choosing it, so the rationale's distinction holds (our reading). Conventions that must apply to every matching file belong in rules; answer A.

**What this teaches:** conventions keyed to a file type across many directories belong in `.claude/rules/` files whose YAML frontmatter `paths` field holds a glob such as `**/*.test.tsx`, because a subdirectory CLAUDE.md is bound to one directory (Task 3.3 in [Domain 3](#domain-3-claude-code-configuration-workflows)).

### Multi-Agent Research System (Questions 7 to 9)

The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) labels these three items "Scenario: Multi-Agent Research System". The setup is in [Scenario 3](#scenario-3-multi-agent-research-system).

#### Question 7

After running the system on the topic "impact of AI on creative industries," you observe that each subagent completes successfully: the web search agent finds relevant articles, the document analysis agent summarizes papers correctly, and the synthesis agent produces coherent output. However, the final reports cover only visual arts, completely missing music, writing, and film production. When you examine the coordinator's logs, you see it decomposed the topic into three subtasks: "AI in digital art creation," "AI in graphic design," and "AI in photography." What is the most likely root cause?

- **A.** The synthesis agent lacks instructions for identifying coverage gaps in the findings it receives from other agents.
- **B.** The coordinator agent's task decomposition is too narrow, resulting in subagent assignments that don't cover all relevant domains of the topic.
- **C.** The web search agent's queries are not comprehensive enough and need to be expanded to cover more creative industry sectors.
- **D.** The document analysis agent is filtering out sources related to non-visual creative industries due to overly restrictive relevance criteria.

??? success "Answer and Anthropic's rationale"

    Anthropic's rationale, verbatim from the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf):

    > Correct Answer: B. The coordinator's logs reveal the root cause directly: it decomposed "creative industries" into only visual arts subtasks (digital art, graphic design, photography), completely omitting music, writing, and film. The subagents executed their assigned tasks correctly: the problem is what they were assigned. Options A, C, and D incorrectly blame downstream agents that are working correctly within their assigned scope.

**What this teaches:** when every subagent succeeds and coverage is still incomplete, look upstream at how the coordinator decomposed the task (Task 1.2 in [Domain 1](#domain-1-agentic-architecture-orchestration)).

#### Question 8

The web search subagent times out while researching a complex topic. You need to design how this failure information flows back to the coordinator agent. Which error propagation approach best enables intelligent recovery?

- **A.** Return structured error context to the coordinator including the failure type, the attempted query, any partial results, and potential alternative approaches.
- **B.** Implement automatic retry logic with exponential backoff within the subagent, returning a generic "search unavailable" status only after all retries are exhausted.
- **C.** Catch the timeout within the subagent and return an empty result set marked as successful.
- **D.** Propagate the timeout exception directly to a top-level handler that terminates the entire research workflow.

??? success "Answer and Anthropic's rationale"

    Anthropic's rationale, verbatim from the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf):

    > Correct Answer: A. Structured error context gives the coordinator the information it needs to make intelligent recovery decisions: whether to retry with a modified query, try an alternative approach, or proceed with partial results. Option B's generic status hides valuable context from the coordinator, preventing informed decisions. Option C suppresses the error by marking failure as success, which prevents any recovery and risks incomplete research outputs. Option D terminates the entire workflow unnecessarily when recovery strategies could succeed.

    **What the rationale faults in Option B.** The generic status at the end, not the retrying (our reading). Task 5.3 in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) wants subagents to "implement local recovery for transient failures and only propagate errors they cannot resolve, including what was attempted and partial results", and says generic error statuses such as "search unavailable" hide valuable context from the coordinator. Task 5.3 also separates access failures, such as this timeout, from valid empty results (a query that succeeded and found nothing); Option C blurs the two by reporting a failure as an empty success.

**What this teaches:** a subagent failure should reach the coordinator as structured context (failure type, attempted query, partial results, alternatives), never as a silent success and never as a workflow-wide abort (Task 5.3 in [Domain 5](#domain-5-context-management-reliability)).

#### Question 9

During testing, you observe that the synthesis agent frequently needs to verify specific claims while combining findings. Currently, when verification is needed, the synthesis agent returns control to the coordinator, which invokes the web search agent, then re-invokes synthesis with results. This adds 2-3 round trips per task and increases latency by 40%. Your evaluation shows that 85% of these verifications are simple fact-checks (dates, names, statistics) while 15% require deeper investigation. What's the most effective approach to reduce overhead while maintaining system reliability?

- **A.** Give the synthesis agent a scoped verify_fact tool for simple lookups, while complex verifications continue delegating to the web search agent through the coordinator.
- **B.** Have the synthesis agent accumulate all verification needs and return them as a batch to the coordinator at the end of its pass, which then sends them all to the web search agent at once.
- **C.** Give the synthesis agent access to all web search tools so it can handle any verification need directly without round-trips through the coordinator.
- **D.** Have the web search agent proactively cache extra context around each source during initial research, anticipating what the synthesis agent might need to verify.

??? success "Answer and Anthropic's rationale"

    Anthropic's rationale, verbatim from the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf):

    > Correct Answer: A. Option A applies the principle of least privilege by giving the synthesis agent only what it needs for the 85% common case (simple fact verification) while preserving the existing coordination pattern for complex cases. Option B's batching approach creates blocking dependencies since synthesis steps may depend on earlier verified facts. Option C over-provisions the synthesis agent, violating separation of concerns. Option D relies on speculative caching that cannot reliably predict what the synthesis agent will need to verify.

**What this teaches:** give a specialist agent one narrow cross-role tool for its frequent simple need, and keep the complex cases routed through the coordinator. Handing it another role's whole tool set is the over-provisioning the rationale rejects; the guide expects each subagent's tools restricted to its role, and its own example of an agent misusing tools outside its specialization is a synthesis agent attempting web searches (Task 2.3 in [Domain 2](#domain-2-tool-design-mcp-integration)).

### Claude Code for Continuous Integration (Questions 10 to 12)

The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) labels these three items "Scenario: Claude Code for Continuous Integration". The setup is in [Scenario 5](#scenario-5-claude-code-for-continuous-integration).

#### Question 10

Your pipeline script runs claude "Analyze this pull request for security issues" but the job hangs indefinitely. Logs indicate Claude Code is waiting for interactive input. What's the correct approach to run Claude Code in an automated pipeline?

- **A.** Add the -p flag: claude -p "Analyze this pull request for security issues"
- **B.** Set the environment variable CLAUDE_HEADLESS=true before running the command
- **C.** Redirect stdin from /dev/null: claude "Analyze this pull request for security issues" < /dev/null
- **D.** Add the --batch flag: claude --batch "Analyze this pull request for security issues"

??? success "Answer and Anthropic's rationale"

    Anthropic's rationale, verbatim from the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf):

    > Correct Answer: A. The -p (or --print) flag is the documented way to run Claude Code in non-interactive mode. It processes the prompt, outputs the result to stdout, and exits without waiting for user input: exactly what CI/CD pipelines require. The other options reference non-existent features (CLAUDE_HEADLESS environment variable, --batch flag) or use Unix workarounds that don't properly address Claude Code's command syntax.

    **Docs check (as of September 2026).** The [headless documentation](https://code.claude.com/docs/en/headless) still says "Add the `-p` (or `--print`) flag to any `claude` command to run it non-interactively." The [CLI reference](https://code.claude.com/docs/en/cli-reference) lists no `--batch` flag, and the [environment variables reference](https://code.claude.com/docs/en/env-vars) lists no `CLAUDE_HEADLESS`, so the rationale's point about non-existent features still holds. The two structured-output flags the guide pairs with `-p` are also current: the CLI reference describes `--output-format` as "Specify output format for print mode (options: `text`, `json`, `stream-json`)" and `--json-schema` as "Get validated JSON output matching a JSON Schema after the agent completes its workflow (print mode only)."

**What this teaches:** in a pipeline, Claude Code runs non-interactively with `-p` (`--print`). Distractors invent flags and environment variables, so know the real ones the guide lists next to `-p`: `--output-format json` and `--json-schema`, for machine-parseable output in CI (Task 3.6 in [Domain 3](#domain-3-claude-code-configuration-workflows)).

#### Question 11

Your team wants to reduce API costs for automated analysis. Currently, real-time Claude calls power two workflows: (1) a blocking pre-merge check that must complete before developers can merge, and (2) a technical debt report generated overnight for review the next morning. Your manager proposes switching both to the Message Batches API for its 50% cost savings. How should you evaluate this proposal?

- **A.** Use batch processing for the technical debt reports only; keep real-time calls for pre-merge checks.
- **B.** Switch both workflows to batch processing with status polling to check for completion.
- **C.** Keep real-time calls for both workflows to avoid batch result ordering issues.
- **D.** Switch both to batch processing with a timeout fallback to real-time if batches take too long.

??? success "Answer and Anthropic's rationale"

    Anthropic's rationale, verbatim from the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf):

    > Correct Answer: A. The Message Batches API offers 50% cost savings but has processing times up to 24 hours with no guaranteed latency SLA. This makes it unsuitable for blocking pre-merge checks where developers wait for results, but ideal for overnight batch jobs like technical debt reports. Option B is wrong because relying on "often faster" completion isn't acceptable for blocking workflows. Option C reflects a misconception: batch results can be correlated using custom_id fields. Option D adds unnecessary complexity when the simpler solution is matching each API to its appropriate use case.

    **Reading note.** Option B's text, as printed, does not contain the words "often faster" that the [rationale](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) quotes. Our reading: the rationale is answering the idea behind B, that a batch usually returns quickly, although nothing guarantees when.

    **Docs detail (as of September 2026).** The [batch processing documentation](https://platform.claude.com/docs/en/build-with-claude/batch-processing) describes asynchronous processing "with most batches finishing in less than 1 hour while reducing costs by 50% and increasing throughput", and also says "Batches expire if processing does not complete within 24 hours." A typical time is not a guarantee, so the answer stands. Option C's worry is real but solved: results may not come back in the order of the requests, and the docs say "To correctly match results with their corresponding requests, always use the `custom_id` field."

**What this teaches:** batch only work that can wait; the Message Batches API suits overnight and weekly jobs, not a check that blocks a merge (Task 4.5 in [Domain 4](#domain-4-prompt-engineering-structured-output)).

#### Question 12

A pull request modifies 14 files across the stock tracking module. Your single-pass review analyzing all files together produces inconsistent results: detailed feedback for some files but superficial comments for others, obvious bugs missed, and contradictory feedback: flagging a pattern as problematic in one file while approving identical code elsewhere in the same PR. How should you restructure the review?

- **A.** Split into focused passes: analyze each file individually for local issues, then run a separate integration-focused pass examining cross-file data flow.
- **B.** Require developers to split large PRs into smaller submissions of 3-4 files before the automated review runs.
- **C.** Switch to a higher-tier model with a larger context window to give all 14 files adequate attention in one pass.
- **D.** Run three independent review passes on the full PR and only flag issues that appear in at least two of the three runs.

??? success "Answer and Anthropic's rationale"

    Anthropic's rationale, verbatim from the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf):

    > Correct Answer: A. Splitting reviews into focused passes directly addresses the root cause: attention dilution when processing many files at once. File-by-file analysis ensures consistent depth, while a separate integration pass catches cross-file issues. Option B shifts burden to developers without improving the system. Option C misunderstands that larger context windows don't solve attention quality issues. Option D would actually suppress detection of real bugs by requiring consensus on issues that may only be caught intermittently.

**What this teaches:** uneven depth and contradictions across many files are attention dilution; split the review into per-file passes plus one cross-file integration pass. A larger context window does not fix attention quality, and requiring agreement between runs suppresses real bugs that only some runs catch (Task 4.6 in [Domain 4](#domain-4-prompt-engineering-structured-output), and Task 1.6 in [Domain 1](#domain-1-agentic-architecture-orchestration)).

### Scenarios with no published samples

The guide gives three samples each for Scenarios 1, 2, 3 and 5, and none for Scenario 4 (Developer Productivity with Claude) or Scenario 6 (Structured Data Extraction). The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says the 4 scenarios on an exam are "picked at random from the full set of the 6 scenarios", so either or both can appear on yours. Prepare them from the objectives instead.

- **Scenario 4.** Its primary domains are Tool Design & MCP Integration, Claude Code Configuration & Workflows, and Agentic Architecture & Orchestration. The scenario names the built-in tools (Read, Write, Bash, Grep, Glob) and MCP servers. By our mapping, those point to Task 2.5 (built-in tools) and Task 2.4 (MCP server integration); Questions 1, 2, 7, 9 and 12 (Domains 1 and 2) and Questions 4, 5, 6 and 10 (Domain 3) exercise its domains, but none of the twelve tests Task 2.4 or 2.5. Read the wording analysis under [Scenario 4](#scenario-4-developer-productivity-with-claude).
- **Scenario 6.** Its primary domains are Prompt Engineering & Structured Output, and Context Management & Reliability. By our mapping, its text (validating output with JSON schemas, maintaining high accuracy) points to Task 4.3 (structured output with tool use and JSON schemas), Task 4.4 (validation, retry and feedback loops) and Task 5.5 (human review and confidence calibration); none of the twelve samples tests those three, while Questions 3, 8, 11 and 12 exercise its two domains. Question 11 is the only sample about the Message Batches API. [Exercise 3](#exercise-3-build-a-structured-data-extraction-pipeline), Build a Structured Data Extraction Pipeline, reinforces the same two domains and practices JSON schemas, validation-retry loops, batch processing and routing low-confidence extractions to human review. Read the wording analysis under [Scenario 6](#scenario-6-structured-data-extraction).

??? tip "Patterns across the twelve rationales (open after you have worked the items)"

    | Decision rule the rationales apply (our wording) | Items |
    |---|---|
    | A rule that must always hold is enforced in code; prompts and examples are probabilistic. | 1 |
    | Fix the root cause the stem describes, not a downstream symptom. | 2, 3, 7, 12 |
    | Prefer the proportionate first step; options that add classifiers, routers or new infrastructure are called over-engineered, and an added fallback is called unnecessary complexity. | 2, 3, 11 |
    | Options that name a flag, variable or config file Claude Code does not have are wrong on sight. | 4, 10 |
    | When the stem already states the complexity, plan before acting. | 5 |
    | Explicit matching beats inference; deterministic loading beats hoping Claude loads something. | 6 |
    | Failures travel as structured context; never mark a failure as success, never abort everything. | 8 |
    | Least privilege: add one scoped tool for the common case, keep the rare case on the existing route. | 9 |
    | Match each workload to the API whose latency guarantee fits it. | 11 |
    | A higher-tier model with a larger context window does not fix attention quality; restructure the work. | 12 |
    | Moving the burden onto people (developers splitting their own PRs) does not improve the system. | 12 |

    In the guide's answer key, ten of the twelve correct answers are A and two are B. Do not read anything into letter position on the live exam. Rules of thumb that candidates have published are tested against these rationales in [What candidates report](#what-candidates-report).

## Official preparation exercises

Section 8 of the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf), Preparation Exercises, sets four builds: "Complete these hands-on exercises to build practical familiarity with the topics covered on the exam. Each exercise is designed to reinforce knowledge across one or more exam domains." Each exercise below gives the guide's objective and steps word for word, then concrete guidance for building and testing it, including the places where the current product documentation differs from the guide.

| Exercise | What you build | Domains reinforced (guide) | Sample items on the same ideas (our mapping) |
|---|---|---|---|
| 1 | A customer-facing agent with tools, structured errors and an escalation hook | 1, 2, 5 | Questions 1 to 3 |
| 2 | A Claude Code setup for a multi-developer repository | 3, 2 | Questions 4 to 6 |
| 3 | A document extraction pipeline with validation, batching and human review | 4, 5 | Question 11 |
| 4 | A coordinator with research subagents, error propagation and provenance | 1, 2, 5 | Questions 7 to 9 |

The scenario each exercise resembles is listed in [The six scenarios](#the-six-scenarios), and the week each one fits is in the [Study plan](#study-plan). The guide writes the steps as bullets; they are numbered here so the guidance can refer to them.

### Exercise 1: Build a Multi-Tool Agent with Escalation Logic

**Objective, from the [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf):** "Practice designing an agentic loop with tool integration, structured error handling, and escalation patterns." **Domains reinforced:** 1, 2 and 5.

**The guide's steps:**

1. Define 3-4 MCP tools with detailed descriptions that clearly differentiate each tool's purpose, expected inputs, and boundary conditions. Include at least two tools with similar functionality that require careful description to avoid selection confusion.
2. Implement an agentic loop that checks stop_reason to determine whether to continue tool execution or present the final response. Handle both "tool_use" and "end_turn" stop reasons correctly.
3. Add structured error responses to your tools: include errorCategory (transient/validation/permission), isRetryable boolean, and human-readable descriptions. Test that the agent handles each error type appropriately (retrying transient errors, explaining business errors to the user).
4. Implement a programmatic hook that intercepts tool calls to enforce a business rule (e.g., blocking operations above a threshold amount), redirecting to an escalation workflow when triggered.
5. Test with multi-concern messages (e.g., requests involving multiple issues) and verify the agent decomposes the request, handles each concern, and synthesizes a unified response.

**How to build it**

- **Step 1: tools.** Reuse Scenario 1's four tools (get_customer, lookup_order, process_refund, escalate_to_human). They already contain a confusable pair: in Question 2, get_customer and lookup_order are misrouted because their descriptions are minimal. For each tool, say what it does, when to use it and when not to, what each parameter means, and its limits, in at least three to four sentences (the description rules and their sources are under [Task 2.1](#task-21-design-effective-tool-interfaces-with-clear-descriptions-and-boundaries)). Then test selection with ambiguous requests, as the guide's How to Prepare list asks; a request like Question 2's "check my order #12345" must reach lookup_order.

- **Step 2: the loop.** Write this loop by hand on the Messages API even if you later move to the Agent SDK, because this step is about stop_reason. While stop_reason is "tool_use", run every requested tool, append the assistant turn, then send one user message holding one tool_result per tool_use block, results before any text. Stop on "end_turn". The [tool use docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works) add that "The loop exits on any other stop reason" (`max_tokens`, `stop_sequence`, `refusal`), and they document `pause_turn` for server-tool loops, so real code handles those too; exam answers use the guide's "tool_use" versus "end_turn" framing. Do not stop because the response contains text: Claude often writes text before its tool_use blocks, and checking for assistant text is one of the anti-patterns the guide names (1.1-S3), alongside parsing natural-language signals and using an arbitrary iteration cap as the primary stopping mechanism.

- **Step 3: structured errors.** Return a failure as an error result (`isError: true` from an MCP tool, `is_error: true` on a Messages API tool_result) whose body carries errorCategory, isRetryable and a readable description. Those fields are the guide's convention inside your payload: the [MCP tools specification](https://modelcontextprotocol.io/specification/2026-07-28/server/tools) defines the `isError` flag but no errorCategory or isRetryable field. The guide lists three errorCategory values, yet its knowledge bullet 2.2-K2 and this step also name business errors, so add business as a fourth value with isRetryable false (our reconciliation of the guide's two lists, not guide wording; 2.2-S2 does ask for a false retry flag on business rule violations). The guide also spells the flag `retriable` in 2.2-S2; pick one spelling. Run this test matrix:

    | Simulate | Category | What to verify |
    |---|---|---|
    | lookup_order times out | transient | The agent retries, as the step expects. |
    | An order number in the wrong format | validation | The agent corrects its input rather than repeating the same call. The MCP spec reports input validation errors in the tool result with `isError: true`, and says clients should pass tool execution errors to the model to enable self-correction. |
    | A refund the policy forbids | business | The agent explains the rule to the customer and does not retry. |
    | The caller lacks access to the account | permission | The agent obeys the isRetryable value you assign and does not retry blindly. |
    | lookup_order finds no matching order | none: a valid empty result | The agent reports that no order matched and does not treat it as a failure. |

- **Step 4: the hook.** In the Agent SDK, register a `PreToolUse` hook whose matcher is the refund tool's full name (custom tools served from an SDK MCP server are named `mcp__{server_name}__{tool_name}`). Above your threshold, return `permissionDecision: "deny"` with a `permissionDecisionReason`, both inside `hookSpecificOutput`; the [SDK hooks docs](https://code.claude.com/docs/en/agent-sdk/hooks) say the reason "tells the model why, so it avoids retrying". Use that reason to redirect: tell Claude to call escalate_to_human with a structured handoff (customer ID, root cause, refund amount, recommended action), because the human agent cannot see the transcript. In a hand-built loop, put the same check in your dispatch code before the tool runs. Add the prerequisite gate from Question 1 as well: block lookup_order and process_refund until get_customer has returned a verified ID. Then try to talk the agent past both rules; neither should ever give way, because a hook is deterministic where a prompt instruction is probabilistic. [Domain 1](#domain-1-agentic-architecture-orchestration) (tasks 1.4 and 1.5) has illustrative Python for both hooks.

- **Step 5: multi-concern messages.** Write test messages that combine two or three of Scenario 1's request types, for example a return and a billing dispute. The agent should list the issues separately, investigate them in parallel with shared context, and send one reply that resolves each. The [parallel tool use docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use) say "Independent, read-only operations are usually safe to run in parallel for lower latency", and all the results go back together in the next user message. The same page adds that tools with side effects, shared state or ordering requirements might be better run sequentially, so parallelize the lookups, not the refunds. The pattern itself is taught under [Task 1.4](#task-14-implement-multi-step-workflows-with-enforcement-and-handoff-patterns).

**Done when**

- [ ] Order questions reach lookup_order and customer questions reach get_customer, including ambiguous phrasings.
- [ ] The loop continues on "tool_use" and ends on "end_turn", with no text sniffing and no iteration cap as the primary stopping rule.
- [ ] Each error category produces the behavior in the matrix, and an empty result is not reported as a failure.
- [ ] No refund above the threshold and no order operation before verification ever runs, whatever the conversation says; blocked refunds arrive at escalate_to_human with a structured handoff.
- [ ] A message with two issues gets one reply that covers both.

Go deeper: [the agentic loop](knowledge/agents-and-agent-sdk.md#the-agentic-loop), [writing tool descriptions that steer selection](knowledge/tool-use-and-mcp.md#writing-tool-descriptions-that-steer-selection), [MCP errors and structured results](knowledge/tool-use-and-mcp.md#mcp-errors-and-structured-results), [hooks in the SDK](knowledge/agents-and-agent-sdk.md#hooks-in-the-sdk), [escalation and ambiguity](knowledge/evaluation-and-reliability.md#escalation-and-ambiguity).

### Exercise 2: Configure Claude Code for a Team Development Workflow

**Objective, from the [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf):** "Practice configuring CLAUDE.md hierarchies, custom slash commands, path-specific rules, and MCP server integration for a multi-developer project." **Domains reinforced:** 3 and 2.

**The guide's steps:**

1. Create a project-level CLAUDE.md with universal coding standards and testing conventions. Verify that instructions placed at the project level are consistently applied across all team members.
2. Create .claude/rules/ files with YAML frontmatter glob patterns for different code areas (e.g., `paths: ["src/api/**/*"]` for API conventions, `paths: ["**/*.test.*"]` for testing conventions). Test that rules load only when editing matching files.
3. Create a project-scoped skill in .claude/skills/ with context: fork and allowed-tools restrictions. Verify the skill runs in isolation without polluting the main conversation context.
4. Configure an MCP server in .mcp.json with environment variable expansion for credentials. Add a personal experimental MCP server in ~/.claude.json and verify both are available simultaneously.
5. Test plan mode versus direct execution on tasks of varying complexity: a single-file bug fix, a multi-file library migration, and a new feature with multiple valid implementation approaches. Observe when plan mode provides value.

The objective names custom slash commands, but no step builds one. Add the shared `/review` command from Question 4, whose answer puts it in `.claude/commands/` so that it reaches everyone through version control. As of September 2026 the [skills docs](https://code.claude.com/docs/en/skills) say "Custom commands have been merged into skills": `.claude/commands/review.md` and `.claude/skills/review/SKILL.md` both create `/review`, and the docs prefer a skill for new work. Expect `.claude/commands/` in exam answers; either file works in practice. The finished repository looks like this:

```text
your-repo/
  CLAUDE.md                          step 1: team standards (or .claude/CLAUDE.md)
  .mcp.json                          step 4: shared server, secrets as ${VAR}
  .claude/
    rules/
      api.md                         step 2: paths: ["src/api/**/*"]
      testing.md                     step 2: paths: ["**/*.test.*"]
    skills/
      <skill-name>/SKILL.md          step 3: context: fork
    commands/
      review.md                      the /review command from Question 4
~/.claude.json                       step 4: your personal experimental server
```

**How to build it**

- **Step 1: project CLAUDE.md.** Commit it at `./CLAUDE.md` or `./.claude/CLAUDE.md`. Anything a teammate needs must not live only in `~/.claude/CLAUDE.md`, which applies to you alone; that is the guide's example of a hierarchy bug (3.1-S1). To check what a session actually loaded, run `/context`. The guide names `/memory` for this job (3.1-S4); the difference is explained in the warning under [Task 3.1](#task-31-configure-claudemd-files-with-appropriate-hierarchy-scoping-and-modular-organization). Expect `/memory` in exam answers; use `/context` in practice. Test from a second user account or a fresh clone.

- **Step 2: path-scoped rules.** Put a `paths` list in each rule file's YAML frontmatter, using the guide's two patterns. To test, start a fresh session and run `/context`, ask Claude to read a file under `src/api/`, then run `/context` again: the API rule should now appear, and the testing rule should not. The test asks Claude to read, not edit, because as of September 2026 the docs say path-scoped rules trigger when Claude reads a matching file, where the guide says "editing" (see the warning under [Task 3.3](#task-33-apply-path-specific-rules-for-conditional-convention-loading)). Answer exam items in the guide's "editing" terms. A rule file without `paths` loads at launch in every session. For a written log, an `InstructionsLoaded` hook can log which rules loaded, when and why.

- **Step 3: a forked skill.** Give the skill a verbose job, such as a codebase analysis, and set `context: fork`. The [skills docs](https://code.claude.com/docs/en/skills) warn: "The subagent doesn't see your conversation history, so the skill's instructions have to stand on their own." To verify isolation, compare `/context` before and after the run: the exploration output should stay out of the main conversation. As of September 2026, the guide and the docs disagree on `allowed-tools`: the guide treats it as a restriction, the docs as a pre-approval, and the field that removes tools is `disallowed-tools` (see the warning under [Task 3.2](#task-32-create-and-configure-custom-slash-commands-and-skills)). Expect the guide's framing on the exam; in a real repository, use `disallowed-tools` to restrict.

- **Step 4: two MCP scopes.** Add the shared server with `claude mcp add --scope project`, which writes `.mcp.json` at the project root, and reference credentials as `${VAR}` (or `${VAR:-default}`) so no secret is committed. Add the personal server with `--scope user`, which stores it in `~/.claude.json`. Run `claude mcp list`, or `/mcp` inside a session, to confirm both are connected at once. Interactive sessions ask for approval before using project-scoped servers. As of September 2026 the docs describe a third scope, local, which is the default (see the warning under [Task 2.4](#task-24-integrate-mcp-servers-into-claude-code-and-agent-workflows)); expect the guide's two scopes on the exam. One more caveat: in a remote server's `url` and `headers`, Claude Code reads certain credential variables (the docs give examples such as `ANTHROPIC_API_KEY`) as empty, so test that your own variable expands.

- **Step 5: plan mode against direct execution.** Enter plan mode with `Shift+Tab` or by starting a prompt with `/plan`; `Ctrl+G` opens the proposed plan in your editor. In plan mode, the [permission modes docs](https://code.claude.com/docs/en/permission-modes) say, Claude reads, explores and writes a plan "but does not edit your source". The guide's task 3.4 bullets predict the outcomes: the single-file bug fix with a clear stack trace suits direct execution (3.4-S2); the multi-file library migration suits planning first, then executing the plan (3.4-S4); the feature with several valid approaches suits plan mode (3.4-K1). The [best practices page](https://code.claude.com/docs/en/best-practices) gives a matching rule of thumb: "If you could describe the diff in one sentence, skip the plan." For each task, note whether the plan changed what Claude did. That note is the deliverable: the step asks you to observe when plan mode provides value.

**Done when**

- [ ] A teammate on a fresh clone gets the same standards you do.
- [ ] `/context` lists the API rule only after Claude reads a file under `src/api/`.
- [ ] The forked skill's exploration output never appears in the main conversation.
- [ ] Both MCP servers are available in one session, and `.mcp.json` contains no secret.
- [ ] You can say, for each of the three tasks, why plan mode did or did not pay off.

Go deeper: [CLAUDE.md and the memory hierarchy](knowledge/claude-code-configuration.md#claudemd-and-the-memory-hierarchy), [path-scoped rules](knowledge/claude-code-configuration.md#path-scoped-rules), [Agent Skills](knowledge/claude-code-configuration.md#agent-skills), [MCP servers in Claude Code](knowledge/claude-code-configuration.md#mcp-servers-in-claude-code), [plan mode or direct execution](knowledge/claude-code-workflows.md#plan-mode-or-direct-execution).

### Exercise 3: Build a Structured Data Extraction Pipeline

**Objective, from the [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf):** "Practice designing JSON schemas, using tool_use for structured output, implementing validation-retry loops, and designing batch processing strategies." **Domains reinforced:** 4 and 5.

**The guide's steps:**

1. Define an extraction tool with a JSON schema containing required and optional fields, an enum with an "other" + detail string pattern, and nullable fields for information that may not exist in source documents. Process documents where some fields are absent and verify the model returns null rather than fabricating values.
2. Implement a validation-retry loop: when Pydantic or JSON schema validation fails, send a follow-up request including the document, the failed extraction, and the specific validation error. Track which errors are resolvable via retry (format mismatches) versus which are not (information absent from source).
3. Add few-shot examples demonstrating extraction from documents with varied formats (e.g., inline citations vs bibliographies, narrative descriptions vs structured tables) and verify improved handling of structural variety.
4. Design a batch processing strategy: submit a batch of 100 documents using the Message Batches API, handle failures by custom_id, resubmit failed documents with modifications (e.g., chunking oversized documents), and calculate total processing time relative to SLA constraints.
5. Implement a human review routing strategy: have the model output field-level confidence scores, route low-confidence extractions to human review, and analyze accuracy by document type and field to verify consistent performance.

**How to build it**

- **Step 1: the extraction tool.** An illustrative definition for invoices (ours, not an official example). Every key except `purchase_order` is listed in `required`; the ones a document may lack are nullable, so the model can answer null instead of inventing a value; `purchase_order` is optional, so it can be left out entirely.

    ```json
    {
      "name": "extract_invoice",
      "description": "Extract invoice fields from the document in the user message. Use it once per document. Return null for any field the document does not state; never estimate a value. Put the total printed on the document in stated_total and your own sum of the line items in calculated_total.",
      "strict": true,
      "input_schema": {
        "type": "object",
        "properties": {
          "invoice_number": {"type": ["string", "null"]},
          "invoice_date": {"type": ["string", "null"]},
          "category": {"type": "string", "enum": ["goods", "services", "subscription", "other", "unclear"]},
          "category_detail": {"type": ["string", "null"]},
          "stated_total": {"type": ["number", "null"]},
          "calculated_total": {"type": ["number", "null"]},
          "conflict_detected": {"type": "boolean"},
          "purchase_order": {"type": "string"}
        },
        "required": ["invoice_number", "invoice_date", "category", "category_detail", "stated_total", "calculated_total", "conflict_detected"],
        "additionalProperties": false
      }
    }
    ```

    `category_detail` carries the free text when `category` is "other"; "unclear" covers documents that do not settle the question. `strict` sits at the top level of the tool definition, next to `name`, `description` and `input_schema`. Strict schemas need `additionalProperties: false` on objects, and the [structured outputs docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) cap all strict schemas in one request at 24 optional parameters and 16 union-typed parameters such as `"type": ["string", "null"]`, so budget nullable and optional fields in large schemas (this example uses five union-typed fields and one optional field). Structured outputs do not guarantee the capitalization of enum values, so compare them case-insensitively. To make Claude call the tool, the guide teaches `tool_choice` set to `"any"` or to the named tool. As of September 2026 some models and settings reject both; use the replacement described in the warning on forced tool choice under [Task 4.3](#task-43-enforce-structured-output-using-tool-use-and-json-schemas) (`"auto"` with strict tool use and an instruction to call the tool, or JSON outputs through `output_config.format`). Expect the guide's `tool_choice` logic on the exam.

- **Step 2: validation and retry.** Validate every extraction with Pydantic. According to the [structured outputs docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs), the Python SDK's `client.messages.parse()` "automatically transforms your Pydantic model, validates the response, and returns a `parsed_output` attribute" for JSON outputs; with tool use, validate the tool's input by constructing Pydantic models from it, as Anthropic's [Pydantic tool-use cookbook](https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/tool_use_with_pydantic.ipynb) does. When validation fails, send one follow-up that carries three things: the original document, the failed extraction and the exact validation error. Keep a tally of outcomes. The guide expects format mismatches and structural errors to clear on retry, and retries to be ineffective when the information is absent from the source; the fix there is a nullable field, not another attempt. Strict schemas remove syntax errors but not semantic ones, so add semantic checks of your own: compare `calculated_total` with `stated_total` in code, and have the model set `conflict_detected` when the source contradicts itself (the guide's self-correction pattern, 4.4-S4).

- **Step 3: few-shot examples.** Choose examples that differ in structure: inline citations against a bibliography, a narrative description against a table. The guide gives 2 to 4 targeted examples for ambiguous scenarios (4.2-S1) and no number for format-variety examples (4.2-S4, 4.2-S5); the [prompting docs](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) suggest three to five, diverse, and wrapped in `<example>` tags "so Claude can distinguish them from instructions." Measure the effect: label a held-out set of documents with the correct values, run it with and without the examples, and count fields that come back empty or null although the document states the value. A null for a value the document lacks is the correct answer (Step 1), so do not count it as a miss.

- **Step 4: the batch.** A batch holds up to 100,000 requests or 256 MB, whichever comes first, so 100 documents normally go in one batch. Give each request a meaningful `custom_id`; the [batch docs](https://platform.claude.com/docs/en/build-with-claude/batch-processing) say it "Must be 1 to 64 characters and contain only alphanumeric characters, hyphens, and underscores". Dry-run one request on the synchronous Messages API first, because validation of each request's `params` runs asynchronously and its errors come back only when the whole batch has ended. Refine the prompt on a small sample before you submit the full set (4.5-S4). Results can arrive in any order, so match them by `custom_id`, and download them within 29 days of creating the batch. This loop is adapted from the batch docs example to collect what needs resubmitting:

    ```python
    fix_then_resend, resend_as_is = [], []
    for result in client.messages.batches.results(batch_id):
        outcome = result.result
        match outcome.type:
            case "succeeded":
                save_extraction(result.custom_id, outcome.message)  # your own storage
            case "errored":
                if outcome.error.error.type == "invalid_request_error":
                    fix_then_resend.append(result.custom_id)  # fix the request body first
                else:
                    resend_as_is.append(result.custom_id)
            case "canceled" | "expired":
                resend_as_is.append(result.custom_id)
    ```

    Resubmit only the failed documents, identified by `custom_id`, splitting any that exceeded the context limit into chunks under new `custom_id` values (4.5-S3). Errored, canceled and expired requests are not billed. For the SLA arithmetic, use the guide's own example, worked under [Task 4.5](#task-45-design-efficient-batch-processing-strategies): submissions every 4 hours plus up to 24 hours of processing give a worst case of 28 hours inside a 30-hour SLA (our arithmetic). A second batch round for resubmitted documents can add up to another 24 hours, which that SLA cannot absorb (our arithmetic); decide in your design how a failed document still meets the deadline.

- **Step 5: human review routing.** Ask for a confidence score next to each extracted field, not one score per document. Route fields below your threshold, and documents whose sources are ambiguous or contradictory, to a reviewer. The guide's pattern is to set that threshold from a labeled validation set rather than by guesswork. Then break accuracy down by document type and by field: a strong overall figure can hide a weak segment, which is why the guide wants every segment checked before human review is reduced. Keep a stratified random sample of high-confidence extractions flowing to reviewers so that new error patterns still get caught. This is the calibrated confidence the guide endorses, as distinct from the raw self-reported confidence that Question 3 rejects as an escalation trigger.

**Done when**

- [ ] Documents that lack a field return null for it, never a plausible invented value.
- [ ] Your retry log separates errors that retries fixed from errors they could not fix.
- [ ] On varied layouts, the few-shot examples measurably reduce fields returned empty or null when the document does state the value, without adding invented values.
- [ ] Only the failed requests, identified by `custom_id`, are resubmitted (with fixes such as chunking where needed), and your worst-case timeline is written against the SLA.
- [ ] You have an accuracy table by document type and field, and review routing based on field-level confidence.

Go deeper: [structured output with tools and JSON schemas](knowledge/prompt-engineering.md#structured-output-with-tools-and-json-schemas), [validation, retry and feedback loops](knowledge/prompt-engineering.md#validation-retry-and-feedback-loops), [few-shot examples](knowledge/prompt-engineering.md#few-shot-examples), [Message Batches](knowledge/claude-api.md#message-batches), [human review and confidence calibration](knowledge/evaluation-and-reliability.md#human-review-and-confidence-calibration).

### Exercise 4: Design and Debug a Multi-Agent Research Pipeline

**Objective, from the [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf):** "Practice orchestrating subagents, managing context passing, implementing error propagation, and handling synthesis with provenance tracking." **Domains reinforced:** 1, 2 and 5.

**The guide's steps:**

1. Build a coordinator agent that delegates to at least two subagents (e.g., web search and document analysis). Ensure the coordinator's allowedTools includes "Task" and that each subagent receives its research findings directly in its prompt rather than relying on automatic context inheritance.
2. Implement parallel subagent execution by having the coordinator emit multiple Task tool calls in a single response. Measure the latency improvement compared to sequential execution.
3. Design structured output for subagents that separates content from metadata: each finding should include a claim, evidence excerpt, source URL/document name, and publication date. Verify that the synthesis subagent preserves source attribution when combining findings.
4. Implement error propagation: simulate a subagent timeout and verify the coordinator receives structured error context (failure type, attempted query, partial results). Test that the coordinator can proceed with partial results and annotate the final output with coverage gaps.
5. Test with conflicting source data (e.g., two credible sources with different statistics) and verify the synthesis output preserves both values with source attribution rather than arbitrarily selecting one, and structures the report to distinguish well-established from contested findings.

**How to build it**

- **Step 1: coordinator and subagents.** Define each subagent with an `AgentDefinition`: a `description` saying when to use it, a `prompt` that acts as its system prompt, and a `tools` list limited to its role (for example, read-only analysis gets Read, Grep and Glob). The guide says `allowedTools` (the TypeScript spelling; Python uses `allowed_tools`) must include "Task". As of September 2026 the spawning tool is named `Agent`, `Task` is still accepted as an alias, and `allowed_tools` auto-approves tools rather than gating them (see the warning under [Task 1.3](#task-13-configure-subagent-invocation-context-passing-and-spawning)). Keep `Agent` in the list, as the docs examples do, and expect "Task" on the exam. The Agent tool's prompt string is the only context a subagent gets from its parent, so when the coordinator calls the synthesis subagent, paste the search results and analysis outputs into that prompt.

- **Step 2: parallel spawning and the measurement.** Parallelism comes from several Agent (Task) calls in one coordinator response, not from calls spread across turns. To confirm it, count the spawning tool_use blocks in a single assistant message; the SDK docs say the tool appears as `"Agent"` in `tool_use` blocks but as `"Task"` in the `system:init` tools list. To measure it, run the same topic twice: once with a coordinator prompt that delegates one subtopic at a time, once asking for all subagents together, and record wall-clock time. The [SDK subagents docs](https://code.claude.com/docs/en/agent-sdk/subagents) explain the gain: "multiple subagents can run concurrently, so independent subtasks finish in the time of the slowest one rather than the sum of all of them." In Anthropic's June 2025 [multi-agent research post](https://www.anthropic.com/engineering/multi-agent-research-system), parallel subagents plus parallel tool calls "cut research time by up to 90% for complex queries"; your figure depends on your tools and topic.

- **Step 3: findings that keep their sources.** Have every subagent return findings in a fixed shape that keeps content apart from metadata. An illustrative shape (our field names, carrying the four items the step lists):

    ```json
    {
      "claim": "One-sentence finding",
      "evidence_excerpt": "The passage from the source that supports it",
      "source": "URL or document name",
      "publication_date": "YYYY-MM-DD"
    }
    ```

    The [SDK subagents docs](https://code.claude.com/docs/en/agent-sdk/subagents) say the parent "may summarize it in its own response" when it receives a subagent's final message, so tell the coordinator to pass findings to synthesis unchanged, and tell the synthesis subagent to keep each claim's source and date. Verify by tracing three claims in the final report back to their sources.

- **Step 4: a forced timeout.** Make the search tool stall past its timeout for one subtopic. The search subagent should first try local recovery (a retry or a reworded query), and only if that fails return structured error context: failure type, attempted query, partial results and possible alternatives. Two SDK behaviors matter while you test this, per the [SDK subagents docs](https://code.claude.com/docs/en/agent-sdk/subagents). First, "An API error that ends the subagent early, such as a rate limit, is never delivered as its result." A stalled tool of your own is a different case, and the structured error context is yours to build. Second, when a subagent stops at its `maxTurns` limit, Claude Code marks its output in the Agent tool result as partial, so Claude knows the run is unfinished. The coordinator should carry on with what it has and state the gap in the report (5.3-S4). The wrong designs are the ones Question 8 rejects: a generic "search unavailable" status that hides the context, an empty result marked as success, and a failure that ends the whole workflow.

- **Step 5: conflicting sources.** Feed in two credible sources that give different figures for the same statistic. The report should show both, each with its source, and separate well-established findings from contested ones. Anthropic's open-source [research subagent prompt](https://github.com/anthropics/claude-cookbooks/blob/main/patterns/agents/prompts/research_subagent.md) gives a matching instruction: "If unable to reconcile facts, include the conflicting information in your final task report for the lead researcher to resolve." Add a second pair whose figures differ only because they come from different years, and check that the report treats it as change over time rather than a contradiction; that is what the publication dates are for.

**Done when**

- [ ] Every subagent prompt contains all the findings it needs; nothing relies on inherited context.
- [ ] One coordinator response spawns two or more subagents, and you have a measured sequential-versus-parallel comparison.
- [ ] Every claim in the final report traces to a source and a date.
- [ ] A forced timeout produces a report with a stated coverage gap, not a crash and not a silent omission.
- [ ] Conflicting figures appear side by side with attribution, and a difference between years is not labeled a conflict.

Go deeper: [multi-agent orchestration](knowledge/agents-and-agent-sdk.md#multi-agent-orchestration), [subagents in the SDK](knowledge/agents-and-agent-sdk.md#subagents-in-the-sdk), [multi-agent handoffs](knowledge/context-engineering.md#multi-agent-handoffs), [error propagation in multi-agent systems](knowledge/evaluation-and-reliability.md#error-propagation-in-multi-agent-systems), [provenance and uncertainty in synthesis](knowledge/context-engineering.md#provenance-and-uncertainty-in-synthesis).

## What is out of scope

The appendix of the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) draws the exam's boundary with three lists: technologies and concepts that "might appear on the exam", topics that "are explicitly tested on the exam", and related topics that "will not appear on the exam". The [program FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) calls the guide "the authoritative source for exam scope". Use the exclusions to decide what not to study for CCAR-F, and the two in-scope lists as a final coverage check.

### Out-of-scope topics

The guide's list, word for word:

- Fine-tuning Claude models or training custom models
- Claude API authentication, billing, or account management
- Detailed implementation of specific programming languages or frameworks (beyond what's needed for tool and schema configuration)
- Deploying or hosting MCP servers (infrastructure, networking, container orchestration)
- Claude's internal architecture, training process, or model weights
- Constitutional AI, RLHF, or safety training methodologies
- Embedding models or vector database implementation details
- Computer use (browser automation, desktop interaction)
- Vision/image analysis capabilities
- Streaming API implementation or server-sent events
- Rate limiting, quotas, or API pricing calculations
- OAuth, API key rotation, or authentication protocol details
- Specific cloud provider configurations (AWS, GCP, Azure)
- Performance benchmarking or model comparison metrics
- Prompt caching implementation details (beyond knowing it exists)
- Token counting algorithms or tokenization specifics

### Where the boundary sits

Several exclusions sit right next to something the [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) does test. Read the qualifiers in parentheses; they mark where the line falls.

| Not tested | Still tested nearby |
|---|---|
| Claude API authentication; OAuth, API key rotation, or authentication protocol details | Keeping credentials out of version control with environment variable expansion in `.mcp.json` (2.4-K2, 2.4-S1) |
| Deploying or hosting MCP servers | Configuring them: project versus user scope, environment variable expansion and several servers at once (in-scope topic 6), and designing their tools and resources (in-scope topic 5) |
| Rate limiting, quotas, or API pricing calculations | The Message Batches API's "50% cost savings" as a design input, weighed in Question 11 against its lack of a guaranteed latency SLA (4.5-K1) |
| Prompt caching implementation details | Knowing that prompt caching exists; the exclusion's own qualifier says so |
| Token counting algorithms or tokenization specifics | Token budgets (an Appendix entry under context window management), and trimming verbose tool output before it accumulates in context (5.1-K3, 5.1-S3) |
| Detailed implementation of specific programming languages or frameworks | What tool and schema configuration needs: JSON Schema design (4.3) and validation-retry loops (4.4), with Pydantic named in the Appendix and in Exercise 3, step 2 |
| Performance benchmarking or model comparison metrics | Question 12 rejects "Switch to a higher-tier model with a larger context window" as the fix for an attention problem; the expected lever is restructuring the work |

### Out of scope here, tested on other exams

Most of these exclusions appear, in some form, in the objectives or recommended experience of the other three Claude exam guides. The wording below is quoted from the [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf), the [CCAR-P guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf) and the [CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf). If you plan a second exam, study these topics for that one; for CCAR-F, leave them.

| Excluded from CCAR-F | Where another guide covers it | That guide's wording |
|---|---|---|
| Streaming; vision | CCDV-F [Domain 2](claude-certified-developer.md#domain-2-applications-and-integration), Claude API Mechanics | "messages, tools, streaming, vision, thinking, caching, invoking Claude through third-party vendors" |
| Prompt caching implementation | CCDV-F [Domain 5](claude-certified-developer.md#domain-5-model-selection-and-optimization), Cost and Token Management; CCAR-P [Domain 2](claude-certified-architect-professional.md#domain-2-claude-models-prompting-context-engineering) | "caching techniques (prompt caching, cache check-pointing)"; "Implement prompt reuse strategies (caching, modular prompts, Skills)" |
| API pricing calculations | CCDV-F Domain 5, Cost and Token Management; CCAR-P [Domain 4](claude-certified-architect-professional.md#domain-4-evaluation-testing-optimization) | "token usage tracking, cost modeling"; "Optimize token usage, latency, and cost-performance trade-offs" |
| Tokenization specifics | CCDV-F Domain 5, LLM Fundamentals (tokens as a concept, not tokenization algorithms) | "Basic understanding of LLMs (tokens, context windows, sampling, non-determinism, next-token generation)" |
| Authentication and API keys | CCDV-F [Domain 7](claude-certified-developer.md#domain-7-security-and-safety), Identity, Secrets, and Key Management; CCAR-P [Domain 3](claude-certified-architect-professional.md#domain-3-integration) | "Managing secrets, credentials, and API keys across Claude development and production environments"; "Analyze authentication and authorization requirements to identify security gaps" |
| Deploying or hosting MCP servers | CCDV-F [Domain 8](claude-certified-developer.md#domain-8-tools-and-mcps), MCP Server Development | "server authoring, deployment, integration with Claude applications" |
| Cloud provider configurations | CCDV-F Domain 2, Claude API Mechanics (calling Claude through vendors, not configuring them) | "invoking Claude through third-party vendors" |
| Embedding models and vector databases | CCAR-P Domain 3, Integration; its Sample 3 rationale also names embeddings | "Design a RAG pipeline with appropriate chunking and indexing strategies"; "a broken re-index or mismatched embeddings" |
| Model comparison | CCDV-F Domain 5, Model Selection and Tradeoffs; CCAO-F [Domain 3](claude-certified-associate.md#domain-3-product-and-model-selection); CCAR-P Domain 2 | "Opus vs. Sonnet vs. Haiku use cases"; "Differentiate between Claude model types (Haiku, Sonnet, Opus)"; "Select appropriate Claude models based on trade-offs" |
| Programming language detail | CCDV-F recommended experience, and Domain 2, Software Engineering Foundations | "Proficiency in Python and/or TypeScript"; "REST APIs, JSON, asynchronous programming" |

None of the other three guides names fine-tuning, Constitutional AI, RLHF or computer use anywhere in its text. For readers sitting more than one exam, the knowledge base covers the shared topics: [streaming](knowledge/claude-api.md#streaming), [vision and PDF input](knowledge/claude-api.md#vision-and-pdf-input), [prompt caching](knowledge/claude-api.md#prompt-caching), [tokens and counting](knowledge/claude-api.md#tokens-context-windows-and-counting), [cost and usage tracking](knowledge/claude-api.md#cost-and-usage-tracking), [Claude on cloud platforms](knowledge/claude-api.md#claude-on-cloud-platforms), [secrets and API keys](knowledge/security-and-governance.md#secrets-and-api-keys), [building an MCP server](knowledge/tool-use-and-mcp.md#building-an-mcp-server), [retrieval-augmented generation](knowledge/solution-architecture.md#retrieval-augmented-generation) and [choosing a model](knowledge/claude-api.md#models-and-how-to-choose-one).

### In-scope topics

The [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) list of topics that "are explicitly tested on the exam", word for word, with the part of this page that teaches each:

| # | Topic, as the guide words it | Where this page teaches it (our mapping) |
|---|---|---|
| 1 | Agentic loop implementation: control flow based on stop_reason, tool result handling, loop termination conditions | [Domain 1](#domain-1-agentic-architecture-orchestration), task 1.1 |
| 2 | Multi-agent orchestration: coordinator-subagent patterns, task decomposition, parallel subagent execution, iterative refinement loops | [Domain 1](#domain-1-agentic-architecture-orchestration), tasks 1.2, 1.3 and 1.6 |
| 3 | Subagent context management: explicit context passing, structured state persistence, crash recovery using manifests | [Domain 1](#domain-1-agentic-architecture-orchestration), task 1.3; [Domain 5](#domain-5-context-management-reliability), task 5.4 |
| 4 | Tool interface design: writing effective tool descriptions, splitting vs consolidating tools, tool naming to reduce ambiguity | [Domain 2](#domain-2-tool-design-mcp-integration), task 2.1 |
| 5 | MCP tool and resource design: resources for content catalogs, tools for actions, description quality for adoption | [Domain 2](#domain-2-tool-design-mcp-integration), task 2.4 |
| 6 | MCP server configuration: project vs user scope, environment variable expansion, multi-server simultaneous access | [Domain 2](#domain-2-tool-design-mcp-integration), task 2.4 |
| 7 | Error handling and propagation: structured error responses, transient vs business vs permission errors, local recovery before escalation | [Domain 2](#domain-2-tool-design-mcp-integration), task 2.2; [Domain 5](#domain-5-context-management-reliability), task 5.3 |
| 8 | Escalation decision-making: explicit criteria, honoring customer preferences, policy gap identification | [Domain 5](#domain-5-context-management-reliability), task 5.2 |
| 9 | CLAUDE.md configuration: hierarchy (user/project/directory), @import patterns, .claude/rules/ with glob patterns | [Domain 3](#domain-3-claude-code-configuration-workflows), tasks 3.1 and 3.3 |
| 10 | Custom commands and skills: project vs user scope, context: fork, allowed-tools, argument-hint frontmatter | [Domain 3](#domain-3-claude-code-configuration-workflows), task 3.2 |
| 11 | Plan mode vs direct execution: complexity assessment, architectural decisions, single-file changes | [Domain 3](#domain-3-claude-code-configuration-workflows), task 3.4 |
| 12 | Iterative refinement: input/output examples, test-driven iteration, interview pattern, sequential vs parallel issue resolution | [Domain 3](#domain-3-claude-code-configuration-workflows), task 3.5 |
| 13 | Structured output via tool_use: schema design, tool_choice configuration, nullable fields to prevent hallucination | [Domain 4](#domain-4-prompt-engineering-structured-output), task 4.3; [Domain 2](#domain-2-tool-design-mcp-integration), task 2.3 |
| 14 | Few-shot prompting: ambiguous scenario targeting, format consistency, false positive reduction | [Domain 4](#domain-4-prompt-engineering-structured-output), task 4.2 |
| 15 | Batch processing: Message Batches API appropriateness, latency tolerance assessment, failure handling by custom_id | [Domain 4](#domain-4-prompt-engineering-structured-output), task 4.5 |
| 16 | Context window optimization: trimming verbose tool outputs, structured fact extraction, position-aware input ordering | [Domain 5](#domain-5-context-management-reliability), task 5.1 |
| 17 | Human review workflows: confidence calibration, stratified sampling, accuracy segmentation by document type and field | [Domain 5](#domain-5-context-management-reliability), task 5.5 |
| 18 | Information provenance: claim-source mappings, temporal data handling, conflict annotation, coverage gap reporting | [Domain 5](#domain-5-context-management-reliability), tasks 5.6 and 5.3 |

### Technologies and concepts that might appear

The guide's list, word for word. The guide separates each area from its contents with a dash, shown here as the column break; for built-in tools, the tool names sit between two dashes and appear here in parentheses in the Area column.

| Area | What the guide lists |
|---|---|
| Claude Agent SDK | agent definitions, agentic loops, stop_reason handling, hooks (PostToolUse, tool call interception), subagent spawning via Task tool, allowedTools configuration |
| Model Context Protocol (MCP) | MCP servers, MCP tools, MCP resources, isError flag, tool descriptions, tool distribution, .mcp.json configuration, environment variable expansion |
| Claude Code | CLAUDE.md configuration hierarchy (user/project/directory), .claude/rules/ with YAML frontmatter path-scoping, .claude/commands/ for slash commands, .claude/skills/ with SKILL.md frontmatter (context: fork, allowed-tools, argument-hint), plan mode, direct execution, /memory command, /compact, --resume, fork_session, Explore subagent |
| Claude Code CLI | -p / --print flag for non-interactive mode, --output-format json, --json-schema for structured CI output |
| Claude API | tool_use with JSON schemas, tool_choice options ("auto", "any", forced tool selection), stop_reason values ("tool_use", "end_turn"), max_tokens, system prompts |
| Message Batches API | 50% cost savings, up to 24-hour processing window, custom_id for request/response correlation, polling for completion, no multi-turn tool calling support |
| JSON Schema | required vs optional fields, enum types, nullable fields, "other" + detail string patterns, strict mode for syntax error elimination |
| Pydantic | schema validation, semantic validation errors, validation-retry loops |
| Built-in tools (Read, Write, Edit, Bash, Grep, Glob) | their purposes and selection criteria |
| Few-shot prompting | targeted examples for ambiguous scenarios, format demonstration, generalization to novel patterns |
| Prompt chaining | sequential task decomposition into focused passes |
| Context window management | token budgets, progressive summarization, lost-in-the-middle effects, context extraction, scratchpad files |
| Session management | session resumption, fork_session, named sessions, session context isolation |
| Confidence scoring | field-level confidence, calibration with labeled validation sets, stratified sampling for error rate measurement |

!!! warning "Where the docs now differ from the guide's terms (as of September 2026)"

    Several entries in this list read differently in the current product documentation, among them the Task tool (now Agent), tool call interception (the `PreToolUse` hook event), `.claude/commands/` (merged into skills), the `tool_choice` options, `allowed-tools`, `/memory`, the Glob and Grep defaults, and MCP scopes. The guide is the authoritative source for exam scope, so answer in its terms. Each difference is listed in [Where current documentation differs from the guide](#where-current-documentation-differs-from-the-guide).

### If a live item seems out of scope

The [certification policies page](https://anthropic-partners.skilljar.com/page/policies-certifications) says: "If a question looks factually wrong, unclear, has more than one defensible answer, or doesn't match the exam guide, report it to Pearson." Anyone can report, pass or fail, and the page adds that "reporting a problem never counts against you or affects your result." Appeals are a separate route. The same page says an appeal of a failed result can cover "a question you think was faulty or an exam that didn't match the published exam guide", and that if Anthropic confirms a question was faulty and it affected the result, "the remedy is a free retake rather than a changed score." The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says "The standard-setting outcome and the content of individual exam items are not subject to appeal." The two documents differ on appeals, while the guide's appeal rule says nothing about reports, so report a suspect item either way. Both documents set the appeal window for a result at 14 days from the exam date.

## What candidates report

Each row below is a candidate who published their own result, linked to their post. Scores are self-reported and copied exactly as each candidate wrote them. The [certification FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says scaled scoring "equates scores across exam forms that may have slightly different difficulty", and neither the four exam guides nor the FAQ publish a pass rate. Read these accounts as individual experiences, not as odds.

!!! warning "Two eras of the same exam"

    On June 30, 2026 the program moved exam delivery to Pearson and digital badges to Credly. Reports about exams sat before then describe the older setup, when the exam was commonly called CCA-F or CCAF: a 6-month lockout after a fail, 6-month validity, no score at the end of the exam, and an official practice exam. None of that applies now. Retakes wait 14, 30 or 90 days, credentials last 12 months, the score shows on screen when you finish, and the practice exam was retired in the move. Failed attempts from before the move were cleared, and credentials earned before it were extended to 12 months. What older reports say about question style may still hold; what they say about logistics does not.

### Reports published after the move to Pearson (July 2026 onward)

Dates are the exam date where the candidate gives one, otherwise the date of the post or video.

| Candidate and source | Date | Result as published | What they add |
|---|---|---|---|
| [ahmdhsn-dev](https://dev.to/ahmdhsn-dev/how-i-scored-9101000-on-ccaf-claude-certification-and-open-sourced-my-last-cheat-sheet-5ae1) (dev.to) | Exam on July 15, 2026 | 910/1000 | Read the official guide at least three times. A third-party mock went from 850/1000 on the first run to "consistently 950+" before booking. |
| [OkRelationship3427](https://www.reddit.com/r/ClaudeAI/comments/1v5zrru/passed_the_ccarf_with_9041000/) (Reddit) | Posted July 25, 2026 | 904/1000 | Took CCAR-F after passing CCAO-F. Saw no multiple-select items: "all of them were single-answer four-choice questions". |
| [bills70](https://www.reddit.com/r/ClaudeAI/comments/1vaxsws/claude_certified_architect_foundations_ccarf_how/) (Reddit) | Posted July 30, 2026 | Passed (no score given) | One shared scenario context covered a series of questions, and each question carried enough detail to answer on its own. Finished 15 minutes early. |
| [guruduttrai](https://www.reddit.com/r/ClaudeAI/comments/1v2x1p7/got_all_4_claude_certifications_ccap_ccaf_ccdvf/) (Reddit) | Exam on August 3, 2026 | Failed, 696/1000 | 24 points under the 720 cut score (our arithmetic). |
| [InRussYouTrust](https://www.youtube.com/watch?v=SYYtM16wXcI) (YouTube, video description) | Video from August 9, 2026 | Failed the first attempt, then "passed weeks later" | Attempt dates not given. Credits a change of approach, including spotting wrong answers that patch symptoms instead of removing the failure mechanism. |
| [cs135dev](https://www.reddit.com/r/ClaudeCertified/comments/1vo909f/passed_ccarp_exam_harder_than_expected/) (Reddit) | Posted August 14, 2026 | 848 (given in a later comment) | Registered for CCAR-P right after passing CCAR-F, sat it a couple of weeks later at a test center, and passed with 885. |
| [Build With Why AI](https://www.youtube.com/watch?v=F2eUnVQOd6Y) (YouTube, video description) | Video from August 23, 2026 | Both Architect exams "over 900 out of 1000" | Four days of preparation. Calls Foundations "a mechanics exam". |
| [WenHao Yu](https://yu-wenhao.com/en/blog/ccar-f-exam-experience/) (blog, Pearson test center) | Published August 29, 2026 | 882 | A bit over a month of focused preparation. Third-party mock scores of 766, 846, 901, 903 and 915. Finished all 60 questions with 15 minutes left. |
| [bluepanda](https://dev.to/bluepanda/i-cleared-all-4-anthropic-claude-certifications-what-you-actually-need-to-know-and-what-to-skip-224m) (dev.to) | Published September 6, 2026 | Passed all four exams (scorecard shown only as an image) | Took CCAR-F first, then the other three online through Pearson VUE over the next seven days. Called CCAR-F "by far the hardest exam of all four" and said its time "feels tighter because of the reading volume". |

### Earlier reports (exams sat before the move to Pearson on June 30, 2026)

| Candidate and source | Date | Result as published | What they add |
|---|---|---|---|
| [abinashteja](https://www.reddit.com/r/ClaudeAI/comments/1ruf70b/just_passed_the_new_claude_certified_architect/) (Reddit) | March 20, 2026 | Passed, with an early-adopter badge | Prepared with the exam guide and the practice test. Found it "not very easy" despite 3+ years of GenAI experience. |
| [Sarvesh Talele](https://newsletter.bigtechcareers.com/p/step-by-step-guide-to-achieve-claude-certification) (guest post, Big Tech Careers) | Published April 2, 2026 | 911 | "roughly 35 to 40 hours of focused preparation" over five days. Used almost all of the exam time. |
| [Kishor Kukreja](https://medium.com/@kishorkukreja/i-passed-anthropics-claude-certified-architect-foundations-exam-with-a-score-of-893-1000-2206c27efd6c) (Medium) and [Suspicious_Low7612](https://www.reddit.com/r/ClaudeAI/comments/1sgn0cf/passed_anthropics_claude_certified_architect/) (Reddit) | April 2026 | 893/1000 each | Both describe building agentic supply-chain systems for enterprise clients, so they may be the same person. Suspicious_Low7612 found the real questions "noticeably trickier" than the practice test. |
| [Suraj Khaitan](https://dev.to/suraj_khaitan_f893c243958/i-passed-the-claude-certified-architect-foundations-cca-f-exam-my-journey-lessons-and-98j) (dev.to) | April 26, 2026 | Passed on the first attempt (no score given) | Finished with 10 minutes to spare. Found that several options were technically correct but only one was production-grade. |
| [service_account](https://www.reddit.com/r/ClaudeAI/comments/1sxac8o/ccaf_questions/) (Reddit) | April 27, 2026 | Failed, 590/1000 | Had scored 1000/1000 on the official practice exam three times. |
| [Upstairs_Muffin_7035](https://www.reddit.com/r/ClaudeAI/comments/1ruf70b/just_passed_the_new_claude_certified_architect/) (Reddit) | April 28, 2026 | 846 | Took the official mock three times until reaching 1000; scored about 780 on a community mock at the first try. |
| [John Weidner](https://verygood.ventures/blog/passing-the-claude-certified-architect-exam/) (Very Good Ventures blog) | Exam on May 30, 2026 | 738 (44 of 60 correct) | Ten days of preparation and three practice runs. Reached the last question with eighteen minutes left and changed about half of the flagged answers on review. |
| [AK (Tan Aik Keong)](https://www.claudeaimalaysia.com/malaysia-claude-ccaf-studyguide.html) (claudeaimalaysia.com) | Exam on June 3, 2026 | 822 (48 of 60 correct) | Finished in under an hour after weeks of preparation. Their company runs the site and sells a prep class. |
| [Matthew Purcell](https://www.linkedin.com/pulse/claude-certification-exams-honest-review-matthew-purcell-byo2e) (LinkedIn) | Exam on June 3, 2026 | 811 out of 1000 | About 6 hours of preparation over 3 days; used about 90 of the 120 minutes. Rated CCAR-F the most difficult of the four exams. Also promotes their own practice exam. |
| [Ihor Sasovets](https://medium.com/@ihor.sasovets/claude-certified-architect-foundations-my-learning-journey-and-exam-experience-826d4c08e664) (Medium) | Published June 6, 2026 | 839/1000 | About 2 weeks of study, starting with minimal Claude Code experience. Scored 882/1000 on the official practice exam the day before. Used 105 of the 120 minutes. |
| [Sofia Lemons](https://www.udacity.com/blog/the-claude-certified-architect-exam-explained-by-someone-who-passed-it/) (Udacity blog) | Published July 2, 2026 (describes the older setup) | Passed (no score given) | About six weeks of preparation; about 90 minutes to finish; results took a week or two. Udacity sells a related Nanodegree. |

### Practice scores ran ahead of real scores

Six of the reports above give both a practice or mock score and a real score. In all six, the best practice figure was higher than the real one (our comparison of the published figures):

| Candidate | Best practice or mock score reported | Real score |
|---|---|---|
| service_account | 1000/1000 (official practice exam) | 590/1000 |
| Upstairs_Muffin_7035 | 1000 (official mock) | 846 |
| John Weidner | About 930 (official practice exam) | 738 |
| Ihor Sasovets | 882/1000 (official practice exam) | 839/1000 |
| ahmdhsn-dev | "consistently 950+" (third-party mock) | 910/1000 |
| WenHao Yu | 915 (third-party mock) | 882 |

Two candidates offer an explanation. [Weidner](https://verygood.ventures/blog/passing-the-claude-certified-architect-exam/) wrote that "The practice exam reuses its question pool, so studying my screenshots was partly teaching me the practice answers rather than the underlying ideas." [AK](https://www.claudeaimalaysia.com/malaysia-claude-ccaf-studyguide.html) warned: "You can score 980 on the mock and still fail if you memorised answers without the reasoning." The official practice exam has since been retired, and the [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) 12 sample questions are "drawn from the practice test", so treat them, and any third-party mock, as a check on your reasoning rather than a forecast of your score. See [Official sample questions](#official-sample-questions).

### What the score reports showed

[Weidner](https://verygood.ventures/blog/passing-the-claude-certified-architect-exam/) published four of their five domain results from a pre-Pearson score report: Agentic Architecture and Orchestration 78%, Tool Design and MCP Integration 70%, Context Management and Reliability 73%, and Claude Code Configuration and Workflows 69%, the one the report flagged "focus here". The post does not give the Prompt Engineering figure. [Purcell's](https://www.linkedin.com/pulse/claude-certification-exams-honest-review-matthew-purcell-byo2e) report also flagged Claude Code as "focus here", although Claude Code is where they spend most of their time. Weidner, a daily user, put it this way: "Daily use had given me fluency with the parts I reach for and blind spots around the parts I do not." If you use Claude Code every day, do not assume [Domain 3](#domain-3-claude-code-configuration-workflows) is covered.

Two candidates published raw counts next to their scaled scores: 44 of 60 correct became 738 (Weidner), and 48 of 60 became 822 (AK). The guide publishes no raw-to-scaled conversion, and it says scaled scoring helps equate scores across exam forms of slightly different difficulty, so two data points do not give you a formula. On the current score report, the per-domain percentages are feedback only: pass or fail rests on the total scaled score.

### How the exam felt, in their words

- **Structure.** [Weidner](https://verygood.ventures/blog/passing-the-claude-certified-architect-exam/) (pre-Pearson) wrote that "The questions come in four blocks of fifteen, each built on its own production scenario." [TFGator1983](https://www.reddit.com/r/ClaudeAI/comments/1wc73tf/ccarf_about_to_take_exam_how_long_are_the/) (comment dated September 10, 2026; their exam date is not given) described each of the four blocks keeping the same scenario, one of the six in the guide, on the left of the screen for all 15 questions. [bills70](https://www.reddit.com/r/ClaudeAI/comments/1vaxsws/claude_certified_architect_foundations_ccarf_how/) found that "The questions themselves contain more information which is enough to answer the question without the context." The guide says the exam has 60 items and presents 4 scenarios picked at random from its 6, each framing a set of questions; it does not say how many items each scenario carries.
- **Item format.** [OkRelationship3427](https://www.reddit.com/r/ClaudeAI/comments/1v5zrru/passed_the_ccarf_with_9041000/) saw only single-answer, four-option items. The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says the exam uses multiple-choice and multiple-response items and that each item states how many responses to select, and all 12 official samples are single-answer. Prepare for both formats. [bluepanda](https://dev.to/bluepanda/i-cleared-all-4-anthropic-claude-certifications-what-you-actually-need-to-know-and-what-to-skip-224m) noted that you can strike out options and flag questions, and reported "no negative marking for incorrect answers"; the July 2026 guide does not mention guessing penalties (the [March 2026 guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F8lsy243ftffjjy1cx9lm3o2bw%2Fpublic%2F1773274827%2FClaude+Certified+Architect+%E2%80%93+Foundations+Certification+Exam+Guide.pdf) said "there is no penalty for guessing").
- **Reading load.** [Talele](https://newsletter.bigtechcareers.com/p/step-by-step-guide-to-achieve-claude-certification): "Constraints are distributed throughout multiple phrases, with the most significant details frequently located in the middle." [WenHao Yu](https://yu-wenhao.com/en/blog/ccar-f-exam-experience/) needed 4 to 5 minutes just to read and understand one question when they started practicing, and by exam day finished with 15 minutes to spare.
- **Close options.** [Talele](https://newsletter.bigtechcareers.com/p/step-by-step-guide-to-achieve-claude-certification) described the core skill as "selecting the right architectural decision when three of the four options seem valid." [TFGator1983](https://www.reddit.com/r/ClaudeAI/comments/1wc73tf/ccarf_about_to_take_exam_how_long_are_the/): "What tripped me up most when I took it was that they tend to ask questions about concepts a lot of different ways."
- **Exact names.** Two candidates said the exam expected exact specifics. [Purcell](https://www.linkedin.com/pulse/claude-certification-exams-honest-review-matthew-purcell-byo2e) listed "CLI/API arguments, parameter values, JSON schemas, and the locations of various Claude files (CLAUDE.md, Skills, Commands)." [bluepanda](https://dev.to/bluepanda/i-cleared-all-4-anthropic-claude-certifications-what-you-actually-need-to-know-and-what-to-skip-224m) wrote that it "asks about exact CLI syntax, file naming conventions, and specific configuration flags." [Purcell's advice](https://www.linkedin.com/posts/purcellmatthew_claude-certified-architect-professional-practice-activity-7482176978008342528-204Q): "Know your stop reasons, your tool_choice options, and your context management."
- **Difficulty.** Opinions differ. [Purcell](https://www.linkedin.com/pulse/claude-certification-exams-honest-review-matthew-purcell-byo2e) and [bluepanda](https://dev.to/bluepanda/i-cleared-all-4-anthropic-claude-certifications-what-you-actually-need-to-know-and-what-to-skip-224m) each ranked CCAR-F the hardest of the four exams. [Leading_Will1794](https://www.reddit.com/r/ClaudeAI/comments/1vaxsws/claude_certified_architect_foundations_ccarf_how/), who took only CCAR-F, wrote in a Pearson-era CCAR-F thread: "I will say it didn't feel like the hardest at all." [Weidner](https://verygood.ventures/blog/passing-the-claude-certified-architect-exam/): "The exam had been harder than I expected".

### Elimination rules candidates published, checked against the official samples

Several candidates shared rules of thumb for ruling out options. Test any such rule against Anthropic's own rationales before you rely on it.

| Rule (who published it) | How it fares on the [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) 12 sample questions (our check against Anthropic's rationales) |
|---|---|
| "Wrong answers have four favorite disguises: add more examples, write a more precise description, switch to a stronger model, tell it to check itself." ([WenHao Yu](https://yu-wenhao.com/en/blog/ccar-f-exam-experience/)) | Breaks twice. In Question 2 the correct answer is to expand each tool's description, and in Question 3 it is explicit escalation criteria with few-shot examples. It holds for the stronger model: Question 12 rejects a higher-tier model because "larger context windows don't solve attention quality issues". The closest official match to "tell it to check itself" is Question 3's option B, a self-reported confidence score, which the rationale rejects because "LLM self-reported confidence is poorly calibrated" (our mapping). |
| The correct answer was almost always "the thing that costs the least while also giving the correct outcome". ([Leading_Will1794](https://www.reddit.com/r/ClaudeAI/comments/1vaxsws/claude_certified_architect_foundations_ccarf_how/)) | Matches the rationales that praise a "low-effort" fix (Question 2) and call the answer "the proportionate first response before adding infrastructure" (Question 3), and that reject options as "over-engineered" (Questions 2 and 3). It needs care when the cheapest option is a prompt instruction for a rule that must always hold, because that option does not reliably give the correct outcome: in Question 1, "programmatic enforcement provides deterministic guarantees that prompt-based approaches cannot". |
| Choosing the bigger model is always wrong. ([UnfortunateHurricane](https://www.reddit.com/r/ClaudeAI/comments/1v5zrru/passed_the_ccarf_with_9041000/), pre-Pearson) | Consistent with Question 12, as above. |
| Wrong answers patch symptoms instead of removing the failure mechanism. ([InRussYouTrust](https://www.youtube.com/watch?v=SYYtM16wXcI)) | Consistent with the rationales for Question 2 ("directly addresses this root cause") and Question 12 ("directly addresses the root cause"). |
| When a question says a team member proposes something, the proposal is almost always wrong. ([WenHao Yu](https://yu-wenhao.com/en/blog/ccar-f-exam-experience/)) | Consistent with Question 11, where the manager's proposal to move both workflows to the Message Batches API is wrong and the answer keeps the blocking pre-merge check on real-time calls. |

A rule of thumb only narrows the field. Confirm the survivor against the decision rule for its task statement in the domain sections above, and work through the [Official sample questions](#official-sample-questions) with the rationales open. The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says its exam items "are written against these objectives", so the task statements, not a heuristic, decide the answer.

### Exam day, as recent candidates describe it

- [WenHao Yu](https://yu-wenhao.com/en/blog/ccar-f-exam-experience/) tested at a Pearson test center, put their phone, watch "and everything else in my pockets" in a locker, saw the score on screen at the end, and would choose in person over online.
- [EnvironmentalPlay440](https://www.reddit.com/r/ClaudeAI/comments/1v5zrru/passed_the_ccarf_with_9041000/), commenting in a CCAR-F thread in July 2026 (their exam date is not given), got a message from the proctor for reading questions aloud in a low voice. [Pearson's OnVUE rules](https://www.pearsonvue.com/us/en/anthropic/onvue.html) list "Speak or read aloud, unless instructed" among the things you must not do.
- [bluepanda](https://dev.to/bluepanda/i-cleared-all-4-anthropic-claude-certifications-what-you-actually-need-to-know-and-what-to-skip-224m) reported that OnVUE check-in took about 10 to 15 minutes.

The practical steps are in the [Exam-day checklist](#exam-day-checklist).

### What to take from these reports

- **Understand; do not memorize.** In the practice-score table above, the three largest gaps between practice and real scores (410, about 192 and 154 points, our arithmetic) belong to the three candidates who took the official practice exam three times each, and Weidner reports that it reused its question pool. The guide's own preparation advice starts with building an agent, not with question banks.
- **Budget reading time.** Two candidates who sat CCAR-F after the move to Pearson singled out reading volume (bluepanda) and reading speed (WenHao Yu). Practice on the long stems of the official samples until you can find the deciding constraint quickly.
- **Do not skip Domain 3 because you use Claude Code daily.** The two score reports quoted above that flag a weak domain both flag Claude Code, and both candidates say they use it heavily.
- **Expect multiple-response items even if the last report you read saw none.** The guide says they are part of the format.

## Study plan

Start with the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf). The [certification FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) calls it "the authoritative source for exam scope", and the guide asks you to read it in full before scheduling your exam. The plan below runs six weeks in blueprint order and pairs each week with hands-on work from the guide, free courses where one fits, the knowledge-base pages on this site, and the official sample questions for that domain. The six-week length is our suggestion. Neither the guide nor the certification FAQ sets a study duration. The guide describes its ideal candidate as a solution architect who "typically has 6+ months of practical experience building with Claude APIs, Agent SDK, Claude Code, and MCP", so the plan assumes you already build with these tools.

### The official course list for CCAR-F

CCAR-F has no dedicated prep path. The Anthropic Partner Academy has a free multi-course prep path for each of the other three exams (8 courses for CCAO-F, 5 for CCDV-F, 5 for CCAR-P); for CCAR-F its [prep courses page](https://anthropic-partners.skilljar.com/page/claude-certified-architect-foundations-prep-courses) lists seven free courses instead. The [certification FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says prep coverage varies by certification and that Anthropic is adding prep content over time, so check that page before you start. The Partner Academy titles differ slightly from the titles on [Claude Academy](https://academy.claude.com/courses), which is free: you can browse the catalog without signing in, and a free Claude account saves your progress.

| Course (Partner Academy title) | On academy.claude.com | Where it helps on CCAR-F (our mapping) |
|---|---|---|
| Building with the Claude API | [Building with the Claude API](https://academy.claude.com/courses/building-with-the-claude-api): 67 lessons, 8 quizzes, 9 hr | Domains 1, 2 and 4: API calls, tool use, MCP, agents and workflows |
| Claude Code in Action | [Claude Code in action](https://academy.claude.com/courses/claude-code-in-action): 9 lessons, 1 quiz, 1 hr | Domain 3: plan mode, compaction, CLAUDE.md, skills, permission modes, hooks, headless runs |
| Introduction to Model Context Protocol | [Introduction to Model Context Protocol](https://academy.claude.com/courses/introduction-to-model-context-protocol): 10 lessons, 1 quiz, 1 hr | Domain 2: tools, resources and prompts, and building a server |
| Claude 101 | [Claude 101](https://academy.claude.com/courses/claude-101): 13 lessons, 1 quiz, 2.5 hr | Orientation: everyday use of Claude, prompting, projects, skills and connected tools |
| AI Fluency: Framework & Foundations | [AI Fluency: Framework and foundations](https://academy.claude.com/courses/ai-fluency-framework-foundations): 14 lessons, 1 quiz, 4 hr | Orientation. The CCAR-F guide does not name its 4D framework (Delegation, Description, Discernment, Diligence). |
| Claude with Amazon Bedrock | [Claude with Amazon Bedrock](https://academy.claude.com/courses/claude-with-amazon-bedrock): 65 lessons, 8 quizzes, 8 hr | Only if you deploy on AWS. The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) lists "Specific cloud provider configurations (AWS, GCP, Azure)" as out of scope. |
| Claude on Google Cloud | [Claude with Google Cloud's Vertex AI](https://academy.claude.com/courses/claude-with-google-cloud-s-vertex-ai): 66 lessons, 9 quizzes, 8.5 hr | Only if you deploy on Google Cloud. Same out-of-scope note. |

Building with the Claude API is the heaviest item, and not all of it is tested. The course page lists seven sections: Accessing Claude with the API (9 lessons), Prompt engineering techniques (6), Tool use with Claude (13), RAG and Agentic Search (7), Model Context Protocol (11), Anthropic apps - Claude Code and computer use (4), and Agents and workflows (8). Its lesson list also includes a "Features of Claude" block (extended thinking, images, PDFs, citations, prompt caching, code execution and the Files API). Give full attention to the API, prompting, tool use, MCP and agents sections, and to the Claude Code lessons in the apps section, since Claude Code is tested in Domain 3. Skim the rest: the RAG section covers embeddings and hybrid search, the apps section also covers computer use, and the features block covers images and prompt caching, while the guide lists embedding and vector database details, computer use, vision and prompt caching implementation details (beyond knowing it exists) as out of scope.

### Optional courses that fit the objectives

These four free Claude Academy courses are not on the official CCAR-F list. Each maps to specific task statements (our mapping from each course's published objectives).

| Course | Length | Why it fits |
|---|---|---|
| [Claude Platform 101](https://academy.claude.com/courses/claude-platform-101) | 13 lessons, 1 quiz, 1.5 hr | You build the agent loop by hand before replacing it with the SDK's Tool Runner (task statement 1.1). |
| [Introduction to subagents](https://academy.claude.com/courses/introduction-to-subagents) | 4 lessons, 45 min, no quiz | How a subagent gets a separate context window and returns a summary (1.2, 1.3, 3.4). |
| [Claude Code 101](https://academy.claude.com/courses/claude-code-101) | 12 lessons, 1 quiz, 1.5 hr | CLAUDE.md, subagents, skills, MCP and hooks, plus `/compact`, `/clear` and `/context` (3.1, 5.4). |
| [Introduction to agent skills](https://academy.claude.com/courses/introduction-to-agent-skills) | 6 lessons, 1 hr, no quiz | SKILL.md frontmatter and choosing between skills, CLAUDE.md, subagents, hooks and MCP servers (3.2). |

The [agent skills course](https://academy.claude.com/courses/introduction-to-agent-skills) teaches you to "restrict tool access with allowed-tools", which matches the guide's wording in 3.2-S3. The current [Claude Code skills documentation](https://code.claude.com/docs/en/skills.md) (as of September 2026) says `allowed-tools` pre-approves tools and "does not restrict which tools are available"; removing tools while a skill is active is the job of `disallowed-tools`. On the exam, answer in the guide's terms (`allowed-tools` restricts tool access); in real projects, follow the docs. [Domain 3](#domain-3-claude-code-configuration-workflows) covers the difference.

### Course hours, with the arithmetic

The [Claude Academy FAQ](https://academy.claude.com/help/faq) calls listed durations "estimates to help you plan", so treat these totals the same way. The sums below are our arithmetic from the durations listed on Claude Academy.

- Technical courses on the official list: 9 + 1 + 1 = 11 hours.
- Orientation courses on the official list: 2.5 + 4 = 6.5 hours.
- Cloud courses on the official list: 8 + 8.5 = 16.5 hours.
- The whole official list: 11 + 6.5 + 16.5 = 34 hours.
- The four optional courses: 1.5 + 0.75 + 1.5 + 1 = 4.75 hours.
- This plan (official list without the cloud courses, plus the optional four): 11 + 6.5 + 4.75 = 22.25 hours of course time.

These figures leave out the hands-on work; neither the guide nor the certification FAQ estimates it. Third-party study-hour estimates disagree with each other and cite no source: one site says 15 to 20 hours for developers with Claude experience and 30 to 40 hours for newcomers, another says 4 to 12 weeks (roughly 40 to 100 hours). The candidates in [What candidates report](#what-candidates-report) describe anything from about 6 hours over three days to about six weeks.

### The guide's own advice, mapped to the weeks

Section 7 of the [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) ("How to Prepare") lists seven activities. Each one asks you to build or practice something.

| How to Prepare item ([guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) wording) | Week in this plan |
|---|---|
| "Build an agent with the Claude Agent SDK: implement a complete agentic loop with tool calling, error handling, and session management. Practice spawning subagents and passing context between them." | Weeks 1 and 2 |
| "Design and test MCP tools: write tool descriptions that clearly differentiate similar tools. Implement structured error responses with error categories and retryable flags. Test tool selection reliability with ambiguous requests." | Week 3 |
| "Configure Claude Code for a real project: set up CLAUDE.md with a configuration hierarchy, create path-specific rules in .claude/rules/, build custom skills with frontmatter options (context: fork, allowed-tools), and integrate at least one MCP server." | Week 4 |
| "Build a structured data extraction pipeline: use tool_use with JSON schemas, implement validation-retry loops, design schemas with optional/nullable fields, and practice batch processing with the Message Batches API." | Week 5 |
| "Practice prompt engineering techniques: write few-shot examples for ambiguous scenarios. Define explicit review criteria to reduce false positives. Design multi-pass review architectures for large code reviews." | Week 5 |
| "Study context management patterns: practice extracting structured facts from verbose tool outputs, implementing scratchpad files for long sessions, and designing subagent delegation to manage context limits." | Week 6 |
| "Review escalation and human-in-the-loop patterns: understand when to escalate (policy gaps, customer requests, inability to progress) versus resolve autonomously. Practice designing human review workflows with confidence-based routing." | Weeks 2 and 6 |

The four Preparation Exercises in Section 8 of the guide turn these into step-by-step builds. The plan places Exercise 1 in Week 2, Exercise 2 in Week 4, Exercise 3 in Weeks 5 and 6, and Exercise 4 in Week 6. The steps are reproduced in [Official preparation exercises](#official-preparation-exercises).

### The plan at a glance

| Week | Focus (blueprint weight) | Task statements | Build | Official samples to work |
|---|---|---|---|---|
| Before week 1 | Orientation | None | Read the whole guide; set up the SDKs and Claude Code | None yet |
| 1 | Domain 1, part 1 (27%) | 1.1, 1.2, 1.3 | First half of the Agent SDK item in How to Prepare | Question 7 |
| 2 | Domain 1, part 2 | 1.4, 1.5, 1.6, 1.7 | Exercise 1 | Question 1 |
| 3 | Domain 2 (18%) | 2.1 to 2.5 | The MCP tools item in How to Prepare | Questions 2 and 9 |
| 4 | Domain 3 (20%) | 3.1 to 3.6 | Exercise 2 | Questions 4, 5, 6 and 10 |
| 5 | Domain 4 (20%) | 4.1 to 4.6 | Exercise 3, steps 1 to 4; the prompt engineering item in How to Prepare | Questions 11 and 12 |
| 6 | Domain 5 (15%), then review | 5.1 to 5.6 | Exercise 4; Exercise 3, step 5; the context management and escalation items in How to Prepare | Questions 3 and 8, then all 12 again |

Domain 1 gets two weeks because it carries the largest weight and the most task statements (seven, against five or six in each other domain). The guide does not label its sample questions by domain; the matching of questions to weeks is ours, based on the objectives each rationale tests.

### Before week 1: orientation

- Read the guide end to end, including [The six scenarios](#the-six-scenarios) and [What is out of scope](#what-is-out-of-scope). The guide asks you to do this before scheduling.
- Get your tools working. Building with the Claude API expects Python, basic JSON handling and an Anthropic API key. Claude Code 101 expects a Claude account (Pro, Max, or Enterprise) or an API key.
- Optional: Claude 101 (2.5 hr) and AI Fluency: Framework and foundations (4 hr).

### Week 1: agentic loops and coordinator-subagent design

- **Objectives:** 1.1 (agentic loops), 1.2 (coordinator-subagent orchestration), 1.3 (subagent invocation, context passing and spawning).
- **Course work:** Building with the Claude API, sections "Accessing Claude with the API" and "Tool use with Claude". Optional: Claude Platform 101 and Introduction to subagents.
- **Build:** a loop that continues while `stop_reason` is `"tool_use"` and ends on `"end_turn"`, with tool results appended to the conversation each turn. Then a coordinator that spawns two subagents and passes each one its context explicitly in the prompt.
- **Watch for, as of September 2026:** the guide names the spawning tool "Task" and says a coordinator's `allowedTools` must include it. Claude Code renamed the Task tool to Agent in version 2.1.63, and `Task(...)` references still work as aliases. The guide also frames loop control as "tool_use" versus "end_turn"; the API docs say the loop ends on any other stop reason (`end_turn`, `max_tokens`, `stop_sequence` or `refusal`) and document `pause_turn` for server-tool loops. Answer in the guide's terms; handle the other stop reasons in real code.
- **Read:** [Domain 1](#domain-1-agentic-architecture-orchestration) on this page; [The agentic loop](knowledge/agents-and-agent-sdk.md#the-agentic-loop); [Stop reasons and the agent loop](knowledge/claude-api.md#stop-reasons-and-the-agent-loop); [Multi-agent orchestration](knowledge/agents-and-agent-sdk.md#multi-agent-orchestration); [Subagents in the SDK](knowledge/agents-and-agent-sdk.md#subagents-in-the-sdk).
- **Self-check:** sample Question 7 (a coordinator whose decomposition is too narrow).

### Week 2: enforcement, hooks, decomposition and sessions

- **Objectives:** 1.4 (enforcement and handoff), 1.5 (Agent SDK hooks), 1.6 (task decomposition), 1.7 (session state, resumption and forking).
- **Course work:** Building with the Claude API, section "Agents and workflows".
- **Build:** Exercise 1 in full. Define 3 to 4 MCP tools, including two similar ones; drive them from a loop that checks `stop_reason`; return structured errors with `errorCategory` (transient, validation or permission), an `isRetryable` boolean and a readable description; add a hook that blocks operations above a threshold and redirects to escalation; test with multi-concern messages. Then resume a named session with `--resume <session-name>` and branch one with `fork_session`. `fork_session` is the Python SDK option name; TypeScript uses `forkSession` and the CLI uses `--fork-session`.
- **Read:** [Domain 1](#domain-1-agentic-architecture-orchestration); [Hooks in the SDK](knowledge/agents-and-agent-sdk.md#hooks-in-the-sdk); [Permissions and enforcement](knowledge/agents-and-agent-sdk.md#permissions-and-enforcement); [Sessions, resumption and forking](knowledge/agents-and-agent-sdk.md#sessions-resumption-and-forking); [Workflow patterns](knowledge/agents-and-agent-sdk.md#workflow-patterns).
- **Self-check:** sample Question 1 (a tool sequence that must always hold).

### Week 3: tool design and MCP integration

- **Objectives:** 2.1 (tool interfaces and descriptions), 2.2 (structured MCP errors), 2.3 (tool distribution and `tool_choice`), 2.4 (MCP servers in Claude Code and agents), 2.5 (built-in tools).
- **Course work:** Introduction to Model Context Protocol (1 hr); Building with the Claude API, section "Model Context Protocol".
- **Build:** the [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) How to Prepare item "Design and test MCP tools": descriptions that separate similar tools, structured errors "with error categories and retryable flags", and tests with ambiguous requests. Practice Grep for content search and Glob for file-name patterns.
- **Watch for, as of September 2026:** the guide lists three `tool_choice` options ("auto", "any" and a forced tool); the docs list four, adding `none`. The guide teaches "any" and a forced tool to guarantee a tool call; the docs say `any` and `tool` return a 400 error on Claude Opus 5.5, Claude Fable 5.1 and Claude Mythos 5.1. The guide contrasts project scope (`.mcp.json`) with user scope (`~/.claude.json`); the docs add a third, local scope, which is the default and is also stored in `~/.claude.json`. The guide teaches Grep and Glob as built-in tools; the docs say Claude Code leaves them out of the default tool set on macOS, Linux and WSL. Answer with the guide's logic. [Domain 2](#domain-2-tool-design-mcp-integration) explains each change.
- **Read:** [Writing tool descriptions that steer selection](knowledge/tool-use-and-mcp.md#writing-tool-descriptions-that-steer-selection); [Controlling tool choice](knowledge/tool-use-and-mcp.md#controlling-tool-choice); [MCP errors and structured results](knowledge/tool-use-and-mcp.md#mcp-errors-and-structured-results); [Designing a tool set](knowledge/tool-use-and-mcp.md#designing-a-tool-set); [MCP in Claude Code](knowledge/tool-use-and-mcp.md#mcp-in-claude-code).
- **Self-check:** sample Questions 2 (minimal tool descriptions) and 9 (a scoped cross-role tool).

### Week 4: Claude Code configuration and workflows

- **Objectives:** 3.1 (CLAUDE.md hierarchy), 3.2 (slash commands and skills), 3.3 (path-specific rules), 3.4 (plan mode or direct execution), 3.5 (iterative refinement), 3.6 (CI/CD).
- **Course work:** Claude Code in action (1 hr). Optional: Claude Code 101 (1.5 hr) and Introduction to agent skills (1 hr).
- **Build:** Exercise 2 in full: a project CLAUDE.md, `.claude/rules/` files with glob `paths`, a project skill in `.claude/skills/` with `context: fork` and `allowed-tools`, a shared server in `.mcp.json` (with environment variable expansion for credentials) next to a personal one in `~/.claude.json`, and plan mode against direct execution on three tasks. Then run a non-interactive `claude -p` call with `--output-format json` and `--json-schema`.
- **Watch for, as of September 2026:** the guide and sample Question 4 put team commands in `.claude/commands/`; the docs say custom commands have been merged into skills, that files in `.claude/commands/` still work, and that skills are preferred for new work. The guide says path-scoped rules load only when you edit a matching file; the docs say they trigger when Claude reads a matching file. The guide uses `/memory` to verify which memory files are loaded; the docs point to `/context` for what actually loaded. The guide says `allowed-tools` restricts tools; the docs say it pre-approves them (see above). Answer in the guide's terms. [Domain 3](#domain-3-claude-code-configuration-workflows) covers each.
- **Read:** [CLAUDE.md and the memory hierarchy](knowledge/claude-code-configuration.md#claudemd-and-the-memory-hierarchy); [Path-scoped rules](knowledge/claude-code-configuration.md#path-scoped-rules); [Slash commands](knowledge/claude-code-configuration.md#slash-commands); [Agent Skills](knowledge/claude-code-configuration.md#agent-skills); [Plan mode or direct execution](knowledge/claude-code-workflows.md#plan-mode-or-direct-execution); [Iterative refinement](knowledge/claude-code-workflows.md#iterative-refinement); [Claude Code in CI/CD](knowledge/claude-code-workflows.md#claude-code-in-cicd).
- **Self-check:** sample Questions 4 (where a team command lives), 5 (plan mode), 6 (conventions for files spread across directories) and 10 (a CI job that hangs).

### Week 5: prompt engineering and structured output

- **Objectives:** 4.1 (explicit criteria), 4.2 (few-shot prompting), 4.3 (structured output with tool use and JSON schemas), 4.4 (validation, retry and feedback loops), 4.5 (batch processing), 4.6 (multi-instance and multi-pass review).
- **Course work:** Building with the Claude API, section "Prompt engineering techniques".
- **Build:** Exercise 3, steps 1 to 4: an extraction tool with required, optional and nullable fields and an "other" + detail enum; a validation-retry loop that records which errors a retry can fix; few-shot examples for varied document formats; a 100-document batch with failures resubmitted by `custom_id`. Add the prompt engineering item from How to Prepare: few-shot examples for ambiguous cases, explicit review criteria, and a multi-pass review.
- **Watch for, as of September 2026:** the guide presents tool use with JSON schemas as the most reliable approach for guaranteed schema-compliant output; the API docs now also offer structured outputs as a separate feature: JSON outputs (`output_config.format`) and strict tool use (`strict: true`). Answer in the guide's terms. The Message Batches API facts the guide tests (50% cost savings, up to 24 hours, `custom_id` correlation) still match the docs, which add that results can come back in any order.
- **Read:** [Explicit criteria and precision](knowledge/prompt-engineering.md#explicit-criteria-and-precision); [Few-shot examples](knowledge/prompt-engineering.md#few-shot-examples); [Structured output with tools and JSON schemas](knowledge/prompt-engineering.md#structured-output-with-tools-and-json-schemas); [Validation, retry and feedback loops](knowledge/prompt-engineering.md#validation-retry-and-feedback-loops); [Batch processing design](knowledge/prompt-engineering.md#batch-processing-design); [Multi-instance and multi-pass review](knowledge/prompt-engineering.md#multi-instance-and-multi-pass-review); [Message Batches](knowledge/claude-api.md#message-batches).
- **Self-check:** sample Questions 11 (batch or real-time) and 12 (a 14-file review).

### Week 6: context management and reliability, then review

- **Objectives:** 5.1 (preserving critical information), 5.2 (escalation and ambiguity), 5.3 (error propagation), 5.4 (large codebase exploration), 5.5 (human review and confidence calibration), 5.6 (provenance and uncertainty).
- **Course work:** the closest course material is directed compaction in Claude Code in action and, optionally, `/compact`, `/clear` and `/context` in Claude Code 101 (our mapping, for 5.1 and 5.4). Most of this week is the builds below.
- **Build:** Exercise 4 in full: a coordinator with at least two subagents, parallel spawning in one response, claim-source structured output, a simulated subagent timeout with structured error context, and conflicting sources preserved with attribution. Then Exercise 3, step 5: field-level confidence, low-confidence routing to human review, and accuracy by document type and field. Add the two remaining How to Prepare items: the context management item (extract structured facts from verbose tool outputs, keep a scratchpad file in a long session, design subagent delegation to manage context limits) and the escalation item (when to escalate, for example on policy gaps, customer requests or inability to progress, versus resolving autonomously; plus a human review workflow with confidence-based routing).
- **Read:** [Domain 5](#domain-5-context-management-reliability); [Preserving critical information in long conversations](knowledge/context-engineering.md#preserving-critical-information-in-long-conversations); [Exploring large codebases](knowledge/context-engineering.md#exploring-large-codebases); [Provenance and uncertainty in synthesis](knowledge/context-engineering.md#provenance-and-uncertainty-in-synthesis); [Escalation and ambiguity](knowledge/evaluation-and-reliability.md#escalation-and-ambiguity); [Error propagation in multi-agent systems](knowledge/evaluation-and-reliability.md#error-propagation-in-multi-agent-systems); [Human review and confidence calibration](knowledge/evaluation-and-reliability.md#human-review-and-confidence-calibration).
- **Self-check:** sample Questions 3 (escalation calibration) and 8 (a subagent timeout).
- **Review days:** work all 12 sample questions again and, for every wrong option, say which part of Anthropic's rationale rules it out. Re-read the out-of-scope list, the [Frequently asked questions](#frequently-asked-questions) on guide wording versus current docs, and the [Exam-day checklist](#exam-day-checklist). The [Quick reference](quick-reference.md#decision-rules) collects the decision rules in one place.

### If you have less time

Merge the weeks in pairs (1 and 2, 3 and 4, 5 and 6) for a three-week plan. Keep every build and every sample question; cut the orientation and optional courses first, since the guide's own advice is all hands-on.

## Exam-day checklist

Tick these off in order. The rules come from the CCAR-F guide, Anthropic's certification pages, or Pearson VUE's pages and candidate rules, as of September 2026; the few items that are our study advice, our arithmetic or a candidate's experience say so. The reasons and the money at stake are explained in [Policies that cost candidates money](index.md#policies-that-cost-candidates-money).

### Two weeks or more before

- [ ] Your registration name matches your government-issued photo ID exactly. If it does not, email certifications-support@anthropic.com before you schedule (the guide's instruction), from your registered address, with the subject line "Name Correction Request" and your name in Latin characters. Corrections typically take 24 to 48 business hours, and the FAQ asks for requests at least 24 hours before the exam.
- [ ] Any accommodation is approved by Pearson VUE before you book. Pearson asks you to allow 10 business days for review, and accommodations cannot be added to an exam that is already scheduled.
- [ ] You have chosen online proctoring (OnVUE) or a Pearson test center. VPNs, corporate networks and public or shared networks are not allowed for OnVUE. If a company laptop or network will not run it, use a personal computer on a personal network, or book a test center.
- [ ] You have run and passed Pearson's System Test on the same device and network you will use on exam day. Passing it does not guarantee a problem-free exam.
- [ ] Your machine meets the OnVUE minimums: Windows 10 or macOS 14 or higher; a working webcam, microphone and speaker (no headphones or headsets); one display only; at least 6 Mbps download and 2 Mbps upload.
- [ ] Your IT team has been asked, early, to allow Pearson's domains and to shut down background applications that block OnVUE. On Windows that list includes the Claude desktop application.

!!! warning "The cancellation deadline is 48 hours, not 24"

    The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says: "You may cancel or reschedule up to 24 hours before your appointment." Anthropic's [policies page](https://anthropic-partners.skilljar.com/page/policies-certifications) and [FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications), and [Pearson's Anthropic page](https://www.pearsonvue.com/us/en/anthropic.html) (for test center appointments), all say 48 hours. Plan on 48. Canceling in Pearson only frees the time slot: to get your money back, email certifications-support@anthropic.com. After any reschedule, make sure a confirmation email for the new date arrives.

### The day before

- [ ] Restart the computer you will test on, and make sure nobody else on your network will be streaming or downloading large files during your slot.
- [ ] Your ID is valid, unexpired, government-issued, carries a recognizable photo, and matches your booking name exactly. Expired, digital, damaged, copied or privately issued IDs are refused.
- [ ] For OnVUE, your desk is empty except for the testing computer, pre-approved items and comfort aids, and a drink in an unmarked container. Books, notes, paper, pens and writing tools are gone, and any whiteboard or note board in the room is wiped.
- [ ] For OnVUE, you have a private, quiet room where you will be alone for the whole session. Offices, libraries and coffee shops are not allowed.
- [ ] Our study advice: close the gaps you found in your last pass through the [Official sample questions](#official-sample-questions) rather than starting a new topic.

### Before the exam starts

- [ ] Online: begin check-in 30 minutes before your appointment. Expect technology checks, a 360° room scan, and photos of you and your ID. If any requirement is not met, you cannot test and the fee is forfeited.
- [ ] At a test center: arrive as early as your confirmation email says. Personal items are not allowed in the testing room; if you refuse to store them, you cannot test and you lose the fee. The administrator provides any materials the exam sponsor authorizes, such as a laminated note board.
- [ ] Either way, do not be late or miss the slot: a no-show, or arriving after the permitted late-arrival window, forfeits the fee and you must re-register.
- [ ] Phone, tablet, watch, headphones, earbuds, stylus, study materials and anything that records are out of reach. Online, every application except OnVUE is closed.
- [ ] Budget about 135 minutes of seat time: the [FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says that total includes "check-in, instructions, and a brief post-exam survey" around the 120-minute exam.
- [ ] Accept the confidentiality and non-disclosure agreement. If you decline it, the session ends and no refund is issued.

### During the exam

- [ ] Read how many responses each item asks for. The guide lists multiple-choice and multiple-response items, and every item states how many to select.
- [ ] Watch the clock at each scenario change. 120 minutes over 4 scenarios is 30 minutes per scenario if the items split evenly (our arithmetic; the guide does not give a per-scenario count).
- [ ] Use the tools you are given. Online, Anthropic's approved aid is Pearson's digital whiteboard; physical whiteboards and writing materials of any kind are not allowed, and the whiteboard is wiped if your connection drops. The [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) mention of "scratch paper provided by the proctor" is only an example of what Pearson VUE may permit; at a test center the administrator provides any authorized materials, such as a laminated note board.
- [ ] Stay in webcam view, stay alone, do not communicate with anyone, and do not read questions aloud. One candidate in a CCAR-F discussion received a proctor message for reading questions in a very low voice.
- [ ] Do not capture, copy, photograph or reproduce any exam content. Misconduct can lead to an invalidated result, a revoked credential and a ban from future exams.
- [ ] Do not use notes, browser translation tools or any AI product or service. The exam is closed book.
- [ ] If a question looks factually wrong, unclear, has more than one defensible answer or does not match the guide, report it to Pearson. Reporting never counts against you. At a test center, the administrator cannot answer content questions, so note the question number for review.
- [ ] If OnVUE freezes or disconnects, close it and relaunch it from your downloads folder. The in-exam chat reaches a proctor, who cannot pause or extend the exam.

### After the exam

- [ ] Note your score. It appears on screen when you finish; test center candidates also get a printed report, and a copy arrives by email.
- [ ] If you passed, accept the Credly badge email, and add a personal email address to your Credly profile so the badge stays with you if you change jobs.
- [ ] Do not share or discuss the questions, including in study groups and online forums.
- [ ] If you did not pass, use the per-domain breakdown to plan your review. The next attempt opens after 14 days (then 30, then 90), each attempt costs the exam fee with any partner discount applied, and you can sit the exam up to four times in a rolling twelve months. The limits apply per exam, so a fail here does not stop you registering for a different exam.
- [ ] If you want to dispute a result, appeal to Pearson VUE support within 14 days of your exam date. A disputed question does not change a pass or fail on its own; if Anthropic confirms a faulty question affected your result, the remedy is a free retake, not a changed score.
- [ ] Put your expiry date in your calendar. The credential is valid for 12 months, and on-time renewal is a free, non-proctored assessment on the Anthropic Partner Academy. Details are in [Renewal](index.md#renewal).

## Resources

Official material comes first, because the guide defines what is tested. Community material follows with notes on where it matches the July 2026 guide and where it does not. Costs, dates and counts are as displayed on each site in September 2026.

### The official exam documents

| Resource | What it gives you |
|---|---|
| [CCAR-F exam guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) | Version 1.0, effective July 2026, 39 pages: the blueprint, the six scenarios, all 30 task statements, How to Prepare, four Preparation Exercises, 12 sample questions with rationales, policies, and the in-scope and out-of-scope lists. It calls itself "the authoritative reference for candidates preparing to sit the exam". |
| [CCAR-F certification page](https://anthropic-partners.skilljar.com/claude-certified-architect-foundations-certification) (Anthropic Partner Academy) | Where you buy the exam (the Purchase button showed &#36;125 in September 2026), with links to the guide, the Certification Terms and Conditions and the Certification Exam Policy to review before your session. |
| [Certification FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) | Current answers on fees and partner discounts, exam time and scoring, the retired practice exam, retakes and renewal. |
| [Certification policies](https://anthropic-partners.skilljar.com/page/policies-certifications) | The 48-hour cancellation rule (the guide says 24 hours, so act at least 48 hours ahead), appeals, question reports, misconduct and renewal details. |
| [Certification Exam Policy (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F34hhd92iyp94a0gtbr15cy5jk%2Fpublic%2F1782870704%2FAnthropic+Certification+Exam+Policy.pdf) | Confidentiality and prohibited conduct, including the use of unauthorized published exam questions and the use of AI products or services during the exam. Last updated June 25, 2026. |
| [Certification Terms and Conditions (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F34hhd92iyp94a0gtbr15cy5jk%2Fpublic%2F1782870634%2FCertification+Terms+and+Conditions.pdf) | Certification term, recertification and confidentiality. |
| [Exam Registration Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542947%2FClaude+Certification+Program+-+Exam+Registration+Guide.pdf) | Step-by-step Partner Academy checkout and Pearson VUE account setup. |
| [Pearson VUE page for Anthropic](https://www.pearsonvue.com/us/en/anthropic.html) and [OnVUE requirements for Anthropic](https://www.pearsonvue.com/us/en/anthropic/onvue.html) | Scheduling, retake limits, support hours, ID rules, system and room requirements. |
| [Computer and network setup](https://anthropic-partners.skilljar.com/page/computer-and-network-setup) (Partner Academy) | Domains to allow and applications to close before an OnVUE exam. |

### Official courses

Anthropic's [CCAR-F prep courses page](https://anthropic-partners.skilljar.com/page/claude-certified-architect-foundations-prep-courses) lists seven existing free courses rather than a dedicated prep course; the [Study plan](#study-plan) gives each one's length and which parts to skim. The same seven appear in the public [Claude Academy](https://academy.claude.com/courses) catalog, some under slightly different titles. Claude Academy is free: you can browse the catalog without signing in, a free Claude account saves your progress, and a personal account works (no paid plan or work account is needed).

### Current documentation for the guide's terms

The guide dates from July 2026, and some product details have changed since. Check each objective against these pages, then answer exam items in the guide's terms (see [Frequently asked questions](#frequently-asked-questions)). The Claude documentation links point to the Markdown version of each page, as listed in the `llms.txt` indexes for [platform.claude.com](https://platform.claude.com/llms.txt) and [code.claude.com](https://code.claude.com/docs/llms.txt).

| Guide area | Documentation page | What to check there (as of September 2026) |
|---|---|---|
| 1.1 agentic loop, `stop_reason` | [How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works.md) | The loop continues on `tool_use` and exits on any other stop reason (`end_turn`, `max_tokens`, `stop_sequence` or `refusal`); `pause_turn` for server-tool loops. The guide frames the loop as continue on `tool_use`, stop on `end_turn`. |
| 1.2, 1.3, 3.4 subagents, the Task tool, Explore | [Agent SDK: subagents](https://code.claude.com/docs/en/agent-sdk/subagents.md), [Claude Code: subagents](https://code.claude.com/docs/en/sub-agents.md) | Task renamed Agent in version 2.1.63; `Task(...)` references still work as aliases; AgentDefinition `description`, `prompt`, `tools`; the only content a parent passes to a subagent is the Agent tool's prompt string (forks excepted); the Explore subagent. |
| 1.5 hooks | [Agent SDK: hooks](https://code.claude.com/docs/en/agent-sdk/hooks.md) | `PreToolUse` blocks with `permissionDecision: "deny"`; `PostToolUse` replaces any tool's output with `updatedToolOutput` (the MCP-only `updatedMCPToolOutput` is deprecated). |
| 1.7 sessions | [Agent SDK: sessions](https://code.claude.com/docs/en/agent-sdk/sessions.md), [CLI reference](https://code.claude.com/docs/en/cli-reference.md) | `fork_session` (Python), `forkSession` (TypeScript), `--fork-session` (CLI); forking copies conversation history, not files; `--resume <name>`. |
| 2.2 MCP errors, 2.4 resources | [MCP specification: tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools.md), [MCP specification: resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources.md) | Tool execution errors, including input validation errors, come back in the tool result with `isError: true`, and clients should pass them to the model so it can self-correct. The guide's `errorCategory`, `isRetryable` and `retriable` fields are application conventions it recommends; the spec's tools page has no fields by those names. Resources expose data such as database schemas. |
| 2.3, 4.3 `tool_choice` | [Define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools.md) | Four options (`auto`, `any`, `tool` and `none`; the guide lists the first three), with `none` the default when no tools are provided; `any` and forced `tool` return a 400 error on Claude Opus 5.5, Claude Fable 5.1 and Claude Mythos 5.1. |
| 2.4 MCP in Claude Code | [Claude Code: MCP](https://code.claude.com/docs/en/mcp.md) | Three scopes (local is the default); `${VAR}` and `${VAR:-default}` expansion; tool search defers tool definitions until needed; resources are referenced with @ mentions (`@server:protocol://resource/path`). |
| 2.5 built-in tools | [Tools reference](https://code.claude.com/docs/en/tools-reference.md) | On macOS, Linux and WSL, Glob and Grep are left out of the default tool set (naming either in `--allowedTools` restores both); a non-unique Edit match is resolved with a longer `old_string` or `replace_all: true` (the guide's answer is Read + Write). |
| 3.1, 3.3 CLAUDE.md and rules | [Claude Code: memory](https://code.claude.com/docs/en/memory.md) | `@path/to/import` imports; path-scoped rules trigger when Claude reads a matching file (the guide says when editing), and rules without `paths` load unconditionally; `/context` shows which CLAUDE.md and rules files loaded (the guide uses `/memory` for this check). |
| 3.2 commands and skills | [Claude Code: skills](https://code.claude.com/docs/en/skills.md) | Commands merged into skills; `allowed-tools` pre-approves rather than restricts; `context: fork` runs the skill in an isolated subagent that does not see the conversation history; `argument-hint` is an autocomplete hint (the guide says it prompts for required parameters). |
| 3.4 plan mode | [Permission modes](https://code.claude.com/docs/en/permission-modes.md) | Plan mode researches and proposes changes without making them; enter it with `Shift+Tab`, or for a single prompt by prefixing it with `/plan`. |
| 3.6 CI/CD | [Headless mode](https://code.claude.com/docs/en/headless.md), [CLI reference](https://code.claude.com/docs/en/cli-reference.md) | `-p`, `--output-format json`, `--json-schema`, and the `structured_output` field. |
| 4.3 structured output | [Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs.md) | JSON outputs (`output_config.format`) and strict tool use (`strict: true`), both backed by constrained decoding. The guide's answer for guaranteed schema-compliant output is tool use with a JSON schema. |
| 4.5 batches | [Batch processing](https://platform.claude.com/docs/en/build-with-claude/batch-processing.md) | 50% lower cost, 24-hour expiry, 29-day result retention, results in any order, `custom_id` format limits; server tools work in batch requests through the same server-side agentic loop as the synchronous Messages API. A tool your own code runs still cannot execute in the middle of a batch request (our reading; the page does not say this in those words). |
| 5.4 `/compact` | [Commands](https://code.claude.com/docs/en/commands.md) | `/compact` with optional focus instructions. |

### Anthropic engineering posts and repositories

The mapping of each item to task statements is ours.

- [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): background on workflows versus agents, prompt chaining and the orchestrator-workers pattern, for task statements 1.1, 1.2 and 1.6. It defines workflows as "systems where LLMs and tools are orchestrated through predefined code paths." The post now carries a note that "Much of the tooling landscape described in this post has changed since December 2024."
- [Writing tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents): background for Domain 2.
- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): background for Domain 5.
- [anthropics/claude-cookbooks](https://github.com/anthropics/claude-cookbooks): copy-able code and guides, actively maintained (last push September 22, 2026). The `claude_agent_sdk` folder has agent notebooks, and the `tool_use` folder includes `tool_choice.ipynb` and `tool_use_with_pydantic.ipynb`. The old name, anthropics/anthropic-cookbook, redirects here.
- [anthropics/skills](https://github.com/anthropics/skills): Anthropic's public Agent Skills repository, useful for real SKILL.md examples.
- [anthropics/claude-quickstarts](https://github.com/anthropics/claude-quickstarts) (starter projects built on the Claude API) and [anthropics/claude-agent-sdk-demos](https://github.com/anthropics/claude-agent-sdk-demos). The demos repository still carries its old description, "Claude Code SDK Demos"; the product is now the Claude Agent SDK.
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers): the MCP reference server implementations, handy when practicing task statement 2.4.

!!! warning "Two official repositories to use with care"

    [anthropics/courses](https://github.com/anthropics/courses) was archived on September 15, 2026, and its notebooks favor Claude 3 Haiku; the claude-cookbooks README still sends API beginners to it. The [prompt engineering interactive tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial) hard-codes `claude-3-haiku-20240307`, a model retired on April 20, 2026; the deprecations page lists `claude-haiku-4-5-20251001` as its replacement. The tutorial's Chapter 5 teaches prefill ("Speaking for Claude"). Current docs say a prefill on the last assistant turn returns a 400 error starting with Claude 4.6 models, although it still works on Claude Haiku 4.5. Use both repositories for the concepts, swap in a current model ID, and do not carry the prefill technique over to current models.

### Community study material

These are third-party resources; check any answer key against Anthropic's rationales before trusting it. A frequent gap: several still describe the March 2026 format of one correct answer and three distractors, while the July 2026 guide includes multiple-response items.

| Resource | Cost and access | What checks out | What to watch |
|---|---|---|---|
| [claudecertificationguide.com](https://claudecertificationguide.com/) | Free, no sign-up | CCAR-F track with 30 lessons, 250+ questions and a full mock exam. | Its study estimate (15 to 20 hours with Claude experience, 30 to 40 if new) cites no source. Independent of Anthropic. |
| [claudecertifiedarchitects.com](https://www.claudecertifiedarchitects.com/) | &#36;49 one-time; free accounts draw from a 55-question sample | A 400-question bank that includes multiple-response items; states correctly that 720 is a scaled score, not a percentage. | Its article on dumps says the guide's rules say nothing about studying from leaked material; the separate Exam Policy lists use of unauthorized published questions as misconduct. |
| [CertSafari](https://www.certsafari.com/anthropic/claude-certified-architect-foundations) | Free | 480-question CCAR-F bank, last updated September 17, 2026; 35 free sample questions with explanations. | Offers multiple choice only. A one-person project. Its curated list includes an April 2026 article that describes the old format. |
| [Tutorials Dojo](https://tutorialsdojo.com/ccar-f-claude-certified-architect-foundations-study-guide/) | Free study guide; practice exams &#36;14.99 | Lists the official domain weights; sample answers cite official docs. | Mixed format signals: the study guide describes one correct answer and three distractors, while the product page lists single-choice and multiple-choice questions. The product page says questions align with "trusted Microsoft resources". A sample keys forced `tool_choice`, which matches a guide skill bullet but returns a 400 error on Claude Opus 5.5, Claude Fable 5.1 and Claude Mythos 5.1. |
| [OpenExamPrep](https://open-exam-prep.com/practice/anthropic-cca-f) | Free, no registration; says it may earn a commission from purchases | Retake rules stated correctly; says its questions are independently developed. | Its 40 to 100 hour estimate cites no source. Its Domain 5 description includes prompt caching and RAG. The guide's out-of-scope list includes prompt caching implementation details (beyond knowing it exists) and embedding model or vector database implementation details. |
| [FlashGenius](https://flashgenius.net/sample-tests/ccar-f) | 10 free questions a day; &#36;14.99 a month | Instant scoring with explanations. | Describes CCAR-F as multiple-choice only and lists prefilling as a prompt-engineering topic. |
| [Sundog Education practice exams](https://www.udemy.com/course/anthropic-claude-certified-architect-3-full-practice-exams/) (Udemy) | Paid | 360 questions; explanations link to official Anthropic docs. | Single-answer format (1 correct, 3 distractors). |
| [Jacob Bushong video course](https://www.udemy.com/course/certified-claude-architect-masterclass-2026/) (Udemy) | Paid | 15h 3m of video, updated September 2026; domain weights match the guide, though it numbers the domains differently. | Its syllabus covers embeddings and hybrid retrieval, while the guide lists embedding model or vector database implementation details as out of scope; uses the old name "claude code sdk"; teaches prefilling. |
| [freeCodeCamp CCAR-F course](https://www.youtube.com/watch?v=reDRM0tqhNs) (YouTube, developed by ExamPro) | Free | 45,516 seconds of video (about 12.6 hours, our arithmetic), uploaded July 20, 2026. | The [announcement article](https://www.freecodecamp.org/news/claude-certified-architect-foundations-prep-for-anthropic-s-new-certification-exam/) calls MCP "Anthropic's custom MCP framework"; MCP is an open standard (created by Anthropic). |
| [Peace Of Code playlist](https://www.youtube.com/playlist?list=PLviC8AFqAj5A9MHkRIn2fU5Ac2lEdJxNf) (YouTube) | Free | 23 videos, including an episode that walks through every official CCAR-F sample question option by option. | [That episode](https://www.youtube.com/watch?v=-NymqBcFy6E) (June 21, 2026) predates the July guide, but the 12 sample questions and their keyed answers are the same in the March and July guides (our comparison). The episode also adds its own practice questions on "ground the official syllabus leaves untested". |
| [daronyondem/claude-architect-exam-guide](https://github.com/daronyondem/claude-architect-exam-guide) | Free, CC BY 4.0 | Teaches the domains with no exam questions, for developers new to LLMs. | Last pushed June 10, 2026, before guide version 1.0; mentions response prefill. |
| [dnacenta/claude-certified-architect](https://github.com/dnacenta/claude-certified-architect) | Free | Refreshed in September 2026; also has overview pages for the other three exams. | Still lists the format as 60 multiple-choice questions. |
| [Amey-Thakur/CLAUDE-CERTIFICATIONS](https://github.com/Amey-Thakur/CLAUDE-CERTIFICATIONS) | Free | Covers all four exams; codes, item counts and fees match the guides; its 35 CCAR-F practice questions are labeled as unofficial, not live items. | Says every course in the program is free on the public Claude Academy with no partner account; the dedicated prep courses for the other three exams sit in the Partner Academy. |
| [paullarionov/claude-certified-architect](https://github.com/paullarionov/claude-certified-architect) | Free | Study notes and Anki decks, pushed August 31, 2026. | Single-answer format; says 4 of 8 scenarios (the guide says 4 of 6); keys prefill as a correct answer; still tells readers to take the retired practice exam. |
| [hamzafarooq/claude-certified-architect](https://github.com/hamzafarooq/claude-certified-architect) | Free | Cheat sheets and a 64-question browser practice exam. | Its README invites questions "from the real exam that you're allowed to share" (its contributing file adds that verbatim NDA questions should not be reproduced); the Exam Policy bars distributing any exam question. |

Candidates' own write-ups are summarized, with links, in [What candidates report](#what-candidates-report). Some of those write-ups come from authors or publishers with something to offer: a paid prep class, a related Nanodegree, or the author's own practice exam. The tables there say which.

!!! danger "Sites that claim to sell real exam questions"

    Some sites advertise "real exam questions" or "Actual Questions" for this exam, or sell "dumps", on their own pages, among them SPOTO, ExamTopics, DumpsBase, CertsHero, SkillCertPro and ExamHeist. They are not linked here. Anthropic's [Exam Policy](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F34hhd92iyp94a0gtbr15cy5jk%2Fpublic%2F1782870704%2FAnthropic+Certification+Exam+Policy.pdf) lists "use of unauthorized publication of Exam questions or answers" as prohibited misconduct, and the sanctions it lists include invalidated results, a revoked certification and a ban from future exams, with no obligation to refund exam fees. Every candidate also agrees before the exam that questions, answer options and scenarios are Anthropic's confidential property. Comment threads under some study posts promote dump vendors too; skip them.

## Frequently asked questions

Short answers on the exam's older names, the scenario draw, item formats, official practice material, guide wording versus current docs, the cloud courses, scoring, difficulty and prerequisites. Where a topic has its own section on this page, the answer links to it.

??? question "Is CCAR-F the same exam as CCA-F or CCAF?"

    Yes. The version 1.0 guide (July 2026) and Pearson VUE's Anthropic page use the code CCAR-F for Claude Certified Architect, Foundations. It launched in March 2026 as the first Claude technical certification and was the only exam until Anthropic added three more in July 2026. In that period candidates, blogs and practice sites often called it CCA-F or CCAF, and many still do. The short form was not only a community label: Anthropic's [early-access page](https://web.archive.org/web/20260314145654/https://anthropic.skilljar.com/claude-certified-architect-foundations-access-request) (archived March 14, 2026) said "Practitioners who pass receive a CCA-F badge". The version 1.0 guide uses neither CCA-F nor CCAF. Check any resource that says "CCA-F" for details from before the program moved to Pearson on June 30, 2026: a &#36;99 fee, 6-month validity, ProctorFree proctoring, single-answer items only, or a 6-month retake lockout (as pre-move candidates described it).

??? question "Which scenarios will I get, and do I need all six?"

    You get 4 scenarios picked at random from the guide's 6, so prepare all six. The guide's 12 sample questions cover only four of them: none is set in Scenario 4 (Developer Productivity with Claude) or Scenario 6 (Structured Data Extraction). Scenario 4 lists Domains 2, 3 and 1 as primary, and its text describes an Agent SDK agent that helps engineers explore unfamiliar codebases, uses the built-in tools (Read, Write, Bash, Grep, Glob) and integrates with MCP servers. For it, practice task statements 2.4 and 2.5 alongside the Domain 3 and Domain 1 material, plus 5.4 for the codebase exploration (our mapping). Scenario 6 lists Domains 4 and 5, and the guide's Exercise 3 builds that kind of extraction pipeline: JSON schemas, tool use for structured output, validation-retry loops and batch processing. [One candidate](https://www.reddit.com/r/ClaudeAI/comments/1wc73tf/ccarf_about_to_take_exam_how_long_are_the/), in a Reddit comment dated September 10, 2026, described four blocks of 15 questions, each keeping one of the six published scenarios on the left of the screen throughout. See [The six scenarios](#the-six-scenarios).

??? question "Will there be multiple-response items?"

    Plan for them. The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) lists the item format as "Multiple-choice and multiple-response items; each item states how many responses to select". All 12 official samples are single-answer with four options, and [one Pearson-era CCAR-F candidate](https://www.reddit.com/r/ClaudeAI/comments/1v5zrru/passed_the_ccarf_with_9041000/) reported that every item on their exam was single-answer, but that is one exam sitting, and the [certification FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) also says "All Claude certification exams use multiple choice and scenario-based multiple response questions." Several third-party practice banks still use only the single-answer format of the March 2026 guide, so practice choosing two or more options on your own.

??? question "Is there an official practice exam?"

    Not any more. The [certification FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says "The practice exam available on the previous platform was retired in the move to Pearson." The [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) 12 sample questions "are drawn from the practice test and include explanations to aid learning", so they are the official practice you have. Third-party mocks exist. In all six candidate reports on this page that give both a practice or mock score and a real score, the best practice figure was higher than the real one (our comparison of the published figures). See [What candidates report](#what-candidates-report) and [Resources](#resources).

??? question "The guide and the current docs disagree. Which answer does the exam want?"

    The guide's. The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says "Exam items are written against these objectives", and neither the guide nor the certification FAQ says whether the live item bank has been updated for any of the differences between the two. Some differences predate the guide itself: the Task tool was renamed Agent in Claude Code 2.1.63 (February 28, 2026), and the July 2026 guide still says Task. Answer in the guide's terms, and learn the current behavior alongside it for real projects. The full list, with the domain section that teaches each difference and how to answer, is in [Where current documentation differs from the guide](#where-current-documentation-differs-from-the-guide).

??? question "Do I need to study Amazon Bedrock and Google Cloud because they are on the prep list?"

    Only if you deploy on them. Anthropic's CCAR-F prep page lists Claude with Amazon Bedrock and Claude on Google Cloud among its seven courses, but the [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) lists "Specific cloud provider configurations (AWS, GCP, Azure)" among the topics that "will not appear on the exam". The same list rules out computer use, vision and image analysis, streaming API implementation, fine-tuning, and prompt caching implementation details (beyond knowing it exists). See [What is out of scope](#what-is-out-of-scope) and the course notes in the [Study plan](#study-plan).

??? question "What score do I need, and how many questions is that?"

    A scaled 720 on a range of 100 to 1,000; the certification FAQ gives the same 720 minimum for all four exams. It is a point on the scaled range, not a raw 72% of items correct. Pass or fail rests on the total scaled score; the per-domain percentages on your score report are there to help you understand your performance and do not decide the result. The guide does not publish how many correct answers a 720 requires, and it says scaled scoring helps equate scores across exam forms of slightly different difficulty. Two candidates published raw counts from pre-Pearson exams: [44 of 60 correct scaled to 738](https://verygood.ventures/blog/passing-the-claude-certified-architect-exam/), and [48 of 60 to 822](https://www.claudeaimalaysia.com/malaysia-claude-ccaf-studyguide.html). Two data points do not give a conversion formula.

??? question "Is CCAR-F harder than the other three Claude exams?"

    Neither the guide nor the certification FAQ ranks the exams by difficulty, and none of the four guides publishes a pass rate. Candidate opinions differ: [Matthew Purcell](https://www.linkedin.com/pulse/claude-certification-exams-honest-review-matthew-purcell-byo2e) (who sat CCAR-F on June 3, 2026, before the move to Pearson, and promotes their own practice exam) and [bluepanda](https://dev.to/bluepanda/i-cleared-all-4-anthropic-claude-certifications-what-you-actually-need-to-know-and-what-to-skip-224m) each ranked CCAR-F the hardest of the four and found CCAR-P easier, [a third candidate on Reddit](https://www.reddit.com/r/ClaudeAI/comments/1v2x1p7/got_all_4_claude_certifications_ccap_ccaf_ccdvf/), who runs a practice site, also found the Professional exam easier than Foundations, and [one CCAR-F candidate](https://www.reddit.com/r/ClaudeAI/comments/1vaxsws/claude_certified_architect_foundations_ccarf_how/) wrote that it "didn't feel like the hardest at all." [Purcell's one-line comparison](https://www.linkedin.com/posts/purcellmatthew_claude-certified-architect-professional-practice-activity-7482176978008342528-204Q): CCAR-F is "much more hands-on with API and SDK mechanics, with less architectural judgement" than the Professional exam. The guide's typical candidate has 6+ months of practical experience building with the Claude APIs, the Agent SDK, Claude Code and MCP.

??? question "Do I need CCAR-F before I can take CCAR-P?"

    No. The [certification FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says Foundations "is the natural place to start and there's no formal prerequisite, so you can take Professional without holding Foundations." It adds that "Foundations does not convert or upgrade to Professional automatically." Each is its own credential with its own Credly badge; see [Claude Certified Architect, Professional (CCAR-P)](claude-certified-architect-professional.md).

??? info "Sources"

    - [Claude Certified Architect, Foundations Exam Guide, Version 1.0 (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): exam details table, intended audience, blueprint weights and their definition, the six scenarios and their primary domains, task statements and objective bullets, preparation exercises, sample question grouping, appendix lists and document control
    - [Claude Certified Architect, Foundations certification page (Anthropic Partner Academy)](https://anthropic-partners.skilljar.com/claude-certified-architect-foundations-certification): where the exam guide PDF is linked
    - [Claude Certification FAQ (Anthropic Partner Academy)](https://anthropic-partners.skilljar.com/page/faq-certifications): partner-only eligibility, the June 30, 2026 price change, partner discounts, seat time, the 48-hour change window, English-only delivery, Foundations versus Professional, partner network eligibility, using the section breakdown before a retake, and the guide as the authoritative source for exam scope
    - [Certification Policies (Anthropic Partner Academy)](https://anthropic-partners.skilljar.com/page/policies-certifications): the 48-hour cancel and reschedule window
    - [Partner certifications (Anthropic Partner Academy)](https://anthropic-partners.skilljar.com/page/partner-certifications): the Associate certification does not count toward partner tier eligibility
    - [Pearson VUE: Anthropic certification program](https://www.pearsonvue.com/us/en/anthropic.html): the 48-hour window for test center appointments
    - [Claude Certified Associate, Foundations Exam Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf): Associate audience, experience, prerequisites line and the boundary with the Architect and Developer credentials
    - [Claude Certified Developer, Foundations Exam Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf): Developer audience, experience, prerequisites line, integration skills and domain weights
    - [Claude Certified Architect, Professional Exam Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf): Professional audience, experience, prerequisites line and single-row document control
    - [Claude Code docs: Subagents](https://code.claude.com/docs/en/sub-agents): the Task tool renamed to Agent in version 2.1.63, with `Task(...)` kept as an alias
    - [Claude Code docs: Skills](https://code.claude.com/docs/en/skills): custom commands merged into skills; `allowed-tools` pre-approves and `disallowed-tools` removes tools
    - [Claude Code docs: MCP](https://code.claude.com/docs/en/mcp): three MCP scopes, with local scope as the default
    - [Claude Code docs: Tools reference](https://code.claude.com/docs/en/tools-reference): Glob and Grep left out of the default tool set on macOS, Linux and WSL
    - [Claude API docs: Define tools](https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools): the four `tool_choice` options and the models on which forced tool use returns a 400 error
    - [Claude API docs: Structured outputs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs): JSON outputs (`output_config.format`) and strict tool use (`strict: true`)
    - [Claude Code changelog](https://code.claude.com/docs/en/changelog.md): version 2.1.63 dated February 28, 2026
    - [Claude Agent SDK: Subagents](https://code.claude.com/docs/en/agent-sdk/subagents): the Agent tool (formerly Task) and its naming in `tool_use` blocks versus `system:init`, the `agents` option, AgentDefinition fields, the prompt string as the only parent-to-subagent content, parallel subagents finishing in the time of the slowest, subagent limits and budgets, resuming a subagent
    - [Claude Agent SDK: Hooks](https://code.claude.com/docs/en/agent-sdk/hooks): what SDK hooks are, `PreToolUse` and `PostToolUse` outputs, `updatedToolOutput`, `permissionDecisionReason`, `systemMessage`, matchers, callback inputs, decision precedence, callback timeouts
    - [Claude Agent SDK: Sessions](https://code.claude.com/docs/en/agent-sdk/sessions): what a session is, resume, continue and fork semantics, `fork_session` / `forkSession`, forks branching history but not files, capturing results into a fresh session
    - [Claude Agent SDK: How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop): the SDK loop running until a response has no tool calls, `max_turns` and `max_budget_usd` with their error subtypes, budgets as a production default, `allowed_tools` as auto-approval, hooks running outside the context window
    - [Claude Agent SDK: Permissions](https://code.claude.com/docs/en/agent-sdk/permissions): permission evaluation order with hooks first, hook deny applying in `bypassPermissions`, `Agent` running without approval in default mode
    - [Claude Agent SDK: Python reference](https://code.claude.com/docs/en/agent-sdk/python): `continue_conversation`, `fork_session`, AgentDefinition camelCase fields and the snake_case `TypeError`, Python hook event list, `Task` accepted as an alias
    - [Claude Agent SDK: Handle approvals and user input](https://code.claude.com/docs/en/agent-sdk/user-input): returning `defer` from a `PreToolUse` hook so a slow human decision can resume later
    - [Claude Code: Sessions](https://code.claude.com/docs/en/sessions): `claude --resume <name>`, naming with `-n` and `/rename`, ambiguous names, `-p` and SDK sessions left out of the picker, restored history, branching with `/branch` and `--fork-session`, resume-from-summary versus resume as-is
    - [Claude Code: CLI reference](https://code.claude.com/docs/en/cli-reference): `--fork-session` and the `--resume` examples
    - [Claude Code: Hooks reference](https://code.claude.com/docs/en/hooks): `PostToolUse` firing after success, `updatedToolOutput` changing only what Claude sees, deny reasons shown to Claude, matcher rules, timeout behavior by hook type, `FileChanged` messages reaching the user, hook context replayed on resume, deprecated `PreToolUse` fields
    - [Claude Code: Hooks guide](https://code.claude.com/docs/en/hooks-guide): hooks as deterministic control
    - [Claude Code: Glossary](https://code.claude.com/docs/en/glossary): hooks firing at fixed lifecycle points rather than at the model's discretion
    - [Claude Code: Best practices](https://code.claude.com/docs/en/best-practices): hooks as deterministic versus advisory CLAUDE.md, descriptive session names, the Explore, Plan, Implement, Commit workflow
    - [Claude Code: Checkpointing](https://code.claude.com/docs/en/checkpointing): outside edits and other sessions' edits not captured
    - [Claude Code: How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works): switching branches changes files but not conversation history
    - [Claude Code: Set up Claude Code in a monorepo or large codebase](https://code.claude.com/docs/en/large-codebases): the plan file re-injected after each compaction
    - [Claude Code: Run agents in parallel](https://code.claude.com/docs/en/agents): subagents reporting back to the spawning conversation versus teammates messaging each other
    - [Claude Code: Orchestrate teams of Claude Code sessions](https://code.claude.com/docs/en/agent-teams): agent teams experimental and disabled by default, subagent coordination by the main agent
    - [How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works): the model never executes tools, the `while stop_reason == "tool_use"` loop, exit on any other stop reason, `pause_turn` for server tools, regex-parsed decisions belonging in tool calls
    - [Handling stop reasons](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons): stop reason values, the manual tool loop snippet, empty `end_turn` responses after text added to tool results, `pause_turn` handling
    - [Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls): `tool_use` and `tool_result` structure, ordering rules, `is_error` results and instructive error messages
    - [Troubleshooting tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/troubleshooting-tool-use): the `tool_use ids were found without tool_result blocks immediately after` error
    - [Working with messages](https://platform.claude.com/docs/en/build-with-claude/working-with-messages): the stateless Messages API and full-history requests
    - [Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview): when Claude calls a tool under `auto`
    - [Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use): multiple tool calls per response, returning all results together, parallel for independent read-only calls
    - [Tool Runner](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner): when to use the manual loop instead
    - [Build a tool-using agent](https://platform.claude.com/docs/en/agents-and-tools/tool-use/build-a-tool-using-agent): how Claude reacts to an `is_error` result
    - [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices): explicit prompt chaining still useful for inspection and fixed pipelines, starting a brand new context window, JSON for state, explaining why an instruction exists
    - [Citations](https://platform.claude.com/docs/en/build-with-claude/citations): citations cannot be combined with `output_config.format`
    - [Ticket routing use-case guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/ticket-routing): multiple issues in one interaction obscuring the primary concern
    - [Commerce agents use-case guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/commerce-agents): staged changes approved outside the chat, "An approval typed in chat approves nothing."
    - [Building effective agents (Anthropic engineering)](https://www.anthropic.com/engineering/building-effective-agents): workflows versus agents, prompt chaining, orchestrator-workers, stopping conditions, separate calls per consideration, adding complexity only when it improves outcomes
    - [How we built our multi-agent research system (Anthropic engineering)](https://www.anthropic.com/engineering/multi-agent-research-system): orchestrator-worker architecture, effort-scaling rules, delegation requirements, vague delegation causing gaps, parallel speedup, heuristics over rigid rules, lead-agent refinement loop, token multipliers, lightweight references
    - [Building multi-agent systems: when and how to use them (Claude blog)](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them): multi-agent token cost versus a single agent, single-agent prompting matching multi-agent results
    - [research_lead_agent.md (Anthropic cookbook prompt)](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_lead_agent.md): mandatory parallel subagent creation, crisp sub-topic boundaries, updating the plan from findings, gap filling and stopping rules, delegation checklist
    - [Customer escalation skill (Anthropic knowledge-work-plugins)](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/customer-support/skills/customer-escalation/SKILL.md): escalation brief structure and guidance, written for human support staff who use Claude
    - [Claude API errors (Claude API docs)](https://platform.claude.com/docs/en/api/errors): the 400 error text for forced `tool_choice` and the recommended substitutes
    - [Thinking (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/thinking): manual extended thinking rejects forced tool use; adaptive thinking allows it on supporting models
    - [Tool search tool (Claude API docs)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool): tool selection accuracy degrades past 30 to 50 tools
    - [Web search tool (Claude API docs)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool): a search with no matches returns an empty list, not an error
    - [Optimizing for cost and intelligence (Claude API docs)](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence): stale instructions also appear in tool descriptions and skills
    - [Prompting Claude Fable 5.1 (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-1): whole-file rewrites cost more output tokens and time than targeted edits
    - [MCP connector (Claude API docs)](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector): the connector supports only tool calls; client-side helpers for resources
    - [Connect to MCP servers (Claude Code docs)](https://code.claude.com/docs/en/mcp-quickstart): where local and user scoped servers are stored in `~/.claude.json`
    - [Debug your configuration (Claude Code docs)](https://code.claude.com/docs/en/debug-your-config): `settings.json` does not read an `mcpServers` key
    - [Configure permissions (Claude Code docs)](https://code.claude.com/docs/en/permissions): permission rules that match MCP tools
    - [Common workflows (Claude Code docs)](https://code.claude.com/docs/en/common-workflows): start broad then narrow, delegate exploration, code intelligence plugins
    - [Give Claude custom tools (Agent SDK docs)](https://code.claude.com/docs/en/agent-sdk/custom-tools): `isError` and `is_error` in handlers, availability versus permission settings
    - [Connect to external tools with MCP (Agent SDK docs)](https://code.claude.com/docs/en/agent-sdk/mcp): `mcp_servers`, explicit permission for MCP tools, wildcard `allowed_tools`
    - [Agent SDK reference: TypeScript](https://code.claude.com/docs/en/agent-sdk/typescript): input shapes for Edit, Glob and Grep
    - [MCP specification: Tools (revision 2026-07-28)](https://modelcontextprotocol.io/specification/2026-07-28/server/tools): protocol errors versus tool execution errors with `isError`, tool name rules, `inputSchema`
    - [MCP specification: Schema (revision 2026-07-28)](https://modelcontextprotocol.io/specification/2026-07-28/schema): `isError` default and why tool errors belong in the result
    - [MCP specification: Changelog (revision 2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/changelog): input validation errors returned as tool execution errors
    - [MCP specification: Resources (revision 2026-07-28)](https://modelcontextprotocol.io/specification/2026-07-28/server/resources.md): no empty contents for a resource that does not exist
    - [Understanding MCP servers (MCP docs)](https://modelcontextprotocol.io/docs/2026-07-28/learn/server-concepts): tools, resources and prompts and who controls each
    - [Architecture overview (MCP docs)](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture): hosts combine tools from all connected servers into one registry
    - [The MCP Registry](https://modelcontextprotocol.io/registry/about): the official registry of public servers, in preview
    - [modelcontextprotocol/servers (GitHub)](https://github.com/modelcontextprotocol/servers): reference servers are educational examples, not production-ready
    - [Writing effective tools for agents (Anthropic Engineering)](https://www.anthropic.com/engineering/writing-tools-for-agents): distinct purpose per tool, overlap confuses agents, unambiguous parameter names, actionable error responses
    - [Effective context engineering for AI agents (Anthropic Engineering)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): Claude Code retrieves files just in time with glob and grep
    - [Research subagent prompt (Anthropic claude-cookbooks)](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_subagent.md): try another tool or query, never repeat the exact same query
    - [Claude Code docs: How Claude remembers your project](https://code.claude.com/docs/en/memory): CLAUDE.md locations and load order, lazy subdirectory loading, `@path` imports, `.claude/rules/` and `paths` frontmatter, rule loading on read, `/memory` and `/context`, managed and local CLAUDE.md, AGENTS.md
    - [Claude Code docs: Explore the .claude directory](https://code.claude.com/docs/en/claude-directory): command-file and skill frontmatter, the `argument-hint` example, the test-file rule example, rules as guidance rather than enforcement
    - [Claude Code docs: Extend Claude Code (features overview)](https://code.claude.com/docs/en/features-overview): choosing between CLAUDE.md, rules, skills, subagents and hooks; context cost of each
    - [Claude Code docs: Choose a permission mode](https://code.claude.com/docs/en/permission-modes): plan mode behavior, entering and leaving it, approval options, `defaultMode`, `dontAsk` for locked-down CI, the locked-down CI command example
    - [Claude Code docs: Explore the context window](https://code.claude.com/docs/en/context-window): what survives compaction, including the plan file and path-scoped rules
    - [Claude Code docs: Model configuration](https://code.claude.com/docs/en/model-config): the `opusplan` alias
    - [Claude Code docs: Manage costs effectively](https://code.claude.com/docs/en/costs): plan mode prevents expensive rework
    - [Claude Code docs: Run Claude Code programmatically](https://code.claude.com/docs/en/headless): `-p`, output formats, `--json-schema` and `structured_output`, invalid-schema error, exit codes, Manual starting mode for `-p`, `--bare`, trust behavior of `-p` runs
    - [Claude Code docs: Commands](https://code.claude.com/docs/en/commands): `/batch` as a bundled skill, `/memory`
    - [Claude Code docs: All settings (settings reference)](https://code.claude.com/docs/en/settings-reference): `~/.claude/commands/` and `.claude/commands/` as current command locations
    - [Claude Code docs: Claude Code GitHub Actions](https://code.claude.com/docs/en/github-actions): CLAUDE.md for code style and review criteria in CI
    - [Claude Code docs: Code Review](https://code.claude.com/docs/en/code-review): tuning with CLAUDE.md or REVIEW.md, push-triggered re-reviews, auto-resolved threads, deduplicated findings
    - [anthropics/claude-code-action: Usage](https://github.com/anthropics/claude-code-action/blob/main/docs/usage.md): `--json-schema` in `claude_args` and the `structured_output` step output
    - [Claude cookbook: Iterate, do, observe, fix (Managed Agents, failing tests)](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/managed_agents/CMA_iterate_fix_failing_tests.ipynb): an example of interacting test failures that resolve once their dependencies are fixed
    - [Claude blog: Prompt engineering best practices for 2026](https://claude.com/blog/best-practices-for-prompt-engineering): when to add examples
    - [Claude Code: Best practices for agentic coding, Anthropic engineering post (archived 2025-06-02 snapshot)](https://web.archive.org/web/20250602202220/https://www.anthropic.com/engineering/claude-code-best-practices): the test-first workflow (confirm tests fail, do not modify tests, check for overfitting) and fixing independent issues one by one
    - [Strict tool use (Claude API docs)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use): `"strict": true` on a tool definition and what non-strict tool calls can get wrong.
    - [Claude Opus 5.5 migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide): replacing forced `tool_choice` with `auto` plus strict tools and a prompt instruction.
    - [Release notes (Claude Platform)](https://platform.claude.com/docs/en/release-notes/overview): dates of the forced `tool_choice` restriction on Claude Fable 5.1, Claude Mythos 5.1 and Claude Opus 5.5.
    - [Batch processing (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/batch-processing): 50% cost, 24-hour expiry, typical completion time, `custom_id` format and correlation, batch size limits, result order, result types and retention, the results-processing example, asynchronous validation, and server tools inside batches.
    - [Context windows (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/context-windows): input that exceeds the context window is rejected as an invalid request.
    - [Prompting Claude Sonnet 5 (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5): how "be conservative" style review instructions affect precision and recall, and the coverage-first finding prompt with confidence and severity.
    - [Prompting Claude Opus 4.8 (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-4-8): a concrete single-pass bar for code review findings.
    - [Prompting Claude Fable 5 (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5): fresh-context verifier subagents compared with self-critique.
    - [Prompting Claude Opus 5 (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5): over-verification on Claude Opus 5 when prompts add explicit verification instructions.
    - [Security guidance plugin (Claude Code docs)](https://code.claude.com/docs/en/security-guidance): a reviewer that is not the Claude instance that wrote the code and starts from the diff.
    - [Environment variables (Claude Code docs)](https://code.claude.com/docs/en/env-vars): `MAX_STRUCTURED_OUTPUT_RETRIES` and its default of 5 attempts.
    - [Structured outputs in the Agent SDK (Claude Code docs)](https://code.claude.com/docs/en/agent-sdk/structured-outputs): optional fields when a task may lack information, re-prompting on schema mismatch, and the `error_max_structured_output_retries` result subtype.
    - [Building agents with the Claude Agent SDK (claude.com blog)](https://claude.com/blog/building-agents-with-the-claude-agent-sdk): feedback that states the rules and explains which ones failed and why.
    - [code-review plugin command (anthropics/claude-code on GitHub)](https://raw.githubusercontent.com/anthropics/claude-code/main/plugins/code-review/commands/code-review.md): false positives erode trust and waste reviewer time.
    - [claude-code-security-review prompts (anthropics on GitHub)](https://raw.githubusercontent.com/anthropics/claude-code-security-review/main/claudecode/prompts.py): a `category` on each security finding.
    - [claude-code-security-review findings filter (anthropics on GitHub)](https://raw.githubusercontent.com/anthropics/claude-code-security-review/main/claudecode/findings_filter.py): exclusion counts kept per reason.
    - [Extracting structured JSON using Claude and tool use (Claude cookbook)](https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/extracting_structured_json.ipynb): extraction tools whose input schema describes the JSON, read from the `tool_use` block.
    - [Context editing](https://platform.claude.com/docs/en/build-with-claude/context-editing): tasks that need exact state across many variables are less suited to compaction
    - [Memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool): client-side memory tool configuration `memory_20250818` and the "ASSUME INTERRUPTION" protocol
    - [On-demand compaction](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand): custom compaction `instructions` of up to 16,384 characters
    - [Search results](https://platform.claude.com/docs/en/build-with-claude/search-results): return a text block describing the outcome when a search fails or returns nothing
    - [Reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations): find a supporting quote for each claim or retract it
    - [Define success criteria and build evaluations](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests): most use cases need multidimensional evaluation
    - [Customer support agent guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/customer-support-chat): escalation accuracy target of 95% or higher; sentiment tracked as a business metric
    - [Content moderation guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/content-moderation): risk tiers that auto-block high risk and flag medium risk for human review
    - [Managed Agents multiagent orchestration](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration): "Escalation" there means consulting a more capable agent or model
    - [Context engineering tools (Claude cookbook)](https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools): a compaction summary may drop specific numbers or exact phrasing
    - [Claude Code workflows](https://code.claude.com/docs/en/workflows): `/deep-research` lists unverifiable claims as unverified rather than refuted
    - [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents): state recovery from `claude-progress.txt` and git history; JSON for structured state
    - [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents): production monitoring for drift; reading sampled transcripts weekly
    - [Language models (mostly) know what they know](https://www.anthropic.com/research/language-models-mostly-know-what-they-know): models struggle with calibration of P(IK) on new tasks
    - [Citations agent prompt (Anthropic cookbook)](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/citations_agent.md): add citations only where sources directly support claims
    - [Human-in-the-loop gate notebook (Anthropic cookbook)](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/managed_agents/CMA_gate_human_in_the_loop.ipynb): escalation calibration matters in both directions
    - [Explore an unfamiliar codebase notebook (Anthropic cookbook)](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/managed_agents/CMA_explore_unfamiliar_codebase.ipynb): the agent writes notes to a NOTES.md file as it explores
    - [claude-code-action test failure analysis example](https://raw.githubusercontent.com/anthropics/claude-code-action/main/examples/test-failure-analysis.yml): routing on a 0.7 confidence threshold, manual review below it
    - [Customer research skill (Anthropic knowledge-work plugins)](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/customer-support/skills/customer-research/SKILL.md): contradictory information across sources as a low-confidence signal
    - [Knowledge synthesis skill (Anthropic knowledge-work plugins)](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/enterprise-search/skills/knowledge-synthesis/SKILL.md): surface conflicts instead of picking one; include dates; keep different time periods separate
    - [MCP specification 2026-07-28: basic protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic/index.md): `resultType` required from this revision, with a missing value from older servers treated as "complete"
    - [Research subagent prompt (Anthropic Claude Cookbooks)](https://github.com/anthropics/claude-cookbooks/blob/main/patterns/agents/prompts/research_subagent.md): handing unreconciled conflicts to the lead researcher
    - [Tool use with Pydantic (Anthropic Claude Cookbooks)](https://github.com/anthropics/claude-cookbooks/blob/main/tool_use/tool_use_with_pydantic.ipynb): validating tool input with Pydantic models
    - [Anthropic Certification Exam Policy (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F34hhd92iyp94a0gtbr15cy5jk%2Fpublic%2F1782870704%2FAnthropic+Certification+Exam+Policy.pdf): confidentiality, prohibited conduct including AI use and unauthorized published questions, sanctions
    - [Certification Terms and Conditions (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F34hhd92iyp94a0gtbr15cy5jk%2Fpublic%2F1782870634%2FCertification+Terms+and+Conditions.pdf): certification term, recertification, confidentiality
    - [Claude Certification Program Exam Registration Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542947%2FClaude+Certification+Program+-+Exam+Registration+Guide.pdf): Partner Academy checkout and Pearson VUE account setup
    - [Claude Certified Architect, Foundations prep courses (Anthropic Partner Academy)](https://anthropic-partners.skilljar.com/page/claude-certified-architect-foundations-prep-courses): the seven official CCAR-F prep courses
    - [Claude certification exam prep courses (Anthropic Partner Academy)](https://anthropic-partners.skilljar.com/page/claude-certification-exam-prep-courses): dedicated prep paths for the other three exams
    - [Computer and network setup (Anthropic Partner Academy)](https://anthropic-partners.skilljar.com/page/computer-and-network-setup): OnVUE delivery, domains to allow, applications that block launch, personal computer or test center options
    - [Pearson VUE: OnVUE requirements for Anthropic](https://www.pearsonvue.com/us/en/anthropic/onvue.html): system requirements, ID rules, desk and room rules, check-in, digital whiteboard, conduct rules
    - [Pearson VUE: OnVUE whiteboard](https://www.pearsonvue.com/us/en/onvue/whiteboard.html): whiteboard wiped on disconnection
    - [Pearson VUE: Anthropic accommodations](https://www.pearsonvue.com/us/en/test-takers/accommodations/pearson_approve.anthropic.html): 10 business days for review, no accommodations added to a scheduled exam
    - [Pearson VUE Candidate Rules Agreement (PDF)](https://www.pearsonvue.com/content/dam/VUE/vue/global/documents/candidate-rules/candidate-rules-agreement.pdf): test center storage, note board, question-number rule
    - [Pearson Professional Center check-in process (PDF)](https://www.pearsonvue.com/content/dam/VUE/vue/en/documents/pearson-professional-center-exam-check-in-process.pdf): arrival time from the confirmation email, note board
    - [Claude Academy courses](https://academy.claude.com/courses): course catalog, lesson counts and durations
    - [Claude Academy: all resources](https://academy.claude.com/all): course lengths and summaries
    - [Claude Academy FAQ](https://academy.claude.com/help/faq): free access with a personal Claude account, durations are estimates
    - [Building with the Claude API (Claude Academy)](https://academy.claude.com/courses/building-with-the-claude-api): sections and lesson counts, prerequisites, RAG and features coverage
    - [Claude Code in action (Claude Academy)](https://academy.claude.com/courses/claude-code-in-action): course objectives
    - [Introduction to Model Context Protocol (Claude Academy)](https://academy.claude.com/courses/introduction-to-model-context-protocol): course summary
    - [Claude Platform 101 (Claude Academy)](https://academy.claude.com/courses/claude-platform-101): builds the agent loop by hand
    - [Claude Code 101 (Claude Academy)](https://academy.claude.com/courses/claude-code-101): context commands, customization lessons, prerequisites
    - [Introduction to subagents (Claude Academy)](https://academy.claude.com/courses/introduction-to-subagents): separate context windows and summaries
    - [Introduction to agent skills (Claude Academy)](https://academy.claude.com/courses/introduction-to-agent-skills): SKILL.md frontmatter and allowed-tools objective
    - [platform.claude.com llms.txt](https://platform.claude.com/llms.txt): Markdown versions of developer docs pages
    - [code.claude.com llms.txt](https://code.claude.com/docs/llms.txt): Agent SDK and Claude Code docs index
    - [Model deprecations (Claude docs)](https://platform.claude.com/docs/en/about-claude/model-deprecations.md): claude-3-haiku-20240307 retirement and replacement
    - [Sonnet 5 migration guide (Claude docs)](https://platform.claude.com/docs/en/models/sonnet-5/migration-guide.md): prefill works on Claude Haiku 4.5
    - [anthropics/claude-cookbooks (GitHub)](https://github.com/anthropics/claude-cookbooks): notebooks, maintenance status, folder contents
    - [anthropics/courses (GitHub)](https://github.com/anthropics/courses): archived September 15, 2026, Claude 3 Haiku
    - [anthropics/prompt-eng-interactive-tutorial (GitHub)](https://github.com/anthropics/prompt-eng-interactive-tutorial): chapters, pinned model ID
    - [anthropics/skills (GitHub)](https://github.com/anthropics/skills): public Agent Skills repository
    - [anthropics/claude-quickstarts (GitHub)](https://github.com/anthropics/claude-quickstarts): starter projects
    - [anthropics/claude-agent-sdk-demos (GitHub)](https://github.com/anthropics/claude-agent-sdk-demos): old "Claude Code SDK Demos" description
    - [WenHao Yu: CCAR-F exam experience](https://yu-wenhao.com/en/blog/ccar-f-exam-experience/): 882 at a Pearson test center, mock progression, timing, elimination rules, test center experience
    - [ahmdhsn-dev: 910/1000 on CCAF (dev.to)](https://dev.to/ahmdhsn-dev/how-i-scored-9101000-on-ccaf-claude-certification-and-open-sourced-my-last-cheat-sheet-5ae1): 910, mock progression, guide reading
    - [OkRelationship3427: passed the CCAR-F with 904/1000 (Reddit)](https://www.reddit.com/r/ClaudeAI/comments/1v5zrru/passed_the_ccarf_with_9041000/): 904, no multiple-select items; UnfortunateHurricane and EnvironmentalPlay440 comments
    - [bills70: CCAR-F thread (Reddit)](https://www.reddit.com/r/ClaudeAI/comments/1vaxsws/claude_certified_architect_foundations_ccarf_how/): shared scenario context, timing; Leading_Will1794 comments
    - [guruduttrai and others: all four certifications thread (Reddit)](https://www.reddit.com/r/ClaudeAI/comments/1v2x1p7/got_all_4_claude_certifications_ccap_ccaf_ccdvf/): 696 fail, Interesting_Ebb_6383's comparison
    - [TFGator1983: CCAR-F thread (Reddit)](https://www.reddit.com/r/ClaudeAI/comments/1wc73tf/ccarf_about_to_take_exam_how_long_are_the/): screen layout, concepts asked different ways
    - [cs135dev: passed CCAR-P (Reddit)](https://www.reddit.com/r/ClaudeCertified/comments/1vo909f/passed_ccarp_exam_harder_than_expected/): 848 on CCAR-F
    - [InRussYouTrust (YouTube)](https://www.youtube.com/watch?v=SYYtM16wXcI): failed then passed, symptom-patching wrong answers
    - [Build With Why AI (YouTube)](https://www.youtube.com/watch?v=F2eUnVQOd6Y): both Architect exams over 900, four days
    - [bluepanda: cleared all four certifications (dev.to)](https://dev.to/bluepanda/i-cleared-all-4-anthropic-claude-certifications-what-you-actually-need-to-know-and-what-to-skip-224m): difficulty ranking, reading volume, check-in time, interface
    - [abinashteja and Upstairs_Muffin_7035: CCA-F thread (Reddit)](https://www.reddit.com/r/ClaudeAI/comments/1ruf70b/just_passed_the_new_claude_certified_architect/): early-adopter pass, 846, mock scores
    - [Sarvesh Talele: step-by-step guide (Big Tech Careers)](https://newsletter.bigtechcareers.com/p/step-by-step-guide-to-achieve-claude-certification): 911, preparation hours, question style
    - [Kishor Kukreja: 893/1000 (Medium)](https://medium.com/@kishorkukreja/i-passed-anthropics-claude-certified-architect-foundations-exam-with-a-score-of-893-1000-2206c27efd6c): 893, background
    - [Suspicious_Low7612: passed with 893 (Reddit)](https://www.reddit.com/r/ClaudeAI/comments/1sgn0cf/passed_anthropics_claude_certified_architect/): 893, practice versus real difficulty
    - [Suraj Khaitan (dev.to)](https://dev.to/suraj_khaitan_f893c243958/i-passed-the-claude-certified-architect-foundations-cca-f-exam-my-journey-lessons-and-98j): first-attempt pass, timing
    - [service_account: CCAF questions (Reddit)](https://www.reddit.com/r/ClaudeAI/comments/1sxac8o/ccaf_questions/): 590 fail after perfect practice scores
    - [John Weidner: passing the Claude Certified Architect exam (Very Good Ventures)](https://verygood.ventures/blog/passing-the-claude-certified-architect-exam/): 738, raw 44 of 60, domain results, practice exam reuse, structure
    - [AK (Tan Aik Keong): CCAF study guide (claudeaimalaysia.com)](https://www.claudeaimalaysia.com/malaysia-claude-ccaf-studyguide.html): 822, raw 48 of 60, mock warning, affiliation
    - [Matthew Purcell: honest review of the Claude certification exams (LinkedIn)](https://www.linkedin.com/pulse/claude-certification-exams-honest-review-matthew-purcell-byo2e): 811, preparation, difficulty, focus-here flag
    - [Matthew Purcell: CCAR-P practice set post (LinkedIn)](https://www.linkedin.com/posts/purcellmatthew_claude-certified-architect-professional-practice-activity-7482176978008342528-204Q): one-line CCAR-F review, study advice
    - [Ihor Sasovets (Medium)](https://medium.com/@ihor.sasovets/claude-certified-architect-foundations-my-learning-journey-and-exam-experience-826d4c08e664): 839, practice 882, timing, preparation
    - [Sofia Lemons (Udacity blog)](https://www.udacity.com/blog/the-claude-certified-architect-exam-explained-by-someone-who-passed-it/): preparation, timing, affiliation
    - [claudecertificationguide.com](https://claudecertificationguide.com/): free CCAR-F track, unsourced study-hour estimate
    - [claudecertifiedarchitects.com](https://www.claudecertifiedarchitects.com/): paid bank with multiple-response items, free sample; [dumps article](https://www.claudecertifiedarchitects.com/blog/ccar-f-exam-dumps/)
    - [CertSafari CCAR-F](https://www.certsafari.com/anthropic/claude-certified-architect-foundations): free bank, format, update date, curated list
    - [Tutorials Dojo CCAR-F study guide](https://tutorialsdojo.com/ccar-f-claude-certified-architect-foundations-study-guide/) and [practice exams](https://portal.tutorialsdojo.com/courses/claude-certified-architect-foundations-ccar-f-practice-exams/): weights, format signals, price, forced tool_choice sample
    - [OpenExamPrep CCA-F](https://open-exam-prep.com/practice/anthropic-cca-f): free access, commissions, study-hour estimate, Domain 5 description
    - [FlashGenius CCAR-F](https://flashgenius.net/sample-tests/ccar-f): pricing, format, prefilling topic
    - [Sundog Education CCAR-F practice exams (Udemy)](https://www.udemy.com/course/anthropic-claude-certified-architect-3-full-practice-exams/): question count, single-answer format, doc links
    - [Jacob Bushong CCAR-F course (Udemy)](https://www.udemy.com/course/certified-claude-architect-masterclass-2026/): length, syllabus notes
    - [freeCodeCamp CCAR-F course (YouTube)](https://www.youtube.com/watch?v=reDRM0tqhNs) and [announcement article](https://www.freecodecamp.org/news/claude-certified-architect-foundations-prep-for-anthropic-s-new-certification-exam/): length, upload date, MCP wording
    - [Peace Of Code playlist (YouTube)](https://www.youtube.com/playlist?list=PLviC8AFqAj5A9MHkRIn2fU5Ac2lEdJxNf) and [sample-question episode](https://www.youtube.com/watch?v=-NymqBcFy6E): video count, sample walkthrough, upload date
    - [daronyondem/claude-architect-exam-guide (GitHub)](https://github.com/daronyondem/claude-architect-exam-guide): no exam questions, license, last push, prefill mention
    - [dnacenta/claude-certified-architect (GitHub)](https://github.com/dnacenta/claude-certified-architect): September 2026 refresh, format wording
    - [Amey-Thakur/CLAUDE-CERTIFICATIONS (GitHub)](https://github.com/Amey-Thakur/CLAUDE-CERTIFICATIONS): coverage, unofficial practice questions, Claude Academy claim
    - [paullarionov/claude-certified-architect (GitHub)](https://github.com/paullarionov/claude-certified-architect): format, scenario count, prefill answer, practice exam advice
    - [hamzafarooq/claude-certified-architect (GitHub)](https://github.com/hamzafarooq/claude-certified-architect): practice exam, invitation to share real questions
    - [dev.to aws-builders CCAR-F overview](https://dev.to/aws-builders/the-claude-certified-architect-exam-5-domains-6-scenarios-and-everything-you-need-to-know-4le3): old single-answer format, dump promotions in comments
    - [March 2026 CCAR-F launch guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F8lsy243ftffjjy1cx9lm3o2bw%2Fpublic%2F1773274827%2FClaude+Certified+Architect+%E2%80%93+Foundations+Certification+Exam+Guide.pdf): earlier single-answer format and no-guessing-penalty statement, unchanged sample stems
    - [Claude 101 (Claude Academy)](https://academy.claude.com/courses/claude-101): lesson count, quiz and duration
    - [AI Fluency: Framework and foundations (Claude Academy)](https://academy.claude.com/courses/ai-fluency-framework-foundations): lesson count, quiz, duration and the 4D framework
    - [Claude with Amazon Bedrock (Claude Academy)](https://academy.claude.com/courses/claude-with-amazon-bedrock): lesson count, quizzes and duration
    - [Claude with Google Cloud's Vertex AI (Claude Academy)](https://academy.claude.com/courses/claude-with-google-cloud-s-vertex-ai): lesson count, quizzes and duration
    - [Claude Certified Architect, Foundations early-access page (archived March 14, 2026)](https://web.archive.org/web/20260314145654/https://anthropic.skilljar.com/claude-certified-architect-foundations-access-request): the "CCA-F badge" wording
