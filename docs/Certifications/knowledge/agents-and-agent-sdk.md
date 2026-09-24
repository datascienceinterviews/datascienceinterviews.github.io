---
title: "Agents and the Claude Agent SDK for the Claude Certifications"
description: Workflows or agents, the five workflow patterns, the agentic loop, multi-agent orchestration and the Claude Agent SDK, taught for CCAR-F, CCDV-F and CCAR-P.
last_reviewed: 2026-09-23
---

# Agents and the Claude Agent SDK

This page teaches how agents built on Claude work: when a fixed workflow beats an autonomous agent, the loop that drives every tool call, how a coordinator delegates to subagents, how the Claude Agent SDK packages Claude Code's loop, tools, hooks, sessions and permissions as a library, where an agent can be hosted, and how third-party agent frameworks compare. It is the depth reference for Domain 1 of the Architect, Foundations exam (weighted 27%), the Agents and Workflows domain of the Developer exam (weighted 14.7%), and the architecture and multi-agent objectives of the Architect, Professional exam. Each section opens with the objectives it serves, including hook, tool, permission and session objectives from other domains, and the [Exam map](#exam-map) at the end collects them.

## Workflows or agents

*Tested in: CCAR-F 1.1, 1.4, 1.6 · CCDV-F Agent Architecture, Agent Patterns and Frameworks · CCAR-P 1.3, 1.5*

Most architecture questions on these exams turn on one decision: should code decide the sequence of steps, or should the model? Anthropic's engineering post [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) notes that "agent" is used for everything from fully autonomous systems to prescriptive implementations that follow predefined workflows. It calls all of these agentic systems and splits them into two kinds: "Workflows are systems where LLMs and tools are orchestrated through predefined code paths." and "Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks."

The post is dated December 19, 2024, and the live page now opens with a note: "Much of the tooling landscape described in this post has changed since December 2024." Two of the exam guides still use the post's terms: CCAR-P objective 1.3 names the same three options (workflow, agentic, augmented LLM), and the CCAR-F technology list includes prompt chaining, the first of the post's workflow patterns. The post's other four pattern names (routing, parallelization, orchestrator-workers, evaluator-optimizer) are not used as pattern names in any of the four guides; [Workflow patterns](#workflow-patterns) maps where the ideas behind them are tested.

### The three options the Professional exam names

CCAR-P objective 1.3 asks you to "Select appropriate architectural patterns (workflow, agentic, augmented LLM)" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). The augmented LLM is the building block underneath both of the others: "The basic building block of agentic systems is an LLM enhanced with augmentations such as retrieval, tools, and memory." ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)).

The table is our summary of [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents); the cost column follows from its statement that agentic systems "trade latency and cost for better task performance".

| Option | Who decides the next step | Choose it when | Cost profile (our summary) |
|---|---|---|---|
| Augmented LLM (one well-built call with retrieval, tools, examples) | No sequence to decide: one call, which may itself search or call a tool | A single call with good retrieval and in-context examples meets the quality bar | Fewest calls, so the lowest latency and cost of the three |
| Workflow (prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer) | Your code | The task is well defined, and you need predictability and consistency | More calls and latency than one call; in prompt chaining the trade buys accuracy by making each call an easier task |
| Agent (the model uses tools in a loop) | The model, from tool results | The number of steps cannot be predicted and no fixed path can be hardcoded | "higher costs, and the potential for compounding errors"; the number of steps, and so the spend, varies per run |

Anthropic's advice is to start at the top of that table and move down only when the evidence says so: "When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all." The same post notes that "For many applications, however, optimizing single LLM calls with retrieval and in-context examples is usually enough." After its pattern catalog it repeats the rule: "you should consider adding complexity only when it demonstrably improves outcomes." ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)).

```text
  simplest                                                        most autonomous
 +------------------+   +------------------+   +--------------+   +------------------+
 | augmented LLM    |-->| workflow         |-->| single agent |-->| multi-agent      |
 | one call with    |   | code fixes the   |   | model picks  |   | coordinator plus |
 | retrieval, tools |   | path             |   | each step    |   | subagents        |
 +------------------+   +------------------+   +--------------+   +------------------+
  more predictable, fewer tokens  <----------------------->  more flexible, more tokens
```

The token cost of moving right is measurable. In Anthropic's multi-agent research data, "agents typically use about 4× more tokens than chat interactions, and multi-agent systems use about 15× more tokens than chats." ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)). The last step, from one agent to several, has its own decision rules in [Multi-agent orchestration](#multi-agent-orchestration).

### How the exams frame the choice

The [Architect, Foundations guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) states the decision in two places:

- **Model-driven vs pre-configured (1.1-K3):** "The distinction between model-driven decision-making (Claude reasons about which tool to call next based on context) and pre-configured decision trees or tool sequences".
- **Fixed vs adaptive decomposition (1.6-K1, 1.6-S1):** "When to use fixed sequential pipelines (prompt chaining) versus dynamic adaptive decomposition based on intermediate findings", resolved as "prompt chaining for predictable multi-aspect reviews, dynamic decomposition for open-ended investigation tasks".

The guide's own example of an open-ended task (1.6-S3 in the [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)) is adding a full set of tests to a legacy codebase, decomposed "by first mapping structure, identifying high-impact areas, then creating a prioritized plan that adapts as dependencies are discovered". 1.6-K3 names the value of such plans: they "generate subtasks based on what is discovered at each step". In our reading, that is an agent's job, because the plan changes with what the agent finds. A per-file review followed by a cross-file integration pass (1.6-K2) is a workflow's job: the guide itself labels it a prompt chaining pattern, and the steps are the same for every pull request.

The [Developer guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) tests the same judgment under Agent Architecture: "the decision criteria for using a workflow versus an agent, the structure of manager/supervisor hierarchies, and the role of subagents in improving task execution." Manager and supervisor hierarchies are covered in [Multi-agent orchestration](#multi-agent-orchestration).

### Decision rules

- **If every run follows the same steps**, build a workflow. In the words of [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents), "workflows offer predictability and consistency for well-defined tasks".
- **If the steps depend on intermediate findings** (research, debugging, exploring an unfamiliar codebase), use an agent. Anthropic's test: "Agents can be used for open-ended problems where it’s difficult or impossible to predict the required number of steps, and where you can’t hardcode a fixed path." ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)).
- **If one step must always happen before another** (identity verification before a refund), enforce the order in code even inside an agent. CCAR-F 1.4-K2 says that when deterministic compliance is required, "prompt instructions alone have a non-zero failure rate" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The mechanics are in [Hooks in the SDK](#hooks-in-the-sdk) and [Permissions and enforcement](#permissions-and-enforcement).
- **If a single call already meets the bar**, stop there. Add a step only when it "demonstrably improves outcomes" ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)).

A real system often mixes both. [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) says its building blocks "aren't prescriptive. They're common patterns that developers can shape and combine to fit different use cases." An illustration of our own: a support agent can choose its tools from context (the model-driven decision-making of CCAR-F 1.1-K3), while code still gates refunds with a programmatic prerequisite of the kind CCAR-F 1.4-S1 describes.

### Where agents earn their cost

From its customer work, Anthropic names two particularly promising applications for agents, customer support and coding, and draws a general lesson from both: "agents add the most value for tasks that require both conversation and action, have clear success criteria, enable feedback loops, and integrate meaningful human oversight." ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)). Those two domains are also the settings of four of the six CCAR-F scenarios: Scenario 1 (Customer Support Resolution Agent), Scenario 2 (Code Generation with Claude Code), Scenario 4 (Developer Productivity with Claude) and Scenario 5 (Claude Code for Continuous Integration).

When you do build an agent, [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) gives three principles: keep the agent's design simple, prioritize transparency by showing the agent's planning steps, and "Carefully craft your agent-computer interface (ACI) through thorough tool documentation and testing." Its rule of thumb is to invest as much effort in agent-computer interfaces as goes into human-computer interfaces. Tool design is taught in [Designing a tool set](tool-use-and-mcp.md#designing-a-tool-set).

### Traps

- **An agent where a workflow fits.** Autonomy "means higher costs, and the potential for compounding errors" ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)). For a fixed, well-defined process, a workflow already gives the predictability and consistency you need.
- **A hard-coded pipeline for open-ended research.** Anthropic's research post says of path-dependent research that "A linear, one-shot pipeline cannot handle these tasks." ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)).
- **A prompt instruction as the guarantee.** CCAR-F sample question 1 rejects an enhanced system prompt (option B) and few-shot examples (option C) and answers with a programmatic prerequisite; its rationale is quoted under [The enforcement ladder](#the-enforcement-ladder), and the gate is built in [A prerequisite gate](#a-prerequisite-gate).
- **A keyword router that replaces the model's tool choice.** Sample question 2 rejects a routing layer that pre-selects the tool from keywords (option C) and answers by expanding each tool's description (option B); its rationale sits with the other routing distractors under [Routing](#routing).
- **Multi-agent before a better prompt.** Anthropic has seen "teams invest months building elaborate multi-agent architectures only to discover that improved prompting on a single agent achieved equivalent results." ([Building multi-agent systems](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)).

!!! note "Associate candidates"

    The [Associate guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) says CCAO-F is "not intended for software developers who build against APIs or design agentic systems". Its related objective is prompt-level decomposition, "Apply task decomposition techniques to structure complex requests" (D1.2), which is taught in [Chain of thought and prompt chaining](prompt-engineering.md#chain-of-thought-and-prompt-chaining).

## Workflow patterns

*Tested in: CCAR-F 1.2, 1.4, 1.6, 4.6 · CCDV-F Agent Architecture, Agent Patterns and Frameworks · CCAR-P 1.3, 1.5*

[Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) names five workflow patterns, each built from LLM calls that the post assumes have the augmented LLM's capabilities, before it reaches autonomous agents. Of the five names, only prompt chaining appears in the exam guides: the [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) lists it among its technologies as "sequential task decomposition into focused passes", and CCAR-F sample question 12 is answered by it. Routing, parallelization, orchestrator-workers and evaluator-optimizer are not pattern names in any of the four guides. CCAR-F uses the word "routing" in other senses (information routing, review routing) and in wrong answer options: sample questions 1, 2 and 3 each offer a routing or classifier option as a wrong answer. The guides describe the other ideas in their own words, such as "parallel subagent execution, iterative refinement loops" in the CCAR-F in-scope topics, so the exam mappings below for those three patterns are ours. For each pattern below: what it is, when Anthropic says to use it, a diagram, and where it shows up on the exams.

### Prompt chaining

"Prompt chaining decomposes a task into a sequence of steps, where each LLM call processes the output of the previous one." You can add programmatic checks on any intermediate step (the post's diagram labels one a "gate") to confirm the process is still on track ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)).

```text
input --> [ call 1 ] --> output 1 --> ( gate: code check ) --pass--> [ call 2 ] --> [ call 3 ] --> result
                                              |
                                             fail --> stop, retry or flag
```

- **When to use it:** "This workflow is ideal for situations where the task can be easily and cleanly decomposed into fixed subtasks. The main goal is to trade off latency for higher accuracy, by making each LLM call an easier task." ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)).
- **Anthropic's examples:** write marketing copy, then translate it into a different language; write an outline, check that it meets certain criteria, then write the document from it.
- **Most common variant:** the Claude prompting guide says the most common chaining pattern is self-correction: "generate a draft → have Claude review it against criteria → have Claude refine based on the review" ([Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)).

**On the exams.** In the [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf), 1.6-K2 is prompt chaining applied to code review: "analyze each file individually, then run a cross-file integration pass". Sample question 12 (a 14-file pull request whose single-pass review gives inconsistent depth and contradictory feedback) is answered by exactly this chain: "analyze each file individually for local issues, then run a separate integration-focused pass examining cross-file data flow." Its rationale names the root cause as "attention dilution when processing many files at once", and rejects switching to a higher-tier model with a larger context window because "larger context windows don't solve attention quality issues." Prompt-level chaining for chat users is taught in [Chain of thought and prompt chaining](prompt-engineering.md#chain-of-thought-and-prompt-chaining).

### Routing

"Routing classifies an input and directs it to a specialized followup task." The post explains the benefit: "Without this workflow, optimizing for one kind of input can hurt performance on other inputs." ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)).

```text
                          +--> [ refund prompt + refund tools ]
input --> [ classifier ] -+--> [ technical support prompt + tools ]
                          +--> [ general questions prompt ]

classifier = an LLM or a traditional classification model
```

- **When to use it:** "Routing works well for complex tasks where there are distinct categories that are better handled separately, and where classification can be handled accurately, either by an LLM or a more traditional classification model/algorithm." ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)).
- **Anthropic's examples:** sending general questions, refund requests and technical support to different downstream prompts and tools; and "Routing easy/common questions to smaller, cost-efficient models like Claude Haiku 4.5 and hard/unusual questions to more capable models like Claude Sonnet 4.5 to optimize for best performance." (both from [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents); the model names are those on the live page as of September 2026).

**On the exams.** A routing or classifier option is a wrong answer in [CCAR-F sample questions](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) 1, 2 and 3: in question 1 because it addresses a different problem, in questions 2 and 3 because it is over-engineered. In sample question 1 a routing classifier that enables only a subset of tools "addresses tool availability rather than tool ordering, which is not the actual problem." In sample question 2 a keyword routing layer is "over-engineered and bypasses the LLM's natural language understanding". In sample question 3 a separate classifier model trained on historical tickets is "over-engineered, requiring labeled data and ML infrastructure when prompt optimization hasn't been tried." Our decision rule from those rationales: choose routing when inputs fall into distinct categories that need different prompts, tools or models; do not choose it to fix ordering, weak tool descriptions or unclear escalation criteria.

### Parallelization

"LLMs can sometimes work simultaneously on a task and have their outputs aggregated programmatically." There are two variants: "Sectioning: Breaking a task into independent subtasks run in parallel." and "Voting: Running the same task multiple times to get diverse outputs." ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)).

```text
Sectioning (different subtasks)            Voting (same task, several runs)

        +--> [ subtask A ] --+                     +--> [ run 1 ] --+
input --+--> [ subtask B ] --+--> aggregate  input --+--> [ run 2 ] --+--> vote / threshold
        +--> [ subtask C ] --+                     +--> [ run 3 ] --+
```

- **When to use it:** "Parallelization is effective when the divided subtasks can be parallelized for speed, or when multiple perspectives or attempts are needed for higher confidence results." ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)).
- **Sectioning examples:** one instance answers the user while another screens for inappropriate content ("This tends to perform better than having the same LLM call handle both guardrails and the core response."); automated evals where each call grades a different aspect.
- **Voting examples:** several different prompts review a piece of code for vulnerabilities and flag the code if they find a problem; several prompts judge content with different vote thresholds "to balance false positives and negatives."

**On the exams.** In our mapping, CCAR-F 1.4-S2 in the [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) is sectioning inside an agent: "Decomposing multi-concern customer requests into distinct items, then investigating each in parallel using shared context before synthesizing a unified resolution". Voting has a trap. In sample question 12, running three independent review passes and flagging only issues that appear in at least two is rejected because it "would actually suppress detection of real bugs by requiring consensus on issues that may only be caught intermittently." Our reading, following the post's point that vote thresholds "balance false positives and negatives": the aggregation rule you choose (any run flags, majority, unanimous) sets that balance, and a consensus rule trades missed findings for fewer false alarms. Independent review instances are taught in [Multi-instance and multi-pass review](prompt-engineering.md#multi-instance-and-multi-pass-review).

### Orchestrator-workers

"In the orchestrator-workers workflow, a central LLM dynamically breaks down tasks, delegates them to worker LLMs, and synthesizes their results." The difference from sectioning is who defines the subtasks. In parallelization your code splits the work; here the "subtasks aren't pre-defined, but determined by the orchestrator based on the specific input." ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)). The post calls the two "topographically similar", so read the stem for who decides the split.

```text
                         +--> [ worker 1: task chosen at runtime ] --+
input --> [ orchestrator ]--> [ worker 2: task chosen at runtime ] --+--> [ orchestrator synthesizes ] --> result
            (plans)      +--> [ worker N ]                         --+
```

- **When to use it:** "This workflow is well-suited for complex tasks where you can’t predict the subtasks needed (in coding, for example, the number of files that need to be changed and the nature of the change in each file likely depend on the task)." ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)).
- **Anthropic's examples:** coding products that make complex changes to multiple files each time; search tasks that gather and analyze information from multiple sources.

**On the exams.** In our mapping, this is the pattern behind the coordinator questions: the CCAR-F hub-and-spoke coordinator (1.2-K1), the "manager/supervisor hierarchies" of the [CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) (Agent Architecture), and Anthropic's own Research system, which "uses a multi-agent architecture with an orchestrator-worker pattern" ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)). Neither Building effective agents nor the research-system post uses the words "manager" or "supervisor", so expect the exam to use the guide's phrase. When each worker is itself an agent with tools, the design is a multi-agent system; the names each source uses are compared, and the design is taught, in [Multi-agent orchestration](#multi-agent-orchestration).

### Evaluator-optimizer

"In the evaluator-optimizer workflow, one LLM call generates a response while another provides evaluation and feedback in a loop." ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)).

```text
input --> [ generator ] --> draft --> [ evaluator ] --> meets criteria? --yes--> result
               ^                                             |
               +------------- feedback ----------------------+ no (bounded number of rounds)
```

- **When to use it:** "This workflow is particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value." The post adds a two-part test: "The two signs of good fit are, first, that LLM responses can be demonstrably improved when a human articulates their feedback; and second, that the LLM can provide such feedback." ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)).
- **Anthropic's examples:** literary translation with an evaluator that critiques nuance; complex search where "the evaluator decides whether further searches are warranted."

**On the exams.** In our mapping, CCAR-F 1.2-S3 is evaluator-optimizer run by a coordinator: the coordinator checks synthesis output for gaps and re-delegates to subagents until coverage is sufficient. The objective's wording and the matching loop in Anthropic's Research system are in [What the coordinator decides](#what-the-coordinator-decides).

Three details make the loop reliable:

- **The evaluator needs criteria it can check.** The Agent SDK blog says "The best form of feedback is providing clearly defined rules for an output, then explaining which rules failed and why." It ranks an LLM judging output against fuzzy rules as the weaker option: it "can have heavy latency tradeoffs", and the post reserves it "for applications where any boost in performance is worth the cost" ([Building agents with the Claude Agent SDK](https://claude.com/blog/building-agents-with-the-claude-agent-sdk)).
- **The evaluator should not share the generator's reasoning.** CCAR-F 4.6-K1 ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)) names the risk: "a model retains reasoning context from generation, making it less likely to question its own decisions in the same session". Claude Managed Agents (a beta API as of September 2026) builds this in for outcomes: "The grader uses a separate context window to avoid being influenced by the main agent's implementation choices." ([Define outcomes](https://platform.claude.com/docs/en/managed-agents/define-outcomes)).
- **The loop needs a bound and an exit.** A Managed Agents outcome (the `user.define_outcome` event) takes an optional `max_iterations`, documented in the code samples as "optional; default 3, max 20". When the cap is hit, the outcome evaluation end event reports `max_iterations_reached`, one final acknowledgment turn follows, the session goes idle, and no further evaluation runs ([Define outcomes](https://platform.claude.com/docs/en/managed-agents/define-outcomes)). Anthropic's research lead-agent prompt gives the model its own exit: once further research has diminishing returns, "STOP FURTHER RESEARCH and do not create any new subagents." ([research_lead_agent.md](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_lead_agent.md)).

A verifier that passes work too early defeats the loop. Anthropic calls this the early victory problem: "The most significant failure mode for verification subagents is marking outputs as passing without thorough testing. The verifier runs one or two tests, observes them pass, and declares success." The post's mitigations are concrete criteria (it contrasts "Run the full test suite and report all failures" with "make sure it works."), checks that cover multiple scenarios and edge cases, negative tests that confirm bad inputs fail, and explicit instructions: the post calls the instruction "You MUST run the complete test suite before marking as passed" essential ([Building multi-agent systems](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)).

### Choosing a pattern

| Pattern | Subtasks known before the run? | Who picks the next step | Cues in a scenario |
|---|---|---|---|
| Prompt chaining | Yes, fixed sequence | Code | A fixed sequence of steps; per-file passes then a cross-file pass; accuracy matters more than latency |
| Routing | Yes, fixed categories | Classifier, then code | Distinct input categories; easy requests could go to a cheaper model |
| Parallelization: sectioning | Yes, independent parts | Code | Independent parts; a request with several concerns; screening that runs alongside the answer |
| Parallelization: voting | Same task repeated | Code (aggregation rule) | Several attempts for higher confidence; a consensus or vote threshold |
| Orchestrator-workers | No, chosen per input | Orchestrator LLM | A coordinator or hub-and-spoke design; subtasks that cannot be predicted in advance |
| Evaluator-optimizer | Yes, generate then evaluate | Evaluator decides whether to loop | Clear evaluation criteria; iterative refinement until a quality or coverage bar is met |
| Agent | No, not even the number of steps | The model, from tool results | Open-ended work; a plan that adapts to what is discovered; no path that can be hardcoded |

!!! tip "How to recognize the right answer"

    Ask two questions of the scenario (our decision rule, built from 1.6-K1, 1.6-S1 and 1.4-K2). First, are the steps known before the run starts? If yes, the answer is a workflow pattern, and the stem's wording (sequence, categories, independent parts, a quality bar to iterate toward) tells you which one. If the subtasks depend on intermediate findings, the answer is dynamic decomposition by an orchestrator or an agent. Second, does something in the flow have to happen every time? If yes, that part belongs in code whatever pattern surrounds it, because "prompt instructions alone have a non-zero failure rate" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf), 1.4-K2).

## The agentic loop

*Tested in: CCAR-F 1.1 and preparation exercise 1, step 2 · CCDV-F Agent Construction with Claude, Agent Patterns and Frameworks, Tool Implementation*

An agent is, in Anthropic's words, "typically just LLMs using tools based on environmental feedback in a loop" ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)). The loop is also a tool-use contract: "The model never executes anything on its own. It emits a structured request, your code (or Anthropic's servers) runs the operation, and the result flows back into the conversation." ([How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works)). CCAR-F 1.1-K1 lists the lifecycle you must know: "sending requests to Claude, inspecting stop_reason ("tool_use" vs "end_turn"), executing requested tools, and returning results for the next iteration" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

### The loop in five steps

The tool-use docs call the canonical shape of the client-tool loop "a `while` loop keyed on `stop_reason`" ([How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works)):

1. Send a request with your `tools` array and the user message.
2. Claude responds with `stop_reason: "tool_use"` and one or more `tool_use` blocks.
3. Execute each tool and format the outputs as `tool_result` blocks.
4. Send a new request containing the original messages, the assistant's response, and a user message with the `tool_result` blocks.
5. Repeat from step 2 while `stop_reason` is `"tool_use"`. The docs add that the loop exits on any other stop reason (`"end_turn"`, `"max_tokens"`, `"stop_sequence"` or `"refusal"`).

This loop is for client tools, the ones your code executes: "The model can't run your code, so every tool call is a round trip: the model asks, you execute, you report back, the model continues." ([How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works)). Server tools run in a separate loop on Anthropic's infrastructure. The same page names four of them (`web_search`, `web_fetch`, `code_execution` and `tool_search`) and says of them: "You never construct a `tool_result` block for these tools." Those four are not the full set: the [tool reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference) also lists the advisor tool (`advisor_20260301`) and the MCP connector (`mcp_toolset`) as server tools. That split is the "client-side vs. server-side tools" pattern in the CCDV-F Tool Implementation skill ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)).

```text
            +-------------------------------------------------------------+
            |                                                             |
            v                                                             |
 messages --> [ Messages API call ] --> stop_reason?                      |
                                          |                               |
                     "tool_use" ----------+--> run each tool_use block    |
                                          |    append assistant turn      |
                                          |    append user turn of        |
                                          |    tool_result blocks --------+
                                          |
      any other value ("end_turn", ...) --+--> exit the loop, then branch on the value
```

The docs' manual loop, reproduced from [Stop reasons and fallback](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons). The docs do not define `execute_tools` / `executeTools`; it stands for your own function that runs the tools and returns one `tool_result` per `tool_use` block.

=== "Python"

    ```python
    def complete_tool_workflow(client, user_query, tools):
        messages = [{"role": "user", "content": user_query}]

        while True:
            response = client.messages.create(
                model="claude-opus-5-5", max_tokens=1024, messages=messages, tools=tools
            )

            if response.stop_reason == "tool_use":
                # Execute tools and continue
                tool_results = execute_tools(response.content)
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                # Final response
                return response
    ```

=== "TypeScript"

    ```typescript
    async function completeToolWorkflow(
      client: Anthropic,
      userQuery: string,
      tools: Anthropic.ToolUnion[]
    ): Promise<Anthropic.Message> {
      const messages: Anthropic.MessageParam[] = [{ role: "user", content: userQuery }];

      while (true) {
        const response = await client.messages.create({
          model: "claude-opus-5-5",
          max_tokens: 1024,
          messages,
          tools
        });

        if (response.stop_reason === "tool_use") {
          // Execute tools and continue
          const toolResults = executeTools(response.content);
          messages.push({ role: "assistant", content: response.content });
          messages.push({ role: "user", content: toolResults });
        } else {
          // Final response
          return response;
        }
      }
    }
    ```

Notice what the loop keys on. It continues on exactly one value, `"tool_use"`, and returns on everything else, so a truncated or refused response ends the loop instead of being treated as a request for more tools. The caller then branches on the returned `stop_reason`. Two cases need extra code. If your requests enable server tools, a `pause_turn` response means the server-side loop reached its iteration limit before the work was finished, and "Your application should handle `pause_turn` in any agent loop that uses server tools." ([Stop reasons and fallback](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)). And if `max_tokens` cut off a `tool_use` block, the same page says "you'll need to retry the request with a higher `max_tokens` value to get the full tool use." It also recommends the SDK tool runner for most use cases because it handles tool execution "with much less code".

### Returning tool results so the model can reason

CCAR-F 1.1-K2 tests "How tool results are appended to conversation history so the model can reason about the next action", and 1.1-S2 the matching skill, "Adding tool results to conversation context between iterations so the model can incorporate new information into its reasoning" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The rules that make the next request valid:

- Send the whole history each time: original messages, the assistant turn that asked for tools, then a user turn with the results.
- Return "one `tool_result` for each `tool_use` block, all together in the next user message." ([Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use)).
- "Tool result blocks must immediately follow their corresponding tool use blocks in the message history." Inside that user message the `tool_result` blocks come first and any text comes after them ([Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)). Text after the tool results is usually allowed but can cause empty responses, as explained under the anti-patterns below. If the assistant turn also called a server tool that has no result block yet, the message must contain only `tool_result` blocks: text ends the turn early, and for a server tool Claude called directly the request fails with a 400 error. Replies to pending programmatic tool calls cannot carry any text either ([Programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)).
- Match each result to its call with `tool_use_id`, and set `is_error: true` when the tool failed. Anthropic's research team found that "letting the agent know when a tool is failing and letting it adapt works surprisingly well." ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)).

Tool results are the agent's ground truth. [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) says "During execution, it's crucial for the agents to gain “ground truth” from the environment at each step (such as tool call results or code execution) to assess its progress." Result formatting, error shapes and parallel calls are taught in [Returning tool results and errors](tool-use-and-mcp.md#returning-tool-results-and-errors) and [Parallel tool calls](tool-use-and-mcp.md#parallel-tool-calls); keeping untrusted output inside `tool_result` blocks is covered in [Prompt injection](security-and-governance.md#prompt-injection).

### What each stop reason means for the loop

A client-tool loop continues only on `"tool_use"`; the docs today (as of September 2026) exit on any other value, and a loop that uses server tools sends a `pause_turn` response back so Claude can finish. The exam guide frames the same control in two values: CCAR-F 1.1-S1 continues on `"tool_use"` and terminates on `"end_turn"`, and the technology appendix lists only those two `stop_reason` values ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)), so answer exam items in the guide's terms. The meaning and handling of all seven values, with the full comparison of guide and docs, are taught in [Stop reasons and the agent loop](claude-api.md#stop-reasons-and-the-agent-loop).

### The three anti-patterns CCAR-F names

CCAR-F 1.1-S3 lists "parsing natural language signals to determine loop termination, setting arbitrary iteration caps as the primary stopping mechanism, or checking for assistant text content as a completion indicator" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

| Anti-pattern | Why it fails | Do instead |
|---|---|---|
| Reading the reply's wording (a sentence saying the work is finished) to decide termination | A decision hidden in prose is fragile. The tool-use docs: "if you're writing a regex to extract a decision from model output, that decision should have been a tool call." ([How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works)) | Continue on `"tool_use"`; terminate on `"end_turn"` |
| An arbitrary iteration cap as the primary stop | A cap counts steps, not completion. In the Agent SDK docs' own example, `max_turns=2` "would have stopped before the edit step" of the four-turn run shown below ([How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop)) | Loop on `stop_reason`; keep a cap or budget only as a safety net |
| Treating the presence of assistant text as proof that the task is complete | A response can carry text and tool calls together, and an `end_turn` response can be empty | Read `stop_reason`, not the presence of text |

The Agent SDK docs describe the model's turn directly: "It may respond with text, request one or more tool calls, or both." ([How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop)). An empty `end_turn` response of two to three tokens has a documented cause: "Adding text blocks immediately after tool results (Claude learns to expect the user to always insert text after tool results, so it ends its turn to follow the pattern)" ([Stop reasons and fallback](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons)).

!!! note "Caps are a backstop, not the stop signal"

    Anthropic's own sources still treat limits as normal practice. [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) says "The task often terminates upon completion, but it’s also common to include stopping conditions (such as a maximum number of iterations) to maintain control.", and the Agent SDK docs say "Setting a budget is a good default for production agents." ([How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop)). The exam objects to a cap as the *primary* mechanism. Terminate on `stop_reason`; let a cap or budget catch runaway sessions.

### Model-driven decisions inside the loop

In the loop your code executes; the model decides. That is the "model-driven decision-making (Claude reasons about which tool to call next based on context)" of CCAR-F 1.1-K3 ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)), as opposed to a pre-configured sequence. The Agent SDK docs walk through a run for the prompt "Fix the failing tests in auth.ts" ([How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop)):

1. Claude calls `Bash` to run `npm test`, and the output shows three failures.
2. Claude calls `Read` on `auth.ts` and `auth.test.ts`.
3. Claude calls `Edit` to fix `auth.ts`, then `Bash` to re-run `npm test`; all three tests pass.
4. Claude returns a text-only response with no tool calls, and the loop ends.

The docs count that as four turns: three with tool calls and one final text-only response.

No code in that run told Claude to read the files after the tests failed. Each step came from the previous tool result. Claude Code's glossary names the pattern "gather context, take action, verify results, and repeat until done" ([Glossary](https://code.claude.com/docs/en/glossary)), and the Agent SDK blog describes the same cycle as "gather context -> take action -> verify work -> repeat." ([Building agents with the Claude Agent SDK](https://claude.com/blog/building-agents-with-the-claude-agent-sdk)).

```text
     +------------------+      +---------------+      +---------------+
 --> |  gather context  | ---> |  take action  | ---> |  verify work  | --+
     |  (search, read)  |      | (edit, run)   |      | (tests, lint) |   |
     +------------------+      +---------------+      +---------------+   |
              ^                                                           |
              +----------------------- repeat until done -----------------+
```

### Who runs the loop

CCDV-F Agent Construction with Claude covers "custom agent loops and harnesses" ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). Claude Code's glossary defines the agentic harness as "The tools, context management, and execution environment that turn a language model into a capable coding agent." and adds: "Claude Code is the harness; Claude is the model inside it." ([Glossary](https://code.claude.com/docs/en/glossary)). The first choice is who writes and hosts that harness.

If you build on a client SDK, there are two ways to drive the loop:

| Option | Who runs the loop | Choose it when |
|---|---|---|
| Manual loop | Your code, as above | You need human-in-the-loop approval, custom logging or conditional execution |
| Tool runner (beta as of September 2026) | The client SDK helper, which "handles the agentic loop, error wrapping, and type safety so you don't have to". "The runner loops until Claude returns a message without a tool use, or until it reaches `max_iterations` if you set it." ([Tool runner (SDK)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner)) | Most other cases: the stop-reasons page recommends it "For most use cases" |

Above the client SDKs, someone else's harness runs the loop. The Agent SDK runs Claude Code's loop and tools inside a process you operate (taught in [The Claude Agent SDK](#the-claude-agent-sdk)). Claude Managed Agents is "A hosted agent harness that runs the agent loop" ([Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)), and instead of your code deciding when the loop is done, the session "emits `session.status_idle` when the agent has nothing more to do." ([Managed Agents migration](https://platform.claude.com/docs/en/managed-agents/migration)). [Deployment models](#deployment-models) compares these options side by side, together with running the Claude Code CLI as a subprocess.

### The same loop in the Agent SDK

The Agent SDK runs this loop for you and reports it as a stream of messages. Three facts from [How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) connect it to the raw loop:

- **A turn is one round trip.** "A turn is one round trip inside the loop: Claude produces output that includes tool calls, the SDK executes those tools, and the results feed back to Claude automatically." This happens without yielding control back to your code, and the loop ends when Claude "produces a response with no tool calls."
- **`max_turns` / `maxTurns` counts tool-use turns only.** Its default is no limit, and so is the default for `max_budget_usd` / `maxBudgetUsd`. Hitting either limit ends the run with the matching error subtype.
- **The end is a `ResultMessage`.** Its `subtype` says whether the run succeeded or stopped at a limit, and it carries the final turn's `stop_reason`; the subtypes and how to read them are under [`query()`: one task, one loop](#query-one-task-one-loop).

!!! note "Two meanings of a turn"

    The Agent SDK docs call each cycle of the loop a turn (the example above has four, the last one text-only), and `max_turns` counts only the turns that include tool use. Claude Code's [glossary](https://code.claude.com/docs/en/glossary) uses the word differently: "One complete response from Claude within a session", which begins when you send a message and ends when Claude finishes responding, with any number of tool calls in between. When a question says `max_turns`, use the SDK meaning.

The SDK's message types, result handling and options are in [The Claude Agent SDK](#the-claude-agent-sdk).

## Multi-agent orchestration

*Tested in: CCAR-F 1.2, 1.3, 1.6, 5.6 · CCDV-F Agent Architecture, Agent Patterns and Frameworks, Context Engineering · CCAR-P 1.4, 3.1*

"A multi-agent system is an architecture where multiple LLM instances run with separate conversation contexts, coordinated through code." ([Building multi-agent systems](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)). The pattern the exams test is the hierarchical one, which the same post describes as "a hierarchical model where a lead agent spawns and manages specialized subagents for specific subtasks." Each source names it differently, so learn the synonyms:

| Source | Name for the lead | Name for the pattern |
|---|---|---|
| [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) | Coordinator | "Hub-and-spoke architecture where a coordinator agent manages all inter-subagent communication, error handling, and information routing" (1.2-K1) |
| [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) | Manager or supervisor | "manager/supervisor hierarchies" (Agent Architecture) |
| [Anthropic Research system](https://www.anthropic.com/engineering/multi-agent-research-system) | Lead agent (LeadResearcher) | "orchestrator-worker pattern" |
| Claude Code and the Agent SDK | Main agent or parent | Subagents, spawned through the `Agent` tool (formerly `Task`) |
| Claude Managed Agents | Coordinator | A coordinator with a declared roster of agents |

### Hub and spoke

```text
                         +----------------------------------+
      user query ------> |           coordinator            | ------> final answer
                         |  decompose, choose subagents,    |
                         |  aggregate, handle errors        |
                         +----------------------------------+
                           |    ^         |    ^         |    ^
                  prompt   |    | result  |    |         |    |
                           v    |         v    |         v    |
                      [ web search ]  [ doc analysis ]  [ synthesis ]
                       own context     own context       own context
                       own tools       own tools         own tools

        no subagent-to-subagent links: everything passes through the hub
```

CCAR-F asks for "Routing all subagent communication through the coordinator for observability, consistent error handling, and controlled information flow" (1.2-S4, [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Subagents in Claude Code and the Agent SDK follow this shape by default: "Subagents report results back to the conversation that spawned them" ([Run agents in parallel](https://code.claude.com/docs/en/agents)); the return path is covered under [Passing context to a subagent](#passing-context-to-a-subagent).

Two current-product details sit outside the exam's model (as of September 2026). The agent teams comparison table notes that subagents Claude named when it spawned them can also message each other. And agent teams themselves are peer-to-peer: "Teammates message each other directly", while the main agent of a subagent setup "manages all work". Agent teams are "experimental and disabled by default", and Claude does not spawn teammates in non-interactive `-p` mode, which includes Agent SDK sessions ([Orchestrate teams of Claude Code sessions](https://code.claude.com/docs/en/agent-teams)). On the exam, an answer that has subagents talk to each other directly breaks the hub-and-spoke rule.

### When a multi-agent design pays off

Anthropic's January 2026 guidance names "three situations where multiple agents consistently outperform a single agent: when context pollution degrades performance, when tasks can run in parallel, and when specialization improves tool selection or task focus. Outside these situations, the coordination costs typically exceed the benefits." ([Building multi-agent systems](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)). The research post agrees: "multi-agent systems excel at valuable tasks that involve heavy parallelization, information that exceeds single context windows, and interfacing with numerous complex tools", while "some domains that require all agents to share the same context or involve many dependencies between agents are not a good fit for multi-agent systems today." ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)).

| Signal in the scenario | Multi-agent? | Why |
|---|---|---|
| One subtask fills the context with detail the rest of the task does not need | Yes: isolate it in a subagent | Context pollution; see [Subagents as context isolation](context-engineering.md#subagents-as-context-isolation) |
| Independent research paths or sources | Yes: parallel subagents | Covers more ground than one context window |
| Many tools across unrelated domains | Consider it, after trying the Tool Search Tool | "An agent with too many tools (often 20+) struggles to select the appropriate one." Tool search "can reduce token usage by up to 85% while improving tool selection accuracy." ([source](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)) |
| Steps share one evolving context (most coding work) | No: one agent | Coordination overhead and "lost context at each handoff" ([source](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)) |
| The goal is lower latency | Not by itself | "The primary benefit of parallelization is thoroughness, not speed." Multi-agent systems "often take longer overall than single-agent systems because of the sheer increase in total computation" ([source](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)) |

State the baseline whenever you quote a cost multiplier. Against a single agent doing the same task: "multi-agent implementations typically use 3-10x more tokens than single-agent approaches for equivalent tasks." Against chat: "multi-agent systems use about 15× more tokens than chats." The payoff can be large when the task fits: in Anthropic's research eval, "a multi-agent system with Claude Opus 4 as the lead agent and Claude Sonnet 4 subagents outperformed single-agent Claude Opus 4 by 90.2%". The same post explains why: on BrowseComp "token usage by itself explains 80% of the variance", and "Multi-agent systems work mainly because they help spend enough tokens to solve the problem." ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system); [Building multi-agent systems](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)).

### Decompose by context, not by job title

"The key insight is to adopt a context-centric view rather than a problem-centric view when decomposing work." ([Building multi-agent systems](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)).

- **Problem-centric (often counterproductive):** "Dividing by type of work (one agent writes features, another writes tests, a third reviews code) creates constant coordination overhead." In one experiment with planner, implementer, tester and reviewer agents, "the subagents spent more tokens on coordination than on actual work."
- **Context-centric (usually effective):** "an agent handling a feature should also handle its tests, because it already possesses the necessary context. Work should only be split when context can be truly isolated."
- **Boundaries that work:** independent research paths ("market trends in Asia" versus "market trends in Europe"), separate components with clean interfaces, and blackbox verification, where a verifier only runs tests and reports results.

The coordinator's decomposition is also where coverage is won or lost. CCAR-F 1.2-K4 names the "Risks of overly narrow task decomposition by the coordinator, leading to incomplete coverage of broad research topics", and 1.2-S2 asks for "Partitioning research scope across subagents to minimize duplication (e.g., assigning distinct subtopics or source types to each agent)" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Anthropic's research lead-agent prompt puts the same rule in one line: "Define extremely clear, crisp, and understandable boundaries between sub-topics to prevent overlap." ([research_lead_agent.md](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_lead_agent.md)).

### What the coordinator decides

CCAR-F 1.2-K3 lists the coordinator's jobs: "task decomposition, delegation, result aggregation, and deciding which subagents to invoke based on query complexity". 1.2-S1 adds that it should "dynamically select which subagents to invoke rather than always routing through the full pipeline" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

Anthropic learned the cost of getting this wrong: "Early agents made errors like spawning 50 subagents for simple queries, scouring the web endlessly for nonexistent sources, and distracting each other with excessive updates." The post's main response was prompting: "Since each agent is steered by a prompt, prompt engineering was our primary lever for improving these behaviors." For overinvestment in simple queries, which it calls "a common failure mode in our early versions", it embedded effort-scaling rules in the prompts: "Simple fact-finding requires just 1 agent with 3-10 tool calls, direct comparisons might need 2-4 subagents with 10-15 calls each, and complex research might use more than 10 subagents with clearly divided responsibilities." ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)).

When results come back, the coordinator judges whether coverage is sufficient and re-delegates if it is not. CCAR-F 1.2-S3 describes the loop: the coordinator "evaluates synthesis output for gaps, re-delegates to search and analysis subagents with targeted queries, and re-invokes synthesis until coverage is sufficient". Anthropic's research system works the same way: "The LeadResearcher synthesizes these results and decides whether more research is needed", and if so it creates more subagents or refines its strategy ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)). A plan that grows from what each round finds is what CCAR-F 1.6-K3 calls "adaptive investigation plans that generate subtasks based on what is discovered at each step". In our mapping, this is the evaluator-optimizer pattern with the coordinator as the evaluator (the guide does not use that name here); the pattern is taught in [Evaluator-optimizer](#evaluator-optimizer).

### Passing context to a subagent

Unless it is a fork (below), a subagent starts with none of the coordinator's conversation. The [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) states it twice: subagents "do not inherit the coordinator's conversation history automatically" (1.2-K2), and "subagents do not automatically inherit parent context or share memory between invocations" (1.3-K2). The Agent SDK docs make the mechanism explicit: "The only content you pass from parent to subagent is the Agent tool's prompt string, so include any file paths, error messages, or decisions the subagent needs directly in that prompt." ([Subagents in the SDK](https://code.claude.com/docs/en/agent-sdk/subagents)).

Besides that string, a non-fork subagent gets only what its own definition and the session supply (its system prompt, tool definitions, project CLAUDE.md, preloaded skills), never the parent's system prompt; the SDK controls are listed in [What crosses the boundary](#what-crosses-the-boundary).

The exception is a fork: "A fork is a subagent that inherits the entire conversation so far instead of starting fresh." As of September 2026, fork mode is on by default in interactive Claude Code (v2.1.232 and later) but off in `-p` mode and in the Agent SDK ([Create custom subagents](https://code.claude.com/docs/en/sub-agents)). Configuration details are in [Subagents in the SDK](#subagents-in-the-sdk).

What to put in the prompt string:

- **The four parts of a good delegation.** "Each subagent needs an objective, an output format, guidance on the tools and sources to use, and clear task boundaries." Without them, "agents duplicate work, leave gaps, or fail to find necessary information." ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)).
- **Complete findings, not a pointer to them.** [CCAR-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) 1.3-S1: "Including complete findings from prior agents directly in the subagent's prompt (e.g., passing web search results and document analysis outputs to the synthesis subagent)".
- **Content separated from metadata.** [CCAR-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) 1.3-S2: "Using structured data formats to separate content from metadata (source URLs, document names, page numbers) when passing context between agents to preserve attribution".
- **Goals and quality criteria, not a script.** [CCAR-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) 1.3-S4: "Designing coordinator prompts that specify research goals and quality criteria rather than step-by-step procedural instructions, to enable subagent adaptability". Anthropic's research team took the same line: "Our prompting strategy focuses on instilling good heuristics rather than rigid rules." ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system))

[CCAR-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) Exercise 4 says "each finding should include a claim, evidence excerpt, source URL/document name, and publication date." One finding in that shape (an illustrative record, not an API schema):

```json
{
  "claim": "<one factual statement>",
  "evidence_excerpt": "<verbatim passage that supports it>",
  "source_url": "<URL or document name>",
  "publication_date": "<date the source gives>"
}
```

The return path matters as much. A subagent "does not see the parent's turns, and only its final response returns to the parent as a tool result", and "The parent receives the subagent's final message as the Agent tool result, but may summarize it in its own response." ([How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop); [Subagents in the SDK](https://code.claude.com/docs/en/agent-sdk/subagents)). A subagent may use tens of thousands of tokens yet return "a condensed, distilled summary of its work (often 1,000-2,000 tokens)" ([Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)). For large outputs, "Subagents call tools to store their work in external systems, then pass lightweight references back to the coordinator." ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)). Handoff design and claim-source mapping are taught in [Multi-agent handoffs](context-engineering.md#multi-agent-handoffs) and [Provenance and uncertainty in synthesis](context-engineering.md#provenance-and-uncertainty-in-synthesis).

### Spawning subagents in parallel

CCAR-F 1.3-S3: "Spawning parallel subagents by emitting multiple Task tool calls in a single coordinator response rather than across separate turns" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Calls emitted together can run together, so "independent subtasks finish in the time of the slowest one rather than the sum of all of them." ([Subagents in the SDK](https://code.claude.com/docs/en/agent-sdk/subagents)). The research post shows the cost of the alternative: its lead agents "execute subagents synchronously, waiting for each set of subagents to complete before proceeding" ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)), so with a coordinator that waits like that, every extra turn used for spawning adds another full wait (our reasoning). Current SDK subagents run in the background by default (see the warning below), but the exam's answer is still one coordinator response carrying several spawn calls.

The shape of one coordinator response that fans out to two subagents (an illustrative message with made-up IDs and a hypothetical `web-researcher` subagent; the block structure is the API's `tool_use` block, and the input field names come from the Agent tool's input schema: `description` is "A short (3-5 word) description of the task", `prompt` is the task itself, `subagent_type` names a defined subagent):

```json
{
  "role": "assistant",
  "content": [
    {"type": "text", "text": "I'll research both regions in parallel."},
    {"type": "tool_use", "id": "toolu_01", "name": "Agent",
     "input": {"description": "Research Asia market trends",
               "prompt": "Objective: ... Output format: ... Sources to use: ... Out of scope: Europe.",
               "subagent_type": "web-researcher"}},
    {"type": "tool_use", "id": "toolu_02", "name": "Agent",
     "input": {"description": "Research Europe market trends",
               "prompt": "Objective: ... Output format: ... Sources to use: ... Out of scope: Asia.",
               "subagent_type": "web-researcher"}}
  ]
}
```

Anthropic's evidence for doing this: "(1) the lead agent spins up 3-5 subagents in parallel rather than serially; (2) the subagents use 3+ tools in parallel. These changes cut research time by up to 90% for complex queries" ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)). The baseline is the earlier version of the same system, which spawned subagents serially and searched sequentially ("Our early agents executed sequential searches, which was painfully slow."), not a single agent; against a single agent, multi-agent systems "often take longer overall" ([Building multi-agent systems](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them)). Its published lead-agent prompt makes it a rule: "You MUST use parallel tool calls for creating multiple subagents (typically running 3 subagents at the same time) at the start of the research, unless it is a straightforward query." ([research_lead_agent.md](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_lead_agent.md)).

Fan-out needs limits. By default the Agent SDK allows three layers of subagents below the main agent and 20 running at once, and a budget cap counts subagent spend; the settings, version requirements and error messages are in [Limits on depth, concurrency and spend](#limits-on-depth-concurrency-and-spend).

!!! warning "Exam guide vs current docs"

    **Tool name.** The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) (Version 1.0, effective July 2026) calls the spawning tool "The Task tool" and names "the requirement that allowedTools must include "Task" for a coordinator to invoke subagents" (1.3-K1). Current docs: "In version 2.1.63, the Task tool was renamed to Agent. Existing `Task(...)` references in settings and agent definitions still work as aliases." The tool shows as `"Agent"` in `tool_use` blocks but as `"Task"` in the `system:init` tools list, so detection code should match both names ([Create custom subagents](https://code.claude.com/docs/en/sub-agents); [Subagents in the SDK](https://code.claude.com/docs/en/agent-sdk/subagents)). Expect "Task" on the exam; write `Agent` in new code.

    **Listing it in allowed tools.** The SDK subagent examples put `"Agent"` in `allowed_tools`, and the SDK feature table maps delegation to "`agents` parameter + `allowedTools: ["Agent"]`". The permissions page adds that allowed tools auto-approve rather than restrict, and that calls needing no approval in `default` mode, including tools like `Agent` "that don't ask before running", run whether or not they are listed ([Configure permissions](https://code.claude.com/docs/en/agent-sdk/permissions)). For the exam, the guide's rule stands: the coordinator's allowed tools include the spawn tool.

    **Waiting for subagents.** The June 2025 [research post](https://www.anthropic.com/engineering/multi-agent-research-system) describes lead agents that "execute subagents synchronously". Claude Code used to pause for subagents too; its release notes for the week of June 29 to July 3, 2026 changed Claude Code's default: "Claude now keeps working while subagents run and picks up their results when they finish, instead of pausing the conversation to wait." ([What's new, week 27](https://code.claude.com/docs/en/whats-new/2026-w27)). The SDK subagents page agrees ("Subagents run in the background by default"), and Claude sets `run_in_background: false` when it needs a result before continuing. The exam's rule about several spawn calls in one response is unaffected.

### Lessons from Anthropic's multi-agent research system

The June 2025 post [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) describes a production coordinator in detail. Its architecture: a LeadResearcher plans and saves the plan to memory, spawns subagents that search with interleaved thinking, synthesizes their results, loops if more research is needed, then hands everything to a CitationAgent.

```text
 query --> [ LeadResearcher ] --save plan--> ( memory )
                 |   ^
       spawn in  |   | condensed findings
       parallel  v   |
          [ subagent 1 ]  [ subagent 2 ]  ...  (search, interleaved thinking)
                 |
      enough? no --> refine strategy, spawn more subagents
              yes --> [ CitationAgent ] --> report with citations
```

What went wrong and what fixed it, in the post's words where quoted ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)):

| Problem Anthropic observed | What fixed it |
|---|---|
| Too many subagents for simple queries, endless searching, agents distracting each other (see [What the coordinator decides](#what-the-coordinator-decides)) | Prompt engineering, including effort-scaling rules in the lead prompt |
| Vague tasks led to duplicated searches and gaps | Every delegation states objective, output format, tools and sources, boundaries |
| "Bad tool descriptions can send agents down completely wrong paths" | Explicit tool heuristics; a tool-testing agent that rewrote flawed descriptions gave "a 40% decrease in task completion time for future agents" |
| Long, specific queries returned few results | "start with short, broad queries, evaluate what’s available, then progressively narrow focus" |
| Sequential searching was slow | 3 to 5 subagents in parallel, each using 3+ tools in parallel |
| Agents preferred "SEO-optimized content farms over authoritative but less highly-ranked sources" | "Adding source quality heuristics to our prompts helped resolve this issue." |
| Context beyond 200,000 tokens would be truncated and the plan lost | The lead saves its plan to memory before delegating |
| Claims needed attribution | A separate CitationAgent locates citations for every claim |
| Errors compound and restarts are expensive | Resume "from where the agent was when the errors occurred", with "retry logic and regular checkpoints" |
| Failures were hard to explain | "Adding full production tracing let us diagnose why agents failed and fix issues systematically." |
| Deploys could break agents mid-run | "rainbow deployments", shifting traffic gradually between versions |
| Relaying every output through the lead degrades it, like a game of telephone | Subagents store work in external systems and return lightweight references to the coordinator |

Two cautions when you use these numbers. The 200,000-token truncation describes the 2025 system, so learn the pattern (persist the plan outside the context window), not the figure. And the synchronous execution the post describes is no longer Claude Code's default (see the warning above). How Anthropic evaluated the system (about 20 real queries to start, an LLM judge with a rubric, human testers) is covered in [Success criteria and test sets](evaluation-and-reliability.md#success-criteria-and-test-sets) and [Grading methods](evaluation-and-reliability.md#grading-methods). The post's separate appendix advice, to judge agents that change persistent state by their end state rather than turn by turn, is in [Evaluating agents](evaluation-and-reliability.md#evaluating-agents).

### How the exam frames it

The [CCAR-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) multi-agent research scenario ("A coordinator agent delegates to specialized subagents: one searches the web, one analyzes documents, one synthesizes findings, and one generates reports") carries three official sample questions (7, 8 and 9), reproduced with rationales in [Official sample questions](../claude-certified-architect-foundations.md#official-sample-questions). The decision rules they teach:

- **Trace a coverage gap to the decomposition first.** In question 7 every subagent did its job, but the coordinator split "creative industries" into three visual-arts subtasks (digital art, graphic design, photography), "completely omitting music, writing, and film." The rationale rejects the other options because they "incorrectly blame downstream agents that are working correctly within their assigned scope." ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).
- **Give a subagent the narrowest tool that covers the common case.** In question 9 the answer gives the synthesis agent "a scoped verify_fact tool for simple lookups, while complex verifications continue delegating to the web search agent through the coordinator." The rationale: it "applies the principle of least privilege by giving the synthesis agent only what it needs for the 85% common case"; giving it "access to all web search tools" (option C) "over-provisions the synthesis agent, violating separation of concerns", and batching every verification to the end (option B) "creates blocking dependencies since synthesis steps may depend on earlier verified facts." ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).
- **Send failures back as structured context.** In question 8 a web search subagent times out, and the answer returns "structured error context to the coordinator including the failure type, the attempted query, any partial results, and potential alternative approaches." The rationale rejects a generic status that "hides valuable context from the coordinator", an empty result "marked as successful" (it "suppresses the error"), and terminating the whole workflow "when recovery strategies could succeed." Error handling in depth is in [Error propagation in multi-agent systems](evaluation-and-reliability.md#error-propagation-in-multi-agent-systems).

The Professional exam frames tool distribution as "Evaluate tool/agent configuration for capability bloat" (objective 3.1), and its multi-agent objective is "Design multi-agent systems and orchestration strategies" (1.4) ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). Its sample 1, tagged to Domain 3 (Integration), has a support agent that can issue refunds and delete accounts although staff only read tickets and draft replies; the answer removes the refund and delete tools, and its rationale is quoted under [The enforcement ladder](#the-enforcement-ladder). The same rule applies to each subagent's tool list (our application of it). See [Least privilege for tools and agents](security-and-governance.md#least-privilege-for-tools-and-agents).

### Traps

- Always running every subagent in a fixed pipeline, instead of choosing subagents by query complexity (1.2-S1).
- Letting subagents talk to each other directly, instead of routing through the coordinator (1.2-S4).
- Assuming a subagent can see the coordinator's history or an earlier subagent's results (1.2-K2, 1.3-K2).
- Spawning subagents one per turn when the tasks are independent (1.3-S3).
- Handing subagents step-by-step procedures instead of goals and quality criteria (1.3-S4).
- Splitting agents by job role (planner, coder, tester, reviewer) when the work shares one context.
- Blaming a subagent for a gap that the coordinator's decomposition created (question 7).
- Giving a subagent every tool so it can skip the coordinator, instead of a scoped tool for the common case (question 9, option C).
- Hiding a subagent failure behind a generic status or an empty result marked as successful, or killing the whole run, instead of returning structured error context (question 8).
- Guarding an unneeded tool with logging or a confirmation step, instead of removing it (CCAR-P sample 1).

### Where the pattern lives in Anthropic products

As of September 2026, from the Claude Code, Agent SDK and Managed Agents docs:

| Product feature | How delegation works | Limits worth knowing |
|---|---|---|
| Claude Code and Agent SDK subagents | The main agent calls the `Agent` tool; results return to the spawner | Three layers of nesting and 20 running at once by default (versions and settings in [Limits on depth, concurrency and spend](#limits-on-depth-concurrency-and-spend)) |
| Claude Code agent teams | "A lead agent supervising peer sessions" ([dynamic workflows](https://code.claude.com/docs/en/workflows)); teammates message each other directly | Experimental, off by default; not spawned in `-p` or Agent SDK sessions |
| `Workflow` tool (dynamic workflows) | "moves the orchestration into a script the runtime executes outside the conversation context" ([Subagents in the SDK](https://code.claude.com/docs/en/agent-sdk/subagents)) | For "runs that coordinate dozens to hundreds of agents"; TypeScript Agent SDK v0.3.149 and later |
| Claude Managed Agents multiagent (beta, `managed-agents-2026-04-01`) | A coordinator delegates to agents in its declared roster; threads persist, so it can follow up with an agent it called earlier | One level deep; roster and thread limits in [Claude Managed Agents (Anthropic-hosted)](#claude-managed-agents-anthropic-hosted) |

Hosting Managed Agents is covered in [Deployment models](#deployment-models); file-based subagents for Claude Code in [Subagents](claude-code-configuration.md#subagents).

## The Claude Agent SDK

*Tested in: CCAR-F 2.5, How to Prepare item 1 and the Agent SDK entries in the technologies list · CCDV-F Agent Construction with Claude, Agentic Customization*

The Agent SDK docs define an agent as "an application that completes a task by planning its own steps and calling tools that read files, run commands, or edit code." The SDK is a library, in Python and TypeScript, that runs the Claude Code binary inside a process you operate, so your application gets the same tools, agent loop and context management as Claude Code, plus its permissions, sessions and hooks ([Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)). The quickstart states what sets it apart: "Claude executes tools directly instead of asking you to implement them." ([Quickstart](https://code.claude.com/docs/en/agent-sdk/quickstart)). With the Client SDK, by contrast, the overview says you write the tool loop yourself or let the client SDK's beta tool runner drive it.

Three of the six CCAR-F scenarios (the exam presents four of them, picked at random) are built with the Agent SDK: customer support, multi-agent research and developer productivity. The guide's technologies list also names agent definitions, agentic loops, `stop_reason` handling, hooks, subagent spawning and `allowedTools` configuration under the SDK. The first "How to Prepare" item is "Build an agent with the Claude Agent SDK: implement a complete agentic loop with tool calling, error handling, and session management." ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

!!! note "Older names you may meet"

    The SDK used to be called the Claude Code SDK. Anthropic's post dated September 29, 2025 announced the change: "To reflect this broader vision, we're renaming the Claude Code SDK to the Claude Agent SDK." ([Building agents with the Claude Agent SDK](https://claude.com/blog/building-agents-with-the-claude-agent-sdk)). The [migration guide](https://code.claude.com/docs/en/agent-sdk/migration-guide) lists the changes: the TypeScript package moved from `@anthropic-ai/claude-code` to `@anthropic-ai/claude-agent-sdk`; the Python package from `claude-code-sdk` to `claude-agent-sdk` (imports `claude_code_sdk` to `claude_agent_sdk`); `ClaudeCodeOptions` became `ClaudeAgentOptions`; and since v0.1.0 "The SDK no longer uses Claude Code's system prompt by default." Tutorials written before the rename can be wrong on all four points.

### Install and authenticate

```bash
# Python 3.10+ (inside a virtual environment)
pip install claude-agent-sdk

# Node.js 18+
npm install @anthropic-ai/claude-agent-sdk

# Both SDKs read the key from the process environment
export ANTHROPIC_API_KEY=your-api-key
```

From the [Quickstart](https://code.claude.com/docs/en/agent-sdk/quickstart) and [Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview):

| Fact | Detail |
|---|---|
| Claude Code install | Not needed in most cases: "Both the TypeScript and Python SDKs bundle a native Claude Code binary, so most installs need no separate Claude Code install." |
| API key | "The SDK reads the key from the environment of the process that runs your agent; it doesn't load `.env` files automatically." |
| Cloud providers | `CLAUDE_CODE_USE_BEDROCK=1` (Amazon Bedrock), `CLAUDE_CODE_USE_ANTHROPIC_AWS=1` plus `ANTHROPIC_AWS_WORKSPACE_ID` (Claude Platform on AWS), `CLAUDE_CODE_USE_VERTEX=1` (Google Cloud's Agent Platform), `CLAUDE_CODE_USE_FOUNDRY=1` (Microsoft Foundry), each with that cloud's credentials configured |
| Login for your users | "Unless previously approved, Anthropic does not allow third party developers to offer claude.ai login or rate limits for their products, including agents built on the Claude Agent SDK." Use API key authentication |
| Other languages | Run the CLI as a subprocess with `-p` and `--output-format json` to drive the same loop |

The [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) lists "Claude API authentication, billing, or account management" as out of scope, so for that exam the table above is background; for CCDV-F, identity and key handling are taught in [Secrets and API keys](security-and-governance.md#secrets-and-api-keys).

### `query()`: one task, one loop

`query()` is "the main entry point that creates the agentic loop. It returns an async iterator, so you use `async for` to stream messages as Claude works." ([Quickstart](https://code.claude.com/docs/en/agent-sdk/quickstart)). In TypeScript it returns a `Query` object that extends `AsyncGenerator<SDKMessage, void>`. Each call starts a new session with no memory of earlier calls unless you pass `continue_conversation=True` or `resume` (Python) or `continue` or `resume` (TypeScript).

The quickstart agent, which reviews a file and fixes crash bugs:

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

#### The message stream

| Message | When it arrives | What to read |
|---|---|---|
| `SystemMessage` | Session lifecycle; subtype `init` at the start, `compact_boundary` after compaction | Session metadata |
| `AssistantMessage` | One per content block in Claude's responses (text or a tool call) | What Claude is doing each turn |
| `UserMessage` | After each tool runs, carrying the tool result sent back to Claude; also for user input you stream mid-loop | Tool output |
| `StreamEvent` | Only when partial messages are enabled (`include_partial_messages` / `includePartialMessages`) | Raw streaming deltas |
| `ResultMessage` | End of the loop | `subtype`, `result`, cost, usage, `session_id` |

Python checks types with `isinstance()`. TypeScript checks `message.type`, and "`AssistantMessage` and `UserMessage` wrap the raw API message in a `.message` field, so content blocks are at `message.message.content`, not `message.content`." ([How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop)).

#### Handling the result

From [How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) and [Track cost and usage](https://code.claude.com/docs/en/agent-sdk/cost-tracking):

- Check `subtype` first: `success`, `error_max_turns`, `error_max_budget_usd`, `error_during_execution` or `error_max_structured_output_retries`. "The `result` field holds the final text output and is only present on the `success` variant, so always check the subtype before reading it."
- "All result subtypes carry `total_cost_usd`, `usage`, `num_turns`, and `session_id` so you can track cost and resume even after errors." The cost fields "are client-side estimates, not authoritative billing data."
- With subagents, pick the right total: `usage` excludes subagent activity, while `total_cost_usd` and `modelUsage` / `model_usage` include it. "Use `modelUsage`, or `model_usage` in Python, for whole-tree token accounting; the `usage` field undercounts as soon as nesting occurs."
- The result also carries the final turn's `stop_reason` (commonly `end_turn`, `max_tokens` or `refusal`); how stop reasons drive the loop is in [The agentic loop](#the-agentic-loop).
- "A single-shot `query()` call yields the final result message, then raises an error that includes the failure text, such as `Reached maximum number of turns`." Wrap the loop in `try` if your code must continue.
- Do not `break` on the result: "iterate the stream to completion rather than breaking on the result."

The docs' combined example puts these together with a turn cap, project settings and an effort level (from [How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop), with some comments shortened):

=== "Python"

    ```python
    import asyncio
    from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

    async def run_agent():
        session_id = None

        try:
            async for message in query(
                prompt="Find and fix the bug causing test failures in the auth module",
                options=ClaudeAgentOptions(
                    allowed_tools=["Read", "Edit", "Bash", "Glob", "Grep"],  # Auto-approved
                    setting_sources=["project"],  # Load CLAUDE.md, skills, hooks from current directory
                    max_turns=30,  # Prevent runaway sessions
                    effort="high",  # Thorough reasoning for complex debugging
                ),
            ):
                if isinstance(message, ResultMessage):
                    session_id = message.session_id  # Save for potential resumption

                    if message.subtype == "success":
                        print(f"Done: {message.result}")
                    elif message.subtype == "error_max_turns":
                        print(f"Hit turn limit. Resume session {session_id} to continue.")
                    elif message.subtype == "error_max_budget_usd":
                        print("Hit budget limit.")
                    else:
                        print(f"Stopped: {message.subtype}")
                    if message.total_cost_usd is not None:
                        print(f"Cost: ${message.total_cost_usd:.4f}")
        except Exception as error:
            # A single-shot query() raises after yielding an error result
            print(f"Session ended with an error: {error}")

    asyncio.run(run_agent())
    ```

=== "TypeScript"

    ```typescript
    import { query } from "@anthropic-ai/claude-agent-sdk";

    let sessionId: string | undefined;

    try {
      for await (const message of query({
        prompt: "Find and fix the bug causing test failures in the auth module",
        options: {
          allowedTools: ["Read", "Edit", "Bash", "Glob", "Grep"], // Auto-approved
          settingSources: ["project"], // Load CLAUDE.md, skills, hooks from current directory
          maxTurns: 30, // Prevent runaway sessions
          effort: "high" // Thorough reasoning for complex debugging
        }
      })) {
        if (message.type === "system" && message.subtype === "init") {
          sessionId = message.session_id;
        }
        if (message.type === "result") {
          if (message.subtype === "success") {
            console.log(`Done: ${message.result}`);
          } else if (message.subtype === "error_max_turns") {
            console.log(`Hit turn limit. Resume session ${sessionId} to continue.`);
          } else if (message.subtype === "error_max_budget_usd") {
            console.log("Hit budget limit.");
          } else {
            console.log(`Stopped: ${message.subtype}`);
          }
          console.log(`Cost: $${message.total_cost_usd.toFixed(4)}`);
        }
      }
    } catch (error) {
      // A single-shot query() throws after yielding an error result
      console.log(`Session ended with an error: ${error}`);
    }
    ```

Resuming that session by its ID is taught in [Sessions, resumption and forking](#sessions-resumption-and-forking).

### `ClaudeSDKClient` and streaming input

In Python, `query()` suits one-off tasks and `ClaudeSDKClient` suits conversations: "Use `ClaudeSDKClient` for interactive applications such as chat interfaces, or when the next action depends on Claude's response." ([Python SDK reference](https://code.claude.com/docs/en/agent-sdk/python)).

| Feature | `query()` | `ClaudeSDKClient` |
|---|---|---|
| Session | Creates a new session by default | Reuses the same session |
| Conversation | Single exchange | Multiple exchanges in the same context |
| Connection | Managed automatically | Manual control |
| Streaming input | Supported | Supported |
| Interrupts | Not supported | Supported |
| Hooks and custom tools | Supported | Supported |
| Continue a chat | Manual, via `continue_conversation` or `resume` | Automatic |
| Use case | One-off tasks | Continuous conversations |

Beyond `query()` and `receive_response()`, the client's methods include `interrupt()` ("Send interrupt signal (only works in streaming mode)"), `set_permission_mode()` and `set_model()` for mid-session changes, and `rewind_files()`, which restores files to an earlier user message and requires `enable_file_checkpointing=True` ([Python SDK reference](https://code.claude.com/docs/en/agent-sdk/python)).

A follow-up question in the same session, shortened from the reference's "Continuing a conversation" example:

```python
import asyncio
from claude_agent_sdk import ClaudeSDKClient, AssistantMessage, TextBlock

async def main():
    async with ClaudeSDKClient() as client:
        await client.query("What's the capital of France?")
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")

        # Follow-up question - the session retains the previous context
        await client.query("What's the population of that city?")
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(f"Claude: {block.text}")

asyncio.run(main())
```

`receive_response()` yields messages "until and including a ResultMessage". The reference warns: "When iterating over messages, avoid using `break` to exit early as this can cause asyncio cleanup issues." ([Python SDK reference](https://code.claude.com/docs/en/agent-sdk/python)). "The TypeScript SDK doesn't have a session-holding client object like Python's `ClaudeSDKClient`" ([Work with sessions](https://code.claude.com/docs/en/agent-sdk/sessions)); in TypeScript you pass `continue: true` or `resume` on later `query()` calls, and the returned `Query` object carries the control methods (`interrupt()`, `setPermissionMode()`, `setModel()`), which the TypeScript reference marks as available only in streaming input mode.

The docs call streaming input the preferred mode: a persistent session that accepts image uploads, queued messages and interruption, and surfaces permission requests. Single message input "does **not** support" direct image attachments, dynamic message queueing, real-time interruption or natural multi-turn conversations. The docs recommend it when you need a one-shot response, do not need image attachments or mid-session control methods, or "You need to operate in a stateless environment, such as a lambda function" ([Streaming input](https://code.claude.com/docs/en/agent-sdk/streaming-vs-single-mode)).

### Options that matter

Python passes a `ClaudeAgentOptions` dataclass; TypeScript passes an `Options` object. The table is a selection (the options this page and the SDK sections below rely on); defaults are from the two SDK references, Python first where they differ.

| Python | TypeScript | Default | What it does |
|---|---|---|---|
| `tools` | `tools` | Not set | Which built-in tools exist in the session; `{"type": "preset", "preset": "claude_code"}` gives Claude Code's defaults |
| `allowed_tools` | `allowedTools` | `[]` | Tools auto-approved without prompting; does not restrict Claude to these tools |
| `disallowed_tools` | `disallowedTools` | `[]` | A bare name (`"Bash"`) removes the tool from context; a scoped rule (`"Bash(rm *)"`) denies matching calls in every mode |
| `permission_mode` | `permissionMode` | `None` / `'default'` | `default`, `dontAsk`, `acceptEdits`, `bypassPermissions`, `plan`, `auto` |
| `can_use_tool` | `canUseTool` | Not set | Callback, invoked only when the permission flow falls through to a prompt |
| `system_prompt` | `systemPrompt` | Minimal prompt | A string, or the `claude_code` preset with optional `append`; also a `custom` object (both SDKs) or a `file` object (Python) |
| `setting_sources` | `settingSources` | All sources (user, project, local); in Python, only user and project when `skills` is set and this is unset | Which filesystem settings, CLAUDE.md files and `.claude/` skills, agents and commands load; `[]` loads none of them (managed policy settings, the global `~/.claude.json` config and auto memory are still read) |
| `mcp_servers` | `mcpServers` | `{}` | MCP servers, including in-process SDK servers for custom tools |
| `agents` | `agents` | Not set | Programmatic subagent definitions (`AgentDefinition`) |
| `hooks` | `hooks` | Not set / `{}` | Callbacks such as `PreToolUse` and `PostToolUse` |
| `max_turns` | `maxTurns` | No limit | Caps tool-use round trips; ends with `error_max_turns` |
| `max_budget_usd` | `maxBudgetUsd` | No limit | Caps the client-side cost estimate, subagents included; ends with `error_max_budget_usd` |
| `effort` | `effort` | Not set | `low`, `medium`, `high`, `xhigh`, `max` |
| `model` | `model` | CLI default | Alias or full model name |
| `cwd` | `cwd` | Current directory | Working directory for the agent |
| `env` | `env` | `{}` / `process.env` | Python merges your entries onto the inherited environment; a TypeScript `env` replaces it, so spread `process.env` to keep `PATH` |
| `resume` | `resume` | Not set | Session ID to resume |
| `continue_conversation` | `continue` | `False` / `false` | Continue the most recent conversation |
| `fork_session` | `forkSession` | `False` / `false` | With `resume`, branch to a new session ID |
| `output_format` | `outputFormat` | Not set | `{"type": "json_schema", "schema": {...}}` for structured output |
| `include_partial_messages` | `includePartialMessages` | `False` / `false` | Emit `StreamEvent` messages |
| (none) | `allowDangerouslySkipPermissions` | `false` | Required in TypeScript to use `bypassPermissions` |
| (none) | `persistSession` | `true` | `false` stops writing the session to disk, so it cannot be resumed |

Three distinctions decide most configuration questions:

- **Availability versus approval.** "`tools` and bare-name `disallowedTools` entries change availability. `allowedTools` and scoped `disallowedTools` rules change permission." ([Give Claude custom tools](https://code.claude.com/docs/en/agent-sdk/custom-tools)). An unlisted tool is still available, so to take a capability away, remove it; what each option does, including the `bypassPermissions` trap, is in [`allowedTools` pre-approves; it does not restrict](#allowedtools-pre-approves-it-does-not-restrict).
- **System prompt versus project instructions.** With no `systemPrompt` the SDK uses a minimal prompt, and "This differs from `claude -p`, which uses the Claude Code system prompt by default." The `claude_code` preset with `append` is the lowest-risk way to customize ("Nothing is removed, so this is the lowest-risk customization"), and "CLAUDE.md loading is controlled by setting sources, not by the `claude_code` preset." ([Modifying system prompts](https://code.claude.com/docs/en/agent-sdk/modifying-system-prompts)).
- **Programmatic versus filesystem settings.** "Programmatic options such as `agents`, `allowed_tools`, and `settings` override user, project, and local filesystem settings. Managed policy settings take precedence over programmatic options." ([Python SDK reference](https://code.claude.com/docs/en/agent-sdk/python)). Set `setting_sources=[]` when local configuration must not leak in: "Isolation is especially important for CI/CD pipelines, deployed applications, test environments, and multi-tenant systems where local customizations should not leak in." ([Migration guide](https://code.claude.com/docs/en/agent-sdk/migration-guide)). The same guide warns that Python SDK 0.1.59 and earlier treated an empty list the same as omitting the option, so upgrade before relying on `setting_sources=[]`. An empty list is still not full isolation: "Managed policy settings and the global `~/.claude.json` config are read regardless of this option", and auto memory at `~/.claude/projects/<project>/memory/` still loads into the system prompt unless you set `autoMemoryEnabled: false` in settings or `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` in `env` ([Use Claude Code features](https://code.claude.com/docs/en/agent-sdk/claude-code-features), which lists the other inputs `settingSources` does not control).

### Built-in tools

The SDK ships the tools that power Claude Code ([How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop)):

| Category | Tools | What they do |
|---|---|---|
| File operations | `Read`, `Edit`, `Write` | Read, modify, and create files |
| Search | `Glob`, `Grep` | Find files by pattern, search content with regex |
| Execution | `Bash` | Run shell commands, scripts, git operations |
| Web | `WebSearch`, `WebFetch` | Search the web, fetch and parse pages |
| Discovery | `ToolSearch` | Find and load tools on demand instead of preloading all of them |
| Orchestration | `Agent`, `Skill`, `AskUserQuestion`, `TaskCreate`, `TaskUpdate` | Spawn subagents, invoke skills, ask the user, track tasks |

The task-tracking tools are not on by default for every model. As of September 2026, the [Track todos](https://code.claude.com/docs/en/agent-sdk/todo-tracking#model-availability) page gives them by default only to Claude 3.x models, Opus 4 through 4.7, Sonnet 4 through 4.6 and Haiku 4.5 (Claude Code v2.1.268 and later). On other models, "Claude Code provides `TaskCreate` and `TaskUpdate` only when you opt in" ([How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop)): name one of the tools in `allowed_tools` / `allowedTools`, list them in `tools`, or set `CLAUDE_CODE_ENABLE_TODO_TOOLS=1` in `env`.

CCAR-F 2.5 tests selection among the file tools. The [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) rules, with the scenario cue that points to each:

| Need | Tool (CCAR-F 2.5 wording) |
|---|---|
| Find callers of a function, an error message, an import | Grep, "for content search (searching file contents for patterns like function names, error messages, or import statements)" |
| Find files by name or extension, such as `**/*.test.tsx` | Glob, "for file path pattern matching (finding files by name or extension patterns)" |
| Load or replace a whole file | Read and Write, "for full file operations" |
| Change a specific passage | Edit, "for targeted modifications using unique text matching" |
| Edit fails because the anchor text is not unique | "using Read + Write as a fallback for reliable file modifications" |
| Understand an unfamiliar codebase | "starting with Grep to find entry points, then using Read to follow imports and trace flows, rather than reading all files upfront" |
| Find every use of a function re-exported through wrapper modules | "first identifying all exported names, then searching for each name across the codebase" |

The guide's task statement 2.5 ("Select and apply built-in tools (Read, Write, Edit, Bash, Grep, Glob) effectively") also names Bash, though none of its bullets is about Bash; the SDK docs describe it as the tool to "Run shell commands, scripts, git operations". Exploration strategy for large repositories is taught in [Exploring large codebases](context-engineering.md#exploring-large-codebases).

The quickstart's presets show how tool lists map to autonomy: `Read`, `Glob`, `Grep` for read-only analysis; `Read`, `Edit`, `Glob` to analyze and modify code; `Read`, `Edit`, `Bash`, `Glob`, `Grep` for full automation. When one response asks for several tools, "Read-only tools (like `Read`, `Glob`, `Grep`, and MCP tools marked as read-only) can run concurrently. Tools that modify state (like `Edit`, `Write`, and `Bash`) run sequentially to avoid conflicts." A denied call does not crash the loop: "Claude receives a rejection message as the tool result and typically attempts a different approach or reports that it couldn't proceed." ([How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop)).

!!! warning "Exam guide vs current docs"

    **Grep and Glob.** The guide treats Grep and Glob as standard built-in tools. Claude Code's tools reference now (as of September 2026) says that on macOS, Linux and WSL, Claude Code leaves Glob and Grep out of the default tool set and Claude searches with `find` and `grep` through the Bash tool instead; those searches reach hooks and permission rules as `Bash` calls. They come back when you name them in `--tools` or `--allowedTools` at startup, or in the equivalent Agent SDK options (the combined example above names both in `allowed_tools`), or when Bash is removed from the session. "An allow rule in a settings file doesn't have this effect." The Agent SDK agent-loop page still lists both in its built-in tools table, and their roles are unchanged, so the exam answers hold ([Tools reference](https://code.claude.com/docs/en/tools-reference)).

    **Edit on a non-unique match.** The guide's fallback is Read + Write. The tools reference describes a different remedy: when `old_string` appears more than once, "Claude either supplies a longer string with enough surrounding context to pin down one occurrence, or sets `replace_all: true` to replace them all." ([Tools reference](https://code.claude.com/docs/en/tools-reference)). Answer with the guide's fallback on the exam.

Beyond the built-ins, an agent can connect MCP servers, define custom tools in an in-process SDK MCP server, and load project skills through setting sources. Custom tools are in [Custom tools in the SDK](#custom-tools-in-the-sdk); choosing among built-in tools, custom tools, Skills and MCP (CCDV-F Agentic Customization) is in [Built-in tools, custom tools, Skills or MCP](tool-use-and-mcp.md#built-in-tools-custom-tools-skills-or-mcp).

### Branding and terms

If you ship a product on the SDK, Claude branding is optional. The overview allows "Claude Agent", "Claude" inside a menu already labeled "Agents", and "{YourAgentName} Powered by Claude"; it does not permit "Claude Code", "Claude Code Agent" or Claude Code-branded ASCII art and visuals. Use of the SDK is governed by Anthropic's Commercial Terms of Service ([Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)).

## Subagents in the SDK

*Tested in: CCAR-F 1.2, 1.3, 2.3, 3.4, 5.4 · CCDV-F Agent Architecture, Agent Patterns and Frameworks, Context Engineering · CCAR-P 1.4, 3.1*

A subagent is a separate agent instance that your main agent spawns to handle a focused subtask. Delegating buys four things: context isolation, parallel execution, specialized instructions, and tool restrictions. Subagents that run in parallel finish in the time of the slowest one, not the sum of all of them. For when a multi-agent design pays off at all, see [Multi-agent orchestration](#multi-agent-orchestration); for the context-budget view, see [Subagents as context isolation](context-engineering.md#subagents-as-context-isolation).

!!! tip "Exam wording: Task is today's Agent tool"

    The CCAR-F guide calls the spawning tool Task and says a coordinator's `allowedTools` must include it (1.3-K1). Current releases call it `Agent` (renamed in Claude Code v2.1.63, with `Task` still accepted as an alias). The full comparison, including what listing it in `allowedTools` does today, is in the "Exam guide vs current docs" warning under [Spawning subagents in parallel](#spawning-subagents-in-parallel). On the exam, answer in the guide's terms: the coordinator lists Task in its allowed tools. In code, write `"Agent"` and detect both names.

### Three ways to define a subagent

| Method | Where it lives | What to know |
|---|---|---|
| Programmatic (recommended) | The `agents` option on `query()`, keyed by agent name | Wins over a filesystem agent with the same name |
| Filesystem | Markdown files with YAML frontmatter in `.claude/agents/` | Read when the matching setting source is enabled (it is when you omit `settingSources`); format is the same as in Claude Code, see [Subagents](claude-code-configuration.md#subagents) |
| Built-in `general-purpose` | Nothing to define | Claude can invoke it at any time through the Agent tool, and an Agent call without `subagent_type` gets it; `CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS=1` removes it, and such a call then fails with `subagent_type is required` |

A coordinator with two programmatic subagents, one read-only and one allowed to run tests (the SDK docs example, with the system prompts shortened to `...`):

=== "Python"

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

=== "TypeScript"

    ```typescript
    import { query } from "@anthropic-ai/claude-agent-sdk";

    for await (const message of query({
      prompt: "Review the authentication module for security issues",
      options: {
        // Auto-approve these tools
        allowedTools: ["Read", "Grep", "Glob", "Agent"],
        agents: {
          "code-reviewer": {
            // description tells Claude when to use this subagent
            description:
              "Expert code review specialist. Use for quality, security, and maintainability reviews.",
            // prompt defines the subagent's behavior and expertise
            prompt: "You are a code review specialist with expertise in security, performance, and best practices. ...",
            // tools restricts what the subagent can do (read-only here)
            tools: ["Read", "Grep", "Glob"],
            // model overrides the default model for this subagent
            model: "sonnet"
          },
          "test-runner": {
            description:
              "Runs and analyzes test suites. Use for test execution and coverage analysis.",
            prompt: "You are a test execution specialist. Run tests and provide clear analysis of results. ...",
            // Bash access lets this subagent run test commands
            tools: ["Bash", "Read", "Grep"]
          }
        }
      }
    })) {
      if ("result" in message) console.log(message.result);
    }
    ```

### AgentDefinition fields

The guide's 1.3-K3 names three parts of an AgentDefinition: descriptions, system prompts, and tool restrictions. Those map to `description`, `prompt` and `tools`. The full field list as of September 2026:

| Field | Required | What it does |
|---|---|---|
| `description` | Yes | Natural-language description of when to use this agent; Claude matches tasks against it |
| `prompt` | Yes | The subagent's system prompt |
| `tools` | No | Allowed tool names; if omitted, the subagent inherits every tool available to subagents |
| `disallowedTools` | No | Tools removed from the subagent's set; `mcp__server`, `mcp__server__*` and `mcp__*` patterns remove MCP tools by server or all at once |
| `model` | No | An alias such as `'fable'`, `'opus'`, `'sonnet'`, `'haiku'`, `'inherit'` (the main model), or a full model ID; the Python reference lists only `"sonnet"`, `"opus"`, `"haiku"` and `"inherit"` |
| `skills` | No | Skill names preloaded into the subagent's context at startup; unlisted skills stay invocable through the Skill tool |
| `memory` | No | Memory source: `'user'`, `'project'` or `'local'` |
| `mcpServers` | No | MCP servers available to this agent, by name or inline config |
| `initialPrompt` | No | First user turn when the agent runs as the main thread; ignored when invoked as a subagent |
| `maxTurns` | No | Turn cap; at the cap Claude Code marks the output partial so the agent can be resumed (partial marking needs Claude Code v2.1.246 or later) |
| `background` | No | Forces background execution for this agent, whatever Claude requests |
| `omitClaudeMd` | No | TypeScript only (Agent SDK v0.3.271 or later): run as a subagent without user, project and local CLAUDE.md files; managed policy files still load |
| `effort` | No | `low`, `medium`, `high`, `xhigh`, `max` or a number; overrides the session's effort level for this subagent |
| `permissionMode` | No | The subagent's permission mode, applied only in some parent modes and never as `bypassPermissions` (rules in [Permission modes](#permission-modes)) |

In Python, `AgentDefinition` is a dataclass and its multi-word fields keep camelCase (`disallowedTools`, `permissionMode`, `maxTurns`) to match the wire format, unlike `ClaudeAgentOptions`, which uses snake_case (`disallowed_tools`, `permission_mode`). Passing a snake_case keyword to `AgentDefinition` raises a `TypeError` at construction time.

### What crosses the boundary

The rule that a non-fork subagent receives only the Agent tool's prompt string from its parent, and what to put in that string, is taught in [Passing context to a subagent](#passing-context-to-a-subagent). The SDK adds a few controls on top of that rule:

| Control | Effect |
|---|---|
| `AgentDefinition.prompt` | The subagent's own system prompt; it does not receive the parent's system prompt |
| `AgentDefinition.skills` | Skill content preloaded at startup; other skills are not preloaded but stay invocable through the Skill tool |
| `omitClaudeMd` (TypeScript) | Runs the subagent without user, project and local CLAUDE.md files; otherwise project CLAUDE.md loads through setting sources |
| `tools` | The subagent's tool definitions: the listed subset, or everything available to subagents (filtered for background runs) |
| Inherited automatically | The main session's extended thinking configuration |

The parent may summarize the subagent's final message (see [Passing context to a subagent](#passing-context-to-a-subagent)), so if users must see the subagent's words verbatim, say so in the main prompt or `systemPrompt`. From Claude Code v2.1.210 the harness scans that final message for instruction-shaped patterns before the parent reads it: it neutralizes imitated control tags (such as a `<system-reminder>` block) and turn markers (`Human:` or `Assistant:` at the start of a line) by inserting a backslash, keeps permission-configuration mentions as written, and prepends a `[harness: ...]` marker line for control-tag or permission-configuration matches. It never removes or rewords the subagent's text. An API error that ends a subagent early, such as a rate limit, is never delivered as its result.

### Invoking subagents, in parallel and in the background

- **Automatic:** Claude decides when to delegate from the task and each subagent's `description`. When Claude answers directly instead of delegating, the docs' two fixes are a description that says exactly when to use the subagent, and explicit invocation.
- **Explicit:** name the subagent in the prompt, for example "Use the code-reviewer agent to check the authentication module" ([subagents page](https://code.claude.com/docs/en/agent-sdk/subagents)). This bypasses automatic matching.
- **Parallel:** several Agent calls in one coordinator response run together; the exam objective (1.3-S3) and an example response are in [Spawning subagents in parallel](#spawning-subagents-in-parallel).
- **Background by default:** an Agent call that omits `run_in_background` launches a background subagent; before v2.1.198 the default was still rolling out, and such a call could run synchronously. The change and how it relates to the exam are in the "Exam guide vs current docs" warning under [Spawning subagents in parallel](#spawning-subagents-in-parallel).

The Agent tool's input and output, as the SDK references document them:

| Direction | Fields |
|---|---|
| Input | `description` (short, 3 to 5 words), `prompt`, `subagent_type`, `model`, `run_in_background`, `name`, `isolation` (`worktree` or `remote`); `mode` and `team_name` are deprecated and ignored |
| Output | Discriminated on `status`: `completed` (finished), `async_launched` (background), `remote_launched` (dispatched to a cloud session) |

To observe delegation in your message stream, look for `tool_use` blocks that call the tool, and for `parent_tool_use_id` on messages produced inside a subagent. Match both names: `tool_use` blocks emitted before Claude Code v2.1.63 say `Task`, and the `system:init` tools list still does. In Python the blocks are on `message.content`; in TypeScript, `SDKAssistantMessage` wraps the API message, so they are on `message.message.content`. The detection loop from the docs example, with `prompt` and `options` standing for your own values:

```typescript
for await (const message of query({ prompt, options })) {
  const msg = message as any;

  // Match both names: older versions emitted "Task", current versions emit "Agent"
  for (const block of msg.message?.content ?? []) {
    if (block.type === "tool_use" && (block.name === "Task" || block.name === "Agent")) {
      console.log(`Subagent invoked: ${block.input.subagent_type}`);
    }
  }

  // Check if this message is from within a subagent's context
  if (msg.parent_tool_use_id) {
    console.log("  (running inside subagent)");
  }
}
```

### Tool restrictions and least privilege

A tool you leave out of `tools` is not in the subagent's session at all: Claude works without it, with no permission prompt and no error. The docs suggest these combinations:

| Role | Tools | Effect |
|---|---|---|
| Read-only analysis | `Read`, `Grep`, `Glob` | Can examine code but not modify or execute |
| Test execution | `Bash`, `Read`, `Grep` | Can run commands and analyze output |
| Code modification | `Read`, `Edit`, `Write`, `Grep`, `Glob` | Full read and write access without command execution |
| Full access | Omit `tools` | Inherits the tools available to subagents |

This is the SDK mechanism behind CCAR-F 2.3-S1 ("Restricting each subagent's tool set to those relevant to its role, preventing cross-specialization misuse") and 2.3-K3 (scoped access, "with limited cross-role tools for specific high-frequency needs"). The guide's reasons: too many tools ("18 instead of 4-5") degrade tool selection (2.3-K1), and agents with tools outside their specialization tend to misuse them, such as a synthesis agent attempting web searches (2.3-K2). CCAR-F sample question 9 applies it with a scoped `verify_fact` tool for the synthesis agent while complex cases route through the coordinator (2.3-S3); the answer and its rationale are under [How the exam frames it](#how-the-exam-frames-it).

Two related limits. `AskUserQuestion` is not available in subagents spawned through the Agent tool, so clarifying questions must route through the parent. And to stop delegation entirely, deny the `Agent` tool itself (`permissions.deny` in settings, or its bare name in `disallowedTools`, which removes it from Claude's context).

### Limits on depth, concurrency and spend

Claude decides on its own when to spawn a subagent and how many, and a subagent can spawn subagents of its own, so one prompt can grow into a tree of agents whose API requests all count toward the query's `total_cost_usd`. Three caps bound that growth, and `maxTurns` bounds each subagent. The docs describe the three caps for TypeScript SDK v0.3.219 and Python SDK v0.2.127 and later (the releases that bundle Claude Code v2.1.219 or later); on earlier releases some are missing or default differently.

| Control | Default | What happens at the limit |
|---|---|---|
| `CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH` (env) | `3` layers below the main agent; `1` stops subagents from spawning their own | Claude Code withholds the Agent tool from subagents at the bottom layer (a fork keeps it, but the call returns an error), so they do their delegated work themselves |
| `CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS` (env) | `20` running at once | The Agent tool returns `Concurrent subagent limit reached` until the running count drops below the limit; sessions with ultracode active are never refused |
| `max_budget_usd` / `maxBudgetUsd` (query option) | No limit; counts the call's own spend, subagent requests included | Spawning another subagent fails with `Budget limit reached`, background subagents still running are stopped, and the query ends with the `error_max_budget_usd` result subtype (cap enforcement needs Claude Code v2.1.217 or later) |
| `maxTurns` on the AgentDefinition | None | Output is marked partial; there is no per-subagent wall-clock deadline |

Set the depth and concurrency limits as environment variables through the `env` option; in TypeScript, spread `process.env` into it (the `env` row of [Options that matter](#options-that-matter) explains why). The docs example turns nesting off, allows five subagents at a time, and caps spend at five US dollars:

=== "Python"

    ```python
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Grep", "Glob", "Agent"],
        # env is merged on top of the inherited environment
        env={
            "CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH": "1",
            "CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS": "5",
        },
        max_budget_usd=5.0,
    )
    ```

=== "TypeScript"

    ```typescript
    const options = {
      allowedTools: ["Read", "Grep", "Glob", "Agent"],
      // env replaces the subprocess environment, so spread process.env to keep PATH
      env: {
        ...process.env,
        CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH: "1",
        CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS: "5",
      },
      maxBudgetUsd: 5,
    };
    ```

Claude Opus 5 delegates to subagents more readily than earlier models, so the docs say these limits matter most on queries that run it. With the `claude_code` system prompt preset and Opus 5, Claude Code adds a line telling Claude not to call the Agent tool unless asked; with a custom prompt, or with no `systemPrompt` at all (the SDK default), that line is absent. Either instruction only steers Claude, so set the limits as well.

!!! warning "Nesting depth: release notes vs current docs"

    The [week 24 release notes](https://code.claude.com/docs/en/whats-new/2026-w24) (June 8 to 12, 2026) described subagent chains as "capped at five levels deep". The current subagents pages give a default of three layers below the main agent, set by `CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH`. Use three. Do not confuse either figure with Claude Managed Agents, where a coordinator delegates only one level deep (see [Deployment models](#deployment-models)).

### Resuming a subagent

Each `query()` call starts a new session by default, and you must resume the same session to reach a subagent's transcript. When a subagent completes, the Agent tool result includes a text block containing `agentId: <id>`. To continue its work:

1. Capture the parent `session_id` from the first query's messages.
2. Parse `agentId` from the Agent tool result text.
3. Pass `resume` with that session ID in the second query's options, name the agent ID in the prompt (for example `Resume agent <id> and ...`), and pass the same custom agent definition in `agents` for both queries.

A resumed subagent keeps its full conversation history, including earlier tool calls, results and reasoning. A subagent stopped by its `maxTurns` cap is marked partial so Claude knows the run is unfinished and can resume it. The built-in `Explore` and `Plan` agents are one-shot and return no `agentId`, so use a custom agent or `general-purpose` when you will need to resume.

### Built-in subagents and forks

| Name | What it is | Why it matters |
|---|---|---|
| `general-purpose` | Handles tasks that need both exploration and modification | What an Agent call without `subagent_type` gets; resumable |
| `Explore` | Fast, read-only codebase search; Write and Edit are denied; Claude specifies a thoroughness level (quick, medium, very thorough); one-shot | CCAR-F 3.4-K4, 3.4-S3 and 5.4-K3: isolate verbose discovery output and return a summary to the main agent |
| `Plan` | Research agent used during plan mode to gather context before presenting a plan; one-shot | Supports the plan mode items in CCAR-F 3.4 |
| Fork | A subagent that inherits the entire conversation so far instead of starting fresh; only its final result returns, and it shares the parent's prompt cache | On by default in interactive sessions, off by default with `-p` and in the Agent SDK (override with `CLAUDE_CODE_FORK_SUBAGENT`); not the guide's `fork_session` |

As of v2.1.198, Explore inherits the main conversation's model (capped at Opus on the Claude API) instead of always running on Haiku. A subagent fork is not the same thing as `fork_session`: a fork subagent is a child that shares the parent's context, while `fork_session` branches a whole session (see [Sessions, resumption and forking](#sessions-resumption-and-forking)).

### Decide and avoid

- If you detect subagent calls in your message stream, match both `Agent` and `Task`; builds before v2.1.63 emit `Task`, and the `system:init` tools list still says `Task`.
- If subagents must not spawn subagents of their own, set `CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH` to `1`; the default allows three layers.
- If a subagent's role does not need a capability, leave the tool out of its `tools`, and give a cross-role tool only in a scoped form for a high-frequency need (a `verify_fact` lookup); not the whole tool family, which the sample question 9 rationale calls over-provisioning that violates separation of concerns.
- If delegation must never happen, deny the `Agent` tool (`permissions.deny`, or its bare name in `disallowedTools`); not leaving `Agent` out of `allowedTools`, which only auto-approves: `Agent` does not ask before running, so it runs whether or not it is listed. On the exam, still list the spawn tool (Task) in the coordinator's allowed tools, as 1.3-K1 requires.
- If you need to continue a subagent later, use a custom or `general-purpose` agent and keep the parent session ID; `Explore` and `Plan` cannot be resumed.
- If a phase produces verbose discovery output ("find all test files," "trace refund flow dependencies" in 5.4-S1), delegate it to a subagent such as Explore so only a summary reaches the main agent; that is the guide's answer for preserving main-conversation context (3.4-S3, 5.4-K3).
- If a run may fan out, set the depth, concurrency and budget caps; a prompt line about delegation only steers Claude, while Claude Code enforces the caps.

## Hooks in the SDK

*Tested in: CCAR-F 1.4, 1.5 · CCDV-F Agent Construction with Claude, Claude Hooks, Guardrails and Safe Deployment · CCAR-P 5.1*

Hooks are callback functions that run your code in response to agent events: a tool about to be called, a tool result coming back, a subagent starting, a session ending. They are deterministic. The [Claude Code glossary](https://code.claude.com/docs/en/glossary) puts it this way: "Hooks are deterministic: they fire at fixed lifecycle points rather than at the model's discretion." That is why the exam guides pair hooks with guarantees: in the [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf), skill 1.5-S3 is "Choosing hooks over prompt-based enforcement when business rules require guaranteed compliance", and 1.5-K3 contrasts "hooks for deterministic guarantees" with "prompt instructions for probabilistic compliance". The CCDV-F Claude Hooks skill (1.0% of that exam) is about guardrails that prevent destructive actions.

Three properties set SDK hooks apart:

- They run in your application process, not inside the agent's context window, so they cost no context.
- The SDK collects two kinds: callbacks you pass in `options.hooks`, and shell command hooks from settings files when the matching setting source is enabled (it is for default `query()` options).
- They sit first in the permission flow: a `PreToolUse` deny applies even in `bypassPermissions` mode (see [Permissions and enforcement](#permissions-and-enforcement)).

### Register a hook

The `hooks` option maps an event name to a list of matchers; each matcher holds an optional pattern and the callbacks to run. This example from the SDK docs blocks writes to `.env` files (the Python version is trimmed to the callback and options):

=== "Python"

    ```python
    from claude_agent_sdk import ClaudeAgentOptions, HookMatcher

    async def protect_env_files(input_data, tool_use_id, context):
        file_path = input_data["tool_input"].get("file_path", "")
        file_name = file_path.split("/")[-1]
        if file_name == ".env":
            return {
                "hookSpecificOutput": {
                    "hookEventName": input_data["hook_event_name"],
                    "permissionDecision": "deny",
                    "permissionDecisionReason": "Cannot modify .env files",
                }
            }
        # Return empty object to allow the operation
        return {}

    options = ClaudeAgentOptions(
        hooks={
            "PreToolUse": [HookMatcher(matcher="Write|Edit", hooks=[protect_env_files])]
        }
    )
    ```

=== "TypeScript"

    ```typescript
    import { query, HookCallback, PreToolUseHookInput } from "@anthropic-ai/claude-agent-sdk";

    const protectEnvFiles: HookCallback = async (input, toolUseID, { signal }) => {
      const preInput = input as PreToolUseHookInput;
      const toolInput = preInput.tool_input as Record<string, unknown>;
      const filePath = toolInput?.file_path as string;
      const fileName = filePath?.split("/").pop();

      if (fileName === ".env") {
        return {
          hookSpecificOutput: {
            hookEventName: preInput.hook_event_name,
            permissionDecision: "deny",
            permissionDecisionReason: "Cannot modify .env files"
          }
        };
      }
      // Return empty object to allow the operation
      return {};
    };

    for await (const message of query({
      prompt: "Create a .env file with the standard local development database configuration",
      options: {
        hooks: {
          PreToolUse: [{ matcher: "Write|Edit", hooks: [protectEnvFiles] }]
        }
      }
    })) {
      if (message.type === "assistant" || message.type === "result") {
        console.log(message);
      }
    }
    ```

### Every hook event

As of September 2026, the Python SDK supports 10 events. The TypeScript SDK supports all of them plus the TypeScript-only events in the table below. `SessionStart` and `SessionEnd` are TypeScript-only as callbacks; a Python application can still use them as shell hooks in `.claude/settings.json` loaded through `setting_sources`. Event names are case-sensitive (`PreToolUse`, not `preToolUse`).

| Event | Python | TypeScript | Fires when |
|---|---|---|---|
| `PreToolUse` | Yes | Yes | A tool call is requested, before it runs (can block or modify) |
| `PostToolUse` | Yes | Yes | A tool call succeeded and returned its result |
| `PostToolUseFailure` | Yes | Yes | A tool call failed |
| `PostToolBatch` | No | Yes | A full batch of parallel tool calls resolved, once per batch, before the next model call |
| `UserPromptSubmit` | Yes | Yes | A prompt is submitted, before Claude processes it |
| `UserPromptExpansion` | No | Yes | A user-typed command or MCP prompt expands into a prompt (can block the expansion) |
| `MessageDisplay` | No | Yes | An assistant message with text completes |
| `Stop` | Yes | Yes | Claude finishes responding |
| `StopFailure` | No | Yes | The turn ends with an API error instead of a normal stop |
| `SubagentStart` | Yes | Yes | A subagent is spawned |
| `SubagentStop` | Yes | Yes | A subagent finishes |
| `PreCompact` | Yes | Yes | Before context compaction |
| `PostCompact` | No | Yes | After compaction completes |
| `PreModelSwitch` | No | Yes | Before a requested model switch (can block it) |
| `PostModelSwitch` | No | Yes | After the session's model changes, including an automatic fallback |
| `PermissionRequest` | Yes | Yes | A tool call needs a permission decision |
| `PermissionDenied` | No | Yes | Auto mode denies a tool call |
| `Notification` | Yes | Yes | The agent sends a status notification |
| `SessionStart` | No (shell hook only) | Yes | A session begins or resumes |
| `SessionEnd` | No (shell hook only) | Yes | A session terminates |
| `Setup` | No | Yes | Session setup or maintenance runs |
| `TeammateIdle` | No | Yes | An agent team teammate is about to go idle |
| `TaskCreated` | No | Yes | A task is created through `TaskCreate` |
| `TaskCompleted` | No | Yes | A task is marked completed |
| `Elicitation` | No | Yes | An MCP server requests user input mid-task |
| `ElicitationResult` | No | Yes | The user answers an MCP elicitation, before the answer returns to the server |
| `ConfigChange` | No | Yes | A configuration file changes during the session |
| `InstructionsLoaded` | No | Yes | A CLAUDE.md or rules file is loaded into context |
| `WorktreeCreate` | No | Yes | A git worktree is created |
| `WorktreeRemove` | No | Yes | A git worktree is removed |
| `CwdChanged` | No | Yes | The working directory changes |
| `FileChanged` | No | Yes | A watched file is modified, created or deleted |
| `DirectoryAdded` | No | Yes | A working directory is added mid-session |

The CCAR-F guide names two hook patterns, and its technology appendix lists them as "hooks (PostToolUse, tool call interception)". `PostToolUse` is named outright (1.5-K1, 1.5-S1: transforming tool results before the model processes them). "Tool call interception" (1.5-K2, 1.5-S2) is the `PreToolUse` event, which fires on a tool call request and can block or modify it. The other events do not appear in the guide's objectives.

### Matchers

SDK matchers follow the same rules as matchers in settings files:

| Matcher value | Meaning |
|---|---|
| Omitted, `""` or `"*"` | Every occurrence of the event |
| Only letters, digits, underscores, hyphens, spaces, commas and the pipe character | An exact tool name, or a list of exact names (commas need Claude Code v2.1.191 or later and hyphens v2.1.195 or later; `FileChanged` and `StopFailure` accept only letters, digits, `_` and `\|` on this path) |
| Anything else (for example `^mcp__`) | An unanchored JavaScript regular expression, so `Edit.*` matches both `Edit` and `NotebookEdit`; write `^Edit$` for a whole-string match |

So `Write|Edit` and (from v2.1.191) `Edit, Write` are exact lists that match either tool, while `^mcp__` is a regular expression that matches every MCP tool.

- Tool events (`PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `PermissionRequest`, `PermissionDenied`) match the tool name. MCP tools are named `mcp__<server>__<action>`, where `<server>` is the key in your `mcpServers` configuration. To match every tool from one server, append `.*` (`mcp__memory__.*`): `mcp__memory` alone is an exact string that matches no tool.
- Matchers see only the tool name, never its arguments. To act on a path, check `tool_input.file_path` inside the callback.
- Other events match other fields: `SubagentStart` and `SubagentStop` match the agent type, `PreCompact` matches `manual` or `auto`, `SessionStart` matches `startup`, `resume`, `clear`, `compact` or `fork`.
- `UserPromptSubmit`, `PostToolBatch`, `Stop`, `TeammateIdle`, `TaskCreated`, `TaskCompleted`, `WorktreeCreate`, `WorktreeRemove`, `MessageDisplay` and `CwdChanged` have no matcher support; a matcher on them is silently ignored.
- Each `HookMatcher` also takes a `timeout` in seconds.

### What a callback receives

Every callback gets three arguments: the input data, the tool use ID, and a context object (in TypeScript it carries an `AbortSignal` named `signal`; in Python it is reserved for future use).

| Input | Carried by | Use |
|---|---|---|
| `session_id`, `cwd`, `hook_event_name` | Every event | Key per-session state; route one callback across events |
| `agent_id`, `agent_type` | Events fired inside a subagent (TypeScript: on every hook input; Python: optional on `PreToolUse`, `PostToolUse`, `PostToolUseFailure` and `PermissionRequest`, required on `SubagentStart` and `SubagentStop`) | Apply rules per subagent |
| `tool_name`, `tool_input`, `tool_use_id` | Tool events | Inspect the call |
| `tool_response` | `PostToolUse` | The result the tool returned, alongside `tool_input` |
| Tool use ID (second argument) | Tool events | Correlate a `PreToolUse` with the `PostToolUse` for the same call |

### What a callback returns

SDK callbacks use the same JSON output format as Claude Code shell hooks. Return `{}` to let the operation proceed unchanged.

| Field | Where | Effect |
|---|---|---|
| `systemMessage` | Top level | A message shown to the user, not the model |
| `continue` (Python `continue_`) | Top level | `false` stops Claude entirely; takes precedence over any event-specific decision |
| `hookSpecificOutput.hookEventName` | Nested | Required whenever you return `hookSpecificOutput` |
| `permissionDecision` | Nested, `PreToolUse` | `"allow"`, `"deny"`, `"ask"` or `"defer"`; `defer` ends the query so you can resume it later |
| `permissionDecisionReason` | Nested, `PreToolUse` | Shown to Claude for `deny`, so it can adapt instead of retrying; shown to the user (not Claude) for `allow` and `ask` |
| `updatedInput` | Nested, `PreToolUse` | Replaces the tool input; pair with `allow` to auto-approve or `ask` to show the user; ignored with `defer` |
| `additionalContext` | Nested | Text added for Claude where the hook fired (for `PostToolUse`, next to the tool result) |
| `updatedToolOutput` | Nested, `PostToolUse` | Replaces what Claude sees as the tool result; works for any tool in both SDKs |
| `decision: "block"` plus `reason` | Top level, on events such as `PostToolUse`, `Stop`, `SubagentStop` and `UserPromptSubmit` | The only value is `"block"`; its meaning depends on the event (next table) |
| `async: true` (Python `async_`) | Top level | The agent proceeds without waiting; the hook can no longer block, modify or add context; `asyncTimeout` optionally bounds the background work in milliseconds |

`additionalContext`, `systemMessage` and plain stdout are each capped at 10,000 characters. The older `updatedMCPToolOutput` field replaced MCP output only and is deprecated. Always return a new object for `updatedInput` rather than mutating `tool_input`, and put it inside `hookSpecificOutput`: at the top level it is ignored.

What "block" means depends on the event:

| Event | Can it block? | What blocking does |
|---|---|---|
| `PreToolUse` | Yes | `deny` prevents the call; Claude receives the reason as the tool result and usually tries another approach |
| `PostToolUse` | No, the tool already ran | `decision: "block"` adds the reason next to the result; Claude still sees the original output unless `updatedToolOutput` replaces it |
| `UserPromptSubmit` | Yes | Prevents processing and erases the prompt from context; the reason goes to the user, not Claude; it cannot rewrite the prompt, only add context |
| `Stop` | Yes | Prevents Claude from stopping; the required `reason` tells Claude why to continue; check `stop_hook_active` to avoid looping, and Claude Code ends the turn anyway after 8 consecutive blocks |
| `SubagentStop` | Yes | Keeps the subagent running and delivers `reason` to it as its next instruction |
| `PermissionRequest` | Through `decision.behavior` | `allow` or `deny`; an `allow` never overrides a matching deny rule |
| `SubagentStart` | No | Can add `additionalContext` to the subagent |
| `SessionStart`, `SessionEnd` | No | `SessionStart` adds context; `SessionEnd` does cleanup only |

When several hooks or rules apply, "`deny` takes priority over `defer`, which takes priority over `ask`, which takes priority over `allow`" ([Agent SDK hooks](https://code.claude.com/docs/en/agent-sdk/hooks)). All matching hooks run in parallel, in no guaranteed order, so write each one to act on its own. If two hooks both return `updatedInput` for the same tool, the last to finish wins, which is non-deterministic: keep one rewriter per tool.

### Blocking a policy violation and redirecting

In the [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf), skill 1.5-S2 is "Implementing tool call interception hooks that block policy-violating actions (e.g., refunds exceeding &#36;500) and redirect to alternative workflows (e.g., human escalation)", and Exercise 1 asks you to build the same kind of hook. In the illustrative example below (our code, built from the docs' hook API), the `deny` stops the call, and the reason, which Claude reads, points it to the escalation path. The server name `billing`, the tool `process_refund` and the `amount` field are illustrative.

=== "Python"

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

=== "TypeScript"

    ```typescript
    import { HookCallback, PreToolUseHookInput } from "@anthropic-ai/claude-agent-sdk";

    const REFUND_LIMIT = 500;

    const enforceRefundLimit: HookCallback = async (input, toolUseID, { signal }) => {
      const preInput = input as PreToolUseHookInput;
      const toolInput = preInput.tool_input as Record<string, unknown>;
      const amount = (toolInput?.amount as number) ?? 0;
      if (amount > REFUND_LIMIT) {
        return {
          // Top-level field: message shown to the user
          systemMessage: "Refund above limit routed to a human agent.",
          hookSpecificOutput: {
            hookEventName: preInput.hook_event_name,
            permissionDecision: "deny",
            // Shown to Claude for "deny", so it can switch to the escalation workflow
            permissionDecisionReason:
              "Refunds over $500 require human approval. Escalate to a human agent with a structured handoff summary instead."
          }
        };
      }
      return {};
    };

    const options = {
      hooks: {
        PreToolUse: [{ matcher: "mcp__billing__process_refund", hooks: [enforceRefundLimit] }]
      }
    };
    ```

The same shape covers the CCDV-F Claude Hooks case, a hook that denies destructive actions. Because a hook checks the call itself, it holds even when injected text has persuaded the model; how CCDV-F sample 2 turns that into an answer on prompt injection is in [Limits of each control](#limits-of-each-control).

### Normalizing tool results before the model sees them

CCAR-F 1.5-S1 asks for `PostToolUse` hooks that normalize heterogeneous formats (Unix timestamps, ISO 8601, numeric status codes) coming from different MCP tools. Match all MCP tools with the regex `^mcp__`, read `tool_response`, and return the cleaned value as `updatedToolOutput`. The example is illustrative (our code, using the docs' field names); `normalize_timestamps` and `normalizeTimestamps` stand for your own conversion code.

=== "Python"

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

=== "TypeScript"

    ```typescript
    import { HookCallback, PostToolUseHookInput } from "@anthropic-ai/claude-agent-sdk";

    const normalizeMcpOutput: HookCallback = async (input, toolUseID, { signal }) => {
      const postInput = input as PostToolUseHookInput;
      const normalized = normalizeTimestamps(postInput.tool_response); // your code
      return {
        hookSpecificOutput: {
          hookEventName: postInput.hook_event_name,
          updatedToolOutput: normalized
        }
      };
    };

    const options = {
      hooks: {
        PostToolUse: [{ matcher: "^mcp__", hooks: [normalizeMcpOutput] }]
      }
    };
    ```

Three limits to remember. `updatedToolOutput` changes only what Claude sees: the tool has already run, so any side effect has already happened. A replacement for a built-in tool must match that tool's output shape, while MCP output is not schema-validated. And if you only want to annotate rather than replace, return `additionalContext` instead, which the docs recommend writing as factual statements rather than imperative instructions.

### Shell hooks from settings files

With the `project` setting source enabled, hooks in `.claude/settings.json` run in SDK sessions too. They follow the Claude Code format of event, matcher group and handler, communicate through exit codes and stdout, and are covered in full in [Hooks](claude-code-workflows.md#hooks). The rules that matter here:

- Exit code 2 is a blocking error on events that can block, and even a JSON `permissionDecision: "allow"` cannot override it. Without valid JSON on stdout, exit code 1 is a non-blocking error, so a policy hook that exits 1 lets the action through. A hook that cannot start (a mistyped script path, for example) is also non-blocking, which leaves the gate silently disabled.
- In `-p` and SDK sessions Claude Code never shows the workspace trust dialog and treats the folder as trusted, so hooks committed in a repository's `.claude/settings.json` run. Before scripting `claude -p` over a repository you did not write, the docs say to review its `.claude/` settings files, start with `--bare`, or turn hooks off for that run with `--settings '{"disableAllHooks": true}'`. In the SDK, `settingSources` decides whether project settings, and so their hooks, load at all: `settingSources: []` limits the session to programmatic configuration, though managed policy settings are still read.

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

In this docs example the `matcher` narrows to Bash calls and the `if` field (one permission rule, evaluated only on tool events) narrows further to Bash subcommands matching `rm *`, so the script spawns only when both match. Setting `args` (here an empty list) runs the command in exec form with no shell, which the docs recommend whenever a path placeholder such as `${CLAUDE_PROJECT_DIR}` is used. The docs call the `if` filter best-effort (when Claude Code cannot tell which commands a Bash input runs, it runs the hook anyway) and say to use the permission system rather than a hook for a hard allow or deny. That is a point about Claude Code's own permission rules; the exam's contrast is between hooks and prompt instructions, where the hook is the answer.

### Timeouts and failure modes

A timed-out callback is canceled and its output discarded; the session continues rather than hanging. Set a longer `timeout` on the `HookMatcher` if a callback needs more time.

| Situation | Behavior |
|---|---|
| Default callback timeout | 600 seconds for most events; 30 seconds for `UserPromptSubmit`, `PreModelSwitch`, `PostModelSwitch`; 10 seconds for `MessageDisplay`; `SessionEnd` callbacks run during shutdown under a 1.5-second default budget |
| An SDK `PreToolUse` callback times out | The tool call does not run; Claude receives a tool result saying the hook did not respond before its timeout (or another hook's explicit deny, if one returned it), and the turn continues |
| An SDK `UserPromptSubmit` or `UserPromptExpansion` callback times out | The prompt is blocked, never let through unscreened |
| An SDK `Stop` or `SubagentStop` callback times out | It counts as no decision, so the agent stops unless another hook on the event blocks |
| A shell, HTTP or MCP-tool `PreToolUse` hook times out | The tool call is not blocked, so a stalled shell hook is not a gate |
| The agent hits `max_turns` | Hooks may not fire, because the session ends first |
| A `UserPromptSubmit` hook spawns subagents | It can loop forever if those subagents trigger the same hook; scope it to the top-level session |

Three of these rules are version-dependent. Before Claude Code v2.1.210, a timed-out `PreToolUse` callback was reported to Claude as a user rejection, which made unattended sessions stop and wait for input. Before v2.1.208, a timed-out `UserPromptSubmit` or `UserPromptExpansion` callback ended the query with `error_during_execution`. Before v2.1.273, a timed-out `Stop` or `SubagentStop` callback counted as a failed hook run, and Claude Code discarded the decisions of the other hooks on that event.

### Decide and avoid

- If a business rule must hold every time (a refund ceiling, identity verified before a financial operation), enforce it in a `PreToolUse` hook or a [prerequisite gate](#a-prerequisite-gate); not a system-prompt rule, few-shot examples or a routing classifier, the three distractors of CCAR-F sample question 1 (their rationales are under [The enforcement ladder](#the-enforcement-ladder) and [Routing](#routing)).
- If the problem is the shape of data coming back from tools, use `PostToolUse` with `updatedToolOutput`; not a prompt asking Claude to interpret three date formats.
- If the model should learn why it was stopped, set `permissionDecisionReason` on the deny; `systemMessage` reaches the user, not Claude.
- If a check must run on auto-approved tools too, use a hook; a `canUseTool` callback never sees auto-approved calls.
- If a rule depends on the call's arguments (a file path, a refund amount), inspect `tool_input` inside the callback; a matcher sees only the tool name.
- If a shell policy hook must block, exit 2; exit 1 without a valid JSON decision lets the action through.
- If you log or send metrics, return `async: true`; not for anything that must block.

## Sessions, resumption and forking

*Tested in: CCAR-F 1.3, 1.7 · CCDV-F Claude Code Operation, Claude Application Design*

"A session is the conversation history the SDK accumulates while your agent works" ([Agent SDK sessions](https://code.claude.com/docs/en/agent-sdk/sessions)): the prompt, every tool call and result, and every response, written to disk automatically under `~/.claude/projects/<encoded-cwd>/*.jsonl` (under `$CLAUDE_CONFIG_DIR/projects/` if that variable is set). Sessions persist the conversation, not the filesystem. Inside one `query()` call the agent already takes as many turns as it needs; session handling matters when a later prompt should share context with an earlier one, because each `query()` call starts fresh unless you tell it otherwise.

### Continue, resume and fork

| Operation | What it does | Python | TypeScript | CLI |
|---|---|---|---|---|
| Continue | Picks up the most recent session in the current directory; you track nothing | `continue_conversation=True` | `continue: true` | `claude --continue` |
| Resume | Picks up one specific session; you track its ID | `resume=session_id` | `resume: sessionId` | `claude --resume <id or name>` |
| Fork | Creates a new session that starts with a copy of the original's history; the original stays unchanged | `resume=...` plus `fork_session=True` | `resume` plus `forkSession: true` | `--fork-session` with `--resume` or `--continue`; `/branch` inside a session |
| Keep nothing on disk | No transcript written, so nothing to resume later | Set `CLAUDE_CODE_SKIP_PROMPT_HISTORY` in the `env` option to suppress transcript writes | `persistSession: false`: the session exists only in memory for the call | `--no-session-persistence` (print mode only), or `CLAUDE_CODE_SKIP_PROMPT_HISTORY` in any mode |

For multi-turn chat inside one process you rarely pass IDs: `ClaudeSDKClient` holds the session in Python, and TypeScript passes `continue: true` on later `query()` calls (see [`ClaudeSDKClient` and streaming input](#claudesdkclient-and-streaming-input)). The experimental TypeScript V2 session API with `createSession()` was removed in TypeScript Agent SDK 0.3.142. To branch from an earlier point rather than the end, both SDKs can load a session only up to a given message UUID: `resume_session_at` in Python (Agent SDK 0.2.137 or later) and `resumeSessionAt` in TypeScript, used with `resume` and usually with fork mode.

!!! tip "Exam wording: fork_session"

    The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) writes `fork_session` (1.7-K2, 1.7-S2 and its technology appendix; 1.3-K4 calls it "Fork-based session management"), which is the Python option name. The same feature is `forkSession` in TypeScript and `--fork-session` on the CLI. The guide's `--resume <session-name>` (1.7-K1, 1.7-S1) is current CLI syntax: name a session with `claude -n auth-refactor` at startup or `/rename` mid-session, then `claude --resume auth-refactor`.

### Capture the ID, resume, fork

Every result message carries `session_id`, whatever its subtype, so you can resume even after an error. TypeScript also exposes it earlier as a direct field on the `init` system message, which is how the docs read a fork's new ID; in Python it is nested inside `SystemMessage.data`. A single-shot `query()` raises after yielding an error result, so catch the error before resuming. The resume and fork calls from the sessions docs (trimmed; the docs' fork example wraps each loop in error handling, while its resume example does not):

=== "Python"

    ```python
    import asyncio
    from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

    session_id = "..."  # captured from an earlier result message

    async def main():
        # Resume: continue the earlier session with full context
        async for message in query(
            prompt="Now implement the refactoring you suggested",
            options=ClaudeAgentOptions(
                resume=session_id,
                allowed_tools=["Read", "Edit", "Write", "Glob", "Grep"],
            ),
        ):
            if isinstance(message, ResultMessage) and message.subtype == "success":
                print(message.result)

        # Fork: branch from session_id into a new session; the original is untouched
        forked_id = None
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

    asyncio.run(main())
    ```

=== "TypeScript"

    ```typescript
    import { query } from "@anthropic-ai/claude-agent-sdk";

    const sessionId = "..."; // captured from an earlier result message

    // Resume: continue the earlier session with full context
    for await (const message of query({
      prompt: "Now implement the refactoring you suggested",
      options: {
        resume: sessionId,
        allowedTools: ["Read", "Edit", "Write", "Glob", "Grep"]
      }
    })) {
      if (message.type === "result" && message.subtype === "success") {
        console.log(message.result);
      }
    }

    // Fork: branch from sessionId into a new session; the original is untouched
    let forkedId: string | undefined;
    for await (const message of query({
      prompt: "Instead of JWT, outline how OAuth2 would work for the auth module",
      options: {
        resume: sessionId,
        forkSession: true,
        maxTurns: 5
      }
    })) {
      if (message.type === "system" && message.subtype === "init") {
        forkedId = message.session_id; // The fork's ID, distinct from sessionId
      }
    }
    ```

The same moves from the CLI, including a scripted follow-up in non-interactive mode:

```bash
claude -n auth-refactor                  # name the session at startup
claude --resume auth-refactor            # resume a named session
claude --continue --fork-session         # branch the most recent session under a new ID
claude --resume abc123 --fork-session
session_id=$(claude -p "Start a review" --output-format json | jq -r '.session_id')
claude -p "Continue that review" --resume "$session_id"
```

What a resumed session brings back: the full history, including tool calls and results, plus, with some exceptions, the model and the agent, and (when resumed from a terminal without `-p`) the permission mode. Three CLI behaviors catch people out. Sessions created with `claude -p` or the Agent SDK are left out of the session picker and out of `claude --continue`, though you can still resume them by ID, and `claude -p --continue` does include them. Resuming the same session in two terminals without forking interleaves both into one transcript. Transcripts are kept for 30 days by default (`cleanupPeriodDays`).

### What a fork does not isolate

"Forking branches the conversation history, not the filesystem." ([Agent SDK sessions](https://code.claude.com/docs/en/agent-sdk/sessions)). If a forked agent edits files, those changes are real and visible to any session working in the same directory. In our reading, that is fine for CCAR-F 1.7-S2's use (comparing two testing strategies or refactoring approaches from one shared codebase analysis) as long as the branches reason and propose. If both branches must write, give each its own working directory: the hosting page says one agent session maps to one subprocess that by default inherits your application's working directory, and to pass a distinct `cwd` when sessions need separate filesystems. To branch and revert file changes instead, the sessions docs point to file checkpointing.

Sessions and file checkpoints are two separate undo systems. File checkpointing tracks changes made through `Write`, `Edit` and `NotebookEdit`, not through `Bash`, and rewinding files does not rewind the conversation. A session fork does not rewind files either.

### Resume, or start fresh with a summary

CCAR-F 1.7-K4 asks "Why starting a new session with a structured summary is more reliable than resuming with stale tool results" ([CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The mechanics explain it. A resumed session restores the full history, including earlier tool calls and results, so a file read before a refactor still shows the old code in context (our inference from that rule). Context that mid-session hooks such as `PostToolUse` or `UserPromptSubmit` injected is replayed from the transcript rather than re-run, so values such as timestamps or commit SHAs go stale. `SessionStart` hooks, by contrast, run again on resume (with `source` set to `"resume"`, or `"fork"` with `--fork-session`), so they can refresh their context. Checkpoints normally do not capture edits made outside Claude Code, and a `FileChanged` hook's message is a terminal notification for the user that does not reach the SDK message stream. Claude Code's Edit tool does notice a file that changed on disk since it was read, but only when an edit is attempted. For any reasoning built on earlier reads, telling the resumed agent which files changed (1.7-K3, 1.7-S4) remains your job.

The sessions docs make a related recommendation for moving work between hosts: capture the results you need (analysis output, decisions, file diffs) as application state and pass them into a fresh session's prompt. The docs describe this as often more dependable than shipping transcript files around.

Our decision table, built from the guide's 1.7 objectives and the sessions docs:

| Situation | Choose | Why |
|---|---|---|
| Follow-up on a finished task, nothing changed since | Resume | CCAR-F 1.7-S3: prior context is mostly valid, so the agent acts on its analysis without re-reading files |
| The run ended with `error_max_turns` or `error_max_budget_usd` | Resume with a higher limit | Every result carries `session_id`, including error results |
| A few files changed since the analysis | Resume and name the changed files for targeted re-analysis | CCAR-F 1.7-S4: avoids a full re-exploration |
| Much has changed, or earlier tool results are no longer true | New session with a structured summary of the findings | CCAR-F 1.7-K4 and 1.7-S3: stale tool results mislead |
| Two approaches to compare from one baseline | Fork | CCAR-F 1.7-K2 and 1.7-S2 |
| Next step runs on another host or an ephemeral container | `SessionStore`, move the `.jsonl` file, or a fresh session with captured results | Session files are local to the machine that wrote them |

### Sessions across hosts

Session files live on the machine that created them. On that machine you can resume from any working directory, because Claude Code searches beyond the current project directory for the ID (before v2.1.223 the lookup covered only the current project directory and its git worktrees). To resume on another host (CI workers, ephemeral containers, serverless), attach a `SessionStore` adapter (`sessionStore` / `session_store`) so transcripts are mirrored to your backend, and resume from a `cwd` matching the original run's, since the store's lookup key derives from the working directory. Alternatively, restore the `<session-id>.jsonl` file inside any directory under `~/.claude/projects/` on the new host before calling `resume`, or skip resume and start a fresh session with captured results. What a `SessionStore` does not carry, and how its writes can fail, is covered with the hybrid hosting pattern that depends on it in [Self-hosting the Agent SDK](#self-hosting-the-agent-sdk). Both SDKs also expose `list_sessions()` / `listSessions()`, `get_session_messages()` / `getSessionMessages()`, and `get_session_info()`, `rename_session()` and `tag_session()` (camelCase in TypeScript) for building session pickers and cleanup jobs.

Cost accounting follows the session too. Claude Code saves the session's totals to the transcript when the process exits normally and restores them when a later call resumes or forks the session, so a resumed call reports the whole session's spend, not just its own. Read the latest result for the session total; summing `total_cost_usd` across resumed calls double-counts. (Before v2.1.277, a session resumed through the SDK or `claude -p` started its totals at zero.) `max_budget_usd` / `maxBudgetUsd` counts only the current call's spend: restored totals do not count against it. Details are in [Cost and usage tracking](claude-api.md#cost-and-usage-tracking).

### Decide and avoid

- If the exam says continue a specific named investigation, the answer is `--resume <session-name>`; `--continue` only finds the most recent session in the directory.
- If you need two alternatives from one analysis, fork; resuming the same session twice in parallel interleaves both into one transcript.
- If files changed after the analysis, tell the resumed session which ones, or start fresh with a summary; resuming silently brings back stale tool results.
- If two forks will both edit files, separate their working directories; a fork copies history, not the disk.
- If your SDK session is missing from `claude --continue`, that is by design: resume it by ID, or use `claude -p --continue`.

## Permissions and enforcement

*Tested in: CCAR-F 1.4, 1.5, 2.3 · CCDV-F Guardrails and Safe Deployment, Claude Hooks, Tool Implementation, Claude Code Operation · CCAR-P 3.1, 5.1, 5.3*

CCAR-F 1.4-K1 draws the central line: "The difference between programmatic enforcement (hooks, prerequisite gates) and prompt-based guidance for workflow ordering". The next bullet, 1.4-K2, gives the reason it matters: "When deterministic compliance is required (e.g., identity verification before financial operations), prompt instructions alone have a non-zero failure rate" ([CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The product docs say the same thing from the other side: "Permission rules are enforced by Claude Code, not by the model" ([Claude Code permissions](https://code.claude.com/docs/en/permissions)), and CLAUDE.md content is treated as context, not enforced configuration.

### The enforcement ladder

Our ordering, strongest first. Each rung is a different mechanism, and a real deployment usually combines several.

| Rung | Mechanism in the Agent SDK | What it guarantees | Where it stops |
|---|---|---|---|
| 1. Remove the capability | Leave a built-in out of `tools` (that option does not affect MCP tools), leave any tool out of an AgentDefinition's `tools`, or put its bare name in `disallowedTools` (for example `mcp__support__process_refund`) | Claude does not see the tool and cannot attempt it, so neither a mistake nor an injected instruction can call it | Too coarse when the role needs the tool for some calls |
| 2. Isolate the environment | Container or sandbox, only the directories the agent needs, network through a proxy, credentials injected by a proxy outside the agent's boundary | Holds even if the agent is compromised through prompt injection, and does not depend on command text | Needs infrastructure; see [Deployment models](#deployment-models) |
| 3. Deny rules | Scoped `disallowedTools` such as `Bash(rm *)`, or `deny` rules in settings | Matching calls are blocked in every mode, including `bypassPermissions` | A Bash rule matches the command as written: `Bash(curl *)` does not stop `/usr/bin/curl` or `sh -c 'curl ...'` |
| 4. Hook or prerequisite gate | A `PreToolUse` callback that checks the call and your own session state | Runs before every other step, on every call, auto-approved or not; its deny holds in `bypassPermissions` | Only as good as your code; a timed-out shell hook from a settings file does not block (see [Timeouts and failure modes](#timeouts-and-failure-modes)) |
| 5. Runtime approval | Permission mode, `ask` rules, the `canUseTool` callback | A person, or your policy code, decides per call | `canUseTool` is never consulted for calls an earlier step already approved |
| 6. Instructions | System prompt, CLAUDE.md, few-shot examples | Shapes what Claude usually does | Probabilistic: a non-zero failure rate |

The rule behind these answers (our summary of CCAR-F 1.4-K2, 1.5-S3 and the sample rationales below): if something must always happen, enforce it in code, with a hook or a prerequisite gate. If it should usually happen, a prompt instruction is proportionate. Two of the three CCAR-F customer-support sample questions (1 to 3) show both halves. Sample question 1 (the agent skips `get_customer` in 12% of cases) is answered with a programmatic prerequisite, because "Options B and C rely on probabilistic LLM compliance, which is insufficient when errors have financial consequences." Sample question 3 (escalation calibrated badly) is answered with explicit criteria and few-shot examples in the prompt, which the rationale calls "the proportionate first response before adding infrastructure." Both quotes are from the [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf).

The top rung is also the Professional exam's least-privilege answer. CCAR-P sample 1 removes the refund and delete tools a support role never uses: "Least privilege means removing capabilities the role does not require, eliminating the attack surface rather than monitoring or guarding it." Its rationale adds that "Logging (A) and confirmations (C) are detective/compensating controls, not removal of unnecessary privilege" ([CCAR-P guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). More on this in [Least privilege for tools and agents](security-and-governance.md#least-privilege-for-tools-and-agents).

### How the SDK evaluates a tool call

```text
tool request
  1. Hooks (PreToolUse)   deny -> blocked.  "allow" does NOT skip steps 2 and 3
  2. Deny rules           match -> blocked, even in bypassPermissions
  3. Ask rules            match -> sent to canUseTool, even in bypassPermissions
  4. Permission mode      bypassPermissions approves (except critical-path rm/rmdir);
                          acceptEdits approves listed file operations; plan sends
                          file edits and shell writes to canUseTool; others fall through
  5. Allow rules          match -> approved (canUseTool never runs)
  6. canUseTool           your callback decides; in dontAsk this step denies instead
```

Two consequences follow. A check that must run on every call belongs in a `PreToolUse` hook: "hooks run before every other step, and a hook deny applies even in `bypassPermissions` mode" ([Agent SDK permissions](https://code.claude.com/docs/en/agent-sdk/permissions)). And a check you put in `canUseTool` is silently skipped for any tool that an allow rule, `acceptEdits` or `bypassPermissions` already approved. Bare-name deny rules are special: they remove the tool from Claude's context before evaluation starts, so only scoped rules are checked at step 2.

### Permission modes

| Mode | Behavior |
|---|---|
| `default` | No mode-based approvals; calls that need approval and match no allow rule go to `canUseTool` |
| `dontAsk` | Anything that would prompt is denied; pre-approved calls run, and so do calls that need no approval in `default`. Tools that require user interaction (such as `AskUserQuestion`), connector tools your organization set to `ask`, and critical-path `rm`/`rmdir` removals are denied even when pre-approved; `canUseTool` is never called |
| `acceptEdits` | File edits and filesystem commands (`mkdir`, `touch`, `rm`, `rmdir`, `mv`, `cp`, `sed`) inside the working directory or `additionalDirectories` are approved automatically; MCP tools and other Bash commands are not |
| `bypassPermissions` | Tools run without prompts, except actions no mode auto-approves; deny rules, `ask` rules and hook denies still apply |
| `plan` | Claude explores and plans without editing source files; edits always go to `canUseTool` |
| `auto` | A model classifier approves or denies permission prompts; available only when your account meets auto mode's requirements (organization setting, model and provider) |

- In TypeScript, `permissionMode: 'bypassPermissions'` also requires `allowDangerouslySkipPermissions: true`. Reserve the mode for CI, containers or other isolated environments; it cannot be used as root on Unix.
- `set_permission_mode()` (Python) or `setPermissionMode()` (TypeScript) changes the mode mid-session, so you can start restrictive and loosen after reviewing Claude's approach.
- A subagent runs in the parent's mode unless its AgentDefinition sets `permissionMode` and the parent is in `default`, `dontAsk` or `plan`. A definition never applies `bypassPermissions`: a subagent runs in that mode only when the parent does (this exception requires Claude Code v2.1.267 or later).

### `allowedTools` pre-approves; it does not restrict

The Python reference describes `allowed_tools` as "Tools to auto-approve without prompting. This does not restrict Claude to only these tools." ([Python reference](https://code.claude.com/docs/en/agent-sdk/python)).

| Option | Layer | Effect |
|---|---|---|
| `allowed_tools=["Read", "Grep"]` | Permission | `Read` and `Grep` run without a prompt; other tools still exist and go through the permission flow |
| `disallowed_tools=["Bash"]` | Availability | The Bash definition is removed; Claude cannot see or attempt it |
| `disallowed_tools=["Bash(rm *)"]` | Permission | Bash stays; calls matching `rm *` as written are denied in every mode; other Bash calls, including `/bin/rm`, fall through to the permission mode |
| `disallowed_tools=["*"]` | Availability | Every tool definition is removed; deny rules accept globs, and `"mcp__*"` matches every MCP tool on every server |
| `tools=["Read", "Grep"]` | Availability | Only the listed built-ins are in context (MCP tools unaffected) |
| `allowed_tools=["Read"]` with `bypassPermissions` | Permission | Every tool is still approved, including `Bash`, `Write` and `Edit` |
| `allowed_tools=["*"]` or `["mcp__*"]` | None | Ignored with a startup warning; globs work only after a literal server prefix such as `mcp__puppeteer__*` |

For a locked-down agent, pair the allow list with `dontAsk`: listed tools run, apart from the exceptions in the `dontAsk` row above, and every other call that would prompt is denied. Calls that need no approval in `default` mode (read-only Bash commands, file reads in the working directories, tools such as `Agent`) still run. To put a tool fully out of reach, add its bare name to `disallowedTools`.

=== "Python"

    ```python
    options = ClaudeAgentOptions(
        allowed_tools=["Read", "Glob", "Grep"],
        permission_mode="dontAsk",
    )
    ```

=== "TypeScript"

    ```typescript
    const options = {
      allowedTools: ["Read", "Glob", "Grep"],
      permissionMode: "dontAsk"
    };
    ```

### Human approval with `canUseTool`

`canUseTool` runs only when the permission flow falls through to a prompt, and it pauses the agent until you answer. It also receives `AskUserQuestion` clarifying questions.

| Response | Python | TypeScript |
|---|---|---|
| Allow | `PermissionResultAllow(updated_input=...)` | `{ behavior: "allow", updatedInput }` |
| Deny | `PermissionResultDeny(message=...)`, optionally `interrupt=True` | `{ behavior: "deny", message }`, optionally `interrupt: true` |

When allowing, the tool runs with the input Claude requested unless you return a modified input (`updated_input` in Python, `updatedInput` in TypeScript).

When a human answer may take a long time, do not hold the callback open. Register a `PreToolUse` hook that returns `permissionDecision: "defer"`: the query ends, the process exits with `stop_reason: "tool_deferred"`, and you resume the persisted session later. The hooks reference adds two conditions: Claude Code honors `defer` only in non-interactive (`-p`) mode, which is how SDK applications run it, and only when Claude makes a single tool call in the turn; with several calls, `defer` is ignored and the tool goes through the normal permission flow. What the human needs in the handoff is covered in [Escalation and ambiguity](evaluation-and-reliability.md#escalation-and-ambiguity).

### A prerequisite gate

CCAR-F 1.4-S1 describes the pattern: "Implementing programmatic prerequisites that block downstream tool calls until prerequisite steps have completed (e.g., blocking process_refund until get_customer has returned a verified customer ID)" ([CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Two hooks do it: a `PostToolUse` hook records that verification succeeded for this session, and a `PreToolUse` hook denies `lookup_order` and `process_refund` until it has, which is the shape of the correct option in CCAR-F sample question 1. `PostToolUse` fires only after a tool call succeeds. The tool names come from the guide's customer-support scenario; the `support` server name and the code are illustrative. A matcher made only of letters, digits, underscores and `|` is a list of exact tool names, and the deny reason is shown to Claude, so it can call `get_customer` and try again.

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
                "permissionDecisionReason": "Call get_customer and verify the customer before looking up orders or issuing a refund.",
            }
        }
    return {}

options = ClaudeAgentOptions(
    hooks={
        "PostToolUse": [
            HookMatcher(matcher="mcp__support__get_customer", hooks=[record_verification])
        ],
        "PreToolUse": [
            HookMatcher(
                matcher="mcp__support__lookup_order|mcp__support__process_refund",
                hooks=[require_verification],
            )
        ],
    }
)
```

Two refinements for production. The `PostToolUse` input carries `tool_response`, so inspect it and record the session only when the lookup actually returned a verified customer. And the set lives in your process's memory: if sessions can resume on another host, keep the gate state wherever you keep session state (see [Sessions across hosts](#sessions-across-hosts)). The TypeScript registration has the same shape as in [Hooks in the SDK](#hooks-in-the-sdk).

### Limits of each control

- A hook `allow` does not skip deny or ask rules, and cannot approve `rm` or `rmdir` removals that target a critical path.
- Tool annotations such as `readOnlyHint` and `destructiveHint` block nothing (see [Annotations and parallel calls](#annotations-and-parallel-calls)).
- A Bash deny rule is not a security boundary around the program; for enforcement that does not depend on command text, use sandboxing or network isolation.
- A timed-out `command`, `http` or `mcp_tool` hook from a settings file does not block the tool call, while a timed-out SDK `PreToolUse` callback does stop it (all timeout rules are in [Timeouts and failure modes](#timeouts-and-failure-modes)).
- A hook or deny rule acts on the tool call itself, whatever text persuaded Claude to make it. That is why the correct option in CCDV-F sample 2 (prompt injection in a summarized web page) is to treat retrieved content as untrusted and "use guardrails or hooks so injected instructions cannot trigger sensitive actions" ([CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)); a system-prompt line asking users not to inject is, in the rationale's words, "not an enforceable control".

Claude Managed Agents replaces modes and `canUseTool` with a per-tool `permission_policy` of `always_allow`, `always_ask` or `auto` (the server evaluates each call), with `user.tool_confirmation` events answering calls that pause for approval. The agent toolset defaults to `always_allow`, MCP toolsets to `always_ask`, and custom tools are not governed by policies because your application runs them. A policy only controls when an enabled tool runs; to remove a tool entirely, disable it. See [Deployment models](#deployment-models).

### Decide and avoid

- If a required tool order is skipped some of the time, add a programmatic prerequisite; not a stronger system prompt or few-shot examples (probabilistic), and not a routing classifier, which the [sample question 1 rationale](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says "addresses tool availability rather than tool ordering".
- If a role never needs a capability, remove the tool; logging and confirmation prompts leave the privilege in place.
- If you run `bypassPermissions`, block what must never happen with `disallowedTools` or hooks; `allowedTools` does not limit that mode.
- If a check must see every call, put it in a `PreToolUse` hook, not in `canUseTool`.
- If an agent must never reach a network destination or read a secret, isolate it (sandbox, proxy, credentials outside the boundary) rather than relying on Bash pattern rules.

## Custom tools in the SDK

*Tested in: CCAR-F 2.2, 2.3, 2.4 · CCDV-F Tool Implementation, MCP Server Development, Agentic Customization · CCAR-P 3.7*

In the Agent SDK a custom tool is an MCP tool served by an in-process MCP server that you create in your own code. The server "runs in-process inside your application, not as a separate process" ([custom tools](https://code.claude.com/docs/en/agent-sdk/custom-tools)), so your handler can reach your database, your APIs or your domain logic directly. This section covers defining such a tool, naming and permitting it, returning errors, annotations, adding external MCP servers, and when a custom tool is the right choice at all.

Every tool has four parts: a name, a description (Claude reads it to decide when to call the tool), an input schema and an async handler. In TypeScript the schema is always a Zod schema and the handler's arguments are typed from it. In Python it is a dict of names to types, like `{"latitude": float}`, which the SDK converts to JSON Schema, or a full JSON Schema dict. Two Python details matter: the dict form treats every key as required, and it does not support enums, so use the full JSON Schema dict for enums, ranges, optional fields or nested objects.

The example below is the docs' weather tool, abridged: the handler body is elided and the annotation comes from the docs' annotations example.

=== "Python"

    ```python
    import asyncio
    from typing import Any
    from claude_agent_sdk import (
        tool,
        create_sdk_mcp_server,
        query,
        ClaudeAgentOptions,
        ToolAnnotations,
    )

    @tool(
        "get_temperature",
        "Get the current temperature at a location",
        {"latitude": float, "longitude": float},
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    async def get_temperature(args: dict[str, Any]) -> dict[str, Any]:
        ...
        return {"content": [{"type": "text", "text": "Temperature: ..."}]}

    weather_server = create_sdk_mcp_server(
        name="weather",
        version="1.0.0",
        tools=[get_temperature],
    )

    async def main():
        options = ClaudeAgentOptions(
            mcp_servers={"weather": weather_server},
            allowed_tools=["mcp__weather__get_temperature"],
        )
        async for message in query(
            prompt="What's the temperature in San Francisco?",
            options=options,
        ):
            ...

    asyncio.run(main())
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
      { annotations: { readOnlyHint: true } }
    );

    const weatherServer = createSdkMcpServer({ name: "weather", version: "1.0.0", tools: [getTemperature] });

    for await (const message of query({
      prompt: "What's the temperature in San Francisco?",
      options: { mcpServers: { weather: weatherServer }, allowedTools: ["mcp__weather__get_temperature"] }
    })) { /* ... */ }
    ```

### Naming and access

The key you use in `mcpServers` becomes the server segment of each tool's full name, `mcp__{server_name}__{tool_name}`. That full name is what you list in `allowedTools`, what hook matchers see, and the name on the `tool_use` blocks Claude emits. `mcp__weather__*` covers every tool on the server. `default` and `acceptEdits` modes do not auto-approve MCP tools, so without an allow rule (or approval from your `canUseTool` callback) Claude sees that the tools are available but cannot call them. `bypassPermissions` would approve them without a rule, but it also removes most other prompts, which is broader than necessary; a wildcard in `allowedTools` grants exactly one server.

| Option | Layer | Effect on custom tools |
|---|---|---|
| `allowedTools: ["mcp__weather__get_temperature"]` | Permission | The tool runs without a prompt |
| `allowedTools: ["mcp__weather__*"]` | Permission | Every tool from that server runs without a prompt |
| `tools: []` | Availability | All built-ins are removed; Claude can use only your MCP tools |
| `tools: ["Read", "Grep"]` | Availability | Only those built-ins remain; MCP tools are unaffected |
| `permissionMode: "acceptEdits"` | Permission | Does not auto-approve MCP tools |

A scoped `disallowedTools` rule blocks matching calls but leaves the tool visible, so Claude may waste a turn trying it. For how bare and scoped `disallowedTools` entries and the `tools` option differ, see [`allowedTools` pre-approves; it does not restrict](#allowedtools-pre-approves-it-does-not-restrict).

### Return values and errors

A handler returns `content` (required), and optionally `structuredContent` and an error flag: `isError: true` in TypeScript, `"is_error": True` in the Python return dict. A handler error never stops the agent loop; in both cases Claude can retry, try a different tool, or explain the failure.

| What happens | What Claude sees |
|---|---|
| The handler throws an uncaught exception | The in-process server converts it to an error result with the raw exception message; the loop continues |
| The handler catches the error and returns the error flag | The message you composed, with whatever context the raw exception lacked |

```python
import json
import httpx
from typing import Any
from claude_agent_sdk import tool

@tool(
    "fetch_data",
    "Fetch data from an API",
    {"endpoint": str},  # Simple schema
)
async def fetch_data(args: dict[str, Any]) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(args["endpoint"])
            if response.status_code != 200:
                # Return the failure as a tool result so Claude can react to it.
                # is_error marks this as a failed call rather than odd-looking data.
                return {
                    "content": [
                        {
                            "type": "text",
                            "text": f"API error: {response.status_code} {response.reason_phrase}",
                        }
                    ],
                    "is_error": True,
                }

            data = response.json()
            return {"content": [{"type": "text", "text": json.dumps(data, indent=2)}]}
    except Exception as e:
        # Composes the message Claude reads. An uncaught exception would
        # reach Claude as the raw str(e) with no context.
        return {
            "content": [{"type": "text", "text": f"Failed to fetch data: {str(e)}"}],
            "is_error": True,
        }
```

This is the SDK side of CCAR-F 2.2-K1 ("The MCP isError flag pattern for communicating tool failures back to the agent", [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). What to put inside the error (category, whether a retry can help, a readable explanation) is taught in [MCP errors and structured results](tool-use-and-mcp.md#mcp-errors-and-structured-results).

Two details about `structuredContent`. When it is set, Claude receives the JSON plus any image or resource blocks, and text blocks in `content` are not forwarded, because they are assumed to duplicate it. The Python `@tool` decorator forwards only `content` and `is_error`, so a Python tool that needs `structuredContent` must run in a standalone MCP server.

### Annotations and parallel calls

Annotations are optional Boolean hints: the fifth argument to `tool()` in TypeScript (`{ annotations: {...} }`), or the `annotations=ToolAnnotations(...)` keyword on the Python `@tool` decorator.

| Annotation | Default | Effect |
|---|---|---|
| `readOnlyHint` | `false` | `true` lets the SDK run the tool in parallel with other read-only tools |
| `destructiveHint` | `true` | Informational only |
| `idempotentHint` | `false` | Informational only |
| `openWorldHint` | `true` | Informational only |

Custom tools default to sequential execution, like the state-changing built-ins `Edit`, `Write` and `Bash`; read-only tools (`Read`, `Glob`, `Grep` and MCP tools marked read-only) can run concurrently. Annotations are "metadata, not enforcement" ([custom tools](https://code.claude.com/docs/en/agent-sdk/custom-tools)): a tool marked `readOnlyHint: true` can still write to disk if its handler does, and marking a tool `destructiveHint` blocks nothing. Keep each annotation accurate to its handler, and put the guard in a hook or a permission rule (see [Permissions and enforcement](#permissions-and-enforcement)).

### External MCP servers in the same agent

MCP servers can run as local processes, connect over HTTP, or run in-process as the SDK MCP servers above. The SDK's rule of thumb: a command to run means stdio, a URL means HTTP or SSE, and tools you build in your own code mean an SDK MCP server. You can pass servers in code or load a project `.mcp.json` through the `project` setting source; `strict_mcp_config=True` (TypeScript: `strictMcpConfig: true`) makes the agent use only the servers you pass in code, ignoring project `.mcp.json`, user settings, plugin-provided servers and claude.ai connectors.

The docs' example connects to the Claude Code documentation server over HTTP. A streamable HTTP server passed in code uses `"type": "http"` (JSON config files also accept the alias `"streamable-http"`), because the SDKs' `McpHttpServerConfig` type declares only `"http"`; an SSE server uses `"type": "sse"`.

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

Limits as of September 2026:

- **Connection timeout:** MCP server connections time out after 30 seconds by default; raise the limit with `MCP_TIMEOUT` (in milliseconds). `MCP_TOOL_TIMEOUT` sets how long a running tool call may take.
- **Large results:** the SDK applies the same MCP output limit as Claude Code. A tool result with no image content that is larger than 25,000 tokens is saved to a file, and Claude gets an error message naming the path so it can read the output back in portions.
- **Tool search:** on by default, with documented exceptions (for example, the SDK turns it off when `ANTHROPIC_BASE_URL` points to a non-first-party host). Tool definitions are withheld from the context window; Claude sees a summary, searches when it needs a capability, and each search loads up to five of the most relevant tools by default. This applies to custom SDK MCP servers as well as remote ones.

CCAR-F 2.4-K3 says "tools from all configured MCP servers are discovered at connection time and available simultaneously to the agent" ([CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Broadly, it still holds: tools from every configured server are available to the agent together. Two details have moved. With tool search on (the current default), full definitions load on demand instead of all up front. And connection is no longer always up front: a remote server with a cached tool list offers its cached tools from the first turn but does not connect until its first tool call, while servers still pending when the first-turn wait ends keep connecting in the background, and their tools are absent until they connect. Answer in the guide's terms. The rest of MCP in the SDK is covered in [MCP in the API and the Agent SDK](tool-use-and-mcp.md#mcp-in-the-api-and-the-agent-sdk).

### Built-in tool, custom tool, MCP server, Skill or hook

The CCDV-F Agentic Customization skill asks for the "Tradeoffs among built-in Tools, custom Tools, Skills, and MCPs for selecting and applying the appropriate approach for a given use case" ([CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). Inside an Agent SDK application the choice looks like this; the cross-surface version is in [Built-in tools, custom tools, Skills or MCP](tool-use-and-mcp.md#built-in-tools-custom-tools-skills-or-mcp).

| You need | Use | Why |
|---|---|---|
| Generic file, search, shell or web work | A built-in tool (`Read`, `Edit`, `Bash`, `Grep`, `WebFetch` and the rest) | Already in the loop, executed by the SDK, covered by permissions and hooks |
| Your own application's logic or data for this agent | A custom tool in an SDK MCP server | Runs in your process; no separate server to deploy |
| A standard integration such as Jira | An existing community MCP server | CCAR-F 2.4-S4 prefers community servers for standard integrations and reserves custom servers for team-specific workflows |
| A capability several Claude applications should share and maintain independently | A standalone MCP server | One server, many clients; this is the CCDV-F sample 3 answer for an internal inventory REST API, where the rationale adds that built-in tools "do not automatically reach arbitrary internal APIs" |
| Know-how about how to do a task, loaded only when relevant | A Skill (`.claude/skills/<name>/SKILL.md`) | Claude invokes a Skill when a request matches its `description`. In the SDK, Skills are files on disk loaded through setting sources; there is no programmatic registration API |
| A rule that must hold every time | A hook | Hooks always fire on their event; Skills and prompts are interpreted by Claude |

Tool design rules still apply to custom tools. Anthropic's advice is to "Poka-yoke your tools. Change the arguments so that it is harder to make mistakes." ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)), and CCAR-F 2.3-S2 gives the exam's example: replace a generic `fetch_url` with a `load_document` tool that validates document URLs. How to write descriptions that steer selection is in [Writing tool descriptions that steer selection](tool-use-and-mcp.md#writing-tool-descriptions-that-steer-selection).

If you move an agent to Claude Managed Agents, `@tool` functions become tools of type `custom`: the session emits `agent.custom_tool_use`, your client runs the function and replies with `user.custom_tool_result`.

### Decide and avoid

- If the handler can fail, catch the error and return the error flag with a message Claude can act on; not a success result with empty data, which Claude cannot tell from a real empty answer (CCAR-F 2.2-S4), and not a uniform "Operation failed" string, which 2.2-K3 says prevents appropriate recovery decisions ([CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).
- If a tool has no side effects, set `readOnlyHint: true` so it can run in parallel; do not treat annotations as enforcement, since they block nothing.
- If only your tools should be available, set `tools: []`; `allowedTools` alone leaves the built-ins in place. Keep each agent's set small and role-scoped: CCAR-F 2.3-K1 says too many tools ("18 instead of 4-5") degrades tool selection reliability ([CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).
- If the capability must be shared across applications, build a standalone MCP server rather than an in-process one; not prompt instructions, pasted data or a built-in tool (the three wrong options in CCDV-F sample 3).
- If Claude can see a custom tool but cannot call it, permit its full `mcp__server__tool` name.

## Deployment models

*Tested in: CCDV-F Agent Construction with Claude · CCAR-P 3.2, 5.4*

The CCDV-F Agent Construction skill (5.3% of that exam) covers "the Claude Agent SDK, custom agent loops and harnesses, managed agent deployment models (self-hosted vs. Anthropic-hosted), and hooks for deterministic actions" ([CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). The Agent SDK overview separates Anthropic's options by "who runs the agent, what comes built in, and how you reach it" ([Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)), and the tool-use docs add that tools differ primarily by where the code executes. Two questions therefore place any deployment (our framing): who runs the agent loop, and where do the tools execute?

### Four ways to run an agent

| Option | Who runs the agent loop | Where tools execute | What you operate |
|---|---|---|---|
| Client SDK on the Messages API | Your code, or the client SDK's beta tool runner | Client tools in your application; server tools on Anthropic's infrastructure | Everything: loop, tools, state |
| Claude Agent SDK | The Claude Code binary, spawned as a subprocess by a library in your process | Your process and container | The process, its disk, scaling, isolation |
| Claude Code CLI | The CLI, interactively or with `-p` and `--output-format json` from any language | Your machine or runner | The runner |
| Claude Managed Agents | Anthropic's hosted harness, configured through the Claude API | An Anthropic-managed cloud sandbox, or a self-hosted sandbox on your infrastructure; custom tools in your application | Your client, which sends and receives events and runs custom tools; with a self-hosted sandbox, also the environment worker and the sandboxes it runs |

!!! warning "Exam guide vs current docs: two meanings of self-hosted"

    The [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) phrase "managed agent deployment models (self-hosted vs. Anthropic-hosted)" maps most directly to Claude Managed Agents environments, which run in either an Anthropic-managed cloud sandbox or a self-hosted sandbox. As of September 2026 Managed Agents is still a beta, and every endpoint requires the `managed-agents-2026-04-01` beta header (the SDKs set it for you). Keep two meanings of "self-hosted" apart. Self-hosting the Agent SDK means the whole loop runs in your containers. A Managed Agents self-hosted sandbox keeps the loop on Anthropic's side and moves only tool execution into your infrastructure. When a question uses the guide's phrase, read the stem for which one it means: whether you operate the whole loop, or only the place where tools run (our decision rule).

### Self-hosting the Agent SDK

The [hosting guide](https://code.claude.com/docs/en/agent-sdk/hosting) opens with the key fact: the SDK "spawns and supervises a `claude` CLI subprocess that owns a shell, a working directory, and session files on disk. Hosting it is not like hosting a stateless API wrapper." One agent session maps to one subprocess. Session transcripts, CLAUDE.md memory files and working-directory artifacts live on local disk, and none of them survive a container restart, a scale-down or a move to another node.

| Session pattern | Container lifetime | Best for | Must have |
|---|---|---|---|
| Ephemeral | One container per user task, destroyed when it completes | One-off tasks | Nothing extra |
| Long-running | Persistent containers, often several SDK processes each | Agents that act autonomously, serve content, or handle high-volume message streams | Memory sized for the maximum concurrent sessions |
| Hybrid | Ephemeral containers that hydrate from a `SessionStore` on startup and persist updates back | Sessions that span many interactions but sit idle between them | A `SessionStore`: without one, shutting down loses the transcript |
| Multi-agent container | Several SDK subprocesses in one container | Agents that must collaborate closely in a shared environment | Separate working directories and isolated settings loading per agent |

A `SessionStore` mirrors transcripts only, not `CLAUDE.md` memory files or other working-directory artifacts; mount a shared volume or sync those separately. Mirror writes are best-effort: when a batch cannot be delivered, the SDK drops it, emits a `mirror_error` system message and continues the query, so alert on those messages if store durability matters.

Sizing and cost, as the hosting guide states them: 1 GiB RAM, 5 GiB disk and 1 CPU per agent is a reasonable starting point for a freshly started instance, memory grows with session length and tool activity, and the 1 GiB figure is a floor, not the per-session ceiling. Token cost typically dominates container cost by an order of magnitude or more: a minimally provisioned container runs roughly &#36;0.05 per hour, while one long session can spend dollars in tokens.

Security at hosting time has three parts:

- **Inbound:** put authentication at a gateway in front of the agent container; "The agent should receive pre-authenticated requests and should not be the component that validates user tokens." ([hosting guide](https://code.claude.com/docs/en/agent-sdk/hosting)).
- **Outbound credentials:** keep them out of the agent's environment. A proxy outside the agent's boundary injects them, so "The agent can make API calls, but it never sees the credential itself." ([secure deployment](https://code.claude.com/docs/en/agent-sdk/secure-deployment)).
- **Least privilege:** mount only the directories the agent needs (read-only where possible), restrict network access to specific endpoints through the proxy, and drop Linux capabilities in containers.

In a container shared by several tenants, the default settings and CLAUDE.md loading can leak one tenant's context into another's session. Isolate each tenant: skip filesystem settings with `setting_sources=[]` (TypeScript: `settingSources: []`), disable auto memory (it loads regardless of setting sources), give each tenant its own `CLAUDE_CONFIG_DIR` and `cwd`, and apply per-tenant egress rules at your proxy. The options from the docs' Python example:

```python
options=ClaudeAgentOptions(
    cwd=tenant_dir,
    setting_sources=[],
    env={
        "CLAUDE_CONFIG_DIR": config_dir,
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    },
)
```

In TypeScript, spread `...process.env` into `env` so the subprocess keeps variables such as `PATH` and `ANTHROPIC_API_KEY` (see the `env` row of [Options that matter](#options-that-matter)). What an empty `setting_sources` list still leaves in place is covered under the same heading.

| Known limitation | What to do |
|---|---|
| No top-level session timeout | Bound tool-use round trips with `maxTurns` / `max_turns` |
| Memory growth over long sessions | Cap session length or recycle subprocesses |
| Wide parallel subagent fan-outs can hit rate limits | Dispatch in smaller batches |
| No per-subagent wall-clock deadline | Cap each subagent with `maxTurns` in its AgentDefinition |

### Claude Managed Agents (Anthropic-hosted)

The Managed Agents launch post, dated April 8, 2026, announced it in public beta and describes it as pairing "an Anthropic-managed harness with production infrastructure for state, memory, permissions, and scheduled execution" ([Claude blog](https://claude.com/blog/claude-managed-agents)). The docs position it against the Messages API: the Messages API is best for "Custom agent loops and fine-grained control", Managed Agents for "Long-running tasks and asynchronous work" ([overview](https://platform.claude.com/docs/en/managed-agents/overview)).

| Concept | What it is |
|---|---|
| Agent | A reusable, versioned configuration: model, system prompt, tools, MCP servers, skills. Passing the ID as a string uses the latest version; pass an object to pin one |
| Environment | Where sessions run: an Anthropic-managed cloud sandbox or a self-hosted sandbox. In a cloud environment, sessions may share the environment but each session gets its own isolated sandbox (a fresh Linux container); in a self-hosted environment, isolation depends on how your worker runs sessions (in-process, or one sandbox per session) |
| Session | A running instance. Statuses are `idle`, `running`, `rescheduling` and `terminated`; a session that finishes its work goes `idle`, not `terminated` |
| Events | How you talk to it: you send events such as `user.message` and `user.tool_confirmation`; results stream back over server-sent events, and the event history is kept server-side |

The quickstart's Python calls, condensed (its `print` lines for IDs are left out): create an agent once, create an environment, start a session, then open the event stream before sending the first `user.message`.

```python
from anthropic import Anthropic

client = Anthropic()

agent = client.beta.agents.create(
    name="Coding Assistant",
    model="claude-opus-5-5",
    system="You are a helpful coding assistant. Write clean, well-documented code.",
    tools=[
        {"type": "agent_toolset_20260401"},
    ],
)
environment = client.beta.environments.create(
    name="quickstart-env",
    config={
        "type": "cloud",
        "networking": {"type": "unrestricted"},
    },
)
session = client.beta.sessions.create(
    agent=agent.id,
    environment_id=environment.id,
    title="Quickstart session",
)
with client.beta.sessions.events.stream(session.id) as stream:
    # Send the user message after the stream opens
    client.beta.sessions.events.send(
        session.id,
        events=[{"type": "user.message", "content": [{"type": "text", "text": "Create a Python script that generates the first 20 Fibonacci numbers and saves them to fibonacci.txt"}]}],
    )
    for event in stream:
        match event.type:
            case "agent.message":
                for block in event.content:
                    if block.type == "text":
                        print(block.text, end="")
            case "agent.tool_use":
                print(f"\n[Using tool: {event.name}]")
            case "session.status_idle":
                print("\n\nAgent finished.")
                break
```

What the platform gives you, as of September 2026:

- **Tools:** `agent_toolset_20260401` enables `bash`, `read`, `write`, `edit`, `glob`, `grep`, `web_fetch` and `web_search`, all on by default. Tool output over 100,000 characters (about 25,000 tokens) is written to a file in the sandbox and the model sees a truncated preview. Custom tools are executed by your application.
- **Permissions:** a per-tool `permission_policy` in place of modes and `canUseTool`; values and defaults are in [Permissions and enforcement](#permissions-and-enforcement).
- **Control and state:** optional session budgets that act as a hard spend ceiling, memory stores that carry information across sessions, vaults that hold per-user third-party credentials referenced by ID, scheduled deployments that start sessions on a cron schedule, and webhooks that deliver only an event's `type` and `id`.
- **Quality loop:** outcomes, where a grader in a separate context window evaluates the work against a rubric.
- **Multiple agents:** a coordinator declares a roster with `multiagent: {type: "coordinator", agents: [...]}`; it delegates one level deep, to at most 20 unique agents (it can call several copies of each), with at most 25 concurrent threads (advisor threads exempt). All agents share the sandbox, filesystem and vault credentials, but each has its own context-isolated session thread.

Check two constraints before choosing it. The [overview](https://platform.claude.com/docs/en/managed-agents/overview) states that Managed Agents is not currently eligible for Zero Data Retention or HIPAA Business Associate Agreement (BAA) coverage, because it is stateful by design; see [Data retention, training and compliance](security-and-governance.md#data-retention-training-and-compliance). And the launch post prices it at standard token rates plus &#36;0.08 per session-hour of active runtime (the post points to the docs for full pricing details).

### Managed Agents with a self-hosted sandbox

"Self-hosted sandboxes keep the orchestration on Anthropic's side but move tool execution into infrastructure you control, so the agent's code, filesystem, and network egress never leave your environment." ([self-hosted sandboxes](https://platform.claude.com/docs/en/managed-agents/self-hosted-sandboxes)). An environment worker, a process you run, claims work from the `self_hosted` environment's queue and executes the tool calls locally. One limit is easy to miss: "Tool inputs and outputs still flow to Anthropic's control plane (where Claude runs) so the model can see results and determine what to do next." (same page).

| | Cloud environment | Self-hosted sandbox |
|---|---|---|
| Where tools run | Anthropic-managed sandboxes | Your infrastructure |
| Network reach | Anthropic's egress controls | Your network policy |
| File and GitHub repo mounting | Managed by Anthropic | Managed by you |
| Memory stores | Mounted by Anthropic at `/mnt/memory/` | Downloaded to `/mnt/memory/` and synced by the SDK worker |
| Lifecycle | Managed by Anthropic | Managed by you |

Self-hosting fits when the agent must work on data that cannot leave your network, reach internal services that are not publicly routable, or run under your own compliance and audit controls. It comes with a shared-responsibility split: Anthropic secures the control plane, and you own image hardening, network egress controls, storage and rotation of the environment service key `ANTHROPIC_ENVIRONMENT_KEY` (in a secrets manager, not in environment files or images), the blast radius of tool execution inside your sandbox, and retention of the conversation content and tool outputs that pass through your worker. Reaching private MCP servers is a separate choice: MCP tunnels, a research preview, connect through an outbound-only gateway.

### Moving an Agent SDK agent to Managed Agents

"The difference is where they run: the SDK runs in a process you operate, while Managed Agents runs in Anthropic's infrastructure." ([migration guide](https://platform.claude.com/docs/en/managed-agents/migration)). Most concepts have a direct counterpart, and a few duties move to your client:

| Agent SDK | Managed Agents |
|---|---|
| `ClaudeAgentOptions(...)` per run | `client.beta.agents.create(...)` once, versioned server-side |
| `ClaudeSDKClient` / `query()` | `client.beta.sessions.create(...)`, then send and receive events |
| `@tool` functions | `{"type": "custom", ...}` tools answered through `agent.custom_tool_use` / `user.custom_tool_result` events |
| Built-in tools on your filesystem | `agent_toolset_20260401` runs the same tools in the session sandbox against `/workspace` |
| `cwd`, `add_dirs` pointing at local paths | Files uploaded or mounted as session resources |
| `system_prompt` and the `CLAUDE.md` hierarchy | A single `system` string on the Agent; each change makes a new server-side version that sessions can pin |
| `mcp_servers` configured and authenticated in one place | Servers declared on the Agent; credentials supplied through a Vault on the Session |
| `permission_mode`, `can_use_tool` | Per-tool `permission_policy` plus `user.tool_confirmation` events |
| `PreToolUse` / `PostToolUse` hooks | Client-side handling of custom tool events, or `always_ask` for built-ins |
| Plan mode | A planning-only session first, then a second session to run the plan |
| `max_turns` | Count turns client-side; there is no server equivalent |

Coming from a hand-written Messages API loop, the migration guide lists four things you stop managing: the history array (the session stores history server-side), running pre-built tools (they run inside the sandbox; you handle only custom tools), provisioning a sandbox for agent-generated code, and deciding when the loop is done (the session emits `session.status_idle` when the agent has nothing more to do).

### Choosing a deployment model

| Requirement | Choose |
|---|---|
| Full control of each step, custom approval logic, your own logging | A Messages API loop, or the Agent SDK |
| Claude Code's tools, permissions, sessions and hooks inside your own product, running on your infrastructure | Agent SDK, self-hosted |
| No agent infrastructure to run; long-running or asynchronous work; scheduled runs | Managed Agents, cloud environment |
| Anthropic runs the loop, but code, files and egress must stay in your network | Managed Agents, self-hosted sandbox (tool inputs and outputs still reach Anthropic's control plane) |
| Zero Data Retention or HIPAA BAA coverage required | Not Managed Agents, as of September 2026 |
| Claude Code's agent loop from a language other than Python or TypeScript | The CLI as a subprocess with `-p` and `--output-format json` (Managed Agents is also reachable from Anthropic's client SDKs, the `ant` CLI or the REST API) |

**Avoid:** assuming a self-hosted sandbox keeps tool results away from Anthropic (they still flow to the control plane); counting on SDK hooks or `max_turns` after moving to Managed Agents (they become your client's job); and treating an Agent SDK container like a stateless web service (its transcripts and memory files live on local disk).

## Agent frameworks

*Tested in: CCDV-F Agent Patterns and Frameworks, Agent Architecture · CCAR-P 3.7*

The CCDV-F Agent Patterns and Frameworks skill (4.9% of that exam) covers "Common agent design patterns (tool-use loops, sub-agents, memory, context-window management) and agentic abstraction frameworks (e.g., Strands, LangGraph, PydanticAI) for building agents and workflows for multi-step tasks." ([CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). The three names are examples. Our reading of the skill: expect questions on what a framework does for you compared with a hand-written loop or the Claude Agent SDK, and what it costs you.

### Anthropic's position on frameworks

[Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) sets out Anthropic's position. Its live page now carries a note that much of the tooling landscape it describes has changed since December 2024 (see [Workflows or agents](#workflows-or-agents)), and the advice quoted here is still on that page. Its core finding: "Consistently, the most successful implementations use simple, composable patterns rather than complex frameworks." Frameworks make it easy to start, "by simplifying standard low-level tasks like calling LLMs, defining and parsing tools, and chaining calls together." The cost: "they often create extra layers of abstraction that can obscure the underlying prompts and responses, making them harder to debug." Hence the advice: "We suggest that developers start by using LLM APIs directly: many patterns can be implemented in a few lines of code. If you do use a framework, ensure you understand the underlying code." The post also names wrong assumptions about what sits under the hood as a common source of customer error, and closes by suggesting you reduce abstraction layers as you move to production.

!!! warning "Exam guide vs current docs: which frameworks"

    The CCDV-F guide names Strands, LangGraph and PydanticAI. The current version of Building effective agents lists the Claude Agent SDK, the Strands Agents SDK by AWS, Rivet and Vellum; LangGraph and Pydantic AI are not in its list. The framework facts below come from each framework's own documentation as of September 2026, and those libraries change quickly: LangChain, for example, now says its `langgraph-supervisor` package is no longer actively maintained. Learn what each framework is for, not its API surface.

### The three named frameworks and the Agent SDK

| Framework | Maker and shape | Multi-agent | Control, state and validation | Reaching Claude |
|---|---|---|---|---|
| Claude Agent SDK | Anthropic; Claude Code's tools, agent loop and context management as a Python and TypeScript library | Subagents through the `agents` option and the Agent tool (called the Task tool in the CCAR-F guide; `Task` is still accepted as an alias) | Sessions, hooks, permissions, `max_turns` and budget caps | Native |
| Strands Agents | AWS, open source; model-driven (an agent is a model, tools and a prompt); Python and TypeScript, runs in-process with no hosted control plane | Agents as tools, Swarm, Graph, Workflow, and Agent-to-Agent (A2A) | Invocation limits on turns and tokens, cancellation, hooks, retry strategies | `AnthropicModel`; Amazon Bedrock is the default provider |
| LangGraph | LangChain Inc (usable without LangChain); a low-level orchestration framework and runtime for long-running, stateful agents, built as graphs of State, Nodes and Edges | LangChain's subagents architecture: a main agent, often called a supervisor, calls subagents as tools | Checkpointers (thread-scoped state for continuity, human-in-the-loop, time travel, fault tolerance) and stores (cross-thread data); recursion limit, default 1000 steps from version 1.0.6 | `ChatAnthropic` from `langchain-anthropic` |
| Pydantic AI | The Pydantic team; agents typed end to end, holding instructions, tools, an output type and dependencies | Agent delegation through a tool; programmatic hand-off; graph-based control flow | Pydantic validation of structured output, `ModelRetry` to ask for another try, `UsageLimits` against runaway cost and tool loops | `AnthropicModel`, or the string `'anthropic:claude-sonnet-4-6'` |

How each one reaches Claude, from its own docs (the Strands snippet is the model setup from its structured-output example):

=== "Strands"

    ```python
    from strands import Agent
    from strands.models.anthropic import AnthropicModel

    model = AnthropicModel(
        client_args={
            "api_key": "<KEY>",
        },
        max_tokens=1028,
        model_id="claude-sonnet-5",
    )

    agent = Agent(model=model)
    ```

=== "LangGraph"

    The model object comes from the `langchain-anthropic` package:

    ```python
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(model="claude-haiku-4-5-20251001")
    ```

    A graph is State, Nodes and Edges, compiled before use. The docs' hello-world node returns a fixed message in place of a model call:

    ```python
    from langgraph.graph import StateGraph, MessagesState, START, END

    def mock_llm(state: MessagesState):
        return {"messages": [{"role": "ai", "content": "hello world"}]}

    graph = StateGraph(MessagesState)
    graph.add_node(mock_llm)
    graph.add_edge(START, "mock_llm")
    graph.add_edge("mock_llm", END)
    graph = graph.compile()

    graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
    ```

=== "Pydantic AI"

    ```python
    from pydantic_ai import Agent

    agent = Agent('anthropic:claude-sonnet-4-6')
    ```

The first examples on the Strands Anthropic page (Python and TypeScript) also set `temperature` to `0.7`; do not copy that line. Anthropic's model deprecations page says `temperature`, `top_p` and `top_k` return a 400 error when set to a non-default value on Claude 4.7 and later models, and recommends prompting instead ([model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations)). The frameworks document the same break: LangChain's Claude integration page lists `claude-opus-5`, `claude-fable-5`, `claude-opus-4-8`, `claude-opus-4-7` and `claude-sonnet-5` and advises removing the parameters rather than adjusting them, and Pydantic AI drops the three keys automatically for `claude-opus-4-7`, `claude-opus-4-8`, `claude-opus-5`, `claude-sonnet-5`, `claude-fable-5` and `claude-mythos-5`. A framework example copied from an older tutorial can fail for this reason alone (our observation); see [Model versions, deprecation and migration](claude-api.md#model-versions-deprecation-and-migration).

### Same pattern, different names

The [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)'s Agent Architecture skill mentions "manager/supervisor hierarchies". Anthropic's names for that shape (coordinator, lead agent, main agent with subagents) are compared in the table at the top of [Multi-agent orchestration](#multi-agent-orchestration), and the underlying pattern is taught in [Orchestrator-workers](#orchestrator-workers). The three named frameworks use their own:

| Framework | Name for one agent that plans, delegates and merges |
|---|---|
| LangChain and LangGraph | A supervisor: a full agent that keeps conversation context and calls subagents as tools (a router, by contrast, is one classification step) |
| Strands | Agents as tools: an orchestrator agent delegating to specialists, which Strands compares to a manager coordinating a team |
| Pydantic AI | Agent delegation: an agent hands work to another through a tool and takes control back when it finishes |

When the pattern pays off, and when it does not, is covered in [Multi-agent orchestration](#multi-agent-orchestration). LangChain's own multi-agent guide makes the same point Anthropic does: a single agent with the right tools and prompt can often achieve similar results.

### Choosing a foundation

| Situation | Choose | Why |
|---|---|---|
| One model, a handful of tools, short and well-scoped tasks | A hand-written loop on the Messages API | Anthropic's default; many patterns take a few lines of code, and nothing hides the prompts |
| You want Claude Code's built-in tools, permissions, sessions and hooks in your product | The Claude Agent SDK | Claude Code's loop, tools and context management come built in |
| An explicit graph mixing fixed steps with LLM steps, with checkpoints, human-in-the-loop and replay | LangGraph | Persistence, human-in-the-loop and memory are its central benefits |
| Typed, validated outputs and dependency injection are central | Pydantic AI | Pydantic builds and validates the schemas |
| An AWS-centered stack with model-driven multi-agent patterns | Strands Agents | Bedrock is its default provider; swarm, graph and workflow are built in |
| Agents built by different teams, frameworks or vendors must work together | A2A between agents, MCP inside each agent for its tools | A2A is an open standard for communication between independent agent systems |

Strands' own comparison page adds a vendor-authored rule of thumb: keep the loop you wrote while the agent is small and intends to stay that way, and reconsider when you catch yourself writing a second token counter or a second provider adapter. Treat it as one vendor's view; the same page cautions that its columns about other libraries may be out of date. For how A2A and MCP divide the work in an enterprise architecture, see [Integration patterns](solution-architecture.md#integration-patterns).

**Avoid:** answering a framework question with a specific API detail (the guide names the frameworks only as examples); assuming a framework removes the need to understand the prompts and tool calls underneath (Anthropic names wrong assumptions about framework internals as a common source of error); and reaching for a multi-agent framework before a single well-prompted agent has been tried.

## Exam map

*Tested in: CCDV-F, CCAR-F and CCAR-P, section by section as mapped below; CCAO-F only at concept level*

Which official objectives each section of this page serves. The objectives, their wording and their order come from the four exam guides (each Version 1.0, effective July 2026); which section serves which objective is our mapping. How to read the labels:

- **[CCAR-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf).** `1.3-K1` is task statement 1.3, first "Knowledge of" bullet, and `S` marks a "Skills in" bullet. The guide numbers its task statements but not the bullets under them, so K and S numbers count bullets in printed order. `Q1` is sample question 1, `EX1-STEP2` is preparation exercise 1, step 2, and How to Prepare item 1 is the first bullet of the guide's How to Prepare section. `APPX-TECH-n` is item n of the appendix list of "technologies and concepts that might appear on the exam", and `APPX-INSCOPE-n` is item n of the appendix list of topics "explicitly tested on the exam".
- **CCDV-F, CCAR-P and CCAO-F.** These guides do not number their objectives (CCDV-F lists named skills with weights; CCAR-P and CCAO-F list bullets), so the numbers follow printed order: CCDV-F D1.2 is the second skill in Domain 1, CCAR-P 3.1 is the first objective in Domain 3, and CCAO-F D3.4 is the fourth objective in Domain 3. "Sample n" in the CCDV-F and CCAR-P columns is that guide's sample question n.
- **Not listed** means the guide has no objective on the section's topic; the notes below the table explain the empty columns.

| Section | CCAO-F | CCDV-F | CCAR-F | CCAR-P |
|---|---|---|---|---|
| [Workflows or agents](#workflows-or-agents) | D1.2 (concept only) | D1.1 Agent Architecture; D1.3 Agent Patterns and Frameworks | 1.1-K3; 1.4-K2, 1.4-S1; 1.6-K1 to 1.6-K3, 1.6-S1, 1.6-S3; Q1, Q2; APPX-TECH-11 | 1.3, 1.5 |
| [Workflow patterns](#workflow-patterns) | D1.2 (concept only) | D1.1; D1.3 | 1.2-K1, 1.2-S3; 1.4-K1, 1.4-K2, 1.4-S2; 1.6-K1, 1.6-K2, 1.6-S1 to 1.6-S3; 4.6-K1 to 4.6-K3, 4.6-S1; Q1, Q2, Q3, Q12; APPX-TECH-11 | 1.3, 1.5 |
| [The agentic loop](#the-agentic-loop) | Not listed | D1.2 Agent Construction with Claude; D1.3; D8.1 Tool Implementation | 1.1-K1 to 1.1-K3, 1.1-S1 to 1.1-S3; EX1-STEP2; APPX-INSCOPE-1, APPX-TECH-5 | Not listed |
| [Multi-agent orchestration](#multi-agent-orchestration) | Not listed | D1.1; D1.3; D6.1 Context Engineering | 1.2-K1 to 1.2-K4, 1.2-S1 to 1.2-S4; 1.3-K1, 1.3-K2, 1.3-S1 to 1.3-S4; 1.6-K3; 5.6-S1; Q7, Q8, Q9; EX4-STEP3; APPX-INSCOPE-2 | 1.4, 3.1; Sample 1 |
| [The Claude Agent SDK](#the-claude-agent-sdk) | Not listed | D1.2; D8.3 Agentic Customization | 2.5-K1 to 2.5-K4, 2.5-S1 to 2.5-S5; APPX-TECH-1, APPX-TECH-9; How to Prepare item 1 | Not listed |
| [Subagents in the SDK](#subagents-in-the-sdk) | Not listed | D1.1; D1.3; D6.1 | 1.2-K2; 1.3-K1 to 1.3-K3, 1.3-S1, 1.3-S3; 2.3-K1 to 2.3-K3, 2.3-S1, 2.3-S3; 3.4-K1, 3.4-K3, 3.4-K4, 3.4-S3; 5.4-K3, 5.4-S1; Q9; EX4-STEP1; APPX-TECH-1, APPX-TECH-3 | 1.4, 3.1 |
| [Hooks in the SDK](#hooks-in-the-sdk) | Not listed | D1.2; D7.2 Guardrails and Safe Deployment; D7.3 Claude Hooks; Sample 2 | 1.4-K1, 1.4-K2, 1.4-S1; 1.5-K1 to 1.5-K3, 1.5-S1 to 1.5-S3; Q1; EX1-STEP4; APPX-TECH-1 | 5.1 |
| [Sessions, resumption and forking](#sessions-resumption-and-forking) | D3.4 (concept only) | D2.5 Claude Application Design (session hygiene); D3.1 Claude Code Operation (session management) | 1.3-K4; 1.7-K1 to 1.7-K4, 1.7-S1 to 1.7-S4; APPX-TECH-3, APPX-TECH-13 | Not listed |
| [Permissions and enforcement](#permissions-and-enforcement) | Not listed | D3.1 (auto-mode); D7.2; D7.3; D8.1 (approval patterns); Sample 2 | 1.4-K1, 1.4-K2, 1.4-S1; 1.5-K3, 1.5-S3; 2.3-K3; Q1, Q3 | 3.1, 5.1, 5.3; Sample 1 |
| [Custom tools in the SDK](#custom-tools-in-the-sdk) | Not listed | D8.1; D8.2 MCP Server Development; D8.3; Sample 3 | 2.2-K1, 2.2-K3, 2.2-S4; 2.3-K1, 2.3-S2; 2.4-K3, 2.4-S4; APPX-TECH-1, APPX-TECH-2 | 3.7 |
| [Deployment models](#deployment-models) | Not listed | D1.2 | Not listed | 3.2, 5.4 |
| [Agent frameworks](#agent-frameworks) | Not listed | D1.1; D1.3 | Not listed | 3.7 |

Notes on the empty cells:

- **CCAO-F.** The Associate guide says the certification "is not intended for software developers who build against APIs or design agentic systems" ([CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)), and none of its 30 objectives names agents, the Agent SDK, hooks or permissions. Three rows share an idea with an Associate objective: Workflows or agents and Workflow patterns with D1.2 ("Apply task decomposition techniques to structure complex requests"), and Sessions, resumption and forking with D3.4 ("Understand and manage context limitations and memory considerations (when to restart, summarize, or persist)"). Associate candidates should study those ideas at the prompt and conversation level: prompt-level decomposition in [Chain of thought and prompt chaining](prompt-engineering.md#chain-of-thought-and-prompt-chaining), and the exam framing in [Domain 1](../claude-certified-associate.md#domain-1-prompting-and-task-execution) and [Domain 3](../claude-certified-associate.md#domain-3-product-and-model-selection) of the CCAO-F page.
- **CCAR-F, deployment and frameworks.** None of the 30 CCAR-F task statements covers where an agent is hosted or which third-party agent framework to use, and the [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)'s out-of-scope list names neither. The closest items on that list are "Deploying or hosting MCP servers (infrastructure, networking, container orchestration)", "Specific cloud provider configurations (AWS, GCP, Azure)" and "Detailed implementation of specific programming languages or frameworks (beyond what's needed for tool and schema configuration)".
- **CCAR-P.** The Professional guide has no Agent SDK objective and never names the Agent SDK; the technologies its objectives do name include MCP, API/CLI and agent-to-agent (3.7), Skills (2.5) and Claude Code (7.1). Its objectives touch this page through architectural patterns (1.3), multi-agent design (1.4), decomposition (1.5), capability bloat and least privilege (3.1, Sample 1), authentication and authorization gaps (3.2), integration mechanisms including agent-to-agent (3.7), guardrails and human-in-the-loop validation (5.1, 5.3) and regulatory compliance (5.4).

Where the weight sits: the blueprint areas that map most directly to this page, with the weights the guides print (the selection is ours). In the [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf), "Skill weights show each skill's share of the overall exam", not of the domain.

| Exam | Blueprint area | Weight | Mapped on this page |
|---|---|---|---|
| CCAR-F | Domain 1, Agentic Architecture & Orchestration | 27% | All seven task statements, 1.1 to 1.7 |
| CCDV-F | Domain 1, Agents and Workflows | 14.7% | All three skills: Agent Architecture 4.5%, Agent Construction with Claude 5.3%, Agent Patterns and Frameworks 4.9% |
| CCDV-F | Claude Hooks, in Domain 7 | 1.0% | Hooks in the SDK; Permissions and enforcement |
| CCDV-F | Agentic Customization, in Domain 8 | 4.1% | The Claude Agent SDK; Custom tools in the SDK |
| CCAR-P | Domain 1, Solution Design & Architecture | 17% | 1.3, 1.4, 1.5 |
| CCAR-P | Domain 3, Integration | 19% | 3.1, 3.2, 3.7 |
| CCAR-P | Domain 5, Governance, Safety & Risk Management | 14% | 5.1, 5.3, 5.4 |

### Guide wording and current docs

Each guide says its exam items are written against its objectives, so expect the guide's wording in stems and options (our reading). The current names are what you will meet in the docs and in code. Where they differ, answer in the guide's terms.

| Guide wording (July 2026) | Current docs, as of September 2026 | Taught in |
|---|---|---|
| CCAR-F: the Task tool spawns subagents, and a coordinator's `allowedTools` must include "Task" (1.3-K1, 1.3-S3, EX4-STEP1, EX4-STEP2, APPX-TECH-1) | Renamed `Agent` in Claude Code v2.1.63; `Task(...)` references in settings and agent definitions still work as aliases, and the SDK's `system:init` tools list still says `"Task"`. Listing it in `allowedTools` auto-approves it, and the docs' own examples still pair `agents` with `allowedTools: ["Agent"]`; but `Agent` does not ask before running, so in `default` and `dontAsk` modes it runs whether or not it is listed. To block delegation, deny `Agent` (`permissions.deny`, or its bare name in `disallowedTools`) | [Spawning subagents in parallel](#spawning-subagents-in-parallel); [Subagents in the SDK](#subagents-in-the-sdk) |
| CCAR-F: loop control continues on stop_reason "tool_use" and terminates on "end_turn" (1.1-K1, 1.1-S1, EX1-STEP2, APPX-TECH-5) | The client-tool loop continues while `stop_reason` is `tool_use` and exits on any other stop reason (the tool-use docs name `end_turn`, `max_tokens`, `stop_sequence` and `refusal`; the API can also return `model_context_window_exceeded`); when the server-side loop of server tools hits its iteration limit, the response comes back with `pause_turn`, which means the work is not finished: send the paused response back so Claude continues. For exam answers, continue on "tool_use" and stop on "end_turn" | [The agentic loop](#the-agentic-loop) |
| CCAR-F: `fork_session` (1.3-K4, 1.7-K2, 1.7-S2, APPX-TECH-13) | Still the Python option name; TypeScript uses `forkSession` and the CLI `--fork-session`. Forking branches the conversation history, not the filesystem | [Sessions, resumption and forking](#sessions-resumption-and-forking) |
| CCAR-F: hooks that intercept outgoing tool calls, and `PostToolUse` hooks that transform tool results (1.5-K1, 1.5-K2, 1.5-S1, 1.5-S2, APPX-TECH-1) | Interception is the `PreToolUse` event, which can block a call with `permissionDecision: "deny"`; a `PostToolUse` hook can replace any tool's output by setting `updatedToolOutput`, and the MCP-only `updatedMCPToolOutput` is deprecated | [Hooks in the SDK](#hooks-in-the-sdk) |
| [CCDV-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf): "auto-mode" (D3.1) | "auto mode", the `auto` permission mode, backed by a classifier | [Permissions and enforcement](#permissions-and-enforcement) |
| [CCDV-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf): "managed agent deployment models (self-hosted vs. Anthropic-hosted)" (D1.2) | Most directly, Claude Managed Agents, whose sessions run in an Anthropic-managed cloud sandbox or a self-hosted sandbox (our reading); still beta (header `managed-agents-2026-04-01`). Self-hosting the Agent SDK, which runs the whole loop in your containers, is the other meaning of self-hosted: read the stem for which one it means | [Deployment models](#deployment-models) |

The exam pages teach every objective in exam framing. For the areas above, start with [CCAR-F Domain 1](../claude-certified-architect-foundations.md#domain-1-agentic-architecture-orchestration), [CCDV-F Domain 1](../claude-certified-developer.md#domain-1-agents-and-workflows), [CCDV-F Domain 7](../claude-certified-developer.md#domain-7-security-and-safety), [CCDV-F Domain 8](../claude-certified-developer.md#domain-8-tools-and-mcps), and CCAR-P [Domain 1](../claude-certified-architect-professional.md#domain-1-solution-design-architecture), [Domain 3](../claude-certified-architect-professional.md#domain-3-integration) and [Domain 5](../claude-certified-architect-professional.md#domain-5-governance-safety-risk-management).

??? info "Sources"

    - [Claude Certified Architect, Foundations: Exam Guide (PDF, July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): task statements 1.1, 1.2, 1.3, 1.4, 1.6, 2.5 and 4.6, the scenarios, Exercise 4, the technologies and out-of-scope lists, and sample questions 1, 2, 3, 7, 8, 9 and 12 with their rationales
    - [Claude Certified Developer, Foundations: Exam Guide (PDF, July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf): Agent Architecture, Agent Construction with Claude, Agent Patterns and Frameworks, Context Engineering, Tool Implementation and Agentic Customization skill descriptions
    - [Claude Certified Architect, Professional: Exam Guide (PDF, July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf): objectives 1.3 (workflow, agentic, augmented LLM), 1.4, 1.5 and 3.1 (capability bloat), and sample question 1 on least privilege
    - [Claude Certified Associate, Foundations: Exam Guide (PDF, July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf): the statement that CCAO-F is not for people who design agentic systems, and objective D1.2 on task decomposition
    - [How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works): the tool-use contract, the five-step client loop keyed on stop_reason, the server-side loop and pause_turn, and the rule that a decision extracted by regex should have been a tool call
    - [Stop reasons and fallback](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons): stop_reason meanings, the pause_turn iteration limit, the empty end_turn cause, and the manual tool loop in Python and TypeScript
    - [Create a Message (API reference)](https://platform.claude.com/docs/en/api/messages/create): the seven stop_reason values and stop_sequence behavior
    - [Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls): tool_use and tool_result fields, is_error, ordering of tool results, keeping untrusted content in tool_result blocks
    - [Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use): returning all tool results together in the next user message
    - [Programmatic tool calling](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling): replies to pending programmatic tool calls carry only tool_result blocks
    - [Tool reference](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference): the advisor tool and the MCP connector listed as server tools
    - [Tool runner (SDK)](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner): the beta loop helper, max_iterations, and when to keep a manual loop
    - [Claude prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices): self-correction as the most common chaining pattern
    - [Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview): what the SDK is, the comparison with the CLI, client SDK and Managed Agents, other languages via the CLI, login restrictions, branding and terms
    - [Agent SDK quickstart](https://code.claude.com/docs/en/agent-sdk/quickstart): prerequisites, install commands, bundled binary, API key and cloud provider variables, query() and the first agent, tool presets
    - [How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop): turns, message types, the four-turn walkthrough, built-in tools, parallel tool execution, max_turns and budget, result subtypes, the combined example
    - [Track todos (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/todo-tracking): the models that get TaskCreate and TaskUpdate by default, and the opt-in for other models
    - [Agent SDK reference: Python](https://code.claude.com/docs/en/agent-sdk/python): query() versus ClaudeSDKClient, client methods, ClaudeAgentOptions fields and defaults, option precedence
    - [Agent SDK reference: TypeScript](https://code.claude.com/docs/en/agent-sdk/typescript): Options fields and defaults, the Query object, env replacement, the Agent tool input schema
    - [Streaming input](https://code.claude.com/docs/en/agent-sdk/streaming-vs-single-mode): streaming input as the preferred mode and the limits of single message input
    - [Work with sessions (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/sessions): TypeScript has no session-holding client; ClaudeSDKClient continues one session
    - [Subagents in the SDK](https://code.claude.com/docs/en/agent-sdk/subagents): the Agent tool, what a subagent receives, parallel completion time, background default, depth, concurrency and budget caps, the Workflow tool, Task and Agent naming
    - [Configure permissions (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/permissions): allowedTools auto-approves only, tools that run without being listed, bypassPermissions with allowed_tools
    - [Give Claude custom tools (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/custom-tools): availability versus permission, in-process SDK MCP servers
    - [Modifying system prompts](https://code.claude.com/docs/en/agent-sdk/modifying-system-prompts): the minimal default prompt, the claude_code preset with append, CLAUDE.md loading through setting sources
    - [Agent SDK migration guide](https://code.claude.com/docs/en/agent-sdk/migration-guide): the rename from the Claude Code SDK, package and type names, the v0.1.0 system prompt change, isolation with empty setting sources
    - [Track cost and usage](https://code.claude.com/docs/en/agent-sdk/cost-tracking): cost fields are client-side estimates
    - [Create custom subagents (Claude Code)](https://code.claude.com/docs/en/sub-agents): the Task to Agent rename in v2.1.63, forks, nesting depth
    - [Run agents in parallel](https://code.claude.com/docs/en/agents): subagents report results to the conversation that spawned them
    - [Orchestrate teams of Claude Code sessions](https://code.claude.com/docs/en/agent-teams): experimental status, teammates messaging each other, no teammates in non-interactive or Agent SDK sessions
    - [Orchestrate subagents at scale with dynamic workflows](https://code.claude.com/docs/en/workflows): a script that decides what runs next, compared with subagents and agent teams
    - [Claude Code glossary](https://code.claude.com/docs/en/glossary): the agentic loop, the harness, and the Claude Code meaning of a turn
    - [Claude Code tools reference](https://code.claude.com/docs/en/tools-reference): Glob and Grep default availability and the Edit remedies for non-unique matches
    - [What's new in Claude Code, week 27 of 2026](https://code.claude.com/docs/en/whats-new/2026-w27): subagents running in the background by default
    - [Managed Agents: migration](https://platform.claude.com/docs/en/managed-agents/migration): session.status_idle ends the loop for you
    - [Managed Agents: multiagent orchestration](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration): coordinator roster, persistent threads, one-level delegation, roster and thread limits
    - [Managed Agents: define outcomes](https://platform.claude.com/docs/en/managed-agents/define-outcomes): a grader in a separate context window and the max_iterations bound
    - [Building effective agents (Anthropic Engineering)](https://www.anthropic.com/engineering/building-effective-agents): workflows versus agents, the augmented LLM, the five workflow patterns with when-to-use guidance and examples, when to use agents, ground truth, stopping conditions, the three principles
    - [How we built our multi-agent research system (Anthropic Engineering)](https://www.anthropic.com/engineering/multi-agent-research-system): orchestrator-worker architecture, token multipliers, the 90.2% result, delegation and effort-scaling rules, parallelism, production lessons
    - [Effective context engineering for AI agents (Anthropic Engineering)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): subagents returning condensed summaries of 1,000 to 2,000 tokens
    - [Building multi-agent systems: when and how to use them (Claude blog)](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them): the definition, the three situations where multiple agents win, 3 to 10 times token cost, context-centric decomposition, the verification subagent and the early victory problem
    - [Building agents with the Claude Agent SDK (Claude blog)](https://claude.com/blog/building-agents-with-the-claude-agent-sdk): the rename, the gather context, take action, verify work loop, and verification methods
    - [research_lead_agent.md (Anthropic Claude Cookbooks)](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_lead_agent.md): the lead agent's rules on parallel subagent creation, sub-topic boundaries and when to stop research
    - [What's new in Claude Code, week 24 of 2026](https://code.claude.com/docs/en/whats-new/2026-w24): the earlier five-level nesting note
    - [Intercept and control agent behavior with hooks (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/hooks): the event table for Python and TypeScript, registration, matchers, callback inputs and outputs, precedence, parallel execution, async output, timeouts, troubleshooting
    - [Hooks reference (Claude Code)](https://code.claude.com/docs/en/hooks): event descriptions, matcher rules, exit codes, decision fields per event, updatedToolOutput and additionalContext behavior, defer, the settings-file format, trust in -p and SDK sessions
    - [Automate actions with hooks (Claude Code)](https://code.claude.com/docs/en/hooks-guide): deterministic control and the rule against several hooks rewriting one tool's input
    - [Manage sessions (Claude Code)](https://code.claude.com/docs/en/sessions): naming and resuming by name, what a resumed session restores, picker and --continue exclusions, interleaving without a fork, transcript retention
    - [CLI reference](https://code.claude.com/docs/en/cli-reference): --fork-session
    - [Run Claude Code programmatically](https://code.claude.com/docs/en/headless): capturing session_id from --output-format json and resuming it
    - [Rewind file changes with checkpointing (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/file-checkpointing): which tools checkpointing tracks and that rewinding files does not rewind the conversation
    - [Checkpointing (Claude Code)](https://code.claude.com/docs/en/checkpointing): edits made outside Claude Code are not captured
    - [Handle approvals and user input (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/user-input): canUseTool responses in Python and TypeScript, AskUserQuestion limits, the defer pattern for long waits
    - [Configure permissions (Claude Code)](https://code.claude.com/docs/en/permissions): permission rules enforced by Claude Code rather than the model, Bash rules as text matches rather than a boundary
    - [Choose a permission mode (Claude Code)](https://code.claude.com/docs/en/permission-modes): critical-path removals that no allow rule or hook can approve
    - [How Claude remembers your project (Claude Code)](https://code.claude.com/docs/en/memory): CLAUDE.md treated as context, not enforced configuration
    - [Connect to external tools with MCP (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/mcp): transports, permissions for MCP tools, acceptEdits and MCP, connection timeout, output limit, the Python structuredContent limit
    - [Scale to many tools with tool search (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/tool-search): tool search on by default and the five-tool default
    - [Use Claude Code features in the SDK](https://code.claude.com/docs/en/agent-sdk/claude-code-features): the agents parameter with allowedTools Agent, setting sources, multi-tenant settings
    - [Extend agents with skills (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/skills): skills as filesystem artifacts with no programmatic registration API
    - [Extend Claude Code](https://code.claude.com/docs/en/features-overview): MCP versus Skills, and hooks for rules that must hold every time
    - [Hosting the Agent SDK](https://code.claude.com/docs/en/agent-sdk/hosting): the subprocess model, local state, the four session patterns, resources, cost, auth, multi-tenant isolation, known limitations, SessionStore
    - [Securely deploying AI agents (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/secure-deployment): the credential proxy pattern and the least-privilege table
    - [Tool use with Claude](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview): client tools versus server tools by where the code runs
    - [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview): positioning against the Messages API, the four concepts, the beta header, ZDR and HIPAA BAA eligibility, best-fit workloads
    - [Get started with Claude Managed Agents](https://platform.claude.com/docs/en/managed-agents/quickstart): creating an agent, environment and session, the agent toolset, streaming events
    - [Managed Agents: define your agent](https://platform.claude.com/docs/en/managed-agents/agent-setup): agents as versioned configurations
    - [Managed Agents: start a session](https://platform.claude.com/docs/en/managed-agents/sessions): latest version versus pinned version
    - [Managed Agents: cloud environment setup](https://platform.claude.com/docs/en/managed-agents/environments): one isolated sandbox per session
    - [Managed Agents: tools](https://platform.claude.com/docs/en/managed-agents/tools): the agent toolset, output truncation, custom tools
    - [Managed Agents: permission policies](https://platform.claude.com/docs/en/managed-agents/permission-policies): always_allow, always_ask, auto and their defaults
    - [Managed Agents: session operations](https://platform.claude.com/docs/en/managed-agents/session-operations): session statuses
    - [Managed Agents: reference](https://platform.claude.com/docs/en/managed-agents/reference): user events and session.status_idle
    - [Managed Agents: session budgets](https://platform.claude.com/docs/en/managed-agents/budgets): session budgets as hard spend ceilings
    - [Managed Agents: using agent memory](https://platform.claude.com/docs/en/managed-agents/memory): memory stores across sessions
    - [Managed Agents: authenticate with vaults](https://platform.claude.com/docs/en/managed-agents/vaults): per-user credentials referenced by ID
    - [Managed Agents: scheduled deployments](https://platform.claude.com/docs/en/managed-agents/scheduled-deployments): sessions started on a schedule
    - [Managed Agents: subscribe to webhooks](https://platform.claude.com/docs/en/managed-agents/webhooks): webhook payloads carry only type and id
    - [Managed Agents: self-hosted sandboxes](https://platform.claude.com/docs/en/managed-agents/self-hosted-sandboxes): orchestration on Anthropic, execution in your infrastructure, tool I/O to the control plane, the comparison table, environment workers, fit
    - [Managed Agents: security model for self-hosted sandboxes](https://platform.claude.com/docs/en/managed-agents/self-hosted-sandboxes-security): the shared responsibility model and environment key storage
    - [Claude Managed Agents (Claude blog, April 8, 2026)](https://claude.com/blog/claude-managed-agents): the public beta launch, what the platform pairs together, and the session-hour pricing
    - [New in Claude Managed Agents: self-hosted sandboxes and MCP tunnels (Claude blog)](https://claude.com/blog/claude-managed-agents-updates): MCP tunnels through an outbound-only gateway
    - [Introducing Strands Agents (AWS Open Source Blog)](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/): Strands as a model-driven open source SDK and its definition of an agent (vendor source)
    - [Strands Agents documentation index](https://strandsagents.com/llms.txt): Python and TypeScript, in-process with no hosted control plane, Bedrock as default provider (vendor source)
    - [Strands: Anthropic model provider](https://strandsagents.com/docs/user-guide/sdk/model-providers/anthropic/index.md): AnthropicModel configuration and the installation extra (vendor source)
    - [Strands: Amazon Bedrock model provider](https://strandsagents.com/docs/user-guide/sdk/model-providers/amazon-bedrock/index.md): BedrockModel as the default provider (vendor source)
    - [Strands: agent loop](https://strandsagents.com/docs/user-guide/sdk/agents/agent-loop/index.md): loop controls (vendor source)
    - [Strands: multi-agent patterns](https://strandsagents.com/docs/user-guide/sdk/multi-agent/multi-agent-patterns/index.md): agents as tools, A2A, Swarm, Graph and Workflow (vendor source)
    - [Strands: agents as tools](https://strandsagents.com/docs/user-guide/sdk/multi-agent/agents-as-tools/index.md): the hierarchical orchestrator pattern and the manager analogy (vendor source)
    - [Strands: choosing an agent foundation](https://strandsagents.com/docs/user-guide/migrate/choosing-an-agent-foundation/index.md): the vendor-authored rule on keeping a hand-written loop, and its own staleness caveat (vendor source)
    - [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview): LangGraph as a low-level orchestration runtime and the hello-world graph (vendor source)
    - [LangGraph Graph API](https://docs.langchain.com/oss/python/langgraph/graph-api): State, Nodes and Edges, StateGraph, compile, the recursion limit (vendor source)
    - [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence): checkpointers and stores (vendor source)
    - [LangChain multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent): a single agent can often achieve similar results (vendor source)
    - [LangChain subagents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents): the supervisor as main agent, supervisor versus router, langgraph-supervisor no longer maintained (vendor source)
    - [LangChain: ChatAnthropic integration](https://docs.langchain.com/oss/python/integrations/chat/anthropic): the langchain-anthropic package and the warning on non-default sampling parameters (vendor source)
    - [Model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations): temperature, top_p and top_k returning a 400 error at non-default values on Claude 4.7 and later models
    - [Pydantic AI documentation index](https://pydantic.dev/docs/ai/llms.txt): typed end to end (vendor source)
    - [Pydantic AI: agents](https://pydantic.dev/docs/ai/core-concepts/agent/index.md): what an agent holds (vendor source)
    - [Pydantic AI: dependencies](https://pydantic.dev/docs/ai/core-concepts/dependencies/index.md): dependency injection (vendor source)
    - [Pydantic AI: output](https://pydantic.dev/docs/ai/core-concepts/output/index.md): Pydantic validation and ModelRetry (vendor source)
    - [Pydantic AI: Anthropic models](https://pydantic.dev/docs/ai/models/anthropic/index.md): AnthropicModel, the model string, dropped sampling parameters (vendor source)
    - [Pydantic AI: multi-agent applications](https://pydantic.dev/docs/ai/guides/multi-agent-applications/index.md): levels of multi-agent complexity, agent delegation, UsageLimits (vendor source)
    - [A2A protocol specification](https://raw.githubusercontent.com/a2aproject/A2A/main/docs/specification.md): A2A as an open standard for communication between independent agent systems
    - [A2A and MCP](https://raw.githubusercontent.com/a2aproject/A2A/main/docs/topics/a2a-and-mcp.md): A2A between agents, MCP inside each agent
