---
title: "Solution Architecture and Enterprise Delivery for the Claude Certifications"
description: Discovery, surface choice, RAG, integration, platforms, capacity, cost, evaluation, governance and Claude Code rollout for the four Claude certification exams.
last_reviewed: 2026-09-23
---

# Solution Architecture and Enterprise Delivery

This page teaches the architect's side of the Claude certifications once: how to run discovery, choose the surface Claude runs on, design retrieval, integrate with enterprise systems and plan platforms and capacity, followed by cost modeling, program-level evaluation, governance, stakeholder work and the Claude Code rollout. It is the core reference for the Claude Certified Architect, Professional (CCAR-P) exam, whose guide describes practitioners who "select appropriate models, architectures, and API patterns; apply prompt and context engineering; integrate Claude into enterprise systems; and incorporate evaluation, security, compliance, and governance considerations into their designs" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). Product facts are as of September 2026; where the documentation has moved since the July 2026 exam guides, a section shows both versions and which wording to expect on the exam (the guide's).

!!! tip "Which exams lean on this page"

    **CCAR-P** draws on every section: Domains 1 (solution design), 3 (integration, including RAG and connection protocols) and 6 (discovery, SLAs and lifecycle) most directly, and Domains 4, 5 and 7 through the evaluation, governance and Claude Code rollout sections. Domain 2 (Claude Models, Prompting & Context Engineering) is taught mainly on the API, prompting and context pages. **CCDV-F** tests the developer's slice, most directly Understanding Requirements, Systems Life Cycle and Claude API Mechanics, which includes "invoking Claude through third-party vendors" and "tradeoffs between realtime and batch API selection" ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)), plus Agent Construction with Claude and Claude Application Design in the surface section, Software Engineering Foundations and Technical Fundamentals in the integration section, the program side of Cost and Token Management, the security and guardrail skills in the governance section, and the team side of Configuration Management and Claude Code Operation. **CCAO-F** tests requirements analysis, workflow integration, model choice and responsible use from the user's side, and its guide draws the line plainly: "Candidates are not expected to design enterprise-scale AI architectures or integrations; that scope belongs to the Claude Architect and Claude Developer credentials, to which Associates escalate more complex or technical work." ([CCAO-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)). **CCAR-F** lists embedding models and vector database implementation details; rate limiting, quotas and API pricing calculations; specific cloud provider configurations; and deploying or hosting MCP servers as out of scope, while Message Batches API appropriateness is in scope; what it does test here is mainly team configuration of Claude Code, Claude Code in CI/CD, the batch trade-off, deterministic enforcement and accuracy checks by segment. Each section below opens with the objectives it serves, and the [Exam map](#exam-map) at the end lists them all.

## Discovery and requirements

*Tested in: CCAR-P 1.1 (translate business problems into Claude-based solutions), 6.1 (structured discovery and requirement gathering), 6.5 (lifecycle phases, starting with discovery) · CCDV-F 2.1 Understanding Requirements · CCAO-F D4.1 (analyze requirements and use cases)*

Discovery turns an ambiguous request into requirements that a design can be traced back to. The official CCAR-P prep course states the skill in one sentence: "Translate an ambiguous business problem into a scoped Claude solution, selecting the reference architecture, model, context strategy, and entry point that keep it accurate and cost-conscious" ([CCAR-P prep course](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional)). Its stakeholder module adds the discipline that makes the result defensible: "translate what you learn into architectural requirements and documented assumptions, so the design traces back to the business case rather than to your own technical preference" ([Stakeholder Engagement, Lifecycle & GTM](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/stakeholder-engagement-lifecycle-gtm)). The CCDV-F guide frames the developer's version as "Functional and infrastructure requirements based on business requirements and solution architecture." ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf))

### What discovery must produce

| Output | What it contains | Source of the rule |
|---|---|---|
| A scoped use case | A defined user, task, output and measurable quality threshold | Anthropic and Accenture, [Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf) |
| A baseline | The answer to "What is the human performance baseline, and how was it established?" | Same guide |
| A work split | What Claude does, what existing systems do, and what humans do | CCAR-P prep course, solution-design module |
| A size estimate | Call volume, token consumption and cost, with a technical feasibility check and a scoped solution architecture with explicit boundary conditions | CCAR-P prep course, integration module |
| Requirements with documented assumptions | Functional and non-functional requirements, each traceable to the business case | CCAR-P prep course, stakeholder module |
| A data map | Where the data lives, its current state, the access rules per source, and its sensitivity class | Pilot-to-production guide |
| Success criteria | Measurable, multidimensional targets covering leading and lagging indicators | Anthropic's success-criteria guidance; pilot-to-production guide |
| A decision process | A go/no-go process documented before the pilot begins | Pilot-to-production guide |

Two of these outputs are taught in depth later on this page: the size estimate feeds the lightweight total cost of ownership model the pilot-to-production guide asks for before the pilot, taught in [Cost modeling](#cost-modeling), and the go/no-go process and ownership questions are in [Stakeholder communication and lifecycle](#stakeholder-communication-and-lifecycle).

### The discovery questions, in order

1. **What is ruled out?** Governance comes first. The solution-design module asks you to "identify which are ruled out by governance or regulated-industry constraints before any other trade-off applies" ([Claude Platform & Solution Design](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/claude-platform-solution-design)). A concrete case: Claude Managed Agents is stateful by design, so it is not currently eligible for Zero Data Retention (ZDR) or HIPAA Business Associate Agreement coverage. The pilot-to-production guide says to involve legal, compliance and security stakeholders before a pilot launches and to treat compliance as a design constraint surfaced pre-pilot.
2. **What is actually required, not assumed?** The [pilot-to-production guide](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf) gives three pre-pilot infrastructure questions: the simplest architecture that meets the actual requirements rather than the assumed ones; "What level of model customization is genuinely necessary versus assumed?"; and "What data residency and security requirements will Legal and Compliance actually enforce?" Its cross-functional "work out" questions add build versus buy (including engineering and maintenance cost), residency and sovereignty, model fit per use case, and cloud, hybrid or on-premises hosting.
3. **Is the data ready?** Pilots usually run on curated data from one system, while production runs on the real data estate, and the [pilot-to-production guide](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf) warns that "data causes more programs to stall than almost any other factor." Audit before setting timelines. Early audits find two kinds of issue: access (the data exists but cannot be reached programmatically, or reaching it needs permissions that take months to get through IT and Legal) and quality (it is reachable but inconsistent, incomplete or formatted in ways that produce unreliable outputs). Production needs data flowing in both directions, reliably and at volume, and in many cases vendor API conversations need to start several months earlier than the pilot team typically thinks to start them.
4. **Who may see what?** Classify the data by sensitivity (public, internal, customer PII, financial, clinical) before setting the pilot architecture, because, in the [same guide's](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf) words, "AI systems inherit the access control requirements of the data they touch, so those requirements need to be mapped before the pilot is built." In the guide's Novo Nordisk example, the team established exactly what data could flow to and from the model, and what controls were needed, before building; clinical study report production then fell from more than 10 weeks to under 10 minutes.
5. **How big and how often?** Estimate call volume, token consumption and cost. Frequency matters: the guide says to start with high-frequency, high-value workflows where ROI compounds, because a process that runs hundreds of times a day scales differently from one that runs once a quarter.

### From a business need to a testable requirement

A **functional requirement** describes "the functions that the software is to execute", and it can be validated by a finite set of test steps. **Non-functional requirements** constrain the solution; SWEBOK also calls them constraints or quality requirements, with types such as performance, maintainability, safety, reliability, security and interoperability ([SWEBOK v3, Software Requirements](http://swebokwiki.org/Chapter_1:_Software_Requirements)). Requirements should be stated clearly and, where appropriate, quantitatively. SWEBOK warns against vague, unverifiable requirements such as "the software shall be reliable". System requirements cover the users' needs and those of other stakeholders, regulators included, and NIST's glossary defines security requirements as derived from applicable laws, directives, policies, standards, regulations or organizational mission and business needs ([NIST glossary: security requirement](https://csrc.nist.gov/glossary/term/security_requirement)).

The CCDV-F description pairs functional requirements with infrastructure requirements. In this page's mapping (ours, not the guide's), the infrastructure side is the set of non-functional requirements that decide where and how Claude runs: residency, retention, capacity, latency and hosting. The pilot-to-production questions above and the lever table below turn each one into a design choice.

Anthropic's version of the same discipline is **success criteria** that are specific, measurable, achievable and relevant, with targets based on industry benchmarks, prior experiments, AI research or expert knowledge; most use cases need several criteria at once ([Define success criteria and build evaluations](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests)). Rewrite every vague requirement before designing anything:

| The stakeholder says | Requirement type | Testable form |
|---|---|---|
| *It has to be fast* | Performance | A latency percentile, such as Anthropic's example of 95% of responses under 200 ms; time to first token for a streaming chat interface |
| *It has to be safe* | Safety | A measured rate, such as Anthropic's example of fewer than 0.1% of outputs flagged for toxicity across 10,000 trials |
| *It can't cost too much* | Price | Cost per call, model size and usage frequency, compared as cost per completed task |
| *It has to be reliable* | Reliability | Uptime (%), plus named reliability patterns: retries, fallbacks, circuit breakers |
| *It has to be accurate* | Task fidelity | A measurable quality threshold on a defined task, agreed by the stakeholders before the pilot |

How to build and grade the test sets behind these criteria is in [Success criteria and test sets](evaluation-and-reliability.md#success-criteria-and-test-sets) and, at program level, in [Evaluation strategy for a program](#evaluation-strategy-for-a-program).

### Map each requirement to a design lever

As of September 2026:

| Requirement | Design lever | Taught in |
|---|---|---|
| Users must see output as it is generated | Streaming; time to first token is the responsiveness metric | [Streaming](claude-api.md#streaming) |
| Tight latency or cost on simple, high-volume tasks | Efficiency-first model choice, starting with Claude Haiku 4.5 | [Models and how to choose one](claude-api.md#models-and-how-to-choose-one) |
| Accuracy outweighs cost | Capability-first model choice, then optimize down | Same section |
| Quality passes but latency or cost fails | Try a model or effort change, not only prompt changes | [Extended thinking, adaptive thinking and effort](claude-api.md#extended-thinking-adaptive-thinking-and-effort) |
| Faster output from Claude Opus 5.5, Opus 5 or Opus 4.8 | Fast mode: up to 2.5x output tokens per second at premium pricing; a research preview on the Claude API only (including Managed Agents), not on Bedrock, Claude Platform on AWS, Google Cloud or Microsoft Foundry | Same section |
| Large volume, nobody waiting for the answer | Message Batches: 50% lower cost, most batches finish in under an hour, but the window is up to 24 hours with no guaranteed latency SLA, so it is inappropriate for a blocking step | [Invocation modes](#invocation-modes) |
| A throughput ceiling | Rate-limit tier plus prompt caching, since for most models only uncached input tokens count toward ITPM | [Deployment platforms and capacity](#deployment-platforms-and-capacity) |
| Guaranteed capacity | Priority Tier capacity commitments are no longer sold; the docs send anyone who needs guaranteed capacity to sales | Same section |
| US-only inference on the Claude API or Claude Platform on AWS | `inference_geo: "us"`, supported on Claude 4.6 and later models (older models return a 400 error) and priced at 1.1x the standard rate | Same section |
| FedRAMP High, IL4, IL5, HIPAA-ready compliance, or AWS as sole data processor, on AWS | Amazon Bedrock rather than Claude Platform on AWS | [Choosing the platform](#choosing-the-platform) |
| ZDR is mandatory | Check every feature against the docs' eligibility table. Not ZDR-eligible: Managed Agents, the MCP connector, Message Batches, the Files API and code execution (under ZDR these are not blocked; using one steps outside the arrangement for that data). Some models are also unavailable under ZDR without Anthropic's express authorization; see [Before the pilot: classify, then rule out](#before-the-pilot-classify-then-rule-out) | [Data retention, training and compliance](security-and-governance.md#data-retention-training-and-compliance) |
| Answers from fresh or permission-restricted documents | A retrieval layer | [Retrieval-augmented generation](#retrieval-augmented-generation) |
| Answers about live state (an order, a balance) | A tool call to the system of record (our example of what the [solution-design module](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/claude-platform-solution-design) calls a job "that live-state should own") | [Retrieval-augmented generation](#retrieval-augmented-generation) |
| Advice, recommendations or decisions directly affecting individuals in a High-Risk Use Case domain (legal; healthcare; insurance; finance; employment and housing; academic testing, accreditation and admissions; media or professional journalistic content) | Review by a qualified professional before dissemination or finalization, plus disclosure of AI involvement at least at the beginning of each session when outputs go directly to individuals | [Governance and risk in delivery](#governance-and-risk-in-delivery) |
| Any consumer-facing chatbot or interactive agent | Disclosure that users are interacting with AI rather than a human | Same section |

### Choosing the first use case

Anthropic's enterprise guide lists the traits of a good first use case: suited to what LLMs do well, measurable success metrics, clear ROI, business critical but low in security risk, abundant data, minimal disruption, and scalable. One way to keep disruption low is to run the AI process in parallel with the existing workflow until it proves reliable ([Building trusted AI in the enterprise](https://www-cdn.anthropic.com/e5c9de22bc8884089970bd262ca0c8b952cb9136.pdf)); the parallel run is one of the instruments in [From pilot to production](#from-pilot-to-production).

For individual and team workflows, which is the CCAO-F angle, Anthropic Academy's advice is to start with Problem Awareness: "Before touching any AI tools, analyze your actual workload." List 5 to 10 repetitive or time-consuming tasks from the past week with how often each happens, how long it takes and how standardized it is ([AI Fluency for nonprofits](https://academy.claude.com/courses/ai-fluency-for-nonprofits/workflow-augmentation)). Then ask what an imperfect result would cost, and pick the top-priority task by time saved, frequency and how straightforward the automation would be ([AI Fluency for small businesses](https://academy.claude.com/courses/ai-fluency-for-small-businesses/tying-it-all-together)). The deciding test is "should AI do this?" and not just "can AI do this?": answering documented questions is a good candidate, while complaints and high-stakes requests stay with people.

### Running discovery with Claude

Claude can run the requirements conversation itself. In Anthropic Academy's [AI-native SDLC playbook](https://academy.claude.com/courses/ai-native-sdlc-playbook/capture-intent), the person with the idea describes the problem in their own words; Claude asks "the questions an analyst would ask: scope, users, constraints, and what success looks like"; Claude drafts an intent document covering the problem, proposed outcome, affected users and systems, constraints and open questions; and the originator corrects anything Claude misunderstood. People who are not engineers can do this from claude.ai or Cowork. Three habits keep the output honest:

- Ask Claude to flag concerns, especially policies that contradict each other, and work through those first, since they are what an analyst would have escalated ([Requirements and design](https://academy.claude.com/courses/ai-native-sdlc-playbook/requirements-and-design)).
- State goals and non-goals: "Calling out goals and non-goals keeps scope honest" ([PRD from a one-pager](https://academy.claude.com/use-cases/prd-from-a-one-pager)).
- Hunt for guesses. The same use case warns that Cowork "fills gaps with reasonable guesses", so fix what you did not decide before the draft goes to review.

When the input is many stakeholders' requests, ask whether different requests point to the same underlying need, and check whether the sample represents the real user base ([Analyze patterns in user feedback](https://academy.claude.com/use-cases/analyze-patterns-in-user-feedback)). The CCAO-F guide page applies this to the exam's scenarios in [Domain 4: Workflow Integration and Solution Design](../claude-certified-associate.md#domain-4-workflow-integration-and-solution-design).

### Decide

- If a regulatory or data-handling constraint applies (ZDR, HIPAA, residency), choose to apply it first and strike out what it excludes; not after comparing cost and latency, because the prep course places governance constraints before any other trade-off.
- If a requirement is vague ("fast", "safe", "reliable"), choose to rewrite it as a measurable criterion; not to pick a model or feature straight away.
- If quality passes but latency or cost fails, test a model or effort change; not only more prompt engineering, because Anthropic notes that latency and cost are sometimes easier to improve by selecting a different model, and that tuning effort is often a better lever than switching models.
- If the work is high volume and straightforward, and speed and cost matter more than deep reasoning, choose a faster, lower-cost model; not the most capable model for every call. The CCAO-F Sample 2 rationale states the rule: "Aligning model selection with task requirements means matching a faster, lower-cost model to straightforward, high-volume work, reserving the most capable model for complex reasoning." ([CCAO-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf))
- If a requirement is latency-tolerant (an overnight report), record that in the requirement, because it decides whether Message Batches is an option; the realtime-versus-batch rule is taught in [Invocation modes](#invocation-modes).
- If the data exists but cannot be reached programmatically, or is inconsistent, choose to fix access and quality before committing to a timeline; not to start the build on curated pilot data alone.
- If two designs both meet the stated requirements, choose the simpler one; the pilot-to-production guide asks for the simplest architecture that meets the actual requirements and says to add complexity only when the simpler approach has demonstrably hit its limits.

### Traps

- **Designing for assumed requirements**, such as a regional deployment nobody asked for, instead of asking Legal and Compliance what they will actually enforce.
- **A goal with no threshold.** Anthropic's [enterprise agents guide](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/69f3af1f0b8ebe5cde42fcda_Claude-Building-AI-Agents-in-the-Enterpise-04302026_v2%20%281%29.pdf) warns that vague goals such as improving productivity produce vague results that are easy to dismiss; its examples of good criteria are concrete, such as call prep time cut by 50 percent.
- **Scoping on the pilot's clean data** and discovering the real data estate at production time.
- **Treating access control as a later security review** when the system inherits the access rules of every source it touches.
- **Accepting a polished requirements draft** without checking which assumptions Claude supplied.
- **Assuming a ZDR arrangement covers every API feature.** Message Batches, the Files API, code execution, the MCP connector and Managed Agents are marked not eligible, and the API does not block them under ZDR.

## Choosing the surface

*Tested in: CCAR-P 1.1 and 1.3 (entry point and architectural pattern), 7.1 (configure Claude tools for teams) · CCDV-F 1.2 Agent Construction with Claude (Agent SDK, managed agent deployment models), 2.5 Claude Application Design (instructions across Claude Code, Desktop, claude.ai, API and SDKs) · CCAO-F D3.1 (select product features) · CCAR-F: the six scenario settings (three built on the Agent SDK, two on Claude Code including CI/CD, one structured extraction system) and the Appendix technologies list (Agent SDK, Claude Code, Claude Code CLI, Claude API, Message Batches API)*

The surface is where Claude meets its user and who runs the agent loop. Anthropic's prep courses and documentation each frame this decision in their own words: the [CCAR-P prep course](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional) says "entry point", and the [CCAO-F prep course](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations) names entry point as the first of four decisions made before any prompt is written (with capability features, model and context). The Agent SDK documentation frames the developer version: "The Agent SDK, the CLI, the Client SDK, and Managed Agents differ in who runs the agent, what comes built in, and how you reach it." ([Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview)). Anthropic's platform introduction describes two ways to build on the API: the Messages API, for custom agent loops and fine-grained control, and Claude Managed Agents, a pre-built harness in managed infrastructure for long-running and asynchronous work ([Intro to Claude](https://platform.claude.com/docs/en/intro)).

The CCAR-P solution-design module splits the decision into layers. It asks you to "Distinguish between the Claude entry points a user sees, the build-time interfaces an engineer codes against, and the delivery routes an enterprise procures", to know where each entry point (Claude.ai, the API, an SDK, Claude Code, or an MCP server) fits and what customization belongs at each layer, and to "Choose between an augmented call, a workflow, and an agent by naming what each choice costs" ([Claude Platform & Solution Design](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/claude-platform-solution-design)). In this page's terms (our mapping): the table below covers entry points and build-time interfaces, and delivery routes are in [Deployment platforms and capacity](#deployment-platforms-and-capacity). The same module asks you to identify which options governance or regulated-industry constraints rule out before any other trade-off applies, which is why the ZDR and HIPAA checks come first in the rules below.

### The surfaces side by side

As of September 2026:

| Surface | Who runs the loop | Built for | Check before choosing |
|---|---|---|---|
| Claude apps: chat and Projects (web, desktop, mobile) | Anthropic's service; custom connectors are reached from Anthropic's cloud, not the user's device | People doing their own knowledge work turn by turn; a Project gives every chat inside it the same instructions and knowledge base | RAG for project knowledge is on paid plans only; on Team and Enterprise an Owner or Primary Owner must enable connectors for the organization; the Team and Enterprise product interfaces are not ZDR-eligible |
| Claude Cowork | Anthropic's servers, in an isolated environment (sessions in the cloud, beta), or, in a local desktop session, the user's computer, with code in an isolated virtual machine; same agentic architecture as Claude Code; started from Claude Desktop, claude.ai or the mobile app, and a cloud session reaches local files or the browser through the desktop app | Handed-off work: multi-step, file-based tasks that span the tools you use; describe an outcome and come back to finished work | Paid plans only (Pro, Max, Team, Enterprise); running Cowork in the cloud is on by default on Team, and off by default on Enterprise until an owner turns it on and grants the capability to groups with custom roles |
| Claude Code (terminal, desktop app, cloud sessions) | The Claude Code harness on the developer's machine; cloud sessions run in an isolated Anthropic-managed VM by default, or on the organization's self-hosted environment when routed there | Building and changing software: daily interactive development, or one-off tasks | Claude for Teams or Enterprise is the default recommendation for organizations; cloud sessions, Routines, Code Review, Remote Control and the Chrome extension need a claude.ai account |
| Claude Code headless and in CI | The CLI on your runner | Pipelines: `claude -p` from any language (flags in [Invocation modes](#invocation-modes)), or `@claude` through Claude Code GitHub Actions | Each Actions run consumes GitHub Actions minutes and tokens, billed to the API, or to a Claude subscription when authenticating with an OAuth token (`CLAUDE_CODE_OAUTH_TOKEN`) |
| Client SDK on the Messages API | Your code, or the client SDK's beta tool runner | Products where Claude is a component: custom agent loops and fine-grained control | You own the loop, tool execution and state; the API is stateless, so you send the full history every time |
| Claude Agent SDK | The Claude Code binary, run as a subprocess by a library in your Python or TypeScript process | Embedding Claude Code's agent (built-in tools, permissions, sessions, hooks) in your own application | You host a process with a shell, working directory and on-disk session files; unless previously approved, third-party products may not offer claude.ai login |
| Claude Managed Agents | Anthropic's hosted harness; tools run in an Anthropic-managed cloud sandbox or a self-hosted sandbox | Long-running and asynchronous work, minimal infrastructure, stateful sessions, scheduled runs | Beta (`managed-agents-2026-04-01` header); not eligible for ZDR or HIPAA BAA coverage; no batch mode; &#36;0.08 per session-hour of `running` time plus tokens |

Amazon Bedrock, Google Cloud, Microsoft Foundry and Claude Platform on AWS are not further surfaces in this sense. They are delivery routes: they decide where the model runs, how you are billed and which features exist, for traffic from the Messages API, the Agent SDK or Claude Code. Comparing them on latency, compliance and cost, as the CCAR-P prep course asks, is taught in [Deployment platforms and capacity](#deployment-platforms-and-capacity).

### A quick way to decide

```text
Who is the user, and who should run the loop?

1. A person doing their own knowledge work
     turn by turn ................................. chat, with a Project for standing context
     hand off an outcome, return later ............ Cowork
2. A developer building or changing software
     interactive .................................. Claude Code
     unattended, in a pipeline .................... claude -p, or Claude Code GitHub Actions
3. Claude inside a product you ship
     one call, a fixed workflow, or your own loop . Messages API through a client SDK
     Claude Code's agent inside your process ...... Claude Agent SDK
     an agent you do not want to host ............. Claude Managed Agents
4. Nobody is waiting for the answer ............... Message Batches (a Messages API feature:
                                                     50% lower cost, up to 24 hours, no latency SLA;
                                                     Managed Agents has no batch mode)
```

### Decision rules

- If ZDR or HIPAA BAA coverage is mandatory, rule out Managed Agents before comparing anything else, because it is stateful by design and not currently eligible. For ZDR, also rule out the Claude Team and Enterprise product interfaces, which are not ZDR-eligible; the documented exception is Claude Code used through Claude Enterprise with ZDR enabled.
- If a single well-built call meets the quality bar, choose the Messages API with that one call; not an agent. Anthropic's guidance reads: "For many applications, however, optimizing single LLM calls with retrieval and in-context examples is usually enough." The same post notes that agentic systems often trade latency and cost for better task performance ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)). Pattern selection itself is taught in [Workflows or agents](agents-and-agent-sdk.md#workflows-or-agents).
- If the loop needs human-in-the-loop approval, custom logging or conditional execution, write it yourself on the client SDK; not the beta tool runner, whose documentation sends those cases to the manual loop.
- If you want Claude Code's built-in tools, permissions, sessions and hooks inside your product, choose the Agent SDK. If you do not need to run the loop on your own infrastructure, the Agent SDK hosting guide itself points to Managed Agents.
- If the calling service is written in a language other than Python or TypeScript and needs Claude Code's agent loop, run the CLI as a subprocess with `-p` and `--output-format json`.
- If the users are knowledge workers whose needs are met by chat, Projects, connectors or Cowork, choose the apps. The CCAO-F guide places enterprise-scale architecture and integration design with the Architect and Developer credentials, to which Associates escalate more technical work. The CCAO-F choice among the guide's named features (Projects, research mode, chat, artifacts) is taught in [Projects](claude-for-work.md#projects), [Search, Research and connectors](claude-for-work.md#search-research-and-connectors) and [Artifacts, files and code execution](claude-for-work.md#artifacts-files-and-code-execution).
- If you are rolling Claude Code out to an organization, Claude for Teams or Enterprise is the documented default; the provider comparison and the rest of the rollout are in [Choose the provider](#choose-the-provider).
- If a framework is on the table, start with the API directly: Anthropic suggests "developers start by using LLM APIs directly" and, if a framework is used, understanding its underlying code, then reducing abstraction layers as the system moves to production. Frameworks are compared in [Agent frameworks](agents-and-agent-sdk.md#agent-frameworks).

### What changes when the surface changes

| Concern | Messages API | Agent SDK | Managed Agents | Claude Code | Claude apps |
|---|---|---|---|---|---|
| Conversation state | Yours: the API is stateless | Transcripts on the container's local disk, lost on restart unless a `SessionStore` mirrors them | Event history persisted server-side | Local transcripts under `~/.claude/projects/`, kept 30 days by default | Held by the service, except local Cowork sessions, which store history on the user's computer; each Project has its own memory |
| Where tools run | Client tools in your application; server tools on Anthropic's infrastructure | Your process and container | An Anthropic-managed or self-hosted sandbox | The developer's machine, or for cloud sessions an Anthropic-managed VM (or the organization's self-hosted environment) | Custom connectors are reached from Anthropic's cloud, not the user's device; desktop extensions (local MCP servers) run locally in Claude Desktop, not on web or mobile; Cowork cloud sessions run on Anthropic's servers, local Cowork sessions on the user's computer in an isolated virtual machine |
| System prompt | Only what you send; the claude.ai system prompt updates do not apply to the API | A minimal prompt unless you use the `claude_code` preset | The agent definition's system prompt | Claude Code's system prompt, with CLAUDE.md delivered as a user message after it | Anthropic's app system prompt, plus account, Project and organization instructions |

How instructions behave on each surface is taught in [XML tags, system prompts and roles](prompt-engineering.md#xml-tags-system-prompts-and-roles). Hosting the Agent SDK yourself and running Managed Agents in a self-hosted sandbox are two different meanings of "self-hosted"; both are in [Deployment models](agents-and-agent-sdk.md#deployment-models).

!!! warning "Exam guide vs current docs"

    The [CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) names the product features as "Projects, research mode, chat, artifacts". On September 16, 2026, Anthropic announced that "Claude Cowork and chat are merging into one Claude", rolling out gradually from Pro and Max, with Team and Free following and Enterprise admins getting at least 30 days' notice ([Cowork is now Claude](https://claude.com/blog/cowork-is-now-claude)). In the new experience Claude can decide which tool a request needs, and an account that has it cannot switch back to separate "Chat" and "Cowork" options ([Claude Cowork and chat are one Claude](https://support.claude.com/en/articles/16761823-claude-cowork-and-chat-are-one-claude)). The separate chat and Cowork rows above describe the experience the guide was written against, which accounts the rollout has not reached (for now including Team, Free and Enterprise) still have. The [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) phrase "managed agent deployment models (self-hosted vs. Anthropic-hosted)" names no product. Our reading is that it maps most closely to Managed Agents environments, which run in an Anthropic-managed cloud sandbox or a self-hosted sandbox; self-hosting the Agent SDK is the other meaning of "self-hosted". Managed Agents is still a beta as of September 2026. Expect the guides' wording on the exam.

### Traps

- **An agent where one call would do.** The added latency and cost buy nothing if a single call with retrieval and examples already meets the bar.
- **Managed Agents for a ZDR or HIPAA workload.** It is not eligible for either.
- **Expecting a batch discount from Managed Agents.** Sessions are stateful and interactive, and there is no batch mode.
- **Assuming cloud-provider credentials give access to every Claude Code feature.** Cloud sessions, Routines, Code Review, Remote Control and the Chrome extension need a claude.ai account.
- **Offering claude.ai login in a product built on the Agent SDK.** Anthropic does not allow it unless previously approved.
- **Scaling the Agent SDK like a stateless web service.** Session transcripts, CLAUDE.md memory files and working-directory artifacts live on local disk and do not survive a container restart, a scale-down or a move to another node; a `SessionStore` mirrors transcripts only.
- **Assuming an API ZDR agreement covers the Claude apps.** The Team and Enterprise product interfaces are not ZDR-eligible; the documented exception is Claude Code used through Claude Enterprise with ZDR enabled.

## Retrieval-augmented generation

*Tested in: CCAR-P 3.5 (RAG pipeline with chunking and indexing), 3.6 (retrieval strategies matched to data shape and query pattern), 3.8 (progressive discovery vs. monolithic context), 4.4 (diagnose system issues; Sample 3 is tagged to Domain 4) · CCDV-F 2.3 Claude API Mechanics (Messages API data access patterns) · CCAO-F D5.1 (Projects with knowledge sources) · CCAR-F: embedding models and vector database implementation details are out of scope*

Anthropic's glossary defines retrieval-augmented generation as retrieving data from an external knowledge base at the moment a query is sent and passing it into the context window; the model does not necessarily do the retrieving itself, although it can through tool use ([Glossary](https://platform.claude.com/docs/en/about-claude/glossary)). The CCAR-P guide expects hands-on experience: its preparation list says to "Build and operate at least one end-to-end Claude solution, including RAG, evaluation, and observability", and its Sample 3 rationale says confident wrong answers after a document refresh "point to retrieval feeding the model poor context, for example a broken re-index or mismatched embeddings" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). The sample itself, with its options, is reproduced on the [CCAR-P exam page](../claude-certified-architect-professional.md#official-sample-questions).

### First decide whether you need a pipeline

| Approach | Choose it when | Watch for |
|---|---|---|
| The whole corpus in the prompt, with prompt caching | The corpus is small and stable, and every user may see all of it | Recall degrades as the window fills ("context rot") |
| A retrieval pipeline (embeddings, BM25, reranking) | Freshness or access control drives the design, or the corpus is too large to send | The index goes stale unless it is maintained, and retrieval needs its own evaluation, separate from end-to-end answers |
| Agentic, just-in-time search | The agent can navigate the data itself: codebases, file trees, stored queries | Runtime exploration is slower than retrieving pre-computed data |
| Hybrid: some context up front, the rest explored | Less dynamic content, such as legal or finance work | Deciding what is always loaded |
| A tool call to the system of record | The answer is live state: an order, a balance, a ticket | Retrieval doing a job that live state should own |
| Claude Projects | Business users with a document set; on paid plans, RAG mode switches on automatically as project knowledge approaches the context limit, expanding capacity by up to 10x | RAG mode is on paid plans only (Pro, Max, Team, Enterprise), although Projects themselves are on every plan (Free: up to five); activation is automatic, with no manual control |

!!! note "A 2024 rule of thumb in a 1M-token era"

    Anthropic's [Contextual Retrieval post](https://www.anthropic.com/engineering/contextual-retrieval) (September 2024) says: "If your knowledge base is smaller than 200,000 tokens (about 500 pages of material), you can just include the entire knowledge base in the prompt that you give the model, with no need for RAG or similar methods." It credits prompt caching with reducing latency by more than 2x and cost by up to 90% for that approach. As of September 2026, Claude Fable 5.1, Claude Opus 5.5 and Claude Sonnet 5 have 1M-token context windows (Claude Haiku 4.5 has 200K), and on Claude 4.6 and later models a 900k-token request is billed at the same per-token rate as a 9k-token one. The 2026 pilot-to-production guide from Anthropic and Accenture ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf)) adds: "Retrieval layers still add genuine value when data freshness or access control are the primary drivers. But many existing pipelines are solving for a context window constraint that no longer applies." Treat 200,000 tokens as the post's 2024 guidance, not a current rule, and remember that recall still degrades as the window fills. The context side of this decision (CCAR-P 3.8) is taught in [Why context is a budget](context-engineering.md#why-context-is-a-budget).

### The pipeline, stage by stage

```text
INGEST (once per document version)
  1. Split each document into chunks (usually no more than a few hundred tokens)
  2. Contextualize each chunk: Claude writes 50 to 100 tokens that situate it in its document
  3. Index twice: embeddings in a vector index, and BM25 in a lexical index

QUERY (every request)
  4. Retrieve candidates from both indexes; merge and deduplicate with rank fusion
  5. Rerank the candidates (Anthropic's test: top 150 in, top 20 out)
  6. Give the chunks to Claude as search results or documents, citations on
  7. Generate the answer, citing the chunks it used

EVALUATE (continuously)
  8. Measure retrieval (recall@k, MRR) separately from end-to-end answer accuracy
```

### Chunking

Standard preprocessing splits the corpus into chunks of "usually no more than a few hundred tokens", and "The choice of chunk size, chunk boundary, and chunk overlap can affect retrieval performance" ([Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)). Anthropic's RAG cookbook chunks by heading, so each chunk holds the content of one subheading. Voyage's server-side auto-chunking resolves `chunk_size` to 512 tokens when it is omitted, `chunk_overlap` defaults to 0, and overlapping tokens are billed as input. Chunking also decides citation granularity: in the Citations feature, plain-text documents are split into sentences, while custom content documents are used as-is, so put each chunk in its own plain-text document if you want sentence-level citations.

### Contextual Retrieval: the published numbers

A chunk such as "The company's revenue grew by 3% over the previous quarter." does not say which company or which quarter, so it retrieves poorly. [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval), published on Anthropic's engineering blog on September 19, 2024, prepends chunk-specific explanatory context to each chunk before embedding it (Contextual Embeddings) and before building the BM25 index (Contextual BM25). The generated context is usually 50 to 100 tokens.

```python
original_chunk = "The company's revenue grew by 3% over the previous quarter."
contextualized_chunk = "This chunk is from an SEC filing on ACME corp's performance in Q2 2023; the previous quarter's revenue was $314 million. The company's revenue grew by 3% over the previous quarter."
```

The [post's](https://www.anthropic.com/engineering/contextual-retrieval) headline claim: "This method can reduce the number of failed retrievals by 49% and, when combined with reranking, by 67%." The measured results, on the post's metric of 1 minus recall@20 (the percentage of relevant documents not retrieved within the top 20 chunks):

| Configuration | Top-20 retrieval failure rate | Reduction |
|---|---|---|
| Starting point in the post's comparison | 5.7% | |
| Contextual Embeddings | 3.7% | 35% |
| Contextual Embeddings + Contextual BM25 | 2.9% | 49% |
| Contextual Embeddings + Contextual BM25 + reranking | 1.9% | 67% |

The conditions behind those numbers matter as much as the numbers:

- **Averaged across domains.** The graphs average the knowledge domains tested (codebases, fiction, ArXiv papers, science papers) using the top-performing embedding configuration, Gemini Text 004, retrieving the top 20 chunks.
- **Reranking setup.** Initial retrieval took the top 150 chunks, the reranker kept the top 20, and the tests used the Cohere reranker; Voyage also offers one, which the post did not test. In the [post's](https://www.anthropic.com/engineering/contextual-retrieval) words, "There is an inherent trade-off between reranking more chunks for better performance vs. reranking fewer for lower latency and cost."
- **How many chunks to pass.** Anthropic tried 5, 10 and 20 chunks, and 20 performed best of those options.
- **Embeddings.** Contextual Retrieval helped every embedding model tested; Gemini and Voyage embeddings were particularly effective, and embeddings plus BM25 beat embeddings alone.
- **Cost.** With prompt caching, the post put the one-time cost to contextualize at &#36;1.02 per million document tokens, assuming 800-token chunks, 8k-token documents, 50-token context instructions and 100 tokens of context per chunk. The post generated that context with Claude 3 Haiku, now retired, and the current cookbook repeats the same figure while using `claude-haiku-4-5`; recompute it at current prices before you budget. As of September 2026, Claude Haiku 4.5 lists &#36;1 per million input tokens, &#36;0.10 per million cache-hit tokens and &#36;5 per million output tokens.
- **What did not work as well.** Adding generic document summaries to chunks gave very limited gains, and summary-based indexing performed poorly.
- **Tuning.** The post says the generic prompt works well and a prompt tailored to your domain or use case may do even better, for example one that includes a glossary of key terms that might only be defined in other documents.

The contextualizer prompt from the post:

```text
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else.
```

The same call through the Messages API, adapted from Anthropic's [contextual embeddings cookbook](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/capabilities/contextual-embeddings/guide.ipynb). The whole document carries `cache_control`, so it is cached once and every later chunk from that document reads it from the cache (the cookbook's note: "Read the document from cache (90% discount on those tokens)"):

```python
import anthropic

client = anthropic.Anthropic()

DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""

def situate_context(doc: str, chunk: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc),
                        "cache_control": {"type": "ephemeral"},  # cache the full document
                    },
                    {
                        "type": "text",
                        "text": CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk),
                    },
                ],
            }
        ],
    )
    return response.content[0].text
```

The main changes from the cookbook: the model name is written inline instead of through its `MODEL_NAME` variable, the function returns the text rather than the whole response, and the cookbook's `temperature=0.0` is left out. The [model deprecations page](https://platform.claude.com/docs/en/about-claude/model-deprecations) explains the last one: "The Python SDK (v1.0 and later) removes `temperature`, `top_p`, and `top_k`, so passing them raises a `TypeError`", and at the API level, Claude 4.7 and later models (not the Claude Haiku 4.5 used here) return a 400 error for a non-default `temperature`. Keep one more constraint in mind: the minimum cacheable prompt is 4,096 tokens on Claude Haiku 4.5 (512 on Claude Fable 5.1 and Claude Opus 5.5), and a shorter prompt is processed without caching and without an error, so short documents get no caching benefit.

!!! note "Models and numbers, then and now"

    The 2024 post generated chunk context with Claude 3 Haiku, which was retired on April 20, 2026; the current cookbook uses `claude-haiku-4-5`, whose own retirement is listed as "Not sooner than October 15, 2026" on the [model deprecations page](https://platform.claude.com/docs/en/about-claude/model-deprecations), so check its status before you build on it. The cookbook also measures its own pipeline differently, with Pass@k (whether the golden chunk appears in the first k results) over 248 queries across 9 codebases, so its figures are not comparable with the post's. Its Pass@10 results: baseline RAG 87.15%, with contextual embeddings 92.34%, with hybrid BM25 search 93.21%, and with Cohere reranking applied to contextual embeddings alone (not the hybrid) 95.26%. Reranking adds about 100 to 200 ms per query ([contextual embeddings cookbook](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/capabilities/contextual-embeddings/guide.ipynb)).

### Embeddings

"Anthropic does not offer its own embedding model." Its [embeddings page](https://platform.claude.com/docs/en/build-with-claude/embeddings) points to Voyage AI while telling readers to assess vendors on dataset size and domain specificity, inference performance, and customization. A selection of the Voyage models it lists, as of September 2026 (the page also lists `voyage-4-nano`, earlier general-purpose models, multimodal models and `voyage-context-3`, and it files the three domain models in its previous-generation table while still recommending them for their domains):

| Model | Context (tokens) | Dimensions | Positioning |
|---|---|---|---|
| `voyage-4-large` | 32,000 | 1024 (default), 256, 512, 2048 | Best quality |
| `voyage-4` | 32,000 | 1024 (default), 256, 512, 2048 | Balanced |
| `voyage-4-lite` | 32,000 | 1024 (default), 256, 512, 2048 | Lowest latency and cost |
| `voyage-code-3` | 32,000 | 1024 (default), 256, 512, 2048 | Code and programming documentation |
| `voyage-law-2` | 16,000 | 1024 | Legal and long-context retrieval |
| `voyage-finance-2` | 32,000 | 1024 | Finance |
| `voyage-context-4` | 120,000 (Voyage's page: 32,000 per chunk, and the 120,000 total across all inputs applies only with auto-chunking; otherwise 32,000) | 1024 (default), 256, 512, 2048 | Contextualized chunk embeddings, called with `contextualized_embed()` instead of `embed()` |

- **Set `input_type`.** Use `"document"` when indexing and `"query"` when searching: "Do not omit `input_type` or set `input_type=None`." ([embeddings page](https://platform.claude.com/docs/en/build-with-claude/embeddings))
- **Similarity.** Voyage embeddings are normalized to length 1, so cosine similarity equals dot product, and the dot product is faster.
- **Storage.** `output_dtype` quantization to int8 or binary cuts storage, memory and cost by 4x and 32x respectively; Matryoshka embeddings can be truncated by keeping the leading dimensions, and the embeddings page's example then re-normalizes the shortened vectors to unit length.
- **Compatibility.** All Voyage 4 series models produce compatible embeddings, so you can embed documents with one and queries with another (Voyage's [Voyage 4 announcement](https://blog.voyageai.com/2026/01/15/voyage-4/) calls this "asymmetric retrieval") and upgrade the query model without re-vectorizing documents.
- **Price.** As of September 2026, Voyage's pricing table makes the first 200 million tokens free for every account on `voyage-4-large`, `voyage-4`, `voyage-4-lite`, `voyage-context-4` and `voyage-code-4`, and the first 50 million on `voyage-finance-2` and `voyage-law-2`; after that `voyage-4-large` costs &#36;0.12 and `voyage-4-lite` &#36;0.02 per million tokens. Voyage's Batch API has a 12-hour completion window and a 33% discount, and the free tokens do not apply to it.
- **Naming drift.** Anthropic's page recommends `voyage-code-3`, while Voyage's own pricing table now lists `voyage-code-4` and moves `voyage-code-3` to its older models, which it says get no free tokens (the page's introductory paragraph still names `voyage-code-3` among the 200-million-token models).

Embedding documents and a query, then scoring by dot product (from the embeddings page, where `documents` and `query` are defined):

```python
import voyageai
import numpy as np

vo = voyageai.Client()  # reads VOYAGE_API_KEY

doc_embds = vo.embed(documents, model="voyage-4", input_type="document").embeddings
query_embd = vo.embed([query], model="voyage-4", input_type="query").embeddings[0]

similarities = np.dot(doc_embds, query_embd)
retrieved_id = np.argmax(similarities)
```

Contextualized chunk embeddings do the contextualizing inside the embedding model, capturing full-document context without manual metadata augmentation. With auto-chunking (`enable_auto_chunking=True`), you pass a flat list of whole documents and `input_type` must be `document`; this example is from Voyage's [contextualized chunk embeddings page](https://docs.voyageai.com/docs/contextualized-chunk-embeddings):

```python
import voyageai

vo = voyageai.Client()

documents = [
    "This is the SEC filing on Leafy Inc.'s Q2 2024 performance.\nThe company's revenue increased by 15% compared to the previous quarter.",
    "This is the SEC filing on Elephant Ltd.'s Q2 2024 performance.\nThe company's revenue decreased by 2% compared to the previous quarter.",
]

result = vo.contextualized_embed(
    model="voyage-context-4",
    inputs=documents,
    input_type="document",
    enable_auto_chunking=True,
    chunk_size=512,
    chunk_overlap=0,
)
```

### Lexical search, hybrid fusion and reranking

BM25 is a ranking function that uses lexical matching to find precise word or phrase matches, which suits unique identifiers and technical terms. The post's example is a query for error code "TS-999": an embedding model can return content about error codes in general and miss the exact match. Hybrid retrieval runs both searches and combines and deduplicates the results with rank fusion. The contextual embeddings cookbook takes the top 150 results from each and fuses them with weighted Reciprocal Rank Fusion, 80% semantic and 20% BM25 by default.

Rerankers are cross-encoders that score the query and each document together, which is why they run on the top candidates from a first-stage search rather than on the whole corpus. Anthropic's [embeddings page](https://platform.claude.com/docs/en/build-with-claude/embeddings) lists `rerank-2.5` ("Highest accuracy. Recommended for most applications.") and `rerank-2.5-lite` (latency and cost), both with a 32,000-token context and called with `rerank()`. Voyage's limits: a request can carry at most 1,000 documents, and the query can be up to 8,000 tokens for `rerank-2.5` and `rerank-2.5-lite`. Claude can also act as the reranker: by default the RAG cookbook retrieves 20 candidates and has Claude pick the most relevant 3.

From Voyage's [reranker page](https://docs.voyageai.com/docs/reranker), where `query` and `documents` are the same six-document example used on the embeddings page:

```python
import voyageai

vo = voyageai.Client()
reranking = vo.rerank(query, documents, model="rerank-2.5", top_k=3)
for r in reranking.results:
    print(f"Document: {r.document}")
    print(f"Relevance Score: {r.relevance_score}")
```

Voyage's own [reranker page](https://docs.voyageai.com/docs/reranker) now labels a preview `rerank-3` as "Highest accuracy" while still listing `rerank-2.5` and `rerank-2.5-lite` as recommended options; Anthropic's page gives that label to `rerank-2.5`.

### Handing retrieved content to Claude

- **Search result blocks.** A `search_result` content block lets Claude cite your own retrieved content with the source and title you supply, the same way it cites web search. It needs no beta header and can arrive in a tool result (dynamic RAG) or as top-level user content (pre-fetched content).
- **Citations on documents.** Put each chunk in its own plain-text document for sentence-level citations, or use a custom content document when you do not want any further chunking. The `cited_text` field does not count toward output tokens.
- **Long documents.** For tasks over long documents (more than 20k tokens), ask Claude to extract word-for-word quotes before doing the task, to reduce hallucinations.
- **Untrusted text.** Retrieved content is third-party content. Anthropic's injection guidance is to deliver it inside `tool_result` blocks, never in `system` prompts or plain user `text` blocks, so returning `search_result` blocks from a retrieval tool meets both the citation guidance and the injection guidance. See [Prompt injection](security-and-governance.md#prompt-injection).
- **An index you already run.** In Claude Code, the large-codebase guidance is to expose an existing code search or RAG index as an MCP tool so Claude queries it instead of reading files directly.

```json
{
  "type": "search_result",
  "source": "https://example.com/article",
  "title": "Article Title",
  "content": [
    { "type": "text", "text": "The actual content of the search result..." }
  ],
  "citations": { "enabled": true }
}
```

Field-level rules for search results, documents and citations are in [Files, citations and search results](claude-api.md#files-citations-and-search-results).

### Retrieval strategy by data shape and query pattern

Data shape is what the data is; query pattern is how people ask for it. Match the retriever to both:

| Data shape or query pattern | Strategy | What it prevents |
|---|---|---|
| Prose, questions phrased many ways | Semantic search over embeddings | Missing passages worded differently from the query |
| Exact terms: error codes, identifiers, function names | BM25, usually inside a hybrid | Embeddings returning general matches and missing the exact one |
| Code, legal or financial text | A domain embedding model (`voyage-code-3`, `voyage-law-2`, `voyage-finance-2`) | A general model where Anthropic's page lists a domain one |
| Relational tables | Text-to-SQL: instructions, the user's query and the schema in the prompt; retrieve only the relevant schema when it is large; let Claude run the SQL and improve it from results or errors | Sending an impractically large schema on every request |
| Live state (orders, balances) | A tool call to the system of record | Answers from a stale copy |
| Codebases and file trees | Agentic search with identifiers loaded just in time; start here and add semantic search only for speed or variety | Stale indexes |
| Catalogs and schemas behind an MCP server | MCP resources and resource templates, which the host application decides how to include | Hard-coding reference data into prompts |
| Breadth-first research questions | Parallel subagents, each starting with short, broad queries and narrowing | Overly long, specific queries |

The [text-to-SQL cookbook](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/capabilities/text_to_sql/guide.ipynb) builds the schema description it sends to Claude like this (SQLite; the cookbook's inline comments removed):

```python
def get_schema_info(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    schema_info = []
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    for (table_name,) in tables:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        table_info = f"Table: {table_name}\n"
        table_info += "\n".join(f"  - {col[1]} ({col[2]})" for col in columns)
        schema_info.append(table_info)
    conn.close()
    return "\n\n".join(schema_info)
```

The cookbook also lists extra context that helps SQL generation (data samples, column statistics, data-quality notes, data-catalog details and business context) and, for growing schemas, suggests filtering the schema lookup to recently queried tables and ranking it by how often tables are queried in production.

### Keeping the index correct

In the CCAR-P Sample 3 scenario (confident wrong answers right after a document refresh, with the model and latency unchanged), the keyed answer is the retrieval and indexing step, and the rationale dismisses the model, temperature and context-window options because they "would not be triggered specifically by a document refresh" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). The question to ask is what changed. Anthropic's RAG cookbook shows one way a refresh goes wrong: in this excerpt of its `VectorDB.load_data` method, the loader reuses a saved index from disk and skips re-embedding, so a refreshed corpus keeps serving the old chunks until the index is rebuilt.

```python
def load_data(self, data):
    if self.embeddings and self.metadata:
        print("Vector database is already loaded. Skipping data loading.")
        return
    if os.path.exists(self.db_path):
        print("Loading vector database from disk.")
        self.load_db()
        return

    texts = [f"Heading: {item['chunk_heading']}\n\n Chunk Text:{item['text']}" for item in data]
    self._embed_and_store(texts, data)
    self.save_db()
```

- **Re-index when documents change**, and re-run retrieval metrics afterward.
- **Keep documents and queries on compatible embeddings.** The cookbook embeds both with the same model; Voyage documents compatibility within the 4 series. Switching the query model outside a compatible series without re-embedding the corpus is one way to get the "mismatched embeddings" of the CCAR-P Sample 3 rationale ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)).
- **Watch input limits.** If results get worse after contextualizing, the embedding model may be truncating the longer chunks; the cookbook's advice is a model with a larger context window.
- **Update schema indexes** when database schemas change.
- **Keep project knowledge current** in Claude Projects, because outdated documents can lead to outdated responses.
- **Or avoid the index.** Just-in-time retrieval with tools such as glob and grep bypasses stale indexing altogether.

### Evaluating retrieval

"When evaluating RAG applications, it's critical to evaluate the performance of the retrieval system and end to end system separately." ([RAG cookbook](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/capabilities/retrieval_augmented_generation/guide.ipynb))

| Metric | Measures | Used in |
|---|---|---|
| Precision | Share of retrieved chunks that are relevant | RAG cookbook |
| Recall | Share of all correct chunks that were retrieved | RAG cookbook |
| F1 | Balance of precision and recall | RAG cookbook |
| Mean Reciprocal Rank (MRR) | How high the first correct result ranks (the average of 1/rank per query), from 0 to 1, where 1 means it always comes first | RAG cookbook |
| End-to-end accuracy | Whether the final answer is correct, judged by an LLM against ground truth | RAG cookbook |
| 1 minus recall@20 | Share of relevant documents missing from the top 20 chunks | Contextual Retrieval post |
| Pass@k | Whether the golden chunk appears in the first k results | Contextual embeddings cookbook |

Many RAG systems favor recall, because the model can ignore less relevant chunks during generation. The RAG cookbook's eval set had 100 synthetic samples (question, expected chunks, correct answer), and moving from basic RAG to summary indexing to reranking with Claude raised MRR from 0.74 to 0.87 and end-to-end accuracy from 71% to 81%. The cookbook's summary indexing has Claude write a 2 to 3 sentence summary of each chunk, then embeds the chunk's heading, that summary and the original text together. The [Contextual Retrieval post](https://www.anthropic.com/engineering/contextual-retrieval) lists summary-based indexing among approaches that performed poorly in its evaluation, without describing how that index was built, so read the two results as separate experiments rather than a contradiction. Program-level evaluation design is in [Evaluation strategy for a program](#evaluation-strategy-for-a-program).

### Decide

- If the corpus is small, stable and open to every user, choose to put it in the prompt with prompt caching; if freshness or per-user access control drives the design, choose retrieval.
- If queries contain exact identifiers, choose hybrid retrieval with BM25; not embeddings alone.
- If chunks lose meaning outside their document, choose contextualized chunks (Contextual Retrieval, or contextualized chunk embeddings); not generic document summaries attached to chunks.
- If precision in the top results matters, choose to retrieve a wide candidate set and rerank; weigh the added latency and cost per query.
- If answers go wrong after a document refresh while the model and latency are unchanged, choose to investigate retrieval and indexing first; not the model, the temperature or the context window.
- If the answer is live state, choose a tool call to the system of record; not a vector index.

### Traps

- **Judging the pipeline only on end-to-end accuracy**, so retrieval regressions stay invisible.
- **Omitting `input_type`**, or mixing embedding models between index and queries outside a compatible series.
- **Quoting the 49% and 67% figures without their conditions**: 1 minus recall@20, averaged across the post's domains, reranking 150 down to 20.
- **Treating the 200,000-token threshold as current guidance.**
- **Letting a vector index stand in for the source of truth** on fast-changing state.
- **Putting retrieved text in the system prompt.** Anthropic's guidance is to deliver third-party content inside `tool_result` blocks, never in system prompts or plain user text blocks.

## Integration patterns

*Tested in: CCAR-P 1.2 (end-to-end architectures: input, processing, output, feedback loops), 3.2 (authentication and authorization gaps), 3.7 (connection protocols: MCP, API/CLI, agent-to-agent) · CCDV-F 2.3 Claude API Mechanics (realtime vs. batch), 2.4 Software Engineering Foundations (REST APIs, JSON, asynchronous programming), 5.2 Technical Fundamentals (SDKs that wrap REST APIs, websockets) · CCAO-F D4.4 (integrate Claude into existing workflows) · CCAR-F 4.5 (the Message Batches API: 4.5-K1 to 4.5-K4, 4.5-S1 to 4.5-S4) and 3.6-K1, 3.6-K2 (Claude Code in CI/CD); CCAR-F lists deploying or hosting MCP servers and streaming API implementation as out of scope*

An integration answers three questions: how a request reaches Claude and how long the caller waits (the invocation mode), how Claude reaches your systems (the connection mechanism), and where credentials and trust boundaries sit. The CCAR-P prep course puts the production version in one line: "Take a solution from proof of concept to production by mapping cost, latency, and reliability to a budget and specifying the integration patterns an enterprise will accept" ([CCAR-P prep course](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional)).

### A reference architecture, end to end

CCAR-P objective 1.2 asks for "input → processing → output → feedback loops" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). The arrangement below, including a separate ACTION stage, is ours; each control in it comes from Anthropic guidance:

```text
INPUT        a user, an event, a schedule or a pipeline trigger
  |          screen untrusted input with a lightweight model such as Claude Haiku 4.5
  v
PROCESSING   assemble context: instructions, retrieved chunks, live data from tools
  |          run the call or the agent loop; authorize each tool call before it runs
  v
OUTPUT       validate structure and content; screen the output
  |          route by risk: automated, sampled, reviewed, or advisory to a human
  v
ACTION       write back to systems of record through narrowly scoped tools
  |
  v
FEEDBACK     log requests and outcomes; monitor for quality drift;
             feed production data back into the offline evaluation set
```

The official [prep course](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional) describes the safety stack the same way: input screening, output screening and tool-call authorization, placed "so the system fails closed rather than open". Each stage is taught elsewhere: screening and review tiers in [Governance and risk in delivery](#governance-and-risk-in-delivery), output checks in [Validation, retry and feedback loops](prompt-engineering.md#validation-retry-and-feedback-loops), monitoring in [Reliability engineering for Claude applications](evaluation-and-reliability.md#reliability-engineering-for-claude-applications), and the evaluation loop in [Evaluation strategy for a program](#evaluation-strategy-for-a-program).

### Invocation modes

| Mode | How | Choose it when | Facts to know |
|---|---|---|---|
| Synchronous | `POST /v1/messages` through an SDK; the API is stateless | A caller is blocked waiting, such as a pre-merge check | The SDKs retry twice by default with backoff; Python's default timeout is 10 minutes |
| Streaming | `"stream": true`, delivered as server-sent events | A person watches the output, or the request is long | Time to first token is the responsiveness metric; avoid a large `max_tokens` without streaming |
| Batch | `POST /v1/messages/batches`, then poll and match results by `custom_id` | Nobody is waiting: overnight reports, weekly audits, nightly test generation | 50% of standard prices; most batches finish in under 1 hour and expire at 24 hours; up to 100,000 requests or 256 MB; results kept 29 days; result order not guaranteed |
| Hosted agent session | Managed Agents: send events, receive server-sent events; history persisted server-side; webhooks and scheduled deployments | Long-running tasks and asynchronous agent work | Webhooks carry only the event `type` and `id`; sessions have no batch mode |
| Headless CLI | `claude -p` with `--output-format json` (add `--json-schema` for schema-conforming output), or Claude Code GitHub Actions | CI/CD pipelines and scripts in any language | `-p` prevents interactive input hangs in CI |

The CI side of the headless row (flags, structured findings, review loops) is taught in [Claude Code in CI/CD](claude-code-workflows.md#claude-code-in-cicd).

CCDV-F Sample 1 (Domain 2) tests this choice and gives the rule in Anthropic's words: "The Message Batches API is designed for latency-tolerant, high-volume workloads at lower cost, which matches an overnight, non-urgent job." It also names the wrong answers: "Sending requests synchronously in parallel (A) does not reduce per-token cost; lowering max_tokens (C) or blindly downsizing the model (D) does not address the batch-versus-realtime tradeoff." ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). Anthropic's cost guide says the same thing as a policy: "Route every request no one is waiting on through a batch, and keep the interactive path for the rest." ([Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence))

A batch can still meet a service level if you schedule submissions around its window. The [CCAR-F guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) example is "4-hour windows to guarantee 30-hour SLA with 24-hour batch processing": a document waits at most 4 hours for the next submission and then up to 24 hours for processing, 28 hours in the worst case, inside the 30-hour commitment (our arithmetic). A request still unprocessed when its batch reaches 24 hours comes back `expired` rather than completed, so the 28 hours bounds when you have either a result or an expired request to resubmit; batch itself has no guaranteed latency SLA. The same task statement also expects you to resubmit only the failed documents, identified by `custom_id`, and to refine the prompt on a sample set before a large batch. Batch request mechanics are in [Message Batches](claude-api.md#message-batches) and batch design in [Batch processing design](prompt-engineering.md#batch-processing-design).

!!! warning "Exam guide vs current docs: tools inside a batch"

    The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says "The batch API does not support multi-turn tool calling within a single request (cannot execute tools mid-request and return results)". As of September 2026, the [batch processing documentation](https://platform.claude.com/docs/en/build-with-claude/batch-processing) says "The batch worker runs the same server-side agentic loop as the synchronous Messages API", so server tools (web search, code execution and the others) can loop inside one batched request, and a `pause_turn` result is continued with a follow-up request (batch or synchronous). Client tools, the ones your code executes, are a round trip by design: the model asks, you execute, you report back. The batch page does not address client tools directly; our reading is that a single batched request cannot provide that round trip mid-request, so the guide's statement still holds for your own tools. Answer CCAR-F items in the guide's wording.

### SDKs, REST and concurrency

The Claude API is a RESTful API at `https://api.anthropic.com`, and the official SDKs wrap it. The Python SDK supports synchronous and asynchronous operation, streaming, and the Bedrock, Claude Platform on AWS, Google Cloud and Foundry integrations; the TypeScript SDK is promise-based and consumes streams with `for await`. Both retry transient failures twice by default; the Python SDK lists connection errors, 408, 409, 429 and 500 and above. Messages API streaming is server-sent events, which are one-way from server to client; WebSockets, by contrast, are full-duplex after an HTTP Upgrade handshake. WebSockets appear in the Claude stack at the edges: an Agent SDK container can expose an HTTP or WebSocket endpoint in front of its sessions (the `claude` subprocess itself talks to the SDK over stdio), and Claude Code accepts WebSocket MCP servers for servers that push events; see [MCP transports](tool-use-and-mcp.md#mcp-transports).

Fan-out is where integrations break. Rate limits can be enforced over intervals shorter than a minute, so an unbounded burst of parallel requests trips 429s, and the Agent SDK hosting guide lists large parallel-subagent fan-outs hitting rate limits as a known limitation, with the advice to break work into smaller batches rather than issuing one wide dispatch. Cap the requests in flight. The pattern below combines the Python SDK's `AsyncAnthropic` client and the promise-based TypeScript client with standard concurrency primitives (`asyncio.Semaphore` and `asyncio.gather` in Python, `Promise.all` over a fixed pool of workers in TypeScript). The limit of 10 is illustrative; size it from your rate limits.

=== "Python"

    ```python
    import asyncio
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic()
    sem = asyncio.Semaphore(10)  # cap in-flight requests

    async def classify(text: str):
        async with sem:
            return await client.messages.create(
                model="claude-opus-5-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": text}],
            )

    async def main(texts):
        return await asyncio.gather(*(classify(t) for t in texts))
    ```

=== "TypeScript"

    ```typescript
    import Anthropic from "@anthropic-ai/sdk";

    const client = new Anthropic();

    async function classifyAll(texts: string[], limit = 10): Promise<Anthropic.Message[]> {
      const results: Anthropic.Message[] = new Array(texts.length);
      let next = 0;
      async function worker(): Promise<void> {
        while (next < texts.length) {
          const i = next++; // claim the next text
          results[i] = await client.messages.create({
            model: "claude-opus-5-5",
            max_tokens: 1024,
            messages: [{ role: "user", content: texts[i] }],
          });
        }
      }
      // at most `limit` requests in flight
      await Promise.all(Array.from({ length: Math.min(limit, texts.length) }, () => worker()));
      return results;
    }
    ```

`asyncio.gather` returns results in the order the awaitables were given; by default it propagates the first exception immediately to the awaiting task while the other awaitables keep running, and `Promise.all` rejects on the first rejected promise. Either way one failure costs you the aggregate result, so catch errors per request (or pass `return_exceptions=True` to `gather`) when one failure should not discard the others. If the work is latency-tolerant, a batch is a better answer than any concurrency limit. And keep production on the Anthropic SDKs: the OpenAI SDK compatibility layer is "primarily intended to test and compare model capabilities, and is not considered a long-term or production-ready solution for most use cases"; through it the function-calling `strict` parameter is ignored and prompt caching is not supported ([OpenAI SDK compatibility](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk)).

### Connecting Claude to your systems

The three-way comparison of direct API calls, CLIs and MCP, with agent-to-agent alongside, is in [Built-in tools, custom tools, Skills or MCP](tool-use-and-mcp.md#built-in-tools-custom-tools-skills-or-mcp). Its architectural consequences, from Anthropic's post on [agents that reach production systems](https://claude.com/blog/building-agents-that-reach-production-systems-with-mcp):

- Direct integrations multiply into the M×N problem: every agent and service pair needs its own auth handling, tool descriptions and edge cases.
- The [post's](https://claude.com/blog/building-agents-that-reach-production-systems-with-mcp) advice: "if your goal is to have production agents in the cloud reach your system, build an MCP server"; a remote MCP server is the only configuration that runs across web, mobile and cloud-hosted agents.
- Anthropic expects mature integrations to ship all three layers: the API as the foundation, a CLI for local-first environments, and MCP for cloud agents. Where a shell is available, Claude Code's best practices still call CLI tools the most context-efficient way to reach external services.
- Group tools around intent rather than mirroring endpoints. For very large APIs, a thin surface that accepts code works: Cloudflare's MCP server covers about 2,500 endpoints with two tools (search and execute) in roughly 1K tokens.

The pattern depends on the Claude surface and on where the system sits, including behind a firewall or an identity provider:

| Situation | Pattern | Constraints (as of September 2026) |
|---|---|---|
| A Messages API application calls a public remote MCP server | The MCP connector: the server in `mcp_servers` plus an `mcp_toolset` entry in `tools`, beta header `mcp-client-2025-11-20` | Publicly exposed HTTP only (Streamable HTTP or SSE), no local stdio servers, tool calls only; not on Amazon Bedrock or Google Cloud; not ZDR-eligible |
| A Managed Agents session or a Messages API application must reach an MCP server inside a private network | MCP tunnels: attach the tunnel hostname to a Managed Agents session, or pass it to the Messages API through the MCP connector | Outbound-only connection, no inbound ports; research preview (access by request), provided "as-is" ([MCP tunnels](https://platform.claude.com/docs/en/agents-and-tools/mcp-tunnels/overview)) with no uptime, support or continuity commitment, dependent on Cloudflare for the transport, and open to change or discontinuation |
| Claude or Cowork users (web, desktop or mobile) need an internal system through a custom connector | A custom connector (remote MCP) | Claude connects from Anthropic's cloud, so the server must be reachable over the public internet from Anthropic's IP ranges; on Team and Enterprise only Owners add custom connectors |
| Claude Desktop users need a local MCP server | A desktop extension (a single-click installable package for a local MCP server) | Runs locally rather than from Anthropic's cloud; available only in Claude Desktop and Claude Code, not on web or mobile |
| Access should follow the organization's identity provider | Enterprise-managed authorization for connectors | Works only for connectors whose provider supports it; beta on Team and Enterprise |
| A Claude Code team shares the same servers | One central team configures them and commits `.mcp.json` | Taught in [MCP servers in Claude Code](claude-code-configuration.md#mcp-servers-in-claude-code) |

The connector request shape, the tool configuration inside `mcp_toolset` and deferred loading for large MCP tool sets are in [MCP in the API and the Agent SDK](tool-use-and-mcp.md#mcp-in-the-api-and-the-agent-sdk).

### Agent-to-agent integration

When the system on the other side is itself an agent, owned by another team, vendor or organization, the CCAR-P guide's third option applies. The guide names no protocol for this option; one open standard for it is A2A: "The Agent2Agent (A2A) Protocol is an open standard designed to facilitate communication and interoperability between independent, potentially opaque AI agent systems." Agents exchange information without access to each other's internal state, memory or tools ([A2A specification](https://raw.githubusercontent.com/a2aproject/A2A/main/docs/specification.md)). A2A is an open-source Linux Foundation project contributed by Google and licensed under Apache 2.0. As of September 2026 its specification page shows 1.0.0 as the latest released version, with bindings to JSON-RPC, gRPC and HTTP/REST.

| Concept | What it is |
|---|---|
| Agent Card | A JSON document the A2A server publishes, describing its identity, capabilities, skills, endpoint and authentication; found at `https://{server_domain}/.well-known/agent-card.json`, in a registry, or by configuration |
| Task | The unit of work, with a unique ID and a stateful lifecycle: `TASK_STATE_SUBMITTED`, `TASK_STATE_WORKING`, `TASK_STATE_INPUT_REQUIRED`, `TASK_STATE_AUTH_REQUIRED`, and the terminal `TASK_STATE_COMPLETED`, `TASK_STATE_FAILED`, `TASK_STATE_CANCELED`, `TASK_STATE_REJECTED`, which accept no further messages |
| Message and Part | One turn with a role of user or agent (serialized as `ROLE_USER` or `ROLE_AGENT`); parts carry text, file references or structured data |
| Artifact | An output made of Parts |
| `contextId` | Groups related tasks and messages; after a terminal state, a refinement starts a new task in the same `contextId` |
| Delivery | Synchronous request/response, streaming over server-sent events, or push notifications to a client webhook |

A basic task over the HTTP+JSON binding, request then response body, from the specification's illustrative examples (section 6.1, Basic Task Execution):

```text
POST /message:send HTTP/1.1
Host: agent.example.com
Content-Type: application/a2a+json
Authorization: Bearer token

{
  "message": {
    "role": "ROLE_USER",
    "parts": [{"text": "What is the weather today?"}],
    "messageId": "msg-uuid"
  }
}
```

```json
{
  "task": {
    "id": "task-uuid",
    "contextId": "context-uuid",
    "status": {"state": "TASK_STATE_COMPLETED"},
    "artifacts": [{
      "artifactId": "artifact-uuid",
      "name": "Weather Report",
      "parts": [{"text": "Today will be sunny with a high of 75°F"}]
    }]
  }
}
```

Security follows ordinary web practice. Identity is handled at the HTTP layer, not inside A2A payloads; production deployments must use HTTPS (TLS for gRPC); clients read the required schemes from the Agent Card's `securitySchemes`, obtain credentials out of band and send them on every request; the server must authenticate every request; and moving a task to `TASK_STATE_AUTH_REQUIRED` does not by itself authorize anything. The A2A project's own summary of the split with MCP: "A2A connects the agents to each other; MCP connects each agent to its own tools." ([A2A and MCP](https://raw.githubusercontent.com/a2aproject/A2A/main/docs/topics/a2a-and-mcp.md)).

Where Claude meets A2A, the protocol support comes from the framework or runtime around the model. Google Cloud describes a registered Claude-powered agent on its Agent Platform delegating tasks to other agents over A2A, Google's Agent Development Kit supports Claude models in Python and Java, and AWS added A2A support to Amazon Bedrock AgentCore Runtime, naming Anthropic Claude among the models whose agents can interoperate. Anthropic and Google Cloud also presented a webinar, dated August 27, 2025, on multi-agent systems using MCP and A2A with Claude on Vertex AI. So the accurate statement is that Claude can power agents that interoperate over A2A through a framework or runtime that implements it. When every agent is your own and they can share one sandbox, Anthropic's first-party option is Managed Agents multiagent orchestration, taught in [Multi-agent orchestration](agents-and-agent-sdk.md#multi-agent-orchestration).

### Credentials and trust boundaries

CCAR-P 3.2 asks you to find authentication and authorization gaps. At integration boundaries the recurring ones are:

| Boundary | Rule |
|---|---|
| API keys | "Set an expiration when you create your API key. Keep the key out of source control, client-side code, and prompts." ([Intro to Claude](https://platform.claude.com/docs/en/intro)); scope keys to one workspace |
| Long-lived keys in workloads | Workload Identity Federation swaps them for short-lived OIDC tokens from your identity provider, and it is only as strong as that provider |
| Browser front ends | The TypeScript SDK disables browser use unless `dangerouslyAllowBrowser` is set; under ZDR, CORS is not supported, so browser apps go through a backend proxy |
| Agent SDK deployments | Authenticate inbound requests at a gateway, because "The agent should receive pre-authenticated requests and should not be the component that validates user tokens"; inject outbound credentials from a proxy so the agent never sees them ([Agent SDK hosting](https://code.claude.com/docs/en/agent-sdk/hosting)) |
| MCP servers | A server must not accept tokens that were not issued for it (no token passthrough); use progressive, least-privilege scopes |
| Multi-tenant files | Never accept `file_id` values from end users; a workspace per tenant gives hard isolation |
| Connectors in Claude Enterprise | Claude inherits each member's own permissions in the source system; organization, role and member gates must all be open; write access deserves a different level of sign-off than read access |
| Central control for Claude Code | For request-level audit logging or routing by data sensitivity, Claude Code's admin guidance places a gateway between developers and the provider; the trade-off is that the gateway becomes infrastructure your organization operates |

The CCAR-P guide's Sample 1 is tagged to Domain 3 (Integration) and tests the tool side of the same boundary: a support agent holds refund and delete tools that its users never need, and the keyed answer removes them. The item and its rationale are worked through in [Design the safety stack to fail closed](#design-the-safety-stack-to-fail-closed).

Secrets handling and least privilege are taught in depth in [Secrets and API keys](security-and-governance.md#secrets-and-api-keys) and [Least privilege for tools and agents](security-and-governance.md#least-privilege-for-tools-and-agents). CCAR-F lists OAuth, API key rotation and authentication protocol details as out of scope. The CCAR-P guide publishes no such exclusion, and its objective 3.2 asks you to analyze authentication and authorization requirements to identify security gaps.

### Fitting Claude into an existing workflow

For CCAO-F D4.4 the integration is a work process rather than an API. The [workflow-augmentation lesson](https://academy.claude.com/courses/ai-fluency-for-nonprofits/workflow-augmentation) in Anthropic Academy's AI Fluency for nonprofits course sorts each task into three groups: AI can handle it; AI can assist and a human decides; or a human should handle it (high-stakes decisions, emotional situations, complex judgment calls). The same lesson says to test the new flow with real past examples and to review outputs before they go out, especially early on. Anthropic's research on productivity gains adds a caution: as AI accelerates some tasks, others may become bottlenecks, so watch the steps you did not change. Once a process works, Cowork can repeat it: do the task once, confirm the output, then type `/schedule`. Artifacts are for prototypes; the Academy says they are best for testing and demonstration, and that at some point you will likely want proper API key management and production-grade infrastructure. Before retiring the old process, run the new one alongside it until it proves reliable (the parallel run in [From pilot to production](#from-pilot-to-production)). See [Domain 4: Workflow Integration and Solution Design](../claude-certified-associate.md#domain-4-workflow-integration-and-solution-design) for the exam framing.

### Decide

- If a caller is blocked waiting (a pre-merge check, a live chat), choose the synchronous API, streamed when a person is watching; if nobody is waiting, choose Message Batches; not parallel synchronous calls, which do not reduce per-token cost.
- If many requests must run at once, choose bounded concurrency or a batch; not an unbounded fan-out that trips short-interval rate limits.
- If production agents in the cloud must reach your system, choose a remote MCP server; if the integration is one agent and one service that will not be reused, direct API calls are proportionate.
- If a Managed Agents session or a Messages API application must reach an MCP server inside a private network, choose MCP tunnels (outbound-only, no inbound firewall ports); not opening the server to the public internet. If the design needs an uptime or support commitment, tunnels do not carry one while they are a research preview, so plan for that gap.
- If claude.ai or Cowork users need the internal system through a custom connector, the server must be reachable over the public internet from Anthropic's IP ranges. Tunnels are not documented as an option there: the [MCP tunnels](https://platform.claude.com/docs/en/agents-and-tools/mcp-tunnels/overview) page lists Claude Managed Agents and the Messages API as the surfaces that reach tunneled servers, and notes that "MCP tunnels created through the Console are not available as connectors in claude.ai".
- If the counterpart is an independent agent from another team or vendor with stateful, multi-turn work, choose an agent-to-agent protocol such as A2A through a framework or runtime that implements it; if it is a well-defined capability, expose it as a tool.
- If a browser needs Claude, put a backend in front of the API; not an API key in client-side code.

### Traps

- **Batching a blocking workflow** such as a pre-merge check. The CCAR-F guide calls batch inappropriate for blocking workflows and notes that it has no guaranteed latency SLA.
- **Expecting batch results in submission order** instead of matching on `custom_id`.
- **Stacking your own retry loop on the SDK's automatic retries** (two by default; `max_retries` in Python, `maxRetries` in TypeScript, and 0 disables them) without adjusting either, which multiplies attempts against a rate limit (our reasoning).
- **Pointing the Messages API MCP connector at a local stdio server.** It connects only to publicly exposed HTTP servers (Streamable HTTP or SSE). For an HTTP MCP server inside a private network, use an MCP tunnel; the tunnel changes how the server is reached, not its transport.
- **Guarding a tool the role never needs** with logging or a confirmation prompt. CCAR-P Sample 1 keys removing the tool; see [Design the safety stack to fail closed](#design-the-safety-stack-to-fail-closed).
- **Saying the Claude API itself speaks A2A.** A2A support with Claude comes from frameworks and runtimes such as Google's ADK and Bedrock AgentCore.
- **Passing a user's token straight through an MCP server** to a downstream API; token passthrough is forbidden by the MCP security guidance.

## Deployment platforms and capacity

*Tested in: CCAR-P 1.6 (performance SLAs as a value pillar), 3.3 (accuracy-latency trade-offs and configuration decisions), 6.3 (expectation alignment, including SLAs) · CCDV-F 2.3 Claude API Mechanics (invoking Claude through third-party vendors) · CCAR-F: out of scope (rate limiting, quotas and pricing calculations; specific cloud provider configurations) · CCAO-F: not listed*

Two decisions finish the architecture: where the model runs, and how much traffic it can take. The CCAR-P certification page lists "Claude with Amazon Bedrock" and "Claude on Google Cloud" among its catalog prep courses, and the stakeholder module of the CCAR-P prep course has architects compare "the direct API, Bedrock, Vertex, and third-party routes on latency, compliance, and cost" ([Stakeholder Engagement, Lifecycle & GTM](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/stakeholder-engagement-lifecycle-gtm)). Client code for each platform is in [Claude on cloud platforms](claude-api.md#claude-on-cloud-platforms) and rate-limit mechanics (errors, headers, the tier table) in [Errors, retries and rate limits](claude-api.md#errors-retries-and-rate-limits); this section is about the architecture those facts produce.

### Five places to run Claude

As of September 2026:

| Platform | Operated by | Data boundary | What it changes for the design |
|---|---|---|---|
| Claude API | Anthropic | Anthropic is your processor under the DPA | Direct access to the latest models and features; fast mode is available only here; Message Batches; `inference_geo` |
| Claude Platform on AWS | Anthropic; AWS provides authentication, IAM and Marketplace billing | Anthropic is the data processor for inference inputs and outputs; ZDR is opt-in; Anthropic's HIPAA-ready program is not available | Same endpoints and model IDs as the Claude API; new models typically launch the same day, and most new features arrive without a separate integration step; its own capacity pool; the `anthropic-workspace-id` header is required on inference requests; no fast mode |
| Amazon Bedrock | AWS, with zero operator access for Anthropic personnel | AWS is the data processor | Current integration (Claude in Amazon Bedrock, Opus 4.7 and later): no server-side tools (code execution, web search, web fetch, advisor), Files API, Agent Skills, MCP connector, programmatic tool calling or Claude Managed Agents; its page lists structured outputs as unsupported, while the structured-outputs page lists them on Bedrock for Opus 4.6, Sonnet 4.6, Sonnet 4.5, Opus 4.5 and Haiku 4.5, so check per model and integration; no Message Batches, Models, Admin, Compliance or Usage and Cost endpoints; 20 MB request limit; regional endpoints cost 10% more; logs go to CloudWatch and CloudTrail |
| Google Cloud (Agent Platform, formerly Vertex AI) | Google | Google is the data processor | `model` goes in the endpoint URL and `anthropic_version: "vertex-2023-10-16"` in the body; web search is supported, but not code execution, web fetch, advisor, the Files API, Agent Skills, the MCP connector, programmatic tool calling or Claude Managed Agents; no Message Batches, Models, Admin, Compliance or Usage and Cost endpoints; 30 MB request payloads; regional and multi-region endpoints cost 10% more |
| Microsoft Foundry | Anthropic, hosted on Azure or hosted on Anthropic | Anthropic acts as an independent processor for Microsoft; on Azure-hosted deployments, prompts and completions stay in Azure; only usage metadata and content flagged by Anthropic's safety systems egress to Anthropic | `model` is your deployment name; billed in Claude Consumption Units at &#36;0.01 each, metered hourly and invoiced monthly in arrears on the Azure bill; no Anthropic rate-limit headers; no Message Batches, Models, Admin or Compliance API, advisor or Claude Managed Agents; Azure-hosted deployments return 400 for code execution, web search and web fetch versions newer than the basic ones, Agent Skills, programmatic tool calling and the Files API |

Three cross-platform facts shape failover and migration plans:

- **Retirement dates differ.** Bedrock and Google Cloud set their own retirement schedules, so the same model can be in a different lifecycle state on each platform; the lifecycle rules are in [The model lifecycle inside the program lifecycle](#the-model-lifecycle-inside-the-program-lifecycle).
- **Model IDs differ.** Claude Platform on AWS uses the Claude API model IDs unchanged, but Bedrock IDs carry the `anthropic.` prefix, Google Cloud uses `@` for dated models, and on Foundry `model` is your deployment name (by default the Claude API model ID), so a failover path that crosses those platforms needs a model-ID map, not just a second endpoint.
- **Caches do not travel.** Prompt caches are isolated per workspace on the Claude API, Claude Platform on AWS and Foundry, and per organization on Bedrock and Google Cloud. Plan a failover platform's capacity as if its cache starts cold (our inference from the isolation rules).

The same `anthropic` Python package ships the platform clients (`AnthropicBedrockMantle`, `AnthropicBedrock`, `AnthropicVertex`, `AnthropicAWS`, `AnthropicFoundry`). Contract terms follow the operator: Anthropic's privacy center says use through a third-party platform is governed by that platform's terms of service, while the Claude Platform on AWS and Foundry pages say customers there are subject to Anthropic's data use terms. What changes for Claude Code on each provider (usage visibility, and managed settings in place of the claude.ai organization's model and effort controls) is covered in [Choose the provider](#choose-the-provider).

### Choosing the platform

| If the requirement is | Choose | Because |
|---|---|---|
| FedRAMP High, IL4, IL5 or HIPAA-ready compliance on AWS, or AWS as the sole data processor | Amazon Bedrock | The Claude Platform on AWS documentation sends these organizations to Bedrock |
| Fast mode, Message Batches, server tools, Agent Skills or the MCP connector | Claude API; Claude Platform on AWS also offers Message Batches, Agent Skills and code execution, but not fast mode; Microsoft Foundry deployments hosted on Anthropic offer code execution, web search, web fetch, Agent Skills and the MCP connector, but not Message Batches, the advisor tool or fast mode | Bedrock and Google Cloud list these as unsupported (Google Cloud does support web search); Foundry has no Message Batches or advisor tool, and its Azure-hosted deployments return 400 for code execution, newer web tool versions and Agent Skills; fast mode is on the first-party Claude API only |
| Prompts and completions must stay inside Azure | Microsoft Foundry, hosted on Azure | The Foundry docs say prompts and completions remain within Azure, though usage metadata and content flagged by Anthropic's safety systems egress to Anthropic; several features return 400 there, and Claude Fable 5.1, Fable 5, Opus 4.7 to 4.5 and Sonnet 4.6 and 4.5 are hosted on Anthropic only |
| Processing in the EU | Bedrock (EU inference profile or an EU regional endpoint) or Google Cloud (`eu` multi-region or a regional endpoint); Anthropic's regional compliance page also names Microsoft Foundry for country-specific European deployments, but the Foundry docs name only Global Standard and US Data Zone Standard deployment types, so confirm a European option before relying on Foundry | `inference_geo` on the Claude API and Claude Platform on AWS offers only `"us"` and `"global"` |
| An existing cloud commitment or consolidated cloud billing | The cloud you already use | The docs' stated reasons for choosing a cloud platform |
| A second capacity pool for failover | Claude Platform on AWS alongside another platform | "You can run workloads on more than one platform and fail over between them." ([Claude Platform on AWS](https://platform.claude.com/docs/en/build-with-claude/claude-platform-on-aws)) |

Regulated-industry rules (FedRAMP authorizations by product, HIPAA and the BAA, GDPR and the DPA) are in [Governance and risk in delivery](#governance-and-risk-in-delivery).

### Data residency by platform

| Platform | How you pin the location | What it costs and constrains |
|---|---|---|
| Claude API, Claude Platform on AWS | `inference_geo` per request (`"global"` default, or `"us"`); on both platforms a workspace can also set `default_inference_geo` and restrict choices with `allowed_inference_geos` | `"us"` costs 1.1x on every token category; Claude 4.6 and later only, older models return 400; rate limits are shared across all `inference_geo` values. On the Claude API, workspace geo (data at rest) is fixed when the workspace is created and is currently only `"us"`; on Claude Platform on AWS it is not configurable, and the AWS region a workspace is bound to does not pin where inference runs |
| Amazon Bedrock | A regional endpoint, which resolves to the single AWS region you specify, or an inference profile that routes within the US, EU, JP or AU | Regional endpoints cost 10% more than global ones; model coverage varies by region (for Claude Fable 5.1, regional endpoints are currently in `us-east-1` only) |
| Google Cloud | A multi-region endpoint (`us` or `eu`) or a regional endpoint | 10% more than global; provisioned throughput requires a regional endpoint; the docs' regional-endpoint code samples note that specific regional endpoints serve Claude Sonnet 4.6 and earlier, while newer models use the global or multi-region endpoints |
| Microsoft Foundry | `inference_geo` does not apply; choose the deployment type: Global Standard, or, for models hosted on Azure, US Data Zone Standard, which keeps inference in the United States | US Data Zone Standard carries the same 1.1x multiplier as `inference_geo: "us"`; Azure-hosted deployments keep prompts and completions in Azure |

A US-only request on the first-party API, from the [data residency docs](https://platform.claude.com/docs/en/manage-claude/data-residency):

```bash
curl https://api.anthropic.com/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-opus-5-5",
    "max_tokens": 1024,
    "inference_geo": "us",
    "messages": [{
      "role": "user",
      "content": "Summarize the key points of this document."
    }]
  }'
```

On the Claude API and Claude Platform on AWS, a residency requirement can therefore rule out a model before price is considered: requests with `inference_geo` on Claude Haiku 4.5 (and on Opus 4.5, Sonnet 4.5 and earlier models) return a 400 error. The cloud platforms pin location through endpoints or deployment types instead (table above).

### How capacity works on the Claude API

The API applies two kinds of limit: **spend limits**, a maximum monthly cost, and **rate limits**, measured per model class in requests per minute (RPM), input tokens per minute (ITPM) and output tokens per minute (OTPM). Both depend on the usage tier: Start, Build and Scale, plus an Evaluation tier with lower limits that new organizations may start in, and a Custom tier whose limits are arranged with the account team. Organizations move up tiers automatically as they build usage history. The docs are explicit that these numbers are ceilings: "All limits described here represent maximum allowed usage, not guaranteed minimums." ([Rate limits](https://platform.claude.com/docs/en/api/rate-limits)). Four rules drive most sizing:

- **Only uncached input counts toward ITPM** on most models: `input_tokens` and `cache_creation_input_tokens` count, `cache_read_input_tokens` does not. The docs' example: with 2,000,000 ITPM and an 80% cache hit rate, you could process 10,000,000 total input tokens per minute.
- **OTPM counts tokens actually generated.** `max_tokens` does not factor into OTPM, so a generous `max_tokens` costs no capacity; a lower cap frees OTPM only by cutting responses short.
- **Limits are per model.** Different models can each run up to their own limits at once. Some models share a bucket: the Opus 4.x limit covers Opus 4.8, 4.7, 4.6 and 4.5 combined, Sonnet 4.6 and 4.5 share the Sonnet 4.x limit, and Fable 5.1 and Fable 5 share the Fable limit, while Opus 5.5, Opus 5 and Sonnet 5 each have their own.
- **Limits bite over short intervals.** The token bucket refills continuously, and 60 RPM might be enforced as 1 request per second, so a burst can return 429s even when the minute's total fits. Sharp increases in traffic can also trip acceleration limits; ramp up gradually.

### Worked sizing example

An internal assistant sends each request an 8,000-token static prefix (instructions and reference text), 500 tokens of varying input, and gets about 500 tokens of output. The target is a peak of 600 requests per minute on Claude Opus 5.5 at the Start tier, whose limits are 1,000 RPM, 2,000,000 ITPM and 400,000 OTPM. All figures below are our arithmetic from those published limits.

| Limit | Per request | At 600 requests per minute | Most requests per minute this limit allows |
|---|---|---|---|
| RPM (1,000) | 1 | 600 | 1,000 |
| ITPM (2,000,000), prefix not cached | 8,500 input tokens | 5,100,000: over the limit | About 235 |
| ITPM (2,000,000), prefix read from cache | 500 input tokens | 300,000 | 4,000 |
| OTPM (400,000) | 500 output tokens | 300,000 | 800 |

Read it as an architect: without caching, input tokens cap the design at about 235 requests per minute, well short of the target; with the prefix cached, output tokens become the binding limit at 800 requests per minute, leaving 25% rate-limit headroom at the 600 peak (our arithmetic). At the Scale tier (10,000 RPM, 10,000,000 ITPM, 2,000,000 OTPM) the same shape allows 4,000 requests per minute, still bound by output (our arithmetic). Two details keep the cached case honest. Cache writes do count toward ITPM, and for concurrent requests a cache entry only becomes available after the first response begins, so the docs say to wait for the first response before sending the rest when parallel requests need cache hits. The 8,000-token prefix clears the 512-token minimum cacheable length on Opus 5.5; below the minimum, prompts are silently not cached.

Rate limits are not the only Start-tier ceiling. With the prefix cached, each request costs about &#36;0.0136 on Opus 5.5 at list prices (&#36;0.0016 cache read, &#36;0.0020 varying input, &#36;0.0100 output), so a sustained 600 requests a minute spends about &#36;8.16 a minute and would reach the Start tier's &#36;500 monthly spend cap in about an hour (our arithmetic). Size the tier on spend as well as on RPM, ITPM and OTPM. The cost side of the same request shape, at 100,000 requests a month, is worked in [Cost modeling](#cost-modeling).

### Levers when a limit binds

| What binds | Lever | Why it works |
|---|---|---|
| ITPM | Prompt caching of the static prefix | Cache reads do not count toward ITPM on most models |
| OTPM | Shorter actual outputs, or part of the traffic on another model | OTPM counts generated tokens; limits are per model |
| One model's limits | Route suitable traffic to another model | Each model has its own limits (watch shared buckets such as Opus 4.x) |
| Volume nobody waits for | Message Batches | Batches have their own limits, shared across models; at the Start tier 200,000 batch requests can wait in the queue and a batch holds up to 100,000 |
| One workload starving another | Workspace rate limits | Set per workspace (not on the Default Workspace, and not available on Claude Platform on AWS); organization limits always apply |
| Developer traffic | The automatic Claude Code workspace | Claude Code usage is rate-limited separately, and admins can cap its share of the organization's limits (this workspace is not available on Claude Platform on AWS) |
| The tier itself | Move up a tier, or a Custom arrangement | Organizations move up automatically with usage history, and **Request rate limit increase** on the Console's Rate limits page asks for more sooner; above Scale, contact sales, and Custom-tier limits are arranged with the account team |
| A need for guaranteed capacity | Talk to sales | Priority Tier capacity commitments are no longer available for purchase |

Anthropic's workspace docs suggest the same partitioning for environments: development with lower rate limits, staging with production-like limits, and production with full rate limits and monitoring ([Workspaces](https://platform.claude.com/docs/en/manage-claude/workspaces)). An organization can have up to 100 workspaces by default (archived ones do not count, and the account team can raise the limit), and batches are isolated to the workspace that created them.

### Capacity on the cloud platforms

- **Amazon Bedrock:** the default quota is 2 million input tokens per minute, and you can request up to 5 million input TPM and 500,000 output TPM without additional Anthropic approval. AWS enforces requests-per-minute limits on the Bedrock side; RPM adjustments go through AWS support.
- **Google Cloud:** global and multi-region endpoints are pay-as-you-go only; provisioned throughput requires a regional endpoint, which carries the 10% premium. The docs' regional-endpoint samples note that specific regional endpoints serve Claude Sonnet 4.6 and earlier, so confirm a newer model is offered regionally before planning provisioned capacity for it.
- **Microsoft Foundry:** responses do not include Anthropic's rate-limit headers, so the docs say to manage rate limiting through Azure's monitoring tools and to request increases through the Azure portal or Azure support. Several deployments of the same model under different names can hold separate configurations or rate limits.
- **Claude Platform on AWS:** the first-party rate limits apply, but organizations start on the Start tier and move up as they build a history of paid AWS Marketplace invoices; higher limits come through the Anthropic account representative or support, and per-workspace rate limits and fast mode are not available. Its capacity pool is separate from both the first-party API and Bedrock, which is what makes it a failover target.

### Priority Tier and the `service_tier` parameter

Three service tiers exist: Priority, Standard and Batch. Standard is the default and is served "with best-effort availability". Priority Tier targets 99.5% uptime and falls back to standard beyond the committed capacity, but "Priority Tier capacity commitments are no longer available for purchase" ([Service tiers](https://platform.claude.com/docs/en/api/service-tiers)); it is available only to organizations with an existing commitment, each defined by input TPM, output TPM, a 1, 3, 6 or 12 month duration and a specific model version, and those commitments run to the end of their contracts. Priority requests also draw on the regular rate limits (a request that would exceed them is declined), and Priority Tier does not support Claude Fable 5.1, Mythos 5.1, Mythos 5, Mythos Preview, Opus 5.5, Opus 5 or Sonnet 5. A request chooses with `service_tier`: `"auto"` (the default) uses Priority capacity when available, `"standard_only"` never does, and the response reports which tier served it:

```python
import anthropic

client = anthropic.Anthropic()

message = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello, Claude!"}],
    service_tier="auto",  # Automatically use Priority Tier when available, fallback to standard
)
print(message.usage.service_tier)
```

Server-side fallback is not a capacity tool either. Only a safety-classifier decline triggers it: "A rate limit, overload, or server error on the requested model is returned to you as-is." ([Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback)). The `fallbacks` parameter is also unavailable on Bedrock, Google Cloud and Foundry, which point to a client-side fallback pattern instead. What the business can be promised, and what each dependency commits to, is in [Stakeholder communication and lifecycle](#stakeholder-communication-and-lifecycle).

!!! note "Names and tiers that changed before September 2026"

    Study material written earlier can be out of date in three places. Usage tiers were consolidated into Start, Build and Scale on June 26, 2026; older release notes refer to "usage tier 1-4 rate limits" ([release notes](https://platform.claude.com/docs/en/release-notes/overview)). Older material may present Priority Tier as a way to buy guaranteed capacity; the current service-tiers page says commitments are no longer sold. And the prep course's "Vertex" is the product the current Anthropic docs call "Google Cloud's Agent Platform, formerly Vertex AI" ([Claude Code third-party integrations](https://code.claude.com/docs/en/third-party-integrations)). None of the four July 2026 exam guides mentions usage tiers, Priority Tier or Vertex by name. Their items are written against objectives worded generically, such as "performance SLAs" and "invoking Claude through third-party vendors", so answer in those terms.

### How the exams frame it

- **CCAR-P** asks you to "Evaluate accuracy-latency trade-offs and justify configuration decisions" and to align solutions to value pillars that include "performance SLAs" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). In our reading, a justified configuration names the platform, the binding limit and the lever, as in the worked example.
- **CCDV-F** lists "invoking Claude through third-party vendors" under Claude API Mechanics; in our reading that means knowing which platform lacks which feature and how the call changes (model in the URL on Google Cloud, `anthropic.` IDs on Bedrock, deployment names on Foundry).
- **CCAR-F** lists rate limiting, quotas, API pricing calculations and specific cloud provider configurations as out of scope.

### Decide

- If a compliance requirement points to a platform (the Claude Platform on AWS docs send FedRAMP High, IL4, IL5 and HIPAA-ready requirements on AWS to Bedrock), settle the platform first; not the one with the most features.
- If the design depends on fast mode, Message Batches, server tools, Agent Skills or the MCP connector, choose the Claude API, Claude Platform on AWS for everything but fast mode after checking its not-supported list, or a Foundry deployment hosted on Anthropic for server tools other than the advisor, Agent Skills and the MCP connector; not Bedrock or Google Cloud, which list them as unsupported (web search aside on Google Cloud).
- If input tokens bind capacity, cache the static prefix; if output tokens bind, shorten real outputs or split traffic across models; not a lower `max_tokens`, which saves OTPM only by truncating answers (the cap itself is not counted toward OTPM).
- If non-urgent volume competes with interactive traffic, move it to Message Batches, which has its own limits; not a bigger synchronous fan-out.
- If one workload must not starve another, give each its own workspace with workspace rate limits (not available on Claude Platform on AWS or the Default Workspace); not one shared workspace.
- If stakeholders need guaranteed capacity on the newest models, go to sales; not Priority Tier, which is no longer sold and does not cover them.
- If you need resilience beyond one provider, run a second platform with its own capacity pool and a model-ID map; not server-side fallback, which returns rate limits and overloads as-is.

### Traps

- **Treating published rate limits as guaranteed capacity.** They are maximums, not minimums, and a 529 `overloaded_error` (the API is temporarily overloaded) can occur when the API experiences high traffic across all users.
- **Sizing on the minute average.** Limits may be enforced per second, and sharp ramps can trip acceleration limits.
- **Expecting `inference_geo: "us"` to work on Claude Haiku 4.5.** On the Claude API and Claude Platform on AWS the parameter needs Claude 4.6 or later; US-only Haiku 4.5 needs another platform: a Bedrock regional endpoint or US inference profile, a Google Cloud `us` multi-region or US regional endpoint, or a Foundry US Data Zone Standard deployment.
- **Assuming `inference_geo` gives a separate rate limit.** Limits are shared across all `inference_geo` values.
- **Planning a failover on the same model ID string.** Claude Platform on AWS takes Claude API model IDs unchanged, but Bedrock (`anthropic.` prefix), Google Cloud's dated models (`@`) and custom-named Foundry deployments need a mapped ID, and Bedrock and Google Cloud set their own retirement dates.
- **Forgetting the platform premiums.** Regional endpoints on Bedrock and regional or multi-region endpoints on Google Cloud cost 10% more; US-only inference through `inference_geo: "us"`, and Foundry's US Data Zone Standard, cost 1.1x.

## Cost modeling

*Tested in: CCAR-P 1.6, 2.5 (caching as a prompt reuse strategy), 3.3, 4.5, 6.2 and Sample 2 · CCDV-F 5.4 Cost and Token Management, 5.3 Model Selection and Tradeoffs · CCAO-F D3.3 and Sample 2 (cost, speed and quality when choosing a model) · CCAR-F 4.5-K1, 4.5-S1, 4.5-S4 and Question 11 (the 50% batch saving and where it does not apply); API pricing calculations are out of scope (APPX-OUTSCOPE-11)*

A cost model is a design input, not a finance afterthought. The CCAR-P prep course ties the move to production to a budget for cost, latency and reliability (its wording is quoted at the top of [Integration patterns](#integration-patterns)), and its integration module starts a use case by estimating call volume, token consumption and cost. Per-token prices, the usage fields and billing reports are taught on the API page under [Cost and usage tracking](claude-api.md#cost-and-usage-tracking); this section is about the program: what goes into the model, how to price a workload you have not built yet, and how to defend the number.

### What goes into the model

The Anthropic and Accenture pilot-to-production guide says to build "a lightweight total cost of ownership model before the pilot begins": API costs, integration overhead, maintenance burden, and the opportunity cost of not deploying. It then says to "revisit it quarterly as AI economics evolve", because an architecture sized on earlier pricing can end up over-engineered ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf)).

| Cost line | What drives it | Where the number comes from |
|---|---|---|
| Model tokens | Requests per month, input and output tokens per request, cache hit share, model | The worked example below; the pricing page |
| Server tools | Web search at &#36;10 per 1,000 searches plus tokens; web fetch has no extra charge beyond tokens; code execution has 1,550 free hours per organization per month, then &#36;0.05 per hour per container, and is free when the request includes `web_search_20260209` or `web_fetch_20260209` (or later) | Pricing page |
| Managed runtime | Claude Managed Agents: tokens plus &#36;0.08 per session-hour of `running` time; sessions have no batch mode | Pricing page |
| Retrieval | Embedding tokens (Anthropic offers no embedding model; Voyage AI's free allowance and per-million-token prices are in [Embeddings](#embeddings)) | Voyage pricing page |
| Integration and maintenance | Engineering to connect systems, keep prompts, evals and indexes current | Your own estimate; the guide lists it as a TCO line |
| Opportunity cost | What the organization loses by not deploying | Your own estimate |

Two framing rules decide where a program starts. Favor volume: "A process that runs hundreds of times daily is a much better candidate, as the economics scale differently" than one that runs quarterly ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf)). And the price criterion in Anthropic's success-criteria guidance is not a single number: it covers "the cost for each API call, the size of the model, and the frequency of usage" ([Define success criteria](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests)).

### The unit is a completed task

Anthropic's cost guide is blunt: "You pay for completed tasks, though, so compare models on cost per completed task." Its second rule: "Price the tail of your workload, not the median: compare models on the hardest tenth of your tasks, not the typical one." The reason is that a failed task still bills its tokens, then the retry, then whatever the failure costs downstream; on one 20-problem WideSearch run, two problems carried 43% of the spend ([Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence)).

Anthropic's own customer-support benchmark shows why. Starting from Claude Opus 4.8 at its default (high) effort, Claude Code's `/claude-api hillclimb` search first moved to Claude Opus 5 at low effort with a prompt audit (98.9% train accuracy, 2.6 cents per ticket), then to Claude Sonnet 5 at low effort: "cheaper still at 1 cent per ticket, but accuracy fell to 88.9%". Adding routing rules and a refund-cap cross-reference to the prompt brought Sonnet 5 back to 98.9% at the same cost, and on 14 held-out tickets the final configuration scored 90.5% against the original 78.6% "at about one fifth the cost" ([Reducing cost and improving performance](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform)). The cheap configuration only became the right answer once its failure rate was fixed.

Token counts are per model, too. Claude 4.7 and later models use a tokenizer that produces approximately 30% more tokens for the same text, while Claude Sonnet 4.6 and earlier models, Claude Haiku 4.5 among them, use the previous tokenizer. Recount a prompt on each candidate model before comparing prices.

### Worked example: an internal policy assistant

The workload below is an assumption chosen to be easy to follow; the prices are the Claude API list prices as of September 2026. All arithmetic is ours.

- 100,000 requests a month.
- Every request sends the same 8,000-token system prompt and policy document, then 500 tokens of varying user input, and receives 500 output tokens.
- The static part is cached with the 5-minute TTL. Traffic is steady, so assume 95% of requests read the prefix from cache and 5% write it.
- Token counts are held equal across models to isolate price; in practice, recount per model (see above).

| Price per million tokens | Input | 5-minute cache write | Cache read | Output |
|---|---|---|---|---|
| Claude Opus 5.5 | &#36;4 | &#36;5 | &#36;0.20 | &#36;20 |
| Claude Sonnet 5 | &#36;2 | &#36;2.50 | &#36;0.20 | &#36;10 |
| Claude Haiku 4.5 | &#36;1 | &#36;1.25 | &#36;0.10 | &#36;5 |

Per request:

| Line | Tokens | Opus 5.5 | Sonnet 5 | Haiku 4.5 |
|---|---|---|---|---|
| Static prefix, uncached | 8,000 | &#36;0.0320 | &#36;0.0160 | &#36;0.0080 |
| Static prefix, cache write | 8,000 | &#36;0.0400 | &#36;0.0200 | &#36;0.0100 |
| Static prefix, cache read | 8,000 | &#36;0.0016 | &#36;0.0016 | &#36;0.0008 |
| Varying input | 500 | &#36;0.0020 | &#36;0.0010 | &#36;0.0005 |
| Output | 500 | &#36;0.0100 | &#36;0.0050 | &#36;0.0025 |

Per month:

| Option | Opus 5.5 | Sonnet 5 | Haiku 4.5 |
|---|---|---|---|
| No caching: 100,000 requests at the uncached rate | &#36;4,400.00 | &#36;2,200.00 | &#36;1,100.00 |
| Caching: 95,000 requests that read the prefix (plus varying input and output) | &#36;1,292.00 | &#36;722.00 | &#36;361.00 |
| Caching: 5,000 requests that write the prefix (plus varying input and output) | &#36;260.00 | &#36;130.00 | &#36;65.00 |
| **Caching total** | **&#36;1,552.00** | **&#36;852.00** | **&#36;426.00** |
| Saving from caching | 64.7% | 61.3% | 61.3% |

What the numbers teach:

- **Caching beats downsizing.** Caching Opus 5.5 (&#36;1,552) costs less than running Sonnet 5 uncached (&#36;2,200). A 5-minute write costs 1.25x input and, in the pricing page's words, caching "pays off after one cache read" ([Pricing](https://platform.claude.com/docs/en/about-claude/pricing)). The 8,000-token prefix is above the minimum cacheable length on all three models (512 tokens on Opus 5.5, 1,024 on Sonnet 5, 4,096 on Haiku 4.5); shorter prompts are processed without caching and no error is returned.
- **Cache reads flatten the model gap.** A cache read costs 5% of input on Opus 5.5, so Opus 5.5 and Sonnet 5 both pay &#36;0.20 per million cached tokens. Once the prefix is cached, the difference between them comes from output and uncached input.
- **Residency is a multiplier, and a constraint.** US-only inference (`inference_geo: "us"`) costs 1.1x on every token category on Claude 4.6 and later: the Sonnet 5 caching total becomes &#36;937.20. On Claude Haiku 4.5 the parameter returns a 400 error, so a US-only requirement met through `inference_geo` rules Haiku 4.5 out before price is considered.
- **Check the tier's spend cap.** Monthly caps are &#36;500 (Start), &#36;1,000 (Build) and &#36;200,000 (Scale); the Custom tier has none. Only Haiku 4.5 with caching fits under Start; Sonnet 5 with caching fits under Build; Opus 5.5 needs Scale (our arithmetic). At the cap, usage pauses until 00:00 UTC on the first day of the next month unless a higher limit is granted sooner, and requests return a 429 whose `error.details.error_code` is `enforced_spend_limit_reached`, with no `retry-after` header.
- **Capacity improves with caching too.** Cache reads do not count toward input tokens per minute on most models; the rate-limit arithmetic for this same request shape is in [Worked sizing example](#worked-sizing-example).

If part of the traffic has no one waiting (a nightly re-check of every policy answer, say), send it through the Message Batches API: 50% off every token, cached ones included. Requests still unprocessed at 24 hours expire unbilled (more of them when demand is high) and need resubmitting, so budget for resubmissions. Caching and batch discounts stack, but batch cache hits are best effort, typically 30% to 98%. The realtime-versus-batch decision itself (the 24-hour window with no latency SLA, and the CCDV-F sample question on it) is in [Invocation modes](#invocation-modes).

### Levers, in the order to test them

Anthropic's cost guide splits levers into free wins that do not touch quality (caching, token hygiene, a prompt audit, batching, workspace spend limits) and trade-offs (model choice, effort, output caps and task budgets, multi-model architectures). Its figures are "Anthropic-internal" and "directional, not guarantees" ([Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence)), but the order is the useful part. The sequence below is ours, built from the guide's two groups and its advice to sweep effort before pricing a stronger model or combining models:

1. **Cache the repeated context.** In Anthropic's measurements caching cut agent-loop cost by a factor of 2.7 to 5.3. Over a full day of real traffic, agent loops read a median 84% of their input from the cache and the top 10% read 94% or more; below about 80%, look for something breaking the cache.
2. **Batch everything nobody waits for.** The cost guide's routing rule, and where batch does not fit, are in [Invocation modes](#invocation-modes).
3. **Sweep effort on the current model.** Anthropic's cost guide calls it "the cheapest experiment" it describes and adds that "most workloads end there" ([Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence)). Returns diminish at the top: on Claude Fable 5.1, the last step up to max effort on Humanity's Last Exam added about half a point for 46% more cost ([Reducing cost and improving performance](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform)).
4. **Price a stronger model at low effort.** "A stronger model at low effort can be cheaper than a weaker model working hard (high effort)." ([Reducing cost and improving performance](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform))
5. **Only then combine models.** An executor that escalates hard decisions to an advisor (the beta advisor tool, available on the Claude API and Claude Platform on AWS, runs the whole strategy "server-side in one `/v1/messages` request"), or an orchestrator that delegates bulk work to cheaper workers. The advisor's payoff depends on the consult rate, which is fragile; the orchestrator "saved money in only two measured situations": a long cost tail on routine work, and work larger than one context window ([Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence)).

Before cutover, run the winner in shadow on a traffic slice and keep the evaluation suite running; the method is in [Evaluation strategy for a program](#evaluation-strategy-for-a-program).

### Seats, subscriptions and chargeback

Apps and Claude Code rollouts are priced differently from API workloads:

- **Claude Enterprise (usage-based plans).** Seats cover access and usage is billed at API rates (quoted, with the seat-based alternative, in [Spend in the rollout plan](#spend-in-the-rollout-plan)). The Academy course describes a typical agreement as a per-member platform fee plus pooled consumption measured in tokens.
- **Claude Code.** Across enterprise deployments, the Claude Code docs put the average at about &#36;13 per developer per active day and &#36;150 to &#36;250 per developer per month, with costs below &#36;30 per active day for 90% of users; budget and rollout are covered in [Rolling out Claude Code to an engineering organization](#rolling-out-claude-code-to-an-engineering-organization).
- **Per-use value.** Anthropic's consumption guide prices individual skills against the value they create: "a call-prep skill costing &#36;0.90 per run against &#36;20 of value returns 20x on every use." ([Claude Enterprise consumption guide](https://support.claude.com/en/articles/14782391-claude-enterprise-consumption-guide))
- **Chargeback.** The pilot-to-production guide cites Accenture research that organizations with formal chargeback accountability "link 32 cents of every dollar of AI token spend to a quantified business outcome, six times more than those with no allocation." ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf))

### How the exams frame it

- **CCAR-P Sample 2** is the model question: an 8,000-token system prompt and policy document repeat on every request, and both latency and cost matter. The keyed answer puts the static content first and enables prompt caching, because that "lets repeated prefixes be reused, reducing both time-to-first-token and per-request cost without discarding required context." The rationale rejects truncation (it "loses needed policy"), downsizing "blindly" (it "risks quality"), and moving the policy into few-shot examples, which "does not create a cacheable, reusable prefix" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)).
- **CCDV-F 5.4** covers "token usage tracking, cost modeling, and caching techniques (prompt caching, cache check-pointing)" ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)); 5.3 adds the quality, latency and cost trade-offs between Opus, Sonnet and Haiku. On "cache check-pointing", see the box below.
- **CCAO-F D3.3** asks you to "Align model selection with task requirements (cost, speed, quality)". Its Sample 2 (a high volume of short customer-reply drafts) keys "Use a faster, lower-cost model suited to straightforward, high-volume tasks", and the rationale adds that "Always using the top model (A) wastes the cost and latency budget" ([CCAO-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)).
- **CCAR-F** lists the batch discount (4.5-K1) but excludes "Rate limiting, quotas, or API pricing calculations" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Its Question 11 tests where the saving stops: a manager wants to move both a blocking pre-merge check and an overnight technical-debt report to the Message Batches API. The keyed answer batches the technical-debt reports only and keeps real-time calls for pre-merge checks, because the API "has processing times up to 24 hours with no guaranteed latency SLA", which "makes it unsuitable for blocking pre-merge checks where developers wait for results, but ideal for overnight batch jobs like technical debt reports." The rationale also rejects a timeout fallback to real-time as unnecessary complexity. Task statement 4.5 adds a cost habit: refine the prompt on a sample set before batch-processing large volumes, to maximize first-pass success and reduce resubmission costs.

!!! warning "Exam guide vs current docs"

    The CCAR-P guide names no model IDs, token prices or API field names; the closest it comes to the API is prompt caching (objective 2.5 and Sample 2) and a temperature distractor (Sample 3). Prices on this page are as of September 2026, after the July 2026 guides: Claude Opus 5.5 launched on September 22, 2026, and Claude Sonnet 5's &#36;2 / &#36;10 price is now standard because "The previously scheduled increase to &#36;3/&#36;15 per million input/output tokens on September 1, 2026 will not occur." ([Pricing](https://platform.claude.com/docs/en/about-claude/pricing)). Learn the method (cache the static prefix, batch what can wait, compare cost per completed task); expect price points to move.

    One wording differs outright. The CCDV-F guide's Cost and Token Management skill (5.4 on this site) says "cache check-pointing"; the prompt caching documentation does not use that term and instead speaks of cache breakpoints (explicit, or automatic through a top-level `cache_control`, with 4 breakpoint slots). Do not confuse it with Claude Code's checkpointing, which rewinds file edits and conversation. Expect the guide's wording on the exam; the guide does not define it, and because it sits in a list of caching techniques, our reading is prompt cache breakpoints.

### Decide

- If a large prefix repeats on every request, put it first and cache it before considering a smaller model; not truncation, because the Sample 2 rationale says truncation loses needed policy.
- If nobody waits for a result, price it at batch rates, because batch halves every token; if someone is blocked waiting, price it at synchronous rates, not batch (the rule is in [Invocation modes](#invocation-modes)).
- If two models are compared, compare cost per completed task on the hardest tenth of your tasks, with each model's own token counts; not list price per token.
- If you are unsure which lever to pull, sweep effort on the current model first; if the sweep shows a gap, price the stronger model alone at low effort; only then consider a multi-model design; not a multi-model design first.
- If the monthly estimate is near the tier's spend cap, arrange the tier increase before launch (a spend limit you set yourself can only sit below the cap); not after the first 429.
- If a data-residency or compliance requirement applies, rule out the models and platforms it excludes before comparing prices.

### Traps

- **API cost as the whole TCO.** Integration, maintenance and opportunity cost are lines in the model too.
- **Pricing the median task.** The tail is where failures and spend concentrate.
- **Assuming cache reads cost 10% of input everywhere.** On Opus 5.5 they cost 5%, and on Fable 5.1 and Mythos 5.1 they cost 2.5%.
- **Forgetting the multipliers.** US-only inference and regional cloud endpoints carry premiums; the rates by platform are in [Data residency by platform](#data-residency-by-platform).
- **A cheap configuration that fails more often.** A failed task still pays for its tokens and its retry.
- **Lowering `max_tokens` to save money.** The cap is invisible to the model, so it does not make the model economize; in Anthropic's measurements it cut cost per attempt without lowering cost per solved task, and the CCDV-F Sample 1 rationale lists it among the answers that miss the batch-versus-realtime trade-off.
- **Counting on the typical batch turnaround.** Most batches finish in under an hour, but the CCAR-F guide describes a processing window of up to 24 hours with no guaranteed latency SLA, and its Question 11 rationale rejects relying on "often faster" completion for blocking workflows.

## Evaluation strategy for a program

*Tested in: CCAR-P 4.1, 4.2, 4.3, 4.4, 4.6 and Sample 3 · CCDV-F 4.1 Debugging and Error Handling, 2.6 Configuration Management (model version pinning, prompt versioning), 5.3 Model Selection and Tradeoffs (breaking behavior changes across releases) · CCAR-F 5.5-K1, 5.5-S2 · CCAO-F D2.1, D7.2 (at the level of checking outputs and adjusting)*

The CCAR-P prep course states the program-level rule: "Build evaluations as acceptance criteria and use them as the gate before any model or architecture change" ([CCAR-P prep course](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional)). The mechanics of a single eval (criteria, test sets, graders, confidence intervals, `pass^k`) are on the evaluation page: [Success criteria and test sets](evaluation-and-reliability.md#success-criteria-and-test-sets), [Grading methods](evaluation-and-reliability.md#grading-methods) and [Evaluating agents](evaluation-and-reliability.md#evaluating-agents). The five metric families CCAR-P 4.1 names ("accuracy, latency, cost, safety, security") are turned into concrete measures on that page too. This section is about running evaluation across a program: agreeing what success means, deciding what the suite gates, moving changes into production safely, and feeding production back into the suite.

### Agree the criteria before the pilot

Evaluation starts from the use case definition. The pilot-to-production guide asks that it include "a defined user, task, output, and a measurable quality threshold" ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf)). The same guide shows why that is harder than it sounds: in its contract-review example, each function wants to track something different.

| Stakeholder | What they want to track in the guide's example |
|---|---|
| Engineering | Whether the system flags non-standard clauses accurately enough that attorney review time drops below manual review |
| Finance | Whether the cost per reviewed contract, counting attorney time on flagged items, is lower than the fully manual process |
| Legal and risk | Whether every output carries a complete audit trail that meets documentation requirements |

Criteria that do not converge before the pilot, the guide warns, "are likely to generate competing narratives about whether the program is working." ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf)) Three more requirements from the same source belong in the plan: a human performance baseline (one of the outputs in [What discovery must produce](#what-discovery-must-produce)), leading and lagging indicators with the lag between them (the guide's example: shorter PR cycle time can be a leading indicator of Claude Code value, while revenue or churn is the lagging one), and the cost of each error type at full production volume.

### Layers, cadence and owners

CCAR-P 4.2 asks for "mixed methodologies", and Anthropic's evals post lists them: automated evals are one method, and "A complete picture includes production monitoring, user feedback, A/B testing, manual transcript review, and systematic human evaluation." No single layer catches every failure; the post compares evaluation to the Swiss Cheese Model and recommends "automated evals for fast iteration, production monitoring for ground truth, and periodic human review for calibration" ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)). At program level, each layer needs a trigger and an owner. The table assembles them from the guidance cited on this page (the arrangement and the Decides column are ours):

| Layer | Runs when | Owner | Decides |
|---|---|---|---|
| Offline suite (capability and regression evals) | Pre-launch and in CI "on each agent change and model upgrade"; for Claude Code, on a schedule and on any change to CLAUDE.md, skills or hooks | A platform engineer assembles 20 to 50 real tasks, each with its expected or accepted outcome; the team that owned an incident writes its eval | Whether a change may ship (configuration changes are gated on the results) |
| Shadow or parallel run | Before cutover, on a traffic slice; or alongside the existing workflow until performance and reliability are proven | Whoever holds decision rights in the go/no-go process documented before the pilot | Cutover |
| A/B test | For significant changes, once traffic is sufficient | The cited guidance names none; assign one | Whether real user outcomes moved |
| Production monitoring | Continuously after launch, to detect distribution drift and unanticipated real-world failures; automated alerts at degradation thresholds, with escalation protocols | A named owner for "continuous monitoring and degradation alerting" | Escalation; rollback (which the SDLC playbook says should be the most rehearsed path) |
| Human review | Feedback triaged constantly; sampled transcripts read weekly; systematic human studies reserved for calibrating LLM graders and subjective outputs | The cited guidance names none; assign one | Grader calibration; new eval cases |

Two consequences follow. First, the suite also gives you operating baselines "for free": "latency, token usage, cost per task, and error rates can be tracked on a static bank of tasks" ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)). Second, the loop closes in both directions: every production incident becomes a regression case, and offline evaluations are updated from production data rather than treated as static. The [SDLC playbook](https://academy.claude.com/courses/ai-native-sdlc-playbook/continuous-evals-in-ci) calls it a live suite: as models improve, cases that once discriminated stop doing so, and new cases come from ongoing monitoring. It also notes that, depending on the use case, some teams may prefer to run the evals offline on a set cadence rather than on every change.

### The gate: what forces a re-run

- **Any change to the agent or its model.** Model IDs are pinned: "Anthropic does not update the weights or configuration of an existing model ID. When an updated version is available, it ships under a new model ID." The guarantee covers model IDs, not the convenience aliases the Claude API accepts for models before the 4.6 generation (an alias such as `claude-sonnet-4-5` resolves to the most recent dated snapshot for that minor version). Call a pinned ID and a model change becomes a deliberate configuration change that the suite can gate ([Model IDs and versions](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions)).
- **Prompt changes.** The [SDLC playbook](https://academy.claude.com/courses/ai-native-sdlc-playbook/continuous-evals-in-ci) names a rewritten prompt as a trigger: "When a new model is swapped in or a prompt is rewritten, the eval suite says whether the agent still does the work to the same standard." Anthropic's older enterprise guidance makes the same point about prompts as assets: "Like code, prompts need version control, testing, and proper documentation." ([Building trusted AI in the enterprise](https://www-cdn.anthropic.com/e5c9de22bc8884089970bd262ca0c8b952cb9136.pdf))
- **Configuration that steers the agent.** Anthropic's AI-native SDLC playbook runs the suite on any change to CLAUDE.md, skills or hooks, "since that configuration steers the agent and deserves the regression testing that code gets." ([AI-native SDLC playbook](https://academy.claude.com/courses/ai-native-sdlc-playbook/continuous-evals-in-ci))
- **New Skill versions.** "Require the full evaluation suite to pass before promoting new versions", and pin production requests to a Skill version ([Skills for enterprise](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise)).
- **Deprecation notices.** Anthropic advises testing newer models "well before the retirement date of your current model." ([Model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations)); notice periods and lifecycle states are in [The model lifecycle inside the program lifecycle](#the-model-lifecycle-inside-the-program-lifecycle).

A migration in Claude Code can start from the bundled Claude API skill (`/claude-api migrate`), which applies the model ID swap and, as needed, breaking parameter changes, then "produces a checklist of items to verify manually" ([Opus 5.5 migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide)). The migration guide's own checklist for moving from Claude Opus 5 ends with the item "Re-baseline cost and latency at your chosen effort level"; its recommended changes include testing "in a development environment before switching production traffic."

```text
/claude-api migrate this project to claude-opus-5-5
```

!!! warning "Exam guide vs current docs"

    The July 2026 guides predate Claude Opus 5.5, which launched on September 22, 2026. Its migration guide (as of September 2026) shows why the gate matters even for an upgrade within one model family: "Effort is the only thinking control on Claude Opus 5.5, and its default is `medium` where Claude Opus 5's is `high`", so a request that omits `effort` runs at a different effort after the switch ([Opus 5.5 migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide)). The guides frame this generically: CCAR-P 4.3 is "Conduct A/B testing and iterative improvements", and CCDV-F 5.3 covers "breaking behavior changes across model releases". Answer in the guide's terms (gate the change on the suite, re-baseline, then roll out), not by recalling this particular default.

### Iterating against a held-out set

Anthropic's cost guide gives a four-step measurement method ([Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence)):

1. Pull a few tasks from production logs, weighted like real traffic, write outcome checks for each (tests pass, ticket closed, row count correct), and record cost per task beside the score.
2. Baseline the model tiers across effort levels, not only the default, and plot score against spend. A multi-model configuration must beat the single model's whole curve.
3. If the curve shows a gap effort cannot close, add the multi-model strategy that fits and re-run the suite.
4. "Run the winner in shadow on a traffic slice before cutover, then keep the suite running."

The Claude API skill in Claude Code automates parts of this loop ([Reducing cost and improving performance](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform)):

| Command | What it does |
|---|---|
| `/claude-api prompt-audit` | Scans the prompts, skills and tool descriptions in the working directory (application code that calls the Claude API, or Claude Code's own CLAUDE.md and skills) and removes common anti-patterns. In Anthropic's test on a customer support benchmark (six legacy prompts, each seeded with one anti-pattern, results averaged), the Opus 4.8 to Opus 5.5 model change alone cut cost by around 18%; running the audit once per prompt cut it by a further 9% and raised accuracy by around 2 percentage points |
| `/claude-api cost-optimize` | Profiles where spend goes, then ranks and applies savings (prompt caching, trimming what each request carries, bounding output, batching unattended work); given an evaluation, shows how savings trade off with performance across effort levels and models |
| `/claude-api hillclimb` | An iterative search over cost and performance: splits your evaluation into train and test sets, proposes changes that aim to cut cost while keeping baseline performance, reads failing train cases, and scores the final configuration on the held-out test set |

The held-out split is the point. In the post's run, the final configuration was scored on "the 14 held-out tickets the search never saw"; a score on the train cases Claude read while tuning would flatter the result (our reading). The worked result is in [Cost modeling](#cost-modeling). One reading rule from [the same post](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform), which applies on a non-saturated evaluation: "a flat performance-cost curve across effort levels suggests that the task is not bound by thinking compute; increasing effort is not beneficial."

### A/B testing a live system

An A/B test compares variants on real user traffic. It measures actual user outcomes ("retention, task completion") and controls for confounds, and it is also slow: "days or weeks to reach significance", it needs enough traffic, it tests only changes you deploy, and it gives less signal on why a metric moved unless you also review transcripts ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)). Run it after offline evals pass, for significant changes. StubHub, in the pilot-to-production guide, tested several models against resolution-rate and satisfaction benchmarks before committing to production.

The prep course's integration module asks you to "Plan and interpret an A/B test or structured experiment on a live Claude system, setting the hypothesis, selecting metrics, estimating the required sample size, and reading a result without overclaiming" ([Enterprise Integration & Production](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/enterprise-integration-production)). A one-page plan built from those requirements (the layout is ours):

```text
Hypothesis:  variant B changes <primary metric> by at least <effect> versus baseline A
Metrics:     primary outcome; guardrails (latency, cost per task, safety flags)
Sample size: from a power analysis for that effect
Exposure:    progressive ramp on a traffic slice; rehearsed rollback path
Read-out:    mean difference with its confidence interval; no claim beyond the tested traffic
```

- **Baseline.** Anthropic's success-criteria page defines A/B testing as a comparison "against a baseline model or earlier version." ([Define success criteria and build evaluations](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests))
- **Statistics.** A 95% confidence interval is the mean plus or minus 1.96 × SEM; paired comparisons and power analysis are covered in [Success criteria and test sets](evaluation-and-reliability.md#success-criteria-and-test-sets) on the evaluation page. On small evals, small real differences "will likely go undetected" ([A statistical approach to model evaluations](https://www.anthropic.com/research/statistical-approach-to-model-evals)).
- **Rollout mechanics.** Anthropic's older enterprise guidance says to roll out progressively and "Set up infrastructure for A/B testing" ([Building trusted AI in the enterprise](https://www-cdn.anthropic.com/e5c9de22bc8884089970bd262ca0c8b952cb9136.pdf)); for stateful agents, Anthropic's research system uses rainbow deployments, "gradually shifting traffic from old to new versions while keeping both running simultaneously." ([How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system))
- **One run is not evidence.** The same guidance lists "Make a decision based on a single evaluation test" among the things not to do ([Building trusted AI in the enterprise](https://www-cdn.anthropic.com/e5c9de22bc8884089970bd262ca0c8b952cb9136.pdf)).

### What production monitoring watches

CCAR-P 4.6 asks you to "Monitor system performance using logging and observability tools". Anthropic's older enterprise guidance defines the scope: track "not just basic metrics like response times and error rates, but also LLM-specific concerns like token usage and output quality" ([Building trusted AI in the enterprise](https://www-cdn.anthropic.com/e5c9de22bc8884089970bd262ca0c8b952cb9136.pdf)). The signals and where each comes from, as of September 2026:

| Signal | Where it comes from |
|---|---|
| Latency, error rate, tokens and cost per task | The eval bank sets the baseline; each response's `usage` object gives the live token counts, and for a Console organization the Usage & Cost Admin API gives "programmatic and granular access to historical API usage and cost data" |
| Share of input read from the prompt cache | Anthropic's production benchmark for agent loops, given in [Levers, in the order to test them](#levers-in-the-order-to-test-them) |
| Refusals | `stop_reason: "refusal"` on newer models; handle it and configure fallback |
| Claude Code usage and cost per developer | OpenTelemetry export, which admins can push to everyone through managed settings; the Claude Code costs page calls it "the only option that streams per-user token and cost metrics into your own observability stack in near real time" |
| What users actually sent and received on Claude Enterprise surfaces (claude.ai, Cowork, and Claude Code through the CLI and Claude Desktop; as of September 2026, cloud sessions in Claude Code are not covered) | The Compliance API: "the Compliance API, not telemetry, is the record to rely on for content" (for a Claude Console organization it returns the Activity Feed only) |
| Output quality | Sampled transcripts read weekly, and quality evals run on production traffic |

Telemetry setup for agents is in [Reliability engineering for Claude applications](evaluation-and-reliability.md#reliability-engineering-for-claude-applications); the organization-wide Claude Code configuration is in [Rolling out Claude Code to an engineering organization](#rolling-out-claude-code-to-an-engineering-organization).

### When production disagrees with the suite

| Symptom | First place to look | Source |
|---|---|---|
| Confident but wrong answers after a document refresh; latency and model version unchanged | Retrieval and indexing: "a broken re-index or mismatched embeddings" | [CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf), Sample 3 |
| Small behavior change on a model ID that has not changed | Serving infrastructure (request router, safety classifiers, sampling logic); "an infrastructure update is the most likely cause" | [Model IDs and versions](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions) |
| Quality slides after a model update, or as prompts drift from their original intent | Drift; compare against the regression suite | [Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf) |
| Suite passes, users report degradation | The suite's sensitivity. Anthropic's postmortem of three infrastructure bugs that intermittently degraded Claude's response quality (August to early September 2025) says "we relied too heavily on noisy evaluations" | [A postmortem of three recent issues](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues) |
| High aggregate accuracy, one segment failing | Break results down by document type and field before reducing review | [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf), Task 5.5 |
| Cost per task rising, quality flat (agent loop) | Share of input read from cache below about 80%: look for something breaking the cache | [Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence) |
| `stop_reason: "refusal"` after a migration | Refusal handling and a configured fallback | [Opus 5.5 migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide) |

In that postmortem Anthropic said it had developed more sensitive evaluations, which it would keep improving, and that it would run quality evaluations continuously on true production systems. CCAR-P 4.4 names three causes to tell apart, "prompt failure, hallucinations, model mismatch"; how to separate them, and how to tell a model-output problem from an integration-layer one, is in [Debugging: model or integration](evaluation-and-reliability.md#debugging-model-or-integration).

### Decide

- If stakeholders define success differently, reconcile the criteria and their thresholds before the pilot; not after launch, when they become competing narratives.
- If any model, prompt, CLAUDE.md, skill or hook changes, run the suite before it ships; not a spot check.
- If the model must not change underneath you, call a pinned model ID and pin Skill versions (serving infrastructure can still shift behavior slightly); not a pre-4.6 alias or an unversioned Skill.
- If a variant wins offline and the change is significant, shadow it or A/B test it on a traffic slice before cutover; not a full switch.
- If an A/B result is small, check the confidence interval and sample size before claiming a win; not the raw difference from one run.
- If a production incident occurs, add it to the suite as a regression case; not only a hotfix.
- If quality drops after a document refresh with the model unchanged, inspect retrieval first ([Keeping the index correct](#keeping-the-index-correct)); not the model.

### Traps

- **Treating the suite as a one-off launch check.** It is the gate for every later change and the source of cost and latency baselines.
- **Optimizing on the cases you read.** Score the final configuration on held-out cases.
- **An A/B test as the first check.** It is slow and reaches users; offline evals come first.
- **Blaming the model when only the data changed.** The CCAR-P Sample 3 distractors (silently changed weights, a temperature that is too low, a shrunken context window) are worked through in [Keeping the index correct](#keeping-the-index-correct).
- **Assuming a pinned model ID can never behave differently.** The weights do not change, but serving infrastructure can.
- **Monitoring with no owner.** The pilot-to-production guide lists "Who owns continuous monitoring and degradation alerting?" among the ownership decisions to assign; alerts nobody is accountable for do not get acted on (our reading).

## Governance and risk in delivery

*Tested in: CCAR-P 5.1, 5.2, 5.3, 5.4, 5.5, 3.2, 6.4 and Sample 1 · CCAO-F D6.1, D6.2, D6.3, D6.4, D2.4 (the user-level view) and Sample 3 · CCDV-F 7.1 AI Application Security, 7.2 Guardrails and Safe Deployment and Sample 2 · CCAR-F 1.4-K1, 1.4-K2 and Question 1*

Governance is cheapest before the pilot. The pilot-to-production guide says programs that treat compliance as a late-stage gate find, during the production build, that regulatory requirements reshape the architecture they designed during the pilot, while "The programs that treat compliance as a design constraint avoid this entirely because they surface those requirements pre-pilot, when changes are decisions rather than delays." It adds that testing alone does not get you there: "Organizations that try to satisfy compliance requirements through testing alone consistently discover the gap at the worst possible time" ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf)). This section is the delivery view: what to decide first, how to prove each control exists, how much human oversight each output needs, and which failure modes to design against. The controls themselves are taught in [Data retention, training and compliance](security-and-governance.md#data-retention-training-and-compliance), [Admin and governance controls](security-and-governance.md#admin-and-governance-controls) and [Anthropic's Usage Policy](security-and-governance.md#anthropics-usage-policy).

### Before the pilot: classify, then rule out

Classification happens in discovery: sort the data by sensitivity, map the access rules the system inherits from it, ask what Legal and Compliance will actually enforce, and bring those reviewers in before the pilot launches. Those steps, with the pilot-to-production guide's wording, are questions 1, 2 and 4 of [The discovery questions, in order](#the-discovery-questions-in-order). One delivery detail to add: in Claude Enterprise, a connector passes through the member's own permissions in the source system.

Then rule out. The prep course's solution-design module puts governance and regulated-industry constraints before any other trade-off (quoted in the same discovery list). As of September 2026, these requirements narrow the choice of surface and platform:

| Requirement | What it rules in or out |
|---|---|
| PHI under HIPAA | A signed BAA and a HIPAA-enabled organization; the API rejects non-eligible features with a 400 (browser use, the one client-side tool the feature eligibility table marks as not blocked, is accepted but still sits outside HIPAA readiness). API HIPAA readiness does not cover Claude Code, Claude Platform on AWS, Microsoft Foundry, beta features not listed as eligible, or third-party integrations. The commercial BAA article lists Batch, Files, Skills, Code Execution, Computer Use and Web Fetch as not covered on a HIPAA-ready API organization, and data sent to external MCP servers is outside it. The API docs' feature eligibility table (as of September 2026) agrees on all of these except computer use, which it marks HIPAA-eligible as a client-side tool. The signed BAA decides. Never put PHI in JSON schema definitions |
| FedRAMP High | Claude for Government, Claude on Amazon Bedrock in AWS GovCloud, or Google Cloud with Assured Workloads (the [Public Sector FAQ](https://support.claude.com/en/articles/13756069-public-sector-faqs) says "Google Vertex"; current docs call it Google Cloud's Agent Platform, formerly Vertex AI). Claude Enterprise bought on AWS Marketplace is not FedRAMP authorized. ITAR data only through Bedrock (IL5) |
| Regulated workloads on AWS (FedRAMP High, IL4, IL5, HIPAA-ready, or AWS as sole data processor) | Bedrock, not Claude Platform on AWS; see [Choosing the platform](#choosing-the-platform) |
| GDPR | The DPA with Standard Contractual Clauses is part of the Commercial Terms; the customer is controller and Anthropic processor. Through a cloud platform, that platform's terms govern. First-party inference geos are only `"us"` and `"global"`; EU residency options run through Bedrock, Google Cloud and Foundry |
| No data retained | ZDR is enabled per organization by the account team. Claude Managed Agents is not covered (transcripts persist until deleted); Covered Models (Fable 5.1, Mythos 5.1, Fable 5, Mythos 5) need 30-day retention unless Anthropic authorizes otherwise; flagged content may be kept up to 2 years. Under ZDR, browser apps need a backend proxy because CORS is not supported |

FedRAMP is a property of the hosting service, not the model: "FedRAMP and DoD Impact Levels are certifications for cloud services (IaaS, PaaS, SaaS). AI models are software components, not cloud services." ([Public sector FAQs](https://support.claude.com/en/articles/13756069-public-sector-faqs)). The same FAQ says Claude models deploy within authorized environments and customers maintain their compliance posture through the hosting platform.

On the exam, CCAR-P 5.4 is "Ensure compliance with regulations (e.g., GDPR, HIPAA, FedRAMP)". Apply the rule-out step above to it: find which options the regulation excludes, then compare what remains. The product specifics in the table are date-sensitive (as of September 2026), so check the current docs before a real engagement.

!!! warning "Sources disagree on HIPAA, ZDR and Claude Code (as of September 2026)"

    The API docs say "If your organization handles PHI, HIPAA readiness is the arrangement to use; you do not also need ZDR." and "Claude Code is not covered under HIPAA readiness." ([API and data retention](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention)). The [Public Sector FAQ](https://support.claude.com/en/articles/13756069-public-sector-faqs) (dated March 25, 2026) says the BAA "requires a Zero Data Retention (ZDR) agreement", and the [commercial BAA article](https://privacy.claude.com/en/articles/8114513-business-associate-agreements-baa-for-commercial-customers) says "Some services, like Claude Code, are only covered under the BAA when ZDR is enabled". The two Claude Code statements describe different arrangements. API HIPAA readiness, which the API docs present as an alternative to ZDR, does not cover Claude Code at all; the commercial BAA article, which covers Claude for Work and the API, lists the Claude Code CLI (through the first-party API or Claude Enterprise OAuth) and Claude Code in the desktop app in local mode as covered only with ZDR enabled; it lists Claude Code on the web, desktop remote mode, Code Review, Security, Computer Use and Remote Control as not covered and incompatible with ZDR. Name the arrangement whenever you cite one of these rules. The full comparison is in [Data retention, training and compliance](security-and-governance.md#data-retention-training-and-compliance).

### Map every obligation to a control, an owner and evidence

The prep course's responsible-AI module sets the standard: "Map each compliance obligation to a named control, an owner, and an evidence artifact, so the architecture can be accurately audited" ([Responsible AI, Safety & Risk for Architects](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/responsible-ai-safety-risk-for-architects)). A register built from controls and owners Anthropic's material names, with product details as of September 2026 (the layout is ours):

| Obligation | Control | Owner the sources name | Evidence artifact |
|---|---|---|---|
| Only the right people reach a connector's data | Three gates (organization, role, member) must all be open; write access gets a different level of sign-off than read access | The course suggests the data-risk owner approve write grants | Connector and role configuration; audit logs (Enterprise only, exporting the past 180 days of events) |
| Content is available for compliance review | Compliance API (Claude Enterprise plans other than Public Sector; Claude Console customers get the Activity Feed only, not content), routed into the existing SIEM review | In Claude Enterprise, the Primary Owner, the only role that can enable it | Activity Feed, retained for 6 years |
| PHI is covered | HIPAA-ready configuration and an accepted BAA | In Claude Enterprise, the Primary Owner activates it and accepts the BAA (the API path differs; see [Data retention, training and compliance](security-and-governance.md#data-retention-training-and-compliance)) | A signed BAA before go-live |
| Claude Code stays inside its boundaries | Managed settings for files, commands and network destinations | The platform lead | `/status`, whose `Setting sources` line shows `Enterprise managed settings` |
| Spend stays within budget | Organization ceiling, group caps and per-member overrides | Whoever commits the budget | The cap table and usage history reviewed with the budget owner; the analytics dashboard for regular review |
| High-risk decisions get professional review | Human approval before release; AI disclosure at least at the start of each session | A qualified professional in the field | Decisions documented with an audit trail; quarterly compliance review |

Two rules make the register real. An owner must exist: "A decision with no owner is one of the quickest ways a rollout stalls." And a record has to be read: the Academy course warns that a record nobody reviews is not a control, which is why the Compliance API belongs in the SIEM review you already run ([Owners and intake](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/owners-and-intake); [Visibility: what you can measure](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/visibility-what-you-can-measure)).

### Match human oversight to consequence

The pilot-to-production guide: "Production governance requires a tiered approach: fast handling for routine, low-consequence outputs and more scrutiny for the ones that carry real weight." ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf))

| Tier | Oversight model | Review in the guide's table |
|---|---|---|
| 1. Automated | No human review; output goes directly to the workflow | Quarterly audit of output quality |
| 2. Sampled | A random subset is reviewed on a regular cadence | A portion of outputs reviewed weekly, starting with a higher rate of reviews and adjusting as error patterns stabilize |
| 3. Reviewed | A human approves every output before it reaches its audience | Every output reviewed before release; the review process audited monthly for catch rate and cost |
| 4. Advisory | AI provides analysis; a human makes the decision and produces the output (the guide's examples include credit and lending recommendations and candidate screening in hiring) | Every decision documented with an audit trail; compliance review quarterly |

- **Anthropic's Usage Policy sets a floor for high-risk uses.** Its High-Risk Use Case Requirements "apply to specific consumer-facing use cases that pose an elevated risk of harm", covering legal, healthcare, insurance, finance, employment and housing, academic testing, accreditation and admissions, and media or professional journalistic content. Where Claude provides advice, recommendations or subjective decisions directly affecting individuals or consumers, "a qualified professional in that field must review the content or decision prior to dissemination or finalization"; where outputs go directly to individuals or consumers, the use of AI must be disclosed: "This disclosure must be provided at a minimum at the beginning of each session." Separately, every consumer-facing chatbot or interactive AI agent must disclose that it is AI, and agentic use cases must still comply ([Usage Policy](https://www.anthropic.com/legal/aup)). In the tier table, the review requirement maps to Reviewed or Advisory (our mapping).
- **Checkpoints are audited too.** "If the catch rate is low and the cost is high, the checkpoint can be automated, sampled, or removed." ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf)). Name who has the authority to remove or automate review checkpoints as the system matures.
- **Deterministic rules need code, not prompts.** CCAR-F 1.4-K2: when deterministic compliance is required, "prompt instructions alone have a non-zero failure rate" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Where a step must always happen, enforce it in code: a hook or a prerequisite gate (the tools 1.4-K1 names), or in Claude Code a permission rule, which Claude Code enforces rather than the model. The CCAR-F Question 1 rationale gives the reason: "programmatic enforcement provides deterministic guarantees that prompt-based approaches cannot."
- **Routing by confidence** (who reviews which item, and how thresholds are calibrated) is taught in [Human review and confidence calibration](evaluation-and-reliability.md#human-review-and-confidence-calibration).

### Design the safety stack to fail closed

The prep course asks you to "Design the full safety stack for a Claude system, placing input screening, output screening, and tool-call authorization so the system fails closed rather than open" ([CCAR-P prep course](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional)). The layers, roughly in the order a request meets them, with the Anthropic guidance behind each (the arrangement is ours):

| Layer | What to do | Effect when it triggers |
|---|---|---|
| Capability | Remove tools the role does not need (least privilege by removal; CCAR-P Sample 1). In Claude Code, a bare tool-name deny rule such as `Bash` removes the tool from Claude's context entirely | A removed tool cannot be called at all |
| Input screening | Pre-screen user input with a lightweight model such as Claude Haiku 4.5, with structured outputs constraining its verdict. Anthropic's agent guidance notes that one model instance processing queries while another screens them tends to perform better than one call handling both. In Claude Enterprise, inference hooks (beta) send every governed prompt to your AI security server for an allow or deny verdict before inference | A denied request never reaches the model |
| Tool-call authorization | Deterministic enforcement for anything that must never happen: hooks and permission rules, which Claude Code enforces rather than the model | The call is blocked whatever the prompt says. A Bash deny rule matches the command text as written, so on its own it "isn't a security boundary around the program" |
| Untrusted content | Deliver third-party content only inside `tool_result` blocks, never in `system` prompts or plain user `text` blocks, and where possible wrap it in a JSON object rather than concatenating it into free-form text | Claude is trained to treat instructions inside tool results with appropriate skepticism (a tendency, not a guarantee), and JSON encoding makes it unambiguous that the payload is data, not a directive |
| Tool outputs | Screen what tools return with the same lightweight classifier before Claude acts on it | Raw content goes back as a `tool_result` only if the screen reports no injection; otherwise return an error or a stripped summary |
| Output screening | Output screening and post-processing (regex, keyword filtering or a prompted LLM, in Anthropic's prompt-leak guidance), plus regular analysis of outputs for signs of successful injection; handle `stop_reason: "refusal"` with a configured fallback | A refusal routes to the fallback rather than an unhandled response |
| Before launch | Red-team with documents, emails and tool outputs that deliberately contain injection attempts | Confirms "that Claude ignores them and that your screening and confirmation steps catch the rest" |

Inference hooks are in beta and, as of September 2026, cover the prompt side only: "Today the only hook event is `prompt`", the verdict timeout defaults to 5 seconds, and response-side enforcement is planned. The organization's failure handling setting decides what happens when the AI security server is unreachable, returns an error or misses the timeout: block the request (fail closed) or let it proceed without inspection (fail open) ([Inference hooks](https://platform.claude.com/docs/en/manage-claude/inference-hooks)). The course's framing for the whole stack is to "Distinguish between what the model's training reduces and what your application layer must still enforce" ([Responsible AI, Safety & Risk for Architects](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/responsible-ai-safety-risk-for-architects)). Techniques are in [Prompt injection](security-and-governance.md#prompt-injection), [Jailbreaks and guardrail layering](security-and-governance.md#jailbreaks-and-guardrail-layering) and [Least privilege for tools and agents](security-and-governance.md#least-privilege-for-tools-and-agents).

**How CCAR-P Sample 1 frames it.** A support agent can read tickets, draft replies, issue refunds and delete accounts, but staff only read and draft. The keyed answer removes the refund and delete tools: "Least privilege means removing capabilities the role does not require, eliminating the attack surface rather than monitoring or guarding it." The rationale rejects the other options: "Logging (A) and confirmations (C) are detective/compensating controls, not removal of unnecessary privilege; model size (D) is unrelated to authorization scope." ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf))

The same logic runs through the other exams' security and governance samples:

- **CCDV-F Sample 2 (Domain 7).** An agent summarizing user-submitted web pages meets hidden text telling it to reveal its system prompt. The keyed answer treats page content as untrusted input, keeps it separate from trusted instructions, and uses guardrails or hooks; the rationale rejects the alternatives because "Temperature (A) is irrelevant to injection; a polite request (C) is not an enforceable control; a more instruction-following model (D) can be more susceptible, not less." ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf))
- **CCAO-F Sample 3 (Domain 6).** A spreadsheet of customer names and account numbers meets a policy against sharing regulated personal data. The keyed answer removes or anonymizes the identifiers before upload; the rationale adds that "instructing the model not to retain data (C) does not satisfy the policy control" ([CCAO-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)).

### Name the failure modes in the design

The prep course's production checklist includes "naming failure modes for the chosen architecture, and articulating the mitigation for each" ([Enterprise Integration & Production](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/enterprise-integration-production)). The pilot-to-production guide defines an error mode as "a specific, recurring category of failure" and asks what each one costs at full production volume ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf)).

| Failure mode | Why it happens | Mitigation |
|---|---|---|
| Compounding errors in autonomous agents | Autonomy brings higher cost and the potential for compounding errors | Sandboxed testing, guardrails, stopping conditions such as a maximum iteration count |
| Context rot | Recall falls as the context window fills | The smallest high-signal context; see [Context Engineering](context-engineering.md#why-context-is-a-budget) |
| Drift | Output quality shifts after model updates or when prompts drift from their original intent | Continuous monitoring with alerts at degradation thresholds |
| Non-determinism breaks QA built for deterministic systems | The same input can produce different outputs | Build behavioral constraints into the architecture rather than relying on testing alone; measure consistency with `pass^k` where it matters |
| Aging model | Deprecated models are likely to be less reliable than active ones | Test newer models well before the retirement date, then migrate through the eval gate |
| Wrong multi-agent split | Agents that must share context or have many dependencies are a poor fit | Decompose by context boundaries, or stay with one agent |
| Stale retrieval | A refresh breaks the index or mismatches embeddings | See [Retrieval-augmented generation](#retrieval-augmented-generation) |

Incident response is part of the design, not the runbook written after the first incident. The guide's questions: who is notified, what the remediation path is, "Who has authority to pause the system?", and who tells affected parties; "Organizations that handle incidents well have documented answers to these questions before go-live." ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf))

### Fairness and transparency

CCAR-P 5.5 asks for "bias, fairness, transparency" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). The prep course's version is to "Identify where unequal outcomes can arise within a system and define the explanations required for users, regulators, and your own debugging team" ([Responsible AI, Safety & Risk for Architects](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/responsible-ai-safety-risk-for-architects)).

- **Make it measurable.** Anthropic's older enterprise guidance turns ethics into a metric: "less than 0.1% of outputs flagged for bias across 10,000 interactions." ([Building trusted AI in the enterprise](https://www-cdn.anthropic.com/e5c9de22bc8884089970bd262ca0c8b952cb9136.pdf))
- **Test by perturbation.** Anthropic's discrimination evaluation generates decision prompts across 70 scenarios and varies the demographic details in each; its released dataset varies age, gender and race for 135 examples per scenario ([discrim-eval dataset](https://huggingface.co/datasets/Anthropic/discrim-eval)). The paper found several prompt interventions "quite effective", especially "Illegal to discriminate", "Ignore demographics" and the two combined. The results are for Claude 2.0 (December 2023), and the authors caution that "we do not believe that performing well on our evaluations is sufficient grounds to warrant the use of models in the high-risk applications we describe here" ([Evaluating and mitigating discrimination](https://arxiv.org/html/2312.03689)).
- **Show the reasoning.** Anthropic's agent principles include making the agent's planning steps visible ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)). For users, disclosure that they are dealing with AI is a Usage Policy requirement for consumer-facing agents.
- **Govern the program.** The same older guidance recommends an AI review board, ethical guidelines, and transparent processes for model evaluation and incident response ([Building trusted AI in the enterprise](https://www-cdn.anthropic.com/e5c9de22bc8884089970bd262ca0c8b952cb9136.pdf)). Anthropic itself holds ISO/IEC 42001:2023 certification for AI management systems ([Anthropic certifications](https://privacy.claude.com/en/articles/10015870-what-certifications-has-anthropic-obtained)).

### Decide

- If a role does not need a capability, remove it; not logging or a confirmation step, because the Sample 1 rationale calls those detective or compensating controls.
- If a regulation applies, rule out the surfaces and platforms it excludes first; not after comparing cost and latency.
- If an obligation lacks a named owner or an evidence artifact, it is not yet a control.
- If a consumer-facing use case in a Usage Policy high-risk domain gives advice or decisions directly affecting individuals, use the Reviewed or Advisory tier (our mapping); not Automated or Sampled.
- If a task does not need regulated identifiers, remove or anonymize them before they reach Claude; not an instruction telling the model not to retain them (CCAO-F Sample 3).
- If a review checkpoint catches little at high cost, automate, sample or remove it, with the authority to do so named in advance.
- If a rule must hold every time, enforce it in code (hook, permission rule, prerequisite gate); not a prompt instruction.
- If untrusted content reaches the agent, isolate it from trusted instructions and gate sensitive tools; not a system-prompt request asking users to behave.

### Traps

- **Treating FedRAMP or HIPAA as a property of the model.** It comes from the hosting service and the arrangement you sign.
- **Assuming ZDR covers every product.** Managed Agents is not covered, and Covered Models require 30-day retention.
- **Assuming the first-party API offers an EU inference geo.** It offers `"us"` and `"global"`.
- **Compliance by testing alone.** Constraints belong in the architecture.
- **A bigger model as a security fix.** CCAR-P Sample 1: model size "is unrelated to authorization scope". CCDV-F Sample 2: a more instruction-following model "can be more susceptible, not less."
- **Asking for good behavior as a control.** A system-prompt line asking users not to include malicious instructions "is not an enforceable control" (CCDV-F Sample 2), and telling the model not to retain data "does not satisfy the policy control" (CCAO-F Sample 3).

## Stakeholder communication and lifecycle

*Tested in: CCAR-P 6.2, 6.3, 6.4, 6.5 (6.1 is in Discovery and requirements), 1.6 (performance SLAs) · CCDV-F 2.2 Systems Life Cycle, 2.6 Configuration Management (model version pinning, prompt versioning) · CCAO-F D4.5 · CCAR-F: not listed*

The CCAR-P prep course describes this work as the conversations "that decide whether a working system actually ships, adopts, and outlasts your involvement" ([CCAR-P prep course](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional)). The technical design can be right and the program can still fail: the pilot-to-production guide cites Accenture data that "42% of organizations rely on shared IT and finance accountability with no single owner responsible for AI costs and outcomes." ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf)). Discovery itself is covered in [Discovery and requirements](#discovery-and-requirements); this section takes the program from design to its long operating life.

### The five phases the guide names

CCAR-P 6.5 lists "discovery, design, handoff, monitoring, iteration" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). What the architect produces in each comes from Anthropic's prep course and delivery guides; the "You are done when" column is our reading of those sources, not an official checklist:

| Phase | What you produce | You are done when |
|---|---|---|
| Discovery | A use case with a defined user, task, output and measurable quality threshold; the human performance baseline; a lightweight total cost of ownership (TCO) model | Stakeholders agree on the success criteria before the pilot, not after launch |
| Design | An architecture specification covering integration patterns for compliance, identity (SSO or OAuth), authorization, data handling and observability instrumentation; named failure modes with mitigations; a documented go/no-go process | The go/no-go owner signs off against written graduation criteria |
| Handoff | Shared configuration, the rollout pattern, the Skills distribution strategy and spend controls; named owners for every running responsibility | The team runs the system without you |
| Monitoring | What to measure, automated alerts at degradation thresholds, escalation protocols | Alerts reach a named owner who acts |
| Iteration | Review triggers, what an SLA breach requires, when to iterate and when to re-architect | The loop is running on a cadence |

The same idea appears in CCDV-F 2.2 as the systems life cycle, which the guide describes as "Systems life cycle management concepts and frameworks used to develop, implement, operate, and maintain IT systems." ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). NIST's glossary defines the system development life cycle as the scope of activities from a system's initiation, through development and acquisition, implementation, and operation and maintenance, to its disposal ([NIST glossary](https://csrc.nist.gov/glossary/term/system_development_life_cycle)). Anthropic's AI-native SDLC playbook groups its plays into six non-linear stages (Plan, Design, Build, Test, Deploy, Maintain) and turns the traditional linear flow into a loop: continuous evals woven through implementation replace QA gates at stage boundaries, each stage ends by committing an artifact (intent.md, spec.md, plan.md, the diff and its tests, the PR with its review findings, the incident record), and "The chain of commits is also the audit trail: who asked for what, what the agent produced, and who approved it." ([AI-native SDLC playbook](https://academy.claude.com/courses/ai-native-sdlc-playbook/introduction)). When agents speed up the build phase, the bottleneck moves to the stages on either side of it, mainly plan, review and test, and deploy, which still run at human speed.

### From pilot to production

Pilots succeed partly because they skip production rigor, so the pilot-to-production guide's fix is to build "the governance, operational ownership, delivery stability, and economics as a part of the pilot design itself" ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf)). Four instruments carry a pilot across:

- **A documented go/no-go process**, written before the pilot: who has decision rights, what triggers escalation, how wins and setbacks are communicated, and the rollout sequence.
- **Graduation criteria.** Anthropic's older enterprise guidance names three groups of potential graduation criteria: performance thresholds (accuracy, speed and latency, cost efficiency), operational readiness (system stability, support infrastructure, team capability) and risk management infrastructure (security compliance, data protection, operational controls).
- **A parallel run.** Anthropic's older enterprise guidance suggests running the AI-enhanced process alongside the existing workflow until performance and reliability are proven ([Building trusted AI in the enterprise](https://www-cdn.anthropic.com/e5c9de22bc8884089970bd262ca0c8b952cb9136.pdf)).
- **An owner with real authority.** The pilot-to-production guide treats decision rights, escalation authority and executive backing as three separate requirements; missing any one of them can stall the program ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf)).

In the pilot-to-production guide's words, mature organizations "stopped treating AI deployment as a project with an end date and started treating it as an ongoing initiative with dedicated governance, metrics, and program management." ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf))

### A handoff that survives your absence

The course's lifecycle outcome is to "design a handoff that survives your absence", because "a design that only the Architect understands falls apart the moment they leave the room." ([CCAR-P prep course](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional)). Before you leave, every running responsibility needs a name. The pilot-to-production guide's "assign" questions make a usable checklist ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf)):

- "Who owns continuous monitoring and degradation alerting?"
- "Who has authority to remove or automate review checkpoints as the system matures?"
- "Who is the human supervisor, exception handler, or system improver at scale?"
- "Who is accountable when model versioning or deprecation affects deployed systems?"
- "Who owns ongoing AI program management as a function?"

The receiving team also needs judgment: the ability to identify incorrect outputs, decide which cases require escalation, and detect system drift before it becomes a production issue, which the pilot-to-production guide says takes training and role design. For internal rollouts, Anthropic's champion kit warns that "Adoption that depends on a single person is fragile." ([Champion kit](https://code.claude.com/docs/en/champion-kit)); its observable signal that the handoff is working is in [Enablement and review discipline (7.2)](#enablement-and-review-discipline-72).

### The model lifecycle inside the program lifecycle

Model IDs have a lifecycle of their own, and the pilot-to-production guide asks who "is accountable when model versioning or deprecation affects deployed systems" ([Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf)), so the program plan has to include model changes. The facts, as of September 2026 ([Model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations), [Model IDs and versions](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions)):

| Fact | What it means for the program |
|---|---|
| Models move through Active, Legacy, Deprecated and Retired; a deprecated model still works but has a recommended replacement and a retirement date; requests to a retired model fail | Track the state of every model ID you use |
| At least 60 days' notice before retiring a publicly released model; impacted customers are notified by email and in the documentation | Budget a migration and an eval cycle inside that window; the docs say to test newer models well before the retirement date |
| Anthropic's dates cover the Claude API, Claude Platform on AWS and Foundry; Amazon Bedrock and Google Cloud set their own retirement schedules | Check the platform you actually run on |
| Each model ID is a pinned version; from Claude 4.6 on, IDs are dateless, and dateless IDs are not evergreen pointers to the latest version | Upgrades happen only when you change the ID, so they can be gated |
| Example: on June 5, 2026 Anthropic notified Claude Opus 4.1 users of its retirement on the Claude API; `claude-opus-4-1-20250805` was retired August 5, 2026, with `claude-opus-4-8` as the replacement. Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) and Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) are Active, with retirement "Not sooner than" September 29, 2026 and October 15, 2026 respectively | A program built on those models needs its migration planned now |
| Find remaining usage by exporting the Console Usage page to CSV, broken down by API key and model | The inventory for the migration |

CCDV-F 2.6 names "model version pinning". In current docs, "Every Claude model ID is a pinned snapshot, including the dateless IDs used from the 4.6 generation on." ([Models overview](https://platform.claude.com/docs/en/models/overview)), so on the API, pinning means recording the exact model ID in configuration and changing it deliberately. Claude Code on a cloud provider pins through environment variables, shown in [Choose the provider](#choose-the-provider). Migration mechanics are in [Model versions, deprecation and migration](claude-api.md#model-versions-deprecation-and-migration); the eval gate is in [Evaluation strategy for a program](#evaluation-strategy-for-a-program).

### Communicating decisions and trade-offs

The stakeholder module's rule: "Present an architectural trade-off in terms a business stakeholder can act on by pairing each choice with a cost, a risk, and what a reversal would take, so executive and procurement reviews reach a decision instead of stalling" ([Stakeholder Engagement, Lifecycle & GTM](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/stakeholder-engagement-lifecycle-gtm)). Different audiences need different things:

| Audience | What they need from you | Source |
|---|---|---|
| Sponsor | An outcome document that compares the direct API, Bedrock, Vertex and third-party routes on latency, compliance and cost, legible to a non-technical reader | Prep course, stakeholder module |
| Executives | A strategic vision tying the initiative to business outcomes; realistic guidance on timelines and impact | Building trusted AI in the enterprise |
| QA and compliance | That outputs are non-deterministic, and whether model behavior can be explained in a regulatory or legal context | Pilot-to-production guide |
| Users, regulators, your debugging team | The explanations each one requires | Prep course, responsible-AI module |
| Everyone | Which questions are "work out" (cross-functional, several owners) and which are "assign" (a CIO or business leader decides) | Pilot-to-production guide |

**Give reversal cost its own line.** Some choices are cheap to undo: a model ID is a configuration change behind an eval gate. Others are not. Anthropic's Claude Enterprise course names four settings that are hard to undo: domain claiming (once on, it can't be reversed), organization topology (merging or splitting organizations later means re-provisioning every affected member; the course's [One organization or many](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/one-organization-or-many) lesson adds that adding a second organization later is cheaper than pulling apart one you consolidated too much), group structure mapped from your identity provider, and data retention (conversations already deleted under a shorter window are gone for good) ([Deploying Claude Enterprise with confidence](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/five-decisions-and-the-frame)). Granting a surface is cheap, while "The expensive direction is taking one away" ([Surfaces each group gets](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/surfaces-each-group-gets)). On the API, "Workspace geo is set when you create a workspace and can't be changed afterward." ([Data residency](https://platform.claude.com/docs/en/manage-claude/data-residency)). More settings that are hard to undo (HIPAA readiness, CMEK, turning off memory for the organization) are listed in [Admin and governance controls](security-and-governance.md#admin-and-governance-controls).

Using the policy-assistant example from [Cost modeling](#cost-modeling), an option statement can look like this. It is illustrative: the layout and the risk and reversal wording are ours, and the figures are that section's arithmetic at list prices as of September 2026:

```text
Option B: Claude Sonnet 5 with prompt caching
  Cost:     about $852 a month at 100,000 requests (Opus 5.5 with caching: about $1,552)
  Risk:     quality on the hardest tenth of requests not yet measured; eval before commit
  Reversal: change the model ID and re-run the eval suite; no data migration
```

State value and limits honestly. The CCAO-F guide asks Associates to "Communicate Claude's value and limitations to stakeholders" ([CCAO-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)), and Anthropic's own material models the tone: usage metrics "tell you Claude is being used, not what that use produced" ([Adoption signals](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/adoption-signals)), and on hallucinations Anthropic says "this is an ongoing challenge for the entire AI field, not at all a solved problem." ([Why do AI models hallucinate?](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate)). Adoption numbers are diagnostics: "The month you turn a diagnostic into a quota, members optimize for the number instead of the work." ([Adoption signals](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/adoption-signals))

### Expectations and SLAs

CCAR-P 6.3 says "(including SLAs)" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). An SLA you give stakeholders is one you build, from what each dependency actually commits to (as of September 2026):

| Dependency | What is committed | Source |
|---|---|---|
| Claude API, standard tier | Best-effort availability; the Commercial Terms provide the services as is and as available and do not warrant uninterrupted use; rate limits are maximums, not guaranteed minimums | [Service tiers](https://platform.claude.com/docs/en/api/service-tiers), [Commercial Terms](https://www.anthropic.com/legal/commercial-terms), [Rate limits](https://platform.claude.com/docs/en/api/rate-limits) |
| Priority Tier | Targets 99.5% uptime, but new commitments can no longer be purchased and it does not support the newest models; details in [Priority Tier and the `service_tier` parameter](#priority-tier-and-the-service_tier-parameter) | [Service tiers](https://platform.claude.com/docs/en/api/service-tiers) |
| Research previews | Can carry no commitment at all: MCP tunnels are provided "as-is" without uptime, support or continuity commitments | [MCP tunnels](https://platform.claude.com/docs/en/agents-and-tools/mcp-tunnels/overview) |
| Amazon Bedrock | AWS publishes a 99.9% Monthly Uptime Percentage per region with service credits (10% below 99.9%, 25% below 99.0%, 100% below 95.0%), applied against future Bedrock payments (or, at AWS's discretion, to the card used for that billing cycle); it covers AWS's service, and the page says "Last Updated: October 4, 2023" | [Amazon Bedrock SLA](https://aws.amazon.com/bedrock/sla/) |
| Voyage AI embeddings | A Service Level Objective of at least 99.5% monthly uptime for `api.voyageai.com` on a commercially reasonable efforts basis, not an SLA; the page says "Last Updated: September 18, 2024" | [Voyage service level objectives](https://docs.voyageai.com/docs/service-level-objectives) |
| Anthropic support | Written and asynchronous; response times vary by plan and severity | [How to get support](https://support.claude.com/en/articles/9015913-how-to-get-support) |

Close the gap with design and measurement: write the SLA as a percentile target (one of Anthropic's example success criteria puts 95% of responses under 200 ms, in [Define success criteria](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests)), specify "reliability patterns (retries, fallbacks, circuit breakers)" as the prep course asks ([Enterprise Integration & Production](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/enterprise-integration-production)), note that the official SDKs retry transient failures twice by default with exponential backoff, honoring the `retry-after` header when present ([Errors](https://platform.claude.com/docs/en/api/errors)), and consider a second platform with its own capacity pool as a failover target ([Choosing the platform](#choosing-the-platform)). Point stakeholders to [status.claude.com](https://status.claude.com/), which as of September 2026 tracks six components (claude.ai, Claude Console, Claude API, Claude Code, Claude Cowork and Claude for Government) and offers email, text message, Slack, Microsoft Teams, webhook, Atom and RSS subscriptions. Capacity mechanics are in [Deployment platforms and capacity](#deployment-platforms-and-capacity).

Then run the feedback loop on a cadence. Anthropic's evals guidance triages user feedback constantly and reads sampled transcripts weekly ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)); the [Building AI Agents in the Enterprise](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/69f3af1f0b8ebe5cde42fcda_Claude-Building-AI-Agents-in-the-Enterpise-04302026_v2%20%281%29.pdf) guide measures adoption weekly in the pilot phase, with qualitative feedback collected alongside the numbers; and the [Claude Enterprise Administrator Guide](https://claude.com/resources/tutorials/claude-enterprise-administrator-guide) suggests a brief retrospective at each phase of rollout and a regular reporting cadence such as quarterly business reviews. The Administrator Guide's example targets include weekly active users at 70% of licensed seats, 3+ hours saved per user per week (user survey) and a satisfaction score of 4.0+ out of 5.0 (quarterly survey), and it says to shift from tracking activity metrics to demonstrating business value as the rollout scales.

### The documentation set

CCAR-P 6.4 asks you to "Document architectures and provide implementation guidance" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). The artifact names and grouping below are ours; what each one records comes from Anthropic's material:

| Artifact | What it records |
|---|---|
| Architecture specification | Integration patterns for compliance, identity, authorization, data handling and observability; failure modes and mitigations |
| Model and prompt register | Exact model IDs (each is a pinned version); prompts under version control with their purpose and expected behavior |
| Skills registry | Per Skill: purpose, owner, version, dependencies and evaluation status; stored in Git with a rollback plan that keeps the previous version; authors do not review their own Skills |
| Team context for Claude Code | A project CLAUDE.md with build and test commands, coding standards, architectural decisions, naming conventions and workflows, reviewed and pruned like code |
| Go/no-go and incident runbooks | The go/no-go process from [From pilot to production](#from-pilot-to-production), written before the pilot, and the incident answers from [Name the failure modes in the design](#name-the-failure-modes-in-the-design), written before go-live. Runbooks, alerts and dashboards can embed `claude-cli://` deep links that open Claude Code in the right repo with a prepared prompt, but GitHub-rendered Markdown does not allow `claude-cli://`, so such a link in a GitHub README, issue or wiki shows only its label |
| Rollout plan | For Claude Enterprise, the five decisions (Structure & Identity, Access, Governance, Spend, Visibility), made in that order because each scopes the ones after it; from the pre-launch checklist, documented data retention policies and a completed security review with IT and information security |
| Compliance register | Obligation, control, owner and evidence (see [Governance and risk in delivery](#governance-and-risk-in-delivery)) |

### Decide

- If a decision will reach an executive or procurement review, present each option with its cost, its risk and what reversing it would take; not a capability comparison.
- If a choice is hard to reverse (domain claiming, organization topology, group structure, retention, workspace geo), make it deliberately and early, with the owner who will live with it.
- If a stakeholder wants an uptime guarantee, state what each dependency commits to and design the gap closed (retries, fallbacks, a second platform) or contract for capacity; not a promise the platform does not make.
- If a model you depend on is deprecated, treat the notice window as a scheduled migration with an eval gate; not an emergency.
- If you are about to hand over, name an owner for monitoring, checkpoints, supervision, deprecations and program management; not a document that only you can interpret.

### Traps

- **Treating the launch as the end.** In the pilot-to-production guide, mature organizations run AI deployment as an ongoing initiative with dedicated governance, metrics and program management.
- **Promising an SLA the platform does not offer.** The standard tier is best-effort.
- **Adoption quotas.** Turn a diagnostic into a quota and members optimize for the number instead of the work.
- **Assuming a dateless model ID updates itself.** It is a pinned snapshot; you move when you change it.
- **Owners without authority.** Decision rights, escalation authority and executive backing are separate requirements.

## Rolling out Claude Code to an engineering organization

*Tested in: CCAR-P 7.1, 7.2, 7.3 · CCDV-F 3.1 Claude Code Operation (CLAUDE.md hierarchy, repository initialization, settings.json), 2.6 Configuration Management (CLAUDE.md files, settings.json, model version pinning) · CCAR-F 3.1-K1, 3.1-K2, 3.1-S1, 3.1-S4, 3.2-K1, 3.2-S1, 2.4-K1, 2.4-K2, 2.4-S1, 2.4-S2, 3.6-K4, Exercise 2 steps 1 and 4 (Configure Claude Code for a Team Development Workflow) and Question 4 (Scenario 2, Code Generation with Claude Code) · CCAO-F: not listed*

This section follows a Claude Code rollout from the provider choice through what the repository carries and what policy enforces, spend, enablement and first-line support. CCAR-P Domain 7 (7% of the exam) is the smallest domain, and its first objective names the tool: "Configure Claude tools and environments for teams (e.g., Claude Code)" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). The matching prep module, Team Enablement & Operational Productivity (listed at 45 minutes), states its goal in one line: "Learn to enable a team to adopt a live Claude system and run it without depending on you." Its three learning objectives pair off with the three Domain 7 objectives (the pairing is ours; both texts are verbatim):

| CCAR-P objective | What the prep module asks you to do ([Team Enablement & Operational Productivity](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/developer-productivity-enablement)) |
|---|---|
| 7.1 Configure Claude tools and environments for teams (e.g., Claude Code) | "Configure Claude tooling and environments for a team, including the shared configuration, the rollout pattern, the Skills distribution strategy, and the spend controls that belong in team setup" |
| 7.2 Improve developer workflows using AI-assisted tooling | "Improve developer workflows with AI tooling and define the review discipline that keeps AI-generated work trustworthy before it reaches production" |
| 7.3 Support debugging and operational issue resolution | "Support debugging and operational issue resolution by connecting symptoms to architecture causes and building the team toward self-sufficiency" |

The configuration mechanics behind the rollout are taught in [Managed settings for organizations](claude-code-configuration.md#managed-settings-for-organizations), [CLAUDE.md and the memory hierarchy](claude-code-configuration.md#claudemd-and-the-memory-hierarchy), [MCP servers in Claude Code](claude-code-configuration.md#mcp-servers-in-claude-code), [Monitoring, usage and cost](claude-code-workflows.md#monitoring-usage-and-cost) and [Claude Code security controls](security-and-governance.md#claude-code-security-controls).

### The decisions, in order

Anthropic's administrator page "walks through the deployment decisions in order" ([Set up Claude Code for your organization](https://code.claude.com/docs/en/admin-setup)). With the pilot and launch steps from Anthropic's cost and enablement material added (the merged sequence is ours):

1. **Choose the API provider:** billing, authentication, the compliance posture you inherit, and which features developers get.
2. **Decide how settings reach devices:** server-managed settings from the admin console, an MDM or OS-level policy, or a `managed-settings.json` file.
3. **Decide what to enforce:** permission rules, sandboxing, MCP and plugin allowlists, login enforcement, model and effort limits.
4. **Set up usage visibility before anyone gets access:** "That’s why you set visibility up before members get access, not after" ([Visibility: what you can measure](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/visibility-what-you-can-measure)).
5. **Review data handling:** retention, ZDR, and whether a gateway is needed for request-level audit logging.
6. **Pilot, then expand in waves:** a small pilot group establishes a usage baseline before the wider rollout ([Manage costs effectively](https://code.claude.com/docs/en/costs)).
7. **Verify, launch and enable:** `/status` on a developer machine, a launch message, a staffed channel, champions.

### Choose the provider

| Provider | Choose it when ([admin setup](https://code.claude.com/docs/en/admin-setup)) | Where you cap spend ([costs](https://code.claude.com/docs/en/costs)) |
|---|---|---|
| Claude for Teams or Enterprise | You want Claude Code and claude.ai under one per-seat subscription with no infrastructure to run; "This is the default recommendation." | Seat-based plans: the seat allowance is the default ceiling, and with usage credits on you set spend limits in the claude.ai admin settings at organization, group or individual member level. Usage-based Enterprise: no per-seat allowance; spend limits (default &#36;0) at organization, group or member level |
| Claude Console | You are API-first or want pay-as-you-go billing | Workspace spend limits; the "Claude Code" workspace, created automatically the first time Claude Code authenticates with a Console account (you cannot create API keys for it), is the only workspace with per-user monthly spend limits |
| Amazon Bedrock, Google Cloud's Agent Platform, Microsoft Foundry | You want to inherit existing AWS, GCP or Azure compliance controls and billing | Your cloud provider's billing console, or a self-hosted Claude apps gateway with per-user spend limits |

What each choice brings with it:

- **Some features need claude.ai accounts.** Cloud sessions, Routines, Code Review, Remote Control and the Chrome extension "aren't available through Console API keys or cloud-provider credentials alone" ([Set up Claude Code for your organization](https://code.claude.com/docs/en/admin-setup)), so a cloud-provider rollout must decide whether developers also need Teams or Enterprise seats. Enterprise adds domain capture, role-based permissions and Compliance API access over Teams.
- **The CLI works everywhere; organization controls do not.** "The Claude Code CLI and everything that runs locally work on every provider." ([Feature availability](https://code.claude.com/docs/en/feature-availability)). Claude Enterprise's organization-level model and effort controls in admin settings do not reach sessions on Bedrock, Google Cloud's Agent Platform, Foundry or Claude Platform on AWS; use managed settings there (`availableModels`, `model`, `maxEffortLevel`). Anthropic's analytics do not see cloud usage either, so use OpenTelemetry or a gateway.
- **A gateway centralizes, and you run it.** Credentials, usage tracking, cost controls and audit logging move to one place, but "the gateway becomes infrastructure your organization operates." ([LLM gateway](https://code.claude.com/docs/en/llm-gateway)). Give each developer their own gateway credential so usage is attributed and offboarding is one revocation. Anthropic's self-hosted Claude apps gateway signs developers in through browser SSO and has no service-token flow, so CI pipelines with no developer to approve the sign-in are configured against the provider directly ([Gateways](https://code.claude.com/docs/en/gateways)).

On Amazon Bedrock, Google Cloud's Agent Platform, Microsoft Foundry or Claude Platform on AWS, pin model versions before the rollout with `ANTHROPIC_DEFAULT_FABLE_MODEL`, `ANTHROPIC_DEFAULT_OPUS_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL` and `ANTHROPIC_DEFAULT_HAIKU_MODEL`. Without pinning, aliases resolve to Claude Code's built-in default for that provider, "which can lag the newest release and may not yet be enabled in your account"; "Pinning lets you control when your users move to a new model." ([Enterprise deployment overview](https://code.claude.com/docs/en/third-party-integrations)). On Bedrock as of September 2026, the unpinned `opus` alias resolves to Opus 5.5 from Claude Code v2.1.280 (older clients resolve it to Opus 5, Opus 4.8 or Opus 4.6, depending on version) and the unpinned `sonnet` alias to Sonnet 4.5. The docs' Bedrock example pins each alias ([Amazon Bedrock](https://code.claude.com/docs/en/amazon-bedrock)):

```bash
export ANTHROPIC_DEFAULT_OPUS_MODEL='us.anthropic.claude-opus-4-8'
export ANTHROPIC_DEFAULT_SONNET_MODEL='us.anthropic.claude-sonnet-4-6'
export ANTHROPIC_DEFAULT_HAIKU_MODEL='us.anthropic.claude-haiku-4-5-20251001-v1:0'
```

### What lives in the repository and what lives in policy

The repository carries team conventions that anyone can improve through review. Managed settings carry the organization's rules: "Claude Code enforces organization policy through managed settings that take precedence over local developer configuration." ([Set up Claude Code for your organization](https://code.claude.com/docs/en/admin-setup))

| What | Where | Owner |
|---|---|---|
| Organization rules: permissions, sandboxing, MCP and plugin allowlists, login method, models, telemetry | Managed settings: server-managed from the admin console (Teams or Enterprise; fetched at startup and hourly), MDM or OS policy, or `managed-settings.json` | The platform lead, who sets file, command and network boundaries for each group |
| Team conventions: build and test commands, coding standards, architectural decisions | Project `CLAUDE.md` (repository root or `.claude/CLAUDE.md`), in Git and reviewed like code | The team, through pull requests |
| Team skills and commands | `.claude/skills/<name>/SKILL.md` and `.claude/commands/`, in Git | The team; when a skill encodes a policy, the policy owner signs off on changes to it |
| Shared MCP servers | `.mcp.json`, with secrets as environment variables such as `${GITHUB_TOKEN}` | One central team |
| Shared project settings, including team hooks | `.claude/settings.json`, committed; non-negotiable hooks go in managed settings instead | The team; the platform or IT admin for managed hooks |
| Personal overrides for one project | `.claude/settings.local.json`; the first time Claude Code writes it in a Git repository that doesn't already ignore it, Claude Code adds it to your global Git excludes file | The individual |
| Personal preferences and experiments | `~/.claude/CLAUDE.md`, `~/.claude/settings.json`, `~/.claude/commands/`, `~/.claude/skills/`, and MCP servers in `~/.claude.json` | The individual; not shared through version control |

When the same key is set in more than one settings file, the value from the highest level wins (array settings such as permission lists merge instead, as below): managed settings, then command-line arguments, then `.claude/settings.local.json`, then `.claude/settings.json`, then `~/.claude/settings.json` ([Settings](https://code.claude.com/docs/en/settings)). Details are in [Settings files and precedence](claude-code-configuration.md#settings-files-and-precedence).

- **Shared means committed.** CCAR-F Question 4 puts a team `/review` command in `.claude/commands/` in the repository because "These commands are version-controlled and automatically available to all developers when they clone or pull the repo." ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The mirror image is CCAR-F 3.1-S1: a new team member misses instructions that live in someone's `~/.claude/CLAUDE.md`.
- **Managed lists only grow.** "Array settings such as `permissions.allow` and `permissions.deny` merge entries from all sources, so developers can extend managed lists but not remove from them." For `fallbackModel`, `availableModels` and `modelPicker`, the managed value replaces lower layers rather than merging. The Windows HKCU registry is writable without elevation, so "treat it as a convenience default rather than an enforcement channel." ([Set up Claude Code for your organization](https://code.claude.com/docs/en/admin-setup))
- **One team owns MCP.** "We recommend that one central team configures MCP servers and checks a `.mcp.json` configuration into the codebase so that all users benefit." ([Enterprise deployment overview](https://code.claude.com/docs/en/third-party-integrations)). CCAR-F contrasts two MCP scopes: project-level `.mcp.json` for shared team tooling and user-level `~/.claude.json` for personal or experimental servers (2.4-K1). The current docs add a third, local scope, which is the default, is also stored in `~/.claude.json`, and is the scope they recommend for experimental configurations. On the exam, use the guide's two-scope model.

Put telemetry in the same policy so it is on from the first session. The docs' organization-wide example for managed settings (metrics and labels are covered in [Monitoring, usage and cost](claude-code-workflows.md#monitoring-usage-and-cost)):

```json
{
  "env": {
    "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
    "OTEL_METRICS_EXPORTER": "otlp",
    "OTEL_LOGS_EXPORTER": "otlp",
    "OTEL_EXPORTER_OTLP_PROTOCOL": "grpc",
    "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector.example.com:4317",
    "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer example-token"
  }
}
```

A managed `OTEL_EXPORTER_OTLP_*` value removes conflicting developer-set variables at startup, which locks the destination.

!!! warning "Exam guide vs current docs: commands and skills"

    The July 2026 guides treat `.claude/commands/` as the home of project slash commands (CCAR-F 3.2-K1 and Question 4; CCDV-F 3.1 lists built-in and custom slash commands among Claude Code's features). The Claude Code docs now say "Custom commands have been merged into skills." A file at `.claude/commands/deploy.md` and a skill at `.claude/skills/deploy/SKILL.md` both create `/deploy`, existing command files keep working, and the docs say to "Prefer a skill for new work" ([Skills](https://code.claude.com/docs/en/skills)). On the exam, answer with the guide's model: project commands in `.claude/commands/`, personal ones in `~/.claude/commands/`.

### Spend in the rollout plan

Say which billing model a statement applies to, because Anthropic's pages describe two. The [Claude Enterprise Administrator Guide](https://claude.com/resources/tutorials/claude-enterprise-administrator-guide) says "Claude Code access requires a Premium seat (legacy model), or a Chat + Code or Claude Enterprise seat (usage-based model)."

- **Usage-based Enterprise:** "the seat fee covers access only, and all usage is billed separately at API rates", and "There are no per-seat usage limits and no included token allowance." ([What is the Enterprise plan?](https://support.claude.com/en/articles/9797531-what-is-the-enterprise-plan)). Spend limits default to &#36;0 when you assign seats, so set them as part of seat assignment ([Claude Enterprise Administrator Guide](https://claude.com/resources/tutorials/claude-enterprise-administrator-guide)).
- **Seat allowance:** Claude Code's cost page says that on Teams and Enterprise plans each member's Claude Code usage "draws from a per-seat allowance that resets on a rolling five-hour window and a weekly window", shared with Claude chat and Cowork and sized by seat tier (Standard or Premium); to let members continue past it, you turn on usage credits and set spend limits ([Manage costs effectively](https://code.claude.com/docs/en/costs)). That description matches seat-based plans, not the usage-based Enterprise plan above.

- **Waves, not day one.** "Giving everyone Claude Code and Cowork access on day one is the fastest way to generate unexpected consumption." Claude Code and Cowork consume tokens much faster than chat. Start with "RBAC group-level limits and per-user limits"; the organization-level limit is a hard ceiling, and hitting it affects everyone simultaneously. A group that consistently approaches its limit should be investigated before the limit is raised ([Claude Enterprise consumption guide](https://support.claude.com/en/articles/14782391-claude-enterprise-consumption-guide)).
- **The default model is a spend lever.** "Because most members run the model they see when they start a new task, the default steers most of your spend" ([Surfaces each group gets](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/surfaces-each-group-gets)). In the course's worked example, lowering the default model on the Engineering role fixed a spend spike and the cap stayed where it was ([Managing spend](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/managing-spend)).
- **Know the usual causes.** On an API or cloud-provider plan, unexpectedly high Claude Code spend "usually traces back to long sessions that were never cleared or to Opus left as the default model." Agent teams are disabled by default, and their token use scales with the number of active teammates ([Manage costs effectively](https://code.claude.com/docs/en/costs)).
- **Size the budget.** Anthropic's averages across enterprise deployments (about &#36;13 per developer per active day) and per-user rate-limit guidance are in [Monitoring, usage and cost](claude-code-workflows.md#monitoring-usage-and-cost); the per-developer ranges sit with the other seat and subscription costs in [Seats, subscriptions and chargeback](#seats-subscriptions-and-chargeback).

### Enablement and review discipline (7.2)

- **Make installation trivial.** The docs say that where the development environment is custom, a "one click" way to install Claude Code is key to growing adoption across an organization ([Enterprise deployment overview](https://code.claude.com/docs/en/third-party-integrations)).
- **Start with guided usage.** The same page suggests new users start with codebase Q&A or smaller bug fixes and feature requests, ask Claude Code to make a plan, and check its suggestions, letting it run more agentically as they learn. For repository initialization, `/init` generates a starting `CLAUDE.md` by analyzing the codebase, and if a `CLAUDE.md` already exists it suggests improvements rather than overwriting it ([Memory](https://code.claude.com/docs/en/memory)).
- **Launch properly.** The communications kit's pre-launch checklist: a `#claude-code` channel linked in the message, the install command tested on at least one machine in your environment, a security and data-handling link, one concrete first task from your own codebase, a named owner for the channel for the first 48 hours ("Unanswered launch-day questions kill momentum"), and a C-suite sponsor to send or co-sign the announcement, because "Exec-sent launches consistently see higher first-week adoption than admin-sent ones" ([Communications kit](https://code.claude.com/docs/en/communications-kit)).
- **Grow champions, then hand off.** A champion shares what they discover, is the person people ask, and grows the circle. The 30-day playbook ends in week 4 by identifying a second champion, and it has worked when "questions in the channel are being answered by people other than you." Security and data-handling questions go to the administrator, and "champions should not improvise this answer." Running `/team-onboarding` in a well-used project produces a setup guide a new teammate can paste as their first message ([Champion kit](https://code.claude.com/docs/en/champion-kit)).

The review discipline in Anthropic's AI-native SDLC playbook ([AI in the PR review loop](https://academy.claude.com/courses/ai-native-sdlc-playbook/ai-in-the-pr-review-loop)): every PR gets the same review passes with findings ranked by severity; "Findings do not approve or block a PR on their own, and branch protection still requires approval from a code owner"; "Anything the agent writes arrives as a PR through branch protection, and the agent has no route to push to main" ([CI/CD integration and deployment](https://academy.claude.com/courses/ai-native-sdlc-playbook/ci-cd-integration-and-deployment)); and a mistake flagged twice becomes a CLAUDE.md correction. CCAR-F 3.6-K4 adds why the reviewer should be a separate instance: "the same Claude session that generated code is less effective at reviewing its own changes compared to an independent review instance" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Review prompts are in [Automated code review that engineers trust](claude-code-workflows.md#automated-code-review-that-engineers-trust).

### First-line support: symptom to cause (7.3)

The checks and fixes below come from Anthropic's docs; pairing them with symptoms in one table is ours.

| Symptom | Likely cause | First check or fix |
|---|---|---|
| "You haven't been added to your organization yet" | The seat does not include Claude Code | Update the seat in the admin console |
| A policy does not seem to apply | A different managed source is in effect | `/status`: the `Setting sources` line shows `Enterprise managed settings` and the source that won |
| The enterprise sign-in option is missing, or the wrong account is in use | Out-of-date client or wrong login | `claude update`; `/logout` then `/login` |
| A new team member does not get the team's instructions | They live in someone's `~/.claude/CLAUDE.md` | Move them to the project `CLAUDE.md`; `/context` shows which memory files loaded (the CCAR-F guide names `/memory` for this check) |
| Telemetry never arrives | The OpenTelemetry exporter settings | Run `claude --debug` and check the debug log |
| `curl` reaches the internet although WebFetch is denied | Bash is allowed; permission rules and the network are different layers | Sandboxing with a network domain allowlist |
| On a cloud provider, sessions start on a model you did not choose, show a fallback notice (Bedrock, Google Cloud's Agent Platform) or fail with errors (Foundry) | No pinned versions, so aliases resolve to Claude Code's built-in default for that provider, which can lag the newest release and may not yet be enabled in your account | Set the `ANTHROPIC_DEFAULT_*_MODEL` variables |
| A group's spend spikes | The most capable model as the group's default (on API or cloud-provider plans, also long uncleared sessions) | Check what the spend is producing and the default model before raising the cap |

The diagnostics a team should know are `/memory`, `/skills`, `/hooks`, `/mcp`, `/permissions` (resolved rules), `/doctor` (a setup checkup), `/debug` and `/status` ([Debug your configuration](https://code.claude.com/docs/en/debug-your-config)); from the terminal, `claude doctor` prints read-only installation diagnostics without starting a session. For self-sufficiency, apply the champion kit to support: answer publicly once and link back to that answer when the question recurs, collect the most common questions and answers in a pinned FAQ message, and identify a second champion to divide the channel responsibilities with ([Champion kit](https://code.claude.com/docs/en/champion-kit)).

On data handling: Anthropic does not train on code or prompts on Team, Enterprise, API or cloud provider plans, and Claude Code ZDR "cannot be enabled from your admin settings" ([Zero data retention](https://code.claude.com/docs/en/zero-data-retention)); it comes through Anthropic for qualified Enterprise accounts, with `forceLoginMethod` and `forceLoginOrgUUID` keeping sessions in the ZDR organization. See [Data retention, training and compliance](security-and-governance.md#data-retention-training-and-compliance).

### Decide

- If developers need Claude Code and claude.ai with no infrastructure to run, choose Teams or Enterprise; if you must inherit AWS, GCP or Azure controls and billing, choose that provider and decide whether developers also need claude.ai seats.
- If every developer must get a piece of configuration on clone, commit it (project `CLAUDE.md`, `.claude/commands/` or `.claude/skills/`, `.mcp.json`, `.claude/settings.json`); not `~/.claude/` or `~/.claude.json`, which are personal.
- If a rule must hold whatever developers configure, put it in managed settings; not in a project file any developer can edit.
- If you deploy on a cloud provider, pin model versions and use managed settings instead: `availableModels` for restrictions, `model` for a default and `maxEffortLevel` for an effort cap; Claude Enterprise's organization model and effort controls do not reach those sessions.
- If a group keeps hitting its cap, check what the spend is producing and the default model first; not an automatic increase.
- If adoption is flat, invest in enablement (champions, guided first tasks); not a settings change, since "The fix is usually enablement, not a settings change" ([Adoption signals](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/adoption-signals)).

### Traps

- **The HKCU registry as enforcement.** It is writable without elevation.
- **Expecting Anthropic's analytics to show Bedrock usage.** Claude Code does not send metrics from your cloud back to Anthropic; use OpenTelemetry, a gateway, or your cloud provider's billing console.
- **Letting AI review findings approve or block a PR.** Branch protection and a code owner's approval still decide.
- **Champions answering security questions.** Refer them to the administrator.
- **A team command in the wrong place.** CCAR-F Question 4's rationale rules out `~/.claude/commands/` ("personal commands that aren't shared via version control") and `CLAUDE.md` ("for project instructions and context, not command definitions").

## Exam map

*Tested in: all four exams, section by section (CCAO-F, CCDV-F, CCAR-F, CCAR-P)*

Which official objective each section of this page serves, taken from the four July 2026 exam guides. How to read the labels: the CCAO-F and CCAR-P guides list their objectives as unnumbered bullets and the CCDV-F guide lists its skills in order under each domain without numbers, so the numbers here are ours. CCAO-F objectives are numbered D1.1 to D7.3 in the order the guide lists them; CCDV-F numbers such as 2.3 are the skill's position within its domain; CCAR-P numbers such as 6.3 are the objective's position within its domain. CCAR-F numbers its own task statements (such as 3.1); the K (knowledge) and S (skill) suffixes are our numbering of the bullets under each task statement, in guide order, and EX, APPX and Q mark preparation exercise steps, appendix items and sample questions.

| Section | CCAO-F | CCDV-F | CCAR-F | CCAR-P |
|---|---|---|---|---|
| [Discovery and requirements](#discovery-and-requirements) | D4.1 | 2.1 | Not listed | 1.1, 6.1, 6.5 |
| [Choosing the surface](#choosing-the-surface) | D3.1 | 1.2, 2.5 | Scenarios 1 to 6; APPX-TECH-1, APPX-TECH-3 to APPX-TECH-6 | 1.1, 1.3, 7.1 |
| [Retrieval-augmented generation](#retrieval-augmented-generation) | D5.1 | 2.3 | Out of scope (APPX-OUTSCOPE-7) | 3.5, 3.6, 3.8, 4.4; Sample 3 (Domain 4) |
| [Integration patterns](#integration-patterns) | D4.4 | 2.3, 2.4, 5.2; Sample 1 (Domain 2) | 4.5-K1 to 4.5-K4, 4.5-S1 to 4.5-S4, 3.6-K1, 3.6-K2; MCP server hosting and streaming out of scope (APPX-OUTSCOPE-4, APPX-OUTSCOPE-10) | 1.2, 3.2, 3.7; Sample 1 (Domain 3) |
| [Deployment platforms and capacity](#deployment-platforms-and-capacity) | Not listed | 2.3 | Out of scope (APPX-OUTSCOPE-11, APPX-OUTSCOPE-13) | 1.6, 3.3, 6.3 |
| [Cost modeling](#cost-modeling) | D3.3; Sample 2 (Domain 3) | 5.4, 5.3 | 4.5-K1, 4.5-S1, 4.5-S4, Q11; API pricing calculations out of scope (APPX-OUTSCOPE-11) | 1.6, 2.5, 3.3, 4.5, 6.2; Sample 2 (Domain 2) |
| [Evaluation strategy for a program](#evaluation-strategy-for-a-program) | D2.1, D7.2 | 4.1, 2.6, 5.3 | 5.5-K1, 5.5-S2 | 4.1, 4.2, 4.3, 4.4, 4.6; Sample 3 (Domain 4) |
| [Governance and risk in delivery](#governance-and-risk-in-delivery) | D6.1 to D6.4, D2.4; Sample 3 (Domain 6) | 7.1, 7.2; Sample 2 (Domain 7) | 1.4-K1, 1.4-K2, Q1 | 5.1 to 5.5, 3.2, 6.4; Sample 1 (Domain 3) |
| [Stakeholder communication and lifecycle](#stakeholder-communication-and-lifecycle) | D4.5 | 2.2, 2.6 | Not listed | 6.2, 6.3, 6.4, 6.5, 1.6 |
| [Rolling out Claude Code to an engineering organization](#rolling-out-claude-code-to-an-engineering-organization) | Not listed | 3.1, 2.6 | 3.1-K1, 3.1-K2, 3.1-S1, 3.1-S4, 3.2-K1, 3.2-S1, 2.4-K1, 2.4-K2, 2.4-S1, 2.4-S2, 3.6-K4, EX2-STEP1, EX2-STEP4, Q4 | 7.1, 7.2, 7.3 |

Where this page carries the most weight (CCDV-F skill weights are shares of the whole exam, as that guide states them):

- **CCAR-P:** every objective in Domain 6, Stakeholder Communication & Lifecycle Management (14%), and Domain 7, Developer Productivity & Operational Enablement (7%), has a section here, and the page also covers Domain 1 (17%: 1.1, 1.2, 1.3, 1.6), Domain 3 (19%: 3.2, 3.3, 3.5 to 3.8), every objective of Domain 4 (16%: 4.1 to 4.6) and every objective of Domain 5 (14%: 5.1 to 5.5). Objectives taught mainly elsewhere: 1.4 and 1.5 on [Agents and the Claude Agent SDK](agents-and-agent-sdk.md#multi-agent-orchestration), Domain 2 (13%) on [Claude API Essentials](claude-api.md#models-and-how-to-choose-one), [Prompt Engineering](prompt-engineering.md#principles-that-decide-most-prompt-questions) and [Context Engineering](context-engineering.md#why-context-is-a-budget) (the caching side of 2.5 and Sample 2 are worked in [Cost modeling](#cost-modeling)), 3.1 in [Designing a tool set](tool-use-and-mcp.md#designing-a-tool-set), and 3.4 in [Reliability engineering for Claude applications](evaluation-and-reliability.md#reliability-engineering-for-claude-applications).
- **CCDV-F:** all of 2.1 Understanding Requirements (3.4%) and 2.2 Systems Life Cycle (2.8%), the vendor, data-access and realtime-versus-batch parts of 2.3 Claude API Mechanics (6.8%), the program side of 5.4 Cost and Token Management (2.8%), and the team-configuration side of 2.6 Configuration Management (4.1%) and 3.1 Claude Code Operation (3.1%).
- **CCAR-F:** little. The appendix puts embedding models or vector database implementation details, rate limiting, quotas or API pricing calculations, and specific cloud provider configurations out of scope. What remains here is team configuration of Claude Code (Domain 3, 20%: 3.1 and 3.2, plus 2.4 from Domain 2, and Exercise 2), Claude Code in CI/CD and independent review (3.6-K1, 3.6-K2, 3.6-K4), the batch trade-off in 4.5 and Question 11, deterministic enforcement in 1.4 and Question 1, and segment-level accuracy before reducing human review in 5.5.
- **CCAO-F:** the user's side only: D4.1, D4.4 and D4.5 from Domain 4 (16%), D3.1 and D3.3, D5.1, D2.1, D2.4 and D7.2, and Domain 6 (15%). The guide leaves enterprise-scale architecture and integration design to the Architect and Developer credentials (its wording is quoted at the top of this page).

??? info "Sources"

    - [Claude Certified Architect, Professional exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf): CCAR-P domains, weights and objectives 1.1 to 7.3, the preparation advice to build an end-to-end solution with RAG, and Samples 1, 2 and 3 with their rationales
    - [Claude Certified Developer, Foundations exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf): CCDV-F skills 1.2, 2.1 to 2.6, 3.1, 4.1, 5.2 to 5.4, 7.1 and 7.2 with their weights, and Samples 1 and 2 with their rationales
    - [Claude Certified Associate, Foundations exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf): CCAO-F objectives D2.1, D2.4, D3.1, D3.3, D4.1, D4.4, D4.5, D5.1, D6.1 to D6.4 and D7.2, Samples 2 and 3, and the escalation boundary for enterprise-scale architecture
    - [Claude Certified Architect, Foundations exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): the six scenarios, task statements 1.4, 2.4, 3.1, 3.2, 3.6, 4.5 and 5.5, Exercise 2, Questions 1, 4 and 11, the technologies list and the out-of-scope appendix
    - [Claude Certified Architect, Professional prep course](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional): learning objectives on scoping, entry points, cost, latency and reliability budgets, evaluations as acceptance criteria, the fail-closed safety stack and the handoff, and module lengths
    - [Claude Platform & Solution Design module](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/claude-platform-solution-design): ruling out options by governance and regulated-industry constraints first, the Claude/systems/humans work split, and retrieval versus live state
    - [Enterprise Integration & Production module](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/enterprise-integration-production): call volume, token and cost estimates, A/B test planning, failure modes with mitigations, reliability patterns and the architecture specification
    - [Stakeholder Engagement, Lifecycle & GTM module](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/stakeholder-engagement-lifecycle-gtm): requirements with documented assumptions, comparing delivery routes on latency, compliance and cost, presenting trade-offs with cost, risk and reversal, and the stakeholder feedback loop
    - [Claude Certified Architect, Professional certification page](https://anthropic-partners.skilljar.com/claude-certified-architect-professional-certification): the catalog prep courses, including Claude with Amazon Bedrock and Claude on Google Cloud
    - [Claude Certified Associate, Foundations prep course](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations): entry point as the first of four decisions before prompting
    - [Deploying AI from pilot to production (Anthropic and Accenture)](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf): use-case definition, baselines, pre-pilot infrastructure questions, data readiness and classification, the Novo Nordisk example, retrieval versus context windows, simplest-architecture guidance, the TCO model, stakeholder success criteria, tiered oversight, incident questions, the go/no-go process, ownership questions, and program versus project
    - [Building AI agents for the enterprise](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/69f3af1f0b8ebe5cde42fcda_Claude-Building-AI-Agents-in-the-Enterpise-04302026_v2%20%281%29.pdf): concrete success criteria versus vague goals, and weekly adoption measurement during a pilot
    - [Building trusted AI in the enterprise](https://www-cdn.anthropic.com/e5c9de22bc8884089970bd262ca0c8b952cb9136.pdf): first use-case traits, parallel deployment, graduation criteria, A/B testing infrastructure, monitoring scope, bias metrics, AI review boards, prompt versioning, and feeding production data back into evaluations
    - [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval): chunking, BM25, rank fusion, the contextualizer prompt, the 35%, 49% and 67% failure reductions and their test conditions, the cost estimate and the 200,000-token guidance
    - [Contextual embeddings cookbook](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/capabilities/contextual-embeddings/guide.ipynb): the current contextualization code with prompt caching, Pass@k results, hybrid fusion weights and reranking latency
    - [Retrieval-augmented generation cookbook](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/capabilities/retrieval_augmented_generation/guide.ipynb): heading-based chunking, the index loader, retrieval metrics, the evaluation set and the MRR and accuracy results
    - [Text-to-SQL cookbook](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/capabilities/text_to_sql/guide.ipynb): schema in the prompt, schema retrieval, self-improvement loop and schema index updates
    - [Embeddings](https://platform.claude.com/docs/en/build-with-claude/embeddings): no first-party embedding model, the Voyage model table, `input_type`, normalization, quantization, Matryoshka truncation and the rerankers Anthropic lists
    - [Search results](https://platform.claude.com/docs/en/build-with-claude/search-results): `search_result` blocks, no beta header, tool-result and top-level delivery
    - [Citations](https://platform.claude.com/docs/en/build-with-claude/citations): chunking of plain-text and custom content documents, and `cited_text` billing
    - [Reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations): extracting quotes first from documents over 20k tokens
    - [Mitigate jailbreaks and prompt injections](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks): input screening with Claude Haiku 4.5, third-party content in `tool_result` blocks, JSON encoding, tool-output screening, red-teaming and least privilege
    - [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): context rot, the smallest high-signal context, just-in-time retrieval, hybrid strategies and stale indexes
    - [Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk): starting with agentic search before semantic search
    - [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system): breadth-first queries, starting searches short and broad, and rainbow deployments
    - [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): single optimized calls, the latency and cost trade-off of agents, starting with the API before frameworks, compounding errors, stopping conditions and visible planning steps
    - [Building agents that reach production systems with MCP](https://claude.com/blog/building-agents-that-reach-production-systems-with-mcp): direct API, CLI and MCP compared, the M×N problem, remote MCP for cloud agents, intent-based tools
    - [Intro to Claude](https://platform.claude.com/docs/en/intro): the Messages API and Managed Agents as the two ways to build, and API key hygiene
    - [Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview): the four ways to build compared, the CLI as a subprocess, and the claude.ai login restriction
    - [Hosting the Agent SDK](https://code.claude.com/docs/en/agent-sdk/hosting): subprocess and on-disk sessions, `SessionStore`, HTTP or WebSocket endpoints, fan-out limits and gateway authentication
    - [Securely deploying AI agents](https://code.claude.com/docs/en/agent-sdk/secure-deployment): credential injection outside the agent
    - [Modifying system prompts in the Agent SDK](https://code.claude.com/docs/en/agent-sdk/modifying-system-prompts): the minimal default prompt versus the `claude_code` preset
    - [Agent SDK quickstart](https://code.claude.com/docs/en/agent-sdk/quickstart): third-party provider environment variables
    - [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview): best-fit workloads, the beta header, sandbox options, server-side event history and ZDR and BAA ineligibility
    - [Managed Agents agent setup](https://platform.claude.com/docs/en/managed-agents/agent-setup): agents as versioned configurations
    - [Managed Agents multiagent orchestration](https://platform.claude.com/docs/en/managed-agents/multiagent): coordinating several agents in one session
    - [Managed Agents webhooks](https://platform.claude.com/docs/en/managed-agents/webhooks) and [scheduled deployments](https://platform.claude.com/docs/en/managed-agents/scheduled-deployments): event delivery and recurring sessions
    - [Tool runner](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner): when to use the manual loop instead
    - [Tool use overview](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview) and [How tool use works](https://platform.claude.com/docs/en/agents-and-tools/tool-use/how-tool-use-works): client versus server tools and the client-tool round trip
    - [MCP connector](https://platform.claude.com/docs/en/agents-and-tools/mcp-connector): beta header, public HTTP servers only, tool calls only, platform availability and ZDR eligibility
    - [MCP tunnels](https://platform.claude.com/docs/en/agents-and-tools/mcp-tunnels/overview): outbound-only connections to private MCP servers, research-preview status with no uptime commitment, and use with Managed Agents and the MCP connector
    - [Message Batches](https://platform.claude.com/docs/en/build-with-claude/batch-processing): the 50% discount, limits, completion times, expiry, result order, batch cache hit rates, server tools inside batches and `pause_turn`
    - [Rate limits](https://platform.claude.com/docs/en/api/rate-limits): tiers, spend caps and spend-limit errors, RPM, ITPM and OTPM, cache-aware ITPM, per-model limits, bursts, acceleration limits, workspace limits, batch limits, limits as maximums, and Claude Platform on AWS limits
    - [Service tiers](https://platform.claude.com/docs/en/api/service-tiers): Priority, Standard and Batch tiers, best-effort availability, Priority Tier targets and the end of its sales, and the `service_tier` parameter
    - [Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback): fallback does not absorb rate limits or overloads
    - [Errors](https://platform.claude.com/docs/en/api/errors): the 529 overloaded error and SDK retries with exponential backoff
    - [Workspaces](https://platform.claude.com/docs/en/manage-claude/workspaces): key scoping, the Claude Code workspace, workspace limits and environment separation
    - [Data residency](https://platform.claude.com/docs/en/manage-claude/data-residency): `inference_geo` values, workspace geo, the 1.1x multiplier and model support
    - [API and data retention](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention): ZDR scope, Covered Models, flagged-content retention, HIPAA readiness and its exclusions, CORS under ZDR, and the cloud provider as data processor
    - [Workload Identity Federation](https://platform.claude.com/docs/en/manage-claude/workload-identity-federation): short-lived OIDC tokens instead of API keys
    - [Claude in Amazon Bedrock](https://platform.claude.com/docs/en/build-with-claude/claude-in-amazon-bedrock): operator access, unsupported features, regional endpoints and inference profiles, quotas and logging
    - [Claude on Google Cloud's Agent Platform](https://platform.claude.com/docs/en/build-with-claude/claude-on-vertex-ai): request format, endpoint types and premiums, provisioned throughput, unsupported features and payload limits
    - [Claude in Microsoft Foundry](https://platform.claude.com/docs/en/build-with-claude/claude-in-microsoft-foundry): hosting options, deployment types, billing, unsupported features and rate-limit management
    - [Claude Platform on AWS](https://platform.claude.com/docs/en/build-with-claude/claude-platform-on-aws): Anthropic operation, the separate capacity pool, failover and the regulated-industry redirect to Bedrock
    - [API overview](https://platform.claude.com/docs/en/api/overview): endpoints, the REST base URL and Anthropic-operated versus partner-operated platforms
    - [Pricing](https://platform.claude.com/docs/en/about-claude/pricing): model and cache prices, cache break-even, long-context billing, fast mode, US-only inference, server tool and Managed Agents charges, Claude Consumption Units, and the Sonnet 5 price note
    - [Prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching): cache isolation by platform, minimum cacheable length and cache availability for parallel requests
    - [Model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations): lifecycle states, 60-day notice, retirement schedules by platform, example dates, usage export, Claude 3 Haiku's retirement, Claude Haiku 4.5's status and sampling-parameter errors
    - [Model IDs and versions](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions): model ID formats on each platform, pinned and dateless IDs, and infrastructure-caused behavior changes
    - [Models overview](https://platform.claude.com/docs/en/models/overview): context windows of the current models
    - [Choosing a model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model): efficiency-first and capability-first strategies, effort as a lever, and executor-advisor and orchestrator-worker patterns
    - [Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence): free wins and trade-offs, cost per completed task, pricing the tail, cache-hit benchmarks, effort sweeps, multi-model strategies, shadow runs, and routing non-urgent work through batches
    - [Fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode): output speed and platform availability
    - [Prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview): model changes as a latency and cost lever
    - [Define success criteria](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests): specific, measurable criteria, latency, price and safety examples, multidimensional criteria, and A/B testing against a baseline
    - [Reduce latency](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-latency): time to first token
    - [Glossary](https://platform.claude.com/docs/en/about-claude/glossary): the definitions of RAG and time to first token
    - [Streaming](https://platform.claude.com/docs/en/build-with-claude/streaming) and [Working with messages](https://platform.claude.com/docs/en/build-with-claude/working-with-messages): server-sent events and the stateless API
    - [Files API](https://platform.claude.com/docs/en/build-with-claude/files): `file_id` handling and workspace-per-tenant isolation
    - [Python SDK](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python) and [TypeScript SDK](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/typescript): async clients, retries, timeouts, platform clients, browser use and `Anthropic.Message`
    - [OpenAI SDK compatibility](https://platform.claude.com/docs/en/cli-sdks-libraries/libraries/openai-sdk): intended for testing, `strict` ignored, no prompt caching
    - [Release notes](https://platform.claude.com/docs/en/release-notes/overview): the June 2026 usage-tier consolidation, earlier numbered tiers, and the Claude Opus 5.5 launch after the exam guides
    - [System prompt release notes](https://platform.claude.com/docs/en/release-notes/system-prompts/overview): claude.ai system prompts do not apply to the API
    - [Set up Claude Code for your organization](https://code.claude.com/docs/en/admin-setup): the ordered deployment decisions, provider choice, features that need claude.ai accounts, managed settings delivery, merging and reach, enforcement controls, gateways, verification with `/status`, login fixes and seat errors
    - [Claude Code third-party integrations](https://code.claude.com/docs/en/third-party-integrations): Teams versus Enterprise, one-click install, guided usage, model pinning on cloud providers, a central team for `.mcp.json`, and the Agent Platform naming
    - [Claude Code feature availability](https://code.claude.com/docs/en/feature-availability), [costs](https://code.claude.com/docs/en/costs) and [LLM gateway](https://code.claude.com/docs/en/llm-gateway): the CLI working on every provider; spend controls by setup, the per-seat allowance, pilot baselines, usual causes of spend, agent teams, metrics on cloud providers and per-user rate-limit guidance; what a gateway centralizes, gateway trade-offs and per-developer credentials
    - [Claude Code on the web](https://code.claude.com/docs/en/claude-code-on-the-web): cloud sessions in Anthropic-managed VMs
    - [Claude Code GitHub Actions](https://code.claude.com/docs/en/github-actions): `@claude` in pull requests and the resources each run consumes
    - [Claude Code data usage](https://code.claude.com/docs/en/data-usage) and [memory](https://code.claude.com/docs/en/memory): local transcript retention, how CLAUDE.md is delivered, and `/init`
    - [Claude Code MCP](https://code.claude.com/docs/en/mcp), [best practices](https://code.claude.com/docs/en/best-practices) and [large codebases](https://code.claude.com/docs/en/large-codebases): WebSocket MCP servers, CLI tools as the context-efficient route, treating CLAUDE.md like code, and exposing an existing index as an MCP tool
    - [What are Projects?](https://support.claude.com/en/articles/9517075-what-are-projects), [How can I create and manage projects?](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects) and [RAG for Projects](https://support.claude.com/en/articles/11473015-retrieval-augmented-generation-rag-for-projects): Project knowledge, instructions, memory and automatic RAG mode
    - [Get started with Claude Cowork](https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork), [Cowork on Team and Enterprise](https://support.claude.com/en/articles/13455879-use-claude-cowork-on-team-and-enterprise-plans) and [Install Claude Desktop](https://support.claude.com/en/articles/10065433-install-claude-desktop): Cowork plans, local and cloud sessions and defaults
    - [Cowork is now Claude](https://claude.com/blog/cowork-is-now-claude) and [Claude Cowork and chat are one Claude](https://support.claude.com/en/articles/16761823-claude-cowork-and-chat-are-one-claude): the September 2026 merge and its rollout
    - [Use connectors](https://support.claude.com/en/articles/11176164-use-connectors-to-extend-claude-s-capabilities) and [custom connectors using remote MCP](https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp): enabling connectors, enterprise-managed auth, and connecting from Anthropic's cloud
    - [Understanding Claude's personalization features](https://support.claude.com/en/articles/10185728-understanding-claude-s-personalization-features) and [Set organization instructions](https://support.claude.com/en/articles/14546867-set-organization-instructions): instruction layers in the apps
    - [Deploying Claude Enterprise with confidence: Connectors](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/connectors): inherited permissions, the three gates, write-access sign-off and enterprise-managed authorization
    - [Claude 101: desktop app](https://academy.claude.com/courses/claude-101/claude-desktop-app-chat-cowork-code) and [Introduction to Projects](https://academy.claude.com/courses/claude-101/introduction-to-projects): chat, Cowork and Code, and keeping project knowledge current
    - [Introduction to Claude Cowork: scheduled tasks](https://academy.claude.com/courses/introduction-to-claude-cowork/scheduled-tasks): tasks that suit Cowork and `/schedule`
    - [Prototype AI-powered apps with Claude artifacts](https://academy.claude.com/tutorials/prototype-ai-powered-apps-with-claude-artifacts): artifacts for prototypes, not production
    - [AI Fluency for nonprofits: workflow augmentation](https://academy.claude.com/courses/ai-fluency-for-nonprofits/workflow-augmentation) and [AI Fluency for small businesses](https://academy.claude.com/courses/ai-fluency-for-small-businesses/tying-it-all-together): workload audits, delegation categories and testing with real examples
    - [AI-native SDLC playbook: capture intent](https://academy.claude.com/courses/ai-native-sdlc-playbook/capture-intent) and [requirements and design](https://academy.claude.com/courses/ai-native-sdlc-playbook/requirements-and-design): running discovery with Claude
    - [PRD from a one-pager](https://academy.claude.com/use-cases/prd-from-a-one-pager) and [Analyze patterns in user feedback](https://academy.claude.com/use-cases/analyze-patterns-in-user-feedback): goals and non-goals, checking guesses, and synthesizing stakeholder input
    - [Estimating productivity gains](https://www.anthropic.com/research/estimating-productivity-gains): accelerated tasks exposing new bottlenecks
    - [Anthropic Usage Policy](https://www.anthropic.com/legal/aup): high-risk use case requirements, professional review, AI disclosure and agentic use
    - [Data Processing Addendum](https://www.anthropic.com/legal/data-processing-addendum) and [How do I view and sign the DPA?](https://privacy.claude.com/en/articles/7996862-how-do-i-view-and-sign-your-data-processing-addendum-dpa): controller and processor roles, the DPA with Standard Contractual Clauses, and third-party platform terms
    - [Regional compliance](https://claude.com/regional-compliance): European deployment options through cloud platforms
    - [Deploying multi-agent systems using MCP and A2A with Claude on Vertex AI (webinar)](https://www.anthropic.com/webinars/deploying-multi-agent-systems-using-mcp-and-a2a-with-claude-on-vertex-ai): Anthropic and Google Cloud on MCP and A2A
    - [MCP security best practices](https://modelcontextprotocol.io/docs/2026-07-28/tutorials/security/security_best_practices), [Enterprise-Managed Authorization](https://modelcontextprotocol.io/extensions/auth/enterprise-managed-authorization) and [MCP resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources): token passthrough, scopes, identity-provider control and resources for reference data
    - [Voyage AI pricing](https://docs.voyageai.com/docs/pricing), [reranker](https://docs.voyageai.com/docs/reranker), [contextualized chunk embeddings](https://docs.voyageai.com/docs/contextualized-chunk-embeddings), [embeddings](https://docs.voyageai.com/docs/embeddings) and [Voyage 4 announcement](https://blog.voyageai.com/2026/01/15/voyage-4/): prices and free tokens, rerank limits, auto-chunking parameters and embedding compatibility
    - [A2A specification](https://raw.githubusercontent.com/a2aproject/A2A/main/docs/specification.md), [README](https://raw.githubusercontent.com/a2aproject/A2A/main/README.md), [Life of a task](https://raw.githubusercontent.com/a2aproject/A2A/main/docs/topics/life-of-a-task.md) and [A2A and MCP](https://raw.githubusercontent.com/a2aproject/A2A/main/docs/topics/a2a-and-mcp.md): Agent Cards, tasks and states, the request example, security rules, governance and the split with MCP
    - [Claude at scale on Google Cloud](https://cloud.google.com/blog/products/ai-machine-learning/claude-at-scale-on-google-cloud-frontier-ai-built-for-enterprise-production) and [ADK: Anthropic models](https://google.github.io/adk-docs/agents/models/anthropic/): Claude-powered agents over A2A on Google Cloud
    - [A2A support in Amazon Bedrock AgentCore Runtime](https://aws.amazon.com/blogs/machine-learning/introducing-agent-to-agent-protocol-support-in-amazon-bedrock-agentcore-runtime/): Claude-powered agents interoperating over A2A on AWS
    - [SWEBOK v3, Software Requirements](http://swebokwiki.org/Chapter_1:_Software_Requirements): functional and non-functional requirements
    - [NIST glossary: security requirement](https://csrc.nist.gov/glossary/term/security_requirement): where security requirements come from
    - [RFC 6455, The WebSocket Protocol](https://www.rfc-editor.org/rfc/rfc6455.html) and [MDN: Using server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events): full-duplex WebSockets versus one-way SSE
    - [Python asyncio synchronization primitives](https://docs.python.org/3/library/asyncio-sync.html), [asyncio tasks](https://docs.python.org/3/library/asyncio-task.html) and [MDN: Promise.all()](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise/all): the semaphore, `gather` and `Promise.all` behavior behind the concurrency example
    - [Prep course: Responsible AI, Safety & Risk for Architects](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/responsible-ai-safety-risk-for-architects): mapping obligations to control, owner and evidence; fairness and transparency; training versus application-layer enforcement
    - [Prep course: Team Enablement & Operational Productivity](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/developer-productivity-enablement): the three learning objectives mapped to CCAR-P 7.1 to 7.3
    - [Token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting): the newer tokenizer producing about 30% more tokens
    - [Claude Opus 5.5 migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide): `/claude-api migrate`, re-baselining cost and latency, the effort default, refusal handling
    - [Skills for enterprise](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise): version pinning, promotion gates and the Skills registry
    - [Inference hooks](https://platform.claude.com/docs/en/manage-claude/inference-hooks): prompt screening for Claude Enterprise and its current limits
    - [Run Claude Code through a gateway](https://code.claude.com/docs/en/gateways): the Claude apps gateway and its missing service-token flow for CI
    - [Monitoring usage](https://code.claude.com/docs/en/monitoring-usage): organization-wide OpenTelemetry through managed settings and `claude --debug`
    - [Managed settings](https://code.claude.com/docs/en/managed-settings): the `/status` label for managed sources
    - [Settings](https://code.claude.com/docs/en/settings): the `Setting sources` line
    - [Debug your configuration](https://code.claude.com/docs/en/debug-your-config): the diagnostic commands
    - [Commands](https://code.claude.com/docs/en/commands): `/team-onboarding` and `/doctor`
    - [Model configuration](https://code.claude.com/docs/en/model-config): pinning models before a cloud-provider rollout
    - [Claude Code on Amazon Bedrock](https://code.claude.com/docs/en/amazon-bedrock): the `ANTHROPIC_DEFAULT_*_MODEL` pinning example
    - [Skills](https://code.claude.com/docs/en/skills): custom commands merged into skills
    - [Zero data retention for Claude Code](https://code.claude.com/docs/en/zero-data-retention): how Claude Code ZDR is enabled and which sessions it covers
    - [Deep links](https://code.claude.com/docs/en/deep-links): `claude-cli://` links in runbooks
    - [What's new, 2026 week 15](https://code.claude.com/docs/en/whats-new/2026-w15): the `/team-onboarding` command
    - [Champion kit](https://code.claude.com/docs/en/champion-kit): the champion role, the 30-day playbook and handoff signal, referring security questions to administrators
    - [Communications kit](https://code.claude.com/docs/en/communications-kit): the pre-launch checklist, 48-hour channel owner, executive-sent launches
    - [What is the Enterprise plan?](https://support.claude.com/en/articles/9797531-what-is-the-enterprise-plan): usage-based Enterprise billing and the HIPAA-ready configuration
    - [Claude Enterprise consumption guide](https://support.claude.com/en/articles/14782391-claude-enterprise-consumption-guide): per-skill ROI, waves, group and per-user limits, investigating before raising limits
    - [Claude Enterprise Administrator Guide](https://claude.com/resources/tutorials/claude-enterprise-administrator-guide): seat types, default spend limits, success metrics and reporting cadence, retrospectives
    - [Public sector FAQs](https://support.claude.com/en/articles/13756069-public-sector-faqs): FedRAMP High options, ITAR, the BAA and ZDR statement
    - [Business Associate Agreements for commercial customers](https://privacy.claude.com/en/articles/8114513-business-associate-agreements-baa-for-commercial-customers): BAA scope and exclusions, Claude Code under ZDR
    - [What certifications has Anthropic obtained?](https://privacy.claude.com/en/articles/10015870-what-certifications-has-anthropic-obtained): ISO/IEC 42001:2023
    - [Access the Compliance API](https://support.claude.com/en/articles/13015708-access-the-compliance-api): availability and who can enable it
    - [Access audit logs](https://support.claude.com/en/articles/9970975-access-audit-logs): 180 days of audit events
    - [How to get support](https://support.claude.com/en/articles/9015913-how-to-get-support): asynchronous support with response times that vary by plan and severity
    - [Anthropic Status](https://status.claude.com/): tracked components and subscription channels
    - [Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms): services provided as is and as available
    - [Deploying Claude Enterprise with confidence](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence), with the lessons [Five decisions and the frame](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/five-decisions-and-the-frame), [One organization or many](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/one-organization-or-many), [Owners and intake](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/owners-and-intake), [Surfaces each group gets](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/surfaces-each-group-gets), [Managing spend](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/managing-spend), [Visibility: what you can measure](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/visibility-what-you-can-measure) and [Adoption signals](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/adoption-signals): the five decisions and their order, hard-to-undo settings, owners, connectors, spend caps, the default model as a spend lever, visibility before access, adoption signals
    - [AI-native SDLC playbook](https://academy.claude.com/courses/ai-native-sdlc-playbook/introduction), with the lessons [Continuous evals in CI](https://academy.claude.com/courses/ai-native-sdlc-playbook/continuous-evals-in-ci), [AI in the PR review loop](https://academy.claude.com/courses/ai-native-sdlc-playbook/ai-in-the-pr-review-loop) and [CI/CD integration and deployment](https://academy.claude.com/courses/ai-native-sdlc-playbook/ci-cd-integration-and-deployment): the six stages as a loop, committed artifacts as the audit trail, continuous evals in CI, the PR review loop, CI/CD integration, CLAUDE.md in Git, skills as versioned policy, hooks as approval gates
    - [Why do AI models hallucinate?](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate): hallucination described as an unsolved problem
    - [Reducing cost and improving performance with Claude Platform](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform): the customer-support hillclimb result, `/claude-api prompt-audit`, `cost-optimize` and `hillclimb`, effort at the top end
    - [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents): layered evaluation, A/B test strengths and limits, baselines from evals, CI gating, `pass^k`
    - [A statistical approach to model evaluations](https://www.anthropic.com/research/statistical-approach-to-model-evals): confidence intervals, power analysis, small-eval sensitivity
    - [A postmortem of three recent issues](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues): noisy evaluations and continuous production evals
    - [Building multi-agent systems: when and how to use them](https://claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them): decomposing by context boundaries
    - [Evaluating and mitigating discrimination in language model decisions](https://www.anthropic.com/research/evaluating-and-mitigating-discrimination-in-language-model-decisions): the 70-scenario perturbation method
    - [Evaluating and mitigating discrimination in language model decisions (paper)](https://arxiv.org/html/2312.03689): effective prompt interventions and the caution on high-risk use
    - [discrim-eval dataset](https://huggingface.co/datasets/Anthropic/discrim-eval) and its [dataset card](https://huggingface.co/datasets/Anthropic/discrim-eval/raw/main/README.md): demographic variations and 135 examples per scenario
    - [Voyage AI service level objectives](https://docs.voyageai.com/docs/service-level-objectives): a 99.5% objective, not an SLA
    - [Amazon Bedrock Service Level Agreement](https://aws.amazon.com/bedrock/sla/): the 99.9% monthly uptime commitment with service credits
    - [NIST glossary: system development life cycle](https://csrc.nist.gov/glossary/term/system_development_life_cycle): the SDLC definition
