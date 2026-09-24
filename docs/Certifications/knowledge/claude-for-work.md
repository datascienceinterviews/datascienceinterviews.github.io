---
title: "Claude Apps for Work: Associate Exam Knowledge Base"
description: "Claude apps for the Associate exam: plans, models, Projects, artifacts, connectors, skills, memory, Chrome, Cowork, admin controls, AI Fluency and verification."
last_reviewed: 2026-09-23
---

# Claude Apps for Work

This page teaches the Claude apps (web, desktop and mobile) the way a working professional uses them: which plan includes what, how to pick a model, how to configure a Project, when to ask for an artifact, how search, Research and connectors bring outside information in, how skills, memory, Claude in Chrome, Cowork and admin controls fit around them, and how the AI Fluency framework and a verification routine help you judge what comes back. It is the main reference for the Claude Certified Associate, Foundations exam (CCAO-F), whose [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) says the certification "is intended for professionals who use Claude as a productivity tool and build Claude Projects in their day-to-day roles", and it gives product context for the other three exams where they touch the Claude apps. Product facts are as of September 2026; where the product has moved since the July 2026 exam guides, the page shows both versions and says which wording to expect on the exam.

## Plans and what they include

*Tested in: CCAO-F 3.1, 6.2, 7.3 · CCAR-P 7.1*

A plan decides three things that show up in exam scenarios: which features exist for the person in the question (Opus, Research, Claude Design), how much work they can do before a limit stops them, and which data terms cover what they type. The CCAO-F objectives never name a plan or a price, so treat the figures below as context for feature and data-handling questions rather than numbers to memorize. Prices and limits are as of September 2026.

### Individual plans: Free, Pro and Max

| | Free | Pro | Max 5x | Max 20x |
|---|---|---|---|---|
| Price (US) | Free | &#36;20 per month, or &#36;17 per month on the annual plan (&#36;200 billed up front) | &#36;100 per month | &#36;200 per month |
| Billing | None | Monthly or annual | Monthly only | Monthly only |
| Session usage | Session limit that resets every five hours | Session limit that resets every five hours (at least 5x Free's usage per session), plus a weekly limit across all models | 5x the Pro per-session allowance, plus a weekly limit | 20x the Pro per-session allowance, plus a weekly limit |
| Haiku | Yes | Yes | Yes | Yes |
| Sonnet | Yes | Yes | Yes | Yes |
| Opus | No | Yes | Yes | Yes |
| Fable models | Not available | Not in plan limits; runs on usage credits from the first message | Up to 50% of the weekly limit at no extra cost, drawn from that limit rather than added to it | Up to 50% of the weekly limit at no extra cost, drawn from that limit rather than added to it |
| Research | No | Yes | Yes | Yes |
| Projects | Up to 5 | Yes | Yes | Yes |
| Custom connectors | One | Yes | Yes | Yes |
| Model training on your chats | Opt-out | Opt-out | Opt-out | Opt-out |
| Usage credits after a limit | No (paid plans only) | Yes | Yes | Yes |

What the Free plan already includes, per the pricing page: chat on web, desktop and mobile; web search, file creation and running code; memory across conversations; connecting apps and tools; and creating artifacts. Pro adds, among other things, handing off and scheduling tasks, Claude Design, Slides and Docs, and Claude in Chrome and Microsoft 365. Max adds higher limits and "priority access to our newest features and models" ([Max plan help article](https://support.claude.com/en/articles/11049741-what-is-the-max-plan)). Pro does not include API usage through the Claude Console; API usage needs its own Console access and is paid for separately. Everyone using Claude must be at least 18.

### Organization plans: Team and Enterprise

| | Team | Enterprise |
|---|---|---|
| Seats | Minimum 2, up to 150 | Minimum 20 (self-serve) or 50 (sales-assisted) |
| Price (US) | Standard seat: &#36;25 per member per month billed monthly, &#36;20 billed annually. Premium seat: &#36;125 monthly, &#36;100 annually | Pricing page: US&#36;20 per seat per month, billed annually; the seat fee covers access only and all usage is billed separately at API rates |
| Usage limits | Per member, not pooled. Standard: 1.25x the Pro per-session allowance. Premium: 6.25x. Both have a weekly limit across all models | Usage-based plans have no plan-level or seat-level limits; admins set spend limits for the organization and for individual users |
| Workplace connectors | Google Drive, Gmail, Google Calendar, GitHub, Microsoft 365 and Slack | Everything in Team |
| Enterprise search | Yes | Yes |
| Organization-wide skills deployment | Yes | Yes |
| Organization instructions | Yes | Yes |
| Role-based access (pricing table) | No | Yes |
| Audit logs (pricing table) | No | Yes |
| Also listed for Enterprise | | SCIM, custom data retention, the Compliance API, customer-managed encryption keys, US-only inference |
| HIPAA-ready configuration | Cannot be enabled | Enterprise only; the Business Associate Agreement is click-to-accept |
| Training on your content | Not used by default | Not used by default (commercial terms) |

Three details are easy to miss. Whoever creates a Team account must use a business email address. Two older Enterprise billing models are ending: seat-based Enterprise plans with Standard and Premium seats, and usage-based Enterprise plans with Chat and Chat + Claude Code seats, cannot continue on that billing model past the organization's next contract renewal. HIPAA-ready Enterprise organizations are the exception: they are provisioned with Chat and Chat + Claude Code seat types and are not eligible for the single Enterprise seat. And enabling HIPAA readiness does not bring every feature under Anthropic's BAA: Cowork is not yet covered, and Claude Code is covered only with zero data retention (ZDR) enabled, on qualified accounts. The day-to-day use of these controls (roles, SCIM, audit logs, retention) is covered in [Admin controls for Team and Enterprise](#admin-controls-for-team-and-enterprise); the data terms themselves are in [Data retention, training and compliance](security-and-governance.md#data-retention-training-and-compliance).

For an engineering rollout (CCAR-P 7.1), the [Claude Code administration docs](https://code.claude.com/docs/en/admin-setup.md) make Claude for Teams or Enterprise the default recommendation when you want "Claude Code and claude.ai under one per-seat subscription with no infrastructure to run". Some Claude Code features (cloud sessions, Routines, Code Review, Remote Control, the Chrome extension) need a claude.ai account and are not available with Console API keys or cloud-provider credentials alone. See [Rolling out Claude Code to an engineering organization](solution-architecture.md#rolling-out-claude-code-to-an-engineering-organization).

!!! note "Sources disagree on two plan details"

    - **Priority access on Pro.** The [Pro help article](https://support.claude.com/en/articles/8325606-what-is-the-pro-plan) lists "Priority access to Claude during high-traffic periods." The [pricing page](https://claude.com/pricing) comparison marks "Priority access at high traffic times" as No for Free and Pro and Yes for both Max tiers.
    - **Role-based access on Team.** The [Team help article](https://support.claude.com/en/articles/9266767-what-is-the-team-plan) lists "Role-based permissioning", but the pricing table marks role-based access as No for Team and Yes for Enterprise, and custom roles are documented as Enterprise-only.

    The four exam guides do not mention either detail. For a purchasing decision, confirm against the current pricing page.

### Usage limits and length limits

Two different limits stop a conversation, and the fix depends on which one you hit. Usage limits "control how much you can use Claude across all your conversations", while length limits "control how long any single conversation can become" ([usage and length limits](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work)). Usage on claude.ai, Claude Code and Claude Desktop all draws on the same usage limit. Your next reset time is shown in Settings > Usage. No plan publishes a fixed message count: how far an allowance goes depends on the length and complexity of your conversations, the model you choose and the features you use.

| You hit | What it means | What to do |
|---|---|---|
| A length limit | One conversation has grown too long for the context window, Claude's working memory for a single chat | Start a new conversation, attach fewer or smaller files, or move the material into a project |
| A usage limit | You have used your allowance for the session or week | Wait for the reset, upgrade the plan, or use usage credits. Pro and Max subscribers enable them in Settings > Usage; they are billed at standard API rates, can be capped with a monthly spend limit and have a daily redemption limit of &#36;2000. On Team and seat-based Enterprise plans an Owner or Primary Owner purchases them and configures members' accounts to use them |
| A limit reset is available (given occasionally to eligible plans) | Depending on the reset shown, it restores either the five-hour session limit or the weekly limit to full right away; weekly limits still reset on their usual day and time | Use it deliberately: once used, it cannot be undone. Claim it with "Reset for free" in Settings > Usage on the web or in Claude Desktop |

What uses your allowance faster, according to the help center:

- Tool use such as Research and web search, and multi-step tasks such as running code, creating files or browsing websites.
- Higher effort levels, which take longer and use more tokens.
- Long conversations that trigger automatic context management.
- Creating files, which uses more of your limit than a normal chat; artifacts also count toward your plan's usage limits.
- On Free accounts, web fetch of a long article (the whole article is pulled into the context window).

What stretches it:

- Group related tasks or questions into a single message, and send whole texts for editing in one message rather than in pieces.
- Keep reusable material in a project, where cached content counts less when reused (the cache caveat is under [Project knowledge](#project-knowledge)).
- Turn off tools and connectors a conversation does not need; they are token-intensive.
- Use Low or Medium effort for routine work, and turn thinking off when a task does not need it, on models that allow it (see [Effort and thinking](#effort-and-thinking)).
- Start a new conversation if you are close to your limit in a long chat.

**How the exam frames it.** Objective 7.3 of the [CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) asks you to "Optimize workflows for efficiency and effectiveness". Of the guide's three sample items, the one on cost and speed is Sample 2 (a Domain 3 item), and it rewards right-sizing: "matching a faster, lower-cost model to straightforward, high-volume work". Applying the same logic to effort levels, batched requests and unneeded tools is our extension, grounded in the help center's usage advice above. For objective 6.2, the data-sensitivity sample item (Sample 3) is worked through under [File uploads](#file-uploads).

**Trap:** applying the other limit's fix. Usage credits let you keep working after your allowance runs out; they do not make one conversation any longer. Starting a new conversation clears a length limit but does not restore a usage allowance, because usage limits apply across all your conversations.

## Choosing a model in the apps

*Tested in: CCAO-F 3.2, 3.3, 3.4, 7.1 · CCDV-F Model Selection and Tradeoffs, LLM Fundamentals · CCAR-P 2.1*

The official [CCAO-F prep course](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations) names "the four decisions that set the quality ceiling for every Claude session before you write a single prompt: entry point, capability features, model, and context." This section covers the model and context decisions. In the apps, the model menu next to the send button controls three settings: which model you are talking to, how much effort it puts into each response, and whether it uses thinking.

### The model families

| Family | Current model in the docs | What the docs say it is for | Relative latency | Share of your limit | Academy guidance: best for | Plans |
|---|---|---|---|---|---|---|
| Haiku | Claude Haiku 4.5 | Fastest model, near-frontier intelligence | Fastest | Lightest | Quick answers, summaries, simple extraction | All plans, including Free |
| Sonnet | Claude Sonnet 5 | Best combination of speed and intelligence | Fast | Moderate | Coding, writing, analysis and multi-step workflows; the versatile default | All plans, including Free |
| Opus | Claude Opus 5.5 | Long-running agentic coding and knowledge work | Moderate | Heavy | Deep research and complex reasoning that needs sustained thinking | Pro and above |
| Fable | Claude Fable 5.1 | Demanding reasoning and long-horizon agentic work | Slower | Heaviest | The largest, most critical projects: long, complex tasks with fewer check-ins | Paid plans. Pro, Team Standard and seat-based Enterprise Standard seats: usage credits only. Usage-based Enterprise: standard API rates. An organization may not have enabled Fable |

The [Academy model tutorial](https://academy.claude.com/tutorials/choosing-the-right-claude-model) frames the choice around your limit (it calls it your rate limit): Haiku is the lightest on it and Fable uses the most, so running a heavy model on a task a lighter one could handle costs tokens for no gain and slows you down. Its practical test is to run a task you already know well on two models and compare where the answers differ, not how long they are.

| Model | Chat context window on paid plans | Training data up to (help center) |
|---|---|---|
| Claude Fable 5.1 | 1M tokens | June 2026 |
| Claude Opus 5.5 | 1M tokens | June 2026 |
| Claude Sonnet 5 | 1M tokens | January 2026 |
| Claude Haiku 4.5 | 200K (the size for every model not on the help center's larger-window list) | July 2025 |
| Claude Opus 5 | 1M tokens | May 2026 |
| Claude Fable 5, Opus 4.8, Opus 4.7 | 500K tokens | January 2026 |
| Claude Opus 4.6, Sonnet 4.6 | 500K tokens | August 2025 |

The [Team plan help article](https://support.claude.com/en/articles/9266767-what-is-the-team-plan) still lists a "200k context window" among Team inclusions. The [context window article](https://support.claude.com/en/articles/8606394-how-large-is-the-context-window-on-paid-claude-plans), which covers Pro, Max, Team and Enterprise, gives the per-model sizes above; use it.

A 200K window holds about 500 pages of text, and part of every window is reserved for Claude's reply, so the longest possible conversation is a little shorter than the full window. Knowledge stops at the cutoff: models "may not be aware of events or information that occurred after their respective cutoff dates" ([training data article](https://support.claude.com/en/articles/8114494-how-up-to-date-is-claude-s-training-data)). The developer docs split Haiku 4.5's date in two: a reliable knowledge cutoff of February 2025 (the date through which its knowledge is most extensive and reliable) and a training data cutoff of July 2025 (the broader range of data used). For anything recent, use [web search](#web-search) rather than reaching for a bigger model; search, retrieval and tool use exist to fill exactly this gap.

### Effort and thinking

| Effort level | Use it for |
|---|---|
| Low, Medium | Routine tasks; they stretch your usage further |
| High | The best overall balance of quality and speed |
| Extra high (xhigh) | Long-running coding and agentic tasks: deeper reasoning than High without the full token cost of Max. Available on Opus 4.7 and newer models |
| Max | Tasks that need the deepest possible reasoning and the most thorough analysis |

- **Thinking is a separate switch.** Thinking and effort can be used in any combination. On models without effort levels, the "Extended" toggle turns thinking on or off. Thinking cannot be turned off on Claude Opus 5.5, Claude Fable 5.1 or Claude Opus 5.
- **Use the thinking trace.** The expandable "Thinking" section shows a summary of Claude's thought process, which helps you check how it reached its conclusion, not just the result. Occasionally the thinking stops before it is complete.
- **Start from the defaults.** Simple questions, basic information requests and general writing need no extra effort or thinking. For complex tasks, raise the effort level, turn on thinking, or both.
- **Read the symptoms.** Too little effort shows up as missed instructions or long work that stops before it is finished. Too much shows up as longer answers that are no better, or scope that grows past what you asked.
- **Tune effort before switching models.** The developer docs say "Tuning effort is often a better lever than switching models" ([choosing a model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model.md)), and the Academy's effort tutorial adds that a frontier model at medium or low effort often beats an older model at high or maximum effort.
- **Change mid-conversation.** Model, effort and thinking can all be changed at any point; the change applies from Claude's next response.

Two behaviors surprise people. First, automatic model switching: on Opus 5 and Opus 5.5, a narrow set of higher-risk requests either falls back to a less capable model or is blocked, and on Fable 5 and Fable 5.1 flagged requests fall back to an Opus model. A fallback is always shown with a notice and a label naming the model that answered. Switching is on by default and can be turned off with the "Switch models when a message is flagged" toggle in Settings > Capabilities; with it off, a flagged request pauses the conversation instead, and you edit and retry or send it to a less capable model yourself. Second, on Enterprise plans a model or effort level you expect may be missing because an administrator turned it off for your role; Haiku models are the exception, since they are always available and cannot be disabled. Administrator model controls are covered in [Admin controls for Team and Enterprise](#admin-controls-for-team-and-enterprise).

### Decision rules for choosing a model

Objective 3.3 asks you to "Align model selection with task requirements (cost, speed, quality)", and the CCAO-F sample item on model choice states the rule in one sentence: "Aligning model selection with task requirements means matching a faster, lower-cost model to straightforward, high-volume work, reserving the most capable model for complex reasoning" ([CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf), Sample 2). The full item is on the [Associate exam page](../claude-certified-associate.md#official-sample-questions). In the table below, the first row is that rule; the other rows come from Academy tutorials and courses and from the developer docs.

| If the task is | Choose | Not | Because |
|---|---|---|---|
| High-volume, short, straightforward work where speed and cost matter more than depth | A faster, lower-cost model (Haiku) | The most capable model for every item | The efficiency-first approach suits high-volume, straightforward tasks; always using the top model wastes cost and latency |
| Everyday drafting, analysis, coding or multi-step work | Sonnet | Opus just to be safe | Sonnet is the versatile default for this work |
| Deep analysis of long, specialized documents, or a problem Sonnet struggled with | Opus | Retrying Sonnet with the same prompt | The Academy maps complex research-paper analysis to Opus, and moving up after Sonnet struggles |
| Work where accuracy outweighs cost | Start capability-first (the most capable model that fits), then optimize down | Starting on the cheapest model | The developer docs' capability-first approach suits applications where accuracy outweighs cost |
| A good model giving thin answers | Raise effort or turn on thinking | Switching model first | Effort is often the better lever |
| A question about recent events | Web search | A bigger model | Model knowledge is frozen at the cutoff |

Two more rules come from the Academy tutorials: judge cost per completed task rather than cost per token, and check a result in proportion to what rides on it (a quick lookup needs a glance, analysis that will shape a decision needs a closer review; the full scale is under [When human review is required](#when-human-review-is-required)).

**Traps**, taken from the official rationales. In the [CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) Sample 2, "Always using the top model (A) wastes the cost and latency budget", and disabling product features or switching to another AI platform "does not address the trade-off". The opposite error is also wrong: in the [CCAR-P guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf) Sample 2, switching "to the smallest available model regardless of task fit" is rejected because "downsizing blindly (B) risks quality". Our summary of the two items: right-sizing means matching the model to the task, in both directions.

### Context limits: restart, summarize or persist

Objective 3.4 asks you to manage context "when to restart, summarize, or persist" ([CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)). Four facts drive the answer:

- When a conversation nears the context limit, Claude summarizes earlier messages so it can continue (automatic context management). It works for users on paid plans with code execution enabled, and it uses more of your usage limit.
- Compaction "is still a summary, and can still occasionally result in lost details" ([Academy: memory and context](https://academy.claude.com/tutorials/parametric-memory-and-context)).
- The whole conversation is sent again on every turn, so a short question late in a long chat can cost more than the same question in a fresh one.
- The [AI Capabilities and Limitations course](https://academy.claude.com/courses/ai-capabilities-and-limitations/working-memory) describes context windows in general: "This property has a cliff rather than a gradient. Silent truncation is the failure mode, and you won't always be warned." It adds that memory features, compaction, projects, larger windows and multi-agent workflows all "exist to push this cliff further out". In the Claude apps, automatic context management (the first point) keeps your full chat history even after summarization; where it does not apply, a message that will not fit gets the length-limit error in the table below. Put the most important instructions at the beginning and end of long context. (Why corrections do not carry over between conversations is covered in [Memory, styles and personalization](#memory-styles-and-personalization).)

| Situation | Do this |
|---|---|
| The conversation has gone off track | Restart: a new chat with a clearer prompt is sometimes faster than redirecting |
| The length-limit error ("Your message will exceed the length limit for this chat", per the [error messages article](https://support.claude.com/en/articles/12466728-troubleshoot-claude-error-messages)) | Start a new conversation, attach fewer or smaller files, summarize or extract the key sections before sending, or ask Claude first to find the relevant portions |
| A long chat is close to your usage limit | Start a new conversation |
| A very long chat where Claude has started agreeing too readily | Start a new conversation: sycophancy is more likely when a conversation gets very long (the other countermeasures are under [Checking for bias and flattery](#checking-for-bias-and-flattery)) |
| The same documents and guidance are needed in every chat | Persist them as project knowledge and instructions (see [Projects](#projects)) |
| Stable facts: your project, your preferences, your working relationships | Persist them in memory or instructions (see [Memory, styles and personalization](#memory-styles-and-personalization)) |
| Ephemeral, granular details such as a head count or a quote | Do not persist them; have Claude look them up fresh when they come up |
| A file you need once | Upload it in the chat; it stays out of project knowledge |

API model IDs, prices and the effort parameter are covered in [Models and how to choose one](claude-api.md#models-and-how-to-choose-one) and [Extended thinking, adaptive thinking and effort](claude-api.md#extended-thinking-adaptive-thinking-and-effort).

!!! warning "Exam guide vs current docs"

    - **A fourth model family.** The [CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) (July 2026) asks you to "Differentiate between Claude model types (Haiku, Sonnet, Opus)". The docs list a fourth family, Fable, above Opus. Claude Fable 5 was released June 9, 2026, before the guide took effect; the current Claude Fable 5.1 was released September 1, 2026, and Claude Opus 5.5 on September 22, 2026. Expect the exam to test the three-family speed, cost and quality trade-off in the guide's wording; treat Fable as the same logic one step further up.
    - **Switching models mid-chat.** The [Claude 101](https://academy.claude.com/courses/claude-101/your-first-conversation-with-claude) video (published December 2025) says "Changing the model will result in a new chat." The [help center](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings) now says you can change the model, effort level or thinking setting at any point in a conversation. Use the help center.
    - **"Extended thinking".** The naming change is covered in the note under [Search, Research and connectors](#search-research-and-connectors).
    - **Default model advice.** The same [Claude 101](https://academy.claude.com/courses/claude-101/your-first-conversation-with-claude) lesson calls Sonnet "our recommended default model"; the [developer docs](https://platform.claude.com/docs/en/models/overview.md) say that if you are unsure which model to use, start with Claude Opus 5.5 for most workloads. The second is developer guidance, not a statement about the app's default. The current [Academy model tutorial](https://academy.claude.com/tutorials/choosing-the-right-claude-model), which already covers Sonnet 5 and Fable 5.1, still says of Sonnet: "If you're not sure which model to pick, start here."

## Projects

*Tested in: CCAO-F 3.1, 3.4, 5.1, 5.2, 5.3, 5.4 · CCDV-F Claude Application Design*

A project is a self-contained workspace with its own chat history and knowledge base. It has two levers you configure. **Project knowledge** is the documents, text, code or other files you upload, which Claude uses as background for every chat in that project. **Project instructions** shape responses across the project, for example a more formal tone or answers from the perspective of a particular role or industry. Projects are available on every plan; Free users can create at most five. The CCAO-F guide's audience statement names this feature (professionals who "build Claude Projects in their day-to-day roles"), and its preparation advice tells candidates to practice it: "configure a Project with instructions and knowledge sources, and evaluate outputs for accuracy and bias" ([CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)).

### Setting one up

1. On the Projects page (claude.ai/projects), click "+ New Project" (upper right).
2. Give it a name and description. Claude cannot see either, so nothing that matters to the output belongs there.
3. On Team and Enterprise plans, choose the visibility: keep it private, or share it with your broader organization where administrators allow that.
4. Click "Set project instructions". Claude applies them to every chat in the project.
5. Add knowledge in the project's Files section: upload files, or add Google Docs from Drive. Docs added from Drive stay synced, so Claude works from the latest version. The Drive option works only in private projects and is disabled for shared ones.

### Project knowledge

| Limit | Value |
|---|---|
| File size | 30MB per file |
| Number of files | Unlimited, but the total content must fit in Claude's context window (RAG extends this on paid plans) |
| What Claude reads | Text extraction only, except multimodal PDFs |

**Retrieval (RAG).** When project knowledge approaches the context window limit, Claude automatically switches the project to RAG mode, which expands capacity by up to 10x. In RAG mode, Claude uses a project knowledge search tool to pull only the relevant parts of your documents into context instead of loading everything. Key facts:

- It is paid-plan only (Pro, Max, Team, Enterprise).
- There is no manual switch; activation depends on the size of the knowledge. If the knowledge later shrinks below the threshold, the project can convert back to loading everything into context.
- A visual indicator shows when a project is RAG-enabled.
- It works with web search, extended thinking and Research.

Because retrieval depends on finding the right file, two habits matter: give files clear, descriptive names, and name the document you mean in your question (for example, *using the Q3 pricing sheet, ...*) so Claude can focus its search. For how retrieval systems work in general, see [Retrieval-augmented generation](solution-architecture.md#retrieval-augmented-generation).

**Keeping knowledge healthy (objective 5.4).**

- Claude 101: "Keep your knowledge base current. Outdated documents can lead to outdated responses." ([Introduction to projects](https://academy.claude.com/courses/claude-101/introduction-to-projects)). Review and update it periodically.
- Remove files you no longer use; they cost context and usage.
- Project content is cached and counts less against your limits when reused, but caches expire after inactivity, so the first message after a long break counts the knowledge in full again.
- A file uploaded directly into a chat stays separate from project knowledge. Use that for one-off context you do not want every future chat to see.
- Google Docs added from Drive sync automatically, so an edit made in Drive reaches the project without a fresh upload. The help center describes that sync only for Docs added from Drive, so treat uploaded files as snapshots to replace when the source changes (our reading).

**What chats in a project share.** Chats in the same project do not share context with each other unless the information is added to the project knowledge base. With memory on, each project also has its own memory, kept separate from other projects and from non-project chats. Project memory, including moving a stray chat out of a project, is covered in [The memory feature](#the-memory-feature).

### Writing project instructions

Project instructions are the everyday place to practice objective 5.3, creating effective system-level instructions. The help center's advice is to use them "for general context around your project, key guidelines, and Claude's role" and to "Reserve task-specific instructions for the chat itself" ([usage and length limits](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work)), and to keep them concise and focused on essential information. Claude 101 groups good instructions into context, process, tone and style, and specific requirements, and notes they can automate a recurring workflow. Three of its examples, combined:

```text
First consider a blog structure that will entice this audience, then write the draft.
Always include a call-to-action at the end of marketing copy.
When I upload a meeting transcript, create a structured summary using this template.
```

The help center also lists defining roles or perspectives Claude should adopt within the project. Where instructions live decides who they affect:

| Layer | Set in | Applies to |
|---|---|---|
| Instructions for Claude | Your account settings | All of your conversations |
| Project instructions | The project | Only chats within that project |
| Organization instructions (Team, Enterprise) | Organization settings > Organization and access, by Owners and Primary Owners | Every conversation in the organization (limits and precedence are under [Organization instructions](#organization-instructions-team-and-enterprise)) |
| Skills | Customize > Skills | Only when the task calls for them (see [Skills in the Claude apps](#skills-in-the-claude-apps)) |

How these layers rank against each other, and how to test that instructions work, are covered in [Instructions for Claude (personal)](#instructions-for-claude-personal) and [Organization instructions](#organization-instructions-team-and-enterprise).

### Sharing and permissions (Team and Enterprise)

On Team and Enterprise plans, projects can be shared with other members of the organization.

| Setting | Options | What it means |
|---|---|---|
| Member permission | Can view | See contents, knowledge and instructions, and chat in the project; cannot edit it |
| | Can edit | Change instructions and knowledge, add or remove members, update member settings |
| Visibility | Public | Everyone in the organization can view and use the project |
| | Private | Only invited members can view and use it |

- Visibility can be changed later from the "Share" button next to the project name.
- Your chats inside a public project stay private unless you share them yourself.
- The Projects page has three tabs: "Your projects," "Organization," and "Shared with you."
- Sharing a project with a group is an Enterprise-only beta; access changes can take up to five minutes to apply.
- Connectors work only in private projects, and chats that contain synced content cannot be shared.
- Archiving a project does not reset its sharing permissions or remove members. An archived project cannot be deleted until you unarchive it.

Admins control this under Organization settings > Data and privacy, in the Sharing section. "Share projects" decides whether users can share projects with others in the organization; its sub-setting "Public projects" decides whether everyone in the organization can see and start chats in public projects. Both are on by default, and only Primary Owners and Owners (plus, on Enterprise, custom roles with the Privacy permission set to "Can manage") can change them.

| Admin turns off | Effect |
|---|---|
| Share projects (this also turns off Public projects) | Users cannot share projects with new users or groups, though they can still change or remove existing access. Projects already shared stay shared and existing access is kept. Existing public projects become private and no new public projects can be created |
| Public projects only | All existing public projects become private and no new ones can be created, but users can still share projects with specific users and groups |
| Project sharing for a role (Enterprise only, under Organization settings > Roles) | Applies to members of that role; role changes can take up to 15 minutes to apply |

### Project, chat, Research, artifact or skill? (objective 3.1)

| You need | Use | Why |
|---|---|---|
| Repeated work on the same body of documents with the same guidance | A project | Static background knowledge that every chat inside it starts with (retrieved as needed once RAG is on) |
| A procedure Claude should follow wherever you are working | A skill | Loads only when relevant and works everywhere across Claude |
| A one-off question or a thought partner, light on context | A plain chat | No setup to maintain |
| A multi-source investigation with citations | Research (see [Search, Research and connectors](#search-research-and-connectors)) | Runs many searches that build on each other |
| A deliverable you will share or keep editing | An artifact (see [Artifacts, files and code execution](#artifacts-files-and-code-execution)) | Opens beside the chat, versioned and shareable |

The fuller comparison of projects, skills, instructions and connectors is in [Skills in the Claude apps](#skills-in-the-claude-apps).

**Traps.** Putting task-specific requests in project instructions (they apply to every chat; task-specific requests belong in the chat itself). Relying on one chat in a project to carry context into another (the help center says context is not shared across chats in a project unless the information is added to the project knowledge base; memory and project-scoped chat search, where on, can surface earlier chats, but adding it to knowledge is the step the help center names). Writing context into the project description (Claude cannot see it). Adding Drive files to a shared project (the option is disabled). Letting knowledge go stale after a policy or price change (outdated documents produce outdated answers).

!!! warning "Exam guide vs current docs"

    - **A new version of projects.** The help center describes a beta in which "a project is one conversation: you say what you need as it comes to you, and Claude breaks the work into parallel threads that run in the cloud", currently for select Pro and Max subscribers who use Claude Code ([What are projects?](https://support.claude.com/en/articles/9517075-what-are-projects)). The [CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) objective 5.1 ("Configure Claude Projects with instructions and knowledge sources") describes the classic model taught above; expect that on the exam.
    - **Project instructions on Free.** The [personalization article](https://support.claude.com/en/articles/10185728-understanding-claude-s-personalization-features) says to use project instructions "(paid plans only)", while the same article and the projects articles say projects are available to all users, with Free capped at five. The exam guide does not address plan availability.

## Artifacts, files and code execution

*Tested in: CCAO-F 2.1, 2.6, 3.1, 4.3, 6.2*

Objective 2.6 asks you to "Organize and curate information and select appropriate output formats (artifacts, inline, structured data)" ([CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)). In the apps today that choice has five practical answers. Two of them, artifacts and downloadable files, depend on one setting: code execution and file creation. Custom visuals need no setting ("You don't need to turn anything on," says the [custom visuals article](https://support.claude.com/en/articles/13979539-custom-visuals-in-chat-and-cowork)).

### Choosing the output format

| Output | Use it when | How to get it |
|---|---|---|
| Inline reply | A quick answer you will read in the conversation, such as "Summarize our Q3 results" | Ask normally |
| Custom visual (beta, web and desktop) | A diagram, chart or interactive element would explain better than text | Ask how something works, or upload a CSV; visuals are ephemeral, so choose "Save as artifact" to keep one |
| Artifact | Significant, self-contained content (typically over 15 lines) that you are likely to edit, iterate on or reuse outside the conversation | Ask for the deliverable, not just the content, or say you want an artifact |
| File (.xlsx, .pptx, .docx, PDF) | Someone needs a file to open in another app | Ask Claude to create it, then download it or save it to Google Drive |
| Structured data | A side-by-side comparison that people need to scan quickly | Ask for a comparison spreadsheet or table and say it must be easy to scan (the Academy's example) |

[Claude 101](https://academy.claude.com/courses/claude-101/creating-with-artifacts) gives the contrast in one example: "Summarize our Q3 results" gets you a chat reply, while "turn our Q3 results into a one-page doc for the leadership team" gets you an artifact you can share and keep working on. If you want to be sure, say "Create this as an artifact." The same lesson separates artifacts from files: "an artifact opens and updates right in Claude and is shared by link, while file creation hands you a file to download and open in other apps."

Two limits shape the choice. Claude does not generate photos or illustrations the way image-generation tools do; its diagrams and charts are built with HTML and SVG, which is why you can interact with them and ask for changes. And in the Academy's comparison-spreadsheet example, the chat preview shows only the basic structure, while the formatting, color coding and cell notes are in the downloaded Excel file.

### Artifacts

An artifact is, in the help center's words, "anything Claude makes for you that you'd put in front of someone: a design, a deck, a document, a dashboard, or a small interactive tool" ([What are artifacts](https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them)). It opens beside the conversation, and you can edit it, come back to it and share it with a link. Common types include single-page HTML websites, diagrams and flowcharts, and interactive React components.

- **Availability.** Free, Pro, Max, Team and Enterprise, and in Claude Code on every plan that includes it. Claude 101 notes that on Free an artifact stays with the conversation that created it.
- **Finding them again.** In the new Claude experience, everything you make is saved to the Artifacts tab automatically. In the previous chat experience, an artifact made in a conversation appears there only after you open it and click "Publish."
- **Prerequisite.** Artifacts are no longer supported without "Code execution and file creation" enabled: in Settings > Capabilities on Free, Pro and Max, or Organization settings > Capabilities on Team and Enterprise.
- **Cost.** Artifacts count toward your plan's usage limits.

**Claude Design, Claude Slides and Claude Docs** are artifact types in beta on paid plans. They are on by default on Pro, Max and Team, and off on Enterprise until an owner turns each one on in Organization settings > Artifacts.

| Type | What it makes | Exports to | Worth knowing |
|---|---|---|---|
| Claude Design | Designs, interactive prototypes, one-pagers and other visual work, on a canvas beside the chat | .pptx, .pdf, standalone HTML or .zip (plus Google Slides from claude.ai/design and handoff to Claude Code) | Not on Free. Use comments for component-level fixes, chat for structural changes, direct edits for quick visual tweaks. No version history yet |
| Claude Slides | Presentations from your notes, reports or the conversation; edit any slide and present without leaving Claude | PowerPoint (.pptx) or PDF | The Claude Design help article points to Slides for presentations |
| Claude Docs | Living documents you write with Claude and your team in real time | Word (.docx), PDF, Markdown (.md) or Google Docs | Not on Free. Mention @Claude in a comment to request an edit; every change is attributed to the person or to Claude; Claude acts with the requester's permissions, so it cannot edit for someone with view-only access; charts and diagrams do not update on their own (ask Claude to pull the latest data); on Team and Enterprise, docs cannot be shared outside the organization yet; not yet available with CMEK, ZDR or a HIPAA-ready configuration; no version history yet |

### Editing and iterating

- For Markdown documents, highlight the text, click "Edit with Claude" and type the change.
- In Claude Design, Claude Slides and Claude Docs, leave a comment on the exact element, slide or passage you want changed, and Claude makes the change there (Claude 101 notes that commenting and direct editing are part of the design, deck and doc experiences).
- Switch between versions with the version selector.
- Editing an earlier chat message creates a different version of the conversation with its own set of artifacts, so you can explore a direction without losing earlier work.
- Your own direct edits do not change Claude's memory of the original content, so tell Claude what you changed if it matters for the next step.
- When something breaks, click the fix button next to the error (the Academy tutorial calls it "Fix with Claude", the help center "Try fixing with Claude") or describe the problem in plain language. The help center adds that a fix is not guaranteed.
- Say who the artifact is for: [Claude 101](https://academy.claude.com/courses/claude-101/creating-with-artifacts) notes that a flowchart "for new employees" comes out differently from one "for the engineering team".

### Prototypes, AI-powered artifacts and where Associate work stops (objective 4.3)

An artifact can call Claude itself: ask Claude to use Claude inside the artifact and no API key is needed. Such an app runs on Anthropic's infrastructure, users sign in with their Claude account, and when others use it, the usage counts against each user's own subscription, not the creator's. MCP integration for artifacts is available on Pro, Max, Team and Enterprise on web and desktop. Each user authenticates the MCP servers themselves, even on shared or published artifacts, and approves an artifact's MCP access on first use. Organization admins can turn artifact MCP access on or off for the whole organization but cannot choose which MCP servers artifacts may use.

| Persistent storage rule | Value |
|---|---|
| Plans | Pro, Max, Team and Enterprise, on web and desktop |
| Storage per artifact | 20 MB |
| What it can hold | Text only: no images, files or binary data |
| Which artifacts get it | Published artifacts only; storage calls fail until the artifact is published |
| Personal or shared | The creator decides: personal storage is private to each user, shared storage is seen by every user |
| On unpublish | All stored data is permanently deleted |
| Before entering sensitive data | Check whether the artifact uses shared storage |

The Academy tutorial on AI-powered artifacts calls them best for testing and demonstration: a production version needs proper API key management and infrastructure, and the path there is to copy Claude's code into an editor. In exam terms (our reading), that is the escalation line: the guide says Associates "are not expected to design enterprise-scale AI architectures or integrations" and that this scope belongs to the Architect and Developer credentials, "to which Associates escalate more complex or technical work" ([CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)). Before sharing a prototype, try to break it with bad inputs and, if it computes, counts or recommends anything, work one example out by hand: a polished app with wrong numbers misleads everyone you share it with. More checks are in [Verifying Claude's output](#verifying-claudes-output).

### Sharing and publishing artifacts

| Plan | Sharing options in the Share dialog |
|---|---|
| Pro, Max | "Only you" or "Anyone with the link" |
| Team, Enterprise | "Only people with access," "Everyone in your organization," or "Anyone with the link" |

- Artifacts start private to you. When you share, you choose whether the link shows the latest version (viewers see your later changes) or a specific version.
- A Claude account is required for anything shared from the Share dialog: people without one cannot open it, even with the link. The one exception is an artifact published from chat (see the "Publish" bullet below).
- Viewers use their own access: an artifact that pulls from connected apps uses the viewer's connections, not the creator's. If a viewer cannot reach a data source, that part shows an error instead of your data.
- On Enterprise, an Owner or Primary Owner must turn on External sharing before "Anyone with the link" works; for chat artifacts and Claude Code artifacts, the help center says Team organizations also share only inside the organization until an owner turns External sharing on. Turning it off later breaks existing public links until it is turned back on. "Anyone with the link" is not offered for artifacts that use connected apps or ask Claude questions, and on Team and Enterprise not yet for Claude Docs.
- Sharing an artifact made in chat also gives viewers the attachments and files from the conversation that created it. Check what else is in that chat before you share. If the artifact came from a project, Team and Enterprise viewers also need access to that project.
- "Publish" is available on Free, Pro and Max for artifacts made in chat in the previous experience. People without a Claude account can view and use a published chat artifact; they are asked to sign up only for advanced features such as AI-powered capabilities. Once you unpublish an artifact, you cannot publish that same artifact again.
- Someone who customizes your shared artifact gets their own copy; your original is unchanged.
- Treat someone else's artifact the way you would treat a file from an unknown sender.

### File uploads

| | Upload to a chat | Project knowledge | Files handled by code execution and file creation |
|---|---|---|---|
| Size per file | 500MB | 30MB | 30MB, for uploads and downloads |
| Number of files | Up to 20 per chat | Unlimited, but total content must fit the context window (on paid plans, RAG mode expands capacity up to 10x) | |
| What Claude reads | PDFs of 100 pages or fewer: text and visuals (images, charts, graphics). PDFs of 101 to 1000 pages: text only. Other documents: text only, embedded images are not read | Text extraction only, except multimodal PDFs | PDFs larger than 30MB can be processed in the computing environment without loading them into context |
| Other limits | Images up to 8000x8000 pixels; PDFs up to 1000 pages; XLSX uploads need code execution and file creation | | |

A worked consequence (our example): a 300-page PDF is read as text only, so its charts are not analyzed; if the charts matter, upload the relevant section of 100 pages or fewer on its own (the help center's own tip for large documents is to divide them into smaller sections).

The two help articles give different upload limits: 500MB per file for chat uploads, and "30MB per file for both uploads and downloads" for code execution and file creation ([file creation article](https://support.claude.com/en/articles/12111783-create-and-edit-files-with-claude)). Project files are 30MB each. Apart from the file creation article's note that PDFs larger than 30MB can be processed in the computing environment without loading them into context, neither article explains how the two limits interact.

**Before you upload (objective 6.2).** The CCAO-F sample item on data sensitivity turns on this step: the correct answer means "redacting or anonymizing regulated identifiers before use, so the analysis can proceed without exposing protected data", and telling the model not to retain the data "does not satisfy the policy control" ([CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf), Sample 3). The Academy's practical version: for pattern analysis you likely do not need names or contact details, so make a copy of the file, replace names with placeholders such as *Customer A*, and delete contact details entirely. See [Untrusted input, PII and data leakage](security-and-governance.md#untrusted-input-pii-and-data-leakage).

### Code execution and file creation

Code execution and file creation is available on every plan, on the web, Claude Desktop and Claude Mobile. It is on by default on Free, Pro and Max, on Team, and for new Enterprise organizations; Team and Enterprise owners can turn it off for the organization. With it, Claude works in a sandboxed computing environment and can create Excel, PowerPoint, Word and PDF files, write Python scripts for data analysis, render charts as image files (PNG), and analyze uploaded CSV, TSV and other data files. Creating files uses more of your limit than a normal chat.

The same setting underpins several other features, so when one of them seems to be missing, check this setting first (our troubleshooting rule):

| Feature | Needs code execution and file creation |
|---|---|
| Artifacts | Yes |
| Skills | Yes |
| Uploading XLSX files | Yes |
| Automatic context management in long chats | Yes |

For calculations, two pieces of guidance apply. The help center says "complex or mission-critical calculations should be verified using specialized mathematical software or manual methods" ([calculations article](https://support.claude.com/en/articles/10366421-how-does-claude-handle-mathematical-equations-and-calculations)), and the Academy's [capabilities and limitations course](https://academy.claude.com/courses/ai-capabilities-and-limitations/when-properties-collide) lists "offload to code execution" among its standard fixes.

**Network access.** On Free, Pro and Max, network access is on, so Claude can install packages from approved sources. For Team and Enterprise, the file creation article's availability list says network access is off by default and owners can enable it in organization settings; its setup steps add that new Enterprise organizations start with Allow network egress off, while Team organizations start with egress on to package managers only. Owners choose among four settings in Organization settings > Capabilities:

| Egress setting | What Claude can reach |
|---|---|
| Off | Pre-installed packages only, no internet access |
| Package managers only (labeled the default) | Approved package managers such as npm, PyPI and GitHub, to install software |
| Package managers and specific domains | Package managers plus domains the owner adds to an allowlist |
| All domains | Full internet access except domains on Anthropic's legal blocklist: the most flexible and the riskiest option |

The risk is prompt injection: Claude "can be tricked into sending information from its context (for example, prompts, projects, data via MCP, Google integrations) to malicious third parties" ([file creation article](https://support.claude.com/en/articles/12111783-create-and-edit-files-with-claude)). Disabling network access keeps data inside the sandbox even if something goes wrong. For a cautious rollout the article recommends a phased approach: start with network access off, then enable package managers, then add specific domains to an allowlist as business needs require. Two caveats: MCP integrations can still reach the network whatever the egress setting, and Anthropic recommends watching Claude while it works and stopping it if it uses or accesses data unexpectedly. See [Prompt injection](security-and-governance.md#prompt-injection).

!!! warning "Exam guide vs current docs"

    - **Artifacts grew.** The [CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) objective 2.6 lists "artifacts, inline, structured data". Since the guide, Anthropic announced on September 16, 2026 that Claude Docs and Claude Slides were new and that Claude Design now works inside conversations. Inline custom visuals (in beta, released March 12, 2026, before the guide) are also missing from the guide's three formats. Expect the guide's three-way choice on the exam. Our decision rule still holds: inline for a quick answer you read in the conversation, an artifact for something you will edit, reuse or show, structured data for things people will scan.
    - **Presentations.** The [Claude Design help article](https://support.claude.com/en/articles/14604416-get-started-with-claude-design) says "To make presentations, use Claude Slides." while an [Academy tutorial](https://academy.claude.com/tutorials/using-claude-design-for-presentations-and-slide-decks) still teaches Claude Design for slide decks.
    - **Viewing shared artifacts.** [Claude 101](https://academy.claude.com/courses/claude-101/creating-with-artifacts), describing how designs, decks and docs are shared, says "Anyone you share with needs a Claude account to open the artifact," while an [Academy tutorial](https://academy.claude.com/tutorials/use-artifacts-to-visualize-and-create-ai-apps-without-ever-writing-a-line-of-code) says published artifacts can be used without signing in (AI-powered ones excepted). The [publish and share article](https://support.claude.com/en/articles/9547008-publish-and-share-artifacts) reconciles them: artifacts shared from the Share dialog need a Claude account, while artifacts published from chat in the previous experience can be viewed by people without one. Answer by asking how the artifact was shared.

## Search, Research and connectors

*Tested in: CCAO-F 3.1, 4.2, 5.2, 6.2 · CCDV-F MCP Server Development · CCAR-P 3.7*

A model knows only what was in its training data, frozen at its cutoff. The Academy's capabilities course puts the fix plainly: "Web search, retrieval (RAG/MCPs), and tool use exist specifically to patch these gaps by giving the model access to information it was never trained on" ([Knowledge lesson](https://academy.claude.com/courses/ai-capabilities-and-limitations/knowledge)). In the apps, those routes show up as web search, Research, and connectors to your own tools. Each adds reach and each adds risk. The CCAO-F guide names these features in two objectives: 3.1 asks you to select product features ("Projects, research mode, chat, artifacts") and 5.2 asks you to "Manage uploaded knowledge and connectors (e.g., Google Drive, Gmail)".

### Web search, extended thinking or Research?

| Feature | Best for | Examples | Usage note |
|---|---|---|---|
| Web search | Straightforward, factual queries answerable with one or two tool calls | The weather, details about a specific company, recent news headlines | Fetching a long page can use a significant portion of your usage limit (flagged for free accounts) |
| Extended thinking | Complex reasoning that does not need recent information from the web | Math problems, debugging code, analyzing philosophical concepts | Turn thinking off when a task does not need it (it cannot be turned off on Opus 5.5, Fable 5.1 or Opus 5); higher effort also uses more tokens |
| Research | Thorough information gathering that needs five or more tool calls over 1 to 3 minutes, synthesized into an in-depth report from the web and your integrations | Comparing business competitors, updating outdated internal documents with information from the web, prioritizing action items from calendar and email | Same limits as other chats, used up faster |
| Enterprise search (Team, Enterprise) | Quick knowledge retrieval across your organization's connected tools | "What is our company's remote work policy?", "Summarize discussions about the Q4 product roadmap." | Counts toward your plan's standard usage limits, like other projects and conversations |

Research and extended thinking can be combined: thinking plans the approach and Research gathers the material, which suits jobs like researching emerging technologies for a business proposal or analyzing multiple scientific papers for a research project. Our decision rule, built on that article: if one or two lookups answer it, use web search; if it needs reasoning but no fresh facts, use thinking; if it needs many sources stitched into a cited report, use Research.

### Web search

- Every response that uses web search includes citations; how to check them is step 3 of the [verification routine](#a-verification-routine).
- On Team and Enterprise, an Owner or Primary Owner must first enable web search for the whole workspace in Organization settings > Capabilities.
- In the new Claude experience there is no web search toggle: Claude searches the web when it helps. In the previous experience, you switch it on per chat from the "+" button at the lower left of the chat, and can switch it off for chats that do not need it.
- With web search on, **web fetch** lets Claude read the full content of pages at URLs you give it. The whole page is retrieved into the context window, so, in the help center's note for free accounts, a long article can use a significant portion of your usage limit.
- To save usage and context, ask Claude not to search when you do not need current information.

### Research

- **Who and where.** Paid plans (Pro, Max, Team, Enterprise) on the web, Claude Desktop and Claude Mobile.
- **How it works.** Claude works agentically, running multiple searches that build on each other and deciding what to investigate next. It covers your connected internal sources (for example Gmail, Google Calendar and Google Docs) as well as the web.
- **Prerequisite.** Web search must be on for Research to work.
- **Starting it.** In the previous experience, click "+" then "Research"; a blue indicator shows it is on. In the new Claude experience, type /deep-research, or click "+" and choose "Research".
- **Steering it.** If Research is on but Claude is not researching, the [Research help article](https://support.claude.com/en/articles/11088861-use-research-on-claude) suggests saying "Claude, please use the research tool to…"; if it is not pulling your internal documents, tell it which source to pull context from.
- **Cost.** It is subject to the same limits as other chats, but uses them faster because it retrieves many sources.
- **Projects.** It works inside RAG-enabled projects.

**Connectors during Research.** Claude can call tools from your connectors automatically during Research, without asking for approval each time. That is why the custom-connector guidance says that when using Research with custom connectors you should disable any tools that can take write actions in external applications, review approval requests carefully, and be mindful of the load from Claude sending a large number of requests to your connectors. The same article notes that advanced research cannot currently invoke tools from local MCP servers. Our rule of thumb: a research task should read, not send, delete or update.

### Enterprise search

Enterprise search (Team and Enterprise) appears as a pre-configured "Ask Your Org" project in the sidebar. It is enabled by default for Team and Enterprise organizations, but an Owner must complete the setup before other members can use it. What sets it apart from a regular project:

- Its system prompt is maintained by Anthropic and optimized for search, instead of instructions you write.
- During setup an Owner must choose a connector for both Documents and Chat; Email is recommended but optional. Only connectors enabled for the organization are available.
- It is permission-aware: each person only sees results from data they can already access in the original systems, and each user authenticates to the connectors with their own credentials.
- No external indexing: results come from MCP calls to the connected services, and no data from connected services is indexed in Anthropic's systems for serving queries.

It is fully functional on Claude Desktop but not available on Claude Mobile. Choose it for quick knowledge retrieval across the organization; choose Research for deep, multi-step research on a specific topic. The article's prompting tips (worth reusing in any search, our suggestion): name the sources ("Search Slack and Google Drive for discussions about the Q4 product launch."), add date ranges, and break complex queries into steps.

### Connectors

Connectors let Claude access your apps and services, retrieve your data, and take actions in them. Claude inherits each person's permissions from the connected service: if someone cannot reach a file, channel or record in the source system, the connector cannot reach it from Claude either. Claude 101's example: connecting your work email gives Claude your inbox, not your CEO's ([Connecting your tools](https://academy.claude.com/courses/claude-101/connecting-your-tools)). Claude sees only what you can see. Connectors are built on the Model Context Protocol (MCP), "an open standard, created by Anthropic, for AI applications to connect to tools and data" ([custom connectors](https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp)); Claude 101 likens it to USB-C for AI.

| Kind | Where it runs | Where it works | How you add it |
|---|---|---|---|
| Directory connector (remote) | A remote server, reached over the internet | All Claude surfaces: Claude, Cowork, Claude Desktop, Claude Mobile | From the Connectors Directory; Anthropic's directory of recommended connectors is at claude.ai/directory |
| Custom connector (remote MCP) | Your remote MCP server, which Claude reaches from Anthropic's cloud, not your device | Claude, Cowork and Claude Desktop, on every plan; Free is limited to one | "Add custom connector", with optional OAuth Client ID and secret under advanced settings |
| Desktop extension (local MCP) | Your computer | Claude Desktop and Claude Code only, not web or mobile | Settings > Extensions in Claude Desktop; custom ones install from a .mcpb file |

Because a custom connector is called from Anthropic's cloud, its MCP server must be reachable over the public internet from Anthropic's IP ranges. To change one, remove it and add it again with the new details. Local servers configured in Claude Desktop's `claude_desktop_config.json` are a separate mechanism: they use your local network but are not available in Cowork or claude.ai. And if you use Claude Code signed in with a claude.ai account, connectors you added in claude.ai are available there automatically. Protocol details (transports, primitives, building a server) are in [MCP architecture](tool-use-and-mcp.md#mcp-architecture) and [Building an MCP server](tool-use-and-mcp.md#building-an-mcp-server).

**Using connectors day to day.**

- Open the connectors menu with the "+" button at the lower left of the chat, or by typing "/", and toggle on the services you want for that conversation.
- Tool access has three modes: **Auto** (the default), where Claude decides which connectors to load based on what you are working on; **Always available**, where all your connectors load at the start of every conversation (suited to fewer than 10 connectors you use constantly, at the cost of more conversation space); and **On demand**, where connectors are not loaded until Claude searches for the right one. The connectors article suggests On demand if you have 10 or more connectors active.
- Tools and connectors are token-intensive: turn off the ones a conversation does not need to save both context and usage.
- Some connectors are interactive and show inline cards (summaries, confirmations, quick actions) in the conversation; look for the Interactive badge in the directory.
- One directory connector can cover several apps: the Atlassian Rovo connector reaches both Jira and Confluence, so search for the vendor rather than the app.
- To review or remove what you have connected, go to Customize > Connectors, where you can disconnect a service or review its permissions. Disconnect services you no longer need.

**Team and Enterprise controls.**

| Control | Rule |
|---|---|
| Enabling a connector | An Owner or Primary Owner must enable it for the organization first |
| Access | Enabling does not grant access: each person still authenticates individually. With Enterprise-managed auth (beta, Team and Enterprise), a connector is authorized once and the team inherits access on first login |
| Custom connectors | Anyone can build and host one, but only Owners can add them to a Team or Enterprise organization |
| Action permissions | Owners can set each permission category (for example read-only tools, write/delete tools) or individual permission to Always allow, Needs approval, or Blocked. The restriction applies org-wide and individual users cannot override it. The article's example: let Claude search and summarize email but prevent it from sending messages |
| Ceiling | Restricting actions in Claude never grants more access than the source system allows; it only narrows it |
| Projects | Connectors work only in private projects, and chats with synced content cannot be shared |

**Connector safety.** The custom-connector article's rules: connect only to servers built and hosted by organizations you trust; review the permissions a server requests during sign-in and limit those scopes where you can; be aware that malicious MCP servers may carry hidden instructions (prompt injection); click "Allow always" only for a server and tool you trust to run unsupervised; disable tools a conversation does not need; and watch for changes in tool behavior, since server developers can update tools unexpectedly. See [Prompt injection](security-and-governance.md#prompt-injection).

### Google Workspace and Microsoft 365 (objective 5.2)

The guide's own examples for objective 5.2 are Google Drive and Gmail. What to know:

| Topic | Google Workspace (Gmail, Calendar, Drive) | Microsoft 365 |
|---|---|---|
| Covers | Gmail, Google Calendar, Google Drive | SharePoint, OneDrive, Outlook and Teams content in your work account |
| Plans | Help center: all users on Claude and Claude Desktop (the claude.com connector docs say Pro, Max, Team and Enterprise) | All plans, Free included |
| Organization setup | On Team and Enterprise, an Owner or Primary Owner enables the connectors before users can authenticate; owners can also disable them in Organization settings > Connectors | On Team and Enterprise, the Claude organization Owner enables Microsoft 365; in every tenant, a Microsoft Entra Global Administrator must also grant a one-time consent |
| Accounts | Your Google account; if Google Workspace blocks access, a Google admin must mark Claude as trusted, and the policy takes about 15 minutes to propagate | Work accounts only; personal accounts such as @outlook.com, @hotmail.com or @live.com cannot be used |

**Gmail.** Ask a question that needs email and Claude detects that it should search your mail; its citations show which emails it used and link back to them. Attachment content is not accessible through Gmail (metadata only). The help center says Claude can send, reply to and forward emails, asking for your approval before each by default, and that on Team and Enterprise owners decide whether members may let those actions run without asking. The same approval default covers sharing, moving and trashing files in Google Drive. The help center also notes that Google's OAuth screen mentions email-sending permissions during sign-in; Claude still sends only with your explicit approval by default.

**Google Drive and Docs.**

- Add a Doc in chat with "+" and "Add from Google Drive"; in projects, Drive works only in private projects.
- From a Google Doc, Claude extracts the main text only: no images, comments or suggestions. The claude.com Drive doc lists Google Docs up to 10MB and Google Sheets and Slides as not currently supported; convert a .docx by opening it in Google Docs and choosing "Save as Google Docs".
- Added Docs stay synced with Drive, so Claude works from the latest version, and you can add only documents you have permission to view. If you lose access, the document preview disappears but the conversation history remains.

**Data handling.** Anthropic states it does not train its models on Gmail, Drive or Calendar connector data. One caveat from the same article: on consumer plans (Free, Pro and Max) where you have allowed your chats to be used for training, content you copy and paste from those services, or Claude responses that include specific information from them, may be used. Claude retrieves the minimum information needed, only when you ask for something that requires it. Data a connector fetches is stored with the chat that fetched it, so deleting the chat deletes that data. To reset a broken Google connection, disconnect it in Customize > Connectors and re-authenticate on next use; if problems persist, remove Claude's access at myaccount.google.com/connections.

**Managing connectors well (objectives 5.2 and 6.2).** Connect only services with relevant data, review the permissions you grant, and switch off connectors a conversation does not need. The CCAO-F guide's three sample questions cover Domains 2, 3 and 6 and none is about connectors, so this is our reading of the objective: the pattern to recognize is least privilege applied by a non-developer. That means the smallest set of connectors, read-only where the task only reads (the rule the custom-connector article gives for Research), and approval kept on for actions that send or change things.

!!! warning "Exam guide vs current docs"

    - **"Research mode".** The [CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) objective 3.1 lists "Projects, research mode, chat, artifacts". The product calls the feature Research (started with /deep-research in the new Claude experience; see the note under [Claude Cowork](#claude-cowork) for what that experience changes). Expect "research mode" on the exam; it means Research.
    - **"Extended thinking".** The [when-to-use article](https://support.claude.com/en/articles/11095361-when-should-i-use-web-search-extended-thinking-and-research) still says "extended thinking"; the [model settings article](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings) now describes an effort level plus a separate thinking setting. In our reading the decision (reasoning without fresh facts) is the same under either name; see [Effort and thinking](#effort-and-thinking).
    - **How long Research takes.** The [when-to-use help article](https://support.claude.com/en/articles/11095361-when-should-i-use-web-search-extended-thinking-and-research) says "over 1-3 minutes"; the [Claude 101 video](https://academy.claude.com/courses/claude-101/your-first-conversation-with-claude) (December 2025) says five to 45 minutes; the current [Claude 101 Research lesson](https://academy.claude.com/courses/claude-101/research-mode-for-deep-dives) says a few minutes or more. The exam guide gives no duration.
    - **Gmail and Calendar actions.** The [help center](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors) says Claude can send, reply to and forward Gmail messages and create, update and delete Calendar events; the claude.com connector docs say Claude "cannot create, send, or modify emails" ([Gmail doc](https://claude.com/docs/connectors/google/gmail.md)) and cannot create, modify or delete events. The two sources also disagree on which file types Drive reads: the help center lists Sheets, Slides, PDFs, images and MS Office files, while the claude.com Drive doc lists only Google Docs as supported, marks Sheets and Slides as not supported, and advises converting a .docx to Google Docs. The exam guide names the connectors but not their actions; for an exam item, our safe principle is that any send or change action should keep human approval (the [Cowork safety article](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely) likewise advises manual approval when mistakes would be hard to undo, like sending messages).

## Skills in the Claude apps

*Tested in: CCAO-F 5.3, 5.4 · CCDV-F Agentic Customization · CCAR-F 3.2 · CCAR-P 2.5, 3.8*

Skills are "folders of instructions, scripts, and resources that Claude loads dynamically to improve performance on specialized tasks" ([What are skills?](https://support.claude.com/en/articles/12512176-what-are-skills)). They need code execution to be enabled. The CCAO-F guide tells candidates to review the documentation for Skills alongside Projects, Artifacts, Memory and Code Execution, although no CCAO-F objective names Skills directly. In our mapping, they fall under writing effective instructions (5.3, "Create effective system-level instructions") and under configurations you maintain and update (5.4). For developers and architects, the same format is the Agent Skills open standard published at agentskills.io.

### How a skill loads: progressive disclosure

Claude does not read every skill in full. It works through progressive disclosure, loading only what the task needs so the context window is not overloaded:

```text
Level 1  Metadata   Name and description of every enabled skill, read at startup      (about 100 tokens each)
Level 2  SKILL.md   The full instructions, loaded when a task matches a description  (under 5k tokens)
Level 3  Resources  Scripts and reference files, loaded only when needed            (nothing until accessed;
                    a script runs through bash and only its output enters context)
```

This is why the description carries so much weight: together with the name, it is all Claude sees of a skill until the skill is triggered. If Claude is not using a skill, the help center's first two checks are that it is toggled on in Customize > Skills and that its description clearly explains when it should be used. For architects (CCAR-P 3.8, "Evaluate progressive discovery vs. monolithic context strategy"), our mapping: skills are the product example of progressive discovery, while project knowledge is always loaded until it approaches the context limit and Claude switches the project to RAG mode.

Kinds of skill you will meet:

- **Anthropic skills**, created and maintained by Anthropic, such as enhanced document creation for Excel, Word, PowerPoint and PDF.
- **Partner skills** in the Skills Directory, from partners such as Notion, Figma and Atlassian.
- **Custom skills** that you or colleagues create.
- **Organization skills** that Owners on Team and Enterprise provision for all users.

### Skill, project, instructions or connector?

| Mechanism | Holds | Loads | Reach |
|---|---|---|---|
| Instructions for Claude | Account-wide preferences | In every conversation | All your chats |
| Project | Static background knowledge and project instructions | Always, for chats inside the project | That project |
| Skill | A procedure: instructions, scripts and resources | Only when the task matches | Everywhere across Claude |
| Connector (MCP) | Access to a tool or data source | When enabled for the conversation; under the default Auto tool access, Claude decides which to load | Where connected |

Two one-liners from official sources settle most questions: "projects store knowledge, skills perform tasks" ([Claude 101](https://academy.claude.com/courses/claude-101/working-with-skills)), and "MCP connections give Claude access to tools, while skills teach Claude how to use those tools effectively" ([What are skills?](https://support.claude.com/en/articles/12512176-what-are-skills)). The current personalization article does not mention styles; it says to use skills "when you want to customize how Claude formats and delivers its responses" ([personalization features](https://support.claude.com/en/articles/10185728-understanding-claude-s-personalization-features); see [Memory, styles and personalization](#memory-styles-and-personalization)).

Decision rules (ours, built on those sources):

- If you keep explaining the same procedure, write it down once as a skill. The Academy's advice for working with agents is the same: "When you explain the same thing twice, turn it into written instructions for the agent." ([Practical ways to get started](https://academy.claude.com/courses/building-effective-human-agent-teams/practical-ways-to-get-started))
- If Claude needs facts about one body of work, use a project; if it needs to follow a method wherever you are, use a skill.
- If Claude cannot reach a system, you need a connector; if it reaches it but uses it badly, you need a skill.
- If a preference applies to everything you do, put it in Instructions for Claude, not in a skill.

The same logic appears in Claude Code for CCAR-F 3.2, which asks you to choose between skills (on-demand, task-specific) and CLAUDE.md (always-loaded universal standards); see [Agent Skills](claude-code-configuration.md#agent-skills). The official rationale for CCAR-F sample Question 6 marks the limit of on-demand loading: a skill "requires manual skill invocation or relies on Claude choosing to load them," so it is the wrong answer when conventions must apply automatically by file path (the correct answer there is rule files in `.claude/rules/` with glob patterns).

### Turning skills on

| Plan | Where | What must be on |
|---|---|---|
| Free, Pro, Max | Settings > Capabilities, then Customize > Skills | "Code execution and file creation", then the individual skills you want |
| Team, Enterprise | Organization settings > Plugins & skills, "Policy" tab | Both "Cloud code execution and file creation" and "Skills". On Team, skills are on at the organization level by default |

### Creating a custom skill

There are three routes:

1. **Talk it through with Claude.** Claude 101 calls this the easiest way to create a custom skill.
2. **Record it.** In Cowork on Claude for Mac, on Pro, Max and Team plans, you can record yourself doing the task and Claude proposes a skill for you to review; a recording can run for about 10 minutes. Recording is not available in chat, on Windows, or on Free and Enterprise plans. Everything on screen and anything you say is captured, so do not type passwords or show sensitive information while recording. The video and audio are not retained afterward; a set of screenshots stays in the Cowork task until you delete the task.
3. **Write it and upload it.** In Customize > Skills, click "+" then "+ Create skill", choose "Upload a skill", and upload a ZIP file containing your skill folder.

The opening of the claude.com skills docs' complete example `SKILL.md` (frontmatter plus the first line of instructions):

```markdown
---
name: brand-guidelines
description: Apply Acme Corp brand guidelines to presentations and documents, including official colors, fonts, and logo usage.
---

# Brand Guidelines

Apply these standards when creating presentations, documents, or marketing materials for Acme Corp.
```

The ZIP must contain the folder, not loose files:

```text
my-skill.zip
  my-skill/
    SKILL.md
    scripts/
```

| Rule | Requirement |
|---|---|
| File structure | `SKILL.md` starts with YAML frontmatter holding the required metadata, followed by Markdown instructions |
| `name` | Lowercase letters, numbers and hyphens only; at most 64 characters; must match the directory name |
| `description` | What the skill does and when to use it; Claude uses it to decide when to invoke the skill; include keywords that help Claude identify relevant tasks; at most 1,024 characters (the Agent Skills specification limit) |
| Length | Keep the main `SKILL.md` under 500 lines and move detailed reference material to separate files |
| Scope | One workflow per skill: "Multiple focused skills compose better than one large skill" ([skills how-to](https://claude.com/docs/skills/how-to.md)) |
| Secrets | Do not hardcode sensitive information such as API keys or passwords |
| Naming for teams | Name specifically to avoid collisions: `sales-customer-renewal-prep` rather than `meeting-prep` |

When uploading through the Claude API instead, the name also may not contain XML tags or the reserved words "anthropic" and "claude", the description must be non-empty with no XML tags, and the maximum upload size is 30 MB (all files combined, uncompressed). A single API request can include up to 20 Skills.

Testing, from the claude.com skills docs and the help center: before uploading, review `SKILL.md` for clarity, check that the description reflects when Claude should use the skill, and confirm every referenced file exists. After uploading, enable the skill in Customize > Skills, try prompts that should trigger it, review Claude's thinking to confirm it loaded the skill, and iterate on the description if Claude is not using it when expected.

!!! note "Sources disagree on skill metadata"

    The help center's [How to create custom skills](https://support.claude.com/en/articles/12512198-how-to-create-custom-skills) article calls the file `skill.md`, describes `name` as a human-friendly name of up to 64 characters, gives the description "(200 characters maximum)", and shows an example with `name: Brand Guidelines`, which has a capital and a space. The claude.com docs and the Claude API docs use `SKILL.md`, a lowercase hyphenated name that matches the folder, and a 1,024-character description; the help center's own provisioning article also says the ZIP "must include a SKILL.md file". Our advice: follow the stricter naming rule and keep the description within 200 characters, which satisfies both sources. Either way, the description must say what the skill does and when to use it.

### Sharing, publishing and organization skills

Custom skills you upload are not visible to colleagues until you share them. Admins can see their names and sharing status, but not their files.

| Action | Control | What recipients get |
|---|---|---|
| Share with people or groups (groups on Enterprise only) | You keep control of the skill | A view-only skill that stays off until they enable it; they can use it but not edit it, and they get your updated version automatically at next use |
| Publish to the organization | You hand the skill to the organization | An entry in the organization library. If publishing requires review, you propose how it is offered (Available to install, Installed by default, or Required) and the reviewer can change that. Later versions go through the same review |

Owner controls on Team and Enterprise (objective 5.4 in our mapping; the pricing table marks organization-wide skills deployment as Yes on Team and Enterprise):

- A skill uploaded through Organization settings > Plugins & skills is provisioned to all users immediately. It is on by default for everyone unless the owner sets it to off by default (the provisioning article advises keeping specialized skills off by default); users can switch it off for themselves but cannot delete it, and only owners can add or remove organization-wide skills. Provisioned skills also load in Claude Code for users who sign in with their Claude account.
- Enterprise can target a group by bundling skills into a plugin and assigning the plugin to that group.
- Turning off "User-created skills" stops users from creating skills in Claude or uploading skill files; provisioned skills and Anthropic's built-in skills stay available.
- Publishing has three settings: "Requires review" (an owner approves each submission, and every later version), "Open" (submitted skills and plugins go to the library without review) or "Off" (users do not see the "Publish to org" button; owners can still add items directly). Team plans start at "Open" (or "Off" if Share with organization was already off). Enterprise plans start with publishing off (or "Open" if Share with organization was on), and an Enterprise organization that has not chosen a setting switches to "Requires review" on October 2, 2026. Anthropic's [Enterprise deployment course](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/governing-customizations), describing the Share with organization toggle, says "There is no in-product review step"; in our reading the Publishing setting is the current control.
- Skill and plugin security scanning is an Enterprise setting. It is off by default until October 2, 2026; from that date it is on by default, where available, for Enterprise organizations that have not set it. A skill or plugin that passes installs normally, one that may carry risk stays usable behind a caution banner, and one with malicious content is blocked. Scanning covers custom skills uploaded in claude.ai and Cowork, not those uploaded through the Skills API or the Claude Console.
- Test a skill on your own account before provisioning it to the organization.

For shared configuration, the Cowork course's habits are worth adopting: one named owner per shared plugin who reviews changes and runs the evals, and a review rhythm ("Quarterly is a reasonable starting point") to retire what has gone stale ([sharing lesson](https://academy.claude.com/courses/introduction-to-claude-cowork/share-what-you-build-with-your-team)). Wider admin controls are in [Admin controls for Team and Enterprise](#admin-controls-for-team-and-enterprise).

### Skill security

The help center names the two main risks: "prompt injection, which allows Claude to be manipulated to execute unintended actions, and data exfiltration, caused by malicious package code or prompt-injected data leaks" ([Use skills in Claude](https://support.claude.com/en/articles/12512180-use-skills-in-claude)). A skill can carry scripts that run, so the platform docs say to treat installing one like installing software: use skills only from trusted sources (ones you created or obtained from Anthropic), audit every bundled file before using one from anywhere else, and be wary of skills that fetch data from external URLs. The claude.com skills docs add: never hardcode API keys or passwords in one. For access to external services, the skills docs say to use MCP connections.

### Where skills work beyond the chat window

| Surface | How custom skills get there | Who can use them | Network access |
|---|---|---|---|
| Claude apps (web, desktop) | Uploaded in Customize > Skills | Help center: you; on Team and Enterprise also people you share with or the organization you publish to. Platform docs: individual user only | Varies with your settings |
| Excel, PowerPoint, Word and Outlook add-ins | Skills enabled in your Claude settings carry over; type / in the sidebar to pick one (for example /deck-check), or describe the task and Claude applies a relevant skill | Your account | Not stated |
| Claude Code | Files in `~/.claude/skills/` or `.claude/skills/`, or plugins; plus a one-way sync from your Claude account (Claude Code v2.1.273 or later; not when signed in with an API key or running on a cloud provider such as Amazon Bedrock) | Personal, project or plugin | Full network access, like any local program |
| Claude API | Uploaded to the Skills API; run through the code execution tool | Everyone in the workspace | None |

The platform [Agent Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) still says the pre-built document skills (PowerPoint, Excel, Word, PDF) are not available in Claude Code. The current [Claude Code skills docs](https://code.claude.com/docs/en/skills.md) say the sync includes "Anthropic's built-in skills such as `pdf` and `xlsx`", and that some of them always sync, so for Claude Code signed in with a claude.ai account, treat the Claude Code docs as current. The Claude Code sync downloads skills into `~/.claude/skills/synced/` when a session starts and checks for changes about every 10 minutes; it reads from your Claude account and never changes anything in it. To stop it, set `syncClaudeAiSkills` to `false` in your user settings (`~/.claude/settings.json`), in `.claude/settings.local.json`, or in managed settings. Only `false` counts (`true` is the same as unset), and a `false` in a repository's shared `.claude/settings.json` is ignored, so a repository cannot turn it off for you:

```json
{
  "syncClaudeAiSkills": false
}
```

API mechanics (the `container` parameter, version pinning, workspace scope) are covered in [Built-in tools, custom tools, Skills or MCP](tool-use-and-mcp.md#built-in-tools-custom-tools-skills-or-mcp), which also serves CCDV-F Agentic Customization.

### Plugins

Plugins are available on all paid plans. Each one bundles skills, connectors and sub-agents into a single package, so a team gets a ready-made setup instead of configuring each piece. You install them from the Plugins tab with "Browse plugins"; Anthropic's Knowledge Work marketplace is added by default. An organization can offer a plugin as **Available to install**, **Installed by default** (members may uninstall), **Required** (members cannot uninstall), or **Not available** (hidden from the catalog, useful for staging or deprecating a plugin). Claude Code's plugin system is covered in [Plugins and marketplaces](claude-code-configuration.md#plugins-and-marketplaces).

**Traps** (our list, drawn from the documentation's own warnings and the rationale for CCAR-F sample Question 6). A skill chosen for something that must happen automatically every time (skills load on demand). One large skill that tries to cover every workflow (focused skills compose better). A vague description (Claude will not know when to load it). A skill from an unknown source (prompt injection and exfiltration). An API key inside the skill. Provisioning to the whole organization before testing on your own account. A preference that belongs in Instructions for Claude packed into a skill, or the reverse. And for reuse questions (CCAR-P 2.5, "Implement prompt reuse strategies (caching, modular prompts, Skills)" in the [CCAR-P guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)), remember that a skill is reused by being loaded on demand, not by being pasted into every prompt.

!!! warning "Sources disagree on where skills reach"

    - **Syncing.** The help center and the [Claude Code skills docs](https://code.claude.com/docs/en/skills.md) describe a one-way sync of your Claude skills (including owner-provisioned ones) into Claude Code, v2.1.273 or later. The platform's [Agent Skills overview](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) still says "Custom Skills do not sync across surfaces" and that "claude.ai does not support centralized admin management or org-wide distribution of custom Skills." For claude.ai to Claude Code, treat the help center and Claude Code docs as current. The help center articles on skills describe no sync with the API, so the platform overview's other two points stand: skills uploaded through the API are not available on claude.ai, and claude.ai uploads must be uploaded again to use them through the API. The exam guides do not mention the sync.
    - **Plans.** The help center says skills are available on Free, Pro, Max, Team and Enterprise, and the pricing table marks Skills Yes for Free; the claude.com skills overview lists Pro, Max, Team and Enterprise only.
    - **Where to switch them on.** Claude 101 points to Settings > Capabilities; the current help center uses Customize > Skills, after turning on code execution in Settings > Capabilities.

## Memory, styles and personalization

*Tested in: CCAO-F 3.4, 5.3, 5.4, 6.2 (the guide's How to Prepare list also names Memory) · CCDV-F Claude Application Design*

Three different things get called "memory" in the Claude apps. Objective 3.4 asks you to "manage context limitations and memory considerations" ([CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)), so keep them apart. The context window is what Claude can see inside one conversation: "The model doesn't learn from your corrections. It only responds to what's currently in context" ([AI Capabilities and Limitations](https://academy.claude.com/courses/ai-capabilities-and-limitations/working-memory)). Project knowledge is background for every chat in one project; chats in a project do not share context unless it goes into that knowledge or, with memory on, into that project's own memory. The memory feature carries saved facts from one conversation to the next; in the Academy's words, "Written memory is how Claude keeps important details in context across multiple conversations" ([Parametric memory and context](https://academy.claude.com/tutorials/parametric-memory-and-context)). Instructions sit on top of all three and shape how Claude responds. This section covers personal and organization instructions, the memory feature, chat search, incognito chats and what happened to styles. The restart, summarize or persist decision table is under Context limits in [Choosing a model in the apps](#choosing-a-model-in-the-apps), and project instructions are in [Projects](#projects).

### Instructions for Claude (personal)

"Instructions for Claude" are account-wide: "Any instructions you add here will be applied to all of your conversations with Claude" ([personalization features](https://support.claude.com/en/articles/10185728-understanding-claude-s-personalization-features)). The same article's rule of thumb: use profile instructions for account-wide settings, and use skills to customize how Claude formats and delivers its responses. The [skills overview](https://support.claude.com/en/articles/12512176-what-are-skills) draws the line the same way: custom instructions apply broadly to all your conversations, while skills are task-specific and load only when relevant.

The article's own examples of what to describe there: your preferred approaches or methods, common terms or concepts you use, typical scenarios you encounter, and general communication instructions.

Where to find them depends on the source. [Claude 101](https://academy.claude.com/courses/claude-101/your-first-conversation-with-claude) points to Settings > Account > 'Instructions for Claude'. In the new Claude experience, Cowork's former Global instructions setting "is now part of Instructions for Claude in Settings > General" ([Cowork and chat are one Claude](https://support.claude.com/en/articles/16761823-claude-cowork-and-chat-are-one-claude)). The effect is the same either way: one set of standing preferences for every conversation.

What belongs there, per the Cowork course ([Giving Cowork context](https://academy.claude.com/courses/introduction-to-claude-cowork/giving-cowork-context)): who you are and what you do, the shorthand and acronyms you use, and how you like output delivered (format, length, tone). Corrections you keep repeating, such as "share the bottom line up front in your responses" or "don't use Oxford commas", are what the course calls "global-instruction candidates". A writing voice that should hold across many drafts is handled differently: the Academy's [voice use case](https://academy.claude.com/use-cases/my-voice) builds a personal /my-voice skill in Cowork and says to "feed the correction back" each time you edit a draft, so that "The skill accumulates your corrections as rules and your edited drafts as reference examples". That matches the help center pointing format and delivery preferences at skills.

Neither Claude 101 nor the personalization article ranks personal instructions against project instructions. Claude 101 says only that project instructions "work alongside any user preferences and styles you've set" ([Introduction to projects](https://academy.claude.com/courses/claude-101/introduction-to-projects)), and the [personalization article](https://support.claude.com/en/articles/10185728-understanding-claude-s-personalization-features) says the features can be used "independently or in combination". The precedence rule the help center does state is organization over individual, below.

### Organization instructions (Team and Enterprise)

| Fact | Value |
|---|---|
| Who can set them | Owners and Primary Owners on Team and Enterprise plans |
| Where | Organization settings > Organization and access |
| Maximum length | 3,000 characters |
| Time to take effect | Up to an hour across Claude products |
| Scope | Included in every message sent by everyone in the organization |
| Conflict with an individual instruction | "If an individual instruction directly contradicts an organization instruction, Claude favors the organization-level instruction." |
| No conflict | "Individual instructions still apply for anything the organization instructions don’t address." |
| How the precedence works | Prompt-level: "In rare edge cases involving directly contradictory instructions, behavior may vary." |
| What they cannot do | Disable Claude's built-in safety guidelines or content policies |

The help center gives six example instructions ([Set organization instructions](https://support.claude.com/en/articles/14546867-set-organization-instructions)). Four of them, copied as written, cover a house style, a length and format default, shared vocabulary and a data-handling reminder (the other two set a team identity and a referral rule):

```text
Communication standards. “Respond in formal English. Use active voice. Avoid contractions and emojis.”
Response formatting. “Prefer concise responses under 300 words. Use bullet points for lists with three or more items.”
Domain context. “Our team works in healthcare claims processing. When users mention ‘claims,’ they’re referring to insurance claims, not legal claims.”
Data handling reminders. “Don’t include customer names, account numbers, or other personally identifiable information in responses or generated artifacts.”
```

The same article's best practices are the checklist to apply when an item asks you to improve an instruction (objectives 5.3 and 5.4):

1. **Be specific.** Replace a vague instruction such as "be professional" with concrete direction.
2. **Keep them short.** They ride along with every message in the organization.
3. **Focus on consistent behaviors.** Organization instructions "work best for instructions that should apply uniformly across every conversation", such as formatting standards, tone requirements or organization-wide context.
4. **Avoid contradictions.** "If your organization instructions contradict each other, Claude may not follow either one reliably."
5. **Do not try to override safety behaviors.** They cannot disable the built-in safety guidelines or content policies.
6. **Test, then maintain.** Start a new conversation and try several types of questions after saving; revisit them as needs change, because "Removing outdated instructions keeps Claude’s responses focused."

**Decide (our rule, from the sources cited).** An instruction is guidance, not enforcement: the help center says precedence "relies on prompt-level instructions". If a behavior should usually happen across the organization, an organization instruction is proportionate. If something must never happen, use a control that removes the capability: a connector permission set to Blocked, a feature switched off, a site blocklist (see [Admin controls for Team and Enterprise](#admin-controls-for-team-and-enterprise)). The data-handling reminder above does not make it acceptable to upload regulated identifiers. In CCAO-F Sample 3 a spreadsheet holds customer names and account numbers, and the rationale says "instructing the model not to retain data (C) does not satisfy the policy control" ([CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)). The CCDV-F Sample 2 rationale makes the same point for developers: "a polite request (C) is not an enforceable control" ([CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)).

### Styles are deprecated

!!! warning "Older material vs current docs: styles"

    [Claude 101](https://academy.claude.com/courses/claude-101/your-first-conversation-with-claude) carries a note under its December 2025 video: the "Use style" menu it shows "has since been deprecated." The current [personalization article](https://support.claude.com/en/articles/10185728-understanding-claude-s-personalization-features) does not mention styles and sends format and delivery preferences to skills instead. A few articles still mention "custom styles" in passing: the [incognito chats](https://support.claude.com/en/articles/12260368-use-incognito-chats) FAQ, and the privacy center's text on how thumbs up / down feedback is stored. None of these articles describes how existing styles carry over, so do not assume a migration path. The CCAO-F guide's How to Prepare list names Projects, Artifacts, Memory, Skills and Code Execution, not styles.

    For tone and format today (our summary of the personalization article): account-wide preferences go in Instructions for Claude, preferences for one body of work go in project instructions, and a reusable format or voice becomes a skill.

### The memory feature

Memory is what lets a fact from Monday's chat show up in Thursday's. Its behavior as of September 2026:

| Question | Answer |
|---|---|
| Who has it on by default? | Free, Pro and Max on the web, Claude Desktop and Claude Mobile. On Team and Enterprise it is off by default and can be turned on by an owner. |
| How does Claude save? | As a set of individual topics while you chat, not by summarizing conversations after they end: "Mention that a deadline moved, and your next conversation already knows." You can also say "remember this". |
| How do I see or change it? | Settings > Memory shows everything Claude remembers, listed under Topics; select a topic to edit or delete it. You can also tell Claude in a chat what to remember, change or forget; the update applies to your next conversation. |
| What it remembers | Everyday context that helps it work with you, such as your role, projects and professional context; the people and places in your work and life; communication preferences and working style; technical preferences and coding style; project details and ongoing work |
| Projects | Each project has its own separate memory space and project summary. On Team and Enterprise plans that use memory, a chat started in the wrong project can be moved out with "Remove from project" so it counts toward non-project memory instead. |
| Cowork | Shared with Cowork sessions that run in the cloud, not with local ones; see [Where Cowork runs, and why it matters](#where-cowork-runs-and-why-it-matters) |
| Sensitive topics | Not stored by default (health, race, ethnicity, religious beliefs, politics, gender identity and similar). Opt in with Include sensitive topics in memory in Settings > Memory; saving then applies going forward, not retroactively. |
| Never saved, even on request | Government ID numbers, criminal history, financial account numbers and immigration status |
| Pause | Keeps existing memory, but Claude neither uses it nor saves new memories |
| Reset | Permanently deletes all memories, including project memories |
| A chat is deleted or expires | Related memory entries are not removed; you can delete individual memories at any time |
| Data exports | Memory entries are included |
| Import | Settings > Memory > "Start import." The import article lists Free, Pro, Max and Team, on the web and Claude Desktop; the getting-started article lists only Free, Pro and Max. The feature is experimental and imports may not always be absorbed. |

**Organization rules for memory.** On Team and Enterprise, Owners and Primary Owners decide whether memory is on, in Organization settings > Capabilities. Memory and sensitive topics are two separate organization controls, both off by default, and even when an organization allows sensitive topics nothing in those categories is saved until each user opts in. "Turning off memory at the organization level permanently deletes all memory data for everyone in your organization." Owners cannot view or edit a user's individual memories. Memory is not available to organizations with HIPAA, public-sector or custom data retention agreements.

**Decide.** Persist what is stable. In the Academy's worked example, "Your project, your preferences, and your working relationships made the cut", while ephemeral details such as an RSVP count and a catering quote did not: "Claude can and should look those things up next time they come up, rather than writing them down and hoping they haven't changed" ([Parametric memory and context](https://academy.claude.com/tutorials/parametric-memory-and-context)). And because the model does not learn from your corrections (above), a correction you want repeated has to be written somewhere that persists: memory, Instructions for Claude, project knowledge or a skill (our list, from the mechanisms in this section).

!!! warning "Exam guide vs current docs"

    The memory mechanism changed just after the CCAO-F guide was issued (the guide is "Effective July 2026" and its PDF was created on July 8, 2026). The [release notes](https://support.claude.com/en/articles/12138966-release-notes) record that on July 10, 2026 memory began working "as a set of individual, categorized entries that Claude reads and updates during your conversations, replacing the previous daily memory summary", and on August 25, 2026: "Memory now works across chat and Cowork in the cloud." The [memory article](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context) still keeps a section because "A small number of Team and Enterprise organizations still use the legacy memory from chats experience", whose synthesis "is updated every 24 hours". In that legacy experience the Enterprise organization-wide memory toggle is "enabled by default"; in the current one, memory is off by default for Team and Enterprise. Older study material may describe the daily summary. The CCAO-F objective itself is about the decision, in its words "when to restart, summarize, or persist", not about the mechanism, so answer in those terms.

### Chat search

Searching past chats is a paid-plan feature (Pro, Max, Team and Enterprise) on the web, Claude Desktop and Claude Mobile. The searches use retrieval-augmented generation and appear as tool calls in the conversation. Inside a project, a search covers only that project's conversations. A typical request from the help center: "Let's continue where we left off with [project]." Claude does not pull information from incognito chats when it searches, and on Enterprise plans that use customer-managed encryption keys, past chats cannot be searched because conversation content is encrypted.

The difference from memory is what comes back: memory holds topics Claude has already saved, while chat search retrieves from the past conversations themselves when you ask.

### Incognito chats

| Fact | Value |
|---|---|
| What it is | A temporary chat, not saved to chat history or to memory |
| Plans | All (Free, Pro, Max, Team, Enterprise) |
| How to start | Click the ghost icon |
| Training | Not used for training, even if Model Improvement is enabled in Privacy Settings |
| Retention | Still retained for 30 days by default, or longer under an Enterprise custom data retention setting |
| Organization visibility | Included in Team and Enterprise organization data exports and in the Compliance API (Enterprise). The incognito article calls the exports "available to account Owners", while the export article limits them to the Primary Owner; see [Oversight and data](#oversight-and-data) |
| Inside projects | Not available; the ghost icon does not appear in a project |
| Afterwards | Cannot be converted to a regular chat or saved to history; cannot be reopened once closed |
| Your existing memory | Not used: starting an incognito chat does not draw on Claude's existing memory, and the chat is not included in future memory entries |
| Your profile information | Still available: the FAQ says Claude can access profile information such as personal preferences in an incognito chat |
| New Claude experience | Opens in the previous chat experience, so Claude cannot create files or run code in it |

**Decide.** Incognito keeps a conversation out of your history and memory. It is not a data-protection control: the chat is still retained and can still appear in exports and the Compliance API. If policy restricts regulated personal data, the answer is the one in CCAO-F Sample 3 ([CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Remove or anonymize the personal identifiers before uploading, consistent with policy." In our reading, choosing incognito for that file is the product version of Sample 3's wrong option C ("Upload the file but instruct Claude not to retain it."): it limits what is kept afterwards without satisfying the policy before upload. Data handling in depth is in [Untrusted input, PII and data leakage](security-and-governance.md#untrusted-input-pii-and-data-leakage).

### Where a fact should live

Our summary table, built from the facts above:

| You want Claude to know... | Put it in | Because |
|---|---|---|
| Your role, preferred format and recurring corrections, in every chat | Instructions for Claude | Applied to all of your conversations |
| A deadline you mentioned, a preference that came up in passing | Memory ("remember this") | Saved as a topic and available in later conversations |
| A rule for everyone in the organization | Organization instructions (an Owner or Primary Owner sets them) | Sent with every message in the organization |
| An ephemeral detail, such as an RSVP count or a catering quote | Nowhere; have Claude look it up fresh | The Academy's advice is to look such details up rather than hope they have not changed |
| Nothing from this conversation | An incognito chat | Not saved to history or memory (but still retained) |

Reference documents for one body of work belong in project knowledge, and a repeatable procedure belongs in a skill; that comparison is in [Skills in the Claude apps](#skills-in-the-claude-apps).

### Traps

- **Tempting:** Claude will remember the correction I gave yesterday. **Actually:** The model does not learn from corrections; it responds only to what is in its current context. A correction carries forward only if it is saved somewhere that persists: memory, instructions, project knowledge or a skill.
- **Tempting:** Pausing memory deletes it. **Actually:** Pause keeps what is stored; Reset deletes, and Reset includes project memories.
- **Tempting:** My new teammate's Claude already knows our context. **Actually:** Memory is personal to each user, and on Team and Enterprise it is off by default until an owner turns it on. Shared context belongs in a project or in organization instructions.
- **Tempting:** The admin can read what Claude remembers about me. **Actually:** Owners cannot view or edit individual memories.
- **Tempting:** Deleting the chat removes what Claude remembered from it. **Actually:** Related memory entries are not removed; delete the individual memory in Settings > Memory.
- **Tempting:** Incognito is the safe way to analyze the customer file. **Actually:** Incognito chats are retained and exportable; anonymize first.
- **Tempting:** Our organization instruction guarantees no PII in outputs. **Actually:** Precedence is prompt-level and behavior may vary; enforcement needs controls, and regulated data needs anonymizing before upload.
- **Tempting:** Set a style so every reply is shorter. **Actually:** The "Use style" menu is deprecated; put the preference in Instructions for Claude, or package a reusable format as a skill.

## Claude in Chrome and other surfaces

*Tested in: CCAO-F 3.1, 4.4, 6.2 · CCDV-F Claude Application Design, AI Application Security · CCAR-F lists computer use (browser automation, desktop interaction) as out of scope*

Claude is not only a chat window at claude.ai. It runs in a desktop app, on phones, inside Chrome, inside Microsoft 365 and in Slack. Usage is pooled across the main surfaces: "your usage of all different Claude product surfaces (claude.ai, Claude Code, Claude Desktop) counts towards the same usage limit" ([usage and length limits](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work)). Claude Tag channel work is the exception covered below: it is billed to the organization. The choice this section helps with is the one [Claude 101](https://academy.claude.com/courses/claude-101/claude-desktop-app-chat-cowork-code) frames for the desktop app: "Turn-by-turn work happens in Chat. Work you hand off runs in Cowork. Building software happens in the Code tab." Cowork has its own section, [Claude Cowork](#claude-cowork).

### Claude in Chrome

The core facts, from [Get started with Claude in Chrome](https://support.claude.com/en/articles/12012173-get-started-with-claude-in-chrome) and [Use Claude in Chrome safely](https://support.claude.com/en/articles/12902428-use-claude-in-chrome-safely), as of September 2026:

| Fact | Value |
|---|---|
| What it is | "a browser extension that allows Claude to read, click, and navigate websites alongside you" |
| Plans | All paid plans (Pro, Max, Team, Enterprise); the pricing page lists it as a Pro addition over Free |
| Where it runs | In Claude Cowork and Claude Code, and in beta in the Chrome side panel |
| Side panel | On Max and Team plans the side panel runs as a Cowork session, rolling out to Pro; on Enterprise, once an admin has enabled Cowork in the cloud (until then, the classic side panel). Cowork side panel sessions are saved to your history and can be picked up on other surfaces. |
| Browser for Cowork on desktop | Cowork also has a browser built into Claude Desktop (rolling out as of September 2026). If you already use Claude in Chrome, it stays your preferred browser; choose in Settings > Cowork > Preferred browser |
| Not supported | Other Chromium-based browsers, and mobile devices |
| How it sees a page | It takes screenshots of the tabs it is working in; whatever is visible in those tabs becomes part of the conversation |

**Permission modes.** The extension side panel and Claude Desktop offer three modes in a drop-down on the chat input, the same three Cowork uses: Manually approve, Automatically approve and Skip all approvals, with their former names and general behavior in the Cowork section's [Permission modes](#permission-modes). What the [permissions guide](https://support.claude.com/en/articles/12902446-claude-in-chrome-permissions-guide) adds for Chrome:

- **Manually approve.** In the classic side panel, Claude first proposes a plan (sites and approach) for you to approve; in the Cowork side panel it asks before each action (Allow all for this website, Allow this time only, or Deny).
- **Automatically approve.** It is the default in the Cowork side panel, and it uses more of your usage limit.
- **Skip all approvals.** Nothing checks Claude's actions: "Only use this when you completely trust every action, connector, file, app, etc. involved in the task."

**Hard limits in every mode.** Whatever the mode, the permissions guide says Claude still needs your explicit permission to modify permission settings, grant authorizations, or input potentially sensitive information into websites. Even on a site you set to "Always allow actions on this site", it still asks before downloading a file, entering potentially sensitive information into a page, or granting authorizations. And it is prohibited, regardless of permissions, from:

- Making purchases or financial transactions
- Creating accounts
- Handling sensitive credit card or ID data
- Downloading files from untrusted sources
- Permanent deletions (emptying trash, deleting emails, files, or messages)
- Providing investment or financial advice
- Executing financial trades or investment transactions
- Modifying system files
- Completing instructions from emails or web content

The [safety article](https://support.claude.com/en/articles/12902428-use-claude-in-chrome-safely) adds that Claude cannot access high-risk site categories such as adult content and known pirated content, asks for permission before accessing financial sites, and is prohibited from stock trading or investment transactions, bypassing captchas, inputting sensitive data, and gathering or scraping facial images.

**The risk to design around.** The help center names it plainly: "The biggest risk facing browser-using AI tools is prompt injection attacks where malicious instructions hidden in web content (websites, emails, documents) could trick Claude into taking unintended actions" ([use Claude in Chrome safely](https://support.claude.com/en/articles/12902428-use-claude-in-chrome-safely)). Safety classifiers screen for it automatically: one checks incoming content for injection attempts and another checks every action before it runs, blocking or pausing it when a risk is flagged. The article is still explicit: "The risk is not zero." Because Claude works from screenshots, it "can’t filter sensitive content out of what it sees", so Anthropic recommends not using it on sensitive sites and considering a separate browser profile without access to sensitive accounts. Claude in Chrome is not available to organizations covered by HIPAA, and the article recommends against using it on pages that contain regulated data.

Three published figures show why the limits exist. They come from different evaluations and different models, so do not read them as one trend. In Anthropic's August 2025 pilot, 123 test cases across 29 attack scenarios produced a 23.6% attack success rate without safety mitigations, reduced to 11.2% with them in autonomous mode ([Claude for Chrome](https://claude.com/blog/claude-for-chrome)). A November 2025 research post adds that even a 1% rate "still represents meaningful risk" and that "No browser agent is immune to prompt injection" ([prompt injection defenses](https://www.anthropic.com/research/prompt-injection-defenses)). The current safety article, as of September 2026, says Anthropic's testing shows Claude Opus 4.8 resisting prompt injection significantly better than previous models, and that "Our current configuration reduces attack success rates to less than 0.08% against our internal testing that combines known effective attack techniques" ([use Claude in Chrome safely](https://support.claude.com/en/articles/12902428-use-claude-in-chrome-safely)), while repeating that the chances of an attack are "still non-zero".

**Admin controls (Team and Enterprise).** Owners and Primary Owners manage the extension in Organization settings > Claude in Chrome ([Claude in Chrome admin controls](https://support.claude.com/en/articles/13065128-claude-in-chrome-admin-controls)).

| Control | Detail |
|---|---|
| Organization toggle | Team: enabled by default. Enterprise: disabled by default, but "Starting September 10, 2026, it turns on by default unless you've already disabled it." |
| Allowlist | Restricts Claude to approved sites only |
| Blocklist | Prevents access to specific sites, regardless of user permissions, on top of Claude's default blocked categories |
| Rollout advice | "Start with a more restrictive allowlist for the security of your organization's data, then expand access over time as you become comfortable with the extension's behavior." |
| Running a pilot | Enable the extension, set a restrictive allowlist of trusted sites, use IT controls to limit who can install it, share the safety article with pilot users, then gather feedback and expand |
| Deployment | Self-service from the Chrome Web Store, or managed deployment through existing Chrome management tools (Google Workspace admin console or MDM) |
| Zero data retention | Not supported for Claude in Chrome, the same as Cowork |
| Relationship to Cowork | "Claude in Chrome and Claude Cowork are managed separately." Whether Claude can use the extension inside Cowork is a separate capability setting, and Claude in Chrome does not inherit a user's Cowork access. On Enterprise, a per-role capability applies to organizations using custom roles |

**Decide (our rules, each resting on the source named).**

- If a task touches banking, health, legal or other regulated pages, keep Claude in Chrome out of it; Anthropic's [launch post](https://claude.com/blog/claude-for-chrome) recommended avoiding sites that involve financial, legal, medical or other sensitive information, and the safety article strongly advises against managing financial accounts, legal documents, medical information or sensitive company data with it.
- If a connector exists for the app, prefer it over the browser. In Cowork, "Claude uses the most precise tool first": connectors, then the browser, then screen interaction ([computer use in Cowork](https://support.claude.com/en/articles/14128542-let-claude-use-your-computer-in-cowork)).
- If you are rolling it out to an organization, start with a restrictive allowlist and widen it as confidence grows.
- If speed matters and the site is trusted, Auto is the middle ground; Skip is only for tasks where you trust everything involved. For work with real consequences (money, messages sent as you, important files), the permissions guide says to stay close or switch back to Manually approve.
- If an item asks how to stop injected page text from triggering actions, the answer is a control, not a request. CCDV-F Sample 2 (a page with hidden instructions) is answered by "isolating untrusted content from trusted instructions and enforcing least-privilege guardrails so injected text cannot invoke sensitive tools" ([CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). Claude in Chrome's fixed prohibition on "Completing instructions from emails or web content" and its per-action classifier are the product version of that answer (our mapping).

!!! warning "Exam guide vs current docs"

    None of the four July 2026 exam guides names Claude in Chrome; for CCAO-F the related objectives are 3.1 (selecting product features) and 6.2 (data sensitivity, regulatory and privacy considerations), and the [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) appendix lists "Computer use (browser automation, desktop interaction)" as out of scope. The product has kept moving: the Enterprise default switched to on from September 10, 2026, and the current help center describes the side panel running as a Cowork session with Automatically approve as its default. Sources also disagree on status: [Claude 101](https://academy.claude.com/courses/claude-101/other-ways-to-work-with-claude) says "Claude in Chrome is generally available on all paid plans", while the [help center](https://support.claude.com/en/articles/12012173-get-started-with-claude-in-chrome) says it is available "in beta in the Chrome side panel". Answer exam items from the stable principles: prompt injection risk, sensitive sites, permissions and allowlists.

### Claude Desktop, mobile and voice

| Surface | What to know |
|---|---|
| Claude Desktop | macOS, Windows and Linux (beta). Requires macOS 11 (Big Sur) or higher, Windows 10 or higher, or Ubuntu 22.04 LTS+ or Debian 12+ on x64 or arm64. Cowork runs in it on all paid plans. |
| Desktop extensions | Local MCP servers packaged as single-click installs; they run locally and work only in Claude Desktop and Claude Code, not on web or mobile (see [Search, Research and connectors](#search-research-and-connectors)) |
| Quick entry (Mac) | All plans, Claude Desktop on macOS: double-tap Option to open Claude from any app; press Caps Lock to dictate |
| Claude for iOS and Android | iOS and iPadOS 18.0 and above; Android 8.0 Oreo and above |
| Voice mode | Beta on all plans, on mobile, desktop and web, "built to work best from your phone" ([voice mode](https://support.claude.com/en/articles/11101966-use-voice-mode)) |
| Dictation vs voice mode | Dictation converts speech to text for a typed prompt; voice mode is a two-way spoken conversation |
| Dictation audio (mobile) | Deleted after conversion to text; not retained or used for training |

### Claude for Microsoft 365

Claude for M365 is a set of Claude add-ins that work inside Microsoft 365 apps. Status as of September 2026: Claude for Excel, PowerPoint and Word are generally available on paid plans, and Claude for Outlook is in beta. One conflict to know: the claude.com docs call Word generally available, while the help center's beta features table (last modified July 7, 2026) still lists "Claude for Word" as Beta. Skills enabled in your Claude settings also work in the add-ins; see [Skills in the Claude apps](#skills-in-the-claude-apps).

Claude for Excel carries the add-in guidance closest to CCAO-F objective 2.4 (when human review is required), because its docs say plainly where it should and should not be trusted ([Claude for Excel](https://claude.com/docs/office-agents/excel.md)):

| What it does | Not recommended for |
|---|---|
| Answers questions about the workbook with cell-level citations you can click | "Final client deliverables without human review." |
| Adjusts assumptions while keeping formula relationships intact | "Audit-critical calculations without verification." |
| Identifies and resolves errors and their root causes | "Models containing highly sensitive or regulated data without proper controls." |

The same page lists two unsupported capabilities, data tables and macros and VBA operations, and several Excel versions the add-in does not run on, including Excel 2016 and 2019 perpetual or volume licenses and Excel on iPad or Android.

Data and governance facts from the same page: Claude for Excel stores chat history locally in the browser (IndexedDB), not on Anthropic's servers and not synced across devices; it does not inherit an organization's custom data retention settings, and its activity is not included in Enterprise audit logs, although Enterprise organizations with the Compliance API enabled get its sessions there (coverage in public beta). Its docs also say to use it only with trusted spreadsheets, because files from external sources "can contain hidden instructions that manipulate the add-in into extracting data, modifying records, or performing destructive actions."

For working across files, you turn on "Let Claude work across files" in each add-in's settings. The toggle is per-device; it is on by default for Pro and Max and off by default for Team and Enterprise, where owners also control access with the organization-wide "Let Claude work across apps" setting (Organization settings, Office agents). Cross-app work is available only when you sign in with your Claude account directly, not through Amazon Bedrock, Google Cloud Vertex AI, Azure AI Foundry or an LLM gateway.

The Microsoft 365 connector is a different thing: it lets Claude search SharePoint, OneDrive, Outlook and Teams content from a chat, and is covered with the other connectors in [Search, Research and connectors](#search-research-and-connectors).

### Claude Tag in Slack

As of September 2026, Claude Tag is available in beta on Team and Enterprise plans and works in Slack. Claude in Slack switched over to Claude Tag on August 3, 2026, after the July 2026 exam guides. Billing depends on where you ask ([What is Claude Tag](https://support.claude.com/en/articles/15594475-what-is-claude-tag)): "Tagging Claude in a channel is billed to your organization. Direct messages are billed to your own Claude account instead." Enterprise custom data retention periods do not apply to Claude Tag.

| Where you use it | Whose identity and access | Billed to | Notes |
|---|---|---|---|
| Tagging @Claude in a channel | The organization's identity, with the tools and access an admin set up for that channel | The organization | The whole exchange stays visible to everyone in the channel, and anyone can steer it |
| Direct message (a private conversation with @Claude) | The capabilities you have enabled in your own Claude account, such as web search and connected tools | Your own Claude account | The assistant panel also uses your own account's capabilities |

Only a Primary Owner or Owner can set up Claude Tag's access and channels (the Admin role cannot). Unlike personal memory, which owners cannot view, Claude Tag keeps context per channel and per workspace, and admins can view, edit and delete that memory. Slack conversations with Claude stay separate from your Claude chat history.

### Traps

- **Tempting:** Use Claude in Chrome on your phone. **Actually:** It is not supported on mobile devices or on other Chromium-based browsers.
- **Tempting:** Grant Skip mode so Claude can finish the checkout. **Actually:** Purchases are prohibited in every mode.
- **Tempting:** Claude will follow the instructions in that email for you. **Actually:** Completing instructions from emails or web content is prohibited; treat such text as untrusted.
- **Tempting:** With Auto mode's safety checks, prompt injection is solved. **Actually:** The safety article says the risk is not zero; keep sensitive sites out of scope and restrict sites with an allowlist.
- **Tempting:** Once a site is set to always allow, Claude can download files there freely. **Actually:** It still asks before downloading a file, entering potentially sensitive information or granting authorizations.
- **Tempting:** Enabling Claude in Chrome also lets Cowork use it. **Actually:** The two are managed separately.
- **Tempting:** The Excel add-in follows our retention policy. **Actually:** It does not inherit custom data retention settings, and its history sits in the browser.
- **Tempting:** Ask the Excel add-in to rewrite the VBA macro. **Actually:** Macros and VBA operations are unsupported.
- **Tempting:** A desktop extension will work on the web too. **Actually:** Desktop extensions run locally and only in Claude Desktop and Claude Code.

## Claude Cowork

*Tested in: CCAO-F 1.4, 3.1, 4.2, 4.4, 7.3 (the guide's candidate profile names "workflow-based interactions") · CCDV-F Claude Application Design*

Cowork is where you hand Claude a whole piece of work instead of working turn by turn. It "uses the same agentic architecture that powers Claude Code, with no terminal required" ([Get started with Claude Cowork](https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork)). You describe an outcome, step away and come back to finished work such as formatted documents, organized files or synthesized research. Cowork is available on paid plans only (Pro, Max, Team, Enterprise): in Claude Desktop on all paid plans, and in beta on the web and mobile for Pro, Max and Team, and for Enterprise where an owner has enabled it ([Use Claude Cowork safely](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely)).

### Chat, Cowork or something else?

Our mapping, built from Claude 101's three-way split and the help articles cited in this section:

| You need | Use | Why |
|---|---|---|
| A quick answer, or drafting back and forth | Chat | Turn-by-turn work happens in Chat |
| A finished deliverable built from several inputs while you do something else | Cowork | Describe the outcome and return to finished work |
| The same job every morning or every Monday | A Cowork scheduled task | Runs on a schedule, remotely, without your computer awake (see the local-files caveat under Scheduled tasks) |
| Work inside desktop apps that have no connector | Cowork with computer use (beta, Pro and Max only) | Claude tries connectors, then the browser, then the screen |
| Building software | The Code tab (Claude Code) | Building software happens in the Code tab |
| A cited, multi-source investigation | Research (see [Search, Research and connectors](#search-research-and-connectors)) | Many searches that build on each other |

The Cowork course gives the recognition test for delegation: "Three patterns cover most of the work Cowork is built for", namely tasks that take several steps, tasks that draw on context from real files, and tasks that span the tools you already use; its own shorthand is "multi-step, file-producing, or tool-using" ([Scheduled tasks lesson](https://academy.claude.com/courses/introduction-to-claude-cowork/scheduled-tasks)). The same course's one-line recap: "Chat is for thinking with Claude. Cowork is for delegating to Claude. Code is for building software with Claude." ([What is Cowork](https://academy.claude.com/courses/introduction-to-claude-cowork/what-is-cowork)). Anthropic's rollout guidance names the failure mode on the other side: "The chat trap. Users default to short prompts with no connectors. Enablement must demo delegation side-by-side with chat" ([Scaling workflows with Claude Cowork](https://academy.claude.com/tutorials/scaling-workflows-with-claude-cowork-at-your-organization)).

### Where Cowork runs, and why it matters

| Session type | Where the work runs | What follows from it |
|---|---|---|
| Cloud (beta) | Anthropic's infrastructure, in an isolated, temporary environment created for the session | Work continues across desktop, web and mobile; scheduled tasks run with the laptop closed; memory is shared with chat; sessions and files are saved to the member's Claude account. Tasks that need local files, the browser or computer use still need Claude Desktop open and connected |
| Local | The user's computer, with code in an isolated virtual machine | Local sessions do not use the memory shared with chat (Cowork project memory is separate and scoped to its project); conversation history is stored on the user's computer |

The governance consequence is on the local side. Local Cowork history is stored on users' computers and, per [Use Claude Cowork on Team and Enterprise plans](https://support.claude.com/en/articles/13455879-use-claude-cowork-on-team-and-enterprise-plans), "is not subject to Anthropic's standard data retention policies, and admins cannot centrally manage or delete it." The same article adds that Enterprise admins can retrieve this content through the Compliance API, but deletion endpoints for local sessions are not available yet. Separately, "Cowork is not yet covered under Anthropic’s BAA" for HIPAA-ready organizations ([HIPAA-ready Enterprise plans](https://support.claude.com/en/articles/13296973-hipaa-ready-enterprise-plans)).

Three more governance facts, as of September 2026:

- **Monitoring.** Per the [Team and Enterprise article](https://support.claude.com/en/articles/13455879-use-claude-cowork-on-team-and-enterprise-plans), Cowork sessions on Claude, Claude Desktop and Claude Mobile are captured in the Compliance API (an Enterprise feature). Team and Enterprise owners can also stream Cowork events to SIEM and observability tools through OpenTelemetry, which "doesn't replace audit logging for compliance purposes."
- **Network egress.** Cowork respects the organization's network egress permissions, applied when a session is created, so a mid-conversation change needs a new conversation. Those permissions do not apply to web fetch, web search or MCPs, including Claude in Chrome; owners turn web search and Claude in Chrome off with their own settings.
- **Isolation is not permission.** Per the [safety article](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely), cloud isolation "doesn't change what Claude can read or do through the access you've granted".

### Permission modes

Cowork asks before it acts according to a mode you choose in the chat box, and you can change it at any time ([Get started with Claude Cowork](https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork)). Deletion is protected in every mode: "Claude requires your explicit permission before permanently deleting any files."

| Mode | Former name | What happens |
|---|---|---|
| Manually approve (Manual) | "Ask before acting" | Claude pauses and asks for approval for actions; you choose Allow or Deny |
| Automatically approve (Auto) | None | Claude keeps working, reviews each action for safety (such as data exfiltration or prompt injection) and blocks what it judges unsafe; when blocked it looks for a safer way or asks you, and if it keeps running into blocks it switches back to asking for each step. It uses more of your usage limit because of this checking |
| Skip all approvals (Skip) | "Act without asking" | Claude does not pause and nothing checks its actions automatically |

The mode interacts with each connector tool's own permission. The help center's matrix ([Get started with Claude Cowork](https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork)):

| Mode | Tool set to "Always allow" | Tool set to "Needs approval" | Tool set to "Blocked" |
|---|---|---|---|
| Manual | Approved | Asks for permission | Denied |
| Auto | Read-only tools are approved; for write or delete tools, Claude decides | Claude decides | Denied |
| Skip | Approved | Approved | Denied |

Read the last column first: a Blocked tool is denied in every mode, which is why (in our reading) blocking is the control and a mode is only a preference. On Team and Enterprise, two organization settings under Permissions in Organization settings > Cowork sit above both ([Use Claude Cowork on Team and Enterprise plans](https://support.claude.com/en/articles/13455879-use-claude-cowork-on-team-and-enterprise-plans)):

- **Allow "Automatically approve" mode** is on by default; when an admin turns it off, the mode disappears from members' mode selector.
- **Allow "Always allow" for connector tools** is off by default. While it is off, members approve write-capable connector tools per task and saved always-allow preferences for write tools are not honored. Read-only tools are exempt only when the connector annotates them as read-only, and most custom connectors do not, so every tool on those connectors is gated. On Enterprise this works alongside custom role grants, and the most restrictive layer wins.

**Decide.** The safety article gives the rule for choosing: switch to Manually approve when "The task touches sensitive files, accounts, or sites", when "You're working with a new tool, plugin, or site for the first time", or when "Mistakes would be hard to undo, like sending messages or making purchases." It also explains why: "For prompt injection attacks to be successful, two things must be true at the same time: Claude can read information outside your trusted boundary, and can perform actions that could compromise the user" ([Use Claude Cowork safely](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely)). Remove either condition (untrusted input, or consequential actions) and, in the article's words, "prompt injection attacks become more difficult". The same article sorts tools into read tools and write tools and says write tools "inherently carry more risk", which is why Cowork treats them differently.

### Giving Cowork a good brief

The one-Claude help article lists what a request should state ([Claude Cowork and chat are one Claude](https://support.claude.com/en/articles/16761823-claude-cowork-and-chat-are-one-claude)):

- **The desired outcome**: what you want to end up with, for example a one-page summary or a spreadsheet with a tab for each region.
- **The format**, such as a Word document, a slide deck, or a message you can paste into Slack.
- **The inputs**: "the files, links, or apps Claude should work from."

Two additions from the Academy: name the deliverable precisely, because "Specifics about format and length save you a regenerate" ([The task loop](https://academy.claude.com/courses/introduction-to-claude-cowork/the-task-loop)), and ask Claude to flag what it is not confident about (step 4 of the [verification routine](#a-verification-routine)). A brief in that shape is the prompt from Anthropic's status-report use case ([Generate project status reports](https://academy.claude.com/use-cases/generate-project-status-reports)), whose point is "being specific about what you need tracked and where to look". It names the inputs, the fields and the deliverable format. The last line is the use case's own missing-information tip, appended here:

```text
I need to consolidate project status from multiple sources into a task tracker.
Pull information from:
- Gmail (past 2 weeks, search "Project Hermes")
- Slack #hermes-sprint channel
- Google Drive "Project Hermes" folder
- Recent calendar meetings
For each task, I need to see:
- Who owns it and what they're working on
- Current status (not started, in progress, blocked, done)
- Any blockers and how long they've been stuck
- Notes from their updates about plans and challenges
Create an Excel tracker and include these features: visual status indicators, cell comments with context from sources (so I can hover and see the details), dropdown menus for status and priority (to make updates easy), and data bars showing progress visually.
The tracker should make it obvious at a glance where the problems are and who needs help.
If you don't find information for a work stream, note that explicitly rather than omitting it.
```

The last line matters for review: the use case says it is better to say "no progress documented" than to have gaps silently smoothed over.

While it runs, steer rather than wait and regenerate. The [task loop lesson](https://academy.claude.com/courses/introduction-to-claude-cowork/the-task-loop): "Cowork is built for course corrections, and the cost of a redirect is low." Watch the plan and progress, and interrupt if it goes off track (wrong source, wrong format, wrong tone); if a run is substantially off, the lesson also allows stopping it, refining the prompt and starting again. If a draft is mostly right, say what to change; if it is wrong in a load-bearing way, "the prompt was missing the load-bearing piece of context", so point Claude to that context. You stay in control throughout: Claude shows its plan before it starts and by default asks before actions that matter (sending, deleting, sharing) ([What is Cowork](https://academy.claude.com/courses/introduction-to-claude-cowork/what-is-cowork)).

### Context that persists: instructions and projects

The Cowork course separates two kinds of carry-over ([Giving Cowork context](https://academy.claude.com/courses/introduction-to-claude-cowork/giving-cowork-context)). Claude's memory "applies in Cowork tasks too" (for cloud sessions; local sessions do not use the memory shared with chat, see [Memory, styles and personalization](#memory-styles-and-personalization)), but "The context that reliably carries from task to task, though, is the context you set up deliberately": global instructions and projects. Outside a project, "each session starts fresh apart from your global instructions". The course names those two; the help center adds folder instructions for local folders:

| Where | What it holds |
|---|---|
| Global instructions | Standing instructions for every session: tone, output format, background on your role. Set in Settings > Cowork; in the new Claude experience they are part of Instructions for Claude in Settings > General |
| Folder instructions | Project-specific context when you select a local folder on desktop; Claude can update them during a session |
| Cowork projects | Workspaces that group related tasks "with their own files, context, instructions, and memory" ([Cowork projects](https://support.claude.com/en/articles/14116274-organize-your-tasks-with-projects-in-claude-cowork)), plus project-specific scheduled tasks. Memory is scoped to the project, so what Claude learns in one does not carry over to others. New projects are saved to your Claude account; projects created from a folder on your computer stay on that computer. On Team and Enterprise, owners cannot restrict project creation |

Plugins bundle skills, connectors and sub-agents into one package for a role or team; they are covered in [Skills in the Claude apps](#skills-in-the-claude-apps). The Cowork safety article adds the risk side: installing one "can significantly expand Claude's scope of action", and local MCP servers bundled with plugins and desktop extensions run on your computer with the same permissions as any other program, so stick to verified extensions and check what each one requests. On Team and Enterprise, plugins are controlled by the same admin toggle as Cowork.

### Scheduled tasks

From [Get started with Claude Cowork](https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork), [Schedule recurring tasks](https://support.claude.com/en/articles/13854387-schedule-recurring-tasks-in-claude-cowork) and [Use Claude Cowork safely](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely):

| Fact | Value |
|---|---|
| How to create one | Type /schedule in any Cowork task, or use Scheduled in the left sidebar; the course's "more natural path" is to do the task once, confirm the output, then schedule that exact process |
| What you set | Task name, prompt, approval mode, cadence, and optionally the model and a working folder |
| Where it runs | Remotely, so it runs on its cadence even when the computer is asleep or the desktop app is closed. Each run is its own Cowork session. The article is not fully consistent about local files: its overview says scheduled tasks "can't be tied to a folder on your computer", while its manual set-up steps say a task needing local files or apps "will only run locally" |
| Cadence (help center) | Hourly, daily, weekly, on weekdays, or manually |
| Safety: start small | "Begin with low-risk tasks like generating summaries or compiling information before automating anything more complex." |
| Safety: what to avoid | "Don't schedule tasks that access sensitive files, send messages on your behalf, make purchases, or take other actions that are difficult to undo." |
| Safety: afterwards | Review outputs after each run, and pause or delete tasks you are not actively using |
| Accountability | "You remain responsible for all actions taken by Claude performed on your behalf", including actions taken by scheduled tasks |

The cadence list varies by source: the [Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/scheduled-tasks) lists "hourly, daily, weekdays, or manual", and an [Academy tutorial](https://academy.claude.com/tutorials/scaling-workflows-with-claude-cowork-at-your-organization) lists "hourly, daily, weekly, weekdays, or on-demand".

### Computer use and Dispatch

From [Let Claude use your computer in Cowork](https://support.claude.com/en/articles/14128542-let-claude-use-your-computer-in-cowork) and [Assign tasks from anywhere](https://support.claude.com/en/articles/13947068-assign-tasks-from-anywhere-in-claude-cowork):

- Computer use is in beta and "Available for Pro and Max plans only"; Team and Enterprise plans do not have it at this time. It works in Cowork and Claude Code in Claude Desktop on macOS and Windows, and the computer must be awake with the desktop app open.
- Claude uses it last: connectors first, then the browser, then screen interaction, because screen interaction is slower and more error-prone.
- "Computer use has no sandbox between Claude and your applications." Claude asks before accessing each application, some sensitive apps (investment and trading platforms, cryptocurrency) are blocked by default, and you can add apps to a blocklist so requests to use them are denied automatically.
- Claude is trained to avoid stock trading, inputting sensitive data and scraping facial images, but the article says these guardrails "aren't absolute. Don't rely on them as a substitute for blocking access to sensitive apps."
- Dispatch lets you message Claude from your phone and have it work on your desktop computer, but it "isn't available to new users"; existing users can keep using it for now.

### Cowork on Team and Enterprise

The organization-level picture as of September 2026, from [Use Claude Cowork on Team and Enterprise plans](https://support.claude.com/en/articles/13455879-use-claude-cowork-on-team-and-enterprise-plans); the toggles live in Organization settings > Cowork:

| Setting | Team | Enterprise |
|---|---|---|
| Cowork itself | On by default; owners can disable it | On by default; owners can disable it, and groups and custom roles can enable it for specific teams |
| "Run Cowork in the cloud" | On by default; an owner can turn it off | Off by default; an owner turns it on, then grants the capability to a group with custom roles |
| Allow "Automatically approve" mode | On by default | On by default |
| Allow "Always allow" for connector tools | Off by default | Off by default; custom role grants cannot override it |
| Built-in browser | On by default as it rolls out | Off by default at launch; "turns on by default starting September 10, 2026, unless you've turned it off" |
| Plugins | Same toggle as Cowork; owners can run plugin marketplaces (installed by default, available, required, or not available) | Same, plus overrides for specific groups |
| Projects | No separate admin control | No separate admin control |
| Computer use | Not available | Not available |

### Rolling out delegated work

The Academy's guidance on handing work to agents points one way: widen autonomy on evidence.

- "Grant autonomy in proportion to demonstrated reliability, then expand it deliberately" ([Building effective human-agent teams](https://academy.claude.com/courses/building-effective-human-agent-teams/what-a-strong-team-looks-like)).
- Start with one visible job: "For the first few days, a person should review the briefing and give the agent feedback on its effectiveness" ([Practical ways to get started](https://academy.claude.com/courses/building-effective-human-agent-teams/practical-ways-to-get-started)).
- "After the agent has handled one kind of task well several times in a row, widen what it may do on that task" (same lesson).
- "AI changes how the work gets produced; accountability for it stays with your people" ([Scaling workflows with Claude Cowork](https://academy.claude.com/tutorials/scaling-workflows-with-claude-cowork-at-your-organization)).

!!! warning "Exam guide vs current docs"

    The [CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) (July 2026) names "Projects, research mode, chat, artifacts" as the features to choose between and never names Cowork, and Claude 101 teaches Chat, Cowork and the Code tab as three places in the desktop app. On September 16, 2026 Anthropic announced "Starting today, Claude Cowork and chat are merging into one Claude" ([Cowork is now Claude](https://claude.com/blog/cowork-is-now-claude)). The new experience is "rolling out gradually, starting with Pro and Max plans on web, desktop, and mobile"; the blog says Team and Free plans will follow soon and Enterprise admins get at least 30 days' notice, and the Team and Enterprise article says those organizations keep chat and Cowork as they are for now. In the new experience, Claude decides which tool a request needs, there is no web search toggle, Research starts with /deep-research (or "+" then Research), the permission setting offers Auto and Manual (Manual is the default), and an account that has the new experience cannot switch back to separate Chat and Cowork options ([Claude Cowork and chat are one Claude](https://support.claude.com/en/articles/16761823-claude-cowork-and-chat-are-one-claude)). Expect the guide's wording (chat, research mode) on the exam and answer in those terms; the underlying choice between turn-by-turn work and handed-off work is the same.

!!! note "Where Cowork runs: two descriptions"

    The Claude Desktop install article says "Cowork runs code in an isolated virtual machine on your computer" ([Install Claude Desktop](https://support.claude.com/en/articles/10065433-install-claude-desktop)), while the current [Cowork article](https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork) says Cowork "runs your tasks in the cloud (in beta)" and the Team and Enterprise article keeps local sessions as the second option. Both exist; which one a session uses decides memory, retention and whether the laptop must stay on.

### Traps

- **Tempting:** Set it to Skip so the weekly send-out runs unattended. **Actually:** Scheduled tasks that send messages on your behalf are exactly what the safety guidance says to avoid; keep consequential actions under Manual approval or out of the schedule.
- **Tempting:** Auto mode means no one is responsible. **Actually:** You remain responsible for everything Claude does on your behalf, scheduled tasks included.
- **Tempting:** Cowork reliably remembers what we did last week. **Actually:** Cloud sessions share your saved memory, but local sessions do not use the memory shared with chat, and the context that reliably carries from task to task is global instructions and projects; outside a project each session starts fresh apart from global instructions.
- **Tempting:** Admins can delete a departed employee's local Cowork history. **Actually:** They cannot manage or delete it centrally; Enterprise admins can retrieve it through the Compliance API, but deletion endpoints for local sessions are not available yet.
- **Tempting:** On our Team plan, Cowork can use computer use to operate the internal app that has no connector. **Actually:** Computer use is Pro and Max only; Team and Enterprise plans do not have it at this time.
- **Tempting:** Claude is trained not to trade, so the brokerage app can stay open to computer use. **Actually:** The help center says not to rely on that training; block sensitive apps.
- **Tempting:** A tool set to Blocked will run in Skip mode. **Actually:** Blocked is denied in every mode.
- **Tempting:** On our Team plan, a write tool I marked "Always allow" never asks again. **Actually:** The organization setting that allows this is off by default, so write-capable connector tools are approved per task.
- **Tempting:** Wait for the draft to finish, then regenerate. **Actually:** Cowork is built for course corrections; redirect mid-run.

## Admin controls for Team and Enterprise

*Tested in: CCAO-F 5.2, 5.4, 6.2, 6.3, 7.1 · CCDV-F Guardrails and Safe Deployment; Identity, Secrets, and Key Management · CCAR-P 3.1, 3.2, 5.1, 5.4, 7.1 (plus Sample 1, which the guide tags to Domain 3)*

The [CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) describes its audience as "professionals who use Claude as a productivity tool and build Claude Projects in their day-to-day roles", not administrators, but objective 6.3 asks you to "Follow organizational AI policies and governance standards", and in the apps those policies arrive as settings an owner controls. Knowing them explains why a feature is missing ("If you're on an Enterprise plan and a model or effort level you expect is missing, your administrator may have turned it off for your role", per the [model settings article](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings)) and which control answers which risk. Identity, audit and retention are taught in more depth in [Admin and governance controls](security-and-governance.md#admin-and-governance-controls) and [Data retention, training and compliance](security-and-governance.md#data-retention-training-and-compliance).

### Roles

| Role | Plans | What sets it apart |
|---|---|---|
| Primary Owner | Team, Enterprise; only one per organization (it can be a service account) | Manages members' Work accounts and all associated data; the only role that can request a Primary Ownership transfer, and the only built-in role that can request organization data exports (for the custom-role exception, see Oversight and data); on Enterprise, the only role that can enable the Compliance API. Seats are a source conflict: the [roles table](https://support.claude.com/en/articles/9267276-roles-and-permissions) checks "Provision new seats" for the Primary Owner only, but the seat articles for [Team](https://support.claude.com/en/articles/12004354-purchase-and-manage-seats-on-team-plans) and [Enterprise](https://support.claude.com/en/articles/13393991-purchase-and-manage-seats-on-enterprise-plans) say "Only Owners and Primary Owners can purchase seats" |
| Owner | Team, Enterprise | Billing (invoices, payment methods); turns features on and off for the organization: connectors, custom connectors, web search and other capabilities, memory, organization instructions, project sharing, organization-wide skills; invites or removes Admins and Owners and changes roles |
| Admin | Team, Enterprise | Invites members and removes members or cancels invitations; on Enterprise, can also view usage analytics |
| User | Team, Enterprise | Creates chats and uses projects with whatever the organization has enabled |
| Custom | Enterprise only | No default permissions: access comes entirely from the custom roles assigned to the member's groups; managed in Organization settings > Roles |

Custom roles follow one rule worth memorizing: "Feature access is determined by a four-level precedence chain, where the most restrictive level wins" ([Manage custom roles](https://support.claude.com/en/articles/13930452-manage-custom-roles-on-enterprise-plans)). The four levels, in order:

1. **Platform-level overrides**: features Anthropic force-enables or force-disables for your organization as part of your contract; they cannot be changed in organization settings.
2. **Organization-level setting**: an Owner or Primary Owner turns the feature on or off for everyone.
3. **Custom role permissions**: if the feature is on for the organization, the member's custom roles decide; if any of their roles grants it, they have it.
4. **User-level setting**: a granted feature stays available unless the member has turned it off in their own settings.

The article's summary: "the organization-level toggle is a main switch. Custom roles are the per-member switches underneath it." If a feature is off at the organization level, no custom role can grant it. Several further details are easy to get wrong. Custom roles affect only members whose role is set to "Custom"; members with the User, Admin or Owner roles take their permissions from those roles. Admin permission areas are set to No access, Can view or Can manage, but not every area offers both levels: Analytics has no Manage level, and User Management, Libraries, Directory management and Claude Design Admin have no View level. As of September 2026 the custom roles article lists eight areas: Identity & Access, Billing, Analytics, Privacy, User Management, Libraries, Directory management and Claude Design Admin (the older [setup article](https://support.claude.com/en/articles/13930458-set-up-role-based-permissions-on-enterprise-plans) lists seven, naming the last one Directory and leaving out Claude Design Admin). A member with Identity & Access set to Can manage can expand their own access, so reserve it for trusted security and IT administrators. Connector permissions per role are Always allow, Needs approval, Blocked or Custom (per tool), and a newly created role defaults to "Needs approval" on every connector.

Whether Team has role-based permissions and domain capture at all is one of the points where sources disagree: the [Team help article](https://support.claude.com/en/articles/9266767-what-is-the-team-plan) lists "Role-based permissioning" and "Single-Sign-On (SSO) and Domain Capture", but the [pricing table](https://claude.com/pricing) marks role-based access and domain capture as No for Team, the [domain claim article](https://support.claude.com/en/articles/14625619-claim-and-migrate-accounts-on-your-domain) says "Domain claiming is supported on Claude Enterprise plans only.", and custom roles are documented as Enterprise-only (for role-based access, see also the "Sources disagree" note under [Plans and what they include](#plans-and-what-they-include)). Treat custom roles and domain capture as Enterprise features.

### Settings that change what members see

Several defaults differ by plan, so learn them as Team and Enterprise pairs. As of September 2026:

| Control | Team default | Enterprise default | Notes |
|---|---|---|---|
| Memory | Off | Off | An owner turns it on; turning it off at the organization level permanently deletes all members' memory data |
| Cowork | On | On | Owners can disable it |
| Run Cowork in the cloud | On | Off; an owner turns it on, then grants it to a group with custom roles | |
| Claude in Chrome | On | Off by default until September 10, 2026; from that date the help center says it turns on by default unless an owner had already disabled it | Allowlists and blocklists; see [Claude in Chrome and other surfaces](#claude-in-chrome-and-other-surfaces) |
| Microsoft 365 add-ins working across apps | "Let Claude work across files" is off by default in each add-in, and the toggle is per-device | Same as Team | Owners control access with "Let Claude work across apps" (Organization settings, Office agents); admins can also manage add-in access in the Microsoft 365 Admin Center; Pro and Max default to on |
| Organization instructions | Owner setting | Owner setting | Up to 3,000 characters; see [Memory, styles and personalization](#memory-styles-and-personalization) |
| Rate chats (thumbs up / down feedback to Anthropic) | Owners and Primary Owners can disable | Owners and Primary Owners can disable | Organization settings > Data and Privacy |

The organization switches for web search, code execution and network egress, connectors and their action permissions, skills and plugins, Claude Design, Slides and Docs, artifact external sharing and project sharing are taught with their features: [Search, Research and connectors](#search-research-and-connectors), [Artifacts, files and code execution](#artifacts-files-and-code-execution), [Skills in the Claude apps](#skills-in-the-claude-apps) and [Projects](#projects).

Enterprise adds controls Team does not have:

| Enterprise control | What it does |
|---|---|
| Default model | One default for the whole organization or per custom role; "Use Anthropic’s recommended default" updates automatically when new models are released; members can still pick a different model for any conversation |
| Model access | Controls which models members can use and caps the effort level they can select; the organization setting is the ceiling for every role; Haiku models are always available and cannot be disabled |
| Groups and group spend limits | Groups are created manually or synced from the identity provider via SCIM, up to 100 per organization; all members of a group can share a per-user spend limit. (Spend limits themselves are not Enterprise-only: Team Owners set monthly spend limits on pre-purchased usage credits, while on usage-based Enterprise plans all usage is billed at API rates and admins set limits for the organization and for individual users.) |
| Session length | A maximum session length of 1, 7, 14 or 28 days, to limit how long a compromised session stays valid |
| IP allowlisting | Checks the source IP address of every authenticated request against the organization's allowlist (CIDR ranges) and blocks the rest. The [IP allowlisting article](https://support.claude.com/en/articles/13200993-restrict-access-to-claude-with-ip-allowlisting) says to send your CIDR ranges to your Anthropic contact or Support; the newer [custom roles article](https://support.claude.com/en/articles/13930452-manage-custom-roles-on-enterprise-plans) (as of September 2026) lists the IP allowlist under Organization settings > Organization and access, editable with Identity & Access set to Can manage |
| US-only inference | On usage-based Enterprise plans, keeps the organization's inference within the United States across the Claude apps, at an increased cost |
| Inference hooks (beta) | Anthropic sends each governed prompt to the organization's own AI security server for an allow or deny verdict before inference, and "a denied request never reaches the model"; the most common use is data loss prevention; verdicts cannot redact or rewrite a prompt |

### Sharing chats: what the plan changes

Chats are private by default; sharing creates a snapshot link ([Share and unshare chats](https://support.claude.com/en/articles/10593882-share-and-unshare-chats)).

| Fact | Value |
|---|---|
| What the snapshot holds | All messages sent before sharing, including artifacts |
| What it leaves out | Raw data from connectors or MCP tool calls (viewers see only Claude's final responses); attached files, unless the chat was shared within your organization, which is the only way Team and Enterprise members can share. That exception is in the newer [Share a chat with specific people](https://support.claude.com/en/articles/16762496-share-a-chat-with-specific-people) article (as of September 2026); the older share article says attached files always stay private |
| Public links | Free, Pro and Max only; Team and Enterprise members can share chats only inside their organization |
| Sharing with specific people by email | All plans, on claude.ai (on Team and Enterprise, only email addresses on the organization's domain); the invite works only for the address entered, and invitees can view the snapshot but cannot reply, copy it or continue the chat |
| Search engines | Every shared chat page carries a "noindex" instruction, but the rule is still: "treat a public link as public." |

### Identity and membership

| Control | Team | Enterprise |
|---|---|---|
| Single sign-on (SSO) | Yes | Yes |
| JIT provisioning | Yes | Yes |
| SCIM provisioning | No | Yes |
| Domain capture (claiming accounts on your domain) | No | Yes; prerequisites are restricted organization creation, DNS domain verification, enforced SSO, and JIT or SCIM; the migration window is 30 days, after which unmigrated personal accounts are deactivated; "Domain capture is a one-way door." ([Claim and migrate accounts](https://support.claude.com/en/articles/14625619-claim-and-migrate-accounts-on-your-domain)) |
| Role-based access and audit logs (pricing page) | No | Yes |

SSO setup needs an Owner or Primary Owner, DNS access to verify the domain, and an identity provider. Once "Require SSO" is on, members must sign in with SSO, so an expired identity-provider certificate locks everyone out: note its expiry date and rotate it ahead of time ([Set up SSO](https://support.claude.com/en/articles/13132885-set-up-single-sign-on-sso)).

Three membership facts: JIT "never removes members automatically", while SCIM removes users who are removed from the identity provider app; a member invitation "expires after 21 days"; and pending invitations "occupy your available seats immediately" ([JIT or SCIM provisioning](https://support.claude.com/en/articles/13133195-set-up-jit-or-scim-provisioning), [Manage members](https://support.claude.com/en/articles/13133750-manage-members-on-team-and-enterprise-plans)).

### Oversight and data

| Tool | Who and where | What it gives you |
|---|---|---|
| Usage analytics | Team: Owners and Primary Owners. Enterprise: Owners, Primary Owners and Admins | Adoption and usage views |
| Audit logs | Enterprise only; Owners and Primary Owners export them from Organization settings > Data and Privacy | A CSV export of the past 180 days, sent as an emailed link active for 24 hours; only unique identifiers of chats and projects, not their titles or content; organizations using customer-managed encryption keys cannot use the export button and get audit log events through the Compliance API |
| Compliance API | Enterprise (excluding Public Sector organizations); only the Primary Owner can enable it | Programmatic access to activity feed events, chat data and file content, including incognito chats; Anthropic's advice is to "Standardize on the Compliance API for ongoing programmatic use" rather than the audit log export |
| Organization data export | Team and Enterprise: the Primary Owner is the only built-in role that can, per the [export article](https://support.claude.com/en/articles/13346720-export-your-organization-s-data); on Enterprise, the newer [custom roles article](https://support.claude.com/en/articles/13930452-manage-custom-roles-on-enterprise-plans) (as of September 2026) also lets a custom role with Privacy set to Can manage "run data exports" | A download link that expires 24 hours after delivery; includes incognito chats |
| Custom data retention | Enterprise; Owners and Primary Owners, per the [retention article](https://support.claude.com/en/articles/10440198-configure-custom-data-retention-controls-for-enterprise-plans); the newer [custom roles article](https://support.claude.com/en/articles/13930452-manage-custom-roles-on-enterprise-plans) also lets a custom role with Privacy set to Can manage "Edit retention periods" | Minimum 30 days (a month counts as 30 days); without a custom period, data is retained indefinitely; for chats inside a project, project retention takes precedence over chat retention; data past its retention period is permanently deleted and cannot be recovered |

Two data-terms facts sit behind these tools. On Claude for Work, the organization's Primary Owner "manages your Work account and all associated data", and Anthropic's consumer terms and privacy policy do not apply where Anthropic acts as the data processor. By default, inputs and outputs from commercial products, including Claude for Work, are not used to train Anthropic's models ([Is my data used for model training?](https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training)).

Adoption numbers need care when you report them. The Academy's enterprise deployment course says usage metrics "tell you Claude is being used, not what that use produced", and warns: "The month you turn a diagnostic into a quota, members optimize for the number instead of the work" ([adoption signals](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/adoption-signals)).

### Where organization controls do not reach

For objective 6.2 questions about whether an organization setting covers a feature:

| Feature or situation | Gap |
|---|---|
| Local Cowork sessions | History is stored on users' computers, is not subject to Anthropic's standard retention, and admins cannot centrally manage or delete it |
| Claude for Excel, PowerPoint, Word and Outlook add-ins | Chat history is stored locally in the browser (IndexedDB in Excel); the add-ins do not inherit custom data retention settings, and their activity is not in Enterprise audit logs or data exports; with the Compliance API enabled, add-in sessions are included there (public beta) |
| Claude Design, Claude Tag, Claude Managed Agents, features built on Claude Code on the web | Custom retention periods do not apply |
| Connected third-party services | They process data on their own infrastructure under their own terms; US-only inference does not change where they operate |
| Organizations with HIPAA, public-sector or custom data retention agreements | Memory is not available |
| Organizations covered by HIPAA (HIPAA-ready Enterprise) | Claude in Chrome is not available; Cowork is not yet covered by Anthropic's BAA |
| Organizations using CMEK, zero data retention or a HIPAA-ready configuration | Claude Docs is not available yet |
| Zero data retention | Not supported for Claude in Chrome, the same as Cowork |

### Decide

- **Least privilege first.** If a group does not need a capability, remove it rather than watch it: set the connector action to Blocked, or leave the feature off for that role. The [CCAR-P guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf) Sample 1 rationale states the principle: "Least privilege means removing capabilities the role does not require, eliminating the attack surface rather than monitoring or guarding it." The same rationale rejects logging and confirmation prompts as "detective/compensating controls, not removal of unnecessary privilege". In admin-console terms (our mapping), Blocked or a feature left off removes the capability; Needs approval is a confirmation step.
- **Start narrow, widen on evidence.** Start Claude in Chrome with a restrictive allowlist, start code execution with network access off ("This is the most secure configuration", per the [file creation article](https://support.claude.com/en/articles/12111783-create-and-edit-files-with-claude)), and test a skill on your own account before provisioning it to everyone.
- **Research plus custom connectors.** During Research, Claude can call connector tools without further approval, so disable tools that can write to external apps (see [Research](#research)).
- **Prompt text is not a control.** Organization instructions guide behavior; settings enforce it (the reasoning is under [Organization instructions](#organization-instructions-team-and-enterprise)).
- **Missing feature?** Check the organization and role settings before assuming the product lacks it.

### Traps

- **Tempting:** Enabling the Gmail connector gives the whole team access to it. **Actually:** Enabling makes it available; each person still authenticates, and Claude inherits only their own permissions (the exception is Enterprise-managed auth, in beta for Team and Enterprise, where a connector is authorized once and the team inherits access on first login).
- **Tempting:** A custom role can give one team Claude in Chrome while it is off for the organization. **Actually:** No custom role can grant a feature disabled at the organization level.
- **Tempting:** The owner can read my memories. **Actually:** Owners cannot view or edit individual memories.
- **Tempting:** The audit log shows what people asked Claude. **Actually:** It carries identifiers only; chat content comes through the Compliance API or the Primary Owner's data export.
- **Tempting:** Any Owner can export the organization's data. **Actually:** Among the built-in roles, only the Primary Owner can. On Enterprise, the newer [custom roles article](https://support.claude.com/en/articles/13930452-manage-custom-roles-on-enterprise-plans) (as of September 2026) also lists "run data exports" under the Privacy area's Can manage level, so a Custom-role member can be given the task; an Owner still cannot.
- **Tempting:** A Team member can post a public link to a chat. **Actually:** Team and Enterprise members can share chats only inside their organization.
- **Tempting:** Our 90-day retention covers everything. **Actually:** Not local Cowork history, the Microsoft 365 add-ins, Claude Design, Claude Tag or Claude Managed Agents.
- **Tempting:** Setting a risky connector action to Needs approval for a role that never uses it is least privilege. **Actually:** A confirmation step is a compensating control in the CCAR-P Sample 1 rationale; if the role does not need the action, block it.
- **Tempting:** A custom role restricts everyone in its group, Owners included. **Actually:** Custom roles affect only members whose role is set to "Custom"; Owners, Admins and Users take their permissions from those roles (organization-level model limits, by contrast, apply to everyone).

## The AI Fluency framework

*Tested in: CCAO-F 1.1, 1.3, 2.1, 2.2, 4.4, 4.5, 6.4 (its vocabulary appears in the Sample 1 rationale) · CCAR-P 5.5 · recommended preparation for all four exams*

The AI Fluency framework is the teaching model behind Anthropic's AI Fluency courses. The course defines the goal in one line: "AI Fluency means engaging with AI systems effectively, efficiently, ethically, and safely" ([Introduction to AI Fluency](https://academy.claude.com/courses/ai-fluency-framework-foundations/introduction-to-ai-fluency)). The course is the result of a partnership between Anthropic and professors Rick Dakan (Ringling College of Art and Design) and Joseph Feller (University College Cork), who developed the framework itself in 2023 to 2024.

Why it matters for the exams:

- The official CCAO-F prep course says: "Before starting this course, we recommend you complete: Claude 101, AI Fluency: Framework & Foundations, and AI Capabilities and Limitations" ([CCAO-F prep course](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations)).
- Two of that prep course's modules are built on it: Prompting & Task Execution is anchored on Description, and Workflow Integration & Solution Design on Delegation, "deciding, for each step, whether the work is AI-appropriate, human-retained, or collaborative."
- AI Fluency: Framework & Foundations is also on the recommended course lists for the CCDV-F and CCAR-P prep paths and on the CCAR-F prep-courses page.
- The CCAO-F guide's Sample 1 rationale calls validating factual claims, especially citations bound for a compliance audience, against an authoritative source "the diligence step required" ([CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)).
- The course itself, AI Fluency: Framework and foundations on Claude Academy (academy.claude.com), lists 14 lessons, 1 quiz, 4 hr, and a completion badge.

### Three ways of working with AI

The course names three modes of engagement, and the framework document (Version 1.1, [Ringling College](https://ringling.libguides.com/ai/framework)) gives examples of each:

| Mode | Course definition ([The 4D framework](https://academy.claude.com/courses/ai-fluency-framework-foundations/the-4d-framework)) | Framework v1.1 examples |
|---|---|---|
| Automation | "AI executes specific tasks based on your instructions" | Emails, summaries, social media posts, basic coding |
| Augmentation | "You and AI collaborate as creative thinking and task execution partners" | Writing stories, essays, research papers, complex coding tasks |
| Agency | "You guide AI to work independently on your behalf, shaping its knowledge and behavior rather than specific actions" | Interactive game characters, tutors, chatbots |

In Claude terms (our mapping, not Anthropic's): a single well-specified request is closest to Automation, iterating on a draft with Claude is Augmentation, and configuring a shared project's instructions or a skill that shapes how Claude behaves for colleagues is closer to Agency, because it defines future behavior rather than one task.

### The 4Ds, with the official definitions

Three official wordings exist: the course lesson, the framework authors' own document and Claude 101. None of them appears in the four exam guides; the AI Fluency course is on the recommended prep lists for all four exams, so learn its wording first. The course lesson ([The 4D framework](https://academy.claude.com/courses/ai-fluency-framework-foundations/the-4d-framework)) and the framework document ([Framework for AI Fluency, Version 1.1](https://ringling.libguides.com/ai/framework)) define each competency as follows:

| Competency | Course definition | Framework v1.1 definition | Sub-competencies (course) |
|---|---|---|---|
| Delegation | "Thoughtfully deciding what work to do with AI vs. doing yourself" | "Creative vision and selection of the right AI tools and techniques to realize that vision." | Problem Awareness, Platform Awareness, Task Delegation |
| Description | "Communicating clearly with AI systems" | "Effectively describing a vision and/or tasks to prompt useful AI behaviors and outputs." | Product, Process and Performance Description |
| Discernment | "Evaluating AI outputs and behavior with a critical eye" | "Accurately assessing the usefulness of AI outputs" | Product, Process and Performance Discernment |
| Diligence | "Ensuring you interact with AI responsibly" | "Taking responsibility and vouching for final products created using AI" | Creation, Transparency and Deployment Diligence |

Claude 101 gives its own phrasing for all four ([Getting better results](https://academy.claude.com/courses/claude-101/getting-better-results)): Delegation is "Deciding on what work should be done by humans, what work should be done by AI, and how to distribute tasks between them", Description is "Effectively communicating with AI systems", Discernment is "Thoughtfully and critically evaluating AI outputs, processes, behaviors and interactions", and Diligence is "Using AI responsibly and ethically. Includes making thoughtful choices about AI systems and interactions, maintaining transparency, and taking accountability for AI-assisted work."

### Delegation

The course splits Delegation into three parts ([A closer look at Delegation](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-delegation)):

- **Problem Awareness**: "Understanding your goals and the work involved to achieve it"
- **Platform Awareness**: "Knowing what different AI systems can do"
- **Task Delegation**: "Strategically dividing work between you and AI"

The same lesson sets the aim: "The goal isn't to automate everything, but to create the most effective human-AI partnership for any given task or goal." Anthropic's AI Fluency for Nonprofits course makes Task Delegation concrete ([Workflow augmentation](https://academy.claude.com/courses/ai-fluency-for-nonprofits/workflow-augmentation)): ask "should AI do this?" not just "can AI do this?", then sort each task into one of three groups:

| Group | Course wording |
|---|---|
| AI can handle | "Standardized responses, documented information, clear processes" |
| AI can assist, human decides | "Tasks where AI can draft or prepare, but you review before action" |
| Human should handle | "High-stakes decisions, emotional situations, complex judgment calls" |

Platform Awareness is where the product knowledge on this page lands: choosing chat, a project, Research, an artifact or Cowork, and choosing a model (see [Choosing a model in the apps](#choosing-a-model-in-the-apps) and [Claude Cowork](#claude-cowork)). The framework document names the first sub-competency differently: "Goal and Task Awareness" instead of the course's Problem Awareness.

### Description

| Type | Course definition ([A closer look at Description](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-description)) |
|---|---|
| Product Description | "clearly defining what you want in terms of outputs, format, audience, and style" |
| Process Description | "guides how the AI approaches your request, which can be as important as specifying the end goal" |
| Performance Description | "defines behavioral aspects like whether the AI should be concise or detailed, challenging or supportive" |

The framework document frames Performance Description more broadly, as "Directive prompting to define future AI behaviors and enable positive user experience"; in our reading that is the Agency mode's kind of prompting, since the framework says Agency "defines the characteristics and future behavior of an AI, rather than a specific task". The lesson's summary line is worth remembering for distractors that treat Claude like a search box: "AI systems are interactive partners, not databases or vending machines".

The course's prompting techniques ([Effective prompting techniques](https://academy.claude.com/courses/ai-fluency-framework-foundations/effective-prompting-techniques)):

| Technique | Course wording |
|---|---|
| Give context | "Be specific about what you want, why you want it, and relevant background" |
| Show examples | "Demonstrate the output style or format you're looking for" |
| Specify constraints | "Clearly define format, length, and other output requirements" |
| Break complex tasks into steps | "Guide the AI through multi-step reasoning" |
| Ask the AI to think first | "Give space for the AI to work through its process" |
| Define the AI's role or tone | "Specify how you want the AI to communicate" |
| The "secret weapon" | "Ask the AI itself to help improve your prompt" |

Claude 101's three-part prompt framework (setting the stage, defining the task, specifying rules) is "adapted from the 4D Framework for AI Fluency" ([Your first conversation with Claude](https://academy.claude.com/courses/claude-101/your-first-conversation-with-claude)). Prompt structure in depth is in [Prompt Engineering and Structured Output](prompt-engineering.md#principles-that-decide-most-prompt-questions).

### Discernment

| Type | Course definition ([A closer look at Discernment](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-discernment)) |
|---|---|
| Product Discernment | "evaluating the quality of actual outputs (accuracy, appropriateness, coherence, relevance)" |
| Process Discernment | "assessing how the AI arrived at its output, looking for logical errors, attention gaps, or inappropriate reasoning" |
| Performance Discernment | "evaluates how the AI behaves within the collaboration process itself, considering whether its communication style is effective for your needs" |

The framework document words two of these differently. Its Process Discernment is "Assessing if the human-AI collaborative dynamic is fruitful or not and how to improve it" (the collaboration, rather than the reasoning behind one output), and its Performance Discernment is "Evaluating if AI-driven independent behaviors enable positive user experiences and how to better direct the AI to improve outcomes" (AI acting independently, in line with the Agency mode). Both wordings are official; when you quote one, name its source.

"Discernment works hand-in-hand with Description in a continuous feedback loop" ([A closer look at Discernment](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-discernment)). The course's [Description-Discernment loop](https://academy.claude.com/courses/ai-fluency-framework-foundations/the-description-discernment-loop) lesson runs it for each task in a project; in our mapping it is the working pattern behind objective 1.3 (iterating prompts):

```text
1. Describe   Product (what you want), Process (how to approach it),
              Performance (how Claude should engage with you)
2. Discern    Product, Process and Performance Discernment of what came back
3. Refine     Give feedback on what worked and what didn't, clarify the
              description: "Request iterations until you're satisfied
              with the result"
4. Integrate  Add your own expertise: "Make the final decisions about what
              to keep, modify, or discard"; take responsibility for the output
        |
        +---> back to 1 for the next task
```

Anthropic's AI Fluency Index shows why the loop matters: "85.7% of the conversations in our sample exhibited iteration and refinement", and those conversations were "5.6x more likely to involve users questioning Claude's reasoning, and 4x more likely to see them identify missing context." Setting the terms of the collaboration was rarer: "In only 30% of conversations do users tell Claude how they'd like it to interact with them" ([The AI Fluency Index](https://academy.claude.com/tutorials/the-ai-fluency-index)). The practical techniques for checking output are in [Verifying Claude's output](#verifying-claudes-output).

### Diligence

| Type | Course definition ([A closer look at Diligence](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-diligence)) |
|---|---|
| Creation Diligence | "being thoughtful about which AI systems we use and how we engage with them" |
| Transparency Diligence | "being honest about AI's role in our work with everyone who needs to know" |
| Deployment Diligence | "taking responsibility for verifying and vouching for the outputs we use or share" |

The framework document adds what Deployment Diligence involves in practice: "thorough fact-checking, testing for accuracy, and validating claims" ([Framework v1.1](https://ringling.libguides.com/ai/framework)). That is the same step the CCAO-F Sample 1 answer requires.

**Diligence statements.** "A diligence statement is a transparent acknowledgment of AI's role in your work, along with your commitment to responsibility for the final output" ([A closer look at Diligence](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-diligence)). The course's example:

```text
In creating this [document/project/content], I collaborated with [AI assistant name] to assist with [specific tasks: drafting, research, editing, etc.]. I affirm that all AI-generated and co-created content underwent thorough review and evaluation. The final output accurately reflects my understanding, expertise, and intended meaning. While AI assistance was instrumental in the process, I maintain full responsibility for the content, its accuracy, and its presentation. This disclosure is made in the spirit of transparency and to acknowledge the role of AI in the creation process.
```

The Academy tutorial on writing one ([Writing an AI diligence statement](https://academy.claude.com/tutorials/writing-an-ai-diligence-statement)) calls it the "methods section" for AI collaboration: what AI assisted with, which tool, what you reviewed, what you changed, and who is responsible. Two warnings from it: "Vague or absent disclosure is what erodes trust, especially if someone discovers the AI involvement later", and "Your AI diligence statement is itself a claim about your process, and it needs to be accurate." Do not claim you verified every citation unless you did.

Transparency also has a machine-readable side. Claude models launched in the EU on or after August 2, 2026 support machine-readable marking at launch: embedded watermarks in generated text, and C2PA Content Credentials in generated files where supported. Anthropic says it is working to add marking to models launched before that date, and warns that "Lack of a detected mark doesn’t mean the content wasn’t AI-generated or processed" ([How Claude marks AI-generated content](https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content)).

### Mapping the 4Ds to the Associate blueprint

The CCAO-F guide does not mention AI Fluency or the 4Ds; the only 4D competency name in it is "diligence", in the Sample 1 rationale. The prep course anchors two modules on Delegation and Description; the rest of this table is our mapping, useful for recognizing which competency an item is testing. It is wider than the Tested in line at the top of this section and the [Exam map](#exam-map), which list only the objectives this section teaches directly:

| Competency | CCAO-F objectives it lines up with |
|---|---|
| Delegation | 3.1 selecting features, 3.3 matching the model to the task, 4.1 analyzing use cases, 4.4 augmenting or redesigning workflows |
| Description | 1.1 to 1.4 prompting, 2.6 choosing the output format, 5.3 system-level instructions |
| Discernment | 2.1 to 2.3 and 2.5 evaluating, fact-checking and refining outputs, 7.1 and 7.2 diagnosing and adjusting |
| Diligence | 2.4 deciding when human review is required, 6.1 to 6.4 governance and ethics, 4.5 communicating limitations |

### Traps

- **Tempting:** The fluent choice is to automate as much as possible. **Actually:** The course says the goal "isn't to automate everything".
- **Tempting:** Discernment is only fact-checking. **Actually:** Product Discernment covers accuracy, appropriateness, coherence and relevance; Process and Performance Discernment look at the reasoning and at how the collaboration is going. Note too that the CCAO-F Sample 1 rationale calls validating a citation against an authoritative source "the diligence step required", and the framework document lists "thorough fact-checking" under Deployment Diligence, so expect checking-before-you-share to be framed as Diligence.
- **Tempting:** A line saying AI was used is enough disclosure. **Actually:** A diligence statement says what AI did, what you reviewed and changed, and who is responsible, and it must be accurate.
- **Tempting:** Ask Claude to vouch for its own output. **Actually:** Deployment Diligence is the human's job: "taking responsibility for verifying and vouching for the outputs we use or share".
- **Tempting:** Treat Claude like a database. **Actually:** The course: "interactive partners, not databases or vending machines".
- **Tempting:** Accept the first response and move on. **Actually:** The loop is describe, discern, refine, integrate; the fluent move is to refine.

## Verifying Claude's output

*Tested in: CCAO-F 2.1 to 2.5, 7.1, with Sample 1 · CCDV-F Output Handling · CCAR-F 5.2-K3, 5.5, 5.6 (the pipeline versions of review and provenance) · CCAR-P 4.4, 5.2, 5.3, 5.5*

Output Evaluation and Validation is the heaviest CCAO-F domain, at 21%. The guide's own sample shows what it rewards. An associate has a confident summary of a new regulation that cites a specific subsection; the keyed answer is to "Verify the cited subsection against the official regulation text before sharing", and the rationale explains: "Language models can fabricate specific-looking details such as citation numbers, a hallucination. Validating factual claims, especially citations bound for a compliance audience, against an authoritative source is the diligence step required. Self-reported confidence (A, C) is not a reliable accuracy signal, and reformatting (D) does not address correctness" ([CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)). The full item is on the [CCAO-F exam page](../claude-certified-associate.md#official-sample-questions). This section covers why confident output can still be wrong, a routine for checking it, how far to trust Claude's own confidence, how to check for bias and flattery, and when a person must review.

### Why confident output can still be wrong

| Failure | What it looks like | Source |
|---|---|---|
| Hallucination | "Claude can display quotes that may look authoritative or sound convincing, but are not grounded in fact." | [Incorrect or misleading responses](https://support.claude.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on) |
| Fabricated specifics | "Fabrication concentrates in specificity: names, dates, statistics, citations, URLs, quotes. The more precise a claim, the more it warrants verification." | [Next Token Prediction](https://academy.claude.com/courses/ai-capabilities-and-limitations/next-token-prediction) |
| False capability claims | Claude says it sent an email or produced an external document; "Even if it claims otherwise, Claude does not have access to other tools or software that are not explicitly integrated, including email, word processors, or file transfers." | [Links and false claims](https://support.claude.com/en/articles/8241188-claude-is-producing-links-that-don-t-work-and-falsely-claiming-that-it-has-sent-emails-or-produced-external-documents-what-s-going-on) |
| Stale knowledge | Knowledge "comes entirely from training data and is frozen at the knowledge cutoff"; models may not know about events after their cutoff dates | [Knowledge](https://academy.claude.com/courses/ai-capabilities-and-limitations/knowledge), [How up-to-date is Claude's training data?](https://support.claude.com/en/articles/8114494-how-up-to-date-is-claude-s-training-data) |
| Sycophancy | When someone "tells you what they think you want to hear, instead of what's true, accurate, or genuinely helpful"; agreeing with your factual error or changing the answer with your phrasing | [What is sycophancy](https://academy.claude.com/tutorials/what-is-sycophancy-in-ai-models) |
| Bias | Stereotyping or political bias, or less directly "defaulting to certain types of answers or perspectives, or providing better quality responses in specific languages" | [Why does bias exist](https://academy.claude.com/tutorials/why-does-bias-exist-in-ai-models) |
| Loose calibration | Fine-tuning on human judgments leaves "fingerprints", among them "loose calibration between stated confidence and actual reliability" | [How AI gets its character](https://academy.claude.com/courses/ai-capabilities-and-limitations/how-ai-gets-its-character) |

The last row fits the Sample 1 rationale, which rejects both options that lean on Claude's confidence: "Self-reported confidence (A, C) is not a reliable accuracy signal." Training-data cutoffs per model are listed in [Choosing a model in the apps](#choosing-a-model-in-the-apps).

Polish is a risk factor of its own. Anthropic's AI Fluency Index found that "in conversations where artifacts are created, users are less likely to identify missing context (-5.2pp), check facts (-3.7pp), or question the model's reasoning by asking it to explain its rationale (-3.1pp)" ([The AI Fluency Index](https://academy.claude.com/tutorials/the-ai-fluency-index)). Its advice: when output looks good, "it's the perfect moment to pause and ask: is this accurate? Is anything missing? Does this reasoning hold up?"

Hallucination is also more likely in predictable places: "if you're asking for specific facts, statistics, or citations, or if the topic is obscure, niche, or very recent, if you're asking about real but not widely known people or places, or when you need exact details like dates, names, or numbers" ([Why do AI models hallucinate](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate)). The same tutorial is candid that this is "not at all a solved problem".

### A verification routine

1. **Find the claims that carry risk.** Names, dates, statistics, citations, URLs and quotes first; anything a reader will act on.
2. **Check them against something outside the conversation.** In Anthropic's interviews with 129 Claude Academy participants, 51% of those interviewed described verifying Claude's output against something external: "source documents, official docs, their own data, another person, or even another AI" ([Discernment toolkit](https://academy.claude.com/tutorials/discernment-toolkit)). For Sample 1 that is the official regulation text.
3. **Make Claude show where each claim came from.** With web search, "Every response includes citations, so you can easily verify sources yourself", but you still have to open them: cross-reference the cited sources and use authoritative sources for critical decisions ([web search](https://support.claude.com/en/articles/10684626-enable-and-use-web-search)).
4. **Make uncertainty visible.** Tell Claude "It's ok if you don't know" ([Why do AI models hallucinate](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate)), and add "flag anything you're not confident about" so you know where to look first ([PRD from a one-pager](https://academy.claude.com/use-cases/prd-from-a-one-pager)). Treat the flags as a map of where to check, not as proof that the rest is right.
5. **Test consistency.** "Run the same question multiple ways and look for consistency. For document-based tasks, ask the AI to quote directly from source material rather than paraphrase" ([Writing an AI diligence statement](https://academy.claude.com/tutorials/writing-an-ai-diligence-statement)).
6. **Get a fresh read.** "If you have an answer you're unsure about, start a new chat and ask the AI to find errors in the answer, and to confirm that the sources support the statements" ([Why do AI models hallucinate](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate)). For subjects outside your expertise, build a "panel of experts" of people who can give a second read; the [Discernment toolkit](https://academy.claude.com/tutorials/discernment-toolkit) found "the biggest barrier to discernment was not time pressure but lack of domain expertise."
7. **Check the reasoning, not only the answer.** Reviewing the expandable Thinking section "can be valuable for verifying how Claude arrived at its conclusion" ([model settings](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings)).
8. **Recompute numbers.** Check complex or mission-critical calculations with specialized software or by hand, as the help center advises (see [Code execution and file creation](#code-execution-and-file-creation)). In Claude for Excel, follow the cell-level citations back to the cells.
9. **When Claude rewrote something, review the diff it gives you.** The Academy's advice is to add a line asking "for a list of what it changed and what it left out. That list is what you review" ([Adapt a textbook page to every reading level](https://academy.claude.com/use-cases/adapt-a-standard-textbook-page-to-every-reading-level)).
10. **Ask the Cowork review questions.** Does it meet the objective, are the facts accurate, and "Does anything sound made up? A specific date, name, or quote that you can't trace to an input is a flag, not a feature" ([The task loop](https://academy.claude.com/courses/introduction-to-claude-cowork/the-task-loop)).

The [Discernment toolkit](https://academy.claude.com/tutorials/discernment-toolkit)'s source-grounding follow-up prompt turns step 3 into one message:

```text
For each factual claim in your answer, tell me where it came from. Quote the exact passage from the source and include the link or page number. If a claim comes from your general knowledge, label it "unsourced" so I know what to check first.
```

### Asking Claude about its confidence

Official sources seem to pull in different directions. Claude 101's fix for confident but wrong answers includes "Ask Claude to cite sources or indicate confidence level" ([Getting better results](https://academy.claude.com/courses/claude-101/getting-better-results)), and the hallucination tutorial suggests you "ask the AI how confident it is, and whether anything might be wrong" ([Why do AI models hallucinate](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate)), yet the Sample 1 rationale says "Self-reported confidence (A, C) is not a reliable accuracy signal." The CCAR-F guide makes the related point for agent escalation: sentiment-based escalation and self-reported confidence scores are "unreliable proxies for actual case complexity" (task statement 5.2).

**Decide.** Asking for confidence is a triage tool: it tells you where to look first. It never replaces checking the claim against a source, and a high self-rating is never the reason to send something. In the CCAR-F pipeline version, task statement 5.5 pairs field-level confidence scores with thresholds calibrated on labeled validation sets; our reading is that a model's confidence becomes usable for routing review only after that calibration. See [Human review and confidence calibration](evaluation-and-reliability.md#human-review-and-confidence-calibration).

### Checking for bias and flattery

For one-sided answers, the Academy's bias tutorial gives five moves: "First, push back if a response feels one-sided. Second, ask it to take a more nuanced and balanced approach. Third, tell it that you're looking for an honest discussion." Then "ask AI to gather evidence and examine the links yourself. Finally, try asking the same questions from different angles" ([Why does bias exist in AI models](https://academy.claude.com/tutorials/why-does-bias-exist-in-ai-models)).

For sycophancy, the tutorial lists the triggers: a subjective truth stated as fact, a cited expert, a question framed from one point of view, a request for validation, emotional stakes, or a very long conversation. The countermeasures: "You can use neutral, fact-seeking language, cross-reference information with trustworthy sources, prompt for accuracy or counter arguments, rephrase questions, start a new conversation, or finally, take a step back from using AI and ask someone that you trust" ([What is sycophancy](https://academy.claude.com/tutorials/what-is-sycophancy-in-ai-models)). The capabilities course suggests a direct test: state a task with a wrong assumption, then try again with this invitation and compare ([How AI gets its character](https://academy.claude.com/courses/ai-capabilities-and-limitations/how-ai-gets-its-character)):

```text
I want you to genuinely disagree with me if you think I'm wrong.
```

### When human review is required

Objective 2.4 asks you to "Determine when human review or additional verification is required". The sources give a consistent scale:

| Situation | Minimum review | Source |
|---|---|---|
| Any high-stakes advice | "Users should not rely on Claude as a singular source of truth and should carefully scrutinize any high-stakes advice given by Claude." | [Incorrect or misleading responses](https://support.claude.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on) |
| Factual claims, especially citations, bound for a compliance audience | Verify each against an authoritative source before sharing | CCAO-F Sample 1 |
| Legal work | Keep "a lawyer in the loop, verify output against primary sources, and document your AI use" | [Claude for legal work](https://support.claude.com/en/articles/15707726-using-claude-for-legal-work-privilege-confidentiality-and-how-to-think-about-configuration) |
| Customer-facing automation | "review outputs before they reach customers, be honest about AI's role, and provide a clear path to a human" | [AI Fluency for small businesses](https://academy.claude.com/courses/ai-fluency-for-small-businesses/tying-it-all-together) |
| Spreadsheet work for clients or audits | Not recommended without review: "Final client deliverables without human review." and "Audit-critical calculations without verification." | [Claude for Excel](https://claude.com/docs/office-agents/excel.md) |
| High-stakes decisions, emotional situations, complex judgment calls | "Human should handle" | [Workflow augmentation](https://academy.claude.com/courses/ai-fluency-for-nonprofits/workflow-augmentation) |
| An agent's first job (the course's example is a morning briefing shared to the whole team) | "For the first few days, a person should review the briefing and give the agent feedback on its effectiveness." | [Practical ways to get started](https://academy.claude.com/courses/building-effective-human-agent-teams/practical-ways-to-get-started) |
| Quick lookups and other low-stakes use | A lighter check: "a quick lookup needs only a glance, while analysis that will shape a decision deserves a closer review"; trust in AI is "a dial you turn, not a switch you flip" | [Choosing the right Claude model](https://academy.claude.com/tutorials/choosing-the-right-claude-model), [Can you trust what AI tells you](https://academy.claude.com/tutorials/can-you-trust-what-ai-tells-you) |

Three background rules frame the table. Anthropic's consumer terms say "You should not rely on any Outputs or Actions without independently confirming their accuracy" ([Consumer terms](https://www.anthropic.com/legal/consumer-terms)). The Commercial Terms of Service put the duty on the customer organization: it must evaluate whether outputs suit its use case, "including where human review is appropriate", and "Customer acknowledges, and must notify its Users, that factual assertions in Outputs should not be relied upon without independently checking their accuracy" ([Commercial terms](https://www.anthropic.com/legal/commercial-terms), section D.3). And Anthropic's discrimination research states: "we do not endorse or permit the use of language models to make automated decisions for the high-risk use cases we study" ([Evaluating and mitigating discrimination](https://www.anthropic.com/research/evaluating-and-mitigating-discrimination-in-language-model-decisions)).

**Decide.** Match the depth of review to the stakes and the audience, not to how confident or polished the output looks. If a specific claim will reach a regulator, a customer or a decision, verify it against the primary source. If a person's rights or wellbeing depend on the outcome, a person decides.

Where this goes deeper for other exams: pipeline-scale human review (confidence calibration, stratified sampling) is in [Human review and confidence calibration](evaluation-and-reliability.md#human-review-and-confidence-calibration), claim-source mappings for multi-source synthesis are in [Provenance and uncertainty in synthesis](context-engineering.md#provenance-and-uncertainty-in-synthesis), and prompt-level techniques are in [Reducing hallucinations](prompt-engineering.md#reducing-hallucinations).

### Traps

These come from the Sample 1 rationale, the guidance above and the capabilities course's Steerability lesson.

- **Tempting:** Send it; Claude was confident. **Actually:** Self-reported confidence is not a reliable accuracy signal (option A in Sample 1).
- **Tempting:** Ask Claude to rate its confidence and send it if the rating is high. **Actually:** Same flaw (option C).
- **Tempting:** Make it sound more formal before sending. **Actually:** Reformatting does not address correctness (option D).
- **Tempting:** It's a finished-looking artifact, so it has been checked. **Actually:** Polished outputs are where users check facts less.
- **Tempting:** Claude confirmed it emailed the client. **Actually:** Without an integration, Claude cannot send email, whatever it says.
- **Tempting:** Repeat the instruction more firmly. **Actually:** When an instruction is followed literally but uselessly, "restate the goal. Repeating the instruction with more force won't close the gap" ([Steerability](https://academy.claude.com/courses/ai-capabilities-and-limitations/steerability)).

## Exam map

*Tested in: all four exams, section by section, as mapped below from the four July 2026 exam guides*

Which official objectives each section of this page serves. The CCAO-F and CCAR-P guides list objectives as unnumbered bullets, so those numbers follow each guide's order: CCAO-F 3.4 is the fourth objective in Domain 3, and CCAR-P 3.2 is the second objective in Domain 3. CCDV-F skills are named without numbers; the position is given in parentheses (Claude Application Design is 2.5, the fifth skill in Domain 2). The CCAR-F guide numbers its own task statements (1.1 to 5.6); codes such as 5.5-K1 add the bullet's position under that statement, K for a "Knowledge of" bullet and S for a "Skills in" bullet. APPX-INSCOPE-n and APPX-OUTSCOPE-n are the nth items of the CCAR-F appendix lists "In-Scope Topics" and "Out-of-Scope Topics". Bullet positions and appendix item numbers are our numbering, not the guides'. The mapping of features to objectives is also ours; the guides name few Claude app features.

| Section | CCAO-F | CCDV-F | CCAR-F | CCAR-P |
|---|---|---|---|---|
| [Plans and what they include](#plans-and-what-they-include) | 3.1, 6.2, 7.3 | Not listed | Not listed | 7.1 |
| [Choosing a model in the apps](#choosing-a-model-in-the-apps) | 3.2, 3.3 (Sample 2), 3.4, 7.1 | Model Selection and Tradeoffs (5.3); LLM Fundamentals (5.1) | Not listed; model comparison metrics are out of scope (APPX-OUTSCOPE-14) | 2.1 |
| [Projects](#projects) | 3.1, 3.4, 5.1, 5.2, 5.3, 5.4 | Claude Application Design (2.5) | Not listed | Not listed |
| [Artifacts, files and code execution](#artifacts-files-and-code-execution) | 2.1, 2.6, 3.1, 4.3, 6.2 | Not listed | Not listed | Not listed |
| [Search, Research and connectors](#search-research-and-connectors) | 3.1, 4.2, 5.2, 6.2 | MCP Server Development (8.2) | Not listed; hosting MCP servers is out of scope (APPX-OUTSCOPE-4) | 3.7 |
| [Skills in the Claude apps](#skills-in-the-claude-apps) | 5.3, 5.4; the How to Prepare list names Skills | Agentic Customization (8.3) | 3.2-K2, 3.2-S5 (skills in Claude Code; skills from a Claude account can sync into Claude Code v2.1.273 or later, and the Agent Skills specification is an open standard) | 2.5, 3.8 |
| [Memory, styles and personalization](#memory-styles-and-personalization) | 3.4, 5.3, 5.4, 6.2; the How to Prepare list names Memory | Claude Application Design (2.5) | Not listed | Not listed |
| [Claude in Chrome and other surfaces](#claude-in-chrome-and-other-surfaces) | 3.1, 4.4, 6.2 | Claude Application Design (2.5); AI Application Security (7.1) | Out of scope: computer use, browser automation (APPX-OUTSCOPE-8) | Not listed |
| [Claude Cowork](#claude-cowork) | 1.4, 3.1, 4.2, 4.4, 7.3 | Claude Application Design (2.5) | Not listed; computer use is out of scope (APPX-OUTSCOPE-8) | Not listed |
| [Admin controls for Team and Enterprise](#admin-controls-for-team-and-enterprise) | 5.2, 5.4, 6.2, 6.3, 7.1 | Guardrails and Safe Deployment (7.2); Identity, Secrets, and Key Management (7.4) | Not listed (the appendix's out-of-scope item APPX-OUTSCOPE-2 covers Claude API authentication, billing, or account management, not the claude.ai admin console) | 3.1, 3.2, 5.1, 5.4, 7.1 (plus Sample 1, which the guide tags to Domain 3) |
| [The AI Fluency framework](#the-ai-fluency-framework) | 1.1, 1.3, 2.1, 2.2, 4.4, 4.5, 6.4; Sample 1 rationale ("the diligence step required") | Not listed; the course is on the recommended prep list | Not listed; the course is on the prep-courses page | 5.5; the course is on the recommended prep list |
| [Verifying Claude's output](#verifying-claudes-output) | 2.1, 2.2, 2.3, 2.4, 2.5, 7.1; Sample 1 | Output Handling (6.3) | 5.2-K3; 5.5-K1 to 5.5-S4 and 5.6-K1 to 5.6-S5, in pipeline form; APPX-INSCOPE-17, APPX-INSCOPE-18 | 4.4, 5.2, 5.3, 5.5 |

### Where the weight sits

This page is the main reference for the CCAO-F exam. Its blueprint weights, and the sections of this page that carry each domain (read off the map above):

| CCAO-F domain | Weight | Sections here |
|---|---|---|
| Domain 1: Prompting and Task Execution | 14% | The AI Fluency framework (Description and the Description-Discernment loop); Claude Cowork (choosing the surface by task type); prompt technique in depth is on the [prompt engineering page](prompt-engineering.md#principles-that-decide-most-prompt-questions) |
| Domain 2: Output Evaluation and Validation | 21% | Verifying Claude's output; The AI Fluency framework (Discernment); Artifacts, files and code execution |
| Domain 3: Product and Model Selection | 12% | Choosing a model in the apps; Projects; Artifacts, files and code execution; Search, Research and connectors; Claude Cowork; Claude in Chrome and other surfaces; Plans and what they include; Memory, styles and personalization (context and memory, objective 3.4) |
| Domain 4: Workflow Integration and Solution Design | 16% | Claude Cowork; Search, Research and connectors; Claude in Chrome and other surfaces; Artifacts, files and code execution (objective 4.3); The AI Fluency framework (Delegation) |
| Domain 5: Configuration and Knowledge Management | 12% | Projects; Search, Research and connectors (connectors, objective 5.2); Skills in the Claude apps; Memory, styles and personalization; Admin controls for Team and Enterprise |
| Domain 6: Governance, Risk, and Responsible Use | 15% | Admin controls for Team and Enterprise; Memory, styles and personalization; Claude in Chrome and other surfaces; Search, Research and connectors; Artifacts, files and code execution; Plans and what they include; The AI Fluency framework (Diligence) |
| Domain 7: Troubleshooting and Optimization | 10% | Choosing a model in the apps; Plans and what they include; Claude Cowork; Verifying Claude's output; Admin controls for Team and Enterprise |

For the other three exams, this page is context rather than core. The closest CCDV-F skill is Claude Application Design (8.6%), whose description in the [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) includes "how Claude interprets instructions across interfaces (Claude Code, Desktop, claude.ai, API, SDKs)". For CCAR-P, the governance and verification material supports Domain 5, Governance, Safety & Risk Management (14%).

### Notes on the mapping

- **Features the guide names, and features it does not.** The [CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) objectives name "Projects, research mode, chat, artifacts" and connectors "Google Drive, Gmail". The How to Prepare list adds "Memory, Skills, and Code Execution", although none of those three is named as a feature in a domain objective (objective 3.4's "memory considerations" is about managing context). Cowork, Claude in Chrome, admin settings and the AI Fluency framework are not named in the guide; they are mapped here to the objectives whose decisions they serve.
- **Vocabulary is from July 2026.** "research mode" and "chat" are the guide's words; expect them in items. What changed in September 2026 is in the notes under [Search, Research and connectors](#search-research-and-connectors) and [Claude Cowork](#claude-cowork).
- **Scope boundary.** The CCAO-F guide says the certification "is not intended for software developers who build against APIs or design agentic systems". API, Claude Code and Agent SDK depth lives on the other knowledge pages.
- **CCAR-F.** Its appendix places computer use (browser automation, desktop interaction) out of scope, so the Chrome and Cowork computer-use material here is background for that exam. The appendix also excludes Claude API authentication, billing and account management, but that item is about the API; the CCAR-F guide does not mention claude.ai admin settings at all, so the admin section is unmapped for that exam rather than excluded.

The exam pages teach each objective in exam framing: [CCAO-F Domain 2](../claude-certified-associate.md#domain-2-output-evaluation-and-validation), [Domain 3](../claude-certified-associate.md#domain-3-product-and-model-selection), [Domain 5](../claude-certified-associate.md#domain-5-configuration-and-knowledge-management) and [Domain 6](../claude-certified-associate.md#domain-6-governance-risk-and-responsible-use); [CCDV-F Domain 2](../claude-certified-developer.md#domain-2-applications-and-integration); [CCAR-P Domain 5](../claude-certified-architect-professional.md#domain-5-governance-safety-risk-management).

??? info "Sources"

    - [Claude Certified Associate, Foundations: Exam Guide (July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf): audience and scope statements, blueprint weights, all objectives used in the exam map, How to Prepare list, and Samples 1, 2 and 3 with their rationales
    - [Claude Certified Architect, Professional: Exam Guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf): objectives used in the exam map (2.1, 2.5, 3.1, 3.2, 3.7, 3.8, 4.4, 5.1 to 5.5, 7.1), the Domain 5 weight, Sample 1 least-privilege rationale and Sample 2 rationale
    - [Claude Certified Developer, Foundations: Exam Guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf): skills used in the exam map (LLM Fundamentals, Model Selection and Tradeoffs, Claude Application Design, Output Handling, AI Application Security, Guardrails and Safe Deployment, Identity, Secrets, and Key Management, MCP Server Development, Agentic Customization) and the Sample 2 rationale
    - [Claude Certified Architect, Foundations: Exam Guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): task statements 3.2 (skills versus always-loaded CLAUDE.md), 5.2, 5.5 and 5.6, Sample Question 6, and the in-scope and out-of-scope appendix lists
    - [Claude Certified Associate, Foundations prep course](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations): the four decisions made before prompting; recommended prerequisite courses including AI Fluency
    - [Claude pricing](https://claude.com/pricing): plan comparison rows (Opus, Sonnet, Haiku, Fable, Research, Projects, model training, role-based access, audit logs, domain capture, skills deployment, org instructions, priority access, Claude in Chrome), Pro and Enterprise prices
    - [What is the Pro plan?](https://support.claude.com/en/articles/8325606-what-is-the-pro-plan): Pro price, session and weekly limits, Settings > Usage, no Console API usage
    - [What is the Max plan?](https://support.claude.com/en/articles/11049741-what-is-the-max-plan): Max 5x and 20x prices and allowances, monthly billing, priority access to new features
    - [What is the Team plan?](https://support.claude.com/en/articles/9266767-what-is-the-team-plan): Team seat prices, 2 to 150 seats, per-member limits, connectors, enterprise search
    - [Get started with the Team plan](https://support.claude.com/en/articles/9267247-get-started-with-the-team-plan): business email for the account creator
    - [What is the Enterprise plan?](https://support.claude.com/en/articles/9797531-what-is-the-enterprise-plan): usage-based billing at API rates, audit logs, SCIM, retention, Compliance API, CMEK, US-only inference, spend limits, legacy billing notes
    - [Purchase and manage seats on Enterprise plans](https://support.claude.com/en/articles/13393991-purchase-and-manage-seats-on-enterprise-plans): 20-seat minimum, HIPAA-ready seat types, who can purchase seats
    - [HIPAA-ready Enterprise plans](https://support.claude.com/en/articles/13296973-hipaa-ready-enterprise-plans): HIPAA is Enterprise-only, click-to-accept BAA, Cowork not yet covered by the BAA, Claude Code covered only with ZDR
    - [Manage custom roles on Enterprise plans](https://support.claude.com/en/articles/13930452-manage-custom-roles-on-enterprise-plans): custom roles are Enterprise-only; precedence chain, main switch analogy, permission areas (including data exports, retention periods and the IP allowlist), Organization settings > Roles
    - [Get started with Claude](https://support.claude.com/en/articles/8114491-get-started-with-claude): Free session limit, minimum age
    - [How do usage and length limits work?](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work): usage versus length limits, one usage limit across surfaces, context window sizes, automatic context management, project and tool tips
    - [Usage limit best practices](https://support.claude.com/en/articles/9797557-usage-limit-best-practices): what consumes usage, project caching, grouping requests
    - [Manage usage credits for paid Claude plans](https://support.claude.com/en/articles/12429409-manage-usage-credits-for-paid-claude-plans): usage credits, API-rate billing, daily redemption limit
    - [What is a limit reset?](https://support.claude.com/en/articles/17007452-what-is-a-limit-reset): limit resets and their irreversibility
    - [Claude Fable models on your plan](https://support.claude.com/en/articles/15424964-claude-fable-models-on-your-plan): Fable availability by plan
    - [Change the model, effort and thinking settings](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings): model menu, effort levels, thinking toggle, mid-conversation changes, missing models on Enterprise roles, the Thinking section as a verification aid
    - [Why Claude switched models in your conversation with Opus 5 or Opus 5.5](https://support.claude.com/en/articles/16049681-why-claude-switched-models-in-your-conversation-with-opus-5-or-opus-5-5): fallback behavior and the toggle that controls it
    - [How large is the context window on paid Claude plans?](https://support.claude.com/en/articles/8606394-how-large-is-the-context-window-on-paid-claude-plans): chat context window per model
    - [How up to date is Claude's training data?](https://support.claude.com/en/articles/8114494-how-up-to-date-is-claude-s-training-data): training data dates per model; models may not know events after their cutoff
    - [Manage model access for your organization](https://support.claude.com/en/articles/15694740-manage-model-access-for-your-organization): model access, effort caps, Haiku models always available
    - [Troubleshoot Claude error messages](https://support.claude.com/en/articles/12466728-troubleshoot-claude-error-messages): length-limit error text and remedies
    - [What are projects?](https://support.claude.com/en/articles/9517075-what-are-projects): project definition, knowledge as background for chats, instructions, Free cap, RAG on paid plans, sharing permissions, the new version of projects
    - [How can I create and manage projects?](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects): setup steps, hidden name and description, no shared context between chats, archiving, project memory and "Remove from project"
    - [Retrieval augmented generation (RAG) for projects](https://support.claude.com/en/articles/11473015-retrieval-augmented-generation-rag-for-projects): automatic RAG mode, 10x capacity, file naming and document references
    - [Manage project visibility and sharing](https://support.claude.com/en/articles/9519189-manage-project-visibility-and-sharing): public and private projects, group sharing
    - [Control project sharing for your organization](https://support.claude.com/en/articles/9927533-control-project-sharing-for-your-organization): admin sharing settings and their effects
    - [Understanding Claude's personalization features](https://support.claude.com/en/articles/10185728-understanding-claude-s-personalization-features): instruction layers, Instructions for Claude scope, project instructions scope and the paid-plans note, skills for format and delivery
    - [Set organization instructions](https://support.claude.com/en/articles/14546867-set-organization-instructions): who sets them, location, 3,000-character limit, propagation time, precedence, examples, writing and testing guidance
    - [Use Claude's chat search and memory](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context): memory defaults, topics, separate project memory spaces, Cowork sharing, sensitive topics, pause and reset, organization controls, chat search, legacy memory
    - [Upload files to Claude](https://support.claude.com/en/articles/8241126-upload-files-to-claude): chat and project upload limits, PDF page handling, XLSX requirement
    - [What are artifacts and how do I use them?](https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them): artifact definition and criteria, code execution requirement, Design, Slides and Docs availability, editing, AI-powered artifacts, persistent storage
    - [Publish and share artifacts](https://support.claude.com/en/articles/9547008-publish-and-share-artifacts): sharing options by plan, account requirement, viewer connections, the External sharing switch, publishing rules
    - [Get started with Claude Docs](https://support.claude.com/en/articles/16923645-get-started-with-claude-docs): comments, attribution, permissions, snapshot charts, availability limits including CMEK, ZDR and HIPAA-ready configurations
    - [Get started with Claude Design](https://support.claude.com/en/articles/14604416-get-started-with-claude-design): availability, editing modes, no version history, Slides for presentations
    - [Custom visuals in chat and Cowork](https://support.claude.com/en/articles/13979539-custom-visuals-in-chat-and-cowork): beta availability, ephemeral visuals, saving as an artifact
    - [Visual and interactive content](https://support.claude.com/en/articles/13641943-visual-and-interactive-content): inline diagrams and charts
    - [Can Claude produce images?](https://support.claude.com/en/articles/9002504-can-claude-produce-images): no photo generation, HTML and SVG visuals
    - [Create and edit files with Claude](https://support.claude.com/en/articles/12111783-create-and-edit-files-with-claude): file types, availability, 30MB limit, network access defaults and egress settings, exfiltration risk, cautious rollout advice
    - [How does Claude handle mathematical equations and calculations?](https://support.claude.com/en/articles/10366421-how-does-claude-handle-mathematical-equations-and-calculations): verifying mission-critical calculations
    - [Enable and use web search](https://support.claude.com/en/articles/10684626-enable-and-use-web-search): citations and how to use them, owner enablement, no toggle in the new experience, web fetch
    - [Use Research on Claude](https://support.claude.com/en/articles/11088861-use-research-on-claude): Research availability, behavior (many searches that build on each other), prerequisites, starting and steering it
    - [When should I use web search, extended thinking, and Research?](https://support.claude.com/en/articles/11095361-when-should-i-use-web-search-extended-thinking-and-research): the three-way decision and examples
    - [Use enterprise search](https://support.claude.com/en/articles/12489464-use-enterprise-search): Ask Your Org project, setup, permission awareness, no indexing, platform support
    - [Claude Cowork and chat are one Claude](https://support.claude.com/en/articles/16761823-claude-cowork-and-chat-are-one-claude): new experience rollout, tool choice, /deep-research, Manual default, instructions location, writing a brief
    - [Use connectors to extend Claude's capabilities](https://support.claude.com/en/articles/11176164-use-connectors-to-extend-claude-s-capabilities): connector basics, enabling versus authenticating, managed auth, action permissions, private projects only, third-party processing
    - [Get started with custom connectors using remote MCP](https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp): MCP definition, custom connector rules (added by Owners), Research auto-invocation and write-capable tools, safety guidance
    - [When to use desktop and web connectors](https://support.claude.com/en/articles/11725091-when-to-use-desktop-and-web-connectors): remote connectors versus desktop extensions; desktop extensions only in Claude Desktop and Claude Code
    - [Getting started with local MCP servers on Claude Desktop](https://support.claude.com/en/articles/10949351-getting-started-with-local-mcp-servers-on-claude-desktop): desktop extensions as single-click local MCP installs and .mcpb files
    - [Manage Claude's tool access](https://support.claude.com/en/articles/13730515-manage-claude-s-tool-access): Auto and On demand modes
    - [Use Google Workspace connectors](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors): Gmail, Calendar and Drive behavior, admin setup, data handling, troubleshooting
    - [Connect to Microsoft 365](https://support.claude.com/en/articles/15183774-connect-to-microsoft-365): the Microsoft 365 connector's scope, plan availability, account and consent requirements
    - [What are skills?](https://support.claude.com/en/articles/12512176-what-are-skills): skill definition, progressive disclosure, kinds of skill, skills versus projects, instructions and MCP, plan availability
    - [Use skills in Claude](https://support.claude.com/en/articles/12512180-use-skills-in-claude): enabling, uploading, sharing and publishing, organization policy settings, risks, add-ins, Claude Code sync
    - [How to create custom skills](https://support.claude.com/en/articles/12512198-how-to-create-custom-skills): recording a skill, and the help center's differing metadata limits
    - [Provision and manage skills for your organization](https://support.claude.com/en/articles/13119606-provision-and-manage-skills-for-your-organization): owner provisioning, testing before provisioning, user-created skills, publishing settings, scanning dates
    - [Use plugins in Claude](https://support.claude.com/en/articles/13837440-use-plugins-in-claude): plugin availability, what a plugin bundles, marketplaces
    - [Use Claude Cowork on Team and Enterprise plans](https://support.claude.com/en/articles/13455879-use-claude-cowork-on-team-and-enterprise-plans): default settings, cloud versus local sessions, local history and retention, plugin distribution options
    - [Is my data used for model training? (commercial)](https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training): commercial products not used for training by default; feedback storage and the Rate chats setting
    - [Cowork is now Claude (blog)](https://claude.com/blog/cowork-is-now-claude): September 16, 2026 merge announcement, rollout order, new Docs and Slides
    - [Skills overview (claude.com docs)](https://claude.com/docs/skills/overview.md): progressive disclosure levels, plan list
    - [Skills how-to (claude.com docs)](https://claude.com/docs/skills/how-to.md): SKILL.md format, name and description rules, length and focus guidance, example and ZIP layout
    - [Google Drive connector (claude.com docs)](https://claude.com/docs/connectors/google/drive.md): Docs size limit, unsupported file types, adding Docs, live sync
    - [Gmail connector (claude.com docs)](https://claude.com/docs/connectors/google/gmail.md): search behavior, citations, and the no-send statement
    - [Google Calendar connector (claude.com docs)](https://claude.com/docs/connectors/google/calendar.md): the no-create statement
    - [Connectors getting started (claude.com docs)](https://claude.com/docs/connectors/getting-started.md): connector best practices
    - [Models overview (Claude Platform docs)](https://platform.claude.com/docs/en/models/overview.md): model descriptions, latency, knowledge cutoffs, starting recommendation
    - [Choosing a model (Claude Platform docs)](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model.md): efficiency-first and capability-first approaches, effort as a lever
    - [Claude Opus 5.5 overview](https://platform.claude.com/docs/en/models/opus-5-5/overview.md): release date
    - [Claude Fable 5.1 overview](https://platform.claude.com/docs/en/models/fable-5-1/overview.md): release date
    - [Agent Skills overview (Claude Platform docs)](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview): cross-surface availability, sharing scope, runtime network, scanning scope, no-sync statement
    - [Skills guide (Claude Platform docs)](https://platform.claude.com/docs/en/build-with-claude/skills-guide): API container, upload limits, version pinning, workspace scope
    - [Claude Code administration setup](https://code.claude.com/docs/en/admin-setup.md): Team or Enterprise as the default Claude Code provider, features that need a claude.ai account
    - [Claude Code MCP documentation](https://code.claude.com/docs/en/mcp.md): claude.ai connectors available in Claude Code
    - [Claude 101: Your first conversation with Claude](https://academy.claude.com/courses/claude-101/your-first-conversation-with-claude): December 2025 model advice, Research timing, deprecated Use style menu, Instructions for Claude path, prompt framework adapted from the 4Ds
    - [Claude 101: Introduction to projects](https://academy.claude.com/courses/claude-101/introduction-to-projects): instruction examples, instructions work alongside user preferences, keeping knowledge current, one-off uploads
    - [Claude 101: Creating with artifacts](https://academy.claude.com/courses/claude-101/creating-with-artifacts): deliverable prompts, artifacts versus files, export formats, end-user descriptions
    - [Claude 101: Working with skills](https://academy.claude.com/courses/claude-101/working-with-skills): projects store knowledge, skills perform tasks; creating skills in conversation; older enable path
    - [Claude 101: Connecting your tools](https://academy.claude.com/courses/claude-101/connecting-your-tools): MCP analogy, connector directory, permissions example
    - [Claude 101: Getting better results](https://academy.claude.com/courses/claude-101/getting-better-results): starting fresh, verifying with web search, the fix for confident but wrong answers, Claude 101 definitions of the 4Ds
    - [Claude 101: Research mode for deep dives](https://academy.claude.com/courses/claude-101/research-mode-for-deep-dives): current Research timing
    - [Academy tutorial: Choosing the right Claude model](https://academy.claude.com/tutorials/choosing-the-right-claude-model): model families, limit use, best-for guidance, task examples, checking in proportion to the stakes
    - [Academy tutorial: How to select the right effort setting](https://academy.claude.com/tutorials/how-to-select-the-right-effort-setting-for-claude-cowork-and-chat): effort symptoms, cost per task
    - [Academy tutorial: Parametric memory and context](https://academy.claude.com/tutorials/parametric-memory-and-context): compaction limits versus written memory, re-sent context, what to persist
    - [Academy tutorial: Prototype AI-powered apps with Claude artifacts](https://academy.claude.com/tutorials/prototype-ai-powered-apps-with-claude-artifacts): Claude inside artifacts, prototype versus production, customized copies
    - [Academy tutorial: Use artifacts to visualize and create AI apps](https://academy.claude.com/tutorials/use-artifacts-to-visualize-and-create-ai-apps-without-ever-writing-a-line-of-code): checking computed outputs, the no-sign-in statement
    - [Academy tutorial: Using Claude Design for presentations](https://academy.claude.com/tutorials/using-claude-design-for-presentations-and-slide-decks): Design for decks
    - [AI Capabilities and Limitations: Knowledge](https://academy.claude.com/courses/ai-capabilities-and-limitations/knowledge): knowledge frozen at the cutoff; retrieval as the fix
    - [Academy: Introduction to Claude Cowork, sharing what you build](https://academy.claude.com/courses/introduction-to-claude-cowork/share-what-you-build-with-your-team): skill naming, owners and quarterly review
    - [Academy: Building effective human-agent teams, practical ways to get started](https://academy.claude.com/courses/building-effective-human-agent-teams/practical-ways-to-get-started): turning repeated explanations into written instructions, human review in the first days, widening scope after good runs
    - [Academy: AI Fluency for small businesses, using data with AI](https://academy.claude.com/courses/ai-fluency-for-small-businesses/using-data-with-ai): replacing names with placeholders
    - [Academy: AI Fluency for nonprofits, understanding privacy and data](https://academy.claude.com/courses/ai-fluency-for-nonprofits/understanding-privacy-and-data): removing identifying information for pattern analysis
    - [CCAO-F prep course: Prompting & Task Execution](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/prompting-task-execution): module anchored on the Description competency
    - [CCAO-F prep course: Workflow Integration & Solution Design](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/workflow-integration-solution-design): module anchored on the Delegation competency
    - [Claude Certified Developer, Foundations prep path](https://anthropic-partners.skilljar.com/path/claude-certified-developer-foundations): AI Fluency on the recommended course list
    - [Claude Certified Architect, Professional prep path](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional): AI Fluency on the recommended course list
    - [Claude Certified Architect, Foundations prep courses](https://anthropic-partners.skilljar.com/page/claude-certified-architect-foundations-prep-courses): AI Fluency on the CCAR-F prep-courses page
    - [Import and export your memory from Claude](https://support.claude.com/en/articles/12123587-import-and-export-your-memory-from-claude): memory import plans, "Start import", experimental status
    - [Use incognito chats](https://support.claude.com/en/articles/12260368-use-incognito-chats): incognito behavior, retention, exports, Compliance API, limits in projects and the new experience
    - [Release notes](https://support.claude.com/en/articles/12138966-release-notes): dated memory changes (July 10 and August 25, 2026)
    - [Get started with Claude in Chrome](https://support.claude.com/en/articles/12012173-get-started-with-claude-in-chrome): what the extension is, plans, surfaces, side panel as a Cowork session, unsupported browsers and devices
    - [Use Claude in Chrome safely](https://support.claude.com/en/articles/12902428-use-claude-in-chrome-safely): prompt injection risk, screenshots, sensitive sites, separate profile, HIPAA exclusion
    - [Claude in Chrome permissions guide](https://support.claude.com/en/articles/12902446-claude-in-chrome-permissions-guide): permission modes, actions requiring explicit permission, prohibited actions, allowlists and blocklists
    - [Claude in Chrome admin controls](https://support.claude.com/en/articles/13065128-claude-in-chrome-admin-controls): Team and Enterprise defaults, September 10, 2026 change, allowlist advice, ZDR, separate management from Cowork
    - [Install Claude Desktop](https://support.claude.com/en/articles/10065433-install-claude-desktop): supported operating systems and versions, Cowork in Desktop, local virtual machine description
    - [Use quick entry with Claude Desktop on Mac](https://support.claude.com/en/articles/12626668-use-quick-entry-with-claude-desktop-on-mac): quick entry availability and shortcuts
    - [Install Claude for iOS](https://support.claude.com/en/articles/9266462-install-claude-for-ios): supported iOS and iPadOS versions
    - [Install Claude for Android](https://support.claude.com/en/articles/9612887-install-claude-for-android): supported Android versions
    - [Use voice mode](https://support.claude.com/en/articles/11101966-use-voice-mode): voice mode availability; dictation versus voice mode
    - [Use dictation on Claude Mobile](https://support.claude.com/en/articles/10065434-use-dictation-on-claude-mobile): dictation audio deletion and no training use
    - [Available beta and research preview features](https://support.claude.com/en/articles/14503520-available-beta-and-research-preview-features): Claude for Word still listed as Beta
    - [What is Claude Tag](https://support.claude.com/en/articles/15594475-what-is-claude-tag): plans, Slack switchover date, billing rules
    - [Get started with Claude Cowork](https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork): architecture, plans, cloud and local requirements, deletion protection, permission modes and connector matrix, scheduling, global and folder instructions
    - [Use Claude Cowork safely](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely): when to use Manual approval, prompt injection conditions, scheduled-task safety, responsibility
    - [Schedule recurring tasks in Claude Cowork](https://support.claude.com/en/articles/13854387-schedule-recurring-tasks-in-claude-cowork): cadence options
    - [Organize your tasks with projects in Claude Cowork](https://support.claude.com/en/articles/14116274-organize-your-tasks-with-projects-in-claude-cowork): Cowork projects and project-scoped memory
    - [Let Claude use your computer in Cowork](https://support.claude.com/en/articles/14128542-let-claude-use-your-computer-in-cowork): computer use plans, tool order, no sandbox, per-app permissions
    - [Assign tasks from anywhere in Claude Cowork](https://support.claude.com/en/articles/13947068-assign-tasks-from-anywhere-in-claude-cowork): Dispatch and its availability
    - [Roles and permissions](https://support.claude.com/en/articles/9267276-roles-and-permissions): one Primary Owner per organization
    - [Set up role-based permissions on Enterprise plans](https://support.claude.com/en/articles/13930458-set-up-role-based-permissions-on-enterprise-plans): custom roles cannot grant disabled features; permission areas; connector default
    - [Set up JIT or SCIM provisioning](https://support.claude.com/en/articles/13133195-set-up-jit-or-scim-provisioning): JIT and SCIM availability, removal behavior, roles by product
    - [Set up single sign-on (SSO)](https://support.claude.com/en/articles/13132885-set-up-single-sign-on-sso): SSO on Team and Enterprise
    - [Claim and migrate accounts on your domain](https://support.claude.com/en/articles/14625619-claim-and-migrate-accounts-on-your-domain): domain capture is Enterprise-only and one-way
    - [Manage members on Team and Enterprise plans](https://support.claude.com/en/articles/13133750-manage-members-on-team-and-enterprise-plans): invitation expiry and seat use
    - [Manage groups and group spend limits on Enterprise plans](https://support.claude.com/en/articles/13799932-manage-groups-and-group-spend-limits-on-enterprise-plans): groups, SCIM sync, 100-group limit
    - [Configuring session security settings](https://support.claude.com/en/articles/13163631-configuring-session-security-settings): maximum session length options
    - [Set a default model for your organization](https://support.claude.com/en/articles/15330088-set-a-default-model-for-your-organization): Enterprise default model options
    - [View usage analytics for Team and Enterprise plans](https://support.claude.com/en/articles/12883420-view-usage-analytics-for-team-and-enterprise-plans): who can see analytics
    - [Access audit logs](https://support.claude.com/en/articles/9970975-access-audit-logs): Enterprise-only, 180 days, identifiers only
    - [Access the Compliance API](https://support.claude.com/en/articles/13015708-access-the-compliance-api): what it returns; Primary Owner enables it
    - [Export your organization's data](https://support.claude.com/en/articles/13346720-export-your-organization-s-data): Primary Owner only; 24-hour link
    - [Configure custom data retention controls for Enterprise plans](https://support.claude.com/en/articles/10440198-configure-custom-data-retention-controls-for-enterprise-plans): minimum period, default, project precedence, exclusions
    - [Who owns and manages the data of my team?](https://support.claude.com/en/articles/9265372-who-owns-and-manages-the-data-of-my-team): Primary Owner manages data; consumer terms do not apply to Claude for Work
    - [Share and unshare chats](https://support.claude.com/en/articles/10593882-share-and-unshare-chats): snapshot contents and exclusions
    - [Public links for shared chats](https://support.claude.com/en/articles/16762437-public-links-for-shared-chats): public links by plan, noindex, "treat a public link as public"
    - [Share a chat with specific people](https://support.claude.com/en/articles/16762496-share-a-chat-with-specific-people): email sharing on all plans and its limits
    - [Claude is providing incorrect or misleading responses](https://support.claude.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on): hallucination, not a single source of truth, review cited sources
    - [Claude is producing links that don't work and falsely claiming it has sent emails](https://support.claude.com/en/articles/8241188-claude-is-producing-links-that-don-t-work-and-falsely-claiming-that-it-has-sent-emails-or-produced-external-documents-what-s-going-on): false capability claims
    - [Using Claude for legal work](https://support.claude.com/en/articles/15707726-using-claude-for-legal-work-privilege-confidentiality-and-how-to-think-about-configuration): lawyer in the loop, primary sources, documenting AI use
    - [How Claude marks AI-generated content](https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content): watermarks, C2PA credentials, missing marks
    - [Claude for Microsoft 365 overview](https://claude.com/docs/office-agents/overview.md): what the add-ins are
    - [Claude for Excel](https://claude.com/docs/office-agents/excel.md): availability, strengths, not-recommended uses, local history, retention, trusted files
    - [Claude for PowerPoint](https://claude.com/docs/office-agents/powerpoint.md): general availability
    - [Claude for Word](https://claude.com/docs/office-agents/word.md): general availability
    - [Claude for Outlook](https://claude.com/docs/office-agents/outlook.md): beta status
    - [Work across apps](https://claude.com/docs/office-agents/work-across-apps.md): per-device cross-file toggle defaults and the organization setting
    - [Claude for Chrome (blog)](https://claude.com/blog/claude-for-chrome): August 2025 pilot figures; sensitive-site advice
    - [Is my data used for model training? (consumer)](https://privacy.claude.com/en/articles/10023580-is-my-data-used-for-model-training): incognito chats are not used for training
    - [Mitigating the risk of prompt injections in browser use (Anthropic research)](https://www.anthropic.com/research/prompt-injection-defenses): residual risk of browser agents
    - [Consumer Terms of Service](https://www.anthropic.com/legal/consumer-terms): do not rely on outputs without independent confirmation
    - [Evaluating and mitigating discrimination in language model decisions (Anthropic research)](https://www.anthropic.com/research/evaluating-and-mitigating-discrimination-in-language-model-decisions): no automated decisions for high-risk use cases
    - [AI Fluency: Framework and foundations (Claude Academy)](https://academy.claude.com/courses/ai-fluency-framework-foundations): course origin, authors, objectives
    - [Claude Academy courses](https://academy.claude.com/courses): course length and badge for AI Fluency
    - [Introduction to AI Fluency](https://academy.claude.com/courses/ai-fluency-framework-foundations/introduction-to-ai-fluency): definition of AI Fluency
    - [The 4D Framework](https://academy.claude.com/courses/ai-fluency-framework-foundations/the-4d-framework): the 4Ds and the three modes of engagement
    - [A closer look at Delegation](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-delegation): Problem, Platform and Task Delegation; the goal of delegation
    - [A closer look at Description](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-description): Product, Process and Performance Description
    - [Effective prompting techniques](https://academy.claude.com/courses/ai-fluency-framework-foundations/effective-prompting-techniques): the six techniques and the "secret weapon"
    - [A closer look at Discernment](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-discernment): Product, Process and Performance Discernment; the feedback loop
    - [The Description-Discernment loop](https://academy.claude.com/courses/ai-fluency-framework-foundations/the-description-discernment-loop): the describe, discern, refine, integrate steps
    - [A closer look at Diligence](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-diligence): Creation, Transparency and Deployment Diligence; diligence statement and template
    - [Claude 101: The Claude desktop app (Chat, Cowork, Code)](https://academy.claude.com/courses/claude-101/claude-desktop-app-chat-cowork-code): the three kinds of desktop work
    - [Claude 101: Other ways to work with Claude](https://academy.claude.com/courses/claude-101/other-ways-to-work-with-claude): Claude in Chrome described as generally available
    - [Introduction to Claude Cowork: What is Cowork](https://academy.claude.com/courses/introduction-to-claude-cowork/what-is-cowork): chat, Cowork and Code recap; staying in control
    - [Introduction to Claude Cowork: Giving Cowork context](https://academy.claude.com/courses/introduction-to-claude-cowork/giving-cowork-context): what goes in global instructions; sessions start fresh outside a project
    - [Introduction to Claude Cowork: The task loop](https://academy.claude.com/courses/introduction-to-claude-cowork/the-task-loop): naming the deliverable, steering mid-run, review questions
    - [Introduction to Claude Cowork: Scheduled tasks](https://academy.claude.com/courses/introduction-to-claude-cowork/scheduled-tasks): the three delegation patterns; do once then schedule; course cadence list
    - [AI Capabilities and Limitations: Next Token Prediction](https://academy.claude.com/courses/ai-capabilities-and-limitations/next-token-prediction): fabrication concentrates in specifics
    - [AI Capabilities and Limitations: Working memory](https://academy.claude.com/courses/ai-capabilities-and-limitations/working-memory): the model does not learn from corrections; the working-memory cliff
    - [AI Capabilities and Limitations: Steerability](https://academy.claude.com/courses/ai-capabilities-and-limitations/steerability): restate the goal rather than repeat the instruction
    - [AI Capabilities and Limitations: How AI gets its character](https://academy.claude.com/courses/ai-capabilities-and-limitations/how-ai-gets-its-character): loose confidence calibration; the disagreement test
    - [AI Fluency for nonprofits: Workflow augmentation](https://academy.claude.com/courses/ai-fluency-for-nonprofits/workflow-augmentation): "should AI do this?" and the task categories
    - [AI Fluency for small businesses: Tying it all together](https://academy.claude.com/courses/ai-fluency-for-small-businesses/tying-it-all-together): review customer-facing outputs, disclose AI's role, path to a human
    - [Building effective human-agent teams: What a strong team looks like](https://academy.claude.com/courses/building-effective-human-agent-teams/what-a-strong-team-looks-like): autonomy in proportion to demonstrated reliability
    - [Deploying Claude Enterprise with confidence: Adoption signals](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/adoption-signals): usage metrics are diagnostics, not quotas
    - [The AI Fluency Index (Academy tutorial)](https://academy.claude.com/tutorials/the-ai-fluency-index): iteration statistics; polished outputs lower scrutiny
    - [Discernment toolkit (Academy tutorial)](https://academy.claude.com/tutorials/discernment-toolkit): external verification, domain-expertise barrier, panel of experts, source-grounding prompt
    - [Why do AI models hallucinate? (Academy tutorial)](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate): high-risk situations, tips, fresh-chat review
    - [What is sycophancy in AI models? (Academy tutorial)](https://academy.claude.com/tutorials/what-is-sycophancy-in-ai-models): definition, triggers, countermeasures
    - [Why does bias exist in AI models? (Academy tutorial)](https://academy.claude.com/tutorials/why-does-bias-exist-in-ai-models): forms of bias and five moves against it
    - [Writing an AI diligence statement (Academy tutorial)](https://academy.claude.com/tutorials/writing-an-ai-diligence-statement): elements of a statement; consistency checks
    - [Can you trust what AI tells you? (Academy tutorial)](https://academy.claude.com/tutorials/can-you-trust-what-ai-tells-you): trust as a dial
    - [Scaling workflows with Claude Cowork at your organization (Academy tutorial)](https://academy.claude.com/tutorials/scaling-workflows-with-claude-cowork-at-your-organization): the chat trap; accountability stays with people; tutorial cadence list
    - [Generate project status reports (Academy use case)](https://academy.claude.com/use-cases/generate-project-status-reports): the status-report brief and explicit gap handling
    - [PRD from a one-pager (Academy use case)](https://academy.claude.com/use-cases/prd-from-a-one-pager): "flag anything you're not confident about"
    - [Adapt a standard textbook page to every reading level (Academy use case)](https://academy.claude.com/use-cases/adapt-a-standard-textbook-page-to-every-reading-level): ask for a list of what changed and what was left out
    - [My voice (Academy use case)](https://academy.claude.com/use-cases/my-voice): feeding corrections back into a voice skill
    - [Framework for AI Fluency, Version 1.1 (Dakan and Feller, Ringling College)](https://ringling.libguides.com/ai/framework): the framework document's 4D definitions, sub-competency names and modality examples
    - [AI Capabilities and Limitations: When properties collide](https://academy.claude.com/courses/ai-capabilities-and-limitations/when-properties-collide): offload to code execution as a standard fix
    - [Claude Code documentation: Skills](https://code.claude.com/docs/en/skills.md): sync of Claude account skills into Claude Code, including built-in skills
    - [Purchase and manage seats on Team plans](https://support.claude.com/en/articles/12004354-purchase-and-manage-seats-on-team-plans): who can purchase seats on Team
    - [Restrict access to Claude with IP allowlisting](https://support.claude.com/en/articles/13200993-restrict-access-to-claude-with-ip-allowlisting): IP allowlisting on Enterprise and how to request it
    - [Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms): customer duty to evaluate outputs and notify users (section D.3)
