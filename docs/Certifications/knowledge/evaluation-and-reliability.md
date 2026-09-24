---
title: "Evaluation, Debugging and Reliability for the Claude Certifications"
description: Success criteria, test sets, graders, agent evals, debugging, error propagation, escalation, human review and model upgrades for the four Claude exams.
last_reviewed: 2026-09-23
---

# Evaluation, Debugging and Reliability

This page teaches how to prove a Claude application works and how to find out why it does not: success criteria and test sets, graders, agent evaluations, and the debugging method that separates integration faults from model behavior, followed by error propagation, escalation, human review, reliability engineering and safe model upgrades. Each section opens with the exam objectives it serves. Product facts are as of September 2026; where the documentation changed after the July 2026 exam guides, the section shows both versions and which wording to expect on the exam (the guide's).

| Exam | Where this topic sits in the blueprint | Weight |
|---|---|---|
| CCAO-F | Domain 2: Output Evaluation and Validation; Domain 7: Troubleshooting and Optimization | 21%; 10% |
| CCDV-F | Domain 4: Eval, Testing, and Debugging (one skill, Debugging and Error Handling); related skills in other domains (D2.2, D2.6, D5.2, D5.3, D6.3, D8.1) are named in the sections below | 2.6% (Domain 4 only) |
| CCAR-F | Domain 5: Context Management & Reliability (Tasks 5.2, 5.3 and 5.5 live here); Task 4.6, multi-instance review, sits in Domain 4: Prompt Engineering & Structured Output; related task statements in other domains are named in the sections below | 15%; 20% |
| CCAR-P | Domain 4: Evaluation, Testing & Optimization; Domain 5: Governance, Safety & Risk Management adds guardrails, failure modes and human-in-the-loop validation (5.1 to 5.3); Domain 7: Developer Productivity & Operational Enablement adds debugging and operational issue resolution (7.3); objectives 1.4, 3.4 and 6.5 are named in the sections below | 16%; 14%; 7% |

The weights understate the reach. The [CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) lists "Designing and running evals, debugging failure modes through trace analysis, validating structured output, and monitoring production quality" among the abilities the credential validates, and its preparation advice asks you to build an application that "includes simple security and evaluation practices". The [CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf) asks you to "Build and operate at least one end-to-end Claude solution, including RAG, evaluation, and observability".

## Success criteria and test sets

*Tested in: CCAR-P 4.1, 4.2, 4.3 · CCDV-F purpose area 5 ("Designing and running evals") and How to Prepare step 3 (Domain 4's only skill, D4.1, covers debugging and error handling, not eval design) · CCAR-F 5.5-K1, 5.5-K4, 5.5-S2, How to Prepare item 3 · CCAO-F D2.1*

An eval is "a test for an AI system: give an AI an input, then apply grading logic to its output to measure success" ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)). You cannot write one until you know what success is. Anthropic's page [Define success criteria and build evaluations](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests) opens with the order of work: "Building a successful LLM-based application starts with clearly defining your success criteria and then designing evaluations to measure performance against them. This cycle is central to prompt engineering." The page's flowchart runs from test cases to a preliminary prompt, then iterative testing and refinement, final validation, and ship. The [prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview) makes the same order a precondition: before you start prompt engineering it assumes you have "A clear definition of the success criteria for your use case", "Some ways to empirically test against those criteria" and a first draft prompt to improve, and if you do not, it says to establish them first.

### Four properties of a good criterion

From [Define success criteria and build evaluations](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests):

| Property | What the docs say | Weak | Strong |
|---|---|---|---|
| Specific | Clearly define what you want to achieve | "good performance" | "accurate sentiment classification" |
| Measurable | Quantitative metrics or well-defined qualitative scales; numbers provide clarity and scalability, and qualitative measures can be valuable if applied consistently along with quantitative ones | "Safe outputs" | "Less than 0.1% of outputs out of 10,000 trials flagged for toxicity by the content filter." |
| Achievable | Targets based on industry benchmarks, prior experiments, AI research or expert knowledge; not unrealistic for current frontier model capabilities | A target beyond what current frontier models can do (our example) | "a 5% improvement over the current baseline" |
| Relevant | Aligned with the application's purpose and user needs | The same bar for every application (our example) | Strong citation accuracy, which might be critical for a medical app but less so for a casual chatbot |

The [docs'](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests) worked example turns "The model should classify sentiments well" into an F1 score of at least 0.85 on a held-out test set of 10,000 diverse Twitter posts, a 5% improvement over the current baseline. Most applications need several such lines at once: "Most use cases need multidimensional evaluation along several success criteria." The multidimensional version of the same example adds that 99.5% of outputs are non-toxic, that 90% of errors would cause inconvenience rather than egregious error, and that 95% of responses arrive in under 200ms, with a note that in practice you would also define what inconvenience and egregious mean.

The [same page](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests) lists eight criteria "that might be important for your use case" and says "This list is non-exhaustive": task fidelity, consistency, relevance and coherence, tone and style, privacy preservation, context utilization, latency, and price.

### Metrics for the five dimensions CCAR-P names

CCAR-P objective 4.1 in the [CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf) is "Define evaluation metrics (accuracy, latency, cost, safety, security)". Each dimension has a sourced way to measure it. The metrics come from Anthropic's material; placing each one under a dimension is our mapping (privacy of health information, for example, could sit under safety or security):

| Dimension | Metrics and methods in Anthropic's material | Example targets and figures |
|---|---|---|
| Accuracy (task fidelity) | Task-specific: F1, BLEU, perplexity. Generic: accuracy, precision, recall. Edge case analysis: the percentage of edge cases handled without errors. For RAG, retrieval precision, recall, F1 and mean reciprocal rank (MRR), measured separately from end-to-end answer accuracy | F1 of at least 0.85; at least 80% routing accuracy on an edge-case set |
| Latency and operations | Response time (ms) and uptime (%), the docs' operational metrics; time to first token (TTFT), particularly relevant when you stream responses | 95% of responses under 200ms |
| Cost | Price: cost per API call, model size and frequency of use; cost per task tracked on a static bank of tasks | Cost per classification cut by 50% on average (across 100 tests) versus the current routing method |
| Safety | Toxicity rate from a content filter; an LLM-graded binary check for protected health information (PHI) in responses | Under 0.1% of 10,000 trials flagged; 99.5% non-toxic |
| Security | Attack success rate when you red-team the workflow with documents, emails and tool outputs that deliberately contain injection attempts | A measured figure, not a target: browser use without Anthropic's safety mitigations (Claude in Chrome pilot, post dated August 2025) showed a 23.6% attack success rate across 123 test cases representing 29 attack scenarios |

For security, a low number is still not zero risk. Anthropic's research post [Mitigating the risk of prompt injections in browser use](https://www.anthropic.com/research/prompt-injection-defenses), a separate November 2025 evaluation against an adaptive Best-of-N attacker (not comparable with the 23.6% pilot figure above), says a 1% attack success rate "still represents meaningful risk" and that no browser agent is immune to prompt injection. The docs page [Mitigate jailbreaks and prompt injections](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks) says to red-team your own agent: before deploying, test the workflow with documents, emails and tool outputs that deliberately contain injection attempts, and "confirm that Claude ignores them and that your screening and confirmation steps catch the rest". The defenses themselves are in [Prompt injection](security-and-governance.md#prompt-injection).

!!! note "Scope differs by exam"

    The [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) lists "Performance benchmarking or model comparison metrics" and "Rate limiting, quotas, or API pricing calculations" among topics that "will not appear on the exam". The CCAR-F exclusion does not carry over to CCAR-P, whose objectives include "Define evaluation metrics (accuracy, latency, cost, safety, security)" (4.1) and "Conduct A/B testing and iterative improvements" (4.3), or to the [CCDV-F credential](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf), which lists "Designing and running evals" among the areas it validates. Neither guide uses the words benchmarking or model comparison; placing the statistical comparison material below under those objectives is our reading. CCAR-F still tests measurement through Task 5.5: segment-level accuracy (5.5-K1, 5.5-K4, 5.5-S2) and confidence calibrated on labeled validation sets (5.5-S3).

### Targets from Anthropic's use-case guides

From the [customer support](https://platform.claude.com/docs/en/about-claude/use-case-guides/customer-support-chat) and [ticket routing](https://platform.claude.com/docs/en/about-claude/use-case-guides/ticket-routing) guides:

| Use case | Metric | How the guide measures it | Target the guide gives |
|---|---|---|---|
| Customer support | Query comprehension accuracy | Review a sample of conversations for the correct interpretation of customer intent | 95% or higher |
| Customer support | Response relevance | LLM-based grading of each response in a set of conversations | 90% or above |
| Customer support | Escalation efficiency | Correctly escalated conversations versus ones that should have been escalated but were not | Escalation accuracy of 95% or higher |
| Customer support | Deflection rate | Share of inquiries handled without human intervention | Typically 70 to 80%, depending on inquiry complexity |
| Ticket routing | Classification consistency | Periodic tests with a set of standardized inputs | 95% or higher |
| Ticket routing | Edge case handling | A dedicated test set of edge cases | At least 80% |
| Ticket routing | Multilingual handling | Routing accuracy measured per language | No more than a 5 to 10% drop for non-primary languages |
| Ticket routing | Bias mitigation | Routing decisions audited across customer groups | Accuracy consistent within 2 to 3% |
| Ticket routing | Explainability score | Human raters score explanations on a scale (for example 1 to 5) | Average of 4 or higher |

These are the guides' examples for their own use cases, not universal thresholds. The [ticket-routing guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/ticket-routing) states the principle behind them: "A proper evaluation requires clear thresholds and benchmarks to determine what is a good result." Its own example thresholds are 95% accuracy out of 100 tests and a 50% average reduction in cost per classification across 100 tests. The [content moderation guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/content-moderation) adds continuous evaluation: regularly assess the system with metrics such as precision and recall, and use the data to refine prompts and criteria.

### Building the test set

The [evals cookbook](https://github.com/anthropics/claude-cookbooks/blob/main/misc/building_evals.ipynb) says evals typically have four parts: an input prompt, the model's output, a golden answer, and a score. Writing questions and golden answers is usually a one-time cost, while grading is paid on every re-run, so "building evals that can be quickly and cheaply graded should be at the center of your design choices."

The [develop-tests page](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests) gives three design principles:

1. **Be task-specific.** Mirror the real-world task distribution and include edge cases. The docs' examples: irrelevant or nonexistent input data; overly long input data or user input; for chat, poor, harmful or irrelevant user input; and ambiguous cases where even humans would struggle to agree on an assessment.
2. **Automate when possible.** Structure questions so they can be graded automatically: multiple-choice, string match, code-graded or LLM-graded.
3. **Prioritize volume over quality.** "More questions with slightly lower signal automated grading is better than fewer questions with high-quality human hand-graded evals."

**Where tasks come from.** [Anthropic's advice](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) is to start early rather than wait for hundreds of tasks: "In reality, 20-50 simple tasks drawn from real failures is a great start." Early changes have large effects, so small sets detect them; more mature agents may need larger, harder evals to detect smaller effects. Evals also get harder to build the longer you wait. Begin with the manual checks you already run during development (the behaviors you verify before each release and common tasks end users try); in production, look at the bug tracker and support queue and convert user-reported failures into test cases, prioritized by user impact. Anthropic's own [multi-agent Research system](https://www.anthropic.com/engineering/multi-agent-research-system) started with "a set of about 20 queries representing real usage patterns". Breaking an interaction into every task Claude performs (the customer support guide's advice) shows you the range of interactions your cases must cover, and Claude can generate more cases from a baseline set of examples. Keep the mix close to the real distribution of questions and difficulties.

**What makes a task usable.** From [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents):

- "A good task is one where two domain experts would independently reach the same pass/fail verdict." Ambiguity in task specifications becomes noise in metrics, and vague rubrics for model-based graders produce inconsistent judgments.
- Everything the grader checks should be clear from the task description; agents should not fail because of an ambiguous spec. With frontier models, a 0% pass rate across many trials is "most often a signal of a broken task, not an incapable agent", and a sign to double-check the task specification and graders.
- Write a reference solution for each task: a known working output that passes every grader, which proves the task is solvable and the graders are configured correctly.
- Balance the set. "Test both the cases where a behavior should occur and where it shouldn't. One-sided evals create one-sided optimization." The post's example: test only whether an agent searches when it should, and you may end up with an agent that searches for almost everything. Anthropic's web search evals for Claude.ai covered both queries where the model should search and queries it should answer from existing knowledge. The [CCAR-F exam guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) preparation advice has a related item for tools: "Test tool selection reliability with ambiguous requests." Connecting that item to balanced sets is our reading.

**Hold data out.** The docs' sentiment criterion is defined on a held-out test set, and Anthropic's tool-building work relied on held-out test sets so it would not overfit to the evaluations it iterated on. As of September 2026, `/claude-api hillclimb` in Claude Code (described in the September 2026 post [Reducing cost and improving performance with Claude Platform](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform)) automates the same discipline: it "splits your evaluation into train and test sets, proposes configuration changes, and reads failing train examples to fix what it finds." The post adds that the final configuration is scored on the held-out test set.

**Tag every case with its segment.** [CCAR-F](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) Task 5.5 warns that "aggregate accuracy metrics (e.g., 97% overall) may mask poor performance on specific document types or fields", 5.5-K4 stresses "validating accuracy by document type and field segment before automating high-confidence extractions", and 5.5-S2 asks you to analyze accuracy by document type and field before reducing human review. A test set can only answer that if each case records its segment: document type and field in the guide's example, and language, customer group or edge-case class in the routing guide's (our list of tags). The routing guide's per-language and per-group targets above are segment targets. How segment results decide where people keep reviewing is in [Human review and confidence calibration](#human-review-and-confidence-calibration).

**For Associates (CCAO-F D2.1).** The [Associate exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) asks you to "Evaluate Claude-generated outputs for accuracy and completeness". The same discipline works on a single output: decide what accurate and complete mean for the task before you read the draft. Two check questions from the rubric Anthropic's [multi-agent Research system](https://www.anthropic.com/engineering/multi-agent-research-system) used for its LLM judge fit this directly: factual accuracy ("do claims match sources?") and completeness ("are all requested aspects covered?"). Applying them to one document is our suggestion. When a person must check before the output is used is covered in [Human review and confidence calibration](#human-review-and-confidence-calibration), and checking habits for the Claude apps in [Verifying Claude's output](claude-for-work.md#verifying-claudes-output).

### Is the difference real?

Model outputs vary between runs, so one run per case tells you little. Anthropic's research post [A statistical approach to model evaluations](https://www.anthropic.com/research/statistical-approach-to-model-evals) treats eval questions as a sample drawn from a "question universe" and makes five recommendations:

| Recommendation (the post's heading) | What to do |
|---|---|
| Use the Central Limit Theorem | Report each score with its standard error of the mean (SEM); a 95% confidence interval is the mean plus or minus 1.96 × SEM |
| Cluster standard errors | When questions come in related groups (several questions about one passage), cluster on the unit of randomization; clustered errors on popular evals "can be over three times as large as naive standard errors", so ignoring clustering can show a difference that does not exist |
| Reduce variance within questions | With chain-of-thought, resample each question several times and use question-level averages as the question scores |
| Analyze paired differences | When two variants answer the same questions, a paired-differences test removes the variance from question difficulty; report mean differences, standard errors, confidence intervals and correlations |
| Use power analysis | Work out how many questions you need to detect the difference you care about (the post's example hypothesis: one model beats another by 3 percentage points); on small evals, "small differences will likely go undetected" |

An offline eval compares variants on your test set. The [develop-tests page](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests) also lists A/B testing, comparing "against a baseline model or earlier version", and [an A/B test on live traffic](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) measures real user outcomes but is slow ("days or weeks to reach significance") and needs enough traffic. Where each method fits over a product's life is in [Evals across the product's life](#evals-across-the-products-life); planning an A/B test for a delivery program is in [Evaluation strategy for a program](solution-architecture.md#evaluation-strategy-for-a-program).

!!! info "The Console evaluation tool (as of September 2026)"

    A July 2024 Anthropic post, [Evaluate prompts in the developer console](https://claude.com/blog/evaluate-prompts), described an Evaluate feature in what was then called the Anthropic Console: add test cases by hand, import them from a CSV, or have Claude generate them with "Generate Test Case"; run all test cases in one click; create new prompt versions and compare the outputs of two or more prompts side by side; and have subject matter experts grade response quality on a 5-point scale. The old docs URLs for the tool (`test-and-evaluate/eval-tool` and `test-and-evaluate/define-success`) now redirect to the combined develop-tests page, and that page does not describe the Console tool. Treat the 2024 description as background, not as a guide to the current interface.

### Decide

- If a stakeholder states a goal such as "safe" or "accurate", turn it into a number, a test set and a threshold before building; not a prompt tweak, because without a measurable criterion no change can be shown to help.
- If you can have many automatically graded cases or a few hand-graded ones, choose the many; not the few, because the docs prioritize volume over quality.
- If the overall score is high, break it down by segment before automating further; not the headline number, because a 97% aggregate can hide a failing document type.
- If two prompt versions differ by a point or two on a small set, check the confidence interval and use a paired comparison; not a ship decision on the raw difference.
- If you are starting from nothing, collect 20 to 50 tasks from real failures now; not a wait for hundreds of cases.

### Traps

- **Vague criteria.** The docs' own examples of criteria to rewrite are "good performance" and "Safe outputs".
- **Targets beyond reach.** The docs say targets should not be unrealistic for current frontier model capabilities; base them on industry benchmarks, prior experiments, AI research or expert knowledge.
- **One-sided sets.** Testing only the cases where a behavior should happen optimizes toward a system that does it almost everywhere, like the agent that searches for almost everything.
- **Tuning on the test set.** Iterating against the same cases you report risks overfitting to them (our wording of why Anthropic relied on held-out sets); keep a held-out set.
- **Blaming the model for a 0% task.** With frontier models, zero passes across many trials most often signals a broken task; double-check the task specification and the graders.

## Grading methods

*Tested in: CCAR-P 4.2 · CCAR-F 4.6-K1, 4.6-K2, 4.6-S1, 5.5-S3, sample question 12 (option D rationale) · CCDV-F purpose area 5 ("Designing and running evals"); Domain 4's only skill, D4.1, covers debugging and error handling rather than grader design · CCAO-F: not listed*

A grader is logic that scores some aspect of a system's performance; a task can have several graders, each holding several assertions, sometimes called checks ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)). CCAR-P objective 4.2 in the [CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf) asks you to "Design evaluation datasets and test frameworks using mixed methodologies". Our reading of that in practice: pick, for each criterion, the fastest and most reliable grader that can measure it, and combine graders where one kind is not enough (the post notes that agent evals typically combine all three kinds). The [develop-tests page](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests) gives the rule of choice: "choose the fastest, most reliable, most scalable method". The [evals post](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) puts it as an order of preference: "We recommend choosing deterministic graders where possible, LLM graders where necessary or for additional flexibility, and using human graders judiciously for additional validation."

### The three kinds of grader

From [develop-tests](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests), [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) and the [evals cookbook](https://github.com/anthropics/claude-cookbooks/blob/main/misc/building_evals.ipynb):

| Aspect | Code-based | LLM-based (model-based) | Human |
|---|---|---|---|
| Methods | Exact match (`output == golden_answer`), string match (`key_phrase in output`), regex or fuzzy match, binary tests (fail-to-pass, pass-to-pass), static analysis (lint, type, security), outcome verification, tool-call verification (tools used, parameters), transcript analysis (turns taken, token usage) | Rubric-based scoring, natural language assertions, pairwise comparison, reference-based evaluation, multi-judge consensus | Subject matter expert (SME) review, crowdsourced judgment, spot-check sampling, A/B testing, inter-annotator agreement |
| Strengths | Fastest and most reliable, extremely scalable; cheap, objective, reproducible, easy to debug; verifies specific conditions | Fast, flexible and scalable, suitable for complex judgment; captures nuance; handles open-ended tasks and freeform output | Most flexible and high quality; gold standard quality that matches expert user judgment; used to calibrate model-based graders |
| Weaknesses | Lacks nuance; brittle to valid variations that do not match expected patterns exactly; limited for some more subjective tasks | Non-deterministic; more expensive than code; requires calibration with human graders for accuracy | Slow and expensive; often requires access to human experts at scale |
| Verdict in the sources | "by far the best grading method if you can design an eval that allows for it" (cookbook) | "Test to ensure reliability first then scale." (develop-tests) | "Avoid if possible." (develop-tests) |

The [cookbook's](https://github.com/anthropics/claude-cookbooks/blob/main/misc/building_evals.ipynb) tactic for moving work into the first column: "Reformatting questions into multiple choice is a common tactic here." If an open question can become a choice among fixed labels, a string comparison can grade it.

### Code-based graders

The [develop-tests](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests) exact-match eval grades 1,000 tweets with human-labeled sentiment, with sarcasm and mixed sentiment as deliberate edge cases. The code below is the page's own, which leaves out the import lines. Note that the completion helper selects the text block by its `type`, not by position; reading `content[0].text` breaks when a thinking block comes before the text:

=== "Python"

    ```python
    tweets = [
        {"text": "This movie was a total waste of time. 👎", "sentiment": "negative"},
        {"text": "The new album is 🔥! Been on repeat all day.", "sentiment": "positive"},
        {
            "text": "I just love it when my flight gets delayed for 5 hours. #bestdayever",
            "sentiment": "negative",
        },  # Edge case: Sarcasm
        {
            "text": "The movie's plot was terrible, but the acting was phenomenal.",
            "sentiment": "mixed",
        },  # Edge case: Mixed sentiment
        # ... 996 more tweets
    ]

    client = anthropic.Anthropic()

    def get_completion(prompt: str):
        message = client.messages.create(
            model="claude-opus-5-5",
            max_tokens=50,
            messages=[{"role": "user", "content": prompt}],
        )
        return next(block.text for block in message.content if block.type == "text")

    def evaluate_exact_match(model_output, correct_answer):
        return model_output.strip().lower() == correct_answer.lower()

    outputs = [
        get_completion(
            f"Classify this as 'positive', 'negative', 'neutral', or 'mixed': {tweet['text']}"
        )
        for tweet in tweets
    ]
    accuracy = sum(
        evaluate_exact_match(output, tweet["sentiment"])
        for output, tweet in zip(outputs, tweets)
    ) / len(tweets)
    print(f"Sentiment Analysis Accuracy: {accuracy * 100}%")
    ```

=== "TypeScript"

    ```typescript
    const tweets = [
      { text: "This movie was a total waste of time. 👎", sentiment: "negative" },
      { text: "The new album is 🔥! Been on repeat all day.", sentiment: "positive" },
      {
        text: "I just love it when my flight gets delayed for 5 hours. #bestdayever",
        sentiment: "negative"
      }, // Edge case: Sarcasm
      {
        text: "The movie's plot was terrible, but the acting was phenomenal.",
        sentiment: "mixed"
      } // Edge case: Mixed sentiment
      // ... 996 more tweets
    ];

    const client = new Anthropic();

    async function getCompletion(prompt: string): Promise<string> {
      const message = await client.messages.create({
        model: "claude-opus-5-5",
        max_tokens: 50,
        messages: [{ role: "user", content: prompt }]
      });
      const textBlock = message.content.find((block) => block.type === "text");
      return textBlock ? textBlock.text : "";
    }

    function evaluateExactMatch(modelOutput: string, correctAnswer: string): boolean {
      return modelOutput.trim().toLowerCase() === correctAnswer.toLowerCase();
    }

    let correctCount = 0;
    for (const tweet of tweets) {
      const output = await getCompletion(
        `Classify this as 'positive', 'negative', 'neutral', or 'mixed': ${tweet.text}`
      );
      if (evaluateExactMatch(output, tweet.sentiment)) {
        correctCount++;
      }
    }
    console.log(`Sentiment Analysis Accuracy: ${(correctCount / tweets.length) * 100}%`);
    ```

Deterministic and metric graders go well beyond exact match, and Anthropic's use-case material often pairs them with an LLM rubric or expert review (the legal and code rows below). From [develop-tests](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests), the [RAG cookbook](https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/retrieval_augmented_generation/guide.ipynb), the [legal summarization guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/legal-summarization), the [text-to-SQL cookbook](https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/text_to_sql/guide.ipynb) and [Demystifying evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents):

| What you measure | Grader | Example in Anthropic's material |
|---|---|---|
| Categorical labels | Exact match after normalizing whitespace and case | 1,000 tweets with human-labeled sentiment |
| Consistency across paraphrases | Cosine similarity of Sentence-BERT embeddings of the answers | 50 groups of paraphrased questions |
| Summary relevance and coherence | ROUGE-L, the longest common subsequence against a reference summary | 200 articles with reference summaries |
| Legal summaries | ROUGE, BLEU, contextual embedding similarity, an LLM rubric, and expert review of a few summaries before production | Legal summarization guide |
| Retrieval in RAG | Precision, recall, F1 and MRR against the expected chunks, scored separately from end-to-end answer accuracy | 100 synthetic samples, each a question, its expected chunks and a correct answer |
| Generated SQL | Execute it against a test database and compare with expected results; also check syntax, semantics and complex queries | Text-to-SQL cookbook |
| Code | Unit tests for correctness, typically paired with an LLM rubric for overall code quality | SWE-bench Verified passes a solution only if it fixes the failing tests without breaking existing ones |

A common way for code graders to fail is by being too strict. Anthropic's tool-building guidance says to "Avoid overly strict verifiers that reject correct responses due to spurious differences like formatting, punctuation, or valid alternative phrasings" ([Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents)). The [evals post](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) gives a cautionary number: Opus 4.5 first scored 42% on CORE-Bench, partly because rigid grading penalized "96.12" when the grader expected "96.124991…"; after fixing grading bugs, ambiguous specs and stochastic tasks, and using a less constrained scaffold, the score rose to 95%. The same post finds the path-level version of this rigidity (requiring a very specific sequence of tool calls) too brittle; that is taught in [Grade the outcome; check the path only where the path is a requirement](#grade-the-outcome-check-the-path-only-where-the-path-is-a-requirement).

### LLM-based graders

The [develop-tests page](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests) gives three rules for LLM graders:

1. **Detailed, clear rubrics.** Its example: "The answer should always mention 'Acme Inc.' in the first sentence. If it does not, the answer is automatically graded as 'incorrect.'" One use case, or even one criterion, may need several rubrics.
2. **Empirical or specific output.** Ask for only 'correct' or 'incorrect', or a score from 1 to 5. "Purely qualitative evaluations are hard to assess quickly and at scale."
3. **Reason first, then discard the reasoning.** This improves grading, particularly for complex judgment.

The page's grader applies all three: a rubric, a reasoning block, and a binary verdict in tags that code can parse. Below is the grading function from the page (imports omitted there too); in the page's usage example, each item's golden answer is what gets passed in as the rubric.

=== "Python"

    ```python
    client = anthropic.Anthropic()

    def build_grader_prompt(answer, rubric):
        return f"""Grade this answer based on the rubric:
        <rubric>{rubric}</rubric>
        <answer>{answer}</answer>
        Think through your reasoning in <thinking> tags, then output 'correct' or 'incorrect' in <result> tags."""

    def grade_completion(output, golden_answer):
        grader_message = client.messages.create(
            model="claude-opus-5-5",
            max_tokens=2048,
            messages=[
                {"role": "user", "content": build_grader_prompt(output, golden_answer)}
            ],
        )
        grader_response = next(
            block.text for block in grader_message.content if block.type == "text"
        )

        return (
            "correct"
            if "<result>correct</result>" in grader_response.lower()
            else "incorrect"
        )
    ```

=== "TypeScript"

    ```typescript
    const client = new Anthropic();

    function buildGraderPrompt(answer: string, rubric: string): string {
      return `Grade this answer based on the rubric:
    <rubric>${rubric}</rubric>
    <answer>${answer}</answer>
    Think through your reasoning in <thinking> tags, then output 'correct' or 'incorrect' in <result> tags.`;
    }

    async function gradeCompletion(output: string, goldenAnswer: string): Promise<string> {
      const graderResponse = await client.messages.create({
        model: "claude-opus-5-5",
        max_tokens: 2048,
        messages: [{ role: "user", content: buildGraderPrompt(output, goldenAnswer) }]
      });
      const textBlock = graderResponse.content.find((block) => block.type === "text");
      const graderText = textBlock ? textBlock.text : "";
      return graderText.toLowerCase().includes("<result>correct</result>")
        ? "correct"
        : "incorrect";
    }
    ```

Every LLM-graded example on the [same page](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests) uses `claude-opus-5-5` both to generate the answer and to grade it, yet the tone, privacy and context-utilization examples carry this comment right above the grading call: "Generally best practice to use a different model to evaluate than the model used to generate the evaluated output". Our reading: treat the shared model ID as sample convenience and follow the comment. Those three examples show the scales an LLM grader can apply: a 1 to 5 Likert scale for tone (100 customer inquiries with a target tone), a binary check for protected health information, which can catch subtle or implicit PHI that rule-based systems might miss (500 simulated patient queries), and a 1 to 5 ordinal scale for context utilization (100 multi-turn conversations with context-dependent questions).

Rules from [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) that make an LLM judge trustworthy:

- **Calibrate against people.** "LLM-as-judge graders should be closely calibrated with human experts to gain confidence that there is little divergence between the human grading and model grading." The post adds that model grading often takes careful iteration to validate.
- **Give the judge a way out,** such as an instruction to return "Unknown" when it does not have enough information, to avoid hallucinated verdicts.
- **Grade dimension by dimension.** The post says it "can also help" to write clear, structured rubrics for each dimension of a task and grade each dimension with an isolated LLM-as-judge rather than one judge for all dimensions.
- **Try it on your task.** The [cookbook](https://github.com/anthropics/claude-cookbooks/blob/main/misc/building_evals.ipynb): "The only way to know if a model-based grader can do a good job grading your task is to try. Try it out and read some samples to see if your task is a good candidate."

!!! note "Two Anthropic findings on judge design"

    The January 2026 [evals post](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) suggests an isolated judge per rubric dimension. The June 2025 [multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) post reported that its team experimented with multiple judges per component but found that, for that system, "a single LLM call with a single prompt outputting scores from 0.0-1.0 and a pass-fail grade was the most consistent and aligned with human judgements." That post adds that the method was especially effective when a test case had a clear answer. The first is general guidance; the second is what worked for one system with a rubric of factual accuracy, citation accuracy, completeness, source quality and tool efficiency. Both measure the judge the same way: against human judgment.

!!! note "How far to trust an LLM judge"

    The sources span a range. The [evals cookbook](https://github.com/anthropics/claude-cookbooks/blob/main/misc/building_evals.ipynb): "It turns out that Claude is highly capable of grading itself". The [develop-tests page](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests): "Test to ensure reliability first then scale." The [Agent SDK post](https://claude.com/blog/building-agents-with-the-claude-agent-sdk), on having another model judge an agent's output against fuzzy rules inside the agent's own feedback loop, calls this generally not a dependable method that can have heavy latency tradeoffs, and says that "for applications where any boost in performance is worth the cost, it can be helpful." The same post ranks rules first: "The best form of feedback is providing clearly defined rules for an output, then explaining which rules failed and why." Our reading of the three together is the order of preference above: rules and code where you can, a calibrated LLM judge where you must.

### Human graders

The [develop-tests page](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests) is blunt: human grading is "Most flexible and high quality, but slow and expensive. Avoid if possible." Anthropic's research-system team saw the other side: "Even in a world of automated evaluations, manual testing remains essential." Under the heading that human evaluation catches what automation misses, the [multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) post reports that human testers noticed early agents "consistently chose SEO-optimized content farms over authoritative but less highly-ranked sources like academic PDFs or personal blogs"; adding source quality heuristics to the prompts helped resolve it. The [January 2026 post](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) fits the two together: triage feedback constantly, sample transcripts to read weekly, and "Reserve systematic human studies for calibrating LLM graders or evaluating subjective outputs where human consensus serves as the reference standard." Once the grading system has proven dependable, the post says occasional human review is sufficient. Some domains still end with a person: the [legal summarization guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/legal-summarization) notes that expert evaluation "is often done on a few summaries as a validation check before deploying to production."

Human labels also calibrate the model's own confidence scores: CCAR-F 5.5-S3 has models output field-level confidence and then calibrates review thresholds on labeled validation sets. That workflow is in [Human review and confidence calibration](#human-review-and-confidence-calibration).

### Keep the grader independent of the generator

CCAR-F Task 4.6 states the self-review limitation: "a model retains reasoning context from generation, making it less likely to question its own decisions in the same session", and adds that independent review instances (without prior reasoning context) are "more effective at catching subtle issues than self-review instructions or extended thinking" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The matching skill, 4.6-S1, is "Using a second independent Claude instance to review generated code without the generator's reasoning context". Anthropic's own evaluator features follow the same design (as of September 2026; the security-guidance plugin is installed from the official Anthropic marketplace):

| Evaluator | How it stays independent |
|---|---|
| Claude Managed Agents outcomes, a beta feature ([docs](https://platform.claude.com/docs/en/managed-agents/define-outcomes)) | Defining an outcome makes the harness provision a grader that checks the artifact against a rubric; "The grader uses a separate context window to avoid being influenced by the main agent's implementation choices." |
| Claude Code `/goal` ([docs](https://code.claude.com/docs/en/goal)) | A separate evaluator checks your condition after every turn, so completion "is decided by a fresh model rather than the one doing the work" |
| Claude Code security-guidance plugin ([docs](https://code.claude.com/docs/en/security-guidance)) | "The plugin does not ask the same Claude instance that wrote the code to grade itself." Its end-of-turn and commit reviews run as a separate Claude call with fresh context and a security-focused prompt |

Review architectures built on this idea (writer and reviewer, per-file plus integration passes) are taught in [Multi-instance and multi-pass review](prompt-engineering.md#multi-instance-and-multi-pass-review); the evaluators above are covered in more detail in [Built-in evaluators (as of September 2026)](#built-in-evaluators-as-of-september-2026).

!!! warning "Exam guide vs current docs"

    **The guide (July 2026):** CCAR-F 4.6-K2 says independent review instances beat "self-review instructions or extended thinking" at catching subtle issues.

    **The docs today (as of September 2026):** the [API errors page](https://platform.claude.com/docs/en/api/errors) says "Claude 4.7 and later models have removed extended thinking". On Opus 5.5, Fable 5.1, Mythos 5.1, Fable 5 and Mythos 5, thinking is adaptive only and always on, and on Opus 5.5 `effort` is the only thinking control.

    **Self-check prompts, by model:** the [prompting best practices page](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) still recommends asking Claude to verify its answer before finishing ("This catches errors reliably, especially for coding and math"), with Claude Opus 5 as the exception. The [Opus 5 prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5) says Claude Opus 5 "verifies its own work without being told to", that explicit verification instructions such as "use a subagent to verify" cause over-verification on that model and should be removed, and that the same applies to legacy harness scaffolding that adds separate verification steps.

    **On the exam:** expect the guide's wording and answer in its terms: an independent reviewer without the generator's reasoning context is the stronger design. Read "extended thinking" as the same model reasoning longer in the same context, which is still not an independent review. The Opus 5 advice is model-specific and is about removing redundant verification the model already does; it does not change the objective's point that a fresh instance is more likely to question generated work than the session that produced it (our reconciliation, not a statement from either document).

### Combining graders into one score

- **Scoring rule per task.** Scoring can be "weighted (combined grader scores must hit a threshold), binary (all graders must pass), or a hybrid" ([Demystifying evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).
- **Partial credit.** For multi-component tasks, a support agent that identifies the problem and verifies the customer but fails to process the refund is meaningfully better than one that fails at once, and the score should show it.
- **Bypass resistance.** Make graders resistant to bypasses or hacks so the agent cannot easily cheat the eval; passing should require solving the problem, not exploiting a loophole.
- **Audit the grader.** When a task fails, the transcript shows whether the system erred or the grader rejected a valid answer; see [Running agent evals that you can trust](#running-agent-evals-that-you-can-trust).
- **Know what an agreement rule costs.** When several review runs feed one verdict, requiring them to agree filters out findings that only some runs catch. The rationale for CCAR-F sample question 12 rejects running three independent review passes and flagging only issues found in at least two, because that "would actually suppress detection of real bugs by requiring consensus on issues that may only be caught intermittently." The correct answer there was per-file passes plus an integration pass ([Multi-instance and multi-pass review](prompt-engineering.md#multi-instance-and-multi-pass-review)).

### Decide

- If the answer is a label, a number, a passing test or a database state, grade it with code; not an LLM judge, because code is faster, cheaper and reproducible.
- If the criterion is tone, coherence or synthesis quality, use an LLM judge with a specific rubric and a discrete output, calibrated against human grades; not free-text commentary, because purely qualitative evaluations are hard to assess quickly and at scale.
- If you need a reviewer for generated work, use an instance without the generator's reasoning context; not a self-review instruction in the same session.
- If a grader rejects answers that a person would accept (a different format, punctuation or valid phrasing), fix the grader; not the prompt, because the eval is then measuring formatting rather than correctness.
- If a task has several components, give partial credit or a weighted score; not all-or-nothing by default, because an agent that gets most of the way is meaningfully better than one that fails at once.

### Traps

- **Human grading as the default.** Human judgment is the gold standard used to calibrate model-based graders, yet the docs still say "Avoid if possible" because it is slow and expensive.
- **An uncalibrated judge.** Model-based graders are non-deterministic and require calibration with human graders for accuracy; until then their scores are unverified.
- **One judge scoring every dimension at once** without checking it against people; the January 2026 post suggests an isolated judge per dimension, and the research system in the June 2025 post used a single judge because, for that system, it proved the most consistent and the most aligned with human judgment.
- **A self-review instruction in the generating session** offered as the reviewer; the guide prefers an independent instance.
- **No way out for the judge.** The post's guard against grader hallucinations is an instruction to return "Unknown" when the judge does not have enough information; leave it out and a judge without the facts is pushed to guess (our reading).
- **Exact-match rigidity** on numbers and phrasing that have several valid forms.

## Evaluating agents

*Tested in: CCAR-P 4.2, 4.3, 4.6 · CCDV-F D4.1 Debugging and Error Handling (trace analysis to identify failure modes) and purpose area 5 · CCAR-F How to Prepare item 3, Exercise 1 steps 3 and 5, Exercise 4 steps 4 and 5 · CCAO-F: no agent-evaluation objective*

An agent eval scores a run, not a reply. Anthropic's [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) (published January 9, 2026) says that when you evaluate an agent you are "evaluating the harness and the model working together". It follows that a change to the system prompt, the tools, the loop code or the model can move the score. The post also says the agent in the eval should function roughly the same as the agent used in production, and the environment itself should not add noise.

For CCDV-F, whose Domain 4 skill (D4.1) names debugging rather than eval design, read this section as the background to D4.1's trace analysis (our reading of the blueprint; where eval design sits in that guide is in the [Exam map](#exam-map)). For CCAR-P it maps directly onto objectives 4.2, 4.3 and 4.6.

### The vocabulary

From the same [post](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents):

| Term | Meaning |
|---|---|
| Task | A single test with defined inputs and success criteria (also called a problem or test case) |
| Trial | One attempt at a task; you run several because model outputs vary between runs |
| Grader | Logic that scores one aspect of performance; a task can have several graders, each with several assertions (checks) |
| Transcript | The complete record of a trial (also called a trace or trajectory): outputs, tool calls, reasoning and intermediate results; for the Anthropic API, "the full messages array at the end of an eval run" |
| Outcome | The final state of the environment: a flight agent may report a booking, but the outcome is whether the reservation exists in the database |
| Evaluation harness | The infrastructure that runs evals end to end: supplies instructions and tools, runs tasks concurrently, records every step, grades and aggregates |
| Agent harness (scaffold) | The system that lets a model act as an agent: it processes inputs, orchestrates tool calls and returns results |
| Evaluation suite | A collection of tasks measuring related capabilities, such as refunds, cancellations and escalations for a support agent |

### Grade the outcome; check the path only where the path is a requirement

Agents reach the same result by different routes, so the default is to judge the result. The [evals post](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) advises that it is often better to "grade what the agent produced, not the path it took", because checking for a sequence of tool calls in the right order produces brittle tests: agents regularly find valid approaches the eval designers did not anticipate. For agents that modify persistent state across many turns, Anthropic's research-system team "found success focusing on end-state evaluation rather than turn-by-turn analysis", breaking complex workflows into discrete checkpoints where specific state changes should have occurred, and asks for methods that judge "whether agents achieved the right outcomes while also following a reasonable process" ([multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)).

Process graders still have a job. Claude Code's plugin-eval docs recommend giving each case "one grader on the result, such as the final message or a produced file, and one on how Claude got there, such as `tool_used` or `tool_order`" ([Test plugins with evals](https://code.claude.com/docs/en/plugin-evals)). Where an order is business logic, such as verifying identity before a refund, CCAR-F sample question 1's answer is "a programmatic prerequisite that blocks lookup_order and process_refund calls until get_customer has returned a verified customer ID" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Pairing that with an eval that asserts the required calls, as the post's illustrative support task below does, is our combination of the two sources: the code guarantees the order, and the eval catches a regression that drops a required call (to assert the order itself, use an order check such as the plugin-eval `tool_order` grader, which passes when both tools were called and the first matching `before` call precedes the first matching `after` call).

### Capability evals and regression evals

| | Capability (quality) evals | Regression evals |
|---|---|---|
| Question | What can this agent do well? | Does the agent still handle everything it used to? |
| Expected pass rate | Start low, targeting tasks the agent struggles with and giving the team a hill to climb | Nearly 100% |
| What the score tells you | Where the agent still falls short | A decline signals that something is broken and needs to be improved |
| Life cycle | After launch and optimization, capability evals with high pass rates can graduate into the regression suite | Run continuously to catch drift, alongside capability work so changes do not cause issues elsewhere |

Two further points from the [evals post](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents). First, "An eval at 100% tracks regressions but provides no signal for improvement": near saturation, large capability improvements show up as small increases in score, so make sure the problems stay hard enough for the model. Second, the post recommends eval-driven development: "build evals to define planned capabilities before agents can fulfill them, then iterate until the agent performs well." Anthropic often builds features that work "well enough" today but are bets on what models can do in a few months; capability evals that start at a low pass rate make those bets visible: when a new model ships, running the suite shows which bets paid off. The upgrade routine itself, and the time evals save there, is in [Upgrading models safely](#upgrading-models-safely).

### Non-determinism: `pass@k` and `pass^k`

Each task has its own success rate, and a task that passes on one run can fail on the next. Two metrics summarize multiple trials:

| Metric | Definition | As k grows | Use it when |
|---|---|---|---|
| `pass@k` | Probability of at least one correct solution in k attempts | Rises | One success is enough, as with a tool that proposes several solutions |
| `pass^k` | Probability that all k trials succeed | Falls | Users expect the agent to work every time, as with customer-facing agents |

At k = 1 the two are identical: both equal the per-trial success rate. The [post's](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) example: "If your agent has a 75% per-trial success rate and you run 3 trials, the probability of passing all three is (0.75)³ ≈ 42%." The post adds that by k = 10 the two "tell opposite stories: pass@k approaches 100% while pass^k falls to 0%." Consistency gets expensive fast: a 90% per-trial rate gives a `pass^10` of about 35% (`0.9^10`, our arithmetic, assuming independent trials). For coding, the post notes that `pass@1`, success on the first try, is often what matters most.

### What to grade, by type of agent

From [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents):

| Agent type | Typical graders | What to remember |
|---|---|---|
| Coding | Deterministic tests (does the code run, do the tests pass), often plus an LLM rubric for code quality; static analysis and transcript checks as needed | "In practice, coding evaluations typically rely on unit tests for correctness verification and an LLM rubric for assessing overall code quality" |
| Conversational (support, sales, coaching) | A state check (was the ticket resolved), a transcript constraint (finished in under 10 turns) and an LLM rubric (was the tone appropriate) | The quality of the interaction is part of what you evaluate, and these evals "often require a second LLM to simulate the user" |
| Research | Groundedness checks (claims supported by retrieved sources), coverage checks (key facts a good answer must include), source quality checks; exact match where one answer is correct | LLM rubrics "should be frequently calibrated against expert human judgment" |
| Computer use | Run in a real or sandboxed environment and verify the outcome, including backend state (the order was placed, not just the confirmation page shown) | Computer use is out of scope for CCAR-F |

The [evals post](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) illustrates a support task (a refund for a frustrated customer) with four graders of different kinds. It titles the example "Theoretical evaluation for a conversational agent", says it "showcases multiple grader types for illustration", and notes that in practice conversational evals typically use model-based graders because many tasks may have more than one correct solution. The grader type names (`llm_rubric`, `state_check`, `tool_calls`, `transcript`) belong to this illustration; they are not the grader types of Claude Code's `claude plugin eval`, which are listed further down. The block below is the example's `graders` section as published.

```yaml
graders:
  - type: llm_rubric
    rubric: prompts/support_quality.md
    assertions:
      - "Agent showed empathy for customer's frustration"
      - "Resolution was clearly explained"
      - "Agent's response grounded in fetch_policy tool results"
  - type: state_check
    expect:
      tickets: {status: resolved}
      refunds: {status: processed}
  - type: tool_calls
    required:
      - {tool: verify_identity}
      - {tool: process_refund, params: {amount: "<=100"}}
      - {tool: send_confirmation}
  - type: transcript
    max_turns: 10
```

The same example also tracks metrics that are not pass/fail: turns, tool calls and total tokens from the transcript, and time to first token, output tokens per second and time to last token for latency. Once a suite exists, the [evals post](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) notes that "latency, token usage, cost per task, and error rates can be tracked on a static bank of tasks" as baselines and regression tests.

### Running agent evals that you can trust

1. **Isolate every trial.** Start each trial from a clean, isolated environment. Unnecessary shared state between runs (leftover files, cached data, resource exhaustion) can cause correlated failures that reflect infrastructure flakiness rather than agent performance; shared state can also inflate scores, as when Claude gained an unfair advantage on some internal tasks by examining the git history from previous trials.
2. **Choose graders deliberately** (see [Grading methods](#grading-methods)), and build in partial credit for multi-step tasks.
3. **Read the transcripts.** "When a task fails, the transcript tells you whether the agent made a genuine mistake or whether your graders rejected a valid solution." Anthropic's rule: "we do not take eval scores at face value until someone digs into the details of the eval and reads some transcripts." ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents))
4. **Watch for saturation** and add harder tasks when scores approach 100%.
5. **Evaluate the tools as well as the agent.** Anthropic's tool guidance: ground tasks in realistic uses (strong tasks may need dozens of tool calls), and beyond accuracy collect runtime per tool call and per task, the number of tool calls, total tokens and tool errors. The same guidance notes that "lots of tool errors for invalid parameters might suggest tools could use clearer descriptions or better examples" ([Writing effective tools for AI agents](https://www.anthropic.com/engineering/writing-tools-for-agents)); the fixes are in [Writing tool descriptions that steer selection](tool-use-and-mcp.md#writing-tool-descriptions-that-steer-selection).
6. **Cover the hard inputs the exam guide names.** The [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)'s preparation advice asks you to "Test tool selection reliability with ambiguous requests". Its exercises turn the rest into test cases:
    - Exercise 1: test that the agent handles each structured error type appropriately ("retrying transient errors, explaining business errors to the user"), and test multi-concern messages to verify that the agent decomposes the request, handles each concern and synthesizes a unified response.
    - Exercise 4: simulate a subagent timeout and verify the coordinator receives structured error context (failure type, attempted query, partial results) and can proceed with partial results while annotating coverage gaps; test with conflicting source data and verify the synthesis preserves both values with source attribution.

### Built-in evaluators (as of September 2026)

| Evaluator | What it scores | Facts to know |
|---|---|---|
| `claude plugin eval` (Claude Code v2.1.269 or later) | A plugin or skill, against cases that each pair a realistic prompt with graders | Each case runs three times by default and passes when its score meets `--threshold` (default `1.0`); by default each case also runs without the plugin, giving `WITH`, `W/OUT` and their difference `Δ`; six grader types, `regex`, `tool_used`, `tool_order` and `file_exists` (computed, no cost) plus `llm` and `baseline` (call a judge model, a small fast model unless you pass `--judge-model`); no custom-code graders; an `llm` grader passes when the judge votes PASS in at least two of three votes; every run and judge call counts against your plan's usage or your API bill |
| Managed Agents outcomes (beta header `managed-agents-2026-04-01`) | A deliverable, against a rubric | Sent as a `user.define_outcome` event with a required markdown rubric; the grader works in a separate context window and scores each criterion independently; `max_iterations` is optional, default 3, maximum 20; results are `satisfied`, `needs_revision`, `max_iterations_reached`, `failed` and `interrupted`; one outcome at a time |
| Claude Code `/goal` | Whether a completion condition holds | After each turn a small fast model checks the condition; it does not run commands or read files, so the condition must be something Claude's output can demonstrate; up to 4,000 characters; a built-in shortcut for a session-scoped prompt-based Stop hook |

A plugin eval case is a directory under the plugin's eval directory (`evals/` unless configured otherwise) containing a `prompt.md`, a `case.yaml`, or both; graders go in a `graders/` folder with one file per grader, in a `graders` list in `case.yaml`, or in both ([Test plugins with evals](https://code.claude.com/docs/en/plugin-evals)). The `prompt.md` body is the prompt; its frontmatter sets the run limits and tools. As of September 2026 the documented limits are `max_turns` (default 10, up to 200), `timeout_seconds` (default 300, up to 3600) and `runs` (default 3, from 1 to 50 per arm), and `allowed_tools` defaults to none. Each run starts in an empty working directory with only the plugin loaded, and the run cannot read the eval directory, so Claude never sees the graders. The first three files below are the docs' manual-case example, written for a skill that drafts commit messages; replace `your-skill-name` with the `name` from your skill's `SKILL.md`.

=== "evals/first-case/prompt.md"

    ```markdown
    ---
    max_turns: 10
    allowed_tools: [Read, Glob, Grep, Skill]
    ---

    Write me a commit message for this change: I renamed getUser to fetchUser and updated the three call sites.
    ```

=== "evals/first-case/graders/criteria.md"

    ```markdown
    ---
    type: llm
    ---

    PASS if <what a correct response contains>.
    FAIL if <what a wrong or missing response looks like>.
    ```

=== "evals/first-case/graders/skill-fired.md"

    ```markdown
    ---
    type: tool_used
    tool: Skill
    input_match: '"skill"\s*:\s*"(?:[\w-]+:)?your-skill-name"'
    ---
    ```

=== "CI invocation"

    ```bash
    claude plugin eval . \
      --trust-plugin \
      --json results.json \
      --threshold 0.8 \
      --model claude-sonnet-5 \
      --judge-model claude-haiku-4-5 \
      --no-publish \
      --max-cost-usd 20
    ```

The CI invocation above is the docs' own example. In CI, the docs say to pass `--trust-plugin` so the job never waits at the trust prompt, pin both models so scores stay comparable over time, keep the report local, and set a cost ceiling; `--max-cost-usd` caps the list-price cost estimate, not plan usage, and runs already in flight when it is reached still finish. Fail the build on the exit code: 0 means every case scored at or above `--threshold` and every case file loaded; 1 means a case scored below the threshold, a case file failed to load, no cases were found, a run could not start, the directory was untrusted without `--trust-plugin`, or an option was invalid; 2 means a partial run (the `--max-cost-usd` ceiling was hit, or the credential was rejected before or at the first run), with `results.json` still written and marked `partial: true`; 130 means interrupted, with partial results written, and 143 means terminated, such as by a CI timeout. Because the default threshold is `1.0`, the command exits 1 whenever any case scores below perfect. If a case's `tool_used: Skill` grader passes but `Δ` is negative, the [plugin-eval docs](https://code.claude.com/docs/en/plugin-evals) say to "suspect the judge before the plugin": a small judge model can mark a correct answer wrong because it is formatted differently from what the rubric describes, so re-run with `--judge-model sonnet` and tighten the rubric so formatting does not decide the verdict. Packaging plugins is covered in [Plugins and marketplaces](claude-code-configuration.md#plugins-and-marketplaces).

A Managed Agents outcome starts with one `user.define_outcome` event sent to the session; the agent begins work on receipt, with no separate user message needed, and the grader's explanation of which criteria passed or failed goes back to the agent for its next iteration ([Define outcomes](https://platform.claude.com/docs/en/managed-agents/define-outcomes)). The body below is the docs' example; the rubric can instead reference an uploaded file with `{"type": "file", "file_id": ...}`:

```json
{
  "events": [
    {
      "type": "user.define_outcome",
      "description": "Build a DCF model for Costco in .xlsx",
      "rubric": {"type": "text", "content": "# DCF Model Rubric\n..."},
      "max_iterations": 5
    }
  ]
}
```

The [Define outcomes](https://platform.claude.com/docs/en/managed-agents/define-outcomes) rubric advice: write explicit, gradeable criteria ("The CSV contains a price column with numeric values" rather than "The data looks good"), because "vague criteria produce noisy evaluations"; if you have no rubric, give Claude a known-good artifact, ask it to analyze what makes it good, and turn the analysis into a rubric. The [evals post](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) makes the same point for any model-based grader: "vague rubrics produce inconsistent judgments." For `/goal` and Stop hooks as completion gates, see [Hooks](claude-code-workflows.md#hooks).

### Evals across the product's life

Automated evals are one layer. [The post](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) compares six ways to understand an agent's performance:

| Method | Best at | Limits |
|---|---|---|
| Automated evals | Fast, reproducible iteration with no user impact; can run on every commit | Up-front investment and upkeep; false confidence if tasks do not match real usage |
| Production monitoring | Real behavior at scale; catches what synthetic evals miss; ground truth on how agents actually perform | Reactive: problems reach users first; noisy signals; needs instrumentation; lacks ground truth for grading |
| A/B testing | Actual user outcomes such as retention and task completion; controls for confounds | Slow ("days or weeks to reach significance") and needs traffic; tests only what you deploy; says little about why metrics moved |
| User feedback | Problems you did not anticipate, with real examples | Sparse and self-selected; skews toward severe issues; users rarely explain why |
| Manual transcript review | Intuition for failure modes; subtle quality issues | Time-intensive; does not scale; inconsistent coverage |
| Systematic human studies | Gold-standard judgments; subjective or ambiguous tasks; signal to improve model graders | Expensive and slow; hard to run often; raters disagree |

In the [same post](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents), they map to stages: automated evals pre-launch and in CI/CD "running on each agent change and model upgrade as the first line of defense against quality problems"; production monitoring after launch for distribution drift and unanticipated real-world failures; A/B tests for significant changes once traffic allows; feedback triage and weekly transcript sampling as ongoing practices; systematic human studies reserved for calibrating LLM graders or evaluating subjective outputs. "Like the Swiss Cheese Model from safety engineering, no single evaluation layer catches every issue." The post's summary: "automated evals for fast iteration, production monitoring for ground truth, and periodic human review for calibration."

Anthropic has published a case where its own evals missed a production problem. Its [postmortem](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues), published September 17, 2025, describes three infrastructure bugs that intermittently degraded Claude's response quality between August and early September. Validation ordinarily relied on benchmarks alongside safety evaluations and performance metrics, with spot checks and small canary deployments, yet "The evaluations we ran simply didn't capture the degradation users were reporting, in part because Claude often recovers well from isolated mistakes." Anthropic added: "More fundamentally, we relied too heavily on noisy evaluations." The remediations were more sensitive evaluations, quality evaluations run continuously on true production systems, and faster debugging tooling. Our takeaway for an architect: a system that recovers from isolated mistakes can pass end-to-end checks while its quality degrades, so production quality needs its own continuous monitoring.

### Decide

- If the requirement is an end state (a record exists, a ticket is resolved), grade the state; not the exact tool sequence, which breaks when the agent finds another valid route.
- If a step order is mandatory business logic, enforce it in code and assert the required calls in the eval; not a prompt instruction checked only by a rubric.
- If users need the agent to succeed every time, report `pass^k`; not `pass@k`, which rises with more attempts.
- If a score drops, read failing transcripts before changing the agent; not a prompt rewrite, because the grader may be rejecting valid solutions.
- If a change is significant and traffic allows, confirm it with an A/B test after offline evals pass; not an A/B test as the first check, because it is slow and reaches users.

### Traps

- **Trusting the agent's report.** A transcript that says "Your flight has been booked" is not a booking ([evals post](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)); check the environment.
- **One trial per task.** A single run of a non-deterministic agent says little; run several.
- **Shared state between trials,** which can cause correlated failures or inflate scores.
- **Reading a saturated suite as proof that nothing can improve.** At 100% it only tracks regressions; add harder tasks.
- **Counting only HTTP errors as failures.** An agent can fail with every request succeeding; see [Debugging: model or integration](#debugging-model-or-integration).

## Debugging: model or integration

*Tested in: CCDV-F D4.1 Debugging and Error Handling, D6.3 Output Handling · CCAR-P 4.4, 7.3, sample question 3 · CCAO-F D7.1, D7.2 · CCAR-F 1.2-K4, Exercise 4, sample question 7*

CCDV-F's only Domain 4 skill names the whole method: "Debugging and error handling techniques for Claude applications, including error type identification, recovery strategy selection, trace analysis to identify failure modes, and problem origin isolation between the integration layer and model output" ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). CCAR-P adds "Diagnose system issues (prompt failure, hallucinations, model mismatch)" and "Support debugging and operational issue resolution" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). Every one of these starts with the same question: is the fault in your integration, or in what the model produced?

### Three kinds of signal

| What you observe | Where the fault usually lives | What it means | First move |
|---|---|---|---|
| An exception, an HTTP 4xx or 5xx, or an error event inside a stream | Your integration, or the platform | The request failed; there is no complete model output to judge | Identify the error type and pick the recovery for that type |
| HTTP 200 with `stop_reason` `max_tokens`, `model_context_window_exceeded`, `pause_turn` or `refusal` | The contract between your code and the model | A valid response that was truncated, paused or declined | Handle the stop reason; see [Stop reasons and the agent loop](claude-api.md#stop-reasons-and-the-agent-loop) |
| HTTP 200, `end_turn`, wrong content | The model's output, or the context your integration gave it | The model did what it could with what it saw | Inspect exactly what the model saw before blaming the model |

The second column (where the fault usually lives) is our triage heuristic, built on the documented split between errors and stop reasons. The [stop reasons page](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons) draws the line: "Unlike errors, which indicate failures in processing your request, `stop_reason` tells you why Claude completed its response generation." Two cases slip past error-rate monitoring. A refusal "is an HTTP 200, so monitoring built on error rates or 5xx responses never sees it" ([Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback)). And with streaming, "an error can occur after the API returns a 200 response" ([API errors](https://platform.claude.com/docs/en/api/errors)), so a 200 status alone does not prove the stream completed.

The integration layer also includes the code that reads responses. Three documented bugs look like model failures and are not:

- **Reading the reply by position.** On Claude Sonnet 5, requests without a `thinking` field run with adaptive thinking, so a response can begin with `thinking` blocks before the first `text` block. The [Sonnet 5 migration guide](https://platform.claude.com/docs/en/models/sonnet-5/migration-guide) says code that reads `content[0].text` "must select content blocks by their `type` field instead".
- **String-matching tool inputs.** Unicode and forward-slash escaping of tool inputs differs between model versions, so parse with `json.loads()` or `JSON.parse()`: "Never do raw string matching on serialized input." ([Troubleshooting tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/troubleshooting-tool-use))
- **Trusting a success subtype.** In the Agent SDK a result can have subtype `success` and still lack `structured_output` (one way to hit this is a schema no output can satisfy); the docs say to treat it as a failure: "Check both that `subtype` is `success` and that `structured_output` is present before using it." ([Agent SDK troubleshooting](https://code.claude.com/docs/en/agent-sdk/troubleshooting))

These three are concrete cases of what CCDV-F D6.3 calls "defensive parsing" and "response validation" (our mapping; [CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)).

### Identify the error type in code

Catch the SDK's typed exceptions, most specific first, instead of string-matching messages ([API errors](https://platform.claude.com/docs/en/api/errors)), and log request IDs: the SDK pages say the `_request_id` property exists "so that you can quickly log failing requests and report them back to Anthropic". The Python tab below joins two examples from the [Python SDK page](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python) (error handling and request IDs), with a client-construction line in place of the page's `# ...`; the TypeScript tab is the [TypeScript SDK page's](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/typescript) error-handling example with its import and client-construction lines added.

=== "Python"

    ```python
    import anthropic

    client = anthropic.Anthropic()

    try:
        message = client.messages.create(
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello, Claude"}],
            model="claude-opus-5-5",
        )
        print(message._request_id)  # e.g., req_018EeWyXxfu5pfWkrYcMdjWG
    except anthropic.APIConnectionError as e:
        print("The server could not be reached")
        print(e.__cause__)  # an underlying Exception, likely raised within httpx2
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

    const client = new Anthropic();

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

Both SDKs map statuses to the same class names: 400 `BadRequestError`, 401 `AuthenticationError`, 403 `PermissionDeniedError`, 404 `NotFoundError`, 409 `ConflictError`, 422 `UnprocessableEntityError`, 429 `RateLimitError`, 500 and above `InternalServerError`, and `APIConnectionError` when the library cannot connect (no status code). Every API response carries a `request-id` header, exposed as `_request_id` in Python and TypeScript, and error bodies carry the same value as a `request_id` field; include it when you contact support about a specific request. Both SDKs read the `ANTHROPIC_LOG` environment variable for debug logging: the Python page documents `debug` or `info`, and TypeScript also accepts a `logLevel` client option (default `warn`). The full list of HTTP errors is in [Errors, retries and rate limits](claude-api.md#errors-retries-and-rate-limits).

### Choose the recovery by error class

| Class | Typical signals | Recovery | Retry the same request? |
|---|---|---|---|
| Transient | Connection errors; 429 with `retry-after`; 500; 529; an `overloaded_error` event mid-stream | Back off and retry; the SDKs already retry twice by default with exponential backoff, honoring `retry-after` | Yes, after waiting |
| Quota | 429 `rate_limit_error` with no `retry-after` header whose `error.details.error_code` is `enforced_spend_limit_reached` on the Messages API (the usage tier's monthly spend cap); 400 `invalid_request_error` when an organization or workspace spend limit you set is reached, except on the Claude Code workspace, whose limit can instead return a 429 that carries a `retry-after` header | Alert a person: request a higher cap, or raise or remove your own limit; otherwise the tier cap pauses usage until 00:00 UTC on the first day of the next month | No: retries, including the SDKs' automatic retries, fail until access resumes |
| Request defect | 400 such as "prompt is too long", a missing or misplaced `tool_result`, edited thinking blocks, or prefill or forced `tool_choice` on models that reject them; 401; 403; 404; 413 | Fix the request, key, permission or payload | No: the same request fails the same way |
| Truncation | `stop_reason` `max_tokens` or `model_context_window_exceeded` | For `max_tokens`, raise `max_tokens` or continue the response, and if a `tool_use` block was cut off, retry with a higher `max_tokens`; for `model_context_window_exceeded`, treat the response as truncated | Only with a changed request |
| Refusal | `stop_reason` `refusal` on an HTTP 200 | Retry on a different Claude model: server-side fallback (beta on the Claude API), the SDK middleware, or a manual retry | No: "Re-sending a refused request to the same model usually earns another refusal." ([Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback)) |
| Tool failure | Your tool threw, timed out or got bad input | Return a `tool_result` with `"is_error": true` and a message that says what went wrong and what to try next | Claude decides; for invalid or missing parameters it retries 2 to 3 times with corrections before apologizing. See [Returning tool results and errors](tool-use-and-mcp.md#returning-tool-results-and-errors) |
| Wrong content | `end_turn` with a wrong answer | Diagnose inputs, prompt and model (below) | Not as a fix: CCAR-F Task 4.4 notes that retries are ineffective when the required information is absent from the source document |

Retry budgets, backoff, fallback chains and idempotency are designed in [Reliability engineering for Claude applications](#reliability-engineering-for-claude-applications); how a failure should travel from a subagent to its coordinator is in [Error propagation in multi-agent systems](#error-propagation-in-multi-agent-systems). Agent SDK runs end with a `ResultMessage` whose `subtype` names the stop (`success`, `error_max_turns`, `error_max_budget_usd`, `error_during_execution`, `error_max_structured_output_retries`); every subtype carries `total_cost_usd`, `usage`, `num_turns` and `session_id`, so you can track cost and resume even after an error, though after a session crash the final `error_during_execution` result's cost fields may be zeroed ([The agentic loop](agents-and-agent-sdk.md#the-agentic-loop)).

### Read the trace

In practice (our working definition), trace analysis means reading the step-by-step record of a run and finding the first step whose output was wrong given correct inputs. Where the records come from (as of September 2026):

| Record | What it contains | Debugging use |
|---|---|---|
| Messages API transcript | The full messages array for the run; log each response's `stop_reason`, `usage` and `request-id` alongside it | Replay exactly what the model saw at each step |
| Agent SDK and Claude Code OpenTelemetry traces (beta) | Spans: `claude_code.interaction` (one agent-loop turn), `claude_code.llm_request` (each API call, with model name, latency and token counts), `claude_code.tool` (with `claude_code.tool.blocked_on_user` and `claude_code.tool.execution` children), and `claude_code.hook` (only with detailed beta tracing, `ENABLE_BETA_TRACING_DETAILED=1` plus `BETA_TRACING_ENDPOINT`); a subagent's spans nest under the parent's `claude_code.tool` span, so the delegation chain is one trace | Find which model call, tool, permission wait, hook or subagent failed or stalled |
| Claude Code monitoring events | `claude_code.api_error` is emitted once per failed request, only when Claude Code gives up; its `attempt` is 11 by default when a transient error exhausted every retry (`CLAUDE_CODE_MAX_RETRIES` defaults to 10 and is capped at 15; on v2.1.199 or later, setting `CLAUDE_CODE_RETRY_WATCHDOG` raises the default and removes the cap), and lower for a non-retryable error such as a 400; `claude_code.api_refusal` covers refusals, which arrive on a successful stream and never fire `api_error` | Tell a recovered session from a stalled one: group by `session.id` and check whether a later `api_request` event exists after the error |
| Headless `claude -p --output-format stream-json` output | `system/api_retry` events with `attempt`, `max_retries`, `retry_delay_ms`, `error_status` and an `error` category; `plugin_errors` and `mcp_server_errors` (`--mcp-config` entries skipped by validation) in the `system/init` event; exit code 0 on success, non-zero on failure | Fail a CI job on a non-empty `plugin_errors` or `mcp_server_errors` array, because a skipped MCP server still lets the run continue and exit cleanly |

The OpenTelemetry traces and monitoring events above exist only once telemetry is turned on; the switches, the content opt-ins and the export pitfalls are in [See failures: tracing and telemetry](#see-failures-tracing-and-telemetry).

**Reading a multi-agent trace.** CCAR-F sample question 7 is a trace-reading item: every subagent completes successfully, yet reports on "impact of AI on creative industries" cover only visual arts, and the coordinator's logs show it decomposed the topic into three visual-arts subtasks. The answer (B) is that the coordinator's task decomposition is too narrow, the risk CCAR-F knowledge bullet 1.2-K4 names; the rationale adds that "Options A, C, and D incorrectly blame downstream agents that are working correctly within their assigned scope" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Our rule of thumb: work back from the symptom until you reach a component whose inputs were right and whose output was wrong; that component is the root cause. Routing all subagent communication through the coordinator, which skill 1.2-S4 names "for observability", is what puts those decomposition decisions in one log. Anthropic's research team made the same point about production: "Adding full production tracing let us diagnose why agents failed and fix issues systematically" ([multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)). Decomposition itself is taught in [Multi-agent orchestration](agents-and-agent-sdk.md#multi-agent-orchestration).

!!! warning "Exam guide vs current docs"

    **The guide (July 2026):** the [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) names "The Task tool as the mechanism for spawning subagents" and says `allowedTools` must include "Task" for a coordinator to invoke subagents (1.3-K1); its appendix lists "subagent spawning via Task tool", and Exercise 4 asks you to ensure the coordinator's `allowedTools` includes "Task".

    **The docs today:** Claude Code v2.1.63 renamed the Task tool to Agent, and `Task(...)` references in settings and agent definitions still work as aliases. The [Agent SDK observability page](https://code.claude.com/docs/en/agent-sdk/observability) describes the case "When the agent spawns a subagent through the Agent tool". The tool appears as `"Agent"` in `tool_use` blocks but as `"Task"` in the `system:init` tools list, and before v2.1.63 `tool_use` blocks also said `"Task"`.

    **On the exam:** expect "Task tool". When you read a current trace or transcript, match both names.

### Isolate the origin: a procedure

1. **Did the request fail?** Check the exception type, the HTTP status or the stream's error event. If it failed, the fault is in your integration or the platform: classify it with the table above. Resending an unchanged request defect only fails again.
2. **It succeeded: what is the `stop_reason`?** Anything other than `end_turn` or `tool_use` is a signal your code must handle, not a quality problem.
3. **`end_turn` but wrong: what did the model see?** Pull the transcript and check the system prompt, the retrieved chunks and the tool results. If the inputs were wrong or stale, the model is not the cause. CCAR-P sample question 3 is this case: after a document refresh a RAG system returns confident but incorrect answers while latency and model version are unchanged. Anthropic's rationale in the [CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf): "Confident-but-wrong answers following a document refresh, with model and latency unchanged, point to retrieval feeding the model poor context, for example a broken re-index or mismatched embeddings." The [RAG cookbook](https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/retrieval_augmented_generation/guide.ipynb) makes this visible in advance: "it's critical to evaluate the performance of the retrieval system and end to end system separately."
4. **Inputs right, output wrong: reproduce it.** Freeze the inputs and run the case several times, since outputs vary and "Inconsistencies across outputs could indicate hallucinations" ([Reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations)). Change one thing at a time so you know what moved the result.
5. **Classify the model-side failure** as a prompt failure, a hallucination or a model mismatch (next table).
6. **Suspect the platform last, and precisely.** "When you use a model ID in an API request, the underlying model remains constant for the lifetime of that ID" ([Model IDs and versioning](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions)); the guarantee covers model IDs, not the convenience aliases the API accepts for some earlier models. So option A of CCAR-P sample question 3 in the [CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf), "The model weights have silently changed.", contradicts the pinning guarantee. The serving infrastructure around the model (request router, safety classifiers, sampling logic) can change, though: "If you notice unexpected behavioral differences on a previously stable model ID, an infrastructure update is the most likely cause." Anthropic's [postmortem](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues) also states: "We never reduce model quality due to demand, time of day, or server load." If you contact support about a specific request, include its request ID.

The other options in CCAR-P sample question 3 (silently changed weights, a too-low temperature, a shrunken context window) fail for the reason the [CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)'s rationale gives: they "would not be triggered specifically by a document refresh". The general rule we draw from it: look first at whatever changed when the symptom began.

### Prompt failure, hallucination or model mismatch

CCAR-P 4.4 names three model-side diagnoses. They have different signs and different fixes:

| Diagnosis | Signs | First fix |
|---|---|---|
| Prompt failure | Output misses behavior the prompt left for Claude to infer, or honors the letter of an instruction but not its intent; a colleague with minimal context would be confused by the prompt | State the instruction explicitly with the reason behind it; add relevant, diverse examples; in Claude Code, if `/context` shows CLAUDE.md loaded but an instruction is ignored, rewrite the instruction |
| Hallucination | Confident, plausible and wrong; authoritative-looking quotes with no grounding; claims of actions Claude has no integrated tool for, such as saying it sent an email | Explicitly allow Claude to admit uncertainty; for long documents (over 20k tokens), have it extract word-for-word quotes first; require a supporting quote or citation for each claim and retract any claim without one; validate critical information, since these techniques reduce hallucinations but do not eliminate them |
| Model mismatch | A well-specified prompt still misses the capability, latency or cost target, or behavior changed after a model upgrade | Try tuning `effort`, which the [model-selection docs](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model) call "often a better lever than switching models"; move up a tier for problems where a smaller model struggled; after an upgrade, re-check prompts and parameters the new model handles differently (for example a different default `effort`, or prefill and forced `tool_choice` now rejected) |

The prompting overview warns against treating every miss as a prompt problem: "Not every success criteria or failing eval is best solved by prompt engineering. For example, you can sometimes improve latency and cost more easily by selecting a different model." ([Prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview)). The test for a prompt failure is the docs' golden rule: "Show your prompt to a colleague with minimal context on the task and ask them to follow it. If they'd be confused, Claude will be too." ([Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)). Go deeper in [Principles that decide most prompt questions](prompt-engineering.md#principles-that-decide-most-prompt-questions) (a symptom-by-symptom table), [Reducing hallucinations](prompt-engineering.md#reducing-hallucinations), [Models and how to choose one](claude-api.md#models-and-how-to-choose-one) and [Upgrading models safely](#upgrading-models-safely).

### Debugging inside Claude Code

- **Loaded but ignored.** "If `/context` confirms the file loaded but Claude still isn't following a particular instruction, the issue is likely how the instruction is written rather than whether it loaded" ([Debug your configuration](https://code.claude.com/docs/en/debug-your-config)).
- **Bisect the customizations.** `claude --safe-mode` launches a session with all customizations disabled, including CLAUDE.md, skills, plugins, hooks, MCP servers, and custom commands and agents; managed hooks and settings policy from your organization still apply. "If the problem disappears in safe mode, one of those surfaces is the cause" ([Debug your configuration](https://code.claude.com/docs/en/debug-your-config)).
- **Guidance or enforcement.** The [same page](https://code.claude.com/docs/en/debug-your-config): "Use permissions or hooks for security boundaries and anything that must never happen, where you need a guarantee instead of guidance."
- **Give Claude the evidence.** For a bug, tell Claude the command that reproduces the issue and gets a stack trace, mention the steps to reproduce it, and "Let Claude know if the error is intermittent or consistent" ([Common workflows](https://code.claude.com/docs/en/common-workflows)).
- **Reset a polluted session.** "After two failed corrections, `/clear` and write a better initial prompt incorporating what you learned." ([Best practices](https://code.claude.com/docs/en/best-practices))

### Troubleshooting in the Claude apps

CCAO-F D7.1 and D7.2 ask business users to "Identify, diagnose, and resolve issues with underperforming prompts or poor outputs" and to "Adjust approach based on feedback and results" ([CCAO-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)). The same isolation logic applies without code. Claude Academy's [capabilities course](https://academy.claude.com/courses/ai-capabilities-and-limitations/when-properties-collide) notes that "Real-world failures are usually two properties interacting, not one", so name both before choosing a fix. Change one thing at a time between attempts. When a conversation has gone wrong, [Claude 101's](https://academy.claude.com/courses/claude-101/your-first-conversation-with-claude) advice is that in the worst case you "restart your conversation in a new chat to fully refresh the context". Symptom-by-symptom fixes are in the prompting table linked above, and checking the output itself is in [Verifying Claude's output](claude-for-work.md#verifying-claudes-output).

### Decide

- If the call raised an exception or returned 4xx or 5xx, fix or retry by error class; not prompt tuning for output quality, because there is no complete model output to judge (a request defect, such as input longer than the model's context window, is fixed by changing the request).
- If the response is HTTP 200 with `stop_reason` `refusal`, treat it as its own signal and retry on a different model; not an error-rate alert, which never fires for it, and not a resend to the same model.
- If answers turned confidently wrong right after a data or index change, with model and latency unchanged, investigate retrieval first; not the model weights, the temperature or the context window.
- If `/context` shows the instruction loaded and Claude still ignores it, rewrite the instruction; not a reload of the file.
- If a well-written prompt still misses a latency or cost target, change the model or `effort`; not more prompt text.

### Traps

- **Monitoring only HTTP errors.** Refusals and mid-stream errors arrive after a 200.
- **Blaming the model first.** Stale retrieval, a position-based parser or a missing `structured_output` check all look like bad answers.
- **Retrying a 400 or a spend-cap 429.** The request fails the same way until you change it, the limit is raised, or access resumes.
- **Assuming a stable model ID can drift in weights.** The ID pins the model; infrastructure is what can change.
- **Blaming a downstream agent.** In sample question 7 the subagents worked correctly within their assigned scope and the coordinator's decomposition was the root cause.

## Error propagation in multi-agent systems

*Tested in: CCAR-F 5.3 (5.3-K1 to 5.3-S4), 2.2-S3, 2.2-S4, 1.2-S4, appendix in-scope topic 7, Exercise 4 step 4, sample question 8 · CCDV-F D4.1 Debugging and Error Handling (recovery strategy selection), D8.1 Tool Implementation (error handling) · CCAR-P 1.4, 5.2 · CCAO-F: not listed*

In a coordinator-subagent system, a subagent's failure becomes information the coordinator has to act on. The coordinator can make a good recovery decision only if the failure reaches it intact: what failed, what was tried, what was found anyway, and what else could work. Anthropic notes that the autonomous nature of agents means higher costs and "the potential for compounding errors" ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)), and its post on the Research system names as one of its production challenges that "Agents are stateful and errors compound" ([multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)). This section follows a failure from a subagent to the final report. How a single tool reports an error (`is_error`, MCP's `isError`, the guide's `errorCategory` and `isRetryable` fields) is in [Returning tool results and errors](tool-use-and-mcp.md#returning-tool-results-and-errors) and [MCP errors and structured results](tool-use-and-mcp.md#mcp-errors-and-structured-results); the coordinator pattern itself is in [Multi-agent orchestration](agents-and-agent-sdk.md#multi-agent-orchestration).

### The rule in one line

Recover locally from what is transient; report what you cannot fix with enough structure for the coordinator to choose a next step; never disguise a failure as success; never let one failure end the whole run. The [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) names the last two together as anti-patterns: "silently suppressing errors (returning empty results as success) or terminating entire workflows on single failures" (5.3-K4).

### The channel is the subagent's final message

In the Agent SDK, a subagent that is not a fork of the parent conversation starts with a fresh context window, and two crossings between parent and subagent decide what anyone learns about an error (as of September 2026):

- **Down:** "The only content you pass from parent to subagent is the Agent tool's prompt string" ([Agent SDK subagents](https://code.claude.com/docs/en/agent-sdk/subagents)). When the coordinator re-delegates after a failure, the failed query, the error and the partial results must be written into the new prompt; a fresh subagent knows none of it. The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) states the same rule: subagent context "must be explicitly provided in the prompt" (1.3-K2).
- **Up:** intermediate tool calls and results stay inside the subagent, and only its final message returns to the parent, which may summarize it. A tool timeout the subagent hit is invisible to the coordinator unless that final message reports it. (An API error that ends the subagent's own run is different: the platform reports it, as the table below shows.)

So the error report format belongs in the subagent's instructions, and the coordinator's instructions must say what to do with it. An illustrative definition (the `AgentDefinition` shape follows the Agent SDK subagents docs; the report format is ours, built from the four elements the guide names in 5.3-K1):

```python
from claude_agent_sdk import AgentDefinition, ClaudeAgentOptions

web_researcher = AgentDefinition(
    description="Web search specialist. Use to find recent public sources on one assigned subtopic.",
    prompt="""You research one subtopic for a coordinator and report back.

If a search fails or times out, retry once with a narrower query.
If it still fails, finish your reply with a FAILURE REPORT section listing:
- failure type (timeout, access denied, rate limited, other)
- every query you ran
- the partial results you did get, each with its source URL
- alternatives the coordinator could try
If a search ran and found nothing, write "no matching sources" and list the queries.
That is a result, not a failure.""",
    tools=["WebSearch", "WebFetch"],
)

options = ClaudeAgentOptions(
    allowed_tools=["Agent", "WebSearch", "WebFetch"],
    agents={"web-researcher": web_researcher},
)
```

The example uses `Agent`, the current name of the tool the CCAR-F guide calls the "Task tool" (1.3-K1); the rename, and which name to expect on the exam, are in the exam-guide-versus-docs note under [Read the trace](#read-the-trace).

### Four outcomes, four different reports

| What happened inside the subagent | What it reports | What the coordinator can do next |
|---|---|---|
| A transient failure that a local retry fixed | Its normal findings | Nothing special |
| An access failure it could not resolve (a timeout, a service still unavailable after retries) | Structured error context: failure type, what was attempted, partial results, potential alternatives | Retry with a modified query, try an alternative approach, or proceed with the partial results (the three options in the sample question 8 rationale) |
| A query that ran and matched nothing | A successful empty result that says so | Accept it as an answer; a retry would return the same nothing |
| A failure no retry can change (permission denied, a business rule) | The error, marked not retryable, with a plain explanation | Explain it to the user or escalate; a retry is wasted (2.2-K4, 2.2-S2) |

The second and third rows are the ones the exam separates: an access failure needs a retry decision, a valid empty result does not (5.3-K2). A report that blurs them forces the coordinator to guess. The guide also rejects a generic status such as "search unavailable" (5.3-K3), and the sample question 8 rationale gives the reason: "Option B's generic status hides valuable context from the coordinator, preventing informed decisions." ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf))

### Say which kind of empty it is

The distinction has to survive into the text of the result. Three reference points in current documentation (as of September 2026):

- **Search tools.** When "a search fails or returns nothing", Anthropic's search-results guidance says to return a plain text block describing the outcome instead of raising an error ([search results](https://platform.claude.com/docs/en/build-with-claude/search-results)). That rule is about the format of the result, not a license to merge the two cases: the text can, and for 5.3-S2 should, say whether the search ran and matched nothing or could not run.
- **MCP resources.** The specification forbids the ambiguous form outright: "Servers **MUST NOT** return an empty `contents` array for a non-existent resource." ([MCP resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources))
- **Claude Code's own history.** "Before v2.1.208, Claude Code reported a rejected input as `No files found` instead of an error, even when the searched-for text existed in the target files." ([tools reference](https://code.claude.com/docs/en/tools-reference)) That is exactly the failure disguised as an empty result that the guide warns about.

In the Agent SDK, the error flag on a custom tool's result (`isError: true` in TypeScript, `"is_error": True` in Python) does the same job; the docs' own Python code comment reads "is_error marks this as a failed call rather than odd-looking data." ([Agent SDK custom tools](https://code.claude.com/docs/en/agent-sdk/custom-tools))

### Three wrong shapes

| Shape | Why it fails |
|---|---|
| **Swallow:** catch the timeout and return an empty result marked successful | The coordinator, and whoever reads the final report, cannot tell *we looked and found nothing* from *we could not look*. No recovery is attempted and the research output is silently incomplete |
| **Blur:** retry with backoff, then return a generic "search unavailable" status | The local retry is fine (the guide asks for local recovery); the context-free status afterwards is the flaw |
| **Abort:** propagate the exception to a top-level handler that ends the run | "Option D terminates the entire workflow unnecessarily when recovery strategies could succeed." ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)) |

### Local recovery, bounded

The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) wants subagents to "implement local recovery for transient failures and only propagate errors they cannot resolve, including what was attempted and partial results" (5.3-S3; 2.2-S3 says the same for MCP tools). Anthropic's published research-subagent prompt shows the behavior in prompt form: "if it seems like some info is not available on the web or some approach is not working, try using another tool or another query", and "NEVER repeatedly use the exact same queries for the same tools, as this wastes resources and will not return new results." ([research subagent prompt](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_subagent.md)) Anthropic's Research system post reports that "letting the agent know when a tool is failing and letting it adapt works surprisingly well" ([multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)).

Local recovery still needs a limit. The same [Research system](https://www.anthropic.com/engineering/multi-agent-research-system) pairs agent adaptability "with deterministic safeguards like retry logic and regular checkpoints". Retries only help when something can change: a transient outage, a narrower query. They cannot recover information that is not there, the same limit the guide states for extraction retries (4.4-K2). Retry budgets by layer are in [Reliability engineering for Claude applications](#reliability-engineering-for-claude-applications).

### What the platforms do when a subagent fails (as of September 2026)

Anthropic's own tooling largely follows the principle: a failure is reported as a failure rather than as findings, and when a Claude Code subagent produced text output before failing, that output is kept.

| Surface | Behavior |
|---|---|
| Claude Code, any subagent (v2.1.199 and later) | A subagent whose run ends on an API error "reports that failure back to Claude instead of returning the error text as if it were the subagent's findings" |
| Claude Code, foreground subagent cut off by a rate limit, overload or server error after producing text | The Agent tool returns the partial output with a note that the subagent was cut off and did not finish |
| Claude Code, foreground subagent that produced nothing, or only tool calls | The call fails with `Agent terminated early due to an API error`, followed by the error detail |
| Claude Code, background subagent | Marked failed; the notice names the API error and includes the last output, so partial work is not lost |
| Claude Code, response cut off mid-stream with text but no tool calls | Claude Code prompts the subagent to continue rather than ending the run; the run ends on the error only once those continuations are used up |
| Claude Code with a fallback model chain | A covered failure switches the subagent to the first model in the chain that accepts the request |
| Agent SDK | "An API error that ends the subagent early, such as a rate limit, is never delivered as its result." Output at a subagent's `maxTurns` limit is marked partial (Claude Code v2.1.246 or later) and the subagent can be resumed. With the SDK releases that bundle Claude Code v2.1.219 or later, at the concurrency limit the SDK refuses to spawn another subagent and returns `Concurrent subagent limit reached`. At the `maxBudgetUsd` (TypeScript) or `max_budget_usd` (Python) cap it refuses to spawn more subagents, returning `Budget limit reached`, stops background subagents still running, and ends the query with the `error_max_budget_usd` result subtype |
| Claude Code dynamic workflows | An `agent()` call resolves to `null` if you stop it mid-run or it hits an unrecoverable API error; `pipeline()` keeps each `null` in its results array |
| Claude Managed Agents | A failed or interrupted advisor consultation "never fails the agent's turn": the agent continues "after a generic notice that the consultation failed"; interrupting a child thread blocked on `requires_action` closes each pending tool call with an error tool result ("Tool execution was interrupted before completion. Please retry.") |

Sources: [Claude Code subagents](https://code.claude.com/docs/en/sub-agents), [Claude Code errors](https://code.claude.com/docs/en/errors), [Agent SDK subagents](https://code.claude.com/docs/en/agent-sdk/subagents), [Claude Code workflows](https://code.claude.com/docs/en/workflows), [Managed Agents multiagent orchestration](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration). Your own code should not undo this: a subagent result marked failed or partial should never be rewrapped as an empty result. The workflows docs' own example ends with `.filter(Boolean)` to drop the `null` entries from `pipeline()`; if the output feeds a report, count those entries before filtering and report them as gaps (our suggestion, following 5.3-K4 and 5.3-S4). Spawn limits (depth, concurrency, budget) and their defaults are in [Subagents in the SDK](agents-and-agent-sdk.md#subagents-in-the-sdk).

### From partial results to an honest report

A run that lost a source can still finish, provided the output says what is missing. The [CCAR-F guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) fourth skill structures synthesis output "with coverage annotations indicating which findings are well-supported versus which topic areas have gaps due to unavailable sources" (5.3-S4), and Exercise 4 simulates a subagent timeout and checks that the coordinator can "proceed with partial results and annotate the final output with coverage gaps" (Exercise 4 step 4). [Anthropic's research-subagent prompt](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_subagent.md) applies the same honesty to source quality: when results have potential issues (speculation, unconfirmed reports, news aggregators instead of original sources), it tells the subagent to "flag these issues when returning your report to the lead researcher rather than blindly presenting all results as established facts." Claude Code's `/deep-research` workflow keeps the same distinction: a claim its verifier agents could not check (after a rate limit or API error) is listed as unverified, not counted as refuted ([Claude Code workflows](https://code.claude.com/docs/en/workflows)).

An illustrative coverage block at the end of a research report (our wording, built on the 5.3-S4 skill and the sample question 7 topic):

```text
COVERAGE
Well supported (several independent sources): visual arts, music
Partial (one source): writing
Gap: film production. The web search subagent timed out twice on
  "AI in film post-production"; no sources were retrieved, so this
  area was not researched.
```

Claim-to-source mappings and conflicting sources are covered in [Provenance and uncertainty in synthesis](context-engineering.md#provenance-and-uncertainty-in-synthesis).

### Keep failures visible

The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) routes "all subagent communication through the coordinator for observability, consistent error handling, and controlled information flow" (1.2-S4). With Agent SDK tracing on, subagent spans nest under the parent's `claude_code.tool` span, "so the full delegation chain appears as one trace" ([Agent SDK observability](https://code.claude.com/docs/en/agent-sdk/observability)). Setup is in [See failures: tracing and telemetry](#see-failures-tracing-and-telemetry).

### Decide

- If a subagent's access failure survives local retries, return structured error context (failure type, attempted query, partial results, alternatives); not a generic status, not an empty result marked successful, not an exception that ends the run.
- If the query ran and matched nothing, report a successful empty result and say so in the text; not an error, because a retry cannot change a valid empty answer.
- If the coordinator receives an error report, choose among a modified retry, an alternative approach, or proceeding with partial results and annotating the gap; not discarding the partial results.
- If you re-delegate after a failure, write the failed query, the error and what is already known into the new subagent prompt; not an assumption that the subagent remembers, because only the prompt string crosses over.
- If some sources stayed unavailable, finish with coverage annotations; not a report presented as complete.

### Traps

- **Retry with exponential backoff, then report a generic status.** It sounds careful; the generic status at the end is what the sample question 8 rationale rejects.
- **Zero results as a stand-in for failure.** Any option that makes a timeout look like an empty search is suppression.
- **One failure, whole run gone.** A top-level handler that terminates the workflow on a single timeout.
- **Assuming the coordinator saw the error.** Tool errors inside a subagent stay there unless its final message reports them.
- **Rewrapping a failed subagent as "no findings".** The platforms mark failed and partial runs; code that flattens them into empty results reintroduces the anti-pattern.

## Escalation and ambiguity

*Tested in: CCAR-F 5.2 (5.2-K1 to 5.2-S5), 1.4-K3, 1.4-S3, 1.5-S2, appendix in-scope topic 8, How to Prepare item 7, Exercise 1 step 4, Scenario 1, sample question 3 · CCAO-F credential purpose (recognize limitations and escalate) · CCDV-F D8.1 Tool Implementation (approval patterns) · CCAR-P 5.1, 5.3*

Escalation here means handing a case to a human. CCAR-F Scenario 1 sets the tension: a support agent with an `escalate_to_human` tool, and "Your target is 80%+ first-contact resolution while knowing when to escalate." ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)) Errors run both ways. Anthropic's human-in-the-loop cookbook puts it plainly: "Calibration matters here, an agent that escalates everything is exhausting to work with, and an agent that escalates nothing is dangerous." ([Managed Agents human-in-the-loop cookbook](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/managed_agents/CMA_gate_human_in_the_loop.ipynb)) The exam page lists the guide's rules situation by situation ([CCAR-F Domain 5](../claude-certified-architect-foundations.md#domain-5-context-management-reliability)); this section explains the design behind them.

### Three layers of an escalation design

| Layer | What it decides | Mechanism | In the guide |
|---|---|---|---|
| Criteria in the system prompt | Judgment calls: resolve or hand over | Explicit escalation criteria plus few-shot examples | 5.2-S1, sample question 3 |
| An escalation tool | How the handoff happens and what it carries | A tool such as `escalate_to_human`, whose input is a structured handoff | Scenario 1, 1.4-S3 |
| A hook for hard limits | Rules that must hold on every call | A tool-call interception hook that blocks a policy-violating call (a refund above a threshold) and redirects to escalation | 1.5-S2, Exercise 1 step 4 |

The split follows the guide's enforcement rule. The sample question 1 rationale in the [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says programmatic enforcement "provides deterministic guarantees that prompt-based approaches cannot", and 1.4-K2 says prompt instructions alone "have a non-zero failure rate" when compliance must be deterministic. A threshold that must never be crossed goes in a hook; whether a case needs a person is judgment, so it goes in the prompt. Hook code that blocks a refund above a threshold and redirects to escalation is in [Hooks in the SDK](agents-and-agent-sdk.md#hooks-in-the-sdk).

### The three triggers

The [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) lists "Appropriate escalation triggers: customer requests for a human, policy exceptions/gaps (not just complex cases), and inability to make meaningful progress" (5.2-K1). The rules below are stated in the guide's own words; learn them in that wording.

**1. The customer asks for a human.** Honor it "immediately without first attempting investigation" (5.2-S2). The guide separates an explicit demand from frustration: when the issue is straightforward and within the agent's capability, the agent acknowledges the frustration while offering the resolution, and escalates "only if the customer reiterates their preference" (5.2-K2, 5.2-S3). Anthropic's customer-support skills (written for human support staff who use Claude) take a similar line: acknowledge the customer's situation before jumping to solutions, and escalate to your manager when a customer "requests direct contact with leadership" ([draft-response skill](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/customer-support/skills/draft-response/SKILL.md)).

**2. Policy needs an exception, or does not cover the request.** The guide's example: a customer asks for competitor price matching when policy only addresses own-site adjustments (5.2-S4). The trigger is what policy allows, not how hard the case looks: in sample question 3 the agent failed by escalating "straightforward cases (standard damage replacements with photo evidence)" while attempting "complex situations requiring policy exceptions" itself. The same skills list "Customer requests exception to policy you can't authorize" as an escalation trigger, and Anthropic's customer support guide adds the matching guardrail: the agent must not make promises or enter into agreements it is not authorized to make ([customer support guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/customer-support-chat)).

**3. The agent cannot make meaningful progress.** The guide names this trigger (5.2-K1; "inability to progress" in How to Prepare item 7) without defining it further. Anthropic's customer-escalation skill has a related trigger: it escalates when "normal support channels aren't progressing" or an issue has been open beyond its SLA; its opposite is a case with "a documented solution or known workaround", which support handles itself ([customer-escalation skill](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/customer-support/skills/customer-escalation/SKILL.md)).

### Signals that look useful and are not triggers

The rationale for sample question 3 in the [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) rejects three of these by name:

| Signal | Why it fails as a trigger | Where it belongs instead |
|---|---|---|
| Negative sentiment | "sentiment doesn't correlate with case complexity, which is the actual issue." | A business metric: Anthropic's support guide aims for sentiment maintained or improved in 90% of interactions |
| Self-reported confidence | "LLM self-reported confidence is poorly calibrated"; the agent "is already incorrectly confident on hard cases" | Review routing, after calibration on labeled data ([Human review and confidence calibration](#human-review-and-confidence-calibration)) |
| A separate classifier trained on past tickets | "Option C is over-engineered, requiring labeled data and ML infrastructure when prompt optimization hasn't been tried." | A later step, if explicit criteria prove insufficient |
| Complexity on its own | The trigger is a policy exception or gap, "not just complex cases" (5.2-K1) | The agent's normal work, when policy covers it |

Sentiment still deserves instructions. Anthropic's ticket-routing guide warns that "When customers express dissatisfaction, Claude may prioritize addressing the emotion over solving the underlying problem", and its fix is to tell Claude when to prioritize sentiment and when not to ([ticket routing guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/ticket-routing)).

### Writing the criteria

The guide's first skill is "Adding explicit escalation criteria with few-shot examples to the system prompt demonstrating when to escalate versus resolve autonomously" (5.2-S1). The sample question 3 rationale calls this the fix that "directly addresses the root cause: unclear decision boundaries" and adds: "This is the proportionate first response before adding infrastructure." How many examples: the guide's few-shot skill says "2-4 targeted few-shot examples for ambiguous scenarios" that show why one action beat the alternatives (4.2-S1); Anthropic's prompting guide recommends 3 to 5 examples for best results, diverse enough to cover edge cases ([prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)). Both ranges include 3 or 4. Few-shot technique in general is in [Few-shot examples](prompt-engineering.md#few-shot-examples).

An illustrative criteria block, built from the guide's triggers and its sample question 3 and 5.2-S4 examples (our wording, not an Anthropic template):

```text
<escalation_policy>
Call escalate_to_human when any of these is true:
1. The customer asks for a human agent. Do it at once; do not investigate first.
2. The request needs an exception to policy, or the policy does not cover it.
3. You cannot make meaningful progress with the tools you have.
Do not escalate only because the customer is upset. Acknowledge the frustration and
offer the fix when the issue is within policy; if they then ask for a person, rule 1 applies.
</escalation_policy>

<examples>
<example>
Customer: My blender arrived cracked. Photo attached. I want a replacement.
Decision: resolve. A standard damage replacement with photo evidence is within policy.
</example>
<example>
Customer: A competitor sells this for less. Match their price.
Decision: escalate (policy gap). Our policy covers price adjustments on our own site only.
</example>
<example>
Customer: This is ridiculous, I have waited a week. Where is my order?
Decision: resolve. Acknowledge the delay, look up the order and give the status.
Escalate only if the customer asks for a person.
</example>
</examples>
```

Explaining the reason behind each decision matters: [Anthropic's prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) says context or motivation behind an instruction helps Claude deliver more targeted responses, and the guide's own few-shot bullet says examples let the model "generalize judgment to novel patterns" (4.2-K3).

### The escalation tool and how a run waits for a person

Anthropic's human-in-the-loop cookbook builds an expense approver around two custom tools: "`decide()` for clear-cut cases and `escalate()` for ambiguous ones". Its system prompt also shows criteria in practice: it names what counts as ambiguous (receipts near thresholds, unclear categories, suspicious notes). The cookbook declares both tools in the Managed Agents agent's `tools=` array, next to the built-in `agent_toolset_20260401`. The two custom tool definitions below are verbatim from the cookbook; only the `tools = [...]` assignment around them is ours:

```python
tools = [
    {
        "type": "custom",
        "name": "decide",
        "description": "Record a final approve/reject for a clear-cut receipt.",
        "input_schema": {
            "type": "object",
            "properties": {
                "receipt_id": {"type": "string"},
                "action": {"type": "string", "enum": ["approve", "reject"]},
                "reason": {"type": "string"},
            },
            "required": ["receipt_id", "action", "reason"],
        },
    },
    {
        "type": "custom",
        "name": "escalate",
        "description": "Surface an ambiguous receipt for human review.",
        "input_schema": {
            "type": "object",
            "properties": {
                "receipt_id": {"type": "string"},
                "question": {"type": "string"},
            },
            "required": ["receipt_id", "question"],
        },
    },
]
```

Making escalation a separate tool gives the handoff an input schema that names its required fields, and makes each escalation a discrete call you can log and count (our reasoning). How the run then waits depends on the surface (as of September 2026):

| Surface | What happens while a person decides |
|---|---|
| Managed Agents custom tool | The session pauses and emits `agent.custom_tool_use`; your application answers with a `user.custom_tool_result` event that passes the event ID as `custom_tool_use_id`. In the cookbook's production pattern, a webhook on `session.status_idled` lets your server put the case in front of a reviewer and post the result whenever the human finishes, "no long-lived connection on your side" (the cookbook's local streaming pattern, by contrast, holds an HTTP connection open while humans think) |
| Managed Agents `always_ask` permission policy | The session idles with `stop_reason.type` `requires_action` and waits indefinitely; answer with a `user.tool_confirmation` event (`result` `"allow"` or `"deny"`, optional `deny_message`) |
| Agent SDK `canUseTool` callback | Execution pauses until the callback returns; it can stay pending indefinitely |
| Agent SDK `PreToolUse` hook returning `defer` | The process exits and resumes later from the persisted session, for users who take longer than the process can stay alive. Claude Code honors `defer` only in non-interactive `-p` mode and only when Claude makes a single tool call in the turn. There is no timeout, but the session file is subject to the `cleanupPeriodDays` retention sweep (30 days by default) |
| Agent SDK custom tool | Connects to existing ticketing, workflow or approval platforms |

For writes with real consequences, Anthropic's commerce-agent blueprint goes further: every write its merchant agent proposes (a listing update, a price change, an inventory action, a promotion or a campaign) is a staged change with a server-generated ID shown to the operator as a preview card, and it applies only after a person approves it outside the conversation. "An approval typed in chat approves nothing." ([commerce agents guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/commerce-agents))

### The handoff: write for someone without the transcript

The guide's handoff skill is "Compiling structured handoff summaries (customer ID, root cause, refund amount, recommended action) when escalating to human agents who lack access to the conversation transcript" (1.4-S3). The human starts cold, so the summary must stand on its own. The guide lists those four items and no fuller schema. Anthropic's customer-escalation skill shows a fuller structure, written for escalations from support to engineering, product or leadership rather than to a human agent taking over the conversation: it packages an issue into "a structured escalation brief" with sections for impact, issue description, what has been tried, reproduction steps, customer communication, what is needed (with a deadline) and supporting context ([customer-escalation skill](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/customer-support/skills/customer-escalation/SKILL.md)).

| The receiving human needs | From the guide (1.4-S3) | From Anthropic's escalation brief |
|---|---|---|
| Who the case is about | Customer ID | Customers affected |
| What is wrong and why | Root cause | Issue description |
| What is at stake | Refund amount | Impact |
| What already happened | | What has been tried |
| What to do next | Recommended action | What is needed: the specific ask, and a deadline |

The brief's own best practices include "Always quantify impact", being clear whether you need someone to "investigate", "fix" or "decide", and keeping ownership of the customer relationship after escalating.

### Ambiguity: ask, or assume and say so

**Customer identity: ask.** When a lookup returns several customers, the guide wants "requesting additional identifiers" rather than "heuristic selection" (5.2-K4, 5.2-S5). The docs support the reasoning: an invalid tool call "usually means that there wasn't enough information for Claude to use the tool correctly" ([handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls)), and the commerce blueprint lets cart writes accept "only product IDs that a catalog or order tool returned in that session", so a guessed product ID cannot reach a cart write.

**Asking in the Agent SDK (as of September 2026).** Claude asks clarifying questions with the `AskUserQuestion` tool when a task has multiple valid approaches. Each call carries 1 to 4 questions with 2 to 4 options each; if you restrict `tools`, include `AskUserQuestion` or Claude cannot ask; and it "is not currently available in subagents spawned via the Agent tool", so a clarifying question has to come from the agent that talks to the user ([Agent SDK user input](https://code.claude.com/docs/en/agent-sdk/user-input)).

**Coding work: often assume, and say so.** Anthropic's prompting guide for Claude Fable 5.1 suggests adding an instruction to coding prompts that keeps changes to what the task asks for; it includes: "Where the task is ambiguous, implement the reading its wording and the surrounding code most directly support, state that assumption in your summary, and don't build for the other readings as well." ([Prompting Claude Fable 5.1](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-1))

The two fit one rule, which a sample prompt in [Anthropic's prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) gives for agent actions: "Consider the reversibility and potential impact of your actions." Take local, reversible actions freely; ask before actions that are hard to reverse, affect shared systems, or could be destructive. Acting on the wrong customer's account is hard to reverse, so ask; a stated coding assumption is reviewed before it matters, so proceed and state it (our reconciliation of the two sources).

### Measuring escalation

Anthropic's [customer support guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/customer-support-chat) aims for escalation accuracy of 95% or higher and, typically, a 70 to 80% deflection rate; both targets and the rule for balanced test sets are in [Success criteria and test sets](#success-criteria-and-test-sets). An eval set for escalation needs cases that must be escalated and cases that must not.

### Two other meanings of escalation

- **Associate level (CCAO-F).** The Associate credential covers the ability to "Recognize limitations and escalate more complex or technical implementations", and the guide places enterprise-scale architectures and integrations with the Architect and Developer credentials, "to which Associates escalate more complex or technical work" ([CCAO-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)). Its preparation list includes practicing "when to escalate or seek human review".
- **Managed Agents multiagent docs.** There, "Escalation" names a pattern in which an agent consults "a more capable agent or model for a subset of complex subtasks" ([multiagent orchestration](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration)). In CCAR-F task 5.2 and Scenario 1, escalation means handing the case to a human.

### Decide

- If escalation is miscalibrated in both directions, add explicit criteria with a few worked examples to the system prompt first; not a confidence threshold, a sentiment trigger or a new classifier.
- If a rule must hold on every call (a refund ceiling), enforce it in a hook that redirects to escalation; not a prompt instruction alone.
- If you escalate, pass a self-contained summary: customer ID, root cause, refund amount, recommended action (plus what was already tried); not a pointer to a transcript the human cannot see.
- If a customer lookup returns several matches, ask for an additional identifier; not the likeliest match.
- If the wait for a human may outlast the process, pause with `defer` or a Managed Agents custom tool; not a callback held open for hours.

### Traps

- Treating an upset customer as a request for a human, or an explicit request as something to investigate first.
- Escalating every complex case: complexity inside policy is the agent's job.
- A handoff that says only *customer needs help*: the human lacks the transcript.
- Accepting a *yes, go ahead* typed in chat as approval for a high-impact write.
- Expecting a subagent to ask the user a clarifying question with `AskUserQuestion`.

## Human review and confidence calibration

*Tested in: CCAR-F 5.5 (5.5-K1 to 5.5-S4), 4.6-S3, 5.2-K3, appendix (confidence scoring; human review workflows), Exercise 3 step 5, How to Prepare item 7 · CCAO-F D2.2, D2.3, D2.4, sample question 1 · CCDV-F D6.3 Output Handling (response validation, skepticism toward confident output) · CCAR-P 4.2, 5.3*

Reviewer capacity is limited, and the guide asks you to spend it where it matters: "prioritizing limited reviewer capacity" (5.5-S4). This section covers where people sit in the loop, what a model's own confidence is worth, how to turn it into a routing signal you can trust, and how to show that automation is safe segment by segment before you reduce review. Calibrating an LLM grader against human graders is in [Grading methods](#grading-methods); independent review instances and multi-pass review are in [Multi-instance and multi-pass review](prompt-engineering.md#multi-instance-and-multi-pass-review).

### Where a person sits in the loop

| Position | What the person does | Examples (as of September 2026) | Taught in depth |
|---|---|---|---|
| Before an action | Approves or denies a tool call or a write | Agent SDK `canUseTool`; Managed Agents `always_ask`; commerce staged changes approved outside the conversation; `permissions.ask` rules for pushes in Claude Code auto mode | [Permissions and enforcement](agents-and-agent-sdk.md#permissions-and-enforcement), [Escalation and ambiguity](#escalation-and-ambiguity) |
| After an output, before it is used | Reviews items routed to them | Low-confidence or ambiguous extractions (5.5-S4); a cited subsection checked before a summary goes to compliance (CCAO-F sample question 1) | This section |
| Continuously, on a sample | Measures error rates nobody would otherwise see | Stratified sampling of high-confidence output (5.5-S1); transcripts read weekly | This section, [Evaluating agents](#evaluating-agents) |
| Over the evaluation system | Keeps automated graders honest | Model-based graders calibrated against human graders | [Grading methods](#grading-methods) |

For actions, the protocol and platform docs put a person before the call. The MCP specification says there SHOULD always "be a human in the loop with the ability to deny tool invocations" ([MCP tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)). Managed Agents warns that under the `auto` policy, end-user input you relay in `user.message` events counts as your intent, and tells you to "Configure `always_ask` on the tools you would not let that end user run without review" ([permission policies](https://platform.claude.com/docs/en/managed-agents/permission-policies)). In Claude Code auto mode, a boundary you state in conversation (for example, wait for my review before deploying) blocks until you lift it, and "Claude's own judgment that a condition was met does not lift it." Such a boundary is not stored as a rule, though: it can be lost if context compaction removes the message that stated it, and for a hard guarantee the docs point to a deny rule ([permission modes](https://code.claude.com/docs/en/permission-modes)).

### What a model's confidence is worth

| Finding | Source |
|---|---|
| Claude models show some introspective awareness, but "this introspective capability is still highly unreliable and limited in scope"; with the best protocol, Claude Opus 4.1 showed it about 20% of the time | [Emergent introspective awareness in LLMs](https://www.anthropic.com/research/introspection) (Oct 29, 2025) |
| Models do not always report their internal states accurately: "in many cases, they are making things up!" | Same |
| Chains of thought are often unfaithful: "Claude 3.7 Sonnet mentioned the hint 25% of the time" when it used one | [Reasoning models don't always say what they think](https://www.anthropic.com/research/reasoning-models-dont-say-think) (Apr 3, 2025) |
| "larger models are well-calibrated on diverse multiple choice and true/false questions when they are provided in the right format", but they "struggle with calibration of P(IK) on new tasks" | [Language models (mostly) know what they know](https://www.anthropic.com/research/language-models-mostly-know-what-they-know) |

The practical reading: a confidence number the model reports about its own output is an uncalibrated signal until you have measured it against labeled outcomes on your own task. The [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) holds both halves of that. It calls self-reported confidence scores "unreliable proxies for actual case complexity" when deciding escalation (5.2-K3), and it uses "Field-level confidence scores calibrated using labeled validation sets for routing review attention" (5.5-K3); the difference is the calibration step. Task 4.6 adds verification passes where the model self-reports confidence alongside each finding "to enable calibrated review routing" (4.6-S3). Read the guide consistently: raw confidence never replaces an escalation rule, and calibrated confidence can route review.

The same holds for a person reading output. A confident tone is not evidence. CCAO-F sample question 1 rejects both sending a summary because "Claude expressed high confidence" and asking Claude to rate itself: "Self-reported confidence (A, C) is not a reliable accuracy signal" ([CCAO-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)). CCDV-F D6.3 asks for "skepticism toward confident output" ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)), and the Claude Help Center warns that "Claude can display quotes that may look authoritative or sound convincing, but are not grounded in fact." ([incorrect or misleading responses](https://support.claude.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on))

### From a raw score to a routing threshold

The CCAR-F skill is "Having models output field-level confidence scores, then calibrating review thresholds using labeled validation sets" (5.5-S3), followed by "Routing extractions with low model confidence or ambiguous/contradictory source documents to human review, prioritizing limited reviewer capacity" (5.5-S4). The guide gives no confidence thresholds, sample sizes or sampling rates for this task. One way to carry it out (our procedure, built on those two skills):

1. **Ask for confidence per field** in the structured output, next to each value. Anthropic's own tools show the shape: the security-review action's prompt defines confidence bands and says "Below 0.7: Don't report (too speculative)", and the claude-code-action `test-failure-analysis.yml` example asks for a `confidence` from 0 to 1 and retries a test automatically only when it is flaky and the confidence is 0.7 or above (see [Automated code review that engineers trust](claude-code-workflows.md#automated-code-review-that-engineers-trust)). Both files use a fixed 0.7 cutoff, and neither describes calibrating it.
2. **Label a validation set** drawn from real traffic, tagged by document type and field (5.5-K4).
3. **Measure observed accuracy per confidence band, per segment.** Set the threshold at the lower edge of the lowest band that, along with every band above it, meets your target. The threshold rests on observed accuracy, not on the model's number itself.
4. **Route below the threshold, and route ambiguous or contradictory sources regardless of score.** Anthropic's customer-research skill lists "Contradictory information found across sources" as a low-confidence signal ([customer-research skill](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/customer-support/skills/customer-research/SKILL.md)).
5. **Keep sampling above the threshold** (see [Sample the output nobody reviews](#sample-the-output-nobody-reviews) below).
6. **Re-measure when the setup changes.** A threshold measured on one model, prompt and document mix describes that setup (our reasoning). [Anthropic's prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) applies the same logic to prompting techniques: where a technique names a specific model, treat it as measured on that model and "re-check it against your own evals before applying it to another." Production monitoring exists "to detect distribution drift and unanticipated real-world failures" ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).

A worked example for one field, `invoice_total` on scanned receipts, with a 97% target for auto-accept (hypothetical figures, our arithmetic):

| Model-reported confidence | Labeled items | Correct | Observed accuracy | Route |
|---|---|---|---|---|
| 0.9 to 1.0 | 400 | 392 | 98% | Auto-accept, keep sampling |
| 0.7 to 0.9 | 250 | 225 | 90% | Human review |
| Below 0.7 | 150 | 105 | 70% | Human review |

A fixed 0.7 cutoff would auto-accept the middle band at 90% accuracy, well short of the target. Calibration moves the cutoff to 0.9 for this field on these documents; another field or document type may land elsewhere.

### Segment before you automate

The guide warns that aggregate accuracy metrics "(e.g., 97% overall) may mask poor performance on specific document types or fields" (5.5-K1) and asks you to analyze "accuracy by document type and field to verify consistent performance across all segments before reducing human review" (5.5-S2). How a 97% can hide a failing segment (hypothetical figures, our arithmetic):

| Segment | Documents | Correct | Accuracy |
|---|---|---|---|
| Typed invoices | 9,000 | 8,865 | 98.5% |
| Handwritten receipts | 1,000 | 835 | 83.5% |
| All documents | 10,000 | 9,700 | 97.0% |

Automating on the 97.0% would send about one handwritten receipt in six through unreviewed with an error (165 of 1,000, our arithmetic). The decision is per segment: automate typed invoices if 98.5% meets the bar, keep reviewing handwritten receipts. Tagging test cases by segment is covered in [Success criteria and test sets](#success-criteria-and-test-sets).

### Sample the output nobody reviews

Items accepted on high confidence are exactly the ones no reviewer sees, so their error rate is unknown unless you measure it. The guide's answer is "Stratified random sampling for measuring error rates in high-confidence extractions and detecting novel error patterns" (5.5-K2, 5.5-S1). Stratifying by segment keeps small document types in the sample instead of letting the largest segment dominate it. Anthropic's evaluation guidance points the same way for agents: triage feedback constantly and "sample transcripts to read weekly", because user feedback is sparse and self-selected ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)).

### Make review fast and honest

- **Give reviewers something to check against.** The API's citations feature returns citations that "are guaranteed to contain valid pointers to the provided documents" ([citations](https://platform.claude.com/docs/en/build-with-claude/citations)), so a reviewer can jump to the cited text. Techniques for grounding output in quotes are in [Reducing hallucinations](prompt-engineering.md#reducing-hallucinations).
- **Tier by risk.** Anthropic's content moderation guide suggests risk levels instead of a binary label, so high-risk content can be blocked automatically "while users with many medium risk queries are flagged for human review" ([content moderation guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/content-moderation)).
- **Watch for polish lowering scrutiny.** Anthropic's AI Fluency Index report found that in conversations where artifacts were created, users were less likely to identify missing context (by 5.2 percentage points), check facts (3.7) or question the model's reasoning (3.1) ([Anthropic Education Report: The AI Fluency Index](https://academy.claude.com/tutorials/the-ai-fluency-index)). A clean-looking output is when a reviewer most needs a checklist.
- **Keep human review next to automated checks.** For coding agents, Anthropic notes that automated tests verify functionality, but "human review remains crucial for ensuring solutions align with broader system requirements" ([Building effective agents](https://www.anthropic.com/engineering/building-effective-agents)).

### For Associates: when must a person check? (CCAO-F D2.4)

CCAO-F D2.4 asks candidates to "Determine when human review or additional verification is required" ([CCAO-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)). A working rule (ours) built from Anthropic's guidance:

- **Always verify** specific facts, statistics, citations, dates, names and numbers, and anything obscure, niche or very recent: Claude Academy lists these among the situations where "hallucinations are most likely to happen" ([Why do AI models hallucinate?](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate)). Sample question 1's specific subsection number is this case: its rationale notes that models "can fabricate specific-looking details such as citation numbers".
- **Always verify** before output reaches a high-stakes or external audience. The Help Center: "Users should not rely on Claude as a singular source of truth and should carefully scrutinize any high-stakes advice given by Claude." The AI Fluency framework calls this Deployment Diligence: taking responsibility for "verifying and vouching for the outputs you use or share" ([AI Fluency: Key Terminology Cheat Sheet](https://www-cdn.anthropic.com/4396730ed190e691a3712cf2fd6bfe35509deca2.pdf)).
- **Verify against an authoritative source,** not by asking Claude to rate itself. Sample question 1's correct answer checks the cited subsection against the official regulation text, and its rationale calls validating citations bound for a compliance audience "against an authoritative source" the required diligence step. Claude Academy's tips include asking a fresh chat to find errors in an answer, but add: "For critical work, you should cross-reference with trusted sources."
- **Flag untraceable specifics.** Anthropic's Cowork course: "A specific date, name, or quote that you can't trace to an input is a flag, not a feature." ([Introduction to Claude Cowork: Hand Claude Cowork your first task](https://academy.claude.com/courses/introduction-to-claude-cowork/the-task-loop))

More on checking output in the apps is in [Verifying Claude's output](claude-for-work.md#verifying-claudes-output).

### Decide

- If a model reports confidence, use it to route review only after calibrating thresholds on a labeled validation set, per field and document type; not the raw number, and not a round default.
- If overall accuracy is high and someone proposes cutting review, break accuracy down by segment first and automate only the segments that clear the bar; not the aggregate.
- If extractions are accepted on high confidence, keep a stratified random sample of them under review; not review of low-confidence items only.
- If reviewers are scarce, send them low-confidence items and items from ambiguous or contradictory sources first; not an even spread.
- If an action is hard to undo, put the person before the action (an approval gate); not after it.

### Traps

- **Treating a confident answer as a correct one,** whether the confidence is in the tone or in a number the model reports.
- **Calibrating once.** A threshold chosen on one model and document mix can stop holding when either changes.
- **Reviewing only what the model flags.** Confident errors and new error patterns stay unmeasured.
- **Asking Claude whether Claude was right** as the verification step for a high-stakes fact.
- **Using a confidence score to decide customer escalation.** That is the 5.2 anti-pattern; escalation follows explicit criteria.

## Reliability engineering for Claude applications

*Tested in: CCDV-F D4.1 Debugging and Error Handling (recovery strategy selection), D2.2 Systems Life Cycle, D5.2 Technical Fundamentals, and purpose areas 2 ("error handling") and 5 ("monitoring production quality") · CCAR-P 3.4, 4.6, 5.2, 7.3 · CCAR-F intended audience ("escalation and reliability decisions"), 1.1-S3, 2.2-S3, 5.4-K4, 5.4-S4; rate limiting and quotas are out of scope · CCAO-F: not listed*

A reliable Claude application gives every failure class an owner and a planned response: a network blip, a capacity spike, an exhausted budget, a refusal, a runaway agent loop, a duplicated side effect, and the hardest one, a quality drop that raises no error at all. The CCAR-P preparation course names the core patterns: "specifying reliability patterns (retries, fallbacks, circuit breakers)" ([Enterprise Integration & Production](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/enterprise-integration-production)). How to recognize each error type in the first place is in [Debugging: model or integration](#debugging-model-or-integration), and the HTTP error catalog is in [Errors, retries and rate limits](claude-api.md#errors-retries-and-rate-limits). This section designs the responses.

### Controls at a glance

| Failure | Control | Subsection |
|---|---|---|
| Transient network or capacity error (connection error, 429 with `retry-after`, 500, 529) | One retry layer, exponential backoff, `retry-after` honored | Retries |
| Spend exhausted (429 with `enforced_spend_limit_reached`, or 400 when a spend limit you set is reached) | Alert a person and stop sending; retries keep failing | Retries |
| A retried call that already had side effects | Writes that are safe to repeat; staged changes | Duplicate side effects |
| Primary model overloaded or unavailable | Availability fallback chain | Fallbacks |
| Refusal (HTTP 200, `stop_reason: "refusal"`) | Retry on a different model; count refusals separately | Fallbacks |
| A dependency that keeps failing | A circuit breaker: stop calling it, apply a fail-closed or fail-open rule, reset when it recovers | Circuit breakers |
| Runaway agent loop | Turn and budget caps; branch on how the run ended | Bound every loop |
| A long run dies midway | Resume from saved state | Resume instead of restarting |
| A failure nobody sees (dropped telemetry, silent quality regression) | Tracing plus continuous evals on production traffic | See failures; Catch the failure that raises no error |
| A subagent fails | Local recovery for transient failures; structured error context to the coordinator for the rest | [Error propagation in multi-agent systems](#error-propagation-in-multi-agent-systems) |

### Retries: bounded, and owned by one layer

Each layer of the stack has its own retry budget (as of September 2026). Know which layer owns a failure before adding another loop:

| Layer | Default behavior | Control |
|---|---|---|
| Anthropic SDKs (Python, TypeScript) | 2 retries with exponential backoff, honoring `retry-after`; both SDKs retry connection errors, 408, 409, 429 and 500 or above; 10-minute default timeout (TypeScript scales it up to 60 minutes for large non-streaming `max_tokens`), and timed-out requests are retried | `max_retries` / `maxRetries`; `timeout` |
| Claude Code | Up to 10 retries with exponential backoff before showing an error | `CLAUDE_CODE_MAX_RETRIES` (default 10, capped at 15 unless the watchdog is set); `CLAUDE_CODE_RETRY_WATCHDOG=1` retries 429 and 529 capacity errors indefinitely in unattended runs such as CI (a standard-speed request whose 429 reports a spend limit or exhausted usage credits still fails at once) and, on v2.1.199 or later, raises the default retry count for other transient errors to 300 |
| Claude Code headless | A `system/api_retry` event before each retry, with `attempt`, `max_retries`, `retry_delay_ms`, `error_status` and an `error` category | `claude -p --output-format stream-json` |
| Claude itself, after an invalid tool call (missing or invalid parameters) | 2 to 3 retries with corrections before apologizing to the user | A `tool_result` that indicates the error (`is_error: true`); `strict: true` eliminates invalid tool calls |
| Structured output in the Claude Code CLI and workflows | 5 attempts (a first attempt plus four retries) | `MAX_STRUCTURED_OUTPUT_RETRIES` |
| Managed Agents outcomes | `max_iterations` default 3, maximum 20 | Per outcome |
| Code Review for GitHub pull requests (Claude Code) | A failed run does not retry on its own | Comment `@claude review`, or click Re-run on the check (not for pull requests from forks) |

Retry only what can change: a transient outage can clear, but a refusal on the same model, a tier spend cap and information missing from a source document do not. The class-by-class verdicts are in [Choose the recovery by error class](#choose-the-recovery-by-error-class); the missing-information case (4.4-K2) is taught in [Validation, retry and feedback loops](prompt-engineering.md#validation-retry-and-feedback-loops). Stacking your own retry loop on top of the SDK's multiplies the attempts for every failed call: three attempts of your own around the SDK's three (one call plus two retries) send up to nine requests (our arithmetic).

CCAR-F frames the same ownership rule for multi-agent systems: a subagent handles transient failures with local recovery and propagates to the coordinator only the errors it cannot resolve, "along with partial results and what was attempted" (2.2-S3; the guide's in-scope list says "local recovery before escalation") ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The coordinator side is taught in [Error propagation in multi-agent systems](#error-propagation-in-multi-agent-systems).

### Duplicate side effects

A retry is safe only if the operation is. Claude Code states the rule for its own requests: when a server error, dropped connection or stalled stream arrives after Claude has completed a block of text or a tool call but before the response finishes, "Claude Code doesn't re-run the request, because that could execute the same tool calls twice." ([Claude Code errors](https://code.claude.com/docs/en/errors)) It keeps what Claude completed, runs any tool calls Claude finished, and continues the turn from their results. Stream recovery has a related limit: tool use and extended thinking blocks cannot be partially recovered, so recovery resumes streaming from the most recent text block ([streaming](https://platform.claude.com/docs/en/build-with-claude/streaming)).

What the protocols give you, and what they do not:

- **MCP `idempotentHint`** means repeated calls with the same arguments have no additional effect; it defaults to `false` and matters only when `readOnlyHint` is `false`. Annotations are hints, not guarantees, and clients should never base tool-use decisions on annotations from untrusted servers ([MCP schema](https://modelcontextprotocol.io/specification/2026-07-28/schema)).
- **In the Agent SDK,** `idempotentHint` is informational only; only `readOnlyHint` changes behavior ([Agent SDK custom tools](https://code.claude.com/docs/en/agent-sdk/custom-tools)).
- **Some tool results ask for a retry.** When you interrupt a Managed Agents child thread that is blocked on `requires_action`, each pending tool call is closed with "Tool execution was interrupted before completion. Please retry." ([multiagent orchestration](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration)), so a call your code had already started can be requested again (our inference).

So make every write tool safe to call twice in your own code (our recommendation, since the annotations enforce nothing): check for the record before creating it, or key the write to an ID an earlier tool call returned. Anthropic's commerce-agent blueprint shows a stronger form: every write is a staged change with a server-generated ID, and guardrails are "checked when the change is staged and again when it is applied" ([commerce agents guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/commerce-agents)).

### Fallbacks: availability and refusal

Several mechanisms carry the name. The two in this table trigger on opposite failures, and a third, in Claude Code, follows it:

| | Availability fallback (Claude Code) | Refusal fallback (Claude API, beta) |
|---|---|---|
| Triggers on | The primary model is overloaded, unavailable, or returns another non-retryable server error | A safety-classifier decline (`stop_reason: "refusal"`). With `"default"`, only in a category that has a recommended fallback (otherwise the refusal stands); with a named list, the next model in the chain runs |
| Never triggers on | Authentication, billing, rate-limit, request-size and transport errors, or a denial by your organization's policy check | Rate limits, overloads and server errors, which are returned as-is |
| Configure | `--fallback-model` (comma-separated) or `fallbackModel` in settings; at most three models | `fallbacks: "default"` with the `server-side-fallback-2026-07-01` beta header, or a list of up to three models |
| Scope | The current turn only; the next message tries the primary again | The request, then, through sticky routing (retained for approximately 1 hour, best-effort), later requests in that conversation that include `fallbacks`, which go straight to the fallback model; it does not propagate into model calls made inside tool execution |
| Not available | | Message Batches; Amazon Bedrock, Google Cloud and Microsoft Foundry (use client-side SDK middleware there) |

Claude Code also has a third, content-based mechanism, automatic model fallback (as of September 2026). Fable models, Opus 5.5 and Opus 5 run with safety classifiers; when a classifier flags a request and the flagged category has a fallback model, Claude Code re-runs the request on that model, shows a notice, and the session continues on the fallback model until you switch back with `/model`. When the category has no fallback model, the request ends with the refusal ([model configuration docs](https://code.claude.com/docs/en/model-config)). It is separate from the availability chain above: one reacts to what the request contains, the other to whether the model is up.

Claude Code availability fallback, from the [model configuration docs](https://code.claude.com/docs/en/model-config):

```bash
claude --fallback-model sonnet,haiku
```

```json
{
  "fallbackModel": ["claude-sonnet-5", "claude-haiku-4-5"]
}
```

Server-side refusal fallback on the Claude API, trimmed from the [refusals and fallback docs](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback) (the model ID is the one the docs use):

=== "Python"

    ```python
    client = Anthropic()

    response = client.beta.messages.create(
        model="claude-fable-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Hello, Claude"}],
        fallbacks="default",
        betas=["server-side-fallback-2026-07-01"],
    )

    # A fallback_message entry in usage.iterations means a fallback model ran;
    # pair it with stop_reason to confirm the fallback served the response.
    fallback_ran = any(
        iteration.type == "fallback_message"
        for iteration in response.usage.iterations or []
    )
    served_by_fallback = fallback_ran and response.stop_reason != "refusal"
    ```

=== "TypeScript"

    ```typescript
    const client = new Anthropic();

    const response = await client.beta.messages.create({
      model: "claude-fable-5",
      max_tokens: 1024,
      messages: [{ role: "user", content: "Hello, Claude" }],
      fallbacks: "default",
      betas: ["server-side-fallback-2026-07-01"]
    });

    // A fallback_message entry in usage.iterations means a fallback model ran;
    // pair it with stop_reason to confirm the fallback served the response.
    const { stop_reason, model, usage } = response;
    const servedByFallback =
      (usage.iterations ?? []).some((entry) => entry.type === "fallback_message") &&
      stop_reason !== "refusal";
    ```

Four of the same page's [common pitfalls](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback) are operating rules: budget refusal retries per request, not per turn or per session (one turn can produce several refusals, for example an agent plus its sub-agents); configure fallback on every request path (retry handlers, error-recovery branches, background workers); give sub-agent calls their own fallback; and instrument refusals separately because "A refusal is an HTTP 200, so monitoring built on error rates or 5xx responses never sees it." In a Message Batch, a refused request comes back as `result.type: "succeeded"` with `stop_reason: "refusal"`, so a batch pipeline that checks only the result type will accept it.

### Circuit breakers

A retry helps when a failure clears; a circuit breaker helps when it does not. After enough consecutive failures, the breaker stops calling the failing dependency, applies a rule decided in advance (fail closed and block, or fail open and continue without it), and resets once the dependency recovers (our summary of the pattern the examples below share). The CCAR-P preparation course lists circuit breakers next to retries and fallbacks. Anthropic's products show the pattern in several places (as of September 2026):

- **Inference hooks (Claude Enterprise, beta).** If the organization's AI security server is unreachable, returns an error or times out, the organization's failure handling setting decides: block the request, or let it proceed without inspection. "Sustained failures attributable to your server trip a circuit breaker": Anthropic stops contacting the server, applies the failure handling setting to every request, and resets the breaker automatically once the server returns verdicts again ([Inference hooks](https://platform.claude.com/docs/en/manage-claude/inference-hooks)).
- **Spend limits on a Claude apps gateway.** The docs call per-developer spend limits "the gateway's per-developer view and circuit breaker on top of that shared bill", because without them "one runaway agent fleet can spend the organization's entire commitment" ([gateway spend limits](https://code.claude.com/docs/en/claude-apps-gateway-spend-limits)).
- **Claude Code's own loop guards (our comparison).** Claude Code overrides a `Stop` hook that keeps blocking once it reaches a cap, and auto mode pauses after repeated classifier blocks; both limits are in the table under [Bound every loop](#bound-every-loop).

Decide the fail direction before the breaker trips (our decision rule). For a safety or policy check, the CCAR-P preparation course wants a system that "fails closed rather than open" ([CCAR-P learning path](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional)); for an optional enrichment step, failing open may be proportionate.

### Bound every loop

Without limits, the Agent SDK loop runs until Claude finishes on its own, which can run long on open-ended prompts; the docs say "Setting a budget is a good default for production agents." ([Agent SDK agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop)) The documented stop limits (as of September 2026):

| Mechanism | Limit | What happens at the limit |
|---|---|---|
| Agent SDK `max_turns` / `maxTurns` | Tool-use turns; no limit by default | `ResultMessage` subtype `error_max_turns` |
| Agent SDK `max_budget_usd` / `maxBudgetUsd` | Spend, subagent requests included; no limit by default | `error_max_budget_usd`; the SDK refuses to spawn more subagents (`Budget limit reached`) |
| Claude Code `Stop` hook that keeps blocking | 8 consecutive blocks by default (`CLAUDE_CODE_STOP_HOOK_BLOCK_CAP`; `0` disables the cap) | Claude Code overrides the hook and ends the turn |
| `/goal` completion condition | Optional turn or time bound written into the condition, such as `or stop after 20 turns` | Claude reports progress against the bound each turn, and the evaluator judges it |
| Claude Code auto mode classifier | 3 blocks in a row or 20 in total (not configurable) | Auto mode pauses and prompting resumes; in a `-p` run without `--permission-prompt-tool`, the blocked action does not run, Claude keeps working and the run does not stop |
| Managed Agents outcome | `max_iterations` default 3, maximum 20 | Result `max_iterations_reached` |
| `claude plugin eval` | `--max-cost-usd` ceiling | Exit code 2, partial `results.json` |

Subagent depth and concurrency caps are in [Subagents in the SDK](agents-and-agent-sdk.md#subagents-in-the-sdk). A cap is a safety net, not the stopping rule. CCAR-F names "setting arbitrary iteration caps as the primary stopping mechanism" as an anti-pattern (1.1-S3); the loop should end when Claude finishes (`stop_reason` `"end_turn"`), with the cap there for when it does not ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Loop mechanics are in [The agentic loop](agents-and-agent-sdk.md#the-agentic-loop).

When a cap or an error ends the run, branch on how it ended. The result subtypes, the `success`-without-`structured_output` case and the `claude -p` exit and init-event checks are listed in [Debugging: model or integration](#debugging-model-or-integration). Three details matter for unattended code (as of September 2026):

- **A single-shot `query()` raises after the error result.** It yields the final result message, then raises an error that includes the failure text, such as `Reached maximum number of turns`, and the Claude Code process exits nonzero ([Agent SDK agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop)). Wrap the call so the failure is handled instead of crashing the caller.
- **On a nonzero CLI exit, Python raises one of two exception classes; TypeScript has no SDK class.** In Python, `ResultError` (the CLI reported an error result) subclasses `ProcessError` (the process exited nonzero without one), so catch `ResultError` first if you handle them differently; before `claude-agent-sdk` 0.2.140, error-result exits raised a plain `Exception` instead of a `ResultError`. In TypeScript, a nonzero CLI exit rejects the `for await` loop with a plain `Error`, so wrap the loop in `try`/`catch` and match on the message ([Agent SDK troubleshooting](https://code.claude.com/docs/en/agent-sdk/troubleshooting)).
- **Transcript mirroring can drop data.** When the SDK cannot deliver a transcript batch to your `SessionStore` (it makes at most three attempts in total), it drops the batch, emits a `system` message with subtype `mirror_error`, and continues the query; alert on it if transcript durability matters ([Agent SDK hosting](https://code.claude.com/docs/en/agent-sdk/hosting), [session storage](https://code.claude.com/docs/en/agent-sdk/session-storage)). Because a retried batch can re-deliver entries that already landed, the session storage docs say to deduplicate by `entry.uuid` in your `append()` implementation: the same safe-to-repeat rule as above.

### Resume instead of restarting

Anthropic's Research system learned that restarting a long agent run after an error is expensive: "Instead, we built systems that can resume from where the agent was when the errors occurred." It combines the agent's adaptability "with deterministic safeguards like retry logic and regular checkpoints" ([multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)). The products support the same pattern:

- **Agent SDK sessions.** Every result, error results included, carries the `session_id` you need to resume, along with `total_cost_usd`, `usage` and `num_turns` ([Sessions, resumption and forking](agents-and-agent-sdk.md#sessions-resumption-and-forking)).
- **Claude Code dynamic workflows.** The runtime saves each agent's result, which makes a run resumable within the session: on relaunch, completed agents return their saved result, and a failed agent reruns along with every agent started after it. "If a script starts A, B, C, and D in that order and B fails, relaunching returns A from cache and runs B, C, and D again." ([Claude Code workflows](https://code.claude.com/docs/en/workflows))
- **Your own state files.** CCAR-F's crash-recovery pattern has each agent export state to a known location and the coordinator load a manifest on resume and inject it into agent prompts (5.4-K4, 5.4-S4); see [Multi-agent handoffs](context-engineering.md#multi-agent-handoffs).

### See failures: tracing and telemetry

Without telemetry, you cannot tell an agent that stalled from one that is still working. The Agent SDK exports OpenTelemetry traces, metrics and log events, but "The SDK does not produce telemetry of its own": the Claude Code CLI child process does the instrumenting, and nothing is exported until `CLAUDE_CODE_ENABLE_TELEMETRY=1` is set and an exporter is chosen; traces (beta) also need `OTEL_TRACES_EXPORTER` plus `CLAUDE_CODE_ENHANCED_TELEMETRY_BETA=1`, and span names and attributes may change between releases while tracing is in beta ([Agent SDK observability](https://code.claude.com/docs/en/agent-sdk/observability)). Configuration travels as environment variables, and the two SDKs differ in one important way: in Python, `env` is merged on top of the inherited environment; in TypeScript, `env` replaces it entirely, so the TypeScript example spreads `process.env` first. Both examples are trimmed from the same page:

=== "Python"

    ```python
    from claude_agent_sdk import query, ClaudeAgentOptions

    OTEL_ENV = {
        "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
        # Required for traces, which are in beta. Metrics and log events do not need this.
        "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
        "OTEL_TRACES_EXPORTER": "otlp",
        "OTEL_METRICS_EXPORTER": "otlp",
        "OTEL_LOGS_EXPORTER": "otlp",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector.example.com:4318",
        "OTEL_EXPORTER_OTLP_HEADERS": "Authorization=Bearer your-token",
    }

    options = ClaudeAgentOptions(env=OTEL_ENV)
    ```

=== "TypeScript"

    ```typescript
    import { query } from "@anthropic-ai/claude-agent-sdk";

    const otelEnv = {
      CLAUDE_CODE_ENABLE_TELEMETRY: "1",
      // Required for traces, which are in beta. Metrics and log events do not need this.
      CLAUDE_CODE_ENHANCED_TELEMETRY_BETA: "1",
      OTEL_TRACES_EXPORTER: "otlp",
      OTEL_METRICS_EXPORTER: "otlp",
      OTEL_LOGS_EXPORTER: "otlp",
      OTEL_EXPORTER_OTLP_PROTOCOL: "http/protobuf",
      OTEL_EXPORTER_OTLP_ENDPOINT: "http://collector.example.com:4318",
      OTEL_EXPORTER_OTLP_HEADERS: "Authorization=Bearer your-token",
    };

    for await (const message of query({
      prompt: "List the files in this directory",
      // env replaces the inherited environment in TypeScript, so spread
      // process.env first to keep PATH, ANTHROPIC_API_KEY, and other variables.
      options: { env: { ...process.env, ...otelEnv } },
    })) {
      console.log(message);
    }
    ```

Three operating facts from the same page (as of September 2026):

- **Export failures are silent by default.** If the collector is unreachable or rejects the data, the agent runs normally and the CLI drops the telemetry; set `CLAUDE_CODE_OTEL_DIAG_STDERR=1` (Claude Code v2.1.179 or later) and read the diagnostics through the SDK's `stderr` callback (Python) or `stderr` option (TypeScript) to surface exporter errors.
- **Do not use the `console` exporter through the SDK.** It writes to standard output, which the SDK uses as its message channel; point `OTEL_EXPORTER_OTLP_ENDPOINT` at a local collector instead.
- **Short-lived runs can lose spans.** The CLI batches telemetry and flushes on a clean exit within a short timeout; a killed process loses whatever is still buffered. By default metrics export every 60 seconds and traces and logs every 5 seconds; lowering the export intervals (`OTEL_METRIC_EXPORT_INTERVAL`, `OTEL_LOGS_EXPORT_INTERVAL`, `OTEL_TRACES_EXPORT_INTERVAL`) narrows both windows.

What the spans and events mean when you read a trace (`claude_code.llm_request`, `claude_code.api_error`, `claude_code.api_refusal` and the rest) is in [Debugging: model or integration](#debugging-model-or-integration). Anthropic's Research system used full production tracing to diagnose failures while monitoring decision patterns and interaction structures, not conversation contents ([multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)); the SDK's defaults match that, since telemetry is structural by default: prompt text, tool inputs and tool output content are recorded only if you opt in with `OTEL_LOG_USER_PROMPTS`, `OTEL_LOG_TOOL_DETAILS` and `OTEL_LOG_TOOL_CONTENT`. Choosing monitoring for a whole estate (analytics APIs, compliance records) is in [Monitoring, usage and cost](claude-code-workflows.md#monitoring-usage-and-cost).

### Catch the failure that raises no error

Some failures never produce an exception. A refusal is an HTTP 200 (above), and a quality regression can arrive as ordinary successful responses too. Anthropic's September 2025 postmortem is the reference case (taught in [Evals across the product's life](#evals-across-the-products-life)): its evaluations missed an intermittent degradation, and the fix included running quality evaluations "continuously on true production systems" ([A postmortem of three recent issues](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues)).

Two habits turn that into routine:

- **Every incident becomes a test.** Anthropic's AI-native SDLC playbook: "Each production incident gets an eval, written by the team that owned the incident, and stays in the suite as a regression test." ([The AI-native SDLC playbook: Continuous evals in CI](https://academy.claude.com/courses/ai-native-sdlc-playbook/continuous-evals-in-ci))
- **Layers, not one check.** Anthropic's evals guidance invokes the Swiss Cheese Model: with automated evals, production monitoring, A/B tests, user feedback and sampled transcript reading combined, "failures that slip through one layer are caught by another" ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents); the layers are laid out in [Evals across the product's life](#evals-across-the-products-life)).

### Decide

- If a failure is transient, let one layer retry it with backoff that honors `retry-after`; not a second loop around the SDK's, and not retries for a malformed request (400 `invalid_request_error`) or a spend cap.
- If a subagent hits a transient failure, recover locally and escalate only what cannot be resolved, with partial results and what was attempted; not a silent success and not an abort of the whole workflow.
- If a response is refused, retry on a different model through fallback and count refusals separately; not a resend to the same model, and not an error dashboard that cannot see HTTP 200s.
- If the primary model is overloaded, switch with an availability fallback; not the refusal fallback, which never fires on overloads.
- If a dependency keeps failing, trip a circuit breaker with a fail direction chosen in advance; not unbounded retries against it.
- If a tool writes, make it safe to call twice; not an assumption that retries will not happen.
- If an agent runs unattended, set turn and budget caps and branch on how the run ended; not the cap as the normal way to stop.
- If quality could drop without errors, run evals continuously on production traffic and read sampled transcripts; not uptime alone.

### Traps

- **Assuming Claude Code, the SDKs and Code Review share one retry default.** They are 10, 2 and none.
- **Confusing the fallbacks.** Claude Code's availability chain never switches on a rate limit; the API's server-side fallback never fires on an overload; Claude Code's automatic model fallback reacts to classifier flags, not outages.
- **Believing silence.** Telemetry export fails silently by default (set `CLAUDE_CODE_OTEL_DIAG_STDERR=1` to surface exporter errors), and refusals never appear as API errors.
- **Marking a refused batch item done** because `result.type` says `succeeded`.
- **Relying on `idempotentHint`.** It is a hint from the server, informational only in the Agent SDK.

## Upgrading models safely

*Tested in: CCDV-F D5.3 Model Selection and Tradeoffs (breaking behavior changes across model releases), D2.6 Configuration Management (model version pinning, prompt versioning), D2.2 Systems Life Cycle · CCAR-P 4.3, 4.4, 6.5 · CCAR-F: not a task statement, and "Performance benchmarking or model comparison metrics" is out of scope · CCAO-F: not listed*

A model upgrade is a new dependency whose behavior you can only confirm by measuring it. Anthropic's evals guidance describes the payoff: teams with evals can determine a new model's strengths, tune their prompts and upgrade in days, while teams without them face weeks of testing, and automated evals run "on each agent change and model upgrade as the first line of defense against quality problems" ([Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)). Model IDs, the lifecycle and the full list of parameter-level breaking changes are in [Model versions, deprecation and migration](claude-api.md#model-versions-deprecation-and-migration); this section is the process that makes a switch safe.

### Loud changes and silent changes

Only some changes announce themselves. Examples from current migration and prompting guides (as of September 2026), sorted by what catches them:

| Change | Example | Fails loudly? | What catches it |
|---|---|---|---|
| A parameter the new model rejects | Forced `tool_choice` (`any`, `tool`) returns 400 on Claude Opus 5.5; a non-default `temperature`, `top_p` or `top_k` returns 400 on Claude Sonnet 5 | Yes, HTTP 400 | Any test that sends the request |
| A changed default | Opus 5.5 defaults to `medium` effort where Opus 5 defaulted to `high`, so a request that omits `effort` runs differently | No | An effort sweep and a cost and latency re-baseline |
| A new tokenizer | Sonnet 5 produces about 30% more tokens than Sonnet 4.6 for the same text | No; budgets and `max_tokens` quietly shift | Recounting tokens; cost per task |
| A new response shape | Code reading `content[0].text` breaks when thinking blocks come first | Sometimes | Parsing by block `type`; integration tests |
| Content moved to another block type | On Opus 5.5, text written between tool calls comes back as progress-update `thinking` blocks, empty at the default display, so an interface that streamed it goes quiet | No request fails | Setting a `display` value that returns the text; tests of what users see |
| Prompt instructions tuned for the old model | Claude Opus 5 verifies its own work, so explicit verification instructions cause over-verification | No; quality or cost drifts | Your eval suite, after re-evaluating model-specific instructions |
| Broader refusal classifiers | Opus 5.5's classifiers cover a broader set of categories than Opus 5's | No; refusals are HTTP 200 | Refusal counts and fallback handling |

Only the first row is certain to fail loudly. The rest are why the docs call a good evaluation set "the most important step in the process" when upgrading or changing models, tested "with your actual prompts and data" ([Choosing the right model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model)).

### Pin everything the behavior depends on

You cannot compare two versions if either one can move underneath you.

| What to pin | How (as of September 2026) |
|---|---|
| The model in API code | Send a full model ID; each ID is a pinned snapshot for its lifetime (details in [Model versions, deprecation and migration](claude-api.md#model-versions-deprecation-and-migration)) |
| The model in Claude Code | Aliases (`opus`, `sonnet`) update over time; pin a full model name or use the `ANTHROPIC_DEFAULT_OPUS_MODEL`-style variables. On Amazon Bedrock, Google Cloud's Agent Platform, Microsoft Foundry or Claude Platform on AWS, pin before rollout so you control when users move |
| A Managed Agents agent | Each config-changing update increments `version`; pin sessions to a version to stage rollouts; a coordinator's roster stays pinned to the versions resolved when it was created or updated |
| Prompts, CLAUDE.md and skills | Keep them in version control and review changes like code; gate configuration changes on eval results; promote a new skill version only when the full evaluation suite passes (prompt versioning is taught in [Prompt versioning and iteration](prompt-engineering.md#prompt-versioning-and-iteration)) |
| The eval harness | Pin both models so scores stay comparable over time; the [plugin evals docs](https://code.claude.com/docs/en/plugin-evals) pin the agent model "so a model rollout isn't mistaken for a plugin regression", and the judge model otherwise defaults to "A small fast model" |
| The Agent SDK | Its bundled CLI is pinned to the SDK version; the SDK follows semver, so take patches continuously and read the changelog before a minor release |

Pinning a Managed Agents session to agent version 1, from the [sessions docs](https://platform.claude.com/docs/en/managed-agents/sessions) (passing only the agent ID as a string uses the latest version):

```python
pinned_session = client.beta.sessions.create(
    agent={"type": "agent", "id": agent.id, "version": 1},
    environment_id=environment.id,
)
```

The docs' CI invocation for `claude plugin eval`, with both models pinned, a pass threshold and a cost ceiling, and the exit codes to fail a build on are in [Built-in evaluators (as of September 2026)](#built-in-evaluators-as-of-september-2026).

### Run the upgrade as an experiment

1. **Baseline the current model.** Run your regression suite (which should pass nearly 100%) and your capability suite, and record latency, token usage, cost per task and error rates. [Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence) adds: "compare models on cost per completed task", not per token.
2. **Migrate on a branch.** In Claude Code, `/claude-api migrate this project to claude-opus-5-5` asks you to confirm the scope, applies the model ID swap and, as needed, breaking parameter changes, prefill replacement and effort calibration, "then produces a checklist of items to verify manually" ([Opus 5.5 migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide)). The same guide notes that Claude Managed Agents users need no changes beyond updating the model name; its breaking changes apply to Messages API code.
3. **Re-tune before you judge.** The Opus 5.5 migration guide's first recommendations are to re-run your effort sweep and to re-evaluate model-specific prompt instructions. [Choosing the right model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model): "Tuning effort is often a better lever than switching models." [Anthropic's prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) adds that a technique measured on one model should be re-checked "against your own evals before applying it to another."
4. **Compare fairly.** Same tasks, several trials each, because outputs vary between runs. For customer-facing agents look at `pass^k` (all k trials succeed), not only `pass@k`. Because both models answer the same questions, use a paired-differences comparison, and size the eval so the difference you care about can show up at all ([Is the difference real?](#is-the-difference-real)). Read transcripts before trusting a score change ([Running agent evals that you can trust](#running-agent-evals-that-you-can-trust)).
5. **Stage the rollout.** Test in a development environment "before switching production traffic" ([Opus 5.5 migration guide](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide)), then "Run the winner in shadow on a traffic slice before cutover" ([Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence)). An A/B test validates significant changes once you have enough traffic, though it takes days or weeks to reach significance ([Evals across the product's life](#evals-across-the-products-life)). For long-running stateful agents, Anthropic's Research system uses rainbow deployments when it ships updates, shifting traffic gradually while old and new versions both run ([multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)).
6. **Re-baseline cost and latency** at the effort level you chose, and recount tokens if the tokenizer changed.
7. **Keep the way back open.** Anthropic's rollback guidance for Skills applies here too: "Maintain the previous version as a fallback" ([Skills for enterprise](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise)). Keep the suite running after cutover.

Do it early. Anthropic gives at least 60 days' notice before retiring a publicly released model, says to test with newer models "well before the retirement date of your current model", and warns that "Deprecated models are likely to be less reliable than active models." ([model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations)) Anthropic's AI-native SDLC playbook treats the swap as a routine event for the eval suite: "When a new model is swapped in or a prompt is rewritten, the eval suite says whether the agent still does the work to the same standard." ([The AI-native SDLC playbook: Continuous evals in CI](https://academy.claude.com/courses/ai-native-sdlc-playbook/continuous-evals-in-ci))

### Reading a result that got worse

CCAR-P 4.4 lists "model mismatch" among the system issues to diagnose ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). Before concluding that the new model is worse:

- **Check the harness.** A 0% pass rate across many trials most often means a broken task, and a failing case can be a grader, or in plugin evals a small judge model, rejecting a valid answer (see [Building the test set](#building-the-test-set) and [Built-in evaluators (as of September 2026)](#built-in-evaluators-as-of-september-2026)).
- **Check what else changed.** If behavior shifts on an unchanged model ID, the [model IDs page](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions) names an infrastructure update as the most likely cause, since the model behind an ID stays constant.
- **Check the settings before the model.** A changed default effort can explain a quality or cost change on its own; Anthropic's cost guidance also notes that "A stronger model at low effort can be cheaper than a weaker model working hard (high effort)." ([Reducing cost and improving performance with Claude Platform](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform))
- **Then decide what kind of fix it needs,** using the three diagnoses in [Prompt failure, hallucination or model mismatch](#prompt-failure-hallucination-or-model-mismatch).

### How the exams frame it

- **CCDV-F** tests "breaking behavior changes across model releases when selecting models for tasks" (D5.3) and names "model version pinning, prompt versioning" in D2.6 ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). Our reading of those lines: expect a request that worked on one model and fails on its successor, or behavior that shifted after a switch.
- **CCAR-P** asks you to "Conduct A/B testing and iterative improvements" (4.3) and to support lifecycle phases through "monitoring, iteration" (6.5) ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)); its preparation course asks candidates to plan an A/B test and read the result "without overclaiming" ([Enterprise Integration & Production](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/enterprise-integration-production)).
- **CCAR-F** places "Performance benchmarking or model comparison metrics" out of scope ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

!!! warning "Exam guide vs current docs"

    The July 2026 [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) lists "extended thinking, adaptive thinking, effort levels" as model options (D5.1) and "adaptive thinking support" as a selection criterion (D5.3). As of September 2026, the [API errors docs](https://platform.claude.com/docs/en/api/errors) say "Claude 4.7 and later models have removed extended thinking", and on Claude Opus 5.5 effort is the only thinking control. Adaptive thinking support still separates models: Claude Haiku 4.5 (like Sonnet 4.5, Opus 4.5 and earlier Claude 4 models) supports only extended thinking.

    The [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) "model version pinning" needs one precision today. On the API, every model ID, dateless 4.6-generation IDs included, is a pinned snapshot. Only the pre-4.6 convenience aliases such as `claude-sonnet-4-5` fall outside that guarantee: each resolves to the most recent dated snapshot for its minor version ([model IDs and versions](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions)). In Claude Code, the `opus` and `sonnet` aliases update over time, so pinning there means a full model name or a variable such as `ANTHROPIC_DEFAULT_OPUS_MODEL`. On the exam, answer in the guide's terms: the guide (effective July 2026) says its skills are what exam items are written against.

### Decide

- If a new model ships, run your suites on it with re-tuned effort and prompts before any production traffic; not a straight ID swap.
- If a switch must be controllable, pin the model, the agent version, the prompts and the judge; not aliases or a "latest" pointer.
- If the change is significant and traffic allows, confirm it with a shadow run or A/B test; not offline scores alone.
- If a score drops, check the harness, the settings and the infrastructure before blaming the model.

### Traps

- **Upgrading on the retirement deadline** instead of well before it.
- **Judging the new model with the old settings,** such as an unset effort that now defaults lower.
- **Letting the judge model float** while comparing agent models, which turns a judge change into an apparent regression.
- **Trusting the absence of errors.** Only rejected parameters are certain to fail loudly; defaults, tokenizers, where content lands and prompt fit change silently.
- **Assuming the weights changed** when a stable model ID starts behaving differently.

## Exam map

*Tested in: all four exams, section by section (CCAO-F, CCDV-F, CCAR-F, CCAR-P)*

Which official objectives each section of this page serves, taken from the four July 2026 exam guides. How to read the labels: CCAO-F objectives are numbered D1.1 to D7.3 in the order the guide lists them; CCDV-F skills such as D4.1 are the skill's position within its domain; CCAR-F codes such as 5.3-K1 mean task statement 5.3, first "Knowledge of" bullet (S marks a "Skills in" bullet); CCAR-P numbers such as 4.2 are the objective's position within its domain. Apart from CCAR-F's task statement numbers (such as 5.3), the guides print these objectives and bullets without numbers.

| Section | CCAO-F | CCDV-F | CCAR-F | CCAR-P |
|---|---|---|---|---|
| [Success criteria and test sets](#success-criteria-and-test-sets) | D2.1 | Domain 4 title; purpose area 5 ("Designing and running evals"); How to Prepare step 3 | 5.5-K1, 5.5-K4, 5.5-S2; How to Prepare item 3 | 4.1, 4.2, 4.3 |
| [Grading methods](#grading-methods) | Not listed | Domain 4 title; purpose area 5 | 4.6-K1, 4.6-K2, 4.6-S1; 5.5-S3; sample question 12 (option D rationale) | 4.2 |
| [Evaluating agents](#evaluating-agents) | Not listed | D4.1 (trace analysis to identify failure modes); purpose area 5 | How to Prepare item 3; Exercise 1 steps 3 and 5; Exercise 4 steps 4 and 5 | 4.2, 4.3, 4.6 |
| [Debugging: model or integration](#debugging-model-or-integration) | D7.1, D7.2 | D4.1; D6.3 | 1.2-K4 and sample question 7; Exercise 4 | 4.4, 7.3; sample question 3 |
| [Error propagation in multi-agent systems](#error-propagation-in-multi-agent-systems) | Not listed | D4.1 (recovery strategy selection); D8.1 (error handling) | 5.3-K1 to 5.3-S4; 2.2-S3, 2.2-S4; 1.2-S4; in-scope topic 7; Exercise 4 step 4; sample question 8 | 1.4, 5.2 |
| [Escalation and ambiguity](#escalation-and-ambiguity) | Purpose bullet 5; How to Prepare item 5 | D8.1 (approval patterns) | 5.2-K1 to 5.2-S5; 1.4-K3, 1.4-S3; 1.5-S2; in-scope topic 8; How to Prepare item 7; Exercise 1 step 4; Scenario 1; sample question 3 | 5.1, 5.3 |
| [Human review and confidence calibration](#human-review-and-confidence-calibration) | D2.2, D2.3, D2.4; sample question 1 | D6.3 | 5.5-K1 to 5.5-S4; 4.6-S3; 5.2-K3; technology list item 14; in-scope topic 17; Exercise 3 step 5; How to Prepare item 7 | 4.2, 5.3 |
| [Reliability engineering for Claude applications](#reliability-engineering-for-claude-applications) | Not listed | D4.1; D2.2; D5.2; purpose areas 2 ("error handling") and 5 ("monitoring production quality") | Audience bullet 7; 1.1-S3; 2.2-S3; 5.4-K4, 5.4-S4; rate limiting and quotas out of scope (out-of-scope item 11) | 3.4, 4.6, 5.2, 7.3 |
| [Upgrading models safely](#upgrading-models-safely) | Not listed | D5.3; D2.6; D2.2 | Not listed; model comparison metrics out of scope (out-of-scope item 14) | 4.3, 4.4, 6.5 |

The objectives and other guide items referenced above, by exam:

| Exam | Label | Wording (as printed in the guide) |
|---|---|---|
| CCAO-F | D2.1 | Evaluate Claude-generated outputs for accuracy and completeness |
| CCAO-F | D2.2 | Identify hallucinations, inconsistencies, and biases in responses |
| CCAO-F | D2.3 | Apply fact-checking and validation techniques |
| CCAO-F | D2.4 | Determine when human review or additional verification is required |
| CCAO-F | D7.1 | Identify, diagnose, and resolve issues with underperforming prompts or poor outputs |
| CCAO-F | D7.2 | Adjust approach based on feedback and results |
| CCAO-F | Purpose bullet 5 | Recognize limitations and escalate more complex or technical implementations |
| CCAO-F | How to Prepare item 5 | Practice responsible-use judgment: data sensitivity, appropriate use cases, and when to escalate or seek human review |
| CCDV-F | D2.2 | Systems Life Cycle (2.8%) |
| CCDV-F | D2.6 | Configuration Management (4.1%) |
| CCDV-F | D4.1 | Debugging and Error Handling (2.6%) |
| CCDV-F | D5.2 | Technical Fundamentals (6.1%) |
| CCDV-F | D5.3 | Model Selection and Tradeoffs (2.7%) |
| CCDV-F | D6.3 | Output Handling (2.6%) |
| CCDV-F | D8.1 | Tool Implementation (4.4%) |
| CCDV-F | Purpose area 2 | Integrating Claude into application code through the API, client SDKs, and third-party integrations, including streaming, error handling, and multi-format input |
| CCDV-F | Purpose area 5 | Designing and running evals, debugging failure modes through trace analysis, validating structured output, and monitoring production quality |
| CCDV-F | How to Prepare step 3 | Build and operate at least one Claude application that exercises the API, integrates one or more tools, applies basic prompt and context engineering, and includes simple security and evaluation practices |
| CCAR-F | Audience bullet 7 | Making sound escalation and reliability decisions, including error handling, human-in-the-loop workflows, and self-evaluation patterns |
| CCAR-F | How to Prepare item 3 | Design and test MCP tools: write tool descriptions that clearly differentiate similar tools. Implement structured error responses with error categories and retryable flags. Test tool selection reliability with ambiguous requests. |
| CCAR-F | How to Prepare item 7 | Review escalation and human-in-the-loop patterns: understand when to escalate (policy gaps, customer requests, inability to progress) versus resolve autonomously. Practice designing human review workflows with confidence-based routing. |
| CCAR-F | In-scope topic 7 | Error handling and propagation: structured error responses, transient vs business vs permission errors, local recovery before escalation |
| CCAR-F | In-scope topic 8 | Escalation decision-making: explicit criteria, honoring customer preferences, policy gap identification |
| CCAR-F | In-scope topic 17 | Human review workflows: confidence calibration, stratified sampling, accuracy segmentation by document type and field |
| CCAR-F | 1.1 | Design and implement agentic loops for autonomous task execution |
| CCAR-F | 1.2 | Orchestrate multi-agent systems with coordinator-subagent patterns |
| CCAR-F | 1.4 | Implement multi-step workflows with enforcement and handoff patterns |
| CCAR-F | 1.5 | Apply Agent SDK hooks for tool call interception and data normalization |
| CCAR-F | 2.2 | Implement structured error responses for MCP tools |
| CCAR-F | 4.6 | Design multi-instance and multi-pass review architectures |
| CCAR-F | 5.2 | Design effective escalation and ambiguity resolution patterns |
| CCAR-F | 5.3 | Implement error propagation strategies across multi-agent systems |
| CCAR-F | 5.4 | Manage context effectively in large codebase exploration |
| CCAR-F | 5.5 | Design human review workflows and confidence calibration |
| CCAR-P | 1.4 | Design multi-agent systems and orchestration strategies |
| CCAR-P | 3.4 | Analyze observability challenges and select monitoring strategies at scale |
| CCAR-P | 4.1 | Define evaluation metrics (accuracy, latency, cost, safety, security) |
| CCAR-P | 4.2 | Design evaluation datasets and test frameworks using mixed methodologies |
| CCAR-P | 4.3 | Conduct A/B testing and iterative improvements |
| CCAR-P | 4.4 | Diagnose system issues (prompt failure, hallucinations, model mismatch) |
| CCAR-P | 4.6 | Monitor system performance using logging and observability tools |
| CCAR-P | 5.1 | Implement guardrails and safety controls |
| CCAR-P | 5.2 | Identify risks, limitations, and failure modes of LLM systems |
| CCAR-P | 5.3 | Apply human-in-the-loop validation strategies |
| CCAR-P | 6.5 | Support lifecycle phases (discovery, design, handoff, monitoring, iteration) |
| CCAR-P | 7.3 | Support debugging and operational issue resolution |

Notes on the cells:

- **CCAO-F.** On the Associate exam, Domain 2, Output Evaluation and Validation, carries 21%, and Domain 7, Troubleshooting and Optimization, carries 10%; both draw on the sections with Associate objectives above. The [Associate guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) says the certification "is not intended for software developers who build against APIs or design agentic systems", so the build-side sections (grading pipelines, agent evals, error propagation, reliability engineering, model upgrades) have no Associate objective.
- **CCDV-F.** Domain 4 is titled "Eval, Testing, and Debugging" and weighted 2.6%, but its only skill, D4.1, describes debugging and error handling. Eval design appears instead in purpose area 5 and How to Prepare step 3 of the [CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) (both in the table above). That is why the first two rows cite the domain title with purpose area 5 (the first also cites How to Prepare step 3), and the third cites D4.1 for its trace analysis alongside purpose area 5.
- **CCAR-F.** Domain 5, Context Management & Reliability, carries 15% and holds task statements 5.2 to 5.5 cited above; 4.6 sits in Domain 4 (20%). No task statement covers eval design or model upgrades as such, and the [CCAR-F guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) out-of-scope list names "Rate limiting, quotas, or API pricing calculations" (item 11) and "Performance benchmarking or model comparison metrics" (item 14).
- **CCAR-P.** Every section maps to at least one Professional objective. Domain 4, Evaluation, Testing & Optimization, carries 16% and appears in every row except Error propagation and Escalation; Domain 5, Governance, Safety & Risk Management (14%), adds guardrails, failure modes and human-in-the-loop validation (5.1 to 5.3).

??? info "Sources"

    - [Claude Certified Architect, Professional exam guide (PDF, July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf): objectives 1.4, 3.4, 4.1 to 4.4, 4.6, 5.1 to 5.3, 6.5 and 7.3, domain weights, sample question 3 and its rationale, How to Prepare
    - [Claude Certified Developer, Foundations exam guide (PDF, July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf): skills D2.2, D2.6, D4.1, D5.1, D5.2, D5.3, D6.3 and D8.1, the Domain 4 weight, purpose areas 2 and 5, and How to Prepare step 3
    - [Claude Certified Architect, Foundations exam guide (PDF, July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): task statements 1.1 to 1.5, 2.2, 4.4, 4.6 and 5.2 to 5.5, sample questions 1, 3, 7, 8 and 12, Scenario 1, Exercises 1, 3 and 4, How to Prepare, the intended audience, the appendix lists and the "Task tool" wording
    - [Claude Certified Associate, Foundations exam guide (PDF, July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf): objectives D2.1 to D2.4, D7.1 and D7.2, sample question 1, the escalation purpose bullet, How to Prepare item 5 and the audience exclusion
    - [Define success criteria and build evaluations](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests): criteria properties, common criteria, design principles, example evals, grading methods, LLM grader tips and code
    - [Prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview): criteria and tests before prompting; not every failing eval is a prompt problem
    - [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices): the golden rule for unclear prompts, explicit instructions, explaining motivation, examples and their number, self-checks, reversibility of actions, re-checking model-specific techniques against your own evals
    - [Prompting Claude Opus 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5): Opus 5 verifying its own work and over-verification from explicit verification instructions
    - [Stop reasons and fallback](https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons): stop reasons versus errors, refusal as HTTP 200, truncation handling
    - [Claude API errors](https://platform.claude.com/docs/en/api/errors): error types and shape, request IDs, mid-stream errors, typed exceptions, spend-cap 429s, SDK automatic retries, forced `tool_choice` rejection, extended thinking removed on Claude 4.7 and later
    - [Rate limits](https://platform.claude.com/docs/en/api/rate-limits): `retry-after`, spend-cap 429 with `enforced_spend_limit_reached`
    - [Python SDK](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python): exception classes, retried status codes, `max_retries`, the 10-minute default timeout, `_request_id`, `ANTHROPIC_LOG`
    - [TypeScript SDK](https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/typescript): `APIError` handling and status-to-class table, default retries and the timeout scaling up to 60 minutes
    - [Refusals and fallback](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback): refusals as HTTP 200 invisible to error-rate monitoring, retry on another model, server-side fallback and its code example, batch refusals, fallback on every request path
    - [Troubleshooting tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/troubleshooting-tool-use): `tool_result` ordering errors, edited thinking blocks, no raw string matching on tool input
    - [Handle tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls): `is_error`, instructive error messages, Claude's 2 to 3 corrective retries, invalid calls signaling missing information
    - [Migrating to Claude Sonnet 5](https://platform.claude.com/docs/en/models/sonnet-5/migration-guide): the new tokenizer (about 30% more tokens) and selecting content blocks by `type`, not position
    - [Migrating to Claude Opus 5.5](https://platform.claude.com/docs/en/models/opus-5-5/migration-guide): breaking changes, the `medium` effort default, `/claude-api migrate`, effort sweep, broader refusal classifiers, testing before switching production traffic, re-baselining cost and latency
    - [Model IDs and versioning](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions): model IDs as pinned snapshots, pre-4.6 aliases, infrastructure updates as the likely cause of behavior changes on a stable ID
    - [Choosing the right model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model): tuning effort before switching models, an evaluation set as the most important step in an upgrade, testing with actual prompts and data
    - [Reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations): "I don't know", quote extraction, citations, best-of-N inconsistency
    - [Reducing latency](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-latency): baseline latency and time to first token
    - [Mitigate jailbreaks and prompt injections](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/mitigate-jailbreaks): red-teaming a workflow with injected documents, emails and tool outputs
    - [Customer support agent (use-case guide)](https://platform.claude.com/docs/en/about-claude/use-case-guides/customer-support-chat): support metrics and targets, including escalation accuracy and deflection; sentiment as a business metric; no unauthorized commitments
    - [Ticket routing (use-case guide)](https://platform.claude.com/docs/en/about-claude/use-case-guides/ticket-routing): routing metrics, segment targets, clear thresholds; emotion over the underlying problem, and directing when to prioritize sentiment
    - [Legal summarization (use-case guide)](https://platform.claude.com/docs/en/about-claude/use-case-guides/legal-summarization): summary metrics and expert review before production
    - [Content moderation (use-case guide)](https://platform.claude.com/docs/en/about-claude/use-case-guides/content-moderation): precision and recall tracking in production; risk tiers with medium-risk cases flagged for human review
    - [Define outcomes (Claude Managed Agents)](https://platform.claude.com/docs/en/managed-agents/define-outcomes): outcome rubrics, grader in a separate context window, `max_iterations` (default 3, maximum 20), evaluation results including `max_iterations_reached`
    - [Test plugins with evals](https://code.claude.com/docs/en/plugin-evals): `claude plugin eval`, grader types, baseline and `Δ`, the pinned CI command and exit codes, suspecting the judge before the plugin
    - [Keep Claude working toward a goal (Claude Code)](https://code.claude.com/docs/en/goal): completion conditions checked by a fresh model; turn or time bounds in a `/goal` condition
    - [Hooks reference (Claude Code)](https://code.claude.com/docs/en/hooks): `/goal` as a session-scoped prompt-based Stop hook; `defer` for a single tool call, with no timeout and the 30-day `cleanupPeriodDays` sweep
    - [Catch security issues as Claude writes code (Claude Code)](https://code.claude.com/docs/en/security-guidance): reviews run in a separate call, not by the instance that wrote the code
    - [Best practices for Claude Code](https://code.claude.com/docs/en/best-practices): reset after two failed corrections
    - [Debug your configuration (Claude Code)](https://code.claude.com/docs/en/debug-your-config): `/context`, `--safe-mode`, guidance versus enforcement
    - [Common workflows (Claude Code)](https://code.claude.com/docs/en/common-workflows): giving Claude reproduction steps and stack traces
    - [Run Claude Code programmatically](https://code.claude.com/docs/en/headless): `system/api_retry` events, `mcp_server_errors` and `plugin_errors` in the `system/init` event, exit codes
    - [Monitoring (Claude Code)](https://code.claude.com/docs/en/monitoring-usage): `api_error` after retries are exhausted, `api_refusal` events, recovered versus stalled sessions
    - [Observability with OpenTelemetry (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/observability): OpenTelemetry spans, telemetry switches and configuration code, content opt-ins, silent export failures, the `console` exporter warning, flush behavior, spans nesting into one trace, Agent tool wording
    - [How the agent loop works (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/agent-loop): `max_turns`, `max_budget_usd`, result subtypes and the fields carried after errors, `query()` raising after an error result
    - [Troubleshoot the Agent SDK](https://code.claude.com/docs/en/agent-sdk/troubleshooting): `success` without `structured_output`, `ResultError` and `ProcessError`
    - [Demystifying evals for AI agents (Anthropic Engineering, January 2026)](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents): eval vocabulary, grader types, capability and regression evals, one-sided evals, `pass@k` and `pass^k`, reading transcripts, agent-type graders, lifecycle methods and the Swiss Cheese Model, weekly transcript sampling, evals on each model upgrade, production monitoring for drift
    - [How we built our multi-agent research system (Anthropic Engineering)](https://www.anthropic.com/engineering/multi-agent-research-system): small starting sets, single-call LLM judge, human testing, end-state evaluation, compounding errors, resuming from failure, retry logic and checkpoints, telling agents when tools fail, production tracing, rainbow deployments
    - [Writing effective tools for AI agents (Anthropic Engineering)](https://www.anthropic.com/engineering/writing-tools-for-agents): realistic tool evals, strict verifiers, held-out test sets, tool metrics
    - [A postmortem of three recent issues (Anthropic Engineering, September 2025)](https://www.anthropic.com/engineering/a-postmortem-of-three-recent-issues): evals that missed intermittent degradation, continuous production evaluation, quality not reduced under load
    - [A statistical approach to model evaluations (Anthropic Research)](https://www.anthropic.com/research/statistical-approach-to-model-evals): SEM, confidence intervals, clustering, resampling, paired differences, power analysis
    - [Mitigating prompt injections in browser use (Anthropic Research)](https://www.anthropic.com/research/prompt-injection-defenses): a 1% attack success rate still represents meaningful risk
    - [Piloting Claude in Chrome (Claude blog, August 2025)](https://claude.com/blog/claude-for-chrome): attack success rate across 123 test cases in 29 scenarios
    - [Building agents with the Claude Agent SDK (Claude blog)](https://claude.com/blog/building-agents-with-the-claude-agent-sdk): rules-based feedback first; limits of LLM-as-judge inside the loop
    - [Evaluate prompts in the developer console (Claude blog, July 2024)](https://claude.com/blog/evaluate-prompts): the Console Evaluate feature as described in 2024
    - [Reducing cost and improving performance with Claude Platform (Claude blog)](https://claude.com/blog/reducing-cost-and-improving-performance-with-claude-platform): `/claude-api hillclimb` train and test split; a stronger model at low effort can cost less than a weaker model at high effort
    - [Building Evals (Claude cookbook)](https://github.com/anthropics/claude-cookbooks/blob/main/misc/building_evals.ipynb): the four parts of an eval, code-based grading preference, multiple-choice reformatting, model-based grading
    - [Retrieval Augmented Generation (Claude cookbook)](https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/retrieval_augmented_generation/guide.ipynb): evaluating retrieval separately from end-to-end accuracy; precision, recall, F1, MRR
    - [Text to SQL with Claude (Claude cookbook)](https://github.com/anthropics/claude-cookbooks/blob/main/capabilities/text_to_sql/guide.ipynb): grading generated SQL against a test database
    - [Claude 101: Your first conversation with Claude (Claude Academy)](https://academy.claude.com/courses/claude-101/your-first-conversation-with-claude): restarting in a new chat to refresh context
    - [AI capabilities and limitations: When properties collide (Claude Academy)](https://academy.claude.com/courses/ai-capabilities-and-limitations/when-properties-collide): failures as two properties interacting
    - [Validating skills for plugins (Claude Academy)](https://academy.claude.com/courses/introduction-to-claude-cowork/validating-skills-for-plugins): change one thing at a time when iterating
    - [Enterprise Integration & Production (Anthropic Partner Academy, CCAR-P path)](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional/enterprise-integration-production): reliability patterns (retries, fallbacks, circuit breakers) and reading an A/B test without overclaiming
    - [Model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations): testing newer models well before retirement, deprecated models being less reliable
    - [Optimizing for cost and intelligence](https://platform.claude.com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence): cost per completed task, shadow run on a traffic slice before cutover
    - [Streaming messages](https://platform.claude.com/docs/en/build-with-claude/streaming): tool use blocks cannot be partially recovered when resuming a stream
    - [Search results](https://platform.claude.com/docs/en/build-with-claude/search-results): returning a plain text block when a search fails or returns nothing
    - [Citations](https://platform.claude.com/docs/en/build-with-claude/citations): citations guaranteed to point into the provided documents
    - [Skills for enterprise](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/enterprise): promoting skill versions only when the evaluation suite passes, keeping the previous version as a fallback
    - [Prompting Claude Fable 5.1](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-fable-5-1): implementing the best-supported reading of an ambiguous coding task and stating the assumption
    - [Commerce agent](https://platform.claude.com/docs/en/about-claude/use-case-guides/commerce-agents): staged changes approved outside the conversation, guardrails checked at staging and apply, product IDs only from tool results
    - [Multiagent orchestration (Managed Agents)](https://platform.claude.com/docs/en/managed-agents/multiagent-orchestration): failed consultations not failing the turn, interrupted child threads, pinned rosters, "Escalation" as consulting a more capable model
    - [Permission policies (Managed Agents)](https://platform.claude.com/docs/en/managed-agents/permission-policies): `always_ask`, `requires_action`, `user.tool_confirmation`
    - [Start a session (Managed Agents)](https://platform.claude.com/docs/en/managed-agents/sessions): pinning a session to an agent version
    - [Subagents in the SDK](https://code.claude.com/docs/en/agent-sdk/subagents): what crosses the parent-subagent boundary, API errors never delivered as results, partial output at `maxTurns`, budget limit, Task and Agent naming
    - [Handle approvals and user input](https://code.claude.com/docs/en/agent-sdk/user-input): `canUseTool`, `defer`, `AskUserQuestion` limits and its absence in subagents, external approval systems
    - [Give Claude custom tools](https://code.claude.com/docs/en/agent-sdk/custom-tools): `is_error` on failed calls, `idempotentHint` informational only
    - [Hosting the Agent SDK](https://code.claude.com/docs/en/agent-sdk/hosting): `mirror_error` messages, SDK semver and bundled CLI pinning
    - [Create custom subagents](https://code.claude.com/docs/en/sub-agents): subagent failure reporting from v2.1.199, partial output, background failures, fallback for subagents, the v2.1.63 rename
    - [Error reference (Claude Code)](https://code.claude.com/docs/en/errors): `Agent terminated early due to an API error`, 10 retries, `CLAUDE_CODE_RETRY_WATCHDOG`, no re-run after completed tool calls
    - [Orchestrate subagents at scale with dynamic workflows](https://code.claude.com/docs/en/workflows): `agent()` resolving to `null`, resumable runs, `/deep-research` unverified claims
    - [Tools reference](https://code.claude.com/docs/en/tools-reference): the pre-v2.1.208 `No files found` behavior
    - [Model configuration](https://code.claude.com/docs/en/model-config): fallback model chains, aliases, pinning before rollout
    - [Choose a permission mode](https://code.claude.com/docs/en/permission-modes): auto mode block thresholds, stated boundaries, `permissions.ask` checkpoints
    - [Environment variables (Claude Code)](https://code.claude.com/docs/en/env-vars): `MAX_STRUCTURED_OUTPUT_RETRIES` default of 5 attempts
    - [Code Review](https://code.claude.com/docs/en/code-review): failed review runs do not retry on their own
    - [MCP specification: Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools): a human in the loop able to deny tool invocations
    - [MCP specification: Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources): no empty `contents` array for a non-existent resource
    - [MCP specification: Schema Reference](https://modelcontextprotocol.io/specification/2026-07-28/schema): `idempotentHint` semantics and annotations as untrusted hints
    - [Building Effective AI Agents](https://www.anthropic.com/engineering/building-effective-agents): compounding errors from autonomy, human review alongside automated tests
    - [Emergent introspective awareness in LLMs](https://www.anthropic.com/research/introspection): introspection as unreliable and limited in scope, about 20% for Claude Opus 4.1
    - [Reasoning models don't always say what they think](https://www.anthropic.com/research/reasoning-models-dont-say-think): Claude 3.7 Sonnet mentioning a used hint 25% of the time
    - [Language models (mostly) know what they know](https://www.anthropic.com/research/language-models-mostly-know-what-they-know): calibration on multiple choice and true/false questions, poor P(IK) calibration on new tasks
    - [Claude is providing incorrect or misleading responses. What's going on? (Claude Help Center)](https://support.claude.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on): not a singular source of truth, authoritative-looking quotes not grounded in fact
    - [AI Fluency: Key Terminology Cheat Sheet (PDF)](https://www-cdn.anthropic.com/4396730ed190e691a3712cf2fd6bfe35509deca2.pdf): Deployment Diligence
    - [Anthropic Education Report: The AI Fluency Index (Claude Academy)](https://academy.claude.com/tutorials/the-ai-fluency-index): lower scrutiny of artifact outputs (missing context, fact checks, questioning reasoning)
    - [Why do AI models hallucinate? (Claude Academy)](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate): situations with the highest hallucination risk
    - [Hand Claude Cowork your first task (Claude Academy, Introduction to Claude Cowork)](https://academy.claude.com/courses/introduction-to-claude-cowork/the-task-loop): untraceable specifics as a flag
    - [Continuous evals in CI (Claude Academy, The AI-native SDLC playbook)](https://academy.claude.com/courses/ai-native-sdlc-playbook/continuous-evals-in-ci): an eval per production incident, evals on model swaps, gating configuration changes on eval results
    - [The CLAUDE.md (Claude Academy, The AI-native SDLC playbook)](https://academy.claude.com/courses/ai-native-sdlc-playbook/claude-md): CLAUDE.md in Git, reviewed like code
    - [Requirements and design (Claude Academy, The AI-native SDLC playbook)](https://academy.claude.com/courses/ai-native-sdlc-playbook/requirements-and-design): prompts and skill versions logged in version control
    - [Gate: human-in-the-loop with custom tools (Claude cookbook)](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/managed_agents/CMA_gate_human_in_the_loop.ipynb): calibration in both directions, `decide` and `escalate` tool definitions, pausing on `agent.custom_tool_use`
    - [Research subagent prompt (Claude cookbook)](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/patterns/agents/prompts/research_subagent.md): local recovery by changing tool or query, no repeated queries, flagging issues to the lead researcher
    - [customer-escalation skill (Anthropic knowledge-work plugins)](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/customer-support/skills/customer-escalation/SKILL.md): the structured escalation brief, escalation triggers and best practices
    - [draft-response skill (Anthropic knowledge-work plugins)](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/customer-support/skills/draft-response/SKILL.md): leading with empathy, policy exceptions and leadership requests as escalation triggers
    - [customer-research skill (Anthropic knowledge-work plugins)](https://raw.githubusercontent.com/anthropics/knowledge-work-plugins/main/customer-support/skills/customer-research/SKILL.md): contradictory sources as a low-confidence signal
    - [claude-code-security-review prompts (GitHub)](https://raw.githubusercontent.com/anthropics/claude-code-security-review/main/claudecode/prompts.py): confidence bands and the 0.7 reporting threshold
    - [claude-code-action test-failure analysis example (GitHub)](https://raw.githubusercontent.com/anthropics/claude-code-action/main/examples/test-failure-analysis.yml): acting automatically only at confidence 0.7 or above
    - [Claude Certified Architect, Professional learning path (Anthropic Partner Academy)](https://anthropic-partners.skilljar.com/path/claude-certified-architect-professional): a safety stack designed to fail closed rather than open
    - [Inference hooks (Claude Enterprise)](https://platform.claude.com/docs/en/manage-claude/inference-hooks): failure handling and the circuit breaker for an unreachable AI security server
    - [Claude apps gateway spend limits](https://code.claude.com/docs/en/claude-apps-gateway-spend-limits): per-developer spend limits as a circuit breaker on a shared bill
    - [Persist sessions to external storage (Agent SDK)](https://code.claude.com/docs/en/agent-sdk/session-storage): deduplicating mirrored transcript entries by `entry.uuid`
