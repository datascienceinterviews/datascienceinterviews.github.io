---
title: "Claude Certified Associate, Foundations (CCAO-F): Free Study Guide"
description: Free study guide for the Claude Certified Associate, Foundations (CCAO-F) exam, covering the format, all 30 objectives, the official samples and a study plan.
last_reviewed: 2026-09-23
---

# Claude Certified Associate, Foundations (CCAO-F)

CCAO-F is Anthropic's proctored exam for professionals who use Claude as a productivity tool and build Claude Projects. It has 60 items in 120 minutes across 7 domains and 30 objectives, the pass mark is a scaled score of 720, and no software-development or API experience is needed. This page follows the official exam guide (Version 1.0, effective July 2026) objective by objective: what to know, the decision each objective asks you to make, and the wrong answers that look right.

!!! abstract "The contract"

    **After this page you can:**

    - State the exam's numbers (60 items, 120 minutes, a pass mark of 720 on a scaled range of 100 to 1,000, a &#36;99 fee, 12 months of validity) and know where the guide and the live program pages disagree.
    - Decide whether CCAO-F fits your role, or which of the other three Claude exams does.
    - Name the seven domains and 30 objectives, and see where the weight sits.
    - Work every objective from the facts, decision rules and traps in its domain section.
    - Reason through Anthropic's three published sample questions, including why each wrong option fails.
    - Recognize where today's Claude apps differ from the guide's July 2026 wording, and answer in the guide's terms.

    **Who it is for:** professionals who use Claude as a productivity tool and build Claude Projects, in roles such as operations, marketing, project management, education and communications, plus external consultants who support implementation, use-case identification and process redesign. The guide recommends regular, hands-on use of Claude in a professional setting. It states that no software-development or API experience is needed. Details are in [Who this exam is for](#who-this-exam-is-for).

    **Who it is not for:** the guide says the certification "is not intended for software developers who build against APIs or design agentic systems, nor for specialists in machine learning, software engineering, or advanced AI system design." If one of the rows below describes you, follow the pointer in its second column.

    | If this describes you | Look at this instead |
    |---|---|
    | You write application code against the Claude API: integration, agents, tools, MCP servers, model tiers, caching, guardrails | [Claude Certified Developer, Foundations (CCDV-F)](claude-certified-developer.md) |
    | You are a solution architect who designs and implements production applications with Claude Code, the Claude Agent SDK, the Claude API and MCP | [Claude Certified Architect, Foundations (CCAR-F)](claude-certified-architect-foundations.md) |
    | You design end-to-end AI systems and lead architectural decisions, including security, legal and executive discussions (that guide recommends 3+ years in systems architecture or platform engineering) | [Claude Certified Architect, Professional (CCAR-P)](claude-certified-architect-professional.md) |
    | Your firm needs the credential to count toward its Claude Partner Network tier | CCAO-F does not count; CCDV-F, CCAR-F and CCAR-P do. See [Pick your exam](index.md#pick-your-exam) |
    | You do not work at a Claude Partner Network organization | Certification is currently open only to partner organizations. See [Who can sit the exams](index.md#who-can-sit-the-exams) |

## Exam at a glance

The official numbers come first, then what they mean on the day, the program rules the guide's table leaves out, and how CCAO-F compares with the other three exams.

### The official details table

This is Section 5 ("Exam Details at a Glance") of the [CCAO-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf), row for row. Numeric ranges are written with "to", and the credential name is written with a comma where the guide prints a dash.

| Field | Value in the guide |
|---|---|
| Credential | Claude Certified Associate, Foundations |
| Exam code | CCAO-F |
| Number of items | 60 |
| Item format | Multiple-choice and multiple-response items; each item states how many responses to select |
| Time limit | 120 minutes |
| Delivery | Proctored: online proctored and/or test center, per program policy |
| Passing score | Scaled score of 720 on a scale of 100 to 1,000 |
| Exam fee | &#36;99 USD |
| Validity period | 12 months from the date the credential is awarded |
| Result reporting | Pass/fail with scaled score (100 to 1,000), plus percent-correct by domain on the score report |

Three more facts frame the table:

- **Which version you are reading.** The guide is Version 1.0, effective July 2026, and says it "is subject to change without notice." Its document-control table records one entry, the initial publication in July 2026. As of September 2026 the [Partner Academy certifications page](https://anthropic-partners.skilljar.com/page/partner-certifications) still links this PDF as the CCAO-F exam guide.
- **No scenario structure.** The table has no "Exam structure" row. The CCAR-F guide is the one that lists "4 scenarios drawn from a bank of 6"; nothing in the CCAO-F guide describes a scenario bank.
- **The name to look for when you schedule.** Pearson VUE lists the exam as "Claude Certified Associate - Foundations (CCAO-F)" on its [Anthropic page](https://www.pearsonvue.com/us/en/anthropic.html).

### What the numbers mean on the day

- **Pacing.** 120 minutes for 60 items is 2 minutes per item (our arithmetic: 120 ÷ 60).
- **Multiple response.** Some items ask for more than one answer, and each item tells you how many to select. The three official samples are all single-answer, so they never show this format (see [What the samples do not show](#what-the-samples-do-not-show)).
- **The pass mark is scaled.** 720 is a point on a 100 to 1,000 scale, not a percentage of items. The [program FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says: "Scaled scoring equates scores across exam forms that may have slightly different difficulty." The guide does not publish how many correct answers a 720 requires.
- **You are measured against a standard, not other candidates.** The guide calls the exam "criterion-referenced": "You pass by demonstrating the knowledge and skills defined in the blueprint, not by outperforming a percentage of peers." Trained subject matter experts set the cut score by judging what a minimally qualified candidate should achieve, which is why the candidate profile in [Who this exam is for](#who-this-exam-is-for) is worth reading closely.
- **Domain percentages are feedback only.** The score report shows the percentage you answered correctly in each domain, but pass or fail rests on the total scaled score alone. See [Scoring, results and badges](index.md#scoring-results-and-badges).
- **Seat time is longer than the clock.** The FAQ says: "Plan for about 135 minutes of total seat time, which includes check-in, instructions, and a brief post-exam survey."

### Beyond the guide's table (as of September 2026)

The details table leaves out several rules that decide whether, when and how you can sit the exam. They come from other sections of the guide and from the live program pages.

| Topic | CCAO-F | More detail |
|---|---|---|
| Fee after discounts | &#36;99 USD list price, paid by credit card on the Anthropic Partner Academy. Registered-tier partners pay full price. Select, Preferred and Global Premier partners get 50% off at checkout, which is &#36;49.50 (our arithmetic). Through December 31, 2026, Global Premier partners get 100% off. | [Fees and partner discounts](index.md#fees-and-partner-discounts) |
| Who can register | People at Claude Partner Network organizations, using a partner email address on a recognized company domain; personal email addresses do not work. To sit the exam you must be at least 18; age is checked against government-issued ID at check-in. | [Who can sit the exams](index.md#who-can-sit-the-exams) |
| Prerequisites | None. No software-development or API experience is needed, and the recommended experience is not required. | [Who this exam is for](#who-this-exam-is-for) |
| Partner standing | Does not count toward Claude Partner Network tier eligibility. CCDV-F, CCAR-F and CCAR-P do. | [Pick your exam](index.md#pick-your-exam) |
| Validity and renewal | 12 months from the date the credential is awarded. On-time renewal is a free, non-proctored assessment on the Anthropic Partner Academy (the [Policies page](https://anthropic-partners.skilljar.com/page/policies-certifications) calls it an "open-book online assessment"). The FAQ says full renewal details will be shared before the first certifications come up for renewal, and the guide says Anthropic may require the full exam instead if content changes significantly. After a lapse, you retake the full exam at the full fee. | [Renewal](index.md#renewal) |
| Retakes | Wait 14 days after a first failed attempt, 30 after a second, 90 after a third. Up to four attempts in a rolling twelve-month period, counted per exam. Every attempt costs the exam fee, and the partner discount applies to retakes. Both the waiting period and the attempt count reset when the exam moves to a new version. If you pass, you cannot retake to raise your score. | [Policies that cost candidates money](index.md#policies-that-cost-candidates-money) |
| Cancel or reschedule | The guide says up to 24 hours before the appointment. The Policies page and the FAQ say 48 hours, and Pearson VUE's Anthropic page gives 48 hours for test center appointments. Plan on 48. Canceling in Pearson frees the slot but does not refund the fee; a refund is requested by email to certifications-support@anthropic.com. | [Policies that cost candidates money](index.md#policies-that-cost-candidates-money) |
| Delivery | Pearson VUE, online through OnVUE or at a Pearson test center. Candidates with a government-issued ID from Belarus, Cuba, North Korea, Russia, Syria or restricted regions of Ukraine are not eligible for OnVUE. From September 8, 2026, Pearson suspended delivery for residents of Iran, both online and at test centers. | [Registration, step by step](index.md#registration-step-by-step) |
| Materials and language | Closed book. Browser translation tools are not permitted, and the [Exam Policy](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F34hhd92iyp94a0gtbr15cy5jk%2Fpublic%2F1782870704%2FAnthropic+Certification+Exam+Policy.pdf) prohibits "Using AI products or services to assist you during the Exam". The exam and prep content are in English only. | [Exam-day checklist](#exam-day-checklist) |
| Practice material | The three sample questions in Section 8 of the guide, which "are not drawn from the live item bank." The practice exam from the previous platform was retired in the move to Pearson. | [Official sample questions](#official-sample-questions) |

!!! warning "The guide's 24-hour cancellation rule is not the one to plan on"

    Step 6 of the registration section in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) says "You may cancel or reschedule up to 24 hours before your appointment." The [Policies page](https://anthropic-partners.skilljar.com/page/policies-certifications) says rescheduling or canceling is free "at least 48 hours before your appointment", and the [FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says "Exams cannot be canceled or rescheduled less than 48 hours prior to your appointment." Pearson VUE's Anthropic page also gives 48 hours for test center appointments. The Policies page adds: "If you reschedule or cancel less than 48 hours before, or you don't show up, you forfeit the fee." You cancel or reschedule through Pearson VUE, so treat 48 hours as the deadline.

### How CCAO-F differs from the other three exams

- **Lower fee.** &#36;99, against &#36;125 for CCDV-F, &#36;125 for CCAR-F and &#36;175 for CCAR-P, all before partner discounts.
- **No partner-tier credit.** CCAO-F does not count toward Claude Partner Network tier eligibility. The FAQ lists CCDV-F, CCAR-F and CCAR-P as the exams that do.
- **Same clock, 60 items.** CCAO-F and CCAR-F have 60 items, CCDV-F has 53 and CCAR-P has 63, all in 120 minutes. That is 2.0 minutes per item here and on CCAR-F, about 2.26 on CCDV-F and about 1.9 on CCAR-P (our arithmetic).
- **Short workplace situations, not a scenario bank.** The [FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says "All Claude certification exams use multiple choice and scenario-based multiple response questions", and the guide's confidentiality section lists "questions, answer options, and scenarios" as exam content. In the CCAO-F samples each stem opens with a situation of one or two sentences: an associate summarizing a new regulation, an associate generating a high volume of short customer-reply drafts, a project manager with a spreadsheet of customer names and account numbers under a policy that restricts regulated personal data.
- **No coding or API background assumed.** The CCAO-F guide says "no software-development or API experience is needed" and describes candidates as having "limited to moderate technical expertise." The CCDV-F guide recommends proficiency in Python and/or TypeScript, the CCAR-F guide's typical candidate has 6+ months building with the Claude APIs, Agent SDK, Claude Code and MCP, and the CCAR-P guide recommends 3+ years in systems architecture or platform engineering.

## Who this exam is for

This section covers who the guide writes the exam for, who it leaves out, and the minimally qualified candidate whose expected performance set the pass mark.

### What the credential says about you

The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) says the certification "validates that an individual can apply Claude to complete business and productivity tasks with minimal guidance." It then spells out four parts: "using built-in platform features, capabilities, and tools to streamline workflows; identifying opportunities to improve processes with Claude; selecting approaches that balance quality, efficiency, and cost; and recognizing limitations and escalating more complex or technical work to Claude Architects and Developers."

The guide says earning it "indicates that the holder has demonstrated the knowledge and skills defined in the exam blueprint." The [Certification Terms and Conditions](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F34hhd92iyp94a0gtbr15cy5jk%2Fpublic%2F1782870634%2FCertification+Terms+and+Conditions.pdf) set the limit: certification "is not a warranty or guarantee of an individual’s abilities regarding Anthropic’s services in general".

### Intended audience

Section 3 of the guide describes "professionals who use Claude as a productivity tool and build Claude Projects in their day-to-day roles." They work "across functions such as operations, marketing, project management, education, communications, and general knowledge work", and the audience includes both internal staff who maintain and optimize ongoing AI-enabled workflows and "external consultants who support implementation, use-case identification, and process redesign."

Two lines place the candidate precisely. Candidates "generally have limited to moderate technical expertise", and they sit "between casual AI prompt users and technical AI practitioners". The official [prep course](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations) describes its audience in practical terms: it "is for people already using AI-powered productivity tools in their work."

**Decide (our rule of thumb, from the guide's audience and profile):** if you use Claude most working days to draft, analyze, research or organize, and you set up Projects for yourself or a team, you are the audience. If you only ask Claude occasional questions, the exam expects more: its minimally qualified candidate "moves beyond basic question-and-answer usage to process reimagination, task automation, and project development."

### One audience, five descriptions

Anthropic's pages, and the Credly and Pearson VUE pages for the program, describe the Associate audience in different words. As of September 2026:

| Source | How it describes the Associate audience |
|---|---|
| [Exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) | "professionals who use Claude as a productivity tool and build Claude Projects in their day-to-day roles" |
| [Partner Academy certifications page](https://anthropic-partners.skilljar.com/page/partner-certifications) | "For consultants, sellers, and delivery leads who guide customers toward the right Claude use cases and set engagements up for success." |
| [Credly badge](https://www.credly.com/org/anthropic/badge/claude-certified-associate-foundations) | "designed for client-facing and delivery practitioners." |
| [Launch post on claude.com](https://claude.com/blog/four-role-based-claude-certifications) | "validates practical, everyday use of Claude for anyone working on Claude-related projects", applying to "a wide range of roles including consultants, project leads, and both business and technical expertise." |
| [Pearson VUE Anthropic page](https://www.pearsonvue.com/us/en/anthropic.html) | Names the three roles "Practitioner, Architect, and Developer", although it lists the exam itself as "Claude Certified Associate - Foundations (CCAO-F)" |

For what the exam tests, go by the guide: the [program FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) calls the exam guide "the authoritative source for exam scope." Our reading: the partner-facing wording overlaps the guide's audience, which already includes external consultants, although the guide names neither sellers nor delivery leads. A seller or delivery lead should prepare as a hands-on user, because every official sample puts you in the seat of someone using Claude for their own work (summarizing a regulation, generating customer-reply drafts, analyzing a spreadsheet).

### No coding required

The guide is explicit on three points:

- **Prerequisites.** "There are no mandatory prerequisites or required courses, and no software-development or API experience is needed."
- **Exclusion.** The certification "is not intended for software developers who build against APIs or design agentic systems, nor for specialists in machine learning, software engineering, or advanced AI system design."
- **Where the boundary sits.** "Candidates are not expected to design enterprise-scale AI architectures or integrations; that scope belongs to the Claude Architect and Claude Developer credentials, to which Associates escalate more complex or technical work."

Two objectives use technical words: D1.1 "Create effective prompts for business and technical tasks" and D4.3 "Use Claude to support solution design, development, and iteration". Our reading: treat both as prompting and assisting skills, not coding skills, because the same guide excludes developers who build against APIs. [Claude 101](https://academy.claude.com/courses/claude-101/claude-desktop-app-chat-cowork-code), one of the three courses the official prep course recommends first, tells non-developers that the Code tab is "a separate tab, and this course doesn't need it".

What the guide does expect is knowing where your own work stops. Recognizing limitations and escalating appears in the credential's purpose, in the intended-audience description ("recognize when human expertise, validation, or escalation is required") and in the guide's How to Prepare list ("when to escalate or seek human review"). The concrete signals that work has moved past the Associate role, such as a prototype that needs proper API key management before it can become a real application, are collected under [D4.3](#d43-use-claude-to-support-solution-design-development-and-iteration).

**Decide (our rule, built from the guide's scope sentences; the guide publishes no list of escalation signals):** if a situation needs API integration, production infrastructure or an enterprise-scale architecture, the Associate's move is to escalate to Claude Architects or Developers, not to build it. For escalation within workflows, see [Domain 4: Workflow Integration and Solution Design](#domain-4-workflow-integration-and-solution-design); for when human review is required, see [Domain 2: Output Evaluation and Validation](#domain-2-output-evaluation-and-validation).

### The minimally qualified candidate

The exam is pitched at the minimally qualified candidate (MQC): "a professional who uses Claude as a core productivity tool and can apply it effectively within real-world workflows to improve efficiency, quality, and outcomes." Trained subject matter experts set the pass mark by judging the performance expected of this person, so the profile is the bar you are measured against (guide Sections 4 and 9).

The MQC "has foundational, applied knowledge of Claude's capabilities, including prompt structuring, task orchestration, and familiarity with features such as Projects, Artifacts, and workflow-based interactions". They are aware of organizational context: "how Claude creates value, where adoption risks exist, and how to align usage with business needs and responsible-AI practices", and they can complete common productivity and workflow tasks on the platform independently.

Section 3 lists six abilities that distinguish candidates. Each maps onto blueprint objectives (the mapping is ours; the objective numbers are this page's labels, explained in [Blueprint](#blueprint)):

| The guide says candidates are distinguished by their ability to... | Where the blueprint tests it (our mapping) |
|---|---|
| "translate business objectives into effective AI interactions" | D4.1 analyze requirements and use cases; D1.1 create effective prompts |
| "select appropriate tools and features" | D3.1 product features; D3.2 model types; D3.3 model selection by cost, speed and quality |
| "create structured prompts" | D1.1 to D1.4; D5.3 system-level instructions |
| "critically evaluate AI-generated content" | D2.1 accuracy and completeness; D2.2 hallucinations, inconsistencies and biases; D2.3 fact-checking |
| "adapt outputs for different audiences" | D2.5 edit and compare outputs for the intended audience; D4.5 communicate value and limitations to stakeholders |
| "recognize when human expertise, validation, or escalation is required" | D2.4 when human review is required; [Domain 6](#domain-6-governance-risk-and-responsible-use), which the How to Prepare list pairs with escalation |

### Recommended experience

The guide recommends, without requiring:

- [ ] Regular, hands-on experience using Claude in a professional setting
- [ ] A foundational understanding of structured problem-solving, workflow design, and digital tool usage
- [ ] Experience in roles such as business analyst, project manager, operations lead, marketing/communications/HR/education professional, consultant, or knowledge worker
- [ ] A practical understanding of AI limitations, including hallucinations, context constraints, and data sensitivity

The last bullet is a preview of the exam. Each limitation it names has its own objective: hallucinations in D2.2, context constraints in D3.4 (when to restart, summarize or persist) and data sensitivity in D6.2. If any of the three is unfamiliar, start with those objectives.

The guide adds: "The experience above is recommended, not required. The credential is awarded based on exam performance alone." Having no experience prerequisites does not mean anyone can register: registration is limited to people at Claude Partner Network organizations, and you must be at least 18 to take the exam. See [Who can sit the exams](index.md#who-can-sit-the-exams).

### Where CCAO-F sits in the program

The [Partner Academy](https://anthropic-partners.skilljar.com/page/partner-certifications) organizes certification into three roles, Associate, Developer and Architect, and says: "Start at Foundations, then advance to Professional where available." For Associate it lists only Foundations, and [Credly](https://www.credly.com/org/anthropic/badge/claude-certified-associate-foundations) lists the CCAO-F badge at level "Foundational". The [launch post](https://claude.com/blog/four-role-based-claude-certifications) says every path "starts with a foundation-level certification and advances to the professional-level", but as of September 2026 the Partner Academy lists no Professional-level Associate exam.

If your work moves into building against the API or designing systems, the guide places that scope in the Developer and Architect credentials: [CCDV-F](claude-certified-developer.md) and [CCAR-F](claude-certified-architect-foundations.md) are the Foundations exams for those roles.

## Blueprint

Section 6 of the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) defines seven domains and their weights: "Weights reflect the relative importance of each domain to competent performance as determined through the job task analysis. The percentages indicate the approximate proportion of scored items drawn from each domain." A single preamble then introduces the seven lists of objectives: "Each domain below lists the tasks a candidate is expected to perform. Exam items are written against these objectives."

The guide lists the objectives as unnumbered bullets. This page numbers them in the guide's order, so D1.1 is the first objective of Domain 1 and D5.2 is the second objective of Domain 5.

### Domains and weights

| Domain | Weight | Items, about (our arithmetic) | Objectives | Official sample | Official prep course module |
|---|---|---|---|---|---|
| [Domain 1: Prompting and Task Execution](#domain-1-prompting-and-task-execution) | 14% | 8.4 | 4 (D1.1 to D1.4) | None | Prompting & Task Execution |
| [Domain 2: Output Evaluation and Validation](#domain-2-output-evaluation-and-validation) | 21% | 12.6 | 6 (D2.1 to D2.6) | [Sample 1](#official-sample-questions): a cited subsection | Evaluating & Validating Claude's Output |
| [Domain 3: Product and Model Selection](#domain-3-product-and-model-selection) | 12% | 7.2 | 4 (D3.1 to D3.4) | [Sample 2](#official-sample-questions): high-volume drafts | Claude Platform & Model Foundations |
| [Domain 4: Workflow Integration and Solution Design](#domain-4-workflow-integration-and-solution-design) | 16% | 9.6 | 5 (D4.1 to D4.5) | None | Workflow Integration & Solution Design |
| [Domain 5: Configuration and Knowledge Management](#domain-5-configuration-and-knowledge-management) | 12% | 7.2 | 4 (D5.1 to D5.4) | None | Configuration & Knowledge Management |
| [Domain 6: Governance, Risk, and Responsible Use](#domain-6-governance-risk-and-responsible-use) | 15% | 9.0 | 4 (D6.1 to D6.4) | [Sample 3](#official-sample-questions): regulated personal data | Governance, Risk & Responsible Use |
| [Domain 7: Troubleshooting and Optimization](#domain-7-troubleshooting-and-optimization) | 10% | 6.0 | 3 (D7.1 to D7.3) | None | Troubleshooting & Optimization |
| **Total** | **100%** | **60** | **30** | **3** | |

The items column is the weight multiplied by 60. It is a planning figure, not a count: the guide calls the weights approximate, applies them to "scored items", and never says whether all 60 items are scored. The last column pairs each domain with a module of the free [Claude Certified Associate - Foundations Prep Course](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations); the pairing is ours, made from the module titles and descriptions. Five modules carry their domain's name, written with "&" (Domains 1, 4, 5, 6 and 7). Domain 2's module is Evaluating & Validating Claude's Output, whose objectives include the D2.4 wording "Determine when human review or additional verification is required". Domain 3 has no namesake: the matching module is Claude Platform & Model Foundations, which covers "entry point, capability features, model, and context" and differentiating Haiku, Sonnet and Opus.

### How to read the weights

- **Domain 2 carries the most weight.** At 21% it is about 12.6 items (our arithmetic). With Domain 4 (16%) and Domain 6 (15%) it makes up 52% of the weight (our arithmetic: 21 + 16 + 15): judging output, fitting Claude into work, and using it responsibly.
- **Each objective averages about two items.** 60 items over 30 objectives is 2.0 per objective (our arithmetic). The guide does not say how items spread across a domain's objectives; if they spread evenly, dividing each domain's items by its objectives gives between 1.8 (Domains 3 and 5) and 2.25 (Domain 6) items per objective (our arithmetic). No objective is safe to skip: at that average, each one is roughly 3% of the exam (our arithmetic: 2 of 60 items).
- **Four domains have no official sample.** The three samples cover Domains 2, 3 and 6. Domains 1, 4, 5 and 7, 52% of the weight together (our arithmetic), have no published item. The guide says its samples show "the style and cognitive level of the exam", so our expectation is the same shape in every domain: a short workplace situation, then a question asking for the most appropriate action or the choice that best fits the task.
- **The domain breakdown is diagnostic, not decisive.** The score report shows the percentage of items you answered correctly in each domain, and the [program FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says: "You can use the section breakdown to decide what to review before a retake." The guide says these percentages "are not used to determine your pass or fail result, which is based on your total scaled score."

### The 30 objectives

The guide's first preparation step is to "Study the exam blueprint in Section 6 and self-assess against each objective". The lists below are the objectives verbatim, as a checklist. Tick an objective only when you can explain its decision rule from its domain section without notes.

**[Domain 1: Prompting and Task Execution](#domain-1-prompting-and-task-execution) (14%)**

- [ ] D1.1 Create effective prompts for business and technical tasks
- [ ] D1.2 Apply task decomposition techniques to structure complex requests
- [ ] D1.3 Iterate prompts to improve output quality
- [ ] D1.4 Adapt prompting strategies based on task type (analysis, research, drafting, brainstorming)

**[Domain 2: Output Evaluation and Validation](#domain-2-output-evaluation-and-validation) (21%)**

- [ ] D2.1 Evaluate Claude-generated outputs for accuracy and completeness
- [ ] D2.2 Identify hallucinations, inconsistencies, and biases in responses
- [ ] D2.3 Apply fact-checking and validation techniques
- [ ] D2.4 Determine when human review or additional verification is required
- [ ] D2.5 Edit, adapt, refine, and compare outputs for the intended audience
- [ ] D2.6 Organize and curate information and select appropriate output formats (artifacts, inline, structured data)

**[Domain 3: Product and Model Selection](#domain-3-product-and-model-selection) (12%)**

- [ ] D3.1 Select appropriate Claude product features (Projects, research mode, chat, artifacts)
- [ ] D3.2 Differentiate between Claude model types (Haiku, Sonnet, Opus)
- [ ] D3.3 Align model selection with task requirements (cost, speed, quality)
- [ ] D3.4 Understand and manage context limitations and memory considerations (when to restart, summarize, or persist)

**[Domain 4: Workflow Integration and Solution Design](#domain-4-workflow-integration-and-solution-design) (16%)**

- [ ] D4.1 Apply Claude to analyze requirements and use cases
- [ ] D4.2 Leverage Claude for research, planning, and process optimization
- [ ] D4.3 Use Claude to support solution design, development, and iteration
- [ ] D4.4 Integrate Claude into existing workflows to augment or redesign them
- [ ] D4.5 Communicate Claude's value and limitations to stakeholders

**[Domain 5: Configuration and Knowledge Management](#domain-5-configuration-and-knowledge-management) (12%)**

- [ ] D5.1 Configure Claude Projects with instructions and knowledge sources
- [ ] D5.2 Manage uploaded knowledge and connectors (e.g., Google Drive, Gmail)
- [ ] D5.3 Create effective system-level instructions
- [ ] D5.4 Inform, maintain, and update Claude configurations, knowledge sources, and instructions

**[Domain 6: Governance, Risk, and Responsible Use](#domain-6-governance-risk-and-responsible-use) (15%)**

- [ ] D6.1 Identify appropriate and inappropriate use cases
- [ ] D6.2 Apply data sensitivity, regulatory, and privacy considerations
- [ ] D6.3 Follow organizational AI policies and governance standards
- [ ] D6.4 Understand the ethical implications of AI usage

**[Domain 7: Troubleshooting and Optimization](#domain-7-troubleshooting-and-optimization) (10%)**

- [ ] D7.1 Identify, diagnose, and resolve issues with underperforming prompts or poor outputs
- [ ] D7.2 Adjust approach based on feedback and results
- [ ] D7.3 Optimize workflows for efficiency and effectiveness

### The named lists in the objectives

Several objectives contain a named list, most of them in parentheses. Our advice: treat these lists as the exam's vocabulary, and learn each item and what separates it from the others in the same list. The third column is ours.

| Objective | What it lists | The distinction to be able to make |
|---|---|---|
| D1.4 | Task types: analysis, research, drafting, brainstorming | How the prompting strategy changes with the type of task |
| D2.2 | Failure types: hallucinations, inconsistencies, biases | Which failure you are looking at, and how you would detect it |
| D2.6 | Output formats: artifacts, inline, structured data | Which format suits the content and the person receiving it |
| D3.1 | Product features: Projects, research mode, chat, artifacts | Which feature fits a given task |
| D3.2 | Model types: Haiku, Sonnet, Opus | How the three differ |
| D3.3 | Task requirements: cost, speed, quality | Which requirement dominates, and therefore which model |
| D3.4 | Context actions: restart, summarize, persist | When each one is the right response to a context limit |
| D5.2 | Example connectors: Google Drive, Gmail | Managing uploaded knowledge versus connected sources |
| D6.2 | Considerations: data sensitivity, regulatory, privacy | Which consideration applies before data reaches Claude |

The How to Prepare section adds a list that is not an objective, "Claude features such as Projects, Artifacts, Memory, Skills, and Code Execution"; how to treat the three that no objective names as features is answered in the [Frequently asked questions](#frequently-asked-questions).

### Ideas the guide repeats across domains

Some ideas appear in several places in the guide: its purpose statement, its intended audience and recommended experience, its preparation advice, its sample questions and objectives in more than one domain. Learn each once and apply the same answer wherever it shows up. The grouping into ideas is ours; the locations are the guide's.

| Idea | Where the guide states it |
|---|---|
| Verify before you rely on an output, and know when human review or escalation is required | Purpose ("Recognize limitations and escalate more complex or technical implementations"); intended audience ("recognize when human expertise, validation, or escalation is required"); D2.3, D2.4; How to Prepare; Sample 1 |
| Match the model and approach to the task's cost, speed and quality needs | Purpose ("Select appropriate approaches to balance quality, efficiency, and cost"); D3.3; D7.3; Sample 2 |
| Protect sensitive data before it reaches Claude | Recommended experience (data sensitivity); D6.2; D6.3; How to Prepare; Sample 3 |
| Know the model's limits | Recommended experience (hallucinations, context constraints); D2.2; D3.4; D4.5 |
| Iterate on prompts, outputs and configurations | D1.3; D2.5 ("refine"); D4.3 ("development, and iteration"); D5.4; D7.2; How to Prepare ("iterating to improve outputs") |
| Write for the person who receives the output | Intended audience ("adapt outputs for different audiences"); D2.5; D4.5 |
| Configure once in a Project instead of repeating yourself in every prompt | Intended audience (builds Claude Projects); D5.1; D5.3; D5.4; How to Prepare ("configure a Project with instructions and knowledge sources"). The prep course's [configuration module](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/configuration-knowledge-management) puts it this way: "A good prompt helps you once, while a good configuration helps you every time." |

!!! warning "Blueprint wording and the product today (as of September 2026)"

    The objectives use July 2026 product vocabulary. Some of it has moved on, and some of it is described differently by different official pages today. Each change is explained once, in the objective it affects:

    - **Models.** D3.2 names "Haiku, Sonnet, Opus"; the current [models overview](https://platform.claude.com/docs/en/models/overview) also lists Claude Fable 5.1, a family the objective does not name. See [D3.2](#d32-differentiate-between-claude-model-types-haiku-sonnet-opus).
    - **Research mode.** D3.1 says "research mode"; the help center calls the feature [Research](https://support.claude.com/en/articles/11088861-use-research-on-claude). See [D3.1](#d31-select-appropriate-claude-product-features-projects-research-mode-chat-artifacts).
    - **Chat.** A new experience that merges Chat and Cowork into one conversation is [rolling out gradually](https://support.claude.com/en/articles/16761823-claude-cowork-and-chat-are-one-claude), starting with Pro and Max plans, so the chat surface D3.1 names may look different on your account. See [D3.1](#d31-select-appropriate-claude-product-features-projects-research-mode-chat-artifacts).
    - **Connectors.** D5.2 gives Google Drive and Gmail as examples, and official pages disagree on what the Gmail connector can do. See [D5.2](#d52-manage-uploaded-knowledge-and-connectors-eg-google-drive-gmail).

    Exam items are written against the guide's objectives, and the program FAQ calls the guide "the authoritative source for exam scope." Answer in the guide's terms and on the concept being tested, such as a faster, lower-cost model for straightforward, high-volume work (Sample 2), rather than on a model name. For the current product behind each term, see [Domain 3](#domain-3-product-and-model-selection) and [Domain 5](#domain-5-configuration-and-knowledge-management).

## Domain 1: Prompting and Task Execution

**Weight:** 14% of scored items. If all 60 items were scored, that is about 8.4 items (14% of 60, our arithmetic); the guide does not say how many of the 60 are scored.

Domain 1 asks whether you can turn a business request into a prompt that returns usable work, break a large request into steps, improve a weak result by iterating, and change your approach for analysis, research, drafting and brainstorming. None of the three official sample questions maps to this domain (they cover Domains 2, 3 and 6), so the decision rules below are our own, drawn from Anthropic's help center, product documentation, blog and engineering posts, and Academy courses. The guide's own How to Prepare list gives the matching practice task: "Practice structuring prompts, decomposing tasks, and iterating to improve outputs" ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)).

The matching module of the free prep course is Prompting & Task Execution (53 minutes). Its [course page](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/prompting-task-execution) says that Description, one of the AI Fluency competencies, "is the backbone of this module and the prompting foundation the rest of the course builds on." The same page sums up the domain in one example: a request to write something about "our Q3 results" tends to return a generic paragraph, while one that names the audience, the three results that matter, the format and the length can return a draft ready to share.

### D1.1 Create effective prompts for business and technical tasks

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Create effective prompts for business and technical tasks". The same guide says the credential "is not intended for software developers who build against APIs", so read "technical tasks" as prompting about technical material (a spreadsheet model, a process map, an error message), not writing code against the API (our reading).

**Know: Claude does not know your situation unless you tell it.** The help center's [prompt design article](https://support.claude.com/en/articles/7996853-introduction-to-prompt-design) says: "Think of Claude as a newly-hired contractor. It doesn’t have any context about you, your task, or your organization." Anthropic's [prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) give a test for any prompt: "Show your prompt to a colleague with minimal context on the task and ask them to follow it. If they'd be confused, Claude will be too."

**Know: a three-part frame.** [Claude 101](https://academy.claude.com/courses/claude-101/your-first-conversation-with-claude) asks you to consider "setting the stage (your role, objectives, and context), defining the task (what action you want Claude to take), and specifying rules (style, tone, and examples)". The course says the frame is adapted from the 4D Framework for AI Fluency. The CCAO-F prep course teaches its own "repeatable component" structure, but that content sits behind registration; use the Claude 101 frame as a public model of the same idea, not as the prep course's framework.

| Part | What it carries | Workplace example (ours) |
|---|---|---|
| Set the stage | Claude 101: your role, objectives and context. Other Academy courses add the audience and the details Claude cannot know (your track record, partnerships, staff expertise) | *I'm the operations lead for a regional delivery team. This goes to our five site managers, who will act on it on Monday.* |
| Define the task | Claude 101: the action you want Claude to take. The claude.com blog adds: lead with a direct verb such as Write, Analyze, Create | *Draft a one-page summary of last month's late deliveries from the attached log.* |
| Specify the rules | Claude 101: style, tone and examples. Other Anthropic guidance adds format, length and what to do when information is missing | *Three short sections, plain language, one table. If a figure is not in the log, write Not specified.* |

**Know: Description, the competency the prep module is built on.** The AI Fluency course defines Description as "Communicating clearly with AI systems" ([the 4D Framework](https://academy.claude.com/courses/ai-fluency-framework-foundations/the-4d-framework)) and splits it three ways ([A closer look at Description](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-description)). The same lesson describes AI systems as "interactive partners, not databases or vending machines".

| Type | What it covers | Example (ours) |
|---|---|---|
| Product Description | Defining "what you want in terms of outputs, format, audience, and style" | *A one-page memo for the finance team, plain language, one table* |
| Process Description | How Claude approaches your request, which the course says "can be as important as specifying the end goal" | *Check the totals against the source sheet before you summarize* |
| Performance Description | Behavior, such as "whether the AI should be concise or detailed, challenging or supportive" | *Be concise, and challenge any assumption that looks weak* |

The course's [effective prompting techniques](https://academy.claude.com/courses/ai-fluency-framework-foundations/effective-prompting-techniques) lesson lists six: give context (what you want, why you want it, and relevant background), show examples, specify constraints (format, length and other output requirements), break complex tasks into steps, ask the AI to think first, and define the AI's role or tone. These six techniques and the three Description types are further public models of good description.

**Know: the techniques Anthropic teaches.**

- **Say why.** Explaining the reason behind an instruction helps, and "Claude is smart enough to generalize from the explanation" ([prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)). Our example: *This goes to the board, so every figure needs its source* gives Claude more to work with than *be accurate*.
- **State the goal as well as the format.** The [Steerability lesson](https://academy.claude.com/courses/ai-capabilities-and-limitations/steerability) separates the two: convincing your team a timeline is realistic is a goal; three bullet points is a format. The lesson asks you to add a goal statement to any task you have been prompting with format alone.
- **Say what to do, not only what to avoid.** The [docs' example](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) replaces "Do not use markdown in your response" with "Your response should be composed of smoothly flowing prose paragraphs."
- **Ask for the action you want.** The [same page](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) warns that if you write "can you suggest some changes", Claude will sometimes give suggestions instead of making the changes.
- **Show an example when the format is easier to show than describe.** The [docs](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) call examples "one of the most reliable ways to steer Claude's output format, tone, and structure" and recommend 3 to 5; the [claude.com blog](https://claude.com/blog/best-practices-for-prompt-engineering) advises starting with one and adding more only if the output still misses. To match a voice, the Academy suggests uploading 2 to 3 representative pieces.
- **Name the deliverable, the format and the inputs.** The help center article [Claude Cowork and chat are one Claude](https://support.claude.com/en/articles/16761823-claude-cowork-and-chat-are-one-claude) lists what a good request states: the desired outcome ("a one-page summary"), the format ("a Word document, a slide deck, or a message you can paste into Slack") and the files, links or apps Claude should work from.
- **Do not over-engineer.** The [claude.com blog](https://claude.com/blog/best-practices-for-prompt-engineering): "Longer, more complex prompts are NOT always better." Claude 101 advises speaking to Claude as you would a coworker: naturally, concisely, conversationally.
- **Template what recurs.** The AI Fluency course suggests template prompts for recurring tasks, with placeholders for the information that changes each time.

**Know: technical tasks without code.** The guide says no software-development or API experience is needed. Claude 101 notes that AI models can help non-specialists write Excel formulas and reformat messy data, and an [Academy use case](https://academy.claude.com/use-cases/understand-and-extend-an-inherited-spreadsheet) for an inherited spreadsheet asks Claude to explain how the tabs connect, what the formulas are doing and anything important that is not documented. The help center article [Give Claude context: CLAUDE.md and better prompts](https://support.claude.com/en/articles/14553240-give-claude-context-claude-md-and-better-prompts), written for Claude Code users (it says its habits "are not generic prompt-engineering tips"), advises: "Paste the full stack trace rather than summarizing it." The same idea applies to a business user (our application): paste the exact error text, not a description of it.

```text
Example prompt (ours), built on the three-part frame:

I lead customer onboarding at a 200-person software company. Our VP of
Sales asked why new customers take longer to go live this quarter.
Using only the attached onboarding tracker, identify the three biggest
causes of delay and how many accounts each affected. Write it as a
half-page note the VP can forward to the sales team: plain language,
one short table, no jargon. If the tracker does not show a cause
clearly, say so rather than guessing.
```

**Decide**

- If the output is for a specific reader, choose a prompt that names the reader, the purpose and the format; not a topic-only request, because missing context about your situation is the cause Claude 101 gives for generic output.
- If the format is easier to show than to describe, choose an example; not a longer description, because examples are among the most reliable ways to steer format and tone.
- If you want Claude to change something, choose an instruction (*Rewrite the second paragraph for a customer audience*); not *can you suggest...*, because Claude may return suggestions only.
- If a prompt keeps growing with every edge case, choose the essentials plus a few diverse, canonical examples; not more rules, because longer prompts are not always better and Anthropic's context-engineering guidance recommends curating a set of diverse, canonical examples instead of stuffing a laundry list of edge cases into a prompt.
- If the same request recurs every week, choose a saved template or standing instructions (see [Domain 5](#domain-5-configuration-and-knowledge-management)); not retyping it from memory each time, because the AI Fluency course recommends template prompts for recurring tasks, with placeholders for the information that changes each time.

**Traps**

- An answer that sends a bare topic (*write something about Q3*) when another option adds audience, format and length.
- Instructions made only of prohibitions, when an option states the wanted behavior positively.
- A format dressed up as a goal (*three bullets*) with no statement of what the output must achieve.
- Assuming Claude knows your team, your acronyms or last week's meeting. Unless memory, your Instructions for Claude or a Project supplies it, Claude has no context about you, your task or your organization; put that context in the prompt, or keep it where Claude can use it, such as Project knowledge (see [D3.4](#d34-understand-and-manage-context-limitations-and-memory-considerations-when-to-restart-summarize-or-persist) for how Projects and memory carry context).
- Using every technique at once. The [claude.com blog](https://claude.com/blog/best-practices-for-prompt-engineering) lists "Don't use every technique at once" among common mistakes.

**Go deeper:** [Principles that decide most prompt questions](knowledge/prompt-engineering.md#principles-that-decide-most-prompt-questions)

### D1.2 Apply task decomposition techniques to structure complex requests

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Apply task decomposition techniques to structure complex requests". The [prep course module](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/prompting-task-execution) states the same skill for "complex, multi-part requests".

**Know: why splitting works.** The help center's advice for unhelpful answers includes "Break down complex requests into substeps." ([help center](https://support.claude.com/en/articles/7996857-my-prompt-isn-t-giving-me-a-helpful-answer)). Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) explains the effect. In its section on parallelization, it says "For complex tasks with multiple considerations, LLMs generally perform better when each consideration is handled by a separate LLM call"; for prompt chaining, the main goal is "to trade off latency for higher accuracy, by making each LLM call an easier task." The docs' consistency page adds that each subtask then gets Claude's full attention.

**Know: the decomposition techniques.**

| Technique | What it looks like in chat | Where the idea comes from |
|---|---|---|
| Staged chain | Ask for an outline, check it against your criteria, then ask for the document | Anthropic's chaining example: outline, check the outline, then write |
| Draft, review, refine | *Draft it*, then *Review the draft against these three criteria*, then *Revise based on your review* | The most common chaining pattern in the docs: self-correction |
| Checkpoint | *Do steps 1 and 2, then stop and show me step 2 before continuing* | The AI Capabilities and Limitations course: insert a checkpoint in multi-step tasks to catch reasoning drift, where small errors compound |
| Quotes first | For a long document: *First extract the exact passages about termination; then, using only those quotes, summarize the risks* | The docs: for documents over 20k tokens, extract word-for-word quotes before the task |
| Chunk and combine | Summarize each section of a very long report, then summarize the summaries | The legal summarization guide's meta-summarization, which often catches details a single summary misses |
| Interview first | *Ask me questions until you understand the problem, then propose a plan* | An [Academy use case](https://academy.claude.com/use-cases/explore-what-claude-can-do-for-you): "If you're not sure what you need, let Claude ask the questions instead." |

**Know: when not to split.** Anthropic's [agent guidance](https://www.anthropic.com/engineering/building-effective-agents) says that for many applications, optimizing single LLM calls "with retrieval and in-context examples is usually enough." The [prompting docs](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) say that with adaptive thinking and subagent orchestration, Claude handles most multistep reasoning internally, and they keep explicit chaining for when "you need to inspect intermediate outputs or enforce a specific pipeline structure." For related but independent questions, the usage guidance points the other way: "If you have multiple related tasks or questions, group them in a single message." ([usage limit best practices](https://support.claude.com/en/articles/9797557-usage-limit-best-practices)).

**Know: handing off a multi-step task (as of September 2026).** [Claude 101](https://academy.claude.com/courses/claude-101/claude-desktop-app-chat-cowork-code) maps "a multi-step task that ends in a finished deliverable, spans your tools, or runs on a schedule" to Cowork, and the [Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/what-is-cowork) says Claude shows you its plan before it starts and, by default, asks before actions that matter (sending, deleting, sharing). The surfaces are changing: in the new Claude experience, which is rolling out gradually starting with Pro and Max plans, Cowork and chat are one conversation, and Claude "can decide which tool to use" ([help center](https://support.claude.com/en/articles/16761823-claude-cowork-and-chat-are-one-claude)). Claude 101's separate Chat and Cowork tabs may therefore not match every account. The same article says "You don't need to break the work into steps or pick a mode first." That describes the product, not the exam: the guide still tests decomposition as a prompting skill, so expect its wording. When you hand off a task in the new experience, the checks between stages move to approving actions (the default Manual permission setting asks before each one) and to stopping or redirecting Claude while it works (our reading). The guide names neither Cowork nor the new experience; treat them as today's surfaces for decomposed work you delegate (see [D1.4](#d14-adapt-prompting-strategies-based-on-task-type-analysis-research-drafting-brainstorming) and [D3.1](#d31-select-appropriate-claude-product-features-projects-research-mode-chat-artifacts)).

```text
Example decomposition (ours): a quarterly vendor review

1. "From the attached delivery log, list every late delivery in a
   table: vendor, date, days late. Quote the log row for any entry
   you are unsure about."
2. You check the table against the log.
3. "Using only that table, rank vendors by total days late and flag
   anything that looks like a data-entry error."
4. "Draft a half-page note to the procurement lead recommending which
   two vendors to review first, citing the table."
```

**Decide**

- If the request has dependent stages (extract, then analyze, then write), choose a sequence of prompts with a check between stages; not one long prompt, because each step then gets full attention and errors are caught before they compound.
- If the questions are related but independent, choose one well-structured message; not a chain, because chaining adds latency and extra messages use more of your limit.
- If the source is a long document, choose quotes first, then analysis from those quotes; not *read this and conclude* in one pass, because the docs say quoting the relevant parts first helps Claude focus on the relevant content and ignore the rest of the document.
- If you do not yet know what you need, choose to let Claude interview you; not a guessed-at complete specification, because an Academy use case recommends letting Claude ask the questions when you are not sure what you need.
- If the task is simple, choose a single prompt; not decomposition for its own sake, because a single optimized call is often enough.

**Traps**

- The mega-prompt option that asks for extraction, analysis, recommendations and a formatted deck in one message with no checkpoint.
- The opposite: splitting a simple, one-step request into a chain, which adds latency when a single well-built call is usually enough.
- Letting a long multi-step run finish unchecked, then blaming the model for an error made at step 2. The [Steerability lesson](https://academy.claude.com/courses/ai-capabilities-and-limitations/steerability) names this failure "reasoning drift (small errors compound)".

**Go deeper:** [Chain of thought and prompt chaining](knowledge/prompt-engineering.md#chain-of-thought-and-prompt-chaining)

### D1.3 Iterate prompts to improve output quality

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Iterate prompts to improve output quality". The [prep course path](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations) lists the skill among its learning objectives: "iterate diagnostically to improve output quality".

**Know: iteration is the normal workflow.** [Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results) says: "Think of your initial prompt as the start of a conversation, not a one-shot request." The AI Fluency course calls this the [Description-Discernment loop](https://academy.claude.com/courses/ai-fluency-framework-foundations/the-description-discernment-loop): describe what you need, discern the quality of what comes back, refine (give feedback, clarify, and "Request iterations until you're satisfied with the result"), then integrate your own expertise, making the final decisions about what to keep, modify or discard. Anthropic's [AI Fluency Index](https://academy.claude.com/tutorials/the-ai-fluency-index) found iteration and refinement in 85.7% of the multi-turn Claude.ai conversations it sampled (one week in January 2026), and those conversations were 5.6x more likely to involve users questioning Claude's reasoning and 4x more likely to identify missing context.

**Know: the iteration moves and when each fits.**

| Move | Use it when | How |
|---|---|---|
| Targeted feedback | The draft is mostly right | Say exactly what to change. The [Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/the-task-loop): "If the draft is mostly right, tell Claude what to change rather than starting over." |
| Add the missing context | The draft is wrong at its core | The [same lesson](https://academy.claude.com/courses/introduction-to-claude-cowork/the-task-loop): "If the draft is wrong in a load-bearing way, the prompt was missing the load-bearing piece of context." Supply it and ask for adjustments. |
| Restate the goal | Claude followed the words and missed the point | The [Steerability lesson](https://academy.claude.com/courses/ai-capabilities-and-limitations/steerability): "When an instruction is followed literally but uselessly, restate the goal. Repeating the instruction with more force won't close the gap." |
| Follow-up question | You need more detail, another angle or a clarification | Build on the response in the same chat |
| Edit an earlier message | You want to try another direction without losing the current one | Claude 101: click the pencil icon on any of your messages to edit and resubmit it. The help center: editing a prior message creates a different version of the conversation, with its own artifacts. An Academy tutorial calls this forking and says you can always return to your original version. As of September 2026, the new Claude experience lists "Branching a conversation from an earlier point isn't available" among its current limitations ([help center](https://support.claude.com/en/articles/16761823-claude-cowork-and-chat-are-one-claude)); there, open a new chat to try another direction (our suggestion) |
| Start fresh | The chat has gone off track | [Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results): "If a conversation has gone off track, sometimes it's faster to open a new chat with a clearer prompt than to try to redirect." |

**Know: what good feedback looks like.** In [Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results), "Make it shorter" is fine, but "Cut the first two paragraphs and make the conclusion more action-oriented" is better. The [Steerability lesson](https://academy.claude.com/courses/ai-capabilities-and-limitations/steerability) pairs the instruction with its purpose: "Make this shorter. My goal is to keep the executive's attention through the key finding on page two." When you iterate on a skill, the [Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/validating-skills-for-plugins) advises: "Change one thing at a time." It means fixing the issue that matters more, re-running, then coming back for another review; the same discipline suits any prompt you reuse (our extension). The AI Fluency course's "secret weapon" is to ask the AI itself to help improve your prompt ([effective prompting techniques](https://academy.claude.com/courses/ai-fluency-framework-foundations/effective-prompting-techniques)).

**Know: what iteration does not do.** From the [AI Capabilities and Limitations course](https://academy.claude.com/courses/ai-capabilities-and-limitations/working-memory): "The model doesn't learn from your corrections. It only responds to what's currently in context." If you want a correction to last, store it where Claude will read it, not only in the chat where you made it (our advice). The [Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/giving-cowork-context) treats corrections you keep repeating as "global-instruction candidates"; where standing instructions live today is in the box under [D5.3](#d53-create-effective-system-level-instructions), and how edits to an artifact interact with Claude is under [D2.5](#d25-edit-adapt-refine-and-compare-outputs-for-the-intended-audience).

**Decide**

- If the draft is mostly right, choose specific feedback in the same conversation; not regenerating or a new chat, because Claude remembers the conversation and edits faster than it regenerates.
- If the output misses the point after a clear instruction, choose to restate the goal; not to repeat the instruction more forcefully, because the Steerability lesson says repeating the instruction with more force will not close the gap.
- If the conversation has gone off track, choose first to steer Claude back with a clarification, then a new chat with a clearer prompt if that does not work; not round after round of redirection, because Claude 101 treats restarting as the worst case, one that fully refreshes the context and is sometimes faster than redirecting.
- If you make the same correction across conversations, choose standing instructions (Instructions for Claude for every chat, project instructions for one Project) or a skill (see [Domain 5](#domain-5-configuration-and-knowledge-management)); not retyping it, because the model does not learn from corrections.
- If you want to compare two directions, choose to edit an earlier message and branch (in the previous chat experience; see the table for the new one); not to overwrite your only good version, because editing a prior message creates a different version of the conversation without losing previous work.

**Traps**

- Vague feedback (*make it better*, *not quite right*) offered as the fix. The [Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/validating-skills-for-plugins) says specific feedback "gives Claude something to act on"; vague feedback does not.
- Changing tone, length and structure in one revision, then not knowing which change helped.
- Believing the model itself learned from last week's corrections. The [AI Capabilities and Limitations course](https://academy.claude.com/courses/ai-capabilities-and-limitations/working-memory) says it "only responds to what's currently in context". As of September 2026, memory (on by default for Free, Pro and Max) can carry saved topics, such as communication preferences, into a new chat ([help center](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context)); a correction that must apply every time still belongs in instructions or a skill.
- Accepting the first response as final when the task matters.

Diagnosing why a prompt underperforms (the prep course names under-specification, context overload, the wrong feature or model, and stale configuration) is the subject of [Domain 7](#domain-7-troubleshooting-and-optimization).

**Go deeper:** [Prompt versioning and iteration](knowledge/prompt-engineering.md#prompt-versioning-and-iteration)

### D1.4 Adapt prompting strategies based on task type (analysis, research, drafting, brainstorming)

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Adapt prompting strategies based on task type (analysis, research, drafting, brainstorming)". The four task types differ in what the prompt must supply, where the work should happen, and what you check before using the result.

**Know: the four task types.**

| Task type | What the prompt must add | Where and how | Check before you use it |
|---|---|---|---|
| Analysis | The question and focus areas, all relevant data in one well-structured message, and the data fields the analysis depends on (in Claude 101's example, without enrollment dates the AI will try to infer them) | Upload the data file (CSV, TSV and others) with Code execution and file creation on (available on every plan), after redacting or anonymizing regulated identifiers where policy restricts sharing them (see [D6.2](#d62-apply-data-sensitivity-regulatory-and-privacy-considerations)); for detailed document analysis or multi-step technical problems, raise effort, turn on thinking, or both | A striking pattern can be real, a confound or a sample quirk; test the approach first on past data whose answer you already know |
| Research | Who you are, who you serve, what exactly you need to know, and a request for citations | Web search for a fact or two; Research for questions that need many sources | Flag regulations, pricing and deadlines for checking against primary sources; question how recent the information is |
| Drafting | Requirements, audience and key points up front; past examples of your voice; the details Claude cannot know | Turn by turn in chat, because your judgment on each turn is the point; the default settings suit general writing | You own the final result and should be able to stand behind every word |
| Brainstorming | Open framing (*What am I not considering?*), a request for a few options (the Claude Design help article suggests 2 to 3 when a direction is unclear) or deliberately different variants, and permission to push back | Turn by turn in chat, because each answer shapes the next question | Honest feedback matters here; avoid asking for validation |

**Know: analysis details.** Adding "flag anything that surprises you" to an analysis request gets interpretation alongside the chart; without it, the [Academy use case](https://academy.claude.com/use-cases/chart-your-data-before-you-commit) says, "you get the matrix and do the reading yourself." Before you trust Claude with a new kind of analysis, test it on past work whose answer you already know: Claude 101's delegation diligence loop, taught under [D2.1](#d21-evaluate-claude-generated-outputs-for-accuracy-and-completeness).

**Know: research details.** Which tool fits a research question (web search for a fact or two, extended thinking for reasoning without fresh information, Research for a synthesis of many sources) is taught under [D3.1](#d31-select-appropriate-claude-product-features-projects-research-mode-chat-artifacts).

For research prompts, Anthropic's [prompting docs](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) recommend clear success criteria and asking Claude to "verify information across multiple sources." Two Academy pages add checks. A stretch goal in an [AI Fluency for Nonprofits lesson](https://academy.claude.com/courses/ai-fluency-for-nonprofits/researching-with-ai) is to ask Claude to track down the original source for one key claim in its summary and compare how accurately it was represented; a [literature review use case](https://academy.claude.com/use-cases/plan-your-literature-review) advises double-checking any quote central to your argument against the original paper before you submit the work.

**Know: drafting and brainstorming details.** Claude 101 sorts desktop work by shape: turn-by-turn work (asking, brainstorming, drafting) happens in chat, and a multi-step task that ends in a finished deliverable is handed off. The [Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/what-is-cowork) sums it up: "Chat is for thinking with Claude. Cowork is for delegating to Claude. Code is for building software with Claude." Where the new Claude experience has merged chat and Cowork into one conversation, read the split as two ways of working rather than two tabs (our reading; see the [D1.2 hand-off note](#d12-apply-task-decomposition-techniques-to-structure-complex-requests)). For brainstorming, set the terms of the collaboration. Across all the multi-turn Claude.ai conversations the Academy's [AI Fluency Index](https://academy.claude.com/tutorials/the-ai-fluency-index) sampled (not brainstorming alone), users told Claude how they would like it to interact with them in only 30%; the Index suggests adding instructions such as "Push back if my assumptions are wrong". For drafting, the Academy's [model tutorial](https://academy.claude.com/tutorials/choosing-the-right-claude-model) calls Sonnet "your versatile default" for coding, writing, analysis and multi-step workflows, and an Academy use case calls Sonnet models the everyday choice for drafting and Haiku models the quick option for a fast answer or a simple rewrite.

**Know: settings that change with the task (as of September 2026).** The help center's [model, effort and thinking article](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings) says the defaults work for everyday tasks and that simple questions and general writing do not need extra effort or thinking; for complex tasks, "raise the effort level, turn on thinking, or both." The same settings article says you can change the model, effort level or thinking setting at any point in a conversation, so you can raise or lower them as a task moves between steps (our reading). The effort levels, the signs of a wrong setting and the older Claude 101 video that says otherwise are under [D3.3](#d33-align-model-selection-with-task-requirements-cost-speed-quality).

!!! note "Product names as of September 2026"

    D1.4 names no product features; the objective is about matching the approach to the task. Where the guide does name the feature, in D3.1, it calls it "research mode", so expect that wording on the exam. How that term and "extended thinking" map to today's apps (Research, a separate thinking setting, and no web search toggle in the new Claude experience) is covered under [D3.1](#d31-select-appropriate-claude-product-features-projects-research-mode-chat-artifacts).

**Decide**

- If you will only know the next question after seeing the answer (brainstorming, drafting, thinking aloud), choose turn-by-turn chat; not a single handed-off task, because you could not have written the whole request up front.
- If the task needs one or two current facts, choose web search; if it needs a synthesis of many sources, choose Research (the guide's "research mode"); if it needs hard reasoning but no fresh information, choose more effort or thinking; not Research for everything, because Research uses limits faster.
- If an analytical task is new to you, choose to test Claude on a past case with a known answer before trusting it on new data; not a leap straight to the live report, because if Claude cannot match your known results, you have learned the task is one you should not delegate.
- If you are brainstorming, choose prompts that ask for options and pushback; not prompts that ask Claude to confirm your idea, because a request for validation is a known trigger for sycophancy.
- If you are drafting in your organization's voice, choose to supply past examples and the facts only you know; not a generic request followed by heavy rewriting, because past work shows Claude your organization's voice, and details such as your track record and partnerships are ones Claude cannot know.

**Traps**

- One prompting style for every task type.
- Research for a single fact that one web search would answer.
- A brainstorming prompt shaped as a request for agreement (*Isn't this a strong plan?*).
- Reading a striking correlation as cause and effect. The [Academy use case](https://academy.claude.com/use-cases/chart-your-data-before-you-commit) suggests asking Claude to quiz you and "catch me if I read causation into a confound."
- Treating research output as verified because it has citations. The [help center](https://support.claude.com/en/articles/10684626-enable-and-use-web-search) advises: "Cross-reference cited sources to understand the full picture."

**Go deeper:** [Search, Research and connectors](knowledge/claude-for-work.md#search-research-and-connectors)

## Domain 2: Output Evaluation and Validation

**Weight:** 21% of scored items, more than any other domain in the blueprint (the guide calls its percentages "the approximate proportion of scored items drawn from each domain"). If all 60 items were scored, that is about 12.6 items (21% of 60, our arithmetic); the guide does not say how many of the 60 are scored.

Domain 2 is the judgment half of using Claude at work: checking what comes back, catching fabricated or skewed content, deciding when a person must review it, adapting it for its reader, and choosing the form it takes. The [exam guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) Intended Audience section names the same skills: to "critically evaluate AI-generated content" and "recognize when human expertise, validation, or escalation is required". Official [Sample 1](#official-sample-questions) (a confident regulation summary that cites a specific subsection) maps to this domain.

The matching prep module is Evaluating & Validating Claude's Output (74 minutes), module 3 of 8 in the Associate prep course. Its [course page](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/evaluating-validating-claudes-output) states the goal as being able to "stand behind every deliverable you put your name on". Its six learning objectives restate the six objectives below almost word for word; the sixth reads "Select appropriate output formats and organize information for optimal results". The [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) own preparation advice is to "evaluate outputs for accuracy and bias".

### D2.1 Evaluate Claude-generated outputs for accuracy and completeness

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Evaluate Claude-generated outputs for accuracy and completeness". Accuracy asks whether what is there is right. Completeness asks whether anything that should be there is missing. In the survey reported in Anthropic's education report [How people check Claude's work](https://academy.claude.com/tutorials/discernment-toolkit), Academy respondents rated their confidence lowest for judging completeness (a mean of 3.5 on a 5-point scale), below judging correctness (3.7).

**Know: the Discernment lens.** The AI Fluency course defines Discernment as "Evaluating AI outputs and behavior with a critical eye" ([4D framework](https://academy.claude.com/courses/ai-fluency-framework-foundations/the-4d-framework)) and splits it three ways ([Discernment lesson](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-discernment)).

| Kind | What you judge | Workplace question (our examples) |
|---|---|---|
| Product Discernment | The output itself: accuracy, appropriateness, coherence, relevance | Are these the right numbers, and does this answer the question I asked? |
| Process Discernment | How the AI reached the output: logical errors, attention gaps, inappropriate reasoning | Did it skip a step, or ignore the second attachment? |
| Performance Discernment | How the AI behaves in the collaboration: is its communication style effective for you | Is it hedging so much that the answer is unusable, or agreeing too readily? |

**Know: a three-question review.** The Cowork course's [task loop lesson](https://academy.claude.com/courses/introduction-to-claude-cowork/the-task-loop) says to review a finished deliverable "the way you'd read a draft from someone you trust but don't yet fully know", and lists three things to check, especially the first few times: does it meet the actual objective (or is it subtly different from what you asked for), are the facts accurate, and does anything sound made up. For facts, it says to start by asking Claude to identify the docs it pulled from, then check them yourself. On the last question: "A specific date, name, or quote that you can't trace to an input is a flag, not a feature."

**Know: what Claude may not have seen.** A completeness failure can start with input Claude never read (our framing). As of September 2026 Anthropic's help center, Academy and docs document these limits.

| Input | What Claude reads | Source |
|---|---|---|
| Google Docs through the Drive connector | "the main text content only and cannot see images, comments, or suggestions" | [Google Workspace connectors](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors) |
| Gmail through the connector | Attachment content is not directly accessible ("metadata only") | [Google Workspace connectors](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors) |
| Uploaded PDF, 100 pages or fewer | Text and visual elements such as images, charts and graphics | [Upload files](https://support.claude.com/en/articles/8241126-upload-files-to-claude) |
| Uploaded PDF, 101 to 1000 pages | Text only; visual elements are not analyzed | [Upload files](https://support.claude.com/en/articles/8241126-upload-files-to-claude) |
| Other uploaded documents (not PDF) | Text only; embedded images are not read | [Upload files](https://support.claude.com/en/articles/8241126-upload-files-to-claude) |
| A very long document or conversation | Working memory is a fixed context window, and the lesson says this property "has a cliff rather than a gradient": silent truncation is the failure mode, "and you won't always be warned" | [Working memory lesson](https://academy.claude.com/courses/ai-capabilities-and-limitations/working-memory) |
| Key facts buried in the middle of long input | The lesson lists this as a separate limitation. Its "Lost in the middle" probe uses a longer document or a few pasted paragraphs, buries one important instruction in the middle, checks whether the AI caught it, then moves the instruction to the very top and compares | [Working memory lesson](https://academy.claude.com/courses/ai-capabilities-and-limitations/working-memory) |
| Anything after the model's training data | Knowledge is frozen at the cutoff. The help center gives Claude Sonnet 5 data up to January 2026 and Claude Haiku 4.5 data up to July 2025. For Haiku 4.5 the developer docs split this into a reliable knowledge cutoff of February 2025 (the date through which its knowledge is most extensive and reliable) and a training data cutoff of July 2025 (the broader range of data used); for Sonnet 5 both are January 2026 | [Training data article](https://support.claude.com/en/articles/8114494-how-up-to-date-is-claude-s-training-data); [models overview](https://platform.claude.com/docs/en/models/overview) |

**Know: make gaps visible.** An Academy status-report use case gives the instruction to add: "If you don't find information for a work stream, note that explicitly rather than omitting it." ([use case](https://academy.claude.com/use-cases/generate-project-status-reports)). Anthropic's [legal summarization guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/legal-summarization) does the same with "Not specified". Why ask: gaps do not stay empty. An Academy use case on drafting a requirements document warns that "Cowork fills gaps with reasonable guesses, and unchecked guesses are what design review catches." Its instruction: "Fix what you didn't decide, then send it." ([PRD use case](https://academy.claude.com/use-cases/prd-from-a-one-pager)).

**Know: test on work you already know.** [Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results) teaches the delegation diligence loop: find past data where you already completed the analysis, have AI reproduce it, evaluate what works and what doesn't, refine, and test again. "If AI can match your known results, you know how to use it and trust it for similar future tasks. And if not, you've learned that this task is something you shouldn't delegate." Its worked example shows why: the AI "correctly identified the correlation between program attendance and job placement, but it missed a critical insight around the combined housing assistance and job placement program" until the analyst asked it to pay attention to program type. The lesson also suggests noting patterns, such as "Claude gets the right numbers but misses the overall patterns". For outputs you will present, a [feedback-analysis use case](https://academy.claude.com/use-cases/analyze-patterns-in-user-feedback) adds an audit layer: a theme-classification tab that shows which feedback got tagged with which themes, described as "your validation layer before presenting findings to stakeholders."

**Know: polished output coincides with less checking.** Anthropic's [AI Fluency Index](https://academy.claude.com/tutorials/the-ai-fluency-index) found that in conversations where artifacts are created, users were less likely to identify missing context (-5.2pp), check facts (-3.7pp) or question the model's reasoning (-3.1pp). This is a correlation, and the Index names "several possible explanations for this pattern": finished-looking work may get treated as finished, artifact tasks may need less factual precision, or users may check artifacts in ways the data cannot see, such as running the code or sharing a draft with a colleague. Its advice: "When AI models produce something that looks good, it's the perfect moment to pause and ask: is this accurate? Is anything missing? Does this reasoning hold up?" For anything that computes, counts or recommends, the [artifacts tutorial](https://academy.claude.com/tutorials/use-artifacts-to-visualize-and-create-ai-apps-without-ever-writing-a-line-of-code) adds: "work one example out by hand and compare", because "An app that looks polished but gets the numbers wrong will mislead the people you share it with".

**Decide**

- If the output must be complete (a status report, a summary of every clause), choose an instruction to mark missing items explicitly and then check coverage against your own list of sources; not a clean-looking document taken at face value, because gaps get filled with reasonable guesses.
- If Claude worked from connected files or long PDFs, choose to confirm what it could read; not the assumption that it saw charts, comments or attachments.
- If you want to delegate an analysis you do regularly, choose to test Claude on past work whose answer you already know; not trust based on how the output reads.
- If the output looks finished and polished, choose more scrutiny, not less.
- If the output contains calculations, choose to recompute at least one by hand; not to accept the total.

**Traps**

- Judging an output by tone and formatting. Sample 1's rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) rejects reformatting because it "does not address correctness".
- Equating length or polish with completeness.
- Assuming Claude read an attachment, an embedded chart or a document comment.

**Go deeper:** [The AI Fluency framework](knowledge/claude-for-work.md#the-ai-fluency-framework)

### D2.2 Identify hallucinations, inconsistencies, and biases in responses

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Identify hallucinations, inconsistencies, and biases in responses". The guide's recommended experience includes "A practical understanding of AI limitations, including hallucinations, context constraints, and data sensitivity".

**Know: hallucination.** The [help center](https://support.claude.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on) calls incorrect or misleading output "hallucinating" information, "a byproduct of some of the current limitations of frontier Generative AI models, like Claude", and warns that "Claude can display quotes that may look authoritative or sound convincing, but are not grounded in fact." The [AI Capabilities and Limitations course](https://academy.claude.com/courses/ai-capabilities-and-limitations/next-token-prediction) tells you where to look: "Fabrication concentrates in specificity: names, dates, statistics, citations, URLs, quotes. The more precise a claim, the more it warrants verification." Sample 1's rationale says the same about citation numbers.

An Academy tutorial on [why AI models hallucinate](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate) lists the high-risk situations: specific facts, statistics or citations; topics that are obscure, niche or very recent; real but little-known people or places; and exact details such as dates, names or numbers. A second kind is the capability hallucination: the help center says that "Even if it claims otherwise, Claude does not have access to other tools or software that are not explicitly integrated, including email, word processors, or file transfers" ([help article](https://support.claude.com/en/articles/8241188-claude-is-producing-links-that-don-t-work-and-falsely-claiming-that-it-has-sent-emails-or-produced-external-documents-what-s-going-on)).

**Know: confidence is not evidence.** The [AI Capabilities and Limitations course](https://academy.claude.com/courses/ai-capabilities-and-limitations/how-ai-gets-its-character) lists among fine-tuning's side effects "loose calibration between stated confidence and actual reliability." Sample 1's rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) states the exam rule: self-reported confidence "is not a reliable accuracy signal". Anthropic's [introspection research](https://www.anthropic.com/research/introspection) points the same way: it found some introspective awareness in Claude models but stresses the capability "is still highly unreliable and limited in scope", and warns that models do not always accurately report their internal states: "in many cases, they are making things up!"

**Know: inconsistency.** Our working definition: an output that disagrees with itself, with the source, or with another run. Anthropic's [hallucination guide](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations) links the two failures: it describes hallucination as text that is "factually incorrect or inconsistent with the given context", and it uses inconsistency across runs as a detector: "Run Claude through the same prompt multiple times and compare the outputs. Inconsistencies across outputs could indicate hallucinations." The Academy's diligence-statement tutorial gives the everyday version: run the same question several ways and look for consistency, and for document tasks ask for direct quotes rather than paraphrase.

**Know: bias.** Anthropic's tutorial [Why does bias exist in AI models](https://academy.claude.com/tutorials/why-does-bias-exist-in-ai-models) says bias includes stereotyping and political bias, but "bias can also be less direct, like defaulting to certain types of answers or perspectives, or providing better quality responses in specific languages." Its worked case, political bias, can be obvious (refusing to explain one side of an issue when asked) or subtle (a more detailed answer for one viewpoint than another). The source is training: from the huge body of internet text a model learns from, it "might pick up a pattern that tilts it to one side of an issue or the other." The [Knowledge lesson](https://academy.claude.com/courses/ai-capabilities-and-limitations/knowledge) lists inherited bias in what counts as "default" among a model's characteristic knowledge failures. Bias can also sit in your input: in a feedback analysis, the Academy suggests asking whether you are over-indexed on hearing from any particular group.

**Know: sycophancy, the agreeable failure.** The guide does not name sycophancy, but Anthropic's Academy teaches it alongside these failures. The [sycophancy tutorial](https://academy.claude.com/tutorials/what-is-sycophancy-in-ai-models) defines it as telling you "what they think you want to hear, instead of what's true, accurate, or genuinely helpful": agreeing with your factual error, changing an answer based on how you phrased the question, or tailoring the response to your preferences. It is most likely when a subjective truth is stated as fact, an expert source is referenced, the question is slanted, validation is requested, emotional stakes are invoked, or the conversation gets very long. The tutorial's line: "We actually want AI models to adapt to your needs, just not when it comes to facts or wellbeing."

The summary below pairs each failure with the checks the sources above recommend; the workplace examples are ours.

| Failure | What it looks like at work | How you catch it |
|---|---|---|
| Hallucinated specific | A subsection number, statistic, quote or URL that is not in any source | Trace it to the source; if you cannot, treat it as unverified |
| Hallucinated capability | *I've sent the email to the team* with no email connector connected | Check that the action actually happened in the real system |
| Inconsistency | The summary's total differs from the table; the answer changes when you rephrase | Re-ask in different ways; reconcile the numbers against the source |
| Stale knowledge | *The current rule is...* for something that changed after the cutoff | Web search or the primary source |
| Bias | More depth for one side; an outsider's idea of what is normal; a skewed sample | Ask for balance and other angles; check the evidence yourself; ask for a breakdown by segment |
| Sycophancy | Claude agrees with a premise you got wrong | Neutral, fact-seeking phrasing; ask for counterarguments or an explicit invitation to disagree; a new conversation; a person you trust |

**Decide**

- If a response contains a precise, checkable detail (number, citation, name, date, quote) that will leave your hands, choose to verify it against its source; not to accept it because it reads confidently, because Sample 1's rationale rejects confidence as an accuracy signal.
- If two runs or two phrasings of the same question disagree, choose to treat that as a hallucination signal and go to the source; not to pick the answer you prefer.
- If a response on a contested topic seems one-sided, choose to push back, ask for a balanced treatment and examine the evidence yourself; not to assume neutrality.
- If Claude readily agrees with your premise, choose a neutral restatement or an explicit invitation to disagree; not to take the agreement as confirmation.

**Traps**

- Stated or self-rated confidence used as proof (options A and C in Sample 1).
- *Claude used web search, so it cannot be wrong.* The [help center](https://support.claude.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on) still says: "When working with web search results, users should review Claude's cited sources."
- Leading questions (*This proves X, right?*) that invite sycophancy, then reading the agreement as evidence.
- Treating bias as only political or offensive content, and missing default perspectives and skewed input data.

**Go deeper:** [Reducing hallucinations](knowledge/prompt-engineering.md#reducing-hallucinations)

### D2.3 Apply fact-checking and validation techniques

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Apply fact-checking and validation techniques". Sample 1's rationale names the core technique: validating factual claims "against an authoritative source is the diligence step required."

**Know: what checking looks like in practice.** In interviews with 129 Claude Academy participants reported in Anthropic's education report [How people check Claude's work](https://academy.claude.com/tutorials/discernment-toolkit), "51% of the participants we interviewed described verifying Claude's output against something external: source documents, official docs, their own data, another person, or even another AI." A read-through without any external validation was the minority practice. Half of those who described missing an error Claude had made said it changed how they work, and they restructured their workflows rather than checking harder: for example, requiring Claude to quote its source text first, before offering a recommendation or analysis, or creating approval gates at each step of a multi-step task instead of reviewing only at the end. The report itself notes the sample skews experienced and technical and makes no prevalence claims about AI users broadly.

**Know: techniques that reduce errors before they happen (in the prompt).**

| Technique | What you write | Source |
|---|---|---|
| Permission to not know | "It's ok if you don't know." Anthropic's guide says explicit permission to admit uncertainty "can drastically reduce false information." | [Hallucination tutorial](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate); [reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations) |
| Stay inside the documents | Tell Claude to use only the provided documents and not its general knowledge | [Reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations) |
| Quotes first | For long documents (over 20k tokens), ask for word-for-word quotes before the task, then work only from those quotes | [Reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations) |
| Ground in current information | Use web search "to ground responses in current information" when the topic is recent | [Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results) |
| Flag uncertainty | Add "flag anything you're not confident about" so you know where to look first | [PRD use case](https://academy.claude.com/use-cases/prd-from-a-one-pager) |

**Know: techniques that check an output after it arrives.**

| Technique | How | Source |
|---|---|---|
| Authoritative source | Check regulations, pricing data and deadlines against primary sources; double-check any quote central to your argument against the original | [AI Fluency for Small Businesses](https://academy.claude.com/courses/ai-fluency-for-small-businesses/researching-with-ai); [literature review use case](https://academy.claude.com/use-cases/plan-your-literature-review) |
| Open the citations | Web search answers include citations; cross-reference them and use authoritative sources for critical decisions | [Web search article](https://support.claude.com/en/articles/10684626-enable-and-use-web-search) |
| Claim by claim | Ask Claude to find a supporting quote for each claim and retract any claim it cannot support | [Reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations) |
| Trace one claim | Have Claude track down the original source for a key claim and compare how accurately it was represented | [AI Fluency for Nonprofits](https://academy.claude.com/courses/ai-fluency-for-nonprofits/researching-with-ai) |
| Ask differently | Run the same question several ways and compare | [Diligence statement tutorial](https://academy.claude.com/tutorials/writing-an-ai-diligence-statement) |
| Fresh eyes | Start a new chat and ask it to find errors in the answer and confirm the sources support the statements | [Hallucination tutorial](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate) |
| Recompute | For each statistic, compare what the source states with what Claude calculates from the raw data; work one example by hand; verify complex or mission-critical calculations with specialized software or manual methods | [Statistics use case](https://academy.claude.com/use-cases/verify-statistics-from-raw-data); [artifacts tutorial](https://academy.claude.com/tutorials/use-artifacts-to-visualize-and-create-ai-apps-without-ever-writing-a-line-of-code); [calculations article](https://support.claude.com/en/articles/10366421-how-does-claude-handle-mathematical-equations-and-calculations) |
| A second person | Build a "panel of experts": know who you would call for a second read in domains outside your expertise, and reach out before you ship | [How people check Claude's work](https://academy.claude.com/tutorials/discernment-toolkit) |

The same report's source-grounding follow-up turns several of these into one prompt ([How people check Claude's work](https://academy.claude.com/tutorials/discernment-toolkit)):

```text
For each factual claim in your answer, tell me where it came from. Quote the exact passage from the source and include the link or page number. If a claim comes from your general knowledge, label it "unsourced" so I know what to check first.
```

**Know: how to use confidence and reasoning without trusting them.** [Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results) suggests asking Claude to "cite sources or indicate confidence level", and the PRD use case asks it to flag what it is not confident about. Use those answers to decide where to look first (our advice). They do not replace the check: Sample 1's rationale says self-reported confidence is not a reliable accuracy signal. The same goes for the expandable "Thinking" section above a response, which shows "Claude's thought process summary and problem-solving approach". The [help center](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings) says reviewing it "can be valuable for verifying how Claude arrived at its conclusion", but Anthropic's [research on reasoning models](https://www.anthropic.com/research/reasoning-models-dont-say-think) cautions: "There’s no specific reason why the reported Chain-of-Thought must accurately reflect the true reasoning process". In that study, the researchers slipped a model a hint about the answer, confirmed it used the hint, and checked whether its reasoning mentioned it: "On average across all the different hint types, Claude 3.7 Sonnet mentioned the hint 25% of the time". Treat the Thinking section as a lead for your review, not as proof.

**Know: the limit.** Anthropic's guide on reducing hallucinations closes with: "Remember, while these techniques significantly reduce hallucinations, they don't eliminate them entirely. Always validate critical information, especially for high-stakes decisions." ([reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations)).

**Decide**

- If a claim is going to a compliance, legal, finance or customer audience, choose verification against the authoritative source; not asking Claude to rate its own confidence, because self-reported confidence is not a reliable accuracy signal.
- If the source document is long (the docs say over 20k tokens), choose quotes first and analysis only from those quotes; not a free summary you then have to audit line by line.
- If the claim concerns something recent, choose web search or the primary source; not the model's training knowledge, which stops at its cutoff.
- If the output contains numbers that matter, choose to recompute at least one; not to trust the arithmetic.
- If you are unsure about an answer, choose an independent check: a fresh chat asked to find errors and confirm the sources support the statements, or the source itself; not a bare *Are you sure?* whose reply you then accept, because self-reported confidence is not a reliable accuracy signal and sycophancy can change an answer based on how a question is phrased.

**Traps**

- The three wrong shapes from Sample 1: send it because Claude sounded sure, ask Claude to rate its confidence and send if high, or reword it to sound more formal.
- Citations taken as proof. A citation shows where a claim may come from; it does not show that the source supports it.
- The Thinking section read as proof of correct reasoning.

**Go deeper:** [Verifying Claude's output](knowledge/claude-for-work.md#verifying-claudes-output)

### D2.4 Determine when human review or additional verification is required

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Determine when human review or additional verification is required". The guide's opening section, About This Certification, adds the escalation half: "recognizing limitations and escalating more complex or technical work to Claude Architects and Developers."

**Know: review scales with stakes.** The Academy tutorial [Can you trust what AI tells you](https://academy.claude.com/tutorials/can-you-trust-what-ai-tells-you) describes trust in AI as "a dial you turn, not a switch you flip," with a doctor analogy: you would not question a recommendation to eat more carrots, but you would probably want a second opinion before surgery. The [model tutorial](https://academy.claude.com/tutorials/choosing-the-right-claude-model) makes it concrete: "check the result in proportion to what's riding on it: a quick lookup needs only a glance, while analysis that will shape a decision deserves a closer review."

**Know: the three-way split.** The AI Fluency for Nonprofits course sorts tasks into what AI can handle, what AI can assist with while a human decides ("Tasks where AI can draft or prepare, but you review before action"), and what a human should handle. For the last group it lists: "High-stakes decisions, emotional situations, complex judgment calls" ([workflow augmentation](https://academy.claude.com/courses/ai-fluency-for-nonprofits/workflow-augmentation)).

| Situation | Minimum check | Source |
|---|---|---|
| A quick lookup for your own use | A glance | [Academy model tutorial](https://academy.claude.com/tutorials/choosing-the-right-claude-model) |
| Analysis that will shape a decision | A closer review (our addition: verify the key figures) | [Academy model tutorial](https://academy.claude.com/tutorials/choosing-the-right-claude-model) |
| Regulatory, compliance or legal content | Verify against the official text; for legal work, keep a lawyer in the loop, verify against primary sources and document your AI use | Sample 1; [legal work article](https://support.claude.com/en/articles/15707726-using-claude-for-legal-work-privilege-confidentiality-and-how-to-think-about-configuration) |
| A final client deliverable, or audit-critical calculations | Human review for the deliverable; verification for the calculations | The [Claude for Excel docs](https://claude.com/docs/office-agents/excel.md) list "Final client deliverables without human review" and "Audit-critical calculations without verification" among uses it is not recommended for |
| Customer-facing messages or automations | Review before output reaches customers, be honest about AI's role, provide a clear path to a human | [AI Fluency for Small Businesses](https://academy.claude.com/courses/ai-fluency-for-small-businesses/tying-it-all-together) |
| High-stakes decisions, emotional situations, complex judgment calls | A person handles it | [AI Fluency for Nonprofits](https://academy.claude.com/courses/ai-fluency-for-nonprofits/workflow-augmentation) |
| A subject outside your expertise | A second read by someone who has it | [How people check Claude's work](https://academy.claude.com/tutorials/discernment-toolkit) |
| More complex or technical work: building against the API, enterprise-scale AI architecture or integration design | Escalate to a Claude Developer or Architect | The [exam guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) opening and audience sections |
| Regulated or personal data | Remove or anonymize identifiers, consistent with policy, before Claude sees them | Sample 3; [Domain 6](#domain-6-governance-risk-and-responsible-use) |
| A new agent or recurring job | A person reviews its output for the first few days; widen what it may do on a task only after it has handled that kind of task well several times in a row | [Building effective human-agent teams](https://academy.claude.com/courses/building-effective-human-agent-teams/practical-ways-to-get-started); [Domain 4](#domain-4-workflow-integration-and-solution-design) |

**Know: why expertise matters.** Anthropic's education report [How people check Claude's work](https://academy.claude.com/tutorials/discernment-toolkit) reports that among the Academy learners interviewed, "the biggest barrier to discernment was not time pressure but lack of domain expertise." It also reports that some non-technical participants were led by Claude into software-development solutions where "they couldn't tell good code or architecture decisions from bad ones." In the guide's terms, that kind of work is for escalation (our reading): the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) says Associates "are not expected to design enterprise-scale AI architectures or integrations", which belong to the Architect and Developer credentials.

**Know: accountability stays with you.** [Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results): "Just remember, validation builds confidence, but it doesn't eliminate responsibility." The AI Fluency course calls this [Deployment Diligence](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-diligence): "taking responsibility for verifying and vouching for the outputs we use or share". Anthropic's [consumer terms](https://www.anthropic.com/legal/consumer-terms) put it plainly: "You should not rely on any Outputs or Actions without independently confirming their accuracy."

**Decide**

- If an error would be costly, public, regulated or hard to reverse, choose human review or independent verification before use; not sending on the strength of Claude's confidence.
- If the question is a low-stakes lookup for yourself, choose a quick check; not a full review, because review should scale with what is riding on the result.
- If you lack the expertise to judge the output, choose a second read from someone who has it, or escalate; not your own sign-off.
- If the work has become more complex or technical (building against the API, or enterprise-scale AI architecture or integration design), choose to escalate to a Claude Developer or Architect; not to keep building it yourself.
- If verification lets the work proceed safely, choose to verify and proceed; not to abandon the task. The guide's samples reward the proportionate control: verify the citation before sharing (Sample 1), anonymize and continue rather than skip the analysis (Sample 3).

**Traps**

- *It's internal, so it doesn't need review.* Internal analysis that shapes a decision still needs a closer look. Sample 3's wrong option A leans on the same excuse ("since the analysis is internal"), and its rationale says uploading as-is "violates policy".
- A reviewer without the relevant expertise, treated as sufficient review.
- The two extremes the samples reject: shipping unchecked output (Sample 1, option A), and dropping the task when a proportionate control would make it safe (Sample 3, option D: "abandoning the task (D) is unnecessary when anonymization enables it").
- Treating a successful first run of a new agent or recurring job as proof it can run unreviewed.

**Go deeper:** [Human review and confidence calibration](knowledge/evaluation-and-reliability.md#human-review-and-confidence-calibration)

### D2.5 Edit, adapt, refine, and compare outputs for the intended audience

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Edit, adapt, refine, and compare outputs for the intended audience". The guide's Intended Audience section lists the same ability: to "adapt outputs for different audiences".

**Know: tell Claude who the reader is.** The AI Fluency course's Product Description covers "outputs, format, audience, and style" ([Description lesson](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-description)). [Claude 101](https://academy.claude.com/courses/claude-101/creating-with-artifacts) adds that "This flowchart is for new employees" leads to different results than "This flowchart is for the engineering team." For tone, [Claude 101's troubleshooting table](https://academy.claude.com/courses/claude-101/getting-better-results) says to describe it in plain language ("This should sound authoritative and formal") and give an example of the style; for length, to be explicit ("Give me a two-paragraph summary"). Anthropic's [latency guide](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-latency) notes that asking for paragraph or sentence count limits works better than an exact word count or word count limit.

**Know: adaptation patterns.**

- **Plain-language rewrite.** An [Academy use case](https://academy.claude.com/use-cases/adapt-content-across-platforms): "Rewrite it for everyday users who want main takeaways, not intricate details."
- **Same content, several levels.** A [reading-level use case](https://academy.claude.com/use-cases/adapt-a-standard-textbook-page-to-every-reading-level) produces three versions of a one-page handout with the same concepts but different vocabulary and sentence length.
- **Executive and detailed versions.** The [diligence-statement tutorial](https://academy.claude.com/tutorials/writing-an-ai-diligence-statement) asks for a 2 to 3 paragraph version for the full internal report and a 2 to 3 sentence version for the executive summary slide deck.
- **Your voice.** The first use case says to upload 2 to 3 pieces that represent your style, and Claude mirrors tone, structure and phrasing patterns.

**Know: review what changed.** An Academy use case gives a one-line instruction for this objective: "Whenever Claude is rewriting, simplifying, or adapting something you gave it, add a line to the prompt asking for a list of what it changed and what it left out. That list is what you review." ([reading-level use case](https://academy.claude.com/use-cases/adapt-a-standard-textbook-page-to-every-reading-level)).

**Know: compare, then refine.** The AI Fluency course's [Discernment lesson](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-discernment) has you ask for three explanations of the same aspect of a topic you know well, judge them with your own expertise, and "Identify the weakest explanation and provide specific feedback on what makes it problematic". An AI Fluency exercise for small businesses asks for one topic explained for three audiences (a customer, a potential business partner, a new employee) and then asks: "Did the shifts in audience actually land, or did it just change vocabulary?" ([exercise](https://academy.claude.com/courses/ai-fluency-for-small-businesses/ai-capabilities-and-limits)). For revisions, an [AI Fluency for K-12 educators exercise](https://academy.claude.com/courses/ai-fluency-for-k-12-educators/creating-high-quality-ai-outputs-in-your-teaching-practice) says to write one concrete revision prompt (not "make it better") and compare the revision to the original. When you build a skill with skill-creator, the Cowork course explains, it produces a pair of outputs for each realistic test prompt, one with your skill and one without, so you judge not just "is this output okay," but "is this output better than what Claude would have done on its own." ([validating skills](https://academy.claude.com/courses/introduction-to-claude-cowork/validating-skills-for-plugins)).

**Know: editing mechanics (as of September 2026).** [Claude 101](https://academy.claude.com/courses/claude-101/creating-with-artifacts) lists three ways to change an artifact, which you can mix: edit it directly, leave a comment for Claude on the exact element, slide or passage you want changed, or just ask in the conversation (direct editing and commenting are part of the design, deck and doc experiences). The [artifacts article](https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them) adds that for Markdown documents you can highlight text, click "Edit with Claude" and type your request, and a version selector switches between versions. In [Claude Docs](https://support.claude.com/en/articles/16923645-get-started-with-claude-docs) you mention @Claude in a comment, and Claude "replies in the thread, makes the change, and explains what it did and why." In its section on the artifact window beside the chat, the artifacts article also notes: "Your edits won't change Claude's memory of the original content." Our inference: for those chat artifacts, tell Claude what you changed before asking for more.

**Know: the final pass is yours.** The AI Fluency for Nonprofits course has you check whether a draft's language reflects your organization's actual voice or sounds generic, and whether it uses deficit-based framing or problematic language about the people you serve. Its standard: "you should be able to stand behind every word because you applied discernment throughout" ([writing with AI](https://academy.claude.com/courses/ai-fluency-for-nonprofits/writing-with-ai)).

!!! note "A reusable voice, as of September 2026"

    For tone you want every time, an Academy [use case](https://academy.claude.com/use-cases/my-voice) builds a personal voice skill in Cowork and improves it as you go: whenever you edit a draft Cowork wrote because you would not say it that way, feed the correction back and tell Cowork to add the rule to the skill. Older material shows a "Use style" menu for tone instead; it has been deprecated (see the box under [D5.3](#d53-create-effective-system-level-instructions), where setting up skills and instructions is covered).

**Decide**

- If the same content goes to several audiences, choose separate versions compared for substance; not one version with swapped vocabulary.
- If Claude rewrote or simplified your material, choose to ask for a list of changes and omissions and review that list; not a line-by-line comparison you will skip.
- If you edited a chat artifact yourself, choose to tell Claude what you changed before asking for more (our inference from the help center's note that your edits don't change Claude's memory of the original); not to assume it saw your edits.
- If a revision is needed, choose concrete instructions (which section, what change, for whom); not *make it better*.
- If you make the same tone corrections every week, choose a skill or standing instructions; not the deprecated Styles menu.

**Traps**

- A simplification that silently drops facts the audience needs.
- An audience shift that changes only the words, not the emphasis or the level of detail.
- Vague revision requests (*this isn't quite right*), which give Claude nothing to act on.
- Sending adapted output you cannot personally stand behind.

**Go deeper:** [Memory, styles and personalization](knowledge/claude-for-work.md#memory-styles-and-personalization)

### D2.6 Organize and curate information and select appropriate output formats (artifacts, inline, structured data)

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Organize and curate information and select appropriate output formats (artifacts, inline, structured data)". The guide's candidate profile expects familiarity with "Projects, Artifacts, and workflow-based interactions".

**Know: when Claude makes an artifact.** The [artifacts article](https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them) describes an artifact as "anything Claude makes for you that you'd put in front of someone", which "opens beside your conversation, and you can edit it, come back to it, and share it with a link." The criteria it lists for when Claude creates one include content that "is significant and self-contained, typically over 15 lines", and content that "is something you're likely to want to edit, iterate on, or reuse outside the conversation." [Claude 101](https://academy.claude.com/courses/claude-101/creating-with-artifacts) says to ask for the deliverable, not just the content: a request to summarize Q3 results gets a chat reply, while a request for a one-page doc for the leadership team gets an artifact. To be sure: "Create this as an artifact."

| Format | Choose it when | Know (as of September 2026) |
|---|---|---|
| Inline reply | A quick answer, or you are still thinking turn by turn | A request for a summary alone gets a chat reply |
| Artifact | The content is substantial and you will edit, iterate on, share or reuse it | Needs Code execution and file creation enabled; counts toward usage limits; a version selector switches between versions; common types include single-page HTML websites, diagrams and flowcharts, and interactive React components |
| Structured data | Others will sort, compare, filter or audit the result, or it feeds another tool | Define the format precisely (JSON, XML or a custom template) and give examples of the output you want; Academy examples include a side-by-side comparison spreadsheet and a decision log with fixed fields |
| File | The reader needs it in Excel, PowerPoint, Word or PDF | Claude creates .xlsx, .pptx, .docx and PDF files, up to 30MB each, to download or save to Google Drive; in one Academy example the chat preview shows only basic structure, while the Excel file carries the formatting, color-coding and cell notes |
| Inline visual | A diagram, chart or interactive element would explain better than text | Custom visuals are in beta on web and desktop and are ephemeral by default; to keep one, copy it as an image, download it as .svg or .html, or save it as an artifact; they are built with HTML and SVG, and Claude doesn't generate photos or illustrations the way image-generation tools do |

**Know: artifact or file.** Claude 101 draws the line: "an artifact opens and updates right in Claude and is shared by link, while file creation hands you a file to download and open in other apps." Artifacts export when you need both: "a deck to .pptx or .pdf, a design to .pptx, .pdf, or .html, a doc to Google Docs or .docx" ([creating with artifacts](https://academy.claude.com/courses/claude-101/creating-with-artifacts)). On the Free plan, an artifact stays with the conversation that created it.

**Know: organizing and curating.** Academy use cases show how to impose structure on many inputs:

- Name the sections and their order, quote the source for any decision already made, and flag anything contradictory (a design spec assembled from a folder).
- Ask for a fixed record with no commentary: decision, owners, open questions, links back to the source messages (a decision log).
- Set a chart limit. In the [metrics narrative use case](https://academy.claude.com/use-cases/metrics-narrative), "Two charts that prove it" keeps output to the charts that carry the argument, "instead of a dashboard dump".
- Charts in a doc are snapshots. The [Claude Docs article](https://support.claude.com/en/articles/16923645-get-started-with-claude-docs) says "Charts and diagrams don't update on their own, even when the data comes from a connected app like Salesforce or Google Sheets"; Claude 101 says to ask Claude to pull fresh data from your connected tools and update the chart.
- Pick the chart's destination. A chart made in chat can be copied as an image for slides, or saved as an artifact "if you want something interactive to share with collaborators" ([chart use case](https://academy.claude.com/use-cases/chart-your-data-before-you-commit)).
- Say when presentation matters. An Academy use case warns that when you simply ask Claude to create a spreadsheet, "it might default to basic formatting" ([design plans use case](https://academy.claude.com/use-cases/turn-inspiration-to-design-plans)).

**Know: sharing consequences (as of September 2026).** The [publishing article](https://support.claude.com/en/articles/9547008-publish-and-share-artifacts) separates two routes:

- **Shared from the Share dialog** (everything made in the new Claude experience, and everything made with Claude Design, Slides, Docs or Claude Code): "People without a Claude account can't open or interact with a shared artifact, even if they have the link." Claude 101 agrees for designs, decks and docs ("Anyone you share with needs a Claude account to open the artifact."), but the same lesson also says that on Pro and Max plans you "can publish a design or a deck as a public artifact". The help center says publishing is "available only for artifacts made in chat". Where the two differ, follow the help center, whose article covers publishing and sharing in detail (our judgment).
- **Published from chat** (Free, Pro and Max, for artifacts made in chat in the previous experience): non-users can view and interact with a published artifact without signing up, and are prompted to sign up only for advanced features such as AI-powered capabilities.

The article sums it up: "Even with External sharing on, people need a Claude account to open any artifact except one published from chat." In its section on sharing chat artifacts on Team and Enterprise plans, it also warns that "When you share an artifact made in chat, viewers also get access to any attachments and files in the conversation that created it", so consider this before sharing from a conversation that holds sensitive documents. That is a data question; see [Domain 6](#domain-6-governance-risk-and-responsible-use).

!!! note "Artifacts as of September 2026"

    The guide (effective July 2026) lists three output formats: "artifacts, inline, structured data". Today the [artifacts article](https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them) defines an artifact as "anything Claude makes for you that you'd put in front of someone: a design, a deck, a document, a dashboard, or a small interactive tool", and says "Claude Design, Claude Slides, and Claude Docs are in beta and available on paid plans": on by default on Pro, Max and Team, and off on Enterprise until an owner turns each one on. Artifacts also now require Code execution and file creation to be enabled, and sharing rules depend on which experience made the artifact (above). On the exam, expect the guide's three labels (artifacts, inline, structured data). The guide does not define an artifact; a working distinction (our reading) is a standalone deliverable you iterate on and share, as opposed to an inline reply or structured data.

**Decide**

- If you will iterate on it or share it, choose an artifact; not an inline reply you will have to copy out.
- If the reader works in Excel, Word or PowerPoint, choose a file (or export the artifact); not a link they may not be able to open.
- If others will sort, compare or audit the result, choose structured data with defined columns or fields and an example; not prose.
- If it is a quick answer or you are mid-thought, choose inline; not an artifact, which is meant for significant, self-contained content (typically over 15 lines) you are likely to edit, iterate on or reuse.
- If a downstream system must parse the output reliably every time, choose to involve a Claude Developer; guaranteed schema conformance comes from the API's Structured Outputs feature, and building against the API or designing enterprise-scale integrations is beyond the Associate scope (our mapping of the guide's escalation rule).

**Traps**

- Everything as an artifact, including one-line answers.
- Expecting a chart in a doc to update itself.
- Sending a Share-dialog artifact link to an external reader who has no Claude account.
- Sharing an artifact made in a chat that also holds sensitive attachments.
- Expecting a custom visual to still be there later when you have not copied, downloaded or saved it as an artifact.
- Expecting Claude to generate photos or illustrations the way image-generation tools do.

**Go deeper:** [Artifacts, files and code execution](knowledge/claude-for-work.md#artifacts-files-and-code-execution)

## Domain 3: Product and Model Selection

**Weight:** 12% of scored items. If all 60 items were scored, that is about 7.2 items (12% of 60, our arithmetic); the guide does not say how many of the 60 are scored.

Domain 3 covers the choices you make before you write a prompt: which Claude feature fits the task, which model, and how to handle the limits of a conversation's context. Official [Sample 2](#official-sample-questions) (high-volume customer-reply drafts where speed and cost matter more than deep reasoning) maps to this domain.

The prep module whose objectives line up with this domain (our mapping) is Claude Platform & Model Foundations (59 minutes), module 1 of 8 in the official prep course. Its [course page](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/claude-platform-model-foundations) frames four decisions made before the first prompt: "entry point, capability features, model, and context". Its four learning objectives are to:

- "Select the appropriate Claude entry point and feature set for a given professional task"
- "Differentiate Haiku, Sonnet, and Opus models by their capability characteristics and task fit"
- "Match model selection to task requirements, including quality, speed, and volume trade-offs"
- "Manage context limitations and use memory features to maintain continuity across sessions"

Some products in this domain have changed since the guide took effect in July 2026: on September 16, 2026 chat and Cowork began merging into one Claude (rolling out first to Pro and Max), and newer model versions have been released. The guide's "research mode" is the feature the help center calls Research. Each objective below teaches the concept in the guide's terms and marks where today's product differs.

### D3.1 Select appropriate Claude product features (Projects, research mode, chat, artifacts)

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Select appropriate Claude product features (Projects, research mode, chat, artifacts)". The guide's intended audience is professionals who "build Claude Projects in their day-to-day roles".

**Know: the four features.**

| Feature | Choose it for | Key facts (as of September 2026) |
|---|---|---|
| Chat | One-off questions, a thought partner, asking, brainstorming, drafting or thinking something through turn by turn; tasks light on context | Available on every plan, including Free |
| Projects | Recurring work that needs the same background documents and instructions in every conversation | Self-contained workspaces with their own chat histories and knowledge bases; instructions apply to every chat in the project (the [personalization article](https://support.claude.com/en/articles/10185728-understanding-claude-s-personalization-features) marks project instructions "paid plans only"; see [D5.1](#d51-configure-claude-projects-with-instructions-and-knowledge-sources)); chats in a project do not share context unless the information is added to project knowledge; each project has its own memory; projects are available on every plan, and Free users can create up to five |
| Research (the guide's research mode) | Questions that need a synthesis of many sources, on the web and in connected apps | Works agentically, running multiple searches that build on each other; needs web search turned on (the new Claude experience has no web search toggle); paid plans only (Pro, Max, Team, Enterprise); help-center guidance positions it for information gathering that needs five or more tool calls over 1 to 3 minutes, while the current [Claude 101 Research lesson](https://academy.claude.com/courses/claude-101/research-mode-for-deep-dives) says a run takes "a few minutes or more, depending on the question"; can use limits faster than ordinary chats |
| Artifacts | Standalone deliverables you will edit, iterate on, share or reuse | Anything Claude makes that you would put in front of someone (a design, deck, document, dashboard or small interactive tool); opens beside the conversation and can be shared by a link; see [D2.6](#d26-organize-and-curate-information-and-select-appropriate-output-formats-artifacts-inline-structured-data) for when to choose one |

**Know: Projects in more detail.** How project knowledge behaves (retrieval on paid plans, caching, what Claude cannot see, what chats in the same project share) and how to set a Project up are taught once, under [D5.1](#d51-configure-claude-projects-with-instructions-and-knowledge-sources). For choosing the feature, [Claude 101](https://academy.claude.com/courses/claude-101/working-with-skills) separates Projects from skills: "projects store knowledge, skills perform tasks."

**Know: Research and its neighbors.** The help center's [Research article](https://support.claude.com/en/articles/11088861-use-research-on-claude) says "Claude operates agentically, conducting multiple searches that build on each other while determining exactly what to investigate next", and that it can draw on connected internal sources (such as Gmail, Google Calendar and Google Docs) as well as the web. Three nearby tools answer different needs (the help article [When should I use web search, extended thinking, and research?](https://support.claude.com/en/articles/11095361-when-should-i-use-web-search-extended-thinking-and-research) compares the first two with Research):

- **Web search:** "Web search is best for straightforward, factual queries that can be answered with one or two tool calls"; every response includes citations.
- **Extended thinking:** "Extended thinking shines when tackling complex reasoning tasks that don't require recent info from the web". The current [settings article](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings) describes this control as the thinking setting in the model menu, which sits beside the model and effort level.
- **Enterprise search** (Team and Enterprise) appears as a pre-configured "Ask Your Org" project that searches the internal sources your organization has connected. The [enterprise search article](https://support.claude.com/en/articles/12489464-use-enterprise-search) describes it as "Optimized for quick knowledge retrieval", while Research is "Designed for deep, multi-step research on specific topics".

During Research, the [custom connectors article](https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp) warns, "Claude can invoke tools from your connectors automatically without further approval", and advises disabling any tools that can take write actions in external applications.

!!! warning "Exam guide vs current docs"

    **The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) (July 2026)** names four features: "Projects, research mode, chat, artifacts".

    **The product today (as of September 2026):** the help center calls the feature Research. Anthropic's [blog](https://claude.com/blog/cowork-is-now-claude) announced on September 16, 2026: "Starting today, Claude Cowork and chat are merging into one Claude." The [help center](https://support.claude.com/en/articles/16761823-claude-cowork-and-chat-are-one-claude) says the new experience is rolling out gradually, starting with Pro and Max, and that once an account has it, you cannot switch back to separate "Chat" and "Cowork" options; the blog says Team and Free plans will follow soon, with Enterprise admins told at least 30 days before anything changes. In the new experience, Claude "can decide which tool to use", there is no web search toggle, and Research starts with `/deep-research` or from the "+" menu. The same announcement introduced Claude Docs and Claude Slides, in beta on paid plans. A new version of Projects, in which a project is one conversation that splits work into parallel threads in the cloud, is in beta for select Pro and Max subscribers who use Claude Code.

    **On the exam, answer in the guide's terms:** chat for turn-by-turn work, Projects for persistent context and instructions, research mode for multi-source investigation, artifacts for standalone deliverables.

**Decide**

- If the same instructions and reference documents are needed across many conversations, choose a Project; not re-uploading the files into each chat.
- If a question needs a synthesis of many sources, choose Research; not a single web search, and not the model's training knowledge.
- If you need one or two current facts, choose web search; not Research, which can use your limits faster.
- If you are thinking, drafting or brainstorming turn by turn, choose chat.
- If the output is a standalone deliverable you will edit and share, choose an artifact.
- If a file matters for one conversation only, choose to upload it in that conversation; [Claude 101](https://academy.claude.com/courses/claude-101/introduction-to-projects) says it "stays separate from your project knowledge."

**Traps**

- Research for a lookup that one search would answer.
- Expecting a new chat in a Project to know what an earlier chat in the same Project discussed; only project knowledge is shared reliably (project memory, and chat search on paid plans, may pick some of it up, but do not count on them).
- Writing instructions into a Project's name or description, which Claude cannot see.
- Expecting Research on the Free plan.
- Running Research with connector tools that can write to external systems, when those tools can run without further approval.

**Go deeper:** [Projects](knowledge/claude-for-work.md#projects)

### D3.2 Differentiate between Claude model types (Haiku, Sonnet, Opus)

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Differentiate between Claude model types (Haiku, Sonnet, Opus)". The Anthropic [pricing page](https://platform.claude.com/docs/en/about-claude/pricing) gives the rule in one sentence: "Choose Haiku for simple tasks, Sonnet for most production workloads, and Opus for the most complex reasoning".

**Know: the three tiers (as of September 2026).**

| Tier (current model) | Description in the [models overview](https://platform.claude.com/docs/en/models/overview) | Comparative latency | Academy tutorial: best for | Academy tutorial: limit use | API list price per million tokens, input / output | In the apps |
|---|---|---|---|---|---|---|
| Haiku (Claude Haiku 4.5) | "The fastest model with near-frontier intelligence" | Fastest | Quick answers, summaries and simple extraction | Lightest | &#36;1 / &#36;5 | Included on Free |
| Sonnet (Claude Sonnet 5) | "The best combination of speed and intelligence" | Fast | Coding, writing, analysis and multi-step workflows; the versatile default | Moderate | &#36;2 / &#36;10 | Included on Free |
| Opus (Claude Opus 5.5) | "For long-running agentic coding and knowledge work" | Moderate | Deep research and complex reasoning; problems where Sonnet struggled | Heavy | &#36;4 / &#36;20 | Not on Free; included on Pro and Max |

The descriptions and latency ratings come from the [models overview](https://platform.claude.com/docs/en/models/overview), and the prices are API list prices from the same page (in the apps you pay through your plan and its usage limits). Sonnet 5's &#36;2 / &#36;10 began as an introductory price; the [pricing page](https://platform.claude.com/docs/en/about-claude/pricing) says a scheduled increase to &#36;3 / &#36;15 will not occur. The "best for" and "limit use" columns come from the Academy tutorial [Choosing the right Claude model](https://academy.claude.com/tutorials/choosing-the-right-claude-model), which describes Opus as for "Deep research and complex reasoning that genuinely needs sustained thinking" and says of Sonnet: "If you're not sure which model to pick, start here." The same tutorial says Free includes Haiku and Sonnet, while Pro and Max add Opus and Fable.

**Know: other differences that show up in questions.**

- **Context window in chat (paid plans):** Claude Opus 5.5 and Claude Sonnet 5 have 1M tokens; models not on the help center's list (Haiku 4.5 is not on it) have 200K, which the [help center](https://support.claude.com/en/articles/8606394-how-large-is-the-context-window-on-paid-claude-plans) describes as "about 500 pages of text or more".
- **Knowledge cutoff:** the help center gives training data up to June 2026 for Opus 5.5, January 2026 for Sonnet 5 and July 2025 for Haiku 4.5, and warns that models may not know about events after their cutoff dates. The [developer docs](https://platform.claude.com/docs/en/models/overview) give a separate, earlier reliable knowledge cutoff of February 2025 for Haiku 4.5. Our inference: a question about recent events is riskier on Haiku without web search.
- **Thinking and effort:** thinking cannot be turned off in Claude on Opus 5.5; in the developer docs, Haiku 4.5 does not support the effort parameter.
- **Fallbacks:** on Opus 5 and Opus 5.5, a narrow set of higher-risk requests falls back to a less capable model or is blocked; for a fallback, a notice appears and the response is labeled with the model that answered.
- **Organization controls:** on Enterprise, an administrator can disable a model or cap its effort level for the whole organization or for a custom role, but Haiku models are always available to every member and cannot be disabled.

!!! warning "Exam guide vs current docs"

    **The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) (July 2026)** asks you to differentiate three model types: "Haiku, Sonnet, Opus". The prep course's model module does the same.

    **The docs today (as of September 2026)** list four headline models: Claude Fable 5.1, Claude Opus 5.5, Claude Sonnet 5 and Claude Haiku 4.5. Fable sits above Opus: the Academy [model tutorial](https://academy.claude.com/tutorials/choosing-the-right-claude-model) calls Fable 5.1 "our most capable model", and the [models overview](https://platform.claude.com/docs/en/models/overview) describes it as "For demanding reasoning and long-horizon agentic work", with "Slower" latency and API prices of &#36;10 / &#36;50 per million tokens. Fable models are available on paid plans only. Claude Opus 5.5 was released on September 22, 2026 and Claude Fable 5.1 on September 1, 2026, both after the guide. The Fable tier itself is older than the guide: Claude Fable 5 was released on June 9, 2026, yet the guide names only Haiku, Sonnet and Opus.

    Advice on the default model also differs by source. [Claude 101's](https://academy.claude.com/courses/claude-101/your-first-conversation-with-claude) December 2025 video calls Sonnet "our recommended default model"; the current Academy model tutorial says to start with Sonnet if unsure; the [developer docs](https://platform.claude.com/docs/en/models/overview) recommend starting with Claude Opus 5.5 "for most workloads", which is advice for building on the API.

    **On the exam, answer in the guide's terms:** Haiku is the fastest, lowest-cost tier for simple, high-volume work; Sonnet is the balanced everyday default; Opus is the most capable of the three, for complex reasoning. Learn the trade-off, not the version numbers.

**Decide**

- If the task is simple, high-volume or speed-sensitive (short answers, categorization, extraction, summaries), choose Haiku; not Opus, which the [Academy tutorial](https://academy.claude.com/tutorials/choosing-the-right-claude-model) says "costs you tokens for no gain" on a task Haiku could handle.
- If the task is everyday writing, analysis or multi-step work, or you are unsure, choose Sonnet.
- If the task is complex reasoning or deep analysis of long, specialized documents, or you tested Sonnet and it struggled, choose Opus.
- If an item describes a tier above Opus, read it as the most capable, slowest and most expensive option and apply the same trade-off logic (our rule of thumb, from the Fable facts above).

**Traps**

- *The biggest model is always best.* Sample 2's rationale rejects always using the top model.
- Confusing a model with a feature: Research, Projects and artifacts are product features; Haiku, Sonnet and Opus are models you pick in the model menu.
- Choosing by version number instead of by the tier's trade-off.
- Assuming Opus is available on the Free plan.

**Go deeper:** [Choosing a model in the apps](knowledge/claude-for-work.md#choosing-a-model-in-the-apps)

### D3.3 Align model selection with task requirements (cost, speed, quality)

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Align model selection with task requirements (cost, speed, quality)". It echoes the guide's purpose statement: to "Select appropriate approaches to balance quality, efficiency, and cost".

**Know: the official rule.** Sample 2's rationale defines the objective: aligning model selection with task requirements means matching a faster, lower-cost model to straightforward, high-volume work and reserving the most capable model for complex reasoning. Anthropic's [choosing a model page](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model) says the same: "Choosing a Claude model means balancing capabilities, speed, and cost." Its efficiency-first path starts with a faster, cheaper model such as Claude Haiku 4.5 for "High-volume, straightforward tasks", and its capability-first path suits "Applications where accuracy outweighs cost considerations".

**Know: effort is a second lever.** The same page says: "Tuning effort is often a better lever than switching models." The [help center](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings) describes the effort levels in the apps:

| Effort level | Use it for |
|---|---|
| Low and Medium | Routine tasks; they stretch your usage further |
| High | The best overall balance of quality and speed |
| Extra high (xhigh) | Long-running coding and agentic tasks; deeper reasoning than High without the full token cost of Max; available on Opus 4.7 and newer models |
| Max | Tasks that need the deepest possible reasoning and most thorough analysis |

Higher effort gives more thorough responses, but they take longer and use more tokens, so you reach usage limits faster. Thinking and effort are separate settings that the help center says can be combined, within what each model allows (thinking cannot be turned off on Opus 5.5, Fable 5.1 or Opus 5). The Academy's [effort tutorial](https://academy.claude.com/tutorials/how-to-select-the-right-effort-setting-for-claude-cowork-and-chat) names the signs of a wrong setting: with too little effort, "Instructions get missed, or long work wraps up before it's finished"; with too much, "Responses become more verbose without improving the quality, or the scope expands past what you asked for." It adds that "A frontier model at medium or low effort often outperforms an older model at high or maximum effort". Judging cost per completed task rather than per token is covered under [D7.3](#d73-optimize-workflows-for-efficiency-and-effectiveness).

**Know: what else uses your limit.** Model choice and effort level are two of several factors, alongside tools such as Research and web search, file creation and conversation length, that draw down one usage limit shared by claude.ai, Claude Code and Claude Desktop. The full list and the levers that save usage are under [D7.3](#d73-optimize-workflows-for-efficiency-and-effectiveness).

**Decide**

| Requirement that dominates | Choose | Not |
|---|---|---|
| High volume of simple output where speed and cost matter more than deep reasoning (Sample 2) | A faster, lower-cost model | The most capable model for every item |
| Accuracy on complex reasoning, where an error is costly | Opus, or higher effort and thinking on the current model | The cheapest model to save usage |
| Instructions get missed or long work stops before it is finished | Raise effort, ask again, and check that the new answer follows your instructions; move up a tier if a tested model still struggles | Leaving effort low and accepting the partial answer |
| Responses grow wordier or wander past the request without getting better | Lower effort | A bigger model |
| You want to save usage on a newer model | Lower effort first | Switching to an older model at maximum effort |
| Sonnet was tested on the task and struggled | Opus | Repeating the same prompt on Sonnet |

A practical test from the Academy [model tutorial](https://academy.claude.com/tutorials/choosing-the-right-claude-model): run a task you know well on Haiku, then in a new chat on Sonnet, and "Compare where the answers differ, not how long they are." If Haiku's answer covers everything you would have flagged, the task is suited to Haiku.

!!! note "Switching models, as of September 2026"

    [Claude 101's](https://academy.claude.com/courses/claude-101/your-first-conversation-with-claude) December 2025 video says "Changing the model will result in a new chat." The current [help center](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings) says: "You can change the model, effort level, or thinking setting at any point in a conversation. Changes apply starting with Claude's next response." Use the help center. Fable models also count differently by plan: on Max and Team Premium seats up to 50% of weekly limits can go to Fable models at no extra cost, while on Pro and Team Standard seats Fable runs on usage credits.

**Traps**

- The three wrong shapes in Sample 2's rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf): always using the top model "wastes the cost and latency budget", and disabling features or switching platforms "does not address the trade-off".
- Comparing models on price per token instead of cost per completed task.
- Forgetting that Research, web search, file creation and high effort draw down the same usage limit.
- Believing you must start a new chat to change the model.

**Go deeper:** [Models and how to choose one](knowledge/claude-api.md#models-and-how-to-choose-one)

### D3.4 Understand and manage context limitations and memory considerations (when to restart, summarize, or persist)

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Understand and manage context limitations and memory considerations (when to restart, summarize, or persist)". The guide's recommended experience lists "context constraints" among the AI limitations a candidate should understand.

**Know: how context works in a conversation.** The help center defines the context window as "the amount of information Claude can work with in a single chat". On paid plans, the newest models support up to 1M tokens and others 500K or 200K; a portion is always reserved for Claude's response ([usage and length limits](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work)). The same article separates two limits: usage limits control "how much you can use Claude across all your conversations", while length limits control "how long any single conversation can become." The Academy's [context tutorial](https://academy.claude.com/tutorials/parametric-memory-and-context) notes that "most of what you pay for in a long conversation is context you didn't write" (the system prompt and material Claude pulls in). Every message also re-sends the whole conversation, which is why, in its words, "starting a new chat can be the cheapest and fastest way to get an answer."

**Know: how context fails.** The [Working memory lesson](https://academy.claude.com/courses/ai-capabilities-and-limitations/working-memory) is blunt: "Silent truncation is the failure mode, and you won't always be warned." It names the limitation zone as "very long documents or conversations, expecting continuity across sessions, burying critical info in the middle of long input", and says "The model doesn't learn from your corrections. It only responds to what's currently in context." The course's [try-it-out lesson](https://academy.claude.com/courses/ai-capabilities-and-limitations/try-it-out-q7hdjm9twcbt) advises: "put your most important instructions at the beginning and end of the context", citing a 2023 Stanford finding that accuracy dropped by more than 30% when a key fact was buried in the middle. Long conversations also bring drift ("long-conversation drift" in the course's [When properties collide](https://academy.claude.com/courses/ai-capabilities-and-limitations/when-properties-collide) lesson), and the Academy's [sycophancy tutorial](https://academy.claude.com/tutorials/what-is-sycophancy-in-ai-models) lists a very long conversation among the conditions in which sycophancy is most likely. The Working memory lesson adds: "Memory features, compaction, projects, larger windows, and multi-agent workflows all exist to push this cliff further out."

**Know: what the app does near the limit.** When a conversation approaches the context window limit, "Claude summarizes earlier messages" so it can continue; this automatic context management requires code execution to be enabled (the [error-messages article](https://support.claude.com/en/articles/12466728-troubleshoot-claude-error-messages) describes it for users on paid plans), "Your full chat history is preserved so Claude can reference it even after summarization", and longer conversations that trigger it consume more of your usage limit ([usage and length limits](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work)). The Academy's [context tutorial](https://academy.claude.com/tutorials/parametric-memory-and-context) draws the line between the two mechanisms: "Compaction is how Claude can continue a single conversation beyond the context limit. Written memory is how Claude keeps important details in context across multiple conversations." It also adds the caveat that compaction "is still a summary, and can still occasionally result in lost details." To make room in the window, the same help article suggests using projects (whose retrieval loads only the relevant content), shortening project instructions, removing unused project files, and turning off tools and connectors a conversation does not need; the usage side of these levers is in [Domain 7](#domain-7-troubleshooting-and-optimization).

**Know: the three moves.**

| Move | When | How (as of September 2026) |
|---|---|---|
| Restart | The chat has gone off track; you are near your usage limit in a long chat; a length-limit error appears; you want an unbiased second look; a new question does not need the earlier history (a short question late in a long chat can cost more than the same question in a fresh one) | Open a new conversation with a clearer prompt. The [length error](https://support.claude.com/en/articles/12466728-troubleshoot-claude-error-messages) itself says: "Try attaching fewer or smaller files or starting a new conversation." For a review, start a new chat and ask it to find errors in the answer. |
| Summarize | The material is bigger than it needs to be; a long conversation must continue | Summarize or extract the key sections before sending, break content into smaller chunks, or use Claude first to find the most relevant portions; automatic context management summarizes earlier messages, with possible loss of detail |
| Persist | Information must survive into future conversations | Project knowledge and instructions; memory (per the [memory article](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context), Claude saves topics as you chat, and you can say "remember this" or tell it what to remember, change or forget); on paid plans, chat search finds past conversations |

**Know: what to persist and what not to.** The Academy's [context tutorial](https://academy.claude.com/tutorials/parametric-memory-and-context) shows what a week of memory keeps: the project, preferences and working relationships are written down, while ephemeral, granular details (its examples are an RSVP count and a catering quote) are not. For those, it says Claude "can and should look those things up next time they come up, rather than writing them down and hoping they haven't changed." The [Working memory lesson](https://academy.claude.com/courses/ai-capabilities-and-limitations/working-memory) asks which tasks "need standing context set up (a project, saved instructions, uploaded reference docs) to be worth running, and which work fine cold?"

**Know: memory facts (as of September 2026).**

- Memory is on by default for Free, Pro and Max; on Team and Enterprise it is off by default until an owner turns it on.
- Each project has its own separate memory space and project summary, kept apart from other projects and non-project chats. A chat started in the wrong project can be moved out with "Remove from project" ([projects article](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects)).
- Settings > Memory shows what Claude remembers. Pausing keeps existing memory but stops its use and stops new memories; resetting permanently deletes all memories, including project memories.
- Incognito chats (the ghost icon) are not saved to chat history or memory, and they are not available inside projects. What memory never stores, and how long incognito chats are still retained, are data questions covered under [D6.2](#d62-apply-data-sensitivity-regulatory-and-privacy-considerations).

!!! note "Memory, as of September 2026"

    The [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) "memory considerations" sits inside a context objective, while its How to Prepare list names Memory as a feature to study. Both readings matter. The help center's [release notes](https://support.claude.com/en/articles/12138966-release-notes) date a change to July 10, 2026: memory now works as individual, categorized entries, "replacing the previous daily memory summary". The current [memory article](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context) says "Claude saves memory as a set of individual topics as you chat, rather than summarizing conversations after they end", and its legacy section still says a small number of Team and Enterprise organizations use the legacy experience, in which a synthesis of chat history is updated every 24 hours; the same article's opening notice, however, says Anthropic has "migrated users off the legacy experience". Older material may describe the legacy behavior.

**Decide**

- If a conversation has drifted or piled up corrections, choose to restart with a clearer prompt that carries only what matters; not more corrections in the same thread.
- If you are near a length or usage limit mid-task, choose a new conversation and move the lasting material into the Project or memory; not pasting the entire history back in.
- If information must survive across conversations (background documents, standing preferences), choose to persist it in project knowledge, instructions or memory; not to count on a new conversation recalling an old one: without saved context, a new conversation starts from zero (the [Working memory lesson's](https://academy.claude.com/courses/ai-capabilities-and-limitations/working-memory) "blank slate" probe), and memory keeps only what Claude or you choose to save.
- If you keep giving the same correction, choose to make it a standing instruction (the [Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/giving-cowork-context) calls such corrections "global-instruction candidates"); not repeating it in every chat.
- If a detail is ephemeral and granular (the context tutorial's examples are an RSVP count and a catering quote), choose to look it up fresh each time; not to persist it.
- If one input is too large, choose to extract the relevant sections first; not to upload everything and hope.
- If a critical instruction sits in the middle of a long prompt, choose to move it to the beginning or the end.

**Traps**

- Assuming chats in the same Project share context (see the [D3.1 traps](#d31-select-appropriate-claude-product-features-projects-research-mode-chat-artifacts)); what later chats need belongs in project knowledge.
- Assuming a correction given once will shape later chats; the model does not learn from corrections, so persist it as an instruction or memory.
- Assuming a bigger context window means nothing gets lost; truncation is silent and the middle of a long input is the weak spot.
- Treating automatic summarization as lossless.
- Persisting volatile facts in memory, so that later conversations use stale values.
- Treating incognito chats as deleted immediately; they are retained for 30 days by default. Data handling is [Domain 6](#domain-6-governance-risk-and-responsible-use).

**Go deeper:** [Compaction, context editing and memory](knowledge/context-engineering.md#compaction-context-editing-and-memory)

## Domain 4: Workflow Integration and Solution Design

**Weight:** 16% of scored items. If all 60 items were scored, that is about 9.6 items (16% of 60, our arithmetic); the guide does not say how many of the 60 are scored.

The official prep module sums this domain up as learning to move from "I use Claude" to "our workflow uses Claude" by deciding which steps to delegate to the model and which to keep with humans. The five objectives cover finding and analyzing the right problems, research and planning, prototyping solutions, fitting Claude into the way work already runs, and telling stakeholders accurately what it can and cannot do. The guide's candidate profile describes someone who "moves beyond basic question-and-answer usage to process reimagination, task automation, and project development", and its audience includes consultants who support "implementation, use-case identification, and process redesign."

That prep module, [Workflow Integration & Solution Design](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/workflow-integration-solution-design) (63 minutes), lists the guide's five Domain 4 objectives as its learning objectives (the fifth with "accurately" added) and names one organizing idea: "The anchoring competency from the AI Fluency Framework for this module is Delegation: deciding, for each step, whether the work is AI-appropriate, human-retained, or collaborative." Its course page also warns: "When the wrong steps are automated, it can backfire." In our reading, all five objectives come back to that step-by-step decision.

None of the guide's three sample questions comes from this domain: they cover Domains 2, 3 and 6. The **Decide** rules below are our own, built from the exam guide's wording and the Claude Academy courses and help articles cited under each objective; Anthropic does not publish them as rules, and none is an official answer key.

Our one-line summary of each objective, drawn from the sources taught in the sections below:

| Objective | The decision it tests | The habit that answers it |
|---|---|---|
| D4.1 Requirements and use cases | Which problem, and is it a good fit for Claude | Let Claude interview you; audit your workload before choosing tools |
| D4.2 Research, planning, process optimization | How to gather evidence, plan work and fix a process | Give the specific current state; require owners and dates; verify key claims |
| D4.3 Solution design, development, iteration | How far to take a prototype yourself | Prototype in artifacts, test it, escalate production work |
| D4.4 Augment or redesign workflows | Which steps Claude does, assists with, or leaves to people | Sort each step; widen Claude's role only on evidence |
| D4.5 Value and limitations | What to tell stakeholders | Outcomes with caveats; specific disclosure of AI's role |

### D4.1 Apply Claude to analyze requirements and use cases

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Apply Claude to analyze requirements and use cases". Two jobs sit under this objective: turning a rough business idea into clear requirements, and deciding which tasks are worth giving to Claude at all.

**Know: capturing requirements with Claude.** Claude Academy's [AI-native SDLC playbook](https://academy.claude.com/courses/ai-native-sdlc-playbook/capture-intent) describes the pattern. Its capture step is set up for "people who are not engineers (claude.ai or Cowork)":

1. The person with the idea (the playbook calls them the originator) describes the problem in their own words, for example what they cannot do today, who is affected, what better looks like, or what is out of scope. "No formal language is required."
2. Claude asks the questions an analyst would ask: "scope, users, constraints, and what success looks like." In a Use Case Gallery example, the prompt "Interview me first" turns a blank-page problem into a guided conversation.
3. Claude writes the result as an intent document using the organization's template. The document can cover "the problem, proposed outcome, affected users and systems, constraints, and open questions."
4. "The originator corrects anything Claude misunderstood."

??? example "The playbook's example intent document, quoted as published"

    ```markdown
    # Intent: claims status self-service
    Author: J. Ortiz (claims operations). Status: draft.
    ## Problem
    Customers phone the contact center to ask where their claim is.
    Handlers spend roughly a third of call time on status-only queries.
    ## Proposed outcome
    Customers see claim status, next step and expected date in the portal.
    ## Affected users and systems
    Claims handlers, portal team, claims-core API.
    ## Constraints
    No new PII in the portal session. Existing authentication only.
    ## Open questions
    Do third-party loss adjusters need access too?
    ```

Three prompt habits make a requirements draft easier to review:

- **Ask for concerns.** A requirements prompt in the playbook says: "Describe clearly any areas of concern, especially where you cannot satisfy contradicting policies." Work through the flagged concerns first, "as they are the points an analyst would have escalated."
- **State goals and non-goals.** "Calling out goals and non-goals keeps scope honest; the things you decided not to do are as load-bearing in review as the things you did."
- **Hunt for guesses.** The PRD example warns that Cowork "fills gaps with reasonable guesses, and unchecked guesses are what design review catches." Its advice: "Fix what you didn't decide, then send it." Adding "flag anything you're not confident about" tells you where to look first.

When the input is many stakeholders' requests or feedback, ask whether different requests point to the same underlying need, and check the sample: "Claude will tell you what's in the data, but you need to interpret whether the sample represents your actual user base."

**Know: finding and ranking use cases.** The AI Fluency: Framework and foundations course splits Delegation into Problem Awareness, Platform Awareness and Task Delegation, and the Academy's advice is to start with Problem Awareness: "Before touching any AI tools, analyze your actual workload." The audit the Academy courses describe:

| Question for each repetitive task | Where it comes from |
|---|---|
| How often does it happen, how long does each instance take, how standardized is it? | AI Fluency for nonprofits: list "5-10 tasks that felt repetitive or time-consuming" from your past week |
| "What's the consequence if this task is done imperfectly?" | AI Fluency for small businesses, Tying it all together lesson |
| Which one first? Weigh "time saved, frequency, and how straightforward the automation would be" | The same small-business lesson |
| Should AI do it at all? Ask "should AI do this?" not just "can AI do this?" | AI Fluency for nonprofits, Workflow augmentation lesson |

The nonprofit Workflow augmentation lesson draws the dividing line: answering documented questions is a good automation candidate, while handling complaints or high-stakes requests should stay with humans. Its exercise then sorts each task into three categories (the buckets in D4.4 below) and picks, from the "AI can handle" and "AI can assist" categories, the task "that would save you the most time" as the automation candidate.

To discover use cases, the Use Case Gallery suggests telling Claude your role (and optionally sharing your working documents), and naming your time sinks, such as "I spend hours on status updates". For work you hand off rather than talk through, the Cowork course names three patterns that cover most of what Cowork is built for: tasks that take several steps, tasks that draw on context from real files, and tasks that span the tools you already use.

**Decide**

- If the idea is still vague, choose to have Claude interview you (scope, users, constraints, success) before it drafts anything; not a request for the finished spec straight away, because Claude fills gaps with reasonable guesses.
- If you are choosing between candidate use cases, choose the frequent, time-consuming, standardized task where an imperfect result is low-consequence; not the high-stakes or emotional one, because Task Delegation asks whether AI should do it, not only whether it can.
- If a requirements draft looks complete, choose to check what you did not decide yourself before sending it; not to forward it as-is.

**Traps**

- Picking a tool or an automation before analyzing the actual workload. The Academy puts Problem Awareness first: "Make a specific list before deciding what to automate".
- Treating "can AI do this?" as the whole test.
- Accepting a polished requirements document without looking for assumptions Claude supplied.
- Asking Claude how confident it is in its own draft and treating a high answer as a check. The guide's Sample 1 rationale (a Domain 2 item) says "Self-reported confidence (A, C) is not a reliable accuracy signal".
- Summarizing stakeholder input without asking whether the sample represents the real user base.

**Go deeper:** [Discovery and requirements](knowledge/solution-architecture.md#discovery-and-requirements)

### D4.2 Research, planning and process optimization

D4.2 asks candidates to use Claude "for research, planning, and process optimization" ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)). Picking research mode as a product feature is tested in [Domain 3](#domain-3-product-and-model-selection), whose D3.1 objective lists "Projects, research mode, chat, artifacts"; this objective is about using research, plans and process analysis inside a piece of work.

**Know: research as a workflow input** (as of September 2026). What Research is, who has it, how to start it and how it differs from web search and enterprise search are covered under [D3.1](#d31-select-appropriate-claude-product-features-projects-research-mode-chat-artifacts). Two points matter when research feeds a piece of work:

- **Connector tools can act without asking.** During Research, "Claude can invoke tools from your connectors automatically without further approval" ([custom connectors](https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp)). For Research with custom connectors, the help center says to disable any tools that can take write actions in external applications and to review Claude's approval request carefully.
- **Citations are where checking starts.** Every web search response includes citations. Anthropic's own advice is to cross-reference cited sources and use authoritative sources for critical decisions. The [AI Fluency for small businesses](https://academy.claude.com/courses/ai-fluency-for-small-businesses/researching-with-ai) course is specific: flag the claims that matter (regulations, pricing data, deadlines) "for verification against primary sources".

The AI Fluency for nonprofits course adds that a research prompt should "explain who you are, who you serve, and what you specifically need to know".

**Know: planning with Claude.** In our reading, the planning patterns in Claude Academy's courses and Use Case Gallery share one rule: a plan is useful only when every item has an owner and a date, and when gaps are named rather than hidden.

| Planning task | What to ask for |
|---|---|
| Project plan | Invite Claude to ask questions until the vision is clear, identify the major tasks together, then "Create a project plan that includes your major tasks and delegation decisions." |
| Status report | Say exactly what to track and where to look, and add: "If you don't find information for a work stream, note that explicitly rather than omitting it." |
| Launch readiness | A checklist "with status and owner per item", a red/yellow/green readiness call, named blockers; point Claude at past retros so it is not "evaluating in isolation" |
| Decisions from a discussion | "Who owns each next step with the date they committed to" |
| Sequencing | What happens first, what cannot start until something else is done, and what a delay does to the timeline |

The launch-readiness page puts the rule in one line: "A checklist with a name on every line is actionable in launch standup; a checklist without owners is a wish list."

??? example "The Use Case Gallery's status-report prompt (abridged, with the page's missing-information tip added)"

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
    If you don't find information for a work stream, note that explicitly rather than omitting it.
    ```

    The page's full prompt also asks for notes from each owner's updates, specifies an Excel tracker (status indicators, cell comments, dropdown menus, data bars) and closes with a line on the tracker's purpose: "The tracker should make it obvious at a glance where the problems are and who needs help." Those lines are left out here. The last line above is the page's own tip for handling missing information.

**Know: process optimization.** The Academy's workflow-improvement planner asks for the current workflow, where "the pain points and bottlenecks are", inputs and outputs, success criteria and constraints. Two instructions matter most:

- **Be specific about volume and frequency.** Not "monthly reports" but "15-page donor report, created by 5th of each month, typically takes development director 8 hours, uses data from 3 different spreadsheets."
- **Be honest about what is broken.** "Takes forever" gives Claude nothing to work on; "requires manually copying 200 donor records from Salesforce into Excel, then reformatting each one" does. In the planner's words, "The more candid you are about pain points, the better the solution."

Also state real constraints (the planner's example: "We can only use free tools, we don't have technical expertise on staff, and I need this to work for someone with limited AI experience."), and remember that "Workflow improvements aren't just about speed": the planner also asks where quality suffers, where knowledge is trapped with one person, where scaling fails and where stress peaks. Speeding up one step can leave another as the bottleneck. Anthropic's productivity research makes a related point about tasks across occupations and the wider economy: "As AI accelerates some tasks, others may become bottlenecks".

For process maps, Claude "identifies decision points, parallel workflows, and conditional logic" from messy narrative, lists or emails. Keep the map current by holding the documents and conventions in a dedicated Project and describing each change in a new conversation there. To show a change worked, the SDLC playbook pairs a leading measure (elapsed time compared with the old requirements-plus-design cycle) with a lagging one ("Requirements rework after build starts").

**Decide**

- If you ask Claude to improve a process, choose to describe the current state concretely (volume, frequency, time, the exact manual steps, constraints); not a bare request to make it faster, because concrete inefficiencies give Claude something specific to fix.
- If a plan or checklist comes back without owners and dates, it is not finished; ask for them.
- If a status report must be trusted, instruct Claude to name the gaps it found; not to smooth over missing workstreams.
- If Research will run with custom connectors that can write, disable the write-capable tools first, because during Research Claude can invoke connector tools without further approval.
- If research feeds a decision, verify the load-bearing specifics (regulations, prices, deadlines) against primary sources before you act.

**Traps**

- Vague process descriptions, such as "Takes forever", that leave Claude guessing.
- Measuring only speed, or speeding one step and ignoring the step that becomes the new bottleneck.
- Plans without named owners: in the Academy's words, "a wish list".
- Treating a cited research summary as verified because it has citations.
- Leaving write-capable connector tools switched on during Research.

**Go deeper:** [Search, Research and connectors](knowledge/claude-for-work.md#search-research-and-connectors)

### D4.3 Use Claude to support solution design, development, and iteration

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Use Claude to support solution design, development, and iteration". As [Who this exam is for](#who-this-exam-is-for) explains, in our reading "development" here is a prompting and assisting skill, not coding against APIs: the guide says no software-development or API experience is needed. In our reading, the skill tested is taking a solution far enough to prove the idea, then knowing when to hand it on.

**Know: prototyping with artifacts.** An artifact is "anything Claude makes for you that you'd put in front of someone: a design, a deck, a document, a dashboard, or a small interactive tool." It opens beside the conversation, and you can edit it, return to it and share it by link. Anthropic's artifacts tutorial reduces the method to two moves: "describe the problem you want solved, then build on Claude's responses until the app fits what you pictured."

1. Explore the problem and let Claude interview you, then ask for the build ("Let's build this now.").
2. Iterate in plain language. When something breaks, click "Fix with Claude" or describe the problem ("the calculator isn't working with decimals"); no need to understand technical error messages.
3. Use the version selector to move between versions.
4. Test before you share: "If your artifact computes, counts, or recommends anything, work one example out by hand and compare." The tutorial's warning: "An app that looks polished but gets the numbers wrong will mislead the people you share it with, and you are the only one who can catch that before they do."

AI-powered artifacts call Claude from inside the artifact: "You can add AI capabilities to your artifact by simply asking Claude to use Claude". When others use a shared AI-powered artifact, "Usage counts against each user's own Claude subscription limits, not yours." If someone customizes your shared artifact, they get their own copy and your original is unchanged.

Before you send a stakeholder a link, check where the artifact was made: people without a Claude account can open only an artifact published from chat in the previous experience (the sharing rules, and where the official sources word them differently, are under [D2.6](#d26-organize-and-curate-information-and-select-appropriate-output-formats-artifacts-inline-structured-data)). The [artifacts tutorial](https://academy.claude.com/tutorials/use-artifacts-to-visualize-and-create-ai-apps-without-ever-writing-a-line-of-code) adds two details for published links: AI-powered artifacts still require sign-in, and "Viewers can copy the content or look at the code, but they can't change your published version."

**Know: designing with Claude Design** (beta, as of September 2026). Claude Design creates "designs, interactive prototypes, one-pagers, and other visual work" through conversation. It is available in beta on Pro, Max, Team and Enterprise, not on Free; it is on by default on Pro, Max and Team and off on Enterprise until an owner turns it on. "Claude Design counts toward the same usage limits as the rest of Claude."

| Situation | What the help article and tutorial advise |
|---|---|
| Writing the first prompt | State the goal (what you are building), the layout, the content to display, and the audience who will use it |
| Changing one component | Use a comment on it ("fix this button") |
| Structural change or new section | Use the chat |
| Quick visual tweak | Edit directly on the canvas |
| Giving feedback | Be specific: "Tighten the spacing between form fields to 8px" rather than "This doesn't look right" |
| Trying a new direction | Say "Save what we have and try a completely different approach"; there is no version history yet |
| Unsure of a direction, or a design review with stakeholders | Ask for 2 to 3 options; generate 2 to 3 alternatives and present them side by side |
| Before handoff to engineering | Ask to see empty, error and loading states and different data volumes |

The tutorial's end-to-end flow finishes with an engineer who "uses Claude Code to implement the feature, starting from the prototype rather than from scratch". In our reading, the Associate's prototype is the input to engineering, not the product.

**Know: the signals that work has left Associate scope.** The guide publishes no escalation checklist. The signals in the left column are our grouping; each statement on the right comes from the source named.

| Signal (our grouping) | Source statement |
|---|---|
| The prototype needs to become a real application | The [prototyping tutorial](https://academy.claude.com/tutorials/prototype-ai-powered-apps-with-claude-artifacts), under the heading "Moving from prototype to production": artifacts are "best for testing and demonstration"; at some point "you'll likely want to implement proper API key management". Its next step is copying Claude's code into a code editor |
| The work means building against APIs or designing agentic systems | The guide excludes "software developers who build against APIs or design agentic systems" from this credential |
| The solution is an enterprise-scale architecture or integration | That scope "belongs to the Claude Architect and Claude Developer credentials" |
| The change is higher risk | In the SDLC playbook the product owner consults "a technical lead for anything the organization classes as higher risk. A human teammate always makes this call" |
| The work needs shared, version-controlled infrastructure | In the SDLC playbook, setting up the shared home for intent documents "is a one-time task for the platform or engineering team" |
| You cannot judge the technical choices | Anthropic's education report [How people check Claude's work](https://academy.claude.com/tutorials/discernment-toolkit) found lack of domain expertise the biggest barrier to checking, including for non-technical participants led into software-development decisions (quoted under [D2.4](#d24-determine-when-human-review-or-additional-verification-is-required)) |

The Developer guide describes that role as translating technical requirements into working systems "through API integration, agent and tool construction, prompt and context engineering"; the Architect guide's ideal candidate "is a solution architect who designs and implements production applications with Claude." Those are the people an Associate escalates to. Claude 101 draws the same line for its own learners: building software happens in the Code tab, and "If you're not a developer, the takeaway is just this: it's a separate tab, and this course doesn't need it".

**Decide**

- If stakeholders need to see how a tool or screen would work, choose an artifact or a Claude Design prototype; not a long written description, because people can try a working prototype and compare alternatives side by side.
- If the direction is unclear, ask for 2 to 3 alternatives and compare them.
- If the artifact computes or recommends anything, verify one example by hand before sharing it.
- If the prototype must run in production with API keys, company systems or real customers, escalate to Developers or Architects; not ship the artifact, because artifacts are best for testing and demonstration.

**Traps**

- Sharing a polished artifact whose numbers nobody checked.
- An Associate designing or building an enterprise integration instead of escalating it.
- Assuming the creator pays for everyone's use of a shared AI-powered artifact.
- Exploring a new design direction without saving first, when Claude Design has no version history.
- Following Claude into software architecture decisions you cannot evaluate.

!!! warning "Exam guide vs current docs"

    The guide (July 2026) names "artifacts" and "Projects, Artifacts, and workflow-based interactions" and does not mention Claude Design, Claude Slides or Claude Docs. Current help articles (as of September 2026) describe Claude Design, Claude Slides and Claude Docs as betas on paid plans, and Anthropic's September 16, 2026 announcement says "Claude Docs and Claude Slides are new today, and Claude Design now works inside your conversations too." The Claude Design help article now sends presentations to Claude Slides ("To make presentations, use Claude Slides."), while an Academy tutorial still teaches Claude Design for slide decks. Answer in the guide's terms (artifacts, Projects, workflow-based interactions): none of these three product names appears in its objectives.

**Go deeper:** [Artifacts, files and code execution](knowledge/claude-for-work.md#artifacts-files-and-code-execution)

### D4.4 Integrate Claude into existing workflows to augment or redesign them

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Integrate Claude into existing workflows to augment or redesign them". The prep module's framing applies directly: for each step, decide whether the work is "AI-appropriate, human-retained, or collaborative."

**Know: three ways to work with AI, and three ways to sort a step.** The AI Fluency framework names three primary ways people engage with AI:

| Mode | Definition in the framework |
|---|---|
| Automation | AI executes specific tasks based on your instructions |
| Augmentation | You and AI collaborate as creative thinking and task execution partners |
| Agency | You guide AI to work independently on your behalf, shaping its knowledge and behavior rather than specific actions |

The Academy's nonprofit workflow lesson sorts each task into one of three buckets: "AI can handle" (standardized responses, documented information, clear processes); "AI can assist, human decides" (AI drafts or prepares, you review before action); and "Human should handle: High-stakes decisions, emotional situations, complex judgment calls". The framework's reminder: "The goal isn't to automate everything, but to create the most effective human-AI partnership for any given task or goal."

To make that call one step at a time, the AI Fluency project-planning lesson has you identify the major tasks with Claude and then, for each task, discuss what skills, knowledge or AI capabilities it needs, "Which parts would benefit from uniquely human strengths?", which parts AI could handle well, and "Where might collaboration have the most impact?"

**Know: augment or redesign.** In our terms, augmenting keeps the process and puts Claude inside its steps, with review; redesigning changes the process itself. The Academy's AI-native SDLC playbook is a redesign example: requirements and design, once separate phases handed between teams, now "happen in a single prompted session." Its Requirements and design stage says the old separation "exists for accountability, but it is slow and lossy", and its Capture stage names the cost of handoffs: "Ownership transfers at each handoff, so what reaches engineering is several steps removed from what the originator meant."

**Know: meeting the work where it already runs** (as of September 2026).

| Where the work happens | Claude surface | One fact to remember |
|---|---|---|
| Multi-step tasks you hand off | Claude Cowork (paid plans) | "describe an outcome, step away, and come back to finished work"; schedule with `/schedule` |
| Recurring jobs | Cowork scheduled tasks | Run remotely, so your computer can be asleep; the help article says they "can't be tied to a folder on your computer", though its Cowork manual-setup steps add that a task needing local files or apps "will only run locally"; the help center lists hourly, daily, weekly, on weekdays, or manually |
| Spreadsheets, decks, documents, email | Claude for M365 add-ins: Claude for Excel, PowerPoint, Word and Outlook (Outlook in beta; the claude.com docs call Word generally available, while the help center's beta table still lists it as beta) | Claude for Excel is not recommended for "Final client deliverables without human review" or "Audit-critical calculations without verification" |
| Team chat | Claude Tag in Slack (Team and Enterprise, beta) | "Tagging Claude in a channel is billed to your organization. Direct messages are billed to your own Claude account instead." |
| Websites | Claude in Chrome (paid plans) | Prompt injection hidden in web content is "The biggest risk facing browser-using AI tools" |

The Academy's sources word the scheduling cadences differently from the help center: the Cowork course lists "hourly, daily, weekdays, or manual", and the Cowork rollout tutorial lists "hourly, daily, weekly, weekdays, or on-demand". Which surface fits which kind of task (turn-by-turn chat or handed-off work) is covered in [Domain 1](#domain-1-prompting-and-task-execution).

**Know: introducing automation safely.** The official guidance is consistent: extend Claude's role on evidence, and keep people accountable.

- **Start small and visible.** The Academy's Building effective human-agent teams course (beta) says: "Give the agent one visible job first." Its suggested first job is a morning briefing shared with the team: "For the first few days, a person should review the briefing and give the agent feedback on its effectiveness." Widen scope only "After the agent has handled one kind of task well several times in a row".
- **Grant autonomy on evidence.** The same course: "Grant autonomy in proportion to demonstrated reliability, then expand it deliberately." The Cowork rollout tutorial applies it per workflow: where output proves reliable, Claude's share widens; "where errors appear, a person takes the step back."
- **Review early output.** "Deployment Diligence means reviewing outputs before they go out (especially early on)."
- **Test with real history.** Use actual past examples to find gaps in your descriptions; the nonprofit lesson's version is "Use actual emails you've received to test the system".
- **Prove it once, then schedule it.** Do the task once in Cowork, confirm the output, then type `/schedule`.
- **Start with low-risk scheduled tasks.** Begin with summaries or compiled information, and "Don't schedule tasks that access sensitive files, send messages on your behalf, make purchases, or take other actions that are difficult to undo."
- **Choose the approval mode by consequence.** In the previous Cowork experience there are three modes: Manually approve (Manual), Automatically approve (Auto, where Claude keeps working but still reviews each action for safety) and Skip all approvals (Skip, where nothing checks its actions). As of September 2026, the new Claude experience offers only Auto and Manual (default). The safety article says to switch to Manually approve when the task "touches sensitive files, accounts, or sites", when "You're working with a new tool, plugin, or site for the first time", or when "Mistakes would be hard to undo, like sending messages or making purchases." It adds that "Claude always asks before permanently deleting files, in any mode."
- **Stay in control of delegated work.** The Cowork course says Claude "shows you its plan before it starts, by default asks before it takes actions that matter (sending, deleting, sharing), and lets you steer at any point."
- **Customer-facing work needs three things:** "review outputs before they reach customers, be honest about AI's role, and provide a clear path to a human."
- **Keep it explainable.** Ask "can we explain what the AI is doing?" If yes, "that's healthy augmentation"; if not, rework the process.
- **Write down what you repeat.** "When you explain the same thing twice, turn it into written instructions for the agent."
- **Review weekly.** "Have agents and humans share what's working and what requires adjustment at the end of every week."
- **Keep accountability with people.** "AI changes how the work gets produced; accountability for it stays with your people." For each automated output, "someone at your org owns that output: they can explain it, defend it, and fix it." The Cowork safety article adds: "You remain responsible for all actions taken by Claude performed on your behalf."

**Decide**

- If a step is a high-stakes decision, an emotional situation or a complex judgment call, keep it with a person; if Claude can draft but a mistake would matter, use "AI can assist, human decides"; if it is documented and repeatable, it is an automation candidate.
- If the process works and the pain is effort inside its steps, augment; if the pain is lost meaning across handoffs, consider redesigning the process shape, as the SDLC playbook does.
- If you are introducing a new automated step, start with one visible job and human review, then widen after several good runs; not full autonomy on day one.
- If a task touches sensitive files, accounts or sites, uses a tool, plugin or site for the first time, or could make a hard-to-undo mistake (sending messages, making purchases), use Manually approve; not Automatically approve or Skip all approvals, because in either of those modes Claude can act on injected instructions before you notice.
- If a recurring task has not yet produced a verified result, run it once and check it before scheduling it.

**Traps**

- Automating everything that can be automated. The framework says the goal "isn't to automate everything".
- Giving an automated workflow full autonomy before it has a track record.
- Putting a customer-facing automation live with no review and no path to a human.
- Using Skip all approvals (where Claude "doesn't pause to ask and nothing checks its actions automatically"), or Automatically approve, for consequential actions, or scheduling tasks that send messages or touch sensitive files.
- Believing accountability moves to the tool once work is automated.
- The "chat trap" named in Anthropic's Cowork rollout tutorial: "Users default to short prompts with no connectors." The same tutorial's fix is to demo delegation side by side with chat.
- Treating a switch-on as a rollout. The same tutorial: "IT flipping the switch and sending an email isn't a rollout."

!!! warning "Exam guide vs current docs"

    The guide (July 2026) speaks of "workflow-based interactions" and "task automation" and does not name Claude Cowork. Since then, chat and Cowork have begun merging into one Claude; the rollout is described in the box under [D3.1](#d31-select-appropriate-claude-product-features-projects-research-mode-chat-artifacts). For this objective, the change that matters is the permission setting: the new experience offers only Auto and Manual, with Manual the default (Skip all approvals belongs to the previous Cowork experience), and Claude 101's three-tab picture (Chat, Cowork, Code) may not match every account. Answer in the guide's terms: the objective is about which steps to automate, augment or keep human, and how to extend trust, not about which menu to click.

**Go deeper:** [Claude Cowork](knowledge/claude-for-work.md#claude-cowork)

### D4.5 Communicate Claude's value and limitations to stakeholders

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Communicate Claude's value and limitations to stakeholders". The prep module's version of this objective adds one word: "Communicate Claude's value and limitations to stakeholders accurately" ([Workflow Integration & Solution Design](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/workflow-integration-solution-design)). In our reading, accuracy runs in both directions: no overclaiming value, and no overstating or hiding limits.

**Know: stating value with its caveats.** Anthropic's own figures show how to present evidence honestly, because each comes with a stated limit:

| Claim | The caveat published with it |
|---|---|
| Anthropic's productivity research sampled one hundred thousand Claude.ai conversations; Claude estimated the tasks would take about 90 minutes without AI and that it speeds individual tasks up by about 80% | The study cannot count time spent outside the conversation, "including validating the quality or accuracy of Claude's work", so the estimates "might overstate current productivity effects to at least some degree"; Anthropic also says its economy-wide extrapolation is not a prediction |
| In the Anthropic Economic Index Survey (launched April 2026, sent to a random sample of Claude users), respondents report "productivity gains in speed, scope, and quality of their work (86%, 82%, and 69%, respectively)" | "The Economic Index Survey is not representative of the general population." |

The same productivity page also gives the average task length as 1.4 hours in its summary, so quote any figure with its own framing.

For your own organization, measure outcomes, not activity:

- Usage metrics "tell you Claude is being used, not what that use produced"; pair them with the outcomes of the work.
- Do not turn adoption numbers into targets: "The month you turn a diagnostic into a quota, members optimize for the number instead of the work."
- Define outcome measures "including net new work completed and qualitative feedback on use cases in addition to productivity measurement."
- Claude 101's simple eval compares Claude's output with 5 to 10 examples of your own past work on a task you do regularly, so you learn where Claude adds the most value and where human review is essential; that gives stakeholders evidence from your own tasks.
- Settle early what happens to the time saved: the nonprofit integration lesson says to "discuss expectations about what happens with time saved to ensure everyone feels good about the work AI is supporting".

**Know: stating limitations accurately.**

- Hallucination is improving but, in Anthropic's words, "not at all a solved problem".
- "Users should not rely on Claude as a singular source of truth and should carefully scrutinize any high-stakes advice given by Claude."
- Anthropic's own terms tell users to confirm outputs independently before relying on them (quoted under [D2.4](#d24-determine-when-human-review-or-additional-verification-is-required)).
- Models "may not be aware of events or information that occurred after their respective cutoff dates."
- Claude may claim abilities it lacks, such as having sent an email with no email integration (see [D2.2](#d22-identify-hallucinations-inconsistencies-and-biases-in-responses)).
- Trust is set by stakes: review scales with what is riding on the output (see [D2.4](#d24-determine-when-human-review-or-additional-verification-is-required)).
- Complex or technical implementation is escalated to Claude Architects and Developers; saying so is part of describing the limits.

**Know: disclosing AI's role.** Transparency Diligence means "being honest about AI's role in our work with everyone who needs to know". Claude Academy's AI diligence statement tutorial says the statement "explains what AI did, what you did, and how you verified the result", and lists five elements: what AI assisted with, which tool, what you reviewed, what you changed, and who is responsible. Three rules from the same tutorial:

- "Vague or absent disclosure is what erodes trust, especially if someone discovers the AI involvement later."
- "Your AI diligence statement is itself a claim about your process, and it needs to be accurate." Only say you verified every citation if you did.
- The statement can also set the reader's level of review, for example: "I asked Claude to summarize this article. Here's the summary; read it with discernment."

??? example "The AI Fluency course's diligence statement template"

    ```text
    In creating this [document/project/content], I collaborated with [AI assistant name] to assist with [specific tasks: drafting, research, editing, etc.]. I affirm that all AI-generated and co-created content underwent thorough review and evaluation. The final output accurately reflects my understanding, expertise, and intended meaning. While AI assistance was instrumental in the process, I maintain full responsibility for the content, its accuracy, and its presentation. This disclosure is made in the spirit of transparency and to acknowledge the role of AI in the creation process.
    ```

    Fill it with project-specific details. The diligence statement tutorial warns: "A copy-pasted generic statement that doesn't describe your actual process provides very little value."

When Claude helps write the stakeholder memo itself, ask for the trade-offs: the Use Case Gallery's vendor memo asks for "what we're giving up by not choosing the others, and what we need to negotiate before signing". Ask it to mark uncertain claims too; a data-report example in the gallery ends its prompt with "Flag where I should hedge."

**Decide**

- If you are asked to show value, choose outcome evidence from real tasks (quality, net new work, a simple eval on your own examples) with its caveats; not login counts or a borrowed headline percentage.
- If a stakeholder asks whether Claude can be trusted, answer by stakes: which outputs need light review and which need full verification; not a flat yes or no.
- If AI contributed to a deliverable, disclose its role specifically and accurately; not vaguely, and not at all.
- If a request is beyond Associate scope, say so and name the escalation path.

**Traps**

- Presenting a research estimate as a promise for your team, without the caveats Anthropic publishes with it.
- Measuring only time saved or logins. The Cowork rollout tutorial lists "Only measuring time-saved or login counts" as a pitfall.
- Turning adoption metrics into quotas.
- Reporting output as reliable because Claude sounded sure. The Sample 1 rationale: self-reported confidence "is not a reliable accuracy signal".
- A disclosure that claims checks nobody ran.
- Telling stakeholders hallucinations are solved, or that Claude sent an email it has no integration to send.

**Go deeper:** [Stakeholder communication and lifecycle](knowledge/solution-architecture.md#stakeholder-communication-and-lifecycle)

## Domain 5: Configuration and Knowledge Management

**Weight:** 12% of scored items. If all 60 items were scored, that is about 7.2 items (12% of 60, our arithmetic); the guide does not say how many of the 60 are scored.

The official prep course module for this domain, [Configuration & Knowledge Management](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/configuration-knowledge-management) (47 minutes), opens its description with the idea the whole domain rests on: "A good prompt helps you once, while a good configuration helps you every time." It describes "the setup work that runs before any single conversation: the Project, the instructions, and the knowledge sources that put the right context in place, so Claude starts informed", notes that this setup "is also what scales a skill from one person to a team", and covers "the maintenance cadence that keeps a configuration from quietly going stale." The guide's own How to Prepare list asks you to "configure a Project with instructions and knowledge sources, and evaluate outputs for accuracy and bias".

### The configuration layers at a glance

The four objectives below refer to the same set of places where standing context lives. Learn them as one map (as of September 2026; the grouping is ours, each row is sourced):

| Layer | Applies to | Who sets it | Facts to know |
|---|---|---|---|
| Instructions for Claude (your settings) | All of your conversations | You | In the new Claude experience, Cowork's Global instructions setting "is now part of Instructions for Claude in Settings > General" ([Cowork and chat are one Claude](https://support.claude.com/en/articles/16761823-claude-cowork-and-chat-are-one-claude)) |
| Project instructions | Every chat in one project | You, or shared members with "Can edit" | Added with "Set project instructions"; Claude cannot see the project's name or description |
| Project knowledge | Every chat in one project, as background | You, or shared members with "Can edit" | 30MB per file; on paid plans RAG expands capacity "by up to 10x" ([RAG for projects](https://support.claude.com/en/articles/11473015-retrieval-augmented-generation-rag-for-projects)) |
| Organization instructions | Every message from everyone in the organization | Owners and Primary Owners, Team and Enterprise | Maximum 3,000 characters; changes can take up to an hour; wins over an individual instruction that directly contradicts it |
| Skills | Loaded only when relevant, anywhere in Claude | You; on Team and Enterprise, Owners can provision skills for all users | Require code execution to be enabled |
| Connectors | Conversations where they are switched on | On Team and Enterprise an Owner or Primary Owner enables them; each person authenticates | Claude inherits each person's permissions in the connected service |
| Memory | Across your chats; each project has its own memory | Claude saves as you chat; you can view and edit it in Settings > Memory (the memory article's legacy section puts Memory under Settings > Capabilities for a small number of Team and Enterprise organizations, while its opening notice says users have been migrated off the legacy experience; see the memory box under [D3.4](#d34-understand-and-manage-context-limitations-and-memory-considerations-when-to-restart-summarize-or-persist)) | On by default for Free, Pro and Max; in the current memory experience, off by default on Team and Enterprise until an owner turns it on (the legacy experience's Enterprise controls describe the organization-wide memory toggle as enabled by default). Context decisions are in [Domain 3](#domain-3-product-and-model-selection) |

Two lines, from Anthropic's [skills overview](https://support.claude.com/en/articles/12512176-what-are-skills) and from [Claude 101](https://academy.claude.com/courses/claude-101/working-with-skills), separate Projects from skills: "Projects provide static background knowledge that's always loaded when you start chats within them. Skills provide specialized procedures that activate dynamically when needed and work everywhere across Claude." In Claude 101's shorthand: "projects store knowledge, skills perform tasks."

### D5.1 Configure Claude Projects with instructions and knowledge sources

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Configure Claude Projects with instructions and knowledge sources". The guide's intended audience "build Claude Projects in their day-to-day roles", and Projects also appear in D3.1 and in the guide's How to Prepare list.

**Know: what a Project is and how to set one up.** The [projects overview](https://support.claude.com/en/articles/9517075-what-are-projects) says: "Projects allow you to create self-contained workspaces with their own chat histories and knowledge bases." They are available on every plan; Free users can create at most five (one line in the personalization article calls project instructions "paid plans only"; see the note below).

The steps, from the [projects help article](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects):

1. On the Projects page, click "+ New Project".
2. Give it a name and description. The help center notes "Claude will not have access to these details", so nothing Claude must follow belongs there.
3. Click "Set project instructions". "Claude will use these instructions for all the chats within the project."
4. Add knowledge: "documents, text, code, or other files", which Claude uses as background for every chat in the project ([projects overview](https://support.claude.com/en/articles/9517075-what-are-projects)).

**Know: how project knowledge behaves.**

| Fact | Detail | Source |
|---|---|---|
| File limits | 30MB per file; unlimited files, "but total content must fit within Claude's context window"; text extraction only, except multimodal PDFs | [Upload files](https://support.claude.com/en/articles/8241126-upload-files-to-claude) |
| RAG | When knowledge nears the context limit, Claude "will automatically enable RAG mode to expand your project's capacity by up to 10x"; paid plans only (Pro, Max, Team, Enterprise) | [RAG for projects](https://support.claude.com/en/articles/11473015-retrieval-augmented-generation-rag-for-projects), [projects overview](https://support.claude.com/en/articles/9517075-what-are-projects) |
| RAG control | None: "RAG activation is handled automatically based on the size of your project knowledge"; no setup is required, a visual indicator shows a RAG-enabled project, and Claude can switch back to context-based processing if knowledge drops below the threshold | [RAG for projects](https://support.claude.com/en/articles/11473015-retrieval-augmented-generation-rag-for-projects) |
| Retrieval quality | "Well-named files help Claude understand and retrieve the right information more effectively"; you can name a document in your question to focus the search. Claude 101's example of a good file name: `Q4-2024-Brand-Guidelines.pdf` rather than `document1.pdf` | [RAG for projects](https://support.claude.com/en/articles/11473015-retrieval-augmented-generation-rag-for-projects), [Claude 101](https://academy.claude.com/courses/claude-101/introduction-to-projects) |
| Chats in the same project | "Context is not shared across chats within a project unless the information is added into the project knowledge base." | [Projects help article](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects) |
| One-off files | A file uploaded in a chat "stays separate from your project knowledge" | [Claude 101](https://academy.claude.com/courses/claude-101/introduction-to-projects) |
| Memory | "Each project has its own memory, kept separate from your non-project chats." | [Projects help article](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects) |
| Usage | "Content in projects is cached and counts less against your limits when reused"; after a long break the first message counts it in full again | [Usage limit best practices](https://support.claude.com/en/articles/9797557-usage-limit-best-practices) |

**Know: writing project instructions.** The help center's [usage and length limits article](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work) advises using project instructions "for general context around your project, key guidelines, and Claude's role. Reserve task-specific instructions for the chat itself." [Claude 101](https://academy.claude.com/courses/claude-101/introduction-to-projects) lists four things good project instructions typically include, with an example of each.

| Element | Claude 101's example |
|---|---|
| Context about what you're working on | "This project is for creating marketing content for our B2B software product." |
| Process instructions | "First consider a blog structure that will entice this audience, then write the draft." |
| Tone and style preferences | "Use a professional but conversational tone. Avoid jargon when possible." |
| Specific requirements | "Always include a call-to-action at the end of marketing copy." |

Separately from that list, the lesson says "You can also use project instructions to automate workflows", with the example "When I upload a meeting transcript, create a structured summary using this template." Every transcript uploaded to that project then gets the same treatment. The lesson's summary: "Think of instructions as programming Claude's behavior for this project." It also names the signs that a Project is worth creating: "Reference materials you'll use repeatedly", "Consistent requirements for how Claude should respond", and "Team collaboration needs where multiple people should work from the same foundation". How to write instructions well is taught under [D5.3](#d53-create-effective-system-level-instructions).

**Know: sharing a Project (Team and Enterprise).**

- Access levels ([projects overview](https://support.claude.com/en/articles/9517075-what-are-projects)): "Can view" members see contents, knowledge and instructions and can chat, but cannot edit; "Can edit" members can change instructions and knowledge and manage members.
- Visibility ([project visibility and sharing](https://support.claude.com/en/articles/9519189-manage-project-visibility-and-sharing)): Public means everyone in the organization can view and use the project; Private means only invited members. Even in a public project, your chats stay private unless you share them. Sharing a project with a group requires Enterprise and is in beta; group access changes can take up to five minutes to apply.
- Connectors ([connectors article](https://support.claude.com/en/articles/11176164-use-connectors-to-extend-claude-s-capabilities), [Google Workspace connectors](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors)): on Team and Enterprise, connectors are only available in private projects, and the Google Drive connector is disabled for shared projects.
- Admin control ([control project sharing](https://support.claude.com/en/articles/9927533-control-project-sharing-for-your-organization)): Owners and Primary Owners control the "Share projects" setting and its sub-setting "Public projects" under Organization settings > Data and privacy; "Both are on by default." On Enterprise, custom roles with the Privacy permission set to "Can manage" can also change them, and project sharing can be turned on or off per role (role changes can take up to 15 minutes to apply). Turning sharing off leaves existing shares in place but makes public projects private, and turning it back on does not reverse that: "Projects that became private stay private."

Cowork has Projects too: they "group related tasks into dedicated workspaces with their own files, context, instructions, and memory", and that memory stays inside the project ([Cowork projects](https://support.claude.com/en/articles/14116274-organize-your-tasks-with-projects-in-claude-cowork)).

**Decide**

- If the same background (brand guide, product sheet, policy) is needed in every chat about one body of work, choose a Project with that material as knowledge; not re-uploading it in each chat, because project knowledge is background for every chat and is cached for reuse.
- If a file matters for one conversation only, upload it in that chat; not to project knowledge, because a chat upload stays separate and keeps the knowledge base uncluttered.
- If a conclusion from one chat must be available in the project's other chats, add it to the project knowledge; not assume the other chats saw it, because the projects help article says context is not shared across chats within a project unless it is added to the project knowledge base (project memory, where it is on, can carry some context, but it is not the documented way to share it; see the box below).
- If a project's knowledge is large, name files descriptively and name the document in your question; not look for a RAG switch, because RAG activation is automatic.
- If you want a procedure available everywhere in Claude, choose a skill; not a Project, because a Project holds background knowledge for one body of work while skills "work everywhere across Claude".
- If a project needs Google Drive files or other connectors, it must be a private project. If the project must be shared with the team, expect the Drive option to be disabled and add uploaded copies instead, because on Team and Enterprise connectors are only available in private projects and the Drive connector is disabled for shared projects.

**Traps**

- Writing guidance into the project name or description, which Claude cannot see.
- Expecting one chat in a project to know what was said in another.
- Loading project instructions with one-off task details that belong in the chat.
- Expecting a manual switch for RAG, or RAG on the Free plan.
- Expecting connectors in a public project, or the Google Drive option in a shared project.

!!! warning "Exam guide vs current docs"

    The guide (July 2026) asks you to configure Projects "with instructions and knowledge sources". The [projects overview](https://support.claude.com/en/articles/9517075-what-are-projects) (as of September 2026) also describes a new version of projects in which "a project is one conversation" and Claude breaks work into parallel threads that run in the cloud; it is "available in beta to select Pro and Max subscribers who use Claude Code." Memory has also changed how context carries between chats. [Claude 101's](https://academy.claude.com/courses/claude-101/introduction-to-projects) video and the [projects help article](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects) say context is not shared across chats in a project unless it is added to the project knowledge, while the [memory article](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context) adds that each project has its own memory space and project summary (and memory is off by default on Team and Enterprise). Answer in the guide's terms: a Project is configured with instructions and knowledge sources, and the documented way to share context across its chats is the project knowledge.

!!! note "The help center disagrees with itself on the Free plan"

    Two projects articles ([overview](https://support.claude.com/en/articles/9517075-what-are-projects), [create and manage](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects)) and the [personalization article](https://support.claude.com/en/articles/10185728-understanding-claude-s-personalization-features) say "Projects are available to all users, including those with free Claude accounts. Free users can create a maximum of five projects." The personalization article's own "Choosing the right feature" list then says to "Use project instructions when you need specific guidance or context for a particular project (paid plans only)." The [pricing page](https://claude.com/pricing) lists Projects as "Up to 5" on Free. Plan details are date-sensitive; the objective itself does not depend on the plan.

**Go deeper:** [Projects](knowledge/claude-for-work.md#projects)

### D5.2 Manage uploaded knowledge and connectors (e.g., Google Drive, Gmail)

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Manage uploaded knowledge and connectors (e.g., Google Drive, Gmail)". In short (our framing): uploaded knowledge is a copy you manage, while a connector is a live link into another system, governed by that system's permissions and by your organization's settings.

**Know: upload limits** (as of September 2026; from [Upload files](https://support.claude.com/en/articles/8241126-upload-files-to-claude) unless noted).

| Where | Limits |
|---|---|
| File in a chat | 500MB per file; up to 20 files per chat; images up to 8000x8000 pixels; PDFs up to 1000 pages |
| PDF content | 100 pages or fewer: text and visual elements; 101 to 1000 pages: text only |
| Other documents | Text only; embedded images are not read |
| Spreadsheets (XLSX) | Code execution and file creation must be enabled to upload them |
| File in project knowledge | 30MB per file; text extraction only, except multimodal PDFs |
| Files Claude creates, and uploads for file creation | 30MB per file ([create and edit files](https://support.claude.com/en/articles/12111783-create-and-edit-files-with-claude)) |

Knowledge hygiene is part of management. The [usage and length limits article](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work) says: "Regularly clean up files you're no longer actively using in your projects." Before you upload, remove what the task does not need: an [AI Fluency lesson](https://academy.claude.com/courses/ai-fluency-for-nonprofits/understanding-privacy-and-data) advises you to "Work backwards from your actual goal to determine what data is truly necessary". [Domain 6](#domain-6-governance-risk-and-responsible-use) covers data sensitivity.

**Know: how connectors work.**

- "Connectors let Claude access your apps and services, retrieve your data, and take actions within connected services." Claude inherits each person's permissions from the connected service ([connectors article](https://support.claude.com/en/articles/11176164-use-connectors-to-extend-claude-s-capabilities)); in [Claude 101's](https://academy.claude.com/courses/claude-101/connecting-your-tools) example, "Connecting your work email doesn't give Claude access to your CEO's inbox".
- On Team and Enterprise, an Owner or Primary Owner must enable a connector for the organization, but "it doesn't automatically grant anyone access. Each person still needs to authenticate individually before they can use it." Enterprise-managed auth (beta, Team and Enterprise) lets an organization authorize a connector once, and the team inherits access on first login.
- Owners on Team and Enterprise can limit which actions a connected service can take. Each permission is set to Always allow, Needs approval, or Blocked, and "Restricting actions in Claude never grants more access than the source system permits". On Enterprise, a newly created role defaults to Needs approval on every connector ([role-based permissions](https://support.claude.com/en/articles/13930458-set-up-role-based-permissions-on-enterprise-plans)).
- You switch connectors on for each conversation from the "+" menu (or by typing "/"). The [Tool access setting](https://support.claude.com/en/articles/13730515-manage-claude-s-tool-access) decides how they load: Auto (the default) lets Claude decide which connectors to load; Always available loads all of them at the start of every conversation; On demand loads none until Claude searches for the right one for your request. With 10 or more active connectors, the connectors article suggests On demand. "Tools and connectors are token-intensive" ([usage and length limits](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work)), so turn off the ones a conversation does not need.
- Remote connectors work across all Claude surfaces; desktop extensions run locally and only in Claude Desktop and Claude Code ([desktop and web connectors](https://support.claude.com/en/articles/11725091-when-to-use-desktop-and-web-connectors)).
- Custom connectors use remote MCP ("an open standard, created by Anthropic, for AI applications to connect to tools and data"). Claude reaches the server from Anthropic's cloud, so it "must be reachable over the public internet from Anthropic's IP ranges". Free users can add one custom connector; on Team and Enterprise only Owners can add them. To change one, remove it and add it again ([custom connectors](https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp)).
- Safety, from the same article: connect only to servers built by organizations you trust, remember that malicious servers can carry hidden instructions (prompt injection), and click "Allow always" only for a server and tool "that you trust to run unsupervised."

**Know: the Google Workspace connectors named in the objective.** Gmail, Google Calendar and Google Drive connectors exist. On Team and Enterprise an Owner or Primary Owner enables them before users can authenticate, and Owners and Primary Owners can disable them in Organization settings > Connectors. Claude mirrors each user's existing Google permissions ([Google Workspace connectors](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors)).

| Connector | Management facts |
|---|---|
| Google Drive | Add a doc from "+" then "Add from Google Drive" ([claude.com Drive docs](https://claude.com/docs/connectors/google/drive.md)). Google Docs added to chats and projects "sync directly from Google Drive, so you're always working with the latest version" ([help center](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors)). You can only sync documents you have permission to view. In projects, Drive works only in private projects. Claude extracts the main text only: no images, comments or suggestions. The claude.com Drive page lists Google Docs "Up to 10MB, text extraction only" and says to convert .docx files with "Save as Google Docs". If you lose access to a doc, its preview disappears but the conversation history stays |
| Gmail | Claude "automatically detects when email data is needed" and cites the emails it used, with links back for verification ([claude.com Gmail docs](https://claude.com/docs/connectors/google/gmail.md)). Attachment content is not directly accessible: "metadata only" ([help center](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors)) |
| Both | "We do not train our models on your Gmail, Drive, or Calendar connector data". Retrieved data "is retained with its associated chat, so you can delete any retrieved data by deleting the chat." ([help center](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors)) |

Privacy detail: Anthropic's [training-data article](https://privacy.claude.com/en/articles/10023580-is-my-data-used-for-model-training) for the consumer plans (Free, Pro and Max, where chats are used for training if you allow it) says the chat data it may use excludes raw content from connectors, "though data may be included if it’s directly copied into your conversation with Claude". When a Google connection misbehaves ([help center](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors)): disconnect it in Customize > Connectors and re-authenticate on next use; if that fails, delete the Claude connection at myaccount.google.com/connections; if your Google Workspace admin blocks Claude, the admin may need to set Claude as "Trusted", and the policy takes about 15 minutes to propagate.

The [Microsoft 365 connector](https://support.claude.com/en/articles/15183774-connect-to-microsoft-365) lets Claude search and analyze content across SharePoint, OneDrive, Outlook and Teams in your work account; it is available on every plan, cannot use personal Microsoft accounts, and needs a one-time consent from a Microsoft Entra Global Administrator in each tenant.

Sharing interacts with connectors: on Team and Enterprise, "Chats with synced content can't be shared" ([connectors article](https://support.claude.com/en/articles/11176164-use-connectors-to-extend-claude-s-capabilities)), and in any shared chat snapshot the raw data from MCP tool calls stays hidden and attached files are not included ([share and unshare chats](https://support.claude.com/en/articles/10593882-share-and-unshare-chats)).

**Decide**

- If a source document changes often and is a Google Doc, add it from Drive; not an uploaded copy, because Google Docs added to chats and projects stay in sync and a copy goes stale (convert a .docx with "Save as Google Docs" first). In a project, that requires a private project.
- If a colleague cannot use a connector the organization has enabled, have them authenticate it themselves; not ask the admin to enable it again, because enabling "doesn't automatically grant anyone access".
- If someone expects Claude to read data they cannot open themselves, the answer is no, because Claude inherits their permissions.
- If a connector can send, delete or change things, choose Needs approval or Blocked for those actions (an Owner's setting on Team and Enterprise); not Always allow, unless you trust that action to run unsupervised (our rule, built from the help center's "Allow always" advice).
- If Claude's answer ignores a Gmail attachment, or images or comments in a Google Doc, suspect the connector's limits (attachment metadata only, main text only); not the prompt first.
- If a custom MCP server sits on an internal network, it will not work until it is reachable over the public internet from Anthropic's IP ranges; that is a network change, not something a prompt can fix.
- If a conversation does not need a connector, switch it off; not leave everything on, because connectors are token-intensive and use up context and usage limits.

**Traps**

- Believing that an admin enabling a connector gives everyone access, or that a restriction in Claude can give more access than the source system.
- Keeping a stale uploaded copy when a synced Google Doc would stay current.
- Expecting Claude to read Gmail attachments or Google Doc comments.
- Adding an untrusted MCP server, or clicking "Allow always" by default.
- Confusing desktop extensions (local, Claude Desktop and Claude Code only) with remote connectors (all surfaces).
- Uploading a full sensitive file and relying on an instruction to ignore the sensitive columns. The Sample 3 rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) rejects the closest shape it tests: "instructing the model not to retain data (C) does not satisfy the policy control". Its correct answer redacts or anonymizes "regulated identifiers before use". Remove or anonymize before uploading.

!!! warning "Official sources disagree on the Google connectors (as of September 2026)"

    The guide names Google Drive and Gmail only as examples and says nothing about what they can do. Anthropic's own pages disagree on the details: the help center's [Google Workspace connectors](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors) article against the claude.com [Gmail](https://claude.com/docs/connectors/google/gmail.md), [Calendar](https://claude.com/docs/connectors/google/calendar.md), [Drive](https://claude.com/docs/connectors/google/drive.md) and [getting-started](https://claude.com/docs/connectors/getting-started.md) pages.

    | Question | Help center | claude.com connector docs |
    |---|---|---|
    | Can Claude send email from Gmail? | "Send, reply to, and forward emails from Gmail. By default, Claude asks for your approval before each of these actions." | "Claude cannot create, send, or modify emails"; the getting-started page says "Claude can search your emails but can't send them" |
    | Can Claude change calendar events? | "Create, update, and delete events with full customization" | "Claude cannot create, modify, or delete calendar events" |
    | Can the Drive connector read Sheets and Slides? | "Read Sheets, Slides, PDFs, images, and MS Office files." | Google Sheets and Google Slides: "Not currently supported" |
    | Who can use the Google connectors? | "available for all users on Claude and Claude Desktop" | "Available on Pro, Max, Team, and Enterprise plans." |

    The help center article was last modified on August 17, 2026, but a date alone does not settle which page is right. Answer in the guide's terms: the objective is about managing uploaded knowledge and connectors (access, permissions, freshness), so do not build an answer on a capability these sources dispute.

**Go deeper:** [Search, Research and connectors](knowledge/claude-for-work.md#search-research-and-connectors)

### D5.3 Create effective system-level instructions

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Create effective system-level instructions". The official prep module words the same objective as "Create effective project-level instructions" ([Configuration & Knowledge Management](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/configuration-knowledge-management)). The guide does not define "system-level". Our mapping: in the Claude apps, the closest equivalents are the standing instructions that apply before you type a prompt, namely your Instructions for Claude, project instructions and organization instructions.

**Know: which layer an instruction belongs in.** The [personalization article's](https://support.claude.com/en/articles/10185728-understanding-claude-s-personalization-features) guidance:

- "Use profile instructions for account-wide settings that affect all your interactions with Claude."
- Use project instructions for project-specific context, workflow guidelines, requirements for a specific set of tasks, and "roles or perspectives Claude should adopt within the project".
- "Use skills when you want to customize how Claude formats and delivers its responses."

The [skills overview](https://support.claude.com/en/articles/12512176-what-are-skills) adds that skills "are task-specific and only load when relevant, making them better for specialized workflows." Organization-wide rules go in organization instructions, set by Owners and Primary Owners on Team and Enterprise under Organization settings > Organization and access ([organization instructions](https://support.claude.com/en/articles/14546867-set-organization-instructions)).

For personal standing instructions, the [Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/giving-cowork-context) suggests three things: who you are and what you do, the shorthand and acronyms you use, and how you like output delivered (format, length, tone). A good signal of what belongs there: corrections you keep repeating, such as "share the bottom line up front in your responses".

**Know: organization instructions** (from the [help center article](https://support.claude.com/en/articles/14546867-set-organization-instructions)).

| Rule | Detail |
|---|---|
| Length | "The maximum length is 3,000 characters." They are "included in every message sent by everyone in your organization, so shorter instructions help keep conversations efficient." |
| Timing | "Changes may take up to an hour to take effect across Claude products." |
| Precedence | "If an individual instruction directly contradicts an organization instruction, Claude favors the organization-level instruction." "Individual instructions still apply for anything the organization instructions don’t address." This prioritization "relies on prompt-level instructions", and in rare edge cases of direct contradiction, behavior may vary |
| Visibility | Only Owners and above can view or edit organization instructions; each user's own instructions are visible only to that user |
| Limits | Organization preferences cannot disable Claude's built-in safety guidelines or content policies |

Four of the article's six examples, quoted as published:

```text
Communication standards. “Respond in formal English. Use active voice. Avoid contractions and emojis.”
Response formatting. “Prefer concise responses under 300 words. Use bullet points for lists with three or more items.”
Domain context. “Our team works in healthcare claims processing. When users mention ‘claims,’ they’re referring to insurance claims, not legal claims.”
Data handling reminders. “Don’t include customer names, account numbers, or other personally identifiable information in responses or generated artifacts.”
```

The organization-instructions article states one precedence rule, organization over individual. [Claude 101](https://academy.claude.com/courses/claude-101/introduction-to-projects) says project instructions "work alongside any user preferences and styles you've set" and names no winner between your personal instructions and a project's.

**Know: what makes an instruction effective.**

1. **Specific, observable behavior.** The [organization instructions article's](https://support.claude.com/en/articles/14546867-set-organization-instructions) contrast: the vague instruction is "be professional"; the concrete version is `Respond in formal English. Don’t use contractions, slang, or emojis.` [Claude 101](https://academy.claude.com/courses/claude-101/introduction-to-projects) puts it plainly: "Vague instructions lead to inconsistent results."
2. **No conflicts.** "If your organization instructions contradict each other, Claude may not follow either one reliably." ([organization instructions](https://support.claude.com/en/articles/14546867-set-organization-instructions))
3. **Short and general.** Keep standing instructions to context, key guidelines and Claude's role; "Reserve task-specific instructions for the chat itself." ([usage and length limits](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work))
4. **Written for a newcomer.** The [prompt design article's](https://support.claude.com/en/articles/7996853-introduction-to-prompt-design) framing: "Think of Claude as a newly-hired contractor." Anthropic's [prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) add that explaining why an instruction exists helps, because "Claude is smart enough to generalize from the explanation."
5. **A role, not a straitjacket.** Setting a role focuses Claude's behavior and tone, but the [claude.com prompting blog](https://claude.com/blog/best-practices-for-prompt-engineering) warns: "Don't over-constrain the role."
6. **Tested.** "After saving, start a new conversation to verify Claude is following your instructions." Try several types of question ([organization instructions](https://support.claude.com/en/articles/14546867-set-organization-instructions)). An [AI Fluency exercise](https://academy.claude.com/courses/ai-fluency-for-students/ai-as-a-learning-partner) configures a study-buddy setup, then tests it by checking whether the AI "maintains its tutoring role or slips into just giving answers".

**Know: when the instruction is really a procedure, make it a skill.** Skills are "folders of instructions, scripts, and resources that Claude loads dynamically" ([skills overview](https://support.claude.com/en/articles/12512176-what-are-skills)); at startup Claude reads only each skill's name and description, and loads the full `SKILL.md` when a task matches the description ([claude.com skills overview](https://claude.com/docs/skills/overview.md)). Skills require code execution. On Free, Pro and Max, turn on "Code execution and file creation" in Settings > Capabilities; on Team and Enterprise, both "Cloud code execution and file creation" and Skills must be enabled in Organization settings > Plugins & skills, under the "Policy" tab. You manage skills in Customize > Skills, where a custom skill is uploaded as a ZIP file containing the skill folder ([Use skills in Claude](https://support.claude.com/en/articles/12512180-use-skills-in-claude)). `SKILL.md` "must start with YAML frontmatter containing required metadata, followed by markdown instructions". The `name` uses lowercase letters, numbers and hyphens only (maximum 64 characters, matching the directory name); the `description` explains what the skill does and when to use it (maximum 1,024 characters). An excerpt from the complete example in the [claude.com skills how-to](https://claude.com/docs/skills/how-to.md):

```markdown
---
name: brand-guidelines
description: Apply Acme Corp brand guidelines to presentations and documents, including official colors, fonts, and logo usage.
---

# Brand Guidelines

Apply these standards when creating presentations, documents, or marketing materials for Acme Corp.
```

If Claude is not using a skill, "Check that the skill's description field clearly explains when it should be used." ([Use skills in Claude](https://support.claude.com/en/articles/12512180-use-skills-in-claude)). Keep skills focused ("Multiple focused skills compose better than one large skill") and never hardcode API keys or passwords in one ([claude.com skills how-to](https://claude.com/docs/skills/how-to.md)). Install skills only from trusted sources: the [help center](https://support.claude.com/en/articles/12512180-use-skills-in-claude) names prompt injection and data exfiltration as the most significant risks.

!!! note "Two official pages give different skill metadata rules (as of September 2026)"

    The claude.com docs use `SKILL.md`, a lowercase hyphenated `name` and a `description` of up to 1,024 characters. The help center's [custom skills article](https://support.claude.com/en/articles/12512198-how-to-create-custom-skills) calls the file `skill.md`, describes `name` as "A human-friendly name for your skill (64 characters maximum)", caps the description at 200 characters, and its example uses `name: Brand Guidelines`. A lowercase hyphenated name of up to 64 characters and a description under 200 characters satisfy both pages (our reading). The guide names neither limit; its D5.3 objective is "Create effective system-level instructions".

**Decide**

- If a preference should apply to all your chats (your role, or bottom line up front), put it in Instructions for Claude; if it applies to one body of work, put it in that project's instructions; if it is a repeatable procedure or output format you want anywhere in Claude, make it a skill; if it must apply to everyone in the organization, it belongs in organization instructions, set by an Owner or Primary Owner.
- If an instruction is vague ("be professional"), rewrite it as behavior you can check; not add more adjectives, because vague instructions "lead to inconsistent results".
- If two instructions conflict, resolve the conflict; not leave Claude to pick one, because Claude may follow neither reliably.
- If a rule must actually be enforced (for example, no customer identifiers leave the team), choose controls that act on the data and the access: remove the data before it reaches Claude, and use admin controls such as connector permissions. Not an instruction alone, because instruction precedence "relies on prompt-level instructions" ([organization instructions](https://support.claude.com/en/articles/14546867-set-organization-instructions)) and the Sample 3 rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) says instructing the model not to retain data "does not satisfy the policy control".
- After changing any standing instruction, test it in a new conversation with several kinds of question, as the organization instructions article advises.

**Traps**

- Vague instructions such as "be professional".
- Contradictory instructions, or long organization instructions that are sent with every message.
- Task-specific details stored as standing instructions.
- Expecting organization instructions to switch off Claude's safety guidelines.
- Treating an instruction as a data-protection control (see the enforcement rule under Decide above, and [Sample 3](#sample-3-regulated-personal-data-in-a-spreadsheet)).
- Reaching for the "Use style" menu to set tone; it has been deprecated (see the box below).

!!! warning "Older course material vs current docs"

    - **Styles.** [Claude 101's](https://academy.claude.com/courses/claude-101/your-first-conversation-with-claude) December 2025 video shows a "Use style" menu; the course now notes that it "has since been deprecated", and the current [personalization article](https://support.claude.com/en/articles/10185728-understanding-claude-s-personalization-features) points to skills for customizing how Claude formats and delivers responses. Claude 101's project lesson still says instructions "work alongside any user preferences and styles you've set."
    - **Where personal instructions live.** Claude 101 points to "Settings > Account > 'Instructions for Claude'"; in the new Claude experience, Cowork's Global instructions setting "is now part of Instructions for Claude in Settings > General" ([Cowork and chat are one Claude](https://support.claude.com/en/articles/16761823-claude-cowork-and-chat-are-one-claude)).

    The guide (July 2026) names no personalization styles and no menu paths. Answer in the guide's terms: the concept tested is what makes a standing instruction effective and where it applies, not a menu location.

**Go deeper:** [Memory, styles and personalization](knowledge/claude-for-work.md#memory-styles-and-personalization)

### D5.4 Inform, maintain, and update Claude configurations, knowledge sources, and instructions

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Inform, maintain, and update Claude configurations, knowledge sources, and instructions". The guide's audience includes "internal staff who maintain and optimize ongoing AI-enabled workflows", and the official prep course's [Troubleshooting & Optimization module](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/troubleshooting-optimization) lists "stale configuration" among the root causes of an underperforming prompt or output, which [Domain 7](#domain-7-troubleshooting-and-optimization) diagnoses.

**Know: why configurations go stale.**

- "Outdated documents can lead to outdated responses. Review and update your project knowledge periodically." ([Claude 101](https://academy.claude.com/courses/claude-101/introduction-to-projects))
- "Treating config as permanent" is a pitfall named in Anthropic's [Cowork rollout tutorial](https://academy.claude.com/tutorials/scaling-workflows-with-claude-cowork-at-your-organization), which says to "revisit connectors and posture with usage evidence". An [AI Fluency course](https://academy.claude.com/courses/ai-fluency-for-creative-work/putting-it-all-together) makes the same point about AI use policies: "A policy without revision triggers goes stale silently."
- Corrections in chat do not fix the configuration: the model does not learn from your corrections (see [D1.3](#d13-iterate-prompts-to-improve-output-quality)), and other chats in the project do not share that context unless it is added to the project knowledge (see [D5.1](#d51-configure-claude-projects-with-instructions-and-knowledge-sources)).
- Models change under you. On Enterprise, the default-model option "Use Anthropic’s recommended default" updates automatically when new models are released, while "Choose a specific model" selects one that won't change when new models are released ([default model settings](https://support.claude.com/en/articles/15330088-set-a-default-model-for-your-organization)). An [Academy tutorial](https://academy.claude.com/tutorials/building-ai-fluent-organizations) says that when people believe "today's AI is the worst you'll ever use", "they keep coming back to retest old assumptions and try new things."

**Know: a maintenance routine built from the official guidance** (the routine is our assembly; each row is sourced).

| Practice | What the source says |
|---|---|
| Name an owner | "Every shared plugin has a named person who reviews changes, runs the evals after edits, and decides when to update or retire it." ([Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/share-what-you-build-with-your-team)). Every quarter "and every time a champion moves teams", check who owns each plugin and scheduled task and "move that ownership on purpose" ([Cowork rollout tutorial](https://academy.claude.com/tutorials/scaling-workflows-with-claude-cowork-at-your-organization)) |
| Set a cadence | "Quarterly is a reasonable starting point to look at what's installed, what's actually getting used, and what's gone stale." ([Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/share-what-you-build-with-your-team)). Organization instructions: "Review and update regularly." ([organization instructions](https://support.claude.com/en/articles/14546867-set-organization-instructions)) |
| Add revision triggers | An [AI policy use case](https://academy.claude.com/use-cases/generate-an-ai-policy) keeps an annual review schedule but adds triggers for interim updates: when adopting new AI tools, after policy violations, when regulations change, or based on sector guidance |
| Gate changes with tests | "Treat the eval loop as the gate": if the cases you care about do not hold up after a change, do not push it to everyone ([Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/share-what-you-build-with-your-team)). A simple eval starts with "5-10 examples of a task you do regularly" ([Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results)) |
| Change one thing at a time | "Change one thing at a time." Pick the problem that matters more, fix it, re-run, then review again ([Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/validating-skills-for-plugins)) |
| Test in a fresh chat | "start a new conversation to verify Claude is following your instructions" ([organization instructions](https://support.claude.com/en/articles/14546867-set-organization-instructions)) |
| Curate | "promote winners to plugins, archive stale skills, merge duplicates" ([Cowork rollout tutorial](https://academy.claude.com/tutorials/scaling-workflows-with-claude-cowork-at-your-organization)); name skills specifically (`sales-customer-renewal-prep`, not `meeting-prep`) ([Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/share-what-you-build-with-your-team)) |
| Automate a refresh | An [Academy use case](https://academy.claude.com/use-cases/my-voice) keeps a writing-voice skill current with a Cowork scheduled task: "Your writing shifts as your role does." Cowork re-reads the last 90 days and merges what's new into the skill "without losing the rules you've added by hand" |
| Clean knowledge | Remove files you no longer use ([usage and length limits](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work)); review and update outdated documents ([Claude 101](https://academy.claude.com/courses/claude-101/introduction-to-projects)); prefer synced Google Docs for sources that change ([Google Workspace connectors](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors)) |

The sources give no single cadence: Claude 101 says "periodically" for project knowledge, the organization instructions article says "regularly" for organization instructions, and the Cowork course suggests quarterly for shared plugins.

**Know: informing configurations and the people who use them.** The guide does not define "Inform". Read as a verb on the same object as "maintain" and "update", it means supplying configurations, knowledge sources and instructions with the information they need (our reading). In a team it also means the people who depend on a configuration know what changed and receive the update (our reading). The help center makes the second point for one setting: "If you turn off project sharing, let your teams know." ([control project sharing](https://support.claude.com/en/articles/9927533-control-project-sharing-for-your-organization)). The mechanics that decide who receives an update:

- **Shared skills** are view-only: recipients can use them but not edit them, and "If you update the skill later, recipients automatically get the updated version at next use." "Sharing gives a skill to specific people or groups, and you keep control of it. Publishing hands it to your organization." ([Use skills in Claude](https://support.claude.com/en/articles/12512180-use-skills-in-claude))
- **Published skills** belong to the organization: "Once published, the skill is managed by your organization. To update it, publish again." Where publishing requires review, "everyone who uses the skill stays on the approved version until the update is approved." ([Use skills in Claude](https://support.claude.com/en/articles/12512180-use-skills-in-claude)). Under the "Requires review" setting, an owner must approve each submitted skill or plugin before it is published, and the organization's Inventory tab offers "View version history" ([provision and manage skills](https://support.claude.com/en/articles/13119606-provision-and-manage-skills-for-your-organization)).
- **Organization-provisioned skills** are provisioned to all users immediately and on by default, but each user can switch one off; only owners can add or remove them. Anthropic's advice in the same article: "Upload and test skills on your own account first to verify they work as expected before distributing them organization-wide."
- **Shared plugins** cannot be edited by teammates; "updates flow from whoever maintains it" ([Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/share-what-you-build-with-your-team)). An organization can mark a plugin Installed by default (members may uninstall), Required (members cannot), or Not available (hidden from the catalog, useful for staging or deprecating) ([Cowork on Team and Enterprise](https://support.claude.com/en/articles/13455879-use-claude-cowork-on-team-and-enterprise-plans)).
- **Timing.** Organization instruction changes can take up to an hour to take effect; on Enterprise, role changes to project sharing can take up to 15 minutes.
- **A written record.** Ask Claude for a change summary, as a [Use Case Gallery example](https://academy.claude.com/use-cases/understand-and-extend-an-inherited-spreadsheet) does for a spreadsheet: "Can you create a summary of every change made to this file? List what was added, what formulas were extended, and what assumptions I should call out as new." As of September 2026, [Claude Docs](https://support.claude.com/en/articles/16923645-get-started-with-claude-docs) and [Claude Design](https://support.claude.com/en/articles/14604416-get-started-with-claude-design) have no version history yet, so keep that record another way.

As of September 2026, two scheduled admin changes show why configuration needs watching (our reading). For Enterprise organizations that have not chosen a Publishing setting for skills and plugins, the setting "switches to “Requires review” on October 2, 2026" ([provision and manage skills](https://support.claude.com/en/articles/13119606-provision-and-manage-skills-for-your-organization)). Skill and plugin scanning "is off by default until October 2, 2026", when it turns on for Enterprise organizations that have not set it ([skill and plugin scanning](https://support.claude.com/en/articles/15927065-get-started-with-skill-and-plugin-scanning)).

**Know: retiring and cleaning up.** "Archiving a project doesn't reset its sharing permissions or remove members", so remove people explicitly to revoke access, and an archived project must be unarchived before it can be deleted ([projects help article](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects)). For memory, Settings > Memory shows what Claude remembers, and you can tell Claude in a chat what to remember, change or forget ([memory article](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context)). For Team and Enterprise plans using memory, the [projects help article](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects) adds that "Remove from project" takes a stray chat out of a project's memory summary so it goes into Claude's non-project memory instead.

**Decide**

- If a source document changes (a new price list, a revised policy), replace it in project knowledge and remove the old version; not correct Claude in one chat, because a chat correction does not change the project knowledge every chat reads, and the projects help article says context is not shared across chats unless it is added to the project knowledge base (memory, where it is on, is no substitute for updating the source).
- If you change a shared skill, instruction set or plugin, run your test cases before publishing; not publish and see, because the Cowork course treats the eval loop "as the gate" and says not to push a change whose cases do not hold up.
- If the owner of a shared configuration changes teams, reassign ownership deliberately; not leave it with nobody, because the rollout tutorial says to check who owns each plugin and scheduled task every quarter and every time a champion moves teams, and to move that ownership on purpose.
- If teammates should receive your updates automatically while you keep control, share the skill; if the organization should own it, publish it, knowing that each later update is published again.
- If access to an archived project must end, remove the members; not archive alone, because archiving does not remove members.
- If you change a configuration that other people rely on, tell them what changed (our rule; the project sharing article gives the same advice for turning sharing off).

**Traps**

- Set-and-forget configuration with no owner, no review date and no revision triggers.
- Fixing stale knowledge with a chat correction instead of updating the source.
- Publishing an edited skill or instruction set without testing it.
- Assuming archiving a project removes its members.
- Leaving superseded files in project knowledge next to their replacements, against the help center's advice to clean up files you no longer use.

**Go deeper:** [Admin controls for Team and Enterprise](knowledge/claude-for-work.md#admin-controls-for-team-and-enterprise)

## Domain 6: Governance, Risk, and Responsible Use

**Weight:** 15% of scored items. If all 60 items were scored, that is about 9.0 items (15% of 60, our arithmetic); the guide does not say how many of the 60 are scored.

The domain has four objectives: identify appropriate and inappropriate use cases; apply data sensitivity, regulatory and privacy considerations; follow organizational AI policies and governance standards; and understand the ethical implications of AI usage. The matching module of the free prep course, Governance, Risk & Responsible Use (55 minutes), sums it up as "Build the judgement to decide what is safe and appropriate to bring to Claude." Its [course page](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/governance-risk-responsible-use) says avoiding governance problems primarily comes down to judgment you can learn: "knowing what's fair game, what your data and industry allow, and which rules your organization has in place". Those three judgments line up with the first three objectives (our mapping), and the page's four learning objectives restate the guide's four.

One step in the guide's preparation advice targets this domain directly: "Practice responsible-use judgment: data sensitivity, appropriate use cases, and when to escalate or seek human review". Its candidate profile expects awareness of "where adoption risks exist, and how to align usage with business needs and responsible-AI practices", and its recommended experience includes "A practical understanding of AI limitations, including hallucinations, context constraints, and data sensitivity" ([CCAO-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)).

!!! tip "One official item, one pattern"

    The guide's three sample questions map to Domains 2, 3 and 6, so Sample 3 (see [Official sample questions](#official-sample-questions)) is the only official item for this domain. Its keyed answer applies a safeguard so the work can go ahead within policy, and its rationale rejects the three other shapes: ignoring the policy, putting an instruction to the model in place of a control, and abandoning the task when a safe route exists. Our rule of thumb: test every option in this domain against those three shapes. The Decide and Traps lists below are our own decision rules, drawn from Anthropic's published guidance and the Sample 3 rationale.

### D6.1 Identify appropriate and inappropriate use cases

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Identify appropriate and inappropriate use cases". In our framing, a use case can be inappropriate for three different reasons: a policy rules it out, Claude's known limits make it unreliable without an added step, or the stakes mean a person has to make the call. Sort each scenario by asking which of the three applies.

**Know: what Anthropic's Usage Policy rules out.** The [Usage Policy](https://www.anthropic.com/legal/aup) (effective September 15, 2025) applies to anyone who can submit inputs to Anthropic's products and services, including through authorized resellers or passthrough access. It has three parts: Universal Usage Standards for all users and use cases, High-Risk Use Case Requirements for specific consumer-facing uses that pose an elevated risk of harm, and Additional Use Case Guidelines (consumer-facing chatbots, products serving minors, agentic use, MCP servers). Prohibitions from the Universal Usage Standards that bear directly on everyday business work (our selection):

- Do not "Impersonate a human by presenting results as human-generated".
- Do not "Plagiarize or submit AI-assisted work without proper permission or attribution".
- Do not misuse, collect, solicit or gain access without permission to private information such as non-public contact details, health data, biometric or neural data (including facial recognition), or confidential or proprietary data.
- Do not intentionally bypass the guardrails in Anthropic's products to make the model produce harmful outputs (for example by jailbreaking or prompt injection) without prior authorization from Anthropic, and do not use inputs and outputs to train an AI model (model scraping or distillation) without prior authorization.
- Do not make determinations in criminal justice applications, such as decisions about parole or sentencing.

**Know: the seven high-risk areas and their two requirements.** When Claude is used to provide advice, recommendations or subjective decisions directly affecting individuals or consumers in these areas, "a qualified professional in that field must review the content or decision prior to dissemination or finalization." The policy adds: "You or your organization are responsible for the accuracy and appropriateness of that information." If model outputs are presented directly to individuals or consumers, AI use must be disclosed, and "This disclosure must be provided at a minimum at the beginning of each session." ([Usage Policy](https://www.anthropic.com/legal/aup))

| High-risk area | What the policy includes |
|---|---|
| Legal | Legal interpretation, legal guidance, decisions with legal implications |
| Healthcare | Healthcare decisions, medical diagnosis, patient care, therapy, mental health, other medical guidance; wellness advice (sleep, stress, nutrition, exercise) is excluded |
| Insurance | Underwriting, claims processing or coverage decisions for health, life, property, disability or other insurance |
| Finance | Investment advice, loan approvals, financial eligibility or creditworthiness |
| Employment and housing | Employability decisions, resume screening, hiring tools, other employment determinations, housing eligibility (including leases and home loans) |
| Academic testing, accreditation and admissions | Standardized testing for school admissions (evaluating, scoring or ranking prospective students), language proficiency or professional certification exams; agencies that evaluate and certify educational institutions |
| Media or professional journalistic content | Automatically generating content and publishing it for external consumption |

Anthropic's research on discrimination in model decisions draws a firm line for the high-risk decisions it studied: "we do not endorse or permit the use of language models to make automated decisions for the high-risk use cases we study" ([Evaluating and mitigating discrimination](https://www.anthropic.com/research/evaluating-and-mitigating-discrimination-in-language-model-decisions)). The Academy's AI-policy use case, written for a youth mental health nonprofit, asks Claude to draft a policy that covers "Prohibited uses (clinical decisions, automated beneficiary assessments)" ([Generate an AI policy](https://academy.claude.com/use-cases/generate-an-ai-policy)).

**Know: allowed is not the same as a good fit.** The [AI Fluency course](https://academy.claude.com/courses/ai-fluency-framework-foundations/the-4d-framework) calls this Delegation: "Thoughtfully deciding what work to do with AI vs. doing yourself." The step-by-step sort (AI can handle, AI can assist while a human decides, human should handle) is covered under [Domain 4](#domain-4-workflow-integration-and-solution-design). For this objective, know the limits that make a task a poor fit unless you add a step. Each is taught in Domain 2: knowledge "frozen at the knowledge cutoff" ([AI Capabilities and Limitations](https://academy.claude.com/courses/ai-capabilities-and-limitations/knowledge)) and fabrication that concentrates in specific details ([D2.2](#d22-identify-hallucinations-inconsistencies-and-biases-in-responses)); calculations that need recomputing ([D2.3](#d23-apply-fact-checking-and-validation-techniques)); and uses Anthropic's own documentation does not recommend without human review or verification, such as final client deliverables and audit-critical calculations in Claude for Excel ([D2.4](#d24-determine-when-human-review-or-additional-verification-is-required)).

A quick empirical test is Claude 101's delegation diligence loop (described under [D2.1](#d21-evaluate-claude-generated-outputs-for-accuracy-and-completeness)): reproduce a past analysis whose answer you already know, refine and test again, and if Claude still cannot match your known results, you have learned that the task is one you should not delegate.

**Know: where Associate scope ends.** A use case can also be inappropriate for an Associate to build: enterprise-scale architecture and integration work belongs to the Claude Architect and Claude Developer credentials. The signals that work has left Associate scope are listed under [D4.3](#d43-use-claude-to-support-solution-design-development-and-iteration).

**Decide**

- If a policy (Anthropic's or your organization's) rules out the purpose, choose not to use Claude for it; not a reworded prompt, because the policy governs what the work is for, not how the request is phrased.
- If the output advises on or decides something about a person in a high-risk area, choose Claude as a drafting and analysis aid with a qualified professional reviewing before anything is finalized; not an automated decision, because the Usage Policy requires that review for its High-Risk Use Cases (specific consumer-facing uses with an elevated risk of harm) and Anthropic's discrimination research says it does not permit automated decisions for the high-risk use cases it studied.
- If the task depends on current facts or exact figures, choose a grounded approach (web search, retrieval through connectors, or source documents you supply) plus verification; not Claude's trained knowledge alone, because that knowledge stops at the cutoff and fabrication clusters in specifics.
- If the work means building a production system or an integration, choose to escalate to a Developer or Architect; not to ship an artifact as the product, because artifacts are best for testing and demonstration.

**Traps**

- Refusing the whole task when a safeguard would make it acceptable. Sample 3's rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) calls abandoning the task "unnecessary when anonymization enables it", and [Claude's constitution](https://www.anthropic.com/constitution) states that unhelpfulness is never trivially safe from Anthropic's perspective.
- Treating Claude's confidence as permission. Sample 1's rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf): "Self-reported confidence (A, C) is not a reliable accuracy signal".
- Automating a decision about people because the model looks accurate on average.
- Passing Claude's output off as human-written, or submitting it without attribution.

**Go deeper:** [Anthropic's Usage Policy](knowledge/security-and-governance.md#anthropics-usage-policy)

### D6.2 Apply data sensitivity, regulatory, and privacy considerations

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Apply data sensitivity, regulatory, and privacy considerations". Sample 3, which the guide maps to Domain 6, tests this objective most directly (our mapping); its rationale defines the skill: "redacting or anonymizing regulated identifiers before use, so the analysis can proceed without exposing protected data" ([CCAO-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)). The Academy's [nonprofit privacy lesson](https://academy.claude.com/courses/ai-fluency-for-nonprofits/understanding-privacy-and-data) makes the same point from the other side: "Safe AI use isn't about avoiding it".

**Know: classify, minimize, match the tool.** The three steps below are our ordering of Anthropic's advice.

1. **Classify.** Mark anything you would not want outside your business: the Academy's list is names, contact details, payment information and proprietary pricing. Anthropic's help-center article for consumer plans adds financial information (SSN, credit card numbers, bank account details), health records or medical information, passwords or private login credentials, and confidential business or personal documents to the details to be thoughtful about sharing. The Academy's sensitivity exercise asks four questions of a dataset: which fields contain personally identifiable information, which information is essential for the analysis, which could be removed or anonymized without losing analytical value, and what the worst case would be if the data were exposed. At organization level, Anthropic and Accenture's guide [Deploying AI from pilot to production](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf) advises classifying data by sensitivity level (public, internal, customer PII, financial and clinical, for example) before the pilot architecture is set.
2. **Minimize.** The Academy's [nonprofit privacy lesson](https://academy.claude.com/courses/ai-fluency-for-nonprofits/understanding-privacy-and-data) says: "Work backwards from your actual goal to determine what data is truly necessary"; "For pattern analysis, you likely don't need names, contact details, or other PII." Work on a copy: replace names with placeholders such as Customer A or Vendor X, remove exact figures you do not need, and delete contact details entirely. The Academy adds that you can often get the full benefit without sharing sensitive information by breaking the task into component parts.
3. **Match the tool.** "higher sensitivity data needs stricter privacy settings" ([Using data with AI](https://academy.claude.com/courses/ai-fluency-for-small-businesses/using-data-with-ai)); tools with more protection allow safer sharing of sensitive data. In practice (our reading), that means the plan, settings and features your organization has approved for that class of data.

**Know: what happens to your data, by plan (as of September 2026).**

| | Free, Pro, Max | Team, Enterprise (Claude for Work) |
|---|---|---|
| Terms | Consumer Terms and Privacy Policy | Commercial Terms, with data processed under the Data Processing Addendum; the organization is the controller and Anthropic the processor, and the Consumer Terms and Privacy Policy do not apply |
| Model training | Opt-out: chats are used if you allow it (Settings > Privacy > "Help Improve our AI models"); even with the setting off, chats flagged for safety review may still be used (including to train models for Anthropic's Safeguards team), and thumbs up or down feedback may be used to train models | Not used by default; explicit feedback (thumbs up or down) is an exception, see below |
| Retention | Deleted chats leave back-end storage within 30 days; if you allow training, de-identified data may be kept for up to 5 years | Enterprise Primary Owners and Owners can set a custom retention period (minimum 30 days); without one, Enterprise data is retained indefinitely |

**Know: features that change the picture (as of September 2026).**

- **Feedback.** A thumbs up or down stores the entire related conversation for up to 5 years, on commercial plans too, and on commercial plans explicitly reported feedback is one of the cases where chats may be used for training. Team and Enterprise Primary Owners and Owners can turn this off with the Rate chats setting.
- **Incognito chats** are not saved to chat history or memory and are not used for training. They are still retained for 30 days by default (longer under an Enterprise custom retention setting), are included in organization data exports on Team and Enterprise and in the Compliance API on Enterprise, and are not available inside projects.
- **Memory** does not store sensitive topics such as health, race, ethnicity, religious beliefs, politics or gender identity by default. It never saves government ID numbers, criminal history, financial account numbers or immigration status, even on request. It is off by default on Team and Enterprise until an owner turns it on.
- **Connectors** inherit each person's permissions in the connected service, so they reach only what that person can already reach. Anthropic does not train on Gmail, Drive or Calendar connector data; retrieved data stays with its chat, so deleting the chat deletes it. Connected services process data on their own infrastructure under their own terms, and the Enterprise US-only inference setting does not change where they operate.
- **Sharing.** The help center's [public links article](https://support.claude.com/en/articles/16762437-public-links-for-shared-chats) says: "The simple rule: treat a public link as public." Public links are available on Free, Pro and Max; Team and Enterprise members can only share chats inside their organization. A shared chat snapshot leaves attached files out, but sharing an artifact made in chat also gives viewers the attachments and files from that conversation. Artifact creators decide which data uses personal or shared storage, so before typing sensitive information into an artifact, check whether it uses shared storage.
- **Code execution and file creation.** Anthropic's [file creation article](https://support.claude.com/en/articles/12111783-create-and-edit-files-with-claude) warns that "Claude can be tricked into sending information from its context (for example, prompts, projects, data via MCP, Google integrations) to malicious third parties." Its advice is to monitor Claude while it works and stop it if it uses or accesses data unexpectedly. Disabling network access keeps data inside Claude's sandbox, although MCP integrations can still reach the network whatever the egress setting. Network access is on for Free, Pro and Max, and new Enterprise organizations start with network egress off. For Team the article is inconsistent: its availability list says network access is off by default, but its setup section says Team starts with egress to package managers only switched on. Check Organization settings > Capabilities.

**Know: regulated data (as of September 2026).** Sector rules sit on top of company policy.

- Among Claude plans, HIPAA readiness is Enterprise-only (the Claude API has its own HIPAA readiness, enabled from the Claude Console); Team, Free, Pro and Max cannot enable it. Enabling it does not bring every feature under the BAA: Cowork is not yet covered, and Claude Code is covered only with zero data retention on qualified accounts. Claude in Chrome is not available to organizations covered by HIPAA, and Anthropic recommends against using it on pages that contain regulated data. Memory is not available to organizations with HIPAA, public-sector or custom data retention agreements.
- Anthropic and Accenture's [pilot-to-production guide](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf), among its industry-specific compliance examples, says healthcare programs touching patient data need HIPAA-compliant infrastructure and "signed business associate agreements before a vendor relationship goes live".
- The Academy's [K-12 course](https://academy.claude.com/courses/ai-fluency-for-k-12-educators/ethics-responsible-use) gives a sector example: "Data protections like FERPA apply to AI tools. Use only district-approved tools for anything with student information."
- For legal work, Anthropic's [help center](https://support.claude.com/en/articles/15707726-using-claude-for-legal-work-privilege-confidentiality-and-how-to-think-about-configuration) treats commercial terms with a DPA and the no-training commitment as "the baseline expectation for legal technology", together with a plan for keeping a lawyer in the loop, verifying output against primary sources and documenting AI use.

If something goes wrong, the Academy's advice is to delete the conversation, request data deletion through the platform's privacy process, and follow your organization's protocols.

**Decide**

- If policy restricts the data, choose to remove or anonymize the identifiers before uploading; not to upload as-is, not to upload with an instruction to Claude, and not to drop the analysis, because anonymizing is the control that lets the work proceed.
- If the task genuinely needs the sensitive data, choose only a plan and configuration your organization has approved for that class of data, and ask your admin or compliance team when unsure; not a personal account, because Free, Pro and Max run under the Consumer Terms with training as an opt-out setting.
- If a feature sits outside your organization's arrangement for regulated data (Cowork is not yet under Anthropic's BAA), choose to keep that data out of it; not to assume every feature inherits the arrangement.
- If content must stay inside the company, choose sharing within the organization; not a public link, because a public link is public.

**Traps**

- Treating internal analysis as exempt from the policy. Sample 3's rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf): "Uploading as-is (A) violates policy".
- Telling Claude not to keep the data. The same rationale in the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) says this "does not satisfy the policy control".
- Using incognito as a workaround. It keeps a chat out of history, memory and training, but the chat is still retained for 30 days by default and, on Team and Enterprise, appears in organization data exports (and in the Compliance API on Enterprise).
- Moving company data into a personal account because it is quicker. Free, Pro and Max all run under the Consumer Terms with model training as an opt-out setting, so a paid personal plan does not bring the data under your organization's commercial terms.
- Assuming shared work carries no files. A shared chat snapshot leaves attached files out, but a shared artifact made in chat gives viewers the conversation's attachments.

**Go deeper:** [Data retention, training and compliance](knowledge/security-and-governance.md#data-retention-training-and-compliance)

### D6.3 Follow organizational AI policies and governance standards

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Follow organizational AI policies and governance standards". Two layers apply at once. Anthropic's Usage Policy binds every user, and the Commercial Terms require customers and their users to use the services in compliance with it. On top of that sits your organization's own AI policy, enforced partly through admin settings in Claude.

**Know: what an organizational AI policy covers.** The Academy's worked example of generating an AI policy (for a youth mental health nonprofit) asks for these sections ([Generate an AI policy](https://academy.claude.com/use-cases/generate-an-ai-policy)):

| Section | What it covers |
|---|---|
| Governance structure | Who approves AI tool adoption, a risk assessment framework, oversight responsibilities |
| Data privacy and protection | What data can or cannot be used with AI tools, beneficiary and donor data safeguards, data retention and deletion protocols |
| Appropriate use cases | Approved applications, prohibited uses, gray areas requiring case-by-case review |
| Staff guidelines | Training requirements, verification responsibilities, when to escalate decisions to humans, documentation requirements |
| Ethical considerations | Bias detection and mitigation, transparency with beneficiaries and donors, mission alignment, community impact |

The same [example](https://academy.claude.com/use-cases/generate-an-ai-policy) tells you to "have legal counsel review the policy to ensure it meets your jurisdiction's requirements before board adoption", and to add triggers for interim updates on top of the annual review: adopting new AI tools, policy violations, regulation changes and sector guidance. The Academy's [creative-work course](https://academy.claude.com/courses/ai-fluency-for-creative-work/putting-it-all-together) makes the same point: "A policy without revision triggers goes stale silently."

**Know: how admins enforce policy in Claude for Work (as of September 2026).** These settings are how an organization's AI policy reaches each member's Claude account.

| Control | What it does |
|---|---|
| Organization instructions | Owners and Primary Owners on Team and Enterprise set instructions that go with every message and win over a directly contradicting individual instruction, but cannot disable Claude's built-in safety guidelines or content policies (limits and precedence under [D5.3](#d53-create-effective-system-level-instructions)) |
| Connectors | An Owner or Primary Owner enables connectors for the organization, each person still authenticates, and Owners can set each permission to Always allow, Needs approval or Blocked, org-wide, which only narrows what the source system already permits (details under [D5.2](#d52-manage-uploaded-knowledge-and-connectors-eg-google-drive-gmail)) |
| Capabilities | An Owner or Primary Owner enables web search on Team and Enterprise; code execution network access is off by default for new Enterprise organizations (the help article contradicts itself on the Team default, see D6.2); memory is off by default; Cowork is on by default and owners can disable it (on Enterprise, Cowork on the web, mobile and the Chrome side panel is available only where an admin has enabled it); the admin controls whether Cowork's Automatically approve mode is available |
| Project sharing | "Share projects" (sharing with others in the organization) and its sub-setting "Public projects" (all users in the organization can see and start chats in public projects) are both on by default; Owners change them in Organization settings > Data and privacy |
| Skills | Owners can provision skills to everyone, turn off user-created skills, and set Publishing to Off, Open (no review) or Requires review; an Enterprise organization that has not chosen a publishing setting switches to "Requires review" on October 2, 2026. Enterprise skill and plugin security scanning, which blocks items with malicious content, is off by default until that date and then on by default where available for organizations that have not set it |
| Claude in Chrome | Allowlists restrict Claude to approved sites and blocklists stop it reaching specific sites; Anthropic recommends starting with a restrictive allowlist |
| Enterprise roles and models | In custom roles the most restrictive level wins; model access settings can disable models and cap effort (Haiku models are always available); a model you expect may be missing because it was turned off for your role |
| Oversight (Enterprise) | Audit log exports cover the past 180 days and contain identifiers, not chat or project titles or content; the Compliance API pulls activity feed events, chat data and file content; custom retention periods |

**Know: obligations that no setting enforces.**

- The Commercial Terms make the customer responsible for evaluating whether outputs are appropriate, including where human review is appropriate, and for telling users that factual assertions in outputs should not be relied on without independent checking.
- Supply chain: install skills only from trusted sources; connect only to MCP servers built and hosted by organizations you trust, and click "Allow always" only for a server and tool you trust to run unsupervised; treat someone else's artifact "the way you'd treat a file from an unknown sender" ([Publish and share artifacts](https://support.claude.com/en/articles/9547008-publish-and-share-artifacts)). The [prep module](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/governance-risk-responsible-use) names "a Skill with too much access" as one of the incidents that can put a hold on Claude for an entire team.
- Accountability for automated work, approval modes and customer-facing safeguards are covered under [Domain 4](#domain-4-workflow-integration-and-solution-design). The help center's [Cowork safety article](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely) states the principle: "You remain responsible for all actions taken by Claude performed on your behalf."

!!! note "Approval-mode names changed (as of September 2026)"

    Cowork's "Manually approve (Manual)" was formerly called "Ask before acting", and "Skip all approvals (Skip)" was formerly "Act without asking". In Skip mode "Claude doesn't pause to ask and nothing checks its actions automatically" ([Get started with Claude Cowork](https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork)). A third mode, Automatically approve (Auto), keeps working without asking at every step while Claude reviews each action for safety. In the new Claude experience that merges chat and Cowork (rolling out gradually, starting with Pro and Max), Manual is the default. Recognize both sets of names.

**Decide**

- If an admin setting blocks something you need, choose to ask the owner or admin through your organization's process; not a workaround such as a personal account, another tool or Skip all approvals, because the setting is how the policy is enforced.
- If an individual instruction directly contradicts an organization instruction, expect Claude to favor the organization's; not your own, because the help center says Claude favors the organization-level instruction.
- If the policy does not address a new tool or use, choose to ask the policy owner before relying on it; not to read silence as permission, because the Academy's AI-policy use case asks for a policy that covers who approves AI tool adoption and gray areas requiring case-by-case review, and advises adding triggers for interim updates, such as adopting new AI tools.
- If Claude asks you to approve a connector or MCP tool call, choose to read each request before approving, and click "Allow always" only for a server and tool you trust to run unsupervised; not to treat an owner's Needs approval or Blocked setting as something to route around, because owners' connector restrictions apply org-wide and individual users can't override them.

**Traps**

- An instruction to Claude presented as a governance control. The Sample 3 rationale rejects exactly this for data retention.
- Believing organization instructions can switch off Claude's built-in safety guidelines (they cannot), or trying to get harmful output by bypassing Claude's guardrails (for example by jailbreaking), which the Usage Policy forbids without Anthropic's prior authorization.
- Installing a skill or connecting a server from an unknown source because it saves time.
- Treating admin restrictions as obstacles to route around instead of controls to request changes to.

**Go deeper:** [Admin and governance controls](knowledge/security-and-governance.md#admin-and-governance-controls)

### D6.4 Understand the ethical implications of AI usage

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Understand the ethical implications of AI usage". In the AI Fluency framework most of this falls under Diligence, which [Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results) defines as "Using AI responsibly and ethically. Includes making thoughtful choices about AI systems and interactions, maintaining transparency, and taking accountability for AI-assisted work."

**Know: the ethical questions and what Anthropic's material says.**

| Question | What Anthropic's material says |
|---|---|
| Who is accountable? | "You must be able to explain and apply everything you submit, even if AI helped" ([AI Fluency for students](https://academy.claude.com/courses/ai-fluency-for-students/ai-as-a-learning-partner)). Ownership "does not transfer because the work is automated" ([Cowork scaling tutorial](https://academy.claude.com/tutorials/scaling-workflows-with-claude-cowork-at-your-organization)), and "validation builds confidence, but it doesn't eliminate responsibility" ([Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results)) |
| Who needs to know AI was involved? | Transparency Diligence means being honest about AI's role with everyone who needs to know. The Usage Policy forbids presenting results as human-generated, and consumer-facing chatbots must tell users they are interacting with AI. How to write a diligence statement is covered under [Domain 4](#domain-4-workflow-integration-and-solution-design) |
| Is it fair to the people affected? | Bias includes stereotyping and political bias, and less direct forms such as defaulting to certain types of answers or perspectives or giving better quality responses in some languages; models can pick up such tilts from patterns in the text they learn from. Anthropic's December 2023 study of 70 decision scenarios found both positive and negative discrimination in Claude 2.0 in select settings when no interventions were applied, and demonstrated "techniques to significantly decrease both positive and negative discrimination through careful prompt engineering" ([Evaluating and mitigating discrimination](https://www.anthropic.com/research/evaluating-and-mitigating-discrimination-in-language-model-decisions)) |
| Is reliance healthy? | "Convergence and skill atrophy are the long-term pitfalls; reaching for AI when you do not know what you want is the immediate one" ([AI Fluency for creative work](https://academy.claude.com/courses/ai-fluency-for-creative-work/delegation-and-diligence)). [Claude's constitution](https://www.anthropic.com/constitution) warns against "fostering problematic forms of complacency and dependence" and counts as acceptable the reliance "a person would endorse on reflection" |
| Should AI do this at all? | In the Academy's AI Fluency for nonprofits course, the [workflow augmentation lesson](https://academy.claude.com/courses/ai-fluency-for-nonprofits/workflow-augmentation) asks whether AI should do a task, not just whether it can; the [integration lesson](https://academy.claude.com/courses/ai-fluency-for-nonprofits/integration) warns that at its worst AI "automates that human touch away", and asks: "When should you choose not to use AI, even if it would be more efficient?" The Academy's [K-12 ethics lesson](https://academy.claude.com/courses/ai-fluency-for-k-12-educators/ethics-responsible-use) also names dependency, de-socialization, student agency, equity of access and environmental impact as tensions to weigh |
| What does it mean for jobs? | Anthropic's [June 2026 Economic Index report](https://www.anthropic.com/research/economic-index-june-2026-report) says users interviewed in December 2025 "reported large productivity gains, but also expressed worry about displacement". In its survey of about 9,700 Claude users (not representative of the general population), 10% rated losing their own jobs in the next 12 months as likely or very likely, and the most common hope was "AI augmentation of work" |

Detecting bias in a particular response is a Domain 2 skill (see [Domain 2](#domain-2-output-evaluation-and-validation)); the ethical question here is what happens to the people an output is about. The Academy's [K-12 ethics lesson](https://academy.claude.com/courses/ai-fluency-for-k-12-educators/ethics-responsible-use) puts it as: "Bias: When might AI reinforce narrow perspectives, stereotypes, or existing inequities?"

**Know: how Claude itself is meant to behave.** [Claude's constitution](https://www.anthropic.com/constitution) says that in cases of apparent conflict Claude should generally prioritize being broadly safe, then broadly ethical, then following Anthropic's guidelines, and otherwise being genuinely helpful, and notes that the vast majority of Claude's interactions are everyday tasks (such as coding, writing and analysis) with no fundamental conflict among these. When Claude declines all or part of a task it need not give its reasons, but it should be transparent that it is not helping, taking the stance of a "transparent conscientious objector" rather than quietly giving a lower-quality answer. Its honesty includes calibration: it "avoids conveying beliefs with more or less confidence than it actually has." By default it keeps professional reticence about sharing its own opinions on hot-button political issues.

**Know: ownership and marking (as of September 2026).** Under the Consumer Terms you are responsible for the inputs you submit, and Anthropic assigns you its rights, if any, in Outputs, subject to your compliance with the terms; under the Commercial Terms the customer owns its Outputs. The help center adds that there are restrictions on using Outputs to train AI models, and the Usage Policy forbids using inputs and outputs to train an AI model without Anthropic's prior authorization. Anthropic has signed the EU AI Act's Article 50(2) Code of Practice on Transparency of AI-Generated Content. Supported Claude models embed an imperceptible watermark in the text they generate, and Claude attaches C2PA Content Credentials to supported file types it creates, such as PNG or JPEG. Text watermarking does not yet cover every model: models launched on or after August 2, 2026 support marking at launch, and Anthropic is adding watermarks to outputs from models released before that date, "with all covered by December 2, 2026" ([How Claude marks AI-generated content](https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content)). A missing mark does not prove that content was not AI-generated.

**Decide**

- If an output will decide something about people in a high-risk area (hiring, housing or credit eligibility, admissions scoring), choose a person as the decision-maker and check results across the groups affected; not an automated decision, because the Usage Policy requires a qualified professional to review such decisions before finalization, Anthropic's 2023 discrimination research says it does not permit automated decisions for the high-risk use cases it studied, and the Academy's AI-policy use case asks for a bias detection and mitigation section.
- If you could not explain or defend an output yourself, choose not to send it; not to rely on Claude having produced it, because accountability stays with you.
- If the value of a task lies in human contact (an emotional situation, a sensitive complaint), choose to keep a person in it; not full automation because it would be faster.
- If AI shaped work that others rely on, choose accurate disclosure of its role; not silence, and not a statement that claims more review than happened.

**Traps**

- Blaming Claude for a mistake in work you sent. Responsibility stays with the person who used the output.
- Treating a missing watermark as proof that content is human-made.
- Treating refusal as the automatically ethical choice. The constitution says unhelpfulness is never trivially safe, and the Sample 3 rationale rejects abandoning a task that a safeguard makes possible.
- Keeping a person nominally in the loop who signs off on outputs they cannot explain.

**Go deeper:** [Responsible use for business users](knowledge/security-and-governance.md#responsible-use-for-business-users)

## Domain 7: Troubleshooting and Optimization

**Weight:** 10% of scored items, the smallest of the seven domains. If all 60 items were scored, that is about 6.0 items (10% of 60, our arithmetic); the guide does not say how many of the 60 are scored.

The domain has three objectives: diagnose and fix underperforming prompts or poor outputs, adjust your approach based on feedback and results, and optimize workflows. The matching module of the free prep course, Troubleshooting & Optimization (30 minutes), says most people react to a disappointing output by giving up or by changing random things until something works, and teaches you instead to "promote a one-time fix into a permanent one" ([course page](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/troubleshooting-optimization)). Its three learning objectives line up with the guide's three, and they give the structure of this section.

None of the three official sample questions covers Domain 7. The closest official guidance on how answers are keyed is the rationales of Samples 1 and 2, which reject options that leave the cause untouched: "reformatting (D) does not address correctness", and "disabling features (C) or switching platforms (D) does not address the trade-off" (see [Official sample questions](#official-sample-questions)). Under each objective below, **Know** lists sourced facts, **Decide** gives our decision rules (derived from those facts and the sample rationales, not published by Anthropic as rules), and **Traps** lists the wrong-answer shapes they rule out.

### D7.1 Identify, diagnose, and resolve issues with underperforming prompts or poor outputs

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Identify, diagnose, and resolve issues with underperforming prompts or poor outputs". The [prep module](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/troubleshooting-optimization)'s objective names the four root causes to trace a failure to: "under-specification, context overload, the wrong feature or model, or stale configuration". Diagnose first, then fix the cause.

**Know: under-specification.** Claude 101 presents "some of the most common challenges and how to address them" as a table ([Getting better results](https://academy.claude.com/courses/claude-101/getting-better-results)), and most of its rows trace back to a prompt that left something out:

| Symptom | What is happening | Fix |
|---|---|---|
| Response too generic | "Your prompt didn't include enough context about your specific situation" | "Add details about your audience, role, or constraints." |
| Too long or too short | "Claude is guessing at appropriate length" | State the length, for example "Give me a two-paragraph summary" |
| Format not followed | "Claude understood what you want but not how you want it presented" | "Show, don't just tell." Give an example of the format or describe the structure |
| Tone not right | "Claude defaults to helpful and professional, which may not match your needs" | Describe the tone in plain language and give an example of the style you want |
| Confident but wrong | "Claude occasionally generates plausible but incorrect information, especially with specific facts or niche topics" | "For high-stakes work, verify key facts independently." Ask for sources you can check and turn on web search. A confidence level Claude gives itself only points to where to look first: the [Sample 1 rationale](#official-sample-questions) says "Self-reported confidence (A, C) is not a reliable accuracy signal" |

The last row is not only a prompt problem; checking factual claims is the core of [Domain 2](#domain-2-output-evaluation-and-validation).

**Know: the other three causes (product behavior as of September 2026).** The prep module names these causes; the mapping of symptoms to causes below is our synthesis, and each symptom and fix comes from the cited Anthropic sources.

| Root cause | Typical symptom | Fix |
|---|---|---|
| Context overload | In a very long chat or input, Claude misses an instruction buried in the middle, or loses guidance given early on. The Academy's [working memory lesson](https://academy.claude.com/courses/ai-capabilities-and-limitations/working-memory) says this limit "has a cliff rather than a gradient": "Silent truncation is the failure mode, and you won't always be warned." Near the context limit, and only when code execution is enabled, Claude summarizes earlier messages so the chat can continue ([automatic context management](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work)), and an [Academy tutorial](https://academy.claude.com/tutorials/parametric-memory-and-context) notes that such a summary "can still occasionally result in lost details" | Put the most important instructions at the beginning and end; start a new chat; use a Project (on paid plans, when project knowledge approaches the context limit, Claude switches the project to retrieval and loads only the relevant content); shorten project instructions, remove unused files, and turn off tools and connectors you do not need |
| Wrong feature | Answers about recent events are outdated or wrong, because knowledge is frozen at the training cutoff | Turn on web search; the Academy's [staleness exercise](https://academy.claude.com/courses/ai-capabilities-and-limitations/knowledge) shows "retrieval in action". Which feature fits which task is covered under [D3.1](#d31-select-appropriate-claude-product-features-projects-research-mode-chat-artifacts) |
| Wrong model or setting | Complex work comes back thin, or, in the [effort tutorial](https://academy.claude.com/tutorials/how-to-select-the-right-effort-setting-for-claude-cowork-and-chat)'s words, "Instructions get missed, or long work wraps up before it's finished." At the other extreme, answers get verbose or wander past the request | Raise effort, turn on thinking, or both; move up a model where a smaller one struggled. Lower effort if output is verbose without being better |
| Stale configuration | Every chat in a Project gives outdated or inconsistent answers, or a skill never triggers | [Claude 101](https://academy.claude.com/courses/claude-101/introduction-to-projects): "Outdated documents can lead to outdated responses", so update project knowledge. Remove contradictions: the [help center](https://support.claude.com/en/articles/14546867-set-organization-instructions) warns that if organization instructions conflict, "Claude may not follow either one reliably." Test changed instructions in a new conversation. For a skill, the [help center](https://support.claude.com/en/articles/12512180-use-skills-in-claude) says: "Check that the skill's description field clearly explains when it should be used." |

**Know: two diagnostic habits.**

- **Name the properties involved.** The [AI Capabilities and Limitations course](https://academy.claude.com/courses/ai-capabilities-and-limitations/when-properties-collide): "Real-world failures are usually two properties interacting, not one." Next Token Prediction combined with Knowledge produces hallucinated specifics; Working Memory combined with Steerability produces long-conversation drift. Naming them "points you straight to the fix: verify specifics, re-supply context, offload to code execution, or invite pushback."
- **Read the prompt as a stranger would.** The [help center's advice for unhelpful answers](https://support.claude.com/en/articles/7996857-my-prompt-isn-t-giving-me-a-helpful-answer): "Pretend you are giving these instructions to someone with no background knowledge about what you are asking." Break complex requests into substeps. The prompt-writing side of this is covered under [Domain 1](#domain-1-prompting-and-task-execution).

**Know: messages that are not prompt problems (as of September 2026).**

- **Length limit.** The [error message](https://support.claude.com/en/articles/12466728-troubleshoot-claude-error-messages) reads: "Your message will exceed the length limit for this chat. Try attaching fewer or smaller files or starting a new conversation." Other fixes: summarize or extract key sections before sending, use Claude first to find the relevant portions, or use a Project to work with larger amounts of information. A usage limit is different: it caps how much you use Claude across all your conversations. On Free and Pro the session-based usage limit resets every five hours; Pro, Max, Team and seat-based Enterprise plans show both the five-hour session limit and weekly limits in Settings > Usage.
- **"Output blocked by content filtering policy."** The [help center](https://support.claude.com/en/articles/10023638-why-am-i-receiving-an-output-blocked-by-content-filtering-policy-error) says these refusals are not a judgment about the propriety of your content; they generally come from Anthropic's efforts to stop Claude being used to replicate or regurgitate pre-existing material, such as copyrighted content.
- **Model switched mid-conversation.** On [Opus 5 and Opus 5.5](https://support.claude.com/en/articles/16049681-why-claude-switched-models-in-your-conversation-with-opus-5-or-opus-5-5), a narrow set of higher-risk requests falls back to a less capable model or is blocked. On [Fable 5 and Fable 5.1](https://support.claude.com/en/articles/15363606-why-claude-switched-models-in-your-conversation-with-fable-5-or-fable-5-1), requests in areas such as offensive cybersecurity and dual-use biology fall back to an Opus model. When a request falls back, a notice appears and the response is labeled with the model that answered; a request blocked outright (on Opus 5 and 5.5, attempts to extract the model's reasoning, for example) gets no fallback answer.
- **Thinking stopped early.** If safety systems cut the thought process short, the [help center](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings) suggests reframing the prompt to approach the problem from a different angle.
- **A claim to have sent an email, or links that do not work.** The [help center](https://support.claude.com/en/articles/8241188-claude-is-producing-links-that-don-t-work-and-falsely-claiming-that-it-has-sent-emails-or-produced-external-documents-what-s-going-on) says Claude "can sometimes hallucinate its capabilities": it has no access to tools that are not explicitly integrated, including email, word processors or file transfers.
- **A model or effort level is missing.** On Enterprise, your administrator may have turned it off for your role (see [Domain 6](#domain-6-governance-risk-and-responsible-use)).

!!! warning "Exam guide vs current docs"

    The [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)'s model objective names three model types, "Haiku, Sonnet, Opus", and mentions no effort or thinking settings. Today's Fable tier is described in the box under [D3.2](#d32-differentiate-between-claude-model-types-haiku-sonnet-opus), and effort, thinking and switching models mid-conversation under [D3.3](#d33-align-model-selection-with-task-requirements-cost-speed-quality). For troubleshooting, an [Academy tutorial](https://academy.claude.com/tutorials/choosing-the-right-claude-model) suggests trying Opus for problems where Sonnet struggled and Fable for problems where Opus struggled. Answer in the guide's terms: a wrong-model failure is fixed by moving between Haiku, Sonnet and Opus to match the task, keeping "the most capable model for complex reasoning" as the [Sample 2 rationale](#official-sample-questions) puts it. Treat Fable and the effort setting as current product detail.

**Decide**

- If output is generic, choose to add the missing context (audience, role, constraints); not a more capable model first, because [Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results) traces generic output to a prompt that "didn't include enough context about your specific situation". If the format is wrong, show an example of it.
- If a long conversation has gone off track and targeted corrections are not bringing it back, choose a new chat with a clearer prompt that re-supplies the key context; not endless rounds of correction, because [Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results) says "sometimes it's faster to open a new chat with a clearer prompt than to try to redirect."
- If the answer depends on information after the training cutoff, choose web search, or a connected or uploaded source that holds the information; not a reworded question, because, in the Academy's [Knowledge lesson](https://academy.claude.com/courses/ai-capabilities-and-limitations/knowledge), the model's knowledge "is frozen at the knowledge cutoff" and web search, retrieval and tool use exist "to patch these gaps".
- If complex work comes back thin on a lighter model, choose a more capable model or more reasoning; not a cheaper model or a lower setting, because the [Sample 2 rationale](#official-sample-questions) reserves the most capable model "for complex reasoning" and the [help center](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings) says that for complex tasks you should "raise the effort level, turn on thinking, or both." The [developer docs](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model) add that "Tuning effort is often a better lever than switching models"; the guide itself frames the choice in model terms.
- If the same failure appears in every chat of a Project, choose to fix the Project's instructions or knowledge (see [Domain 5](#domain-5-configuration-and-knowledge-management)); not a correction in each chat, because project instructions apply to all the chats within the project and, per [Claude 101](https://academy.claude.com/courses/claude-101/introduction-to-projects), outdated project documents "can lead to outdated responses".

**Traps**

- Concluding "Claude can't do this", or changing random things until something works: the two reactions the [prep module](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/troubleshooting-optimization) sets out to replace.
- Cosmetic fixes. Rewording or reformatting an answer does not make it correct, which is the point of the Sample 1 rationale on option D.
- Drastic moves that skip diagnosis, such as switching platforms or turning features off. The [Sample 2 rationale](#official-sample-questions) says they do "not address the trade-off".
- Writing instructions into a project's name or description. Claude cannot see either.
- Expecting a correction made in one chat to fix future chats without saving it to instructions, a Project, a skill or memory. The Academy's [working memory lesson](https://academy.claude.com/courses/ai-capabilities-and-limitations/working-memory): "The model doesn't learn from your corrections."

**Go deeper:** [Principles that decide most prompt questions](knowledge/prompt-engineering.md#principles-that-decide-most-prompt-questions)

### D7.2 Adjust approach based on feedback and results

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Adjust approach based on feedback and results". The [prep module](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/troubleshooting-optimization)'s version is to adjust "based on the output you get, turning recurring corrections into fixes that persist instead of repeating the same manual work". The moves for improving a single response (targeted feedback, restating the goal, branching, starting fresh) are covered under [Domain 1](#domain-1-prompting-and-task-execution); this objective is about what you change once the feedback repeats or the results come in.

**Know: make recurring corrections persist.** The model does not learn from your corrections (see [D1.3](#d13-iterate-prompts-to-improve-output-quality)), so a correction typed into one chat does not carry into the next unless it is saved somewhere Claude reads, such as instructions or memory. The [Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/giving-cowork-context) calls the corrections you keep giving "global-instruction candidates", and Anthropic's [human-agent teams course](https://academy.claude.com/courses/building-effective-human-agent-teams/practical-ways-to-get-started) says: "When you explain the same thing twice, turn it into written instructions for the agent." Where each kind of correction belongs:

| Recurring correction | Where it persists | Scope |
|---|---|---|
| A preference for all your work, such as "share the bottom line up front in your responses" (the [Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/giving-cowork-context)'s example) | Instructions for Claude (Settings); in Cowork on accounts without the new Claude experience, Cowork's Global instructions | Every conversation |
| A rule for one body of work | Project instructions | Every chat in that project |
| Reference material Claude keeps getting wrong | Project knowledge | Background for every chat in that project |
| A procedure or output format you reuse | A skill; the Academy's personal voice skill improves as you feed back each correction you make to a draft | Loads when relevant, across Claude |
| A fact about you or your work | Memory: tell Claude to "remember this" ([help center](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context)) | Your next conversations |

On Team and Enterprise, memory is off until an owner turns it on, so a correction that depends on memory may not persist there.

**Know: check that the adjustment helped.**

- **Compare against a baseline.** When you test a skill, the Cowork course's skill-creator shows each test prompt answered with and without the skill, so you judge whether the output is better than what Claude would do on its own, not just whether it is acceptable.
- **Change one variable per round,** so you can tell which change made the difference (the iteration mechanics are under [Domain 1](#domain-1-prompting-and-task-execution)).
- **Gate shared changes.** For configurations others use, the [Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/share-what-you-build-with-your-team) says: "if the cases you care about don't hold up after a change, don't push it to everyone."
- **Move the human boundary on evidence.** Widen Claude's share where results prove reliable and step a person back in where errors appear; the rollout practices are under [Domain 4](#domain-4-workflow-integration-and-solution-design).
- **Test with known answers, and retest old conclusions.** Check Claude against a task whose results you already know (Claude 101's delegation diligence loop, under [D2.1](#d21-evaluate-claude-generated-outputs-for-accuracy-and-completeness)). Anthropic's [Academy tutorial on AI-fluent organizations](https://academy.claude.com/tutorials/building-ai-fluent-organizations) says people who treat today's AI as the worst they will ever use "keep coming back to retest old assumptions". Our inference: a task that failed a known-answer test months ago may be worth testing again.

**Know: feedback that leaves the conversation.** The thumbs down button reports an unhelpful response to Anthropic. A thumbs up or down stores the entire related conversation for up to 5 years, on commercial plans too, so mind what the chat contains; Team and Enterprise owners can turn feedback off with the Rate chats setting. Feedback to Anthropic is not a fix for your own workflow: the model does not learn from your corrections, so a fix you need tomorrow belongs in your own configuration.

**Decide**

- If you have given the same correction twice, choose to promote it into Instructions for Claude, project instructions or a skill; not to retype it each session, because the model does not learn from corrections.
- If a reviewer's feedback is vague ("this isn't quite right"), choose to turn it into a specific change before re-prompting; not to pass the vague version to Claude, because, as the [Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/validating-skills-for-plugins) puts it, specific feedback gives Claude something to act on.
- If an adjustment is meant to improve a shared skill or template, choose to re-run the test cases you care about and compare the output with a baseline; not to publish on the strength of one good output, because the Cowork course makes the eval loop the gate before every publish.
- If results on a task stay unreliable after the prompt, feature and model have been checked, choose to keep a person doing that task or reviewing it closely; not to keep tuning indefinitely, because a failed known-answer test tells you the task is one "you shouldn't delegate" ([Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results)), and where errors appear "a person takes the step back" ([Academy tutorial](https://academy.claude.com/tutorials/scaling-workflows-with-claude-cowork-at-your-organization)).

**Traps**

- Regenerating the same request and hoping. The [Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/the-task-loop) tells you to resist the chat habit of waiting and regenerating, because "Cowork is built for course corrections, and the cost of a redirect is low."
- Declaring an adjustment an improvement without comparing it to what came before.
- Sending feedback to Anthropic and expecting your own results to change.
- Widening an automation after a single good run.

**Go deeper:** [Prompt versioning and iteration](knowledge/prompt-engineering.md#prompt-versioning-and-iteration)

### D7.3 Optimize workflows for efficiency and effectiveness

Official wording ([exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)): "Optimize workflows for efficiency and effectiveness". As working definitions (ours, not the guide's), efficiency is doing the work with less effort, usage and waiting; effectiveness is ending up with results you can use. The [prep module](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/troubleshooting-optimization)'s route is "promoting shared context, format, and verification steps into Projects, Skills, and standing instructions". Process redesign and scheduling are covered under [Domain 4](#domain-4-workflow-integration-and-solution-design), and model choice under [Domain 3](#domain-3-product-and-model-selection).

**Know: where each reusable piece belongs.**

| Reusable piece | Home | Anthropic's reasoning |
|---|---|---|
| Background documents that many chats need | Project knowledge | Projects "provide static background knowledge that's always loaded" ([help center](https://support.claude.com/en/articles/12512176-what-are-skills)); project content is cached and counts less against your limits when reused |
| Rules and a role for one body of work | Project instructions | They apply to every chat in the project and can automate a workflow, for example a structured summary whenever a meeting transcript is uploaded |
| A procedure or format used across contexts | A skill | Skills "provide specialized procedures that activate dynamically when needed and work everywhere across Claude" ([help center](https://support.claude.com/en/articles/12512176-what-are-skills)); in [Claude 101](https://academy.claude.com/courses/claude-101/working-with-skills)'s shorthand, "projects store knowledge, skills perform tasks" |
| Preferences for everything you do | Instructions for Claude | Applied to all of your conversations |
| A prompt you rerun with new details | A template | The [AI Fluency course](https://academy.claude.com/courses/ai-fluency-framework-foundations/additional-activities): "Include placeholders or notes for variable information you'll need to add each time" |
| A verification aid | Built into the request | Ask Claude to "flag anything you're not confident about" ([Academy use case](https://academy.claude.com/use-cases/prd-from-a-one-pager)) so you know where to look first, then check those points yourself; or ask for an audit layer such as a [theme-classification tab](https://academy.claude.com/use-cases/analyze-patterns-in-user-feedback) you can check before presenting findings |

Keep project instructions to general context, key guidelines and Claude's role. The [help center](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work): "Reserve task-specific instructions for the chat itself." Setting up and maintaining these configurations is covered under [Domain 5](#domain-5-configuration-and-knowledge-management).

**Know: usage levers (as of September 2026).** Usage limits cap how much you can use Claude across all conversations, and claude.ai, Claude Code and Claude Desktop draw on the same limit.

- The [usage-limit best practices](https://support.claude.com/en/articles/9797557-usage-limit-best-practices): "If you have multiple related tasks or questions, group them in a single message."
- Tools and connectors are token-intensive; turn off the ones a conversation does not need. The [usage-limit best practices](https://support.claude.com/en/articles/9797557-usage-limit-best-practices) list tool usage (such as Research and web search) and multi-step tasks (running code, creating files, browsing websites) among the factors that affect usage limits, along with message length, file attachment size, conversation length, model choice and effort level. Research sessions can use up your limits faster than normal chats, and creating files uses more of your limit than a normal chat.
- Choose lower effort for routine tasks and, on models that allow it, turn thinking off when a task does not need it; save higher settings for complex work. Thinking cannot be turned off in Claude on Opus 5.5, Fable 5.1 or Opus 5 ([help center](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings)), so on those models effort is the lever.
- Project content is cached and counts less against your limits when reused, but caches expire after a period of inactivity, so the first message after a long break counts project content in full again.
- In Cowork, Automatically approve (Auto) uses more of your usage limit than the other modes because Claude does extra safety checking of each action.

**Know: judge the whole task, not the prompt.** Anthropic's [effort tutorial](https://academy.claude.com/tutorials/how-to-select-the-right-effort-setting-for-claude-cowork-and-chat) says "the unit that matters more is the cost per task completed, which can diverge significantly from cost per token." Anthropic's [productivity research](https://www.anthropic.com/research/estimating-productivity-gains) notes that its time-saving estimates cannot count time spent outside the conversation, including validating Claude's work. Our inference from both: a change that makes drafting faster but checking slower may not be an optimization at all.

Where these standing instructions live in today's apps, and why the "Use style" menu is no longer the place for tone, is in the box under [D5.3](#d53-create-effective-system-level-instructions).

**Decide**

- If you paste the same context into many chats, choose a Project; if the same procedure or format recurs across projects, choose a skill; if it is a preference for everything, choose Instructions for Claude; not repeated pasting, because each home loads the material for you.
- If you are running short of usage, choose to cut waste first (group related messages, turn off unneeded tools, lower effort on routine work); not a downgrade to an older model for the work that matters, because the help center's usage tips include grouping related messages, turning off unneeded tools and lowering effort for routine tasks, and the Academy's [effort tutorial](https://academy.claude.com/tutorials/how-to-select-the-right-effort-setting-for-claude-cowork-and-chat) says "A frontier model at medium or low effort often outperforms an older model at high or maximum effort." Model choice is itself a usage factor, so moving straightforward work to a lighter model is still the right call (next rule).
- If the work is high-volume and straightforward, choose a faster, lower-cost model and keep the most capable model for complex reasoning, as the [Sample 2 rationale](#official-sample-questions) does; not the top model for everything, which "wastes the cost and latency budget".
- If a change speeds up production but adds checking, choose to measure the whole task including verification; not the drafting time alone, because time spent validating Claude's work happens outside the conversation and is easy to leave out of the count.

**Traps**

- The top model (or the highest effort) for everything. The [Sample 2 rationale](#official-sample-questions): "Always using the top model (A) wastes the cost and latency budget".
- Packing task-specific detail into project instructions, or leaving every connector switched on in case it is needed.
- Treating a faster draft as a better workflow when the time saved is spent checking the output.

**Go deeper:** [Skills in the Claude apps](knowledge/claude-for-work.md#skills-in-the-claude-apps)

## Official sample questions

Section 8 of the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) introduces its three samples as items that "show the style and cognitive level of the exam", and adds: "They are not drawn from the live item bank."

There is no official practice exam to supplement them. The [certification FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says: "The practice exam available on the previous platform was retired in the move to Pearson." It points candidates to these samples as the reference for format and style.

!!! info "Before you start"

    - Each of the three samples has four options and one correct answer. The live exam also uses multiple-response items, and the guide says "each item states how many responses to select".
    - The three items cover Domains 2, 3 and 6. The other four domains have no published sample; see [Blueprint](#blueprint) for what that means for your preparation.
    - Stems, options and rationales are reproduced word for word from the guide. Only the item headings are reworded.
    - Commit to an answer before you open each answer block.

| Sample | Domain in the guide (weight) | Objectives it exercises (our mapping) |
|---|---|---|
| [Sample 1](#sample-1-a-confident-summary-bound-for-compliance) | Domain 2: Output Evaluation and Validation (21%) | D2.2 hallucinations, D2.3 fact-checking, D2.4 when to verify |
| [Sample 2](#sample-2-high-volume-customer-reply-drafts) | Domain 3: Product and Model Selection (12%) | D3.2 model types, D3.3 cost, speed and quality |
| [Sample 3](#sample-3-regulated-personal-data-in-a-spreadsheet) | Domain 6: Governance, Risk, and Responsible Use (15%) | D6.2 data sensitivity and privacy, D6.3 organizational policy |

### Sample 1: a confident summary bound for compliance

The guide files this item under Domain 2, Output Evaluation and Validation.

An associate asks Claude to summarize a new regulation, and Claude produces a confident summary citing a specific subsection number. Before sending the summary to the compliance team, what is the most appropriate action?

- **A.** Send it as-is, since Claude expressed high confidence.
- **B.** Verify the cited subsection against the official regulation text before sharing.
- **C.** Ask Claude to rate its own confidence and send it if the rating is high.
- **D.** Reword the summary to sound more formal, then send it.

??? success "Answer and Anthropic's rationale"

    **Correct answer: B.** Verify the cited subsection against the official regulation text before sharing.

    Anthropic's rationale, verbatim from the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf):

    > Sample 1: B. Language models can fabricate specific-looking details such as citation numbers, a hallucination. Validating factual claims, especially citations bound for a compliance audience, against an authoritative source is the diligence step required. Self-reported confidence (A, C) is not a reliable accuracy signal, and reformatting (D) does not address correctness.

**What this teaches:** a precise-looking detail such as a subsection number is exactly what to check against the primary source before it reaches a high-stakes audience, because neither Claude's confident tone nor its own rating of itself counts as verification ([Domain 2](#domain-2-output-evaluation-and-validation), D2.2 to D2.4).

### Sample 2: high-volume customer-reply drafts

The guide files this item under Domain 3, Product and Model Selection.

An associate needs to generate a high volume of short customer-reply drafts where speed and cost matter more than deep reasoning. Which choice best fits the task?

- **A.** Use the most capable, highest-cost model for every reply to maximize quality.
- **B.** Use a faster, lower-cost model suited to straightforward, high-volume tasks.
- **C.** Disable all product features to reduce cost.
- **D.** Switch to a different AI platform.

??? success "Answer and Anthropic's rationale"

    **Correct answer: B.** Use a faster, lower-cost model suited to straightforward, high-volume tasks.

    Anthropic's rationale, verbatim from the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf):

    > Sample 2: B. Aligning model selection with task requirements means matching a faster, lower-cost model to straightforward, high-volume work, reserving the most capable model for complex reasoning. Always using the top model (A) wastes the cost and latency budget; disabling features (C) or switching platforms (D) does not address the trade-off.

**What this teaches:** read the stem for the stated priority ("speed and cost matter more than deep reasoning") and pick the model tier that serves it, keeping the most capable model for work that needs complex reasoning ([Domain 3](#domain-3-product-and-model-selection), D3.2 and D3.3).

### Sample 3: regulated personal data in a spreadsheet

The guide files this item under Domain 6, Governance, Risk, and Responsible Use.

A project manager wants to upload a spreadsheet containing customer names and account numbers so Claude can analyze trends. Organizational policy restricts sharing regulated personal data. What is the most appropriate action?

- **A.** Upload the file as-is, since the analysis is internal.
- **B.** Remove or anonymize the personal identifiers before uploading, consistent with policy.
- **C.** Upload the file but instruct Claude not to retain it.
- **D.** Skip the analysis entirely.

??? success "Answer and Anthropic's rationale"

    **Correct answer: B.** Remove or anonymize the personal identifiers before uploading, consistent with policy.

    Anthropic's rationale, verbatim from the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf):

    > Sample 3: B. Applying data-sensitivity and privacy safeguards means redacting or anonymizing regulated identifiers before use, so the analysis can proceed without exposing protected data. Uploading as-is (A) violates policy; instructing the model not to retain data (C) does not satisfy the policy control; abandoning the task (D) is unnecessary when anonymization enables it.

**What this teaches:** when policy restricts a kind of data, take it out before it reaches Claude so the work can still go ahead; asking Claude not to keep it is not a policy control, and dropping the task is not the safe default ([Domain 6](#domain-6-governance-risk-and-responsible-use), D6.2 and D6.3).

### Patterns in Anthropic's rationales

In our reading of the three rationales, the right answer is the proportionate step that deals with the real risk and still lets the work go ahead. The wrong answers fall into a few recognizable shapes (our grouping), and each rationale names why it fails.

| Wrong-answer shape | Where it appears | Why the rationale rejects it |
|---|---|---|
| Trust Claude's confidence, or its rating of itself | Sample 1, A and C | "Self-reported confidence (A, C) is not a reliable accuracy signal" |
| Change the form, not the substance | Sample 1, D | "reformatting (D) does not address correctness" |
| Maximize one dimension whatever it costs | Sample 2, A | the top model for everything "wastes the cost and latency budget" |
| A drastic move that sidesteps the actual trade-off | Sample 2, C and D | disabling features or switching platforms "does not address the trade-off" |
| An instruction to the model in place of a control | Sample 3, C | "does not satisfy the policy control" |
| Ignore the policy because the use is internal | Sample 3, A | "Uploading as-is (A) violates policy" |
| Abandon the task when a safe route exists | Sample 3, D | "abandoning the task (D) is unnecessary when anonymization enables it" |

Our suggestion: use the table as a first filter on any option. If it matches one of these shapes, look for the option that fixes the underlying problem instead. The [Quick reference](quick-reference.md#decision-rules) collects decision rules like these across all four exams.

### What the samples do not show

- **Multiple-response items.** All three samples are single-answer, but the guide's item format includes multiple-response items, and each one states how many responses to select. The [certification FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) describes the exams as using "multiple choice and scenario-based multiple response questions". Neither the guide nor the FAQ says whether a multiple-response item gives partial credit. Read the selection instruction on every item.
- **Four of the seven domains.** There is no sample for [Domain 1](#domain-1-prompting-and-task-execution), [Domain 4](#domain-4-workflow-integration-and-solution-design), [Domain 5](#domain-5-configuration-and-knowledge-management) or [Domain 7](#domain-7-troubleshooting-and-optimization). Work those from the objectives and the traps in each domain section.
- **Answer positions.** All three keys are B. Three items are too few to say anything about answer positions on the live exam.
- **Difficulty.** One candidate who passed found the samples easier than the live items; see [What candidates report](#what-candidates-report).

## What candidates report

Each row below is a candidate who published their own CCAO-F result, linked to the post. Scores are self-reported and copied exactly as each candidate wrote them. The exam is criterion-referenced: the [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) says each candidate "is measured against a fixed performance standard, not against other candidates". Read these as individual experiences, not as odds.

All of these reports come from the Pearson era. CCAO-F did not exist before the program moved to Pearson and Credly on June 30, 2026, so none of the older logistics (ProctorFree, 6-month validity, a practice exam) applies.

!!! warning "What the confidentiality rules mean for these reports"

    The [policies page](https://anthropic-partners.skilljar.com/page/policies-certifications) says: "By taking an exam you agree not to share, reproduce, or discuss the questions in any form, including in study groups and online forums." Several of the posts below also describe which topics came up. This summary keeps to results, preparation, pacing, item style and logistics. One candidate, alberto3333, wrote simply: "Not describing any questions; the agreement forbids it."

### The reports

| Candidate and source | Date | Result as published | What they add |
|---|---|---|---|
| [OkRelationship3427](https://www.reddit.com/r/ClaudeAI/comments/1uv1wi2/passed_the_ccaof_today/) (Reddit) | Exam on July 13, 2026 | Passed, "really close to the passing line" (no score given) | "Honestly the exam was harder than I expected." Sat it before the Skilljar prep course was released. The same person later posted passes on CCAR-F (904/1000), CCDV-F (941/1000) and CCAR-P (840/1000). |
| [Matthew Purcell](https://www.linkedin.com/pulse/claude-certification-exams-honest-review-matthew-purcell-byo2e) (LinkedIn) | Exam on July 16, 2026 (article published July 21, 2026) | 967 | About 45 minutes of preparation with their own practice exam. Booked CCAO-F for 15 minutes after finishing CCDV-F the same day, and finished in about an hour. Promotes the practice exam they wrote. |
| [Interesting_Ebb_6383](https://www.reddit.com/r/ClaudeAI/comments/1v2x1p7/got_all_4_claude_certifications_ccap_ccaf_ccdvf/) (Reddit) | Posted July 21, 2026 | Passed all four exams (no scores given) | "Honestly, I did not prepare at all for any of these exams." Launched a practice site after passing. |
| [OkFan7308](https://www.reddit.com/r/ClaudeAI/comments/1vaxsws/claude_certified_architect_foundations_ccarf_how/) (Reddit comment in a CCAR-F thread) | Posted August 5, 2026 (exam on August 3) | 851 | Their comments say they chose CCAO-F. Relied only on the official Associate prep course, and "had to sit through the entire 2 hours." |
| [Hefty_Ad_6638](https://www.reddit.com/r/claudeskills/comments/1vuxasi/failed_ccaof_am_i_the_only_one/) (Reddit) | Posted August 22, 2026 | Failed (no score given) | Took the exam because their company asked. "Mock tests are different from what you see in the exam. Its super hard and 2hrs is not enough." |
| [Hot_Entrepreneur671](https://www.reddit.com/r/claudeskills/comments/1vssz2i/passed_my_claude_certified_associate_foundations/) (Reddit comment) | August 23, 2026 | Passed (no score given) | The real questions "were nowhere close to the mocks i had seen during my prep". |
| [royalkaku](https://www.reddit.com/r/claudeskills/comments/1vssz2i/passed_my_claude_certified_associate_foundations/) (Reddit comment in another candidate's thread) | Posted September 3, 2026 | "Passed 934/1000. Got two questions incorrect." | Called it "fairly straightforward". |
| [bluepanda](https://dev.to/bluepanda/i-cleared-all-4-anthropic-claude-certifications-what-you-actually-need-to-know-and-what-to-skip-224m) (dev.to) | Published September 6, 2026 | Passed all four exams (scorecard shown only as an image) | Took CCAR-F first, then the other three online through Pearson VUE within seven days. Ranked the Associate exam the easiest of the four and CCAR-F the hardest. |
| [alberto3333](https://www.reddit.com/r/claudeskills/comments/1wa1rf5/passed_ccdvf_and_ccaof_i_spent_two_weeks_studying/) (Reddit) | Posted September 7, 2026 | 835/1000 on Associate (and 926/1000 on CCDV-F) | Two weeks of study covering both exams. Runs a paid practice site and says so in the post: "treat this as the disclosure it is". |
| [Broad-Service-7733](https://www.reddit.com/r/claudeskills/comments/1we020n/claude_associate_foundation_certification_ccaof/) (Reddit) | Posted September 12, 2026 | Passed (no score given) | On September 2, with the exam scheduled for the following week, [they described](https://www.reddit.com/r/claudeskills/comments/1w558ch/ccaof/) their level as "Basic understanding of claude". |
| [imstillwhite](https://www.reddit.com/r/ClaudeAI/comments/1wjnvrw/i_passed_the_ccaof_claude_associate_foundation/) (Reddit) | Posted September 18, 2026 | 917/1000 | Took it online with OnVUE. Their preparation ranking and logistics advice are below. |

OkRelationship3427 later wrote that they sat the exam before the Skilljar prep course was released, so they could not use it.

### Difficulty: the reports disagree

- **Harder than expected.** [OkRelationship3427](https://www.reddit.com/r/ClaudeAI/comments/1uv1wi2/passed_the_ccaof_today/) found it harder than they expected and passed close to the line. [Purcell](https://www.linkedin.com/pulse/claude-certification-exams-honest-review-matthew-purcell-byo2e) wrote that "after I saw the first question I realised that it was actually going to be slightly trickier than I expected", while adding "It is not hard" and "you can't sleepwalk through it either." [Hefty_Ad_6638](https://www.reddit.com/r/claudeskills/comments/1vuxasi/failed_ccaof_am_i_the_only_one/) failed and called it "super hard".
- **Not an entry-level exam.** [Interesting_Ebb_6383](https://www.reddit.com/r/ClaudeAI/comments/1v2x1p7/got_all_4_claude_certifications_ccap_ccaf_ccdvf/): "My experience says CCAO-F is not entry level compared to other 3."
- **Straightforward.** [royalkaku](https://www.reddit.com/r/claudeskills/comments/1vssz2i/passed_my_claude_certified_associate_foundations/) called it "fairly straightforward", and [bluepanda](https://dev.to/bluepanda/i-cleared-all-4-anthropic-claude-certifications-what-you-actually-need-to-know-and-what-to-skip-224m) ranked it the easiest of the four.

The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) places the intended candidate "between casual AI prompt users and technical AI practitioners". Several of the candidates above also passed one or more of the technical exams, so (our inference) their sense of difficulty may not match that of the candidate the guide describes.

### Item style, in their words

- **Scenarios, not definitions.** [royalkaku](https://www.reddit.com/r/claudeskills/comments/1vssz2i/passed_my_claude_certified_associate_foundations/): "it's 100% scenario based. They won't ask definitions or basically anything from memory. 80% of the question will have one obvious correct answer while 20% has subtle close wrongs."
- **Judgment over technical fact.** [Purcell](https://www.linkedin.com/pulse/claude-certification-exams-honest-review-matthew-purcell-byo2e): "I thought the answers were less clear-cut and required more judgement compared with the technical questions in the other exams." They added that "Many questions adopt a particular non-technical persona".
- **Two plausible options.** [OkRelationship3427](https://www.reddit.com/r/ClaudeAI/comments/1uv1wi2/passed_the_ccaof_today/): "The official sample questions are too easy, you can pick the right answer just from the choices alone. The real exam narrows it down to two, and that's where it gets hard." Practice on the [Official sample questions](#official-sample-questions) by explaining why each wrong option fails, not just by finding the right one.
- **Long, business-flavored reading.** [imstillwhite](https://www.reddit.com/r/ClaudeAI/comments/1wjnvrw/i_passed_the_ccaof_claude_associate_foundation/): "Also, the exam is heavy on business English", and "Not strictly question dense but questions are long and often complex. You don't have much time."
- **An elimination habit.** [OkFan7308](https://www.reddit.com/r/ClaudeAI/comments/1vaxsws/claude_certified_architect_foundations_ccarf_how/): "One thing that helped me was trying to find the odd one out among the four options."
- **Interface.** [bluepanda](https://dev.to/bluepanda/i-cleared-all-4-anthropic-claude-certifications-what-you-actually-need-to-know-and-what-to-skip-224m) reported that you can strike out options and flag questions for review, with "no negative marking for incorrect answers". The CCAO-F guide does not address guessing penalties either way; the March 2026 CCAR-F guide, an older document for a different exam, said "there is no penalty for guessing".

### Time

The guide gives 120 minutes for 60 items, which is 2 minutes per item (our arithmetic). The [FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) adds: "Plan for about 135 minutes of total seat time, which includes check-in, instructions, and a brief post-exam survey." Experiences at that pace varied:

- [Purcell](https://www.linkedin.com/pulse/claude-certification-exams-honest-review-matthew-purcell-byo2e) finished "in about an hour".
- [OkFan7308](https://www.reddit.com/r/ClaudeAI/comments/1vaxsws/claude_certified_architect_foundations_ccarf_how/) used the full two hours, and [Hefty_Ad_6638](https://www.reddit.com/r/claudeskills/comments/1vuxasi/failed_ccaof_am_i_the_only_one/) wrote that "2hrs is not enough".
- [imstillwhite](https://www.reddit.com/r/ClaudeAI/comments/1wjnvrw/i_passed_the_ccaof_claude_associate_foundation/): "You don't have much time."
- [bluepanda](https://dev.to/bluepanda/i-cleared-all-4-anthropic-claude-certifications-what-you-actually-need-to-know-and-what-to-skip-224m) judged 120 minutes "enough time for most exams except probably CCAR-F".

### Preparation: what they credit

- **The official prep course.** [OkFan7308](https://www.reddit.com/r/ClaudeAI/comments/1vaxsws/claude_certified_architect_foundations_ccarf_how/) relied on it alone: "From Anthropic academy - official Prep course for the Associate exam, this is the best and it is perfectly in sync with questions asked in the exam." [imstillwhite](https://www.reddit.com/r/ClaudeAI/comments/1wjnvrw/i_passed_the_ccaof_claude_associate_foundation/) ranked it first: "the preparation efficiency scale is: Anthropic official prep course > everyday use > mock tests > other courses." They also found that "There were a few questions not really covered by the prep course".
- **Third-party mocks.** Three reports found the real exam unlike the mocks they used. [imstillwhite](https://www.reddit.com/r/ClaudeAI/comments/1wjnvrw/i_passed_the_ccaof_claude_associate_foundation/): "Mock tests are useful for testing your knowledge, but I found the official exam harder and more closely aligned with the prep course." Hot_Entrepreneur671 and Hefty_Ad_6638 said the same in the quotes above.
- **Time spent.** The reports range from no preparation at all (Interesting_Ebb_6383) and about 45 minutes (Purcell) to two weeks shared between two exams (alberto3333).

The course they credit is the free [Claude Certified Associate - Foundations Prep Course](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations) on the Anthropic Partner Academy. The [Study plan](#study-plan) builds on it.

### What the score reports showed

[Purcell](https://www.linkedin.com/pulse/claude-certification-exams-honest-review-matthew-purcell-byo2e) wrote that spotting likely hallucinations was something that, "according to the score report, I wasn't too good at". The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) says the score report shows "the percentage of items you answered correctly within each content domain". Identifying hallucinations, inconsistencies and biases is objective D2.2, in [Domain 2](#domain-2-output-evaluation-and-validation), the most heavily weighted domain at 21%.

One report ties a score to wrong answers: [royalkaku](https://www.reddit.com/r/claudeskills/comments/1vssz2i/passed_my_claude_certified_associate_foundations/) says two questions wrong became 934. The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) publishes no raw-to-scaled conversion, and the [FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says scaled scoring "equates scores across exam forms that may have slightly different difficulty", so one self-reported data point is not a formula. The per-domain percentages on your own report are feedback only: pass or fail rests on the total scaled score.

### Online or test center

[imstillwhite](https://www.reddit.com/r/ClaudeAI/comments/1wjnvrw/i_passed_the_ccaof_claude_associate_foundation/) took the exam on OnVUE and still recommends a test center: "I took the OnVUE online proctored exam, but I suggest you go to a Pearson-certified center to take it in person because it's much less strict and stressful." Services on their company laptop kept getting flagged in the system tests, so they switched to a personal laptop, and "For me, the best fix was launching OnVUE with admin privileges" when OnVUE lagged. [bluepanda](https://dev.to/bluepanda/i-cleared-all-4-anthropic-claude-certifications-what-you-actually-need-to-know-and-what-to-skip-224m) wrote that "The check-in takes about 10 to 15 minutes."

The official setup page matches the laptop experience. It lists applications that can stop OnVUE from launching, including the Claude desktop application on Windows, and its fallback options are a personal computer on a personal network or a Pearson test center. Pearson asks OnVUE candidates to begin check-in 30 minutes before the appointment. The [Exam-day checklist](#exam-day-checklist) turns this into steps.

!!! note "One claim to check before you rely on it"

    imstillwhite also wrote: "If English isn't your first language, you may be entitled to 30 extra minutes on top of the normal 135, but this only applies to the in-person test and must be requested through your company, afaik." Treat it as unconfirmed. In our reading, the "normal 135" matches the FAQ's seat-time guidance (about 135 minutes including check-in), not the 120-minute exam time. The [FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says accommodations are requested through Pearson's accommodations process (the claim says through your company), and asks you to request them "10 days or more before you plan on taking your exam".

## Study plan

Start with the [exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf). The [certification FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) calls it "the authoritative source for exam scope", and the guide itself says: "Read it in full before scheduling." It also says "There is no single required course" and "Anthropic does not guarantee that any particular resource ensures a passing result."

The plan below runs four weeks and follows the order of Anthropic's own prep course, which moves through the domains one module at a time. The four-week length is our suggestion: the guide's How to Prepare section lists activities but gives no study duration. Plan and feature availability in this section is as of September 2026.

### The official prep course

The Anthropic Partner Academy has a free, dedicated path for this exam, the [Claude Certified Associate - Foundations Prep Course](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations). Its page describes the course as "turning everyday business problems into reliable Claude workflows you can stand behind" and shows "Register | FREE". The path exists only on the [Partner Academy](https://anthropic-partners.skilljar.com/), which "requires additional validation at login", and [Pearson VUE's Anthropic page](https://www.pearsonvue.com/us/en/anthropic.html) says "Training is available to members of the Claude Partner Network."

| Module | Minutes as listed | Pairs with | What the module page says it covers |
|---|---|---|---|
| 1. [Claude Platform & Model Foundations](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/claude-platform-model-foundations) | 59 | [Domain 3](#domain-3-product-and-model-selection) | The "four decisions" before any prompt: "entry point, capability features, model, and context"; differentiating Haiku, Sonnet and Opus |
| 2. [Prompting & Task Execution](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/prompting-task-execution) | 53 | [Domain 1](#domain-1-prompting-and-task-execution) | Anchored on Description, a competency from the AI Fluency Framework |
| 3. [Evaluating & Validating Claude's Output](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/evaluating-validating-claudes-output) | 74 | [Domain 2](#domain-2-output-evaluation-and-validation) | Identifying hallucinations, inconsistencies and biases; deciding when human review is required |
| 4. [Workflow Integration & Solution Design](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/workflow-integration-solution-design) | 63 | [Domain 4](#domain-4-workflow-integration-and-solution-design) | Anchored on Delegation: deciding whether each step is "AI-appropriate, human-retained, or collaborative" |
| 5. [Configuration & Knowledge Management](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/configuration-knowledge-management) | 47 | [Domain 5](#domain-5-configuration-and-knowledge-management) | "A good prompt helps you once, while a good configuration helps you every time." |
| 6. [Governance, Risk & Responsible Use](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/governance-risk-responsible-use) | 55 | [Domain 6](#domain-6-governance-risk-and-responsible-use) | "Build the judgement to decide what is safe and appropriate to bring to Claude." |
| 7. [Troubleshooting & Optimization](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/troubleshooting-optimization) | 30 | [Domain 7](#domain-7-troubleshooting-and-optimization) | Tracing a weak output to "under-specification, context overload, the wrong feature or model, or stale configuration" |
| 8. [Course Summary & Next Steps](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/course-summary-next-steps) | 8 | All domains | Connecting the seven skills from the earlier modules, then a brief overview of the CCAO-F exam |

The [path page](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations) recommends three courses first: "Claude 101, AI Fluency: Framework & Foundations, and AI Capabilities and Limitations." The prep-course list on the [CCAO-F exam page](https://anthropic-partners.skilljar.com/claude-certified-associate-foundations-certification) is slightly different: AI Fluency: Framework & Foundations, Claude 101 and Introduction to Claude Cowork. That makes four courses in all, and all four are free on [Claude Academy](https://academy.claude.com/courses), where a free Claude account saves your progress. The Academy [FAQ](https://academy.claude.com/help/faq) says you do not need a paid plan, but Introduction to Claude Cowork lists one as a prerequisite, and trying some features that Claude 101 covers needs one too. Titles there differ slightly from the Partner Academy's.

| Course on academy.claude.com | Listed size | Why it is in this plan |
|---|---|---|
| [Claude 101](https://academy.claude.com/courses/claude-101) | 13 lessons, 1 quiz, 2.5 hr | Lessons include Introduction to projects, Creating with artifacts, Working with skills and Connecting your tools, plus lessons on Enterprise search and Research. Works on any plan (Free, Pro, Max, Team or Enterprise), though a few features it covers need more to try hands-on: Cowork and creating designs, decks and documents as artifacts need a paid plan (on Team and Enterprise an admin may need to turn them on), and Enterprise Search needs Team or Enterprise. |
| [AI Fluency: Framework and foundations](https://academy.claude.com/courses/ai-fluency-framework-foundations) | 14 lessons, 1 quiz, 4 hr | The 4D framework of Delegation, Description, Discernment and Diligence, which modules 2 and 4 of the prep course build on. |
| [AI capabilities and limitations](https://academy.claude.com/courses/ai-capabilities-and-limitations) | 13 lessons, 1 quiz, 3.5 hr | Four properties (Next Token Prediction, Knowledge, Working Memory, Steerability) and the context window as a hard-edged limit. Assumes no technical background. |
| [Introduction to Claude Cowork](https://academy.claude.com/courses/introduction-to-claude-cowork) | 14 lessons, 1 quiz, 2.5 hr | On the prep-course list of the CCAO-F exam page, not the path's recommended three. Handing off multi-step work. The course page lists a paid Claude plan (Pro, Max, Team or Enterprise) as a prerequisite, plus the desktop app for the lessons that work on local folders. |

### Course hours, with the arithmetic

The prep path lists minutes per module but no total, and the [Claude Academy FAQ](https://academy.claude.com/help/faq) calls its listed durations "estimates to help you plan". Treat these sums the same way.

- Prep course: 59 + 53 + 74 + 63 + 47 + 55 + 30 + 8 = 389 minutes, which is 6 hours 29 minutes.
- The three recommended courses: 2.5 + 4 + 3.5 = 10 hours.
- Introduction to Claude Cowork: 2.5 hours.
- Everything in this plan: 6 hours 29 minutes + 10 hours + 2.5 hours = 18 hours 59 minutes of course time.

The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) encourages candidates "to combine hands-on experience with the resources below", and that practice comes on top of the course time. The candidates in [What candidates report](#what-candidates-report) describe anything from no preparation to two weeks of study shared between two exams.

### The guide's own advice, mapped to the weeks

Section 7 of the [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf), How to Prepare, lists six activities.

| How to Prepare item (guide wording) | Where it happens in this plan |
|---|---|
| "Study the exam blueprint in Section 6 and self-assess against each objective" | Before week 1, and again at the end of week 4 |
| "Review official Anthropic documentation and help articles for Claude features such as Projects, Artifacts, Memory, Skills, and Code Execution" | Weeks 1 to 3; the articles are linked in [Resources](#resources) |
| "Practice structuring prompts, decomposing tasks, and iterating to improve outputs" | Week 1 |
| "Build real workflows: configure a Project with instructions and knowledge sources, and evaluate outputs for accuracy and bias" | Weeks 2 and 3 |
| "Practice responsible-use judgment: data sensitivity, appropriate use cases, and when to escalate or seek human review" | Week 4 |
| "Complete the sample questions in Section 8 to familiarize yourself with item style" | Weeks 1, 2 and 4 |

Skills and Code Execution appear in that list but in none of the 30 objectives, and Memory appears only as D3.4's "memory considerations", in the sense of context management. Study all three as the tools behind objectives that are listed: output formats (D2.6), context and memory (D3.4) and configuration (Domain 5). The [Frequently asked questions](#frequently-asked-questions) cover this gap.

### The plan at a glance

| Week | Focus (weight) | Objectives | Prep course modules (minutes) | Other courses | Official sample |
|---|---|---|---|---|---|
| Before week 1 | Orientation | All 30, as a self-assessment | None | Claude 101 (2.5 hr), AI Fluency: Framework and foundations (4 hr), AI capabilities and limitations (3.5 hr) | None yet |
| 1 | Domain 3 (12%), then Domain 1 (14%) | D3.1 to D3.4, D1.1 to D1.4 | 1 (59) and 2 (53) | None | Sample 2 |
| 2 | Domain 2 (21%) | D2.1 to D2.6 | 3 (74) | None | Sample 1 |
| 3 | Domain 4 (16%), then Domain 5 (12%) | D4.1 to D4.5, D5.1 to D5.4 | 4 (63) and 5 (47) | Introduction to Claude Cowork (2.5 hr) | None published |
| 4 | Domain 6 (15%) and Domain 7 (10%), then review | D6.1 to D6.4, D7.1 to D7.3 | 6 (55), 7 (30) and 8 (8) | None | Sample 3, then all three |

Course time per week (our arithmetic): 150 + 240 + 210 = 600 minutes before week 1; 112 minutes in week 1; 74 minutes in week 2; 110 + 150 = 260 minutes in week 3; 93 minutes in week 4. That totals 1,139 minutes, the same 18 hours 59 minutes as above. All three recommended courses come before week 1 because the prep path recommends completing them before its modules, and module 2 is built on AI Fluency's Description competency. Week 4 is light on courses so that it has room for review. Domain 2 gets a week to itself because it carries the largest weight.

### Before week 1: orientation

- Read the guide from start to finish, including its policies. It asks you to do this before scheduling.
- Rate yourself on all 30 objectives using the checklist in [Blueprint](#blueprint): confident, shaky or new.
- Register for the free prep path on the Partner Academy, which "requires additional validation at login".
- Take the three courses the [prep path](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations) recommends before its modules: Claude 101, AI Fluency: Framework and foundations, and AI capabilities and limitations. While you take Claude 101, check your plan: projects are available on every plan (Free allows up to five), but the [personalization article](https://support.claude.com/en/articles/10185728-understanding-claude-s-personalization-features) marks project instructions "(paid plans only)", and Research needs Pro, Max, Team or Enterprise.

### Week 1: product, model and context choices, then prompting

- **Objectives:** D3.1 to D3.4 (features, model types, cost, speed and quality, context and memory), then D1.1 to D1.4 (prompts, decomposition, iteration, task type).
- **Course work:** prep modules 1 and 2.
- **Build, Domain 3:** run the same short task on Haiku and on Sonnet and compare. Anthropic's [model tutorial](https://academy.claude.com/tutorials/choosing-the-right-claude-model) says "Free includes Haiku and Sonnet", so this works on any plan; it lists Haiku for "Quick answers, summaries, and simple extraction" and calls Sonnet "your versatile default". This is the rule in [Sample 2's rationale](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) in practice: match "a faster, lower-cost model to straightforward, high-volume work, reserving the most capable model for complex reasoning." On Sonnet, try the effort setting too. As of September 2026 the [help center](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings) lists the effort selector for Sonnet 5, Sonnet 4.6 and recent Opus and Fable models, not for Haiku, and says "Low and Medium work well for routine tasks and stretch your usage further."
- **Build, Domain 3, context and memory:** run the [blank-slate probe](https://academy.claude.com/courses/ai-capabilities-and-limitations/working-memory) from AI capabilities and limitations: teach Claude something, open a brand-new conversation, ask a question that assumes it remembers, and "Watch it start from zero." In the Claude apps the result depends on memory. Memory is on by default on Free, Pro and Max, and the [memory article](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context) says Claude "saves memory as a set of individual topics as you chat", so an ordinary new chat may already know what you taught it. Repeat the probe in an [incognito chat](https://support.claude.com/en/articles/12260368-use-incognito-chats), which "won’t use Claude’s existing memory". Our framing: the gap between the two runs is the restart-or-persist choice that D3.4 asks you to manage.
- **Build, Domain 1:** the [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) "Practice structuring prompts, decomposing tasks, and iterating to improve outputs". Write a weekly task of yours as a structured prompt. Split a multi-step request and, as the [steerability lesson](https://academy.claude.com/courses/ai-capabilities-and-limitations/steerability) suggests, ask Claude "to stop and show you the result of step 2 before continuing". Iterate with specific feedback: [Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results) rates "Cut the first two paragraphs and make the conclusion more action-oriented" above "Make it shorter", and the [Cowork course](https://academy.claude.com/courses/introduction-to-claude-cowork/validating-skills-for-plugins) advises "Change one thing at a time."
- **Watch for, as of September 2026:** objective D3.1 says "research mode", while the [help center](https://support.claude.com/en/articles/11088861-use-research-on-claude) calls the feature Research. [Domain 3](#domain-3-product-and-model-selection) covers how the guide's wording maps to today's apps, including how web search works in the new Claude experience.
- **Read:** [Domain 3](#domain-3-product-and-model-selection) and [Domain 1](#domain-1-prompting-and-task-execution) on this page; [Choosing a model in the apps](knowledge/claude-for-work.md#choosing-a-model-in-the-apps); [Memory, styles and personalization](knowledge/claude-for-work.md#memory-styles-and-personalization); [Why context is a budget](knowledge/context-engineering.md#why-context-is-a-budget); [Compaction, context editing and memory](knowledge/context-engineering.md#compaction-context-editing-and-memory); [Principles that decide most prompt questions](knowledge/prompt-engineering.md#principles-that-decide-most-prompt-questions); [Chain of thought and prompt chaining](knowledge/prompt-engineering.md#chain-of-thought-and-prompt-chaining); [Prompt versioning and iteration](knowledge/prompt-engineering.md#prompt-versioning-and-iteration).
- **Self-check:** [Sample 2](#sample-2-high-volume-customer-reply-drafts).

### Week 2: output evaluation and validation

- **Objectives:** D2.1 to D2.6 (accuracy and completeness, hallucinations and bias, fact-checking, when to verify, adapting for an audience, output formats).
- **Course work:** prep module 3, the longest at 74 minutes. Revisit Discernment and the Description-Discernment loop in AI Fluency: Framework and foundations.
- **Build:** the evaluation half of the [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) "Build real workflows" item. Ask Claude to summarize, with citations, a long document you know well, and check every cited point against the source: [Sample 1's rationale](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) calls validating citations against an authoritative source "the diligence step required." For an answer you are unsure of, the Academy's [hallucination tutorial](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate) says to "start a new chat and ask the AI to find errors in the answer, and to confirm that the sources support the statements." Add "flag anything you're not confident about" to a drafting prompt, as an [Academy use case](https://academy.claude.com/use-cases/prd-from-a-one-pager) suggests, and see where it points you. Finally, for D2.6, produce one piece of content in each of its three formats: inline, as an artifact, and as structured data such as a table or a spreadsheet. Code execution and file creation is available on every plan, and Claude can create .xlsx, .pptx, .docx and PDF files.
- **Keep in mind:** the [help center](https://support.claude.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on) says "Users should not rely on Claude as a singular source of truth and should carefully scrutinize any high-stakes advice given by Claude."
- **Read:** [Domain 2](#domain-2-output-evaluation-and-validation); [Verifying Claude's output](knowledge/claude-for-work.md#verifying-claudes-output); [Reducing hallucinations](knowledge/prompt-engineering.md#reducing-hallucinations); [Human review and confidence calibration](knowledge/evaluation-and-reliability.md#human-review-and-confidence-calibration); [Artifacts, files and code execution](knowledge/claude-for-work.md#artifacts-files-and-code-execution).
- **Self-check:** [Sample 1](#sample-1-a-confident-summary-bound-for-compliance). Say why A, C and D each fail before you open the rationale.

### Week 3: workflow integration, then configuration and knowledge

- **Objectives:** D4.1 to D4.5 (requirements, research and planning, solution design, augmenting or redesigning workflows, communicating value and limits), then D5.1 to D5.4 (Projects, knowledge and connectors, system-level instructions, maintenance).
- **Course work:** prep modules 4 and 5; Introduction to Claude Cowork if your plan allows it.
- **Build, Domain 5:** the configuration half of the guide's "Build real workflows" item. Create a project ("+ New Project"; Claude cannot see its name or description), click "Set project instructions" (on Free this may not be offered: see the paid-plan note in [Before week 1](#before-week-1-orientation)), and upload knowledge files. Then test the rule from the [projects article](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects): "Context is not shared across chats within a project unless the information is added into the project knowledge base." As of September 2026 the [memory article](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context) adds that "Each project has its own separate memory space and dedicated project summary", so with memory on, something one chat in the project learned can reappear in the next. To test the knowledge-base rule on its own, pause memory first (Claude then "won't use memory or make new memories"); incognito chats are not available inside projects. On paid plans, when project knowledge nears the context limit, Claude turns on RAG mode by itself; the help center reserves this enhanced project knowledge for Pro, Max, Team and Enterprise. If your organization allows it, add a Google Drive file; Drive files can only be added to private projects.
- **Build, Domain 4:** map one real workflow step by step and label each step the way [module 4](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/workflow-integration-solution-design) does: "AI-appropriate, human-retained, or collaborative". Then decide which steps Claude should augment and which the workflow should be redesigned around.
- **Watch for, as of September 2026:** official pages disagree on which plans get the Google Workspace connectors; check what your own plan offers, and see [Domain 5](#domain-5-configuration-and-knowledge-management) for both sides. The exam uses the [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) wording, "e.g., Google Drive, Gmail".
- **Read:** [Domain 4](#domain-4-workflow-integration-and-solution-design) and [Domain 5](#domain-5-configuration-and-knowledge-management); [Projects](knowledge/claude-for-work.md#projects); [Search, Research and connectors](knowledge/claude-for-work.md#search-research-and-connectors); [Skills in the Claude apps](knowledge/claude-for-work.md#skills-in-the-claude-apps); [Claude Cowork](knowledge/claude-for-work.md#claude-cowork); [XML tags, system prompts and roles](knowledge/prompt-engineering.md#xml-tags-system-prompts-and-roles); [Discovery and requirements](knowledge/solution-architecture.md#discovery-and-requirements); [Stakeholder communication and lifecycle](knowledge/solution-architecture.md#stakeholder-communication-and-lifecycle).
- **Self-check:** neither domain has an official sample. Re-rate yourself on their nine objectives, and for each one state the decision rule from its domain section without notes.

### Week 4: governance and troubleshooting, then review

- **Objectives:** D6.1 to D6.4 (appropriate use cases, data sensitivity and regulation, organizational policy, ethics), then D7.1 to D7.3 (diagnosing weak outputs, adjusting to feedback, optimizing workflows).
- **Course work:** prep modules 6, 7 and 8.
- **Build, Domain 6:** the [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) "Practice responsible-use judgment" item. Take a spreadsheet of customer records and, before anything is uploaded, follow the [AI Fluency for small businesses](https://academy.claude.com/courses/ai-fluency-for-small-businesses) lesson on using data: mark "anything you wouldn't want outside your business", then make a copy that replaces names with placeholders, removes exact figures that are not needed and deletes contact details entirely. Check that the analysis still works on the copy. That is [Sample 3's answer](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) in practice: redact or anonymize regulated identifiers "so the analysis can proceed without exposing protected data." Read your organization's AI policy and list what it restricts.
- **Build, Domain 7:** take a prompt that gave a generic answer and fix it with [Claude 101's troubleshooting table](https://academy.claude.com/courses/claude-101/getting-better-results), which traces generic output to "Your prompt didn't include enough context about your specific situation". Re-run a question about recent events with web search on (the new Claude experience has no toggle and searches when it helps) and compare, as the [knowledge lesson](https://academy.claude.com/courses/ai-capabilities-and-limitations/knowledge) of AI capabilities and limitations suggests. Classify each failure you find by [module 7's](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/troubleshooting-optimization) four root causes.
- **Read:** [Domain 6](#domain-6-governance-risk-and-responsible-use) and [Domain 7](#domain-7-troubleshooting-and-optimization); [Untrusted input, PII and data leakage](knowledge/security-and-governance.md#untrusted-input-pii-and-data-leakage); [Anthropic's Usage Policy](knowledge/security-and-governance.md#anthropics-usage-policy); [Data retention, training and compliance](knowledge/security-and-governance.md#data-retention-training-and-compliance); [Responsible use for business users](knowledge/security-and-governance.md#responsible-use-for-business-users); [Escalation and ambiguity](knowledge/evaluation-and-reliability.md#escalation-and-ambiguity); [Debugging: model or integration](knowledge/evaluation-and-reliability.md#debugging-model-or-integration).
- **Self-check:** [Sample 3](#sample-3-regulated-personal-data-in-a-spreadsheet), then all three samples again, naming for every wrong option the part of Anthropic's rationale that rules it out. Re-rate all 30 objectives. Finish with the [Exam-day checklist](#exam-day-checklist) and the [Quick reference](quick-reference.md#decision-rules).

### If your time or background differs

- **Two weeks instead of four:** merge weeks 1 and 2, then weeks 3 and 4. Keep every build and every sample question, and drop Introduction to Claude Cowork first, since it is not among the three courses the prep path recommends.
- **New to Claude:** add a week of everyday use before week 1. The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) recommends "Regular, hands-on experience using Claude in a professional setting", and the [prep path](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations) is written "for people already using AI-powered productivity tools in their work."

## Exam-day checklist

Work through these in order. Each item comes from the CCAO-F guide, Anthropic's certification pages or Pearson VUE's pages and candidate documents, as of September 2026; the pacing arithmetic is ours. The rules that can cost you the fee are explained in [Policies that cost candidates money](index.md#policies-that-cost-candidates-money).

### Two weeks or more before

- [ ] The name on your registration matches your government-issued photo ID exactly. If it does not, email certifications-support@anthropic.com before you schedule; the [FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) asks you to write from the address you registered with, with the subject line "Name Correction Request" and your name in Latin characters only. Corrections typically take 24 to 48 business hours.
- [ ] Any accommodation is approved before you book. Pearson VUE must approve it first, asks you to allow 10 business days for review, and cannot add accommodations to an exam that is already scheduled.
- [ ] You have chosen online proctoring (OnVUE) or a Pearson test center. Candidates with a government-issued ID from Belarus, Cuba, North Korea, Russia, Syria or restricted regions of Ukraine must use a test center, and Pearson suspended delivery for residents of Iran, online and at test centers, from September 8, 2026.
- [ ] For OnVUE, you have run and passed Pearson's System Test on the same device and network you will use on exam day. Passing it does not guarantee a problem-free exam.
- [ ] Your machine meets the OnVUE minimums: Windows 10 or macOS 14 or higher; a working webcam, microphone and speaker (no headphones or headsets); one display only; at least 6 Mbps download and 2 Mbps upload. [Pearson's OnVUE page for Anthropic](https://www.pearsonvue.com/us/en/anthropic/onvue.html) also lists as prohibited technology virtual machines or beta operating systems, secondary or touchscreen displays ("disconnect and cover if not removable"), and VPNs, corporate networks and public or shared networks.
- [ ] If you plan to use a work laptop, your IT team has been asked, early, to allow Pearson's domains and stop background applications that block OnVUE. Anthropic's [setup page](https://anthropic-partners.skilljar.com/page/computer-and-network-setup) says some of these "run as background services you may not be able to stop yourself". The Windows list includes Microsoft Edge, Google Chrome, Zoom, Microsoft Outlook, the Microsoft Teams updater and the Claude desktop application, so quit Claude before check-in even if it is part of your working day. Anthropic's setup page tells you to contact your corporate network administrator if the system test flags a domain, while Pearson's page lists corporate networks as prohibited; our reading is that a personal network avoids the conflict. If the company machine or network still will not cooperate, Anthropic recommends a personal computer on a personal network or a Pearson test center.

!!! warning "Plan on 48 hours to cancel or reschedule, not 24"

    The [CCAO-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) says: "You may cancel or reschedule up to 24 hours before your appointment." Anthropic's [policies page](https://anthropic-partners.skilljar.com/page/policies-certifications) and [FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) say 48 hours, and so does [Pearson VUE's Anthropic page](https://www.pearsonvue.com/us/en/anthropic.html) for test-center appointments. Use 48. Canceling in Pearson only frees the time slot: to get your money back, email certifications-support@anthropic.com. After any reschedule, make sure an email confirming the new date arrives.

### The day before

- [ ] Restart the computer you will test on, and make sure nobody else on your network will be streaming or making large downloads during your slot.
- [ ] Your ID is valid, unexpired, government-issued, carries a recognizable photo, and matches your booking name exactly. Expired, digital, damaged, copied and privately issued IDs are refused.
- [ ] For OnVUE, your desk is empty except for the testing computer, pre-approved items and comfort aids, and a drink in an unmarked container. Books, notes, paper, pens and writing tools are gone, and any whiteboard or note board in the room is wiped.
- [ ] For OnVUE, you have a quiet room where you will be alone for the whole session. Public spaces such as offices, libraries and coffee shops are not allowed.

### Before the exam starts

- [ ] Online: begin check-in 30 minutes before your appointment. Expect technology checks, a 360° room scan, and photos of you and your ID. If any requirement is not met, you cannot test and the fee is forfeited.
- [ ] At a test center: arrive as early as your confirmation email says. Personal items go into a locker or other secure area, since they are not permitted in the testing room, and the administrator provides any materials the test sponsor authorizes, such as a laminated note board.
- [ ] Your phone, smart watch, headphones and anything that records are out of reach, and every application except OnVUE is closed.
- [ ] You have budgeted about 135 minutes of seat time for a 120-minute exam: the [FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says that total "includes check-in, instructions, and a brief post-exam survey".
- [ ] You are ready to accept the confidentiality and non-disclosure agreement. If you do not accept it, the session ends and no refund is issued.

### During the exam

- [ ] Read how many responses each item asks for. The exam mixes multiple-choice and multiple-response items, and each item states how many to select.
- [ ] Keep to about 2 minutes per item: 60 items in 120 minutes (our arithmetic). Check the clock at the halfway mark, when about 30 items should be done.
- [ ] Use only the tools you are given. Online, the allowance that [Pearson's OnVUE page for Anthropic](https://www.pearsonvue.com/us/en/anthropic/onvue.html) lists as pre-approved for all candidates is a digital whiteboard, and it "does not grant permission to use physical whiteboards or writing materials of any kind"; the whiteboard is wiped if your connection drops. At a test center, the administrator supplies whatever the test sponsor authorizes, such as a laminated note board. The [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) mention of "scratch paper provided by the proctor" is only an example of what Pearson VUE may specify.
- [ ] Stay in webcam view, stay alone, and do not speak or read aloud unless instructed.
- [ ] No notes, no browser translation tools and no AI products or services. The exam is closed book, and the [Exam Policy](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F34hhd92iyp94a0gtbr15cy5jk%2Fpublic%2F1782870704%2FAnthropic+Certification+Exam+Policy.pdf) lists "Using AI products or services to assist you during the Exam" as prohibited.
- [ ] No unscheduled breaks unless requested and approved in advance.
- [ ] If a question looks factually wrong, unclear, has more than one defensible answer or does not match the guide, note its question number, if one is shown, and report it after the exam (see below). Online, you can jot it on the digital whiteboard, but Pearson's [whiteboard page](https://www.pearsonvue.com/us/en/onvue/whiteboard.html) says your work there is accessible "in every section during your entire exam" and is wiped if the connection drops, so keep the number in mind for your report; at a test center, staff cannot answer content questions.
- [ ] If OnVUE freezes or disconnects, close it and relaunch it from your downloads folder. The in-exam chat reaches a proctor, who cannot pause or extend the exam.

### After the exam

- [ ] Note your score. It appears on screen when you finish, and test center candidates also receive a printed score report.
- [ ] If you passed, accept the badge from the Credly email, and add a personal email address to your Credly profile so the badge stays with you if you change jobs. The credential is yours, not your employer's.
- [ ] Do not share or discuss the questions, including in study groups and online forums.
- [ ] If you did not pass, use the percent-correct by domain on your score report to plan your review (those percentages are diagnostic; pass or fail rests on the total scaled score). The wait is 14 days after a first failed attempt, 30 after a second and 90 after a third; you can sit the exam up to four times in a rolling twelve-month period, and each attempt costs the exam fee, with any partner discount applied as on the first attempt.
- [ ] To dispute a result, appeal to Pearson VUE support within 14 days of your exam date.
- [ ] To flag a faulty question you noted, report it to Pearson. The [policies page](https://anthropic-partners.skilljar.com/page/policies-certifications) says anyone can do this "whether they passed or failed", that reporting "never counts against you or affects your result", and that a report is different from an appeal.
- [ ] Put your expiry date in your calendar. The credential is valid for 12 months from the date it is awarded, and on-time renewal is a free, non-proctored assessment on the Anthropic Partner Academy; if it lapses, you retake the full exam at the full fee. Details are in [Renewal](index.md#renewal).

## Resources

Official material comes first, in the order you are likely to need it. Community resources follow, with what is known about each. Details are as of September 2026; links and course contents change, and the [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) itself "is subject to change without notice."

### Official: the exam and its rules

| Resource | What it gives you |
|---|---|
| [CCAO-F exam guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) | Version 1.0, effective July 2026, 11 pages: audience, format, the blueprint and all 30 objectives, How to Prepare, three sample questions with rationales, scoring and policies. The guide says it "is subject to change without notice." |
| [CCAO-F certification page](https://anthropic-partners.skilljar.com/claude-certified-associate-foundations-certification) | Where you purchase the exam on the Anthropic Partner Academy (listed at &#36;99). |
| [Partner Certifications page](https://anthropic-partners.skilljar.com/page/partner-certifications) | Links to the prep courses, the exam guide and registration for each exam, and notes that CCAO-F "Does not count toward Claude Partner Network tier eligibility." |
| [Certifications FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) | Eligibility, prices and partner discounts, seat time, results, retakes, name corrections and refunds. |
| [Certification policies page](https://anthropic-partners.skilljar.com/page/policies-certifications) | Rescheduling, retakes, renewal, misconduct, appeals and faulty-question reports. Registering means agreeing to the Terms, the Exam Policy and the Anthropic Usage Policy, and the page says those documents apply where it conflicts with them. |
| [Anthropic Certification Exam Policy (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F34hhd92iyp94a0gtbr15cy5jk%2Fpublic%2F1782870704%2FAnthropic+Certification+Exam+Policy.pdf) | The list of prohibited conduct you agree to when you register, including the use of AI products during the exam. Last updated June 25, 2026. |
| [Certification Terms and Conditions (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F34hhd92iyp94a0gtbr15cy5jk%2Fpublic%2F1782870634%2FCertification+Terms+and+Conditions.pdf) | The program terms, including the rule against presenting yourself as certified after expiry. |
| [Exam Registration Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542947%2FClaude+Certification+Program+-+Exam+Registration+Guide.pdf) | The registration steps, which are the same for every Claude certification exam. |
| [Computer and network setup](https://anthropic-partners.skilljar.com/page/computer-and-network-setup) | OnVUE preparation: domains to allow, applications that block the launch, and what to do on a company machine. |
| [Pearson VUE: Anthropic](https://www.pearsonvue.com/us/en/anthropic.html) and [OnVUE for Anthropic](https://www.pearsonvue.com/us/en/anthropic/onvue.html) | Scheduling, support contacts, OnVUE requirements, acceptable IDs and testing-space rules. |
| [Pearson VUE accommodations for Anthropic](https://www.pearsonvue.com/us/en/test-takers/accommodations/pearson_approve.anthropic.html) | Where to request accommodations. The guide's printed text shows a generic Pearson address, but its link points here. |
| [CCAO-F badge on Credly](https://www.credly.com/org/anthropic/badge/claude-certified-associate-foundations) | The badge you receive on passing, with its eight listed skills. |

### Official: courses

| Resource | What it gives you |
|---|---|
| [Claude Certified Associate - Foundations Prep Course](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations) | Anthropic's free, dedicated prep path, on the Partner Academy only: seven modules that follow the seven domains plus a summary module, 389 minutes in total by our sum. See [Study plan](#study-plan). |
| [Claude 101](https://academy.claude.com/courses/claude-101) | Recommended before the prep path. Everyday use of the Claude apps, from prompting to projects, artifacts, skills and connected tools. 2.5 hr. |
| [AI Fluency: Framework and foundations](https://academy.claude.com/courses/ai-fluency-framework-foundations) | Recommended before the prep path. The 4D framework of Delegation, Description, Discernment and Diligence. 4 hr. |
| [AI capabilities and limitations](https://academy.claude.com/courses/ai-capabilities-and-limitations) | Recommended before the prep path. What language models can and cannot do, including knowledge limits and the context window. 3.5 hr. |
| [Introduction to Claude Cowork](https://academy.claude.com/courses/introduction-to-claude-cowork) | On the prep-course list of the [CCAO-F exam page](https://anthropic-partners.skilljar.com/claude-certified-associate-foundations-certification), though not among the path's recommended three. Delegating multi-step work. 2.5 hr; the course page lists a paid plan as a prerequisite. |
| [Choosing the right Claude model: Haiku, Sonnet, Opus, or Fable](https://academy.claude.com/tutorials/choosing-the-right-claude-model) | Academy tutorial on matching each model to the task, and on checking results in proportion to the stakes. As of September 2026 it covers four models, including Fable; objective D3.2 names three (Haiku, Sonnet, Opus), and the exam uses the guide's wording. |
| [How to select the right effort setting for Claude Cowork and Chat](https://academy.claude.com/tutorials/how-to-select-the-right-effort-setting-for-claude-cowork-and-chat) | Academy tutorial on the signs of too little and too much effort. |
| [Why do AI models hallucinate?](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate) | Academy tutorial, including the fresh-chat check for an answer you are unsure of. |

Claude Academy is free: you can browse the catalog without signing in, and a free Claude account saves your progress and earns completion badges. The dedicated prep path exists only on the separate [Anthropic Partner Academy](https://anthropic-partners.skilljar.com/), which "requires additional validation at login". Claude Academy's own [FAQ](https://academy.claude.com/help/faq) separates the two: its course badges are "free and are earned by passing a course’s quizzes", while exam credentials come from the paid, proctored Pearson and Credly program.

### Official: help-center articles for the features the guide names

The [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) How to Prepare section tells you to review "official Anthropic documentation and help articles for Claude features such as Projects, Artifacts, Memory, Skills, and Code Execution". These are the matching articles on the Claude help center, plus the ones behind the other objectives.

| Feature | Article | Objectives it serves |
|---|---|---|
| Projects | [What are projects?](https://support.claude.com/en/articles/9517075-what-are-projects) and [How can I create and manage projects?](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects) | D3.1, D5.1, D5.4 |
| Project knowledge at scale | [Retrieval augmented generation (RAG) for projects](https://support.claude.com/en/articles/11473015-retrieval-augmented-generation-rag-for-projects) | D3.4, D5.2 |
| Artifacts | [What are artifacts and how do I use them?](https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them) | D2.6, D3.1 |
| Memory | [Use Claude's chat search and memory to build on previous context](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context) | D3.4 |
| Skills | [What are skills?](https://support.claude.com/en/articles/12512176-what-are-skills) and [Use skills in Claude](https://support.claude.com/en/articles/12512180-use-skills-in-claude) | D5.3, D5.4 |
| Code execution | [Create and edit files with Claude](https://support.claude.com/en/articles/12111783-create-and-edit-files-with-claude) | D2.6 |
| Research | [Use research on Claude](https://support.claude.com/en/articles/11088861-use-research-on-claude) | D3.1, D4.2 |
| Connectors | [Use connectors to extend Claude's capabilities](https://support.claude.com/en/articles/11176164-use-connectors-to-extend-claude-s-capabilities) and [Use Google Workspace connectors](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors) | D5.2 |
| Models, effort and thinking | [Change the model, effort, and thinking settings](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings) | D3.2, D3.3 |
| Context and usage | [How do usage and length limits work?](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work) | D3.4, D7.3 |
| Hallucinations | [Claude is providing incorrect or misleading responses. What's going on?](https://support.claude.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on) | D2.2, D2.3 |
| Acceptable use | [Anthropic Usage Policy](https://www.anthropic.com/legal/aup) | D6.1, D6.3 |

The objective mapping in the last column is ours. Where the help center and other official pages disagree, such as on connector availability and Gmail actions, the domain sections show both sides.

### Official: repositories

These are optional for an Associate candidate: the [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) lists "no mandatory prerequisites or required courses", and no API experience is needed. The first one is dated.

- [anthropics/prompt-eng-interactive-tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial): 9 chapters of prompt-engineering exercises plus an appendix. It is built on claude-3-haiku-20240307, which was retired on April 20, 2026 (the deprecations page names claude-haiku-4-5-20251001 as the replacement), and its newest commit is dated 2024-04-08. Chapter 5, "Formatting Output & Speaking for Claude", covers speaking for Claude (prefill), which current docs say is no longer supported on the last assistant turn starting with the Claude 4.6 models.
- [anthropics/skills](https://github.com/anthropics/skills): Anthropic's public Agent Skills repository. Useful for seeing what a skill looks like, which supports the Skills item in How to Prepare.

### Community resources

None of these is published by Anthropic, and CertSafari and the Mistry course both state that they are not affiliated with or endorsed by Anthropic. Treat their practice questions as unverified. Figures such as question counts and student numbers are as displayed on September 23, 2026.

| Resource | Cost and access | What to know |
|---|---|---|
| [Ravikiran's CCAO-F study guide](https://ravikirans.com/claude-certified-associate-foundations-study-guide/) | Free | Maps every domain of the blueprint to Anthropic's own documentation and help-center pages. Dated July 14, 2026, so check product details against the current help center. |
| [Tutorials Dojo CCAO-F study guide](https://tutorialsdojo.com/ccao-f-claude-certified-associate-foundations-study-guide/) | Free | Its sample questions cite Claude help-center articles, which makes them easy to check. The questions are Tutorials Dojo's own, not Anthropic's. |
| [CertSafari CCAO-F practice bank](https://www.certsafari.com/anthropic/claude-certified-associate-foundations) | Free | 757 practice questions, bank created August 29, 2026. Its CCAR-F page names a single person as builder and maintainer, and lists the formats CertSafari offers as multiple choice only, while the real exam also has multiple-response items. |
| [Claude Certified Associate Foundations (CCAO-F) Exam Guide](https://www.udemy.com/course/claude-certified-associate-foundations-ccao-f-exam-guide/) by Ankit Mistry and Ajay Gadhave (Udemy) | Paid | 48 lectures, 5h 31m total length, plus 35 practice quiz questions with every answer explained. It states that no course can promise you will pass. |
| [Claude Certified Associate Foundations CCAO-F Practice Tests](https://www.udemy.com/course/claude-associate/) by Mike Wheeler (Udemy) | Paid | Says every question maps to one of the seven domains of the Version 1.0 guide. 325 students as displayed. |
| [CCAO-F study-guide video](https://www.youtube.com/watch?v=lcjm0TvsKH0) by Program Strategy HQ (YouTube) | Free | 2,491 seconds long (about 41.5 minutes, our arithmetic). Links the official prep path. |
| [Amey-Thakur/CLAUDE-CERTIFICATIONS](https://github.com/Amey-Thakur/CLAUDE-CERTIFICATIONS) (GitHub) | Free | Cheat sheets, practice questions, mocks and flashcards across all four exams; last pushed September 22, 2026. It says every course in the program is free on the public Claude Academy with no partner account needed, but the dedicated CCAO-F prep path exists only on the Partner Academy. |
| [dnacenta/claude-certified-architect](https://github.com/dnacenta/claude-certified-architect) (GitHub) | Free | An Architect-focused repository that also carries overview pages for the other three exams, sourced from the Version 1.0 guides. Says it was last refreshed in September 2026. |
| [Matthew Purcell's review of all four exams](https://www.linkedin.com/pulse/claude-certification-exams-honest-review-matthew-purcell-byo2e) (LinkedIn) | Free | A first-hand account, summarized in [What candidates report](#what-candidates-report). Treat it as opinion; the author also promotes a practice exam they wrote. |

Third-party practice questions are not Anthropic's items. They can be a useful check on your reasoning, but the candidates in [What candidates report](#what-candidates-report) who compared mocks with the real exam found the two different; [one](https://www.reddit.com/r/ClaudeAI/comments/1wjnvrw/i_passed_the_ccaof_claude_associate_foundation/) found the official exam "harder and more closely aligned with the prep course." The [FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says the practice exam on the previous platform "was retired in the move to Pearson" and points to the guide's sample questions for the format and style; the guide has three.

!!! danger "Sites that sell real exam questions"

    Some sites advertise actual exam content. One, p2pexams, says its CCAO-F questions were "submitted by actual Anthropic CCAO-F exam candidates". The [Exam Policy](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F34hhd92iyp94a0gtbr15cy5jk%2Fpublic%2F1782870704%2FAnthropic+Certification+Exam+Policy.pdf) lists as prohibited "Seeking or obtaining unauthorized access to any Exam or Exam Content, including use of unauthorized publication of Exam questions or answers". The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) warns that misconduct "may result in invalidation of your result, revocation of your credential, and a ban from future exams", and the [policies page](https://anthropic-partners.skilljar.com/page/policies-certifications) adds that serious cases "may be referred for legal action." This page does not link to such sites.

## Frequently asked questions

Short answers to the questions candidates ask most, each pointing to the section of this page that covers it in full.

??? question "Do I need to code, or to know the Claude API?"

    No. The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) says "There are no mandatory prerequisites or required courses, and no software-development or API experience is needed", and that the certification is not intended for developers "who build against APIs or design agentic systems". What it does expect is judgment about when work goes beyond you: Associates escalate "more complex or technical work" to Claude Architects and Developers. Read in that light, objectives such as D1.1 ("prompts for business and technical tasks") and D4.3 ("solution design, development, and iteration") are about prompting Claude and supporting that work, not writing code (our reading of the guide's Section 3).

??? question "Who can register, and does passing count toward my firm's partner tier?"

    The [FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) limits registration to people at Claude Partner Network organizations, with a partner email on a recognized company domain; personal email addresses do not work, and you must be at least 18. The list price is &#36;99 USD before partner-tier discounts, which are covered in [Fees and partner discounts](index.md#fees-and-partner-discounts). On partner standing, the program-level copy is general: the FAQ's overview says certification "counts toward your firm's standing in the Claude Partner Network", and [Pearson VUE's Anthropic page](https://www.pearsonvue.com/us/en/anthropic.html) says it "counts toward partner program standing." For this exam, though, the same FAQ lists Claude Certified Associate under "Exams that do not count towards Claude Partner Network eligibility", and the [Partner Certifications page](https://anthropic-partners.skilljar.com/page/partner-certifications) says it "Does not count toward Claude Partner Network tier eligibility." Take CCAO-F as not counting toward your firm's tier.

??? question "Is there an official practice exam?"

    No. The [FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says the practice exam on the previous platform "was retired in the move to Pearson", and that the exam guide's sample questions show the format and style. The guide has three, reproduced with Anthropic's rationales in [Official sample questions](#official-sample-questions). Anything else calling itself a CCAO-F practice test is a third party's work; see [Resources](#resources).

??? question "Will I see multiple-response items?"

    Plan for them. The [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) item format is "Multiple-choice and multiple-response items; each item states how many responses to select". None of the three published samples shows the format, and neither the guide nor the [program FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says whether partial credit is given; see [What the samples do not show](#what-the-samples-do-not-show).

??? question "Memory, Skills and Code Execution are in How to Prepare, but no objective names them as features. Should I study them?"

    Yes, as supporting knowledge. The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) says "Exam items are written against these objectives", and none of the 30 objectives names those three features. But How to Prepare tells you to review help articles for "Projects, Artifacts, Memory, Skills, and Code Execution", and they sit behind listed objectives: D3.4 covers "memory considerations (when to restart, summarize, or persist)", D2.6 covers output formats including "structured data", and Domain 5 covers configuration. Our reading: these features are more likely to appear as part of a situation than as the thing an item asks you to define, so learn what each one does and when you would choose it; the [Study plan](#study-plan) fits them into weeks 1 to 3.

??? question "Are the guide's 'research mode' and its Google Drive and Gmail connectors still the product names?"

    As of September 2026 the [help center](https://support.claude.com/en/articles/11088861-use-research-on-claude) calls the feature Research. It is available on paid plans and needs web search turned on, though in the new Claude experience the [web search article](https://support.claude.com/en/articles/10684626-enable-and-use-web-search) says "there's no web search toggle. Claude searches the web when it helps." Gmail and Google Drive are covered, alongside Google Calendar, by the help center's [Google Workspace connectors article](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors). Official pages disagree on some connector details: that article says the connectors "are available for all users on Claude and Claude Desktop", while [claude.com's Drive documentation](https://claude.com/docs/connectors/google/drive.md) says "Available on Pro, Max, Team, and Enterprise plans." On the exam, expect the [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) July 2026 wording. [Domain 3](#domain-3-product-and-model-selection) and [Domain 5](#domain-5-configuration-and-knowledge-management) map each term to today's apps.

??? question "Is CCAO-F the easy one of the four?"

    Candidates disagree. [One who passed all four](https://dev.to/bluepanda/i-cleared-all-4-anthropic-claude-certifications-what-you-actually-need-to-know-and-what-to-skip-224m) ranked it the easiest of the four; [another who passed all four](https://www.reddit.com/r/ClaudeAI/comments/1v2x1p7/got_all_4_claude_certifications_ccap_ccaf_ccdvf/) wrote that "CCAO-F is not entry level compared to other 3", and [one who took it at their company's request](https://www.reddit.com/r/claudeskills/comments/1vuxasi/failed_ccaof_am_i_the_only_one/) failed and called it "super hard". The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) itself places its candidate "between casual AI prompt users and technical AI practitioners", and the pass standard is fixed: a scaled 720 on a range of 100 to 1,000, the same cut score as the other three exams. [What candidates report](#what-candidates-report) has the individual accounts.

??? question "What does the score report tell me if I fail?"

    Your pass or fail result, a scaled score from 100 to 1,000, and the percentage of items you answered correctly in each content domain. The [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) says those domain percentages "are not used to determine your pass or fail result, which is based on your total scaled score"; use them to choose what to review. The wait before a retake is 14 days after a first failed attempt, 30 after a second and 90 after a third, with up to four attempts in a rolling twelve-month period, and each attempt costs the exam fee. The limits apply per exam, so a fail does not stop you registering for a different Claude exam. More in [Scoring, results and badges](index.md#scoring-results-and-badges).

??? question "How long does the credential last?"

    12 months from the date it is awarded; the [guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf) gives the reason: "Because the underlying technology evolves rapidly, the credential is time-limited so that holders maintain current knowledge." On-time renewal is free: you review what has changed and complete a non-proctored assessment on the Anthropic Partner Academy. If the credential lapses, you retake the full exam at the full fee. The [policies page](https://anthropic-partners.skilljar.com/page/policies-certifications) describes the renewal as an "open-book online assessment" that you can retake as often as you need and that extends the credential by 12 months from your current expiration date; the [FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications) says full details "will be shared before the first certifications come up for renewal." See [Renewal](index.md#renewal).

??? question "Is there a Professional level for the Associate track?"

    Not at present: as of September 2026 the [Partner Certifications page](https://anthropic-partners.skilljar.com/page/partner-certifications) lists only Foundations for the Associate role. How that squares with the launch post's foundation-to-professional wording is in [Where CCAO-F sits in the program](#where-ccao-f-sits-in-the-program).

??? info "Sources"

    - [Claude Certified Associate, Foundations Exam Guide, Version 1.0 (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf): exam details table, purpose, intended audience, minimally qualified candidate, recommended experience and prerequisites, blueprint weights and all 30 objectives, How to Prepare, sample question stems and domains, scoring, retake and renewal rules, 24-hour cancellation wording, document control
    - [Anthropic Partner Academy: Partner Certifications](https://anthropic-partners.skilljar.com/page/partner-certifications): three roles and their descriptions, "where available" Professional path, Associate listed at Foundations only, no credit toward Claude Partner Network tier eligibility, link to the current exam guide
    - [Anthropic Partner Academy: Certification FAQ](https://anthropic-partners.skilljar.com/page/faq-certifications): eligibility and minimum age, partner discounts and payment, list prices of all four exams, exams that count toward partner eligibility, seat time, closed book and translation tools, English only, scaled scoring and equating, section breakdown for retakes, 48-hour cancellation, refunds by email, OnVUE country restrictions and the Iran suspension, retired practice exam, exam guide as the authoritative source for scope, scenario-based item wording
    - [Anthropic Partner Academy: Certification Policies](https://anthropic-partners.skilljar.com/page/policies-certifications): 48-hour free cancellation and refund, retake discount, no retake after a pass, open-book online renewal assessment
    - [Anthropic Partner Academy: Computer and network setup](https://anthropic-partners.skilljar.com/page/computer-and-network-setup): online exams delivered through Pearson OnVUE
    - [Claude Certified Associate - Foundations Prep Course](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations): free prep path, its audience, recommended prior courses, module names and the four platform decisions
    - [Prep course module: Claude Platform & Model Foundations](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/claude-platform-model-foundations): objective to differentiate Haiku, Sonnet and Opus
    - [Anthropic Certification Exam Policy (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F34hhd92iyp94a0gtbr15cy5jk%2Fpublic%2F1782870704%2FAnthropic+Certification+Exam+Policy.pdf): ban on using AI products or services during the exam
    - [Certification Terms and Conditions (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F34hhd92iyp94a0gtbr15cy5jk%2Fpublic%2F1782870634%2FCertification+Terms+and+Conditions.pdf): certification is not a warranty of an individual's abilities
    - [Pearson VUE: Anthropic program page](https://www.pearsonvue.com/us/en/anthropic.html): exam listing name, "Practitioner" role name, 48-hour rescheduling for test center appointments
    - [Credly: Claude Certified Associate - Foundations badge](https://www.credly.com/org/anthropic/badge/claude-certified-associate-foundations): badge audience description
    - [Credly: Anthropic badge templates (JSON)](https://www.credly.com/organizations/anthropic/badges.json): badge level "Foundational"
    - [Claude blog: four role-based Claude certifications](https://claude.com/blog/four-role-based-claude-certifications): launch description of the Associate credential and the foundation-to-professional path
    - [Claude Certified Developer, Foundations Exam Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf): Developer audience, recommended Python and/or TypeScript proficiency, item count and time limit for comparison
    - [Claude Certified Architect, Foundations Exam Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): solution-architect audience, typical 6+ months of experience, scenario structure row, time limit for comparison
    - [Claude Certified Architect, Professional Exam Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf): mid- to senior-level audience, stakeholder responsibilities, recommended 3+ years in systems architecture or platform engineering, item count and time limit for comparison
    - [Claude 101: Claude desktop app, Chat, Cowork and Code](https://academy.claude.com/courses/claude-101/claude-desktop-app-chat-cowork-code): non-developers do not need the Code tab
    - [Anthropic Academy: Prototype AI-powered apps with Claude artifacts](https://academy.claude.com/tutorials/prototype-ai-powered-apps-with-claude-artifacts): artifacts are best for testing and demonstration; production needs API key management and infrastructure
    - [Claude models overview](https://platform.claude.com/docs/en/models/overview.md): current model lineup as of September 2026
    - [Claude Help Center: Use Google Workspace connectors](https://support.claude.com/en/articles/10166901-use-google-workspace-connectors): Gmail connector can send, reply and forward with approval
    - [Claude docs: Gmail connector](https://claude.com/docs/connectors/google/gmail.md): Gmail connector cannot create, send or modify emails
    - [Prep course module: Prompting & Task Execution](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/prompting-task-execution): Description as the backbone of Domain 1; the Q3 results example; the four learning objectives.
    - [Prep course module: Evaluating & Validating Claude's Output](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/evaluating-validating-claudes-output): the module goal and learning objectives for Domain 2.
    - [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices): the colleague test, explaining motivation, positive instructions, action phrasing, examples (3 to 5), when explicit chaining is useful, self-correction chains, research prompting.
    - [Reduce hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations): permission to admit uncertainty, quotes first for long documents, claim-by-claim verification, best-of-N comparison, restricting to provided documents, the limit of these techniques.
    - [Models overview](https://platform.claude.com/docs/en/models/overview): current model descriptions, relative latency, API list prices, reliable knowledge cutoffs, the Opus 5.5 starting-point advice.
    - [Choosing a model](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model): balancing capability, speed and cost; efficiency-first and capability-first paths; effort as a lever; cost per completed task.
    - [Pricing](https://platform.claude.com/docs/en/about-claude/pricing): Haiku for simple tasks, Sonnet for most production workloads, Opus for the most complex reasoning.
    - [Legal summarization use case guide](https://platform.claude.com/docs/en/about-claude/use-case-guides/legal-summarization): marking missing information as "Not specified"; meta-summarization.
    - [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents): prompt chaining trades latency for accuracy; separate calls per consideration; single optimized calls are often enough.
    - [Change the model, effort and thinking settings](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings): effort levels, thinking settings, changing models mid-conversation, the Thinking section, models where thinking cannot be turned off.
    - [When should I use web search, extended thinking, and Research](https://support.claude.com/en/articles/11095361-when-should-i-use-web-search-extended-thinking-and-research): the split between web search, extended thinking and Research.
    - [Use Research on Claude](https://support.claude.com/en/articles/11088861-use-research-on-claude): how Research works, plan availability, web search requirement, internal sources, usage.
    - [Enable and use web search](https://support.claude.com/en/articles/10684626-enable-and-use-web-search): citations, cross-referencing sources, no toggle in the new Claude experience.
    - [Use enterprise search](https://support.claude.com/en/articles/12489464-use-enterprise-search): the "Ask Your Org" project and how it differs from Research.
    - [Get started with custom connectors using remote MCP](https://support.claude.com/en/articles/11175166-get-started-with-custom-connectors-using-remote-mcp): connector tools running without further approval during Research; disabling write tools.
    - [Claude Cowork and chat are one Claude](https://support.claude.com/en/articles/16761823-claude-cowork-and-chat-are-one-claude): the new experience rollout, no switching back, Research via /deep-research, what a good brief contains.
    - [Cowork is now Claude (blog)](https://claude.com/blog/cowork-is-now-claude): the September 16, 2026 merge announcement; Team and Free plans to follow.
    - [What are Projects](https://support.claude.com/en/articles/9517075-what-are-projects): project knowledge, RAG on paid plans, Free plan limit, new version of Projects in beta.
    - [How can I create and manage projects](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects): instructions apply to all chats, no shared context across chats, project name and description hidden from Claude, Remove from project.
    - [What are artifacts and how do I use them](https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them): when Claude creates an artifact, requirements, usage, version selector, edits and Claude's memory, Claude Design, Slides and Docs.
    - [Publish and share artifacts](https://support.claude.com/en/articles/9547008-publish-and-share-artifacts): Claude account requirement; attachments shared with a chat-made artifact.
    - [Get started with Claude Docs](https://support.claude.com/en/articles/16923645-get-started-with-claude-docs): charts and diagrams do not update on their own; @Claude comments.
    - [Upload files to Claude](https://support.claude.com/en/articles/8241126-upload-files-to-claude): what Claude reads from PDFs by page count and from other documents.
    - [How do usage and length limits work](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work): usage versus length limits, context window sizes, automatic context management, starting a new conversation or using projects.
    - [Usage limit best practices](https://support.claude.com/en/articles/9797557-usage-limit-best-practices): grouping related questions, what uses limits faster.
    - [How large is the context window on paid Claude plans](https://support.claude.com/en/articles/8606394-how-large-is-the-context-window-on-paid-claude-plans): context window by model; 200K as about 500 pages.
    - [How up to date is Claude's training data](https://support.claude.com/en/articles/8114494-how-up-to-date-is-claude-s-training-data): training data dates for Opus 5.5, Sonnet 5 and Haiku 4.5.
    - [Use Claude's chat search and memory](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context): how memory saves topics, defaults by plan, project memory, what is never saved, legacy memory.
    - [Troubleshoot Claude error messages](https://support.claude.com/en/articles/12466728-troubleshoot-claude-error-messages): the length-limit error text and remedies.
    - [Understanding Claude's personalization features](https://support.claude.com/en/articles/10185728-understanding-claude-s-personalization-features): skills for customizing format and delivery.
    - [Claude is providing incorrect or misleading responses](https://support.claude.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on): hallucination, authoritative-looking quotes, reviewing cited sources, high-stakes advice.
    - [Claude is producing links that don't work and falsely claiming it has sent emails](https://support.claude.com/en/articles/8241188-claude-is-producing-links-that-don-t-work-and-falsely-claiming-that-it-has-sent-emails-or-produced-external-documents-what-s-going-on): capability hallucination.
    - [Introduction to prompt design](https://support.claude.com/en/articles/7996853-introduction-to-prompt-design): Claude as a newly-hired contractor with no context.
    - [My prompt isn't giving me a helpful answer](https://support.claude.com/en/articles/7996857-my-prompt-isn-t-giving-me-a-helpful-answer): breaking complex requests into substeps.
    - [Give Claude context: CLAUDE.md and better prompts](https://support.claude.com/en/articles/14553240-give-claude-context-claude-md-and-better-prompts): pasting full error text rather than summarizing.
    - [How does Claude handle mathematical equations and calculations](https://support.claude.com/en/articles/10366421-how-does-claude-handle-mathematical-equations-and-calculations): verifying complex or mission-critical calculations.
    - [Using Claude for legal work](https://support.claude.com/en/articles/15707726-using-claude-for-legal-work-privilege-confidentiality-and-how-to-think-about-configuration): a lawyer in the loop, primary sources, documenting AI use.
    - [Claude for Excel](https://claude.com/docs/office-agents/excel.md): final client deliverables and audit-critical calculations not recommended without review or verification.
    - [Claude 101: Your first conversation with Claude](https://academy.claude.com/courses/claude-101/your-first-conversation-with-claude): the three-part prompt frame; Sonnet and Opus guidance from the December 2025 video; Styles deprecation; changing models in the older video.
    - [Claude 101: Getting better results](https://academy.claude.com/courses/claude-101/getting-better-results): iteration, specific feedback, starting fresh, troubleshooting tone and length, the delegation diligence loop, validation and responsibility.
    - [Claude 101: Creating with artifacts](https://academy.claude.com/courses/claude-101/creating-with-artifacts): asking for the deliverable, artifact versus file creation, export formats, describing the end user.
    - [Claude 101: Introduction to projects](https://academy.claude.com/courses/claude-101/introduction-to-projects): files uploaded in a conversation stay out of project knowledge.
    - [Claude 101: Working with skills](https://academy.claude.com/courses/claude-101/working-with-skills): projects store knowledge, skills perform tasks.
    - [Introduction to Claude Cowork: What is Cowork](https://academy.claude.com/courses/introduction-to-claude-cowork/what-is-cowork): Chat for thinking, Cowork for delegating, Code for building software.
    - [Introduction to Claude Cowork: The task loop](https://academy.claude.com/courses/introduction-to-claude-cowork/the-task-loop): the three-question review, targeted feedback versus missing context.
    - [Introduction to Claude Cowork: Giving Cowork context](https://academy.claude.com/courses/introduction-to-claude-cowork/giving-cowork-context): repeated corrections as global-instruction candidates.
    - [Introduction to Claude Cowork: Validating skills for plugins](https://academy.claude.com/courses/introduction-to-claude-cowork/validating-skills-for-plugins): specific feedback; changing one thing at a time.
    - [AI Fluency: The 4D framework](https://academy.claude.com/courses/ai-fluency-framework-foundations/the-4d-framework): the definition of Discernment.
    - [AI Fluency: A closer look at Description](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-description): Product Description covers outputs, format, audience and style.
    - [AI Fluency: A closer look at Discernment](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-discernment): Product, Process and Performance Discernment; comparing three explanations.
    - [AI Fluency: A closer look at Diligence](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-diligence): Deployment Diligence.
    - [AI Fluency: The Description-Discernment loop](https://academy.claude.com/courses/ai-fluency-framework-foundations/the-description-discernment-loop): requesting iterations and making the final decisions.
    - [AI Capabilities and Limitations: Next token prediction](https://academy.claude.com/courses/ai-capabilities-and-limitations/next-token-prediction): fabrication concentrates in specific details.
    - [AI Capabilities and Limitations: Knowledge](https://academy.claude.com/courses/ai-capabilities-and-limitations/knowledge): knowledge frozen at the cutoff; characteristic knowledge failures including inherited bias.
    - [AI Capabilities and Limitations: Working memory](https://academy.claude.com/courses/ai-capabilities-and-limitations/working-memory): silent truncation, no learning from corrections, standing context.
    - [AI Capabilities and Limitations: Try it out (working memory)](https://academy.claude.com/courses/ai-capabilities-and-limitations/try-it-out-q7hdjm9twcbt): important instructions at the beginning and end; the 2023 Stanford finding.
    - [AI Capabilities and Limitations: Steerability](https://academy.claude.com/courses/ai-capabilities-and-limitations/steerability): goal versus format, restating the goal, checkpoints, reasoning drift.
    - [AI Capabilities and Limitations: How AI gets its character](https://academy.claude.com/courses/ai-capabilities-and-limitations/how-ai-gets-its-character): loose calibration between stated confidence and reliability; sycophancy.
    - [AI Capabilities and Limitations: When properties collide](https://academy.claude.com/courses/ai-capabilities-and-limitations/when-properties-collide): long-conversation drift.
    - [AI Fluency for Nonprofits: Workflow augmentation](https://academy.claude.com/courses/ai-fluency-for-nonprofits/workflow-augmentation): the three-way split of tasks; what a human should handle.
    - [AI Fluency for Nonprofits: Writing with AI](https://academy.claude.com/courses/ai-fluency-for-nonprofits/writing-with-ai): drafting inputs, draft-review checks, owning the final result.
    - [AI Fluency for Nonprofits: Researching with AI](https://academy.claude.com/courses/ai-fluency-for-nonprofits/researching-with-ai): research context and tracing a claim to its original source.
    - [AI Fluency for Small Businesses: Researching with AI](https://academy.claude.com/courses/ai-fluency-for-small-businesses/researching-with-ai): verifying regulations, pricing and deadlines against primary sources.
    - [AI Fluency for Small Businesses: AI capabilities and limits](https://academy.claude.com/courses/ai-fluency-for-small-businesses/ai-capabilities-and-limits): checking whether an audience shift actually landed.
    - [AI Fluency for Small Businesses: Tying it all together](https://academy.claude.com/courses/ai-fluency-for-small-businesses/tying-it-all-together): review of customer-facing outputs and a path to a human.
    - [Choosing the right Claude model](https://academy.claude.com/tutorials/choosing-the-right-claude-model): the tier guidance for Haiku, Sonnet, Opus and Fable, plan availability, checking in proportion to stakes, trying a task on two models.
    - [How to select the right effort setting](https://academy.claude.com/tutorials/how-to-select-the-right-effort-setting-for-claude-cowork-and-chat): signs of too little or too much effort; lower effort before older models; cost per task.
    - [Parametric memory and context](https://academy.claude.com/tutorials/parametric-memory-and-context): compaction versus written memory, the cost of long conversations, what to persist.
    - [The AI Fluency Index](https://academy.claude.com/tutorials/the-ai-fluency-index): iteration rates, setting interaction terms, polished outputs and reduced scrutiny.
    - [Why do AI models hallucinate](https://academy.claude.com/tutorials/why-do-ai-models-hallucinate): high-risk situations, permission not to know, fresh-chat review.
    - [Why does bias exist in AI models](https://academy.claude.com/tutorials/why-does-bias-exist-in-ai-models): forms and sources of bias; tips for balance.
    - [What is sycophancy in AI models](https://academy.claude.com/tutorials/what-is-sycophancy-in-ai-models): definition, triggers and countermeasures.
    - [Discernment toolkit](https://academy.claude.com/tutorials/discernment-toolkit): the source-grounding prompt, domain expertise as the main barrier, a panel of experts.
    - [Writing an AI diligence statement](https://academy.claude.com/tutorials/writing-an-ai-diligence-statement): consistency checks; executive and detailed versions.
    - [Can you trust what AI tells you](https://academy.claude.com/tutorials/can-you-trust-what-ai-tells-you): trust as a dial.
    - [Use artifacts to visualize and create AI apps](https://academy.claude.com/tutorials/use-artifacts-to-visualize-and-create-ai-apps-without-ever-writing-a-line-of-code): working one computed example by hand; forking a conversation.
    - [Use case: Explore what Claude can do for you](https://academy.claude.com/use-cases/explore-what-claude-can-do-for-you): letting Claude ask the questions; brainstorming prompts.
    - [Use case: Chart your data before you commit](https://academy.claude.com/use-cases/chart-your-data-before-you-commit): asking Claude to flag surprises; confounds; quizzing yourself.
    - [Use case: Generate project status reports](https://academy.claude.com/use-cases/generate-project-status-reports): noting missing information explicitly.
    - [Use case: PRD from a one-pager](https://academy.claude.com/use-cases/prd-from-a-one-pager): Claude fills gaps with reasonable guesses; flagging uncertainty.
    - [Use case: Adapt content across platforms](https://academy.claude.com/use-cases/adapt-content-across-platforms): plain-language rewrites; matching a voice from samples.
    - [Use case: Adapt a standard textbook page to every reading level](https://academy.claude.com/use-cases/adapt-a-standard-textbook-page-to-every-reading-level): asking for a list of what changed and what was left out.
    - [Use case: Verify statistics from raw data](https://academy.claude.com/use-cases/verify-statistics-from-raw-data): comparing stated and calculated figures.
    - [Use case: Plan your literature review](https://academy.claude.com/use-cases/plan-your-literature-review): checking central quotes against the original.
    - [Use case: Metrics narrative](https://academy.claude.com/use-cases/metrics-narrative): limiting charts to those that carry the argument.
    - [Prompt engineering best practices for 2026 (claude.com blog)](https://claude.com/blog/best-practices-for-prompt-engineering): starting with one example; not over-engineering; not using every technique at once.
    - [Anthropic research: Introspection](https://www.anthropic.com/research/introspection): introspective capability described as highly unreliable and limited in scope.
    - [Anthropic research: Reasoning models don't always say what they think](https://www.anthropic.com/research/reasoning-models-dont-say-think): reported chain of thought may not reflect the true reasoning.
    - [Anthropic consumer terms](https://www.anthropic.com/legal/consumer-terms): do not rely on outputs without independently confirming their accuracy.
    - [Claude 101: the Research lesson](https://academy.claude.com/courses/claude-101/research-mode-for-deep-dives): the current estimate of how long a Research run takes.
    - [AI Fluency: Effective prompting techniques](https://academy.claude.com/courses/ai-fluency-framework-foundations/effective-prompting-techniques): the six prompting techniques; asking the AI to help improve your prompt.
    - [AI Fluency for K-12 Educators: Creating high-quality AI outputs](https://academy.claude.com/courses/ai-fluency-for-k-12-educators/creating-high-quality-ai-outputs-in-your-teaching-practice): one concrete revision prompt, then comparing revision and original.
    - [Reduce latency](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-latency): paragraph or sentence counts work better than word counts.
    - [Claude Help Center: Release notes](https://support.claude.com/en/articles/12138966-release-notes): the July 10, 2026 memory change; model release dates.
    - [Claude Fable 5.1 overview](https://platform.claude.com/docs/en/models/fable-5-1/overview): Fable 5.1 release date.
    - [Claude Fable 5 overview](https://platform.claude.com/docs/en/models/fable-5/overview): Fable 5 release date, before the guide took effect.
    - [Claude Fable models on your plan](https://support.claude.com/en/articles/15424964-claude-fable-models-on-your-plan): Fable on paid plans only; how Fable usage counts by plan.
    - [Custom visuals in chat and Cowork](https://support.claude.com/en/articles/13979539-custom-visuals-in-chat-and-cowork): custom visuals in beta, ephemeral by default, how to keep one.
    - [Prep course module: Workflow Integration & Solution Design](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/workflow-integration-solution-design): Delegation as the anchoring competency; the module's D4.5 wording with "accurately"
    - [Prep course module: Configuration & Knowledge Management](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/configuration-knowledge-management): "a good configuration helps you every time"; maintenance cadence; "project-level instructions" wording
    - [Prep course module: Troubleshooting & Optimization](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/troubleshooting-optimization): "stale configuration" as a root cause (D5.4)
    - [Manage project visibility and sharing (Claude Help Center)](https://support.claude.com/en/articles/9519189-manage-project-visibility-and-sharing): public and private projects; chats stay private
    - [Control project sharing for your organization (Claude Help Center)](https://support.claude.com/en/articles/9927533-control-project-sharing-for-your-organization): Share projects and Public projects settings, defaults, role changes timing
    - [Retrieval-augmented generation (RAG) for projects (Claude Help Center)](https://support.claude.com/en/articles/11473015-retrieval-augmented-generation-rag-for-projects): automatic RAG up to 10x, indicator, file naming, naming documents in questions
    - [Create and edit files with Claude (Claude Help Center)](https://support.claude.com/en/articles/12111783-create-and-edit-files-with-claude): 30MB file-creation limit
    - [Set organization instructions (Claude Help Center)](https://support.claude.com/en/articles/14546867-set-organization-instructions): who can set them, 3,000-character limit, timing, precedence, safety limits, best practices and examples
    - [Use connectors to extend Claude's capabilities (Claude Help Center)](https://support.claude.com/en/articles/11176164-use-connectors-to-extend-claude-s-capabilities): inherited permissions, enabling versus authenticating, Enterprise-managed auth, action restrictions, per-conversation toggles, private projects only
    - [When to use desktop and web connectors (Claude Help Center)](https://support.claude.com/en/articles/11725091-when-to-use-desktop-and-web-connectors): remote connectors on all surfaces, desktop extensions local only
    - [Manage Claude's tool access (Claude Help Center)](https://support.claude.com/en/articles/13730515-manage-claude-s-tool-access): Auto as the default tool access mode
    - [Connect to Microsoft 365 (Claude Help Center)](https://support.claude.com/en/articles/15183774-connect-to-microsoft-365): services covered, plan availability, account and consent requirements
    - [Set up role-based permissions on Enterprise plans (Claude Help Center)](https://support.claude.com/en/articles/13930458-set-up-role-based-permissions-on-enterprise-plans): new custom roles default to Needs approval on every connector
    - [Share and unshare chats (Claude Help Center)](https://support.claude.com/en/articles/10593882-share-and-unshare-chats): attached files and raw MCP data excluded from shared snapshots
    - [What are skills? (Claude Help Center)](https://support.claude.com/en/articles/12512176-what-are-skills): skills definition, code execution requirement, provisioning, projects versus skills, custom instructions versus skills
    - [Use skills in Claude (Claude Help Center)](https://support.claude.com/en/articles/12512180-use-skills-in-claude): enabling code execution, Customize > Skills, ZIP upload, description troubleshooting, sharing versus publishing, trusted sources and risks
    - [Provision and manage skills for your organization (Claude Help Center)](https://support.claude.com/en/articles/13119606-provision-and-manage-skills-for-your-organization): organization provisioning, owner-only removal, test before provisioning, October 2, 2026 publishing and scanning defaults
    - [Get started with skill and plugin scanning (Claude Help Center)](https://support.claude.com/en/articles/15927065-get-started-with-skill-and-plugin-scanning): scanning off by default until October 2, 2026
    - [Use Claude Cowork on Team and Enterprise plans (Claude Help Center)](https://support.claude.com/en/articles/13455879-use-claude-cowork-on-team-and-enterprise-plans): plugin distribution options (Installed by default, Required, Not available)
    - [Set a default model for your organization (Claude Help Center)](https://support.claude.com/en/articles/15330088-set-a-default-model-for-your-organization): the self-updating recommended-default option
    - [Get started with Claude Design (Claude Help Center)](https://support.claude.com/en/articles/14604416-get-started-with-claude-design): availability, design loop, prompt elements, comments versus chat versus direct edits, no version history, presentations moved to Claude Slides
    - [Get started with Claude Cowork (Claude Help Center)](https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork): Cowork purpose, paid plans, /schedule, cloud scheduled tasks, Manual, Auto and Skip modes
    - [Schedule recurring tasks in Claude Cowork (Claude Help Center)](https://support.claude.com/en/articles/13854387-schedule-recurring-tasks-in-claude-cowork): scheduling cadences
    - [Use Claude Cowork safely (Claude Help Center)](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely): low-risk scheduled tasks, when to use Manual approval, user responsibility
    - [Organize your tasks with projects in Claude Cowork (Claude Help Center)](https://support.claude.com/en/articles/14116274-organize-your-tasks-with-projects-in-claude-cowork): Cowork projects and project-scoped memory
    - [What is Claude Tag? (Claude Help Center)](https://support.claude.com/en/articles/15594475-what-is-claude-tag): availability and billing in Slack
    - [Get started with Claude in Chrome (Claude Help Center)](https://support.claude.com/en/articles/12012173-get-started-with-claude-in-chrome): what the extension does, paid plans
    - [Use Claude in Chrome safely (Claude Help Center)](https://support.claude.com/en/articles/12902428-use-claude-in-chrome-safely): prompt injection as the biggest risk
    - [Is my data used for model training? (Anthropic Privacy Center)](https://privacy.claude.com/en/articles/10023580-is-my-data-used-for-model-training): raw connector content excluded from training data unless copied into the conversation
    - [Claude pricing](https://claude.com/pricing): Projects "Up to 5" on the Free plan
    - [Skills: how-to (claude.com docs)](https://claude.com/docs/skills/how-to.md): SKILL.md frontmatter, example skill, focused skills, no hardcoded secrets
    - [Skills: overview (claude.com docs)](https://claude.com/docs/skills/overview.md): progressive disclosure (metadata, then SKILL.md)
    - [Connectors: getting started (claude.com docs)](https://claude.com/docs/connectors/getting-started.md): Gmail search-only statement in the capability conflict
    - [Google Drive connector (claude.com docs)](https://claude.com/docs/connectors/google/drive.md): adding docs, private projects, live sync, permission-based access, file-type table, .docx conversion, plan availability
    - [Google Calendar connector (claude.com docs)](https://claude.com/docs/connectors/google/calendar.md): the "cannot create, modify, or delete calendar events" statement
    - [Claude for Microsoft 365 overview (claude.com docs)](https://claude.com/docs/office-agents/overview.md): the add-ins inside Microsoft 365 apps
    - [Claude for PowerPoint (claude.com docs)](https://claude.com/docs/office-agents/powerpoint.md): availability
    - [Claude for Word (claude.com docs)](https://claude.com/docs/office-agents/word.md): availability
    - [Claude for Outlook (claude.com docs)](https://claude.com/docs/office-agents/outlook.md): beta availability
    - [Capture as intent.md, The AI-native SDLC playbook (Claude Academy)](https://academy.claude.com/courses/ai-native-sdlc-playbook/capture-intent): requirements capture steps, example intent document, cost of handoffs
    - [Requirements and design, The AI-native SDLC playbook (Claude Academy)](https://academy.claude.com/courses/ai-native-sdlc-playbook/requirements-and-design): flagged concerns, single-session redesign, technical lead for higher-risk changes, lagging indicator
    - [A closer look at Delegation, AI Fluency: Framework and foundations (Claude Academy)](https://academy.claude.com/courses/ai-fluency-framework-foundations/a-closer-look-at-delegation): Problem, Platform and Task Delegation; not automating everything
    - [Project planning and Delegation, AI Fluency: Framework and foundations (Claude Academy)](https://academy.claude.com/courses/ai-fluency-framework-foundations/project-planning-and-delegation): planning with Claude and delegation decisions
    - [Integration, AI Fluency for nonprofits (Claude Academy)](https://academy.claude.com/courses/ai-fluency-for-nonprofits/integration): the "can we explain what the AI is doing?" test
    - [Understanding privacy and data, AI Fluency for nonprofits (Claude Academy)](https://academy.claude.com/courses/ai-fluency-for-nonprofits/understanding-privacy-and-data): remove identifying information the task does not need
    - [AI as a learning partner, AI Fluency for students (Claude Academy)](https://academy.claude.com/courses/ai-fluency-for-students/ai-as-a-learning-partner): testing a configured role for slippage
    - [Putting it all together, AI Fluency for creative work (Claude Academy)](https://academy.claude.com/courses/ai-fluency-for-creative-work/putting-it-all-together): policies without revision triggers go stale
    - [What a strong human-agent team looks like (Claude Academy)](https://academy.claude.com/courses/building-effective-human-agent-teams/what-a-strong-team-looks-like): autonomy in proportion to demonstrated reliability
    - [Some practical ways to get started, Building effective human-agent teams (Claude Academy)](https://academy.claude.com/courses/building-effective-human-agent-teams/practical-ways-to-get-started): human review at first, widening scope after good runs, writing down repeated explanations
    - [Connecting your tools, Claude 101 (Claude Academy)](https://academy.claude.com/courses/claude-101/connecting-your-tools): connectors see only what you can see
    - [Introduction to Claude Cowork: scheduled tasks lesson (Claude Academy)](https://academy.claude.com/courses/introduction-to-claude-cowork/scheduled-tasks): do a task once, then /schedule; the course's cadence list
    - [Share what you build with your team, Introduction to Claude Cowork (Claude Academy)](https://academy.claude.com/courses/introduction-to-claude-cowork/share-what-you-build-with-your-team): one owner, evals as the gate, quarterly review, naming, maintainers control updates
    - [Adoption signals, Deploying Claude Enterprise with confidence (Claude Academy)](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/adoption-signals): usage metrics versus outcomes; diagnostics are not quotas
    - [Using Claude Design for prototypes and UX (Claude Academy)](https://academy.claude.com/tutorials/using-claude-design-for-prototypes-and-ux): alternatives side by side, edge states before handoff, Claude Code implementation
    - [Using Claude Design for presentations and slide decks (Claude Academy)](https://academy.claude.com/tutorials/using-claude-design-for-presentations-and-slide-decks): the older tutorial side of the presentations freshness note
    - [Scaling workflows with Claude Cowork at your organization (Claude Academy)](https://academy.claude.com/tutorials/scaling-workflows-with-claude-cowork-at-your-organization): moving the human and AI boundary, accountability, chat trap, outcome measures, treating config as permanent, curation, ownership
    - [Building AI-fluent organizations (Claude Academy)](https://academy.claude.com/tutorials/building-ai-fluent-organizations): retesting old assumptions
    - [Analyze patterns in user feedback (Claude Academy use case)](https://academy.claude.com/use-cases/analyze-patterns-in-user-feedback): same underlying need, sample representativeness
    - [Launch readiness sweep (Claude Academy use case)](https://academy.claude.com/use-cases/launch-readiness): owners per item, readiness call, past retros, checklist without owners
    - [Thread to decision doc (Claude Academy use case)](https://academy.claude.com/use-cases/thread-to-decision): owner and committed date per next step
    - [Turn inspiration into design plans (Claude Academy use case)](https://academy.claude.com/use-cases/turn-inspiration-to-design-plans): sequencing and delay impact
    - [Workflow improvement planner (Claude Academy use case)](https://academy.claude.com/use-cases/workflow-improvement-planner): intake, specificity, honesty about what is broken, constraints, improvements beyond speed
    - [Create a process flowchart (Claude Academy use case)](https://academy.claude.com/use-cases/create-a-process-flowchart): decision points from messy documentation, keeping maps current in a Project
    - [Compare and analyze competing options (Claude Academy use case)](https://academy.claude.com/use-cases/compare-and-analyze-competing-options): stakeholder memo with trade-offs
    - [Understand and extend an inherited spreadsheet (Claude Academy use case)](https://academy.claude.com/use-cases/understand-and-extend-an-inherited-spreadsheet): change-summary prompt
    - [Generate an AI policy (Claude Academy use case)](https://academy.claude.com/use-cases/generate-an-ai-policy): revision triggers between annual reviews
    - [Estimating AI productivity gains (Anthropic research)](https://www.anthropic.com/research/estimating-productivity-gains): sample size, 90-minute and 80% estimates, 1.4-hour figure, limitations, bottleneck shift, not a prediction
    - [Anthropic Economic Index report (Anthropic research)](https://www.anthropic.com/research/economic-index-june-2026-report): survey productivity percentages and non-representativeness
    - [How to create custom skills (Claude Help Center)](https://support.claude.com/en/articles/12512198-how-to-create-custom-skills): the help center's skill metadata rules, which differ from the claude.com docs
    - [Available beta and research preview features (Claude Help Center)](https://support.claude.com/en/articles/14503520-available-beta-and-research-preview-features): beta status of the Claude for Microsoft 365 add-ins
    - [Governance, Risk & Responsible Use (prep course module)](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/governance-risk-responsible-use): the module summary, its description of the three judgments, and the "Skill with too much access" example
    - [Anthropic Usage Policy](https://www.anthropic.com/legal/aup): who the policy applies to, its three parts, the universal standards cited, the seven high-risk areas, the human-in-the-loop and disclosure requirements, and chatbot disclosure
    - [Anthropic Commercial Terms of Service](https://www.anthropic.com/legal/commercial-terms): customer ownership of outputs, no training on customer content, the DPA, compliance with the Usage Policy, and the customer's duty to evaluate outputs
    - [Anthropic Privacy Policy](https://www.anthropic.com/legal/privacy): consumer training on inputs and outputs unless the user opts out
    - [Updates to Consumer Terms and Privacy Policy (Anthropic news)](https://www.anthropic.com/news/updates-to-our-consumer-terms): which plans the consumer training choice covers, and 5-year versus 30-day retention
    - [Is my data used for model training? (commercial products, privacy center)](https://privacy.claude.com/en/articles/7996868-is-my-data-used-for-model-training): no training by default on commercial products; feedback stores the conversation for up to 5 years; the Rate chats setting
    - [How do I change my model improvement privacy settings? (privacy center)](https://privacy.claude.com/en/articles/12109829-how-do-i-change-my-model-improvement-privacy-settings): the Help Improve our AI models toggle
    - [How long do you store my data? (privacy center)](https://privacy.claude.com/en/articles/10023548-how-long-do-you-store-my-data): deletion from back-end storage within 30 days and de-identified retention for up to 5 years
    - [Does Anthropic act as a data processor or controller? (privacy center)](https://privacy.claude.com/en/articles/9267385-does-anthropic-act-as-a-data-processor-or-controller): controller and processor roles for Claude for Work
    - [Who owns and manages the data of my team? (help center)](https://support.claude.com/en/articles/9265372-who-owns-and-manages-the-data-of-my-team): consumer terms and privacy policy do not apply to Claude for Work
    - [Sensitive data in chats: who can view my conversations? (help center)](https://support.claude.com/en/articles/8325621-i-would-like-to-input-sensitive-data-into-my-chats-with-claude-who-can-view-my-conversations): the list of sensitive details to be thoughtful about sharing
    - [Use incognito chats (help center)](https://support.claude.com/en/articles/12260368-use-incognito-chats): incognito behavior, 30-day retention, inclusion in exports and the Compliance API, no incognito inside projects
    - [Public links for shared chats (help center)](https://support.claude.com/en/articles/16762437-public-links-for-shared-chats): treat a public link as public; Team and Enterprise share inside the organization only
    - [HIPAA-ready Enterprise plans (help center)](https://support.claude.com/en/articles/13296973-hipaa-ready-enterprise-plans): HIPAA readiness is Enterprise-only; Cowork is not yet under the BAA
    - [Claude in Chrome permissions guide (help center)](https://support.claude.com/en/articles/12902446-claude-in-chrome-permissions-guide): allowlists and blocklists
    - [Claude in Chrome admin controls (help center)](https://support.claude.com/en/articles/13065128-claude-in-chrome-admin-controls): start with a restrictive allowlist
    - [Manage custom roles on Enterprise plans (help center)](https://support.claude.com/en/articles/13930452-manage-custom-roles-on-enterprise-plans): the most restrictive level wins
    - [Manage model access for your organization (help center)](https://support.claude.com/en/articles/15694740-manage-model-access-for-your-organization): model and effort caps; Haiku models always available
    - [Access audit logs (help center)](https://support.claude.com/en/articles/9970975-access-audit-logs): Enterprise-only audit logs, 180-day export, identifiers without chat content
    - [Configure custom data retention controls for Enterprise plans (help center)](https://support.claude.com/en/articles/10440198-configure-custom-data-retention-controls-for-enterprise-plans): 30-day minimum and indefinite default
    - [Access the Compliance API (help center)](https://support.claude.com/en/articles/13015708-access-the-compliance-api): what the Compliance API pulls
    - [Get started with Claude (help center)](https://support.claude.com/en/articles/8114491-get-started-with-claude): five-hour session limit on the Free plan
    - [What is the Pro plan? (help center)](https://support.claude.com/en/articles/8325606-what-is-the-pro-plan): five-hour reset and weekly limit on Pro
    - [What is the Team plan? (help center)](https://support.claude.com/en/articles/9266767-what-is-the-team-plan): weekly limit on Team seats
    - [Why Claude switched models in your conversation with Opus 5 or Opus 5.5 (help center)](https://support.claude.com/en/articles/16049681-why-claude-switched-models-in-your-conversation-with-opus-5-or-opus-5-5): fallback behavior and the visible notice
    - [Why am I receiving an Output blocked by content filtering policy error? (help center)](https://support.claude.com/en/articles/10023638-why-am-i-receiving-an-output-blocked-by-content-filtering-policy-error): the cause of the content filtering error
    - [How Claude marks AI-generated content (help center)](https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content): EU AI Act Article 50(2) Code of Practice, watermarks and C2PA, a missing mark proves nothing
    - [Choosing a model (Claude Platform docs)](https://platform.claude.com/docs/en/about-claude/models/choosing-a-model.md): tuning effort as a better lever than switching models
    - [Claude's constitution](https://www.anthropic.com/constitution): value priorities, unhelpfulness never trivially safe, transparent conscientious objector, calibration, political reticence, acceptable reliance
    - [Evaluating and mitigating discrimination in language model decisions (Anthropic research)](https://www.anthropic.com/research/evaluating-and-mitigating-discrimination-in-language-model-decisions): discrimination findings, prompt-engineering mitigation, no automated decisions for the high-risk uses studied
    - [AI Fluency: Additional activities (Anthropic Academy)](https://academy.claude.com/courses/ai-fluency-framework-foundations/additional-activities): reusable prompt templates with placeholders
    - [AI Fluency for Small Businesses: Using data with AI (Anthropic Academy)](https://academy.claude.com/courses/ai-fluency-for-small-businesses/using-data-with-ai): marking sensitive data, pseudonymizing a copy, matching the tool to sensitivity
    - [AI Fluency for K-12 Educators: Ethics and responsible use (Anthropic Academy)](https://academy.claude.com/courses/ai-fluency-for-k-12-educators/ethics-responsible-use): FERPA example and the list of ethical tensions, including bias
    - [AI Fluency for Creative Work: Delegation and diligence (Anthropic Academy)](https://academy.claude.com/courses/ai-fluency-for-creative-work/delegation-and-diligence): convergence and skill atrophy
    - [My voice (Anthropic Academy use case)](https://academy.claude.com/use-cases/my-voice): feeding corrections back into a voice skill
    - [Deploying AI from pilot to production (Anthropic and Accenture, PDF)](https://cdn.prod.website-files.com/6889473510b50328dbb70ae6/6aa46440db7c5ad962ef7ef2_Claude-Accenture-Deploying-AI-from-pilot-to-production-09112026%20%281%29.pdf): classifying data by sensitivity level before a pilot
    - [Why Claude switched models in your conversation with Fable 5 or Fable 5.1 (Claude Help Center)](https://support.claude.com/en/articles/15363606-why-claude-switched-models-in-your-conversation-with-fable-5-or-fable-5-1): requests that fall back from Fable to an Opus model
    - [Anthropic Partner Academy: CCAO-F certification page](https://anthropic-partners.skilljar.com/claude-certified-associate-foundations-certification): exam purchase page and price
    - [Anthropic Partner Academy home](https://anthropic-partners.skilljar.com/): validation required at login
    - [Claude Certification Program: Exam Registration Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542947%2FClaude+Certification+Program+-+Exam+Registration+Guide.pdf): same registration process for every exam
    - [Pearson VUE: OnVUE for Anthropic](https://www.pearsonvue.com/us/en/anthropic/onvue.html): system requirements, ID rules, testing space, check-in steps, in-exam conduct, digital whiteboard
    - [Pearson VUE: Accommodations for Anthropic](https://www.pearsonvue.com/us/en/test-takers/accommodations/pearson_approve.anthropic.html): 10 business days for review, no accommodations added to a scheduled exam
    - [Pearson VUE: OnVUE whiteboard](https://www.pearsonvue.com/us/en/onvue/whiteboard.html): whiteboard wiped when the connection drops
    - [Pearson VUE: Candidate Rules Agreement (PDF)](https://www.pearsonvue.com/content/dam/VUE/vue/global/documents/candidate-rules/candidate-rules-agreement.pdf): test-center personal items, note board, reporting a question number
    - [Pearson VUE: Test center check-in process (PDF)](https://www.pearsonvue.com/content/dam/VUE/vue/en/documents/pearson-professional-center-exam-check-in-process.pdf): arrival time given in the confirmation email
    - [Claude Academy FAQ](https://academy.claude.com/help/faq): free access, durations as estimates, course badges versus certification
    - [Claude Academy: Claude 101](https://academy.claude.com/courses/claude-101): size, lessons, plan prerequisite, iteration and troubleshooting guidance
    - [Claude Academy: AI Fluency: Framework and foundations](https://academy.claude.com/courses/ai-fluency-framework-foundations): size, 4D framework, Description-Discernment loop
    - [Claude Academy: AI capabilities and limitations](https://academy.claude.com/courses/ai-capabilities-and-limitations): size, four properties, context window, blank-slate probe, checkpoint exercise, staleness probe
    - [Claude Academy: Introduction to Claude Cowork](https://academy.claude.com/courses/introduction-to-claude-cowork): size, paid-plan prerequisite, change one thing at a time
    - [Claude Academy: All resources](https://academy.claude.com/all): course sizes and summaries
    - [anthropics/prompt-eng-interactive-tutorial](https://github.com/anthropics/prompt-eng-interactive-tutorial): chapters, model used, last commit
    - [Claude docs: Model deprecations](https://platform.claude.com/docs/en/about-claude/model-deprecations.md): retirement of claude-3-haiku-20240307 and its replacement
    - [Claude docs: Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices.md): prefill no longer supported on recent models
    - [anthropics/skills](https://github.com/anthropics/skills): public Agent Skills repository
    - [Reddit: OkRelationship3427 on passing CCAO-F](https://www.reddit.com/r/ClaudeAI/comments/1uv1wi2/passed_the_ccaof_today/): pass close to the line, difficulty, samples versus live items, sat before the prep course
    - [Matthew Purcell: review of all four Claude certification exams (LinkedIn)](https://www.linkedin.com/pulse/claude-certification-exams-honest-review-matthew-purcell-byo2e): 967, preparation, timing, item style, score-report weak area, own practice exam
    - [Reddit: Interesting_Ebb_6383 on passing all four exams, with comments by OkFan7308](https://www.reddit.com/r/ClaudeAI/comments/1v2x1p7/got_all_4_claude_certifications_ccap_ccaf_ccdvf/): no preparation, "not entry level", practice site; OkFan7308 chose CCAO-F
    - [Reddit: CCAR-F thread with OkFan7308's CCAO-F comments](https://www.reddit.com/r/ClaudeAI/comments/1vaxsws/claude_certified_architect_foundations_ccarf_how/): 851, full two hours, prep course, elimination habit
    - [Reddit: Hefty_Ad_6638 on failing CCAO-F](https://www.reddit.com/r/claudeskills/comments/1vuxasi/failed_ccaof_am_i_the_only_one/): fail, mocks unlike the exam, time
    - [Reddit: CCAO-F pass thread with comments by royalkaku and Hot_Entrepreneur671](https://www.reddit.com/r/claudeskills/comments/1vssz2i/passed_my_claude_certified_associate_foundations/): royalkaku's 934 and item style, Hot_Entrepreneur671's comparison with mocks
    - [dev.to: bluepanda on clearing all four Claude certifications](https://dev.to/bluepanda/i-cleared-all-4-anthropic-claude-certifications-what-you-actually-need-to-know-and-what-to-skip-224m): difficulty order, time, check-in, interface
    - [Reddit: alberto3333 on passing CCDV-F and CCAO-F](https://www.reddit.com/r/claudeskills/comments/1wa1rf5/passed_ccdvf_and_ccaof_i_spent_two_weeks_studying/): 835, two weeks, disclosure, not describing questions
    - [Reddit: Broad-Service-7733 asking about CCAO-F](https://www.reddit.com/r/claudeskills/comments/1w558ch/ccaof/): starting level before the exam
    - [Reddit: Broad-Service-7733 on passing CCAO-F](https://www.reddit.com/r/claudeskills/comments/1we020n/claude_associate_foundation_certification_ccaof/): the pass
    - [Reddit: imstillwhite on passing CCAO-F](https://www.reddit.com/r/ClaudeAI/comments/1wjnvrw/i_passed_the_ccaof_claude_associate_foundation/): 917, preparation ranking, mocks, item style, OnVUE experience, extra-time claim
    - [Reddit: OkRelationship3427, CCAR-F 904](https://www.reddit.com/r/ClaudeAI/comments/1v5zrru/passed_the_ccarf_with_9041000/), [CCDV-F 941](https://www.reddit.com/r/ClaudeAI/comments/1v6sjc5/completed_the_claude_foundations_trilogy_ccaof/) and [CCAR-P 840](https://www.reddit.com/r/ClaudeAI/comments/1ve3x4u/passed_claude_certified_architect_professional/): the same candidate's later results
    - [Ravikiran Srinivasulu: CCAO-F study guide](https://ravikirans.com/claude-certified-associate-foundations-study-guide/): free guide mapping domains to official docs, date
    - [Tutorials Dojo: CCAO-F study guide](https://tutorialsdojo.com/ccao-f-claude-certified-associate-foundations-study-guide/): free guide whose samples cite help-center articles
    - [CertSafari: CCAO-F practice questions](https://www.certsafari.com/anthropic/claude-certified-associate-foundations): bank size and creation date
    - [CertSafari: CCAR-F practice questions](https://www.certsafari.com/anthropic/claude-certified-architect-foundations): formats offered, one-person project
    - [CertSafari: Disclaimer](https://www.certsafari.com/disclaimer): not affiliated with Anthropic
    - [Udemy: Claude Certified Associate Foundations (CCAO-F) Exam Guide](https://www.udemy.com/course/claude-certified-associate-foundations-ccao-f-exam-guide/): length, quiz questions, no pass guarantee
    - [Udemy: Claude Certified Associate Foundations CCAO-F Practice Tests](https://www.udemy.com/course/claude-associate/): mapping to the Version 1.0 guide, students shown
    - [YouTube: Program Strategy HQ, CCAO-F study guide](https://www.youtube.com/watch?v=lcjm0TvsKH0): length, link to the prep path
    - [GitHub: Amey-Thakur/CLAUDE-CERTIFICATIONS](https://github.com/Amey-Thakur/CLAUDE-CERTIFICATIONS): contents, last push, claim about Claude Academy
    - [GitHub: dnacenta/claude-certified-architect](https://github.com/dnacenta/claude-certified-architect): overview pages for the other exams, refresh date
    - [Claude Academy: course catalog](https://academy.claude.com/courses): the four recommended courses are free on Claude Academy
    - [Claude Academy: AI Fluency for small businesses](https://academy.claude.com/courses/ai-fluency-for-small-businesses): the course behind the week 4 data-handling exercise
    - [Prep course module: Course Summary & Next Steps](https://anthropic-partners.skilljar.com/path/claude-certified-associate-foundations/course-summary-next-steps): the closing module and its brief exam overview
