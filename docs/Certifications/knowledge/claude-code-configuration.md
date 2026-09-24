---
title: "Claude Code Configuration for the Claude Certifications"
description: How Claude Code loads CLAUDE.md, rules, settings, permissions, commands, skills, subagents, plugins and MCP servers, and which layer wins, for the Claude exams.
last_reviewed: 2026-09-23
---

# Claude Code Configuration

Claude Code takes its behavior from layered files. Some layers are context that Claude reads and tries to follow (CLAUDE.md, rules, skills, commands); others are applied by the Claude Code client whatever Claude decides (settings, permission rules, hooks, managed settings). Each layer is taught once (hooks in [Hooks](claude-code-workflows.md#hooks), everything else below) with its file locations, precedence and the decisions the CCAR-F, CCDV-F and CCAR-P guides test; product details are as of September 2026, and where the docs have moved on from the July 2026 exam guides, both versions are shown with the wording to expect on the exam.

## The configuration layers at a glance

*Tested in: CCAR-F 3.1, 3.2 (3.2-S5), 3.3 (3.3-S3), 2.4-K1, APPX-TECH-3 · CCDV-F D2.6 Configuration Management, D3.1 Claude Code Operation, D8.3 Agentic Customization · CCAR-P 3.8, 7.1*

The Claude Code objectives keep asking one question in different forms: which layer should hold this instruction or control? CCAR-F asks you to choose between skills and CLAUDE.md, and between path-specific rules and subdirectory CLAUDE.md files; CCDV-F asks for the tradeoffs among built-in tools, custom tools, Skills and MCPs. Start every such question with one split. CLAUDE.md, rules, skills and command files are context: Claude reads them and usually follows them, but nothing forces it to. Settings, permission rules and hooks are applied by the Claude Code client itself, so they hold whatever Claude decides.

### Which layer for which need

| You need | Layer | Team copy (commit it) | Personal copy | Loads or runs | Kind |
|---|---|---|---|---|---|
| Build commands, conventions and "always do X" rules in every session | CLAUDE.md | `./CLAUDE.md` or `./.claude/CLAUDE.md` | `~/.claude/CLAUDE.md`, or `./CLAUDE.local.md` (gitignored) for one project | Full content at session start; costs context on every request | Guidance |
| Conventions for one file type or one area of the code | Rules | `.claude/rules/*.md` | `~/.claude/rules/*.md` | At launch, or only when Claude reads a file matching the rule's `paths` | Guidance |
| Default model, environment variables, hooks, plugins and other client behavior | Settings | `.claude/settings.json` | `~/.claude/settings.json`, or `.claude/settings.local.json` for one project | Read at start; most edits apply to the running session | Enforced by the client |
| Allow, ask about or block specific tool calls | Permission rules (the `permissions` key in settings) | `.claude/settings.json` | `~/.claude/settings.json` or `.claude/settings.local.json` | Checked on each tool call | Enforced by the client |
| A saved prompt you start by name | Custom command (the older format) | `.claude/commands/<name>.md` | `~/.claude/commands/<name>.md` | When you type `/<name>` | Guidance |
| A procedure or reference material needed only for some tasks | Skill | `.claude/skills/<name>/SKILL.md` | `~/.claude/skills/<name>/SKILL.md` | Description on every request; full body only when invoked | Guidance |
| A side task whose output would flood the conversation | Subagent | `.claude/agents/*.md` | `~/.claude/agents/*.md` | When spawned, in its own context window; a summary comes back | Isolated context |
| Something that must happen every time | Hook | `hooks` in `.claude/settings.json` | `hooks` in `~/.claude/settings.json` | On its lifecycle event, outside the model | Deterministic |
| Tools and data from an external system | MCP server | `.mcp.json` at the project root | `~/.claude.json` (local and user scopes) | Tool names and server instructions at session start, full schemas only when needed (tool search is on by default) | Capability |
| The same setup in a second repository | Plugin | `enabledPlugins` in `.claude/settings.json` | `enabledPlugins` in `~/.claude/settings.json` | Installs a bundle of skills, agents, hooks and MCP servers | Packaging |

Two more layers sit on top of these. Output styles (`.claude/output-styles/*.md`, or `~/.claude/output-styles` for you) set the role, tone and format of every response; they are covered in [Models and output styles](#models-and-output-styles). Managed settings sit above all of these and let an organization enforce policy on every machine it deploys them to; see [Managed settings for organizations](#managed-settings-for-organizations).

### The trigger for each layer

The Claude Code features overview pairs each layer with the moment it becomes worth adding. These make good decision rules for scenario questions:

- Claude gets a convention or command wrong twice: add it to CLAUDE.md.
- You keep asking Claude to be shorter, explain more, or answer in the same format: set an output style.
- You keep typing the same prompt to start a task: save it as a user-invocable skill (or, in the exam guide's terms, a custom slash command).
- You paste the same playbook or multi-step procedure for the third time: capture it as a skill.
- You keep copying data from a browser tab Claude can't see: connect that system as an MCP server.
- A side task floods the conversation with output you won't reference again: route it through a subagent.
- Something must happen every time without asking: write a hook.
- A second repository needs the same setup: package it as a plugin.

### How the layers combine when two levels define the same thing

| Layer | When several levels define it |
|---|---|
| CLAUDE.md files and rules | Additive: every level loads and contributes. Nothing overrides; project content appears after user content, and conflicting lines leave Claude to choose |
| Settings keys | The highest level that sets the key wins: managed, then command line, then `.claude/settings.local.json`, then `.claude/settings.json`, then `~/.claude/settings.json` |
| List keys such as `permissions.allow` | Merged across files, so each file adds entries (four keys that hold model lists or per-model entries follow their own rules: `fallbackModel`, `modelPicker`, `availableModels` and `modelSettings`) |
| Permission rules | Merged; a deny at any level cannot be allowed at another |
| Skills | By name: enterprise over personal, personal over project; plugin skills are namespaced and never collide |
| A skill and a command file with the same name | The skill wins |
| Subagents | By name: managed settings, then the `--agents` flag, then `.claude/agents/`, then `~/.claude/agents/`, then plugins |
| MCP servers | By name: local, then project, then user |
| Hooks | Merged: every registered hook fires for its matching events, whatever file it came from; managed hooks can't be removed |

### Guidance or guarantee

A rule worth memorizing comes from Anthropic's [Debug your configuration](https://code.claude.com/docs/en/debug-your-config) guide:

> Use CLAUDE.md for "we do it this way here." Use permissions or hooks for security boundaries and anything that must never happen, where you need a guarantee instead of guidance.

The [features overview](https://code.claude.com/docs/en/features-overview) gives the standard example:

> An instruction like "never edit `.env`" in CLAUDE.md or a skill is a request, not a guarantee. A `PreToolUse` hook that blocks the edit is enforcement.

Decision rule: if a behavior should usually happen, a CLAUDE.md line, rule or skill is proportionate. If it must always happen or must never happen, put it where the client enforces it: a permission rule, a hook, or managed settings for an organization. The CCAR-F guide draws the same line for agents, contrasting hooks' deterministic guarantees with the probabilistic compliance of prompt instructions; hooks themselves are taught in [Hooks](claude-code-workflows.md#hooks).

### Where the files live

```text
your-repo/
├── CLAUDE.md                      # team instructions (or .claude/CLAUDE.md)
├── CLAUDE.local.md                # your notes for this repo; gitignore it
├── .mcp.json                      # project-scoped MCP servers
└── .claude/
    ├── settings.json              # team settings: permissions, hooks, env, plugins
    ├── settings.local.json        # your overrides for this repo; keep it out of git
    ├── rules/*.md                 # topic rules, optional paths: frontmatter
    ├── skills/<name>/SKILL.md     # skills, run as /<name>
    ├── commands/*.md              # older single-file commands, run as /<name>
    ├── agents/*.md                # subagents
    └── output-styles/*.md         # output styles

~/.claude/                         # yours, applies to every project
├── CLAUDE.md
├── settings.json
└── rules/  skills/  commands/  agents/  output-styles/

~/.claude.json                     # app state: sign-in, MCP servers (local and
                                   # user scopes), per-project trust decisions
```

### Things that look like configuration but are not

These are the misplacements the docs and the official rationales call out. Each is a plausible wrong answer; each link goes to the correct layer.

- **`.claude/config.json` with a commands array.** A mechanism that doesn't exist in Claude Code; Anthropic's rationale for CCAR-F sample Question 4 rejects it (see [Slash commands](#slash-commands)).
- **Commands defined in CLAUDE.md.** CLAUDE.md holds project instructions and context, not command definitions; commands live in `.claude/commands/` or `.claude/skills/` (see [Custom commands](#custom-commands)).
- **`permissions`, `hooks` or `env` in `~/.claude.json`.** That file holds app state and UI toggles; those keys belong in `~/.claude/settings.json` (see [The files](#the-files)).
- **An `mcpServers` key in `settings.json`.** Settings files don't read it; project MCP servers go in `.mcp.json` at the repository root (see [Details that decide answers](#details-that-decide-answers)).
- **A standalone hooks file for a project.** There is none; hooks go under `"hooks"` in a settings file. Only plugins use `hooks/hooks.json` (see [Hooks](claude-code-workflows.md#hooks) and [Plugin layout](#plugin-layout)).
- **A skill saved as `.claude/skills/name.md`.** A skill is a folder with a `SKILL.md` inside: `.claude/skills/name/SKILL.md` (see [Where skills live](#where-skills-live)).
- **A parent directory's `.claude/settings.json` applying in a subdirectory.** Project settings are not inherited from parent directories the way CLAUDE.md files are (see [What a committed `.claude/settings.json` can and cannot do](#what-a-committed-claudesettingsjson-can-and-cannot-do)).

## CLAUDE.md and the memory hierarchy

*Tested in: CCAR-F 3.1 (3.1-K1 to 3.1-K4, 3.1-S1 to 3.1-S4), 3.6-K3, 3.6-S5, Exercise 2 step 1, APPX-INSCOPE-9 · CCDV-F D3.1 Claude Code Operation (CLAUDE.md hierarchy, repository initialization), D2.6 Configuration Management (CLAUDE.md files) · CCAR-P 3.8, 7.1*

Every Claude Code session starts with a fresh context window. CLAUDE.md files carry project knowledge from one session to the next: Markdown instructions that Claude Code loads at the start of every conversation. The [memory docs](https://code.claude.com/docs/en/memory) put the purpose in one line: "Treat CLAUDE.md as the place you write down what you'd otherwise re-explain."

### What CLAUDE.md is, and what it is not

- **Context, not configuration.** Claude treats CLAUDE.md as context, not enforced configuration. To block an action whatever Claude decides, use a permission rule or a `PreToolUse` hook.
- **A user message, not the system prompt.** CLAUDE.md content is delivered as a user message after the system prompt, so strict compliance is not guaranteed. For instructions at system-prompt level, launch with `--append-system-prompt`, which the docs describe as better suited to scripts and automation.
- **Also read by the auto mode classifier.** The classifier reads the same CLAUDE.md content Claude loads, so an instruction such as "never force push" in the project CLAUDE.md steers both Claude and the classifier.
- **Loaded whole, up to a size cap.** Claude Code loads a CLAUDE.md of up to 4 MiB in full and skips a larger file. The 200-line or 25KB limit you may see quoted applies only to auto memory's `MEMORY.md`. Within the cap, shorter files still produce better adherence.
- **Maintainer comments are free.** Block-level HTML comments in CLAUDE.md are stripped before the content reaches Claude (comments inside code blocks are kept), so they cost no context.

### The locations

The docs list the locations in load order, broadest first, "so a project instruction appears in context after a user instruction" ([memory docs](https://code.claude.com/docs/en/memory)).

| Scope | Location | Shared with | Use it for |
|---|---|---|---|
| Managed policy | macOS `/Library/Application Support/ClaudeCode/CLAUDE.md`; Linux and WSL `/etc/claude-code/CLAUDE.md`; Windows `C:\Program Files\ClaudeCode\CLAUDE.md` | Every user on the machine; individual settings cannot exclude it | Organization-wide instructions managed by IT or DevOps |
| User | `~/.claude/CLAUDE.md` | Just you, in every project | Personal preferences |
| Project | `./CLAUDE.md` or `./.claude/CLAUDE.md` | The team, through source control | Build commands, conventions, architecture |
| Local | `./CLAUDE.local.md`, added to `.gitignore` | Just you, in this project | Personal project notes such as sandbox URLs or preferred test data |

!!! warning "Exam guide vs current docs"

    CCAR-F Task 3.1 names three levels: "user-level (~/.claude/CLAUDE.md), project-level (.claude/CLAUDE.md or root CLAUDE.md), and directory-level (subdirectory CLAUDE.md files)" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). As of September 2026 the docs also describe a managed policy CLAUDE.md, `CLAUDE.local.md`, the managed `claudeMd` setting, user-level rules in `~/.claude/rules/` and, from v2.1.277, `AGENTS.md`. None of these contradicts the guide's three levels. Expect the three-level model on the exam and treat the rest as detail.

### How the files are found and combined

1. **Walk up at launch.** Claude Code loads `CLAUDE.md` and `CLAUDE.local.md` from the working directory and every directory above it. Started in `foo/bar/`, it loads `foo/bar/CLAUDE.md` and `foo/CLAUDE.md`.
2. **Concatenate, never override.** All discovered files go into context together. They are ordered from the filesystem root down to the working directory, so the file nearest the launch directory is read last. Within one directory, `CLAUDE.local.md` is appended after `CLAUDE.md`.
3. **Load subdirectories lazily.** A CLAUDE.md in a subdirectory below the working directory is not loaded at launch. It loads when Claude reads a file in that subdirectory with the Read tool, and not when Claude writes or creates files there.
4. **Skip added directories unless asked.** Directories added with `--add-dir` don't contribute their CLAUDE.md files by default.

To load instruction files from an added directory, set the environment variable at launch. This loads `CLAUDE.md`, `.claude/CLAUDE.md`, `.claude/rules/*.md` and `CLAUDE.local.md` from it:

```bash
CLAUDE_CODE_ADDITIONAL_DIRECTORIES_CLAUDE_MD=1 claude --add-dir ../shared-config
```

Three consequences show up in exam scenarios:

- **The closest file does not win.** Because files are concatenated, two contradicting instructions both reach Claude, and Claude may pick one arbitrarily. Resolve conflicts by editing the files, not by relying on order.
- **Where you start matters in a monorepo.** Starting at the repository root loads only the root CLAUDE.md at launch; starting in a package loads that package's CLAUDE.md plus every ancestor's. To skip other teams' files, list them in `claudeMdExcludes` (glob patterns matched against absolute paths, settable at any settings layer; managed policy files can't be excluded).
- **Compaction treats levels differently.** After `/compact`, Claude re-reads the project-root CLAUDE.md from disk. Nested CLAUDE.md files reload only when Claude next reads a file they apply to. More in [Compaction, context editing and memory](context-engineering.md#compaction-context-editing-and-memory).

```json
{
  "claudeMdExcludes": [
    "**/monorepo/CLAUDE.md",
    "/home/user/monorepo/other-team/.claude/rules/**"
  ]
}
```

### User-level instructions are not shared

This is the fact behind CCAR-F skill 3.1-S1, diagnosing configuration hierarchy issues. `~/.claude/CLAUDE.md` lives in one person's home directory, outside the repository, so version control never delivers it to anyone else. The guide's example symptom is a new team member not receiving instructions because they sit in user-level rather than project-level configuration. The fix is to move team-wide instructions into the project CLAUDE.md and commit it. Anthropic's Claude Academy course *The AI-native SDLC playbook* gives the same advice: "Check CLAUDE.md into Git at the repo root so the whole team shares one version and changes are reviewed like code." ([Claude Academy](https://academy.claude.com/courses/ai-native-sdlc-playbook/claude-md))

`CLAUDE.local.md` has a related limit: being gitignored, it exists only in the worktree where you created it. To reuse personal instructions across worktrees of one repository, keep them in your home directory and import them:

```text
# Individual Preferences
- @~/.claude/my-project-instructions.md
```

### Modular CLAUDE.md with `@` imports

A CLAUDE.md can pull in other files with `@path/to/import`. The imported files are expanded and loaded alongside the CLAUDE.md that references them.

```text
See @README for project overview and @package.json for available npm commands for this project.

# Additional Instructions
- git workflow @docs/git-instructions.md
```

| Rule | Detail |
|---|---|
| Path types | Relative and absolute paths both work |
| Relative to what | The file containing the import, not the working directory |
| Nesting | Imported files can import others, to a maximum depth of four hops |
| Code is ignored | Paths inside Markdown code spans and fenced code blocks are not imported |
| Context cost | None saved: imports help organization, but imported files still load at launch |
| Paths outside the working directory | In a project file, the first external import shows a one-time approval dialog; if declined, those imports stay disabled and the dialog doesn't return. Imports in your own user-level files load without it, except in Cowork sessions on your desktop, which skip user-file imports that resolve outside the session's working directory |

The CCAR-F guide's use of imports is per package: each package's CLAUDE.md imports only the standards files that apply to it, chosen by the people who maintain that package. Remember the cost rule when a question is about context size rather than organization: splitting into imports doesn't shrink what loads. Path-scoped rules and skills do.

### Writing a CLAUDE.md that Claude follows

- **Size.** Target under 200 lines per file; longer files consume more context and reduce adherence. When a file approaches that, split it into [path-scoped rules](#path-scoped-rules) or move reference material into [skills](#agent-skills).
- **Specificity.** Write instructions concrete enough to verify. The [memory docs](https://code.claude.com/docs/en/memory) contrast "Use 2-space indentation" with "Format code properly", and "Run `npm test` before committing" with "Test your changes".
- **Consistency.** If two rules contradict each other, Claude may pick one arbitrarily. Review CLAUDE.md files, nested files and `.claude/rules/` together.
- **Emphasis.** If Claude keeps skipping one instruction, the [best practices guide](https://code.claude.com/docs/en/best-practices) says to add emphasis such as "IMPORTANT" to that line alone. Emphasize many lines and none stands out.
- **Fixed points belong in hooks.** An instruction that must run at a specific point, such as before every commit or after each file edit, should be a hook.
- **Compaction guidance.** You can steer what compaction keeps; the [best practices guide](https://code.claude.com/docs/en/best-practices) gives the example "When compacting, always preserve the full list of modified files and any test commands".

The [best practices guide](https://code.claude.com/docs/en/best-practices) gives a test for every line: "Would removing this cause Claude to make mistakes?" If not, cut it. Its include and exclude lists:

| Include | Leave out |
|---|---|
| Bash commands Claude can't guess | Anything Claude can figure out by reading code |
| Code style rules that differ from defaults | Standard language conventions Claude already knows |
| Testing instructions and preferred test runners | Detailed API documentation (link to docs instead) |
| Repository etiquette (branch naming, PR conventions) | Information that changes frequently |
| Architectural decisions specific to your project | Long explanations or tutorials |
| Developer environment quirks (required env vars) | File-by-file descriptions of the codebase |
| Common gotchas or non-obvious behaviors | Self-evident practices like "write clean code" |

For CI, CLAUDE.md is how a pipeline-invoked Claude Code learns the project's testing standards, fixture conventions and review criteria. The CCAR-F guide expects you to document testing standards, valuable test criteria and available fixtures there so the tests Claude writes improve; the pipeline side is in [Claude Code in CI/CD](claude-code-workflows.md#claude-code-in-cicd).

### Commands that create, inspect and debug memory

| Tool | What it does |
|---|---|
| `/init` | Generates a starting CLAUDE.md by analyzing the codebase (build commands, test instructions, conventions). If a CLAUDE.md exists, it suggests improvements instead of overwriting. It folds in Cursor rules (`.cursor/rules/`, `.cursorrules`) and Copilot rules (`.github/copilot-instructions.md`) |
| `CLAUDE_CODE_NEW_INIT=1` | Makes `/init` an interactive flow that asks which artifacts to set up (CLAUDE.md files, skills, hooks) and shows a proposal before writing; the commands reference adds that it also walks through personal memory files |
| `/memory` | Lists CLAUDE.md, CLAUDE.local.md and other memory locations across user and project scopes, including files that don't exist yet; toggles auto memory; opens a file in your editor |
| `/context` | Shows what is actually in the context window, including which CLAUDE.md and rules files loaded |
| `InstructionsLoaded` hook | Logs which CLAUDE.md and rules files load, when and why; useful for path-scoped rules and lazily loaded files |
| `/doctor` | Proposes trims for a checked-in CLAUDE.md, cutting content Claude can derive from the codebase (v2.1.206 or later) |

!!! warning "Exam guide vs current docs"

    CCAR-F skill 3.1-S4 reads: "Using the /memory command to verify which memory files are loaded and diagnose inconsistent behavior across sessions". The current docs describe `/memory` as a list of memory file locations (including files not yet created) that opens them for editing, and say: "To check which `CLAUDE.md` and rules files loaded into the current session, run `/context`." ([memory docs](https://code.claude.com/docs/en/memory)). On the exam, pick `/memory` when the question is about verifying memory files. In practice, confirm with `/context`.

To add a memory during a session, ask for it in words. Asking Claude to remember something, such as the [memory docs'](https://code.claude.com/docs/en/memory) example "always use pnpm, not npm", saves it to auto memory. To change CLAUDE.md instead, ask Claude directly (the docs' phrasing is "add this to CLAUDE.md") or edit the file through `/memory`. The old `#` prefix no longer does this (see [Slash commands](#slash-commands)).

### Organization-wide instructions and AGENTS.md

- **Managed CLAUDE.md.** It applies to every user on the machine and cannot be excluded; its paths are in [The locations](#the-locations), and how to deploy it, the managed `claudeMd` key and when to use managed settings instead are in [Managed instructions: CLAUDE.md or settings](#managed-instructions-claudemd-or-settings).
- **AGENTS.md (newer than the exam guides).** From v2.1.277, Claude Code reads `AGENTS.md` as project instructions, but by default only when there is no `CLAUDE.md`, `.claude/CLAUDE.md` or `CLAUDE.local.md` in the working directory or above; your `~/.claude/CLAUDE.md`, a managed CLAUDE.md and `.claude/rules/` files don't count for that check. The **Project instructions** setting in `/config` changes the default (`claude-md-or-agents-md`) to `claude-md-and-agents-md`, `claude-md` or `managed-only`. In sessions that can't read `AGENTS.md` directly, such as those on Amazon Bedrock or with telemetry disabled, import it from a CLAUDE.md with `@AGENTS.md`.

### Auto memory: the notes Claude writes itself

|  | CLAUDE.md | Auto memory |
|---|---|---|
| Written by | You | Claude |
| Holds | Instructions and rules | Learnings and patterns |
| Scope | Managed, user, project, local | Per repository, shared across its worktrees, on this machine only |
| Loads | Every session, in full up to 4 MiB (subdirectory files on demand) | Every session: the first 200 lines or first 25KB of `MEMORY.md`, whichever comes first; topic files on demand |

Auto memory lives in `~/.claude/projects/<project>/memory/` and is on by default; turn it off in `/memory`, with `autoMemoryEnabled: false`, or with `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`. Subagents don't receive the main conversation's auto memory (a fork is the exception); a subagent keeps its own through the `memory` field, the CCDV-F guide's "Agent Memory" (see [Subagents](#subagents)).

### Decide

- If every developer must receive an instruction, put it in the project CLAUDE.md and commit it; not in `~/.claude/CLAUDE.md`, because user-level files are not shared through version control.
- If an instruction matters only for one part of the codebase, make it a [path-scoped rule](#path-scoped-rules). If it is a multi-step procedure, make it a [skill](#agent-skills).
- If an instruction must run at a fixed point or must never be broken, use a hook or a permission rule, not CLAUDE.md.
- If a CLAUDE.md has grown too long, move content into path-scoped rules or skills. Imports and rules without `paths` still load at launch, so they improve organization without saving context.
- If behavior differs between sessions or teammates, check which memory files loaded (`/memory` in the guide's wording, `/context` in the docs) before rewriting instructions.

### Traps

- *A subdirectory CLAUDE.md overrides the root file.* Wrong: files are concatenated.
- *Imports reduce context.* Wrong: imported files load at launch.
- *Relative import paths resolve from where Claude was started.* Wrong: they resolve from the importing file.
- *CLAUDE.md is part of the system prompt, so it is enforced.* Wrong: it arrives as a user message and is guidance.
- *A nested CLAUDE.md loads as soon as Claude creates a file in that folder.* Wrong: it loads when Claude reads a file there.

## Path-scoped rules

*Tested in: CCAR-F 3.1-K4, 3.1-S3, 3.3 (3.3-K1 to 3.3-K3, 3.3-S1 to 3.3-S3), sample question 6, Exercise 2 step 2, APPX-INSCOPE-9 · CCDV-F D3.1 Claude Code Operation (Rules) · CCAR-P 2.4, 7.1*

Rules are Markdown files in `.claude/rules/`. They do two jobs. First, they break a large CLAUDE.md into topic files that are easier for a team to maintain. Second, with a `paths` field in their frontmatter, they load only when Claude works with files that match, which keeps unrelated conventions out of the context window.

### Setting up rules

Each rule file should cover one topic and have a descriptive name such as `testing.md` or `api-design.md`. Claude Code discovers every `.md` file under `.claude/rules/` recursively, so subfolders like `frontend/` or `backend/` work. A rule with no `paths` field loads at launch with the same priority as `.claude/CLAUDE.md`: unconditionally, for all files.

```text
your-project/
├── .claude/
│   ├── CLAUDE.md           # Main project instructions
│   └── rules/
│       ├── code-style.md   # Code style guidelines
│       ├── testing.md      # Testing conventions
│       └── security.md     # Security requirements
```

### Scoping a rule to paths

Add YAML frontmatter with a `paths` field listing glob patterns:

```markdown
---
paths:
  - "src/api/**/*.ts"
---

# API Development Rules

- All API endpoints must include input validation
- Use the standard error response format
- Include OpenAPI documentation comments
```

The CCAR-F guide writes the same idea as a one-line YAML list, `paths: ["terraform/**/*"]`. Both forms are YAML lists, and `paths` also accepts a comma-separated string.

```yaml
---
paths: ["terraform/**/*"]
---
```

The CCAR-F preparation exercise uses the same form: `paths: ["src/api/**/*"]` for API conventions and `paths: ["**/*.test.*"]` for testing conventions, then asks you to test that the rules load only when editing matching files.

A rule for test files wherever they sit, the case CCAR-F uses to contrast rules with directory-level CLAUDE.md files (the guide's own glob is `**/*.test.tsx`):

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

| Pattern | Matches |
|---|---|
| `**/*.ts` | All TypeScript files in any directory |
| `src/**/*` | All files under `src/` |
| `*.md` | Markdown files in the project root |
| `src/components/*.tsx` | React components in one directory |
| `src/**/*.{ts,tsx}` | Brace expansion: `.ts` and `.tsx` files under `src/` |

A rule can list several patterns, and brace groups multiply: `src/*.{ts,tsx}` expands to two patterns, and `{a,b}/{c,d}/*.{ts,tsx}` to eight. To keep expansion bounded, a rule's whole `paths` list shares one budget of 1,000 expanded patterns and 4 MiB; patterns without braces don't count against it. A pattern that would exceed the budget is used unexpanded, and its literal braces match no files.

### Rule frontmatter reference

- `paths` is the only field Claude Code reads from a rule. Any other field is ignored without an error, and the frontmatter is removed before the rule enters context.
- `paths` takes a YAML list or a comma-separated string. YAML lists of globs have been accepted since v2.1.84 (March 26, 2026).
- If the YAML between the `---` markers doesn't parse, Claude Code ignores the frontmatter and loads the rule as if it had no `paths`, which means everywhere. Run `claude --debug` to see the parse error.

### When a scoped rule loads, and when it drops out

- **Trigger.** A path-scoped rule loads when Claude reads a file matching one of its patterns, not on every tool use ([memory docs](https://code.claude.com/docs/en/memory)).
- **Compaction.** Compaction summarizes path-scoped rules and nested CLAUDE.md files away; they come back when Claude next reads a matching file. The docs' advice: "If a rule must persist across compaction, drop the `paths:` frontmatter or move it to the project-root CLAUDE.md." ([Context window docs](https://code.claude.com/docs/en/context-window))
- **Verification.** `/context` shows which rules files are loaded right now. The `InstructionsLoaded` hook logs when and why each rules file loads, which the docs single out for debugging path-specific rules.

!!! warning "Exam guide vs current docs"

    CCAR-F knowledge bullet 3.3-K2 reads: "How path-scoped rules load only when editing matching files, reducing irrelevant context and token usage". The current docs say: "Path-scoped rules trigger when Claude reads files matching the pattern, not on every tool use." ([memory docs](https://code.claude.com/docs/en/memory)). The two usually coincide, because the Edit tool's read-before-edit check means Claude has normally read a file in the conversation before editing it (the tools reference says older models always require the read, while newer models can sometimes edit an unread file). Answer exam items in the guide's terms (rules load for matching files, saving context); know that reading is the actual trigger.

### Rules, subdirectory CLAUDE.md files or skills?

| Situation | Choose | Why |
|---|---|---|
| A convention applies to one file type scattered across the tree, such as tests next to the code they test | A path-scoped rule with a glob like `**/*.test.tsx` | Glob patterns match by file type regardless of directory |
| Conventions for one directory that its own team maintains | A CLAUDE.md in that directory (a rule scoped to it also works) | The file lives with the code its owners maintain |
| You want every convention in one central place | `.claude/rules/` | Central files, one topic each |
| A standard that applies to all work in the project | The root CLAUDE.md or an unscoped rule | Loads every session |
| A procedure to run on request, or reference used only for some tasks | A skill | Loads only when invoked or judged relevant |

Anthropic's rationale for the CCAR-F sample question on this topic (test files spread through a codebase that must all follow one convention) states the decision directly. The rules answer "allows conventions to be automatically applied based on file paths regardless of directory location, essential for test files spread throughout the codebase". The skills option "requires manual skill invocation or relies on Claude choosing to load them", and the per-directory option fails "since CLAUDE.md files are directory-bound" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The full item is on the [CCAR-F page](../claude-certified-architect-foundations.md#official-sample-questions).

### Personal rules and shared rule sets

- **User-level rules** in `~/.claude/rules/` apply to every project on your machine. They load before project rules, so project rules appear later in context, but neither set overrides the other: on a conflict Claude may follow either.
- **Symlinks** in `.claude/rules/` let several projects share one rule set; circular links are detected. A link whose target is outside the working directory is treated like an external import: the linked rules don't load until you approve external imports for the project, and even then only the linked rules without `paths` load. Claude Code asks for that approval only when a project memory file has an external `@path` import, not for symlinks alone, so to share rules without it, keep them in `~/.claude/rules/`.
- **Scripts that skip project settings** skip project rules too: excluding `project` from `--setting-sources` drops them.

### Decide

- If a convention applies to files by type across many directories, choose a `.claude/rules/` file with a `paths` glob; not subdirectory CLAUDE.md files, because those are directory-bound.
- If CLAUDE.md is long and mixes unrelated topics, split it into topic files in `.claude/rules/` (the guide's examples are `testing.md`, `api-conventions.md`, `deployment.md`).
- If a rule must survive compaction, leave out `paths` or move the text into the root CLAUDE.md.
- If the convention must be enforced every time, a rule is the wrong layer: rules, like CLAUDE.md, are guidance Claude reads, not configuration Claude Code enforces. Use a hook or a permission rule.
- When the sample-question rationale speaks of the need for "deterministic" application based on file paths, read it as describing when a rule loads (matched by file path, with no invocation or inference), not whether Claude obeys it once loaded (our reading of the rationale and the docs together).

### Traps

- *Consolidate all conventions under headers in the root CLAUDE.md and let Claude infer which applies.* Anthropic's rationale in the [CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) rejects this as relying "on inference rather than explicit matching, making it unreliable."
- *A rule with a YAML typo is simply skipped.* Wrong: it loads everywhere, as if unscoped.
- *Extra frontmatter fields in a rule change its behavior.* Wrong: only `paths` is read.
- *Scoped rules stay loaded for the rest of the session.* Wrong: compaction removes them until a matching file is read again.

## Settings files and precedence

*Tested in: CCDV-F D2.6 Configuration Management (settings.json), D3.1 Claude Code Operation (settings.json) · CCAR-P 7.1*

Settings are the JSON keys that change how Claude Code behaves: "which model it starts with, what it can run without asking, which files it can't read, how it looks in your terminal, and what your organization enforces" ([settings docs](https://code.claude.com/docs/en/settings)). Unlike CLAUDE.md, settings are applied by the client, not interpreted by Claude.

### The files

| Scope | File | Who it affects | Use it for |
|---|---|---|---|
| User | `~/.claude/settings.json` | You, in every project on this machine | Personal preferences: theme, editor mode, default model, your own permission rules |
| Shared project | `.claude/settings.json` | Everyone working in the folder; in a git repository, commit it | Team permissions, hooks, plugins and the environment variables the project needs |
| Project local | `.claude/settings.local.json` | You, in this one project; Claude Code keeps it out of git when it creates the file (add it to `.gitignore` yourself if you create it by hand) | Personal overrides for one project, and testing before you share |
| Managed | `managed-settings.json` and other managed sources | Everyone the organization deploys it to; nothing you set overrides it, apart from a few security-sensitive exceptions | Security policy and compliance requirements |

A fifth file is easy to confuse with these. `~/.claude.json` holds your sign-in session, MCP server configurations, per-project state such as trust decisions, and the global config keys that `/config` writes. It holds app state and UI toggles, not settings: `permissions`, `hooks` and `env` belong in `~/.claude/settings.json`, and the debugging guide lists putting them in `~/.claude.json` as a common cause of settings that don't apply. On Windows, `~/.claude` means `%USERPROFILE%\.claude`, and `CLAUDE_CONFIG_DIR` moves the whole home-directory configuration elsewhere.

### Precedence

When the same key appears in more than one place, Claude Code uses the value from the highest level that sets it:

```text
highest   1. Managed settings   managed-settings.json, MDM, or the claude.ai console
          2. Command line       claude --settings, and flags such as --model (one session)
          3. Project local      .claude/settings.local.json
          4. Shared project     .claude/settings.json
lowest    5. User               ~/.claude/settings.json
```

- **Nothing you set overrides managed settings,** not even `--settings`. The exceptions (a managed `model` is only a default, and a few security keys accept a stricter value from a lower scope) are in [How managed values combine with everyone else's](#how-managed-values-combine-with-everyone-elses).
- **Closer beats broader below managed.** The debugging guide puts it as local, then project, then user.

Worked example (our illustration of the documented order): your `~/.claude/settings.json` sets `"model": "sonnet"` and a repository's committed `.claude/settings.json` sets `"model": "opus"`. A session launched at that repository's root starts on `opus`, because shared project sits above user; sessions launched anywhere else start on `sonnet`, including one launched in a subdirectory of that repository, because the shared file is read only from the primary working directory. Every higher level also overrides both files: a `model` in `.claude/settings.local.json`, one passed with `--settings`, or a managed `model`. So do an exported `ANTHROPIC_MODEL` and `--model` for one session, which beats the variable too.

### Three rules that settle most precedence questions

1. **Single values: highest level wins.** As above.
2. **Lists merge.** A list key such as `permissions.allow`, set in several files, is combined rather than replaced, so each file adds entries. The exceptions are the model lists and per-model entries: `fallbackModel`, `modelPicker`, `availableModels` and `modelSettings` follow their own rules. Hooks also merge across files. For permission rules, a deny from any level still wins over an allow from any other (see [Permissions and permission modes](#permissions-and-permission-modes)).
3. **Environment variables are not a level.** Whether a shell variable or a settings key applies is decided per pair. `ANTHROPIC_MODEL` exported in your shell applies over the `model` key from any file, while `ANTHROPIC_DEFAULT_MODEL` applies only when no file sets `model`. An `env` block inside a settings file is an ordinary key and follows the levels above. A variable set there applies to every session and to the subprocesses Claude Code starts, and it wins over the same variable exported in your shell.

### Changing a setting for one session

| Method | Example | Notes |
|---|---|---|
| `--settings` | `claude --settings '{"model": "claude-opus-5-5"}'` | Inline JSON or a path to a file (a regular file no larger than 2 MiB); applied above user, project and local, below managed. It can set any key a user settings file can, but not `Managed` or `Global config` keys |
| A key-specific flag | `claude --model opus` | Flags exist for keys such as the model and effort |
| The paired environment variable | `ANTHROPIC_MODEL` | Lasts for that terminal session |
| `--setting-sources` | `claude --setting-sources user,project` | Loads only the listed sources (`user`, `project`, `local`) |

### What a committed `.claude/settings.json` can and cannot do

A shared file arrives with every clone, so Claude Code limits what it can do on arrival:

- **Some keys wait for workspace trust.** `permissions.allow`, `permissions.additionalDirectories`, `extraKnownMarketplaces` and most `env` values apply only after each teammate trusts the folder. `deny` and `ask` rules apply straight away, since they only restrict.
- **Some values never apply from a project file.** The `permissions.defaultMode` values `auto` and `bypassPermissions` don't take effect from project or local settings (what happens instead is in [Setting and switching the mode](#setting-and-switching-the-mode)), and the auto mode classifier's `autoMode` block is not read from them either.
- **Some environment variables are off limits.** Project and local settings can't set variables a checked-out repository shouldn't control, such as `CLAUDE_CONFIG_DIR`, `HOME`, `TMPDIR` and `XDG_*`; Claude Code drops them with a warning.
- **It is read from one place.** The shared file comes from the session's primary working directory, so start Claude Code at the repository root to use a root-level file; project settings are not inherited from parent directories. In a git repository the local file is read and written at the repository root even when you start in a subdirectory (the documented exceptions include Windows and a repository root that is your home directory; there it stays next to `.claude/settings.json`).

The settings docs' own example, with the schema line that gives editors autocomplete and validation. The docs show it in `~/.claude/settings.json`; the same keys work in a committed `.claude/settings.json`, where the `allow` rules wait for workspace trust:

```json
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "allow": [
      "Bash(npm run lint)",
      "Bash(npm run test *)"
    ],
    "deny": [
      "Read(./.env)",
      "Read(./.env.*)"
    ]
  }
}
```

Settings files are strict JSON: a `//` comment or a trailing comma is a syntax error. The public schema can lag behind the newest CLI releases, so a validation warning on a recently documented key doesn't by itself mean the configuration is invalid.

### How the local file fills up

When you answer a Bash prompt with "Yes, and don't ask again", Claude Code saves an `allow` rule to `.claude/settings.local.json` at the repository root. The first time it writes that file in a git repository that doesn't already ignore it, it adds `**/.claude/settings.local.json` to your global git excludes file, so personal approvals stay out of commits. While the file stays untracked, its `allow` rules apply without the workspace trust step, because the file is yours rather than the repository's. An `allow` saved there does not outrank an `ask` rule from a project or managed file.

### When edits apply, and how to check what loaded

- **Live reload.** Claude Code watches the settings files and applies most edits, including `permissions`, `hooks` and credential helpers such as `apiKeyHelper`, to the running session. `model` and effort settings (`effortLevel`, `modelSettings`) are read only at session start; use `/model` and `/effort` to change them mid-session.
- **`/config`.** Opens the settings menu; most choices save to `~/.claude/settings.json`. Pass `key=value` to set one option directly, for example `/config verbose=true`.
- **`/status`.** Its `Setting sources` line lists each settings file loaded for the session, such as `User settings` or `Project local settings`. It does not say which file supplied each key; `/permissions` does show the source file of every permission rule.
- **Broken files.** Invalid JSON or a rejected value raises a Settings Error dialog at the next start; an individual bad entry, such as a malformed permission rule, raises a Settings Warning and is skipped while the rest of the file applies. A `-p` run shows no dialog and skips broken values. `claude doctor` lists rejected entries from the terminal.
- **Cloud sessions.** A cloud session reads the shared `.claude/settings.json` (in single-repository sessions) but not user or project-local settings, and receives only server-managed settings from the organization.

Model version pinning, which CCDV-F lists under Configuration Management, is set through these same files and environment variables; it is taught in [Models and output styles](#models-and-output-styles).

### Decide

- If the whole team needs a setting, put it in `.claude/settings.json` and commit it.
- If only you need it in one project, use `.claude/settings.local.json`; in every project, `~/.claude/settings.json`.
- If nobody may override it, deliver it as managed settings (see [Managed settings for organizations](#managed-settings-for-organizations)).
- If you need it for one run only, pass `--settings` or the key's flag rather than editing a file.

### Traps

- *A key in `.claude/settings.local.json` can override managed policy.* Wrong: managed sits above every settings file and `--settings`. The only lower-scope values that count are stricter values for a few security keys, such as `disableClaudeAiConnectors: true`.
- *When two files both set `permissions.allow`, the higher file's list replaces the lower one.* Wrong: lists merge.
- *A committed `"defaultMode": "bypassPermissions"` puts the team in bypass mode.* Wrong as of September 2026: since v2.1.257 that value is ignored in project and local files (earlier versions honored it from any file).
- *`//` comments are fine in a settings file.* Wrong: settings files are strict JSON, and a comment is a syntax error.
- *Project settings in a parent folder cascade into subfolders like CLAUDE.md does.* Wrong: they are not inherited.

## Permissions and permission modes

*Tested in: CCDV-F D3.1 Claude Code Operation (auto-mode, settings.json), D7.2 Guardrails and Safe Deployment (least privilege) · CCAR-F 3.4-K3 (plan mode as a mode) · CCAR-P 3.1, 5.1, 7.1, Sample 1 (least privilege)*

Permission rules decide which tool calls Claude Code runs silently, which it asks about, and which it refuses. The [permissions docs](https://code.claude.com/docs/en/permissions) are explicit that this is the client's job: "Permission rules are enforced by Claude Code, not by the model." A line in CLAUDE.md can shape what Claude tries to do, but it does not change what Claude Code allows.

### What asks by default

In Manual mode (the `default` mode), Claude Code applies a tiered system:

| Tool type | Example | Asks first? | "Yes, and don't ask again" lasts |
|---|---|---|---|
| Read-only | File reads, Grep | No, within the working and additional directories | Not applicable |
| Bash commands | Shell execution | Yes, except a built-in set of read-only commands | Permanently, per repository and command |
| File modification | Edit and write files | Yes | Until the session ends |
| Web fetch | WebFetch | Yes, except preapproved documentation domains | Permanently, per repository and domain |
| Web search | WebSearch | Yes | Permanently, per repository |

The read-only Bash set runs without a prompt in every mode and is not configurable. It includes `ls`, `cat`, `echo`, `pwd`, `head`, `tail`, `grep`, `find`, `wc`, `which`, `diff`, `stat`, `du`, `cd`, and read-only forms of `git`. The exceptions are paths fenced by `permissions.blockReadsOutsideWorkingDirectories` and, in Manual mode, a few risky forms, such as an unquoted glob on a command with write-capable flags (`find`, `sort`, `sed`, `git`). To make one of these commands prompt, add an `ask` or `deny` rule for it. Permanent approvals are saved as allow rules in `.claude/settings.local.json` at the root of the git repository (see [Settings files and precedence](#settings-files-and-precedence)); a file-modification approval is never saved and lasts until the session ends.

### Allow, ask and deny

- **Allow** rules let a matching call run without approval. **Ask** rules prompt for confirmation whenever a call matches. **Deny** rules refuse the call.
- **Order.** From the [permissions docs](https://code.claude.com/docs/en/permissions): "Rules are evaluated in order: deny, then ask, then allow. The first match in that order determines the outcome, and rule specificity doesn't change the order." So a deny on `Bash(aws *)` blocks `aws s3 ls` even if `Bash(aws s3 ls)` is allowed: an allow rule can't carve an exception out of a deny.
- **Across levels.** "If a tool is denied at any level, no other level can allow it." ([permissions docs](https://code.claude.com/docs/en/permissions)). A managed deny beats `--allowedTools`; a user-level deny beats a project-level allow.
- **Ask beats modes.** An ask rule prompts even in modes that would otherwise approve the call, such as `acceptEdits` or `bypassPermissions`. In `dontAsk`, a matching ask rule is denied.
- **Bare names remove tools.** A deny naming only a tool, like `Bash`, removes the tool from Claude's context entirely. A scoped deny like `Bash(rm *)` leaves the tool available and blocks matching calls.
- **Where rules come from.** Rules merge across settings files. `/permissions` lists every rule and the file it comes from, and a change there applies from Claude's next tool call. `--allowedTools` and `--disallowedTools` add allow and deny rules for one session.

The settings reference's own example of a `permissions` block, with one rule of each kind and a starting mode:

```json
{
  "permissions": {
    "allow": ["Bash(npm run *)"],
    "ask": ["Bash(git push *)"],
    "deny": ["Read(./.env)"],
    "defaultMode": "acceptEdits"
  }
}
```

### Rule syntax

A rule is `Tool` or `Tool(specifier)`; `Bash(*)` means the same as `Bash`.

| Rule | Matches | Watch for |
|---|---|---|
| `Bash(npm run build)` | Exactly that command | No `*` means an exact match |
| `Bash(git log *)` | `git log` with any arguments | `*` matches any text, spaces included; `Bash(git *)` would allow every git command |
| `Bash(ls *)` | `ls` and `ls -la`, not `lsof` | `Bash(ls*)` also matches `lsof`; `Bash(ls:*)` is the same as `Bash(ls *)` |
| `Read(./.env)` | The `.env` file in the current directory | Read and Edit rules use gitignore syntax |
| `Edit(docs/**)` | Any edit under `docs/` | `Edit` covers all built-in editing tools; path rules on `Write`, `NotebookEdit` or `Glob` are never consulted |
| `Read(//Users/alice/secrets/**)` | An absolute path | `//` anchors at the filesystem root; `/path` anchors at the settings source; `~/path` at home |
| `WebFetch(domain:example.com)` | Fetches from that domain | `domain:*.example.com` matches subdomains but not `example.com` itself |
| `mcp__puppeteer` | Every tool from the `puppeteer` MCP server | `mcp__puppeteer__puppeteer_navigate` matches one tool |
| `Agent(Explore)` | The built-in Explore subagent | Deny it to disable Explore |
| `Skill(name)`, `Skill(name *)` | One skill, or that skill with any arguments | Deny `Skill` to disable all skills |
| `Agent(model:opus)` | Agent calls whose `model` parameter is literally `opus` | The `Tool(param:value)` form works in deny and ask rules only, on top-level parameters of built-in tools |

Three details trip people up. A pattern like `/Users/alice/file` is not absolute: a single leading slash anchors at the settings source, which is `~/.claude/` for user settings and the primary working directory for project and local settings. A relative pattern with one directory segment matches at different depths by rule type: as an allow rule, `Edit(src/**)` matches only `<cwd>/src`, but as a deny or ask rule it matches a `src` directory at any depth. And tool-name globs differ by rule type: deny and ask rules accept glob patterns in the tool-name position, such as `"*"` and `"mcp__*"`, but allow rules accept a glob only after a literal `mcp__<server>__` prefix.

Bash matching understands shell structure. The separators `&&`, `||`, `;`, `|`, `|&`, `&` and newlines split a command, and each part must match a rule independently, so `Bash(safe-cmd *)` does not permit `safe-cmd && other-cmd`. Before matching, Claude Code strips the wrappers `timeout`, `time`, `nice`, `nohup` and `stdbuf`, the builtins `command` and `builtin`, and zsh's `noglob`; environment runners such as `npx`, `docker exec` and `devbox run` are not stripped.

### What permission rules cannot do

- **A Bash deny is not a security boundary.** `Bash(rm *)` does not stop `/bin/rm -rf build/` or `bash -c 'rm -rf build/'`. Patterns that try to constrain arguments are fragile; to limit network access, the docs suggest denying `curl` and `wget`, allowing `WebFetch(domain:github.com)` for permitted domains, and pairing the deny with the sandbox network allowlist when the restriction must hold.
- **Read and Edit denies cover Claude's tools, not every process.** They apply to Claude's built-in file tools, to file commands Claude Code recognizes in Bash (such as `cat`, `head`, `tail`, `sed` and `tee`) and to redirect targets such as `> file`. They don't stop a subprocess such as a Python script from opening a file, or a command that reads files without naming them, such as `grep -r pattern .` run from the directory that holds the file.
- **For OS-level enforcement, add the sandbox.** Sandboxing restricts Bash, PowerShell and Monitor commands and their child processes at the operating-system level; the docs recommend permissions and sandboxing together for defense in depth. With the sandbox on and `sandbox.autoAllowBashIfSandboxed` at its default `true`, sandboxed Bash commands run without a prompt even under a bare `Bash` ask rule, except in plan mode (from v2.1.212); deny rules and content-scoped ask rules such as `Bash(git push *)` still apply. Details are in [Claude Code security controls](security-and-governance.md#claude-code-security-controls).
- **Hooks sit in front of the rules, not above them.** A `PreToolUse` hook runs before the permission prompt and can deny, force a prompt or skip it, but a hook's `"allow"` does not bypass deny and ask rules. A hook that exits with code 2 blocks the call before rules are evaluated.

### Workspace trust and working directories

Which project keys wait for the workspace trust dialog is covered in [What a committed `.claude/settings.json` can and cannot do](#what-a-committed-claudesettingsjson-can-and-cannot-do): allow rules wait, deny and ask rules don't. The dialog appears only in interactive sessions. In a `claude -p` or SDK run in a folder that was never trusted, the project's `permissions.allow` rules are not used and Claude Code prints a `this workspace has not been trusted` warning to stderr, which is one reason the CI recipe below passes its allowlist on the command line (our reading). To reach files outside the start directory, use `--add-dir <path>` at launch, `/add-dir` during a session, or `additionalDirectories` in settings. The settings key grants file access only; `--add-dir` and `/add-dir` directories also load their skills, `.claude/commands/` and `.claude/agents/`.

### The six permission modes

A permission mode sets the baseline for what runs without asking; rules then adjust it.

| Mode (config value) | Status bar | Runs without asking | Best for |
|---|---|---|---|
| `default` (labeled Manual) | `⏸ manual mode on` | Reads only | Reviewing every action yourself, sensitive work |
| `acceptEdits` | `⏵⏵ accept edits on` | Reads, file edits and common filesystem commands | Iterating on code you're reviewing |
| `plan` | `⏸ plan mode on` | Reads, plus classifier-approved commands when auto mode is available | Exploring a codebase before changing it |
| `auto` | `⏵⏵ auto mode on` | Everything, with background safety checks | Long tasks, reducing prompt fatigue |
| `dontAsk` | `⏵⏵ don't ask on` | Reads and pre-approved tools; anything that would prompt is denied | Locked-down CI and scripts |
| `bypassPermissions` | `⏵⏵ bypass permissions on` | Everything | Isolated containers and VMs only |

Deny rules block in every mode, including `bypassPermissions`; allow rules have no effect in `bypassPermissions`.

!!! note "Names as of September 2026"

    The mode that reviews every action is labeled **Manual** in the CLI, `claude --help`, the VS Code and JetBrains extensions and the desktop app. Its config value is still `default`, which is what hooks and SDK integrations use; from v2.1.200 the CLI also accepts `manual` as an alias wherever you type the value, such as `--permission-mode manual` or `"defaultMode": "manual"`. Apart from plan mode and CCDV-F's "auto-mode", the four exam guides don't name permission modes. If an item says "default mode" and describes prompting before actions, read it as this mode (the config value `default`), not as whichever mode a plan starts in: on Pro, Max and Team plans the built-in starting mode is now `auto` (our reading).

### Setting and switching the mode

1. **At launch**, the first of these that applies wins: the `--permission-mode` flag (or `--dangerously-skip-permissions`), then `permissions.defaultMode` in a settings file, then the built-in default.
2. **The built-in default**, as of September 2026, is `auto` for Pro, Max and Team plans in a terminal or the VS Code extension (v2.1.228 or later on macOS, Linux and WSL; v2.1.233 or later on native Windows; on earlier versions the built-in default is Manual). It is `default` under `claude -p`, the Agent SDK, Amazon Bedrock, Google Cloud's Agent Platform, Microsoft Foundry, Claude Platform on AWS, a signed-in Claude apps gateway session, an Enterprise plan or a Console API key; whenever a settings file sets `disableAutoMode` to `"disable"`; when feature-flag fetching is off; and in the first session after you install Claude Code or upgrade to a version that adds the `auto` default (unless, after a fresh install, Claude Code fetches the flags in time). When auto mode is selected but isn't available to the session, the session starts in Manual.
3. **During a session**, `Shift+Tab` cycles `default`, `acceptEdits`, `plan` and back to `default`; `bypassPermissions` and then `auto` join the cycle after `plan` when available. `dontAsk` never appears in the cycle; set it with `--permission-mode dontAsk`.
4. **As a project default**, set `defaultMode` in `.claude/settings.json`. `default`, `acceptEdits`, `plan` and `dontAsk` work there; `auto` and `bypassPermissions` don't take effect from project or local settings (before v2.1.257, `bypassPermissions` did): an `auto` there makes the session use the built-in default rather than a `defaultMode` from `~/.claude/settings.json`, and a `bypassPermissions` there starts it in Manual. Put them in user or managed settings or pass the flag. The VS Code extension never reads project settings for the starting mode.

```bash
claude --permission-mode plan
claude --permission-mode acceptEdits
claude -p "run the test suite" --permission-mode dontAsk --allowedTools "Bash(npm test)" "Read"
```

### What each mode adds

- **`acceptEdits`** auto-approves file creation and edits plus `mkdir`, `touch`, `rm`, `rmdir`, `mv`, `cp` and `sed` for paths in the working directory or `additionalDirectories`. Paths outside that scope, protected paths, removals of critical paths and every other Bash command outside the read-only set still prompt.
- **`plan`** has Claude research and propose changes without editing source: it reads files, runs shell commands to explore and writes a plan, and edits stay blocked until you approve. Enter it with `Shift+Tab`, by starting one prompt with `/plan`, or with `--permission-mode plan`; `Ctrl+G` opens the proposed plan in your editor. The approval prompt offers **Yes, and use auto mode** (**Yes, auto-accept edits** when auto mode isn't available), **Yes, manually approve edits** and **No, keep planning**. It is not strictly read-only: when auto mode is available and `useAutoModeDuringPlan` is on (the default), the classifier reviews shell commands during planning and approved ones run; otherwise commands outside the read-only set prompt. In interactive terminal sessions where bypass permissions are available, Claude Code doesn't enforce plan mode's blocks: Claude is still told to plan without editing, but an edit or command it attempts runs without a prompt. For the exam, keep the guide's framing (CCAR-F 3.4-K3): plan mode is for safe exploration and design before committing to changes. When to choose it over direct execution is taught in [Plan mode or direct execution](claude-code-workflows.md#plan-mode-or-direct-execution).
- **`auto`** replaces most prompts with a separate classifier model that reviews actions before they run and blocks anything that escalates beyond the request, targets unrecognized infrastructure, or appears driven by hostile content Claude read. Explicit ask rules still prompt. Default blocks include `curl | bash`, sending sensitive data externally, production deploys and migrations, and force pushes. On entry it drops broad allow rules that grant arbitrary code execution, such as `Bash(*)` or `Bash(python*)`; narrow ones like `Bash(npm test)` stay, and the dropped rules return when you leave auto mode. After 3 blocks in a row or 20 in total, it pauses and prompting resumes; these thresholds are not configurable. A boundary you state in conversation, such as "don't push", is not stored as a rule and can be lost to compaction, so the [permission modes docs](https://code.claude.com/docs/en/permission-modes) say to add a deny rule for a hard guarantee; an ask rule is the way to keep a human checkpoint. Administrators turn auto mode off with `permissions.disableAutoMode: "disable"` in managed settings. In the same docs' words: "Auto mode reduces permission prompts but does not guarantee safety."
- **`dontAsk`** auto-denies every call that would prompt. File reads inside your working directories, read-only Bash, `permissions.allow` matches and calls a `PreToolUse` hook approves still run, so the session never waits for input. Explicit ask-rule matches, `AskUserQuestion` and MCP tools marked `requiresUserInteraction` are denied even if an allow rule matches them. That makes it the mode for CI with an exact allowlist.
- **`bypassPermissions`** disables prompts and safety checks, including protected-path writes. Use it only in isolated containers or VMs; it offers no protection against prompt injection. `--dangerously-skip-permissions` is equivalent. On Linux and macOS, Claude Code refuses it under root or sudo (the check is skipped inside a recognized sandbox), and you can't switch into it mid-session unless the session started with it enabled (`--allow-dangerously-skip-permissions` adds it to the cycle without turning it on). `permissions.disableBypassPermissionsMode: "disable"` blocks it; the key works from any settings scope, and managed settings keep it from being overridden.

### How auto mode decides

CCDV-F D3.1 lists "auto-mode" among Claude Code features; the product calls it auto mode, the `auto` permission mode. Each action goes through a fixed order, and the first matching step wins:

1. Your allow, ask and deny rules resolve the action, with exceptions that include these: protected-path writes go to the classifier even when an allow rule matches, and an ask rule that matches on a command's content, such as `Bash(git push *)`, falls back to a permission prompt.
2. Read-only actions and file edits in the working directory are auto-approved, except protected-path writes and the first read outside the working directories, which prompts you.
3. Everything else goes to the classifier.
4. If the classifier blocks, Claude receives the reason and tries an alternative.

What else to know about the classifier:

- **What it sees.** In the requests Claude Code itself sends, user messages, tool calls other than read-only lookups, and your CLAUDE.md content. Tool results are stripped from its requests, so hostile content in a file or web page can't manipulate it directly.
- **What it trusts.** By default, only the working directory and the remotes configured when the session started; a remote added or repointed mid-session with `git remote add` or `git remote set-url` isn't trusted (before v2.1.200, remotes added mid-session were). Administrators add trusted repos, buckets and services in `autoMode.environment`, and the classifier doesn't read `autoMode` from project `.claude/settings.json` or `.claude/settings.local.json`, so a cloned repository can't widen it. `claude auto-mode defaults` prints the built-in rule lists as JSON.
- **Where it is available (as of September 2026).** All plans; on Team and Enterprise it is available by default, and administrators can turn it off with `permissions.disableAutoMode`. On the Anthropic API and Claude Platform on AWS it needs Claude Opus 4.6 or later, Sonnet 4.6 or later, or a Fable model; on Amazon Bedrock, Google Cloud's Agent Platform, Microsoft Foundry and signed-in Claude apps gateway sessions, only Claude Sonnet 5, Opus 4.7 or later, and the Fable models. Sonnet 4.5, Opus 4.5, Haiku and claude-3 models are not supported on any provider.
- **Subagents.** The classifier also reviews work delegated to subagents; how a subagent's mode follows the parent's is in [Tool control](#tool-control).

### Checks that modes and allow rules don't skip

- **Protected paths.** Writes to directories such as `.git`, `.vscode`, `.idea`, `.husky` and `.claude` (except `.claude/worktrees`), and to files such as `.gitconfig`, `.bashrc`, `.zshrc`, `.npmrc`, `.mcp.json` and `.claude.json`, are never auto-approved: they prompt in Manual and `acceptEdits`, go to the classifier in `auto`, and are denied in `dontAsk`. An allow rule can't pre-approve them, because the safety check runs first: `Edit(.claude/**)` changes nothing. Only `bypassPermissions` (and plan mode in an interactive terminal session where bypass permissions are available) skips this check.
- **Never auto-approved in any mode, including `bypassPermissions`:** explicit ask-rule matches, `AskUserQuestion`, MCP tools marked `requiresUserInteraction`, connector tools the organization set to `ask`, cross-session messaging safeguards, reads outside the working directories while `blockReadsOutsideWorkingDirectories` is on, and `rm` or `rmdir` aimed at a critical path (such as the filesystem root, top-level directories, home, and the working directory and its parents). No allow rule or `PreToolUse` hook `"allow"` gets past that last one; the docs call it a circuit breaker that guards against model error. For a critical-path removal, what happens instead depends on the mode: Manual, `acceptEdits` and `bypassPermissions` ask you, `plan` asks you (or sends it to the classifier when auto mode is available during planning and bypass permissions aren't), `auto` sends it to the classifier, and `dontAsk` denies it. Other items on the list take a different path in `auto`: an explicit ask-rule match asks you, and MCP tools marked `requiresUserInteraction` and connector tools the organization set to `ask` prompt you directly and never reach the classifier. In `dontAsk` all of these are denied, and so is `AskUserQuestion`.

### Decide

- If a CI job must never hang and must do only named things, run `claude -p` with `--permission-mode dontAsk` and an explicit `--allowedTools` list.
- If Claude's file tools and recognized file commands must not read a file for anyone on the team, commit a `Read(...)` deny in `.claude/settings.json` (deny rules apply without workspace trust). For a hard guarantee that also covers commands that read without naming the file, such as `grep -r pattern .`, and scripts Claude runs, add the sandbox.
- If a rule must bind everyone in the organization, deliver it as a managed deny, which no lower level can allow (see [Managed settings for organizations](#managed-settings-for-organizations)).
- If a role never needs a tool, remove the tool rather than guard it: a bare-name deny such as `Bash`, or a tool-name glob deny such as `mcp__github__*`, takes it out of Claude's context. Anthropic's rationale for CCAR-P Sample 1 is the rule to remember: "Least privilege means removing capabilities the role does not require, eliminating the attack surface rather than monitoring or guarding it." ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)).
- If you want long autonomous runs with a human checkpoint on a tool the role does need, use auto mode with ask rules such as `Bash(git push *)`; an ask rule prompts even in auto mode, though it matches only the command as written.
- If a run must be fully unattended without an allowlist, so every call runs without approval checks, `bypassPermissions` is acceptable only inside an isolated container or VM. Even there, explicit ask-rule matches and critical-path removals still prompt, and a `-p` run denies them instead.

### Traps

- *A more specific allow overrides a broader deny.* Wrong: deny is checked first and specificity doesn't matter.
- *`Write(docs/**)` restricts writes to `docs/`.* Wrong: `Write` path rules are never consulted; use `Edit(docs/**)`.
- *`/Users/alice/secrets` in a rule is an absolute path.* Wrong: absolute paths start with `//`.
- *Denying `Bash(rm *)` makes deletion impossible.* Wrong: other invocations get through; use the sandbox or a hook.
- *`bypassPermissions` ignores deny rules.* Wrong: deny rules block in every mode.
- *`dontAsk` asks when unsure.* Wrong: it denies anything that would prompt.
- *A CLAUDE.md line forbidding `rm` guarantees the command is blocked.* Wrong: CLAUDE.md is guidance, not an enforced rule. In auto mode the classifier reads it too and may block the command, but that is a judgment call; for enforcement use a deny rule, and the sandbox or a hook for a hard guarantee.
- *`"defaultMode": "auto"` in the committed `.claude/settings.json` is how to make auto mode the team default.* Wrong: `auto` doesn't take effect from project or local settings. Claude Code falls back to the built-in default instead, skipping any `defaultMode` in `~/.claude/settings.json`. That default is `auto` only on Pro, Max and Team plans; Enterprise and Console API key sessions start in Manual. Put `auto` in user or managed settings.
- *Keeping a dangerous tool the role never uses, behind a confirmation prompt, is least privilege.* Wrong: Anthropic's CCAR-P Sample 1 rationale calls logging and confirmations "detective/compensating controls, not removal of unnecessary privilege" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). Remove the tool.

## Slash commands

*Tested in: CCAR-F 3.2-K1, 3.2-S1 (sample Question 4), 3.1-S4, 5.4-S5, APPX-TECH-3, APPX-INSCOPE-10 · CCDV-F D3.1 Claude Code Operation (built-in and custom slash commands, repository initialization) · CCAR-P 7.1*

Slash commands start work by name. Some are built into the CLI, some are skills or workflows that ship with Claude Code, and the rest come from your own files, plugins or MCP servers. CCAR-F tests where a team's custom command belongs (sample Question 4), and CCDV-F lists built-in and custom slash commands under Claude Code Operation.

A slash command is recognized only at the start of a message: `/`, a name, and any text after the name as its arguments. Type `/` alone to see what is available, or `/` plus a few letters to filter. A command sent while Claude is responding is queued until the turn ends; a few, such as `/status` and `/usage`, run immediately. Skills are the exception to the one-command rule: since v2.1.199 you can chain up to six at the start of a message, as in `/skill-a /skill-b do XYZ`.

### What can appear in the `/` menu

| Kind | What it is | Examples |
|---|---|---|
| Built-in command | Fixed logic coded into the CLI | `/clear`, `/compact`, `/config`, `/permissions` |
| Bundled skill | A prompt handed to Claude, which then orchestrates the work with its tools; marked **Skill** in the commands reference. Turn them off with `disableBundledSkills` (from v2.1.205, `/doctor` stays typable) | `/code-review`, `/batch`, `/doctor`, `/debug`, `/loop`, `/claude-api` |
| Bundled workflow | A dynamic workflow that fans work out across many subagents and runs in the background; marked **Workflow** in the commands reference | `/deep-research` |
| Your own command or skill | A Markdown prompt in `.claude/commands/`, `.claude/skills/` or their `~/.claude/` equivalents | `/fix-issue 123`, `/deploy` |
| Synced skill | A skill synced from your claude.ai account; the `/` menu marks it as coming from claude.ai | `/anthropic-skills:<name>`, or `/<name>` while no other command uses that name |
| Plugin skill | A skill shipped in a plugin, always namespaced | `/plugin-name:skill-name` |
| MCP prompt | A prompt an MCP server exposes, listed as `/servername:promptname (MCP)`; typing `/mcp__servername__promptname` also runs it | `/mcp__github__pr_review 456` |

### Commands worth knowing

| Command | What it does |
|---|---|
| `/init` | Initializes the project with a CLAUDE.md guide; its options are in [Commands that create, inspect and debug memory](#commands-that-create-inspect-and-debug-memory) |
| `/memory` | Edits CLAUDE.md files, turns auto memory on or off, and shows auto memory entries |
| `/context [all]` | Shows current context usage as a colored grid, with suggestions for context-heavy tools and memory bloat |
| `/compact [instructions]` | Frees context by summarizing the conversation so far; optional focus instructions, such as `/compact focus on the auth bug fix` |
| `/clear [name]` | Starts a new conversation with empty context while keeping project memory (aliases `/reset`, `/new`) |
| `/rewind` | Rewinds the conversation, the code or both to an earlier point, or summarizes from a chosen message (aliases `/checkpoint`, `/undo`) |
| `/resume [session]` | Resumes a conversation by ID or name, or opens the session picker (alias `/continue`) |
| `/plan [description]` | Enters plan mode from the prompt, for example `/plan fix the auth bug` |
| `/config [key=value ...]` | Opens the settings interface, or sets keys directly, for example `/config model=sonnet` (alias `/settings`) |
| `/status` | Opens settings on the Status tab: version, model, account, connectivity; works while Claude is responding |
| `/permissions` | Manages allow, ask and deny rules and working directories (alias `/allowed-tools`) |
| `/model [model]` | Switches the model and saves it as your default for new sessions; press `s` on a row to switch for this session only |
| `/effort [level]` | Sets the effort level (`low` to `xhigh`, `max`, `ultracode` or `auto`); `max` and `ultracode` last for the session only |
| `/hooks`, `/mcp`, `/skills`, `/plugin` | View hook configuration; manage MCP connections and OAuth; list skills; manage plugins |
| `/add-dir <path>` | Adds a working directory for the current session |
| `/agents` | Since v2.1.198, prints a reminder to ask Claude to manage subagents or edit `.claude/agents/` directly (the wizard is gone) |
| `/code-review` | Bundled skill: reviews the current diff, or a PR number, branch or path, for correctness bugs; `/review` is its alias |
| `/security-review` | Analyzes the current branch's changes for security vulnerabilities |
| `/batch <instruction>` | Bundled skill: splits a large change into 5 to 30 independent units, each handled by a background subagent in its own git worktree |
| `/usage` | Session cost, plan usage limits and activity stats (aliases `/cost`, `/stats`) |
| `/doctor` | Bundled skill that runs a setup checkup and can fix issues (alias `/checkup`); `claude doctor` in a terminal prints read-only diagnostics instead |

The docs suggest a first-session order in a new repository: `/init` to generate a starter CLAUDE.md, then `/memory` to refine it, `/mcp` for servers and `/permissions` for approval rules. For CCAR-F, `/memory` (verifying memory files) and `/compact` (reducing context during long exploration) are the two built-ins the guide names in its objectives. The docs point to `/context`, not `/memory`, for checking which memory files actually loaded; answer in the guide's terms (see [CLAUDE.md and the memory hierarchy](#claudemd-and-the-memory-hierarchy)). When to compact, clear or rewind is taught in [Managing context in large codebases](claude-code-workflows.md#managing-context-in-large-codebases).

### Input prefixes that are not commands

- **`!` at the start** switches to shell mode: the command runs directly, its output is added to the session, and Claude responds to it. Shell-mode commands run outside the sandbox even when sandboxing is on (strict sandbox mode aside), because the sandbox applies to commands Claude runs.
- **`@`** mentions a file path, with autocomplete.
- **`#` no longer adds memories.** It was the quick-memory prefix from v0.2.54 (April 2, 2025) until v2.0.70 (December 15, 2025) removed it; tell Claude to edit CLAUDE.md instead. Older material may still show it.

### Custom commands

A custom command is a Markdown file whose body is the prompt. Where you save it decides who gets it:

| Location | Who gets it | Exam framing |
|---|---|---|
| `.claude/commands/` in the repository | Everyone who clones or pulls the repository | Project-scoped, shared through version control |
| `~/.claude/commands/` | Only you, in all your projects | User-scoped, personal |

The command name comes from the file name without `.md`: `.claude/commands/deploy.md` becomes `/deploy`. Each subdirectory becomes a `:` namespace, so `.claude/commands/frontend/component.md` becomes `/frontend:component`. The `/project:` prefix seen in 2025 material is gone.

The docs' own example of a command file, with an argument hint, live context from a shell command, and the user's input in `$ARGUMENTS` (`.claude/commands/fix-issue.md`, run as `/fix-issue 123`, which makes the `!` line run `gh issue view 123`):

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

What this shows:

- `$ARGUMENTS` receives everything typed after the name, so `/fix-issue 123` runs `gh issue view 123`.
- The `` !`<command>` `` line runs before the prompt is sent to Claude and is replaced by the command's output. Such commands never prompt: outside auto mode, a command that isn't allowed aborts the whole invocation. `gh` is not among the read-only commands the permissions docs list, so outside auto mode pre-approve it, for example with `allowed-tools: Bash(gh issue view *)` in the frontmatter, as the docs' own `pr-summary` skill does with `allowed-tools: Bash(gh *)` (our reading).
- `argument-hint` is shown during autocomplete to indicate the expected arguments.
- Command files accept the same frontmatter as skills except `name` and `paths`. The general rules for placeholders and injected commands are in [Arguments and dynamic context](#arguments-and-dynamic-context), the full field list in [Every frontmatter field](#every-frontmatter-field), and what the exam guide says about `argument-hint` and `allowed-tools` in [`allowed-tools` and `argument-hint`: what the guide says and what the docs say](#allowed-tools-and-argument-hint-what-the-guide-says-and-what-the-docs-say).

### How commands relate to skills today

Claude Code v2.1.3 (January 9, 2026) "Merged slash commands and skills, simplifying the mental model with no change in behavior" ([changelog](https://code.claude.com/docs/en/changelog)). A file at `.claude/commands/deploy.md` and a skill at `.claude/skills/deploy/SKILL.md` both create `/deploy` and work the same way, and existing command files keep working.

|  | Command file | Skill |
|---|---|---|
| Path | `.claude/commands/<name>.md` | `.claude/skills/<name>/SKILL.md` |
| Invocation | `/<name>`; Claude can also invoke it (the docs disagree on whether it loads automatically) | `/<name>`; Claude can also load it automatically when relevant |
| Supporting files | None: a single file | Reference files and scripts in the skill folder |
| Frontmatter | Skill fields except `name` and `paths` | All skill fields |
| Same name as the other | Loses | Wins |
| Status | Older format, still supported | Preferred for new work |

A skill beats a command file of the same name. How skills at different levels shadow each other, and why CCAR-F 3.2-K4 gives personal variants different names, is in [Where skills live](#where-skills-live).

The command in CCAR-F's sample question needs care today. As of September 2026, `/review` is an alias of the bundled `/code-review` skill; before v2.1.223 it was a separate command that reviewed a GitHub pull request by number in a single read-only pass, except that from v2.1.186 through v2.1.201 it ran the same multi-agent engine as `/code-review medium`. The docs' collision table covers a local skill that takes a bundled skill's name (it replaces the bundled command but not its aliases), not one that takes an alias's name, so a team building a `/review` checklist today avoids the question by giving it a distinct name (our suggestion).

!!! warning "Exam guide vs current docs"

    The guide (3.2-K1 and sample Question 4) places team commands in `.claude/commands/` and personal ones in `~/.claude/commands/`. Anthropic's rationale: "Project-scoped custom slash commands should be stored in the .claude/commands/ directory within the repository." It rejects the alternatives because "Option B (~/.claude/commands/) is for personal commands that aren't shared via version control. Option C (CLAUDE.md) is for project instructions and context, not command definitions." ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The current docs call command files the older format and recommend skills for new work, but `.claude/commands/` still works. On the exam, answer with the guide's locations. The full item is on the [CCAR-F page](../claude-certified-architect-foundations.md#official-sample-questions).

### Controlling who can run what

- Claude's skill invocations go through the Skill tool, so permission rules restrict them: deny `Skill` to disable all skills, or use `Skill(name)` for an exact match and `Skill(name *)` for a prefix match with arguments.
- A few built-ins are reachable through the Skill tool, including `/init` and `/security-review`; others, such as `/compact`, are not.
- For a command with side effects that only a person should trigger, set `disable-model-invocation: true` in its frontmatter; see [Agent Skills](#agent-skills).

### Decide

- If every developer should get a command when they clone or pull, commit it in the repository: `.claude/commands/` in the guide's wording, `.claude/skills/<name>/SKILL.md` for new work today.
- If only you want it, keep it in `~/.claude/commands/` or `~/.claude/skills/`, under a name that doesn't shadow a team command.
- If the content is a standard Claude should always follow rather than a task you start, it belongs in CLAUDE.md or a rule, not a command.
- If a context-heavy session is still on the same task, `/compact`; if you are switching to an unrelated task, `/clear`.

### Traps

- *Define the command in CLAUDE.md.* Wrong: CLAUDE.md is for instructions and context, not command definitions.
- *Add a `.claude/config.json` with a commands array.* Wrong: Anthropic's rationale for CCAR-F sample Question 4 says this option "describes a configuration mechanism that doesn't exist in Claude Code" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).
- *Put the team command in `~/.claude/commands/`.* Wrong: that directory holds personal commands that aren't shared through version control.
- *Stop a pipeline run from hanging with `claude --batch`.* Wrong: Anthropic's rationale for CCAR-F sample Question 10 calls the `--batch` flag a non-existent feature. The documented way to run non-interactively is `claude -p` (or `--print`). The `/batch` bundled skill does exist, but it is a slash command that fans a large change out to subagents, not a CLI flag for non-interactive runs.
- *Open `/agents` to build a subagent in a wizard.* Wrong since v2.1.198: `/agents` now prints a reminder to ask Claude or to edit `.claude/agents/` directly.
- *`/model` changes the model for this session only.* Wrong: in an interactive session it also saves the choice as your default for new sessions unless you press `s` (in a `-p` run, from v2.1.205, it applies to that session only).

## Agent Skills

*Tested in: CCAR-F 3.2 (3.2-K2, 3.2-K3, 3.2-K4, 3.2-S2 to 3.2-S5), Exercise 2 step 3, APPX-TECH-3, APPX-INSCOPE-10 · CCDV-F D3.1 Claude Code Operation, D8.3 Agentic Customization · CCAR-P 2.5, 3.8, 7.1*

A skill is a `SKILL.md` file of instructions that Claude adds to its toolkit. Claude uses a skill when it is relevant, or you run it directly with `/skill-name`. The property behind the skill-or-CLAUDE.md choice (CCAR-F 3.2-S5): unlike CLAUDE.md, a skill's body loads only when the skill is used, so long reference material costs almost nothing until it is needed. Claude Code skills follow the Agent Skills open standard (agentskills.io) and add invocation control, subagent execution and dynamic context injection on top of it.

**Skills as prompt reuse (CCAR-P 2.5).** The CCAR-P guide lists Skills next to caching and modular prompts as prompt reuse strategies. The skills documentation gives the trigger for writing one: you keep pasting the same instructions, checklist or multi-step procedure into chat, or a section of CLAUDE.md has grown into a procedure rather than a fact.

### Skills and custom commands are one mechanism

Custom commands have been merged into skills, so a command file in `.claude/commands/` and a skill folder in `.claude/skills/` are two formats for the same `/name` command. Everything in this section is written for the skill format, which the docs prefer for new work. How the two formats differ (frontmatter, supporting files, which one wins a name clash) and the CCAR-F sample question about a shared `/review` command are taught in [Slash commands](#slash-commands).

### Where skills live

| Level | Path | Who gets it |
|---|---|---|
| Enterprise | `.claude/skills/<skill-name>/SKILL.md` in the managed settings directory | Every user on machines where the organization deploys it |
| Personal | `~/.claude/skills/<skill-name>/SKILL.md` | You, in all your projects on this machine (not Cowork or cloud sessions) |
| Project | `.claude/skills/<skill-name>/SKILL.md` | Everyone working in the repository, once it is committed |
| Nested | `<subdir>/.claude/skills/<skill-name>/SKILL.md` | Sessions started in or below `<subdir>`; a session started above it loads the skill once Claude works on files there |
| Additional directory | `.claude/skills/<skill-name>/SKILL.md` inside a directory passed with `--add-dir` | That session |
| Plugin | `<plugin>/skills/<skill-name>/SKILL.md` | Wherever the plugin is enabled, invoked as `/plugin-name:skill-name` |
| claude.ai account | Skills enabled for your claude.ai account | Cowork, cloud sessions, and terminal sessions signed in with that account |

Rules that follow from the table:

- **Same name, different levels.** Enterprise beats personal, and personal beats project. With `deploy` in both `~/.claude/skills/` and the project's `.claude/skills/`, `/deploy` runs your personal copy. Plugin skills never collide, because they are namespaced as `/plugin-name:skill-name`.
- **Nested skills do not shadow.** With a `deploy` skill at the repository root and another in `apps/web/.claude/skills/`, both stay available: `/deploy` runs the root skill and `/apps/web:deploy` runs the nested one.
- **Bundled skills.** A local skill with the same name as a bundled skill replaces the bundled command but not its aliases; what that means for a team `/review` command is in [Slash commands](#slash-commands).
- **Discovery.** Project skills load from `.claude/skills/` in the start directory and every parent up to the repository root. Claude Code watches skill directories, so added, edited or removed skills take effect in the running session (except in bare mode). A top-level skills directory created after the session started needs a restart before it is watched.
- **Command name.** The directory name becomes the command: `.claude/skills/deploy-staging/SKILL.md` gives `/deploy-staging`. In a personal or project skill, the `name` field only sets the display label.
- **The folder is required.** A file at `.claude/skills/name.md` is a common mistake; the skill must be `.claude/skills/name/SKILL.md`.

**Personal variants (CCAR-F 3.2-K4).** The guide asks for personal variants in `~/.claude/skills/` "with different names to avoid affecting teammates" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). In Claude Code terms, give the variant a different name, such as `deploy-verbose` next to the team's `deploy`: under the shadowing rule above, a personal skill with the team's name would run in place of the team's skill for you, so you would stop seeing the version your teammates use.

### SKILL.md anatomy

Every skill is a directory with a `SKILL.md` inside it. The file has YAML frontmatter between `---` markers, which tells Claude when to use the skill, and a markdown body with the instructions Claude follows when it runs. The skills documentation's deploy example below is a task skill that only you can trigger and that runs in a forked subagent.

```yaml
---
name: deploy
description: Deploy the application to production
context: fork
disable-model-invocation: true
---

Deploy the application:
1. Run the test suite
2. Build the application
3. Push to the deployment target
```

Three parsing rules decide whether your frontmatter is read at all:

- In Claude Code every field is optional. Only `description` is recommended, and unknown field names are ignored without an error.
- Frontmatter is read only when the opening `---` is the file's first line.
- If the YAML is malformed, the skill still loads, with no fields set: `/skill-name` still works, but Claude cannot match your request against the missing `description`. Running with `--debug` shows the parse error.

### Every frontmatter field

| Field | What it does | Default and notes |
|---|---|---|
| `name` | Display label in skill listings. In a plugin skill it sets the last segment of the command | The command in a personal or project skill comes from the directory name |
| `description` | What the skill does and when to use it; Claude decides whether to load the skill from this | Recommended. If omitted, the first non-empty markdown line is used |
| `when_to_use` | Extra trigger context, such as example requests, appended to `description` | `description` plus `when_to_use` is truncated at 1,536 characters in the skill listing |
| `argument-hint` | Hint shown during autocomplete, for example `[issue-number]` or `[filename] [format]` | See the guide-versus-docs note below |
| `arguments` | Named positional arguments for `$name` substitution | Space-separated string or YAML list |
| `disable-model-invocation` | `true` stops Claude from loading the skill automatically; you run it with `/name` | Default `false`. Also stops the skill being preloaded into subagents |
| `user-invocable` | `false` hides the skill from the `/` menu, so only Claude can invoke it | Default `true`. Use for background knowledge |
| `allowed-tools` | Tools Claude may use without a permission prompt during the turn that invokes the skill | The grant clears when you send your next message. It does not remove any tool |
| `disallowed-tools` | Tools removed from Claude's available pool while the skill is active | Clears on your next message. Example: `AskUserQuestion` for a background loop |
| `model` | Model for the rest of the current turn; accepts `/model` values or `inherit` | Not saved to settings. A model blocked by `availableModels` is ignored. With `context: fork`, it sets the forked subagent's model instead |
| `effort` | Effort level while the skill is active: `low`, `medium`, `high`, `xhigh`, `max` | Available levels depend on the model |
| `context` | `fork` runs the skill in a forked subagent context | See `context: fork` below |
| `agent` | Subagent type for `context: fork`: `Explore`, `Plan`, `general-purpose`, or a custom agent from `.claude/agents/` | Default `general-purpose` |
| `background` | Only with `context: fork`: `false` waits for the subagent's result in the invoking turn | Default `true`; requires v2.1.218 or later |
| `hooks` | Hooks registered when the skill is invoked; they keep running for the rest of the session | Same configuration format as settings-based hooks. `once: true` on a hook removes it after its first successful run |
| `paths` | Glob patterns that limit automatic activation to matching files | Same format as path-scoped rules |
| `shell` | Shell for `!` commands in the skill | `bash` (default) or `powershell` |
| `metadata` | Free-form YAML map for your own tooling | Claude Code does not act on it |
| `license` | License field from the Agent Skills spec | Accepted, not acted on |
| `compatibility` | Environment requirements from the Agent Skills spec | Up to 500 characters; accepted, not acted on |

**Portability.** Outside Claude Code (claude.ai uploads, the Skills API, `package_skill.py`), only six keys are allowed: `name`, `description`, `license`, `compatibility`, `metadata` and `allowed-tools`. Any other key is a hard error, for example `Unexpected key(s) in SKILL.md frontmatter: argument-hint`. The platform spec also constrains values: `name` has at most 64 characters of lowercase letters, numbers and hyphens, no XML tags and neither of the reserved words `anthropic` and `claude`; `description` is non-empty, at most 1,024 characters, with no XML tags. The platform overview lists `name` and `description` as required, while Claude Code treats both as optional. Claude Code-only body features, such as dynamic context injection, do not work in claude.ai chat or through the API. A skill meant for several surfaces should stay inside the six shared keys and fill in both required ones.

### Who can invoke a skill, and what it costs in context

| Frontmatter | You can invoke | Claude can invoke | What sits in context |
|---|---|---|---|
| (default) | Yes | Yes | Description always; full skill when invoked |
| `disable-model-invocation: true` | Yes | No | Description not in context; full skill when you invoke it |
| `user-invocable: false` | No | Yes | Description always; full skill when invoked |

- **Decide:** if a skill has side effects (deploys, commits, sends messages), set `disable-model-invocation: true`. Only you can trigger it, and its description stops costing context. If a skill is background knowledge that users should never call directly, set `user-invocable: false`.
- **`user-invocable: false` is not a block on Claude.** It hides the skill from the `/` menu so you cannot run it, but Claude still can. To keep Claude from invoking a skill, set `disable-model-invocation: true`.
- **Permissions on the Skill tool:** `Skill`, `Skill(name)` and `Skill(name *)` rules control which skills can run; the syntax is in [Slash commands](#slash-commands).
- **Visibility per skill:** the `skillOverrides` setting sets a state for each skill name without editing its `SKILL.md`. The `/skills` menu writes it to `.claude/settings.local.json`, a skill missing from it counts as `"on"`, and plugin skills are unaffected.

| `skillOverrides` value | Listed to Claude | In the `/` menu |
|---|---|---|
| `"on"` | Name and description | Yes |
| `"name-only"` | Name only | Yes |
| `"user-invocable-only"` | Hidden | Yes |
| `"off"` | Hidden | Hidden |

The documentation's example collapses one skill to its name and turns another off entirely:

```json
{
  "skillOverrides": {
    "legacy-context": "name-only",
    "deploy": "off"
  }
}
```

### Progressive disclosure

Skills are built for progressive disclosure: Claude loads information in stages as it needs it instead of paying for everything up front. The platform docs define three levels; their table labels the third one Level 3+.

| Level | When it loads | Token cost | What loads |
|---|---|---|---|
| 1. Metadata | Always, at startup | About 100 tokens per skill | `name` and `description` from the frontmatter |
| 2. Instructions | When the skill is triggered | Under 5k tokens | The `SKILL.md` body |
| 3+. Resources | As needed | None until accessed | Bundled files: reference files load into context when read; scripts run through bash and only their output enters context |

How that plays out in Claude Code:

- In a regular session, skill descriptions sit in context so Claude knows what exists; full content loads only on invocation. A subagent that lists skills in its `skills` field gets their full content at startup instead.
- The skill listing has a budget of 1% of the model's context window (fallback 8,000 characters), adjustable with `skillListingBudgetFraction` or `SLASH_COMMAND_TOOL_CHAR_BUDGET`. The budget never removes a skill's name from the listing (skills hidden with `disable-model-invocation: true`, or with `skillOverrides` set to `"user-invocable-only"` or `"off"`, are not listed at all); when it overflows, Claude Code shortens descriptions, dropping them first for the skills you invoke least. Separately, each entry's `description` plus `when_to_use` is capped at 1,536 characters (configurable with `skillListingMaxDescChars`).
- Once invoked, the rendered `SKILL.md` enters the conversation as a single message and stays there; Claude Code does not re-read the file on later turns. The `allowed-tools` grant does not persist the same way: it clears when you send your next message.
- After auto-compaction, Claude Code re-attaches the most recent invocation of each skill, keeping the first 5,000 tokens of each within a combined budget of 25,000 tokens. The budget is filled from the most recently invoked skill, so older skills can drop out entirely if you invoked many.

Because the description is what Claude matches your request against, it must say both what the skill does and when to use it. The platform authoring guidance adds: write the description in the third person (it is injected into the system prompt), treat `SKILL.md` as a table of contents that points Claude to detailed files, and keep references one level deep from `SKILL.md`. Both the Claude Code docs and the platform guidance say to keep `SKILL.md` under 500 lines. In Claude Code, reference each supporting file from `SKILL.md` so Claude knows what it contains and when to load it; scripts are executed, not loaded.

```text
my-skill/
├── SKILL.md (required - overview and navigation)
├── reference.md (detailed API docs - loaded when needed)
├── examples.md (usage examples - loaded when needed)
└── scripts/
    └── helper.py (utility script - executed, not loaded)
```

**Progressive discovery versus monolithic context (CCAR-P 3.8).** The guide names the trade-off without naming features; in Claude Code terms (our mapping), CLAUDE.md is the monolithic option, because its full content is sent with every request, and a skill is the progressive option, because only its description is sent until it is used. The docs' rule: if Claude should always know something (coding conventions, build commands, project structure, rules about what never to do), it belongs in CLAUDE.md; reference material and invocable workflows belong in skills. The size guidance for CLAUDE.md is in [Writing a CLAUDE.md that Claude follows](#writing-a-claudemd-that-claude-follows).

### Arguments and dynamic context

| Placeholder | Expands to |
|---|---|
| `$ARGUMENTS` | Everything typed after the command |
| `$ARGUMENTS[N]` or `$N` | One argument by 0-based index; `$0` is the first |
| `$name` | A named argument declared in the `arguments` frontmatter |
| `${CLAUDE_SKILL_DIR}` | The directory holding this skill's `SKILL.md` |
| `${CLAUDE_PROJECT_DIR}` | The project root (v2.1.196 or later) |
| `${CLAUDE_SESSION_ID}`, `${CLAUDE_EFFORT}` | The session ID and the current effort level |
| `${CLAUDE_PLUGIN_ROOT}`, `${CLAUDE_PLUGIN_DATA}` | Plugin install directory and persistent data directory (plugin skills only) |

Indexed arguments use shell-style quoting, so wrap a multi-word value in quotes: `/my-skill "hello world" second` makes `$0` expand to `hello world` and `$1` to `second`, while `$ARGUMENTS` always expands to the full string as typed. An indexed placeholder with no matching argument (`$2` when only one was passed) stays in the content unchanged, and a named placeholder with no matching argument expands to an empty string. If you pass arguments and no placeholder receives them, Claude Code appends `ARGUMENTS: <your input>` to the skill content so Claude still sees them; an unmatched indexed placeholder such as that `$2` doesn't count as receiving them.

```yaml
---
name: migrate-component
description: Migrate a component from one language to another
---

Migrate the $0 component from $1 to $2.
Preserve all existing behavior and tests.
```

Running `/migrate-component SearchBar JavaScript TypeScript` (the documentation's example) fills the three positions in order.

**Dynamic context injection.** The `` !`<command>` `` syntax runs a shell command before the skill content is sent to Claude and replaces the placeholder with the command's output, so Claude receives the data, not the command. The inline form is recognized only at the start of a line or right after whitespace. For several lines, use a fenced code block whose opening fence is followed directly by `!` (example below). Substitution runs once: command output is inserted as plain text and is not scanned again for placeholders.

- **Failure aborts everything.** A failed command aborts the whole skill invocation, not only its placeholder, and Claude never sees the skill content for that invocation. With the default `bash` shell any non-zero exit code counts as a failure, apart from exit code 1 from search and comparison commands; append `|| true` to a command you expect to exit non-zero.
- **Permissions are checked, never prompted.** Each injected command is checked against your permission rules first; a deny rule aborts the invocation, and outside auto mode so does any result other than allow, including a rule that would normally ask you. Pre-approving the command in `allowed-tools` keeps it from aborting; deny and ask rules still override `allowed-tools`. In auto mode, a command that would otherwise need approval doesn't abort: the skill loads with an instruction for Claude to run the command first, and that call goes through auto mode's usual checks. A forked skill that sets `agent` still aborts.
- **Policy switch.** Setting `"disableSkillShellExecution": true` replaces each command with `[shell command execution disabled by policy]` for user, project, plugin and additional-directory skills (bundled and managed skills are unaffected); it is most useful in managed settings, where users cannot override it.
- **Deeper reasoning.** Including `ultrathink` anywhere in a skill requests deeper reasoning when it runs.

````markdown
## Environment
```!
node --version
git status --short
```
````

The documentation's pull request summary skill combines injection with a forked, read-only agent:

```yaml
---
name: pr-summary
description: Summarize changes in a pull request
context: fork
agent: Explore
allowed-tools: Bash(gh *)
---

## Pull request context
- PR diff: !`gh pr diff`
- PR comments: !`gh pr view --comments`
- Changed files: !`gh pr diff --name-only`

## Your task
Summarize this pull request...
```

**CCAR-F Exercise 2 step 3 in current Claude Code.** The exercise asks for a project skill in `.claude/skills/` "with context: fork and allowed-tools restrictions" and a check that it runs "in isolation without polluting the main conversation context" ([CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The `pr-summary` skill above has that shape. Under the current docs, each line does a specific job: `context: fork` keeps the diff and comments in the subagent's context and returns only its summary; `agent: Explore` picks an agent whose tools are read-only (Write and Edit denied), which is what keeps this skill from editing files; and `allowed-tools: Bash(gh *)` pre-approves the `gh` commands, so the three injected commands pass the permission check instead of aborting the invocation. The check the exercise asks for: only the summary should appear in your main conversation.

### `context: fork`

With `context: fork`, Claude Code starts a new subagent of the type named in `agent` and hands it the skill content as its prompt. The subagent does not see your conversation history, so the skill's instructions must stand on their own. Despite the name, this is not a fork of the current conversation (a conversation fork would carry the history). Only the subagent's result comes back, which is exactly what the CCAR-F guide wants: isolating verbose output, such as codebase analysis, or exploratory context, such as brainstorming, from the main session.

Details that decide answers:

- `context: fork` only makes sense for a skill with explicit task instructions. A skill that holds only guidelines, such as "use these API conventions" in the [skills documentation](https://code.claude.com/docs/en/skills) example, gives the subagent no actionable prompt, and it returns without meaningful output.
- With `agent: Explore` or `agent: Plan`, the subagent skips CLAUDE.md and git status, so it sees only the skill content and the agent's own system prompt.
- The forked subagent runs in the background by default and its result arrives when it finishes. Set `background: false` to wait for it in the same turn. Claude Code also waits without that setting in non-interactive mode (`-p` or the Agent SDK), when `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` is `1`, when an earlier invocation of the same skill is still running, and when a scheduled task fires with the skill as its prompt.
- A forked skill that runs in the background gets the narrower tool set that applies to background subagents. Its subagent is a regular agent type, so the exemption that gives conversation forks the full tool pool (see [Tool control](#tool-control)) does not cover it. Set `background: false` if the skill needs a tool outside that set. Its edits also fall outside your session's checkpoints, so `/rewind` does not undo them; revert with git.

Skills and subagents combine in two directions, and the documentation contrasts them like this:

| Approach | System prompt | Task | Also loads |
|---|---|---|---|
| Skill with `context: fork` | From the agent type | The `SKILL.md` content | CLAUDE.md, per the agent's startup context |
| Subagent with a `skills` field | The subagent's markdown body | Claude's delegation message | The preloaded skills plus CLAUDE.md, per the subagent's startup context |

In both cases the subagent starts without your conversation history. The subagent side is taught in [Subagents](#subagents).

!!! warning "Exam guide vs current docs: forked skills now run in the background"

    The CCAR-F guide describes `context: fork` only as isolation, and that is still correct. As of September 2026, the docs say the fork runs in the background by default in interactive sessions; before v2.1.218, forked skills always blocked the turn until they finished. On the exam, choose `context: fork` when the scenario asks to keep a skill's verbose or exploratory output out of the main conversation.

### `allowed-tools` and `argument-hint`: what the guide says and what the docs say

!!! warning "Exam guide vs current docs: `allowed-tools`"

    The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) (skill 3.2-S3) lists "Configuring allowed-tools in skill frontmatter to restrict tool access during skill execution (e.g., limiting to file write operations to prevent destructive actions)". The [current skills documentation](https://code.claude.com/docs/en/skills) says the opposite about restriction: "It does not restrict which tools are available: every tool remains callable". As of September 2026, `allowed-tools` pre-approves the listed tools for the turn that invokes the skill, and your permission settings still govern every tool that is not listed. To pre-approve tools for a whole session instead, the docs point to allow rules in permission settings.

    To actually remove tools, use one of these:

    - `disallowed-tools` in the skill's frontmatter, which removes tools from Claude's available pool while the skill is active and clears on your next message;
    - `permissions.deny` rules, which block in every permission mode and apply across all skills and prompts;
    - `--tools` on the command line, which restricts the built-in tools available in a session (it does not affect MCP tools).

    Which wording to expect: the CCAR-F guide says exam items are written against its objectives, and none of the four guides mentions `disallowed-tools`. Our reading: an item built on 3.2-S3 would use the guide's framing, with `allowed-tools` as the skill frontmatter for restricting tool access during the skill, so answer in that framing. In a real repository, when a tool must be unavailable during a skill, use `disallowed-tools` or a deny rule.

The documentation's commit skill pre-approves only the git commands it needs, so they run without per-use approval whenever you invoke it:

```yaml
---
name: commit
description: Stage and commit the current changes
disable-model-invocation: true
allowed-tools: Bash(git add *) Bash(git commit *) Bash(git status *)
---
```

!!! danger "Review checked-in skills before you run Claude Code in a repository"

    Workspace trust does not gate a project skill's `allowed-tools`: Claude Code applies it whenever the skill is invoked, including in a `-p` run in a folder you have never trusted. A committed skill can grant itself broad tool access, so read the `allowed-tools` of any skills in a repository you did not write before starting a session there.

!!! warning "Exam guide vs current docs: `argument-hint`"

    The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) (3.2-S4) describes "Using argument-hint frontmatter to prompt developers for required parameters when they invoke the skill without arguments". The [skills documentation](https://code.claude.com/docs/en/skills) defines it as a "Hint shown during autocomplete to indicate expected arguments." Expect the guide's framing on the exam: `argument-hint` is the frontmatter that tells a developer which parameters a skill expects.

### Choosing a skill over the alternatives

- **Skill or CLAUDE.md (3.2-S5).** If every session needs it, CLAUDE.md; if it is a task-specific workflow or reference, a skill. CLAUDE.md cannot trigger a workflow; a skill can, with `/<name>`.
- **Skill or path-scoped rule.** If conventions must apply automatically to files by path (tests spread across the codebase), choose `.claude/rules/` with `paths`; the CCAR-F sample question 6 rationale for rejecting skills is in [Rules, subdirectory CLAUDE.md files or skills?](#rules-subdirectory-claudemd-files-or-skills). A skill's own `paths` field does not change that answer: it only limits when Claude may load the skill automatically, which is still Claude choosing to load it (our reading of the rationale against the frontmatter reference).
- **Skill or hook.** A skill is interpreted by Claude and its outcome can vary; a hook fires on its event every time. Guardrails belong in hooks. See [Hooks](claude-code-workflows.md#hooks).
- **Skill or subagent.** A skill adds to your main context window; a subagent works in its own window and returns a summary. They combine: a subagent can preload skills, and a skill can run in isolation with `context: fork`.
- **Skill and MCP.** MCP connects Claude to an external service; a skill teaches Claude how to use that service well, for example your database schema and query patterns. The wider trade-off (built-in tools, custom tools, Skills, MCP) is in [Built-in tools, custom tools, Skills or MCP](tool-use-and-mcp.md#built-in-tools-custom-tools-skills-or-mcp).

**Traps.**

- Putting a skill at `.claude/skills/deploy.md` instead of `.claude/skills/deploy/SKILL.md`.
- Expecting a skill to apply itself to matching files the way a path-scoped rule does.
- Treating `allowed-tools` as a sandbox in a real deployment. It pre-approves; `disallowed-tools` and deny rules remove.
- Giving a personal skill the same name as the team's and then wondering why the team version never runs for you.
- Using `context: fork` for a guidelines-only skill: the subagent has no task and returns nothing useful.

### Sharing skills, and skills on other surfaces

Share a skill by committing `.claude/skills/`, by shipping a `skills/` directory in a [plugin](#plugins-and-marketplaces), or by deploying it through [managed settings](#managed-settings-for-organizations). Skills behave differently on other surfaces: the platform overview says custom skills do not sync between claude.ai, the Claude API and Claude Code, and that on the API they run with no network access, while in Claude Code they have the same network access as any other program on the machine. The Claude Code skills page adds that terminal sessions signed in with a claude.ai account load that account's enabled skills (v2.1.273 or later), including Anthropic's built-in skills such as `pdf` and `xlsx`; the help center says organization-provisioned skills also load in Claude Code unless `syncClaudeAiSkills` is set to `false` in Claude Code managed settings (the Claude Code page adds that the same key in your user settings stops syncing on one machine). The platform overview still says the pre-built document Skills are not available in Claude Code; for Claude Code, the Claude Code page is the more current source. When a synced skill's short name matches another command, `/<name>` runs the other command and the synced skill runs only as `/anthropic-skills:<name>`. For skills in the Claude apps see [Skills in the Claude apps](claude-for-work.md#skills-in-the-claude-apps).

## Subagents

*Tested in: CCAR-F 1.2-K2, 1.3-K1, 1.3-K2, 1.3-K3, 1.3-S3, 2.3-S1, 3.4-K4, 3.4-S3, 5.4-K3, 5.4-S1, APPX-TECH-3 · CCDV-F D1.1 Agent Architecture, D1.3 Agent Patterns and Frameworks, D3.1 Claude Code Operation (Agents, Agent Memory), D6.1 Context Engineering · CCAR-P 1.4, 3.1, 7.1*

A subagent is a specialized assistant for a side task that would otherwise flood the main conversation. Each one runs in its own context window with a custom system prompt, its own tool access and independent permissions, does the work there, and returns only a summary. The docs list five reasons to use one: preserve context, enforce constraints by limiting tools, reuse configurations, specialize behavior, and control costs by routing work to faster, cheaper models such as Haiku. Define a custom subagent when you keep spawning the same kind of worker with the same instructions. Subagents work inside a single session.

This section covers Claude Code's file-based subagents. The Agent SDK's programmatic form (`AgentDefinition`, the coordinator's tool list) is in [Subagents in the SDK](agents-and-agent-sdk.md#subagents-in-the-sdk); orchestration patterns are in [Multi-agent orchestration](agents-and-agent-sdk.md#multi-agent-orchestration); the context-budget argument is in [Subagents as context isolation](context-engineering.md#subagents-as-context-isolation).

### Built-in subagents

| Subagent | Job | Tools | Model | Loads CLAUDE.md and git status |
|---|---|---|---|---|
| Explore | Fast, read-only search and analysis of a codebase. Claude picks a thoroughness level: quick, medium or very thorough | Read-only; Write and Edit denied | Inherits the main conversation's model, capped at Opus on the Claude API | No |
| Plan | Research during plan mode, so exploration output stays in a separate context window while the main conversation remains read-only | Read-only; Write and Edit denied | Inherits the main conversation's model | No |
| general-purpose | Complex, multi-step tasks that need both exploration and action | Every tool available to subagents | `CLAUDE_CODE_SUBAGENT_MODEL` if you set it and nothing assigns a model another way, otherwise the main conversation's model | Yes |
| claude | Catch-all for a task that fits no more specialized agent; also the default agent for a dispatched background session | Every tool available to subagents | None of its own; follows the subagent model order | Yes |
| statusline-setup | Runs when you use `/statusline` | Not stated in the subagents docs | Sonnet | Yes |
| claude-code-guide | Answers questions about Claude Code features | Not stated in the subagents docs | Haiku | Yes |

Built-in subagents inherit the parent conversation's permissions, and most run with a restricted tool set. Explore and Plan are one-shot: they return no agent ID, so Claude cannot resume them; use general-purpose or a custom subagent when work must continue.

**Explore and the CCAR-F guide (3.4-K4, 3.4-S3).** Claude delegates to Explore when it needs to search or understand a codebase without changing it, which keeps the exploration output out of your main conversation. That is the guide's point: use Explore for verbose discovery phases so the main context window is not exhausted during a multi-phase task. The decision between plan mode and direct execution is taught in [Plan mode or direct execution](claude-code-workflows.md#plan-mode-or-direct-execution).

!!! warning "Older material vs current docs: Explore's model"

    Older material says Explore always runs on Haiku. As of v2.1.198, Explore inherits the main conversation's model; on the Claude API the inherited model is capped at Opus, and on other providers it is inherited directly. A user or project subagent named `Explore` overrides the built-in and keeps its own `model` field, so a definition with `model: haiku` keeps exploration on a lower-cost model. The exam guides do not name Explore's model; they test what Explore is for.

Turning subagents off:

| Goal | Configuration |
|---|---|
| Disable one subagent (built-in or custom) | `"permissions": {"deny": ["Agent(Explore)"]}`, or `claude --disallowedTools "Agent(Explore)"` |
| Remove only built-in Explore and Plan | `CLAUDE_CODE_DISABLE_EXPLORE_PLAN_AGENTS=1` (v2.1.198 or later); Claude then reads and explores files directly |
| Stop all delegation | Deny the `Agent` tool itself |
| Remove every built-in type in `-p` and the Agent SDK | `CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS=1`, then supply only your own |
| Keep fork mode on but stop conversation forks | Deny `Agent(fork)` |

### Subagent files

A subagent is a Markdown file with YAML frontmatter. The markdown body becomes its system prompt, and the subagent receives only that prompt plus basic environment details such as the working directory, not Claude Code's own system prompt. The documentation's code reviewer, saved as `.claude/agents/code-reviewer.md`:

```markdown
---
name: code-reviewer
description: Reviews code for quality and best practices
tools: Read, Glob, Grep
model: sonnet
---

You are a code reviewer. When invoked, analyze the code and provide
specific, actionable feedback on quality, security, and best practices.
```

When two definitions share a name, the higher-priority location wins:

| Priority | Location | Scope |
|---|---|---|
| 1 (highest) | Managed settings (`.claude/agents/` inside the managed settings directory) | Organization-wide |
| 2 | `--agents` CLI flag (JSON) | That session only; not saved to disk |
| 3 | `.claude/agents/` | The project; check it into version control |
| 4 | `~/.claude/agents/` | You, in all your projects |
| 5 (lowest) | A plugin's `agents/` directory | Wherever the plugin is enabled |

- Project subagents are found by walking up from the working directory to the repository root; when nested `.claude/agents/` directories define the same `name`, the one closest to the working directory wins (v2.1.178 or later). Two files with the same `name` in one directory tree load only one of them, chosen by filesystem read order, so keep names unique.
- Claude Code scans agent directories recursively, and in project and user scopes identity comes only from the `name` field, not the subfolder path or the filename. In a plugin, a subfolder does become part of the scoped identifier: `agents/review/security.md` in `my-plugin` registers as `my-plugin:review:security`. Names cannot contain `:`, which is reserved for those plugin-scoped identifiers.
- Only `name` and `description` are required. Field names are camelCase (`maxTurns`, `disallowedTools`), and an unrecognized field is ignored without an error.
- Claude Code watches `~/.claude/agents/` and `.claude/agents/`; an edited definition is used by the next delegation within a few seconds, with no restart. A newly created `agents` directory, and `.claude/agents/` inside an `--add-dir` directory, still need a restart, and a session started with `--disable-slash-commands` does not watch these directories at all.
- Each description sits in context so Claude can decide when to delegate. When the combined descriptions of your custom subagents exceed 15,000 tokens (the docs call this the 15,000-token limit), Claude Code shows a startup warning with the total and still loads every subagent. The fix the docs give: trim the `description` fields and move detail into each system prompt, which loads only when that subagent runs.

A subagent file is less forgiving than a skill. A skill with malformed YAML still loads with no fields set; a project, user or managed subagent file is skipped without any notice in the session when it has no `name`, has a `name` but no `description`, has YAML that does not parse, has an opening `---` that is not the first line, or has a `name` that starts with `-` or contains `:`. A file with no `name`, or whose opening `---` is not the first line, is treated as documentation; for the other cases the reason goes to the debug log, which `--debug` shows. A plugin subagent with no `name` or unparseable frontmatter still loads, under its filename.

!!! warning "Current docs: the `/agents` wizard is gone"

    On v2.1.197 and earlier, `/agents` opened an interactive wizard with a Running tab and a Library tab. As of v2.1.198 (July 1, 2026), `/agents` only prints a reminder: ask Claude to create or manage subagents, or edit `.claude/agents/` or `~/.claude/agents/` directly. Those are the project and user locations in the priority table above.

### Frontmatter fields

| Field | What it controls |
|---|---|
| `name` (required) | Unique identifier, such as `code-reviewer`; hooks receive it as `agent_type`. The filename does not have to match |
| `description` (required) | When Claude should delegate to this subagent |
| `tools` | Allowlist, as a comma-separated string or YAML list. If omitted, the subagent inherits every tool available to subagents. To preload skills, use `skills` rather than listing `Skill` here |
| `disallowedTools` | Tools removed from the inherited or specified list. An entry with a specifier, such as `Bash(git push *)`, still removes the whole tool |
| `model` | `sonnet`, `opus`, `haiku`, `fable`, a full model ID such as `claude-opus-5-5`, or `inherit` |
| `permissionMode` | `default`, `acceptEdits`, `auto`, `dontAsk`, `bypassPermissions`, `plan`, or `manual` (alias for `default`, v2.1.200 or later) |
| `maxTurns` | Maximum agentic turns; at the limit the output comes back marked as partial and Claude can resume the subagent |
| `skills` | Skills whose full content is preloaded at startup. It does not limit access: the subagent can still invoke unlisted skills through the Skill tool. Skills with `disable-model-invocation: true` cannot be preloaded |
| `mcpServers` | Name of an already-configured server, or an inline server definition |
| `hooks` | Lifecycle hooks scoped to this subagent |
| `memory` | Persistent memory scope: `user`, `project` or `local` |
| `background` | `true` keeps the subagent in the background even when Claude asks to run it in the foreground |
| `effort` | `low`, `medium`, `high`, `xhigh` or `max`, overriding the session; available levels depend on the model |
| `isolation` | `worktree` runs it in a temporary git worktree, branched by default from your default branch rather than the parent session's `HEAD`, and removed automatically if the subagent makes no changes |
| `omitClaudeMd` | `true` skips the user, project and local CLAUDE.md files (managed policy files still load); ignored when the agent runs as the main session agent; v2.1.271 or later |
| `color` | Display color in the task list and transcript: `red`, `blue`, `green`, `yellow`, `purple`, `orange`, `pink` or `cyan` |
| `initialPrompt` | A first user turn auto-submitted when the agent runs as the main session agent (`--agent` or the `agent` setting); ignored for plugin subagents |
| `experimental` | Map of experimental options; its `cacheTtl` key takes `5m` or `1h` for this subagent's prompt cache lifetime (v2.1.248 or later) |

The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)'s "AgentDefinition configuration including descriptions, system prompts, and tool restrictions" (1.3-K3) is the SDK form of the same three things: `description`, the system prompt (the file body here, `prompt` in JSON and in `AgentDefinition`) and `tools`. The `--agents` flag takes the same three in JSON, with each top-level key as the agent's name and `prompt` standing in for the markdown body. It also accepts most frontmatter fields (`disallowedTools`, `model`, `permissionMode`, `mcpServers`, `hooks`, `maxTurns`, `skills`, `initialPrompt`, `memory`, `effort`, `background`, `omitClaudeMd`, `isolation`); `color` and `experimental` are ignored there. The documentation's example:

```bash
claude --agents '{
  "code-reviewer": {
    "description": "Expert code reviewer. Use proactively after code changes.",
    "prompt": "You are a senior code reviewer. Focus on code quality, security, and best practices.",
    "tools": ["Read", "Grep", "Glob", "Bash"],
    "model": "sonnet"
  },
  "debugger": {
    "description": "Debugging specialist for errors and test failures.",
    "prompt": "You are an expert debugger. Analyze errors, identify root causes, and provide fixes."
  }
}'
```

### Tool control

- **Allowlist.** `tools: Read, Grep, Glob, Bash` gives a subagent that cannot edit files, write files or use any MCP tool. In Claude Code terms (our mapping), this is how you apply CCAR-F 2.3-S1 (restrict each subagent's tools to its role).
- **Denylist.** `disallowedTools: Write, Edit` inherits the pool minus those two, so the subagent keeps Bash, MCP tools and the rest. Both fields also take MCP server patterns: `mcp__<server>` or `mcp__<server>__*` covers every tool of that server, and `mcp__*` in `disallowedTools` removes every MCP tool.
- **Both lists set.** `disallowedTools` is applied first, then `tools` is resolved against what remains; a tool named in both is removed.
- **Blocking one command, not the tool.** Because a `disallowedTools` entry such as `Bash(git push *)` removes all of Bash, keep Bash and add `Bash(git push *)` to `permissions.deny` in settings instead; that rule applies to the main conversation and to subagents.
- **Always removed from subagents,** even when listed in `tools`: `AskUserQuestion`, `EndConversation`, `EnterPlanMode`, `ExitPlanMode` (unless `permissionMode` is `plan`), `ScheduleWakeup`, `WaitForMcpServers`, `Workflow`, and `Agent` once the depth limit is reached.
- **Background subagents get fewer built-in tools.** A subagent running in the background (the default) keeps every MCP tool but, apart from `Agent` and `ExitPlanMode` (which follow the rules above), only these built-in tools: `Read`, `Grep`, `Glob`, `LSP`, `Bash`, `PowerShell`, `Edit`, `Write`, `NotebookEdit`, `WebFetch`, `WebSearch`, `TodoWrite`, `Skill`, `ToolSearch`, `EnterWorktree`, `ExitWorktree`, `Monitor`, `TaskStop`, `SendMessage` and `Artifact`, plus `SubagentHandback` for a subagent that reports through it. The same definition can therefore resolve to different tools in the foreground and the background. Forks skip both filters and get the main conversation's exact tool pool.
- **Which subagents a main agent may spawn.** When an agent runs as the main session with `claude --agent`, `tools: Agent(worker, researcher), Read, Bash` limits the subagent types it can spawn; leaving `Agent` out entirely means it cannot spawn any. In an ordinary subagent definition, the type list inside the parentheses is ignored.
- **MCP servers scoped to one subagent.** Defining a server inline in `mcpServers` keeps its tool descriptions out of the main conversation's context. Inline servers from a project's `.claude/agents/` load only after you trust the folder the agent file came from.
- **Conditional rules.** A `PreToolUse` hook in the subagent's frontmatter can validate each call, for example allowing only read-only SQL. Frontmatter hooks run only while that subagent is active, and a `Stop` hook there is converted to `SubagentStop`. Hooks in `settings.json` also fire inside subagents, and `SubagentStart` and `SubagentStop` fire at start and finish. Frontmatter hooks of a project subagent run only after you accept the workspace trust dialog for its folder; a `-p` session does not count.
- **Plugin subagents** ignore `hooks`, `mcpServers` and `permissionMode` for security reasons.

The documentation's read-only database subagent runs a validation script before every Bash call:

```yaml
---
name: db-reader
description: Execute read-only database queries
tools: Bash
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate-readonly-query.sh"
---
```

**Permission mode inheritance.** If `permissionMode` is unset, the subagent runs in the main conversation's mode (how that mode is chosen at launch is in [Setting and switching the mode](#setting-and-switching-the-mode)). When the parent is in `bypassPermissions`, `acceptEdits` or `auto`, the subagent runs in that same mode and its own `permissionMode` is ignored; in auto mode the classifier checks the delegated task at spawn, each action while the subagent runs, and its final report. When the parent is in `default`, `dontAsk` or `plan`, the subagent uses the mode it sets, except that a subagent declaring `bypassPermissions` keeps the parent's mode (v2.1.267 or later).

!!! warning "Exam guide vs current docs: Task tool or Agent tool"

    The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) (1.3-K1) names "The Task tool as the mechanism for spawning subagents" and says `allowedTools` must include "Task" for a coordinator to invoke subagents. The [subagents documentation](https://code.claude.com/docs/en/sub-agents) says: "In version 2.1.63, the Task tool was renamed to Agent." Existing `Task(...)` references in settings and agent definitions still work as aliases. Answer exam items in the guide's vocabulary ("Task"), and read `Agent(...)` in current configuration as the same tool.

### Choosing the model

A subagent's model is resolved in this order: the per-invocation `model` parameter, then the frontmatter `model`, then `CLAUDE_CODE_SUBAGENT_MODEL`, then the main conversation's model. Before v2.1.251, `CLAUDE_CODE_SUBAGENT_MODEL` came first and overrode the other two, so older material may describe it as an override. Setting that variable alone does not change the model of the built-in Explore and Plan subagents. To force one model onto every subagent, set `CLAUDE_CODE_SUBAGENT_MODEL` and also `CLAUDE_CODE_SUBAGENT_MODEL_FORCE=1` (v2.1.257 or later); a fork still runs on the main conversation's model. An organization's `availableModels` allowlist also applies to subagent models: a blocked family alias runs on the newest version of that family the allowlist permits where that substitution applies, and otherwise the subagent falls back to the inherited model (see [Models and output styles](#models-and-output-styles)).

### Delegation

- **Automatic.** Claude decides from your request, each subagent's `description` and the current context. Phrases such as "use proactively" in a description encourage delegation ([subagents documentation](https://code.claude.com/docs/en/sub-agents)).
- **Explicit.** Name the subagent in natural language (Claude still decides), @-mention it (guarantees it runs for one task), or run a whole session as it with `--agent <name>` or the `agent` setting. An @-mention picks the subagent, not its prompt: your full message still goes to Claude, which writes the subagent's task prompt.
- **Session-wide agents.** With `--agent`, the subagent's system prompt replaces Claude Code's default system prompt entirely, the way `--system-prompt` does; CLAUDE.md still loads. Setting `agent` in `.claude/settings.json` makes it the default for every session in the project, and the CLI flag overrides it.
- **Foreground or background.** A foreground subagent blocks the main conversation until it finishes. Background subagents run concurrently and raise their permission prompts in the main session, naming the subagent that asks. Claude Code picks the mode from the first rule that applies: a subagent spawned by an in-process agent team teammate runs in the foreground; with `CLAUDE_CODE_DISABLE_BACKGROUND_TASKS` set to `1`, subagents run in the foreground in every kind of session, whether or not fork mode is on; where fork mode is on (the default in interactive sessions), Claude Code runs the subagents Claude spawns in the background and Claude cannot ask for the foreground; where it is off (`-p` and the Agent SDK by default), Claude runs them in the background by default and in the foreground when it needs the result first. Press `Ctrl+B` to background a running task.
- **Limits.** By default a subagent can spawn subagents up to three layers below the main conversation (`CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH`; `1` turns nesting off), and spawning fails with `Concurrent subagent limit reached` when 20 are running (`CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS`, v2.1.217 or later). There is no limit on the total number of subagents over a session.

Parallel research works best when the research paths do not depend on each other; for multi-step work, ask Claude to use subagents in sequence so each result feeds the next. Many subagents that each return detailed results can still fill the main context, and running several at once multiplies token usage. The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) skill of spawning parallel subagents "by emitting multiple Task tool calls in a single coordinator response" (1.3-S3) is the SDK-level view of the same fan-out; see [Multi-agent orchestration](agents-and-agent-sdk.md#multi-agent-orchestration).

### What a subagent can see

The [subagents documentation](https://code.claude.com/docs/en/sub-agents) is explicit: "Each subagent starts with a fresh, isolated context window. It doesn't see your conversation history, the skills you've already invoked, or the files Claude has already read." Claude writes a delegation message that summarizes the task. The main conversation still has your full CLAUDE.md when it reads the subagent's result, so most rules need not reach the subagent; if one must, such as "ignore the `vendor/` directory", restate it in the prompt you give Claude when delegating.

| Loaded at subagent startup | Not loaded |
|---|---|
| Its own system prompt plus environment details (not Claude Code's) | Your conversation history |
| The task message Claude writes when delegating | Skills invoked earlier in the main conversation, and files Claude already read |
| Every CLAUDE.md level the main conversation loads (none for Explore and Plan; with `omitClaudeMd`, only managed policy files, or none for a managed subagent) | The main conversation's auto memory |
| A git status snapshot (skipped only by Explore and Plan) | The main conversation's output style |
| Full content of skills listed in `skills` | |

A subagent's context window is also sized by its own model, not the parent's, so delegating to a model with a smaller window gives that subagent the smaller window.

This is what CCAR-F 1.2-K2 and 1.3-K2 test: subagents do not inherit the coordinator's history, so context must be passed explicitly in the prompt. A fork is the one exception (below). The guide's 1.3-K2 also says subagents do not "share memory between invocations", and that matches the default here: each invocation is a new instance. Resuming one instance with `SendMessage` and the opt-in `memory` field (below) are Claude Code features layered on top; in our reading they do not change the exam answer that the coordinator passes context in the prompt.

Since v2.1.210, Claude Code also scans each subagent's final report before Claude reads it: text that imitates Claude Code's own output gets a backslash inserted, and a marker line flags reports that imitate tags or mention permission settings. The scan does not judge whether content is malicious and is not a substitute for restricting what a subagent can reach.

### Resuming subagents and forks

- Each invocation creates a new instance. A resumed subagent keeps its full history of tool calls, results and reasoning; Claude resumes it with the `SendMessage` tool, passing the agent's ID or name as `to`, and the resumed run continues in the background without a new `Agent` call.
- Transcripts live at `~/.claude/projects/{project}/{sessionId}/subagents/`, one `agent-{agentId}.jsonl` per subagent, and main-conversation compaction does not touch them. They are deleted after the `cleanupPeriodDays` retention period (30 days by default). Subagents auto-compact with the same logic as the main conversation.
- A **fork** is a subagent that inherits the entire conversation so far (same system prompt, tools, model and history); only its final result comes back. Start one with `/subtask <task>` (v2.1.212 or later; the command was `/fork` on v2.1.161 through v2.1.211). When agent view is off, `/subtask` is not available and `/fork` starts the forked subagent; otherwise `/fork` copies the whole session into a new background session. A fork's first request reuses the parent's prompt cache, so it is cheaper than a fresh subagent that needs the same context. A fork cannot spawn further forks. Fork mode is on by default in interactive sessions (v2.1.232 or later) and off in `-p` and the Agent SDK; `CLAUDE_CODE_FORK_SUBAGENT` set to `1` or `0` overrides that.

Forking a whole session (`--fork-session`, `/branch`, and the SDK's `fork_session`) is a different feature, taught in [Sessions: continue, resume, fork and rewind](claude-code-workflows.md#sessions-continue-resume-fork-and-rewind).

### Agent memory (CCDV-F D3.1)

The `memory` field gives a subagent a persistent directory that survives across conversations, for patterns, debugging insights and architectural decisions.

| Scope | Directory | Use when |
|---|---|---|
| `user` | `~/.claude/agent-memory/<name-of-agent>/` | The knowledge applies across all your projects |
| `project` | `.claude/agent-memory/<name-of-agent>/` | The knowledge is project-specific and should be shared through version control (the recommended default) |
| `local` | `.claude/agent-memory-local/<name-of-agent>/` | The knowledge is project-specific but should stay out of version control |

Subagent memory is part of auto memory. If auto memory is off (the `autoMemoryEnabled` setting or `CLAUDE_CODE_DISABLE_AUTO_MEMORY`), the `memory` field has no effect. When it is on, the subagent's system prompt includes instructions for its memory directory and the first 200 lines or 25KB of its own `MEMORY.md`, whichever comes first, and Read, Write and Edit are enabled so it can manage those files. The docs suggest asking the subagent to check its memory before a task and to save what it learned afterward.

### Decide and traps

**Decide.**

- If a side task produces large output (test runs, log searches, documentation fetches) and can return a summary, use a subagent; the verbose output stays in its context.
- If the main agent must keep the high-level picture while specific questions get answered (the [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)'s examples are "find all test files," "trace refund flow dependencies"), spawn a subagent per question and let the main agent coordinate (5.4-K3, 5.4-S1).
- If the task needs frequent back-and-forth, shares context across phases, is a quick targeted change, or latency matters, stay in the main conversation.
- If you want a reusable prompt or workflow that runs in the main conversation's context, write a [skill](#agent-skills) instead.
- If a worker should only read, restrict it with a `tools` allowlist; limiting tool access is one of the documented best practices, alongside focused design, descriptions that single out one subagent, and checking project subagents into version control.

**Capability bloat (CCAR-P 3.1).** The CCAR-P guide asks you to evaluate tool and agent configuration for capability bloat. The Claude Code levers for that (our mapping onto the objective) are all documented above: grant each subagent only the tools its role needs; keep descriptions short, since every custom description sits in context and the startup warning fires past 15,000 tokens; move detail into the system prompt, which loads only when that subagent runs; and define an MCP server inline in the one subagent that needs it, so its tool descriptions stay out of the main conversation.

**Traps.**

- Assuming a subagent already knows what the main conversation discussed or which files were read.
- Expecting Explore or Plan to follow rules that live only in CLAUDE.md; they skip it.
- Planning to resume an Explore run; it is one-shot.
- Putting `hooks`, `mcpServers` or `permissionMode` in a plugin subagent and expecting them to apply.
- Adding `Bash(git push *)` to `disallowedTools` to block one command; it removes all of Bash from the subagent.
- Treating the `skills` field as an access limit; it only preloads, and the subagent can still invoke other skills.
- Trusting older material that says subagents cannot spawn their own subagents. The default has changed across versions (five layers on v2.1.172 through v2.1.216, one on v2.1.217 and v2.1.218); since v2.1.219 it is three.

## Plugins and marketplaces

*Tested in: CCDV-F D2.5 Claude Application Design (plugin management), D2.6 Configuration Management (plugin dependencies) · CCAR-P 7.1 (configuring Claude tools and environments for teams) · of the four guides, only the CCDV-F guide names plugins*

A plugin is a self-contained directory of components that extends Claude Code: skills, agents, hooks, MCP servers, LSP servers and monitors. It is the packaging layer: it bundles what you already know how to configure into one installable unit, so you can reuse a setup across repositories or distribute it through a marketplace. The documented trigger is simple: when a second repository needs the same setup, package it as a plugin.

| Approach | Skill names | Best for |
|---|---|---|
| Standalone `.claude/` directory | `/hello` | Personal workflows, project-specific customizations, quick experiments |
| Plugin | `/plugin-name:hello` | Sharing with teammates, distributing to the community, versioned releases, reuse across projects |

Start standalone for quick iteration and convert to a plugin when you are ready to share. Plugin skills are always namespaced, which prevents conflicts when two plugins ship a skill with the same name; the prefix comes from `name` in `plugin.json`. Because of the namespace, a plugin skill never replaces a same-named standalone skill: both stay available. Plugin skills are model-invoked: Claude uses them automatically based on the task context.

### Plugin layout

| At the plugin root | Holds |
|---|---|
| `.claude-plugin/plugin.json` | The manifest. Only `plugin.json` goes inside `.claude-plugin/` |
| `skills/` | Skills as `<name>/SKILL.md` directories |
| `commands/` | Skills as flat Markdown files; use `skills/` for new plugins |
| `agents/` | Custom agent (subagent) definitions |
| `hooks/hooks.json` | Hook handlers, in the same format as hooks in `settings.json` |
| `.mcp.json` | MCP server configurations |
| `.lsp.json` | LSP servers for code intelligence |
| `monitors/` | Background monitor configurations |
| `output-styles/` | Output styles |
| `bin/` | Executables added to the Bash tool's `PATH` while the plugin is enabled |
| `settings.json` | Defaults applied when the plugin is enabled; only the `agent` and `subagentStatusLine` keys are supported |

The plugin root is the plugin's own directory, never `~/.claude/`: Claude Code does not read a `.mcp.json` placed at `~/.claude/.mcp.json`. A plugin hook file uses the settings format under a top-level `hooks` key:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [{ "type": "command", "command": "jq -r '.tool_input.file_path' | xargs npm run lint:fix" }]
      }
    ]
  }
}
```

### The `plugin.json` manifest

The manifest is optional. Without one, Claude Code discovers components in their default locations and takes the plugin name from the directory name. If you include one, `name` is the only required field (kebab-case, no spaces), and it namespaces every component: an agent `agent-creator` in plugin `plugin-dev` appears as `plugin-dev:agent-creator`.

```json
{
  "name": "my-first-plugin",
  "description": "A greeting plugin to learn the basics",
  "version": "1.0.0",
  "author": {
    "name": "Your Name"
  }
}
```

| Group (our grouping) | Fields |
|---|---|
| Metadata | `$schema`, `displayName`, `version`, `description`, `author`, `homepage`, `repository`, `license`, `keywords`, `metadata`, `defaultEnabled` (default `true`) |
| Component paths | `skills` (adds to the default scan), `commands`, `agents`, `workflows`, `hooks`, `mcpServers`, `outputStyles`, `lspServers`, `experimental.themes`, `experimental.monitors`, `experimental.evals` |
| Other fields in the same reference table | `userConfig`, `channels`, `dependencies` |

- **`version` is a release gate.** Setting it pins the plugin: users receive updates only when you bump it (a `command` source and a plugin loaded in place are the exceptions). Pushing new commits without a bump has no effect, and `/plugin update` reports that the plugin is already at the latest version. If both `plugin.json` and the marketplace entry set a version, `plugin.json` wins.
- **Without `version`.** Claude Code takes the marketplace entry's `version`, then the git commit SHA for `github`, `url`, `git-subdir` and git-hosted relative-path sources, then the SHA-256 digest for `archive` sources; `npm` sources resolve to `unknown`. Omitting `version` everywhere gives commit-SHA versioning, where users get an update whenever the resolved commit changes; the docs suggest it for internal or team plugins under active development.
- **Unknown fields are ignored.** `claude plugin validate` reports them as warnings; `--strict` turns warnings into errors.
- **Path variables.** `${CLAUDE_PLUGIN_ROOT}` is the installation directory; `${CLAUDE_PLUGIN_DATA}` is a persistent directory that survives updates (`~/.claude/plugins/data/{id}/`); `${CLAUDE_PROJECT_DIR}` is the project root.
- **Cache.** Marketplace plugins are copied into `~/.claude/plugins/cache` (unless loaded in place); old versions are removed roughly 14 days after an update or uninstall.

### Build, test and install

```bash
claude --plugin-dir ./my-first-plugin                           # load a plugin under development, no install
claude plugin validate ./my-plugin --strict                     # check syntax and schema; fail on warnings too
claude plugin install formatter@my-marketplace --scope project  # record it in .claude/settings.json for the team
```

`--plugin-dir` also accepts a `.zip` and can be repeated; `/reload-plugins` picks up changes without a restart. `claude plugin init my-tool` scaffolds `~/.claude/skills/my-tool/` with a manifest, which loads as `my-tool@skills-dir` with no marketplace or install step. Inside a session, `/plugin` opens the plugin menu or takes a subcommand such as `list`, `install`, `enable` or `disable`. From the shell, `claude plugin` (alias `claude plugins`) manages plugins.

| Install scope | Written to | Use |
|---|---|---|
| `user` (default) | `~/.claude/settings.json` | Personal plugins across all projects |
| `project` | `.claude/settings.json` | Team plugins shared through version control |
| `local` | `.claude/settings.local.json` | Project-specific plugins for you only |
| `managed` | Managed settings | Managed plugins; read-only, update only |

Installing writes to `enabledPlugins`, an object keyed `plugin-name@marketplace-name` with a Boolean value. Project settings take precedence over user settings here, so to opt out of a project-enabled plugin on your machine, set it to `false` in `.claude/settings.local.json`. A managed `false` blocks installation at every scope.

### Managing installed plugins

| Task | Command |
|---|---|
| List installed plugins with version, source marketplace and enable status | `claude plugin list` |
| Turn a plugin off without uninstalling it, or back on | `claude plugin disable <plugin>`, `claude plugin enable <plugin>` |
| Update a plugin to the latest version | `claude plugin update <plugin>` |
| Remove a plugin | `claude plugin uninstall <plugin>` |
| Remove auto-installed dependencies that no installed plugin needs any more | `claude plugin prune` |
| Load plugin changes into the running session | `/reload-plugins` |

Marketplaces can also auto-update in the background after startup. `claude-plugins-official`, most other official Anthropic marketplaces and marketplaces added from claude.ai have auto-update on by default; other third-party and local marketplaces have it off. You toggle it per marketplace in `/plugin`, and administrators can set `"autoUpdate": true` on an `extraKnownMarketplaces` entry in managed settings. The `DISABLE_AUTOUPDATER` environment variable turns off automatic updates for Claude Code and for marketplace plugins.

### Marketplaces

A marketplace is a catalog of plugins. Using one takes two steps: add the marketplace, then install individual plugins from it.

- The official Anthropic marketplace, `claude-plugins-official`, is added automatically the first time you start Claude Code interactively; otherwise run `/plugin marketplace add anthropics/claude-plugins-official`. Install from it with `/plugin install <name>@claude-plugins-official`.
- `/plugin marketplace add` accepts a GitHub `owner/repo`, a git URL, a local path, a remote URL to a `marketplace.json`, or a claude.ai-hosted marketplace.
- `/plugin install plugin-name@marketplace-name` lets you pick User, Project (adds the plugin to `.claude/settings.json`) or Local scope.
- Names such as `claude-plugins-official`, `claude-code-marketplace`, `anthropic-plugins` and `agent-skills` are reserved for Anthropic; third-party marketplaces cannot use them.
- Plugins submitted through the submission forms go to the community marketplace (`claude-community`), not the official one; Anthropic curates `claude-plugins-official` at its discretion and it has no application process.

A marketplace is defined by `.claude-plugin/marketplace.json` at the repository root. The required fields are `name`, `owner` and `plugins`, and each plugin entry needs at least a `name` and a `source`. Sources can be a relative path (`./...`), `github` (`repo`, optional `ref` and `sha`), `url`, `git-subdir`, `npm`, `archive` or `command`.

```json
{
  "name": "company-tools",
  "owner": {
    "name": "DevTools Team",
    "email": "devtools@example.com"
  },
  "plugins": [
    {
      "name": "code-formatter",
      "source": "./plugins/formatter",
      "description": "Automatic code formatting on save",
      "version": "2.1.0",
      "author": {
        "name": "DevTools Team"
      }
    },
    {
      "name": "deployment-tools",
      "source": {
        "source": "github",
        "repo": "company/deploy-plugin"
      },
      "description": "Deployment automation tools"
    }
  ]
}
```

**Team marketplaces.** Put `extraKnownMarketplaces` (and `enabledPlugins` for the plugins the team should use) in the project's committed `.claude/settings.json`. After a teammate trusts the folder, Claude Code adds the marketplace without a further prompt; repository entries are honored only after workspace trust. Adding the marketplace does not install its plugins from external sources: as of Claude Code v2.1.195, a plugin that only the project's `.claude/settings.json` enables, and that comes from an external source such as a GitHub repository or npm package, does not load until each teammate installs it once. Until then, Claude Code reports it as not installed and shows the `claude plugin install` command to run. The keys below are trimmed from the documentation's example of a team's committed `.claude/settings.json`:

```json
{
  "extraKnownMarketplaces": {
    "acme-tools": {
      "source": {
        "source": "github",
        "repo": "acme-corp/claude-plugins"
      }
    }
  },
  "enabledPlugins": {
    "code-formatter@acme-tools": true
  }
}
```

### Plugin dependencies (CCDV-F D2.6)

A plugin can depend on other plugins, listed in the `dependencies` array of `plugin.json` or in its marketplace entry. Each entry is a bare name or an object with `name` (required), `version` (a semver range such as `~2.1.0`, `^2.0`, `>=1.4` or `=2.1.0`) and an optional `marketplace`.

```json
{
  "name": "deploy-kit",
  "version": "3.1.0",
  "dependencies": [
    "audit-logger",
    { "name": "secrets-vault", "version": "~2.1.0" }
  ]
}
```

- **Unconstrained means latest.** By default a dependency tracks the latest available version, so an upstream release can change it under your plugin without warning. A version range holds it at a tested range until you choose to move.
- **Installed automatically.** Installing a plugin resolves and installs its declared dependencies, except a dependency whose marketplace entry has a `command` source or a `headersHelper`; you install those yourself first.
- **Same marketplace by default.** Claude Code refuses to auto-install a dependency from a different marketplace unless the root marketplace lists that marketplace in `allowCrossMarketplaceDependenciesOn` in its `marketplace.json`.
- **Enable and disable follow the graph.** `claude plugin enable` turns on a marketplace plugin's dependencies at the same scope; `claude plugin disable` fails while another enabled plugin depends on the target.
- **Bundles.** A manifest with only `name` and a `dependencies` array packages a curated plugin set behind one install.
- **Pre-releases are opt-in.** Versions such as `2.0.0-beta.1` are excluded unless the range opts in, for example `^2.0.0-0`.

### Trust and organizational control

The [plugin discovery documentation](https://code.claude.com/docs/en/discover-plugins) warns: "Plugins and marketplaces are highly trusted components that can execute arbitrary code on your machine with your user privileges." Install only from sources you trust. Two details to know: plugin subagents ignore their `hooks`, `mcpServers` and `permissionMode` fields (see [Tool control](#tool-control)), and MCP tools from a plugin carry the full name `mcp__plugin_<plugin-name>_<server-name>__<tool-name>`.

Administrators control plugins through [managed settings](#managed-settings-for-organizations):

| Managed key | Effect |
|---|---|
| `strictKnownMarketplaces` | Unset: no restriction. `[]`: complete lockdown, including the official marketplace. A list: only those marketplace sources |
| `blockedMarketplaces` | Blocklist of marketplace sources, checked before anything is downloaded |
| `enabledPlugins` with `false` | Blocks installation of that plugin at every scope |
| `strictPluginOnlyCustomization` | Blocks skills, agents, hooks and MCP servers from user and project sources, so they can only come from plugins or managed settings; `true` locks all four, an array names which |
| `disableSideloadFlags` | Rejects `--plugin-dir`, `--plugin-url`, `--agents` and `--mcp-config` at startup |
| `disableCommandPluginSources` | Blocks plugins whose marketplace entry installs by running a command |

All of these except `enabledPlugins` are read only from a managed source; placing them in user or project settings has no effect. `strictKnownMarketplaces` matches the marketplace a plugin comes from, not the entries inside it, so users can still install a plugin with a `command` source from an allowed marketplace; to block those as well, set `disableCommandPluginSources`. It also does not cover plugins synced from claude.ai; `syncClaudeAiPlugins: false` in managed settings stops those.

```json
{
  "strictKnownMarketplaces": [
    {
      "source": "github",
      "repo": "anthropics/claude-plugins-official"
    }
  ]
}
```

On Enterprise plans, the help center describes giving skills to only some users by bundling them into a plugin and assigning that plugin to a group. For plugin evals run in CI with `claude plugin eval`, the docs advise pinning the model under test with `--model` so that a model rollout is not mistaken for a plugin regression.

### Decide and traps

**Decide** (our decision rules, each following from the documented behavior above).

- If one repository's team needs a skill, subagent or hook, commit it under `.claude/`. If several repositories need the same setup, or you want versioned releases, package a plugin and publish it in a marketplace.
- If a plugin depends on another plugin's behavior, declare the dependency with a version range rather than relying on the latest release.
- If the organization must control where plugins come from, use `strictKnownMarketplaces` in managed settings rather than asking developers not to add marketplaces.

**Traps** (our list; the CCDV-F guide names plugin management and plugin dependencies but no plugin anti-patterns, so each trap follows from a documented rule above).

- Putting `skills/`, `agents/`, `commands/` or `hooks/` inside `.claude-plugin/`; only `plugin.json` belongs there.
- Pushing new commits without bumping `version` and expecting users to receive them.
- Expecting a plugin skill to be invoked as `/hello`; it is `/plugin-name:hello`.
- Editing the team's `.claude/settings.json` to switch off a project-enabled plugin for yourself, instead of setting it to `false` in `.claude/settings.local.json`.
- Putting `strictKnownMarketplaces` in a project's `.claude/settings.json` and expecting it to restrict anything.
- Treating a marketplace as vetted by default; plugins run code with your privileges.

## MCP servers in Claude Code

*Tested in: CCAR-F 2.4-K1, 2.4-K2, 2.4-K3, 2.4-S1, 2.4-S2, Exercise 2 step 4, APPX-INSCOPE-6 · CCDV-F D8.2 MCP Server Development (integration with Claude applications) · CCAR-P 3.7, 7.1*

An MCP server gives Claude Code tools and data from an external system, and you choose a scope for each server you add: one project, every project of yours, or the whole team through a committed file. This section covers where that configuration lives and how it is shared. Transports, OAuth, output limits and tool search tuning are taught in [MCP in Claude Code](tool-use-and-mcp.md#mcp-in-claude-code); writing a server is taught in [Building an MCP server](tool-use-and-mcp.md#building-an-mcp-server).

### Three scopes

| Scope | Stored in | Who gets the server |
|---|---|---|
| Local (the default) | `~/.claude.json`, under the project's path | You, in the current project only |
| Project | `.mcp.json` at the project root | Everyone on the repository, once the file is committed |
| User | `~/.claude.json`, under the top-level `mcpServers` key | You, in all your projects |

When the same server is defined in more than one place, Claude Code connects to it once, using this order: local, project, user, plugin-provided servers, then claude.ai connectors. The three scopes match duplicates by name; plugins and connectors match by endpoint (the same URL or command). The whole entry from the winning source is used; fields are not merged across scopes. A server the organization provides through the `managedMcpServers` managed setting ranks above all of these (v2.1.259 or later). The MCP local scope is not the same thing as the general local settings file `.claude/settings.local.json`.

!!! warning "Exam guide vs current docs: two scopes or three"

    The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) (2.4-K1) contrasts "project-level (.mcp.json) for shared team tooling vs user-level (~/.claude.json) for personal/experimental servers". The docs add a third scope, local, which is the default, is also stored in `~/.claude.json`, and is the scope the [MCP documentation](https://code.claude.com/docs/en/mcp) recommends for "personal development servers, experimental configurations, or servers with credentials you don't want in version control". Answer in the guide's terms: shared servers go in the committed `.mcp.json`, personal and experimental ones in `~/.claude.json` (user scope when they should follow you into every project). The practical catch is that `claude mcp add` without `--scope user` does not make a server available in all your projects.

### Adding servers

```bash
claude mcp add --transport http notion https://mcp.notion.com/mcp                 # remote HTTP (local scope by default)
claude mcp add --env AIRTABLE_API_KEY=YOUR_KEY --transport stdio airtable \
  -- npx -y airtable-mcp-server                                                   # stdio: server command after --
claude mcp add --transport http shared-server --scope project https://example.com/mcp   # writes .mcp.json
claude mcp add --transport http hubspot --scope user https://mcp.hubspot.com/anthropic  # all projects
claude mcp list
claude mcp reset-project-choices # reset .mcp.json approvals
```

HTTP is the recommended transport for remote servers, and the SSE transport is deprecated. For stdio servers, everything after `--` goes to the server command untouched. `claude mcp list`, `claude mcp get <name>` and `claude mcp remove <name>` manage servers from the shell; inside a session, `/mcp` shows server status and handles OAuth sign-in. Those two views are how you confirm that a project server and a personal server are both available at once, the check Exercise 2 step 4 asks for.

### Sharing a server without sharing its secret

CCAR-F 2.4-K2 and 2.4-S1 test environment variable expansion in `.mcp.json`: commit the file so the team gets the same servers, and keep the credential in each developer's environment.

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

- `${VAR}` expands to the variable; `${VAR:-default}` uses `default` when `VAR` is unset.
- Expansion works in `command`, `args`, `env`, `url` and `headers`.
- A variable that is unset and has no default does not stop the config loading: Claude Code warns and leaves the literal `${VAR}` text in place.
- In a remote server's `url` and `headers`, credential variables such as `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` and `AWS_BEARER_TOKEN_BEDROCK` read as empty, so a project `.mcp.json` cannot send your Claude Code credentials to a server it names.
- A JSON entry with a `url` but no `type` is a configuration error, because an entry with no `type` is read as a stdio server.

### Approval of project servers

In interactive sessions, Claude Code asks for approval before using servers from a project's `.mcp.json`, and a cloned, untrusted repository cannot approve its own servers: approvals committed to its `.claude/settings.json` are ignored until the workspace trust dialog is accepted. In `claude -p` runs, Agent SDK sessions and cloud sessions no prompt can be shown, so project servers load without asking. To keep a project server out of such a run, add it to `disabledMcpjsonServers`, exclude project settings with `--setting-sources`, or start with `--strict-mcp-config`. The settings `enableAllProjectMcpServers`, `enabledMcpjsonServers` and `disabledMcpjsonServers` pre-record the choice; a rejection takes precedence over both approval keys.

### Details that decide answers

- **Where the config goes.** `settings.json` does not read an `mcpServers` key. Project servers go in `.mcp.json` at the repository root.
- **Permission rule names.** MCP tools are named `mcp__<server>__<tool>`; the rule forms that match one server's tools or a single tool are in [Rule syntax](#rule-syntax).
- **Scoped to one subagent.** Defining a server inline in a subagent's `mcpServers` keeps its tool descriptions out of the main conversation; see [Tool control](#tool-control).
- **Organization control.** `managed-mcp.json`, `managedMcpServers`, `allowedMcpServers` and `deniedMcpServers` are covered in [Managed settings for organizations](#managed-settings-for-organizations).

!!! warning "Exam guide vs current docs: when MCP tools are loaded"

    CCAR-F 2.4-K3 says "That tools from all configured MCP servers are discovered at connection time and available simultaneously to the agent" ([CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). As of September 2026, tool search is on by default in Claude Code: "Only tool names and server instructions load at session start" ([MCP documentation](https://code.claude.com/docs/en/mcp)), and full definitions load when Claude needs a tool. Claude Code turns tool search off by itself in a few setups, for example when `ANTHROPIC_BASE_URL` points to a non-first-party host. The exam statement is still right that every configured server's tools are available at once; what changed is that their full schemas are no longer placed in context up front. Answer 2.4-K3 items in the guide's terms.

## Models and output styles

*Tested in: CCDV-F D2.6 Configuration Management (model version pinning), D5.1 LLM Fundamentals (fast mode, extended thinking, adaptive thinking, effort levels), D5.3 Model Selection and Tradeoffs · CCAR-P 2.1, 7.1 · output styles are not named in any of the four guides*

Two settings shape every response in a session: the model, chosen by an alias or a model name, and the output style, which sets Claude's role, tone and response format. This section covers how Claude Code picks, pins and restricts the model for a session, and how output styles change the way it responds. Which model family suits which task is taught in [Models and how to choose one](claude-api.md#models-and-how-to-choose-one).

### Model aliases

The `model` setting accepts an alias or a model name. On the Anthropic API the name is the full model ID; on Amazon Bedrock it is an inference profile ARN, on Microsoft Foundry a deployment name, and on Google Cloud's Agent Platform a version name.

| Alias | What it selects |
|---|---|
| `default` | Not an alias: clears any model override and returns to your account's runtime default |
| `best` | The model `fable` resolves to where Fable is available to you, otherwise the same model as `opus` |
| `fable` | The Fable model for your provider, for the hardest and longest-running tasks |
| `sonnet` | The latest Sonnet model, for daily coding tasks |
| `opus` | The latest Opus model, for complex reasoning tasks |
| `haiku` | The fast, efficient Haiku model, for simple tasks |
| `sonnet[1m]`, `opus[1m]` | Sonnet or Opus with a 1 million token context window, for long sessions; `sonnet[1m]` has no effect when `sonnet` already resolves to Sonnet 5, whose window is natively 1M |
| `opusplan` | `opus` during plan mode, then `sonnet` for execution |

As of September 2026, on the Anthropic API `opus` resolves to Opus 5.5 and `sonnet` to Sonnet 5. The mapping differs by provider: `sonnet` resolves to Sonnet 4.6 on Claude Platform on AWS and to Sonnet 4.5 on Amazon Bedrock and Agent Platform, and `opus` resolves to Opus 4.6 on Microsoft Foundry. `default` resolves to Opus 5.5 on Pro, Max, Team, Enterprise, the Anthropic API, Claude Platform on AWS, Amazon Bedrock and Agent Platform, and to Sonnet 4.5 on Microsoft Foundry, unless an administrator has set an organization default model. Opus 5.5 became the default Opus model in v2.1.280 (September 22, 2026); before that release `default` resolved to Sonnet 5 on Pro and Team Standard, and to Opus 5 (from v2.1.219) on Max, Team Premium, Enterprise, the Anthropic API, Claude Platform on AWS, Amazon Bedrock and Agent Platform. Treat any alias-to-version mapping as a snapshot.

### Pinning a model version (CCDV-F D2.6)

Aliases point to the recommended version for your provider and change over time. To pin, use the full model name, for example `claude-opus-5-5`, or set the matching environment variable such as `ANTHROPIC_DEFAULT_OPUS_MODEL`. On the API side, every Claude model ID, including the dateless IDs used from the 4.6 generation on, is a pinned snapshot; so in Claude Code, the thing that moves is the alias (our reading of the two facts together).

| Variable | Sets the model for |
|---|---|
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | `opus`, and `opusplan` while plan mode is active |
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | `sonnet`, and `opusplan` outside plan mode |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | `haiku`, and background functionality (replaces the deprecated `ANTHROPIC_SMALL_FAST_MODEL`) |
| `ANTHROPIC_DEFAULT_FABLE_MODEL` | `fable` |
| `CLAUDE_CODE_SUBAGENT_MODEL` | Subagents, agent team teammates and workflow agents that are not assigned a model another way |

For Amazon Bedrock, Agent Platform, Microsoft Foundry and Claude Platform on AWS, the docs say to pin model versions before rolling out to users, by setting these variables to specific version IDs as part of initial setup, because "Pinning lets you control when your users move to a new model." ([model configuration](https://code.claude.com/docs/en/model-config)). On Bedrock, for example: `export ANTHROPIC_DEFAULT_OPUS_MODEL='us.anthropic.claude-opus-4-8'`. To move users to a new version, update the variables and redeploy. Append `[1m]` to a pinned ID (`claude-opus-4-8[1m]`) to enable the extended context window, only when the underlying model supports 1M context; Claude Code strips the suffix before sending the ID. `ANTHROPIC_CUSTOM_MODEL_OPTION` adds one custom entry to the `/model` picker without replacing the built-in aliases, and `ANTHROPIC_BASE_URL` changes where requests go, not which model answers.

### Which model setting wins

1. `/model` during the session.
2. `claude --model` at startup.
3. The `ANTHROPIC_MODEL` environment variable.
4. The `model` field in a settings file.
5. `ANTHROPIC_DEFAULT_MODEL`, the default for new sessions (v2.1.236 or later).

```json
{
  "model": "opus"
}
```

- `/model` switches and saves your choice as the default for new sessions by writing `model` into your user settings. In the picker, `Enter` switches and saves; `s` switches for the current session only.
- `--model` and `ANTHROPIC_MODEL` apply only to the session they launch, so different terminals can run different models.
- Resumed sessions (`--resume`, `--continue`, `/resume`) keep the model they were using when the transcript was saved, regardless of the current `model` setting. A model picked for the new launch with `--model` or `ANTHROPIC_MODEL` still takes precedence (as of v2.1.195, so does a family variable such as `ANTHROPIC_DEFAULT_OPUS_MODEL`), and a restored model that has been retired or is excluded by `availableModels` falls through to the normal order. On Amazon Bedrock, Agent Platform and Microsoft Foundry the transcript model is not restored at all; the session resolves its model through the normal order.
- `model` is read only at session start; editing it mid-session does nothing until the next session, so use `/model`.
- A managed `model` is a default, not a lock: `--model` and `ANTHROPIC_MODEL` still pick the session model. To restrict choice, an organization deploys `availableModels` (see [Managed settings for organizations](#managed-settings-for-organizations)).

**Restricting models.** `availableModels` in managed or policy settings limits what users can select. Entries match a family (`sonnet`), a version prefix (`claude-sonnet-4-5`) or a full model ID. The allowlist applies to `/model`, `--model`, `ANTHROPIC_MODEL`, the `model` setting, subagent and teammate models, and the `model` frontmatter of skills and commands; a blocked skill or command override is ignored and it runs on the session model. Setting `enforceAvailableModels: true` alongside a non-empty `availableModels` in managed settings (v2.1.175 or later) extends the allowlist to the Default option. On Claude Enterprise plans there is a second mechanism, organization model restrictions: admins disable individual models in the claude.ai admin console, the restriction arrives with the account's entitlements when Claude Code authenticates (v2.1.187 or later), separate from any `availableModels` list, and the server enforces it independently when a session is created.

```json
{
  "availableModels": ["sonnet", "haiku"]
}
```

### Effort, thinking, fast mode and fallback models

| Control | How it works |
|---|---|
| Effort level | Controls adaptive reasoning. The default is `high` on every model that supports effort, except Opus 5.5 (`medium`) and Opus 4.7 (`xhigh`). Set it with `/effort`, the `/model` slider, `--effort`, `CLAUDE_CODE_EFFORT_LEVEL`, the `modelSettings` or `effortLevel` settings, or `effort` frontmatter in a skill or subagent |
| `max` effort | Applies to the current session only, unless set through `CLAUDE_CODE_EFFORT_LEVEL`; the `effortLevel` and `modelSettings` keys do not accept `max` |
| `CLAUDE_CODE_EFFORT_LEVEL` | Takes precedence over `--effort`, `/effort`, `modelSettings` and `effortLevel` |
| Organization caps | A managed `effortLevel` is a default users can still change with `/effort` or `--effort`; `maxEffortLevel` caps the level instead |
| `ultrathink` | Anywhere in a prompt, requests deeper reasoning for that turn without changing the session's effort. Phrases such as `think hard` are not keywords |
| Extended thinking | Toggle with `Option+T` or `Alt+T`; `/config` saves a default as `alwaysThinkingEnabled`; `MAX_THINKING_TOKENS=0` turns it off on the Anthropic API. Thinking cannot be turned off on Opus 5.5 or the Fable models |
| Fast mode | `/fast` toggles it (saved as `fastMode`). It is the same Opus model with an API configuration that trades cost for speed, up to 2.5x faster at a higher price per token; supported on Opus 5.5, Opus 5 and Opus 4.8, as a research preview |
| Fallback chain | `--fallback-model sonnet,haiku` for one session, or a `fallbackModel` array in settings. It triggers when the primary model is overloaded, unavailable or returns another non-retryable server error, never on authentication, billing or rate-limit errors. Chains cap at three models, and a switch lasts for the current turn only |
| 1M context | `CLAUDE_CODE_DISABLE_1M_CONTEXT=1` turns it off |
| Checking the model | The status line, or `/status` |

```json
{
  "fallbackModel": ["claude-sonnet-5", "claude-haiku-4-5"]
}
```

The API-level view of thinking, effort and fast mode is in [Extended thinking, adaptive thinking and effort](claude-api.md#extended-thinking-adaptive-thinking-and-effort).

### Output styles

An output style is a set of instructions that sets Claude's role, tone and response format for every response in a session. Claude Code ships four built-in styles besides Default, and you can write your own.

| Style | Behavior |
|---|---|
| Default | No output style selected; Claude works from Claude Code's standard system prompt for software engineering |
| Proactive | Starts work and makes reasonable assumptions. It does not change your permission mode |
| Concise | Leads with the result, with no preamble, narration or recap |
| Explanatory | Adds short `Insight` blocks that explain the choices behind the code it writes |
| Learning | Adds `Insight` blocks and leaves `TODO(human)` pieces for you to write |

- **Switching.** Use `/output-style <style>` (for example `/output-style concise`), `/config` then Output style, or the `outputStyle` settings field. The command and the menus save the choice to `.claude/settings.local.json`. A switch mid-session applies from your next message (before v2.1.251 it applied only after `/clear` or a new session).
- **Exact names.** `outputStyle` is case-sensitive: `Proactive`, `Concise`, `Explanatory`, `Learning`. A value such as `explanatory` gives you Default.
- **Custom styles.** Markdown files in `~/.claude/output-styles` (user), `.claude/output-styles` (project), or `.claude/output-styles` inside the managed settings directory. Plugins can ship them in `output-styles/`. Frontmatter fields: `name`, `description`, `keep-coding-instructions` (default `false`) and `force-for-plugin` (plugin styles only).
- **Coding instructions drop out.** The four built-in styles keep Claude Code's software engineering instructions and add their own. A custom style leaves those instructions out (how to scope changes, write comments, verify work) unless `keep-coding-instructions: true`.
- **Scope.** Styles apply to the main conversation and to forks, not to other subagents, which run their own system prompt.

A custom style file, trimmed from the example in the output styles documentation:

```markdown
---
name: Diagrams first
description: Lead every explanation with a diagram
keep-coding-instructions: true
---

When explaining code, architecture, or data flow, start with a Mermaid diagram showing the structure, then explain in prose.
```

An output style shapes behavior; it does not guarantee it. The [output styles documentation](https://code.claude.com/docs/en/output-styles) says of a style: "It's an instruction Claude follows, so nothing enforces it."

| If you want | Use |
|---|---|
| A voice and format for every response | An output style |
| Project conventions Claude should always know | CLAUDE.md |
| Instructions for one kind of task | A skill |
| Something that must happen every time | A hook |
| A helper with its own separate context | A subagent |
| Extra text appended to the system prompt at launch | `--append-system-prompt` |

!!! warning "Current docs: `/output-style` came back"

    Output styles arrived in v1.0.81 (August 14, 2025), were deprecated in v2.0.30 and restored in v2.0.32. The `/output-style` command was deprecated in v2.1.73 in favor of `/config` and added back in v2.1.269 (September 11, 2026); the Concise style arrived in v2.1.237 (August 20, 2026). Material written between March and September 2026 may tell you to use `/config` instead. Both work as of September 2026.

### Traps

Our list; the guides name model version pinning and effort levels but no Claude Code anti-patterns for them, so each trap follows from a documented rule above.

- Assuming a managed `model` locks the model; it is only a default. Restrict choice with `availableModels` (plus `enforceAvailableModels: true` to cover the Default option) or, on Claude Enterprise, with organization model restrictions in the claude.ai admin console.
- Expecting `opus` to mean the same version next quarter; pin a full ID when behavior must not change.
- Writing `"outputStyle": "explanatory"` in lowercase and getting Default.
- Expecting an output style to change how a subagent responds.
- Putting `max` in `effortLevel` (the key does not accept it) or expecting `/effort max` to carry into the next session; only `CLAUDE_CODE_EFFORT_LEVEL` makes `max` stick.
- Treating a managed `effortLevel` as a ceiling; `maxEffortLevel` is the cap.

## Managed settings for organizations

*Tested in (our mapping; none of the four guides names managed settings): CCAR-P 5.1, 7.1 · CCDV-F D2.6 Configuration Management (settings.json), D7.2 Guardrails and Safe Deployment*

Managed settings are the organization's layer. They are deployed to every developer's machine and apply above every other level, so no user, project, local or `--settings` value overrides them, apart from a few security-sensitive keys described below. They bind Claude Code only: a developer who calls the API from another tool is not covered by them. The threat model and the wider governance picture are in [Claude Code security controls](security-and-governance.md#claude-code-security-controls) and [Admin and governance controls](security-and-governance.md#admin-and-governance-controls); rollout planning is in [Rolling out Claude Code to an engineering organization](solution-architecture.md#rolling-out-claude-code-to-an-engineering-organization).

### Four ways to deliver a policy

| Mechanism, highest priority first | How it is delivered | When Claude Code reads it |
|---|---|---|
| Server-managed settings | Configured by organization Owners in the claude.ai admin console (Admin Settings > Claude Code > Managed settings), which needs a Claude for Teams or Enterprise plan and network access to `api.anthropic.com`, or on a self-hosted Claude apps gateway | Fetched at startup and polled hourly |
| MDM or OS-level policy | macOS configuration profile in the managed preferences domain `com.anthropic.claudecode`; on Windows, a `Settings` value (`REG_SZ` or `REG_EXPAND_SZ`) under `HKLM\SOFTWARE\Policies\ClaudeCode` | Read at startup, checked every 30 minutes |
| File-based | `managed-settings.json` in a system directory (paths below) | Read at startup, reloaded when the file changes |
| HKCU registry (Windows) | The same `Settings` value under `HKCU\SOFTWARE\Policies\ClaudeCode` | Read at startup, checked every 30 minutes, and used only when no other managed source delivers a policy key; writable without elevation, so a convenience default rather than an enforcement channel |

| OS | File-based path |
|---|---|
| macOS | `/Library/Application Support/ClaudeCode/managed-settings.json` |
| Linux and WSL | `/etc/claude-code/managed-settings.json` |
| Windows | `C:\Program Files\ClaudeCode\managed-settings.json` |

- **Legacy Windows path.** `C:\ProgramData\ClaudeCode\managed-settings.json` has not been read since v2.1.75.
- **Drop-in fragments.** A `managed-settings.d/` directory lets separate teams deploy independent fragments: `managed-settings.json` merges first, then every `*.json` file in the directory in alphabetical order (for example `10-telemetry.json`, then `20-security.json`). When two files set the same single value, the later file wins; lists combine.
- **Several sources on one machine.** By default (`managedSourcesBehavior: "first-wins"`), Claude Code uses the highest-ranked source that delivers at least one policy key and ignores the rest instead of merging them. `"merge"` (v2.1.242 or later) applies every admin source that delivers a policy key (the server, MDM or OS-level policy, and the file) and combines them by kind of key; the user-writable HKCU registry is not an admin source and never merges with another source.
- **Server or endpoint.** Endpoint-managed settings (MDM or the file) give stronger guarantees because the file can be protected from modification at the OS level. Server-managed settings suit organizations without MDM, and they are the only kind that reaches Anthropic-hosted cloud sessions. They are fetched only over a direct connection to `api.anthropic.com`: a developer who exports a `CLAUDE_CODE_USE_*` provider variable or a non-default `ANTHROPIC_BASE_URL` skips the fetch, so the docs say to use the endpoint-managed channel to enforce policy without relying on developers' shells. For Bedrock, Agent Platform, Foundry and Claude Platform on AWS, a self-hosted Claude apps gateway provides the equivalent remote delivery.
- **Verify.** After deploying a file, `/status` shows `Enterprise managed settings (file)` on the `Setting sources` line.

### How managed values combine with everyone else's

- **Nothing overrides them.** A key passed with `--settings` does not override the same managed key, and `--model` picks only from the models the organization allows.
- **Lists merge.** Array settings such as `permissions.allow` and `permissions.deny` combine across all sources, so developers can extend a managed list but cannot remove entries from it. Model lists are the exception: when managed settings define `availableModels`, that list applies as-is and entries users add are ignored, and `fallbackModel` is taken whole from the highest-precedence file.
- **A deny anywhere wins,** so a managed deny cannot be overridden by `--allowedTools`; the rule is taught in [Allow, ask and deny](#allow-ask-and-deny).
- **Hooks.** Hooks from managed settings cannot be removed by other files, and `disableAllHooks` set outside managed settings cannot disable managed hooks.
- **A restrictive value can come from below.** For a few keys that restrict a session, Claude Code honors the stricter value from a scope that otherwise could not override managed settings: for example `disableClaudeAiConnectors: true` from any scope, or a lower `maxEffortLevel` cap from any scope. `disableBypassPermissionsMode` also works from any scope.
- **A managed `model` is a default.** `--model` and `ANTHROPIC_MODEL` still choose the session model; restrict choice with `availableModels` (see [Which model setting wins](#which-model-setting-wins)).

### Keys only a managed source can set

Claude Code reads the keys below (a selection from the documented list) only from a managed source; placing them in user or project settings files has no effect. Most are locks: the value they govern, such as permission rules, is an ordinary key any level can set, and the lock tells Claude Code to honor only the managed value.

| Key | Effect |
|---|---|
| `allowManagedPermissionRulesOnly` | Managed settings become the only source of permission rules; rules from user, project and local files and from `--allowedTools` are ignored |
| `allowManagedHooksOnly` | Only managed hooks run (plus Agent SDK hooks and hooks from plugins that managed settings force-enable) |
| `allowManagedMcpServersOnly` | Only the managed `allowedMcpServers` list is respected; `deniedMcpServers` still merges from all sources |
| `managedMcpServers` | Remote MCP servers provided to every user alongside their own; it provides servers rather than locking anything down (v2.1.259 or later) |
| `strictKnownMarketplaces`, `blockedMarketplaces`, `strictPluginOnlyCustomization`, `disableSideloadFlags` | Control which marketplaces, customization sources and sideloading flags users can use; each is explained in [Trust and organizational control](#trust-and-organizational-control) |
| `forceRemoteSettingsRefresh` | Blocks startup until remote managed settings are freshly fetched, and exits if the fetch fails |
| `sandbox.network.allowManagedDomainsOnly` | Honors only managed `allowedDomains` and `WebFetch(domain:...)` allow rules; other domains are blocked without a prompt |
| `sandbox.filesystem.allowManagedReadPathsOnly` | Honors only managed `filesystem.allowRead` paths; `denyRead` still merges from all sources |

### A typical policy file

This `managed-settings.json`, from the managed settings documentation, blocks two file reads, turns off bypass mode, and makes Claude Code ignore permission rules from user, project and local files and from `--allowedTools`:

```json
{
  "permissions": {
    "deny": [
      "Read(./.env)",
      "Read(./secrets/**)"
    ],
    "disableBypassPermissionsMode": "disable"
  },
  "allowManagedPermissionRulesOnly": true
}
```

| Goal | Managed configuration |
|---|---|
| No bypass or auto mode | `permissions.disableBypassPermissionsMode: "disable"` (also rejects `--dangerously-skip-permissions`); `permissions.disableAutoMode: "disable"` |
| Require the sandbox | `sandbox.enabled: true`, `failIfUnavailable: true`, `allowUnsandboxedCommands: false` |
| Limit models | `availableModels`, plus `enforceAvailableModels: true` to cover the Default option |
| Cap reasoning effort | `maxEffortLevel` (a managed `effortLevel` is only a default) |
| Limit sign-in | `forceLoginMethod` (`"claudeai"`, `"console"` or `"gateway"`) with `forceLoginOrgUUID` |
| Organization-wide instructions | A managed CLAUDE.md file, or the `claudeMd` key |
| Organization-wide skills and subagents | Skills under `.claude/skills/` in the managed settings directory; a subagent defined in managed settings takes priority over any same-named definition |
| Startup messages | `companyAnnouncements`, one shown at random per session |

```json
{
  "sandbox": {
    "enabled": true,
    "failIfUnavailable": true,
    "allowUnsandboxedCommands": false
  }
}
```

### Managed instructions: CLAUDE.md or settings

A managed CLAUDE.md applies to every user on the machine and cannot be excluded by individual settings, including `claudeMdExcludes`. It sits at the managed policy path for each OS listed in [The locations](#the-locations) and is distributed with MDM, Group Policy, Ansible or similar tools. The `claudeMd` key puts the same kind of text inside `managed-settings.json`; it is honored only in managed and policy settings and loads ahead of user and project CLAUDE.md files.

```json
{
  "claudeMd": "Always run `make lint` before committing.\nNever push directly to main."
}
```

The decision rule, in the words of the [memory documentation](https://code.claude.com/docs/en/memory): "Settings rules are enforced by the client regardless of what Claude decides to do. CLAUDE.md instructions shape Claude's behavior but are not a hard enforcement layer." Put technical enforcement (`permissions.deny`, `sandbox.enabled`, `env`, `forceLoginMethod`) in managed settings, and behavioral guidance in a managed CLAUDE.md.

### Managed MCP control

By default anyone running Claude Code can connect any MCP server. `managed-mcp.json` gives the organization exclusive control of the server set; it lives in the same system directories as `managed-settings.json` and, being a standalone file, cannot be delivered through server-managed settings. To give every user some servers without taking exclusive control, list them under `managedMcpServers` in any managed source (v2.1.259 or later); users keep the servers they add themselves. Alternatively, `allowedMcpServers` and `deniedMcpServers` filter what users add: a denylist match blocks a server and nothing overrides it, an unset allowlist allows everything, and an empty allowlist allows no servers other than the organization's own.

```json
{
  "allowManagedMcpServersOnly": true,
  "allowedMcpServers": [
    { "serverUrl": "https://api.githubcopilot.com/*" },
    { "serverUrl": "https://*.internal.example.com/*" }
  ]
}
```

Match on `serverUrl` or `serverCommand`, not `serverName`: users choose server names, so a `serverName` entry in either list is not a security control.

### Decide and traps

**Decide** (our decision rules, each following from the documented behavior above).

- If a control must hold whatever developers configure and whatever Claude decides, put it in managed settings, not in CLAUDE.md or a project `.claude/settings.json`.
- If devices are enrolled in MDM, prefer endpoint-managed settings, and configure server-managed settings as well when developers run cloud sessions; if devices are not enrolled, use server-managed settings. If developers use a third-party provider or a custom `ANTHROPIC_BASE_URL`, server-managed settings are skipped, so use the endpoint-managed channel (or a Claude apps gateway).
- If only the organization's list should count (permission rules, hooks, MCP servers), set the matching `allowManaged...Only` key rather than only adding managed entries, because lists otherwise merge with developers' own.

**Traps** (our list; each follows from a documented rule above).

- Putting `allowManagedHooksOnly` or `strictKnownMarketplaces` in a project settings file and expecting it to apply.
- Treating a managed `model` as a lock.
- Using the Windows HKCU registry as the enforcement channel.
- Allowlisting MCP servers by `serverName`.
- Assuming managed settings also govern API calls made from other tools.
- Deploying to the legacy `C:\ProgramData\ClaudeCode\` path.

## Exam map

Which objectives each section of this page serves. The objectives come from the four exam guides (Version 1.0, effective July 2026); which section serves which objective is our mapping. CCAR-F codes such as 3.2-K2 mean task statement 3.2, second "Knowledge of" bullet in the [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) (S for "Skills in"); the task statements are numbered in the guide, the bullets are not, so the bullet numbers are ours, in the guide's order. CCDV-F codes such as D2.6 mean domain 2, sixth skill, and CCAR-P codes such as 7.1 mean domain 7, first objective bullet; both guides print these without numbers, so those numbers are ours too, in the order printed. In the CCAR-F column, `APPX-TECH-n` and `APPX-INSCOPE-n` are items in the guide's appendix lists of technologies and concepts and of in-scope topics, "Exercise 2 step N" is a step of Preparation Exercise 2 (Configure Claude Code for a Team Development Workflow), and "sample question N" is the guide's sample Question N; the appendix items and exercise steps are unnumbered bullets in the guide, so those numbers are ours, in the guide's order. In the CCAR-P column, "sample 1" is the guide's Sample 1, the least-privilege question tagged to Domain 3.

| Section | CCAO-F | CCDV-F | CCAR-F | CCAR-P |
|---|---|---|---|---|
| [The configuration layers at a glance](#the-configuration-layers-at-a-glance) | None | D2.6 Configuration Management; D3.1 Claude Code Operation; D8.3 Agentic Customization | 3.1; 3.2-S5; 3.3-S3; 2.4-K1; APPX-TECH-3 | 3.8, 7.1 |
| [CLAUDE.md and the memory hierarchy](#claudemd-and-the-memory-hierarchy) | None | D2.6 (CLAUDE.md files); D3.1 (CLAUDE.md hierarchy, repository initialization) | 3.1-K1 to 3.1-K4, 3.1-S1 to 3.1-S4; 3.6-K3, 3.6-S5; Exercise 2 step 1; APPX-INSCOPE-9 | 3.8, 7.1 |
| [Path-scoped rules](#path-scoped-rules) | None | D3.1 (Rules) | 3.1-K4, 3.1-S3; 3.3-K1 to 3.3-K3; 3.3-S1 to 3.3-S3; sample question 6; Exercise 2 step 2; APPX-INSCOPE-9 | 2.4, 7.1 |
| [Settings files and precedence](#settings-files-and-precedence) | None | D2.6 (settings.json); D3.1 (settings.json configuration) | Not named | 7.1 |
| [Permissions and permission modes](#permissions-and-permission-modes) | None | D3.1 (auto-mode, settings.json); D7.2 (least privilege) | 3.4-K3 (plan mode) | 3.1, 5.1, 7.1; sample 1 (least privilege) |
| [Slash commands](#slash-commands) | None | D3.1 (Commands; built-in and custom slash commands, repository initialization) | 3.2-K1, 3.2-S1; sample question 4; 3.1-S4 (`/memory`); 5.4-S5 (`/compact`); APPX-TECH-3, APPX-INSCOPE-10 | 7.1 |
| [Agent Skills](#agent-skills) | Not an objective; How to Prepare names Skills | D3.1 (Skills); D8.3 Agentic Customization | 3.2-K2, 3.2-K3, 3.2-K4, 3.2-S2 to 3.2-S5; Exercise 2 step 3; APPX-TECH-3, APPX-INSCOPE-10 | 2.5, 3.8, 7.1 |
| [Subagents](#subagents) | None | D1.1 Agent Architecture; D1.3 Agent Patterns and Frameworks; D3.1 (Agents, Agent Memory); D6.1 Context Engineering | 1.2-K2; 1.3-K1, 1.3-K2, 1.3-K3, 1.3-S3; 2.3-S1; 3.4-K4, 3.4-S3; 5.4-K3, 5.4-S1; APPX-TECH-3 | 1.4, 3.1, 7.1 |
| [Plugins and marketplaces](#plugins-and-marketplaces) | None | D2.5 Claude Application Design (plugin management); D2.6 (plugin dependencies) | Not named | 7.1 |
| [MCP servers in Claude Code](#mcp-servers-in-claude-code) | None | D8.2 MCP Server Development (integration with Claude applications) | 2.4-K1, 2.4-K2, 2.4-K3, 2.4-S1, 2.4-S2; Exercise 2 step 4; APPX-INSCOPE-6 | 3.7, 7.1 |
| [Models and output styles](#models-and-output-styles) | None | D2.6 (model version pinning); D5.1 (fast mode, extended thinking, adaptive thinking, effort levels); D5.3 Model Selection and Tradeoffs | Not named | 2.1, 7.1 |
| [Managed settings for organizations](#managed-settings-for-organizations) | None | D2.6 (settings.json); D7.2 Guardrails and Safe Deployment | Not named | 5.1, 7.1 |

Where the weight sits:

- **CCAR-F:** Domain 3, Claude Code Configuration & Workflows, carries 20% of the exam, and Exercise 2 (Configure Claude Code for a Team Development Workflow) has the stated objective of practicing CLAUDE.md hierarchies, custom slash commands, path-specific rules and MCP server integration.
- **CCDV-F:** Domain 3, Claude Code, is weighted 3.1%, but Claude Code configuration also sits in D2.5 Claude Application Design (8.6%) and D2.6 Configuration Management (4.1%), so the Domain 3 weight alone understates it (our reading of the blueprint).
- **CCAR-P:** Domain 7, Developer Productivity & Operational Enablement, is weighted 7%, and its first objective is to configure Claude tools and environments for teams, with Claude Code as the example.
- **CCAO-F:** the guide does not mention Claude Code. Its How to Prepare list names Skills among the features to review; Skills and connectors in the apps are taught in [Skills in the Claude apps](claude-for-work.md#skills-in-the-claude-apps) and [Search, Research and connectors](claude-for-work.md#search-research-and-connectors).

Exam-specific framing for these objectives is on the exam pages: [CCAR-F Domain 3](../claude-certified-architect-foundations.md#domain-3-claude-code-configuration-workflows), [CCDV-F Domain 3](../claude-certified-developer.md#domain-3-claude-code) and [CCAR-P Domain 7](../claude-certified-architect-professional.md#domain-7-developer-productivity-operational-enablement).

??? info "Sources"

    - [Claude Certified Architect, Foundations: Exam Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): CCAR-F Task Statements 1.2, 1.3, 2.3, 2.4, 3.1 to 3.4, 3.6 and 5.4, Preparation Exercise 2, sample Questions 4, 6 and 10 with Anthropic's rationales, appendix technology and in-scope lists; the `allowed-tools`, `argument-hint`, Task tool and MCP scope wording quoted on this page
    - [Claude Certified Developer, Foundations: Exam Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf): CCDV-F skills D1.1, D1.3, D2.5, D2.6, D3.1, D5.1, D5.3, D6.1, D7.2, D8.2 and D8.3, their descriptions and weights
    - [Claude Certified Architect, Professional: Exam Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf): CCAR-P objectives 1.4, 2.1, 2.4, 2.5, 3.1, 3.7, 3.8, 5.1 and 7.1, the Domain 7 weight, and Sample 1 with Anthropic's rationale
    - [Claude Certified Associate, Foundations: Exam Guide (PDF)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf): the How to Prepare list that names Skills
    - [Claude Code: How Claude remembers your project](https://code.claude.com/docs/en/memory): CLAUDE.md locations and load order, lazy loading, imports, `.claude/rules/` and path-scoped rules, `/init`, `/memory` versus `/context`, managed CLAUDE.md and `claudeMd`, `claudeMdExcludes`, AGENTS.md, auto memory, the settings versus CLAUDE.md enforcement quote
    - [Claude Code: Settings files and precedence](https://code.claude.com/docs/en/settings): the four settings files, `~/.claude.json`, precedence order, managed settings overriding `--settings` and `--model`, restrictive exceptions, list merging, environment variables, one-session overrides, workspace trust, live reload, `/status`, broken files, cloud sessions
    - [Claude Code: All settings](https://code.claude.com/docs/en/settings-reference): `permissions` keys and values, `defaultMode`, `env`, `model`, `outputStyle`, `enabledPlugins`, `extraKnownMarketplaces`, `disableBypassPermissionsMode`, `disableAutoMode`, `forceLoginMethod`, `companyAnnouncements`, `claudeMd`, hooks merging, example permission blocks
    - [Claude Code: Example settings files](https://code.claude.com/docs/en/settings-example): the team `.claude/settings.json` example with `extraKnownMarketplaces` and `enabledPlugins`
    - [Claude Code: Configure permissions](https://code.claude.com/docs/en/permissions): tiered defaults, allow/ask/deny evaluation, deny at any level, rule syntax for Bash, Read, Edit, WebFetch, MCP, Agent and Skill, limits of Bash and file rules, hooks and rules, workspace trust, working directories and `--add-dir`
    - [Claude Code: Choose a permission mode](https://code.claude.com/docs/en/permission-modes): the six modes, labels, starting-mode resolution, `Shift+Tab` cycle, plan mode, auto mode behavior and subagents, `dontAsk`, `bypassPermissions`, disabling auto mode, protected paths and checks no mode skips
    - [Claude Code: Configure auto mode](https://code.claude.com/docs/en/auto-mode-config): classifier reads CLAUDE.md, `autoMode` not read from project settings, ask rules as human checkpoints
    - [Claude Code: Commands](https://code.claude.com/docs/en/commands): how commands are recognized and queued, built-in commands and aliases, bundled skills, `/review` alias change, `/agents`, `/mcp`, `/plugin`, `/output-style`, `/effort`, MCP prompts as commands
    - [Claude Code: Interactive mode](https://code.claude.com/docs/en/interactive-mode): `!` shell mode and `@` mentions
    - [Claude Code: Extend Claude Code](https://code.claude.com/docs/en/features-overview): triggers for each layer, how layers combine, context cost by feature, guidance versus enforcement, subagent startup context
    - [Claude Code: Explore the .claude directory](https://code.claude.com/docs/en/claude-directory): file map, the test-file rule and `fix-issue` command examples, rules as guidance, frontmatter fields for skills and subagents
    - [Claude Code: Debug your configuration](https://code.claude.com/docs/en/debug-your-config): CLAUDE.md versus permissions, common misplacements such as the skill folder mistake and `mcpServers` in `settings.json`, lazy loading of subdirectory CLAUDE.md, merge order
    - [Claude Code: Set up Claude Code in a monorepo or large codebase](https://code.claude.com/docs/en/large-codebases): root versus subdirectory start, per-directory CLAUDE.md versus path-scoped rules, settings not inherited
    - [Claude Code: Explore the context window](https://code.claude.com/docs/en/context-window): what survives compaction, path-scoped rules and nested CLAUDE.md after compaction
    - [Claude Code: Best practices for Claude Code](https://code.claude.com/docs/en/best-practices): what to include in and leave out of CLAUDE.md, emphasis, compaction instructions, hooks as deterministic
    - [Claude Code: CLI reference](https://code.claude.com/docs/en/cli-reference): `--settings`, `--setting-sources`, `--permission-mode`, `--allowedTools`, `--disallowedTools`, `--tools`, `--add-dir`, `--append-system-prompt`
    - [Claude Code: Environment variables](https://code.claude.com/docs/en/env-vars): shell versus settings `env` precedence, `CLAUDE_CODE_DISABLE_AUTO_MEMORY`, `SLASH_COMMAND_TOOL_CHAR_BUDGET`, `CLAUDE_CODE_EFFORT_LEVEL`, `ANTHROPIC_DEFAULT_MODEL`
    - [Claude Code: Hooks reference](https://code.claude.com/docs/en/hooks): hook configuration locations and merging, hooks in skill frontmatter, MCP tool names in matchers
    - [Claude Code: Configure the sandboxed Bash tool](https://code.claude.com/docs/en/sandboxing): OS-level enforcement for Bash, PowerShell and Monitor commands, the managed configuration that requires the sandbox
    - [Claude Code: Tools reference](https://code.claude.com/docs/en/tools-reference): Edit tool read-before-edit check, Read deny rules and Grep
    - [Claude Code changelog](https://code.claude.com/docs/en/changelog): version history for the skills and commands merge (v2.1.3), the `#` shortcut, the `/agents` wizard removal, the YAML list for `paths`, output styles, `managed-settings.d/`, the Windows legacy path, Opus 5.5 as default Opus
    - [Claude Academy, AI-native SDLC playbook: CLAUDE.md](https://academy.claude.com/courses/ai-native-sdlc-playbook/claude-md): check CLAUDE.md into Git and review it like code
    - [Claude Code best practices, Anthropic engineering blog (archived copy)](https://web.archive.org/web/20250602202220/https://www.anthropic.com/engineering/claude-code-best-practices): the former `~/.claude/commands` guidance and the retired `/project:` command prefix
    - [Claude Code: Extend Claude with skills](https://code.claude.com/docs/en/skills): commands merged into skills, SKILL.md structure, every frontmatter field, command-file frontmatter, skill locations and precedence, invocation control, arguments, dynamic context injection, `context: fork`, `allowed-tools` and `disallowed-tools` behavior, `Skill` permission rules, claude.ai skill sync
    - [Claude Code: Create custom subagents](https://code.claude.com/docs/en/sub-agents): built-in subagents, file format and frontmatter, scope priority, tool control, the Task to Agent rename, delegation, context isolation, resumption, forks, agent memory
    - [Agent SDK: Subagents in the SDK](https://code.claude.com/docs/en/agent-sdk/subagents): `AgentDefinition` fields, the SDK form of a subagent's description, prompt and tools
    - [Claude Code: Create plugins](https://code.claude.com/docs/en/plugins): standalone versus plugin, plugin layout, testing with `--plugin-dir`, `claude plugin init` and `validate`
    - [Claude Code: Plugins reference](https://code.claude.com/docs/en/plugins-reference): `plugin.json` fields, versioning, path variables, plugin cache, installation scopes and `enabledPlugins`
    - [Claude Code: Discover and install prebuilt plugins through marketplaces](https://code.claude.com/docs/en/discover-plugins): marketplaces, the official marketplace, install scopes, auto-update, team marketplaces, the trust warning quoted on this page
    - [Claude Code: Create and distribute a plugin marketplace](https://code.claude.com/docs/en/plugin-marketplaces): `marketplace.json`, plugin sources, `enabledPlugins`, `strictKnownMarketplaces`, reserved names
    - [Claude Code: Constrain plugin dependency versions](https://code.claude.com/docs/en/plugin-dependencies): the `dependencies` array, semver ranges, automatic installation, pre-release handling
    - [Claude Code: Test plugins with evals](https://code.claude.com/docs/en/plugin-evals): pinning the model when evaluating plugins in CI
    - [Claude Code: Connect Claude Code to tools via MCP](https://code.claude.com/docs/en/mcp): MCP scopes and precedence, `claude mcp` commands, `.mcp.json` variable expansion, project server approval, tool search
    - [Claude Code: Connect to MCP servers](https://code.claude.com/docs/en/mcp-quickstart): where local- and user-scoped servers are stored in `~/.claude.json`
    - [Claude Code: Control MCP server access for your organization](https://code.claude.com/docs/en/managed-mcp): `managed-mcp.json`, allowlists and denylists, `serverName` versus `serverUrl` and `serverCommand`
    - [Claude Code: Model configuration](https://code.claude.com/docs/en/model-config): aliases and their resolution as of September 2026, pinning, selection priority, `availableModels`, effort, extended thinking, fallback models
    - [Claude Code: Speed up responses with fast mode](https://code.claude.com/docs/en/fast-mode): what fast mode trades and which models support it
    - [Claude Code: Output styles](https://code.claude.com/docs/en/output-styles): built-in and custom styles, file locations, switching, `keep-coding-instructions`, scope, the non-enforcement statement quoted on this page
    - [Claude Code: Deploy managed settings](https://code.claude.com/docs/en/managed-settings): delivery mechanisms, file paths, `managed-settings.d/`, `managedSourcesBehavior`, managed settings above every other level, managed-only keys, the example policy file
    - [Claude Code: Configure server-managed settings](https://code.claude.com/docs/en/server-managed-settings): admin console delivery, plan requirement, server versus endpoint trade-off
    - [Claude Code: Set up Claude Code for your organization](https://code.claude.com/docs/en/admin-setup): source priority, HKCU caveat, list merging, `strictPluginOnlyCustomization`, verifying with `/status`
    - [Claude Code: Run agents in parallel](https://code.claude.com/docs/en/agents): token usage multiplying with parallel sessions and subagents
    - [Claude Code: How Claude Code works](https://code.claude.com/docs/en/how-claude-code-works): session forking with `--fork-session` and `/branch`
    - [Agent SDK: Work with sessions](https://code.claude.com/docs/en/agent-sdk/sessions): `fork_session` in Python
    - [Agent Skills (Claude Platform docs)](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview): progressive disclosure levels, required fields on the platform, surface differences, no sync across surfaces
    - [Skill authoring best practices (Claude Platform docs)](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices): `name` and `description` limits, third-person descriptions, SKILL.md as a table of contents
    - [Models overview (Claude Platform docs)](https://platform.claude.com/docs/en/models/overview): every model ID is a pinned snapshot
    - [Provision and manage skills for your organization (Claude Help Center)](https://support.claude.com/en/articles/13119606-provision-and-manage-skills-for-your-organization): provisioned skills loading in Claude Code, `syncClaudeAiSkills`, plugins assigned to groups on Enterprise
