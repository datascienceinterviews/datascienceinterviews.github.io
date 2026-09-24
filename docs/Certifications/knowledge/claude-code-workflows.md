---
title: "Claude Code Workflows, Hooks and CI/CD for the Claude Certifications"
description: Claude Code hooks, plan mode, iterative refinement, sessions, context control, headless runs and CI/CD, taught for all four Claude certification exams.
last_reviewed: 2026-09-23
---

# Claude Code Workflows, Hooks and CI/CD

This page teaches how Claude Code work is run, both at the terminal and in pipelines: hooks that enforce rules deterministically, the choice between plan mode and direct execution, iterative refinement, session resumption, forking and rewind, context management in large repositories, headless runs, CI/CD, automated code review, the working practices the Claude Code docs recommend, and usage and cost monitoring. Each section opens with the exam objectives it serves, and the Exam map at the end collects them for all four exams. Product details are as of September 2026; where the July 2026 exam guides use older wording, the relevant section shows both and says which wording to expect on the exam.

## Hooks

*Tested in: CCAR-F 1.4-K1, 1.4-K2, 1.4-S1, 1.5 (Scenario 1, sample question 1, Exercise 1 step 4), appendix Claude Agent SDK item · CCDV-F Claude Hooks (Domain 7 Sample 2), Agent Construction with Claude · CCAR-P 5.1, 7.1*

A hook is a handler that Claude Code runs automatically at a fixed point in its lifecycle. The [hooks guide](https://code.claude.com/docs/en/hooks-guide) states the point of the feature: hooks give "deterministic control" so that "certain actions always happen rather than relying on the LLM to choose to run them." A handler can be a shell command, an HTTP endpoint, an MCP tool call, an LLM prompt or a subagent, and hooks fire in the terminal, IDE extensions, the Desktop app and cloud sessions.

This section covers hooks configured in Claude Code settings files. The Agent SDK exposes the same idea as callback functions in your own process (passed in `options.hooks`), and it also runs shell command hooks from settings files when the matching `settingSources` (TypeScript) or `setting_sources` (Python) entry is enabled, which it is for default `query()` options; the SDK form, and the SDK's permission pipeline, are taught in [Hooks in the SDK](agents-and-agent-sdk.md#hooks-in-the-sdk).

### Hook or instruction: the decision the exams test

The CCAR-F guide frames the whole topic as a contrast. Task 1.5 names "The distinction between using hooks for deterministic guarantees versus relying on prompt instructions for probabilistic compliance", and Task 1.4 adds that "When deterministic compliance is required (e.g., identity verification before financial operations), prompt instructions alone have a non-zero failure rate" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Skill 1.5-S3 is the decision itself: "Choosing hooks over prompt-based enforcement when business rules require guaranteed compliance". CCDV-F tests the same idea in its Claude Hooks skill (hooks as guardrails and safety controls that prevent destructive actions) and in Agent Construction with Claude, which names "hooks for deterministic actions".

Two official sample questions turn on this contrast. In CCAR-F sample question 1 the agent skips `get_customer` in 12% of cases, and the correct answer is a programmatic prerequisite that blocks `lookup_order` and `process_refund` until `get_customer` has returned a verified customer ID. The rationale: "programmatic enforcement provides deterministic guarantees that prompt-based approaches cannot", while a stronger system prompt and few-shot examples "rely on probabilistic LLM compliance, which is insufficient when errors have financial consequences" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). In CCDV-F Sample 2 (Domain 7), an agent summarizing web pages meets hidden injected instructions, and the correct answer includes using "guardrails or hooks so injected instructions cannot trigger sensitive actions"; the rationale rejects the prompt-only option because "a polite request (C) is not an enforceable control" ([CCDV-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf)). Both items, with every option, are on the [CCAR-F](../claude-certified-architect-foundations.md#official-sample-questions) and [CCDV-F](../claude-certified-developer.md#official-sample-questions) pages.

The Claude Code docs agree. The [memory docs](https://code.claude.com/docs/en/memory) say Claude treats CLAUDE.md and auto memory as context, "not enforced configuration", and point to a `PreToolUse` hook to block an action regardless of what Claude decides. The [best-practices page](https://code.claude.com/docs/en/best-practices) adds: "Unlike CLAUDE.md instructions which are advisory, hooks are deterministic and guarantee the action happens." The [features overview](https://code.claude.com/docs/en/features-overview) puts it as a rule: "If a rule must hold every time, make it a hook rather than a prompt instruction."

The table below is our summary of those sources.

| Requirement | Mechanism | Why |
|---|---|---|
| Must hold every time (never edit `.env`, block `rm -rf`, format after every edit, tests must pass before Claude stops) | Hook (a permission deny rule only when the whole tool or a fixed path is banned) | Runs on every matching event regardless of what Claude decides |
| A tool call that must never be allowed, whatever its arguments | Permission deny rule | Deny rules block in every mode, including `bypassPermissions` |
| A conditional check that inspects the call (the command text, the file path, an amount) | `PreToolUse` hook | The hook reads `tool_input` and can deny, ask or rewrite |
| A convention Claude should usually follow (use Bun, not npm) | CLAUDE.md or a rule file | Guidance is proportionate; enforcement is overhead |

Three facts decide layering questions. A `PreToolUse` hook fires before any permission-mode check, so a hook `deny` blocks the tool even in `bypassPermissions` mode or with `--dangerously-skip-permissions`. Hooks can tighten restrictions but not loosen them: a hook `allow` does not override a matching deny or ask rule. And a Bash deny rule such as `Bash(rm *)` is not a security boundary around the program (it does not stop `/bin/rm -rf build/`), which is where sandboxing or a `PreToolUse` hook comes in.

### Where hooks are configured

Hooks live under a `"hooks"` key in JSON settings files. There is no standalone hooks file for project or user configuration; only plugins use `hooks/hooks.json`.

| Location | Scope | Shareable |
|---|---|---|
| `~/.claude/settings.json` | All your projects | No, local to your machine |
| `.claude/settings.json` | Single project | Yes, can be committed to the repo |
| `.claude/settings.local.json` | Single project | No, gitignored when Claude Code saves a setting to it |
| Managed policy settings | Organization-wide | Yes, admin-controlled |
| Plugin `hooks/hooks.json` | When the plugin is enabled | Yes, bundled with the plugin |
| Skill frontmatter `hooks:` | The rest of the session once the skill is invoked | Yes, defined in the skill file |
| Subagent frontmatter `hooks:` | While that subagent is running | Yes, defined in the subagent file |

- Hook entries **merge** across settings levels instead of replacing each other, and an identical handler defined in more than one file runs once.
- Hooks load from the current working directory's `.claude/` with no parent-directory fallback.
- Hooks from settings, managed policy and plugins also run inside subagents; the input then carries `agent_id` and `agent_type`. In subagent frontmatter, a `Stop` hook is converted to `SubagentStop`.
- Cloud sessions do not read your local `~/.claude/settings.json`; their hooks come from the repository's `.claude/settings.json` (in a session with one repository), plugins synced from your claude.ai account, and your organization's server-managed settings.
- Plugin subagents ignore a `hooks` field in their frontmatter, for security.
- Team hooks belong in `.claude/settings.json` in Git; non-negotiable hooks belong in managed settings, where engineers cannot switch them off.

### Configuration shape

The configuration has three levels of nesting: the **event**, one or more **matcher groups**, and the **handlers** inside each group. This example from the hooks reference blocks `rm -rf` with a structured JSON decision:

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

```bash
#!/bin/bash
# .claude/hooks/block-rm.sh
COMMAND=$(jq -r '.tool_input.command')

if echo "$COMMAND" | grep -q 'rm -rf'; then
  jq -n '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: "Destructive command blocked by hook"
    }
  }'
else
  exit 0  # no decision; normal permission flow applies
fi
```

`${CLAUDE_PROJECT_DIR}` is the project root where the session started (it stays there even if Claude enters a worktree, while the input's `cwd` follows Claude). Plugins also get `${CLAUDE_PLUGIN_ROOT}` and `${CLAUDE_PLUGIN_DATA}`. Because `args` is set, this handler runs in exec form with no shell; without `args` a command hook runs in shell form (`sh -c` on macOS and Linux).

### Matchers and the `if` filter

| Matcher value | Meaning |
|---|---|
| `"*"`, `""` or omitted | Match every occurrence of the event |
| Only letters, digits, `_`, `-`, spaces, commas and `\|` | Exact string, or a list of exact strings: `Bash`, `Edit\|Write`, `Edit, Write` |
| Any other character | Unanchored JavaScript regular expression: `Edit.*` matches `Edit` and `NotebookEdit`; use `^Edit$` for a whole-string match |
| `mcp__memory__.*` | Every tool from the `memory` MCP server (tools are named `mcp__<server>__<tool>`; `mcp__memory` alone is an exact string that matches no tool) |

Two version and event caveats from the reference: hyphens in the exact-match set need Claude Code v2.1.195 or later (earlier versions treat `code-reviewer` as a regular expression that also fires for `senior-code-reviewer`), and `FileChanged` and `StopFailure` use a narrower exact-match set of letters, digits, `_` and `|` only.

What the matcher is compared against depends on the event:

| Event | Matches on | Values |
|---|---|---|
| `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `PermissionRequest`, `PermissionDenied` | Tool name | `Bash`, `Edit`, `Write`, MCP tool names |
| `SessionStart` | How the session started | `startup`, `resume`, `clear`, `compact`, `fork` |
| `Setup` | Which CLI flag triggered setup | `init`, `maintenance` |
| `SessionEnd` | Why it ended | `clear`, `resume`, `logout`, `prompt_input_exit`, `other` |
| `SubagentStart`, `SubagentStop` | Agent type | `general-purpose`, `Explore`, `Plan`, custom names, plugin-scoped names |
| `PreCompact`, `PostCompact` | Trigger | `manual` (`/compact`) or `auto` |
| `ConfigChange` | Configuration source | `user_settings`, `project_settings`, `local_settings`, `policy_settings`, `skills` |
| `FileChanged` | Literal filenames to watch | `.envrc\|.env` |
| `StopFailure` | Error type | `rate_limit`, `overloaded`, `authentication_failed`, `billing_error`, `server_error`, `max_output_tokens`, `unknown` and others |
| `InstructionsLoaded` | Load reason | `session_start`, `nested_traversal`, `path_glob_match`, `include`, `compact` |
| `Notification` | Notification type | `permission_prompt`, `idle_prompt`, `auth_success`, elicitation types and others |
| `UserPromptExpansion` | Command name | Your skill or command names |
| `Elicitation`, `ElicitationResult` | MCP server name | Your configured MCP server names |
| `DirectoryAdded` | How the directory was added | `slash_command`, `register_repo_root` |
| `PreModelSwitch`, `PostModelSwitch` | Canonical name of the model being switched to | `claude-opus-5`, `.*opus.*` |

`UserPromptSubmit`, `PostToolBatch`, `Stop`, `TeammateIdle`, `TaskCreated`, `TaskCompleted`, `WorktreeCreate`, `WorktreeRemove`, `MessageDisplay` and `CwdChanged` have no matcher support: they fire on every occurrence, and a matcher on them is silently ignored. Matchers are case-sensitive.

The per-handler `if` field narrows a tool-event handler with permission rule syntax, such as `"Bash(git *)"` or `"Edit(*.ts)"`. It holds exactly one rule (no `&&`, `||` or lists; define one handler per condition), and it is best-effort, so the docs say to use the permission system rather than a hook for a hard allow or deny.

### The hook events

The reference describes three cadences: per session (`SessionStart`, `SessionEnd`), per turn (`UserPromptSubmit`, `Stop`, `StopFailure`) and per tool call inside the agentic loop (`PreToolUse`, `PostToolUse`). The table lists every event in the hooks reference as of September 2026 (33 events, our count of the reference table); the grouping is ours and loosely follows those cadences.

| Group | Event | Fires |
|---|---|---|
| Session | `SessionStart` | When a session begins or resumes |
| Session | `Setup` | Only with `--init-only`, or `--init` / `--maintenance` in `-p` mode; for one-time preparation in CI or scripts |
| Session | `SessionEnd` | When a session terminates |
| Turn | `UserPromptSubmit` | When you submit a prompt, before Claude processes it |
| Turn | `UserPromptExpansion` | When a user-typed command expands into a prompt, before it reaches Claude; can block the expansion |
| Turn | `Stop` | When Claude finishes responding |
| Turn | `StopFailure` | When the turn ends due to an API error |
| Tool loop | `PreToolUse` | Before a tool call executes; can block it |
| Tool loop | `PermissionRequest` | When a tool call needs a permission decision |
| Tool loop | `PermissionDenied` | When auto mode denies a tool call |
| Tool loop | `PostToolUse` | After a tool call succeeds |
| Tool loop | `PostToolUseFailure` | After a tool call fails |
| Tool loop | `PostToolBatch` | After a full batch of parallel tool calls resolves, before the next model call |
| Delegation | `SubagentStart`, `SubagentStop` | When a subagent is spawned; when it finishes |
| Delegation | `TaskCreated`, `TaskCompleted` | When a task is created via `TaskCreate`; when it is marked completed |
| Delegation | `TeammateIdle` | When an agent team teammate is about to go idle |
| Context and config | `InstructionsLoaded` | When a CLAUDE.md or `.claude/rules/*.md` file loads, at session start and on lazy loads |
| Context and config | `ConfigChange` | When a configuration file changes during a session |
| Context and config | `PreCompact`, `PostCompact` | Before compaction; after it completes |
| Context and config | `PreModelSwitch`, `PostModelSwitch` | Before a requested model switch (can block it); after the model changes |
| Environment | `CwdChanged` | When the working directory changes (useful with tools like direnv) |
| Environment | `DirectoryAdded` | When a directory is added mid-session via `/add-dir` or the SDK `register_repo_root` request |
| Environment | `FileChanged` | When a watched file changes on disk |
| Environment | `WorktreeCreate`, `WorktreeRemove` | When a worktree is created (replaces default git behavior); when one is removed |
| Interface and MCP | `Notification`, `MessageDisplay` | When Claude Code sends a notification; while assistant text is displayed |
| Interface and MCP | `Elicitation`, `ElicitationResult` | When an MCP server requests user input; after the user responds, before the response goes back |

Study five events first. `PreToolUse` is the CCAR-F guide's tool call interception of outgoing calls, and `PostToolUse` is where tool results are transformed before the model processes them; the docs' redaction advice uses the same split (intercept at `PreToolUse` for outbound tool inputs, at `PostToolUse` for inbound tool results). `Stop` is the deterministic gate from the best-practices page: the hook runs your check as a script and blocks the turn from ending until it passes. `SessionStart` injects context, including after compaction. `UserPromptSubmit` adds context alongside a prompt.

### Handler types

| Type | What runs | Default timeout | Notes |
|---|---|---|---|
| `command` | A shell command; JSON input on stdin; results through exit codes and stdout | 600 s | Fields include `command`, `args` (exec form), `async`, `asyncRewake`, `shell` (`"bash"` or `"powershell"`) |
| `http` | POSTs the event JSON to a URL; the response body uses the same JSON output format | 600 s | Cannot block through status codes alone; header env vars resolve only if listed in `allowedEnvVars` |
| `mcp_tool` | Calls a tool on an MCP server | 600 s | The server must already be connected; the hook never triggers OAuth or a connection flow |
| `prompt` | Sends the input and your prompt to a Claude model (Haiku by default) for a single-turn decision | 30 s | `$ARGUMENTS` is replaced by the hook input JSON; optional `model` |
| `agent` | Spawns a subagent that can use Read, Grep and Glob for up to 50 turns | 60 s | Experimental; the docs prefer command hooks for production |

- Every handler accepts `type`, `if`, `timeout`, `statusMessage` (custom spinner text) and `once` (remove after the first successful run; honored only in skill frontmatter).
- The 600-second default for `command`, `http` and `mcp_tool` drops to 30 seconds on `UserPromptSubmit`, `PreModelSwitch` and `PostModelSwitch`, and to 10 on `MessageDisplay`. `SessionEnd` hooks share a 1.5-second budget, raised to match a longer per-hook timeout up to 60 seconds.
- `SessionStart` and `Setup` support only `command` and `mcp_tool` handlers. An `mcp_tool` hook on `SessionStart` is skipped at launch (MCP servers are not yet connected) and one on `Setup` is always skipped, so use a `command` hook for anything the first turn needs and for all `Setup` work.
- The events that accept all five types are `PreToolUse`, `PostToolUse`, `PostToolUseFailure`, `PostToolBatch`, `PermissionDenied`, `Stop`, `SubagentStop`, `TaskCreated`, `TaskCompleted`, `TeammateIdle`, `UserPromptExpansion` and `UserPromptSubmit`. `PermissionRequest` accepts every type except `agent`. The remaining events, apart from `SessionStart` and `Setup`, accept `command`, `http` and `mcp_tool` only.
- All matching hooks run in parallel, and one hook's `deny` does not stop sibling hooks from running to completion.

### What a hook receives

Command hooks read the event's JSON on stdin; HTTP hooks receive it as the POST body. A `PreToolUse` input for a Bash call looks like this:

```json
{
  "session_id": "abc123",
  "prompt_id": "550e8400-e29b-41d4-a716-446655440000",
  "transcript_path": "/home/user/.claude/projects/.../transcript.jsonl",
  "cwd": "/home/user/my-project",
  "scratchpad_dir": "/tmp/claude-1000/-home-user-my-project/abc123/scratchpad",
  "permission_mode": "default",
  "hook_event_name": "PreToolUse",
  "tool_name": "Bash",
  "tool_input": {
    "command": "npm test",
    "description": "Run test suite",
    "timeout": 120000,
    "run_in_background": false
  },
  "tool_use_id": "toolu_01ABC123..."
}
```

- `permission_mode` is one of `default`, `plan`, `acceptEdits`, `auto`, `dontAsk` or `bypassPermissions`; the mode labeled Manual in the interface arrives as `default`.
- Tool events add `tool_name`, `tool_input` and `tool_use_id`; `PostToolUse` also adds `tool_response`. For `Write`, `Edit` and `Read`, `tool_input.file_path` is always absolute.
- Inside a subagent or with `--agent`, the input also carries `agent_type` (and `agent_id` for subagent calls).
- `Stop` and `SubagentStop` both receive `stop_hook_active` and `last_assistant_message` (the text of Claude's final response, so the hook need not parse the transcript).
- `PreToolUse` fires only when Claude calls a tool. A file referenced with `@` in your prompt is inserted without a tool call, so no `PreToolUse` hook sees it; block such reads with a `Read` deny rule.

### Exit codes

| Exit | Meaning | Effect |
|---|---|---|
| `0` | Success | Stdout is parsed as JSON only if it starts with `{` and ends with `}`. Otherwise stdout goes to the debug log, except on `UserPromptSubmit`, `UserPromptExpansion`, `SessionStart` and `PostModelSwitch`, where plain stdout is added as context Claude can see |
| `2` | Blocking error | On events that can block, blocks whether or not JSON is printed; even a JSON `permissionDecision` of `"allow"` cannot override it. The message is the JSON reason if present, otherwise stderr |
| Any other, including `1` | Non-blocking error for most events | With JSON that passes validation, the JSON alone decides (on events that use the standard decision model). Otherwise the action proceeds, and a hook error notice shows the first line of stderr (plain or empty stdout) or the parse or validation message (JSON that fails) |

The hooks reference is blunt about exit codes: "If your hook is meant to enforce a policy, use `exit 2`." Exit code 1, the usual Unix failure code, lets the action through when stdout carries no valid JSON. Two more failure modes also let it through: a hook that cannot start (for example exit 127 from a wrong path), where "a mistyped path in `settings.json` leaves the gate silently disabled", and a stalled one, since "A timed-out `command`, `http`, or `mcp_tool` hook doesn't block the tool call" ([hooks reference](https://code.claude.com/docs/en/hooks)). HTTP hooks cannot block by status code at all: return a 2xx response with a JSON decision body.

What exit code 2 does depends on the event:

| Event | Effect of exit 2 |
|---|---|
| `PreToolUse` | Blocks the tool call |
| `UserPromptSubmit` | Blocks prompt processing and erases the prompt |
| `Stop` / `SubagentStop` | Prevents Claude (or the subagent) from stopping; the conversation continues |
| `PostToolBatch` | Stops the agentic loop before the next model call |
| `PreCompact` | Blocks compaction |
| `ConfigChange` | Blocks the change, except `policy_settings` changes |
| `TaskCreated` / `TaskCompleted` | Rolls back the task / prevents completion |
| `TeammateIdle` | Stderr becomes feedback and the teammate keeps working |
| `WorktreeCreate` | Any non-zero exit fails worktree creation |
| `PostToolUse` | Cannot block (the tool already ran); stderr is shown to Claude |
| `PermissionRequest` | Not honored; deny through the decision object instead |
| `SessionStart`, `SessionEnd`, `SubagentStart` | Stderr is shown to the user only |

The exit-code style, as the hooks reference shows it:

```bash
#!/bin/bash
# Reads JSON input from stdin, checks the command
input=$(cat)
command=$(jq -r '.tool_input.command' <<<"$input")

if [[ "$command" == rm* ]]; then
  echo "Blocked: rm commands are not allowed" >&2
  exit 2  # Blocking error: tool call is prevented
fi

exit 0  # No decision: the normal permission flow applies
```

### JSON output

Pick one approach per hook: exit codes alone, or exit 0 and print JSON. When printing JSON, stdout must contain only the JSON object; unconditional `echo` lines in a shell profile get prepended to it and are a common reason hook JSON is ignored.

| Field | Applies to | Effect |
|---|---|---|
| `continue` | All events | Default `true`. `false` stops Claude entirely and takes precedence over any event-specific decision |
| `stopReason` | All events | Message shown to the user when `continue` is `false` |
| `systemMessage` | All events | Warning shown to the user |
| `suppressOutput` | All events | Accepted but has no effect |
| `terminalSequence` | All events | An allowlisted terminal escape sequence (notification, title, bell); hooks have no `/dev/tty` |
| `decision: "block"` plus `reason` | `UserPromptSubmit`, `UserPromptExpansion`, `PostToolUse`, `PostToolUseFailure`, `PostToolBatch`, `Stop`, `SubagentStop`, `ConfigChange`, `PreCompact`, `TaskCreated` (cancels the task), `PreModelSwitch` (cancels the switch; it also accepts `hookSpecificOutput`) | `"block"` is the only value; to allow, omit it |
| `hookSpecificOutput` | Events that need richer control | Must include `hookEventName` set to the event name |
| `additionalContext` | Inside `hookSpecificOutput` | Wrapped in a system reminder where the hook fired; Claude reads it on the next request |

`additionalContext`, `systemMessage`, `initialUserMessage` and plain stdout are each capped at 10,000 characters (overflow is saved to a file and replaced by its path and a preview of up to 2,000 characters; Claude is not asked to read the file, so keep what Claude must see within the cap). Write `additionalContext` as factual statements; text phrased as imperative system instructions can trip prompt-injection defenses.

Three output shapes from the hooks reference: stop Claude entirely; block with a top-level `decision`; and a `PreToolUse` decision that also rewrites the input and adds context.

```json
{ "continue": false, "stopReason": "Build failed, fix errors before continuing" }
```

```json
{
  "decision": "block",
  "reason": "Test suite must pass before proceeding"
}
```

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "allow",
    "permissionDecisionReason": "My reason here",
    "updatedInput": {
      "field_to_modify": "new value"
    },
    "additionalContext": "Current environment: production. Proceed with caution."
  }
}
```

#### Event-specific decisions

| Event | Output | Behavior to remember |
|---|---|---|
| `PreToolUse` | `hookSpecificOutput.permissionDecision`: `allow`, `deny`, `ask`, `defer`; `permissionDecisionReason`; `updatedInput` | `allow` skips the prompt (except for actions no mode auto-approves, and for `AskUserQuestion` and `ExitPlanMode`, which need `updatedInput` with it), but deny and ask rules still apply. The reason goes to Claude on `deny`, to the user on `allow` and `ask`. `updatedInput` replaces the whole input, so include unchanged fields. Across hooks: `deny` > `defer` > `ask` > `allow` |
| `PermissionRequest` | `hookSpecificOutput.decision.behavior`: `allow` or `deny`; `updatedInput` / `updatedPermissions` with allow; `message` / `interrupt` with deny | Runs only when Claude Code is about to ask for permission (`PreToolUse` runs before every call). An `allow` does not override a matching deny rule |
| `PostToolUse` | `decision: "block"` with `reason`; `updatedToolOutput` | `block` adds the reason next to the result, and Claude still sees the original output. `updatedToolOutput` must match the tool's output shape and changes only what Claude sees; the tool already ran |
| `UserPromptSubmit` | `decision: "block"`; `additionalContext` | `block` erases the prompt and shows the reason to the user. It cannot replace the prompt |
| `Stop`, `SubagentStop` | `decision: "block"` with a required `reason` | The reason tells Claude why to continue. After 8 consecutive blocks Claude Code ends the turn anyway (`CLAUDE_CODE_STOP_HOOK_BLOCK_CAP`, `0` disables the cap) |
| `SessionStart` | `additionalContext`, `initialUserMessage`, `sessionTitle`, `watchPaths`, `reloadSkills` | Can write `export` lines to `CLAUDE_ENV_FILE` to persist environment variables for later Bash commands |
| `SubagentStart` | `additionalContext` | Cannot block creation; injects context into the subagent |
| `PermissionDenied` | `hookSpecificOutput.retry: true` | Tells the model it may retry the call auto mode denied |

`defer` pauses a run so an outside system can decide later. It is honored only in non-interactive `-p` mode, only when Claude makes a single tool call in the turn; the process exits with `stop_reason: "tool_deferred"` and the pending call preserved, and the integration resumes with `claude -p --resume <session-id>`.

### Prompt, agent and async hooks

Prompt and agent hooks let a model make the call instead of a script. A prompt hook's model returns `{"ok": true|false, "reason": ..., "impossible": true|false}`, with `reason` required when `ok` is `false`; an agent hook returns `{ "ok": true }` or `{ "ok": false, "reason": "..." }` and does not support `impossible`.

What `ok: false` does depends on the event:

- `Stop` and `SubagentStop`: the reason is fed back to Claude as its next instruction and the turn continues, unless a prompt hook also sets `impossible: true`, in which case the turn ends.
- `PreToolUse`: the call is denied. For a prompt hook the turn then ends by default, with the reason shown as a warning line; the prompt-hook field `continueOnBlock: true` instead returns the reason to Claude as the tool error so it can adjust. Claude Code handles an agent hook's `ok: false` the way it handles a prompt hook with `continueOnBlock: true`.
- `PermissionRequest`: `ok: false` has no effect; deny from a command hook's decision object instead.

The `/goal` command is a built-in shortcut for a session-scoped prompt-based `Stop` hook.

=== "Prompt hook"

    ```json
    {
      "hooks": {
        "Stop": [
          {
            "hooks": [
              {
                "type": "prompt",
                "prompt": "Evaluate if Claude should stop: $ARGUMENTS. Check if all tasks are complete."
              }
            ]
          }
        ]
      }
    }
    ```

=== "Agent hook"

    ```json
    {
      "hooks": {
        "Stop": [
          {
            "hooks": [
              {
                "type": "agent",
                "prompt": "Verify that all unit tests pass. Run the test suite and check the results. $ARGUMENTS",
                "timeout": 120
              }
            ]
          }
        ]
      }
    }
    ```

A command `Stop` hook that keeps blocking should check `stop_hook_active` (true when Claude is already continuing because of a stop hook) and exit early:

```bash
#!/bin/bash
INPUT=$(cat)
if [ "$(echo "$INPUT" | jq -r '.stop_hook_active')" = "true" ]; then
  exit 0  # Allow Claude to stop
fi
# ... rest of your hook logic
```

`Stop` hooks fire whenever Claude finishes responding, not only at task completion, and not on user interrupts; an API error fires `StopFailure` instead.

`"async": true` (command hooks only) runs a hook in the background. Async hooks cannot block or control behavior; their `additionalContext` and `systemMessage` reach Claude on the next turn, and in `-p` mode any async hook still running at teardown is killed. `asyncRewake: true` also runs in the background but wakes Claude on exit code 2.

### Worked examples

**Protect sensitive files.** This example from the hooks guide is a `PreToolUse` hook on `Edit|Write` that exits 2 when the target path contains `.env`, `package-lock.json` or `.git/`. On macOS and Linux, hook scripts must be executable (`chmod +x .claude/hooks/protect-files.sh`).

```bash
#!/bin/bash
# protect-files.sh

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')
# Normalize Windows backslash separators so the patterns below match
FILE_PATH="${FILE_PATH//\\//}"

PROTECTED_PATTERNS=(".env" "package-lock.json" ".git/")

for pattern in "${PROTECTED_PATTERNS[@]}"; do
  if [[ "$FILE_PATH" == *"$pattern"* ]]; then
    echo "Blocked: $FILE_PATH matches protected pattern '$pattern'" >&2
    exit 2
  fi
done

exit 0
```

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/protect-files.sh"
          }
        ]
      }
    ]
  }
}
```

**Format after every edit.** A `PostToolUse` hook on `Edit|Write` runs Prettier on the edited path:

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "jq -r '.tool_input.file_path' | xargs npx prettier --write"
          }
        ]
      }
    ]
  }
}
```

**Scope a hook to a skill.** Hooks in skill frontmatter register when the skill is invoked and keep running for the rest of the session:

```yaml
---
name: secure-operations
description: Perform operations with security checks
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/security-check.sh"
---
```

Two more examples sit with the workflows they serve: auto-approving `ExitPlanMode` in [Plan mode or direct execution](#plan-mode-or-direct-execution), and re-injecting context after compaction in [Managing context in large codebases](#managing-context-in-large-codebases).

**Mapping the exam's scenarios.** Two of CCAR-F Task 1.5's three skills are hook implementations (the third, 1.5-S3, is choosing hooks over prompt-based enforcement), and Scenario 1's customer support agent (with MCP tools such as `process_refund` and `escalate_to_human`) is where they fit: intercept tool calls that break policy (the guide's example is refunds exceeding &#36;500) and redirect them to an alternative workflow such as human escalation (1.5-S2); and normalize Unix timestamps, ISO 8601 dates and numeric status codes from different MCP tools before the agent processes them (1.5-S1). Exercise 1 step 4 asks you to build the first one. The mapping to hook events is ours, following the docs' naming: the first is a `PreToolUse` hook matched to the refund tool that inspects `tool_input` and returns `deny` with a `permissionDecisionReason` (shown to Claude, so it can name the escalation path); the second is a `PostToolUse` hook that replaces the result with `updatedToolOutput`. Skill 1.4-S1's prerequisite gate (block `process_refund` until `get_customer` has returned a verified customer ID) is the same kind of interception, keyed on what has already happened in the session. The guide places Task 1.5 under the Agent SDK, where the same events are registered as callbacks; see [Hooks in the SDK](agents-and-agent-sdk.md#hooks-in-the-sdk).

### Security, trust and organizational control

- Command hooks run with your full user permissions; review and test them before adding them. The docs' checklist: validate inputs, quote shell variables, block path traversal (`..`), use absolute paths, skip sensitive files such as `.env` and `.git/`.
- In interactive sessions, Claude Code holds back hooks from every settings file until you accept the workspace trust dialog. In `-p` and SDK sessions there is no dialog and the folder is treated as trusted, so hooks committed to a repository's `.claude/settings.json` run in a folder you never trusted. Before scripting `claude -p` over a repository you did not write, review its `.claude/` settings, use `--bare`, or pass `--settings '{"disableAllHooks": true}'`.
- `"disableAllHooks": true` turns hooks off; there is no way to disable one hook while keeping it configured. Set in user, project or local settings, it cannot disable managed hooks; only `disableAllHooks` at the managed level can.
- `allowManagedHooksOnly` (managed settings only) blocks user, project, local and plugin hooks; hooks from plugins force-enabled in managed settings are exempt. `allowedHttpHookUrls` restricts which URLs HTTP hooks may call, and `httpHookAllowedEnvVars` restricts which environment variables may be interpolated into their headers.

Hook governance for teams is covered further in [Claude Code security controls](security-and-governance.md#claude-code-security-controls) and [Managed settings for organizations](claude-code-configuration.md#managed-settings-for-organizations).

### Debugging hooks

- `/hooks` opens a read-only browser of configured hooks (event, matcher, type, source). To change a hook, edit the settings JSON.
- Test a script by piping sample input into it and checking the exit status: `echo '{"tool_name":"Bash","tool_input":{"command":"ls"}}' | ./my-hook.sh`.
- Execution details go to the debug log: run `claude --debug-file <path>`, or `claude --debug` and read `~/.claude/debug/<session-id>.txt`.
- Command hooks cannot trigger `/` commands or tool calls.
- If several `PreToolUse` hooks return `updatedInput` for the same call, the last to finish wins, non-deterministically; let only one hook rewrite a given input.

!!! warning "Exam guide vs current docs"

    The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) (July 2026) describes hooks by pattern (tool call interception, `PostToolUse` hooks that transform results) and places them under the Agent SDK. Current docs add detail the guide does not mention: five handler types (`command`, `http`, `mcp_tool`, `prompt`, `agent`); `updatedToolOutput` replaces any tool's output, while the older MCP-only `updatedMCPToolOutput` is marked deprecated in the Agent SDK hooks docs (the Claude Code hooks reference says to prefer `updatedToolOutput`); on `PreToolUse`, the old top-level `decision` / `reason` are deprecated in favor of `hookSpecificOutput.permissionDecision`, with `"approve"` and `"block"` mapping to `"allow"` and `"deny"`. On the exam, answer with the guide's concepts: interception before the call to block, a `PostToolUse` hook to normalize results, hooks over prompts when compliance must be guaranteed.

**Traps**

- Answers that add a firmer instruction to the system prompt or CLAUDE.md, or few-shot examples, when the requirement is guaranteed compliance. The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) rules this out: prompt instructions alone have "a non-zero failure rate", and sample question 1 rejects both options as probabilistic compliance.
- A system prompt line asking users not to include malicious instructions. CCDV-F Sample 2's rationale: a polite request is not an enforceable control.
- A routing classifier that changes which tools are available when the problem is the order of tool calls. Sample question 1's rationale: it "addresses tool availability rather than tool ordering".
- A hook that exits with `exit 1` to block. Only `exit 2` (or a JSON `deny` / `block`) blocks; exit 1 proceeds.
- `PostToolUse` to prevent an action. It runs after the tool ran; prevention belongs in `PreToolUse`.
- A hook `allow` to get past a deny rule. Hooks tighten; they do not loosen.
- A matcher on `Stop` or `UserPromptSubmit` to filter them. It is silently ignored.

## Plan mode or direct execution

*Tested in: CCAR-F 3.4 (Scenario 2, sample question 5, Exercise 2 step 5), appendix in-scope plan mode item · CCDV-F Claude Code Operation (our mapping: its description lists session management, slash commands, headless mode, streaming mode and auto-mode, and does not name plan mode) · CCAR-P 7.1, 7.2*

The [permission modes page](https://code.claude.com/docs/en/permission-modes) defines the feature in one line: "Plan mode tells Claude to research and propose changes without making them." In plan mode Claude reads files, runs shell commands to explore and writes a plan, but does not edit your source; edits stay blocked until you approve the plan (the exception, interactive terminal sessions with bypass permissions available, is covered in the box below). Direct execution, in the exam guide's sense, is the opposite choice: let Claude make the change straight away.

### The decision rule

The CCAR-F guide states the rule in Task 3.4, and the Claude Code best-practices page gives the same rule in everyday terms.

| Signal in the task | Choose | Examples from the guide and docs |
|---|---|---|
| Large-scale change across many files | Plan mode | Microservice restructuring; a library migration affecting 45+ files |
| Several valid approaches, or an architectural decision | Plan mode | Choosing between integration approaches with different infrastructure requirements |
| Unfamiliar code or an uncertain approach | Plan mode | The docs: planning is most useful when you are uncertain about the approach, the change modifies multiple files, or you are unfamiliar with the code |
| Simple, well-scoped change, or a well-understood one with clear scope | Direct execution | Adding a single validation check to one function; a single-file bug fix with a clear stack trace; adding a date validation conditional |
| A trivial edit | Direct execution | The docs: fix a typo, add a log line, rename a variable |

The best-practices page compresses it into one test: "If you could describe the diff in one sentence, skip the plan" ([best practices](https://code.claude.com/docs/en/best-practices)). The reason to plan at all is the [guide's](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) 3.4-K3: plan mode "enables safe codebase exploration and design before committing to changes, preventing costly rework". The cost page makes the same point about money: planning prevents "expensive re-work when the initial direction is wrong" ([costs](https://code.claude.com/docs/en/costs)). The trade-off is that plan mode "adds overhead", so for small tasks with a clear scope the best-practices page says to ask Claude to do it directly.

Task 3.4 also covers the Explore subagent for verbose discovery phases (3.4-K4, 3.4-S3). Explore is a built-in subagent rather than a permission mode, and it is taught in [Managing context in large codebases](#managing-context-in-large-codebases).

**Combine them.** [Task 3.4-S4](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) is "Combining plan mode for investigation with direct execution for implementation (e.g., planning a library migration, then executing the planned approach)". The docs' recommended workflow has the same shape in four phases: Explore in plan mode, Plan (ask for a detailed implementation plan), Implement (leave plan mode by approving the plan or pressing `Shift+Tab`, then code against the plan), Commit (a descriptive message and a PR).

CCAR-F sample question 5 tests exactly this with a monolith-to-microservices restructuring that involves changes across dozens of files and decisions about service boundaries; the correct answer is to enter plan mode first. Read it with Anthropic's rationale in [the CCAR-F sample questions](../claude-certified-architect-foundations.md#official-sample-questions). Its rationale names the three wrong-answer shapes: starting directly "risks costly rework when dependencies are discovered late", detailed upfront instructions "assumes you already know the right structure without exploring the code", and starting directly with a switch later "ignores that the complexity is already stated in the requirements, not something that might emerge later" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

[Exercise 2 step 5](../claude-certified-architect-foundations.md#official-preparation-exercises) in the same guide is the hands-on version: try plan mode and direct execution on a single-file bug fix, a multi-file library migration, and a feature with multiple valid implementation approaches, and observe when plan mode provides value.

### Using plan mode

| Action | How |
|---|---|
| Plan a single prompt | Prefix it with `/plan`, for example `/plan fix the auth bug` |
| Toggle during a session | `Shift+Tab` cycles `default` → `acceptEdits` → `plan` → back to `default` (optional modes slot in after `plan`); the status bar shows `⏸ plan mode on` |
| Start a session in plan mode | `claude --permission-mode plan` |
| Make plan mode the project default | `"defaultMode": "plan"` under `permissions` in `.claude/settings.json` (terminal sessions; VS Code extension conversations use `claudeCode.initialPermissionMode` instead) |
| Leave without approving | Press `Shift+Tab` again |
| Edit the plan yourself | `Ctrl+G` opens it in your default text editor |
| Approve | **Yes, and use auto mode** (it reads **Yes, auto-accept edits** when auto mode is not available to the session), **Yes, manually approve edits**, or **No, keep planning** |
| Approve and drop the planning context | Enable `showClearContextOnPlanAccept` to add that option |

```bash
claude --permission-mode plan
```

```json
{
  "permissions": {
    "defaultMode": "plan"
  }
}
```

What happens around the plan:

- When Claude needs to understand the codebase during planning, it delegates research to the built-in **Plan** subagent, so exploration output stays in a separate context window while the main conversation remains read-only. Plan (like Explore) skips CLAUDE.md files and the git status snapshot to stay fast and inexpensive.
- Approving a plan exits plan mode and switches the session to the permission mode the chosen option describes, so Claude starts editing. It also gives the session a generated title based on the plan, unless you already named it.
- The plan Claude wrote is re-injected from disk after compaction, so it survives where conversation history may not.
- Plan mode keeps its blocks in non-interactive `-p` runs, Agent SDK sessions and the VS Code chat panel; hooks see `permission_mode` set to `"plan"`.

A `PermissionRequest` hook can approve `ExitPlanMode` automatically; when it does, Claude Code exits plan mode and restores the permission mode that was active before, and it always keeps the current conversation (the hook path cannot clear context the way the approval dialog can). The hooks guide's example below goes in `~/.claude/settings.json`. Keep the matcher narrow: the [hooks guide](https://code.claude.com/docs/en/hooks-guide) warns that `.*` or an empty matcher "would auto-approve every tool permission prompt, including file writes and shell commands." A `PreToolUse` hook on `ExitPlanMode` also receives the plan itself (`plan`, in Markdown) and `planFilePath`, so a hook can inspect the plan before Claude leaves plan mode.

```json
{
  "hooks": {
    "PermissionRequest": [
      {
        "matcher": "ExitPlanMode",
        "hooks": [
          {
            "type": "command",
            "command": "echo '{\"hookSpecificOutput\": {\"hookEventName\": \"PermissionRequest\", \"decision\": {\"behavior\": \"allow\"}}}'"
          }
        ]
      }
    ]
  }
}
```

For teams, plan mode doubles as a review gate. Claude Academy's [AI-native SDLC playbook](https://academy.claude.com/courses/ai-native-sdlc-playbook/plan-mode) wants design review to happen before any code is generated, and notes that "Plan mode enforces this itself, since Claude cannot edit files until the engineer accepts the plan". It recommends committing the approved plan as `plan.md` so the plan joins the audit trail. The full set of permission modes, and which mode a session starts in, is covered in [Permissions and permission modes](claude-code-configuration.md#permissions-and-permission-modes).

!!! warning "Exam guide vs current docs"

    The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) describes plan mode as enabling "safe codebase exploration and design before committing to changes". Current docs add two nuances. When auto mode is available and `useAutoModeDuringPlan` is on (the default), a classifier reviews shell commands during planning and approved ones run. In interactive terminal sessions with bypass permissions available, plan mode's blocks are not enforced: Claude is still instructed to plan without editing, but an edit or shell command it attempts runs without prompting. On the exam, treat plan mode as the guide does: explore and design first, with no source edits until the plan is approved.

**Traps**

- Starting in direct execution and letting the implementation reveal the boundaries (sample question 5, option B). The rationale in the [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): it "risks costly rework when dependencies are discovered late".
- Direct execution with exhaustive upfront instructions instead of planning (option C). Detailed instructions assume you already know the right structure; plan mode is for finding it.
- Starting in direct execution and switching to plan mode only if complexity appears (option D). When the task description already states the complexity, plan first.
- Plan mode for a one-line fix. It adds overhead with nothing to decide.
- Treating the Explore subagent and plan mode as the same thing. Plan mode is a permission mode; Explore is a subagent that keeps discovery output out of the main context (see [Managing context in large codebases](#managing-context-in-large-codebases)).

## Iterative refinement

*Tested in: CCAR-F 3.5, appendix in-scope iterative refinement item · CCDV-F Prompt Engineering (iterative refinement) · CCAO-F D1.3, D7.2 (concept only, in the Claude apps) · CCAR-P 7.2*

Iterative refinement means getting from a first attempt to a correct result through a deliberate loop rather than hoping the first prompt lands. CCAR-F Task 3.5, "Apply iterative refinement techniques for progressive improvement", names four techniques, and each one answers a different failure.

| Technique | Use it when | Guide objective |
|---|---|---|
| Concrete input/output examples | Prose descriptions of a transformation are interpreted inconsistently | 3.5-K1, 3.5-S1, 3.5-S4 |
| Test-driven iteration | Correct behavior can be checked by tests | 3.5-K2, 3.5-S2 |
| The interview pattern | The domain is unfamiliar and there are considerations you may not have anticipated | 3.5-K3, 3.5-S3 |
| One message or one at a time | Several problems need fixing: interacting ones together, independent ones sequentially | 3.5-K4, 3.5-S5 |

### Show the transformation with examples

The guide calls concrete input/output examples "the most effective way to communicate expected transformations when prose descriptions are interpreted inconsistently", and the matching skill is "Providing 2-3 concrete input/output examples" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The same move fixes edge cases: 3.5-S4 gives specific test cases with example input and expected output, such as null values in migration scripts.

The Claude Code [best-practices page](https://code.claude.com/docs/en/best-practices) shows the pattern. Instead of "implement a function that validates email addresses", give the cases and the check:

```text
write a validateEmail function. example test cases: user@example.com is true, invalid is false, user@.com is false. run the tests after implementing
```

Few-shot technique in general (how many examples, how to vary them, where to put them) is taught in [Few-shot examples](prompt-engineering.md#few-shot-examples).

### Test-driven iteration

The guide's version: write test suites covering expected behavior, edge cases and performance requirements before implementation, then iterate by sharing test failures. The reason it works is that it gives Claude a target it can check; the docs' version of that practice, and the four ways to make a turn wait on the check, are in [Give Claude a way to verify its work](#give-claude-a-way-to-verify-its-work).

Anthropic's engineering post on Claude Code best practices laid the loop out step by step, calling it "an Anthropic-favorite workflow for changes that are easily verifiable with unit, integration, or end-to-end tests". Its URL now redirects to the docs page; the recipe below is from the [June 2025 archived copy](https://web.archive.org/web/20250602202220/https://www.anthropic.com/engineering/claude-code-best-practices):

1. Ask Claude to write tests from expected input/output pairs, and say explicitly that you are doing test-driven development so it avoids mock implementations.
2. Have Claude run the tests and confirm they fail, without writing implementation code yet.
3. Commit the tests.
4. Ask Claude to write code that passes the tests without modifying them, and to keep going until all pass. It usually takes a few write, run, adjust cycles.
5. Optionally, have independent subagents check that the implementation is not overfitting to the tests; then commit the code.

Current guidance keeps the same core. For bug fixes, the [best-practices page](https://code.claude.com/docs/en/best-practices) shows an example prompt that ends "write a failing test that reproduces the issue, then fix it". One Claude can write tests, then another write code to pass them. The [prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) warn that Claude can focus too heavily on making tests pass at the expense of more general solutions; their sample prompt asks for a solution that works for all valid inputs, not just the test cases, and says: "Tests are there to verify correctness, not to define the solution."

Two hooks make the loop enforceable rather than a request. A `Stop` hook can run the test suite and block the turn from ending until it passes. And Claude Academy's [AI-native SDLC playbook](https://academy.claude.com/courses/ai-native-sdlc-playbook/give-claude-a-feedback-loop) protects the check itself: have Claude reproduce the bug as a test and confirm it fails, commit that test, then have Claude make it pass without editing it, with a hook blocking edits to test files during the fix. See [Hooks](#hooks).

!!! warning "Exam guide vs current docs: where test-driven iteration comes from"

    CCAR-F 3.5-K2 teaches "Test-driven iteration: writing test suites first, then iterating by sharing test failures to guide progressive improvement" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The docs' best-practices page, where the engineering post's URL now redirects, has no standalone TDD section: it keeps the pieces described above (the verification loop, the failing-test bug fix, separate test and code writers), and the full five-step recipe survives only in the archived copy. Use the guide's wording on the exam (as of September 2026).

### The interview pattern

The guide's 3.5-K3 is "having Claude ask questions to surface considerations the developer may not have anticipated before implementing", and skill 3.5-S3 names cache invalidation strategies and failure modes as design considerations to surface before implementing in unfamiliar domains ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The Claude Code docs describe the same flow for larger features: start with a minimal prompt, ask Claude to interview you using the `AskUserQuestion` tool, have it write the complete spec to `SPEC.md`, then start a fresh session to execute the spec with clean context.

`AskUserQuestion` is on the short list of tools removed from every subagent, so an interview that uses it runs in the main conversation, not inside a delegated task.

### All issues at once, or one at a time

The guide's rule (3.5-K4, 3.5-S5): when problems **interact**, address all of them in a single detailed message; when problems are **independent**, fix them sequentially. The reason, in our words: when fixes interact, a fix for one changes what the other needs, so Claude has to see them together.

The two Anthropic sources below do not discuss the interacting-versus-independent distinction directly; in our reading, they are the nearest examples. For a batch of separate issues, the [archived post](https://web.archive.org/web/20250602202220/https://www.anthropic.com/engineering/claude-code-best-practices) has Claude write lint errors to a Markdown checklist and "address each issue one by one, fixing and verifying before checking it off". For linked failures, the [Managed Agents cookbook on fixing failing tests](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/managed_agents/CMA_iterate_fix_failing_tests.ipynb) plants two bugs so that a third test, `test_mean`, is downstream of the other two: once `add` and `divide` are fixed it passes on its own, with no edit to `mean()`, which the notebook says "quietly teaches the agent not to over-fix". The archived post also notes that thoroughly explaining the task at the beginning gives the best results, though you can also course-correct at any time.

### Correcting course

In Claude Code, the docs' course-correction habits (be specific up front, stop with `Esc`, rewind, and `/clear` after repeated corrections) are taught in [Be specific and give rich context](#be-specific-and-give-rich-context) and [Communicate and manage the session](#communicate-and-manage-the-session).

The same habits apply in the Claude apps, which is what CCAO-F tests. Claude Academy's [Claude 101](https://academy.claude.com/courses/claude-101/getting-better-results) advice: treat the first prompt as the start of a conversation, not a one-shot request, and give specific feedback ("Cut the first two paragraphs and make the conclusion more action-oriented" beats "Make it shorter"). The [Help Center's artifacts article](https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them) adds that editing a prior message creates a different version of the conversation, so you can explore another direction without losing previous work. Prompt iteration as a discipline is covered in [Prompt versioning and iteration](prompt-engineering.md#prompt-versioning-and-iteration).

**Traps**

- Re-explaining a transformation in longer prose after Claude misreads it. The guide's answer is 2 to 3 concrete input/output examples.
- Fixing interacting problems one message at a time. The guide's skill is a single detailed message when fixes interact; sequential fixes are for independent issues.
- Accepting a result as finished without a check Claude can run, or letting Claude edit the tests to make them pass.
- Pushing on through a long, corrected session instead of `/clear` and a better prompt.

## Sessions: continue, resume, fork and rewind

*Tested in: CCAR-F 1.7, appendix session management item · CCDV-F Claude Code Operation (session management), Claude Application Design (session hygiene) · CCAO-F D3.4 (concept only: when to restart, summarize, or persist)*

A Claude Code session is a saved conversation tied to a project directory. Every message, tool use and result is written to a plaintext JSONL file under `~/.claude/projects/`, which is what makes resume, fork and rewind possible. A new session starts with a fresh context window and none of the previous sessions' history; what carries across sessions is auto memory and the instructions you put in CLAUDE.md.

### Commands at a glance

| Goal | Command |
|---|---|
| Reopen the most recent conversation in this directory | `claude --continue` (`-c`); prints `No conversation found to continue` if there is none |
| Choose from a list | `claude --resume` (opens the session picker) |
| Resume a named session | `claude --resume <name>` |
| Resume by ID or name, with a new prompt | `claude -r "<session>" "query"` |
| Switch conversations inside a session | `/resume [session]` (alias `/continue`) |
| Find sessions linked to a pull request | `claude --from-pr <number>` |
| Name a session | `claude -n auth-refactor` at startup, `/rename auth-refactor` during it, or `Ctrl+R` in the picker |
| Branch inside a session | `/branch [name]` |
| Branch from the command line | `claude --continue --fork-session` or `claude --resume <id> --fork-session` |
| Roll back | `/rewind` (aliases `/checkpoint`, `/undo`) or `Esc` twice on an empty prompt |
| Export | `/export` copies the conversation to the clipboard or saves it as plain text |

```bash
claude --continue
claude --resume
claude --resume auth-refactor
claude -n auth-refactor
claude --continue --fork-session
claude --resume abc123 --fork-session
claude -p --resume <session-id> --output-format json "summarize what we changed" | jq -r '.result'
```

Scripted use of sessions (capturing a `session_id` from `--output-format json` and resuming it with `-p`) is covered in [Headless mode and the CLI](#headless-mode-and-the-cli).

### Named sessions

The exam's 1.7-K1 is "Named session resumption using `--resume <session-name>` to continue a specific prior conversation", and 1.7-S1 applies it to continuing named investigation sessions across work sessions. Current Claude Code matches both: `claude --resume <name>` resumes the named session directly ([sessions docs](https://code.claude.com/docs/en/sessions)).

- Set the name with `claude -n <name>` (`--name`) or `/rename <name>`; return with `claude --resume <name>` or `/resume <name>`.
- An ambiguous name behaves differently by entry point: `claude --resume <name>` opens the picker with the name pre-filled; `/resume <name>` reports an error.
- An unnamed session gets a generated title (a short summary of the first prompt) that also works as a resume handle; the default display name (such as `my-app-3f`) does not. Accepting a plan in plan mode also generates a title, unless you already named the session. A `claude -p` run started directly from a shell or script gets no generated title.
- The docs' habit: give sessions descriptive names like `oauth-migration` and treat them like branches: "each workstream gets its own persistent context" ([best practices](https://code.claude.com/docs/en/best-practices)).

### What a resume brings back, and what it does not

| Restored on resume | Not restored |
|---|---|
| The full conversation history, including tool calls and results (a tool that was still running when the old process ended does not finish or run again) | Flags such as `--mcp-config`, `--settings`, `--plugin-dir`, `--fallback-model`, and directories added with `--add-dir` (pass them again); settings files such as `settings.json` are re-read at launch anyway |
| The model (unless it was retired, is not allowed by `availableModels`, a `--model` flag or `ANTHROPIC_MODEL`-family variable picks one at launch, or you are on a provider that uses provider-specific deployment IDs, such as Amazon Bedrock, Google Cloud's Agent Platform or Microsoft Foundry) and the agent | "Allow for this session" grants, when you fork into a new process with `--fork-session` (a `/branch` in the same process keeps them) |
| The permission mode, when you resume from a terminal with `claude --continue`, `claude --resume <session-id>` or an unambiguous `claude --resume <name>`, with exceptions (a session that ended in `bypassPermissions` or `plan` starts in the mode a new session would; one that ended in `auto` gets `auto` back only when your account still meets the auto mode requirements; one that ended in Manual starts in a settings-file `defaultMode` when one takes effect) | The permission mode on other paths: a `claude -p` resume starts in the mode a new `-p` run would use (except that, on v2.1.246 and later, a session that ended in plan mode resumes in plan mode when you pass `--permission-prompt-tool`, pass none of `--permission-mode`, `--dangerously-skip-permissions` or `--fork-session`, and do not start the run through channels), and the picker and `/resume` do not restore the stored mode |
| The active goal and unexpired scheduled tasks | Background Bash and monitor tasks |
| Context that mid-session hooks (such as `PostToolUse`) added in past turns, replayed rather than re-run, so values like timestamps or commit SHAs in it go stale; `SessionStart` hooks run again with `source` set to `resume` (or `fork`) | New system prompt flag text: by default the prompt recorded on the conversation's first request is reused, and different flag text takes effect only after compaction or in a new conversation |
| Checkpoints, so `/rewind` still works | |

- Sessions created with `claude -p` or the Agent SDK are left out of the picker and `claude --continue`, but `claude --resume <session-id>` still resumes them, and `claude -p --continue` includes them.
- Resuming the same session in two terminals without forking interleaves both conversations into one transcript. Fork instead.
- On a Pro or Max plan, resuming a session that has been inactive for more than about an hour and is over 100,000 tokens opens a dialog: **Resume from summary** (runs `/compact` immediately, keeping a summary, the most recent exchanges and up to five recently read files), **Resume full session as-is** (every detail kept, at a per-request cost that scales with the conversation's size), or **Don't ask me again**. The prompt cache has expired by then, so the next request processes the full history once whichever option you pick.

### Resume or start fresh: the exam's rule

This is the judgment CCAR-F Task 1.7 tests. The skill is "Choosing between session resumption (when prior context is mostly valid) and starting fresh with injected summaries (when prior tool results are stale)", and the knowledge behind it is "Why starting a new session with a structured summary is more reliable than resuming with stale tool results" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

The mechanism is the table above: a resumed session brings back its old tool calls and results as they were recorded, and even replayed hook context can carry stale values. Claude Code offers only partial safety nets. Checkpoints normally do not capture edits made outside Claude Code or by other sessions. Switching git branches changes the files Claude sees but not the conversation history. A `FileChanged` hook's message goes to the user as a terminal notification, not to Claude. The Edit tool notices that a file changed on disk only when an edit is attempted. So telling the resumed session what changed remains your job.

| Situation | Do this | Guide objective |
|---|---|---|
| Earlier analysis is mostly valid; a few known files changed | Resume by name and say exactly which files changed, so Claude re-analyzes those instead of re-exploring everything | 1.7-K3, 1.7-S4 |
| Earlier tool results are stale (large merge, branch switch, many edits since) | Start a new session and inject a structured summary of the findings that still hold | 1.7-K4, 1.7-S3 |
| You want to try two approaches from the same analysis | Fork, so each branch starts from the shared baseline | 1.7-K2, 1.7-S2 |
| Long session, mostly valid, expensive to carry | `/compact` with a focus, or **Resume from summary** when the resume dialog described above offers it | Docs |

A targeted re-analysis prompt names the change and bounds the work. An illustration (our example, not from the docs):

```text
Since our last session, src/auth/refresh.ts and src/auth/session.ts changed.
Re-read those two files and update your analysis of the token refresh flow;
the rest of the module is unchanged.
```

The start-fresh option has a documented counterpart. For resuming work on another host, the [Agent SDK sessions docs](https://code.claude.com/docs/en/agent-sdk/sessions) list not relying on session resume at all: capture the results you need (analysis output, decisions, file diffs) as application state and pass them "into a fresh session's prompt", which the docs say is often more dependable than shipping transcript files around. The interview pattern ends the same way: once the spec is complete, a fresh session executes it.

The appendix's session management item also names "session context isolation". The guide uses the same phrase in 3.6-K4: the session that generated code is less effective at reviewing its own changes than an independent review instance. That point is taught in [Automated code review that engineers trust](#automated-code-review-that-engineers-trust).

### Forking: fork_session, --fork-session and /branch

Branching "creates a copy of the conversation so far and switches you into it, leaving the original intact" ([sessions docs](https://code.claude.com/docs/en/sessions)). The fork gets its own session ID. The exam uses it for "independent branches from a shared analysis baseline to explore divergent approaches", for example comparing two testing strategies or two refactoring approaches after one shared codebase analysis ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

One concept, several names:

| Where | How to fork |
|---|---|
| Exam guide | `fork_session` |
| Inside a Claude Code session | `/branch [name]` |
| Claude Code CLI | `--fork-session` together with `--resume` or `--continue` |
| Agent SDK, Python | `resume=session_id` plus `fork_session=True` in `ClaudeAgentOptions` |
| Agent SDK, TypeScript | `resume: sessionId` plus `forkSession: true` in the options |

The two tabs below are excerpts from the fork example in the [Agent SDK sessions docs](https://code.claude.com/docs/en/agent-sdk/sessions); imports and error handling are omitted, and `session_id` / `sessionId` holds the ID captured from an earlier run's result message.

=== "Python"

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

=== "TypeScript"

    ```typescript
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

Two facts about forks carry most of the traps. A fork branches the conversation history, **not the filesystem**: if a forked agent edits files, those edits are real and visible to any session working in the same directory. For parallel code-changing sessions, the docs' tool is git worktrees (each one a separate checkout on its own branch); in the SDK, file checkpointing branches and reverts file changes. And a fork started with `--fork-session` in a new process does not inherit "Allow for this session" grants; you re-approve there.

Session forks are different from **subagent** forks. A subagent fork inherits the whole conversation, runs as a subagent, and returns only its final result; its first request reuses the parent's prompt cache. Current Claude Code starts one with `/subtask <task>` (v2.1.212 and later); on v2.1.161 through v2.1.211 that command was `/fork`. In current versions `/fork` instead copies the conversation into a new background session, except when agent view is turned off, where `/subtask` is unavailable and `/fork` still starts a forked subagent. The SDK side of sessions is covered in [Sessions, resumption and forking](agents-and-agent-sdk.md#sessions-resumption-and-forking).

### Checkpoints and /rewind

Claude Code captures the state of your code before each prompt that starts a turn and keeps file snapshots for the 100 most recent checkpoints in a session. Open the rewind menu with `/rewind` or `Esc` twice on an empty prompt.

| Rewind action | Effect |
|---|---|
| Restore code and conversation | Both go back to the checkpoint |
| Restore conversation | Conversation only; files stay as they are |
| Restore code | Files only; conversation stays |
| Summarize from here | Compress the conversation from this point forward, freeing context |
| Summarize up to here | Compress everything before this point, keeping later messages intact |
| Never mind | Return to the message list without making changes |

The two code restore options appear only when the selected checkpoint has tracked file changes to revert. Summarizing does not change files on disk, and the original messages stay in the session transcript.

Summarize keeps you in the same session and works like a targeted `/compact`; to branch instead, use `/branch` or `claude --continue --fork-session`. If you ran `/clear` earlier in the same process, the menu also offers `/resume <session-id> (previous session)`.

What checkpoints do **not** cover:

- Files changed by Bash commands (`rm`, `mv`, `cp`).
- Subagent edits, except those of a forked skill running in the foreground; revert the rest with git.
- Manual edits outside Claude Code and edits from other concurrent sessions (normally not captured).
- Actions on remote systems (databases, APIs, deployments).
- Symlinked and hard-linked files, which a code restore skips with a warning.

Checkpoints are for quick, session-level recovery and are not a replacement for version control.

### Where sessions are stored

Transcripts are JSONL files at `~/.claude/projects/<project>/<session-id>.jsonl`. The entry format is internal and changes between versions, so scripts that parse these files directly can break on any release; the docs point to `/export` or the script interfaces (`claude -p` with JSON output, `claude -p --resume`, the Agent SDK) instead. Retention defaults to 30 days (`cleanupPeriodDays` changes it, and also keeps checkpoint snapshots longer), and `--no-session-persistence` stops a `-p` run from being saved or resumed at all.

!!! warning "Exam guide vs current docs"

    The CCAR-F guide says `fork_session`, which is the Python Agent SDK option name. The Claude Code CLI calls the same thing `--fork-session`, the TypeScript SDK `forkSession`, and the in-session command `/branch`. Older material may use `/fork` for a subagent fork; since v2.1.212 that is `/subtask`, and `/fork` now copies the session into a background session (unless agent view is turned off). The guide's `--resume <session-name>` still matches current behavior. On the exam, use the guide's terms.

**Traps**

- Resuming a session after large code changes and trusting its earlier tool results. Stale results are the guide's reason to start fresh with a summary.
- Expecting a fork to isolate file changes. It isolates conversation history only.
- Expecting `--resume` to bring back `--mcp-config`, `--settings` or `--add-dir`. Pass them again.
- Treating `/rewind` as undo for everything, including Bash changes and deployments, or as a substitute for git.
- Opening the same session in two terminals instead of forking.

## Managing context in large codebases

*Tested in: CCAR-F 5.4, 3.4-K4 and 3.4-S3, Scenario 2 (Code Generation with Claude Code), appendix context window management item · CCDV-F Context Engineering · CCAR-P 2.4, 3.8*

The [best-practices page](https://code.claude.com/docs/en/best-practices) opens with the constraint behind everything in this section: "Claude's context window fills up fast, and performance degrades as it fills." As it fills, Claude may start forgetting earlier instructions or making more mistakes. The CCAR-F guide describes the same degradation in extended sessions: inconsistent answers, and references to typical patterns instead of the specific classes discovered earlier. This section covers the Claude Code controls; the general theory (context as a budget, crash-recovery manifests, handoffs) is in [Context Engineering and Long-Running Work](context-engineering.md#exploring-large-codebases).

### See what is using context

Run `/context` for a live breakdown of usage by category, as a colored grid with optimization suggestions, including which CLAUDE.md and auto memory files loaded. For continuous monitoring, the status line JSON exposes `context_window.used_percentage`.

### The context commands

| Command | What it does | Use it when |
|---|---|---|
| `/compact [instructions]` | Summarizes the conversation so far, optionally with focus instructions | Exploration has filled context with verbose output; before starting a long new task (`/compact focus on the auth bug fix`) |
| `/clear [name]` | Starts a new conversation with empty context (aliases `/reset`, `/new`); the previous one stays saved | Switching to an unrelated task |
| `/btw` | Asks a side question; the answer never enters conversation history | Checking a detail without growing context |
| `/rewind`, then Summarize from here / up to here | Compresses part of the conversation | Only one stretch of the conversation is bloated |
| `/autocompact [auto|<tokens>]` | Sets how full the window gets before automatic compaction (v2.1.221 and later) | Tuning when auto-compaction happens |

The CCAR-F skill 5.4-S5 is exactly the first row: "Using /compact to reduce context usage during extended exploration sessions when context fills with verbose discovery output" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Choose between the first two by what you need next. `/compact` keeps continuity, but it reads the whole conversation it summarizes, so compacting a large context is itself a large request; `/clear` costs nothing and is right when the old conversation would only crowd out the files you need next and cost tokens on every message.

Claude Code also compacts on its own as you approach the limit, so a full window does not end the session. When context fills it clears older tool outputs first, then summarizes the conversation if needed. Three environment variables tune this: `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE` (a percentage, 1 to 100, of the auto-compact window) makes auto-compaction trigger earlier and cannot raise the threshold, and it applies only in sessions that compact before the model's context limit; `DISABLE_AUTO_COMPACT=1` turns off automatic compaction but keeps `/compact`; `DISABLE_COMPACT=1` turns off both. If one large file or output refills the window after each summary, Claude Code stops after a few attempts with an `Autocompact is thrashing` error; read in chunks, compact with a focus, delegate to a subagent, or `/clear`.

### What survives compaction

| Content | After compaction |
|---|---|
| System prompt and output style | Still apply |
| Project-root CLAUDE.md and unscoped rules | Re-injected from disk |
| Auto memory | Re-injected |
| The plan written in plan mode | Re-injected from disk |
| Path-scoped rules and nested CLAUDE.md files | Summarized away; reload only when matching files are read again |
| Git status snapshot | A fresh one is read from the repository |
| Files Claude read or edited | Up to five, most recently modified first, are re-read; a file over 5,000 tokens comes back as a path reference only |
| Invoked skills | Re-injected, capped at 5,000 tokens per skill and 25,000 tokens total, oldest dropped first |
| Background commands and background subagents | Keep running; Claude is reminded which ones are still running |
| Context that hooks added earlier | Summarized with the rest; `SessionStart` hooks matching `compact` run again |
| Instructions given only in conversation | May be lost; put persistent rules in CLAUDE.md |

The decision rule follows from the table: if a rule must persist across compaction, drop the `paths:` frontmatter or move it into the project-root CLAUDE.md. For cross-package changes, plan first, because the plan file is re-injected after each compaction.

Three ways to control what compaction keeps:

1. **Compaction instructions in CLAUDE.md.** The docs' examples: "When compacting, always preserve the full list of modified files and any test commands" ([best practices](https://code.claude.com/docs/en/best-practices)), or a `# Compact instructions` section ([costs page](https://code.claude.com/docs/en/costs)).
2. **A focus on the command:** `/compact Focus on the API changes`.
3. **A `SessionStart` hook with the `compact` matcher**, which re-injects critical context after every compaction, automatic or manual.

Both snippets below are the docs' own: the first goes in the project-root CLAUDE.md ([costs page](https://code.claude.com/docs/en/costs)), the second in `.claude/settings.json` ([hooks guide](https://code.claude.com/docs/en/hooks-guide)), where plain text the command writes to stdout is added to Claude's context.

```markdown
# Compact instructions

When you are using compact, please focus on test output and code changes
```

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

Around the compaction itself, a `PreCompact` hook can block it with exit code 2 (its input carries `trigger` and any `custom_instructions` passed to `/compact`), and a `PostCompact` hook receives the generated `compact_summary`.

### Delegate exploration to subagents

Two guide objectives point here. 3.4-K4 names "The Explore subagent for isolating verbose discovery output and returning summaries to preserve main conversation context", and 5.4-S1 is spawning subagents to investigate specific questions, such as finding all test files or tracing refund flow dependencies, while the main agent keeps high-level coordination ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

- **Explore** is a built-in, fast, read-only subagent for searching and analyzing codebases (Write and Edit are denied). Claude delegates to it to search or understand code without making changes, which keeps exploration results out of the main context. When invoking it, Claude specifies a thoroughness level: quick, medium or very thorough.
- Explore and Plan skip CLAUDE.md files and the git status snapshot, and any non-fork subagent starts fresh, without your conversation history, invoked skills or files already read. If a rule must reach it (the docs' example: ignore the `vendor/` directory), restate it in the delegation prompt.
- Explore and Plan are one-shot and cannot be resumed; use general-purpose or a custom subagent for work you will continue.
- Delegation is not free: many subagents each returning detailed results can still consume significant context in the main conversation. Ask for findings, not transcripts.

Delegation prompts from the docs:

```text
Use subagents to investigate how our authentication system handles token
refresh, and whether we have any existing OAuth utilities I should reuse.
```

```text
Use a subagent to run the test suite and report only the failing tests with their error messages
```

For multi-phase investigations, the guide adds 5.4-S3: summarize the key findings of one phase before spawning subagents for the next, and inject that summary into their initial context. Subagent configuration (tools, models, frontmatter) is covered in [Subagents](claude-code-configuration.md#subagents), and the isolation principle in [Subagents as context isolation](context-engineering.md#subagents-as-context-isolation).

!!! warning "Exam guide vs current docs"

    The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) calls the subagent-spawning tool "Task". In Claude Code v2.1.63 it was renamed Agent, and `Task(...)` references still work as aliases. Older material also says Explore runs on Haiku; as of v2.1.198 it inherits the main conversation's model (capped at Opus on the Claude API), and a user or project subagent named `Explore` with `model: haiku` keeps exploration on the lower-cost model. Answer with the guide's term, and do not rely on a model detail for Explore.

### Keep findings outside the window

The guide's 5.4-K2 and 5.4-S2 are about scratchpad files: agents record key findings in a file and consult it for later questions, which counteracts context degradation. Anthropic's own sources describe the same practice under other names: structured note-taking, where "the agent regularly writes notes persisted to memory outside of the context window" ([effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)); a Markdown checklist as a working scratchpad for large multi-step tasks (from an Anthropic engineering post on Claude Code best practices, [archived June 2025](https://web.archive.org/web/20250602202220/https://www.anthropic.com/engineering/claude-code-best-practices), whose live URL now redirects to the docs); and a `claude-progress.txt` file plus git history that lets a fresh context window recover the state of the work. In that long-running harness, the feature list whose status fields the agent updates was kept in JSON, because "the model is less likely to inappropriately change or overwrite JSON files compared to Markdown files" ([effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents)). State files for crash recovery (5.4-K4, 5.4-S4) are taught in [Exploring large codebases](context-engineering.md#exploring-large-codebases).

### Read less in the first place

| Technique | How |
|---|---|
| Keep generated and vendored code out | Permission deny rules on `Read` for build output, generated files and `vendor/` (the docs' `.claude/settings.json` example is below) |
| Navigate by symbol, not by scanning | A code intelligence plugin gives Claude go-to-definition and find-references navigation; in a large codebase, finding where a symbol is defined or used by scanning can cost many file reads and grep calls |
| Use the search you already have | Expose an existing code search or RAG index as an MCP tool so Claude queries it instead of reading files |
| Scope the session | Starting in a subdirectory loads that directory's CLAUDE.md plus every ancestor's; starting at the root loads only the root CLAUDE.md, with subdirectory files on demand |
| Pre-filter output with a hook | A hook can grep a 10,000-line log for `ERROR` and return only matching lines, cutting context from tens of thousands of tokens to hundreds |
| Give Claude a map | A skill can carry domain knowledge so Claude does not have to explore; the docs' example is a "codebase-overview" skill describing architecture, key directories and naming conventions |
| Prefer CLIs | The docs call CLI tools like `gh`, `aws` and `gcloud` the most context-efficient way to reach external services; unlike MCP servers, they add no per-tool listing to context |
| Keep CLAUDE.md lean | Its full content loads on every request; aim for under 200 lines and move specialized workflows into skills |
| Scope the question | Start broad, then narrow; the infinite-exploration failure pattern is an unscoped investigation that reads hundreds of files |

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

Claude Code's own loading already follows progressive discovery, the strategy CCAR-P 3.8 asks you to weigh against a monolithic context. CLAUDE.md loads in full on every request, but skills cost only their descriptions until used, MCP servers expose tool names until a tool is used, subdirectory CLAUDE.md files load on demand, and hooks add nothing unless they return context. The trade-off in general terms is in [Why context is a budget](context-engineering.md#why-context-is-a-budget).

**Traps**

- Letting the main conversation read the whole repository instead of delegating discovery to Explore or a scoped subagent.
- Expecting a path-scoped rule or a nested CLAUDE.md to survive compaction. They are summarized away; project-root CLAUDE.md and unscoped rules are re-injected.
- Forgetting that a delegated subagent has not seen the conversation (and that Explore and Plan also skip CLAUDE.md): rules and findings must be restated in its prompt.
- Using `/clear` when you need continuity, or `/compact` when a clean start would do.

## Headless mode and the CLI

*Tested in: CCAR-F 3.6 (3.6-K1, 3.6-K2, 3.6-S1, 3.6-S2), the appendix's Claude Code CLI item, sample question 10 · CCDV-F Claude Code Operation (headless mode, streaming mode, auto-mode), Output Handling · CCAR-P 3.7 (API/CLI as an integration mechanism), 7.1 · CCAO-F: not listed*

Headless mode means running Claude Code with `-p` (long form `--print`). Claude takes the prompt, runs its normal agent loop without the terminal interface, prints the response and exits. You can add `-p` to any `claude` command, though not every option combines with it: Claude Code rejects `--bg`, and rejects `--cloud` with a task description. The docs page for this is now titled "Run Claude Code programmatically" ([headless docs](https://code.claude.com/docs/en/headless)) and presents the CLI as the Agent SDK in command-line form, available for scripts and CI/CD alongside the Python and TypeScript packages. The exam guides use other names for the same thing: the [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) says "non-interactive mode" (a term the docs still use for `-p`) and the [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) says "headless mode" (which survives in the page's URL, `/docs/en/headless`). All three names mean `claude -p`.

### The flag behind sample question 10

CCAR-F sample question 10 describes a pipeline that runs `claude "Analyze this pull request for security issues"` and hangs because Claude Code is waiting for interactive input. The correct answer is to add `-p`. The official rationale says "The -p (or --print) flag is the documented way to run Claude Code in non-interactive mode", calls a `CLAUDE_HEADLESS` environment variable and a `--batch` flag non-existent features, and rejects redirecting stdin from `/dev/null` as a Unix workaround ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The full item is reproduced on the [CCAR-F page](../claude-certified-architect-foundations.md#official-sample-questions).

!!! warning "Exam guide vs current docs: --batch and /batch"

    The guide is right that Claude Code has no `--batch` flag. Current Claude Code does have a `/batch <instruction>` slash command: a bundled skill that splits one large change across 5 to 30 worktree-isolated subagents that each open a pull request. On the exam, an option that adds `--batch` to a CLI call is still wrong. Do not stretch that into a belief that Claude Code has no batching feature at all (as of September 2026).

### What a `-p` run does

- **Reads stdin.** You can pipe data in and redirect the answer out. Piped stdin is capped at 10MB; past the cap Claude Code exits with an error and a non-zero status, so write large input to a file and reference its path instead.
- **Signals success by exit code.** Claude Code exits 0 on success and non-zero when the run fails, so scripts can branch on it. Stopping a run with SIGTERM exits with code 143 and leaves the in-progress turn unfinished.
- **Still saves a session.** A `-p` run creates a resumable session unless you pass `--no-session-persistence`.
- **Starts in Manual permission mode on every plan** unless you pass `--permission-mode` or a settings file sets `permissions.defaultMode`. Manual's config value is `default`. As of September 2026, interactive sessions on Pro, Max and Team start in auto mode by default (Claude Code v2.1.228 or later on macOS, Linux and WSL, v2.1.233 or later on native Windows; earlier versions start in Manual), so pass the mode you want to a `-p` run.
- **Loads the same context as an interactive session unless you pass `--bare`.** Without it, `claude -p` auto-discovers hooks, skills, custom commands, subagents, plugins, MCP servers, auto memory and CLAUDE.md from the working directory and `~/.claude`.
- **Runs your skills.** User-invoked skills and custom commands work when you put `/skill-name` in the prompt. Terminal-only built-ins such as `/login` do not.
- **Waits for background work.** If Claude starts a background subagent or workflow, the process stays open until it completes, with a default ceiling of 10 minutes of continuous idle waiting (`CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS`; `0` waits without a ceiling). A background Bash task such as a dev server is terminated about five seconds after the final result instead.

```bash
# Non-interactive run with pre-approved tools
claude -p "Find and fix the bug in auth.py" --allowedTools "Read,Edit,Bash"

# Pipe data in and redirect the answer out
cat build-error.txt | claude -p 'concisely explain the root cause of this build error' > output.txt
git log --oneline -20 | claude -p "summarize these recent commits"
```

### Output formats

| `--output-format` | What you get | Typical consumer |
|---|---|---|
| `text` (default) | Plain text | People and logs |
| `json` | One JSON object: the text in `result`, the `session_id`, and metadata such as usage, `total_cost_usd` and a per-model cost breakdown (client-side estimates that can differ from the bill) | Scripts that parse with `jq` |
| `stream-json` | Newline-delimited JSON, one event per line, normally starting with the `system/init` event (startup events such as `hook_started` or `plugin_install` can come first); the last line is a `result` message with the final text, cost and session metadata | Live progress displays and CI gates |

Details worth knowing for `stream-json`:

- Add `--verbose` for streaming JSON, and `--include-partial-messages` for token-level events. `--include-partial-messages` requires `--print` and `--output-format stream-json`.
- The `system/init` event reports the model, tools, MCP servers and loaded plugins. Its `plugin_errors` key lists plugins that failed to load, and `mcp_server_errors` (v2.1.219 and later) lists `--mcp-config` entries skipped by config validation; both keys are omitted when nothing failed, so a CI gate can fail the build on a non-empty array. Without that check, a run whose MCP server never loaded still exits cleanly.
- A retryable API failure emits a `system/api_retry` event with `attempt`, `max_retries`, `retry_delay_ms`, `error_status` and `error`.
- Messages from subagents carry `parent_tool_use_id` set to the ID of the tool call that spawned them; main-conversation messages carry `null`.
- Input can stream too: `--input-format` accepts `text` or `stream-json`, and `--replay-user-messages` echoes stdin user messages back on stdout (it needs `stream-json` for both input and output).

!!! note "What the CCDV-F guide means by streaming mode"

    The [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) lists "streaming mode" under Claude Code Operation without defining it. Two documented features fit the words (our reading): `--output-format stream-json` from `claude -p`, and the Agent SDK's streaming input mode, which the SDK docs call the preferred way to use the SDK (a persistent interactive session, as opposed to single-message input for one-shot queries). Learn both.

### Structured output with `--json-schema`

CCAR-F 3.6-K2 names the pair of flags to know: `--output-format json` and `--json-schema`, "for enforcing structured output in CI contexts" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). The CLI reference describes `--json-schema` as returning validated JSON that matches a JSON Schema after the agent completes its workflow, in print mode only. The response still carries metadata (session ID, usage), and the schema-conforming data sits in the `structured_output` field.

```bash
claude -p "Extract function names from auth.py" \
  --output-format json \
  --json-schema '{"type":"object","properties":{"functions":{"type":"array","items":{"type":"string"}}},"required":["functions"]}' \
  | jq '.structured_output'
```

- An invalid schema makes `claude` exit with `Error: --json-schema is not a valid JSON Schema`. The `format` keyword is accepted but not enforced.
- `MAX_STRUCTURED_OUTPUT_RETRIES` sets how many attempts a `-p` run gets when the response fails schema validation: 5 by default, a first attempt plus four retries. After that many failures with no valid output, the run fails.
- To read the plain text answer instead, use `jq -r '.result'`.

!!! warning "Exam guide vs current docs: invalid schemas"

    The guide's advice still holds: use `--output-format json` with `--json-schema`. What changed is failure behavior. Before v2.1.205, Claude Code silently ignored an invalid schema and returned unstructured text, and treated any schema containing `format` as invalid; current versions exit with an error on an invalid schema and accept `format` as an unenforced annotation. So a pipeline run on an older version could have continued with no structured output at all (our inference; as of September 2026).

### Permissions in unattended runs

Nobody is present to answer a prompt in CI, so decide in advance what Claude may do.

| Setting | Effect in a `-p` run |
|---|---|
| `--allowedTools "Bash(npm test)" "Read"` | Listed tools run without a prompt. Permission rule syntax: the trailing space and `*` in `Bash(git diff *)` enable prefix matching; the space matters, because `Bash(git diff*)` would also match `git diff-index` |
| `--disallowedTools` | Deny rules. A bare tool name removes the tool from Claude's context; a scoped rule such as `Bash(rm *)` denies only matching calls |
| `--tools` | Restricts which built-in tools exist at all (`""` disables all, `"default"` restores the default set); it does not affect MCP tools, which `--disallowedTools "mcp__*"` removes |
| `--permission-mode dontAsk` | Auto-denies every call that would otherwise prompt; actions that need no approval in Manual mode (such as reads in the working directories) and actions your allow rules cover still run, except `AskUserQuestion`, connector tools your organization set to `ask`, and MCP tools marked `requiresUserInteraction`, which are denied even when an allow rule matches. The docs list it for locked-down CI and scripts |
| `--permission-mode acceptEdits` | Auto-approves file edits and common filesystem commands (`mkdir`, `touch`, `rm`, `rmdir`, `mv`, `cp`, `sed`) in the working directory |
| `--permission-mode auto` | A separate classifier model reviews actions before they run; it reduces prompts but does not guarantee safety |
| `--permission-prompts none` | Denies anything that would prompt (unless a `PermissionRequest` hook allows it) when nobody can answer, for example a scheduled job, and tells Claude not to retry; v2.1.259 and later |
| `--permission-prompt-tool` | Hands permission prompts to an MCP tool you supply |
| `--dangerously-skip-permissions` | Same as `--permission-mode bypassPermissions`; for isolated containers and VMs only, and refused when running as root or under sudo on Linux and macOS (the check is skipped inside a recognized sandbox) |

`--allowedTools` approves tools; it does not remove the others. To shrink what Claude can see, use `--tools` for built-in tools or a bare-name `--disallowedTools` (including `"mcp__*"` for MCP tools). The docs' locked-down CI example combines an exact allowlist with `dontAsk`:

```bash
claude -p "run the test suite" --permission-mode dontAsk --allowedTools "Bash(npm test)" "Read"
```

The "auto-mode" that the [CCDV-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf) lists is, in current docs, the `auto` permission mode (spelled "auto mode"). As of September 2026 it is the default starting mode for interactive sessions on Pro, Max and Team, but a `-p` run starts in Manual on every plan (versions in [What a `-p` run does](#what-a-p-run-does)), so pass `--permission-mode auto` when you want it. Full coverage of modes and rules is on the configuration page under [Permissions and permission modes](claude-code-configuration.md#permissions-and-permission-modes).

### Bare mode: the same result on every machine

`--bare` skips auto-discovery of hooks, skills, custom commands, subagents, plugins, MCP servers, auto memory and CLAUDE.md. The docs say it is "useful for CI and scripts where you need the same result on every machine", and that "`--bare` is the recommended mode for scripted and SDK calls, and will become the default for `-p` in a future release" ([Run Claude Code programmatically](https://code.claude.com/docs/en/headless)). It sets `CLAUDE_CODE_SIMPLE`, and `CLAUDE_CODE_SIMPLE=1` is the environment-variable equivalent. The same page advises adding `--bare` in CI and other scripted environments so the run starts without the host's hooks, plugins, auto memory or CLAUDE.md.

Three consequences follow:

- **Authentication.** Bare mode never reads OAuth credentials or the system keychain. For the Anthropic API, set `ANTHROPIC_API_KEY` (or supply an `apiKeyHelper` through `--settings`). Amazon Bedrock, Google Cloud's Agent Platform and Microsoft Foundry keep reading their own provider credentials.
- **A small tool set.** Claude has the Bash, file read and file edit tools; MCP tools from `--mcp-config` are still available.
- **Context loads only by flag.** Pass what the run needs explicitly: `--append-system-prompt` or `--append-system-prompt-file` for instructions, `--settings` for settings, `--mcp-config` for MCP servers, `--agents` for custom agents, `--plugin-dir` or `--plugin-url` for a plugin. One partial exception: skills in a directory you pass with `--add-dir` still load.

```bash
# Set ANTHROPIC_API_KEY first: bare mode does not use a subscription login
claude --bare -p "Summarize README.md" --allowedTools "Read"
```

!!! warning "Exam guide vs current docs: CLAUDE.md in CI and `--bare`"

    CCAR-F 3.6-K3 names CLAUDE.md as "the mechanism for providing project context (testing standards, fixture conventions, review criteria) to CI-invoked Claude Code" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). That is the answer to choose on the exam. The guide does not mention `--bare`, and bare mode skips CLAUDE.md, so a bare CI run only sees those standards if you pass them with `--append-system-prompt-file` or similar. The trade-off: without `--bare`, a `-p` session runs the hooks in the project's `.claude/settings.json` and connects the servers in its `.mcp.json` even in a folder you never trusted, with no trust dialog (as of September 2026).

### System prompt flags

| Flag | Effect | Use when |
|---|---|---|
| `--append-system-prompt`, `--append-system-prompt-file` | Appends text to the default system prompt | Claude should remain a coding assistant with extra rules |
| `--system-prompt`, `--system-prompt-file` | Replaces the entire system prompt (the two are mutually exclusive) | You take responsibility for everything the task needs, because replacing drops the default tool guidance and safety instructions |
| `--append-subagent-system-prompt`, `--append-subagent-system-prompt-file` | Appends text to every subagent's system prompt, nested subagents included; a forked subagent keeps the conversation's own prompt | Subagents in a `-p` run need extra rules (the flags apply only with `-p`) |

By default the system prompt is built once, on a conversation's first request, and recorded in the session. If you resume with different system-prompt flag text, the change takes effect only after compaction or in a new conversation. Passing `--system-prompt-snapshot off` (v2.1.257 and later) rebuilds the prompt on every request instead, and bare mode (outside cloud sessions) leaves recording off unless you pass `--system-prompt-snapshot on`.

### Sessions from scripts

A script can capture the session ID from JSON output and continue that exact conversation later:

```bash
session_id=$(claude -p "Start a review" --output-format json | jq -r '.session_id')
claude -p "Continue that review" --resume "$session_id"
```

- `--continue` picks up the most recent conversation in the directory; `--resume` with an ID picks a specific one.
- Which sessions the picker and `claude -p --continue` see, and what a `-p` resume restores (flags, permission mode), are in [What a resume brings back, and what it does not](#what-a-resume-brings-back-and-what-it-does-not).
- `--fork-session` with `--resume` or `--continue` gives the continuation a new session ID. `--session-id` sets a specific ID, which must be a valid UUID.
- Since v2.1.223 Claude Code finds a session ID in any project on the machine, so the resume can run from a different directory than the first call. `--resume` also accepts the absolute path to a session's `.jsonl` transcript file.

Everything else about resuming, naming and forking is in [Sessions: continue, resume, fork and rewind](#sessions-continue-resume-fork-and-rewind).

### CLI flag reference for automation

| Flag | What it does | Know this |
|---|---|---|
| `-p`, `--print` | Prints the response without interactive mode | The fix for a CI job that hangs on input |
| `--output-format` | `text`, `json` or `stream-json` | Print mode |
| `--json-schema` | Validated JSON matching a schema, in `structured_output` | Print mode only |
| `--input-format` | `text` or `stream-json` | Print mode |
| `--include-partial-messages` | Partial streaming events | Needs `--print` and `stream-json` output |
| `--verbose` | Full turn-by-turn output | Used with `stream-json` |
| `--bare` | Minimal mode, no auto-discovery | For the Anthropic API, needs `ANTHROPIC_API_KEY` or `apiKeyHelper` |
| `--allowedTools`, `--allowed-tools` | Tools that run without prompting | Does not restrict availability |
| `--disallowedTools`, `--disallowed-tools` | Deny rules | Bare name removes the tool |
| `--tools` | Restricts built-in tools | `""` disables all; MCP tools are unaffected |
| `--permission-mode` | `default`, `acceptEdits`, `plan`, `auto`, `dontAsk`, `bypassPermissions`, or `manual` (an alias for `default`, v2.1.200 and later) | Overrides `defaultMode` from settings; `-p` uses `default` when nothing is configured |
| `--max-turns` | Limits agentic turns; exits with an error at the limit | Print mode only; no limit by default |
| `--max-budget-usd` | Maximum dollar spend on API calls before stopping | Print mode only; subagent spend counts |
| `--model` | Session model by alias (`sonnet`, `opus`, `haiku`, `fable`) or full name | Overrides the `model` setting and `ANTHROPIC_MODEL` |
| `--fallback-model` | Falls back to the listed models when the primary is overloaded or unavailable | Comma-separated list, tried in order |
| `--effort` | `low`, `medium`, `high`, `xhigh`, `max`, or `ultracode` | Session effort level; available levels depend on the model |
| `--mcp-config`, `--strict-mcp-config` | Loads MCP servers from JSON; strict ignores every other MCP configuration | Pair with the `mcp_server_errors` gate |
| `--settings` | Settings file path or inline JSON for this session | `'{"disableAllHooks": true}'` turns hooks off for one run |
| `--agents`, `--agent` | Defines subagents as JSON; picks the agent for the session | `--agent` overrides the `agent` setting |
| `--no-session-persistence` | Session is not saved and cannot be resumed | Print mode only |
| `--init`, `--maintenance`, `--init-only` | Run `Setup` hooks with the `init` or `maintenance` matcher (print mode only); `--init-only` runs `Setup` and `SessionStart` hooks, then exits | One-time preparation in CI or scripts |
| `--debug`, `--debug-file` | Debug logging, optionally to a file path | Troubleshooting hooks and telemetry |
| `--exclude-dynamic-system-prompt-sections` | Moves per-machine sections into the first user message | Better prompt-cache reuse across machines in `-p` workloads; ignored with `--system-prompt` |

### The same controls in the Agent SDK

The docs present `claude -p` and the Python and TypeScript packages as the same Agent SDK (same tools, agent loop and context management), and the automation flags below have direct SDK options. The [Agent SDK page](agents-and-agent-sdk.md#the-claude-agent-sdk) teaches the SDK itself. To drive the same agent loop from a language other than Python or TypeScript, the [Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) says to run the CLI as a subprocess with `-p` and `--output-format json`; in our mapping, that is the "CLI" option when CCAR-P 3.7 asks you to choose an integration mechanism (MCP, API/CLI, agent-to-agent).

| CLI | Python (`ClaudeAgentOptions`) | TypeScript (`Options`) |
|---|---|---|
| `--allowedTools` | `allowed_tools` | `allowedTools` |
| `--disallowedTools` | `disallowed_tools` | `disallowedTools` |
| `--permission-mode` | `permission_mode` | `permissionMode` |
| `--max-turns` | `max_turns` | `maxTurns` |
| `--max-budget-usd` | `max_budget_usd` | `maxBudgetUsd` |
| `--json-schema` | `output_format={"type": "json_schema", "schema": ...}` | `outputFormat: { type: "json_schema", schema }` |
| `--resume ID --fork-session` | `resume=session_id`, `fork_session=True` | `resume: sessionId`, `forkSession: true` |

**Decide**

- If a pipeline job hangs waiting for input, add `-p`; not a `CLAUDE_HEADLESS` environment variable or a `--batch` flag (Anthropic's rationale calls both non-existent), and not a stdin redirect from `/dev/null` (a Unix workaround), because `-p` is the documented way to run non-interactively.
- If a later step must parse the result, use `--output-format json` with `--json-schema` and read `structured_output`; not free text, because text has no schema to validate.
- If you need live progress or want to fail fast on a missing MCP server, use `--output-format stream-json --verbose` and inspect `system/init`.
- If the run must behave the same on every runner, use `--bare` and pass context by flag; if the exam asks where CI project context lives, answer CLAUDE.md.
- If no person can answer prompts, set `--permission-mode dontAsk` with an explicit `--allowedTools` list; not `bypassPermissions` outside an isolated container or VM.
- If you want Claude Code's defaults plus a few rules, use `--append-system-prompt`; not `--system-prompt`, which drops the default tool guidance and safety instructions.

**Traps**

- Treating `--allowedTools` as a restriction. It pre-approves; unlisted tools remain available and fall through to the permission mode.
- Writing `Bash(git diff*)` when you meant `Bash(git diff *)`. Without the space the rule also matches `git diff-index`.
- Assuming a `-p` session shows up in `claude --continue` from an interactive terminal. It does not; resume it by ID.
- Assuming `-p` inherits the auto mode your interactive sessions use. Its built-in starting mode is Manual on every plan; pass `--permission-mode` explicitly.
- Scripting `claude -p` over a repository you did not write without checking its `.claude/` settings. Hooks committed there run with no trust dialog; use `--bare` or `--settings '{"disableAllHooks": true}'`.

## Claude Code in CI/CD

*Tested in: CCAR-F 3.6 (every knowledge and skill bullet), Scenario 5, 4.5-K2 and 4.5-S1 (blocking pre-merge checks versus overnight work), sample questions 10 and 11 · CCDV-F Software Engineering Foundations (SDLC integration, code review), Identity, Secrets, and Key Management · CCAR-P 3.2, 7.1, 7.2 · CCAO-F: not listed*

CCAR-F Scenario 5 sets the task: a pipeline that "runs automated code reviews, generates test cases, and provides feedback on pull requests", where "You need to design prompts that provide actionable feedback and minimize false positives" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Its primary domains are Claude Code Configuration & Workflows and Prompt Engineering & Structured Output. This section covers the plumbing: how Claude Code gets into a pipeline, how it authenticates, and how a job is bounded. The review prompts themselves are in [Automated code review that engineers trust](#automated-code-review-that-engineers-trust).

### Choose the integration route

| Route | What it is | Status (as of September 2026) | Choose it when |
|---|---|---|---|
| GitHub Actions: `anthropics/claude-code-action@v1` | A GitHub Action built on the Agent SDK. It runs Claude Code on your own runner, answers `@claude` mentions, or runs a `prompt` on any GitHub event | Documented; v1 replaced the beta inputs | You want to own the workflow file, prompt, model and triggers |
| GitLab CI/CD | One job in `.gitlab-ci.yml` that runs `claude -p`, built on the Claude Code CLI and Agent SDK | Beta; the integration is maintained by GitLab | Your code lives on GitLab |
| `claude -p` in any pipeline | The CLI called from a script step | Documented | Any other CI system, or you need full control of flags |
| Code Review (managed) | A fleet of specialized agents reviews PR changes in the context of the full codebase and posts inline comments | Research preview for Team and Enterprise; not available with Zero Data Retention | You want PR review without maintaining a workflow file |
| Auto-fix for pull requests (cloud sessions) | Watches a PR for CI failures and review comments; `/autofix-pr` turns it on from the terminal | Runs in cloud sessions, a research preview for Pro, Max and Team users and for Enterprise users with premium or Chat + Claude Code seats; requires the Claude GitHub App on the repository | You want Claude to respond to failing checks and review comments on a PR |

### What CCAR-F 3.6 asks you to do

| Objective | The rule | Where it is taught |
|---|---|---|
| 3.6-K1, 3.6-S1 | Run Claude Code in CI with `-p` so the job cannot hang on interactive input | [Headless mode and the CLI](#headless-mode-and-the-cli) |
| 3.6-K2, 3.6-S2 | Use `--output-format json` with `--json-schema` to produce machine-parseable findings that a later step posts as inline PR comments | Below, and in the review section |
| 3.6-K3, 3.6-S5 | Put project context (testing standards, fixture conventions, review criteria, valuable test criteria, available fixtures) in CLAUDE.md, which CI-invoked Claude Code reads (a `--bare` run is the exception; see below) | Below |
| 3.6-K4 | Review with an independent instance, not the session that generated the code | [Automated code review that engineers trust](#automated-code-review-that-engineers-trust) |
| 3.6-S3 | On re-runs after new commits, include prior findings and report only new or still-unaddressed issues | Review section |
| 3.6-S4 | Give test generation the existing test files so it does not suggest scenarios already covered | Below |

The product docs back the CLAUDE.md rule for both platforms. The GitHub Actions page says to "Create a `CLAUDE.md` file in your repository root to define code style guidelines, review criteria, project-specific rules, and preferred patterns" ([Claude Code GitHub Actions](https://code.claude.com/docs/en/github-actions)), and the GitLab page lists CLAUDE.md for coding standards, security requirements and project conventions, which Claude reads during runs. The GitHub Actions page also lists a concise CLAUDE.md among its cost controls, because Claude reads it on every run.

### GitHub Actions

**Setup.** The quick path is to run `/install-github-app` inside Claude Code. It installs the Claude GitHub App, adds the authentication secret and prepares a pull request with the workflow. It works only with github.com repositories and needs the GitHub CLI authenticated with `gh auth login`. The manual path is to install the app, add the secret yourself and copy `examples/claude.yml` into `.github/workflows/`. For either path you need admin access to the repository. The action relies on three of the app's permissions: Contents, Issues and Pull requests, each read and write.

Installing the official app grants its full permission set, which is shared with Code Review and auto-fix, and GitHub does not let you accept a subset. An organization that wants only the three permissions the action uses can create a custom GitHub App instead; that custom app covers only the action, and Code Review and web auto-fix still need the official app.

**Authentication.** Pick one credential and pass it to the matching input.

| Credential | Stored as | Workflow input | Notes |
|---|---|---|---|
| Claude Console API key | Repository secret `ANTHROPIC_API_KEY` | `anthropic_api_key` | Use this for a secret shared across repositories |
| Subscription OAuth token (Pro, Max, Team, Enterprise) | Repository secret `CLAUDE_CODE_OAUTH_TOKEN` | `claude_code_oauth_token` | Generate it locally with `claude setup-token`; it is tied to the subscription of the person who generated it, and runs use that subscription instead of API billing |
| Workload identity federation | No long-lived secret | `anthropic_federation_rule_id` and `anthropic_organization_id` (plus `anthropic_service_account_id`, optional because the federation rule already targets a service account, and `anthropic_workspace_id`, optional when the rule targets a single workspace), with `id-token: write` | GitHub's OIDC token is exchanged for Claude API access through a Console service account |
| Amazon Bedrock, Google Cloud's Agent Platform, Microsoft Foundry | Cloud trust for the workflow's OIDC token, plus repository secrets for provider identifiers (such as `AWS_ROLE_TO_ASSUME` or `GCP_WORKLOAD_IDENTITY_PROVIDER`) and, with a custom GitHub App, `APP_ID` and `APP_PRIVATE_KEY` | `use_bedrock: "true"`, `use_vertex: "true"`, `use_foundry: "true"`, with `id-token: write` | No static cloud credentials in the repository; `anthropic_api_key` is not used |

**Two modes, detected automatically.** With no `prompt` input the action runs in interactive mode: it waits for the trigger phrase (`@claude` by default) and replies in a comment on the triggering issue or PR. With a `prompt` input it runs in automation mode on any event without waiting for a mention, and results go to the workflow run log by default; Claude can post to the issue or PR when the prompt directs it to and it has a tool that can post.

**Who can trigger a run.** Two checks run in both modes, and the run fails if either rejects the actor:

- **Write access.** On issue and pull request events the triggering user must have write access. `allowed_non_write_users` admits named users without it, and works only when you pass your own `github_token`. Events that no user authors, such as a `schedule` trigger, skip this check.
- **Human actor.** On every event a bot actor is rejected unless it is listed in `allowed_bots`, whose default empty string allows no bots. Scheduled runs are attributed to a repository user, usually whoever last changed the `cron` schedule, so if that user is a bot, list it.

Respond to `@claude` in issue and PR comments:

```yaml
name: Claude Code
on:
  issue_comment:
    types: [created]
  pull_request_review_comment:
    types: [created]
jobs:
  claude:
    if: contains(github.event.comment.body, '@claude')
    runs-on: ubuntu-latest
    permissions:
      contents: write
      pull-requests: write
      issues: write
      id-token: write
      actions: read
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 1
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
```

`id-token: write` is required for the action's default GitHub App authentication, `actions: read` lets Claude read CI results on the PR, and the `if:` line keeps runners from starting on comments that do not mention `@claude`.

Structured output for a later step (the pattern behind 3.6-S2), from the action's usage docs. Pass `--json-schema` in `claude_args`; the result is validated against the schema and all fields come back in a single `structured_output` JSON string output that later steps read with `fromJSON(...)`:

```yaml
- name: Detect flaky tests
  id: analyze
  uses: anthropics/claude-code-action@v1
  with:
    anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
    prompt: |
      Check the CI logs and determine if this is a flaky test.
      Return: is_flaky (boolean), confidence (0-1), summary (string)
    claude_args: |
      --json-schema '{"type":"object","properties":{"is_flaky":{"type":"boolean"},"confidence":{"type":"number"},"summary":{"type":"string"}},"required":["is_flaky"]}'

- name: Retry if flaky
  if: fromJSON(steps.analyze.outputs.structured_output).is_flaky == true
  run: gh workflow run CI
```

A scheduled run with CLI arguments. For a plain-text prompt, Claude has no shell or GitHub API access until you grant tools, either with `--allowedTools` in `claude_args` or with a `permissions.allow` rule in the `settings` input; a skill invocation can instead use the tools its `allowed-tools` frontmatter grants. `claude_args` accepts any Claude Code CLI argument. GitHub runs scheduled workflows only from the default branch and, in public repositories, disables the schedule after 60 days without repository activity. This docs example writes its report to the workflow run log at 09:00 UTC and skips the checkout step because Claude reads commits and issues through the two GitHub MCP tools it is allowed.

```yaml
name: Daily Report
on:
  schedule:
    - cron: "0 9 * * *"
jobs:
  report:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      issues: read
      id-token: write
    steps:
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          prompt: "Generate a summary of yesterday's commits and open issues"
          claude_args: |
            --model claude-opus-5-5
            --allowedTools "mcp__github__list_commits,mcp__github__list_issues"
```

The review workflow that installs the `code-review` plugin and posts inline comments is shown in the [review section](#automated-code-review-that-engineers-trust).

**Inputs to recognize.**

| Input | Purpose |
|---|---|
| `prompt` | Instructions as plain text or a skill invocation (`/skill-name` for a skill in `.claude/skills/`, which needs a checkout step first, or `/plugin-name:skill-name` for a plugin skill). Omit it for `@claude` mode |
| `claude_args` | Any Claude Code CLI arguments, for example `--max-turns 5 --model claude-sonnet-5 --mcp-config /path/to/config.json`. Without `--model`, the action uses the Claude Code default model |
| `anthropic_api_key`, `claude_code_oauth_token` | The credential |
| `github_token` | A GitHub token for GitHub operations; when omitted, the action authenticates as the Claude GitHub App. The action's usage docs say to include it only when you connect a custom GitHub App of your own |
| `plugin_marketplaces`, `plugins` | Newline-separated marketplace Git URLs and plugin names (`plugin-name@marketplace-name`) to install before the run |
| `settings` | Claude Code settings as a JSON string or a path to a settings file, for example a `permissions.allow` rule |
| `trigger_phrase` | Default `@claude` |
| `use_bedrock`, `use_vertex`, `use_foundry` | Route inference through a cloud provider |
| `use_sticky_comment`, `track_progress` | Deliver PR comments through a single comment; force tag mode with tracking comments on pull request and issue events. Both default to `false` |
| `branch_prefix` | Prefix for Claude's branches, default `claude/` |
| `allowed_bots`, `allowed_non_write_users` | Bot usernames allowed to trigger runs; users without write access allowed to trigger runs (the usage docs mark this one as risky, and it needs your own `github_token`) |

**Operating notes from the docs.**

- GitHub does not trigger workflows on commits made with the default `GITHUB_TOKEN`. If you pass `github_token: ${{ secrets.GITHUB_TOKEN }}` and Claude's pushes do not start CI, remove it so the action authenticates as the Claude GitHub App, or pass a custom app token instead.
- The comment must contain `@claude` as a complete word, not `/claude` or `@claude-bot`.
- On public repositories GitHub withholds secrets from runs triggered by fork pull requests, so the review workflow runs only on pull requests from branches in the same repository.
- The inline-comment MCP server starts only when `--allowedTools` in `claude_args` names `mcp__github_inline_comment__create_inline_comment`, even if the skill's own `allowed-tools` frontmatter names the same tool.
- Never commit API keys or OAuth tokens; store them as GitHub Secrets. Grant the workflow only the permissions it needs, and review Claude's changes before merging.

!!! warning "Old tutorials: `claude-code-action@beta`"

    Workflows written for the beta are stale. To upgrade: change `@beta` to `@v1`, remove the `mode` input (v1 detects the mode), replace `direct_prompt` with `prompt`, and move options such as `max_turns` and `model` into `claude_args`. `custom_instructions` has no same-name flag and becomes `--append-system-prompt` (as of September 2026).

### GitLab CI/CD

The [GitLab integration](https://code.claude.com/docs/en/gitlab-ci-cd) is in beta and maintained by GitLab. Quick setup is one masked CI/CD variable (`ANTHROPIC_API_KEY`, under Settings > CI/CD > Variables) plus one job. Each interaction runs in a container on your GitLab runners, and every change flows through a merge request, so reviewers see the diff and approvals still apply. The docs' quick-setup job:

```yaml
stages:
  - ai

claude:
  stage: ai
  image: node:24-alpine3.21
  # Adjust rules to fit how you want to trigger the job:
  # - manual runs
  # - merge request events
  # - web/API triggers when a comment contains '@claude'
  rules:
    - if: '$CI_PIPELINE_SOURCE == "web"'
    - if: '$CI_PIPELINE_SOURCE == "merge_request_event"'
  variables:
    GIT_STRATEGY: fetch
  before_script:
    - apk update
    - apk add --no-cache git curl bash
    - curl -fsSL https://claude.ai/install.sh | bash
    # The installer places claude in ~/.local/bin, which isn't on PATH in this image
    - export PATH="$HOME/.local/bin:$PATH"
  script:
    # Optional: start a GitLab MCP server if your setup provides one
    - /bin/gitlab-mcp-server || true
    # Use AI_FLOW_* variables when invoking via web/API triggers with context payloads
    - echo "$AI_FLOW_INPUT for $AI_FLOW_CONTEXT on $AI_FLOW_EVENT"
    - >
      claude
      -p "${AI_FLOW_INPUT:-'Review this MR and implement the requested changes'}"
      --permission-mode acceptEdits
      --allowedTools "Bash Read Edit Write mcp__gitlab"
      --debug
```

- **Tokens for GitLab operations.** Manual setup, which the docs recommend for production, uses `CI_JOB_TOKEN` by default or a Project Access Token with `api` scope stored (masked) as `GITLAB_ACCESS_TOKEN`.
- **`@claude` mentions (optional).** Add a project webhook for "Comments (notes)" ([Claude Code GitLab CI/CD](https://code.claude.com/docs/en/gitlab-ci-cd)) that feeds your event listener; when a comment contains `@claude`, the listener calls the pipeline trigger API with variables such as `AI_FLOW_INPUT` and `AI_FLOW_CONTEXT`.
- **Providers.** Claude API, Amazon Bedrock or Google Cloud's Agent Platform. Bedrock uses OIDC role assumption (`AWS_ROLE_TO_ASSUME`, `AWS_REGION`) with no static keys; `ANTHROPIC_API_KEY` is needed only for the Claude API.
- **Per-job prompts.** Guide Claude with CLAUDE.md plus a task-specific `-p` prompt for each job, for example separate review, implement and refactor jobs.
- **Security.** Never commit API keys or cloud credentials; use masked CI/CD variables and provider OIDC where possible, limit job permissions and network egress, and review Claude's merge requests like any other contributor's.
- **Cost.** Use specific `@claude` commands, set `--max-turns` and a job-level `timeout` (for example `timeout: 30m`), and limit concurrency.

### Any other CI system: call the CLI

A script step needs the same decisions the actions make for you. Each flag below is explained in [Headless mode and the CLI](#headless-mode-and-the-cli).

1. **Run non-interactively** with `-p`.
2. **Authenticate.** Use a Console `ANTHROPIC_API_KEY`, or set `CLAUDE_CODE_OAUTH_TOKEN` to the one-year subscription token that `claude setup-token` prints. A `--bare` run never reads OAuth credentials, so it needs the API key (or an `apiKeyHelper`).
3. **Pre-decide permissions.** A `-p` run starts in Manual mode, so pass `--permission-mode dontAsk` with an exact `--allowedTools` list; keep `bypassPermissions` for isolated containers and VMs.
4. **Bound the run** with `--max-turns`, `--max-budget-usd` and a job timeout.
5. **Make the output machine-readable** with `--output-format json` and `--json-schema`, branch on the exit code, and fail the job when the `system/init` event reports MCP servers or plugins that failed to load.
6. **Decide what context loads.** The docs recommend `--bare` for scripted calls, but a bare run does not see CLAUDE.md unless you pass it (for example with `--append-system-prompt-file`). On the exam, CI project context still lives in CLAUDE.md.

The headless docs' security review script pipes a PR diff into Claude with an appended system prompt and JSON output. The same page notes, for a separate script that pipes a diff, that piping means Claude doesn't need Bash permission to read it. Piped stdin has a size cap (see [Headless mode and the CLI](#headless-mode-and-the-cli)), so write a very large diff to a file and reference its path in the prompt.

```bash
gh pr diff "$1" | claude -p \
  --append-system-prompt "You are a security engineer. Review for vulnerabilities." \
  --output-format json
```

The CCAR-P guide's own least-privilege rationale applies directly to CI allowlists: "Least privilege means removing capabilities the role does not require, eliminating the attack surface rather than monitoring or guarding it" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). Our application to CI: a review job that only reads a piped diff does not need `Edit` or unrestricted `Bash`, so restrict the built-in tools with `--tools`, or remove a tool from Claude's context with a bare-name `--disallowedTools` entry. `--allowedTools` alone does not do this, because it approves tools rather than removing the others.

### Test generation in the pipeline

- **Show Claude the tests that exist.** CCAR-F 3.6-S4: provide existing test files so generation avoids suggesting scenarios the suite already covers. The docs add that Claude examines existing test files to match their style, frameworks and assertion patterns, and that you should be specific about the behavior you want verified.
- **Say what a valuable test is.** CCAR-F 3.6-S5: document testing standards, valuable test criteria and available fixtures in CLAUDE.md to improve quality and reduce low-value output. 3.6-K3 adds fixture conventions to the context CLAUDE.md carries.
- **Treat Claude's configuration like code.** Anthropic's AI-native SDLC playbook runs an eval suite of real tasks non-interactively in CI on a schedule and on any change to CLAUDE.md, skills or hooks, and gates configuration changes on the results.
- **Keep changes behind review.** In the same playbook, anything the agent writes arrives as a PR through branch protection, and the agent has no route to push to main. In production the agent only prepares the release: the release manager authorizes it, and a hook enforces the gate.

### Blocking or not: where each job belongs

| Job | Someone waiting? | Run it as | Why |
|---|---|---|---|
| Pre-merge check | Yes, developers cannot merge until it finishes | A synchronous call in the pipeline (`claude -p` or the Messages API) | The Message Batches API has up to a 24-hour processing window and no guaranteed latency SLA |
| Overnight technical debt report, weekly audit, nightly test generation | No | A scheduled job; if it calls the API directly, the Message Batches API at 50% of the cost | 4.5-K2 names these non-blocking, latency-tolerant workloads as the fit for batches |

That table is CCAR-F sample question 11 in one line: batch the overnight report, keep the pre-merge check real-time. The official rationale rejects the alternatives: relying on batches finishing faster is not acceptable for a blocking workflow, batch result ordering is not a reason to avoid batches because `custom_id` correlates requests and responses, and a timeout fallback to real-time adds complexity when the simpler answer is to match each API to its use case. Batch mechanics are on the API page under [Message Batches](claude-api.md#message-batches).

For scheduling Claude Code itself, the docs list four options: routines (cloud, Anthropic-managed by default, and able to run while your computer is off), Desktop scheduled tasks (your machine), GitHub Actions (your CI pipeline, for repository events or cron schedules that live with your workflow config) and `/loop` (the current CLI session). Scheduled prompts run autonomously and cannot ask clarifying questions, so state what success looks like and what to do with the results.

### Security and cost

- **Trust.** A `-p` or SDK session never shows the workspace trust dialog, so a repository's committed hooks run without it; what to check and how to switch them off are in [Security, trust and organizational control](#security-trust-and-organizational-control). More in [Claude Code security controls](security-and-governance.md#claude-code-security-controls) and [Secrets and API keys](security-and-governance.md#secrets-and-api-keys).
- **Untrusted PRs.** Anthropic's `claude-code-security-review` action warns that it is not hardened against prompt injection and should review only trusted PRs. Its README recommends the repository setting that requires approval for all external contributors, so workflows run only after a maintainer has reviewed the PR.
- **Cost.** Each GitHub Actions run consumes Actions minutes and API tokens (or, with an OAuth token, your Claude subscription instead of API billing); a GitLab job consumes runner minutes and tokens. The documented controls are specific `@claude` requests, issue templates, a concise CLAUDE.md, `--max-turns` in `claude_args`, workflow timeouts and concurrency limits. Tracking spend is covered in [Monitoring, usage and cost](#monitoring-usage-and-cost).

**Decide**

- If a CI job must give Claude project standards, put them in the root CLAUDE.md; not in `~/.claude/CLAUDE.md`, which applies only to that user and is not shared through version control (our inference: a CI runner does not have a developer's user-level file).
- If a later step must post findings as inline comments, emit JSON with `--output-format json` and `--json-schema`; not free text that a script has to parse.
- If developers are waiting on the result, run it synchronously; if nobody is waiting, schedule it and consider batches.
- If a shared secret serves many repositories, use a Console API key; not a personal subscription token.
- If Claude's commits must trigger CI, let the action authenticate as the Claude GitHub App (or a custom app token); not the default `GITHUB_TOKEN`.
- If a CI job only reads and reports, remove the tools it does not need; do not rely on logging or confirmations to guard capabilities it never required.

**Traps**

- Setting `CLAUDE_HEADLESS=true`, adding a `--batch` flag, or redirecting stdin from `/dev/null`: the Q10 rationale calls the first two non-existent features and the third a Unix workaround; the fix for a hanging job is `-p` (see [The flag behind sample question 10](#the-flag-behind-sample-question-10), which also separates `--batch` from the real `/batch` command).
- Moving the pre-merge check to batches to save 50%: wrong for a blocking workflow, because batches have no guaranteed latency SLA.
- Adding a timeout fallback from batches to real-time calls: the Q11 rationale calls it unnecessary complexity.
- Reviewing in the same session that wrote the code: CCAR-F 3.6-K4 says that session is less effective at reviewing its own changes than an independent review instance.
- Granting a scheduled prompt no tools and expecting it to read GitHub: a plain-text prompt has no shell or GitHub API access until you grant it.

## Automated code review that engineers trust

*Tested in: CCAR-F 4.1 (4.1-K1 to 4.1-S3), 4.6 (4.6-K1 to 4.6-S3), 3.6-K4, 3.6-S2, 3.6-S3, 1.6-K2 and 1.6-S2, 4.2-S2 and 4.2-S3, 4.4-K3 and 4.4-S3, 5.5-S3, Scenario 5, sample question 12, How to Prepare item 5 · CCDV-F Software Engineering Foundations (code review), Output Handling (skepticism toward confident output) · CCAR-P 5.3, 7.2 · CCAO-F: not listed*

An automated reviewer is only useful if engineers act on its comments. CCAR-F 4.1-K3 names the failure mode: "high false positive categories undermine confidence in accurate categories" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Anthropic's open-source code-review command puts it the same way to its own review agents: "If you are not certain an issue is real, do not flag it. False positives erode trust and waste reviewer time." ([code-review command](https://raw.githubusercontent.com/anthropics/claude-code/main/plugins/code-review/commands/code-review.md)). Four design levers get you there: explicit criteria, false-positive controls, an independent reviewer, and multiple focused passes. The general prompting techniques are taught in [Explicit criteria and precision](prompt-engineering.md#explicit-criteria-and-precision) and [Multi-instance and multi-pass review](prompt-engineering.md#multi-instance-and-multi-pass-review); this section applies them to a review pipeline.

### Lever 1: criteria the model can apply

The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) contrasts the two kinds of instruction:

| Vague instruction | Explicit criterion (what to write instead) |
|---|---|
| "check that comments are accurate" | "flag comments only when claimed behavior contradicts actual code behavior" |
| "be conservative", "only report high-confidence findings" | Specific categorical criteria: report bugs and security issues; skip minor style and local patterns |
| Severity left to judgment (our contrast for 4.1-S3) | Explicit severity criteria with a concrete code example for each level |

The Opus 4.8 and Sonnet 5 prompting guides give a concrete bar for a reviewer that must self-filter in a single pass, instead of qualitative terms like "important": "report any bugs that could cause incorrect behavior, a test failure, or a misleading result; only omit nits like pure style or naming preferences." ([Prompting Claude Opus 4.8](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-4-8)). The Opus 4.8 guide adds: iterate on review prompts against a subset of your evals or test cases to validate recall or F1 gains.

!!! warning "Exam guide vs current docs: be conservative"

    The [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) (4.1-K2) says general instructions like "be conservative" or "only report high-confidence findings" fail to improve precision compared with specific categorical criteria. The current [Opus 4.8](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-4-8) and [Sonnet 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5) prompting guides describe a different effect of the same kind of instruction: these models may follow "only report high-severity issues" or "be conservative" more faithfully than earlier models, so precision typically rises but measured recall can fall. They recommend that the finding stage report everything, with a confidence level and an estimated severity for each finding, and leave filtering to a separate verification step. The [Opus 5](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5) guide says the model "may follow that instruction literally and report less" and advises: "ask it to report everything and filter in a separate pass instead." Both sources steer away from a vague qualitative filter in the finding prompt. On the exam, choose specific categorical criteria (as of September 2026).

In Claude Code's managed Code Review, review-only rules live in `REVIEW.md` at the repository root. The agents that find and verify findings receive it as the repository's review instructions, and the agents that rank and report findings consult it. Anthropic's example for a backend service recalibrates severity, caps nits, skips generated files and adds repo-specific checks:

```markdown
# Review instructions

## What Important means here

Reserve Important for findings that would break behavior, leak data,
or block a rollback: incorrect logic, unscoped database queries, PII
in logs or error messages, and migrations that aren't backward
compatible. Style, naming, and refactoring suggestions are Nit at
most.

## Cap the nits

Report at most five Nits per review. If you found more, say "plus N
similar items" in the summary instead of posting them inline. If
everything you found is a Nit, lead the summary with "No blocking
issues."

## Do not report

- Anything CI already enforces: lint, formatting, type errors
- Generated files under `src/gen/` and any `*.lock` file
- Test-only code that intentionally violates production rules

## Always check

- New API routes have an integration test
- Log lines don't include email addresses, user IDs, or request bodies
- Database queries are scoped to the caller's tenant
```

The other patterns the docs list for `REVIEW.md` are a verification bar (evidence required before a class of finding is posted), a re-review convergence rule, and a summary shape, such as a one-line tally at the top of the review body. For paths that deserve some review but not full scrutiny, set a higher bar instead of skipping them: "in `scripts/`, only report if near-certain and severe." The docs warn that "a long `REVIEW.md` dilutes the rules that matter most" ([Code Review](https://code.claude.com/docs/en/code-review)), so keep general project context in CLAUDE.md.

### Lever 2: false-positive controls

| Control | How it works | Source of the rule |
|---|---|---|
| Report and skip lists | Name the categories to report (bugs, security) and the ones to skip (minor style, local patterns) | CCAR-F 4.1-S1 |
| Switch off a noisy category | Temporarily disable a high false-positive category to restore trust while you improve its prompt; `REVIEW.md` skip rules can suppress whole finding categories, paths or branch patterns | CCAR-F 4.1-S2; Code Review docs |
| Severity with examples | Define each level with a concrete code example; Code Review uses Important, Nit and Pre-existing | CCAR-F 4.1-S3; Code Review docs |
| Verification bar | Require evidence before a class of finding is posted, for example "behavior claims need a `file:line` citation in the source, not an inference from naming" | [Code Review docs](https://code.claude.com/docs/en/code-review) |
| Verification pass | Check each candidate against the actual code before posting; drop what does not validate | Code Review docs; code-review command |
| Scope to the change | Report issues the PR introduces. Managed Code Review still posts bugs the PR did not introduce, tagged Pre-existing; the security-review action tells Claude not to comment on existing security concerns | Code Review docs; security-review action |
| Skip what CI already enforces | Make lint, formatting, type errors and other CI-enforced checks a skip rule; the code-review command also treats issues a linter will catch as false positives | Code Review docs; code-review command |
| Few-shot contrast | Examples that separate acceptable code patterns from genuine issues, which reduces false positives while still letting the model generalize | CCAR-F 4.2-S3 |
| Learn from dismissals | Add a `detected_pattern` field to each finding so you can analyze which code constructs trigger the findings developers dismiss | CCAR-F 4.4-K3, 4.4-S3 |

Anthropic's code-review command lists what its agents must treat as false positives: pre-existing issues, something that appears to be a bug but is actually correct, pedantic nitpicks a senior engineer would not flag, issues a linter will catch, general code quality concerns (such as lack of test coverage) unless CLAUDE.md explicitly requires them, and issues mentioned in CLAUDE.md but explicitly silenced in the code, for example with a lint ignore comment. It flags only high-signal issues: code that will fail to compile or parse, code that will definitely produce wrong results, and clear CLAUDE.md violations where the exact rule can be quoted. Managed Code Review collects feedback from people instead: each finding arrives with thumbs-up and thumbs-down reactions, Anthropic collects the reaction counts after the PR merges and uses them to tune the reviewer, and you dismiss a finding by resolving its thread (a reply does not dismiss it).

### Lever 3: an independent reviewer

CCAR-F 4.6-K1 states the reason: a model "retains reasoning context from generation, making it less likely to question its own decisions in the same session" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). 3.6-K4 applies the same idea to CI: the session that generated code is less effective at reviewing its own changes than an independent review instance. 4.6-K2 adds that independent review instances without prior reasoning context are more effective at catching subtle issues than self-review instructions or extended thinking. The skill (4.6-S1) is to use a second, independent Claude instance to review generated code without the generator's reasoning context.

The Claude Code docs apply this in several places:

- **Writer and Reviewer.** "A fresh context improves code review since Claude won't be biased toward code it just wrote." ([Best practices](https://code.claude.com/docs/en/best-practices)) The same pattern works for tests: one Claude writes tests, another writes code to pass them.
- **Adversarial review.** Before treating a task as done, have a subagent review the diff in a fresh context and report gaps. The bundled `/code-review` skill reviews the current diff for bugs in a fresh subagent and returns findings to the session. Tell the reviewer to flag only gaps that affect correctness or the stated requirements, and treat the rest as optional, since chasing every finding leads to over-engineering.
- **Security guidance plugin.** It "does not ask the same Claude instance that wrote the code to grade itself" ([security guidance](https://code.claude.com/docs/en/security-guidance)): its end-of-turn and commit reviews run as a separate Claude call with fresh context and a security-focused prompt, and that reviewer starts from the diff, has no investment in the original approach, and is told only to find problems.

One current doc adds a caveat for a single model: the [Opus 5 prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5) says Claude Opus 5 verifies its own work unprompted and that explicit instructions such as "use a subagent to verify" cause over-verification and should be removed, while the same guide says the model coordinates subagents "with effective writer-verifier patterns" (details in [Multi-instance and multi-pass review](prompt-engineering.md#multi-instance-and-multi-pass-review)). On the exam, answer with the guide's independent reviewer (as of September 2026).

Our application to a pipeline: independence means a separate `claude -p` invocation (each new session starts with a fresh context window, without earlier sessions' history) that receives the diff and the review criteria, not a `--continue` or `--resume` of the session that wrote the change.

!!! warning "Exam guide vs current docs: extended thinking"

    4.6-K2 in the [CCAR-F guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf) contrasts independent review with "extended thinking". In current API docs, extended thinking is the manual mode configured as `thinking.type: "enabled"` with `budget_tokens`: it is deprecated on the Claude 4.6 models, and Claude 4.7 and later models reject it with a 400 error; the docs point to adaptive thinking instead. Either way, more thinking happens inside the same context as the generation, so our reading is that the guide's point stands: extra reasoning in the generating session is not a substitute for an independent reviewer. On the exam, answer in the guide's terms (as of September 2026).

### Lever 4: several focused passes

CCAR-F sample question 12 describes a single-pass review of 14 files that gives detailed feedback on some files and superficial comments on others, misses obvious bugs, and contradicts itself across files (flagging a pattern in one file while approving identical code in another). The correct answer is to split the review: analyze each file individually for local issues, then run a separate integration-focused pass for cross-file data flow. The rationale names the root cause as attention dilution when many files are processed at once. It rejects three tempting alternatives: making developers split their PRs shifts the burden without improving the system; a bigger model or context window fails because "larger context windows don't solve attention quality issues"; and flagging only issues found by at least two of three runs "would actually suppress detection of real bugs by requiring consensus on issues that may only be caught intermittently" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).

Our assembly of those pieces into one pipeline (illustrative), using only the stages the guide and the docs describe:

```text
PR diff + review criteria (CLAUDE.md, REVIEW.md)
        |
        v
Finding passes: one per changed file, local issues only
        |
        v
Integration pass: cross-file data flow across the changed files
        |
        v
Verification pass: check each candidate against the actual code,
keep validated findings, record a confidence for each
        |
        v
Deduplicate, rank by severity, post inline comments
```

Anthropic's own review tools follow a similar shape. In managed Code Review, multiple agents analyze the diff and surrounding code in parallel, each looking for a different class of issue; a verification step checks candidates against actual code behavior; and the results are deduplicated, ranked by severity and posted as inline comments. The open-source code-review command first checks whether to stop (closed or draft PR, no review needed, or Claude has already commented), gathers the relevant CLAUDE.md paths and a summary of the change, then launches four review agents in parallel (two for CLAUDE.md compliance, two for bugs), launches parallel validation subagents for the flagged issues, and filters out anything not validated.

If you split finding from filtering, the finding stage should favor coverage. The [Sonnet 5 prompting guide](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5) recommends this language for that stage (the Opus 4.8 guide gives the same text). The guide adds that the prompt can be used without an actual second step, but moving confidence filtering out of the finding step often helps, and that a harness with a separate verification, deduplication or ranking stage should tell the model its job at the finding stage is coverage rather than filtering:

```text
Report every issue you find, including ones you are uncertain about or consider low-severity. Do not filter for importance or confidence at this stage - a separate verification step will do that. Your goal here is coverage: it is better to surface a finding that later gets filtered out than to silently drop a real bug. For each finding, include your confidence level and an estimated severity so a downstream filter can rank them.
```

**Confidence routing.** CCAR-F 4.6-S3 has the verification pass self-report confidence alongside each finding "to enable calibrated review routing" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)), and 5.5-S3 calibrates review thresholds using labeled validation sets. Anthropic's examples route on fixed thresholds: the security-review action's prompt says not to report findings below 0.7, and a claude-code-action example workflow retries a flaky test automatically only at confidence 0.7 or higher and recommends manual review below it. Neither example documents a calibration step, and [Anthropic research](https://www.anthropic.com/research/language-models-mostly-know-what-they-know) reports that models predict well whether they know an answer but struggle with calibrating that prediction on new tasks. Our reading, consistent with 5.5-S3: calibrate against labeled outcomes for your own task before trusting a threshold. Calibration and sampling are taught in [Human review and confidence calibration](evaluation-and-reliability.md#human-review-and-confidence-calibration).

The guide also warns in the other direction: 5.2-K3 calls self-reported confidence scores unreliable proxies for actual case complexity when deciding whether to escalate a customer case. The two positions fit together once calibration is counted: raw self-confidence is a poor escalation trigger, while calibrated confidence can route review work.

!!! warning "Two descriptions of the same code-review plugin"

    The code-review plugin's README describes scoring each issue from 0 to 100 and filtering out issues below 80. The plugin's command file in the same repository describes validation subagents that filter out unvalidated issues and never mentions a confidence score. Treat the README as possibly stale and the command file as the current behavior (as of September 2026).

### Structured findings for inline comments

CCAR-F 3.6-S2 pairs `--output-format json` with `--json-schema` so a later step can post each finding as an inline PR comment. 4.2-S2 uses few-shot examples to make the output format consistent (location, issue, severity, suggested fix), and 4.4-S3 adds a `detected_pattern` field. An illustrative schema built from those names (our example, not from the docs), with the diff piped in; the schema-conforming result arrives in the `structured_output` field:

```bash
gh pr diff "$1" | claude -p "Review this diff against the review criteria in CLAUDE.md. Report only issues this change introduces." \
  --output-format json \
  --json-schema '{"type":"object","properties":{"findings":{"type":"array","items":{"type":"object","properties":{"file":{"type":"string"},"line":{"type":"integer"},"severity":{"type":"string","enum":["Important","Nit","Pre-existing"]},"issue":{"type":"string"},"suggested_fix":{"type":"string"},"detected_pattern":{"type":"string"},"confidence":{"type":"number"}},"required":["file","line","severity","issue"]}}},"required":["findings"]}' \
  | jq '.structured_output.findings'
```

For comparison, Anthropic's `claude-code-security-review` action asks for each finding in this shape (the finding object from its prompt, with the Python format string's doubled braces shown as single braces):

```json
{
  "file": "path/to/file.py",
  "line": 42,
  "severity": "HIGH",
  "category": "sql_injection",
  "description": "User input passed to SQL query without parameterization",
  "exploit_scenario": "Attacker could extract database contents by manipulating the 'search' parameter with SQL injection payloads like '1; DROP TABLE users--'",
  "recommendation": "Replace string formatting with parameterized queries using SQLAlchemy or equivalent",
  "confidence": 0.95
}
```

Inside GitHub Actions you can skip the parsing step: the `code-review` plugin posts inline comments itself when the prompt passes `--comment` and `claude_args` names its inline-comment MCP tool. Without `--comment`, Claude posts nothing and the findings appear only in the workflow run log. The docs' review workflow:

```yaml
name: Code Review
on:
  pull_request:
    types: [opened, synchronize, ready_for_review, reopened]
jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: read
      issues: read
      id-token: write
    steps:
      - uses: actions/checkout@v6
        with:
          fetch-depth: 1
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          plugin_marketplaces: "https://github.com/anthropics/claude-code.git"
          plugins: "code-review@claude-code-plugins"
          prompt: "/code-review:code-review --comment ${{ github.repository }}/pull/${{ github.event.pull_request.number }}"
          claude_args: '--allowedTools "mcp__github_inline_comment__create_inline_comment"'
```

Claude posts an inline comment on each issue it finds, or one summary comment when it finds none. It skips draft and closed pull requests, pull requests it judges not to need a review (such as automated or trivial ones), and pull requests that already have a comment from Claude. Fork pull requests are covered in [Claude Code in CI/CD](#claude-code-in-cicd).

### Re-running a review after new commits

CCAR-F 3.6-S3 is precise: include the prior review findings in context when re-running after new commits, and instruct Claude to report only new or still-unaddressed issues, so the PR does not collect duplicate comments. An illustrative prompt in that spirit (our wording, not from the docs):

```text
<prior_findings>
...JSON findings from the previous review of this PR...
</prior_findings>
For each prior finding, state whether the new commits fixed it or it is still present.
Report a new finding only if it is not already in prior_findings.
Do not post a still-present prior finding again as a new comment.
```

Anthropic's tools solve the same problem in different ways, and none of their docs describes passing prior findings into the prompt as such:

| Tool | Re-review behavior |
|---|---|
| Managed Code Review, "After every push" | Reviews every push, catching new issues as the PR evolves and auto-resolving threads when you fix flagged issues |
| `REVIEW.md` convergence rule | For example "after the first review, suppress new nits and post Important findings only" ([Code Review](https://code.claude.com/docs/en/code-review)) |
| Open-source code-review command | Stops if Claude has already commented on the PR (the GitHub Actions review workflow documents the same skip), so by our reading a re-run after new commits posts nothing; posts only one comment per unique issue |
| `claude-code-action` | `use_sticky_comment` delivers PR comments through a single comment |
| `claude-code-security-review` | Its `run-every-commit` input (default `false`) skips the cache check, with a warning that it may increase false positives on PRs with many commits |

On the exam, answer with the guide's own technique: prior findings in context, and only new or still-unaddressed issues reported.

### Managed Code Review at a glance (as of September 2026)

| Item | Detail |
|---|---|
| Availability | Research preview for Team and Enterprise subscriptions; not available with Zero Data Retention. An Owner enables it once for the organization and selects repositories |
| Triggers per repository | Once after PR creation, After every push, or Manual. A pull request from a fork is reviewed only when someone comments `@claude review` |
| Comment commands | `@claude review` starts a single review; `@claude review always` also subscribes the PR to push-triggered reviews. Before a July 2026 update, `@claude review` also subscribed the PR |
| Default focus | Correctness: bugs that would break production, not formatting preferences or missing test coverage |
| Severity tags | Important (a bug to fix before merging), Nit (minor, not blocking), Pre-existing (a bug not introduced by this PR) |
| Merge gating | Findings do not approve or block; the check run always completes neutral, so parse the severity counts in its output in your own CI if you want a gate |
| Tuning | CLAUDE.md (newly introduced violations are flagged as nits), plus review-only rules in `REVIEW.md` |
| Cost and time | Anthropic states an average of &#36;15 to &#36;25 per review, billed on token usage through usage credits separately from plan usage, and 20 minutes on average; After every push multiplies cost by the number of pushes |
| Failures | A failed run never blocks the PR and does not retry on its own; comment `@claude review`, or (except on a fork PR) re-run the check |
| Local equivalent | `/code-review` reviews a diff in your terminal without the GitHub App; it follows CLAUDE.md but not `REVIEW.md`, and in a `-p` run it waits and includes the findings in the response. At `low` and `medium` effort it reports only its most confident findings; `high` through `max` broaden coverage. `/review` is now an alias (it was a separate command before v2.1.223) |
| Deeper option | `/code-review ultra` (ultrareview, research preview) runs a fleet of reviewer agents in a cloud sandbox and independently reproduces and verifies each reported finding; it runs only when invoked. From CI or a script, the `claude ultrareview` subcommand launches it and blocks until the findings arrive; `claude -p '/code-review ultra'` only launches it and prints a tracking link, and stops before launching when the review would bill usage credits (Team and Enterprise plans include no free runs) |

### People stay accountable

Anthropic's AI-native SDLC playbook gives every PR the same set of review passes with findings ranked by severity, so human attention moves up a level: whether the change does what the plan intended, and whether the risk is acceptable. Findings do not approve or block a PR on their own, and branch protection still requires a code owner's approval; a platform engineer who wants to gate merges can read the severity counts the check run publishes. Findings also feed back into CLAUDE.md: when a review flags the same mistake a second time, the correction goes into CLAUDE.md, and because review reads CLAUDE.md, later PRs are checked against it. Once a month the tech lead rates findings and caps nit volume in `REVIEW.md`.

**Decide**

- If developers stop trusting the bot because one category is noisy, disable that category temporarily and improve its prompt; not an instruction to be conservative, which the guide says fails to improve precision compared with specific categorical criteria.
- If severity labels are inconsistent, define each level explicitly with a concrete code example.
- If a review of many files is shallow or contradictory, split it into per-file passes plus an integration pass; not a bigger model or context window, not majority voting across runs, not forcing smaller PRs.
- If Claude wrote the code, review it with a second, independent instance that lacks the generator's reasoning context; not a self-review instruction in the same session.
- If findings must become inline comments, emit schema-validated JSON with a consistent format (location, issue, severity, suggested fix).
- If a PR is re-reviewed after new commits, pass the prior findings and ask for new or still-unaddressed issues only.

**Traps**

- **Confidence instead of categories.** A confidence-based filter such as "only report high-confidence findings" in place of categorical criteria is the anti-pattern 4.1-K2 and 4.1-S1 name ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)).
- **Consensus voting.** Requiring a finding to appear in two of three runs suppresses real bugs that may only be caught intermittently (Q12 rationale).
- **A bigger context window.** The Q12 rationale says larger context windows do not solve attention quality issues.
- **A self-review instruction or more thinking in the same session.** 4.6-K2 ranks independent review instances above both self-review instructions and extended thinking.
- **Trusting raw confidence scores.** 4.6-S3 uses self-reported confidence for calibrated review routing, 5.5-S3 calibrates review thresholds using labeled validation sets, and 5.2-K3 calls self-reported confidence an unreliable proxy for case complexity.

## Practices from the Claude Code documentation

*Tested in: CCAR-F 3.4, 3.5, 5.4 (and How to Prepare item 2, configuring Claude Code for a real project) · CCDV-F Claude Code Operation, Claude Application Design (session hygiene), Configuration Management (CLAUDE.md files) · CCAR-P 7.1, 7.2 · CCAO-F: not listed*

As of September 2026, Anthropic's Claude Code best-practices guidance lives in the docs: the old engineering blog URL redirects to [Best practices for Claude Code](https://code.claude.com/docs/en/best-practices). The page explains its own organizing idea: "Most best practices are based on one constraint: Claude's context window fills up fast, and performance degrades as it fills." It calls the context window "the most important resource to manage". In our reading, almost every practice below is either a way to keep context clean or a way to give Claude a reliable signal of success. Several practices have their own sections on this page, linked where they come up.

### Give Claude a way to verify its work

The [docs'](https://code.claude.com/docs/en/best-practices) first practice: "Give Claude a check it can run: tests, a build, a screenshot to compare." Without one, "looks done" is the only signal and you become the verification loop. With a check that produces a pass or fail, Claude does the work, runs the check, reads the result and iterates until the check passes. The docs' example prompt, a `validateEmail` request that lists test cases and asks Claude to run the tests, is shown in [Show the transformation with examples](#show-the-transformation-with-examples).

Once the check exists, the docs list four ways to decide how hard it gates the stop. Each step trades setup for attention:

| How hard it gates the stop | Mechanism | What to know |
|---|---|---|
| In one prompt | Ask Claude to run the check and iterate in the same message | Works on any task today |
| Across a session | Set the check as a `/goal` condition | A separate evaluator re-checks it after every turn; `/goal` is a built-in shortcut for a session-scoped prompt-based Stop hook |
| As a deterministic gate | A Stop hook runs your check as a script | Blocks the turn from ending until the check passes; Claude Code overrides the hook and ends the turn after 8 consecutive blocks |
| By a second opinion | A verification subagent, or a dynamic workflow that checks its own findings | A fresh model tries to refute the result, so the agent doing the work is not the one grading it |

The docs add that the `/goal` and Stop hook versions are what let an unattended run finish correctly without you. Ask for evidence, not assertions: test output, the commands that ran and what they returned, screenshots. The [failure-patterns list](https://code.claude.com/docs/en/best-practices) sums it up as "If you can't verify it, don't ship it." Hooks are taught in [Hooks](#hooks), and test-first work in [Iterative refinement](#iterative-refinement).

### Separate exploring from building

The docs' four-phase workflow (explore, plan, implement, commit), their one-sentence test for skipping the plan, and the exam's decision rule are taught together in [Plan mode or direct execution](#plan-mode-or-direct-execution).

### Be specific and give rich context

"The more precise your instructions, the fewer corrections you'll need." ([Best practices](https://code.claude.com/docs/en/best-practices)) The docs name four strategies: scope the task, point to sources, reference existing patterns, and describe the symptom (the symptom, the likely location, and what "fixed" looks like). Their bug-fix example asks Claude to write a failing test that reproduces the issue, then fix it.

Ways to hand Claude context: reference files with `@`, paste images, give URLs, pipe data in (`cat error.log | claude`), or let Claude fetch what it needs. An `@` file reference includes the file's full content and also adds any CLAUDE.md files from that file's directory and its parents.

### Configure the environment once

| Practice | What to do |
|---|---|
| Start CLAUDE.md with `/init` | Generate a starter file from the project structure, then refine it over time; run `/context` to confirm Claude loaded it |
| Keep CLAUDE.md lean | For each line ask "Would removing this cause Claude to make mistakes?" ([Best practices](https://code.claude.com/docs/en/best-practices)); bloated files cause Claude to ignore your actual instructions. If Claude already behaves correctly without a line, delete it or convert it to a hook |
| Use hooks for zero-exception actions | CLAUDE.md is advisory; hooks are deterministic. Claude can write the hook for you, for example "Write a hook that runs eslint after every file edit" ([Best practices](https://code.claude.com/docs/en/best-practices)) |
| Prefer CLI tools for external services | The docs call CLI tools such as `gh`, `aws`, `gcloud` and `sentry-cli` the most context-efficient way to interact with external services |
| Guard skills with side effects | Set `disable-model-invocation: true` so a workflow like `/fix-issue 1234` runs only when you trigger it |

The [features overview](https://code.claude.com/docs/en/features-overview) adds two matching rules. Output styles, like CLAUDE.md, are followed as instructions rather than enforced, so the hook rule in [Hook or instruction: the decision the exams test](#hook-or-instruction-the-decision-the-exams-test) covers them too. A side task whose output would flood the conversation belongs in a subagent. The configuration layers themselves are covered on the [configuration page](claude-code-configuration.md#the-configuration-layers-at-a-glance).

The CCAR-F guide turns this into hands-on preparation. Its How to Prepare item 2 reads: "Configure Claude Code for a real project: set up CLAUDE.md with a configuration hierarchy, create path-specific rules in .claude/rules/, build custom skills with frontmatter options (context: fork, allowed-tools), and integrate at least one MCP server." ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)). Those mechanics are taught in [CLAUDE.md and the memory hierarchy](claude-code-configuration.md#claudemd-and-the-memory-hierarchy), [Path-scoped rules](claude-code-configuration.md#path-scoped-rules), [Agent Skills](claude-code-configuration.md#agent-skills) and [MCP servers in Claude Code](claude-code-configuration.md#mcp-servers-in-claude-code).

### Communicate and manage the session

- **Interview first for larger features.** Have Claude interview you with the `AskUserQuestion` tool, write the result to `SPEC.md`, then start a fresh session to execute it. See [Iterative refinement](#iterative-refinement).
- **Course-correct early.** `Esc` stops Claude mid-action with context preserved; `Esc` twice or `/rewind` opens the rewind menu; `/clear` resets context between unrelated tasks.
- **Restart instead of repeating yourself.** After correcting Claude more than twice on the same issue in one session, the context is cluttered with failed approaches. The [docs'](https://code.claude.com/docs/en/best-practices) conclusion: "A clean session with a better prompt almost always outperforms a long session with accumulated corrections."
- **Keep side questions out of context.** `/btw` answers a question without adding it to the conversation history; a prompt like "use subagents to investigate X" ([Best practices](https://code.claude.com/docs/en/best-practices)) keeps research out of the main conversation.
- **Treat sessions like branches.** Name them with `/rename`; `claude --continue` picks up where you left off and `claude --resume` chooses from a list.

Sessions, checkpoints and compaction are taught in [Sessions: continue, resume, fork and rewind](#sessions-continue-resume-fork-and-rewind) and [Managing context in large codebases](#managing-context-in-large-codebases).

### Automate and scale

- **Non-interactive runs.** Use `claude -p "prompt"` in CI, pre-commit hooks or scripts; the run still creates a resumable session unless you pass `--no-session-persistence`. Covered in [Headless mode and the CLI](#headless-mode-and-the-cli).
- **Writer and reviewer.** A fresh context improves code review because Claude is not biased toward code it just wrote; see [Automated code review that engineers trust](#automated-code-review-that-engineers-trust).
- **Fan out across files.** In a git repository, `/batch <instruction>` splits a change across 5 to 30 subagents, each in its own worktree and opening its own pull request. Or loop over `claude -p` yourself, scoping permissions with `--allowedTools`, which matters when the run is unattended. The [docs'](https://code.claude.com/docs/en/best-practices) advice: "Refine your prompt based on what goes wrong with the first 2-3 files, then run on the full set."
- **Parallel sessions.** Worktrees give each session a separate git checkout so parallel sessions do not edit the same files (`claude --worktree feature-auth`). Agent teams are experimental and disabled by default; agent view (`claude agents`) is a research preview. Running several sessions or subagents at once multiplies token usage.

The docs' fan-out loop, run after Claude has written the list of files to migrate to `files.txt`:

```bash
for file in $(cat files.txt); do
  claude -p "Migrate $file from Python 2 to Python 3. Return OK or FAIL." \
    --allowedTools "Edit,Bash(git commit *)"
done
```

### Failure patterns the docs name

| Pattern | What goes wrong | Fix |
|---|---|---|
| The kitchen sink session | Unrelated tasks pile up in one context | `/clear` between unrelated tasks |
| Correcting over and over | Repeated corrections leave the context polluted with failed approaches | After two failed corrections, `/clear` and write a better initial prompt that uses what you learned |
| The over-specified CLAUDE.md | A file that is too long gets half ignored because important rules get lost in the noise | Ruthlessly prune; if Claude already does something correctly without the instruction, delete it or convert it to a hook |
| The trust-then-verify gap | Claude produces a plausible-looking implementation that does not handle edge cases | Always provide verification (tests, scripts, screenshots): if you cannot verify it, do not ship it |
| The infinite exploration | An unscoped "investigate" request makes Claude read hundreds of files, filling the context | Scope investigations narrowly or use subagents |

### Recipes from the Common workflows page

| Task | Recipe |
|---|---|
| Understand an unfamiliar codebase | Start with broad questions, then narrow down to specific areas |
| Fix a bug | Tell Claude the command that reproduces the issue and get a stack trace |
| Refactor | Work in small, testable increments |
| Write tests | Claude examines existing test files to match their style, frameworks and assertion patterns; ask it to identify edge cases you might have missed |
| Plan before editing | `claude --permission-mode plan`: Claude reads files and proposes a plan but makes no edits until you approve |
| Delegate research | Ask in plain words, for example "use a subagent to investigate how our auth system handles token refresh" ([Common workflows](https://code.claude.com/docs/en/common-workflows)) |
| Find the session behind a PR | `claude --from-pr 1234` (sessions link to a PR when Claude creates it with `gh pr create` or `glab mr create`, and when Claude works on an existing PR) |
| Feed a script's output to Claude | Pipe it in: `git log --oneline -20` piped into `claude -p "summarize these recent commits"` |

### How the exam frames these practices

The guides test these habits through their own objective wording, not through the docs' headings. Our mapping of each practice to the objective it serves:

| Docs practice | Guide objective it matches | Taught in depth |
|---|---|---|
| Give Claude a check it can run | CCAR-F 3.5-K2, test-driven iteration: write the tests first, then share test failures | [Iterative refinement](#iterative-refinement) |
| Explore in plan mode, then implement | CCAR-F 3.4-K3 (plan mode "enables safe codebase exploration and design before committing to changes, preventing costly rework") and 3.4-S4 (plan mode for investigation, direct execution for implementation) | [Plan mode or direct execution](#plan-mode-or-direct-execution) |
| Let Claude interview you | CCAR-F 3.5-K3, the interview pattern | [Iterative refinement](#iterative-refinement) |
| Use subagents for investigation | CCAR-F 5.4-K3, "Subagent delegation for isolating verbose exploration output while the main agent coordinates high-level understanding" | [Managing context in large codebases](#managing-context-in-large-codebases) |
| Hooks for zero-exception actions, CLAUDE.md for guidance | CCAR-F 1.5-K3, "The distinction between using hooks for deterministic guarantees versus relying on prompt instructions for probabilistic compliance" | [Hooks](#hooks) |
| Writer and reviewer in separate contexts | CCAR-F 3.6-K4, session context isolation: the session that generated code is less effective at reviewing it than an independent review instance | [Automated code review that engineers trust](#automated-code-review-that-engineers-trust) |
| `/init`, CLAUDE.md, sessions and slash commands | CCDV-F Claude Code Operation, whose description lists session management, built-in and custom slash commands, the CLAUDE.md hierarchy and repository initialization | [Configuration page](claude-code-configuration.md#claudemd-and-the-memory-hierarchy) |

**Decide**

- If a task has an objective pass or fail check, give Claude that check and ask for evidence; if the check must pass before every turn ends, make it a Stop hook rather than a prompt line.
- If you have corrected the same mistake twice, `/clear` and restart with a better prompt; not a third correction.
- If an instruction must hold with zero exceptions, it belongs in a hook; if it is a convention, CLAUDE.md is proportionate.
- If a change touches many files the same way, fan out: `/batch` (review and approve its plan before it spawns subagents) or a `claude -p` loop that you prove on the first 2 to 3 files before running the full set.

**Traps**

- Relying on a CLAUDE.md line or a prompt instruction for something that must happen every time. The guide pairs prompts with "probabilistic compliance" and hooks with deterministic guarantees (1.5-K3).
- Letting open-ended exploration run in the main conversation. The guide's answer is subagent delegation for verbose exploration output (5.4-K3); the docs name the same failure "the infinite exploration".
- Asking the session that wrote the code to review it. A fresh context reviews better (3.6-K4).
- Using plan mode for a one-sentence diff, or skipping it for a multi-file architectural change. The guide says plan mode is designed for large-scale, multi-file and architectural work, and direct execution is appropriate for simple, well-scoped changes (3.4-K1, 3.4-K2).

## Monitoring, usage and cost

*Tested in: CCDV-F Cost and Token Management · CCAR-P 3.4, 4.5, 4.6, 7.1 · CCAR-F: not an objective (the guide lists rate limiting, quotas and API pricing calculations as out of scope) · CCAO-F: not listed*

Claude Code charges by API token consumption, so monitoring it means watching tokens: per session while you work, per run in automation, and per user across an organization. This section covers the Claude Code tools for each level. API-level usage reporting is on the API page under [Cost and usage tracking](claude-api.md#cost-and-usage-tracking), and program budgeting under [Cost modeling](solution-architecture.md#cost-modeling).

### Where to look

For one developer or one run:

| Scope | Tool | What it shows | Know this |
|---|---|---|---|
| Your session | `/usage` (aliases `/cost`, `/stats`) | Session cost, plan usage limits and activity stats | The dollar figure is an estimate computed locally from token counts at list price, unless an administrator has set a `modelPricing` table in managed settings; `modelPricing` changes what Claude Code reports, not what Anthropic charges. The Session block is intended for API users: for Pro and Max subscribers the session cost figure is not relevant for billing |
| Your context | `/context` | A live breakdown of context usage by category, including which CLAUDE.md and auto memory files loaded | Token costs scale with context size |
| One scripted run | `claude -p --output-format json` | `total_cost_usd` and a per-model cost breakdown | Client-side estimates that can differ from your actual bill |
| A cap on one run | `--max-budget-usd`, `--max-turns` | Stops the run at a dollar amount, or exits with an error at a turn count | Print mode only; subagent spend counts toward the budget cap; `--max-turns` has no limit by default |
| Every user, near real time | OpenTelemetry export | Per-user token and cost metrics in your own observability stack | Works on every setup, whatever the provider |

For an organization, where you see spend, cap it and pull per-user numbers depends on how developers access Claude Code (the docs' mapping, as of September 2026):

| Your setup | See spend | Cap spend | Per-user reporting |
|---|---|---|---|
| Claude for Teams or Enterprise | Spend report in org analytics | Spend limits in admin settings | Spend report CSV; the Enterprise Analytics API on Enterprise |
| Claude Console (API) | Console usage page | Workspace spend limits | Console dashboard; the Claude Code Analytics API |
| Amazon Bedrock, Google Cloud's Agent Platform or Microsoft Foundry | Your cloud billing console | Your cloud's budget controls | OpenTelemetry or an LLM gateway (the docs' Cloud providers section lists a self-hosted Claude apps gateway as a third option) |

- **Team and seat-based Enterprise.** Each member's Claude Code usage draws from a per-seat allowance that resets on a rolling five-hour window and a weekly window, shared with Claude chat and Cowork. The seat allowance is the default ceiling; to let members continue past it, an admin turns on usage credits and sets spend limits at the organization, group or member level. Usage inside the seat allowance is not metered in dollars.
- **Usage-based Enterprise.** On a usage-based Enterprise plan (including self-serve Enterprise) there are no per-seat usage limits; usage is based on consumption and billed at API rates ([Use Claude Code with your Team or Enterprise plan](https://support.claude.com/en/articles/11845131-use-claude-code-with-your-team-or-enterprise-plan)). On either plan type, the controls live in the claude.ai admin console, not the Claude Console.
- **Console organizations.** The first time Claude Code authenticates with a Console account, a workspace called "Claude Code" is created automatically. You cannot create API keys for it, and it is the only workspace that supports per-user monthly spend limits. The Claude Code Analytics Admin API returns daily aggregated per-user metrics (sessions, lines of code, commits, pull requests, tool usage, and estimated cost by model) and needs an Admin API key; the Claude Enterprise Analytics API needs a different key type, and neither key can call the other API.
- **Cloud providers.** Claude Code does not send metrics from your cloud back to Anthropic, so the analytics dashboards and the Claude Code Analytics API do not cover this usage. Per-user attribution comes from OpenTelemetry or a gateway.

One more gap: the Claude Code contribution metrics (public beta, shown on the Teams and Enterprise analytics dashboard) need the GitHub integration, are deliberately conservative (an underestimate), and are not available to organizations with Zero Data Retention enabled.

### OpenTelemetry setup

Claude Code exports metrics as time series, events through the logs protocol, and optionally distributed traces. Telemetry is off until you set `CLAUDE_CODE_ENABLE_TELEMETRY=1` and choose at least one exporter. The quick start from the [Monitoring docs](https://code.claude.com/docs/en/monitoring-usage):

```bash
# 1. Enable telemetry
export CLAUDE_CODE_ENABLE_TELEMETRY=1

# 2. Choose exporters (both are optional - configure only what you need)
export OTEL_METRICS_EXPORTER=otlp       # Options: otlp, prometheus, console, none
export OTEL_LOGS_EXPORTER=otlp          # Options: otlp, console, none

# 3. Configure OTLP endpoint (for OTLP exporter)
export OTEL_EXPORTER_OTLP_PROTOCOL=grpc
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317

# 4. Set authentication (if required)
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Bearer your-token"

# 5. For debugging: reduce export intervals, and reset them for production use
export OTEL_METRIC_EXPORT_INTERVAL=10000  # 10 seconds (default: 60000ms)
export OTEL_LOGS_EXPORT_INTERVAL=5000     # 5 seconds (default: 5000ms)

# 6. Run Claude Code
claude
```

- **Defaults.** `OTEL_METRIC_EXPORT_INTERVAL` defaults to 60000 ms and `OTEL_LOGS_EXPORT_INTERVAL` to 5000 ms. Claude Code has no default OTLP protocol, so set `OTEL_EXPORTER_OTLP_PROTOCOL` (or the per-signal variable) for each `otlp` exporter you enable.
- **Check it works.** Look for the `claude_code.session.count` metric, emitted when a session starts, or the `claude_code.user_prompt` event for logs. If nothing arrives, run `claude --debug` and check the debug log.
- **Roll it out centrally.** Administrators can set these variables for every user through the managed settings file; a managed `OTEL_EXPORTER_OTLP_*` value removes conflicting developer-set variables at startup, which locks the destination.
- **Label for chargeback.** `OTEL_RESOURCE_ATTRIBUTES="department=engineering,team.id=platform,cost_center=eng-123"` adds team or cost-center labels. Each custom key becomes a label on every metric series, so high-cardinality values increase storage cost in your metrics backend; set `OTEL_METRICS_INCLUDE_RESOURCE_ATTRIBUTES=false` to send custom attributes in the resource block only.
- **Privacy defaults.** Prompt content is redacted unless `OTEL_LOG_USER_PROMPTS=1`, and tool parameters such as Bash commands and MCP server and tool names are logged with `OTEL_LOG_TOOL_DETAILS=1`. Both default to disabled. `OTEL_LOG_RAW_API_BODIES` (also off by default) exports full API request and response bodies, conversation history included, so treat it as turning on both.
- **Scope.** Claude Code does not pass `OTEL_*` variables to the subprocesses it spawns, including the Bash tool, hooks, MCP servers and language servers.
- **Traces (beta).** Off by default. Set `CLAUDE_CODE_ENABLE_TELEMETRY=1` and `CLAUDE_CODE_ENHANCED_TELEMETRY_BETA=1`, then choose `OTEL_TRACES_EXPORTER`. Bash and PowerShell subprocesses then inherit a `TRACEPARENT` variable for end-to-end tracing (by default only when Claude Code talks to the Anthropic API directly; behind a custom `ANTHROPIC_BASE_URL` proxy, set `CLAUDE_CODE_PROPAGATE_TRACEPARENT=1`).

**Metrics to know.**

| Metric | What it counts | Unit (as the docs list it) |
|---|---|---|
| `claude_code.session.count` | CLI sessions started | none |
| `claude_code.lines_of_code.count` | Lines of code modified | none |
| `claude_code.pull_request.count` | Pull requests created | none |
| `claude_code.commit.count` | Git commits created | none |
| `claude_code.cost.usage` | Cost of the Claude Code session | USD |
| `claude_code.token.usage` | Tokens used, with a `type` attribute of `input`, `output`, `cacheRead` or `cacheCreation` | tokens |
| `claude_code.code_edit_tool.decision` | Code editing tool permission decisions | none |
| `claude_code.active_time.total` | Total active time | s |

The cost and token counters are incremented after each API request and carry `query_source` (`main`, `subagent` or `auxiliary`), so you can see how much spend comes from delegation. Every event produced while handling one user prompt shares a `prompt.id`.

**Events for reliability.** Claude Code retries failed API requests internally and emits one `claude_code.api_error` event only after it gives up, so that event is the terminal signal; when retries on a transient error are exhausted, its `attempt` is 11 by default. Refusals arrive on a successful stream and have their own `claude_code.api_refusal` event. To tell a session that recovered from one that stalled, group events by `session.id` and look for a later `api_request` after the error.

**Traces and the Agent SDK.** Trace spans include `claude_code.interaction` (one turn of the agent loop), `claude_code.llm_request` (each API call, with model, latency and token counts) and `claude_code.tool`; a `claude_code.hook` span is added only with detailed beta tracing. Subagent spans nest under the parent's tool span, so a delegation chain appears as one trace. The Agent SDK produces no telemetry of its own; the Claude Code CLI child process it runs does the exporting, configured through the same environment variables.

Telemetry tells you how Claude is running. For a record of content, the Claude Enterprise course points to the Compliance API instead: "the Compliance API, not telemetry, is the record to rely on for content" ([Deploying Claude Enterprise with confidence](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/visibility-what-you-can-measure)). That advice is scoped to Claude Enterprise. The Compliance API is available to Enterprise plan organizations (excluding Public Sector organizations), and its coverage does not include Claude Code cloud sessions, Claude Code accessed through the Claude Platform, or sessions run on Amazon Bedrock or Google Vertex AI ([Access the Compliance API](https://support.claude.com/en/articles/13015708-access-the-compliance-api)). Governance controls are covered in [Admin and governance controls](security-and-governance.md#admin-and-governance-controls).

### What it costs (Anthropic-stated figures, as of September 2026)

| Figure | Value |
|---|---|
| Average across enterprise deployments | About &#36;13 per developer per active day and &#36;150 to &#36;250 per developer per month; below &#36;30 per active day for 90% of users |
| Background usage (summarization for `claude --resume`, status checks such as `/usage`) | Typically under &#36;0.04 per session |
| Agent teams | About 7x the tokens of a standard session when teammates run in plan mode |
| Code Review (research preview, Team and Enterprise) | Averages &#36;15 to &#36;25 per review, billed separately through usage credits and not counted against the plan's included usage |

These are averages the docs state, not guarantees; per-developer costs vary widely with model selection, codebase size and usage patterns such as running multiple instances or automation. To estimate your own spend, the docs recommend a small pilot group to establish a baseline before a wider rollout. On an API or cloud-provider plan, the docs say unexpectedly high spend usually traces back to long sessions that were never cleared or to Opus left as the default model.

Rate-limit planning for Console organizations uses per-user tokens per minute (TPM) and requests per minute (RPM) recommendations that fall as teams grow, because fewer people use Claude Code at the same moment in larger organizations. The limits apply at the organization level, not per individual user, so one user can temporarily use more than their share when others are idle:

| Team size | TPM per user | RPM per user |
|---|---|---|
| 1 to 5 users | 200k to 300k | 5 to 7 |
| 5 to 20 users | 100k to 150k | 2.5 to 3.5 |
| 20 to 50 users | 50k to 75k | 1.25 to 1.75 |
| 50 to 100 users | 25k to 35k | 0.62 to 0.87 |
| 100 to 500 users | 15k to 20k | 0.37 to 0.47 |
| 500+ users | 10k to 15k | 0.25 to 0.35 |

The docs' worked example: with 200 users, you might request 20k TPM for each user, or 4 million TPM in total. They also note that unusually high concurrent usage, such as a live training session with a large group, may need higher per-user allocations.

### Reduce token usage

| Lever | Why it works |
|---|---|
| `/clear` between tasks | Stale context wastes tokens on every later message, and `/clear` itself costs nothing; `/rename` first so you can `/resume` the session later |
| `/compact` with a focus, or a "Compact Instructions" section in CLAUDE.md | Keeps what matters when history is summarized; when to prefer `/clear` is in [The context commands](#the-context-commands) |
| Pick the model for the job | Sonnet handles most coding tasks and costs less than Opus; reserve Opus for complex architectural decisions or multi-step reasoning, and set `model: haiku` for simple subagent tasks |
| Lower effort for simple work | Thinking tokens are billed as output tokens; lower the effort level with `/effort` or in `/model`, or turn thinking off in `/config` (as of September 2026, thinking cannot be turned off on Opus 5.5 or the Fable models) |
| Keep MCP overhead down | MCP tool definitions are deferred by default, so only tool names and server instructions enter context until Claude uses a specific tool; disable unused servers with `/mcp` |
| Delegate verbose work | A subagent keeps test runs, log reading and doc fetching out of the main context and returns only a summary |
| Write specific prompts | Vague requests such as "improve this codebase" trigger broad scanning; a specific request lets Claude work with minimal file reads |
| Plan complex work first | Plan mode prevents expensive rework when the initial direction is wrong |
| Mind the cache lifetime | The first message after a break longer than the cache lifetime misses the cache and reprocesses the full context. The lifetime is an hour on a subscription (five minutes once drawing on usage credits) and five minutes by default on an API key or cloud provider |

Three context levers from the same costs page also cut spend: pre-filtering output with a hook, keeping CLAUDE.md under 200 lines, and preferring CLI tools such as `gh` over MCP servers for external services. They are taught in [Read less in the first place](#read-less-in-the-first-place).

For CI, the GitHub Actions and GitLab docs' cost advice is to set `--max-turns` (in `claude_args` for the GitHub action), a workflow or job timeout, and concurrency limits, as described in [Claude Code in CI/CD](#claude-code-in-cicd). A `claude -p` step can also take `--max-budget-usd`. Prompt caching itself is taught on the API page under [Prompt caching](claude-api.md#prompt-caching).

### How the exam frames it

The objectives name outcomes, not Claude Code commands. CCDV-F's Cost and Token Management skill covers "token usage tracking, cost modeling, and caching techniques (prompt caching, cache check-pointing)"; CCAR-P asks candidates to "Analyze observability challenges and select monitoring strategies at scale" (3.4), "Optimize token usage, latency, and cost-performance trade-offs" (4.5) and "Monitor system performance using logging and observability tools" (4.6). CCAR-P's How to Prepare list includes "Build and operate at least one end-to-end Claude solution, including RAG, evaluation, and observability".

Of the published sample items, CCAR-P Sample 2 is the closest to this section: an application resends the same 8,000-token system prompt and policy document on every request, and latency and cost are both concerns. The correct answer places the static content before the dynamic content and enables prompt caching; Anthropic's rationale: "Ordering stable content first and enabling prompt caching lets repeated prefixes be reused, reducing both time-to-first-token and per-request cost without discarding required context." The rejected options truncate needed context, downsize the model blindly, or move the policy into a few-shot block, which "does not create a cacheable, reusable prefix" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)). Claude Code applies prompt caching automatically, so inside Claude Code the day-to-day version of this lesson is the cache-lifetime lever above. Two other sample items are about cost through the Message Batches API rather than monitoring: CCAR-F sample question 11 and CCDV-F Sample 1 (see [Claude Code in CI/CD](#claude-code-in-cicd) and [Message Batches](claude-api.md#message-batches)). CCAR-F, by contrast, lists "Rate limiting, quotas, or API pricing calculations" among its out-of-scope topics.

**Decide**

- If you need per-user token and cost data in your own dashboards as it happens, export OpenTelemetry; if you need daily per-user adoption and cost reports for a Console organization, use the Claude Code Analytics Admin API.
- If developers run Claude Code through Bedrock or another cloud provider, rely on OpenTelemetry or a gateway, because Anthropic's analytics dashboards and the Claude Code Analytics API do not see that usage.
- If spend on an API or cloud-provider plan is unexpectedly high, look first at long sessions that were never cleared and at Opus left as the default model: the docs say that is what it usually traces back to.
- If a job runs unattended, give it a hard ceiling (our rule, built from the flags above): `--max-turns`, `--max-budget-usd` and a timeout.

**Traps**

- Treating the `/usage` or `total_cost_usd` figure as the bill. Both are estimates computed on the client, and for Pro and Max subscribers the session cost figure is not relevant for billing at all.
- Expecting hooks or MCP servers to inherit your `OTEL_*` settings. Claude Code does not pass them to subprocesses.
- Putting high-cardinality values in `OTEL_RESOURCE_ATTRIBUTES`. Every custom key becomes a label on every metric series, and high-cardinality values increase storage cost in your metrics backend.
- Forgetting that prompt logging is opt-in. Prompt content is redacted by default; `OTEL_LOG_USER_PROMPTS=1` includes it (and `OTEL_LOG_RAW_API_BODIES`, also off by default, exports full API request bodies with the conversation history). For a record of content on Claude Enterprise, the Compliance API is the one to rely on; its coverage does not include Claude Code accessed through the Claude Platform, cloud sessions, or sessions run on Amazon Bedrock or Google Vertex AI.

## Exam map

Which objective of which exam each section of this page serves, taken from the four exam guides (Version 1.0, July 2026). "none" means that guide has no objective for the section.

| Section | CCAO-F | CCDV-F | CCAR-F | CCAR-P |
|---|---|---|---|---|
| [Hooks](#hooks) | none | Claude Hooks (1.0%); Agent Construction with Claude (5.3%, hooks for deterministic actions); Sample 2 (its correct option names guardrails or hooks) | 1.4-K1, 1.4-K2, 1.4-S1; 1.5 (1.5-K1 to 1.5-S3); Scenario 1; sample question 1; Exercise 1 step 4; appendix Claude Agent SDK item (hooks) | 5.1, 7.1 |
| [Plan mode or direct execution](#plan-mode-or-direct-execution) | none | Claude Code Operation (3.1%) | 3.4 (3.4-K1 to 3.4-S4); Scenario 2; sample question 5; Exercise 2 step 5; appendix in-scope plan mode item | 7.1, 7.2 |
| [Iterative refinement](#iterative-refinement) | D1.3, D7.2 (concept only) | Prompt Engineering (4.6%, iterative refinement) | 3.5 (3.5-K1 to 3.5-S5); appendix in-scope iterative refinement item | 7.2 |
| [Sessions: continue, resume, fork and rewind](#sessions-continue-resume-fork-and-rewind) | D3.4 (concept only: when to restart, summarize, or persist) | Claude Code Operation (3.1%, session management); Claude Application Design (8.6%, session hygiene) | 1.7 (1.7-K1 to 1.7-S4); appendix session management item | none |
| [Managing context in large codebases](#managing-context-in-large-codebases) | none | Context Engineering (3.8%, compaction, context isolation through subagents) | 5.4 (5.4-K1 to 5.4-S5); 3.4-K4, 3.4-S3; Scenario 2; appendix context window management item | 2.4, 3.8 |
| [Headless mode and the CLI](#headless-mode-and-the-cli) | none | Claude Code Operation (3.1%, headless mode, streaming mode, auto-mode); Output Handling (2.6%) | 3.6-K1, 3.6-K2, 3.6-S1, 3.6-S2; appendix Claude Code CLI item; sample question 10 | 3.7, 7.1 |
| [Claude Code in CI/CD](#claude-code-in-cicd) | none | Software Engineering Foundations (7.4%, SDLC integration, code review); Identity, Secrets, and Key Management (1.6%) | 3.6 (3.6-K1 to 3.6-S5); Scenario 5; 4.5-K2, 4.5-S1; sample questions 10 and 11 | 3.2, 7.1, 7.2 |
| [Automated code review that engineers trust](#automated-code-review-that-engineers-trust) | none | Software Engineering Foundations (7.4%, code review); Output Handling (2.6%, skepticism toward confident output) | 4.1 (4.1-K1 to 4.1-S3); 4.6 (4.6-K1 to 4.6-S3); 3.6-K4, 3.6-S2, 3.6-S3; 1.6-K2, 1.6-S2; 4.2-S2, 4.2-S3; 4.4-K3, 4.4-S3; 5.5-S3; Scenario 5; sample question 12; How to Prepare item 5 | 5.3, 7.2 |
| [Practices from the Claude Code documentation](#practices-from-the-claude-code-documentation) | none | Claude Code Operation (3.1%); Claude Application Design (8.6%); Configuration Management (4.1%, CLAUDE.md files) | 3.4, 3.5, 5.4; How to Prepare item 2 | 7.1, 7.2 |
| [Monitoring, usage and cost](#monitoring-usage-and-cost) | none | Cost and Token Management (2.8%); Sample 1 (Message Batches cost) is related | none (rate limiting, quotas and API pricing calculations are listed as out of scope); sample question 11 (Message Batches cost, 4.5-K1) is related | 3.4, 4.5, 4.6, 7.1; Sample 2 (prompt caching) is related |

How to read the labels:

- **CCAR-F.** Task statement numbers (3.6) are the guide's own. Bullet labels such as 3.6-K1 (knowledge bullet 1) and 3.6-S2 (skill bullet 2) count the guide's unnumbered bullets in order. Scenario 5 is "Claude Code for Continuous Integration" ([CCAR-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf)); its primary domains are Domain 3, Claude Code Configuration & Workflows (20%), and Domain 4, Prompt Engineering & Structured Output (20%).
- **CCDV-F.** Skill names and percentages are the guide's own; each percentage is that skill's share of the whole exam. The whole Claude Code domain is 3.1%, but in our mapping Claude Code topics also appear under other skills: Claude Application Design names Claude Code among the interfaces and lists session hygiene, Configuration Management lists CLAUDE.md files, and Claude Hooks and Context Engineering cover hooks, compaction and subagent isolation. "Sample 1" and "Sample 2" mean the guide's Domain 2 and Domain 7 sample items.
- **CCAR-P.** Objective numbers such as 7.1 count the guide's bullets in order within each domain; "Sample 2" is the guide's second sample item, tagged to Domain 2. Domain 7, Developer Productivity & Operational Enablement, is 7% of the exam and includes "Configure Claude tools and environments for teams (e.g., Claude Code)" ([CCAR-P exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf)).
- **CCAO-F.** Labels such as D3.4 count the guide's bullets in order within each domain. The guide says the certification "is not intended for software developers who build against APIs or design agentic systems" ([CCAO-F exam guide](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf)), and the guide does not mention Claude Code. The cells marked *concept only* share a judgment with this page (iterating on output, and deciding when to restart, summarize or persist a conversation), taught for the Claude apps on the [CCAO-F page](../claude-certified-associate.md).

Exam pages for the heaviest overlaps: [CCAR-F Domain 3](../claude-certified-architect-foundations.md#domain-3-claude-code-configuration-workflows), [CCAR-F Domain 4](../claude-certified-architect-foundations.md#domain-4-prompt-engineering-structured-output), [CCDV-F Domain 3](../claude-certified-developer.md#domain-3-claude-code) and [CCAR-P Domain 7](../claude-certified-architect-professional.md#domain-7-developer-productivity-operational-enablement).

??? info "Sources"

    - [Claude Certified Architect, Foundations: Exam Guide (PDF, Version 1.0, July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542750%2FClaude+Certified+Architect+%E2%80%93+Foundations+Exam+Guide.pdf): Task Statements 1.4, 1.5, 1.6, 1.7, 3.4, 3.5, 3.6, 4.1, 4.2, 4.4, 4.5, 4.6, 5.2, 5.4 and 5.5; Scenarios 1, 2 and 5; sample questions 1, 5, 10, 11 and 12 with their rationales; Exercise 1 step 4 and Exercise 2 step 5; How to Prepare items 2 and 5; the appendix technology, in-scope and out-of-scope lists
    - [Claude Certified Developer, Foundations: Exam Guide (PDF, Version 1.0, July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542875%2FClaude+Certified+Developer+%E2%80%93+Foundations+Exam+Guide.pdf): skill names, weights and descriptions for Claude Hooks, Agent Construction with Claude, Claude Code Operation, Software Engineering Foundations, Claude Application Design, Configuration Management, Cost and Token Management, Context Engineering, Prompt Engineering, Output Handling, and Identity, Secrets, and Key Management; Sample 1 (Message Batches) and Sample 2 (injected instructions and hooks)
    - [Claude Certified Architect, Professional: Exam Guide (PDF, Version 1.0, July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542810%2FClaude+Certified+Architect+%E2%80%93+Professional+Exam+Guide.pdf): objectives 2.4, 3.2, 3.4, 3.7, 3.8, 4.5, 4.6, 5.1, 5.3, 7.1 and 7.2, the Domain 7 weight, the How to Prepare list, Sample 1's least-privilege rationale and Sample 2 (prompt caching)
    - [Claude Certified Associate, Foundations: Exam Guide (PDF, Version 1.0, July 2026)](https://everpath-course-content.s3-accelerate.amazonaws.com/instructor%2F6nizmqk8tpzpfjvt6qmmav7rh%2Fpublic%2F1783542847%2FClaude+Certified+Associate+%E2%80%93+Foundations+Exam+Guide.pdf): the audience exclusion for software developers; objectives D1.3 (iterating prompts), D3.4 (when to restart, summarize or persist) and D7.2 (adjusting to feedback)
    - [Hooks reference (Claude Code docs)](https://code.claude.com/docs/en/hooks): hook events, configuration locations, matchers and the `if` field, handler types and timeouts, input fields, exit codes, JSON output, decision control, `defer`, prompt, agent and async hooks, `/goal` as a prompt-based Stop hook, the Stop hook block cap, Setup hooks with `--init`, workspace trust and `-p` or SDK sessions, `disableAllHooks`, debugging
    - [Automate actions with hooks (Claude Code docs)](https://code.claude.com/docs/en/hooks-guide): deterministic control, worked examples (protect files, auto-format, re-inject context after compaction, auto-approve ExitPlanMode), hook-deny precedence over permission modes, Stop hook loop guard, troubleshooting
    - [Choose a permission mode (Claude Code docs)](https://code.claude.com/docs/en/permission-modes): plan mode definition, entering and leaving it, plan approval options, `defaultMode`, auto mode during planning, where plan mode's blocks apply; Manual as `default`, auto mode as the interactive starting mode on Pro, Max and Team, `dontAsk` for locked-down CI, `acceptEdits` scope, the auto mode classifier, `bypassPermissions` limits
    - [Best practices for Claude Code (Claude Code docs)](https://code.claude.com/docs/en/best-practices): context as the key constraint, verification checks and the ways to gate on them, explore, plan, implement, commit, when to skip planning, specificity, the interview pattern, course correction, environment configuration, compaction instructions, session naming, delegating research, automation and fan-out, Writer and Reviewer, adversarial review, failure patterns
    - [Manage sessions (Claude Code docs)](https://code.claude.com/docs/en/sessions): continue, resume, naming, branching, what a resume restores, `-p` sessions and the picker, permission mode on a `-p` resume, the resume-from-summary dialog, storage and retention
    - [Checkpointing (Claude Code docs)](https://code.claude.com/docs/en/checkpointing): automatic checkpoints, the 100-checkpoint limit, rewind actions, what checkpoints do not track
    - [Commands (Claude Code docs)](https://code.claude.com/docs/en/commands): `/compact`, `/clear`, `/context`, `/btw`, `/autocompact`, `/plan`, `/branch`, `/fork` and `/subtask`, `/resume`, `/rewind` aliases, `/usage` and its aliases, `/batch`
    - [Explore the context window (Claude Code docs)](https://code.claude.com/docs/en/context-window): `/context` breakdown by category, what survives compaction, file and skill re-injection limits, compacting with a focus, clearing between tasks
    - [CLI reference (Claude Code docs)](https://code.claude.com/docs/en/cli-reference): `--continue`, `--resume`, `--fork-session`, `--name`, `--no-session-persistence`, every flag in the automation flag table, append versus replace system prompt flags, system prompt recording on resume
    - [Create custom subagents (Claude Code docs)](https://code.claude.com/docs/en/sub-agents): Explore and Plan subagents, isolating high-volume operations, restating rules in delegation prompts, subagent forks and `/subtask`, Task renamed to Agent, Explore model change
    - [How Claude Code works (Claude Code docs)](https://code.claude.com/docs/en/how-claude-code-works): session JSONL files, a fresh context window for each new session, resume versus fork, how auto-compaction clears and summarizes, a "Compact Instructions" section in CLAUDE.md, branch switching, remote actions and checkpoints
    - [Manage costs effectively (Claude Code docs)](https://code.claude.com/docs/en/costs): token-based charging, `/usage` estimates and `modelPricing`, plan mode preventing re-work, `/compact` versus `/clear` cost, compact instructions in CLAUDE.md, hook pre-filtering, CLAUDE.md size, enterprise cost averages, pilot baselines, workspaces, rate-limit recommendations, token reduction levers, background usage, cache lifetime, agent team cost, seat allowances, cloud providers not reporting metrics
    - [How Claude remembers your project (Claude Code docs)](https://code.claude.com/docs/en/memory): CLAUDE.md as context rather than enforced configuration, size guidance, persistence across compaction
    - [Environment variables (Claude Code docs)](https://code.claude.com/docs/en/env-vars): `CLAUDE_AUTOCOMPACT_PCT_OVERRIDE`, `DISABLE_AUTO_COMPACT`, `DISABLE_COMPACT`, `CLAUDE_CODE_STOP_HOOK_BLOCK_CAP`, `CLAUDE_CODE_SIMPLE`, `MAX_STRUCTURED_OUTPUT_RETRIES`
    - [Extend Claude Code (Claude Code docs)](https://code.claude.com/docs/en/features-overview): hook versus CLAUDE.md versus skill, CLAUDE.md and output styles as instructions rather than enforcement, subagents for side tasks, context cost by feature
    - [Common workflows (Claude Code docs)](https://code.claude.com/docs/en/common-workflows): recipes for exploring, bug fixing, refactoring, tests, planning, delegation and piping; `--from-pr`; worktrees for parallel sessions; code intelligence plugins; scheduling options
    - [Set up Claude Code in a monorepo or large codebase (Claude Code docs)](https://code.claude.com/docs/en/large-codebases): deny rules for generated and vendored code, code search as an MCP tool, symbol lookup cost, launch directory and CLAUDE.md loading, plan file re-injection
    - [Configure permissions (Claude Code docs)](https://code.claude.com/docs/en/permissions): Bash deny rules are not a security boundary around a program, hooks load from the working directory's `.claude/`
    - [All settings (Claude Code docs)](https://code.claude.com/docs/en/settings-reference): `allowManagedHooksOnly`
    - [Debug your configuration (Claude Code docs)](https://code.claude.com/docs/en/debug-your-config): no standalone hooks file for project or user config, CLAUDE.md versus permissions and hooks
    - [Extend Claude with skills (Claude Code docs)](https://code.claude.com/docs/en/skills): hooks in skill frontmatter
    - [Troubleshooting (Claude Code docs)](https://code.claude.com/docs/en/troubleshooting): the autocompact thrashing error and recovery
    - [Customize your status line (Claude Code docs)](https://code.claude.com/docs/en/statusline): `context_window.used_percentage`
    - [Work with sessions (Agent SDK docs)](https://code.claude.com/docs/en/agent-sdk/sessions): continue, resume and fork options in Python and TypeScript (`fork_session`, `forkSession`), forking history not the filesystem, passing captured results into a fresh session
    - [Intercept and control agent behavior with hooks (Agent SDK docs)](https://code.claude.com/docs/en/agent-sdk/hooks): SDK callback hooks, `updatedToolOutput`, deprecated `updatedMCPToolOutput`, deny reasons reaching the model
    - [Prompting best practices (Claude Platform docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices): tests verify correctness rather than define the solution
    - [The AI-native SDLC playbook: Claude Code plan mode as the default starting point (Claude Academy)](https://academy.claude.com/courses/ai-native-sdlc-playbook/plan-mode): plan mode as a design review gate
    - [The AI-native SDLC playbook: Give Claude a feedback loop (Claude Academy)](https://academy.claude.com/courses/ai-native-sdlc-playbook/give-claude-a-feedback-loop): failing test first, hook that blocks test-file edits
    - [The AI-native SDLC playbook: Hooks as approval gates (Claude Academy)](https://academy.claude.com/courses/ai-native-sdlc-playbook/hooks-as-approval-gates): team hooks in Git versus non-negotiable managed hooks
    - [Claude 101: Getting better results (Claude Academy)](https://academy.claude.com/courses/claude-101/getting-better-results): iterating in the Claude apps, specific feedback
    - [What are artifacts and how do I use them? (Claude Help Center)](https://support.claude.com/en/articles/9487310-what-are-artifacts-and-how-do-i-use-them): editing an earlier message to branch a conversation
    - [Effective context engineering for AI agents (Anthropic Engineering)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents): structured note-taking outside the context window
    - [Effective harnesses for long-running agents (Anthropic Engineering)](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents): progress file plus git history for fresh context windows, JSON for structured state
    - [Claude Code best practices, archived copy of the 2025 engineering post (Wayback Machine, June 2, 2025)](https://web.archive.org/web/20250602202220/https://www.anthropic.com/engineering/claude-code-best-practices): the full test-driven iteration recipe the current docs no longer carry, a Markdown checklist as a scratchpad, one-by-one fixes from a checklist, explaining the task up front
    - [Claude Code best practices (Anthropic Engineering, original URL)](https://www.anthropic.com/engineering/claude-code-best-practices): now redirects to the Claude Code docs best-practices page
    - [CMA_iterate_fix_failing_tests notebook (Claude Cookbooks, Managed Agents)](https://raw.githubusercontent.com/anthropics/claude-cookbooks/main/managed_agents/CMA_iterate_fix_failing_tests.ipynb): interacting test failures that resolve once dependencies are fixed
    - [Run Claude Code programmatically (Claude Code docs)](https://code.claude.com/docs/en/headless): `-p`, stdin piping and its 10MB cap, exit codes and SIGTERM, output formats, `--json-schema` and `structured_output`, invalid-schema behavior, `stream-json` events (`system/init`, `system/api_retry`, `parent_tool_use_id`), `--bare`, permission modes for `-p`, `--permission-prompts none`, `--allowedTools` syntax, system prompt flags, session capture and resume, the security review script
    - [Authentication (Claude Code docs)](https://code.claude.com/docs/en/authentication): `claude setup-token` and the one-year `CLAUDE_CODE_OAUTH_TOKEN` for CI and scripts
    - [Model configuration (Claude Code docs)](https://code.claude.com/docs/en/model-config): turning thinking off in `/config`, and the models where it cannot be turned off
    - [Claude Code GitHub Actions (Claude Code docs)](https://code.claude.com/docs/en/github-actions): quick and manual setup, credentials and inputs, interactive and automation modes, trigger checks, example workflows, the code-review workflow and inline-comment tool, schedules and `claude_args`, `GITHUB_TOKEN` behavior, fork PR secrets, cost controls, upgrading from beta, CLAUDE.md for review criteria
    - [Use Claude Code GitHub Actions with cloud providers (Claude Code docs)](https://code.claude.com/docs/en/github-actions-cloud-providers): OIDC trust for Bedrock, Agent Platform and Foundry
    - [Claude Code GitLab CI/CD (Claude Code docs)](https://code.claude.com/docs/en/gitlab-ci-cd): beta status and maintainer, the quick-setup job, tokens, mention triggers, providers, security and cost guidance, CLAUDE.md
    - [Use Claude Code in the cloud (Claude Code docs)](https://code.claude.com/docs/en/claude-code-on-the-web): cloud sessions and their research-preview availability, auto-fix for pull requests
    - [Code Review (Claude Code docs)](https://code.claude.com/docs/en/code-review): availability, verification step, severity tags, neutral check run, triggers and comment commands, `REVIEW.md` rules (severity, nit cap, skip rules, verification bar, re-review convergence), reactions and dismissals, cost, the local `/code-review` command and effort levels
    - [Find bugs with ultrareview (Claude Code docs)](https://code.claude.com/docs/en/ultrareview): `/code-review ultra` reproduces each finding and runs only when invoked
    - [Catch security issues as Claude writes code (Claude Code docs)](https://code.claude.com/docs/en/security-guidance): a separate reviewer with fresh context instead of self-grading
    - [Run agents in parallel (Claude Code docs)](https://code.claude.com/docs/en/agents): worktrees, agent teams, agent view, `/batch`, token multiplication
    - [Track team usage with analytics (Claude Code docs)](https://code.claude.com/docs/en/analytics): contribution metrics are conservative and unavailable with Zero Data Retention
    - [Monitoring (Claude Code docs)](https://code.claude.com/docs/en/monitoring-usage): OpenTelemetry quick start, exporters, intervals, managed settings, subprocess scope, metrics and attributes, prompt and tool logging defaults, traces beta, `OTEL_RESOURCE_ATTRIBUTES` and cardinality, `api_error` and `api_refusal` events
    - [Get structured output from agents (Agent SDK docs)](https://code.claude.com/docs/en/agent-sdk/structured-outputs): `output_format` and `outputFormat`
    - [Streaming input (Agent SDK docs)](https://code.claude.com/docs/en/agent-sdk/streaming-vs-single-mode): streaming input mode as the preferred mode
    - [How the agent loop works (Agent SDK docs)](https://code.claude.com/docs/en/agent-sdk/agent-loop): `max_turns`, `max_budget_usd`, `allowed_tools` and `disallowed_tools` behavior
    - [Configure permissions (Agent SDK docs)](https://code.claude.com/docs/en/agent-sdk/permissions): `allowedTools` with `permissionMode: "dontAsk"`, permission mode option names
    - [Agent SDK reference for Python (Agent SDK docs)](https://code.claude.com/docs/en/agent-sdk/python): `allowed_tools` and `output_format` descriptions
    - [Track cost and usage (Agent SDK docs)](https://code.claude.com/docs/en/agent-sdk/cost-tracking): `maxBudgetUsd` and `max_budget_usd`
    - [Observability with OpenTelemetry (Agent SDK docs)](https://code.claude.com/docs/en/agent-sdk/observability): the CLI process does the exporting, trace span names, content logging opt-ins
    - [Agent SDK overview (Agent SDK docs)](https://code.claude.com/docs/en/agent-sdk/overview): driving the CLI as a subprocess with `-p` and `--output-format json`
    - [Prompting Claude Opus 4.8 (Claude Platform docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-4-8): a concrete single-pass review bar; validating review prompts against evals
    - [Prompting Claude Sonnet 5 (Claude Platform docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-sonnet-5): "be conservative" lowers recall on newer models; the coverage-first finding-stage prompt
    - [Prompting Claude Opus 5 (Claude Platform docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/prompting-claude-opus-5): report everything and filter in a separate pass
    - [Extended thinking (Claude Platform docs)](https://platform.claude.com/docs/en/build-with-claude/extended-thinking): Claude 4.7 and later reject extended-thinking requests
    - [Errors (Claude API docs)](https://platform.claude.com/docs/en/api/errors): extended thinking removed from Claude 4.7 and later
    - [Claude Code Analytics API (Claude API docs)](https://platform.claude.com/docs/en/manage-claude/claude-code-analytics-api): daily per-user Claude Code metrics
    - [Analytics APIs (Claude API docs)](https://platform.claude.com/docs/en/manage-claude/analytics-api): Admin keys and Analytics keys are not interchangeable
    - [Workspaces (Claude API docs)](https://platform.claude.com/docs/en/manage-claude/workspaces): the Claude Code workspace and per-user monthly spend limits
    - [Use Claude Code with your Team or Enterprise plan (Claude Help Center)](https://support.claude.com/en/articles/11845131-use-claude-code-with-your-team-or-enterprise-plan): usage-based Enterprise billing with no per-seat usage limits, controls in the claude.ai admin console
    - [What is the Enterprise plan? (Claude Help Center)](https://support.claude.com/en/articles/9797531-what-is-the-enterprise-plan): no per-seat usage limits on usage-based Enterprise
    - [Access the Compliance API (Claude Help Center)](https://support.claude.com/en/articles/13015708-access-the-compliance-api): Enterprise availability and what the Compliance API does not cover
    - [claude-code-action README (anthropics on GitHub)](https://github.com/anthropics/claude-code-action/blob/main/README.md): automatic mode detection, runs on your own runner, v1 unified inputs, admin rights for quick setup
    - [claude-code-action usage docs (anthropics on GitHub)](https://github.com/anthropics/claude-code-action/blob/main/docs/usage.md): structured outputs through `--json-schema` and `structured_output`, the flaky-test example, other inputs and `allowed_bots`
    - [claude-code-action action.yml (anthropics on GitHub)](https://raw.githubusercontent.com/anthropics/claude-code-action/main/action.yml): `use_sticky_comment` and `track_progress`
    - [claude-code-action test-failure-analysis example (anthropics on GitHub)](https://raw.githubusercontent.com/anthropics/claude-code-action/main/examples/test-failure-analysis.yml): routing on a 0.7 confidence threshold
    - [code-review plugin command (anthropics/claude-code on GitHub)](https://raw.githubusercontent.com/anthropics/claude-code/main/plugins/code-review/commands/code-review.md): parallel review agents, validation subagents, high-signal criteria, the false-positive list, one comment per issue, stopping when Claude has already commented
    - [code-review plugin README (anthropics/claude-code on GitHub)](https://raw.githubusercontent.com/anthropics/claude-code/main/plugins/code-review/README.md): the older 0 to 100 confidence description
    - [claude-code-security-review README (anthropics on GitHub)](https://raw.githubusercontent.com/anthropics/claude-code-security-review/main/README.md): not hardened against prompt injection, approval for external contributors, every-commit warning
    - [claude-code-security-review prompts.py (anthropics on GitHub)](https://raw.githubusercontent.com/anthropics/claude-code-security-review/main/claudecode/prompts.py): the finding shape, the 0.7 reporting threshold, scoping to issues the PR adds
    - [The AI-native SDLC playbook: AI in the PR review loop (Claude Academy)](https://academy.claude.com/courses/ai-native-sdlc-playbook/ai-in-the-pr-review-loop): identical review passes, findings that neither approve nor block, feeding findings back into CLAUDE.md
    - [The AI-native SDLC playbook: CI/CD integration and deployment (Claude Academy)](https://academy.claude.com/courses/ai-native-sdlc-playbook/ci-cd-integration-and-deployment): agent changes arrive as PRs, a hook enforces the production gate
    - [The AI-native SDLC playbook: Continuous evals in CI (Claude Academy)](https://academy.claude.com/courses/ai-native-sdlc-playbook/continuous-evals-in-ci): an eval suite that runs in CI on changes to CLAUDE.md, skills or hooks
    - [Deploying Claude Enterprise with confidence: Visibility: what you can measure (Claude Academy)](https://academy.claude.com/courses/deploying-claude-enterprise-with-confidence/visibility-what-you-can-measure): the Compliance API, not telemetry, as the content record
    - [Language models (mostly) know what they know (Anthropic Research)](https://www.anthropic.com/research/language-models-mostly-know-what-they-know): calibration is weaker on new tasks, the reason to calibrate confidence thresholds on your own data
