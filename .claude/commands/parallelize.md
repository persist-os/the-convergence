---
description: Set up git worktrees for parallel development
allowed-tools: Bash, Read, Write, AskUserQuestion, Task, TodoWrite
---

# Parallelize Command

Set up git worktrees for parallel development across independent work streams.

## Steps

### 1. Analyze Task and Extract Streams

Parse the user's request to identify independent work streams:

```
Example:
USER: "Add OAuth, billing, and notifications"

STREAMS:
1. OAuth Authentication - Files: src/auth/, models/user.py
2. Billing Integration - Files: src/billing/, services/stripe.py
3. Email Notifications - Files: src/notifications/, workers/email.py
```

### 2. Ask for Coordination Mode

```
question: "How should I coordinate work across the worktrees?"
options:
  1. Independent - Create worktrees, provide terminal commands
  2. Coordinated - Create worktrees, spawn agents, handle merging
```

### 3. Create Worktrees

```bash
PROJECT_NAME=$(basename $(pwd))
git worktree add -b [branch-name] ../${PROJECT_NAME}-[stream-name]
```

### 4. Provide Instructions

For independent mode, provide copy-paste terminal commands.
For coordinated mode, spawn agents in each worktree.

### 5. Cleanup Commands

```bash
git worktree remove ../project-feature-name
git branch -d feature-name
```

## Success Criteria

- All worktrees created successfully
- Each on correct branch
- Commands provided are copy-paste ready
- Clear merge/cleanup instructions given
