# forge-v2

A coding agent harness for small local models, built on one rule: never trust the model's claims, only its evidence.

Runs `qwen2.5-coder:3b` through Ollama. A 3B model will happily tell you it fixed something it didn't touch, so the harness treats "the model said it worked" as worth nothing. Tasks only complete when the required evidence actually exists: a file was really read, a test really ran, a command really exited 0.

## how a task runs

1. `run_task(prompt)` derives a success definition up front: what evidence would prove this task is done (file listing, file contents, test result, command result).
2. The model emits tool calls as JSON. Tools: `list_files`, `read_file`, `write_file`, `search_files`, `run_command`, `apply_change_with_test`, `remember`, `recall`. Each carries a schema and a risk tier.
3. Tool results are recorded as evidence in typed buckets. Failures count too: a failed test run is evidence, recorded separately, not discarded.
4. The loop stops on verified completion, the step budget (20), or the stall detector (3 iterations with no new evidence).
5. The final answer is grounded in the collected evidence. If completion wasn't verified, the answer is labeled `[INCOMPLETE - not verified]` with exactly what's missing, instead of a confident guess.

## the parts that make it honest

**Real sandboxing, honestly reported.** `run_command` goes through one of two backends. If a Docker daemon is reachable: short-lived container, `--network none`, memory/pids/cpu caps, no-new-privileges, auto-removed. If not: a restricted local backend with a scrubbed environment (credentials stripped), timeouts, output caps and POSIX rlimits. The fallback is explicitly *not* isolation (no filesystem or network confinement) and every result records which backend ran it and what that backend actually guarantees. Nothing is ever reported as "sandboxed" when it wasn't.

**Transactional edits.** `apply_change_with_test` makes edit -> test -> keep/rollback one atomic operation. Snapshot the file, write, run the test through the sandbox. Exit 0 keeps the change, anything else restores the exact previous state. A bad edit never just sits there.

**Redundant call prevention.** An exact repeat of a tool call is served from cache, marked as cached, and doesn't count as new evidence, so a looping model can't fake progress.

**Durable run records.** Every run writes `runs/<id>.json`: task, model, every tool call with arguments, result, timing and status, the completion check, the final answer. Failed calls are recorded, not dropped.

**Project memory.** SQLite-backed, scoped per workspace. Facts, assumptions and decisions are distinct kinds with provenance (who said it, which run). Contradicted entries are superseded, never overwritten, so the project keeps a decision history. Recalled memory is injected as a labeled block, and facts are never shown with the same confidence as assumptions.

**An eval harness.** `eval_harness.py` defines reproducible task cases (real workspace, real tool execution, scripted model responses) and compares baseline vs candidate implementations. A change is only promotable with zero regressions; "same pass count but different failures" doesn't count.

## run it

```bash
# needs python 3.10+, and ollama with the model pulled
ollama pull qwen2.5-coder:3b
pip install requests pytest
python forge.py
```

Drop files into `workspace/` and type a task. Docker is optional but gives you the actually-isolated backend.

Run the tests (60 of them, no live model needed):

```bash
python -m pytest
```

## what it doesn't do

- Completion checking is keyword heuristics (`define_success` matches words like "test" or "run" to evidence types). It works for the tasks it's built for; it's not a general judge.
- The tool protocol is tolerant JSON parsing over free-text model output. A 3B model still malforms calls sometimes, and that burns steps.
- The restricted-local fallback is guardrails, not a sandbox. On a machine without Docker, treat `run_command` output accordingly.
- Single workspace, single model, no streaming. This is a harness for making one small model reliable, not a framework.
