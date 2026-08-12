# Target Design v4.2 Editorial Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the eight remaining low-severity specification ambiguities without changing the v4.2 product boundary or runtime schema.

**Architecture:** Keep `docs/target-design-v2/src/index.template.html` as the only authoritative design source. Protect the three drift-prone contracts with section-local verifier pins, add mutation-sensitive tests before editing the source, rebuild both generated HTML artifacts, and commit the complete editorial batch once all checks pass.

**Tech Stack:** HTML design source, Python document linter/tests, Mermaid CLI offline renderer, Git.

## Global Constraints

- Baseline is branch `feat/unified-llm-modelspec` at `0165ded` on 2026-08-12.
- Do not change the 18-gate set, the ten Chapter 0 non-goals, or production result schemas.
- Edit `src/index.template.html`; never hand-edit `index.html` or `artifact.html`.
- Add strong verifier pins for comparison NotRequested semantics, ExecEvent self-reference, and Structure/Runtime snapshot isolation.
- Preserve all existing user changes and produce one low-risk editorial commit only after full verification.

---

### Task 1: Add mutation-sensitive editorial contract tests

**Files:**
- Modify: `docs/target-design-v2/tools/verify_gates.py`
- Modify: `docs/target-design-v2/tools/test_verify_gates.py`
- Modify: `docs/target-design-v2/tools/test_module_contracts.py`

**Interfaces:**
- Consumes: existing `validate(...)`, module-section extraction, `REQUIRED_TEXT`, and `MODULE_CONTRACT_REQUIRED_TEXT`.
- Produces: section-local pins for the three drift-prone contracts and forbidden-residue checks for the two arm-level NotRequested phrasings.

- [ ] **Step 1: Add failing pins and mutation cases**

  Require the following normative contracts in their owning modules:

  ```text
  NotRequested iff metric not in ComparisonRequest.requested_metrics
  require event.resolved_semantic_ref == event.event_id
  StructureRegistrySnapshot and RuntimeRegistrySnapshot are distinct frozen snapshots
  runtime_registry_digest is excluded from model_input_digest
  ```

  Reject the stale phrases `两侧都未请求才是 NotRequested` and `仅单侧 NotRequested`.

- [ ] **Step 2: Run the focused verifier tests and confirm RED**

  Run:

  ```powershell
  python tools/verify_gates.py
  python -m unittest tools.test_verify_gates.VerifyGatesContractTest.test_editorial_closure_contracts_are_section_local_and_mutation_sensitive -v
  ```

  Expected: failure only because the authoritative source still lacks the new contracts or retains the stale phrases.

### Task 2: Apply the eight source-faithful editorial fixes

**Files:**
- Modify: `docs/target-design-v2/src/index.template.html`
- Modify: `docs/target-design-v2/HANDOFF.md`

**Interfaces:**
- Consumes: the current Chapter 0, code-ir, runtime-events, comparison, gate-system, and Chapter 13 contracts.
- Produces: one internally consistent v4.2 specification with no arm-level request ambiguity.

- [ ] **Step 1: Fix comparison and release-fixture wording**

  Make `ComparisonRequest.requested_metrics` the sole NotRequested authority, remove the unreachable one-sided NotRequested branch, and explain that release boundary fixtures may assert either rejection or the correct scoped disposition.

- [ ] **Step 2: Close runtime-event and schedule identities**

  Add `require event.resolved_semantic_ref == event.event_id` and replace the premature cross-stream motivation with the stream-agnostic actual-issue/completion rule.

- [ ] **Step 3: Close CodeIR construction and registry snapshot staging**

  Populate `logical_rank_order` before `per_rank`, compute `model_digest` only after every non-derived CodeIR field is fixed, and state that Structure and Runtime are two distinct snapshots whose digests enter model and runtime input stages respectively.

- [ ] **Step 4: Align the product wording**

  Replace “multi-configuration comparison interface” with K-configuration sweep summary plus sealed left/right comparison; N-way deltas are composed by callers from pairwise requests.

- [ ] **Step 5: Run focused GREEN checks**

  Run `python tools/verify_gates.py` and the affected verifier/module-contract test classes. Expected: all pass.

### Task 3: Rebuild, verify, and commit the editorial batch

**Files:**
- Regenerate: `docs/target-design-v2/index.html`
- Regenerate: `docs/target-design-v2/artifact.html`

**Interfaces:**
- Consumes: the edited authoritative template.
- Produces: deterministic generated artifacts and one verified editorial commit.

- [ ] **Step 1: Build twice and compare SHA-256 hashes**

  Run `python tools/build_doc.py` twice. The `index.html` hash must match across runs, and the `artifact.html` hash must match across runs.

- [ ] **Step 2: Run the standard regression**

  Run:

  ```powershell
  python tools/verify_gates.py
  python -m unittest discover -s tools -p "test_*.py" -v
  ```

  Expected: all required contracts, all 18 structural gates, and all tests pass.

- [ ] **Step 3: Audit and commit**

  Require `git diff --check` to pass, confirm stale wording is absent, inspect the scoped diff, and create one editorial commit containing only this plan, source/HANDOFF edits, verifier/tests, and generated artifacts.
