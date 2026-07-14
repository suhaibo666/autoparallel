# Memory Evaluator Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the review findings into reproducible local failures and a bounded two-card Ascend measurement.

**Architecture:** Keep evidence separate from production estimator code. Local pytest cases encode semantic invariants as strict expected failures; an NPU runner observes actual optimizer-boundary gradient buffers without changing MindFormers. Remote files are copied only to `/tmp` and removed after collection.

**Tech Stack:** Python 3, pytest, MindSpore 2.10, MindFormers PyNative, Ascend `msrun`, PowerShell/SSH.

## Global Constraints

- Use at most two Ascend NPUs and at most three training steps.
- Inspect `npu-smi info` before every run; never kill another user's process.
- Compare evaluator allocated peak with `ms.runtime.max_memory_allocated()` and report reserved separately.
- Preserve all pre-existing dirty local and remote files.
- Do not change estimator production code during evidence collection.

---

### Task 1: Local semantic counterexamples

**Files:**
- Create: `tests/test_review_evidence.py`

**Interfaces:**
- Consumes: `_build_parallel`, `build_llm_spec`, `ShapeEval`, `Evaluator`.
- Produces: six `pytest.mark.xfail(strict=True)` regression cases runnable as real failures with `--runxfail`.

- [ ] **Step 1: Add config and reshard failures**

Assert that `ulysses`, interleave 4, and reshard `never` survive import. Compare the complete `always` and `never` gather timelines and require them to differ.

- [ ] **Step 2: Add router, TP, and tensor-identity failures**

Require a non-empty router parameter list, TP=2 embedding/head local numel equal to half of TP=1, and one resolved size per tensor name per layer.

- [ ] **Step 3: Add the NPU-backed accumulated-gradient failure**

Require `optstep.breakdown.grad_buf > 0`, because all reduced gradients were measured immediately before the real optimizer.

- [ ] **Step 4: Verify intended failures**

Run: `python -m pytest --runxfail -q tests/test_review_evidence.py`

Expected: six assertion failures with concrete actual values, not import or setup errors.

- [ ] **Step 5: Verify normal-suite behavior**

Run: `python -m pytest -q -rxX tests/test_review_evidence.py`

Expected: six XFAIL results and exit code 0.

### Task 2: Runtime gradient evidence

**Files:**
- Create: `analysis/realmachine/run_main_grad_probe.py`
- Create: `analysis/realmachine/run_main_grad_probe.sh`

**Interfaces:**
- Consumes: the existing DSv4 4-layer YAML and `trainer.model` at `_optimizer_update()`.
- Produces: per-rank `[MAIN_GRAD]` and `[MEMPROBE]` evidence lines.

- [ ] **Step 1: Observe the real optimizer boundary**

Subclass the PyNative Trainer, snapshot `param.main_grad` or `param.grad` before calling the original `_optimizer_update`, then snapshot again after training/zero-grad.

- [ ] **Step 2: Bound the Ascend run**

Use physical cards `6,7`, unique ports, two workers, fused DSA, and one step per probe iteration. Stop after three one-step runs.

- [ ] **Step 3: Transfer without text transcoding**

Use `scp` to host `/tmp`, followed by `docker cp` to container `/tmp`; do not pipe source containing non-ASCII text through PowerShell.

- [ ] **Step 4: Compare logical and physical gradient bytes**

For FSDP-2, verify the allocated difference before/after zero-grad equals half of the logical FP32 gradient bytes.

- [ ] **Step 5: Clean remote assets**

Resolve and verify all dedicated `/tmp/codex_*` paths before deleting only those paths.

### Task 3: Evidence report and verification

**Files:**
- Create: `analysis/realmachine/review_evidence_2026-07-14.md`

**Interfaces:**
- Consumes: local failure output and NPU measurement lines.
- Produces: confirmed/contradicted/unverified classifications with reproduction commands.

- [ ] **Step 1: Record commits, config, cards, commands, and measured values**

- [ ] **Step 2: Correct claims contradicted or narrowed by NPU evidence**

- [ ] **Step 3: Run full verification**

Run: `python -m pytest -q`

Run: `python -m compileall -q cost_eval analysis/realmachine tests/test_review_evidence.py`

- [ ] **Step 4: Check diffs and incomplete markers**

Run: `git diff --check`
