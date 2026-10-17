> Historical audit snapshot (2026-10-17). Read the release findings (`/docs/findings.md`, retained internal evidence) for later fixes and final decisions. Private source, feature caches and raw run dumps cited below are not distributed.

# DSpine training audit, shared brief (2026-10-17 02:45 EDT)

We are training a DSpine drafter (paper: `docs/reference/dspine-paper-2609.36173v2.txt`, Sections 4.x,
Appendix A, Table 4) for Qwen3.8-Flash-Next (arch `qwen4_exp`: 48 layers, Gated-DeltaNet/QSA, 4 hyper-connection
streams, hidden 2560, vocab 248320). No public DSpine training code exists. Ours in `dspine/` is the first
implementation, so we must find out whether it is right. Contract: `docs/SPEC.md`. Code docs: `dspine/README.md`,
`dspine/STATUS.md`, `docs/notes/trainer.md`.

## Symptom

The full run (frozen config `dspine/configs/qwen38-flash-next-full.json`; installed `runs/rolling/run.json` on the workstation)
warm-started from a ~200-step pilot. Window 0 of 14 is done (step 4516 of 55,877, 8% of the 540M-token schedule).
Each step is 4 sequences x 256 anchors = 1,024 draft blocks of 16, L=6, peak LR 6e-4 with 4% warmup by response
tokens, then cosine decay.

| step | complete-round tau b6/b8 (T=1) | opening tau b6/b8 | T0 tau b6 |
|---|---|---|---|
| 0 | 1.109 / 1.125 | 1.999 / 1.999 | 1.125 |
| 1000 | 1.352 / 1.337 | 2.459 / 2.460 | 1.373 |
| 2000 | 1.408 / 1.405 | 2.557 / 2.575 | 1.477 |
| 3000 | 1.519 / 1.516 | 2.582 / 2.676 | |
| 4516 | 1.570 / 1.590 | 2.838 / 2.870 | |

(Eval: 18 held-out sequences, about 64 anchors per domain, exact q/p rejection with Monte Carlo at T=1.)
Post-warmup gains are slow (+0.09 from step 2500 to 4516). Complete-round tau is far below opening tau.
Loss terms, mean over 100 steps: `ce` (position-weighted SUM over 15 positions) 50.4 -> 37.8; `l1` 12.2 -> 9.85;
refinement ~4.3; `embedding` 25.3 -> 21.2; `lambda` -> ~17-19, so lambda*emb ~ 360 against backbone ~13.
For MTP parity we need complete-round tau ~2.7 (b6) / ~2.95 (b8). The goal is ~3.75-4.1.

Validation that already exists: our modeling/propose/eval code loads tencent/Qwen3-8B-DSpine-b16 and reproduces
T0 tau ~10.7 on Qwen3-8B GSM8K (dense target, reference path). This validates inference on that path only. It does
NOT validate training, the loss, our target's feature pipeline, the packed/FlexAttention training path, or the
189,090-token restricted proposal head.

## Known suspect, already being tested (do not duplicate)

`dspine/losses.py` computes `lambda = beta * sg(ce_backbone)`, where `ce_backbone` is the position-weighted SUM
(sum of weights ~6.63). The paper says lambda = beta_t sg(CE_bb), "the token cross-entropy of the backbone output".
The TRAINER agent is preparing a lambda A/B and is editing `dspine/losses.py`, `dspine/train.py`,
`dspine/ab_lambda.py`, `dspine/forks.py` and `dspine/scripts/ab_lambda_job.sh`. Do not edit those files.

## Rules for every agent

- Never touch the production `qwen3.8-flash-next` service, unit, catalog entry or app directory on the workstation. Never print
  or copy secrets (HF token, cloud credentials).
- the workstation (`ssh compute-host`) is busy: the GPU is running the training job, and host RAM is ~15 GB free while the target is
  loaded. Do not use the GPU. Do not start ai-jobs, systemd units or containers. Do not reserve the GPU or touch
  `runs/rolling`, `ckpt/full` or `ai-dspine-*` units. Reads are fine. Any remote python must use < 2 GB RAM and run with
  `nice -n 19`. Prefer copying small samples (<= 2 GB total) to the Mac under `/tmp/audit-<you>/` and analysing them there.
  the workstation code mirror: `./work/code/dspine/`. Data: `./work/{gen,features,ckpt,runs}`.
- Write only your own paths (given in your task). Commit only those paths with `git add <paths>` and a clear message.
  Do not revert or reformat other files.
- Rank findings by their expected impact on acceptance (tau). Back each one with file:line evidence and, where
  possible, a numeric reproduction. Mark each as CONFIRMED BUG, DEVIATION FROM PAPER, RISK or OK. Give each an exact
  proposed fix as a diff in your report; do not apply it unless your task says so. Propose a decisive GPU test of
  <= 20 minutes on an RTX PRO 6000 for anything you cannot settle on CPU.
- Report back briefly within ~60 minutes, by 03:50 EDT. Put full details in your report file.
