> Historical audit snapshot (2026-10-17). Read the release findings (`/docs/findings.md`, retained internal evidence) for later fixes and final decisions. Private source, feature caches and raw run dumps cited below are not distributed.

# Clean-room interpretation register

Written during phase 1, before opening `dspine/*.py`. Sources are the local
paper, Section 4, Appendix A, Table 4, and `docs/SPEC.md`. Reference defaults
follow the paper's explicit equations and the target specification.

| Question | Default | Switch / alternative |
|---|---|---|
| What is “token cross-entropy” in lambda? | Unweighted mean CE over valid predicted tokens | `lambda_ce=weighted_mean`: position-weighted CE divided by valid weight sum; `position_sum`: position-weighted sum averaged over blocks. The paper does not specify this reduction. |
| Reduction across independent sampled blocks | Mean of per-block position sums | `reduction=sum` or `token_mean` (divide by valid position weights). Eq. 6 and 7 explicitly sum positions and layers, without saying how batching is reduced. |
| Refinement losses where masks exclude positions | Zero excluded terms, retain original block denominator | `refinement_mask_reduction=conditional` divides each masked sum by its included weight. Appendix A says “computed only where” but does not specify a changed denominator. |
| Does the per-layer input norm also act on fused context? | No: context fusion already includes `hidden_norm`, and SPEC says fused context supplies each layer's K/V | `context_input_norm=True`. Qwen3 attention/MLP wiring follows SPEC; the paper alone does not describe its norm placement. |
| Apply K head RMSNorm to context as well as block keys? | Yes, per SPEC's Qwen3 q/k norm | `context_k_norm=False`. |
| Epsilon in the injection's scalar RMS | None: literal Eq. 5 | `rms_write_eps=True` uses the configured RMSNorm epsilon. Mathematical RMS is separate from learned RMSNorm. |
| Which layers use correct predecessors with L=6? | Layers 1--3, literally Appendix A and consistent with SPEC's ceil(L/2) | `curriculum_layers` can implement a different layer count. One uniform draw per block is always reused for every selected layer. |
| Readout for embedding loss | Post-injection h at every layer, before final norm | `embedding_post_injection=False` or `embedding_after_final_norm=True`. These alternatives contradict Eq. 6 but make suspected implementation differences testable. |
| PCA details | Center row-normalized E, centered Gram SUM, eigenvalue exponent -1/2, normalize projected rows | `whiten_center` / `whiten_power`. Covariance mean vs Gram sum differs by a global scalar canceled by row normalization. Signs and rotations inside degenerate eigenspaces are arbitrary and require consistent transformations of all transfer-space weights. |

Unambiguous choices: sqrt(r) multiplies predicted readout features and known
token table rows for every injection/refinement; C itself has unit row norm.
The embedding loss is cosine alignment, so sqrt(r) cancels there. Refinement
starts from cached pre-injection final states, replaces the immediate correct
predecessor in both gate and message, uses the unchanged model-derived
receiver, and scores the backbone's fixed candidate set. Position 1's
predecessor is always the anchor. CE refinement masks membership; L1 masks
candidate target mass strictly greater than 1e-4, independently of CE.

Context window membership and absolute block positions are explicit inputs.
The paper does not state the context/next-token storage offset. That must be
checked against the feature data contract in phase 2 rather than guessed.

Qwen3 transformer implementation details (pre-norm residuals, SwiGLU,
split-half rotary positions, GQA, epsilon and theta) use the target shape
contract. The paper does not give all these operations algebraically.
