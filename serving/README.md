# Portable serving references

This directory contains code authored for this project:

- `sampling.py`: joint cutoff-tie-preserving target processing, sparse-q exact
  rejection, residual correction, proposal-cache walking.
- `oracle.py`: hash-checked frozen conditional tables for controlled testing.
- `pit.py`: independently randomized probability integral transforms.
- `profile/joint_exactness.py`: joint-law goodness-of-fit analysis.
- `hybrid/test_exactness.py`: exhaustive toy-law check for heterogeneous proposals.

Read [EXACTNESS_PROTOCOL.md](EXACTNESS_PROTOCOL.md) for the historical preregistered
test design, and [the hybrid note](../docs/notes/hybrid-mtp-dspine.md) for an MTP-first
proposal design. The private runtime, scheduler, target patches, copied overlays
and kernels are not distributed. Historical paths in those documents identify
retained internal evidence, not runnable release entry points.
