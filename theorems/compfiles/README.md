# Compfiles Problem Library

This directory contains Lean formalizations of olympiad-style mathematics problems sourced from the `dwrensha/compfiles` project, a catalog of contest problems and solutions maintained for Lean 4 users.<sup>[1](https://github.com/dwrensha/compfiles)</sup>

- Each file mirrors a corresponding problem module from the upstream library, making it easy to cross-reference the original statements and proofs.
- The collection targets Lean `v4.15.0`.
- Where available, the problem declarations are preserved so they can be imported directly or adapted for automated problem extraction workflows.

When updating the directory, pull changes from the upstream repository against the same Lean compiler version to avoid breaking imports, and document any local deviations so they can be reconciled during future syncs.
