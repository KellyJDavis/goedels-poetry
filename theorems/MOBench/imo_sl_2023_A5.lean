
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2023 A5

Let $N > 0$ be an integer and $(a_0, a_1, \dots, a_N)$ be a permutation of $(0, 1, \dots, N)$.
Suppose that the sequence of absolute differences $(|a_0 - a_1|, \dots, |a_{N - 1} - a_N|)$
is a permutation of $(1, 2, \dots, N)$.

Prove that $\max\{a_0, a_N\} \ge \lfloor (N + 1)/4 \rfloor + 1$.
-/
/- special open -/ open Fin
/--
A `NicePerm N` is a structure containing the permutations that satisfy the problem's conditions.
- `toPerm`: The permutation `a` of `{0, ..., N}`.
- `distPerm`: The permutation of `{1, ..., N}` given by the absolute differences.
- `distPerm_spec`: The proof that the differences `|aᵢ - aᵢ₊₁|` match the `distPerm`.
-/
structure NicePerm (N : ℕ) where
  toPerm : Equiv (Fin (N + 1)) (Fin (N + 1))
  distPerm : Equiv (Fin N) (Fin N)
  distPerm_spec : ∀ i : Fin N,
    Nat.dist (toPerm i.castSucc).val (toPerm i.succ).val = (distPerm i).val + 1

theorem imo_sl_2023_A5 (N : ℕ) (p : NicePerm N) :
  (if N = 0 then 0 else (N + 1) / 4 + 1) ≤ max (p.toPerm (last N)).val (p.toPerm 0).val := by sorry
