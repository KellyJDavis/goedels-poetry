
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2009 N1 (P1)

Let $n$ be a positive integer.
Let $a_1, a_2, …, a_k$ be distinct integers in $\{1, 2, …, n\}$, with $k > 1$.
Prove that there exists $i ≤ k$ such that $n$ does not divide $a_i (a_{i + 1} - 1)$.
Here, we denote $a_{k + 1} = a_1$.
-/
theorem imo_sl_2009_N1 (hk : 1 < Nat.succ k) {a : Fin (Nat.succ k) → ℤ}
    (ha : a.Injective) {n : ℕ} (ha0 : ∀ i, 0 < a i ∧ a i ≤ n) :
    ¬∀ i, (n : ℤ) ∣ a i * (a (i + 1) - 1) := by sorry
