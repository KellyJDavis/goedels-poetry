
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2021 A1

Let $n$ be an integer and $A$ be a subset of $\{0, 1, …, 5^n\}$ of size $4n + 2$.
Prove that there exists $a, b, c ∈ A$ such that $a < b < c$ and $c + 2a > 3b$.
-/
/- special open -/ open List
theorem imo_sl_2021_A1 (hn : n ≠ 0) {A : Finset ℕ}
    (hA : A.card = 4 * n + 2) (hA0 : ∀ a ∈ A, a ≤ 5 ^ n) :
    ∃ a ∈ A, ∃ b ∈ A, ∃ c ∈ A, a < b ∧ b < c ∧ 3 * b < c + 2 * a := by sorry
