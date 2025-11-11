import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- For a sequence of nonnegative real numbers $a_1, a_2, \ldots, a_n$ such that $\sum_{i=1}^n a_i = n$, show that $\prod_{i=1}^n a_i \leq 1$.-/
theorem algebra_amgm_sum1toneqn_prod1tonleq1 (a : ℕ → NNReal) (n : ℕ)
    (h₀ : (∑ x in Finset.range n, a x) = n) : (∏ x in Finset.range n, a x) ≤ 1 := by sorry
