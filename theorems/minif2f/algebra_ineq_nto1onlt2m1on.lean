import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Let $n$ be a positive natural number. Show that $n^{1/n} \leq 2 - 1/n$.-/
theorem algebra_ineq_nto1onlt2m1on (n : ℕ) : (n : ℝ) ^ ((1 : ℝ) / n) ≤ 2 - 1 / n := by sorry
