import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Show that for any positive integer $n$, we have $\prod_{k=1}^n (1 + 1/k^3) \leq 3 - 1/n$.-/
theorem induction_prod1p1onk3le3m1onn (n : ℕ) (h₀ : 0 < n) :
    ∏ k in Finset.Icc 1 n, (1 + (1 : ℝ) / k ^ 3) ≤ (3 : ℝ) - 1 / ↑n := by sorry
