import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Show that for any real number $x$ and any natural number $n$, if $x > -1$, then $(1+nx)\leq (1+x)^n$.-/
theorem induction_1pxpownlt1pnx (x : ℝ) (n : ℕ) (h₀ : -1 < x) (h₁ : 0 < n) :
    1 + ↑n * x ≤ (1 + x) ^ (n : ℕ) := by sorry
