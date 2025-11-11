import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The first and ninth terms of an arithmetic sequence are $\frac23$ and $\frac45$, respectively. What is the fifth term? Show that it is \frac{11}{15}.-/
theorem mathd_algebra_452 (a : ℕ → ℝ) (h₀ : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n)
    (h₁ : a 1 = 2 / 3) (h₂ : a 9 = 4 / 5) : a 5 = 11 / 15 := by sorry
