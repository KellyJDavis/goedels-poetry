import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Show that if $x$ and $y$ are positive integers such that $x^{y^2} = y^x$, then $(x, y)$ is equal to $(1, 1)$, $(16, 2)$, or $(27, 3)$.-/
theorem imo_1997_p5 (x y : ℕ) (h₀ : 0 < x ∧ 0 < y) (h₁ : x ^ y ^ 2 = y ^ x) :
    (x, y) = (1, 1) ∨ (x, y) = (16, 2) ∨ (x, y) = (27, 3) := by sorry
