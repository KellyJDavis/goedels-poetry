import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Solve for $a$: $\dfrac{8^{-1}}{4^{-1}}-a^{-1}=1$. Show that it is -2.-/
theorem mathd_algebra_129 (a : ℝ) (h₀ : a ≠ 0) (h₁ : 8⁻¹ / 4⁻¹ - a⁻¹ = 1) : a = -2 := by sorry
