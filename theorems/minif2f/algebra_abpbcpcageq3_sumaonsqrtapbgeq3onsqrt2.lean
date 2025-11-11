import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- For positive real numbers a, b, c, such that $3 \leq ab+bc+ca$, show that $3/\sqrt{2} \leq a/\sqrt{a+b} + b/\sqrt{b+c} + c/\sqrt{c+a}$.-/
theorem algebra_abpbcpcageq3_sumaonsqrtapbgeq3onsqrt2 (a b c : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c)
    (h₁ : 3 ≤ a * b + b * c + c * a) :
    3 / Real.sqrt 2 ≤ a / Real.sqrt (a + b) + b / Real.sqrt (b + c) + c / Real.sqrt (c + a) := by sorry
