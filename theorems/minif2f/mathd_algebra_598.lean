import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Suppose that  $4^{a}=5$, $5^{b}=6$, $6^{c}=7,$ and  $7^{d}=8$. What is $a\cdot b\cdot c\cdot d$? Show that it is \frac{3}{2}.-/
theorem mathd_algebra_598 (a b c d : ℝ) (h₁ : (4 : ℝ) ^ a = 5) (h₂ : (5 : ℝ) ^ b = 6)
    (h₃ : (6 : ℝ) ^ c = 7) (h₄ : (7 : ℝ) ^ d = 8) : a * b * c * d = 3 / 2 := by sorry
