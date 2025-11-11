import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Assume that $m$ and $n$ are both positive reals, $m^3 = 2$, $n^3 = 4$, and $a + bm + cn = 0$ for rational numbers $a$, $b$, and $c$.
Show that $a = b = c = 0$.-/
theorem algebra_apbmpcneq0_aeq0anbeq0anceq0 (a b c : ℚ) (m n : ℝ) (h₀ : 0 < m ∧ 0 < n)
    (h₁ : m ^ 3 = 2) (h₂ : n ^ 3 = 4) (h₃ : (a : ℝ) + b * m + c * n = 0) : a = 0 ∧ b = 0 ∧ c = 0 := by sorry
