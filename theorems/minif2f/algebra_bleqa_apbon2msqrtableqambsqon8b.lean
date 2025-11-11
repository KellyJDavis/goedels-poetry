import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Let $a$ and $b$ be positive real numbers such that $b \leq a$. Show that $\frac{a+b}{2} - \sqrt{ab} \leq \frac{(a-b)^2}{8b}$.-/
theorem algebra_bleqa_apbon2msqrtableqambsqon8b (a b : ℝ) (h₀ : 0 < a ∧ 0 < b) (h₁ : b ≤ a) :
    (a + b) / 2 - Real.sqrt (a * b) ≤ (a - b) ^ 2 / (8 * b) := by sorry
