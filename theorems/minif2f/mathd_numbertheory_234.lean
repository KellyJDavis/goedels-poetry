import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- When the two-digit integer $``\text{AB}"$ is cubed, the value is $912,\!673$. What is $A + B$? Show that it is 16.-/
theorem mathd_numbertheory_234 (a b : ℕ) (h₀ : 1 ≤ a ∧ a ≤ 9 ∧ b ≤ 9)
    (h₁ : (10 * a + b) ^ 3 = 912673) : a + b = 16 := by sorry
