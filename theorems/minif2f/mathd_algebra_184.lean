import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- We have two geometric sequences of positive real numbers: $$6,a,b\text{ and }\frac{1}{b},a,54$$Solve for $a$. Show that it is 3\sqrt{2}.-/
theorem mathd_algebra_184 (a b : NNReal) (h₀ : 0 < a ∧ 0 < b) (h₁ : a ^ 2 = 6 * b)
    (h₂ : a ^ 2 = 54 / b) : a = 3 * NNReal.sqrt 2 := by sorry
