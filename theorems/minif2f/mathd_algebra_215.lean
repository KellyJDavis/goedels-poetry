import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- What is the sum of the two values of $x$ for which $(x+3)^2 = 121$? Show that it is -6.-/
theorem mathd_algebra_215 (S : Finset ℝ) (h₀ : ∀ x : ℝ, x ∈ S ↔ (x + 3) ^ 2 = 121) :
    (∑ k in S, k) = -6 := by sorry
