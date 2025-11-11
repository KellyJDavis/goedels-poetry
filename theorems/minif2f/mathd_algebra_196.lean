import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Find the sum of all solutions of the equation $|2-x|= 3$. Show that it is 4.-/
theorem mathd_algebra_196 (S : Finset ℝ) (h₀ : ∀ x : ℝ, x ∈ S ↔ abs (2 - x) = 3) :
    (∑ k in S, k) = 4 := by sorry
