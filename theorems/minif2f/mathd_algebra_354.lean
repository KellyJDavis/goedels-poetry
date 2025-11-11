import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- In an arithmetic sequence, the 7th term is 30, and the 11th term is 60. What is the 21st term of this sequence? Show that it is 135.-/
theorem mathd_algebra_354 (a d : ℝ) (h₀ : a + 6 * d = 30) (h₁ : a + 10 * d = 60) :
    a + 20 * d = 135 := by sorry
