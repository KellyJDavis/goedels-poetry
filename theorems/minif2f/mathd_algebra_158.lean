import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The sum of 5 consecutive even integers is 4 less than the sum of the first 8 consecutive odd counting numbers. What is the smallest of the even integers? Show that it is 8.-/
theorem mathd_algebra_158 (a : ℕ) (h₀ : Even a)
    (h₁ : ∑ k in Finset.range 8, (2 * k + 1) - ∑ k in Finset.range 5, (a + 2 * k) = (4 : ℤ)) :
    a = 8 := by sorry
