
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1976, Problem 4

Determine, with proof, the largest number which is the product
of positive integers whose sum is 1976.
-/
abbrev solution : ℕ := 2 * 3^658

theorem imo1976_p4 :
    IsGreatest
      { n | ∃ s : Finset ℕ, ∑ i ∈ s, i = 1976 ∧ ∏ i ∈ s, i = n }
      solution := by sorry
