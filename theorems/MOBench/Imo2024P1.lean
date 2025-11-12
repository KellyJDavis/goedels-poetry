
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2024, Problem 1

Determine all real numbers α such that, for every positive integer n, the
integer

     ⌊α⌋ + ⌊2α⌋ + ... + ⌊nα⌋

is a multiple of n.
-/
abbrev solutionSet : Set ℝ := {α : ℝ | ∃ m : ℤ, α = 2 * m}

theorem imo2024_p1 (α : ℝ) :
  α ∈ solutionSet ↔
  ∀ n : ℕ, 0 < n → (n : ℤ) ∣ ∑ i ∈ Finset.Icc 1 n, ⌊i * α⌋ := by sorry
