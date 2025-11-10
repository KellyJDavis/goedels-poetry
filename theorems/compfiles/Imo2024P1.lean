import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2024, Problem 1

Determine all real numbers α such that, for every positive integer n, the
integer

     ⌊α⌋ + ⌊2α⌋ + ... + ⌊nα⌋

is a multiple of n.
-/

namespace Imo2024P1

/- determine -/ abbrev solutionSet : Set ℝ := sorry

theorem imo2024_p1 (α : ℝ) :
  α ∈ solutionSet ↔
  ∀ n : ℕ, 0 < n → (n : ℤ) ∣ ∑ i ∈ Finset.Icc 1 n, ⌊i * α⌋ := sorry

end Imo2024P1
