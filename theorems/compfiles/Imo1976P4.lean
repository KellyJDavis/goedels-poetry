import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1976, Problem 4

Determine, with proof, the largest number which is the product
of positive integers whose sum is 1976.
-/

namespace Imo1976P4

/- determine -/ abbrev solution : ℕ := sorry

theorem imo1976_p4 :
    IsGreatest
      { n | ∃ s : Finset ℕ, ∑ i ∈ s, i = 1976 ∧ ∏ i ∈ s, i = n }
      solution := sorry


end Imo1976P4
